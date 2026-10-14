# pool/base.py
# Copyright (C) 2005-2026 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php


"""Base constructs for connection pools."""

from __future__ import annotations

from collections import deque
import dataclasses
from enum import Enum
import os
import threading
import time
import traceback
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Protocol
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref

from .. import event
from .. import exc
from .. import log
from .. import util
from ..util.typing import Self

if TYPE_CHECKING:
    from ..engine.interfaces import DBAPIConnection
    from ..engine.interfaces import DBAPICursor
    from ..engine.interfaces import Dialect
    from ..event import _DispatchCommon
    from ..event import _ListenerFnType
    from ..event import dispatcher
    from ..sql._typing import _InfoType


@dataclasses.dataclass(frozen=True)
class PoolResetState:
    """describes the state of a DBAPI connection as it is being passed to
    the :meth:`.PoolEvents.reset` connection pool event.

    .. versionadded:: 2.0.0b3

    """

    __slots__ = ("transaction_was_reset", "terminate_only", "asyncio_safe")

    transaction_was_reset: bool
    """Indicates if the transaction on the DBAPI connection was already
    essentially "reset" back by the :class:`.Connection` object.

    This boolean is True if the :class:`.Connection` had transactional
    state present upon it, which was then not closed using the
    :meth:`.Connection.rollback` or :meth:`.Connection.commit` method;
    instead, the transaction was closed inline within the
    :meth:`.Connection.close` method so is guaranteed to remain non-present
    when this event is reached.

    """

    terminate_only: bool
    """indicates if the connection is to be immediately terminated and
    not checked in to the pool.

    This occurs for connections that were invalidated, as well as asyncio
    connections that were not cleanly handled by the calling code that
    are instead being garbage collected.   In the latter case,
    operations can't be safely run on asyncio connections within garbage
    collection as there is not necessarily an event loop present.

    """

    asyncio_safe: bool
    """Indicates if the reset operation is occurring within a scope where
    an enclosing event loop is expected to be present for asyncio applications.

    Will be False in the case that the connection is being garbage collected.

    """


class ResetStyle(Enum):
    """Describe options for "reset on return" behaviors."""

    reset_rollback = 0
    reset_commit = 1
    reset_none = 2


_ResetStyleArgType = Union[
    ResetStyle,
    Literal[True, None, False, "commit", "rollback"],
]
reset_rollback, reset_commit, reset_none = list(ResetStyle)


class LeaseAction(Enum):
    """Describe the action taken by the pool when a checked-out
    connection exceeds its configured lease.

    .. versionadded:: 2.1

    """

    #: the expired connection is recorded in the expired lease list and
    #: the :meth:`_events.PoolEvents.lease_expired` event is emitted;
    #: nothing else is done.
    log = "log"

    #: the expired connection is forcibly reclaimed by the pool: its
    #: capacity is released back to the pool and the connection handle
    #: still held by the borrower is invalidated, raising
    #: :class:`~sqlalchemy.exc.LeaseExpiredError` on further use.
    reclaim = "reclaim"

    #: the expired connection is marked suspect and left with the
    #: borrower; it is terminated and recycled (rather than returned to
    #: the idle pool) when the borrower finally releases it.
    invalidate = "invalidate"


lease_log, lease_reclaim, lease_invalidate = list(LeaseAction)

_LeaseActionArgType = Union[
    LeaseAction, Literal["log", "reclaim", "invalidate"]
]


@dataclasses.dataclass(frozen=True)
class LeaseStatus:
    """Immutable snapshot of the state of a single checked-out connection
    lease.

    Instances are delivered to the :meth:`_events.PoolEvents.lease_expired`
    event and returned by :meth:`_pool.Pool.checked_out_leases` and
    :meth:`_pool.Pool.expired_leases`.

    .. versionadded:: 2.1

    """

    __slots__ = (
        "connection_record",
        "dbapi_connection",
        "checked_out_at",
        "lease_time",
        "expires_at",
        "age",
        "thread_ident",
        "source",
        "traceback",
        "expired",
        "state",
    )

    connection_record: ConnectionPoolEntry
    """the :class:`.ConnectionPoolEntry` to which the lease applies."""

    dbapi_connection: Optional[DBAPIConnection]
    """the DBAPI connection associated with the lease at the time the
    snapshot was taken; this is ``None`` if the connection was already
    reclaimed."""

    checked_out_at: float
    """``time.time()`` at which the connection was checked out."""

    lease_time: float
    """configured lease duration, in seconds."""

    expires_at: float
    """``time.time()`` value at which the lease is considered expired."""

    age: float
    """seconds elapsed since checkout at the time the snapshot was taken."""

    thread_ident: int
    """identifier of the thread that checked the connection out."""

    source: str
    """short description (``filename:lineno in callable``) of the most
    recent caller outside of SQLAlchemy on the checkout call stack."""

    traceback: str
    """formatted checkout call stack leading to the most recent caller
    outside of SQLAlchemy, useful for diagnosing which borrower holds the
    connection."""

    expired: bool
    """True if the lease had expired when the snapshot was taken."""

    state: Literal["active", "expired", "invalidated", "reclaimed", "returned"]
    """disposition state of the lease:

    ``"active"``
      the lease has not expired;

    ``"expired"``
      the lease has expired and is being recorded only
      (:class:`.LeaseAction.log`);

    ``"invalidated"``
      the lease has expired with :class:`.LeaseAction.invalidate`; the
      connection is still with the borrower but will be recycled when it
      is returned;

    ``"reclaimed"``
      the connection was forcibly reclaimed by the pool
      (:class:`.LeaseAction.reclaim`);

    ``"returned"``
      an expired lease was resolved when the borrower finally released
      the connection.

    """

    def status_line(self) -> str:
        """Return a one-line human readable description of the lease."""

        return "Lease age=%.2fs limit=%.2fs state=%s source=%s thread=%s" % (
            self.age,
            self.lease_time,
            self.state,
            self.source,
            self.thread_ident,
        )


class _Lease:
    """Mutable per-checkout lease bookkeeping (internal)."""

    __slots__ = (
        "starttime",
        "lease_time",
        "expires_at",
        "source",
        "traceback_text",
        "thread_ident",
        "event_loop",
        "expired",
        "disposition",
        "reclaimed",
    )

    starttime: float
    lease_time: float
    expires_at: float
    source: str
    traceback_text: str
    thread_ident: int
    event_loop: Any
    expired: bool
    disposition: Optional[LeaseAction]
    reclaimed: bool

    def __init__(
        self,
        lease_time: float,
        source: str,
        traceback_text: str,
        thread_ident: int,
        event_loop: Any,
    ):
        self.starttime = time.time()
        self.lease_time = lease_time
        self.expires_at = self.starttime + lease_time
        self.source = source
        self.traceback_text = traceback_text
        self.thread_ident = thread_ident
        self.event_loop = event_loop
        self.expired = False
        self.disposition = None
        self.reclaimed = False


def _current_event_loop() -> Any:
    """Return the running asyncio event loop, if any, else None."""

    try:
        import asyncio

        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None
    except ImportError:  # pragma: no cover
        return None


def _checkout_stack_frames() -> List[traceback.FrameSummary]:
    """Return the frames of the current checkout call stack.

    When the pool is driven from an asyncio engine, checkout runs inside
    a worker greenlet whose own frame chain never contains the calling
    user coroutine.  The user code is suspended on the parent (asyncio)
    greenlet, so its frame chain is appended.

    """

    frames = list(traceback.extract_stack()[:-1])

    # asyncio checkout runs inside a worker greenlet; this walks the
    # suspended parent (asyncio) greenlet to reach the calling coroutine.
    # greenlet is an optional dependency (sync-only installs may not
    # have it), and sync checkout has no parent greenlet to walk.
    try:
        import greenlet
    except ImportError:
        greenlet = None  # type: ignore[assignment]

    if greenlet is not None:
        current = greenlet.getcurrent()
        if getattr(current, "__sqlalchemy_greenlet_provider__", False):
            frame = getattr(current.parent, "gr_frame", None)
            while frame is not None:
                frames.append(
                    traceback.FrameSummary(
                        frame.f_code.co_filename,
                        frame.f_lineno,
                        frame.f_code.co_name,
                        lookup_line=False,
                    )
                )
                frame = frame.f_back
    return frames


def _checkout_caller_source() -> Tuple[str, str]:
    """Derive ``(short_source, traceback_text)`` describing the most
    recent non-SQLAlchemy caller on the current checkout stack."""

    package_prefix = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    frames = _checkout_stack_frames()
    external = [
        frame
        for frame in frames
        if not os.path.abspath(frame.filename).startswith(package_prefix)
    ]
    if external:
        chosen = external[-1]
        relevant = external[-25:]
    elif frames:
        chosen = frames[-1]
        relevant = [chosen]
    else:  # pragma: no cover
        return "<unknown>", ""

    lineno = chosen.lineno if chosen.lineno is not None else 0
    return (
        "%s:%d in %s" % (chosen.filename, lineno, chosen.name),
        "".join(traceback.format_list(relevant)),
    )


class _LeaseReaper(threading.Thread):
    """Background daemon thread that enforces expired checkout leases."""

    def __init__(self, pool: Pool):
        super().__init__(daemon=True)
        self.name = "sqlalchemy-pool-lease-reaper-%#x" % id(pool)
        self._pool_ref = weakref.ref(pool)
        self._stop_event = threading.Event()

    def run(self) -> None:
        pool = self._pool_ref()
        if pool is None:
            return
        interval = pool._lease_reap_interval
        while not self._stop_event.wait(interval):
            pool = self._pool_ref()
            if pool is None:
                return
            try:
                pool._reap_expired_leases()
            except BaseException:
                pool.logger.error(
                    "Unhandled exception in lease reaper thread",
                    exc_info=True,
                )

    def stop(self) -> None:
        self._stop_event.set()


class _ConnDialect:
    """partial implementation of :class:`.Dialect`
    which provides DBAPI connection methods.

    When a :class:`_pool.Pool` is combined with an :class:`_engine.Engine`,
    the :class:`_engine.Engine` replaces this with its own
    :class:`.Dialect`.

    """

    is_async = False
    has_terminate = False

    def do_rollback(self, dbapi_connection: PoolProxiedConnection) -> None:
        dbapi_connection.rollback()

    def do_commit(self, dbapi_connection: PoolProxiedConnection) -> None:
        dbapi_connection.commit()

    def do_terminate(self, dbapi_connection: DBAPIConnection) -> None:
        dbapi_connection.close()

    def do_close(self, dbapi_connection: DBAPIConnection) -> None:
        dbapi_connection.close()

    def _do_ping_w_event(self, dbapi_connection: DBAPIConnection) -> bool:
        raise NotImplementedError(
            "The ping feature requires that a dialect is "
            "passed to the connection pool."
        )

    def get_driver_connection(self, connection: DBAPIConnection) -> Any:
        return connection


class _AsyncConnDialect(_ConnDialect):
    is_async = True


class _CreatorFnType(Protocol):
    def __call__(self) -> DBAPIConnection: ...


class _CreatorWRecFnType(Protocol):
    def __call__(self, rec: ConnectionPoolEntry) -> DBAPIConnection: ...


class Pool(log.Identified, event.EventTarget):
    """Abstract base class for connection pools."""

    dispatch: dispatcher[Pool]
    echo: log._EchoFlagType

    _orig_logging_name: Optional[str]
    _dialect: Union[_ConnDialect, Dialect] = _ConnDialect()
    _creator_arg: Union[_CreatorFnType, _CreatorWRecFnType]
    _invoke_creator: _CreatorWRecFnType
    _invalidate_time: float

    def __init__(
        self,
        creator: Union[_CreatorFnType, _CreatorWRecFnType],
        recycle: int = -1,
        echo: log._EchoFlagType = None,
        logging_name: Optional[str] = None,
        reset_on_return: _ResetStyleArgType = True,
        events: Optional[List[Tuple[_ListenerFnType, str]]] = None,
        dialect: Optional[Union[_ConnDialect, Dialect]] = None,
        pre_ping: bool = False,
        lease_time: float = -1,
        lease_on_expire: _LeaseActionArgType = "log",
        lease_reap_interval: Optional[float] = None,
        _dispatch: Optional[_DispatchCommon[Pool]] = None,
    ):
        """
        Construct a Pool.

        :param creator: a callable function that returns a DB-API
          connection object.  The function will be called with
          parameters.

        :param recycle: If set to a value other than -1, number of
          seconds between connection recycling, which means upon
          checkout, if this timeout is surpassed the connection will be
          closed and replaced with a newly opened connection. Defaults to -1.

        :param logging_name:  String identifier which will be used within
          the "name" field of logging records generated within the
          "sqlalchemy.pool" logger. Defaults to a hexstring of the object's
          id.

        :param echo: if True, the connection pool will log
         informational output such as when connections are invalidated
         as well as when connections are recycled to the default log handler,
         which defaults to ``sys.stdout`` for output..   If set to the string
         ``"debug"``, the logging will include pool checkouts and checkins.

         The :paramref:`_pool.Pool.echo` parameter can also be set from the
         :func:`_sa.create_engine` call by using the
         :paramref:`_sa.create_engine.echo_pool` parameter.

         .. seealso::

             :ref:`dbengine_logging` - further detail on how to configure
             logging.

        :param reset_on_return: Determine steps to take on
         connections as they are returned to the pool, which were
         not otherwise handled by a :class:`_engine.Connection`.
         Available from :func:`_sa.create_engine` via the
         :paramref:`_sa.create_engine.pool_reset_on_return` parameter.

         :paramref:`_pool.Pool.reset_on_return` can have any of these values:

         * ``"rollback"`` - call rollback() on the connection,
           to release locks and transaction resources.
           This is the default value.  The vast majority
           of use cases should leave this value set.
         * ``"commit"`` - call commit() on the connection,
           to release locks and transaction resources.
           A commit here may be desirable for databases that
           cache query plans if a commit is emitted,
           such as Microsoft SQL Server.  However, this
           value is more dangerous than 'rollback' because
           any data changes present on the transaction
           are committed unconditionally.
         * ``None`` - don't do anything on the connection.
           This setting may be appropriate if the database / DBAPI
           works in pure "autocommit" mode at all times, or if
           a custom reset handler is established using the
           :meth:`.PoolEvents.reset` event handler.

         * ``True`` - same as 'rollback', this is here for
           backwards compatibility.
         * ``False`` - same as None, this is here for
           backwards compatibility.

         For further customization of reset on return, the
         :meth:`.PoolEvents.reset` event hook may be used which can perform
         any connection activity desired on reset.

         .. seealso::

            :ref:`pool_reset_on_return`

            :meth:`.PoolEvents.reset`

        :param events: a list of 2-tuples, each of the form
         ``(callable, target)`` which will be passed to :func:`.event.listen`
         upon construction.   Provided here so that event listeners
         can be assigned via :func:`_sa.create_engine` before dialect-level
         listeners are applied.

        :param dialect: a :class:`.Dialect` that will handle the job
         of calling rollback(), close(), or commit() on DBAPI connections.
         If omitted, a built-in "stub" dialect is used.   Applications that
         make use of :func:`_sa.create_engine` should not use this parameter
         as it is handled by the engine creation strategy.

        :param pre_ping: if True, the pool will emit a "ping" (typically
         "SELECT 1", but is dialect-specific) on the connection
         upon checkout, to test if the connection is alive or not.   If not,
         the connection is transparently re-connected and upon success, all
         other pooled connections established prior to that timestamp are
         invalidated.     Requires that a dialect is passed as well to
         interpret the disconnection error.

        :param lease_time: number of seconds a connection may remain checked
         out before its lease is considered expired.  Defaults to ``-1``,
         which disables checkout lease governance entirely; when leases are
         disabled the pool behaves exactly as it does without the lease
         feature.  When set to a value greater than or equal to zero, every
         checked-out connection receives a lease that records its checkout
         time and calling source, and expired leases are detected by a
         background reaper thread as well as opportunistically on further
         checkouts.  Expired leases are listed by
         :meth:`_pool.Pool.expired_leases` and emit the
         :meth:`_events.PoolEvents.lease_expired` event exactly once.  The
         ``lease_time`` limit constrains how long a borrower may *hold* a
         connection; it is distinct from the :paramref:`_pool.Pool.timeout`
         waiting limit and does not change checkout waiting or ``pre_ping``
         behavior.

         .. versionadded:: 2.1

        :param lease_on_expire: selects what the pool does once a checked-out
         lease is expired.  Defaults to ``"log"``.  Valid values are:

         * ``"log"`` - record the expired lease in
           :meth:`_pool.Pool.expired_leases` and emit
           :meth:`_events.PoolEvents.lease_expired`; the borrower keeps the
           connection.
         * ``"reclaim"`` - in addition to recording and emitting the event,
           forcibly reclaim the connection: its capacity is returned to the
           pool, and the connection handle held by the original borrower is
           invalidated such that any further use raises
           :class:`~sqlalchemy.exc.LeaseExpiredError`, rather than affecting
           a connection that may already have been handed to another
           borrower.
         * ``"invalidate"`` - mark the expired connection suspect; it is
           left with the borrower, but when it is released (including via
           garbage collection) it is terminated and recycled instead of
           being returned to the idle pool, so a possibly stale connection
           is never handed to the next borrower.

         .. versionadded:: 2.1

        :param lease_reap_interval: optional interval in seconds at which the
         background lease reaper thread scans checked-out leases.  Defaults
         to ``None``, which selects half of ``lease_time`` (with a minimum of
         0.05 seconds).

         .. versionadded:: 2.1

        """
        if logging_name:
            self.logging_name = self._orig_logging_name = logging_name
        else:
            self._orig_logging_name = None

        log.instance_logger(self, echoflag=echo)
        self._creator = creator
        self._recycle = recycle
        self._invalidate_time = 0
        self._pre_ping = pre_ping
        self._reset_on_return = util.parse_user_argument_for_enum(
            reset_on_return,
            {
                ResetStyle.reset_rollback: ["rollback", True],
                ResetStyle.reset_none: ["none", None, False],
                ResetStyle.reset_commit: ["commit"],
            },
            "reset_on_return",
        )

        self.echo = echo

        if lease_time != -1 and lease_time < 0:
            raise exc.ArgumentError(
                "lease_time must be -1 (disabled) or a non-negative number "
                "of seconds, got: %r" % (lease_time,)
            )
        self._lease_time = float(lease_time)
        parsed_lease_on_expire = util.parse_user_argument_for_enum(
            lease_on_expire,
            {
                LeaseAction.log: ["log"],
                LeaseAction.reclaim: ["reclaim"],
                LeaseAction.invalidate: ["invalidate"],
            },
            "lease_on_expire",
        )
        if parsed_lease_on_expire is None:
            raise exc.ArgumentError(
                "lease_on_expire must be one of 'log', 'reclaim' or "
                "'invalidate', got: %r" % (lease_on_expire,)
            )
        self._lease_on_expire = parsed_lease_on_expire
        if lease_reap_interval is not None:
            self._lease_reap_interval = float(lease_reap_interval)
        elif self._lease_time >= 0:
            self._lease_reap_interval = max(0.05, self._lease_time / 2.0)
        else:
            self._lease_reap_interval = 1.0

        self._leases: Dict[_ConnectionRecord, _Lease] = {}
        self._lease_history: Dict[int, LeaseStatus] = {}
        self._lease_lock = threading.RLock()
        self._reaper: Optional[_LeaseReaper] = None

        if _dispatch:
            self.dispatch._update(_dispatch, only_propagate=False)
        if dialect:
            self._dialect = dialect
        if events:
            for fn, target in events:
                event.listen(self, target, fn)

    @util.hybridproperty
    def _is_asyncio(self) -> bool:
        return self._dialect.is_async

    @util.hybridproperty
    def _lease_enabled(self) -> bool:
        return self._lease_time >= 0

    @property
    def _creator(self) -> Union[_CreatorFnType, _CreatorWRecFnType]:
        return self._creator_arg

    @_creator.setter
    def _creator(
        self, creator: Union[_CreatorFnType, _CreatorWRecFnType]
    ) -> None:
        self._creator_arg = creator

        # mypy seems to get super confused assigning functions to
        # attributes
        self._invoke_creator = self._should_wrap_creator(creator)

    @_creator.deleter
    def _creator(self) -> None:
        # needed for mock testing
        del self._creator_arg
        del self._invoke_creator

    def _should_wrap_creator(
        self, creator: Union[_CreatorFnType, _CreatorWRecFnType]
    ) -> _CreatorWRecFnType:
        """Detect if creator accepts a single argument, or is sent
        as a legacy style no-arg function.

        """

        try:
            argspec = util.get_callable_argspec(self._creator, no_self=True)
        except TypeError:
            creator_fn = cast(_CreatorFnType, creator)
            return lambda rec: creator_fn()

        if argspec.defaults is not None:
            defaulted = len(argspec.defaults)
        else:
            defaulted = 0
        positionals = len(argspec[0]) - defaulted

        # look for the exact arg signature that DefaultStrategy
        # sends us
        if (argspec[0], argspec[3]) == (["connection_record"], (None,)):
            return cast(_CreatorWRecFnType, creator)
        # or just a single positional
        elif positionals == 1:
            return cast(_CreatorWRecFnType, creator)
        # all other cases, just wrap and assume legacy "creator" callable
        # thing
        else:
            creator_fn = cast(_CreatorFnType, creator)
            return lambda rec: creator_fn()

    def _close_connection(
        self, connection: DBAPIConnection, *, terminate: bool = False
    ) -> None:
        self.logger.debug(
            "%s connection %r",
            "Hard-closing" if terminate else "Closing",
            connection,
        )
        try:
            if terminate:
                self._dialect.do_terminate(connection)
            else:
                self._dialect.do_close(connection)
        except BaseException as e:
            self.logger.error(
                f"Exception {'terminating' if terminate else 'closing'} "
                f"connection %r",
                connection,
                exc_info=True,
            )
            if not isinstance(e, Exception):
                raise

    def _create_connection(self) -> ConnectionPoolEntry:
        """Called by subclasses to create a new ConnectionRecord."""

        return _ConnectionRecord(self)

    def _invalidate(
        self,
        connection: PoolProxiedConnection,
        exception: Optional[BaseException] = None,
        _checkin: bool = True,
    ) -> None:
        """Mark all connections established within the generation
        of the given connection as invalidated.

        If this pool's last invalidate time is before when the given
        connection was created, update the timestamp til now.  Otherwise,
        no action is performed.

        Connections with a start time prior to this pool's invalidation
        time will be recycled upon next checkout.
        """
        rec = getattr(connection, "_connection_record", None)
        if not rec or self._invalidate_time < rec.starttime:
            self._invalidate_time = time.time()
        if _checkin and getattr(connection, "is_valid", False):
            connection.invalidate(exception)

    def recreate(self) -> Pool:
        """Return a new :class:`_pool.Pool`, of the same class as this one
        and configured with identical creation arguments.

        This method is used in conjunction with :meth:`dispose`
        to close out an entire :class:`_pool.Pool` and create a new one in
        its place.

        """

        raise NotImplementedError()

    def dispose(self) -> None:
        """Dispose of this pool.

        This method leaves the possibility of checked-out connections
        remaining open, as it only affects connections that are
        idle in the pool.

        .. seealso::

            :meth:`Pool.recreate`

        """

        raise NotImplementedError()

    def connect(self) -> PoolProxiedConnection:
        """Return a DBAPI connection from the pool.

        The connection is instrumented such that when its
        ``close()`` method is called, the connection will be returned to
        the pool.

        """
        return _ConnectionFairy._checkout(self)

    def _return_conn(self, record: ConnectionPoolEntry) -> None:
        """Given a _ConnectionRecord, return it to the :class:`_pool.Pool`.

        This method is called when an instrumented DBAPI connection
        has its ``close()`` method called.

        """
        self._do_return_conn(record)

    def _do_get(self) -> ConnectionPoolEntry:
        """Implementation for :meth:`get`, supplied by subclasses."""

        raise NotImplementedError()

    def _do_return_conn(self, record: ConnectionPoolEntry) -> None:
        """Implementation for :meth:`return_conn`, supplied by subclasses."""

        raise NotImplementedError()

    def status(self) -> str:
        """Returns a brief description of the state of this pool."""
        raise NotImplementedError()

    # ------------------------------------------------------------------
    # checkout lease governance
    # ------------------------------------------------------------------

    def _lease_snapshot(
        self,
        lease: _Lease,
        record: ConnectionPoolEntry,
        state: str,
        now: Optional[float] = None,
    ) -> LeaseStatus:
        if now is None:
            now = time.time()
        return LeaseStatus(
            connection_record=record,
            dbapi_connection=record.dbapi_connection,
            checked_out_at=lease.starttime,
            lease_time=lease.lease_time,
            expires_at=lease.expires_at,
            age=max(0.0, now - lease.starttime),
            thread_ident=lease.thread_ident,
            source=lease.source,
            traceback=lease.traceback_text,
            expired=lease.expired or now >= lease.expires_at,
            state=state,  # type: ignore[arg-type]
        )

    def _record_lease_history(
        self,
        lease: _Lease,
        record: ConnectionPoolEntry,
        state: str,
        now: Optional[float] = None,
    ) -> LeaseStatus:
        snapshot = self._lease_snapshot(lease, record, state, now)
        history = self._lease_history
        history[id(lease)] = snapshot
        if len(history) > 100:
            for key in tuple(history)[:-100]:
                del history[key]
        return snapshot

    def _lease_state_for(self, lease: _Lease) -> str:
        if not lease.expired:
            return "active"
        elif lease.disposition is lease_invalidate:
            return "invalidated"
        elif lease.disposition is lease_reclaim:
            return "reclaimed"
        else:
            return "expired"

    def checked_out_leases(self) -> List[LeaseStatus]:
        """Return :class:`.LeaseStatus` snapshots for all connections
        currently checked out from this pool under a lease.

        Each snapshot includes the current occupancy time (``age``) and
        the most recent checkout source (``source`` and ``traceback``).
        Returns an empty list when checkout leases are not configured.

        .. versionadded:: 2.1

        """

        if not self._lease_enabled:
            return []
        with self._lease_lock:
            now = time.time()
            return [
                self._lease_snapshot(
                    lease, record, self._lease_state_for(lease), now
                )
                for record, lease in self._leases.items()
            ]

    def expired_leases(self) -> List[LeaseStatus]:
        """Return the current expired-lease list.

        The list contains one :class:`.LeaseStatus` entry per lease that
        has ever expired, updated with its current disposition
        (``"expired"``, ``"invalidated"``, ``"reclaimed"`` or
        ``"returned"``).  The list is bounded to the 100 most recent
        expired leases.

        .. versionadded:: 2.1

        """

        if not self._lease_enabled:
            return []
        with self._lease_lock:
            return list(self._lease_history.values())

    def reap_expired_leases(self) -> None:
        """Run an immediate enforcement pass over checked-out leases.

        This is normally performed by the background reaper thread and
        opportunistically at checkout; it is exposed to allow deterministic
        enforcement by applications and tests.

        .. versionadded:: 2.1

        """

        if self._lease_enabled:
            self._reap_expired_leases()

    def _lease_begin(
        self, record: _ConnectionRecord, fairy: _ConnectionFairy
    ) -> None:
        if not self._lease_enabled:
            return

        source, traceback_text = _checkout_caller_source()
        event_loop = _current_event_loop() if self._is_asyncio else None

        lease = _Lease(
            self._lease_time,
            source,
            traceback_text,
            threading.get_ident(),
            event_loop,
        )
        record._lease = lease
        with self._lease_lock:
            self._leases[record] = lease
            self._start_reaper_locked()

        # enforce other, already outstanding leases; the brand new lease
        # is not itself reaped on this pass.
        self._reap_expired_leases(_except_lease=lease)

    def _lease_end(self, record: _ConnectionRecord) -> None:
        if not self._lease_enabled:
            return
        with self._lease_lock:
            lease = self._leases.pop(record, None)
            record._lease = None
            record._lease_invalid_on_return = False
            if lease is not None and lease.expired:
                state = (
                    "reclaimed"
                    if lease.reclaimed or lease.disposition is lease_reclaim
                    else "returned"
                )
                self._record_lease_history(lease, record, state)

    def _start_reaper_locked(self) -> None:
        reaper = self._reaper
        if reaper is None or not reaper.is_alive():
            reaper = _LeaseReaper(self)
            self._reaper = reaper
            reaper.start()

    def _stop_reaper(self) -> None:
        if not self._lease_enabled:
            return
        with self._lease_lock:
            reaper = self._reaper
            self._reaper = None
        if reaper is not None and reaper.is_alive():
            reaper.stop()
            reaper.join(timeout=1.0)

    def _reap_expired_leases(
        self, _except_lease: Optional[_Lease] = None
    ) -> None:
        now = time.time()
        newly_expired: List[Tuple[_ConnectionRecord, _Lease]] = []
        with self._lease_lock:
            for record, lease in list(self._leases.items()):
                if (
                    lease is not _except_lease
                    and not lease.expired
                    and now >= lease.expires_at
                ):
                    lease.expired = True
                    lease.disposition = self._lease_on_expire
                    newly_expired.append((record, lease))

        for record, lease in newly_expired:
            self._enforce_expired_lease(record, lease)

    def _enforce_expired_lease(
        self, record: _ConnectionRecord, lease: _Lease
    ) -> None:
        now = time.time()
        with self._lease_lock:
            initial_state = (
                "invalidated"
                if lease.disposition is lease_invalidate
                else "expired"
            )
            snapshot = self._record_lease_history(
                lease, record, initial_state, now
            )

        # the event hook fires exactly once per expired lease
        try:
            self.dispatch.lease_expired(
                record.dbapi_connection, record, snapshot
            )
        except BaseException:
            self.logger.error(
                "Exception raised by lease_expired event handler",
                exc_info=True,
            )

        self.logger.warning(
            "Connection %r checkout lease expired after %.2fs "
            "(limit %.2fs); action=%s source=%s",
            record.dbapi_connection,
            snapshot.age,
            lease.lease_time,
            lease.disposition.value if lease.disposition is not None else None,
            lease.source,
        )

        if lease.disposition is lease_invalidate:
            self._mark_lease_invalid_on_return(record)
        elif lease.disposition is lease_reclaim:
            self._reclaim_lease(record, lease)

    def _mark_lease_invalid_on_return(self, record: _ConnectionRecord) -> None:
        with self._lease_lock:
            record._lease_invalid_on_return = True
        ref = record.fairy_ref
        fairy = ref() if ref is not None else None
        if fairy is not None:
            fairy._lease_invalidated = True

    def _lease_expired_error(
        self, record: ConnectionPoolEntry, lease: _Lease, action: str
    ) -> exc.LeaseExpiredError:
        return exc.LeaseExpiredError(
            "The pooled connection %r was %s by the pool after its "
            "checkout lease of %.2f seconds expired (the connection was "
            "held for %.2f seconds; checkout source: %s).  Further use of "
            "this connection handle is not permitted."
            % (
                record.dbapi_connection,
                action,
                lease.lease_time,
                time.time() - lease.starttime,
                lease.source,
            )
        )

    def _reclaim_lease(self, record: _ConnectionRecord, lease: _Lease) -> None:
        error = self._lease_expired_error(record, lease, "forcibly reclaimed")

        # 1. cut the borrower's connection handle FIRST, so no further
        # read or write can land on the connection being reclaimed
        ref = record.fairy_ref
        fairy = ref() if ref is not None else None
        if fairy is not None:
            fairy._reclaimed_exc = error
            fairy._lease_invalidated = True

        # 2. for asyncio pools, the record must be enqueued back onto the
        # event loop the asyncio queue is bound to; the handle was already
        # invalidated above regardless of whether scheduling succeeds.
        if self._is_asyncio:
            loop = lease.event_loop
            loop_is_open = loop is not None and not loop.is_closed()
            if loop_is_open:
                try:
                    loop.call_soon_threadsafe(
                        self._reclaim_lease_on_loop, record, lease
                    )
                    return
                except RuntimeError:
                    pass

            # no usable event loop: cannot safely terminate or re-enqueue
            # from another thread; degrade to recycle-on-return.
            self.logger.warning(
                "Connection %r lease expired but it cannot be forcibly "
                "reclaimed outside of its event loop; the connection "
                "will be recycled when returned by the borrower",
                record.dbapi_connection,
            )
            with self._lease_lock:
                lease.disposition = lease_invalidate
                record._lease_invalid_on_return = True
                self._record_lease_history(lease, record, "invalidated")
            return

        self._reclaim_lease_sync(record, lease, fairy)

    def _reclaim_lease_sync(
        self,
        record: _ConnectionRecord,
        lease: _Lease,
        fairy: Optional[_ConnectionFairy],
    ) -> None:
        # 3. detach and terminate the raw connection (best effort;
        # _close_connection logs rather than raising on failure), so the
        # next checkout establishes a brand new connection
        raw_connection = record.dbapi_connection
        record.finalize_callback.clear()
        record.dbapi_connection = None
        if raw_connection is not None:
            self._close_connection(raw_connection, terminate=True)

        # 4. give the record (its capacity) back to the pool
        if fairy is not None:
            fairy.dbapi_connection = None  # type: ignore[assignment]
            # this runs the record checkin (with no reset, as the
            # connection is already gone) and enqueues it back to the pool
            fairy._checkin()
        # if the fairy was already garbage collected, its weakref
        # finalizer performs the checkin itself; the record no longer
        # references a connection so nothing stale can be re-used.

        with self._lease_lock:
            self._leases.pop(record, None)
            record._lease = None
            lease.reclaimed = True
            self._record_lease_history(lease, record, "reclaimed")

    def _reclaim_lease_on_loop(
        self, record: _ConnectionRecord, lease: _Lease
    ) -> None:
        """Running on the asyncio event loop thread: terminate and
        check in a forcibly reclaimed connection."""

        # if the borrower already returned the connection (and the record
        # possibly started a new lease in the meantime), nothing is left
        # to reclaim here
        with self._lease_lock:
            if self._leases.get(record) is not lease:
                return
        if not lease.expired:
            return

        raw_connection = record.dbapi_connection
        if raw_connection is not None and self._dialect.has_terminate:
            self._close_connection(raw_connection, terminate=True)

        ref = record.fairy_ref
        fairy = ref() if ref is not None else None
        record.finalize_callback.clear()
        record.dbapi_connection = None
        if fairy is not None:
            fairy.dbapi_connection = None  # type: ignore[assignment]
            try:
                fairy._checkin()
            except BaseException:
                self.logger.error(
                    "Exception checking in reclaimed async connection %r"
                    % (raw_connection,),
                    exc_info=True,
                )
                return
        elif record.in_use:
            record.checkin()

        with self._lease_lock:
            self._leases.pop(record, None)
            record._lease = None
            lease.reclaimed = True
            self._record_lease_history(lease, record, "reclaimed")


class ManagesConnection:
    """Common base for the two connection-management interfaces
    :class:`.PoolProxiedConnection` and :class:`.ConnectionPoolEntry`.

    These two objects are typically exposed in the public facing API
    via the connection pool event hooks, documented at :class:`.PoolEvents`.

    .. versionadded:: 2.0

    """

    __slots__ = ()

    dbapi_connection: Optional[DBAPIConnection]
    """A reference to the actual DBAPI connection being tracked.

    This is a :pep:`249`-compliant object that for traditional sync-style
    dialects is provided by the third-party
    DBAPI implementation in use.  For asyncio dialects, the implementation
    is typically an adapter object provided by the SQLAlchemy dialect
    itself; the underlying asyncio object is available via the
    :attr:`.ManagesConnection.driver_connection` attribute.

    SQLAlchemy's interface for the DBAPI connection is based on the
    :class:`.DBAPIConnection` protocol object

    .. seealso::

        :attr:`.ManagesConnection.driver_connection`

        :ref:`faq_dbapi_connection`

    """

    driver_connection: Optional[Any]
    """The "driver level" connection object as used by the Python
    DBAPI or database driver.

    For traditional :pep:`249` DBAPI implementations, this object will
    be the same object as that of
    :attr:`.ManagesConnection.dbapi_connection`.   For an asyncio database
    driver, this will be the ultimate "connection" object used by that
    driver, such as the ``asyncpg.Connection`` object which will not have
    standard pep-249 methods.

    .. versionadded:: 1.4.24

    .. seealso::

        :attr:`.ManagesConnection.dbapi_connection`

        :ref:`faq_dbapi_connection`

    """

    @util.ro_memoized_property
    def info(self) -> _InfoType:
        """Info dictionary associated with the underlying DBAPI connection
        referred to by this :class:`.ManagesConnection` instance, allowing
        user-defined data to be associated with the connection.

        The data in this dictionary is persistent for the lifespan
        of the DBAPI connection itself, including across pool checkins
        and checkouts.  When the connection is invalidated
        and replaced with a new one, this dictionary is cleared.

        For a :class:`.PoolProxiedConnection` instance that's not associated
        with a :class:`.ConnectionPoolEntry`, such as if it were detached, the
        attribute returns a dictionary that is local to that
        :class:`.ConnectionPoolEntry`. Therefore the
        :attr:`.ManagesConnection.info` attribute will always provide a Python
        dictionary.

        .. seealso::

            :attr:`.ManagesConnection.record_info`


        """
        raise NotImplementedError()

    @util.ro_memoized_property
    def record_info(self) -> Optional[_InfoType]:
        """Persistent info dictionary associated with this
        :class:`.ManagesConnection`.

        Unlike the :attr:`.ManagesConnection.info` dictionary, the lifespan
        of this dictionary is that of the :class:`.ConnectionPoolEntry`
        which owns it; therefore this dictionary will persist across
        reconnects and connection invalidation for a particular entry
        in the connection pool.

        For a :class:`.PoolProxiedConnection` instance that's not associated
        with a :class:`.ConnectionPoolEntry`, such as if it were detached, the
        attribute returns None. Contrast to the :attr:`.ManagesConnection.info`
        dictionary which is never None.


        .. seealso::

            :attr:`.ManagesConnection.info`

        """
        raise NotImplementedError()

    def invalidate(
        self, e: Optional[BaseException] = None, soft: bool = False
    ) -> None:
        """Mark the managed connection as invalidated.

        :param e: an exception object indicating a reason for the invalidation.

        :param soft: if True, the connection isn't closed; instead, this
         connection will be recycled on next checkout.

        .. seealso::

            :ref:`pool_connection_invalidation`


        """
        raise NotImplementedError()


class ConnectionPoolEntry(ManagesConnection):
    """Interface for the object that maintains an individual database
    connection on behalf of a :class:`_pool.Pool` instance.

    The :class:`.ConnectionPoolEntry` object represents the long term
    maintenance of a particular connection for a pool, including expiring or
    invalidating that connection to have it replaced with a new one, which will
    continue to be maintained by that same :class:`.ConnectionPoolEntry`
    instance. Compared to :class:`.PoolProxiedConnection`, which is the
    short-term, per-checkout connection manager, this object lasts for the
    lifespan of a particular "slot" within a connection pool.

    The :class:`.ConnectionPoolEntry` object is mostly visible to public-facing
    API code when it is delivered to connection pool event hooks, such as
    :meth:`_events.PoolEvents.connect` and :meth:`_events.PoolEvents.checkout`.

    .. versionadded:: 2.0  :class:`.ConnectionPoolEntry` provides the public
       facing interface for the :class:`._ConnectionRecord` internal class.

    """

    __slots__ = ()

    @property
    def in_use(self) -> bool:
        """Return True the connection is currently checked out"""

        raise NotImplementedError()

    def close(self) -> None:
        """Close the DBAPI connection managed by this connection pool entry."""
        raise NotImplementedError()


class _ConnectionRecord(ConnectionPoolEntry):
    """Maintains a position in a connection pool which references a pooled
    connection.

    This is an internal object used by the :class:`_pool.Pool` implementation
    to provide context management to a DBAPI connection maintained by
    that :class:`_pool.Pool`.   The public facing interface for this class
    is described by the :class:`.ConnectionPoolEntry` class.  See that
    class for public API details.

    .. seealso::

        :class:`.ConnectionPoolEntry`

        :class:`.PoolProxiedConnection`

    """

    __slots__ = (
        "__pool",
        "fairy_ref",
        "finalize_callback",
        "fresh",
        "starttime",
        "dbapi_connection",
        "_lease",
        "_lease_invalid_on_return",
        "__weakref__",
        "__dict__",
    )

    finalize_callback: Deque[Callable[[DBAPIConnection], None]]
    fresh: bool
    fairy_ref: Optional[weakref.ref[_ConnectionFairy]]
    starttime: float
    _lease: Optional[_Lease]
    _lease_invalid_on_return: bool

    def __init__(self, pool: Pool, connect: bool = True):
        self.fresh = False
        self.fairy_ref = None
        self.starttime = 0
        self.dbapi_connection = None
        self._lease = None
        self._lease_invalid_on_return = False

        self.__pool = pool
        if connect:
            self.__connect()
        self.finalize_callback = deque()

    dbapi_connection: Optional[DBAPIConnection]

    @property
    def driver_connection(self) -> Optional[Any]:  # type: ignore[override]  # mypy#4125  # noqa: E501
        if self.dbapi_connection is None:
            return None
        else:
            return self.__pool._dialect.get_driver_connection(
                self.dbapi_connection
            )

    @property
    @util.deprecated(
        "2.0",
        "The _ConnectionRecord.connection attribute is deprecated; "
        "please use 'driver_connection'",
    )
    def connection(self) -> Optional[DBAPIConnection]:
        return self.dbapi_connection

    _soft_invalidate_time: float = 0

    @util.ro_memoized_property
    def info(self) -> _InfoType:
        return {}

    @util.ro_memoized_property
    def record_info(self) -> Optional[_InfoType]:
        return {}

    @classmethod
    def checkout(cls, pool: Pool) -> _ConnectionFairy:
        if TYPE_CHECKING:
            rec = cast(_ConnectionRecord, pool._do_get())
        else:
            rec = pool._do_get()

        try:
            dbapi_connection = rec.get_connection()
        except BaseException as err:
            with util.safe_reraise():
                rec._checkin_failed(err, _fairy_was_created=False)

            # not reached, for code linters only
            raise

        echo = pool._should_log_debug()
        fairy = _ConnectionFairy(pool, dbapi_connection, rec, echo)

        # assign fairy_ref to the ConnectionRecord; note that under StaticPool,
        # this could be swapping out an existing fairy on that ConnectionRecord
        rec.fairy_ref = weakref.ref(fairy)

        if echo:
            pool.logger.debug(
                "Connection %r checked out from pool", dbapi_connection
            )
        return fairy

    def _checkin_failed(
        self, err: BaseException, _fairy_was_created: bool = True
    ) -> None:
        try:
            self.invalidate(e=err)
        finally:
            if self.dbapi_connection is not None:
                # invalidate() was itself interrupted, such as by a second
                # cancellation, before it could close the connection.  keep
                # the connection so that its close can be retried, but
                # have the next checkout recycle it rather than use it
                self._soft_invalidate_time = time.time()

            # the record must be returned in any case; nothing else will
            # ever return it, as no _ConnectionFairy refers to it
            self.checkin(
                _fairy_was_created=_fairy_was_created,
            )

    def checkin(self, _fairy_was_created: bool = True) -> None:
        if self.fairy_ref is None and _fairy_was_created:
            # _fairy_was_created is False for the initial get connection phase;
            # meaning there was no _ConnectionFairy and we must unconditionally
            # do a checkin.
            #
            # otherwise, if fairy_was_created==True, if fairy_ref is None here
            # that means we were checked in already, so this looks like
            # a double checkin.
            util.warn("Double checkin attempted on %s" % self)
            return
        self.fairy_ref = None
        connection = self.dbapi_connection
        pool = self.__pool

        # close out the checkout lease (idempotent), consuming any
        # pending invalid-on-return flag before the record is reused
        pool._lease_end(self)

        while self.finalize_callback:
            finalizer = self.finalize_callback.pop()
            if connection is not None:
                finalizer(connection)
        if pool.dispatch.checkin:
            pool.dispatch.checkin(connection, self)

        pool._return_conn(self)

    @property
    def in_use(self) -> bool:
        return self.fairy_ref is not None

    @property
    def needs_gc(self) -> bool:
        ref = self.fairy_ref
        return ref is not None and ref() is None

    @property
    def last_connect_time(self) -> float:
        return self.starttime

    def close(self) -> None:
        if self.dbapi_connection is not None:
            self.__close()

    def invalidate(
        self, e: Optional[BaseException] = None, soft: bool = False
    ) -> None:
        # already invalidated
        if self.dbapi_connection is None:
            return
        if soft:
            self.__pool.dispatch.soft_invalidate(
                self.dbapi_connection, self, e
            )
        else:
            self.__pool.dispatch.invalidate(self.dbapi_connection, self, e)
        if e is not None:
            self.__pool.logger.info(
                "%sInvalidate connection %r (reason: %s:%s)",
                "Soft " if soft else "",
                self.dbapi_connection,
                e.__class__.__name__,
                e,
            )
        else:
            self.__pool.logger.info(
                "%sInvalidate connection %r",
                "Soft " if soft else "",
                self.dbapi_connection,
            )

        if soft:
            self._soft_invalidate_time = time.time()
        else:
            self.__close(terminate=True)
            self.dbapi_connection = None

    def get_connection(self) -> DBAPIConnection:
        recycle = False

        # NOTE: the various comparisons here are assuming that measurable time
        # passes between these state changes.  however, time.time() is not
        # guaranteed to have sub-second precision.  comparisons of
        # "invalidation time" to "starttime" should perhaps use >= so that the
        # state change can take place assuming no measurable  time has passed,
        # however this does not guarantee correct behavior here as if time
        # continues to not pass, it will try to reconnect repeatedly until
        # these timestamps diverge, so in that sense using > is safer.  Per
        # https://stackoverflow.com/a/1938096/34549, Windows time.time() may be
        # within 16 milliseconds accuracy, so unit tests for connection
        # invalidation need a sleep of at least this long between initial start
        # time and invalidation for the logic below to work reliably.

        if self.dbapi_connection is None:
            self.info.clear()
            self.__connect()
        elif (
            self.__pool._recycle > -1
            and time.time() - self.starttime > self.__pool._recycle
        ):
            self.__pool.logger.info(
                "Connection %r exceeded timeout; recycling",
                self.dbapi_connection,
            )
            recycle = True
        elif self.__pool._invalidate_time > self.starttime:
            self.__pool.logger.info(
                "Connection %r invalidated due to pool invalidation; "
                + "recycling",
                self.dbapi_connection,
            )
            recycle = True
        elif self._soft_invalidate_time > self.starttime:
            self.__pool.logger.info(
                "Connection %r invalidated due to local soft invalidation; "
                + "recycling",
                self.dbapi_connection,
            )
            recycle = True

        if recycle:
            self.__close(terminate=True)
            self.info.clear()

            self.__connect()

        assert self.dbapi_connection is not None
        return self.dbapi_connection

    def _is_hard_or_soft_invalidated(self) -> bool:
        return (
            self.dbapi_connection is None
            or self.__pool._invalidate_time > self.starttime
            or (self._soft_invalidate_time > self.starttime)
        )

    def __close(self, *, terminate: bool = False) -> None:
        self.finalize_callback.clear()
        if self.__pool.dispatch.close:
            self.__pool.dispatch.close(self.dbapi_connection, self)
        assert self.dbapi_connection is not None
        self.__pool._close_connection(
            self.dbapi_connection, terminate=terminate
        )
        self.dbapi_connection = None

    def __connect(self) -> None:
        pool = self.__pool

        # ensure any existing connection is removed, so that if
        # creator fails, this attribute stays None
        self.dbapi_connection = None
        try:
            self.starttime = time.time()
            self.dbapi_connection = connection = pool._invoke_creator(self)
            pool.logger.debug("Created new connection %r", connection)
            self.fresh = True
        except BaseException as e:
            with util.safe_reraise():
                pool.logger.debug("Error on connect(): %s", e)
        else:
            try:
                # in SQLAlchemy 1.4 the first_connect event is not used by
                # the engine, so this will usually not be set
                if pool.dispatch.first_connect:
                    pool.dispatch.first_connect.for_modify(
                        pool.dispatch
                    ).exec_once_unless_exception(self.dbapi_connection, self)

                # init of the dialect now takes place within the connect
                # event, so ensure a mutex is used on the first run
                pool.dispatch.connect.for_modify(
                    pool.dispatch
                )._exec_w_sync_on_first_run(self.dbapi_connection, self)
            except BaseException:
                # the connection is established but the events that
                # configure it did not complete; nothing else has a
                # reference to it, so close it here or it is stranded.
                # for an asyncio driver in particular there is no other
                # opportunity, as the garbage collector cannot close a
                # connection that needs the event loop
                with util.safe_reraise():
                    try:
                        pool._close_connection(connection, terminate=True)
                    except BaseException:
                        pool.logger.exception(
                            "Exception closing connection %r stranded by a "
                            "failed connect event",
                            connection,
                        )
                    self.dbapi_connection = None


def _finalize_fairy(
    dbapi_connection: Optional[DBAPIConnection],
    connection_record: Optional[_ConnectionRecord],
    pool: Pool,
    echo: Optional[log._EchoFlagType],
    transaction_was_reset: bool = False,
    fairy: Optional[_ConnectionFairy] = None,
    is_gc_cleanup: bool = False,
) -> None:
    """Cleanup for a :class:`._ConnectionFairy` whether or not it's already
    been garbage collected.

    When using an async dialect no IO can happen here (without using
    a dedicated thread), since this is called outside the greenlet
    context and with an already running loop. In this case function
    will only log a message and raise a warning.
    """

    if is_gc_cleanup:
        assert connection_record is not None

        # check connection record to see that we're the current
        # fairy for this record.  if not, then return; assume the
        # record is either checked in, or another fairy supersedes us
        # (can happen with StaticPool)
        if not connection_record.needs_gc:
            return
        assert dbapi_connection is None
        dbapi_connection = connection_record.dbapi_connection

    # null pool is not _is_asyncio but can be used also with async dialects
    dont_restore_gced = pool._dialect.is_async

    if dont_restore_gced:
        detach = connection_record is None or is_gc_cleanup
        can_manipulate_connection = not is_gc_cleanup
        can_close_or_terminate_connection = (
            not pool._dialect.is_async or pool._dialect.has_terminate
        )
        requires_terminate_for_close = (
            pool._dialect.is_async and pool._dialect.has_terminate
        )

    else:
        detach = connection_record is None
        can_manipulate_connection = can_close_or_terminate_connection = True
        requires_terminate_for_close = False

    if dbapi_connection is not None:
        if connection_record and echo:
            pool.logger.debug(
                "Connection %r being returned to pool", dbapi_connection
            )

        try:
            if not fairy:
                assert connection_record is not None
                fairy = _ConnectionFairy(
                    pool,
                    dbapi_connection,
                    connection_record,
                    echo,
                )
            assert fairy.dbapi_connection is dbapi_connection

            # a lease that expired with the "invalidate" disposition
            # marks this connection as suspect; it must not be reset
            # (the borrower may have been stuck on exactly that call) or
            # returned to the idle pool - terminate/recycle it instead.
            lease_invalidated = can_manipulate_connection and (
                fairy._lease_invalidated
                or (
                    connection_record is not None
                    and connection_record._lease_invalid_on_return
                )
            )

            if lease_invalidated:
                assert connection_record is not None
                connection_record.invalidate(
                    e=exc.LeaseExpiredError(
                        "Connection %r is being returned to the pool "
                        "after its checkout lease expired with "
                        'lease_on_expire="invalidate"; the connection '
                        "is being recycled" % (dbapi_connection,)
                    )
                )
                fairy.dbapi_connection = None  # type: ignore[assignment]
                dbapi_connection = None
            else:
                fairy._reset(
                    pool,
                    transaction_was_reset=transaction_was_reset,
                    terminate_only=detach,
                    asyncio_safe=can_manipulate_connection,
                )

                if detach:
                    if connection_record:
                        fairy._pool = pool
                        fairy.detach()

                    if can_close_or_terminate_connection:
                        if pool.dispatch.close_detached:
                            pool.dispatch.close_detached(dbapi_connection)

                        pool._close_connection(
                            dbapi_connection,
                            terminate=requires_terminate_for_close,
                        )

        except BaseException as e:
            pool.logger.error(
                "Exception during reset or similar", exc_info=True
            )
            if not isinstance(e, Exception):
                # a cancellation or similar propagates from here, so the
                # checkin() below is not reached; return the record now,
                # rather than leaving it to garbage collection of the fairy
                if connection_record:
                    connection_record._checkin_failed(e)
                raise
            if connection_record:
                connection_record.invalidate(e=e)
        finally:
            if detach and is_gc_cleanup and dont_restore_gced:
                message = (
                    "The garbage collector is trying to clean up "
                    f"non-checked-in connection {dbapi_connection!r}, "
                    f"""which will be {
                        'dropped, as it cannot be safely terminated'
                        if not can_close_or_terminate_connection
                        else 'terminated'
                    }.  """
                    "Please ensure that SQLAlchemy pooled connections are "
                    "returned to "
                    "the pool explicitly, either by calling ``close()`` "
                    "or by using appropriate context managers to manage "
                    "their lifecycle."
                )
                pool.logger.error(message)
                util.warn(message)

    if connection_record and connection_record.in_use:
        connection_record.checkin()

    # give gc some help.  See
    # test/engine/test_pool.py::PoolEventsTest::test_checkin_event_gc[True]
    # which actually started failing when pytest warnings plugin was
    # turned on, due to util.warn() above
    if fairy is not None:
        # don't need the finalizer anymore since we are cleaning up here
        fairy._finalizer.detach()
        fairy.dbapi_connection = None  # type: ignore[assignment]
        fairy._connection_record = None
    del dbapi_connection
    del connection_record
    del fairy


class PoolProxiedConnection(ManagesConnection):
    """A connection-like adapter for a :pep:`249` DBAPI connection, which
    includes additional methods specific to the :class:`.Pool` implementation.

    :class:`.PoolProxiedConnection` is the public-facing interface for the
    internal :class:`._ConnectionFairy` implementation object; users familiar
    with :class:`._ConnectionFairy` can consider this object to be equivalent.

    .. versionadded:: 2.0  :class:`.PoolProxiedConnection` provides the public-
       facing interface for the :class:`._ConnectionFairy` internal class.

    """

    __slots__ = ()

    if typing.TYPE_CHECKING:

        def commit(self) -> None: ...

        def cursor(self, *args: Any, **kwargs: Any) -> DBAPICursor: ...

        def rollback(self) -> None: ...

        def __getattr__(self, key: str) -> Any: ...

    @property
    def is_valid(self) -> bool:
        """Return True if this :class:`.PoolProxiedConnection` still refers
        to an active DBAPI connection."""

        raise NotImplementedError()

    @property
    def is_detached(self) -> bool:
        """Return True if this :class:`.PoolProxiedConnection` is detached
        from its pool."""

        raise NotImplementedError()

    def detach(self) -> None:
        """Separate this connection from its Pool.

        This means that the connection will no longer be returned to the
        pool when closed, and will instead be literally closed.  The
        associated :class:`.ConnectionPoolEntry` is de-associated from this
        DBAPI connection.

        Note that any overall connection limiting constraints imposed by a
        Pool implementation may be violated after a detach, as the detached
        connection is removed from the pool's knowledge and control.

        """

        raise NotImplementedError()

    def close(self) -> None:
        """Release this connection back to the pool.

        The :meth:`.PoolProxiedConnection.close` method shadows the
        :pep:`249` ``.close()`` method, altering its behavior to instead
        :term:`release` the proxied connection back to the connection pool.

        Upon release to the pool, whether the connection stays "opened" and
        pooled in the Python process, versus actually closed out and removed
        from the Python process, is based on the pool implementation in use and
        its configuration and current state.

        """
        raise NotImplementedError()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
        return None


class _AdhocProxiedConnection(PoolProxiedConnection):
    """provides the :class:`.PoolProxiedConnection` interface for cases where
    the DBAPI connection is not actually proxied.

    This is used by the engine internals to pass a consistent
    :class:`.PoolProxiedConnection` object to consuming dialects in response to
    pool events that may not always have the :class:`._ConnectionFairy`
    available.

    """

    __slots__ = ("dbapi_connection", "_connection_record", "_is_valid")

    dbapi_connection: DBAPIConnection
    _connection_record: ConnectionPoolEntry

    def __init__(
        self,
        dbapi_connection: DBAPIConnection,
        connection_record: ConnectionPoolEntry,
    ):
        self.dbapi_connection = dbapi_connection
        self._connection_record = connection_record
        self._is_valid = True

    @property
    def driver_connection(self) -> Any:  # type: ignore[override]  # mypy#4125
        return self._connection_record.driver_connection

    @property
    def connection(self) -> DBAPIConnection:
        return self.dbapi_connection

    @property
    def is_valid(self) -> bool:
        """Implement is_valid state attribute.

        for the adhoc proxied connection it's assumed the connection is valid
        as there is no "invalidate" routine.

        """
        return self._is_valid

    def invalidate(
        self, e: Optional[BaseException] = None, soft: bool = False
    ) -> None:
        self._is_valid = False

    @util.ro_non_memoized_property
    def record_info(self) -> Optional[_InfoType]:
        return self._connection_record.record_info

    def cursor(self, *args: Any, **kwargs: Any) -> DBAPICursor:
        return self.dbapi_connection.cursor(*args, **kwargs)

    def __getattr__(self, key: Any) -> Any:
        return getattr(self.dbapi_connection, key)


class _ConnectionFairy(PoolProxiedConnection):
    """Proxies a DBAPI connection and provides return-on-dereference
    support.

    This is an internal object used by the :class:`_pool.Pool` implementation
    to provide context management to a DBAPI connection delivered by
    that :class:`_pool.Pool`.   The public facing interface for this class
    is described by the :class:`.PoolProxiedConnection` class.  See that
    class for public API details.

    The name "fairy" is inspired by the fact that the
    :class:`._ConnectionFairy` object's lifespan is transitory, as it lasts
    only for the length of a specific DBAPI connection being checked out from
    the pool, and additionally that as a transparent proxy, it is mostly
    invisible.

    .. seealso::

        :class:`.PoolProxiedConnection`

        :class:`.ConnectionPoolEntry`


    """

    __slots__ = (
        "dbapi_connection",
        "_connection_record",
        "_finalizer",
        "_echo",
        "_pool",
        "_counter",
        "_reclaimed_exc",
        "_lease_invalidated",
        "__weakref__",
        "__dict__",
    )

    pool: Pool
    dbapi_connection: DBAPIConnection
    _echo: log._EchoFlagType
    _reclaimed_exc: Optional[BaseException]
    _lease_invalidated: bool

    def __init__(
        self,
        pool: Pool,
        dbapi_connection: DBAPIConnection,
        connection_record: _ConnectionRecord,
        echo: log._EchoFlagType,
    ):
        self._pool = pool
        self._counter = 0
        self.dbapi_connection = dbapi_connection
        self._connection_record = connection_record
        self._echo = echo
        self._reclaimed_exc = None
        self._lease_invalidated = False

        # use weakref.finalize as a destructor
        self._finalizer = weakref.finalize(
            self,
            _finalize_fairy,
            None,
            connection_record,
            pool,
            echo,
            transaction_was_reset=False,
            is_gc_cleanup=True,
        )

    _connection_record: Optional[_ConnectionRecord]

    @property
    def driver_connection(self) -> Optional[Any]:  # type: ignore[override]  # mypy#4125  # noqa: E501
        if self._connection_record is None:
            return None
        return self._connection_record.driver_connection

    @property
    @util.deprecated(
        "2.0",
        "The _ConnectionFairy.connection attribute is deprecated; "
        "please use 'driver_connection'",
    )
    def connection(self) -> DBAPIConnection:
        return self.dbapi_connection

    @classmethod
    def _checkout(
        cls,
        pool: Pool,
        threadconns: Optional[threading.local] = None,
        fairy: Optional[_ConnectionFairy] = None,
    ) -> _ConnectionFairy:
        if not fairy:
            fairy = _ConnectionRecord.checkout(pool)

            if threadconns is not None:
                threadconns.current = weakref.ref(fairy)

        assert (
            fairy._connection_record is not None
        ), "can't 'checkout' a detached connection fairy"
        assert (
            fairy.dbapi_connection is not None
        ), "can't 'checkout' an invalidated connection fairy"

        fairy._counter += 1
        if (
            not pool.dispatch.checkout and not pool._pre_ping
        ) or fairy._counter != 1:
            if fairy._counter == 1:
                # the lease starts only once the connection is fully
                # ready to hand to the borrower (pre-ping / checkout
                # events above would otherwise consume lease time);
                # nested re-checkouts keep the original lease
                assert fairy._connection_record is not None
                pool._lease_begin(fairy._connection_record, fairy)
            return fairy

        # Pool listeners can trigger a reconnection on checkout, as well
        # as the pre-pinger.
        # there are three attempts made here, but note that if the database
        # is not accessible from a connection standpoint, those won't proceed
        # here.

        attempts = 2

        while attempts > 0:
            connection_is_fresh = fairy._connection_record.fresh
            fairy._connection_record.fresh = False
            try:
                if pool._pre_ping:
                    if not connection_is_fresh:
                        if fairy._echo:
                            pool.logger.debug(
                                "Pool pre-ping on connection %s",
                                fairy.dbapi_connection,
                            )
                        result = pool._dialect._do_ping_w_event(
                            fairy.dbapi_connection
                        )
                        if not result:
                            if fairy._echo:
                                pool.logger.debug(
                                    "Pool pre-ping on connection %s failed, "
                                    "will invalidate pool",
                                    fairy.dbapi_connection,
                                )
                            raise exc.InvalidatePoolError()
                    elif fairy._echo:
                        pool.logger.debug(
                            "Connection %s is fresh, skipping pre-ping",
                            fairy.dbapi_connection,
                        )

                pool.dispatch.checkout(
                    fairy.dbapi_connection, fairy._connection_record, fairy
                )
                pool._lease_begin(fairy._connection_record, fairy)
                return fairy
            except exc.DisconnectionError as e:
                if e.invalidate_pool:
                    pool.logger.info(
                        "Disconnection detected on checkout, "
                        "invalidating all pooled connections prior to "
                        "current timestamp (reason: %r)",
                        e,
                    )
                    fairy._connection_record.invalidate(e)
                    pool._invalidate(fairy, e, _checkin=False)
                else:
                    pool.logger.info(
                        "Disconnection detected on checkout, "
                        "invalidating individual connection %s (reason: %r)",
                        fairy.dbapi_connection,
                        e,
                    )
                    fairy._connection_record.invalidate(e)
                try:
                    fairy.dbapi_connection = (
                        fairy._connection_record.get_connection()
                    )
                except BaseException as err:
                    with util.safe_reraise():
                        fairy._connection_record._checkin_failed(
                            err,
                            _fairy_was_created=True,
                        )

                        # prevent _ConnectionFairy from being carried
                        # in the stack trace.  Do this after the
                        # connection record has been checked in, so that
                        # if the del triggers a finalize fairy, it won't
                        # try to checkin a second time.
                        del fairy

                    # never called, this is for code linters
                    raise

                attempts -= 1
            except BaseException as be_outer:
                with util.safe_reraise():
                    rec = fairy._connection_record
                    if rec is not None:
                        rec._checkin_failed(
                            be_outer,
                            _fairy_was_created=True,
                        )

                    # prevent _ConnectionFairy from being carried
                    # in the stack trace, see above
                    del fairy

                # never called, this is for code linters
                raise

        pool.logger.info("Reconnection attempts exhausted on checkout")
        fairy.invalidate()
        raise exc.InvalidRequestError("This connection is closed")

    def _checkout_existing(self) -> _ConnectionFairy:
        return _ConnectionFairy._checkout(self._pool, fairy=self)

    def _checkin(self, transaction_was_reset: bool = False) -> None:
        _finalize_fairy(
            self.dbapi_connection,
            self._connection_record,
            self._pool,
            self._echo,
            transaction_was_reset=transaction_was_reset,
            fairy=self,
        )

    def _close(self) -> None:
        self._checkin()

    def _reset(
        self,
        pool: Pool,
        transaction_was_reset: bool,
        terminate_only: bool,
        asyncio_safe: bool,
    ) -> None:
        if pool.dispatch.reset:
            pool.dispatch.reset(
                self.dbapi_connection,
                self._connection_record,
                PoolResetState(
                    transaction_was_reset=transaction_was_reset,
                    terminate_only=terminate_only,
                    asyncio_safe=asyncio_safe,
                ),
            )

        if not asyncio_safe:
            return

        if pool._reset_on_return is reset_rollback:
            if transaction_was_reset:
                if self._echo:
                    pool.logger.debug(
                        "Connection %s reset, transaction already reset",
                        self.dbapi_connection,
                    )
            else:
                if self._echo:
                    pool.logger.debug(
                        "Connection %s rollback-on-return",
                        self.dbapi_connection,
                    )
                pool._dialect.do_rollback(self)
        elif pool._reset_on_return is reset_commit:
            if self._echo:
                pool.logger.debug(
                    "Connection %s commit-on-return",
                    self.dbapi_connection,
                )
            pool._dialect.do_commit(self)

    @property
    def _logger(self) -> log._IdentifiedLoggerType:
        return self._pool.logger

    @property
    def is_valid(self) -> bool:
        return (
            self.dbapi_connection is not None and self._reclaimed_exc is None
        )

    @property
    def is_detached(self) -> bool:
        return self._connection_record is None

    @util.ro_memoized_property
    def info(self) -> _InfoType:
        if self._connection_record is None:
            return {}
        else:
            return self._connection_record.info

    @util.ro_non_memoized_property
    def record_info(self) -> Optional[_InfoType]:
        if self._connection_record is None:
            return None
        else:
            return self._connection_record.record_info

    def invalidate(
        self, e: Optional[BaseException] = None, soft: bool = False
    ) -> None:
        if self._reclaimed_exc is not None:
            # the pool already took this connection back; there is nothing
            # left for the borrower to invalidate
            return
        if self.dbapi_connection is None:
            util.warn("Can't invalidate an already-closed connection.")
            return
        if self._connection_record:
            self._connection_record.invalidate(e=e, soft=soft)
        if not soft:
            # prevent any rollback / reset actions etc. on
            # the connection
            self.dbapi_connection = None  # type: ignore[assignment]

            # finalize
            self._checkin()

    def cursor(self, *args: Any, **kwargs: Any) -> DBAPICursor:
        if self._reclaimed_exc is not None:
            raise self._reclaimed_exc
        assert self.dbapi_connection is not None
        return self.dbapi_connection.cursor(*args, **kwargs)

    def __getattr__(self, key: str) -> Any:
        # note: this is only invoked when normal attribute lookup fails,
        # so _reclaimed_exc itself (a real slot attribute) does not recurse
        reclaimed_exc = self._reclaimed_exc
        if reclaimed_exc is not None:
            raise reclaimed_exc
        return getattr(self.dbapi_connection, key)

    def detach(self) -> None:
        if self._reclaimed_exc is not None:
            # a reclaimed handle cannot be detached by the former borrower
            return
        if self._connection_record is not None:
            rec = self._connection_record
            rec.fairy_ref = None

            # cancel the finalizer, as current behavior is that there's no
            # GC "cleanup" for a detached connection.  not sure if this
            # is the most appropriate decision; see issue #13570
            self._finalizer.detach()

            rec.dbapi_connection = None
            self._pool._lease_end(rec)
            # TODO: should this be _return_conn?
            self._pool._do_return_conn(self._connection_record)

            # can't get the descriptor assignment to work here
            # in pylance.  mypy is OK w/ it
            self.info = self.info.copy()  # type: ignore[misc]

            self._connection_record = None

            if self._pool.dispatch.detach:
                self._pool.dispatch.detach(self.dbapi_connection, rec)

    def close(self) -> None:
        if self._reclaimed_exc is not None:
            # the pool already checked this connection in on our behalf;
            # the borrower's close() is a no-op so the record can't be
            # returned twice
            return
        self._counter -= 1
        if self._counter == 0:
            self._checkin()

    def _close_special(self, transaction_reset: bool = False) -> None:
        if self._reclaimed_exc is not None:
            return
        self._counter -= 1
        if self._counter == 0:
            self._checkin(transaction_was_reset=transaction_reset)
