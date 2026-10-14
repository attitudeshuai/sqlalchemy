# event/api.py
# Copyright (C) 2005-2026 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Public API functions for the event system."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

from . import attr
from .base import _registrars
from .registry import _ET
from .registry import _EventKey
from .registry import _listener_label
from .registry import _ListenerFnType
from .registry import _state_for_stored_fn
from .registry import FAILURE_RAISE
from .registry import ListenerFailure
from .. import exc
from .. import util

if TYPE_CHECKING:
    from .registry import EventTarget

CANCEL = util.symbol("CANCEL")
NO_RETVAL = util.symbol("NO_RETVAL")


class ListenerEntry(NamedTuple):
    """Read-only description of one listener as seen on a target.

    .. versionadded:: 2.1.5

    """

    event: str
    """name of the event the listener is registered for."""

    listener: Callable[..., Any]
    """the listener function as originally passed to :func:`.listen`."""

    listener_name: str
    """identifier of the listener, e.g. its qualified name."""

    on_failure: str
    """failure policy: ``"raise"``, ``"skip"`` or ``"log"``."""

    fail_count: int
    """cumulative number of times the listener raised while being
    dispatched. Only ever non-zero for the ``"skip"`` and ``"log"``
    failure policies."""

    failures: Tuple[ListenerFailure, ...]
    """recent failure records for the listener, oldest first.  Retained
    records are bounded, while :attr:`fail_count` is cumulative."""


class ListenerRegistryView(Mapping[str, Tuple[ListenerEntry, ...]]):
    """Read-only view of the listeners registered on one target.

    Maps event names to a tuple of :class:`ListenerEntry` in registration
    (dispatch) order.  The view cannot be used to alter the registry; use
    :func:`.listen` and :func:`.remove` for that purpose.

    .. versionadded:: 2.1.5

    """

    __slots__ = ("_target", "_entries")

    def __init__(
        self,
        target: "EventTarget",
        entries: Dict[str, Tuple[ListenerEntry, ...]],
    ):
        self._target = target
        self._entries = MappingProxyType(entries)

    def __getitem__(self, key: str) -> Tuple[ListenerEntry, ...]:
        return self._entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return "registered_listeners(%r, %r)" % (
            self._target,
            dict(self._entries),
        )

    @property
    def failures(self) -> Tuple[ListenerFailure, ...]:
        """All retained :class:`ListenerFailure` records for every event
        and listener on the target, in view and registration order."""

        return tuple(
            failure
            for entries in self._entries.values()
            for entry in entries
            for failure in entry.failures
        )


def _iter_instance_storage(
    collection: Any,
) -> Iterator[Tuple[Any, _ListenerFnType]]:
    if isinstance(collection, attr._JoinedListener):
        yield from _iter_instance_storage(collection.local)
        yield from _iter_instance_storage(
            getattr(collection.parent_dispatch, collection.name)
        )
    elif isinstance(collection, attr._ListenerCollection):
        for fn in collection.parent_listeners:
            yield collection.parent, fn
        for fn in collection.listeners:
            yield collection, fn
    elif isinstance(collection, attr._EmptyListener):
        for fn in collection.parent_listeners:
            yield collection.parent, fn
    elif isinstance(collection, attr._ClsLevelDispatch):
        # a class-level collection reached through a join; it can only be
        # resolved with a concrete target class, which is unavailable here
        return
    else:
        # unknown collection type; fall back to its own iteration
        for fn in collection:
            yield collection, fn


def _iter_target_storage(
    target: Any, dispatch: Any, identifier: str
) -> Iterator[Tuple[Any, _ListenerFnType]]:
    collection = getattr(dispatch, identifier)

    if isinstance(collection, attr._ClsLevelDispatch):
        # class-level target: dispatch is the root _Dispatch and each
        # event attribute is a _ClsLevelDispatch keyed by target class
        deque = collection._clslevel.get(target)
        if deque is not None:
            for fn in deque:
                yield collection, fn
    else:
        yield from _iter_instance_storage(collection)


def _entry_for_storage(
    identifier: str, owner: Any, stored_fn: _ListenerFnType
) -> ListenerEntry:
    state = _state_for_stored_fn(owner, stored_fn)
    if state is None:
        # defensive: listeners that reached a collection without going
        # through the standard registration path are shown using the
        # default policy and carry no failure history.
        return ListenerEntry(
            identifier,
            stored_fn,
            _listener_label(stored_fn),
            FAILURE_RAISE,
            0,
            (),
        )

    fail_count, failures = state.snapshot()
    return ListenerEntry(
        identifier,
        state.fn,
        state.fn_name,
        state.on_failure,
        fail_count,
        failures,
    )


def _event_key(
    target: _ET, identifier: str, fn: _ListenerFnType
) -> _EventKey[_ET]:
    for evt_cls in _registrars[identifier]:
        tgt = evt_cls._accept_with(target, identifier)
        if tgt is not None:
            return _EventKey(target, identifier, fn, tgt)
    else:
        raise exc.InvalidRequestError(
            "No such event '%s' for target '%s'" % (identifier, target)
        )


def listen(
    target: Any, identifier: str, fn: Callable[..., Any], *args: Any, **kw: Any
) -> None:
    """Register a listener function for the given target.

    The :func:`.listen` function is part of the primary interface for the
    SQLAlchemy event system, documented at :ref:`event_toplevel`.

    e.g.::

        from sqlalchemy import event
        from sqlalchemy.schema import UniqueConstraint


        def unique_constraint_name(const, table):
            const.name = "uq_%s_%s" % (table.name, list(const.columns)[0].name)


        event.listen(
            UniqueConstraint, "after_parent_attach", unique_constraint_name
        )

    :param bool insert: The default behavior for event handlers is to append
      the decorated user defined function to an internal list of registered
      event listeners upon discovery. If a user registers a function with
      ``insert=True``, SQLAlchemy will insert (prepend) the function to the
      internal list upon discovery. This feature is not typically used or
      recommended by the SQLAlchemy maintainers, but is provided to ensure
      certain user defined functions can run before others, such as when
      :ref:`Changing the sql_mode in MySQL <mysql_sql_mode>`.

    :param bool named: When using named argument passing, the names listed in
      the function argument specification will be used as keys in the
      dictionary.
      See :ref:`event_named_argument_styles`.

    :param bool once: Private/Internal API usage. Deprecated.  This parameter
      would provide that an event function would run only once per given
      target. It does not however imply automatic de-registration of the
      listener function; associating an arbitrarily high number of listeners
      without explicitly removing them will cause memory to grow unbounded even
      if ``once=True`` is specified.

    :param bool propagate: The ``propagate`` kwarg is available when working
      with ORM instrumentation and mapping events.
      See :class:`_ormevent.MapperEvents` and
      :meth:`_ormevent.MapperEvents.before_mapper_configured` for examples.

    :param bool retval: This flag applies only to specific event listeners,
      each of which includes documentation explaining when it should be used.
      By default, no listener ever requires a return value.
      However, some listeners do support special behaviors for return values,
      and include in their documentation that the ``retval=True`` flag is
      necessary for a return value to be processed.

      Event listener suites that make use of :paramref:`_event.listen.retval`
      include :class:`_events.ConnectionEvents` and
      :class:`_ormevent.AttributeEvents`.

    :param str on_failure: failure policy for this listener, selecting what
      happens when the listener raises an exception while an event is being
      dispatched.

      * ``"raise"`` (the default): the exception propagates to the caller and
        the listener chain is interrupted. This is the historical behavior
        and is unchanged when :paramref:`_event.listen.on_failure` is not
        specified.

      * ``"skip"``: the exception is caught, a failure record is retained
        (see :func:`.listener_failures` and
        :attr:`ListenerEntry.failures`), the failing listener is skipped and
        dispatch continues with the listeners registered after it.

      * ``"log"``: like ``"skip"``, but a warning including the exception is
        also logged to the ``sqlalchemy.event`` logger.

      Failure records contain the event name, a listener identifier and an
      exception summary. Failure bookkeeping is thread-safe under concurrent
      dispatch.

      For events that consume listener return values (those registered with
      ``retval=True``), a skipped listener contributes ``None`` and the
      dispatch proceeds with the remaining listeners.

      .. versionadded:: 2.1.5

    .. note::

        The :func:`.listen` function cannot be called at the same time
        that the target event is being run.   This has implications
        for thread safety, and also means an event cannot be added
        from inside the listener function for itself.  The list of
        events to be run are present inside of a mutable collection
        that can't be changed during iteration.

        Event registration and removal is not intended to be a "high
        velocity" operation; it is a configurational operation.  For
        systems that need to quickly associate and deassociate with
        events at high scale, use a mutable structure that is handled
        from inside of a single listener.

    .. seealso::

        :func:`.listens_for`

        :func:`.remove`

    """

    _event_key(target, identifier, fn).listen(*args, **kw)


def listens_for(
    target: Any, identifier: str, *args: Any, **kw: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a function as a listener for the given target + identifier.

    The :func:`.listens_for` decorator is part of the primary interface for the
    SQLAlchemy event system, documented at :ref:`event_toplevel`.

    This function generally shares the same kwargs as :func:`.listen`.

    e.g.::

        from sqlalchemy import event
        from sqlalchemy.schema import UniqueConstraint


        @event.listens_for(UniqueConstraint, "after_parent_attach")
        def unique_constraint_name(const, table):
            const.name = "uq_%s_%s" % (table.name, list(const.columns)[0].name)

    A given function can also be invoked for only the first invocation
    of the event using the ``once`` argument::

        @event.listens_for(Mapper, "before_configure", once=True)
        def on_config():
            do_config()

    .. warning:: The ``once`` argument does not imply automatic de-registration
       of the listener function after it has been invoked a first time; a
       listener entry will remain associated with the target object.
       Associating an arbitrarily high number of listeners without explicitly
       removing them will cause memory to grow unbounded even if ``once=True``
       is specified.

    .. seealso::

        :func:`.listen` - general description of event listening

    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        listen(target, identifier, fn, *args, **kw)
        return fn

    return decorate


def remove(target: Any, identifier: str, fn: Callable[..., Any]) -> None:
    """Remove an event listener.

    The arguments here should match exactly those which were sent to
    :func:`.listen`; all the event registration which proceeded as a result
    of this call will be reverted by calling :func:`.remove` with the same
    arguments.

    e.g.::

        # if a function was registered like this...
        @event.listens_for(SomeMappedClass, "before_insert", propagate=True)
        def my_listener_function(*arg):
            pass


        # ... it's removed like this
        event.remove(SomeMappedClass, "before_insert", my_listener_function)

    Above, the listener function associated with ``SomeMappedClass`` was also
    propagated to subclasses of ``SomeMappedClass``; the :func:`.remove`
    function will revert all of these operations.

    .. note::

        The :func:`.remove` function cannot be called at the same time
        that the target event is being run.   This has implications
        for thread safety, and also means an event cannot be removed
        from inside the listener function for itself.  The list of
        events to be run are present inside of a mutable collection
        that can't be changed during iteration.

        Event registration and removal is not intended to be a "high
        velocity" operation; it is a configurational operation.  For
        systems that need to quickly associate and deassociate with
        events at high scale, use a mutable structure that is handled
        from inside of a single listener.

    .. seealso::

        :func:`.listen`

    """
    _event_key(target, identifier, fn).remove()


def contains(target: Any, identifier: str, fn: Callable[..., Any]) -> bool:
    """Return True if the given target/ident/fn is set up to listen."""

    return _event_key(target, identifier, fn).contains()


def registered_listeners(target: Any) -> ListenerRegistryView:
    """Return a read-only view of the listeners registered on ``target``.

    The returned :class:`ListenerRegistryView` maps event names to tuples of
    :class:`ListenerEntry` in registration (dispatch) order.  Each entry
    exposes the original listener function, its failure policy, the
    cumulative number of recorded failures and the recent failure records.

    e.g.::

        view = event.registered_listeners(some_engine)
        for entry in view.get("before_cursor_execute", ()):
            print(entry.listener_name, entry.on_failure, entry.fail_count)

    .. versionadded:: 2.1.5

    .. seealso::

        :func:`.listener_failures`

        :func:`.listen`

    """

    dispatch = target.dispatch

    try:
        event_names = dispatch._events.dispatch._event_names
    except AttributeError as attr_err:
        raise exc.InvalidRequestError(
            "target %r does not provide an event dispatch registry" % (target,)
        ) from attr_err

    entries: Dict[str, Tuple[ListenerEntry, ...]] = {}
    for identifier in event_names:
        storage = [
            _entry_for_storage(identifier, owner, stored_fn)
            for owner, stored_fn in _iter_target_storage(
                target, dispatch, identifier
            )
        ]
        if storage:
            entries[identifier] = tuple(storage)

    return ListenerRegistryView(target, entries)


def listener_failures(
    target: Any, identifier: Optional[str] = None
) -> Tuple[ListenerFailure, ...]:
    """Return retained listener failure records for ``target``.

    Returns a tuple of :class:`ListenerFailure` recorded by listeners
    registered with the ``"skip"`` or ``"log"`` failure policy, in
    registration order.  If ``identifier`` is given, only failures for that
    event are returned.

    Records are bounded per listener, while the cumulative failure count
    remains available via :func:`.registered_listeners`.

    .. versionadded:: 2.1.5

    """

    view = registered_listeners(target)

    if identifier is None:
        return view.failures

    if identifier not in target.dispatch._events.dispatch._event_names:
        raise exc.InvalidRequestError(
            "No such event '%s' for target '%s'" % (identifier, target)
        )

    return tuple(
        failure
        for entry in view.get(identifier, ())
        for failure in entry.failures
    )
