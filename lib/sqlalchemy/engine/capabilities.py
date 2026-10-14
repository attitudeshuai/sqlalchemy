# engine/capabilities.py
# Copyright (C) 2005-2026 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Runtime capability negotiation for dialects.

Dialects historically declare their capabilities as hardcoded booleans
(and occasionally tuples) on the dialect class.  A handful of them are
recomputed from :attr:`.Dialect.server_version_info` inside
:meth:`.Dialect.initialize`, which runs exactly once, on the first
physical connection.  The server actually reached at runtime may disagree
with those declarations - a feature gate may be compiled out, a
compatibility level may be in effect, a fork or a load balancer may hand
the pool connections to different servers - and execution branches in
different places also tended to re-derive the same version comparison on
their own.

This module provides a single, queryable registry for capability
judgement:

* dialects :meth:`~.CapabilityRegistry.declare` capabilities with their
  class-level default and may :meth:`~.CapabilityRegistry.add_version_rule`
  declare how a capability varies over *version ranges*;

* dialects may additionally register a runtime *probe* - a callable that
  talks to the actual server - via
  :meth:`~.CapabilityRegistry.register_probe`;

* a :class:`.CapabilityNegotiation` is computed against what a given
  physical connection actually reports, cached per server identity, and
  stamped onto the connection record.  When a probe succeeds, its value
  takes precedence over the declared / version-derived value; the
  disagreement is both logged and retained on the resulting
  :class:`.CapabilityRecord`;

* a probe that fails (or explicitly reports "undetermined") degrades to
  the deterministic declared / version-derived baseline - never to an
  exception - and the failure is recorded on the negotiation;

* negotiation runs for every newly established physical connection, so a
  reconnect that reaches a different server renegotiates against that
  new server;

* code paths which never negotiate (e.g. a dialect used only to compile
  statements) keep observing the class-level declarations unchanged.

The registry is exposed on every dialect as
:attr:`.Dialect.capabilities`::

    if connection.dialect.capabilities.supports(
        "supports_identity_columns", connection
    ):
        ...

    # boolean-free capabilities work as well:
    style = connection.dialect.capabilities.get_value(
        "aggregate_order_by_style", connection
    )

.. versionadded:: 2.1
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import enum
import logging
from types import MappingProxyType
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
import weakref

logger = logging.getLogger(__name__)


class CapabilitySource(enum.Enum):
    """Describe where the effective value of a negotiated capability
    came from.

    .. versionadded:: 2.1

    """

    DECLARED = "DECLARED"
    """the class-level declared value is in effect - either no version
    rule matched (this includes the case where the server version is
    unknown) or no probe was registered."""

    VERSION_RANGE = "VERSION_RANGE"
    """a dialect-declared version range rule matched the observed server
    version."""

    PROBED = "PROBED"
    """a runtime probe against the actual server produced the effective
    value.  :attr:`.CapabilityRecord.conflict` indicates whether the
    probe disagreed with the declared / version-derived baseline."""

    PROBE_FAILED = "PROBE_FAILED"
    """a runtime probe was attempted but failed, or explicitly reported
    the capability as undetermined.  The deterministic declared /
    version-derived baseline is in effect, and
    :attr:`.CapabilityRecord.probe_error` describes the failure."""


class CapabilityUndetermined(Exception):
    """Raise from a capability probe to report that support for the
    capability could not be determined at runtime.

    Raising this exception is equivalent to returning
    :data:`.CAPABILITY_UNDETERMINED`: negotiation records the probe as
    failed and deterministically falls back to the declared /
    version-derived baseline.

    .. versionadded:: 2.1

    """


class _CapabilityUndeterminedToken:
    __slots__ = ()

    def __repr__(self) -> str:
        return "CAPABILITY_UNDETERMINED"


CAPABILITY_UNDETERMINED = _CapabilityUndeterminedToken()
"""Sentinel a capability probe may return to indicate that support for
the capability could not be determined at runtime.

Equivalent to raising :class:`.CapabilityUndetermined`.

.. versionadded:: 2.1

"""


class CapabilityRule(NamedTuple):
    """Declare the value of a capability for a range of server versions.

    ``min_version`` is inclusive and ``max_version`` is exclusive, so
    that ``CapabilityRule((10,), None, True)`` models "supported on
    version 10 and later".  Rules are evaluated in declaration order and
    the first rule whose range contains the observed server version
    wins; if no rule matches, the class-level declared default is used.

    When the server version is unknown, no rule ever matches.

    .. versionadded:: 2.1

    """

    min_version: Optional[Tuple[Any, ...]]
    """minimum server version for which this rule applies,
    inclusive; ``None`` means no lower bound."""

    max_version: Optional[Tuple[Any, ...]]
    """server version at which this rule stops applying,
    exclusive; ``None`` means no upper bound."""

    value: Any
    """the capability value within the version range."""

    @classmethod
    def since(
        cls, version: Tuple[Any, ...], value: Any = True
    ) -> "CapabilityRule":
        """Build a rule applying ``value`` at ``version`` and later."""
        return cls(version, None, value)

    @classmethod
    def below(
        cls, version: Tuple[Any, ...], value: Any = True
    ) -> "CapabilityRule":
        """Build a rule applying ``value`` below ``version``."""
        return cls(None, version, value)

    @classmethod
    def between(
        cls,
        min_version: Tuple[Any, ...],
        max_version: Tuple[Any, ...],
        value: Any = True,
    ) -> "CapabilityRule":
        """Build a rule applying ``value`` from ``min_version`` (inclusive)
        up to ``max_version`` (exclusive)."""
        return cls(min_version, max_version, value)

    def matches(self, version: Tuple[Any, ...]) -> bool:
        """Return True if the rule's range contains ``version``.

        A version tuple shorter than a boundary compares the way plain
        Python tuple comparison dictates (e.g. ``(10,) >= (10, 2)`` is
        ``False`` and ``(11,) >= (10, 2)`` is ``True``).  Versions whose
        elements cannot be compared against the boundary (mixed strings
        and integers) deterministically do not match.

        """
        if self.min_version is not None and not _version_at_least(
            version, self.min_version
        ):
            return False
        if self.max_version is not None and not _version_below(
            version, self.max_version
        ):
            return False
        return True


def _version_at_least(
    version: Tuple[Any, ...], boundary: Tuple[Any, ...]
) -> bool:
    try:
        return bool(version >= boundary)
    except TypeError:
        # mixed int/str version elements - cannot compare
        return False


def _version_below(
    version: Tuple[Any, ...], boundary: Tuple[Any, ...]
) -> bool:
    try:
        return bool(version < boundary)
    except TypeError:
        return False


_UNSET = object()


class CapabilityDeclaration(NamedTuple):
    """Declarative description of a capability, used by dialect classes.

    Dialects expose a sequence of these as
    :attr:`.DefaultDialect.capability_declarations`; they may also be
    added imperatively to a live registry using
    :meth:`.CapabilityRegistry.declare` and
    :meth:`.CapabilityRegistry.add_version_rule`.

    ``default`` defaults to the current value of the dialect attribute
    named ``name`` (or ``attribute``); it must be provided explicitly
    for capabilities that have no backing dialect attribute.

    ``attribute`` controls whether negotiation writes the effective
    value back onto a dialect attribute so that pre-existing code paths
    keep observing it.  It defaults to ``name`` when such an attribute
    exists and to ``None`` otherwise; pass ``None`` explicitly to
    disable write-back.

    .. versionadded:: 2.1

    """

    name: str
    rules: Sequence[CapabilityRule] = ()
    probe: Optional[Callable[[Any], Any]] = None
    default: Any = _UNSET
    attribute: Any = _UNSET
    description: Optional[str] = None


@dataclass(frozen=True)
class CapabilityRecord:
    """The outcome of negotiating one capability for one server.

    .. versionadded:: 2.1

    """

    name: str
    """name of the capability."""

    value: Any
    """the effective value - what execution branches must use."""

    declared_value: Any
    """the class-level declared value (before version rules and
    probes)."""

    baseline_value: Any
    """the value in effect before a probe ran: the declared value or the
    value of the matching version range rule."""

    source: CapabilitySource
    """where :attr:`value` came from."""

    server_version_info: Optional[Tuple[Any, ...]]
    """the server version observed for this negotiation, or ``None`` if
    it could not be determined."""

    rule: Optional[CapabilityRule] = None
    """the version range rule that matched, if any."""

    probe_error: Optional[str] = None
    """when :attr:`source` is :attr:`.CapabilitySource.PROBE_FAILED`, a
    description of the failure."""

    conflict: bool = False
    """``True`` when a successful probe produced a value different from
    the declared / version-derived baseline.  The probe value is in
    effect."""


@dataclass(frozen=True)
class CapabilityNegotiation:
    """The capabilities negotiated against one observed server.

    Returned by :meth:`.CapabilityRegistry.negotiate` and cached per
    server identity.  Iterating yields :class:`.CapabilityRecord`
    objects.

    .. versionadded:: 2.1

    """

    server_version_info: Optional[Tuple[Any, ...]]
    records: Mapping[str, CapabilityRecord]
    detection_error: Optional[str] = None

    def value_of(self, name: str) -> Any:
        """Return the effective value of capability ``name``."""
        return self.records[name].value

    def supports(self, name: str) -> bool:
        """Return the truthiness of capability ``name``."""
        return bool(self.records[name].value)

    def record(self, name: str) -> CapabilityRecord:
        """Return the full :class:`.CapabilityRecord` for ``name``."""
        return self.records[name]

    def __iter__(self) -> Iterator[CapabilityRecord]:
        return iter(self.records.values())

    def __len__(self) -> int:
        return len(self.records)


@dataclass
class _CapabilitySpec:
    name: str
    default: Any
    attribute: Optional[str]
    description: Optional[str] = None
    rules: list[CapabilityRule] = field(default_factory=list)
    probe: Optional[Callable[[Any], Any]] = None


@dataclass
class _RecordState:
    # the negotiation in effect for one pool connection record, along
    # with the DBAPI connection that was observed when stamping it;
    # comparing the two lets a reconnect on a recycled record trigger a
    # new negotiation without mutating the user-visible record.info
    dbapi_connection: Any
    negotiation: CapabilityNegotiation


def _connection_record(connection: Any) -> Tuple[Any, Any]:
    """Return the pool ConnectionPoolEntry behind a SQLAlchemy Connection,
    or ``None`` for connections that are not pool-managed (e.g. a mock
    connection)."""

    pool_proxied = getattr(connection, "connection", None)
    if pool_proxied is None:
        return None, None
    return (
        getattr(pool_proxied, "_connection_record", None),
        getattr(pool_proxied, "dbapi_connection", None),
    )


class CapabilityRegistry:
    """Holds capability declarations and per-server negotiated results
    for one dialect.

    A registry is created automatically for every dialect and is
    available as :attr:`.Dialect.capabilities`.  Dialect authors populate
    it via :attr:`.DefaultDialect.capability_declarations` or the
    imperative :meth:`declare`, :meth:`add_version_rule` and
    :meth:`register_probe` methods; runtime code queries it via
    :meth:`supports`, :meth:`get_value` and :meth:`records`.

    .. versionadded:: 2.1

    """

    def __init__(
        self,
        dialect: Any,
        declarations: Sequence[CapabilityDeclaration] = (),
    ) -> None:
        self._dialect = dialect
        self._specs: dict[str, _CapabilitySpec] = {}
        # negotiated for servers actually observed; keyed by server
        # version tuple (None is a valid key)
        self._negotiations: dict[Any, CapabilityNegotiation] = {}
        # baseline-only negotiations, used for connection-less queries
        self._static_negotiations: dict[Any, CapabilityNegotiation] = {}
        # most recent negotiation against a real connection
        self._current: Optional[CapabilityNegotiation] = None
        # per pool connection record: the negotiation stamped for the
        # physical DBAPI connection currently held by that record.
        # keyed weakly so records discarded by the pool take their
        # negotiation state with them, and the user-visible
        # ConnectionPoolEntry.info dictionary is never mutated.
        self._record_state: weakref.WeakKeyDictionary[
            Any, _RecordState
        ] = weakref.WeakKeyDictionary()

        for declaration in declarations:
            self.add_declaration(declaration)

    def __repr__(self) -> str:
        return "%s(dialect=%r, capabilities=%r)" % (
            type(self).__name__,
            getattr(self._dialect, "dialect_description", self._dialect),
            sorted(self._specs),
        )

    # ------------------------------------------------------------------
    # declaration

    def add_declaration(self, declaration: CapabilityDeclaration) -> None:
        """Register a :class:`.CapabilityDeclaration`."""

        self.declare(
            declaration.name,
            default=declaration.default,
            attribute=declaration.attribute,
            description=declaration.description,
        )
        for rule in declaration.rules:
            self.add_version_rule(
                declaration.name,
                rule.value,
                min_version=rule.min_version,
                max_version=rule.max_version,
            )
        if declaration.probe is not None:
            self.register_probe(declaration.name, declaration.probe)

    def declare(
        self,
        name: str,
        default: Any = _UNSET,
        *,
        attribute: Any = _UNSET,
        description: Optional[str] = None,
    ) -> None:
        """Declare a capability and its class-level default.

        When ``default`` is omitted, the current value of the dialect
        attribute of the same name is captured, so that the existing
        class-level declaration remains the baseline.

        ``attribute`` selects the dialect attribute (if any) onto which
        negotiated values are written back; see
        :class:`.CapabilityDeclaration`.

        """

        if name in self._specs:
            raise ValueError(
                f"Capability {name!r} is already declared on dialect "
                f"{self._dialect.dialect_description!r}"
            )

        if default is _UNSET:
            default = getattr(self._dialect, name, _UNSET)
            if default is _UNSET:
                raise ValueError(
                    f"Capability {name!r} has no dialect attribute named "
                    f"{name!r}; an explicit default value is required"
                )

        if attribute is _UNSET:
            attribute = name if hasattr(self._dialect, name) else None

        self._specs[name] = _CapabilitySpec(
            name=name,
            default=default,
            attribute=attribute,
            description=description,
        )

    def add_version_rule(
        self,
        name: str,
        value: Any,
        *,
        min_version: Optional[Tuple[Any, ...]] = None,
        max_version: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        """Declare that capability ``name`` has ``value`` for servers
        whose version is ``>= min_version`` (inclusive) and
        ``< max_version`` (exclusive).

        Either bound may be ``None``.  Rules are evaluated in the order
        in which they are added.

        """

        self._require(name).rules.append(
            CapabilityRule(min_version, max_version, value)
        )

    def register_probe(
        self, name: str, probe: Callable[[Any], Any]
    ) -> None:
        """Register a runtime probe for capability ``name``.

        ``probe`` is passed the live SQLAlchemy Connection and must
        return the capability value observed on the actual server.  It
        may raise :class:`.CapabilityUndetermined` or return
        :data:`.CAPABILITY_UNDETERMINED` to report that support could
        not be determined; negotiation then deterministically falls back
        to the declared / version-derived baseline.  Any other
        exception is treated the same way and recorded.

        A successful probe takes precedence over the baseline; when its
        value differs from the baseline the disagreement is logged at
        warning level and retained on the resulting
        :class:`.CapabilityRecord`.

        """

        self._require(name).probe = probe

    def declared_value(self, name: str) -> Any:
        """Return the class-level declared default for ``name``."""

        return self._require(name).default

    @property
    def capability_names(self) -> Sequence[str]:
        """the names of all declared capabilities."""

        return tuple(sorted(self._specs))

    # ------------------------------------------------------------------
    # negotiation

    def negotiate(
        self,
        connection: Any = None,
        *,
        server_version_info: Any = _UNSET,
        force: bool = False,
    ) -> CapabilityNegotiation:
        """Negotiate capabilities against the server a connection sees.

        The result is cached per observed server identity, so repeated
        calls for the same server reuse probe results; a reconnect that
        reaches a server reporting a different identity produces a new
        negotiation.  Pass ``force=True`` to discard cached results and
        renegotiate unconditionally.

        This method never raises: a failure to observe the server or to
        run a probe degrades deterministically to the declared values
        and is recorded instead.

        """

        try:
            return self._negotiate(
                connection, server_version_info, force
            )
        except Exception as err:
            # negotiation must never break connection establishment;
            # fall back to whatever is deterministically declarable
            logger.warning(
                "Capability negotiation for dialect %s failed; falling "
                "back to class-level capability declarations: %r",
                getattr(
                    self._dialect, "dialect_description", self._dialect
                ),
                err,
                exc_info=True,
            )
            fallback = self._static_negotiation(
                getattr(self._dialect, "server_version_info", None)
            )
            if self._current is None:
                self._current = fallback
                self._write_back(fallback)
            self._stamp(connection, fallback)
            return fallback

    def _stamped(
        self, record: Any, dbapi_connection: Any
    ) -> Optional[CapabilityNegotiation]:
        state = self._record_state.get(record)
        if (
            state is not None
            and state.dbapi_connection is dbapi_connection
        ):
            return state.negotiation
        return None

    def _negotiate(
        self, connection: Any, server_version_info: Any, force: bool
    ) -> CapabilityNegotiation:
        record, dbapi_connection = _connection_record(connection)

        # reuse a negotiation already established for this exact
        # physical connection
        if record is not None and not force:
            stamped = self._stamped(record, dbapi_connection)
            if stamped is not None:
                return stamped

        version: Optional[Tuple[Any, ...]]
        detection_error: Optional[str] = None

        if server_version_info is not _UNSET:
            version = server_version_info
        elif connection is not None:
            version, detection_error = self._detect_server_version(connection)
        else:
            version = getattr(self._dialect, "server_version_info", None)

        if not force and version in self._negotiations:
            negotiation = self._negotiations[version]
        else:
            negotiation = self._build_negotiation(
                version, detection_error, connection
            )
            self._negotiations[version] = negotiation

        self._current = negotiation
        self._write_back(negotiation)
        self._stamp(connection, negotiation)

        return negotiation

    #: historical alias, kept readable at call sites that specifically
    #: mean "talk to the server again"
    renegotiate = negotiate

    def _detect_server_version(
        self, connection: Any
    ) -> Tuple[Optional[Tuple[Any, ...]], Optional[str]]:
        try:
            return self._dialect._get_server_version_info(connection), None
        except NotImplementedError:
            # dialect does not publish a server version
            return None, None
        except Exception as err:
            error = "%s: %s" % (type(err).__name__, err)
            logger.debug(
                "Server version detection failed for dialect %s; "
                "capabilities will use class-level declarations (%s)",
                getattr(
                    self._dialect, "dialect_description", self._dialect
                ),
                error,
            )
            return None, error

    def _build_negotiation(
        self,
        version: Optional[Tuple[Any, ...]],
        detection_error: Optional[str],
        connection: Any,
    ) -> CapabilityNegotiation:
        records: dict[str, CapabilityRecord] = {}

        for name, spec in self._specs.items():
            baseline, rule, source = self._baseline(spec, version)

            probe_error: Optional[str] = None
            conflict = False
            value = baseline

            if spec.probe is not None and connection is not None:
                value, source, probe_error, conflict = self._run_probe(
                    spec, connection, baseline, source
                )

            records[name] = CapabilityRecord(
                name=name,
                value=value,
                declared_value=spec.default,
                baseline_value=baseline,
                source=source,
                server_version_info=version,
                rule=rule,
                probe_error=probe_error,
                conflict=conflict,
            )

        return CapabilityNegotiation(
            server_version_info=version,
            records=MappingProxyType(records),
            detection_error=detection_error,
        )

    def _baseline(
        self,
        spec: _CapabilitySpec,
        version: Optional[Tuple[Any, ...]],
    ) -> Tuple[Any, Optional[CapabilityRule], CapabilitySource]:
        if version is not None:
            for rule in spec.rules:
                if rule.matches(version):
                    return rule.value, rule, CapabilitySource.VERSION_RANGE
        return spec.default, None, CapabilitySource.DECLARED

    def _run_probe(
        self,
        spec: _CapabilitySpec,
        connection: Any,
        baseline: Any,
        baseline_source: CapabilitySource,
    ) -> Tuple[Any, CapabilitySource, Optional[str], bool]:
        probe = spec.probe
        if probe is None:
            return baseline, baseline_source, None, False
        try:
            value = probe(connection)
        except CapabilityUndetermined as err:
            return (
                baseline,
                CapabilitySource.PROBE_FAILED,
                "undetermined: %s" % err,
                False,
            )
        except Exception as err:
            error = "%s: %s" % (type(err).__name__, err)
            logger.debug(
                "Capability probe for %r on dialect %s failed; using "
                "declared value %r (%s)",
                spec.name,
                getattr(
                    self._dialect, "dialect_description", self._dialect
                ),
                baseline,
                error,
            )
            return baseline, CapabilitySource.PROBE_FAILED, error, False

        if value is CAPABILITY_UNDETERMINED:
            return (
                baseline,
                CapabilitySource.PROBE_FAILED,
                "undetermined",
                False,
            )

        if value != baseline:
            logger.warning(
                "Capability %r observed on dialect %s (server version "
                "%s) is %r, which disagrees with the %s value %r; using "
                "the observed runtime value",
                spec.name,
                getattr(
                    self._dialect, "dialect_description", self._dialect
                ),
                getattr(self._dialect, "server_version_info", None),
                value,
                baseline_source.value,
                baseline,
            )
            return value, CapabilitySource.PROBED, None, True

        return value, CapabilitySource.PROBED, None, False

    def _write_back(self, negotiation: CapabilityNegotiation) -> None:
        for name, spec in self._specs.items():
            attribute = spec.attribute
            if attribute is None:
                continue
            if not hasattr(self._dialect, attribute):
                continue
            record = negotiation.records[name]
            # only write when the value would actually change, to avoid
            # turning class attributes into instance attributes needlessly
            if getattr(self._dialect, attribute) != record.value:
                setattr(self._dialect, attribute, record.value)

    def _stamp(
        self, connection: Any, negotiation: CapabilityNegotiation
    ) -> None:
        if connection is None:
            return
        record, dbapi_connection = _connection_record(connection)
        if record is None:
            return
        try:
            self._record_state[record] = _RecordState(
                dbapi_connection, negotiation
            )
        except TypeError:
            # the record does not support weak references; it simply
            # won't carry a stamped negotiation and queries fall back to
            # dialect-current / declared values
            logger.debug(
                "Could not stamp capability negotiation on connection "
                "record %r",
                record,
            )

    # ------------------------------------------------------------------
    # connection lifecycle hooks

    def handle_connect(self, connection: Any) -> bool:
        """Negotiate for a newly established physical connection.

        Installed as a pool ``connect`` listener, this runs for every
        new DBAPI connection - including connections created after
        invalidation - and renegotiates whenever the new connection
        reports a different server identity.  A connection already
        stamped during :meth:`.Dialect.initialize` is left untouched.

        Dialects which declare no capabilities at all are short
        circuited so that the connect event performs no extra work and
        no extra server round trip.  Returns True if a negotiation
        actually took place.

        """

        if not self._specs:
            return False

        record, dbapi_connection = _connection_record(connection)
        if record is not None:
            if self._stamped(record, dbapi_connection) is not None:
                return False

        self.negotiate(connection)
        return True

    def invalidate(self) -> None:
        """Forget all cached negotiations.

        The next capability query renegotiates against the observed
        server.  This is normally unnecessary - new physical
        connections are negotiated automatically - but is useful for
        tests and for dialects that know their runtime capabilities have
        changed on the live connection.

        """

        self._negotiations.clear()
        self._static_negotiations.clear()
        self._record_state.clear()
        self._current = None

    # ------------------------------------------------------------------
    # queries

    def supports(self, name: str, connection: Any = None) -> bool:
        """Return True if capability ``name`` is supported (truthy).

        With a ``connection`` argument the answer reflects the
        negotiation for the server that physical connection sees;
        without one, the most recently negotiated answer is used when
        available, falling back to the class-level declaration (which
        keeps compile-only and un-negotiated dialects behaving exactly
        as before).

        """

        return bool(self.get_value(name, connection))

    def get_value(self, name: str, connection: Any = None) -> Any:
        """Return the effective value of capability ``name``.

        Capability values are not restricted to booleans; tuples,
        enums and other values are supported.

        """

        return self._negotiation_for(connection).value_of(name)

    def get_record(
        self, name: str, connection: Any = None
    ) -> CapabilityRecord:
        """Return the :class:`.CapabilityRecord` for ``name``, including
        its provenance."""

        return self._negotiation_for(connection).record(name)

    def records(
        self, connection: Any = None,
    ) -> Mapping[str, CapabilityRecord]:
        """Return all capability records in effect for ``connection``
        (or for the dialect's current / declared state when no
        connection is given)."""

        return self._negotiation_for(connection).records

    def _negotiation_for(self, connection: Any) -> CapabilityNegotiation:
        if connection is not None:
            record, dbapi_connection = _connection_record(connection)
            if record is not None:
                negotiation = self._stamped(record, dbapi_connection)
                if negotiation is not None:
                    return negotiation
                # a real, pool-managed connection that has not been
                # negotiated yet: negotiate lazily on its behalf
                if dbapi_connection is not None:
                    return self.negotiate(connection)

        if self._current is not None:
            return self._current
        return self._static_negotiation(
            getattr(self._dialect, "server_version_info", None)
        )

    def _static_negotiation(
        self, version: Optional[Tuple[Any, ...]]
    ) -> CapabilityNegotiation:
        if version in self._static_negotiations:
            return self._static_negotiations[version]

        records: dict[str, CapabilityRecord] = {}
        for name, spec in self._specs.items():
            baseline, rule, source = self._baseline(spec, version)
            records[name] = CapabilityRecord(
                name=name,
                value=baseline,
                declared_value=spec.default,
                baseline_value=baseline,
                source=source,
                server_version_info=version,
                rule=rule,
            )

        negotiation = CapabilityNegotiation(
            server_version_info=version,
            records=MappingProxyType(records),
        )
        self._static_negotiations[version] = negotiation
        return negotiation

    # ------------------------------------------------------------------

    def _require(self, name: str) -> _CapabilitySpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(
                "Capability %r is not declared on dialect %r; declared "
                "capabilities are: %s"
                % (
                    name,
                    getattr(
                        self._dialect,
                        "dialect_description",
                        self._dialect,
                    ),
                    ", ".join(sorted(self._specs)) or "(none)",
                )
            ) from None
