# engine/test_capabilities.py
# Copyright (C) 2005-2026 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Tests for runtime dialect capability negotiation."""

from __future__ import annotations

import logging

from sqlalchemy import create_engine
from sqlalchemy import create_mock_engine
from sqlalchemy import inspect
from sqlalchemy.engine import capabilities as capabilities_module
from sqlalchemy.engine import CapabilityDeclaration
from sqlalchemy.engine import CapabilityRule
from sqlalchemy.engine import CapabilitySource
from sqlalchemy.engine import CapabilityUndetermined
from sqlalchemy.engine.default import DefaultDialect
from sqlalchemy.testing import assert_raises
from sqlalchemy.testing import eq_
from sqlalchemy.testing import fixtures
from sqlalchemy.testing import is_false
from sqlalchemy.testing import is_true


class _DeclaredDialect(DefaultDialect):
    name = "capability_test"
    driver = "testdriver"

    some_flag = True
    other_flag = False
    some_value = "default"


def _registry(dialect=None):
    if dialect is None:
        dialect = _DeclaredDialect()
    return dialect, dialect.capabilities


class CapabilityRuleTest(fixtures.TestBase):
    def test_rule_since(self):
        rule = CapabilityRule.since((10,), True)
        is_false(rule.matches((9, 9)))
        is_true(rule.matches((10,)))
        is_true(rule.matches((10, 0)))
        is_true(rule.matches((11, 2)))

    def test_rule_below(self):
        rule = CapabilityRule.below((9, 2), False)
        is_true(rule.matches((9, 1)))
        is_true(rule.matches((9,)))
        is_false(rule.matches((9, 2)))
        is_false(rule.matches((10,)))

    def test_rule_between(self):
        rule = CapabilityRule.between((10,), (12,), "mid")
        is_false(rule.matches((9, 99)))
        is_true(rule.matches((10,)))
        is_true(rule.matches((11, 99)))
        is_false(rule.matches((12,)))

    def test_rule_order_first_match_wins(self):
        dialect, reg = _registry()
        reg.declare("some_flag")
        reg.add_version_rule("some_flag", "first", min_version=(8,))
        reg.add_version_rule("some_flag", "second", min_version=(10,))
        eq_(
            reg.negotiate(None, server_version_info=(9,)).value_of(
                "some_flag"
            ),
            "first",
        )
        eq_(
            reg.negotiate(None, server_version_info=(11,)).value_of(
                "some_flag"
            ),
            "first",
        )

    def test_rule_uncomparable_version_does_not_match(self):
        # mixed int / str version elements cannot be ordered against a
        # boundary that reaches the mixed element; such rules
        # deterministically do not match
        rule = CapabilityRule.since((10, 0), True)
        is_false(rule.matches((10, "beta", 1)))
        is_true(CapabilityRule.below((9,), False).matches((8, "x")))


class CapabilityRegistryDeclarationTest(fixtures.TestBase):
    def test_declared_default_captured_from_attribute(self):
        dialect, reg = _registry()
        reg.declare("some_flag")
        is_true(reg.declared_value("some_flag"))
        # un-negotiated query keeps the class-level declaration
        is_true(reg.supports("some_flag"))
        rec = reg.get_record("some_flag")
        eq_(rec.source, CapabilitySource.DECLARED)
        eq_(rec.value, True)

    def test_explicit_default_required_without_attribute(self):
        dialect, reg = _registry()
        assert_raises(ValueError, reg.declare, "no_such_attribute")
        # ...but an explicit default works, with write-back disabled
        reg.declare("no_such_attribute", default="x", attribute=None)
        eq_(reg.get_value("no_such_attribute"), "x")

    def test_duplicate_declaration_raises(self):
        dialect, reg = _registry()
        reg.declare("some_flag")
        assert_raises(ValueError, reg.declare, "some_flag")

    def test_unknown_capability_raises(self):
        dialect, reg = _registry()
        try:
            reg.supports("nope")
            assert False
        except KeyError as err:
            assert "nope" in str(err)

    def test_declaration_via_capability_declaration(self):
        class _Dialect(_DeclaredDialect):
            capability_declarations = (
                CapabilityDeclaration(
                    "other_flag",
                    (CapabilityRule.since((10,), True),),
                ),
            )

        d2 = _Dialect()
        is_false(d2.capabilities.supports("other_flag"))
        d2.capabilities.negotiate(None, server_version_info=(10,))
        is_true(d2.capabilities.supports("other_flag"))

    def test_attribute_writeback_disabled(self):
        dialect, reg = _registry()
        reg.declare("some_flag", default=True, attribute=None)
        reg.add_version_rule("some_flag", False, max_version=(10,))
        dialect.some_flag = True
        reg.negotiate(None, server_version_info=(8,))
        # no backing attribute -> no write even though name coincides
        eq_(dialect.some_flag, True)


class CapabilityNegotiationTest(fixtures.TestBase):
    def _versioned(self):
        dialect, reg = _registry()
        reg.declare("other_flag")
        reg.declare("some_value")
        reg.add_version_rule(
            "other_flag", True, min_version=(10,), max_version=(15,)
        )
        reg.add_version_rule("some_value", "new", min_version=(12,))
        return dialect, reg

    def test_version_range_applied_and_recorded(self):
        dialect, reg = self._versioned()

        neg = reg.negotiate(None, server_version_info=(11, 2))
        is_true(neg.supports("other_flag"))
        rec = neg.record("other_flag")
        eq_(rec.source, CapabilitySource.VERSION_RANGE)
        eq_(rec.rule, CapabilityRule((10,), (15,), True))
        eq_(rec.declared_value, False)
        eq_(rec.server_version_info, (11, 2))

        # outside the range: class default, declared source
        neg = reg.negotiate(None, server_version_info=(16,))
        is_false(neg.supports("other_flag"))
        eq_(neg.record("other_flag").source, CapabilitySource.DECLARED)

        # non-boolean capability values work
        eq_(
            reg.negotiate(None, server_version_info=(11,)).value_of(
                "some_value"
            ),
            "default",
        )
        eq_(
            reg.negotiate(None, server_version_info=(12,)).value_of(
                "some_value"
            ),
            "new",
        )

    def test_unknown_version_uses_declared(self):
        dialect, reg = self._versioned()
        neg = reg.negotiate(None, server_version_info=None)
        is_false(neg.supports("other_flag"))
        for rec in neg:
            eq_(rec.source, CapabilitySource.DECLARED)

    def test_writeback_updates_dialect_attributes(self):
        dialect, reg = self._versioned()
        is_false(dialect.other_flag)

        reg.negotiate(None, server_version_info=(11,))
        is_true(dialect.other_flag)

        # re-negotiating to an older server downgrades again
        reg.negotiate(None, server_version_info=(9,), force=True)
        is_false(dialect.other_flag)

    def test_current_and_static_queries(self):
        dialect, reg = self._versioned()

        # before any negotiation, static declaration is used
        is_false(reg.supports("other_flag"))

        reg.negotiate(None, server_version_info=(11,))
        # no connection: most recent negotiation
        is_true(reg.supports("other_flag"))

    def test_negotiation_cached_per_server_identity(self):
        dialect, reg = self._versioned()
        calls = []

        def probe(connection):
            calls.append(connection)
            return True

        reg.register_probe("other_flag", probe)

        fake = object()
        n1 = reg.negotiate(fake, server_version_info=(11,))
        n2 = reg.negotiate(fake, server_version_info=(11,))
        eq_(len(calls), 1)
        assert n1 is n2

        # a different server identity negotiates again
        reg.negotiate(fake, server_version_info=(11, 5))
        eq_(len(calls), 2)

        # force re-runs probes unconditionally
        reg.negotiate(fake, server_version_info=(11, 5), force=True)
        eq_(len(calls), 3)

    def test_per_connection_negotiation_scoped_to_connection(self):
        dialect, reg = self._versioned()

        class FakeRec:
            def __init__(self):
                self.info = {}

        class FakeFairy:
            def __init__(self, rec):
                self._connection_record = rec
                self.dbapi_connection = object()

        class FakeConn:
            def __init__(self, rec):
                self.connection = FakeFairy(rec)

        conn_a = FakeConn(FakeRec())
        conn_b = FakeConn(FakeRec())

        reg.negotiate(conn_a, server_version_info=(9,))
        reg.negotiate(conn_b, server_version_info=(13,))

        is_false(reg.supports("other_flag", conn_a))
        is_true(reg.supports("other_flag", conn_b))
        # dialect-current reflects the most recent negotiation
        is_true(reg.supports("other_flag"))

    def test_probe_agrees_source_probed(self):
        dialect, reg = self._versioned()
        reg.register_probe("other_flag", lambda conn: True)
        # 11.x is within the 10-15 range, so the probe agrees with it
        rec = reg.negotiate(object(), server_version_info=(11,)).record(
            "other_flag"
        )
        eq_(rec.source, CapabilitySource.PROBED)
        is_false(rec.conflict)
        is_true(rec.value)

    def test_probe_conflict_wins_and_is_recorded(self, caplog):
        dialect, reg = self._versioned()
        reg.register_probe("other_flag", lambda conn: True)

        caplog.set_level(
            logging.WARNING,
            logger="sqlalchemy.engine.capabilities",
        )

        neg = reg.negotiate(object(), server_version_info=(9,))
        # probe wins over the declared False
        is_true(neg.value_of("other_flag"))
        rec = neg.record("other_flag")
        eq_(rec.source, CapabilitySource.PROBED)
        is_true(rec.conflict)
        eq_(rec.baseline_value, False)
        eq_(rec.declared_value, False)

        # disagreement is logged and identifies the capability
        warning = "".join(
            rec.message for rec in caplog.records if rec.levelname == "WARNING"
        )
        assert "other_flag" in warning
        assert "disagrees" in warning

    def test_probe_exception_deterministic_fallback(self):
        dialect, reg = self._versioned()

        def broken(connection):
            raise RuntimeError("server says no")

        reg.register_probe("other_flag", broken)
        neg = reg.negotiate(object(), server_version_info=(11,))
        rec = neg.record("other_flag")
        eq_(rec.source, CapabilitySource.PROBE_FAILED)
        is_true(rec.value)  # version-range baseline still in effect
        assert "RuntimeError" in rec.probe_error

    def test_probe_undetermined_sentinel_and_exception(self):
        dialect, reg = self._versioned()

        reg.register_probe(
            "other_flag",
            lambda conn: capabilities_module.CAPABILITY_UNDETERMINED,
        )
        rec = reg.negotiate(object(), server_version_info=(11,)).record(
            "other_flag"
        )
        eq_(rec.source, CapabilitySource.PROBE_FAILED)
        is_true(rec.value)

        def undetermined(connection):
            raise CapabilityUndetermined("feature gate not visible")

        reg.register_probe("other_flag", undetermined)
        rec = reg.negotiate(
            object(), server_version_info=(11,), force=True
        ).record("other_flag")
        eq_(rec.source, CapabilitySource.PROBE_FAILED)
        assert "feature gate" in rec.probe_error

    def test_version_detection_failure_degrades_to_declared(self):
        dialect = _DeclaredDialect()

        def boom(connection):
            raise NotImplementedError()

        dialect._get_server_version_info = boom
        reg = dialect.capabilities
        reg.declare("other_flag")
        reg.add_version_rule("other_flag", True, min_version=(10,))

        fake = _FakePoolConnection()
        neg = reg.negotiate(fake)
        is_false(neg.supports("other_flag"))
        eq_(neg.server_version_info, None)
        eq_(neg.record("other_flag").source, CapabilitySource.DECLARED)

    def test_unexpected_version_detection_error_recorded(self):
        dialect = _DeclaredDialect()

        def boom(connection):
            raise RuntimeError("connection reset")

        dialect._get_server_version_info = boom
        reg = dialect.capabilities
        reg.declare("other_flag")

        neg = reg.negotiate(_FakePoolConnection())
        eq_(neg.server_version_info, None)
        assert "connection reset" in neg.detection_error
        # still a deterministic declared answer, no exception escaped
        is_false(neg.supports("other_flag"))

    def test_records_exposes_full_provenance(self):
        dialect, reg = self._versioned()
        reg.declare("some_flag")
        reg.register_probe("other_flag", lambda conn: True)
        reg.negotiate(object(), server_version_info=(11,))
        records = reg.records()
        eq_(set(records), {"some_flag", "other_flag", "some_value"})
        eq_(records["other_flag"].source, CapabilitySource.PROBED)

    def test_invalidate_forgets_cached_negotiations(self):
        dialect, reg = self._versioned()
        reg.negotiate(None, server_version_info=(11,))
        is_true(reg.supports("other_flag"))
        reg.invalidate()
        is_false(reg.supports("other_flag"))


class _FakeRec:
    def __init__(self):
        self.info = {}


class _FakeFairy:
    def __init__(self):
        self._connection_record = _FakeRec()
        self.dbapi_connection = object()


class _FakePoolConnection:
    def __init__(self):
        self.connection = _FakeFairy()


class CapabilityEngineTest(fixtures.TestBase):
    def test_sqlite_connection_scoped_negotiation(self):
        engine = create_engine("sqlite://")
        with engine.connect() as conn:
            is_true(
                conn.supports_capability("supports_table_xinfo_pragma")
            )
            is_true(
                engine.dialect.supports_capability(
                    conn, "supports_table_xinfo_pragma"
                )
            )
            eq_(
                conn.get_capability("supports_table_xinfo_pragma"), True
            )
            rec = engine.dialect.capabilities.get_record(
                "supports_table_xinfo_pragma", conn
            )
            eq_(rec.source, CapabilitySource.VERSION_RANGE)
        # writeback happened onto the dialect attribute
        is_true(engine.dialect.supports_table_xinfo_pragma)

    def test_reflection_uses_negotiated_branch(self):
        engine = create_engine("sqlite://")
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "CREATE TABLE t (a INTEGER, b INTEGER GENERATED ALWAYS "
                "AS (a * 2) VIRTUAL)"
            )
        inspector = inspect(engine)
        cols = {c["name"]: c for c in inspector.get_columns("t")}
        is_true(cols["b"]["computed"])

    def test_mock_connection_uses_declared(self):
        engine = create_mock_engine("sqlite://", lambda *a, **k: None)
        # no pool-managed connection -> class-level declaration
        is_false(
            engine.dialect.capabilities.supports(
                "supports_table_xinfo_pragma", engine
            )
        )

    def test_reconnect_to_new_server_renegotiates(self):
        engine = create_engine("sqlite://")
        dialect = engine.dialect
        versions = iter([(3, 30, 0), (3, 31, 0)])
        dialect._get_server_version_info = lambda conn: next(versions)

        with engine.connect() as conn:
            is_false(
                conn.supports_capability("supports_table_xinfo_pragma")
            )
        is_false(dialect.supports_table_xinfo_pragma)

        # hard invalidate; the next physical connection reaches a
        # "new server" reporting 3.31 -> renegotiation
        with engine.connect() as conn:
            conn.invalidate()

        with engine.connect() as conn:
            is_true(
                conn.supports_capability("supports_table_xinfo_pragma")
            )
        is_true(dialect.supports_table_xinfo_pragma)
        eq_(
            dialect.capabilities.get_record(
                "supports_table_xinfo_pragma"
            ).server_version_info,
            (3, 31, 0),
        )

    def test_reconnect_to_same_server_reuses_probe(self):
        engine = create_engine("sqlite://")
        dialect = engine.dialect
        probe_calls = []
        dialect.capabilities.register_probe(
            "supports_table_xinfo_pragma",
            lambda conn: probe_calls.append(1) or True,
        )

        with engine.connect() as conn:
            is_true(
                conn.supports_capability("supports_table_xinfo_pragma")
            )
        with engine.connect() as conn:
            conn.invalidate()
        with engine.connect() as conn:
            is_true(
                conn.supports_capability("supports_table_xinfo_pragma")
            )

        # first connect probed once; reconnect reached the same server
        # and reused the cached negotiation
        eq_(len(probe_calls), 1)

        # an explicit renegotiation against the observed server probes
        # again
        dialect.capabilities.negotiate(
            _FakePoolConnection(),
            server_version_info=dialect.server_version_info,
            force=True,
        )
        eq_(len(probe_calls), 2)

    def test_probe_conflict_on_live_connection(self, caplog):
        engine = create_engine("sqlite://")
        engine.dialect.capabilities.register_probe(
            "supports_table_xinfo_pragma", lambda conn: False
        )
        caplog.set_level(
            logging.WARNING,
            logger="sqlalchemy.engine.capabilities",
        )

        with engine.connect() as conn:
            is_false(
                conn.supports_capability("supports_table_xinfo_pragma")
            )
            rec = conn.dialect.capabilities.get_record(
                "supports_table_xinfo_pragma", conn
            )
            is_true(rec.conflict)
            eq_(rec.baseline_value, True)

        is_false(engine.dialect.supports_table_xinfo_pragma)

    def test_failed_version_detection_on_initialize(self):
        engine = create_engine("sqlite://")

        def no_version(conn):
            raise NotImplementedError()

        engine.dialect._get_server_version_info = no_version
        # initialize must not blow up; capabilities degrade to the
        # class-level declarations
        with engine.connect() as conn:
            is_false(
                conn.supports_capability("supports_table_xinfo_pragma")
            )
            eq_(
                conn.dialect.capabilities.get_record(
                    "supports_table_xinfo_pragma", conn
                ).source,
                CapabilitySource.DECLARED,
            )


class BuiltinDialectDeclarationTest(fixtures.TestBase):
    """Version-range declarations must produce exactly the values the
    old scattered ``server_version_info >= (...)`` assignments did."""

    def test_postgresql_initialize_equivalence(self):
        from sqlalchemy.dialects.postgresql.base import PGDialect

        cases = [
            # version, smallserial, drop_idx_conc, identity, jsonb_sub,
            # virtual, indnkeyatts, nullsnotdistinct
            ((8, 4), False, False, False, False, False, False, False),
            ((9, 1), False, False, False, False, False, False, False),
            ((9, 2), True, True, False, False, False, False, False),
            ((9, 5), True, True, False, False, False, False, False),
            ((10, 0), True, True, True, False, False, False, False),
            ((13, 9), True, True, True, False, False, True, False),
            ((14, 0), True, True, True, True, False, True, False),
            ((17, 99), True, True, True, True, False, True, True),
            ((18, 0), True, True, True, True, True, True, True),
            ((20, 0), True, True, True, True, True, True, True),
        ]

        for version, *expected in cases:
            dialect = PGDialect()
            dialect.capabilities.negotiate(
                None, server_version_info=version, force=True
            )
            actual = (
                dialect.supports_smallserial,
                dialect._supports_drop_index_concurrently,
                dialect.supports_identity_columns,
                dialect._supports_jsonb_subscripting,
                dialect.supports_virtual_generated_columns,
                dialect.capabilities.supports(
                    "supports_index_including_key_attributes"
                ),
                dialect.capabilities.supports(
                    "supports_index_nulls_not_distinct"
                ),
            )
            eq_(actual, tuple(expected), version)

    def test_postgresql_compile_only_keeps_class_declarations(self):
        from sqlalchemy.dialects.postgresql.base import PGDialect

        dialect = PGDialect()
        # un-negotiated dialect keeps observing class-level values
        is_true(dialect.capabilities.supports("supports_identity_columns"))
        is_true(dialect.supports_smallserial)
        is_false(
            dialect.capabilities.supports(
                "supports_index_including_key_attributes"
            )
        )

    def test_sqlite_declarations(self):
        from sqlalchemy.dialects.sqlite.base import SQLiteDialect

        dialect = SQLiteDialect()
        dialect.capabilities.negotiate(
            None, server_version_info=(3, 30, 1), force=True
        )
        is_false(dialect.supports_table_xinfo_pragma)

        dialect.capabilities.negotiate(
            None, server_version_info=(3, 31, 0), force=True
        )
        is_true(dialect.supports_table_xinfo_pragma)

        dialect.capabilities.negotiate(
            None, server_version_info=None, force=True
        )
        is_false(dialect.supports_table_xinfo_pragma)


class CapabilityPublicApiTest(fixtures.TestBase):
    def test_public_symbols(self):
        import sqlalchemy.engine as engine

        for name in (
            "CapabilityRegistry",
            "CapabilityDeclaration",
            "CapabilityRule",
            "CapabilityRecord",
            "CapabilityNegotiation",
            "CapabilitySource",
            "CapabilityUndetermined",
        ):
            assert hasattr(engine, name), name
        # the undetermined sentinel lives on the capabilities module
        assert hasattr(
            capabilities_module, "CAPABILITY_UNDETERMINED"
        )

    def test_registry_repr_lists_names(self):
        dialect, reg = _registry()
        reg.declare("some_flag")
        text = repr(reg)
        assert "some_flag" in text

    def test_module_marker(self):
        eq_(repr(capabilities_module.CAPABILITY_UNDETERMINED),
            "CAPABILITY_UNDETERMINED")
