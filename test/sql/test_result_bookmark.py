# sql/test_result_bookmark.py
# Copyright (C) 2005-2026 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Tests for result set bookmarks - resumable streaming reads."""

from __future__ import annotations

import pickle

from sqlalchemy import create_engine
from sqlalchemy import exc
from sqlalchemy import Integer
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import text
from sqlalchemy.engine import cursor as _cursor
from sqlalchemy.engine import ResultBookmark
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Session
from sqlalchemy.testing import config
from sqlalchemy.testing import eq_
from sqlalchemy.testing import expect_raises
from sqlalchemy.testing import expect_raises_message
from sqlalchemy.testing import fixtures
from sqlalchemy.testing import is_false
from sqlalchemy.testing import is_instance_of
from sqlalchemy.testing import is_true
from sqlalchemy.testing.schema import Column
from sqlalchemy.testing.schema import Table


class ResultBookmarkFixtureTest(fixtures.TablesTest):
    __backend__ = True
    __sparse_driver_backend__ = True

    nrows = 100

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "result_bookmark_test",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("data", String(30)),
        )

    def setup_test(self):
        with config.db.begin() as conn:
            conn.execute(self.tables.result_bookmark_test.delete())
            conn.execute(
                self.tables.result_bookmark_test.insert(),
                [
                    {"id": i, "data": "row %d" % i}
                    for i in range(self.nrows)
                ],
            )

    @property
    def table(self):
        return self.tables.result_bookmark_test

    def _select_all_text(self):
        return text(
            "SELECT id, data FROM result_bookmark_test ORDER BY id"
        )

    def _select_all_core(self):
        return select(
            self.table.c.id, self.table.c.data
        ).order_by(self.table.c.id)

    def _resume(self, conn, statement, bookmark, **extra_opts):
        opts = {"bookmark": bookmark}
        opts.update(extra_opts)
        return conn.execute(statement, execution_options=opts)

    def _tracked(self, conn, statement, params=None, **extra_opts):
        opts = {"bookmark_tracking": True}
        opts.update(extra_opts)
        return conn.execute(statement, params, execution_options=opts)


class TrackingSetupTest(ResultBookmarkFixtureTest):
    def test_untracked_result_has_original_strategy(self, connection):
        result = connection.execute(self._select_all_text())
        is_false(
            isinstance(
                result.cursor_strategy,
                _cursor._BookmarkTrackingFetchStrategy,
            )
        )

    def test_untracked_bookmark_raises(self, connection):
        result = connection.execute(self._select_all_text())
        result.fetchone()
        with expect_raises(exc.ResultBookmarkError):
            result.bookmark()

    def test_tracking_installs_counter(self, connection):
        result = self._tracked(connection, self._select_all_text())
        is_instance_of(
            result.cursor_strategy,
            _cursor._BookmarkTrackingFetchStrategy,
        )
        eq_(result.cursor_strategy.rows_consumed, 0)

    def test_resume_option_also_enables_tracking(self, connection):
        result = self._tracked(connection, self._select_all_text())
        result.fetchmany(5)
        bookmark = result.bookmark()

        result2 = self._resume(connection, self._select_all_text(), bookmark)
        is_instance_of(
            result2.cursor_strategy,
            _cursor._BookmarkTrackingFetchStrategy,
        )
        # the fast-forwarded rows are counted in the new result
        eq_(result2.cursor_strategy.rows_consumed, 5)
        result2.close()


class BookmarkResumeTest(ResultBookmarkFixtureTest):
    def _expected(self, start):
        return [(i, "row %d" % i) for i in range(start, self.nrows)]

    def test_resume_after_fetchmany_text(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_text())
            first = result.fetchmany(30)
            bookmark = result.bookmark()
            eq_(len(first), 30)

        with config.db.connect() as conn:
            resumed = self._resume(conn, self._select_all_text(), bookmark)
            eq_(resumed.all(), self._expected(30))

    def test_resume_after_fetchone(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_core())
            for _ in range(20):
                result.fetchone()
            bookmark = result.bookmark()

        with config.db.connect() as conn:
            resumed = self._resume(conn, self._select_all_core(), bookmark)
            eq_([r[0] for r in resumed.fetchmany(10)], list(range(20, 30)))
            eq_(resumed.fetchall(), self._expected(30))

    def test_resume_after_iteration_break(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_text())
            count = 0
            for row in result:
                count += 1
                if count == 15:
                    break
            bookmark = result.bookmark()
            eq_(bookmark.consumed_rows, 15)

        with config.db.connect() as conn:
            resumed = self._resume(conn, self._select_all_text(), bookmark)
            seen = list(resumed)
            eq_([r[0] for r in seen], list(range(15, self.nrows)))

    def test_resume_after_partition(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_core())
            partitions = result.partitions(25)
            first = next(partitions)
            bookmark = result.bookmark()
            eq_(len(first), 25)

        with config.db.connect() as conn:
            resumed = self._resume(conn, self._select_all_core(), bookmark)
            eq_(resumed.all(), self._expected(25))

    def test_resume_with_core_statement_and_params(self):
        stmt = select(self.table.c.id, self.table.c.data).where(
            self.table.c.id < 50
        ).order_by(self.table.c.id)

        with config.db.connect() as conn:
            result = self._tracked(conn, stmt, {"id_1": 50})
            result.fetchmany(10)
            bookmark = result.bookmark()

        with config.db.connect() as conn:
            resumed = self._resume(conn, stmt, bookmark)
            eq_([r[0] for r in resumed.all()], list(range(10, 50)))

    def test_resume_from_zero_position(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_text())
            bookmark = result.bookmark()
            eq_(bookmark.consumed_rows, 0)

        with config.db.connect() as conn:
            resumed = self._resume(conn, self._select_all_text(), bookmark)
            eq_(len(resumed.all()), self.nrows)

    def test_resume_with_exec_driver_sql_positional(self):
        raw_sql = (
            "SELECT id, data FROM result_bookmark_test "
            "WHERE id > ? ORDER BY id"
        )
        with config.db.connect() as conn:
            result = conn.exec_driver_sql(
                raw_sql,
                (40,),
                execution_options={"bookmark_tracking": True},
            )
            result.fetchmany(5)
            bookmark = result.bookmark()

        with config.db.connect() as conn:
            resumed = conn.exec_driver_sql(
                raw_sql,
                (40,),
                execution_options={"bookmark": bookmark},
            )
            eq_([r[0] for r in resumed.all()], list(range(46, self.nrows)))

    def test_multi_chunk_seek(self):
        with config.db.begin() as conn:
            conn.execute(self.table.delete())
            conn.execute(
                self.table.insert(),
                [{"id": i, "data": "r%d" % i} for i in range(2500)],
            )

        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_text())
            consumed = 0
            while consumed < 2100:
                consumed += len(result.fetchmany(100))
            bookmark = result.bookmark()
            eq_(consumed, 2100)

        with config.db.connect() as conn:
            resumed = self._resume(conn, self._select_all_text(), bookmark)
            rows = resumed.all()
            eq_(len(rows), 400)
            eq_(rows[0][0], 2100)
            eq_(rows[-1][0], 2499)


class BookmarkValidationTest(ResultBookmarkFixtureTest):
    def test_statement_mismatch_rejected(self, connection):
        result = self._tracked(connection, self._select_all_text())
        result.fetchmany(5)
        bookmark = result.bookmark()
        result.close()

        with config.db.connect() as conn:
            with expect_raises(exc.ResultBookmarkError):
                conn.execute(
                    text(
                        "SELECT id, data FROM result_bookmark_test "
                        "ORDER BY data"
                    ),
                    execution_options={"bookmark": bookmark},
                )
            # connection remains usable after rejection
            eq_(
                conn.execute(
                    text("SELECT count(*) FROM result_bookmark_test")
                ).scalar(),
                self.nrows,
            )

    def test_parameters_mismatch_rejected(self):
        stmt = text(
            "SELECT id, data FROM result_bookmark_test "
            "WHERE id < :max ORDER BY id"
        )
        with config.db.connect() as conn:
            result = self._tracked(conn, stmt, {"max": 50})
            result.fetchmany(5)
            bookmark = result.bookmark()

        with config.db.connect() as conn:
            with expect_raises_message(
                exc.ResultBookmarkError, "bound parameters"
            ):
                conn.execute(
                    stmt,
                    {"max": 51},
                    execution_options={"bookmark": bookmark},
                )

    def test_same_params_resume_works(self):
        stmt = text(
            "SELECT id, data FROM result_bookmark_test "
            "WHERE id < :max ORDER BY id"
        )
        with config.db.connect() as conn:
            result = self._tracked(conn, stmt, {"max": 50})
            result.fetchmany(10)
            bookmark = result.bookmark()

        with config.db.connect() as conn:
            resumed = conn.execute(
                stmt,
                {"max": 50},
                execution_options={"bookmark": bookmark},
            )
            eq_([r[0] for r in resumed.all()], list(range(10, 50)))

    def test_engine_mismatch_rejected(self):
        # a second, empty in-memory sqlite engine has a different URL
        other = create_engine("sqlite://")
        try:
            with config.db.connect() as conn:
                result = self._tracked(conn, self._select_all_text())
                result.fetchone()
                bookmark = result.bookmark()

            with other.connect() as conn:
                with expect_raises(exc.ResultBookmarkError):
                    conn.execute(
                        text("SELECT 1"),
                        execution_options={"bookmark": bookmark},
                    )
        finally:
            other.dispose()

    def test_executemany_resume_rejected(self):
        with config.db.begin() as conn:
            result = self._tracked(conn, self._select_all_text())
            result.fetchone()
            bookmark = result.bookmark()

        with config.db.connect() as conn:
            with expect_raises(exc.ResultBookmarkError):
                conn.execute(
                    self.table.insert(),
                    [
                        {"id": 1000, "data": "x"},
                        {"id": 1001, "data": "y"},
                    ],
                    execution_options={"bookmark": bookmark},
                )

    def test_dml_result_bookmark_rejected(self, connection):
        result = connection.execute(
            self.table.update().where(self.table.c.id == 1).values(
                data="updated"
            ),
            execution_options={"bookmark_tracking": True},
        )
        with expect_raises(exc.ResultBookmarkError):
            result.bookmark()

    def test_unique_filter_rejected(self, connection):
        result = self._tracked(connection, self._select_all_text())
        unique = result.unique()
        unique.fetchone()
        with expect_raises(exc.ResultBookmarkError):
            unique.bookmark()

    def test_bookmark_option_wrong_type(self, connection):
        with expect_raises(exc.ResultBookmarkError):
            connection.execute(
                self._select_all_text(),
                execution_options={"bookmark": "not a bookmark"},
            )


class BookmarkStateTest(ResultBookmarkFixtureTest):
    def test_terminal_bookmark_after_all(self, connection):
        result = self._tracked(connection, self._select_all_text())
        rows = result.all()
        eq_(len(rows), self.nrows)

        bookmark = result.bookmark()
        is_true(bookmark.exhausted)
        eq_(bookmark.consumed_rows, self.nrows)

        resumed = self._resume(
            connection, self._select_all_text(), bookmark
        )
        eq_(resumed.all(), [])

    def test_terminal_bookmark_after_iteration(self, connection):
        result = self._tracked(connection, self._select_all_core())
        for _ in result:
            pass
        bookmark = result.bookmark()
        is_true(bookmark.exhausted)

    def test_hard_closed_bookmark_raises(self, connection):
        result = self._tracked(connection, self._select_all_text())
        result.fetchone()
        result.close()
        with expect_raises(exc.ResourceClosedError):
            result.bookmark()

    def test_in_progress_bookmark_not_exhausted(self, connection):
        result = self._tracked(connection, self._select_all_text())
        result.fetchmany(10)
        bookmark = result.bookmark()
        is_false(bookmark.exhausted)
        eq_(bookmark.consumed_rows, 10)

    def test_bookmark_does_not_consume_or_close(self, connection):
        result = self._tracked(connection, self._select_all_text())
        result.fetchmany(10)
        result.bookmark()
        is_false(result._soft_closed)
        eq_(result.fetchone()[0], 10)


class SerializationTest(ResultBookmarkFixtureTest):
    def test_dumps_loads_roundtrip(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_text())
            result.fetchmany(12)
            bookmark = result.bookmark()

        token = bookmark.dumps()
        is_instance_of(token, bytes)

        restored = ResultBookmark.loads(token)
        eq_(restored.consumed_rows, 12)
        is_false(restored.exhausted)

        with config.db.connect() as conn:
            resumed = self._resume(
                conn, self._select_all_text(), restored
            )
            eq_([r[0] for r in resumed.all()], list(range(12, self.nrows)))

    def test_pickle_roundtrip(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_core())
            result.fetchone()
            bookmark = result.bookmark()

        restored = pickle.loads(pickle.dumps(bookmark))
        eq_(restored.consumed_rows, 1)

        with config.db.connect() as conn:
            resumed = self._resume(
                conn, self._select_all_core(), restored
            )
            eq_(len(resumed.all()), self.nrows - 1)

    def test_corrupt_token_rejected(self):
        with expect_raises(exc.ResultBookmarkError):
            ResultBookmark.loads(b"this is not a pickle")

    def test_wrong_type_payload_rejected(self):
        token = pickle.dumps({"v": 1, "n": 3})
        with expect_raises(exc.ResultBookmarkError):
            ResultBookmark.loads(token)

    def test_negative_position_rejected_at_loads(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_text())
            result.fetchmany(10)
            bookmark = result.bookmark()

        bookmark._data = dict(bookmark._data, n=-5)
        with expect_raises(exc.ResultBookmarkError):
            ResultBookmark.loads(bookmark.dumps())

    def test_tampered_payload_rejected_at_resume(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_text())
            result.fetchmany(10)
            bookmark = result.bookmark()

        # bypass the deserialization validation by mutating the
        # in-memory bookmark; the resume path must still reject it
        bookmark._data = dict(bookmark._data, n=-5)

        with config.db.connect() as conn:
            with expect_raises(exc.ResultBookmarkError):
                self._resume(conn, self._select_all_text(), bookmark)

    def test_constructor_is_restricted(self):
        with expect_raises(exc.ResultBookmarkError):
            ResultBookmark()

    def test_repr(self, connection):
        result = self._tracked(connection, self._select_all_text())
        result.fetchmany(3)
        eq_(repr(result.bookmark()), "<ResultBookmark (at row 3)>")
        result.all()
        eq_(repr(result.bookmark()), "<ResultBookmark (exhausted)>")


class ConcurrentResumeTest(ResultBookmarkFixtureTest):
    def test_concurrent_resumes_are_independent(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_text())
            result.fetchmany(10)
            bookmark = result.bookmark()

        conn_a = config.db.connect()
        conn_b = config.db.connect()
        try:
            ra = self._resume(conn_a, self._select_all_text(), bookmark)
            rb = self._resume(conn_b, self._select_all_text(), bookmark)

            eq_([r[0] for r in ra.fetchmany(4)], [10, 11, 12, 13])
            eq_([r[0] for r in rb.fetchmany(7)], list(range(10, 17)))

            # each result tracks its own progress
            eq_(ra.bookmark().consumed_rows, 14)
            eq_(rb.bookmark().consumed_rows, 17)

            eq_([r[0] for r in ra.all()], list(range(14, self.nrows)))
            eq_([r[0] for r in rb.all()], list(range(17, self.nrows)))
        finally:
            conn_a.close()
            conn_b.close()

    def test_repeated_resume_same_bookmark(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_text())
            result.fetchmany(50)
            bookmark = result.bookmark()

        for _ in range(3):
            with config.db.connect() as conn:
                resumed = self._resume(
                    conn, self._select_all_text(), bookmark
                )
                eq_(len(resumed.all()), self.nrows - 50)

    def test_incremental_bookmark_chain(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_text())
            result.fetchmany(10)
            bookmark = result.bookmark()

        expected_starts = [10, 30, 50]
        page_sizes = [20, 20, 30]
        for expected_start, page_size in zip(
            expected_starts, page_sizes
        ):
            with config.db.connect() as conn:
                resumed = self._resume(
                    conn, self._select_all_text(), bookmark
                )
                rows = resumed.fetchmany(page_size)
                eq_(
                    [r[0] for r in rows],
                    list(
                        range(
                            expected_start, expected_start + page_size
                        )
                    ),
                )
                bookmark = resumed.bookmark()
                eq_(
                    bookmark.consumed_rows, expected_start + page_size
                )

        with config.db.connect() as conn:
            resumed = self._resume(
                conn, self._select_all_text(), bookmark
            )
            eq_([r[0] for r in resumed.all()], list(range(80, 100)))


class FilteredResultTest(ResultBookmarkFixtureTest):
    def test_scalars_bookmark_delegates(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_core())
            scalars = result.scalars()
            seen = []
            for value in scalars:
                seen.append(value)
                if len(seen) == 8:
                    break
            bookmark = scalars.bookmark()
            eq_(bookmark.consumed_rows, 8)

        with config.db.connect() as conn:
            resumed = self._resume(
                conn, self._select_all_core(), bookmark
            )
            eq_(list(resumed.scalars())[:3], [8, 9, 10])

    def test_mappings_bookmark_delegates(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_core())
            mappings = result.mappings()
            first = mappings.fetchmany(5)
            bookmark = mappings.bookmark()
            eq_([m["id"] for m in first], list(range(5)))

        with config.db.connect() as conn:
            resumed = self._resume(
                conn, self._select_all_core(), bookmark
            )
            eq_(
                [m["id"] for m in resumed.mappings().fetchmany(3)],
                [5, 6, 7],
            )

    def test_columns_slice_bookmark(self):
        with config.db.connect() as conn:
            result = self._tracked(conn, self._select_all_core())
            sliced = result.columns("data")
            rows = sliced.fetchmany(3)
            eq_([r[0] for r in rows], ["row 0", "row 1", "row 2"])
            bookmark = sliced.bookmark()
            eq_(bookmark.consumed_rows, 3)

        # resume on the raw statement, then apply the same slice
        with config.db.connect() as conn:
            resumed = self._resume(
                conn, self._select_all_core(), bookmark
            )
            rows = resumed.columns("data").fetchmany(2)
            eq_([r[0] for r in rows], ["row 3", "row 4"])

    def test_scalar_unique_rejected(self, connection):
        result = self._tracked(connection, self._select_all_core())
        scalars = result.scalars().unique()
        with expect_raises(exc.ResultBookmarkError):
            scalars.bookmark()


class FetchStrategyTest(ResultBookmarkFixtureTest):
    def test_stream_results_counts_consumed_not_buffered(
        self, connection
    ):
        result = self._tracked(
            connection,
            self._select_all_text(),
            stream_results=True,
            max_row_buffer=20,
        )
        for _ in range(3):
            result.fetchone()
        bookmark = result.bookmark()
        # exactly three consumed, even though more rows were buffered
        eq_(bookmark.consumed_rows, 3)
        result.close()

    def test_yield_per_counts(self, connection):
        result = self._tracked(
            connection,
            self._select_all_core(),
            yield_per=25,
        )
        rows = result.fetchmany(40)
        eq_(len(rows), 40)
        eq_(result.bookmark().consumed_rows, 40)
        result.close()

    def test_yield_per_method_counts(self, connection):
        result = self._tracked(
            connection, self._select_all_text()
        ).yield_per(15)
        result.fetchmany(35)
        eq_(result.bookmark().consumed_rows, 35)
        result.close()


class UnaffectedSemanticsTest(ResultBookmarkFixtureTest):
    def test_freeze_untracked_unchanged(self, connection):
        result = connection.execute(self._select_all_core())
        frozen = result.freeze()
        first = frozen()
        eq_([r[0] for r in first.all()], list(range(self.nrows)))
        eq_([r[0] for r in frozen().all()], list(range(self.nrows)))

    def test_freeze_tracked_unchanged(self, connection):
        result = self._tracked(connection, self._select_all_core())
        frozen = result.freeze()
        eq_([r[0] for r in frozen().all()], list(range(self.nrows)))
        # freezing consumed everything: terminal bookmark available
        bookmark = result.bookmark()
        is_true(bookmark.exhausted)
        eq_(bookmark.consumed_rows, self.nrows)

    def test_partitions_untracked_unchanged(self, connection):
        result = connection.execute(self._select_all_text())
        collected = [
            row for partition in result.partitions(17) for row in partition
        ]
        eq_([r[0] for r in collected], list(range(self.nrows)))

    def test_iterator_result_bookmark_rejected(self, connection):
        result = connection.execute(self._select_all_core())
        frozen = result.freeze()
        unfrozen = frozen()
        with expect_raises(exc.ResultBookmarkError):
            unfrozen.bookmark()


class OrmResultBookmarkTest(fixtures.TestBase):
    __backend__ = True
    __sparse_driver_backend__ = True

    def setup_test(self):
        class Base(DeclarativeBase):
            pass

        class OrmThing(Base):
            __tablename__ = "result_bookmark_orm"
            id = mapped_column(Integer, primary_key=True)
            data = mapped_column(String(30))

        Base.metadata.create_all(config.db)
        self._Base = Base
        self._OrmThing = OrmThing

        with Session(config.db) as session:
            session.add_all(
                OrmThing(id=i, data="row %d" % i) for i in range(20)
            )
            session.commit()

    def teardown_test(self):
        self._Base.metadata.drop_all(config.db)

    def test_orm_object_result_bookmark_rejected(self):
        # ORM object-hydrating results don't map raw cursor positions
        # 1:1 to entities; the rejection must be deterministic rather
        # than silently restarting the query
        with Session(config.db) as session:
            result = session.execute(
                select(self._OrmThing),
                execution_options={"bookmark_tracking": True},
            )
            result.fetchmany(5)
            with expect_raises(exc.ResultBookmarkError):
                result.bookmark()

