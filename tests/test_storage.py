"""Tests for the SQLite storage module."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from metareason.storage.store import RunStore


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_runs.db"


@pytest.fixture
def store(db_path):
    s = RunStore(db_path)
    yield s
    s.close()


class TestRunLifecycle:
    def test_start_run_returns_id(self, store):
        run_id = store.start_run("test-spec", n_variants=5, n_oracles=2)
        assert isinstance(run_id, int)
        assert run_id > 0

    def test_start_and_finish_run(self, store):
        run_id = store.start_run("test-spec", n_variants=5, n_oracles=2)
        store.finish_run(run_id)
        run = store.get_run(run_id)
        assert run["status"] == "completed"
        assert run["finished_at"] is not None

    def test_finish_run_with_failed_status(self, store):
        run_id = store.start_run("test-spec", n_variants=5, n_oracles=2)
        store.finish_run(run_id, status="failed")
        run = store.get_run(run_id)
        assert run["status"] == "failed"

    def test_list_runs(self, store):
        store.start_run("spec-a", n_variants=3, n_oracles=1)
        store.start_run("spec-b", n_variants=5, n_oracles=2)
        runs = store.list_runs()
        assert len(runs) == 2
        # Most recent first
        assert runs[0]["spec_id"] == "spec-b"

    def test_list_runs_with_limit(self, store):
        for i in range(5):
            store.start_run(f"spec-{i}", n_variants=1, n_oracles=1)
        runs = store.list_runs(limit=3)
        assert len(runs) == 3

    def test_get_run_not_found(self, store):
        assert store.get_run(999) is None


class TestPipelineStages:
    def test_save_pipeline_stages(self, store):
        run_id = store.start_run("test-spec", n_variants=1, n_oracles=1)
        stages = [
            {
                "model": "gpt-4",
                "adapter": "openai",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000,
            },
            {
                "model": "gpt-3.5-turbo",
                "adapter": "openai",
                "temperature": 0.5,
            },
        ]
        store.save_pipeline_stages(run_id, stages)

        # Verify via raw SQL
        conn = store._get_conn()
        rows = conn.execute(
            "SELECT * FROM pipeline_stages WHERE run_id = ? ORDER BY stage_index",
            (run_id,),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["model"] == "gpt-4"
        assert rows[1]["model"] == "gpt-3.5-turbo"


class TestSampleStorage:
    def test_save_and_get_sample(self, store):
        run_id = store.start_run("test-spec", n_variants=1, n_oracles=1)
        sample_id = store.save_sample(
            run_id=run_id,
            sample_index=0,
            sample_params={"temp": 0.7, "style": "formal"},
            original_prompt="Write a poem",
            final_response="Roses are red...",
            evaluations={
                "quality": {"score": 4.5, "explanation": "Good poem"},
            },
        )
        assert sample_id > 0

        samples = store.get_samples(run_id)
        assert len(samples) == 1
        assert samples[0]["sample_params"] == {"temp": 0.7, "style": "formal"}
        assert samples[0]["original_prompt"] == "Write a poem"
        assert samples[0]["evaluations"]["quality"]["score"] == 4.5

    def test_save_multiple_samples(self, store):
        run_id = store.start_run("test-spec", n_variants=3, n_oracles=2)
        for i in range(3):
            store.save_sample(
                run_id=run_id,
                sample_index=i,
                sample_params={"idx": i},
                original_prompt=f"Prompt {i}",
                final_response=f"Response {i}",
                evaluations={
                    "oracle_a": {"score": 3.0 + i * 0.5, "explanation": f"Ok {i}"},
                    "oracle_b": {"score": 4.0 - i * 0.3},
                },
            )

        samples = store.get_samples(run_id)
        assert len(samples) == 3
        assert len(samples[0]["evaluations"]) == 2

    def test_save_sample_without_explanation(self, store):
        run_id = store.start_run("test", n_variants=1, n_oracles=1)
        store.save_sample(
            run_id=run_id,
            sample_index=0,
            sample_params={},
            original_prompt="Hi",
            final_response="Hello",
            evaluations={"judge": {"score": 3.0}},
        )
        samples = store.get_samples(run_id)
        assert samples[0]["evaluations"]["judge"]["explanation"] is None


class TestAnalysisStorage:
    def test_save_and_get_analysis(self, store):
        run_id = store.start_run("test-spec", n_variants=5, n_oracles=1)
        result = {
            "population_mean": 4.2,
            "hdi_lower": 3.8,
            "hdi_upper": 4.6,
            "oracle_noise_mean": 0.3,
        }
        store.save_analysis(run_id, "quality_judge", result)

        analyses = store.get_analysis(run_id)
        assert len(analyses) == 1
        assert analyses[0]["oracle_name"] == "quality_judge"
        assert analyses[0]["result"]["population_mean"] == 4.2

    def test_save_multiple_analyses(self, store):
        run_id = store.start_run("test-spec", n_variants=5, n_oracles=2)
        store.save_analysis(run_id, "oracle_a", {"mean": 4.0})
        store.save_analysis(run_id, "oracle_b", {"mean": 3.5})

        analyses = store.get_analysis(run_id)
        assert len(analyses) == 2


class TestScoreQueries:
    def _populate(self, store):
        run_id = store.start_run("test", n_variants=3, n_oracles=2)
        for i in range(3):
            store.save_sample(
                run_id=run_id,
                sample_index=i,
                sample_params={"idx": i},
                original_prompt=f"P{i}",
                final_response=f"R{i}",
                evaluations={
                    "judge_a": {"score": 3.0 + i},
                    "judge_b": {"score": 4.0},
                },
            )
        return run_id

    def test_get_all_scores(self, store):
        run_id = self._populate(store)
        scores = store.get_scores(run_id)
        assert len(scores) == 6  # 3 samples x 2 oracles

    def test_get_scores_filtered(self, store):
        run_id = self._populate(store)
        scores = store.get_scores(run_id, oracle_name="judge_a")
        assert len(scores) == 3
        assert all(s["oracle_name"] == "judge_a" for s in scores)


class TestExportForFinetuning:
    def _populate(self, store):
        run_id = store.start_run("test", n_variants=3, n_oracles=1)
        for i, score in enumerate([2.0, 4.0, 5.0]):
            store.save_sample(
                run_id=run_id,
                sample_index=i,
                sample_params={"idx": i},
                original_prompt=f"Prompt {i}",
                final_response=f"Response {i}",
                evaluations={"judge": {"score": score}},
            )
        return run_id

    def test_export_default_threshold(self, store):
        self._populate(store)
        pairs = store.export_for_finetuning()
        assert len(pairs) == 2  # scores 4.0 and 5.0
        assert all(p["score"] >= 4.0 for p in pairs)

    def test_export_custom_threshold(self, store):
        self._populate(store)
        pairs = store.export_for_finetuning(min_score=5.0)
        assert len(pairs) == 1

    def test_export_by_run_id(self, store):
        run_id = self._populate(store)
        # Create another run with a high score
        run_id2 = store.start_run("other", n_variants=1, n_oracles=1)
        store.save_sample(
            run_id=run_id2,
            sample_index=0,
            sample_params={},
            original_prompt="Other",
            final_response="Other resp",
            evaluations={"judge": {"score": 5.0}},
        )

        pairs = store.export_for_finetuning(run_id=run_id)
        assert all(p["prompt"].startswith("Prompt") for p in pairs)

    def test_export_format(self, store):
        self._populate(store)
        pairs = store.export_for_finetuning(min_score=5.0)
        assert len(pairs) == 1
        p = pairs[0]
        assert "prompt" in p
        assert "response" in p
        assert "score" in p
        assert "oracle_name" in p
        assert "sample_params" in p


class TestContextManager:
    def test_context_manager(self, db_path):
        with RunStore(db_path) as store:
            run_id = store.start_run("test", n_variants=1, n_oracles=1)
            store.finish_run(run_id)
        # Verify data persisted
        with RunStore(db_path) as store:
            runs = store.list_runs()
            assert len(runs) == 1


class TestSchemaIdempotency:
    def test_reopen_db(self, db_path):
        store1 = RunStore(db_path)
        store1.start_run("test", n_variants=1, n_oracles=1)
        store1.close()

        store2 = RunStore(db_path)
        runs = store2.list_runs()
        assert len(runs) == 1
        store2.close()

    def test_wal_mode(self, db_path):
        store = RunStore(db_path)
        conn = store._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        store.close()

    def test_foreign_keys_enabled(self, db_path):
        store = RunStore(db_path)
        conn = store._get_conn()
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        store.close()
