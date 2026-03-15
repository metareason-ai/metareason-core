"""SQLite-based storage for evaluation run data.

Stores prompts, responses, evaluation scores, and analysis results
for future fine-tuning and dashboard use.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


_SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    n_variants INTEGER,
    n_oracles INTEGER,
    status TEXT NOT NULL DEFAULT 'running',
    spec_yaml TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_stages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    stage_index INTEGER NOT NULL,
    model TEXT NOT NULL,
    adapter TEXT NOT NULL,
    temperature REAL,
    top_p REAL,
    max_tokens INTEGER
);

CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    sample_index INTEGER NOT NULL,
    sample_params TEXT NOT NULL,
    original_prompt TEXT NOT NULL,
    final_response TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id INTEGER NOT NULL REFERENCES samples(id),
    oracle_name TEXT NOT NULL,
    score REAL NOT NULL,
    explanation TEXT
);

CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    oracle_name TEXT NOT NULL,
    analysis_type TEXT NOT NULL DEFAULT 'population_quality',
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_samples_run_id ON samples(run_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_sample_id ON evaluations(sample_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_oracle ON evaluations(oracle_name);
CREATE INDEX IF NOT EXISTS idx_analysis_run_id ON analysis_results(run_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_run_id ON pipeline_stages(run_id);
"""


class RunStore:
    """SQLite store for evaluation run data.

    Args:
        db_path: Path to the SQLite database file. Created if it doesn't exist.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self):
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)

        # Check/set schema version
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,)
            )
        conn.commit()

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Write operations ──────────────────────────────────────────────

    def start_run(
        self,
        spec_id: str,
        n_variants: int,
        n_oracles: int,
        spec_yaml: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """Create a new run record and return its ID."""
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO runs (spec_id, started_at, n_variants, n_oracles,
                                status, spec_yaml, metadata)
               VALUES (?, ?, ?, ?, 'running', ?, ?)""",
            (
                spec_id,
                datetime.now().isoformat(),
                n_variants,
                n_oracles,
                spec_yaml,
                json.dumps(metadata) if metadata else None,
            ),
        )
        conn.commit()
        return cur.lastrowid

    def save_pipeline_stages(self, run_id: int, stages: List[dict]):
        """Save pipeline stage configurations for a run."""
        conn = self._get_conn()
        for i, stage in enumerate(stages):
            conn.execute(
                """INSERT INTO pipeline_stages
                   (run_id, stage_index, model, adapter, temperature, top_p, max_tokens)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    i,
                    stage["model"],
                    stage["adapter"],
                    stage.get("temperature"),
                    stage.get("top_p"),
                    stage.get("max_tokens"),
                ),
            )
        conn.commit()

    def save_sample(
        self,
        run_id: int,
        sample_index: int,
        sample_params: dict,
        original_prompt: str,
        final_response: str,
        evaluations: Dict[str, dict],
    ) -> int:
        """Save a sample result with its evaluations. Returns sample ID."""
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO samples (run_id, sample_index, sample_params,
                                   original_prompt, final_response)
               VALUES (?, ?, ?, ?, ?)""",
            (
                run_id,
                sample_index,
                json.dumps(sample_params),
                original_prompt,
                final_response,
            ),
        )
        sample_id = cur.lastrowid

        for oracle_name, eval_data in evaluations.items():
            conn.execute(
                """INSERT INTO evaluations (sample_id, oracle_name, score, explanation)
                   VALUES (?, ?, ?, ?)""",
                (
                    sample_id,
                    oracle_name,
                    eval_data["score"],
                    eval_data.get("explanation"),
                ),
            )
        conn.commit()
        return sample_id

    def finish_run(self, run_id: int, status: str = "completed"):
        """Mark a run as finished."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE runs SET finished_at = ?, status = ? WHERE id = ?",
            (datetime.now().isoformat(), status, run_id),
        )
        conn.commit()

    def save_analysis(
        self,
        run_id: int,
        oracle_name: str,
        result: dict,
        analysis_type: str = "population_quality",
    ):
        """Save analysis results for a run/oracle."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO analysis_results
               (run_id, oracle_name, analysis_type, result_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                run_id,
                oracle_name,
                analysis_type,
                json.dumps(result),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()

    # ── Read operations ───────────────────────────────────────────────

    def list_runs(self, limit: int = 20) -> List[dict]:
        """List recent runs."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, spec_id, started_at, finished_at, n_variants,
                      n_oracles, status
               FROM runs ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: int) -> Optional[dict]:
        """Get a single run by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_samples(self, run_id: int) -> List[dict]:
        """Get all samples for a run, including evaluations."""
        conn = self._get_conn()
        samples = conn.execute(
            """SELECT id, sample_index, sample_params, original_prompt, final_response
               FROM samples WHERE run_id = ? ORDER BY sample_index""",
            (run_id,),
        ).fetchall()

        result = []
        for s in samples:
            sample = dict(s)
            sample["sample_params"] = json.loads(sample["sample_params"])

            evals = conn.execute(
                """SELECT oracle_name, score, explanation
                   FROM evaluations WHERE sample_id = ?""",
                (s["id"],),
            ).fetchall()
            sample["evaluations"] = {
                e["oracle_name"]: {"score": e["score"], "explanation": e["explanation"]}
                for e in evals
            }
            result.append(sample)
        return result

    def get_scores(
        self, run_id: int, oracle_name: Optional[str] = None
    ) -> List[dict]:
        """Get scores for a run, optionally filtered by oracle."""
        conn = self._get_conn()
        if oracle_name:
            rows = conn.execute(
                """SELECT e.oracle_name, e.score, s.sample_index, s.sample_params
                   FROM evaluations e
                   JOIN samples s ON e.sample_id = s.id
                   WHERE s.run_id = ? AND e.oracle_name = ?
                   ORDER BY s.sample_index""",
                (run_id, oracle_name),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT e.oracle_name, e.score, s.sample_index, s.sample_params
                   FROM evaluations e
                   JOIN samples s ON e.sample_id = s.id
                   WHERE s.run_id = ?
                   ORDER BY s.sample_index, e.oracle_name""",
                (run_id,),
            ).fetchall()

        return [
            {
                "oracle_name": r["oracle_name"],
                "score": r["score"],
                "sample_index": r["sample_index"],
                "sample_params": json.loads(r["sample_params"]),
            }
            for r in rows
        ]

    def get_analysis(self, run_id: int) -> List[dict]:
        """Get analysis results for a run."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT oracle_name, analysis_type, result_json, created_at
               FROM analysis_results WHERE run_id = ?""",
            (run_id,),
        ).fetchall()
        return [
            {
                "oracle_name": r["oracle_name"],
                "analysis_type": r["analysis_type"],
                "result": json.loads(r["result_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def export_for_finetuning(
        self,
        run_id: Optional[int] = None,
        min_score: float = 4.0,
        oracle_name: Optional[str] = None,
    ) -> List[dict]:
        """Export prompt/response pairs for fine-tuning.

        Returns pairs where at least one oracle scored >= min_score.

        Args:
            run_id: Filter to a specific run. If None, searches all runs.
            min_score: Minimum score threshold for inclusion.
            oracle_name: Filter to a specific oracle's scores.
        """
        conn = self._get_conn()

        query = """
            SELECT DISTINCT s.original_prompt, s.final_response,
                   s.sample_params, e.oracle_name, e.score
            FROM samples s
            JOIN evaluations e ON e.sample_id = s.id
            WHERE e.score >= ?
        """
        params: list = [min_score]

        if run_id is not None:
            query += " AND s.run_id = ?"
            params.append(run_id)
        if oracle_name is not None:
            query += " AND e.oracle_name = ?"
            params.append(oracle_name)

        query += " ORDER BY e.score DESC"
        rows = conn.execute(query, params).fetchall()

        return [
            {
                "prompt": r["original_prompt"],
                "response": r["final_response"],
                "sample_params": json.loads(r["sample_params"]),
                "oracle_name": r["oracle_name"],
                "score": r["score"],
            }
            for r in rows
        ]
