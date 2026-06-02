"""
GPUOS Experiment Logger (for baselines, agent ranking, benches, PhaseN work).

JSON-first (zero setup, like sudoku-era v2 dashboard). Optional SurrealDB fallback
if 'surrealdb' package + server available (for cross-project sharing with old
Transformer-VM experiments).

Usage (standalone or from harness/demos/benches):
    from experiments.gpuos_experiment_logger import ExperimentLogger
    logger = ExperimentLogger()
    exp_id = logger.create_experiment(name="agent-ranking-512-v1", track="agent-ranking", phase="2")
    run_id = logger.log_run(
        experiment_id=exp_id,
        kernel_ms=5.0,
        vram_delta_mib=2,
        sync_processed=256,
        num_candidates=512,
        recall_at_k=0.67,
        latency_per_turn_ms=0.7,
        success=True,
        config={"arch": "compute_120", "mode": "baseline"},
    )
    logger.add_observation(run_id, "counters", "heartbeat_peak", "3900")
    logger.add_observation(run_id, "verification", "baseline_clean", "true")

Also has helpers:
- parse_jit_baseline_output(stdout) -> dict for auto-log from persistent_jit runs
- log_from_benchmark_result(...)
- context manager for auto success/failure.

Results go to experiments/results/ (or override). Dashboard picks them up.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

try:
    from surrealdb import Surreal
except ImportError:
    Surreal = None  # type: ignore

# Results dir (relative to this file's experiments/ parent = GPUOS root / experiments/results)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class ExperimentLogger:
    def __init__(
        self,
        url: str | None = None,
        namespace: str = "gpuos_experiments",
        database: str = "gpuos_experiments",
        username: str = "root",
        password: str = "root",
        results_dir: Path | None = None,
    ):
        self.url = url or os.getenv("SURREAL_URL", "http://localhost:8001")
        self.namespace = namespace
        self.database = database
        self.username = username
        self.password = password
        self.results_dir = results_dir or RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._db: Any = None
        self._surreal_available = Surreal is not None

    def connect(self):
        if not self._surreal_available:
            raise RuntimeError("surrealdb Python package not installed (optional). pip install surrealdb ; or rely on JSON results/ (preferred, zero-setup)")
        self._db = Surreal(self.url)
        self._db.signin({"username": self.username, "password": self.password})
        self._db.use(self.namespace, self.database)
        return self._db

    def _ensure_connected(self):
        if self._db is None and self._surreal_available:
            try:
                self.connect()
            except Exception as e:
                print(f"[logger] Surreal connect failed (using JSON only): {e}")
                self._db = None

    # ------------------------------------------------------------------
    # Core logging (JSON always; Surreal best-effort)
    # ------------------------------------------------------------------
    def create_experiment(
        self,
        name: str,
        track: str,
        phase: str = "2",
        problem_id: str | None = None,
        notes: str | None = None,
    ) -> str:
        """Create experiment record. Returns id (for json: the name or generated)."""
        exp_id = name.replace(" ", "_").replace("/", "_")
        data = {
            "id": exp_id,
            "name": name,
            "track": track,
            "phase": phase,
            "problem_id": problem_id,
            "status": "running",
            "notes": notes,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # Always write a json "experiment header" (lightweight)
        exp_path = self.results_dir / f"{exp_id}.json"
        if not exp_path.exists():
            with open(exp_path, "w") as f:
                json.dump({"experiment": data, "runs": []}, f, indent=2)
        # Surreal best effort
        self._ensure_connected()
        if self._db is not None:
            try:
                self._db.create("experiments", data)
            except Exception:
                pass
        return exp_id

    def log_run(
        self,
        experiment_id: str,
        *,
        kernel_ms: Optional[float] = None,
        wall_ms: Optional[float] = None,
        vram_pre_mib: Optional[float] = None,
        vram_post_mib: Optional[float] = None,
        vram_delta_mib: Optional[float] = None,
        sync_processed: Optional[int] = None,
        sync_submitted: Optional[int] = None,
        sync_heartbeat: Optional[int] = None,
        num_candidates: Optional[int] = None,
        k: Optional[int] = None,
        recall_at_k: Optional[float] = None,
        latency_per_turn_ms: Optional[float] = None,
        speedup: Optional[float] = None,
        candidates_per_sec: Optional[float] = None,
        engine: str = "persistent_jit",
        success: bool = True,
        config: Optional[Dict[str, Any]] = None,
        notes: str | None = None,
        raw_output: str | None = None,
    ) -> str:
        """Log a run. Returns run-ish id (timestamp based for json). Always writes JSON."""
        ts = datetime.now(timezone.utc)
        run_id = f"{experiment_id}_{ts.strftime('%Y%m%d_%H%M%S')}"
        run_data = {
            "id": run_id,
            "experiment_id": experiment_id,
            "engine": engine,
            "kernel_ms": kernel_ms,
            "wall_ms": wall_ms,
            "vram_pre_mib": vram_pre_mib,
            "vram_post_mib": vram_post_mib,
            "vram_delta_mib": vram_delta_mib,
            "sync_processed": sync_processed,
            "sync_submitted": sync_submitted,
            "sync_heartbeat": sync_heartbeat,
            "num_candidates": num_candidates,
            "k": k,
            "recall_at_k": recall_at_k,
            "latency_per_turn_ms": latency_per_turn_ms,
            "speedup": speedup,
            "candidates_per_sec": candidates_per_sec,
            "success": success,
            "config": config or {},
            "notes": notes,
            "created_at": ts.isoformat(),
            "raw_output_snippet": (raw_output or "")[:2000] if raw_output else None,
        }
        # Append to the experiment json (or create)
        exp_path = self.results_dir / f"{experiment_id}.json"
        if exp_path.exists():
            with open(exp_path, "r") as f:
                exp_doc = json.load(f)
        else:
            exp_doc = {"experiment": {"id": experiment_id}, "runs": []}
        exp_doc.setdefault("runs", []).append(run_data)
        with open(exp_path, "w") as f:
            json.dump(exp_doc, f, indent=2)

        # Surreal best-effort
        self._ensure_connected()
        if self._db is not None:
            try:
                self._db.create("runs", run_data)
            except Exception:
                pass

        print(f"[GPUOS logger] Run logged: {run_id} (json in {exp_path})")
        return run_id

    def add_observation(self, run_id: str, category: str, key: str, value: Any, note: str | None = None):
        """Add observation (appended to the run's experiment json under 'observations')."""
        # Find which exp json has this run (simple scan; fine for small #)
        for jf in self.results_dir.glob("*.json"):
            with open(jf) as f:
                doc = json.load(f)
            for r in doc.get("runs", []):
                if r.get("id") == run_id:
                    r.setdefault("observations", []).append({
                        "category": category, "key": key, "value": str(value), "note": note,
                        "created_at": datetime.now(timezone.utc).isoformat()
                    })
                    with open(jf, "w") as f:
                        json.dump(doc, f, indent=2)
                    # surreal optional
                    self._ensure_connected()
                    if self._db is not None:
                        try:
                            self._db.create("observations", {"run_id": run_id, "category": category, "key": key, "value": str(value), "note": note})
                        except Exception:
                            pass
                    return
        print(f"[GPUOS logger] Warning: run {run_id} not found for observation")

    # ------------------------------------------------------------------
    # Helpers for common GPUOS sources
    # ------------------------------------------------------------------
    def parse_jit_baseline_output(self, stdout: str) -> Dict[str, Any]:
        """Parse the structured output from persistent_jit (baseline block + Phase1 prints)."""
        data: Dict[str, Any] = {}
        # kernel time
        m = re.search(r"processed \d+/\d+ in ([\d.]+) ms", stdout)
        if m: data["kernel_ms"] = float(m.group(1))
        # wall from FINAL
        m = re.search(r"Exited cleanly after kernel baseline in ([\d.]+) ms", stdout)
        if m: data["wall_ms"] = float(m.group(1))
        # vram
        m = re.search(r"post-sync-layer.*?([\d.]+) MiB", stdout)
        if m: data["vram_post_mib"] = float(m.group(1))
        m = re.search(r"pre-sync-layer.*?([\d.]+) MiB", stdout)
        if m: data["vram_pre_mib"] = float(m.group(1))
        if data.get("vram_pre_mib") and data.get("vram_post_mib"):
            data["vram_delta_mib"] = data["vram_post_mib"] - data["vram_pre_mib"]
        # sync counters
        m = re.search(r"sync counters: processed=(\d+) submitted=(\d+) heartbeat=(\d+)", stdout)
        if m:
            data["sync_processed"] = int(m.group(1))
            data["sync_submitted"] = int(m.group(2))
            data["sync_heartbeat"] = int(m.group(3))
        data["success"] = "FINAL (baseline only)" in stdout and "Exited cleanly" in stdout
        return data

    def log_from_jit_run(self, experiment_name: str, stdout: str, config: Optional[Dict] = None, notes: str | None = None):
        exp_id = self.create_experiment(name=experiment_name, track="jit-baseline", phase="0/1")
        parsed = self.parse_jit_baseline_output(stdout)
        return self.log_run(exp_id, **parsed, config=config or {"mode": "baseline-force"}, notes=notes, raw_output=stdout)

    # (similar helpers for bench/agent can be added; harness uses them)

@contextmanager
def experiment_run(logger: ExperimentLogger, name: str, track: str, phase: str = "2", **create_kwargs):
    """Context manager: auto log success/fail + duration."""
    exp_id = logger.create_experiment(name, track, phase, **create_kwargs)
    start = datetime.now(timezone.utc)
    success = False
    run_id = None
    try:
        yield exp_id
        success = True
    finally:
        wall = (datetime.now(timezone.utc) - start).total_seconds()
        run_id = logger.log_run(exp_id, wall_ms=wall*1000, success=success)
        if not success:
            logger.add_observation(run_id, "status", "error", "exception or timeout")
