"""Private, bounded HTTP facade for the existing synthetic CPU runner.

No arbitrary code, uploaded scenarios, training, GPU or real MaleCNS execution.
Deploy behind Cloud Run IAM; the bearer secret is an additional application gate.
"""
from __future__ import annotations

import csv
import hmac
import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict

RECIPES = {
    "loom": "00_no_train_loom_escape",
    "saber": "01_photon_saber_readout",
    "market": "02_market_arena_synthetic",
}
ROOT = Path(__file__).resolve().parents[1]


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipe: Literal["loom", "saber", "market"]


def create_app(token: str | None = None) -> FastAPI:
    secret = token or os.environ.get("CNS_API_TOKEN", "")
    if len(secret) < 32:
        raise RuntimeError("CNS_API_TOKEN must contain at least 32 characters")
    app = FastAPI(title="CNS Engine private preview", docs_url=None, redoc_url=None,
                  openapi_url=None)
    gate = threading.Lock()

    def authorize(authorization: str = Header(default="")) -> None:
        if not hmac.compare_digest(authorization.encode(), ("Bearer " + secret).encode()):
            raise HTTPException(401, "Authentication required")

    @app.get("/healthz")
    def health():
        return {"status": "ok", "runtime": "synthetic-cpu-preview"}

    @app.get("/v1/capabilities", dependencies=[Depends(authorize)])
    def capabilities():
        return {"recipes": list(RECIPES), "training": False, "gpu": False,
                "malecns_mounted": False, "signal": "leaky-rate activity, not spikes",
                "max_parallel_runs_per_instance": 1, "seed": 7,
                "arbitrary_tasks": False, "biological_validation": False}

    @app.post("/v1/runs", dependencies=[Depends(authorize)])
    def execute(request: RunRequest):
        if not gate.acquire(blocking=False):
            raise HTTPException(429, "Runner busy; retry later")
        try:
            with tempfile.TemporaryDirectory(prefix="cns-preview-") as folder:
                recipe = ROOT / "recipes" / RECIPES[request.recipe] / "scenario.yaml"
                try:
                    subprocess.run(
                        [sys.executable, "-m", "cns_tinker.cli", "run", str(recipe),
                         "--out", folder], cwd=ROOT, check=True, timeout=45,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                except subprocess.TimeoutExpired:
                    raise HTTPException(504, "Run exceeded the 45-second limit") from None
                except subprocess.CalledProcessError:
                    raise HTTPException(500, "Run failed") from None
                out = Path(folder)
                manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
                with (out / "action_trace.csv").open(encoding="utf-8", newline="") as handle:
                    actions = list(csv.DictReader(handle))
                with (out / "stimulus_trace.csv").open(encoding="utf-8", newline="") as handle:
                    stimuli = list(csv.DictReader(handle))
                with np.load(out / "neural_trace.npz", allow_pickle=False) as trace:
                    # Every channel and original time point is retained, with explicit units.
                    activity = trace["activity"].tolist()
                    time_ms = trace["t_ms"].tolist()
                return {"run_id": uuid.uuid4().hex, "mode": "executed-synthetic-preview",
                        "manifest": manifest, "seed": 7, "actions": actions,
                        "recipe": request.recipe, "stimuli": stimuli,
                        "stimulus_units": "normalized synthetic inputs; t_ms in milliseconds",
                        "signal": {"kind": "leaky-rate", "units": "arbitrary",
                                   "time_units": "ms", "t_ms": time_ms,
                                   "channels": [f"synthetic_{i}" for i in range(len(activity[0]))],
                                   "activity": activity},
                        "retention": "response only; temporary files removed"}
        finally:
            gate.release()

    return app
