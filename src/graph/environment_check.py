from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT_DIR / "reports" / "environment" / "g1_dependency_stack.json"

FRAMEWORK_DECISION = "hand_rolled_pytorch"
FRAMEWORK_ALTERNATIVES_CONSIDERED = ["pytorch_geometric", "deep_graph_library"]
FRAMEWORK_DECISION_RATIONALE = (
    "The temporal contract (src/graph/temporal_contract.py) rules out using a "
    "library's stock neighbour sampler: hop-2 admissibility is target-anchored, "
    "not the intermediate node's own timestamp, which is not how a generic "
    "k-hop sampler works. The sampler is hand-written regardless of the "
    "framework, and a GraphSAGE mean-aggregator is small enough to hand-roll "
    "directly on tensors, which removes a dependency whose sampler semantics "
    "would otherwise need auditing for temporal safety on top of the one being "
    "written anyway. This environment runs Python 3.14.0, ahead of what the "
    "graph-library ecosystem has caught up with: torch_geometric's own wheel is "
    "pure Python and installs cleanly, but its accelerated extensions "
    "(torch-scatter, pyg-lib) are not published on PyPI at all -- they ship "
    "from a custom index keyed to specific torch/CUDA build combinations, and "
    "no build exists yet for torch 2.14 or cp314 -- so the richer sampler "
    "ecosystem PyG is normally chosen for is not actually usable here. DGL is "
    "not published as an installable PyPI wheel at all (only two ancient, "
    "unrelated 0.1.x releases exist under that name). Plain torch==2.14.0 has "
    "an official cp314 win_amd64 wheel on PyPI and was verified to install and "
    "run a forward/backward pass in this environment. No CUDA device is "
    "available here, so the CPU build is used."
)
PYTHON_314_BLOCKED_CHOSEN_STACK = False


def run_minimal_forward_backward_pass() -> dict[str, Any]:
    """A tiny 2-layer MLP forward+backward pass, used to verify the install."""
    torch.manual_seed(0)
    x = torch.randn(8, 4, requires_grad=True)
    w1 = torch.randn(4, 6, requires_grad=True)
    w2 = torch.randn(6, 1, requires_grad=True)
    hidden = torch.relu(x @ w1)
    output = hidden @ w2
    loss = output.sum()
    loss.backward()

    if x.grad is None or w1.grad is None or w2.grad is None:
        raise AssertionError("Backward pass did not populate gradients.")
    grads = (x.grad, w1.grad, w2.grad)
    if not all(torch.isfinite(grad).all() for grad in grads):
        raise AssertionError("Backward pass produced non-finite gradients.")

    return {
        "forward_output_shape": list(output.shape),
        "loss": float(loss.item()),
        "w2_grad_norm": float(w2.grad.norm().item()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def build_report() -> dict[str, Any]:
    pass_result = run_minimal_forward_backward_pass()
    return {
        "framework_decision": FRAMEWORK_DECISION,
        "alternatives_considered": FRAMEWORK_ALTERNATIVES_CONSIDERED,
        "decision_rationale": FRAMEWORK_DECISION_RATIONALE,
        "python_3_14_blocked_chosen_stack": PYTHON_314_BLOCKED_CHOSEN_STACK,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "forward_backward_pass": pass_result,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    report = build_report()
    write_json(REPORT_PATH, report)
    print(f"Framework decision: {report['framework_decision']}")
    print(f"torch version: {report['versions']['torch']}")
    print(f"Forward/backward pass OK: {report['forward_backward_pass']}")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
