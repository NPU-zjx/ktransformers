"""Verify the four real runs; keep failed diagnostic attempts in the audit trail."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from diagnostics import collect


def read(path):
    return json.loads(path.read_text())


def verify(root):
    formal = root/"formal-03"
    torch = root/"torch-01"
    nsys = root/"nsys-01"
    ncu = root/"ncu-03"
    for run in (formal, torch, nsys, ncu):
        assert read(run/"completion.json")["exit_code"] == 0, run
    for run in (formal, torch, nsys):
        assert read(run/"ledger.json")["status"] == "complete"
        assert len(list((run/"phase").glob("phase-TP*.json"))) == 8
    assert len(list((torch/"phase").glob("*.trace.json.gz"))) == 32
    assert (torch/"derived/torch-calls.csv").stat().st_size > 0
    cov = read(nsys/"analysis/coverage.json")
    assert cov["gpu_devices"] == 8 and cov["kernel_events"] == cov["matched_launch_events"] > 0
    collect("ncu", ncu)
    with (ncu/"raw-metrics.csv").open() as src:
        headers, units, values = list(csv.reader(src))
    raw = dict(zip(headers, values))
    assert "_fused_qk_rmsnorm_rope_gate_kernel" in raw["Kernel Name"]
    assert "kt_extend" in " ".join(values)
    selected = {}
    for key in ["gpu__time_duration.sum","sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__warps_active.avg.pct_of_peak_sustained_active"]:
        v = float(raw[key].replace(",",""))
        assert math.isfinite(v) and v > 0, (key, raw[key])
        selected[key] = dict(value=v, unit=units[headers.index(key)])
    (ncu/"selected-metrics.json").write_text(json.dumps(selected, indent=2))
    passes = sorted(ncu.glob("pass-*"))
    assert passes
    for p in passes:
        ledger = read(p/"ledger.json")
        assert ledger["status"] == "complete" and ledger["requests"] == 32
        workers = ledger["workers"]
        assert len(workers)==8 and all(w["replays"]>0 for w in workers)
    files = [formal/"analysis/client-summary.json", formal/"phase/summary.json",
             formal/"derived/resource-summary.json", torch/"derived/torch-calls.csv",
             nsys/"rollout.nsys-rep", nsys/"rollout.sqlite", ncu/"hotspot.ncu-rep",
             ncu/"raw-metrics.csv"]
    result = dict(formal_rollouts=1, formal="formal-03", torch_traces=32,
                  nsys_kernels=cov["kernel_events"], ncu_diagnostic_application_passes=len(passes),
                  ncu_kernel=raw["Kernel Name"], ncu_selected_metrics=selected,
                  evidence_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  recovered=["formal-01: Ray import", "formal-02: callable serialization",
                             "torch-01: outer launcher exit 2; container exit 0; traces parsed offline",
                             "ncu-01 and ncu-02: kernel replay UnknownError, exit 9"])
    (root/"suite-verification.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
