"""Generate a compact live status report from current processes and inventories."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

def processes():
    """Return the current process table; never rely on recycled scheduler PIDs."""
    out = subprocess.run(["ps", "-eo", "pid=,stat=,comm=,args="], capture_output=True,
                         text=True, check=True).stdout.splitlines()
    rows = []
    for line in out:
        parts = line.strip().split(None, 3)
        if len(parts) == 4:
            rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows


def matching(rows, predicate):
    return [(pid, stat) for pid, stat, comm, args in rows if predicate(comm, args)]

def main():
    progress = json.loads((OUT / "classification_progress.json").read_text())
    eta = json.loads((OUT / "tpl_tail_eta.json").read_text())
    rows = processes()
    jobs = {
        "TPL UB XGB": matching(rows, lambda c, a: "python" in c and "classify.py" in a and "--datasets tpl_ub" in a),
        "TPL B XGB": matching(rows, lambda c, a: "python" in c and "classify.py" in a and "--datasets tpl_b" in a),
        "deferred GPU scheduler": matching(rows, lambda c, a: "deferred_gpu_safe.sh" in a),
        "CPU factorial XGB": matching(rows, lambda c, a: "python" in c and "classify.py" in a and "--study factorial" in a and "--models xgb" in a),
    }
    lines = ["# Live benchmark status", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
             "## Queue health", "", "| Process | PID | State |", "|---|---:|---|"]
    for name, found in jobs.items():
        if found:
            value = ", ".join(f"{pid} ({stat})" for pid, stat in found)
        else:
            value = "none"
        lines.append(f"| {name} | {value} |")
    lines += ["", "## Completion", "", "| Study/model | Completed | Expected | Percent |", "|---|---:|---:|---:|"]
    for study in ("benchmark", "factorial"):
        for model, v in progress[study]["completion"].items():
            lines.append(f"| {study} {model} | {v['completed']} | {v['expected']} | {v['percent']:.2f}% |")
    lines += ["", "## TPL tail", "", f"- Missing cells: UB {eta['missing_cells']['tpl_ub']}, B {eta['missing_cells']['tpl_b']}",
              f"- Estimated remaining wall time: **{eta['wall_hours_if_both_workers_remain_active']} hours**",
              f"- Active cells: `{eta['active_cell']}`", ""]
    (OUT / "LIVE_STATUS.md").write_text("\n".join(lines))
    print("Wrote", OUT / "LIVE_STATUS.md")

if __name__ == "__main__":
    main()
