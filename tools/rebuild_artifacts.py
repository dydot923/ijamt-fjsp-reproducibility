"""Regenerate the paper's numerical tables and 11 figures inside this project."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT.parent / "04_修订与核查/工具/python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path[:0] = [str(ROOT / "replication"), str(ROOT / "figures"), str(ROOT / "tables")]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document


def support_figures(old):
    step = pd.read_csv(ROOT / "replication/results_support/single_step.csv")
    means = step.groupby(["parent_seed", "focus", "version"]).relative_target.mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(old.WIDTH, 3.0))
    fig.subplots_adjust(left=.09, right=.975, top=.84, bottom=.22, wspace=.45)
    for index, (ax, focus, title) in enumerate(zip(axes, ("Cmax", "TEC", "Lmax"), ("完工方向", "能源方向", "占用方向"))):
        old.blue_boxplot(ax, [100 * means[(means.focus == focus) & (means.version == version)].relative_target
                             for version in ("original", "matched")])
        ax.set(xticks=(1, 2), xticklabels=("Original", "Matched"))
        ax.axhline(0, color="gray", linewidth=.4, linestyle="--")
        if index == 0:
            ax.set_ylabel("Target objective change / %")
        old.panel(ax, chr(97 + index), title)
    fig.text(.09, .04, "Each box summarizes 10 common parents; 20 proposals per direction and mechanism.", fontsize=8)
    old.save(fig, "fig10_reproduced_single_step")

    rows = json.loads((ROOT / "replication/results_support/factorial_runs.json").read_text("utf8"))
    data = {}
    for scenario in ("CF0", "E1", "CFM", "E2"):
        data[scenario] = np.array([np.asarray(row["front"]).min(axis=0) for row in sorted(
            [r for r in rows if r["instance"] == scenario], key=lambda r: r["seed"])])
    fig, axes = plt.subplots(1, 3, figsize=(old.WIDTH, 3.1))
    fig.subplots_adjust(left=.09, right=.975, top=.84, bottom=.26, wspace=.6)
    for ax, column, title, ylabel in zip(axes[:2], (0, 2), ("完工端点", "占用端点"),
                                       (r"$C_{max}^{min}$ / min", r"$L_{max}^{min}$ / min")):
        for names, color, marker, label in ((("CF0", "E1"), "#0000FF", "o", "No maintenance"),
                                            (("CFM", "E2"), "#FF0000", "s", "Maintenance")):
            values = [data[name][:, column] for name in names]
            ax.errorbar((0, 1), [v.mean() for v in values], yerr=[v.std(ddof=1) for v in values],
                        color=color, marker=marker, mfc="none", lw=.65, ms=4, capsize=2, label=label)
        ax.set(xticks=(0, 1), xticklabels=("No setup", "Setup"), ylabel=ylabel)
        old.panel(ax, "a" if column == 0 else "b", title)
    axes[0].legend(fontsize=7)
    interaction = data["E2"] - data["E1"] - data["CFM"] + data["CF0"]
    rng = np.random.default_rng(20261007)
    sampled = interaction[rng.integers(0, 12, (20000, 12))].mean(axis=1)
    for x, column, color in ((0, 0, "#0000FF"), (1, 2, "#009600")):
        values = interaction[:, column]
        lo, hi = np.quantile(sampled[:, column], [.025, .975])
        axes[2].plot(np.full(12, x), values, "o", mfc="none", mec=color, ms=3)
        axes[2].errorbar(x, values.mean(), yerr=[[values.mean() - lo], [hi - values.mean()]],
                         marker="D", color=color, mfc="none", ms=4, capsize=3)
    axes[2].axhline(0, color="gray", lw=.4, linestyle="--")
    axes[2].set(xticks=(0, 1), xticklabels=(r"$C_{max}^{min}$", r"$L_{max}^{min}$"), ylabel="Interaction / min", xlim=(-.5, 1.5))
    old.panel(axes[2], "c", "配对交互")
    fig.text(.09, .035, "(a, b) Mean ± SD. (c) Twelve seeds and paired 95% bootstrap intervals.", fontsize=8)
    old.save(fig, "fig11_reproduced_constraints")


def main():
    import analyze_upgrade
    analyze_upgrade.main()
    import make_upgrade_tables
    make_upgrade_tables.main()
    manuscript = ROOT.parent / "01_当前稿/三目标柔性作业车间调度_理论与叙事增强稿_20261029.docx"
    if manuscript.exists():
        for i, table in enumerate(Document(manuscript).tables, 1):
            with (ROOT / "tables" / f"Table{i:02d}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                csv.writer(stream).writerows([["".join(cell._tc.xpath(".//w:t/text()|.//m:t/text()"))
                                              for cell in row.cells] for row in table.rows])
    import make_figures as old
    import make_upgrade_figures as new
    destination = ROOT / "figures/reproduced"
    destination.mkdir(exist_ok=True)
    old.F = destination
    new.F = destination
    (destination / "data").mkdir(exist_ok=True)
    # Outlined SVG glyphs prevent browser font substitution. PDF retains
    # embedded fonts and the plotting sources retain editable text labels.
    plt.rcParams["svg.fonttype"] = "path"
    original_save = old.save

    def export(fig, name):
        original_save(fig, name)
        fig.savefig(destination / f"{name}.png", dpi=600, facecolor="white")
        fig.savefig(destination / f"{name}.eps", dpi=600, facecolor="white")
        old.REPORT[-1]["dpi"] = 600

    old.save = export
    # Preserve complete SD whiskers in the small-instance HV-ratio panel.
    oracle_code = Path(new.__file__).read_text("utf8").replace("lows=[]", "lows=[]; highs=[]")
    oracle_code = oracle_code.replace("lows.extend((g.mean()-g.std()).tolist())",
                                      "lows.extend((g.mean()-g.std()).tolist()); highs.extend((g.mean()+g.std()).tolist())")
    oracle_code = oracle_code.replace("/1000,1.005)", "/1000,max(1.005, max(highs)+.001))")
    # Execute the original source in its module namespace; only the view limit
    # differs, with all measured data and statistics unchanged.
    exec(compile(oracle_code, new.__file__, "exec"), new.__dict__)
    new.F = destination
    old.toy()
    old.mechanism()
    frame = pd.read_csv(ROOT / "replication/results_v3/metrics.csv")
    new.validation()
    new.oracle()
    new.confirm(frame)
    new.benchmarks(frame)
    new.budgets(frame)
    new.tradeoffs()
    new.schedule()
    support_figures(old)
    assert len(old.REPORT) == 11
    mapping = ["fig1_encoding", "fig2_mechanism", "fig3_upgrade_validation", "fig4_upgrade_oracle",
               "fig5_upgrade_confirmation", "fig7_upgrade_benchmarks", "fig6_upgrade_budget",
               "fig8_upgrade_tradeoffs", "fig9_upgrade_schedule", "fig10_reproduced_single_step", "fig11_reproduced_constraints"]
    records = []
    for i, name in enumerate(mapping, 1):
        record = {"figure": i, "original_stem": name, "files": {}}
        for ext in ("png", "svg", "pdf", "eps"):
            path = destination / f"{name}.{ext}"
            final = destination / f"Fig{i:02d}.{ext}"
            path.replace(final)
            record["files"][ext] = {"file": final.name, "sha256": hashlib.sha256(final.read_bytes()).hexdigest()}
        records.append(record)
    (ROOT / "reports/rebuilt-artifacts.json").write_text(json.dumps({"figures": records, "layout_audit": old.REPORT,
        "timing_note": "Figure 3 uses the fresh serial timing; manuscript Figure 3 retains the separately identified original batch",
        "manuscript_assets_changed": False}, ensure_ascii=False, indent=2), "utf8")
    print("Rebuilt all 13 table CSVs and 11 figures in PNG/SVG/PDF/EPS inside the new code project")


if __name__ == "__main__":
    main()
