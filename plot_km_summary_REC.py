#!/usr/bin/env python3
"""Summary KM figure for treatment recommendation: TabSurv + all baselines.

For each gene-set scenario, draws one grid figure with a pooled 20-year
Kaplan-Meier panel (Followed vs Not Followed) per method: TabSurv first, then
the REC baselines in config order (DeepHS, DeepSurv, ENCox, LH, MTLR, PCHazard,
PMF, RSF, SurvITE, BITES). Each panel pools the test-set predictions saved by
tabsurv_REC.py / baselines_REC.py across the selected seeds.

Run after tabsurv_REC.py and baselines_REC.py have produced prediction files.

Examples
--------
    python plot_km_summary_REC.py                       # Nik_panel and allgenes
    python plot_km_summary_REC.py --scenario Nik_panel
    python plot_km_summary_REC.py --models RSF SurvITE BITES
"""

import argparse

from experiment_config import (
    SEEDS,
    REC_SCENARIOS,
    REC_PLOTS_DIR,
    REC_ALL_BASELINE_MODELS,
    selected_rec_scenarios,
)
from utils import plot_km_summary_grid

TABSURV_NAME = "TabSurv"

# Wording used in the figure title for each scenario.
SCENARIO_TITLES = {
    "Nik_panel": "the Nik panel",
    "Mills_panel": "the Mills panel",
    "Pereira_panel": "the Pereira panel",
    "allgenes": "all genes",
}


def _seed_files(prediction_dir, dataset, model, seeds):
    files = []
    for seed in seeds:
        path = prediction_dir / f"{dataset}_{model}_seed{seed}_predict.csv"
        if path.is_file():
            files.append((seed, str(path)))
        else:
            print(f"[KM summary] Missing: {path}")
    return files


def make_summary(scenario, seeds, models, include_tabsurv=True, ncols=3, title=None):
    cfg = REC_SCENARIOS[scenario]
    outputs = []
    for dataset in cfg["datasets"]:
        method_files = []
        if include_tabsurv and "TabSurv_REC" in cfg:
            method_files.append((
                TABSURV_NAME,
                _seed_files(cfg["TabSurv_REC"]["prediction_dir"], dataset, TABSURV_NAME, seeds),
            ))
        if "baselines_REC" in cfg:
            for model in models:
                method_files.append((
                    model,
                    _seed_files(cfg["baselines_REC"]["prediction_dir"], dataset, model, seeds),
                ))
        else:
            print(f"[KM summary] Scenario {scenario} has no baselines_REC config; TabSurv only.")

        suptitle = title or (
            "Kaplan–Meier survival curves for treatment recommendation using "
            f"{SCENARIO_TITLES.get(scenario, scenario)}"
        )
        suffix = "" if len(cfg["datasets"]) == 1 else f"_{dataset}"
        out_path = REC_PLOTS_DIR / f"KM_summary_{scenario}{suffix}_{len(seeds)}seeds_pooled.png"
        plotted = plot_km_summary_grid(method_files, suptitle, out_path, ncols=ncols)
        if plotted:
            outputs.append(out_path)
    return outputs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--scenario",
        choices=[*REC_SCENARIOS.keys(), "both"],
        default="both",
        help="Gene-set scenario. 'both' = Nik_panel + allgenes. Default: both.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help=f"Seeds to pool. Default: shared SEEDS={SEEDS}.")
    p.add_argument("--models", nargs="+", choices=REC_ALL_BASELINE_MODELS,
                   default=REC_ALL_BASELINE_MODELS, help="Baseline panels, in display order.")
    p.add_argument("--no-tabsurv", action="store_true", help="Leave out the TabSurv panel.")
    p.add_argument("--ncols", type=int, default=3, help="Panels per row. Default: 3.")
    p.add_argument("--title", default=None, help="Override the figure title.")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = list(SEEDS if args.seeds is None else args.seeds)
    written = []
    for scenario in selected_rec_scenarios(args.scenario):
        written += make_summary(
            scenario, seeds, list(args.models),
            include_tabsurv=not args.no_tabsurv, ncols=args.ncols, title=args.title,
        )
    if not written:
        print("[KM summary] No figures written (no prediction files found).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
