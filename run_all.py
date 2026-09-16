#!/usr/bin/env python3
"""Master runner for TabSurv survival and treatment-recommendation experiments.

For the final survival report, use:

    python run_all.py --methods final-report --scenario both

This runs TabSurv_M, TabSurv_P, TabSurv_A plus all eight survival-prediction baselines. InD and OOD
both report all three TabSurv variants plus the baselines; Stability_CI remains
OOD-only. It then runs paired Wilcoxon comparisons for TabSurv_M versus the
baselines for RFS-InD, RFS-OOD, DMFS-InD and DMFS-OOD
(and the secondary combined 10-cohort OOD analysis).
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from experiment_config import (
    BASELINE_MODELS,
    PROPOSED_METHOD,
    REC_ALL_BASELINE_MODELS,
    REC_TEST_SIZE,
    SEEDS,
    check_ckpt,
    tabpfn_model_source,
)

SCRIPT_DIR = Path(__file__).resolve().parent

OOD_METHODS = ["TabSurv_A", "TabSurv_M", "TabSurv_P", "baselines"]
FINAL_REPORT_METHODS = ["TabSurv_A", PROPOSED_METHOD, "TabSurv_P", "baselines"]
REC_METHODS = ["TabSurv_REC", "baselines_REC"]

METHOD_SCRIPTS = {
    "TabSurv_A": "tabsurv_A.py",
    "TabSurv_M": "tabsurv_M.py",
    "TabSurv_P": "tabsurv_P.py",
    "baselines": "baselines.py",
    "TabSurv_REC": "tabsurv_REC.py",
    "baselines_REC": "baselines_REC.py",
}


def _run(cmd, dry_run=False):
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return completed.returncode


def _expand_methods(requested):
    expanded = []
    for item in requested:
        if item == "all":
            expanded.extend(OOD_METHODS)
        elif item == "final-report":
            expanded.extend(FINAL_REPORT_METHODS)
        elif item == "all-rec":
            expanded.extend(REC_METHODS)
        elif item == "everything":
            expanded.extend(OOD_METHODS)
            expanded.extend(REC_METHODS)
        else:
            expanded.append(item)
    return list(dict.fromkeys(expanded))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument(
        "--scenario",
        choices=["RFS", "DMFS", "both"],
        default="both",
        help="Endpoint scenario(s) for survival experiments.",
    )
    p.add_argument(
        "--setting",
        choices=["InD", "OOD", "both"],
        default="both",
        help=(
            "Survival evaluation setting. InD runs held-out source-cohort evaluation; "
            "OOD runs external-cohort evaluation; both runs both. For TabSurv_M, OOD "
            "reuses the same 50% source-training split/model used for InD."
        ),
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "all", "final-report", "all-rec", "everything",
            "TabSurv_A", "TabSurv_M", "TabSurv_P", "baselines",
            "TabSurv_REC", "baselines_REC",
        ],
        default=["all"],
        help=(
            "'final-report' = TabSurv_M + TabSurv_P + TabSurv_A + all baselines; 'all' = all survival "
            "variants + baselines; 'all-rec' = REC methods; 'everything' = both families."
        ),
    )
    p.add_argument(
        "--baseline-models",
        nargs="+",
        choices=BASELINE_MODELS,
        default=BASELINE_MODELS,
        help="Survival baseline subset forwarded to baselines.py and statistics.",
    )
    p.add_argument(
        "--tabsurv-p-device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Compute device forwarded only to tabsurv_P.py.",
    )
    p.add_argument(
        "--tabsurv-p-predict-batch-size",
        type=int,
        default=32,
        help="Prediction batch size forwarded only to tabsurv_P.py.",
    )

    # Treatment recommendation ------------------------------------------------
    p.add_argument(
        "--rec-scenario",
        choices=["72genes", "allgenes", "both"],
        default="both",
        help="Treatment-recommendation gene-set scenario(s). Default: both.",
    )
    p.add_argument(
        "--rec-baseline-models",
        nargs="+",
        choices=REC_ALL_BASELINE_MODELS,
        default=REC_ALL_BASELINE_MODELS,
    )
    p.add_argument(
        "--rec-datasets",
        nargs="+",
        default=None,
        help="Optional custom REC dataset override; use only with a single --rec-scenario.",
    )
    p.add_argument("--rec-seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--rec-test-size", type=float, default=REC_TEST_SIZE)
    p.add_argument("--rec-full-reseed", action="store_true")

    # Reporting/control -------------------------------------------------------
    p.add_argument("--skip-evaluation", action="store_true", help="Skip evaluations.py.")
    p.add_argument("--skip-statistics", action="store_true", help="Skip paired Wilcoxon comparisons.")
    p.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Skip model fitting; evaluate existing prediction files, then run statistics when applicable.",
    )
    p.add_argument(
        "--statistics-only",
        action="store_true",
        help="Skip model fitting and evaluation; run statistics from existing evaluation files.",
    )
    p.add_argument(
        "--strict-statistics",
        action="store_true",
        help="Require all five seeds and identical InD patient sets before Wilcoxon reporting.",
    )
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _check_rec_causal_dependencies(selected_rec_models):
    selected = set(selected_rec_models)
    errors = []
    if "BITES" in selected and importlib.util.find_spec("geomloss") is None:
        errors.append("BITES requires Python package 'geomloss' (pip install geomloss).")
    if errors:
        raise RuntimeError("REC causal-baseline preflight failed:\n  - " + "\n  - ".join(errors))


def main():
    args = parse_args()
    methods = _expand_methods(args.methods)

    if args.tabsurv_p_predict_batch_size < 1:
        raise SystemExit("--tabsurv-p-predict-batch-size must be >= 1")
    if not 0.0 < args.rec_test_size < 1.0:
        raise SystemExit("--rec-test-size must be between 0 and 1")
    if args.rec_datasets is not None and args.rec_scenario == "both":
        raise SystemExit("--rec-datasets can only be used with --rec-scenario 72genes or allgenes")
    if args.evaluation_only and args.statistics_only:
        raise SystemExit("Choose at most one of --evaluation-only and --statistics-only.")

    failures = []
    skip_model_runs = args.evaluation_only or args.statistics_only

    # Resolve TabPFN model source before launching a long job. On Gadi, the
    # PBS environment sets TABPFN_REQUIRE_LOCAL=1/HF_HUB_OFFLINE=1, so this
    # remains a strict local-checkpoint preflight. On an internet-connected
    # machine, a missing local checkpoint falls back to TabPFN's cache/download.
    tabpfn_methods = {"TabSurv_A", "TabSurv_M", "TabSurv_P", "TabSurv_REC"}
    if not skip_model_runs and not args.dry_run and any(m in tabpfn_methods for m in methods):
        print(f"TabPFN model source: {tabpfn_model_source()}")

    if not skip_model_runs and not args.dry_run and "baselines_REC" in methods:
        _check_rec_causal_dependencies(args.rec_baseline_models)

    if not skip_model_runs:
        for method in methods:
            script = SCRIPT_DIR / METHOD_SCRIPTS[method]

            if method in OOD_METHODS:
                cmd = [sys.executable, str(script), "--scenario", args.scenario, "--setting", args.setting]
                if method == "TabSurv_P":
                    cmd += [
                        "--device", args.tabsurv_p_device,
                        "--predict-batch-size", str(args.tabsurv_p_predict_batch_size),
                    ]
                elif method == "baselines":
                    cmd += ["--models", *args.baseline_models]
            elif method == "TabSurv_REC":
                cmd = [
                    sys.executable, str(script),
                    "--scenario", args.rec_scenario,
                    "--seeds", *[str(x) for x in args.rec_seeds],
                    "--test-size", str(args.rec_test_size),
                ]
                if args.rec_datasets is not None:
                    cmd += ["--datasets", *args.rec_datasets]
                if args.rec_full_reseed:
                    cmd.append("--full-reseed")
            elif method == "baselines_REC":
                cmd = [
                    sys.executable, str(script),
                    "--scenario", args.rec_scenario,
                    "--seeds", *[str(x) for x in args.rec_seeds],
                    "--test-size", str(args.rec_test_size),
                    "--models", *args.rec_baseline_models,
                ]
                if args.rec_datasets is not None:
                    cmd += ["--datasets", *args.rec_datasets]
                if args.rec_full_reseed:
                    cmd.append("--full-reseed")
            else:
                raise RuntimeError(f"Unhandled method: {method}")

            rc = _run(cmd, dry_run=args.dry_run)
            if rc != 0:
                failures.append((method, rc))
                if not args.continue_on_error:
                    return rc

    ood_methods = [m for m in methods if m in OOD_METHODS]

    # Evaluation -------------------------------------------------------------
    evaluation_ran = False
    if not args.statistics_only and not args.skip_evaluation and ood_methods:
        eval_models = []
        for method in ood_methods:
            if method == "baselines":
                eval_models.extend(args.baseline_models)
            else:
                eval_models.append(method)
        eval_models = list(dict.fromkeys(eval_models))

        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "evaluations.py"),
            "--scenario", args.scenario,
            "--setting", args.setting,
            "--models", *eval_models,
        ]
        rc = _run(cmd, dry_run=args.dry_run)
        if rc != 0:
            failures.append(("evaluations", rc))
            if not args.continue_on_error:
                return rc
        else:
            evaluation_ran = True
    elif not args.statistics_only and not args.skip_evaluation and not ood_methods:
        print("\nNo survival/OOD methods selected; evaluations.py is not needed.")

    # Statistical comparison ------------------------------------------------
    # Automatically run when the selected survival experiment contains both
    # TabSurv_M and baselines. statistics-only also uses the requested baseline
    # subset against already-generated evaluation files.
    can_compare = PROPOSED_METHOD in ood_methods and "baselines" in ood_methods
    should_run_statistics = not args.skip_statistics and can_compare
    if should_run_statistics:
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "statistical_comparisons.py"),
            "--analysis", "all" if args.scenario == "both" else args.scenario,
            "--setting", args.setting,
            "--baselines", *args.baseline_models,
        ]
        if args.strict_statistics:
            cmd += ["--strict-seeds"]
            if args.setting in {"InD", "both"}:
                cmd += ["--strict-ind-pairing"]
        rc = _run(cmd, dry_run=args.dry_run)
        if rc != 0:
            failures.append(("statistical_comparisons", rc))
            if not args.continue_on_error:
                return rc
    elif not args.skip_statistics and ood_methods and not can_compare:
        print(
            f"\nPaired statistics skipped: select both {PROPOSED_METHOD} and baselines "
            "(or use --methods final-report)."
        )

    if failures:
        print("\nCompleted with failures:")
        for name, rc in failures:
            print(f"  - {name}: exit code {rc}")
        return 1

    print("\nAll requested runs/reports completed successfully.")
    if can_compare:
        scenario_names = ["RFS", "DMFS"] if args.scenario == "both" else [args.scenario]
        setting_names = ["InD", "OOD"] if args.setting == "both" else [args.setting]
        covered = ", ".join(f"{sc}-{st}" for sc in scenario_names for st in setting_names)
        print(
            f"Final survival report covers {covered} with "
            f"{PROPOSED_METHOD} as the proposed method."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
