
# TabSurv: Tabular Foundation Model for Breast Cancer Prognosis using Gene Expression Data
A Python implementation of TabSurv: Foundation Model-Based Survival Analysis method.


# Infrastructure used to run experiments:
* OS: MacOS, version 26.6.2.
* CPU: Apple M5.
* RAM: 24 GB.

## Datasets

TabSurv is evaluated on 12 breast cancer cohorts with recurrence-free survival (**RFS**) or distant metastasis-free survival (**DMFS**) endpoints.

| Dataset | Sample size | Events | Event (%) | Min time | Max time | End point |
|---|---:|---:|---:|---:|---:|:---:|
| METABRIC | 1,980 | 647 | 32.68 | 0.00 | 29.60 | RFS |
| GEO | 736 | 349 | 47.40 | 0.00 | 18.52 | RFS |
| GSE6532 | 401 | 139 | 34.66 | 0.022 | 16.85 | RFS |
| GSE19783 | 100 | 43 | 43.00 | 0.69 | 10.62 | RFS |
| TCGA500 | 500 | 45 | 9.00 | 0.00 | 17.91 | RFS |
| UK | 207 | 77 | 37.20 | 0.39 | 10.00 | RFS |
| UPP | 235 | 54 | 23.08 | 0.08 | 12.75 | RFS |
| NKI | 320 | 109 | 34.17 | 0.02 | 18.35 | DMFS |
| MAINZ | 200 | 46 | 23.00 | 0.08 | 19.72 | DMFS |
| HEL | 115 | 25 | 21.74 | 0.00 | 5.00 | DMFS |
| UNT | 133 | 28 | 21.05 | 0.17 | 14.53 | DMFS |
| TRANSBIG | 198 | 62 | 31.31 | 0.34 | 29.60 | DMFS |

Due to GitHub file-size limits, the larger datasets (>25MB) are not distributed in this repository. They are available upon request.

For the survival experiments, **METABRIC** is the source cohort for the RFS scenario, with TCGA500, GEO, GSE6532, GSE19783, UK, and UPP used as external cohorts. **NKI** is the source cohort for the DMFS scenario, with HEL, UNT, TRANSBIG, and MAINZ used as external cohorts.

## Methods

The final survival experiments include three TabSurv variants (**TabSurv_M**, **TabSurv_P**, and **TabSurv_A**) and eight baselines:

- DeepHitSingle (DeepHS)
- DeepSurv
- Elastic-Net Cox (ENCox)
- LogisticHazard (LH)
- Multi-Task Logistic Regression (MTLR)
- Piecewise Constant Hazard (PCHazard)
- Probability Mass Function (PMF)
- Random Survival Forest (RSF)

**TabSurv_M** is the proposed method used for paired statistical comparisons with the survival baselines.

Treatment-recommendation experiments additionally include **SurvITE** and **BITES** as causal-survival baselines.

## Installation

The experiments were implemented in Python. Create a virtual environment and install the required packages:

```bash
python -m venv tabsurv_env
source tabsurv_env/bin/activate
pip install -r requirements.txt
```

The exact package versions used in the experiments are provided in `requirements.txt`.

TabSurv uses the TabPFN v2.5 regressor checkpoint. If a local checkpoint is available, specify it with:

```bash
export TABPFN_CKPT_PATH=/path/to/tabpfn-v2.5-regressor-v2.5_default.ckpt
```

Otherwise, on an internet-connected machine, TabPFN can resolve/download the checkpoint through its cache.

## Reproducing the Experiments

`run_all.py` is the main entry point for the experimental pipeline.

### Survival prediction

Run the final RFS and DMFS experiments under both InD and OOD settings:

```bash
python run_all.py --methods final-report --scenario both --setting both
```

This runs the three TabSurv configurations and all eight survival baselines, followed by evaluation and paired statistical comparisons where applicable.

To evaluate existing prediction files without retraining:

```bash
python run_all.py --methods final-report --scenario both --setting both --evaluation-only
```

### Treatment recommendation

Run TabSurv and all treatment-recommendation baselines for both gene-set scenarios:

```bash
python run_all.py --methods all-rec --rec-scenario both
```

The default treatment-recommendation experiments use five random seeds (`40, 41, 42, 43, 44`) and a 30% test split.

### Run all experiments

```bash
python run_all.py --methods everything --scenario both --setting both --rec-scenario both
```

Results and prediction files are written to the `output/` directory.

## Reproducibility

Shared experimental settings, dataset scenarios, random seeds, model lists, and output paths are defined in `experiment_config.py`. The master script `run_all.py` provides a common interface for model execution, evaluation, and statistical comparison.

    

