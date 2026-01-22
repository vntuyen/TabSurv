
# Tabular Foundation Model for Breast Cancer Prognosis using Gene Expression Data
A Python implementation of TabSurv: Foundation Model-Based Survival Analysis method.


# Infrastructure used to run experiments:
* High performance computing (HPC)
* GPU: NVIDIA GPU (Tesla V100S-PCIE-32 GB).
* RAM: 128 GB.

# Datasets

| Dataset   | Sample Size | # Events | Event (%) | Min Time | Max Time |
|-----------|-------------|----------|-----------|----------|----------|
| METABRIC  | 1,980       |  647     | 32.89%    | 0.00     | 29.60    |
| TCGA500   | 500         |  45      | 9.00%     | 0.0027   | 17.91    |
| GEO       | 736         |  349     | 47.40%    | 0.00     | 18.52    |
| GSE6532   | 401         |  139     | 34.66%    | 0.022    | 16.85    |
| GSE19783  | 100         |  43      | 43.00%    | 0.69     | 10.62    |
| HEL       | 115         |  25      | 21.74%    | 0.00     | 5.00     |
| UNT       | 133         |  28      | 21.05%    | 0.17     | 14.53    |
| NKI       | 320         |  109     | 34.17%    | 0.02     | 18.35    |
| TRANSBIG  | 198         |  62      | 31.31%    | 0.34     | 29.60    |
| UK        | 207         |  77      | 37.20%    | 0.39     | 10.00    |
| MAINZ     | 200         |  46      | 23.00%    | 0.08     | 19.72    |
| UPP       | 235         |  54      | 23.08%    | 0.08     | 12.75    |


Due to the large file sizes, the datasets (input) are not included in this repository. They are available upon request.

# Baselines: 7 methods

* LogisticHazard
* PMF (Probability Mass Function)
* DeepHitSingle
* PCHazard
* MTLR (Multi-Task Logistic Regression))
* DeepSurv
* RSF (Random Survival Forest)


# Installation
**Installation requirements:**

* Python = 3.10
* cuda = 12.3.2
* numpy 1.23.5
* pandas 1.5.3
* tabpfn 6.3.0
* pycox 0.3.0
* scikit-survival 0.25.0
* torchtuples 0.2.2
* lifelines 0.24.3
* scikit-learn 1.5.2
* scipy 1.15.3
* matplotlib 3.7.1
* seaborn 0.12.2

**Detailed Guidelines for Python Environment Setup**

***1. Create a Python Environment***

Load modules in the High Performance Computing (HPC) and Create a new python environment named tabsurv_env:
     
    module purge
    module load python3/3.10.4 cuda/12.3.2
    python -m venv ~/tabsurv_env
    source ~/tabsurv_env/bin/activate

***2. Install Python Packages:***
     Install essential Python packages using pip: (bash)
     
    pip install numpy pandas scikit-learn scipy matplotlib seaborn tabpfn pycox scikit-survival torchtuples lifelines




# Reproducing the Paper Results:


**1. Run 7 models for survival analysis for 12 datasets.**

    python baselines.py

**2. Run the TabSurv model for 12 datasets**

    python tabsurv.py
    
**3. Generate the evaluation results in the paper**

    python evaluations.py
