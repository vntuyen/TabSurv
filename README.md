
# TabSurv: Tabular Foundation Model for Breast Cancer Prognosis using Gene Expression Data
A Python implementation of TabSurv: Foundation Model-Based Survival Analysis method.


# Infrastructure used to run experiments:
* OS: Ubuntu, version 25.10.
* CPU: Intel(R) Core(TM) Ultra 7 268V   2.20 GH.
* RAM: 32 GB.

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


Due to the large file sizes (> 25MB), the big datasets (METABRIC, TCGA500, GEO, GSE6532, NKI, UPP, UK, MAINZ, TRANSBIG) are not included in this repository. They are available upon request.

# Baselines: 7 methods

* LogisticHazard
* PMF (Probability Mass Function)
* DeepHS (DeepHitSingle)
* PCHazard
* MTLR (Multi-Task Logistic Regression))
* DeepSurv
* RSF (Random Survival Forest)


# Installation
**Installation requirements:**

* Python==3.13
* huggingface-hub==0.36.0
* lifelines==0.30.0
* numpy==2.1.3
* pandas==2.3.3
* pycox==0.3.0
* scikit-learn==1.6.1
* scikit-survival==0.25.0
* scipy==1.15.3
* sklearn-pandas==2.2.0
* sympy==1.14.0
* tabpfn==6.3.0
* tabpfn-extensions==0.2.2
* tensorboard==2.20.0
* torch==2.7.1
* torchmetrics==1.7.4
* torchtuples==0.2.2
* torchvision==0.22.1
* transformers==4.49.0


**Detailed Guidelines for Python Environment Setup**

***1. Create a Python Environment***

Create a new python environment named tabsurv_env:
     
    python -m venv ~/tabsurv_env
    source ~/tabsurv_env/bin/activate

***2. Install Python Packages:***
     Install essential Python packages using pip: (bash)
     
    pip install -r requirements.txt




# Reproducing the Paper Results:

***1. Survival Prediction for Prognosis.***


 Run 7 models for survival analysis for 12 datasets.

    python baselines_OOD.py

 Run the TabSurv model for 12 datasets 

    python tabsurv_OOD.py

 Generate the evaluation results in the paper

    python evaluations.py

    
***2. Treatment Recommendation Evaluation***

    python tabsurv_REC.py
    python baselines_REC.py
    

