# SIADS 696 Team 06

Link to Final Report Google Docs Link: https://docs.google.com/document/d/1fEONCNopRJT-BAK175rt4tA-QZsGhMV0eOzdT0e9y8E/edit?usp=sharing

This project asks whether tumor-infiltrating immune cell proportions improve the prediction of three-year overall survival in kidney renal clear cell carcinoma (TCGA-KIRC) beyond clinical variables. It uses TCGA clinical and survival data together with CIBERSORT estimates of 22 immune cell types.

Team: Qunshan Jia, Fang Yuan, Affan Atif

## Repository layout

```text
data/
  raw/          the two input files
  processed/    patient-level files built from the raw data
figures/        figures used in the report
source/
  workbench/    Colab notebooks (.ipynb)
  python/       Python versions of the notebooks (.py)
requirements.txt
```

Each team member's work has a notebook in `source/workbench` and a matching file in `source/python`:

| File | Contents |
|---|---|
| `696_Fang` | Data cleaning, exploratory analysis, and the supervised models with nested cross-validation |
| `696_Qunshan` | Supervised models and failure analysis |
| `696_Affan` | Unsupervised analysis: k-means and hierarchical clustering on the immune profiles, with a PCA check |

## Data

Raw files are in `data/raw`:

- `TCGA-CDR-SupplementalTableS1.xlsx` (clinical and survival data, sheet `TCGA-CDR`)
- `TCGA.Kallisto.fullIDs.cibersort.relative.tsv` (CIBERSORT immune cell proportions)

Two processed files are in `data/processed` for reference:

- `kirc_patient_level_merged.csv`: 533 KIRC patients, one primary tumor sample each
- `kirc_supervised_3year.csv`: the 404 of those patients with a known three-year outcome

The supervised models use the 404-patient file. The unsupervised analysis uses all 533 patients.

## Install

From the repo folder:

```bash
pip install -r requirements.txt
```

In Colab, run `!pip install -r requirements.txt` in a cell, or let each notebook install what it needs.

## Run on local

Every file checks whether it is running in Colab. Outside Colab it skips Google Drive and reads `data/raw` from this repo. From the repo folder:

```bash
python source/python/696_Fang.py
python source/python/696_Qunshan.py
python source/python/696_Affan.py
```

The notebooks in `source/workbench` also run locally in Jupyter when they are opened from inside the repo.

Outputs are written inside the repo folder:

- `696_Fang` writes the two processed files to `data/processed`, and its figures and tables to `outputs/`.
- `696_Qunshan` builds the two processed files if they are missing. Its model search can take a long time.
- `696_Affan` writes its figures, tables, and cluster labels to `SIADS_696_outputs/`.

## Run in Google Colab

1. Open the notebook from `source/workbench` in Colab.
2. Allow the Google Drive mount when Colab asks.
3. Make sure the data files are in the Drive folder that notebook expects:
   - `696_Fang` and `696_Qunshan` read from `MyDrive/SIADS_696_project/data/raw` and write to `data/processed` and `outputs` in that folder.
   - `696_Affan` reads the two raw files from the top level of `MyDrive` and writes results to `MyDrive/SIADS_696_outputs`.
4. Run the cells from the top.

`696_Qunshan` has a later cell that points `PROJECT_DIR` at the team's shared Drive shortcut. Leave that cell as it is when running in Colab.
