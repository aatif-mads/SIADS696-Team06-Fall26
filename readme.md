# SIADS 696 Team 06

Supervised modeling is in `source/workbench/696_Qunshan.ipynb`. The same code can be run from `source/python/696_Qunshan.py`. It predicts three-year overall survival for TCGA-KIRC from clinical variables and CIBERSORT immune proportions.

## Data

Raw files are in `data/raw`:

- `TCGA-CDR-SupplementalTableS1.xlsx`
- `TCGA.Kallisto.fullIDs.cibersort.relative.tsv`

The notebook trains on two processed files, which are already in `data/processed` for reference:

- `data/processed/kirc_patient_level_merged.csv` (533 patients)
- `data/processed/kirc_supervised_3year.csv` (404 patients)

On Google Drive those same files are in `SIADS_696_project/data/processed`.

## Run it in Google Colab

1. Upload `source/workbench/696_Qunshan.ipynb` to Colab, or open it from Drive.
2. Use a runtime that can install packages. In a Colab cell, run `!pip install -r requirements.txt`.
3. Run the notebook cells from the top.
4. Allow the Drive mount when Colab asks.
5. The first cell uses `MyDrive/SIADS_696_project` when it is running in Colab.
6. A later cell points `PROJECT_DIR` at the shared Drive shortcut. Leave that cell as it is when you are in Colab.

## Run it on your computer

From the repo folder, install the dependencies, then run the script:

```bash
pip install -r requirements.txt
python source/python/696_Qunshan.py
```

The script runs the same code cells as the notebook in `source/workbench`. It does not mount Google Drive. It finds `data/raw` in this repo, and if `data/processed` does not already contain the two training files, it creates them. Drive-only cells are skipped. The model search can take a long time.
