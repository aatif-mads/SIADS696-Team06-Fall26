"""Run the supervised notebook in Colab or as a local script.

From the repo folder:

    pip install -r requirements.txt
    python source/python/696_Qunshan.py

In Colab, open source/workbench/696_Qunshan.ipynb and run the cells from the top.
This script runs those same code cells. On a computer without Google Drive
it reads data/raw and creates the processed files if they are not there yet.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

NOTEBOOK = Path(__file__).resolve().parents[1] / "workbench" / "696_Qunshan.ipynb"
SHORTCUT = Path(
    "/content/drive/.shortcut-targets-by-id/1Etb6PgEnIDI846xzjqpiUS4nA8LNJh3f/SIADS_696_project"
)


def running_in_colab():
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def prepare_cell(source, in_colab):
    text = source.strip()
    if not text:
        return None

    if not in_colab and "drive.mount" in text and "PROJECT_DIR = Path('/content/drive/MyDrive/SIADS_696_project')" not in text and "search_folders" not in text:
        print("Skipping Drive mount. This computer is not running Colab.")
        return None

    if not in_colab and "/content/drive/MyDrive" in text and "os.walk" in text:
        print("Skipping the Drive file search.")
        return None

    if "shortcut-targets-by-id" in text and not SHORTCUT.exists():
        print("Drive shortcut was not found. Keeping the local project folder.")
        return "from pathlib import Path\nimport pandas as pd\n"

    lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("%pip") or stripped.startswith("!pip"):
            lines.append("# skipped in the script: " + stripped)
            continue
        if stripped.startswith("%"):
            lines.append("# skipped in the script: " + stripped)
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = text.replace(
        "from google.colab import files",
        "files = None",
    )
    text = text.replace(
        "from IPython.display import display",
        "pass  # display is already defined",
    )
    text = re.sub(
        r"files\.download\((.*?)\)",
        r'print("Saved", \1)',
        text,
    )
    return text


def ensure_processed_files(namespace):
    """Create the two training files from data/raw when they are missing."""
    import numpy as np
    import pandas as pd

    raw_dir = namespace["RAW_DIR"]
    processed_dir = namespace["PROCESSED_DIR"]
    merged_path = processed_dir / "kirc_patient_level_merged.csv"
    supervised_path = processed_dir / "kirc_supervised_3year.csv"
    if merged_path.exists() and supervised_path.exists():
        print("Using processed files in", processed_dir)
        return

    print("Processed files were not found. Building them from data/raw.")
    clinical = pd.read_excel(raw_dir / "TCGA-CDR-SupplementalTableS1.xlsx", sheet_name="TCGA-CDR")
    immune = pd.read_csv(raw_dir / "TCGA.Kallisto.fullIDs.cibersort.relative.tsv", sep="\t")

    clinical = clinical.loc[clinical["type"] == "KIRC"].copy()
    clinical = clinical.rename(columns={"bcr_patient_barcode": "patient_id"})
    immune = immune.loc[immune["CancerType"] == "KIRC"].copy()
    immune["sample_barcode"] = immune["SampleID"].astype(str).str.replace(".", "-", regex=False)
    immune["patient_id"] = immune["sample_barcode"].str.extract(r"^(TCGA-[^-]+-[^-]+)")
    immune["sample_type"] = immune["sample_barcode"].str.extract(r"^TCGA-[^-]+-[^-]+-(\d{2})")
    immune = immune.loc[immune["sample_type"] == "01"].copy()
    immune = immune.sort_values(
        ["patient_id", "P.value", "RMSE", "Correlation", "SampleID"],
        ascending=[True, True, True, False, True],
    )
    immune = immune.drop_duplicates("patient_id", keep="first")

    clinical = clinical[
        [
            "patient_id",
            "age_at_initial_pathologic_diagnosis",
            "gender",
            "ajcc_pathologic_tumor_stage",
            "histological_grade",
            "OS",
            "OS.time",
        ]
    ].copy()
    clinical = clinical.replace(["[Not Available]", "[Discrepancy]", "GX"], np.nan)
    for column in ["age_at_initial_pathologic_diagnosis", "OS", "OS.time"]:
        clinical[column] = pd.to_numeric(clinical[column], errors="coerce")

    merged = clinical.merge(immune, on="patient_id", how="inner", validate="one_to_one")
    died = (merged["OS"] == 1) & (merged["OS.time"] <= 365 * 3)
    lived = (merged["OS.time"] > 365 * 3) | ((merged["OS"] == 0) & (merged["OS.time"] >= 365 * 3))
    merged["survival_3y"] = np.select([died, lived], [0, 1], default=np.nan)
    supervised = merged.loc[merged["survival_3y"].notna()].copy()
    supervised["survival_3y"] = supervised["survival_3y"].astype(int)

    processed_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_path, index=False)
    supervised.to_csv(supervised_path, index=False)
    print("Saved", merged_path.name, len(merged))
    print("Saved", supervised_path.name, len(supervised))


def run(stop_after=None):
    notebook = json.loads(NOTEBOOK.read_text())
    in_colab = running_in_colab()
    namespace = {"__name__": "__main__"}

    def display(value):
        print(value)

    namespace["display"] = display
    prepared = False

    code_index = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        runnable = prepare_cell(source, in_colab)
        if runnable is None:
            code_index += 1
            continue
        if (not prepared) and "kirc_supervised_3year.csv" in runnable and "PROCESSED_DIR" in namespace:
            ensure_processed_files(namespace)
            prepared = True
        print(f"\n--- notebook code cell {code_index} ---")
        exec(compile(runnable, f"cell_{code_index}", "exec"), namespace)
        if stop_after is not None and code_index >= stop_after:
            print(f"Stopped after notebook code cell {code_index}.")
            return namespace
        code_index += 1
    return namespace


if __name__ == "__main__":
    stop = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("--until="):
        stop = int(sys.argv[1].split("=", 1)[1])
    run(stop_after=stop)
