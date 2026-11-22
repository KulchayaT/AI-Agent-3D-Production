"""Create data/assets.csv (synthetic proof-of-concept data, see qa_model/data.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qa_model.config import DATA_PATH  # noqa: E402
from qa_model.data import make_dataset  # noqa: E402

if __name__ == "__main__":
    df = make_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"wrote {len(df)} rows to {DATA_PATH}")
    print(df["label"].value_counts().rename({0: "PASS", 1: "FAIL"}).to_string())
