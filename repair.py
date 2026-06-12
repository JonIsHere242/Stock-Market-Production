import os
import traceback
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

DATA_DIRS = [
    r"Data\PriceData",
    r"Data\RFpredictions"
]

def repair_file(file_path):
    print(f"Repairing: {file_path}")
    table = None

    # Strategy 1: PyArrow legacy dataset reader (works in many cases)
    try:
        table = pq.read_table(file_path, use_legacy_dataset=True)
        print("  -> Success with use_legacy_dataset=True")
    except TypeError:
        # use_legacy_dataset was removed in newer PyArrow versions; try plain read without threads
        try:
            table = pq.read_table(file_path, use_threads=False)
            print("  -> Success with use_threads=False")
        except Exception:
            pass
    except Exception:
        pass

    # Strategy 2: Read with pandas using PyArrow engine, which sometimes uses a different code path
    if table is None:
        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
            table = pq.Table.from_pandas(df)
            print("  -> Success with pd.read_parquet(engine='pyarrow')")
        except Exception:
            pass

    # Strategy 3: fastparquet (most tolerant)
    if table is None:
        try:
            df = pd.read_parquet(file_path, engine='fastparquet')
            table = pq.Table.from_pandas(df)
            print("  -> Success with fastparquet")
        except Exception:
            pass

    if table is None:
        print(f"  -> FAILED to read {file_path}. The file may be irrecoverable.")
        return False

    # Write back a clean copy – this removes the broken histogram
    try:
        pq.write_table(table, file_path)
        print(f"  -> Successfully rewrote clean file.")
        return True
    except Exception as e:
        print(f"  -> Failed to write repaired file: {e}")
        return False

def main():
    for data_dir in DATA_DIRS:
        if not os.path.isdir(data_dir):
            continue
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".parquet"):
                    full_path = os.path.join(root, f)
                    repair_file(full_path)

if __name__ == "__main__":
    main()


    