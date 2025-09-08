import argparse
import os
import sys
import subprocess

#!/usr/bin/env python3
"""
data.py

Download the HF dataset Skywork/Skywork-OR1-RL-Data, filter out code examples using heuristics,
and save the filtered dataset to disk.

Usage:
    python data.py --dataset Skywork/Skywork-OR1-RL-Data --output_dir ./data
"""


try:
    from datasets import load_dataset
    import pandas as pd
except Exception:
    # try to install required packages if missing
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "pandas", "huggingface-hub"])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Skywork/Skywork-OR1-RL-Data", help="HuggingFace dataset id")
    parser.add_argument("--output_dir", default=os.path.expanduser("~/data"), help="Directory to save filtered dataset")
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset} ...")
    ds = load_dataset(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        code_ds = ds['code']
    except Exception as e:
        print(f"Dataset has no 'code' split/column: {e}")
        sys.exit(1)
    try:
        df = code_ds.to_pandas()
        # shuffle then split 90/10
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n_train = int(len(df) * 0.9)
        train_df = df.iloc[:n_train]
        test_df = df.iloc[n_train:]
        save_train = os.path.join(args.output_dir, "train.parquet")
        save_test = os.path.join(args.output_dir, "test.parquet")
        print(f"Saving train ({len(train_df)}) -> {save_train} and test ({len(test_df)}) -> {save_test}")
        train_df.to_parquet(save_train, index=False)
        test_df.to_parquet(save_test, index=False)
    except Exception as e2:
        print("Fallback parquet save failed:", e2)
        raise


if __name__ == "__main__":
    main()