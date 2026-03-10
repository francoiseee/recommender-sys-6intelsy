"""
get_data.py — Download and prepare the dataset.
TODO (Week 1/2): Implement actual download logic once dataset is finalized.
"""

import os
import argparse

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download_data():
    """Download dataset from source."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[get_data] Data directory: {DATA_DIR}")
    # TODO: Add download logic here (e.g., requests, kaggle API, HuggingFace datasets)
    # Example:
    # from datasets import load_dataset
    # dataset = load_dataset("mind_small")
    # dataset.save_to_disk(DATA_DIR)
    print("[get_data] TODO: Implement download logic in Week 1/2.")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if data exists")
    args = parser.parse_args()

    if os.path.exists(DATA_DIR) and not args.force:
        print(f"[get_data] Data already exists at {DATA_DIR}. Use --force to re-download.")
    else:
        download_data()


if __name__ == "__main__":
    main()
