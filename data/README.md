# Data

## Dataset

**Planned Dataset:** [MIND (Microsoft News Dataset)](https://msnews.github.io/) or [MovieLens](https://grouplens.org/datasets/movielens/)

> ⚠️ **No raw data is stored in this repository.** Run the download script to obtain data locally.

## How to Obtain Data

```bash
python data/get_data.py
```

This script will:
1. Download the dataset from the official source
2. Extract and organize files into `data/raw/`
3. Run basic integrity checks

## Dataset Details (to be finalized in Week 1 Proposal)

| Field | Details |
|-------|---------|
| Source | TBD (MIND / MovieLens / other) |
| License | TBD |
| Size | TBD |
| PII / Consent | No PII; publicly available research dataset |
| Representativeness | To be analyzed in EDA notebook |

## Splits

| Split | Size | Notes |
|-------|------|-------|
| Train | ~70% | Stratified |
| Val   | ~15% | Stratified |
| Test  | ~15% | Held out — no leakage |

## Notes

- Raw data files are **gitignored** (see `.gitignore`)
- Processed/intermediate files are saved to `data/processed/` (also gitignored)
- Only this README and the download script are version-controlled
