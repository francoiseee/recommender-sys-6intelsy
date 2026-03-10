# Model Card — Personalized Recommendation with Contextual Bandits

> **Status:** Draft (Week 1) — to be completed by Week 3

---

## Model Details

| Field | Details |
|-------|---------|
| Model Name | Text-CNN + Contextual Bandit Recommender |
| Version | v0.1 (Proposal) |
| Type | Ranking / Recommendation |
| Developed By | 6INTELSY Team (Holy Angel University) |
| Date | 2025–2026 |
| Framework | PyTorch |

---

## Intended Use

- **Primary use:** Offline simulation of personalized news/item recommendation
- **Intended users:** Researchers and students studying recommender systems
- **Out-of-scope:** Not intended for production deployment or real user data

---

## Dataset

| Field | Details |
|-------|---------|
| Source | TBD (MIND / MovieLens) |
| License | TBD |
| PII | None — publicly available research dataset |
| Splits | 70% train / 15% val / 15% test |

---

## Model Architecture

- **CNN Component:** Text-CNN (Kim 2014) for item text feature extraction
- **NLP Component:** Tokenizer + embedding layer for text preprocessing
- **RL Component:** ε-Greedy / LinUCB Contextual Bandit for adaptive ranking
- **Core DL:** Embedding-based ranker (dot product scoring)

---

## Metrics

*(To be filled after evaluation — Week 3)*

| Metric | Value |
|--------|-------|
| nDCG@10 | TBD |
| Hit@10 | TBD |
| RL Cumulative Reward | TBD |

---

## Ethical Considerations

- **Filter Bubbles:** Bandit exploration helps mitigate over-personalization
- **Privacy:** No real user behavioral data is collected
- **Consent:** Dataset is publicly available with no PII
- **Fairness:** Exposure fairness across items to be evaluated (stretch goal)

---

## Caveats & Limitations

- Offline simulation does not reflect real-world online performance
- RL reward is simulated; real click data not available
- Model is trained/evaluated on a single dataset — generalization not guaranteed

---

## Deployment Guidance

> ⚠️ This model is for academic use only. Not for production deployment.
