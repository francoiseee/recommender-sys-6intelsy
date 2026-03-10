# Personalized Recommendation with Contextual Bandits
### 6INTELSY – Intelligent Systems | Final Project | CS - 302

> **Task:** Rank items/news using text embeddings; adapt recommendations via contextual bandit (NLP + RL).

---

## Team Members

|              Name                |            Role            |
|----------------------------------|----------------------------|
| Gurango, Christine Francoise O.  | Project Lead / Integration |
| Apostol, Lance Jezreel B.        | Data & Ethics Lead         |
| Maninang, Allein Dane G.         | Modeling Lead              |
| Parungao, Nikko S.               | Evaluation & MLOps Lead    |

---

## Project Overview

This project builds a **personalized recommendation system** that:
1. Encodes item/news text using **NLP embeddings** (CNN-based or transformer-based)
2. Ranks items using an **embedding-based ranker**
3. Adapts recommendations over time using a **Contextual Bandit** (ε-greedy / LinUCB) in offline simulation

### Components
| Requirement | Implementation |
|-------------|---------------|
| Core DL Model | Text embedding ranker (NLP) |
| CNN Component | Text-CNN for feature extraction |
| NLP Component | Text classification / embedding pipeline |
| RL Component | Contextual Bandit (ε-greedy / LinUCB) |

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/recommender-sys-6intelsy.git
cd recommender-sys-6intelsy
```

### 2. Set up the environment
```bash
pip install -r requirements.txt
# or with conda:
conda env create -f environment.yml
conda activate recommender-sys
```

### 3. Download the data
```bash
python data/get_data.py
```

### 4. Reproduce all results
```bash
make repro
# or:
bash run.sh
```

---

## Results Summary

> *(To be updated after training — Week 3)*

| Model | nDCG@10 | Hit@10 | Cumulative Reward |
|-------|---------|--------|-------------------|
| Random Baseline | — | — | — |
| Popularity Baseline | — | — | — |
| Text-CNN Ranker | — | — | — |
| + ε-greedy Bandit | — | — | — |
| + LinUCB Bandit | — | — | — |

---

## Repository Structure

```
recommender-sys-6intelsy/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── Makefile
├── run.sh
├── data/
│   ├── README.md           # How to obtain data (no raw PII in repo)
│   └── get_data.py
├── src/
│   ├── data_pipeline.py    # Preprocessing & data loading
│   ├── models/             # CNN / NLP architectures
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation script
│   ├── rl_agent.py         # Contextual bandit agent
│   └── utils/
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory Data Analysis
├── experiments/
│   ├── configs/            # Hyperparameter configs (YAML/JSON)
│   ├── logs/               # Training logs & learning curves
│   └── results/            # Tables, plots, figures
└── docs/
    ├── proposal.pdf
    ├── checkpoint.pdf
    ├── final_report.pdf
    ├── slides.pdf
    ├── model_card.md
    └── ethics_statement.md
```

---

## Timeline

| Week | Milestone | Status |
|------|-----------|--------|
| Week 1 | Proposal, repo setup, GitHub release v0.1 | In Progress |
| Week 2 | Data acquired, baselines trained, CNN + NLP scaffolded, RL stubbed | Upcoming |
| Week 3 | Final model, ablations, report, defense | Upcoming |

---

## License

[MIT License](LICENSE)

---

## Citation / Acknowledgements

> *(To be filled in — cite datasets, pretrained models, and any external code used)*
