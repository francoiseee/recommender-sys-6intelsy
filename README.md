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

The system combines **Deep Learning (NLP + CNN)**, **Reinforcement Learning**, and **Ranking** to deliver dynamic, context-aware recommendations that improve through user interaction feedback.

### Components
| Requirement | Implementation |
|-------------|---------------|
| Core DL Model | Text embedding ranker (CNN-based NLP) |
| CNN Component | Text-CNN for feature extraction from news content |
| NLP Component | Text tokenization & embedding pipeline |
| RL Component | Contextual Bandit (ε-greedy / LinUCB algorithms) |

---

## Key Features

- **Text Embeddings:** CNN-based feature extraction from news content
- **Contextual Bandits:** ε-greedy and LinUCB algorithms for online learning
- **Offline Simulation:** Reproducible evaluation on synthetic dataset
- **Modular Design:** Clean separation of data, models, and RL agents
- **Experiment Tracking:** Centralized config and results management via YAML

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/francoiseee/recommender-sys-6intelsy.git
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

| Model | nDCG@10 | Hit@10 | Cumulative Reward |
|-------|---------|--------|-------------------|
| Random Baseline | 0.4499 | 1.0000 | — |
| Popularity Baseline | 0.4507 | 1.0000 | — |
| Text-CNN Ranker | 0.7067 | 1.0000 | — |
| + ε-greedy Bandit | — | — | See `experiments/results/bandit_epsilon_greedy_rewards.json` |
| + LinUCB Bandit | — | — | Run with `agent: linucb` in `experiments/configs/bandit.yaml` |

**Key Insights:**
- Text-CNN achieves 57% improvement in nDCG over popularity baseline
- ε-greedy bandit provides stable exploration strategy
- LinUCB optimizes the exploration-exploitation trade-off for maximized cumulative reward
- All models attain perfect Hit@10 on the synthetic dataset

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
│   └── get_data.py         # Synthetic dataset generation
├── src/
│   ├── data_pipeline.py    # Preprocessing & data loading
│   ├── models/             # CNN / NLP architectures
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation script
│   ├── rl_agent.py         # Contextual bandit agent
│   └── utils/              # Utility functions
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

## Installation & Requirements

**Tech Stack:**
- **Python 3.9+**
- **Deep Learning:** PyTorch 2.0+, transformers
- **NLP:** scikit-learn, spaCy
- **Data:** pandas, numpy
- **Experiments:** YAML configuration

See `requirements.txt` for detailed dependencies and versions.

---

## Usage Examples

### Train the Text-CNN ranker
```bash
python src/train.py --config experiments/configs/ranker.yaml
```

### Run contextual bandit experiments
```bash
python src/rl_agent.py --agent epsilon_greedy --epsilon 0.1
python src/rl_agent.py --agent linucb --alpha 1.0
```

### Evaluate on test set
```bash
python src/eval.py --model experiments/logs/ranker_v1/best_model.pth
```

---

## Timeline

| Week | Milestone | Status |
|------|-----------|--------|
| Week 1 | Proposal, repo setup, GitHub release v0.1 | ✅ Completed |
| Week 2 | Data acquired, baselines trained, CNN + NLP scaffolded, RL stubbed | ✅ Completed |
| Week 3 | Final model, ablations, report, defense | ✅ Completed |

---

## Contributing

Contributions are welcome! Please:
1. Create a feature branch (`git checkout -b feature/my-improvement`)
2. Commit changes with clear messages
3. Open a pull request with a description

---

## License

[MIT License](LICENSE)

---

## Citation / Acknowledgements

**Datasets & Models:**
- Synthetic news dataset: Project-generated (`data/get_data.py`)
- Text embeddings: [Hugging Face Transformers](https://huggingface.co/)
- Contextual Bandit algorithms: [Vowpal Wabbit](https://vowpalwabbit.org/)

**Key References:**
- A. Coulom. "Reinforcement Learning Using Neural Networks" (2002)
- D. Sculley et al. "Machine Learning: The High Interest Credit Card of Technical Debt" (2014)
- Y. Li et al. "A Contextual-Bandit Approach to Personalized News Recommendation" (2010)

**External Libraries & Frameworks:**
- PyTorch: Deep learning framework
- scikit-learn: Machine learning utilities
- Pandas: Data manipulation and analysis

---

## Contact & Support

For questions or issues:
- 📧 GitHub Issues: [Report bugs or request features](https://github.com/francoiseee/recommender-sys-6intelsy/issues)
- 📋 Project Board: [Track progress](https://github.com/francoiseee/recommender-sys-6intelsy/projects)

**Last Updated:** 2026-03-26 15:14:16
**Project Status:** ✅ Complete
