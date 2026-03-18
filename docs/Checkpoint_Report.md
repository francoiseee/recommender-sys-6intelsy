# Checkpoint Report (1-2 pages)

## Project
Personalized Recommendation with Contextual Bandits (NLP + RL)

## Scope of This Checkpoint
This checkpoint focuses on establishing a reproducible training/evaluation pipeline, generating initial baseline metrics, and preparing documentation artifacts for final report integration.

## Data Status
- Data acquisition completed using MIND-style files in `data/`:
  - `news.tsv`
  - `behaviors.tsv`
  - `entity_embedding.vec`
  - `relation_embedding.vec`
- Data loading and preprocessing implemented in `src/data_pipeline.py`.
- Train/validation split currently uses deterministic ratio split in `src/train.py` (`--train-ratio`, default `0.8`).

## EDA Status
- EDA notebook scaffold completed in `notebooks/01_eda.ipynb`.
- Notebook includes:
  - dataset loading checks
  - row count sanity checks
  - top categories/subcategories
  - history and impression length statistics

## Baselines and Experiments
### 1. Contextual Bandit (LinUCB)
- Implemented in `src/rl_agent.py` and trained through `src/train.py`.
- Includes:
  - progress bars with elapsed/ETA per stage
  - checkpoint save/resume support (`experiments/checkpoints/`)

### 2. Baseline Benchmark Suite
- Implemented in `src/benchmark.py`.
- Baselines included:
  - Popularity baseline
  - Content-similarity baseline
  - LinUCB baseline
  - Lightweight DL baseline (NumPy MLP)

### 3. CNN Component (Prototype Scaffold)
- CNN experiment scaffold added in `src/models/cnn_experiment.py`.
- Provides model construction and smoke-test path when PyTorch is available.
- Current environment does not include PyTorch, so this remains scaffold/prototype at this checkpoint.

## Initial Metrics Logged
Generated artifacts:
- Metrics table: `results/tables/baseline_metrics.csv`
- Learning curve table: `results/tables/linucb_learning_curve.csv`
- Learning curve figure: `results/figures/linucb_learning_curve.svg`
- Baseline comparison figure: `results/figures/baseline_ctr_comparison.svg`

Current benchmark sample run (`max-news=3000`, `max-events=500`) produced:
- popularity: CTR@1=0.0100, avg_reward=0.0100, avg_regret=0.0200
- content_similarity: CTR@1=0.0100, avg_reward=0.0100, avg_regret=0.0200
- linucb: CTR@1=0.0100, avg_reward=0.0100, avg_regret=0.0200
- mlp_dl_baseline: CTR@1=0.0100, avg_reward=0.0100, avg_regret=0.0200

Note: these are very early/noisy results from a small run intended for pipeline validation. Larger runs already showed stronger LinUCB behavior in terminal experiments.

## RL Reward Design
- Current reward: binary click label from impression token suffix (`news_id-1` clicked, `news_id-0` not clicked).
- Immediate reward objective: maximize click-through in top-1 recommendation.
- Logged metrics: CTR@1, avg_reward, avg_regret.

## Ethics and Responsible AI (Checkpoint)
- Draft ethics statement created in `docs/Ethics_Statement.md`.
- Current concerns tracked:
  - click bias and feedback loops
  - popularity amplification
  - privacy of behavioral history
  - fairness across categories/user groups

## Challenges and Risks
- CNN experiment is currently scaffolded only; full training requires PyTorch setup.
- Current split is deterministic-by-order; may need temporal split validation for stronger claims.
- Additional reproducibility enhancements are needed (fixed run configs, multi-seed aggregate report).

## Next Steps Before Final Submission
1. Run benchmark suite on larger event counts and export updated tables/figures.
2. Add multi-seed comparison and confidence intervals.
3. Add temporal split experiment for stronger offline validation.
4. Finalize Model Card and ethics mitigations with concrete evidence.
5. Integrate screenshots of figures and terminal runs into final document.

## Evidence Files for Screenshots
- `results/figures/linucb_learning_curve.svg`
- `results/figures/baseline_ctr_comparison.svg`
- `results/tables/baseline_metrics.csv`
- `experiments/checkpoints/run2_final.json`
- `experiments/checkpoints/run3_final.json`
