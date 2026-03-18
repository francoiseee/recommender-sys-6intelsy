# Model Card (Draft)

## Model Details
- Model name: LinUCB Contextual Bandit Recommender
- Location: `src/rl_agent.py`, `src/train.py`
- Input: candidate and user-context feature vectors derived from news text/category/entity features
- Output: ranking score per candidate item (top-1 selected)

## Intended Use
- Personalized news recommendation research prototype.
- Offline experimentation with logged impression/click data.

## Not Intended Use
- High-stakes decision making.
- Production deployment without online guardrails, fairness review, and privacy/legal review.

## Data
- Source format: MIND-style files in `data/`.
- Fields used:
  - item text and metadata from `news.tsv`
  - historical behavior and impressions from `behaviors.tsv`
- Reward label: click/non-click from impression suffix.

## Metrics
- Primary: CTR@1, average reward, average regret.
- Secondary (planned): coverage, diversity, calibration, subgroup performance.

## Quantitative Snapshot
See:
- `results/tables/baseline_metrics.csv`
- `results/tables/linucb_learning_curve.csv`

## Ethical Considerations
- Known risks:
  - click-driven feedback loops
  - popularity bias amplification
  - topic imbalance and potential filter bubbles
- Mitigations planned:
  - add diversity/coverage constraints
  - compare subgroup/category outcomes
  - maintain transparent logging and auditability

## Limitations
- Offline evaluation only (no online A/B test).
- Reward is short-term click proxy and may not align with long-term utility.
- CNN component is currently scaffolded and not yet fully trained in this environment.

## Maintenance Plan
- Track versioned checkpoints in `experiments/checkpoints/`.
- Store reproducible benchmark outputs in `results/tables/` and `results/figures/`.
- Update this model card as new experiments are completed.
