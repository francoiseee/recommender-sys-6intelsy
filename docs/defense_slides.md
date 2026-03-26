# Defense Slides Draft (Markdown)

## Slide 1: Title
- Personalized Recommendation with Contextual Bandits
- 6INTELSY Final Project AY 2025-2026

## Slide 2: Problem and Motivation
- Need to rank relevant items for user preference contexts.
- Evaluate recommendation quality with robust retrieval metrics.

## Slide 3: System Architecture
- Data generation and preprocessing pipeline
- Text-CNN click-ranker
- Baselines: random, popularity
- RL bandit simulation (epsilon-greedy / LinUCB)

## Slide 4: Results Overview
|   nDCG@10 |   Hit@10 |   num_impressions | model               |
|----------:|---------:|------------------:|:--------------------|
|  0.449916 |        1 |               360 | random_baseline     |
|  0.450666 |        1 |               360 | popularity_baseline |
|  0.717259 |        1 |               360 | cnn_ranker          |

## Slide 5: Ablations
- Compare architecture and regularization choices.
- Best model by nDCG@10: **cnn_ranker**.

## Slide 6: Slice and Error Analysis
- Per user subgroup and per category subgroup metrics
- Miss-rate analysis and hard-failure cases
- See generated plots in experiments/results/plots

## Slide 7: Ethics and Responsible Use
- Synthetic dataset, no PII
- Filter-bubble and exposure-bias risks documented
- Explicit non-production use disclaimer

## Slide 8: Demo and Reproducibility
- One-command flow via `make repro`
- UI available via Streamlit for interactive demo

## Slide 9: Conclusions and Next Work
- Pipeline is reproducible and rubric-complete for required components
- Next: evaluate on real dataset and online feedback loop
