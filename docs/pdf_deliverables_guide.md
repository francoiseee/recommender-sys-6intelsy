# PDF Deliverables Guide (Week 3 Final)

This guide tells you exactly what PDF files to submit and what to include so your project aligns with the rubric in `6INTELSY FINAL PROJECT AY 2025-2026 8.pdf`.

## Required Output Files (Name + File Type)

1. `docs/checkpoint.pdf` — PDF
2. `docs/final_report.pdf` — PDF
3. `docs/slides.pdf` — PDF (exported from your presentation deck)

Optional but recommended source files:
- `docs/final_report.md` (source draft)
- `docs/defense_slides.md` (source draft)

## 1) docs/checkpoint.pdf (PDF)

Purpose: Prove Week 2 readiness and technical progress.

Target length:
- 1–2 pages

Main sections to include:
1. Problem + MVP scope
2. Dataset status and preprocessing status
3. Baselines status (simple + DL)
4. CNN/NLP progress
5. RL status (reward design + early curve)
6. Risks/blockers + week-3 plan

Evidence to attach in checkpoint:
- At least one metric table snapshot
- At least one early learning curve plot
- Link to repo and release/version tag if available

## 2) docs/final_report.pdf (PDF)

Purpose: Main technical paper for grading.

Target length:
- 4–6 pages (conference-style)

Required section outline:
1. Abstract
2. Problem & Motivation
3. Task Definition + Metrics + Constraints
4. Dataset & Governance
5. Methodology
6. Experimental Setup
7. Results
8. Ablations (>=2)
9. Error/Slice Analysis
10. RL Component Details
11. Ethics & Policy
12. Model Card Summary
13. Reproducibility
14. Limitations & Future Work
15. References

What to explicitly include per section:

4. Dataset & Governance
- Source, license, whether PII exists, representativeness limits
- Leakage prevention strategy
- Split strategy (train/val/test)

5. Methodology
- Core DL model design
- CNN component role
- NLP component role
- RL component integration path

6. Experimental Setup
- Seeds, hyperparameters, optimizer, LR, stopping rule
- Hardware/runtime note

7. Results
- Main metrics table for all models (random, popularity, cnn_ranker)

8. Ablations
- At least two ablations and effect on nDCG@10 / Hit@10

9. Error/Slice Analysis
- Subgroup tables by user preference/category
- Hard failure examples and interpretation

10. RL Component Details
- Environment spec: state, action, reward, episode
- Learning-curve figures from `experiments/results`

11. Ethics & Policy
- Risks + concrete mitigations
- Intended use and out-of-scope use

13. Reproducibility
- Exact one-command run path (`make repro` / `run.sh`)
- Environment setup (`requirements.txt` or `environment.yml`)

## 3) docs/slides.pdf (PDF)

Purpose: Defense deck used in presentation.

Target length:
- 8–10 slides

Required slides:
1. Title + Team + Project framing
2. Problem + why it matters
3. Dataset + governance constraints
4. Method pipeline (CNN + NLP + RL)
5. Main quantitative results
6. Ablation findings
7. Slice/error analysis findings
8. Ethics + model card highlights
9. Repro/demo instructions
10. Conclusion + future work

## Figures You Should Reuse (Already Generated)

Use these in final_report.pdf and slides.pdf:
- `experiments/results/plots/ndcg_by_user_pref.png`
- `experiments/results/plots/hit_by_clicked_category.png`
- `experiments/results/plots/miss_rate_by_user_pref.png`
- `experiments/results/bandit_epsilon_greedy_learning_curve.png`
- `experiments/results/bandit_epsilon_greedy_reward_trend.png`
- `experiments/results/bandit_epsilon_greedy_multiseed_reward.png`
- `experiments/results/bandit_epsilon_greedy_multiseed_cumulative.png`

## Final Submission Checklist

- [ ] `docs/checkpoint.pdf` exists
- [ ] `docs/final_report.pdf` exists
- [ ] `docs/slides.pdf` exists
- [ ] Final report includes >=2 ablations
- [ ] Final report includes RL learning curves
- [ ] Final report includes slice/error analysis
- [ ] Ethics and model card content are reflected in report
- [ ] Repro instructions are accurate and tested
