# Defense Presentation Keypoints Guide

Use this guide to make your video/live presentation satisfy the rubric expectations.

## Target Output Files (Name + File Type)

1. `docs/slides.pdf` — PDF
2. Optional recording file (you handle this): MP4

## Recommended Timing

- 8–10 minutes total presentation
- 8–10 minutes Q&A preparation

## Slide-by-Slide Focus (8–10 slides)

### Slide 1: Title and Team
- Project name
- Team roles
- One-sentence project objective

### Slide 2: Problem and Success Metrics
- Define ranking objective clearly
- Explain metrics (nDCG@10, Hit@10, RL cumulative reward)

### Slide 3: Dataset and Governance
- Data source
- License and PII statement
- Split strategy and no-leakage statement

### Slide 4: Architecture Overview
- Show end-to-end pipeline
- Clearly label CNN, NLP, and RL components

### Slide 5: Main Results
- Baseline vs CNN comparison table
- Explain practical performance gain

### Slide 6: Ablation Findings
- Show >=2 ablations and what changed
- Explain what you learned from each

### Slide 7: RL Learning Curves
- Show learning-curve plots
- Explain reward trend and seed variance

### Slide 8: Slice and Error Analysis
- Subgroup performance differences
- Hard-failure examples and interpretation

### Slide 9: Ethics and Model Card Summary
- Top risks and mitigations
- Intended use, limitations, and safeguards

### Slide 10: Repro + Demo + Closing
- One-command run path
- Demo flow
- Key takeaway and future work

## Figures You Should Show

- `experiments/results/metrics_summary.csv` table snapshot
- `experiments/results/ablation_summary.json` summary table
- `experiments/results/plots/ndcg_by_user_pref.png`
- `experiments/results/plots/hit_by_clicked_category.png`
- `experiments/results/plots/miss_rate_by_user_pref.png`
- `experiments/results/bandit_epsilon_greedy_learning_curve.png`
- `experiments/results/bandit_epsilon_greedy_multiseed_reward.png`

## What Panelists Usually Ask (Prep Prompts)

1. Why these metrics and not others?
2. How did you prevent leakage?
3. Why does CNN outperform baselines?
4. What does RL improve in your setup?
5. Which subgroup is weakest and why?
6. What are your ethical risks in real deployment?
7. How reproducible is your pipeline?

## Delivery Checklist

- [ ] Every rubric-required component is explicitly named in slides
- [ ] Quantitative results are visible and readable
- [ ] RL curve is shown and interpreted
- [ ] Ethics section includes risks + mitigations
- [ ] Repro command is shown exactly
- [ ] Demo path is clear and short
