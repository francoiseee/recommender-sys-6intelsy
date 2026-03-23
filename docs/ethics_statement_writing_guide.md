# Ethics Statement Writing Guide

This guide helps you write a strong ethics statement that matches the project rubric.

## Target Output File (Name + File Type)

- `docs/ethics_statement.md` — Markdown source
- Include this content in `docs/final_report.pdf` (PDF section)

## Required Length

- 1–2 pages worth of content in report form

## Recommended Structure

1. Intended Use
2. Out-of-Scope / Non-Intended Use
3. Data Governance
4. Risk Register
5. Fairness Checks
6. Privacy and Consent
7. Misuse Scenarios and Safeguards
8. Limitations and Residual Risk
9. Responsible Deployment Guidance

## Key Points You Must Cover

### 1) Intended Use
- What your recommender is for (offline educational prototype)
- Who should use it (students/researchers)

### 2) Out-of-Scope Use
- Not for production deployment
- Not for high-stakes decision-making
- Not for profiling real users without consent

### 3) Data Governance
- Dataset source and licensing
- Presence/absence of PII
- Data representativeness limits
- Leakage prevention and split strategy

### 4) Risk Register (Minimum 3)
For each risk include:
- Risk description
- Likelihood (low/medium/high)
- Impact (low/medium/high)
- Mitigation implemented
- Remaining risk

Recommended risks for your project:
1. Filter bubbles / over-personalization
2. Popularity bias / exposure imbalance
3. Privacy risk if moved to real user data

### 5) Fairness Checks
- Subgroup performance differences (user_pref/category)
- Exposure distribution by category
- Where results come from (`experiments/results/slice_metrics_*.csv`)

### 6) Privacy and Consent
- Clarify that current dataset is synthetic and no PII is used
- State requirements if real behavioral data is used later (explicit consent, minimization, retention limits)

### 7) Misuse and Safeguards
- Potential misuse (engagement manipulation, narrow content loops)
- Proposed safeguards (diversity constraints, exposure caps, auditing)

### 8) Limitations and Residual Risk
- Offline eval limits
- Synthetic data limits external validity
- RL reward simulation mismatch risk

### 9) Responsible Guidance
- Human oversight requirement
- Periodic subgroup audits
- Clear disclaimers in deployment docs

## Suggested Evidence to Include

- Slice tables and plots from `experiments/results/plots/`
- Error analysis examples from `experiments/results/error_cases_top_rank_misses.csv`
- RL trend figures to show policy behavior over steps

## Quality Checklist

- [ ] 3+ concrete risks with mitigations
- [ ] Fairness section uses subgroup evidence
- [ ] Privacy/consent policy is explicit
- [ ] Intended-use and out-of-scope are clearly separated
- [ ] Limitations and residual risks are acknowledged
