# Ethics & Policy Statement
### Personalized Recommendation with Contextual Bandits
#### 6INTELSY Final Project — AY 2025–2026

**Version:** Week 2 Update  
**Last Updated:** March 18, 2026

---

## 1. Intended Use & Limitations

This system is an **academic research prototype** for personalized news or item recommendation using NLP text embeddings and a Contextual Bandit reinforcement learning agent.

**Intended users:** Researchers and students studying recommender systems and NLP/RL pipelines.

**Intended use:**
- Offline simulation of personalized recommendation
- Academic study of contextual bandit behavior
- Demonstrating CNN + NLP + RL integration

**Out of scope / NOT intended for:**
- Real-world production deployment
- Use with real personal user data or behavioral logs
- Clinical, legal, financial, or safety-critical decision making
- Targeting or profiling real individuals

---

## 2. Dataset & Privacy Statement

| Field | Details |
|-------|---------|
| Dataset | [Fill in: MIND / MovieLens / other] |
| Source | https://www.kaggle.com/datasets/arashnic/mind-news-dataset?fbclid=IwY2xjawQnJ1pleHRuA2FlbQIxMABicmlkETFCMjJzaGhLMlFKNXhYb1Boc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHuBKiyPStGKkNHNO-rD1rHM7C0PbmjvWx4kAqNYwc-Eg5pCMj8m9k6dwM_gF_aem_E3ZiBbApOz-McA_GhHfaWw |
| License | [Fill in: e.g. MIT / CC BY 4.0 / Research only] |
| PII Present | No — publicly available research dataset |
| Consent | Not required — no personal data collected |
| Data Collection | No hidden or undisclosed data collection |

**Privacy measures taken:**
- No raw PII stored in the repository
- No real user behavioral data collected during development or demo
- All data obtained from publicly available, licensed research sources
- Dataset license documented in `data/README.md`

---

## 3. Ethics Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | **Filter Bubbles / Echo Chambers** — system over-personalizes and narrows what users see | High | High | Bandit exploration (ε-greedy / LinUCB) promotes diversity by forcing random exploration; evaluate exposure fairness across item categories |
| 2 | **Privacy & Consent for Behavioral Data** — using click/engagement data raises consent concerns | Medium | High | Use only publicly available datasets with no PII; document dataset license clearly; no real user data collected |
| 3 | **Popularity Bias / Unfair Item Exposure** — popular items dominate, rare items never shown | Medium | Medium | Evaluate recommendation diversity; report per-category exposure metrics; consider exposure fairness constraints as stretch goal |
| 4 | **Simulation Gap** — offline reward does not reflect real-world user behavior | High | Medium | Clearly document results are from offline simulation only; include disclaimers in Model Card and README |
| 5 | **Misuse for Manipulation** — bandit optimization repurposed to maximize engagement at cost of user wellbeing | Low | High | Academic-only use; no deployment pipeline provided; clear disclaimer in README and Model Card |

---

## 4. Fairness Checks (Week 2 Update)

The following fairness checks have been planned and are being implemented:

- **Category exposure fairness:** Are items from all categories recommended at roughly equal rates?
- **Popularity bias check:** Do popular items dominate? Measure % of recommendations from top-10% most popular items.
- **Text length bias:** Does the model perform better for items with longer, richer text descriptions?

> *(Results to be added in Week 3 after full evaluation)*

---

## 5. Transparency & Reproducibility

- All model architecture decisions documented in `docs/final_report.pdf`
- All hyperparameters logged in `experiments/configs/` and `experiments/logs/`
- Fixed random seeds (seed=42) used throughout for reproducibility
- Full pipeline reproducible via `make repro` or `bash run.sh`
- Code is open-source under MIT License
- Failure cases and slice analyses documented in Week 3 final report

---

## 6. Misuse Considerations

This system **must not** be used to:
- Manipulate user behavior or maximize engagement at the expense of user wellbeing
- Build echo chambers or reinforce harmful content loops
- Target, profile, or surveil individuals
- Make consequential decisions (hiring, lending, health, legal)
- Deploy in systems targeting vulnerable populations

---

## 7. Mitigations Summary

| Risk | Concrete Mitigation Implemented |
|------|---------------------------------|
| Filter bubbles | ε-greedy exploration randomly recommends non-optimal items; LinUCB balances exploration vs exploitation |
| Privacy | Public dataset only; no PII in repo; no real user data |
| Popularity bias | Diversity metrics (coverage, exposure) measured in evaluation |
| Simulation gap | Clear disclaimers in README, Model Card, and Final Report |
| Misuse | Academic-only disclaimer; no deployment pipeline |

---

## 8. Team Acknowledgement

| Role | Name | 
|------|------|
| Project Lead | Gurango, Christine Francoise O. |
| Data & Ethics Lead | Apostol, Lance Jezreel B. |
| Modeling Lead | Maninang, Allein Dane G. |
| Evaluation & MLOps Lead | Parungao, Nikko S. |

*This document will be finalized with actual fairness check results, slice analysis findings, and full evaluation metrics in Week 3.*
