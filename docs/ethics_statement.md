# Ethics & Policy Statement
### Personalized Recommendation with Contextual Bandits
#### 6INTELSY Final Project AY 2025–2026

> **Status:** Draft (Week 1 — Ethics Risk Register) — to be expanded to full 1–2 page statement by Week 3

---

## 1. Intended Use & Limitations

This system is designed as an **academic research prototype** for personalized news/item recommendation using NLP and reinforcement learning. It is **not intended for deployment** in production systems or use with real personal user data.

---

## 2. Ethics Risk Register (Top 3 Risks)

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | **Filter Bubbles / Echo Chambers** | High | High | Bandit exploration (ε-greedy/LinUCB) promotes diversity; plan to evaluate exposure fairness across items |
| 2 | **Privacy & Consent for Behavioral Data** | Medium | High | Use only publicly available datasets with no PII; no real user behavioral data collected; document dataset license clearly |
| 3 | **Popularity Bias / Unfair Exposure** | Medium | Medium | Evaluate recommendation diversity; consider exposure fairness constraints as stretch goal; report per-item slice metrics |

---

## 3. Fairness Checks (Planned)

- Analyze recommendation distribution across item categories
- Check if bandit unfairly favors high-popularity items (popularity bias)
- Measure exposure fairness: are items from all categories recommended at fair rates?

---

## 4. Privacy & Consent

- Dataset: Public research dataset (MIND / MovieLens or similar) — no PII
- No hidden data collection during development or demo
- No real user behavioral data will be used without explicit consent
- All data sources will be cited with their licenses

---

## 5. Misuse Considerations

- This model should **not** be used to manipulate user behavior or maximize engagement at the cost of user wellbeing
- It should **not** be deployed in systems targeting vulnerable populations
- Exploration mechanisms are designed for fairness, not adversarial exploitation

---

## 6. Transparency

- All model architecture decisions, training procedures, and evaluation results will be documented
- Failure cases and slice analyses will be included in the final report
- Code is open-source and reproducible under the MIT License

---

*This statement will be expanded to 1–2 pages with full risk analysis, concrete mitigations, and fairness evaluation results in Week 3.*
