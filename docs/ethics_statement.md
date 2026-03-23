# Ethics & Policy Statement
### Personalized Recommendation with Contextual Bandits
#### 6INTELSY Final Project AY 2025–2026

---

## 1. Intended Use

This system is an **academic research prototype** for personalized news/item recommendation using NLP (CNN-based ranker) and reinforcement learning (contextual bandit simulation). It is designed for use by **students and researchers** exploring offline evaluation of recommendation pipelines. The system was evaluated on `synthetic_newsrec_v1`, a project-generated dataset with no real user data.

The pipeline supports three model variants — random baseline, popularity baseline, and CNN ranker — and is evaluated using nDCG@10 and Hit@10 across subgroups defined by user preference and clicked category.

---

## 2. Out-of-Scope / Non-Intended Use

This system is **not intended for**:

- Production deployment in any live recommendation service
- High-stakes decision-making (e.g., content moderation, hiring, financial decisions)
- Profiling, tracking, or targeting real users without explicit informed consent
- Any application involving sensitive populations or vulnerable users
- Maximizing engagement at the expense of user wellbeing

---

## 3. Data Governance

- **Dataset:** `synthetic_newsrec_v1`, generated via `data/get_data.py` — fully synthetic, no PII, no real user behavioral data
- **Licensing:** Project-generated; no third-party data licenses apply to the synthetic set. All data sources, if referenced, will be cited with their licenses.
- **Split Strategy:** Train/val/test split by impression time ordering (70/15/15), preventing temporal data leakage
- **Representativeness Limits:** The synthetic dataset covers five topic categories (business, health, politics, sports, technology) with a controlled user preference distribution. It does not reflect the diversity of real-world user behavior or content ecosystems.
- **No hidden data collection** occurred during development or demo phases.

---

## 4. Risk Register

| # | Risk | Likelihood | Impact | Mitigation Implemented | Residual Risk |
|---|------|-----------|--------|------------------------|---------------|
| 1 | **Filter Bubbles / Echo Chambers** | High | High | Bandit exploration (ε-greedy/LinUCB) promotes diversity; exposure fairness evaluated across item categories | Exploration alone may not fully prevent over-personalization in dense preference clusters |
| 2 | **Popularity Bias / Unfair Exposure** | Medium | Medium | Per-category slice metrics reported (nDCG@10 by user_pref and clicked_category); popularity baseline compared against CNN ranker | CNN ranker shows slight variation across categories (e.g., nDCG 0.695–0.758); bias may persist in production-scale data |
| 3 | **Privacy Risk if Extended to Real User Data** | Medium | High | Current pipeline uses only synthetic data; no behavioral data collected; consent requirements documented | If adapted to real data, re-identification risk and behavioral profiling risk would require full re-assessment |

---

## 5. Fairness Checks

Fairness was evaluated through subgroup slice analysis across two dimensions:

**5.1 By User Preference Subgroup**

The CNN ranker shows consistent performance across all user preference groups, with nDCG@10 ranging from **0.698 (technology)** to **0.741 (health)**. The popularity baseline shows greater variance (0.416–0.497), suggesting it is less equitable across preference groups. No subgroup had a miss rate greater than 0 under the CNN ranker, indicating reliable retrieval across all groups.

**5.2 By Clicked Category Subgroup**

nDCG@10 for the CNN ranker ranges from **0.696 (technology/sports)** to **0.758 (politics)**. The spread of ~0.06 nDCG points across categories is modest but warrants monitoring. The popularity baseline disadvantages health and technology users most (nDCG ~0.41–0.42), reinforcing that popularity-based ranking is less fair.

**5.3 Error Analysis**

Hard failure cases (lowest nDCG impressions) are concentrated in the random baseline, where the clicked item lands at rank 10 (nDCG ~0.289). These cases span all user preference and category combinations, confirming the random baseline's unreliability rather than systematic model bias in the CNN ranker.

**Planned Fairness Expansions:**
- Exposure fairness constraints as a stretch goal
- Evaluation of recommendation diversity beyond subgroup accuracy metrics

---

## 6. Privacy and Consent

- The current system uses **fully synthetic data** (`synthetic_newsrec_v1`) with no PII and no real user behavioral signals.
- No user data was collected, stored, or processed during development or demonstration.
- If this system is extended to real behavioral data in the future, the following requirements apply:
  - **Explicit informed consent** from all users before data collection
  - **Data minimization**: collect only what is necessary for the recommendation task
  - **Retention limits**: define and enforce a data retention policy
  - **Right to erasure**: users must be able to request deletion of their behavioral data
  - Full re-review of the risk register and fairness checks under real-data conditions

---

## 7. Misuse Scenarios and Safeguards

**Potential Misuse:**
- Deploying the bandit policy to maximize engagement metrics (e.g., clicks, dwell time) at the cost of content diversity or user wellbeing
- Using the user preference modeling to build narrow filter bubbles or segment users for targeted manipulation
- Adapting the pipeline for profiling vulnerable populations (e.g., minors, politically targeted groups)

**Proposed Safeguards:**
- Diversity constraints: enforce minimum exposure rates per content category in ranked outputs
- Exposure caps: prevent any single category from dominating a user's recommendation feed
- Regular subgroup audits: re-run slice analyses on any new dataset before deployment
- Human oversight: require human review of bandit policy updates before rollout
- Clear disclaimers in all deployment documentation that this system is a research prototype

---

## 8. Limitations and Residual Risk

- **Offline evaluation gap:** nDCG@10 and Hit@10 measure retrieval quality in a static setting. They do not capture how users would respond to recommendations in a live system, where feedback loops, novelty effects, and behavioral drift occur.
- **Synthetic data limits external validity:** The controlled category distribution and user preference assignments in `synthetic_newsrec_v1` do not reflect the complexity of real-world news consumption patterns. Results should not be generalized beyond the prototype setting.
- **RL reward simulation mismatch:** The bandit simulation uses click signals as a proxy for user satisfaction. In reality, clicks do not reliably indicate value, and optimizing for clicks can reinforce harmful patterns (sensationalism, outrage content).
- **Small evaluation set:** With 360 test impressions, statistical confidence in subgroup results (as few as 54–85 impressions per category) is limited.
- **Residual popularity bias:** The CNN ranker outperforms the popularity baseline but does not eliminate popularity effects. Items with more training signal may still receive disproportionate exposure.

---

## 9. Responsible Deployment Guidance

If this prototype is ever adapted beyond its academic scope:

1. **Require human oversight** for any policy changes to the bandit or ranker
2. **Conduct subgroup audits** before and after each model update using real slice metrics
3. **Replace synthetic data** with consented, representative real-world data and re-run the full risk assessment
4. **Include user controls**: allow users to inspect and adjust their preference profiles
5. **Publish evaluation results** including failure cases and slice analyses — do not report only aggregate metrics
6. **Comply with applicable data protection regulations** (e.g., Data Privacy Act of 2012 in the Philippines, GDPR where applicable)

---

## 10. Transparency

- All model architecture decisions, training procedures, and evaluation results are documented in `final_report.md`
- Failure cases and slice analyses are included (see Section 5.3 and `experiments/results/error_cases_top_rank_misses.csv`)
- Code is open-source and reproducible under the **MIT License** via `make repro`
- This ethics statement will be included as a section in `docs/final_report.pdf`

---

*Ethics statement updated to reflect Week 3 final evaluation results from `synthetic_newsrec_v1`. Aligned with the Ethics Statement Writing Guide and project rubric requirements.*
