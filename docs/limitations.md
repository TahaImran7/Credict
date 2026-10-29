
---

## FILE 6 — `docs/limitations.md`

```markdown
# CreditLens — Consolidated Limitations

This document consolidates all known limitations of CreditLens in one
place. Every claim in the pitch, README, and report should be read in the
context of these limitations.

---

## 1. Dataset Limitations

### 1.1 Temporal and geographic scope
- **Data source:** Taiwanese bank, April–September 2005
- **Implication:** Behaviour patterns in 2005 Taiwan may not reflect
  current Pakistani credit card users
- **Mitigation:** Documented explicitly; model is a demonstration, not a
  production deployment

### 1.2 Class imbalance
- **Distribution:** 22.12% positive, 77.88% negative
- **Implication:** Accuracy alone is misleading; dumb baseline is 77.88%
- **Mitigation:** PR-AUC used as primary metric; observed default rates
  per risk band confirm monotone discrimination

### 1.3 No identifier or temporal features
- **Missing:** Customer tenure, employment history, external credit
  bureau data
- **Implication:** Real bank deployments would likely include these for
  better performance
- **Mitigation:** CreditLens uses only the 23 UCI features; document as
  scope

### 1.4 Undocumented category codes
- **EDUCATION codes 0, 5, 6:** Not in the UCI data dictionary
- **MARRIAGE code 0:** Not in the UCI data dictionary
- **Mitigation:** Merged to "others" (4 for education, 3 for marriage);
  documented in `src/data/preprocessing.py`

---

## 2. Model Limitations

### 2.1 Accuracy not state-of-the-art
- **Achieved:** 79.50% (calibrated RF)
- **Published SOTA:** 80–83% (XGBoost, LightGBM on this dataset)
- **Implication:** ~2–3 percentage point gap
- **Mitigation:** Justified via explainability-first positioning; XGBoost
  would improve accuracy marginally but not change core value

### 2.2 Calibration trades ranking for probability quality
- **Brier improvement:** −11.4% (0.1536 → 0.1361)
- **ROC-AUC drop:** −0.1% (0.7740 → 0.7732)
- **PR-AUC drop:** −2.4% (0.5523 → 0.5391)
- **Cause:** Isotonic regression step function creates ties
- **Justification:** Brier improvement matters more for a probability-
  displaying application; ranking drop is negligible

### 2.3 One-hot encoding cannot extrapolate
- **Issue:** Unseen category value at inference becomes all-zeros for
  that feature
- **Implication:** Customer with an unusual category value may receive
  misleading prediction
- **Mitigation:** `handle_unknown='ignore'` prevents crash; documented as
  limitation

### 2.4 Linear baseline is not deployed
- **LR accuracy:** 77.40% vs RF 79.70%
- **Implication:** The interpretable baseline underperforms; only SHAP
  explanations provide interpretability
- **Trade-off:** Accepted for accuracy gains

### 2.5 No model drift monitoring
- **Issue:** Deployed model reflects 2005 training data
- **Implication:** Real customer behaviour changes over years; model may
  degrade silently
- **Mitigation for production:** Recommend quarterly retraining, model
  performance monitoring — outside project scope

---

## 3. Explainability Limitations

### 3.1 SHAP is not causal
- **What SHAP shows:** Feature contributions to the model's output under
  its explanation framework
- **What SHAP does NOT show:** Causal reasons a customer defaulted
- **Implication:** "PAY_0 increases risk" means the model uses PAY_0 to
  justify a higher score — not that late payment causes default (though
  it correlates)
- **Repetition:** This caveat appears in `src/explainability/shap_explainer.py`
  docstring, in every SHAP chart title, and in the UI disclaimer

### 3.2 SHAP values are on the raw model
- **Issue:** SHAP computed on uncalibrated RF, not the calibrated
  deployed model
- **Why:** TreeExplainer requires a tree-based estimator; the calibrated
  wrapper is not a tree
- **Justification:** Calibration is monotonic — feature ranking and sign
  preserved. Only absolute magnitudes scale. Documented in
  `shap_explainer.py`

### 3.3 Aggregation obscures one-hot details
- **Issue:** SHAP values summed back to original feature names hides
  which category value drove the contribution
- **Trade-off:** Cleaner UI vs loss of granularity
- **Mitigation:** Full expanded SHAP available in `ShapExplanation.expanded_features`
  and `expanded_shap_values` for programmatic access

---

## 4. Fairness Limitations

### 4.1 Descriptive, not prescriptive
- **What the audit does:** Reports group-level metrics across 4
  dimensions
- **What the audit does NOT do:** Certify legal compliance, prove or
  disprove discrimination
- **Implication:** Group differences may reflect historical bias, sample
  size, or genuine population differences
- **Caveat:** This is a common misreading of fairness audits; documented
  in UI disclaimers

### 4.2 Small groups flagged but included
- **Issue:** EDUCATION=4 has 82 test records; AGE=60+ has 76
- **Implication:** Metrics for these groups are less stable
- **Mitigation:** `small_group = True` flag; metrics still shown with
  warning

### 4.3 No statistical significance testing
- **Issue:** Audit reports point estimates without confidence intervals
  or significance tests
- **Implication:** Small differences between groups may not be
  statistically meaningful
- **Mitigation for future:** Bootstrap confidence intervals would
  strengthen the audit

### 4.4 Only four dimensions audited
- **Dimensions:** SEX, EDUCATION, MARRIAGE, AGE_BAND
- **Missing:** Income (not in dataset), geography (not in dataset),
  employment status (not in dataset)
- **Implication:** CreditLens cannot audit dimensions the UCI dataset
  does not contain

---

## 5. Validation Limitations

### 5.1 No primary user research
- **Approach used:** Secondary research (academic literature, SBP
  regulation, competitive analysis)
- **Not done:** Interviews with bank risk officers, compliance officers,
  or end customers
- **Justification:** Semester-scope project; user validation would be
  Phase 2 activity
- **Documented in:** `docs/secondary_research.md`

### 5.2 No external peer review
- **Approach used:** Self-audit via 147 automated tests + manual UI
  verification
- **Not done:** Third-party code review, external model validation
- **Justification:** Appropriate for academic project; production
  deployment would require external audit
- **Documented in:** `docs/self_audit.md`

---

## 6. Deployment Limitations

### 6.1 No security hardening
- **Not done:** Authentication, RBAC, audit logging, input sanitization
  beyond basic validation
- **Implication:** Not ready for multi-user deployment without
  Streamlit-level auth
- **Mitigation for production:** Add authentication (Streamlit-Authenticator
  or reverse proxy), audit logging

### 6.2 No load testing
- **Not done:** Concurrent user testing, throughput benchmarking
- **Implication:** Unknown behaviour under multi-user load
- **Mitigation for production:** Load test before deploying to multiple
  analysts

### 6.3 No cloud deployment
- **Design choice:** Local-only execution
- **Trade-off:** No cloud scaling, but data never leaves the bank —
  aligns with SBP data localization
- **Reversibility:** Deployable on cloud VM if needed; Streamlit is
  portable

### 6.4 No regulatory certification
- **Not done:** SBP compliance audit
- **Implication:** Cannot be used for actual lending decisions without
  regulatory review
- **Positioning:** Decision-support, not automated approval
- **Mitigation:** Would require SBP Innovation Challenge or Regulatory
  Sandbox participation

---

## 7. Financial Feasibility Limitations

### 7.1 No willingness-to-pay validation
- **Pricing tiers:** Proposed, not validated by customer research
- **Implied assumption:** Mid-size banks would pay PKR 1.5M/year for the
  value delivered
- **Risk:** Actual willingness-to-pay may differ
- **Mitigation for Phase 2:** Interview 5–10 bank procurement teams

### 7.2 Market size is approximate
- **Pakistan credit card figures:** Industry estimates, not verified
  from primary SBP reports
- **Implication:** TAM/SAM/SOM calculations are illustrative
- **Mitigation:** Explicitly recommended to verify from SBP Payment
  Systems Review before commercial claim

### 7.3 Savings estimates are upper-bound
- **"PKR 30M savings"** assumes full 2% relative reduction
- **Reality:** Banks may capture only part of the tool's signal
- **Mitigation:** "Conservative" scenario is actually moderate; further
  conservatism would be prudent

---

## 8. Scope Limitations

### 8.1 Out-of-scope features
- Real bank API integrations
- Automated loan approval/rejection
- Cloud deployment
- Multi-user authentication
- Live financial-market data
- Chatbot or natural-language interface
- Mobile application

### 8.2 Technical scope
- Only 2 base models trained (LR, RF); XGBoost skipped
- Only isotonic calibration; Platt scaling not compared
- No feature engineering
- No nested cross-validation

---

## 9. Ethical Considerations

### 9.1 CreditLens must not be used to approve/reject real applicants
- **Reason:** Academic prototype, not regulatory certified
- **Guardrail:** UI disclaimers on every prediction page
- **Risk:** If used improperly, could cause real financial harm

### 9.2 SHAP explanations can be misread as causal
- **Guardrail:** Every SHAP chart title and UI text includes causality
  caveat
- **Risk:** A user may believe "changing PAY_0 fixes credit risk" —
  false

### 9.3 Fairness audit is not a fairness guarantee
- **Guardrail:** `FAIRNESS_LIMITATION` string in every JSON export
- **Risk:** A user may read group differences as proof of discrimination
  or non-discrimination — neither is valid

---

## 10. What Would Strengthen CreditLens

Priority-ordered list of enhancements for Phase 2:

| Enhancement | Impact | Effort |
|---|---|---|
| XGBoost as third model | +1–2pp accuracy | 30 min |
| Feature engineering (utilization, delay count) | +1pp accuracy | 1–2 hrs |
| User validation interviews | Addresses a rubric gap | 1–2 days |
| Bootstrap confidence intervals for fairness | Stronger audit | 2 hrs |
| Model drift monitoring | Production-ready | 1 day |
| SBP regulatory sandbox application | Regulatory pathway | Weeks–months |

---

## 11. Honest Summary

CreditLens is a **complete, tested, working academic prototype**. It
demonstrates:
- End-to-end ML pipeline
- Probability calibration
- SHAP-based explainability
- Fairness auditing
- Interactive dashboard with 8 pages
- 147 automated tests passing

It is **not**:
- Production-ready for real lending decisions
- Regulatory certified
- Validated with real bank stakeholders
- Monitored for model drift
- Hardened for security or multi-user deployment

Every claim in the pitch deck and README is bounded by these limitations.
No metrics are invented. No features are claimed without implementation.
The trade-offs (calibration vs ranking, explainability vs accuracy) are
documented, not hidden.