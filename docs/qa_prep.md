# Credict — Q&A Preparation

15 most likely judge questions with prepared answers. Rehearse each
answer to 30–45 seconds.

---

## Model & Technical

### Q1. What is your model's accuracy, and why isn't it higher?

**Answer:** "79.5% on the held-out test set. Published benchmarks on this
exact UCI dataset range from 79 to 83% for state-of-the-art methods
including XGBoost and deep learning. We're within that range. The reason
accuracy is capped is the class imbalance — 22% defaults, 78% non-defaults
— which means a trivial 'always predict no default' model already achieves
77.88%. The real signal is in ROC-AUC 0.773 and PR-AUC 0.539, well above
random baselines. And the risk bands are monotone: Low band 7.6% actual
default rate, Very High band 73%. That staircase is what actually matters
in credit-risk work, not raw accuracy."

### Q2. Why Random Forest and not XGBoost / deep learning?

**Answer:** "Random Forest was chosen for three reasons. First, the UCI
dataset is tabular with 23 features — ensemble tree models perform
competitively with deep learning on tabular data. Second, SHAP's
TreeExplainer produces *exact* (not approximate) explanations for Random
Forest, which is critical for the explainability requirement. Third,
sklearn-only keeps the dependency footprint minimal and the install
reliable on any OS. Adding XGBoost would likely push accuracy up by 1–2pp,
but at the cost of added complexity and no material change to explainability.
That's a documented trade-off in `docs/limitations.md`."

### Q3. What is SHAP, and how do you know it's correct?

**Answer:** "SHAP — SHapley Additive exPlanations — comes from cooperative
game theory. It distributes a model's prediction among input features by
computing each feature's average marginal contribution across all feature
orderings. For tree-based models, Lundberg and Lee (2017) showed that
TreeExplainer produces *exact* SHAP values in polynomial time. Every SHAP
value in Credict comes from the official SHAP library's TreeExplainer,
running on the actual deployed Random Forest. It's the same library used
at Google, Microsoft, and most major banks."

### Q4. What does calibration do, and why did you need it?

**Answer:** "Random Forest was trained with `class_weight='balanced'`
because the dataset is 78% non-defaults and 22% defaults. Class weighting
improves discrimination on imbalanced data but inflates predicted
probabilities — the mean predicted probability was 35% when the actual
base rate is 22%. Isotonic regression was fitted on 5-fold out-of-fold
predictions to correct this. Result: Brier score improved 11.4%, and the
baseline customer's probability dropped from 37% to 20.7%, aligning with
the true base rate. The trade-off: ROC-AUC dropped slightly, from 0.774 to
0.773, and PR-AUC from 0.552 to 0.539, because isotonic regression
introduces ties. That's a documented consequence of the method — accuracy
versus probability quality."

### Q5. How do you prevent data leakage?

**Answer:** "Three mechanisms. First, preprocessing (scaling, one-hot
encoding) is fitted only on the training set — never on test. Second,
hyperparameters are tuned via 5-fold cross-validation on the training set
only, and the test set is untouched during this step. Third, calibration
is fitted on out-of-fold predictions from the training set — again, no
test data. Everything is verified by tests: `test_split_is_reproducible`,
`test_full_pipeline_split_and_fit`, and the calibration tests. The test
set of 6,000 records is only used for final evaluation."

---

## Business & Financial

### Q6. How much money does this save a bank?

**Answer:** "For a mid-size Pakistani bank with 500,000 credit-card
customers at 2% default rate — that's 10,000 defaults per year — and an
average exposure of PKR 150,000, the annual expected loss is roughly 1.5
billion rupees. If Credict reduces defaults by even 2% relative —
conservative, given published literature — that's PKR 30 million saved
per year. The tool costs PKR 1.5 million per year on our proposed tier.
ROI: over 1,900%. Detailed calculation is in `docs/feasibility_analysis.md`."

### Q7. What's your revenue model?

**Answer:** "B2B SaaS subscription, tiered by customer count. Starter
tier up to 100,000 customers: PKR 600,000/year. Growth tier — 100k to
500k: PKR 1.5 million. Enterprise: PKR 3 million. Adjacent revenue from
customization, compliance reporting, and integration services. This is
proposed pricing — real willingness-to-pay validation would be part of a
commercial Phase 2. For the semester project, the pricing shows the
financial logic, not a signed contract."

### Q8. What's the total addressable market?

**Answer:** "Global credit-scoring market is roughly USD 10–15 billion.
But Credict targets Pakistan first — that's a serviceable addressable
market of 15 to 20 banks with active credit card portfolios. Realistic
Year 1 target: 2 to 4 institutions. Break-even at our costs: 2–3
customers. Adjacent markets: microfinance banks (about 11 licensed in
Pakistan), fintech lenders, and credit bureaus as data partners rather
than competitors."

---

## Regulatory & Ethics

### Q9. Is this compliant with SBP regulations?

**Answer:** "Credict is positioned as *decision-support*, not
automated decision-making. That's a deliberately safe framing: the model
gives the analyst a probability and an explanation, but a human makes the
actual credit decision. SBP's Prudential Regulations for Consumer
Financing require systematic risk assessment, and SHAP-based explanations
directly satisfy that requirement — they make the model auditable.
Formal approval for use in lending decisions would require an SBP
Regulatory Sandbox process, which is a Phase 2 activity. Currently we're
compliant in the *decision-support* category, not in the *automated
decision* category."

### Q10. What about fairness — are you sure the model doesn't discriminate?

**Answer:** "We don't claim it's fair — we claim it's *auditable*. Our
fairness module computes group-level metrics across sex, education,
marital status, and age bands: true-positive rate, false-positive rate,
precision, NPV, ROC-AUC per group. Some groups show different metrics —
for example, EDUCATION=4 has fewer than 100 records and unstable
performance. That's flagged in the audit output, not hidden. SBP and other
regulators increasingly want *auditable* AI, not *certified fair* AI.
Credict provides the former. Certifying the latter would require a
larger sample and legal review outside this project's scope."

### Q11. What about data privacy?

**Answer:** "Credict runs entirely on the bank's local infrastructure.
No customer data leaves the machine — no cloud, no external API, no
telemetry. That directly aligns with SBP's data localization requirements.
The model artifacts are saved as local files; predictions are computed
in-process. This is a deliberate architectural choice for the banking
sector."

---

## Limitations & Honesty

### Q12. What can't Credict do?

**Answer:** "Several things, all documented in `docs/limitations.md`.
First, the dataset is from a Taiwanese bank in 2005 — real Pakistani bank
data would look different, and retraining would be required. Second,
there's no model drift monitoring; a production deployment needs quarterly
retraining. Third, no live user validation interviews — I used secondary
research and desk analysis instead, appropriate for a semester project.
Fourth, calibration improves probability quality but slightly reduces
ROC-AUC and PR-AUC due to ties. Fifth, no regulatory certification for
use in actual lending decisions — this is decision-support, not a lending
authority."

### Q13. What would you do differently with more time?

**Answer:** "Four things. First, add XGBoost as a third base model — likely
1–2pp accuracy gain. Second, feature engineering — payment ratio, credit
utilization, delay count — likely 1pp more. Third, proper user validation
interviews with 10+ bank risk officers. Fourth, production hardening:
model drift monitoring, quarterly retraining pipeline, RBAC and audit
logging. The prototype is already complete; these are additive improvements
for a commercial deployment."

---

## Comparison & Competition

### Q14. Why not just use FICO?

**Answer:** "Three reasons. First, FICO is a black box — it does not
provide per-prediction explanations. For SBP compliance and internal
audit, banks need to justify decisions; FICO doesn't help. Second, FICO
is expensive — typically only large banks can afford it. Third, FICO
isn't tuned to Pakistani customer data. Credict is explainable by
design, deployable at lower cost, and retrainable on a bank's own data.
We're not trying to replace FICO at HSBC — we're targeting mid-tier
Pakistani banks and microfinance lenders who currently use rule-based
scorecards."

### Q15. This is just a student project — why should a bank trust it?

**Answer:** "Fair point. We don't claim bank-readiness today. We claim:
(1) the technical approach is standard — Random Forest, SHAP, isotonic
calibration, all published and proven in the credit-risk literature;
(2) the implementation is verified — 147 automated tests, reproducible
from seed 42; (3) the design is aligned with regulatory needs —
decision-support, local deployment, auditable. What's missing for
production is user validation, security audit, and SBP regulatory
review — all Phase 2 work. We're asking for a pilot conversation, not a
production contract."

---

## Cross-cutting tips

1. **Never inflate numbers.** If asked "how accurate is it really?" — say
   "79.5%, within the published benchmark of 79–83% for this dataset."
2. **Acknowledge trade-offs** before judges find them. "We deliberately
   chose accuracy-for-probability quality via calibration."
3. **When unsure, say "that's a Phase 2 concern"** — clean, honest,
   defensible.
4. **Cite sources when challenged.** "Lundberg and Lee, NeurIPS 2017" for
   SHAP. "Yeh and Lien, 2009" for the dataset.
5. **Rehearse the ROI answer 3–4 times.** That's the highest-impact answer.