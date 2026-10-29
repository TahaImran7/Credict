# Credict — Demo Script (5 Minutes)

Live walkthrough for pitch day. Practice twice before the actual
presentation.

**Before you start:**
- App running: `streamlit run app.py`
- Browser open at http://localhost:8501
- Light theme (default)
- Terminal window behind browser (for showing tests if asked)
- Baseline customer values in your head: `LIMIT_BAL=50000, AGE=35, all
  PAY=0, all BILL=1000, all PAY_AMT=500`

---

## 0:00 – 0:20 — Introduction

> "This is Credict — an explainable AI platform for credit risk
> assessment. I'll show you the four most important things: prediction
> with explanation, what-if analysis, model comparison, and fairness
> audit. Five minutes total."

*Land on the Overview page.*

---

## 0:20 – 1:00 — Overview (40 seconds)

**Point at the four KPI cards:**

> "Top row shows the portfolio: 30,000 customers, 22.12% historical
> default rate, Random Forest selected as the deployed model, ROC-AUC of
> 0.773 on the held-out test set."

**Point at the two charts:**

> "Left — how many customers fall in each predicted risk band. Right —
> the actual default rate for each band. Notice the staircase: 7.6% for
> Low risk, 16.4% for Moderate, 46.9% for High, and 73% for Very High.
> That staircase is the single most important proof that the model works
> — it produces genuinely informative risk bands."

**Point at the probability histogram:**

> "The probability distribution. Most customers are scored below 30% —
> which matches the 22% actual default rate in this dataset."

---

## 1:00 – 2:30 — Risk Assessment with SHAP (90 seconds)

*Click Risk Assessment in sidebar.*

**Fill the form (have values ready to type):**
- LIMIT_BAL: 50000
- SEX: Male, EDUCATION: University, MARRIAGE: Single
- AGE: 35
- All PAY_* fields: Revolving credit (0)
- All BILL_AMT: 1000, All PAY_AMT: 500

**Click "Predict and explain". Wait ~2 seconds.**

> "The model gives this customer a 20.7% probability of default in the
> next month — that's the Moderate band. Now — the interesting part."

**Point at the SHAP waterfall:**

> "The chart below shows *why*. Two things pushed risk up: the credit
> limit of 50,000 is relatively low, and the recent payment history is
> neutral but not fully clear. Three things pushed risk down: PAY_0 shows
> the customer is current, PAY_2 also current, and PAY_AMT payments were
> consistent."

**Point at "Decision threshold" row:**

> "This is important — the decision threshold is 0.31, not 0.50. That's
> because we applied isotonic calibration. The model was trained with
> class weights to handle the 22% class imbalance, which inflated
> probabilities. Calibration corrected this. A 20.7% predicted
> probability actually means 'about 21 out of 100 customers with this
> profile default' — that's the whole point of calibration."

**Point at download buttons:**

> "Every assessment is downloadable as JSON or CSV — for audit logs or
> customer correspondence."

---

## 2:30 – 3:15 — What-if Simulator (45 seconds)

*Click What-if Simulator in sidebar.*

**Select features:** `PAY_0`, `LIMIT_BAL`

**Set new values:** `PAY_0 = 3`, `LIMIT_BAL = 20000`

**Click "Run scenario".**

> "Same customer. Now change PAY_0 to 3 — meaning the customer is 3
> months late on the most recent payment — and reduce the limit to
> 20,000. The model updates instantly: risk jumps by 11.55 percentage
> points. Still in the Moderate band, but heading toward High."

**Point at warning if any appears:**

> "If any combination is unrealistic — for example, holding all other
> PAY_* fields at zero while changing just PAY_0 to 3 — a warning appears
> telling the analyst the combination is unusual in real customer data."

**Point at disclaimer at bottom:**

> "And every scenario includes a disclaimer: this is a model-based
> simulation, not a financial guarantee."

---

## 3:15 – 4:00 — Model Performance (45 seconds)

*Click Model Performance in sidebar.*

**Point at the comparison table:**

> "Three models compared on the 6,000 held-out test set. Logistic
> Regression: 77.4% accuracy. Random Forest: 79.7%. Calibrated Random
> Forest — the deployed model — 79.5% accuracy with Brier score 0.136,
> which is 11% better than the uncalibrated model. The trade-off:
> calibration slightly reduced ROC-AUC and PR-AUC due to ties in the
> isotonic step function, but improved probability quality. That's the
> right trade for a tool that displays probabilities."

**Point at ROC curves, PR curves, calibration curve:**

> "Standard evaluation charts. The calibration curve is the one that
> matters most here — you can see the calibrated model's curve is closer
> to the perfect diagonal than the raw model's."

**Point at confusion matrices:**

> "Confusion matrices for both models. Both catch roughly the same
> proportion of defaulters at their respective thresholds — but the
> calibrated model's probabilities are meaningful, not just its
> classifications."

---

## 4:00 – 4:30 — Fairness Audit (30 seconds)

*Click Fairness Audit in sidebar.*

**Leave dimension at "SEX".**

> "Group-level metrics across gender. Both groups — male and female — are
> sampled here from the test set. True positive rate is similar: 0.533
> for male, 0.539 for female. False positive rate: 0.146 vs 0.118. Not
> identical, but no red flags."

**Click through EDUCATION dimension:**

> "Education dimension shows more variation — the 'Others' group has only
> 82 records, so metrics there are flagged as low-confidence. That's the
> audit being honest about what it can and cannot confidently say."

**Point at the disclaimer at the bottom:**

> "And critically — this is a *descriptive* audit. It reports what the
> model does; it doesn't certify that the model is fair. That distinction
> matters for a regulator."

---

## 4:30 – 5:00 — Close (30 seconds)

*Click Reports page (optional) or back to Overview.*

> "Everything I've shown comes from actual runs on a 6,000-customer
> held-out test set. 147 automated tests pass. The entire pipeline is
> reproducible from seed 42. The model, the SHAP explainer, and the
> calibration are all saved as artefacts — the analyst never retrains."

**Pause.**

> "That's Credict. Happy to answer questions."

---

## Backup material (if judges ask for more)

**If asked to show tests:**
- Switch to terminal, run `pytest tests -v | tail -5`
- Show `147 passed` line

**If asked to show a high-risk customer:**
- Risk Assessment → Load from dataset → enter a high-risk row index
  from `test_predictions.csv` (have 1–2 prepared)
- Predict → shows Very High band

**If asked about calibration numbers:**
- Model Performance page → point at Brier column

**If asked about limitations:**
- Reference `docs/limitations.md` directly

---

## Timing drift corrections

| If you're at | and are supposed to be at | Recovery |
|---|---|---|
| Overview at 1:30 | 1:00 | Skip the probability histogram mention |
| Risk Assessment at 2:45 | 2:30 | Skip the download buttons |
| What-if at 3:30 | 3:15 | Skip the unrealistic-combination warning |
| Model Performance at 4:15 | 4:00 | Skip confusion matrices, just say "standard, both models perform comparably" |
| Fairness at 4:45 | 4:30 | Skip EDUCATION — stay on SEX only |

**If running long overall:** drop the Fairness slide from the demo — jump
from Model Performance directly to close. Mention fairness in one
sentence: "Fairness audit is also available; the app reports group-level
metrics across four dimensions."