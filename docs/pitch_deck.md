# Credict — Pitch Deck

**Format:** 10 slides
**Target length:** 5–7 minutes speaking time + 3–5 min Q&A
**Slide template:** Each slide lists (a) on-slide content, (b) speaker
notes, (c) suggested timing.

**Rubric alignment:** Per the FinTech project rubric, judges weight
Innovation (4 marks) and Research/Feasibility (6 marks combined) the
highest. Slides 3, 5, 8, 9 carry the most weight.

---

## Slide 1 — Title

**On slide:**
- **Credict**
- Explainable AI Credit Risk Assessment
- Your name, roll number, date

**Speaker notes:**
> "Credict is an explainable AI credit-risk assessment platform. It
> predicts next-month credit-card default probability and, crucially,
> explains *why* — feature by feature. Built end-to-end in Python, runs
> locally, 147 automated tests pass."

**Time:** 15 seconds

---

## Slide 2 — The Problem

**On slide:**
- Credit scoring today is a **black box**
- Banks face regulatory pressure to justify decisions
- **Analyzing analysts** can't explain AI outputs to auditors
- **Customers** get rejected without a reason
- A single mid-size Pakistani bank loses ~**PKR 1.5 billion/year** in
  expected credit card default losses

**Speaker notes:**
> "Credit scoring is a fifty-year-old problem. FICO, Experian — accurate
> but opaque. When a customer is rejected, no one can explain why. SBP's
> prudential regulations require banks to justify credit decisions. A
> black-box model doesn't help. A mid-size bank with 500,000 credit card
> customers loses roughly 1.5 billion rupees per year in expected
> defaults. Even small improvements mean crores saved."

**Time:** 45 seconds

---

## Slide 3 — Research & Market Evidence

**On slide:**
- **Published benchmark:** 79–83% accuracy on the UCI credit default
  dataset (2015–2023 papers)
- **SHAP** (Lundberg & Lee, 2017) — standard for tree model
  explainability, thousands of citations
- **Market:** ~33 scheduled banks in Pakistan, ~15–20 with active credit
  card portfolios
- **Regulatory:** SBP Prudential Regulations for Consumer Financing
  require systematic risk assessment
- **Gap:** No local or open-source tool combines prediction + explanation
  + fairness audit

**Speaker notes:**
> "We researched this thoroughly. The problem is well-established in
> academia — Yeh and Lien's dataset is used in hundreds of papers with
> published accuracy of 79 to 83%. SHAP is the industry standard for
> explainability. In Pakistan, SBP regulations require systematic credit
> risk assessment but no local or open-source tool combines prediction
> with explanation and fairness audit. That's the gap Credict fills."

**Time:** 45 seconds

---

## Slide 4 — Solution Overview

**On slide:**
- **Credict** — 8-page interactive dashboard
- Predicts default probability with explanation
- What-if simulator for scenario analysis
- Model comparison (LR vs RF)
- Fairness audit across customer groups
- Downloadable reports
- Runs locally, no cloud dependency

**Speaker notes:**
> "Credict is a full application, not a notebook. Eight pages: risk
> assessment with SHAP-based explanation, what-if simulator, model
> comparison, portfolio analytics, fairness audit, downloadable reports.
> Runs entirely on the analyst's local machine — no data leaves the bank.
> Light and dark themes."

**Time:** 40 seconds

---

## Slide 5 — Live Demo (Screenshots)

**On slide:**
- Screenshot 1: Risk Assessment showing 20.7% probability + SHAP
  waterfall
- Screenshot 2: Model comparison table (3 models)
- Screenshot 3: Fairness audit (SEX dimension)
- One-line caption per screenshot

**Speaker notes:**
> "Here's the app. Left — a customer with a 20.7% predicted default
> probability. The SHAP waterfall below shows which features pushed the
> risk up or down — PAY_0 reduces it because the customer is current,
> low credit limit increases it. Right — model comparison: Logistic
> Regression, Random Forest, and a calibrated Random Forest. Calibration
> improved Brier score by 11%, that's probability quality. Below —
> fairness audit shows group-level true-positive rates across gender,
> education, marital status, age bands. Every number comes from actual
> runs on a 6,000-customer held-out test set."

**Time:** 1 min 30 seconds

---

## Slide 6 — How It Works

**On slide:**
- Data → Preprocessing → Model Training → Calibration → Deployment
- 30,000 UCI records (22% default rate, imbalanced)
- Stratified 80/20 split, seed 42
- 5-fold cross-validated hyperparameter tuning
- Isotonic calibration on out-of-fold predictions
- SHAP TreeExplainer for per-prediction explanations

**Speaker notes:**
> "The pipeline: UCI dataset of 30,000 credit card customers, 22% default
> rate. We split 80/20 with seed 42. Trained Logistic Regression and
> Random Forest. Hyperparameters tuned by 5-fold cross-validation.
> Random Forest was overconfident due to class balancing, so we applied
> isotonic regression calibration — that's what shifted the baseline
> customer from 37% to 20.7%, and improved Brier score by 11%. SHAP
> explains every individual prediction. No data leakage: preprocessing
> fitted on training data only."

**Time:** 45 seconds

---

## Slide 7 — Results

**On slide:**

| Model | Accuracy | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| LR | 77.4% | 0.759 | 0.527 | 0.186 |
| RF raw | 79.7% | 0.774 | 0.552 | 0.154 |
| **RF calibrated** | **79.5%** | **0.773** | **0.539** | **0.136** |

- Monotone observed default rates by risk band: 7.6% → 16.4% → 46.9% →
  73.1%
- 147 automated tests pass
- Published benchmark: 79–83% on this dataset

**Speaker notes:**
> "Results on the 6,000 held-out test set. Logistic Regression: 77.4%
> accuracy. Random Forest raw: 79.7%. Calibrated Random Forest: 79.5%
> accuracy with the best Brier score at 0.136 — that's our deployed
> model. Crucially, the risk bands are *informative*: customers scored in
> the Low band default 7.6% of the time, Very High band default 73% of
> the time. That monotone staircase is the real proof the model works.
> Compared to published benchmarks of 79 to 83%, we're competitive. 147
> automated tests pass."

**Time:** 40 seconds

---

## Slide 8 — Market & Financials

**On slide:**
- **Market:** ~15–20 banks in Pakistan with credit card portfolios
- **SAM:** conservative — 2–4 institutions in Year 1
- **Customer value:** PKR 30M+ conservative annual savings per mid-size
  bank
- **Cost to build (commercial):** ~PKR 2.3M (~USD 8,200)
- **Price:** PKR 1.5M/year per bank
- **Break-even:** 2–3 customers in Year 1

**Speaker notes:**
> "Market: fifteen to twenty Pakistani banks have meaningful credit card
> portfolios. Microfinance and fintech lending are adjacent segments. For
> a single mid-size bank, 500,000 customers at 2% default rate means 1.5
> billion rupees of expected annual loss. Even a 2% relative reduction —
> that's PKR 30 million saved per year. The tool costs PKR 1.5 million
> per year. ROI: over 1,900%. Break-even for us: two to three customers
> in Year 1. Commercial build cost is roughly 2.3 million rupees."

**Time:** 45 seconds

---

## Slide 9 — Innovation & Differentiation

**On slide:**
- Only Pakistani / open-source product combining:
  - Prediction + explanation + fairness audit + what-if simulation
  - Calibrated probabilities (Brier 0.136)
  - Local deployable (SBP data localization compatible)
- **vs FICO/Experian:** they are black boxes; we explain every prediction
- **vs open-source:** sklearn, SHAP, Fairlearn — each does one thing; we
  integrate all into one analyst tool
- Positioned as **decision-support**, not automated approval — regulator-
  friendly

**Speaker notes:**
> "What makes Credict distinct? FICO is accurate but opaque. SHAP is
> powerful but requires a data scientist. Fairlearn computes fairness
> but doesn't integrate with prediction. Credict is the only tool
> that combines all of it, locally deployable, into one interface that an
> analyst can use without code. Crucially — we positioned it as
> decision-support, not automated approval. That's regulator-friendly,
> SBP-aligned, and safer."

**Time:** 40 seconds

---

## Slide 10 — Ask & Close

**On slide:**
- Complete working prototype
- 147 automated tests passing
- Reproducible from seed 42
- Open to: deployment pilot with a Pakistani bank, or extension to
  XGBoost + feature engineering for production
- **Ask:** feedback on real-bank pilot opportunity
- Contact: [your email]

**Speaker notes:**
> "We've built a complete, tested, working prototype. All numbers are
> from actual runs. Everything is reproducible. We're not asking for
> funding — we're asking for a pilot opportunity with a Pakistani bank,
> or feedback on the direction for production hardening. Thank you."

**Time:** 20 seconds

---

## Total timing

| Slide | Time |
|---|---|
| 1 | 0:15 |
| 2 | 0:45 |
| 3 | 0:45 |
| 4 | 0:40 |
| 5 | 1:30 |
| 6 | 0:45 |
| 7 | 0:40 |
| 8 | 0:45 |
| 9 | 0:40 |
| 10 | 0:20 |
| **Total** | **6:25** |

Within 5–7 minute target. Rehearse twice before pitch day — time yourself
with a stopwatch. If running over 6:30, trim slide 6 (How It Works) to 30
seconds.

---

## Pitch delivery tips

1. **Don't read slides.** Slides are anchors; you talk.
2. **Pause after "1.5 billion rupees"** on slide 2 — let it land.
3. **On slide 5 (demo), point at the screen.** "See this? This is PAY_0
   reducing risk."
4. **On slide 7 (results), emphasize monotone bands** — "73% of Very High
   actually defaulted. That's the real proof."
5. **If time pressure, drop slide 6** — jump from 5 to 7. Pipeline detail
   can be in Q&A.
6. **End with confidence.** "Thank you" — then stop talking. Let judges
   ask.

---

## Slide design guidelines

- **Colour palette:** navy (#0B1729), teal (#0D9488), off-white (#F8FAFC)
  — matches app UI
- **Font:** Inter (or system sans-serif)
- **No emoji**
- **One idea per slide**
- **Charts:** use app screenshots, not recreated charts
- **Tables:** keep to 3–4 rows max on screen