# CreditLens — Financial Feasibility Analysis

**Document type:** Feasibility and business viability assessment
**Prepared for:** FinTech Project Evaluation
**Last updated:** 29 - October - 2026

**Note on data sources:** Figures marked "verify:" are industry estimates
from publicly available sources. Exact verification from SBP quarterly
reports is recommended before any commercial claim. For this semester
project, the estimates are reasonable and defensible.

---

## 1. Executive Summary

CreditLens is an explainable AI credit risk assessment platform. It predicts
next-month payment default probability for credit card customers, explains
each prediction using SHAP values, supports what-if scenario analysis, and
audits model fairness across customer groups.

This document assesses:
- Cost to build and operate
- Market opportunity in Pakistan and globally
- Revenue model options
- Concrete financial value to a single mid-size bank
- Break-even and risk analysis

**Key finding:** A Pakistani bank with 500,000 credit card customers faces
approximately **PKR 1.5 billion per year** in expected default losses
(assuming a 2% default rate and average exposure of PKR 150,000). Even a
**2% relative reduction** in defaults — conservative given published
literature on AI credit scoring — would save **PKR 30 million per year**.
CreditLens needs to save only a fraction of this to justify its cost.

---

## 2. Market Size (TAM / SAM / SOM)

### 2.1 Total Addressable Market (TAM)

Global market for credit scoring software and services.

- Credit scoring market size (global): approximately **USD 10–15 billion**
  annually (Allied Market Research, MarketsandMarkets — verify exact figure)
- Global explainable AI in finance: growing segment, projected to grow at
  ~20% CAGR through 2030
- Pakistan's share of global banking assets: <0.5% (Pakistan does not
  contribute significantly to global TAM)

*Interpretation: the product is not designed to compete globally in its
current state. It targets Pakistan first.*

### 2.2 Serviceable Addressable Market (SAM) — Pakistan

- **Scheduled banks in Pakistan:** approximately 33 (verify: SBP Annual Report)
- **Commercial banks with active credit card portfolios:** roughly 15–20
- **Total bank branches in Pakistan:** approximately 17,000+ (verify: SBP)
- **Credit card customers in Pakistan:** estimates vary; low single-digit
  millions (verify: SBP Payment Systems Review)

**SAM interpretation:** ~15–20 banks with meaningful credit card portfolios
represent the initial target segment.

### 2.3 Serviceable Obtainable Market (SOM) — Realistic Year 1–2

Given the product's stage (academic MVP, no production hardening yet), the
realistic first target is **small to mid-tier banks and NBFCs** that:

- Have credit card or consumer lending products
- Are under regulatory pressure to improve risk transparency
- Have smaller internal data science teams

Realistic Year 1–2 target: **2–4 institutions** in Pakistan.

### 2.4 Additional adjacent markets

- **Microfinance banks** — Pakistan has ~11 licensed microfinance banks;
  strong need for automated risk scoring given small manual teams
- **Fintech lenders** — Pakistan's fintech sector includes several digital
  lending startups (verify: SECP NBFC registry)
- **Credit bureaus** — companies like eCIB (SBP) provide the data side;
  CreditLens complements, does not replace

---

## 3. Cost to Build

### 3.1 Development cost (one-time)

Assumes a 2-person team (1 ML engineer, 1 full-stack engineer), 3–4 months.

| Item | Assumption | Cost (PKR) |
|---|---|---|
| ML engineer | PKR 250,000/month × 4 months | 1,000,000 |
| Full-stack engineer | PKR 200,000/month × 4 months | 800,000 |
| Data labeling / validation | 1 week of analyst time | 150,000 |
| Cloud compute (training + dev) | AWS/GCP small instances | 50,000 |
| Legal + compliance (initial review) | SBP compliance consultant | 300,000 |
| **Total one-time build cost** | | **~2,300,000 (~USD 8,200)** |

*Note: This is the full commercial build cost. The current project was built
by a single student as an academic capstone over ~3 months — the student
version's **cash cost is near zero**, and this figure represents what a
commercial team would need.*

### 3.2 Operating cost (annual, at low scale)

For a single-bank deployment, small customer base:

| Item | Cost (PKR/year) |
|---|---|
| Cloud infrastructure (compute + storage) | 600,000 |
| Model retraining + monitoring | 400,000 (part-time data scientist) |
| Support + customer success | 500,000 (part-time) |
| Compliance audits | 200,000 |
| **Total annual operating cost** | **~1,700,000 (~USD 6,000)** |

### 3.3 Cost per customer at scale

At 10 deploying banks, operating cost per customer drops to roughly
PKR 170,000 per year — high enough that the product would need to charge
PKR 500,000+ per customer per year to remain viable, which is realistic
for a B2B SaaS deal but requires strong value proof.

---

## 4. Revenue Model

### 4.1 Options considered

| Model | Description | Fit for Pakistan |
|---|---|---|
| **B2B SaaS subscription** | Fixed annual fee per bank | Best — predictable |
| **Per-API-call pricing** | Charge per credit decision | Weak — banks have low decision volume |
| **Enterprise license** | One-time fee + support | Workable for large banks |
| **Freemium + premium features** | Free basic, paid advanced | Weak — banks don't use freemium |
| **Consulting + customization** | Charged project fees | Works as add-on |

### 4.2 Recommended model

**B2B SaaS subscription**, tiered by customer count:

| Tier | Customers | Annual price (PKR) |
|---|---|---|
| Starter | Up to 100,000 customers | 600,000 (~USD 2,200) |
| Growth | 100,000–500,000 | 1,500,000 (~USD 5,400) |
| Enterprise | 500,000+ | 3,000,000 (~USD 10,800) |

*Note: These are proposed prices, not validated by customer willingness-
to-pay research (which would be part of a commercial Phase 2).*

### 4.3 Adjacent revenue

- **Custom model development** — PKR 500,000–1,500,000 per project
- **Compliance and audit reporting** — PKR 300,000/year
- **Training and integration services** — PKR 200,000 per engagement

---

## 5. Value to a Single Mid-Size Bank (Concrete Calculation)

### 5.1 Assumptions

- Bank has **500,000 credit card customers**
- Average credit card outstanding balance: **PKR 150,000** (a mix of low
  and high users; typical mid-size Pakistani bank)
- Annual **default rate**: **2.0%** (benchmark: Pakistani credit card NPLs
  have historically run between 1.5% and 3%; verify current rate from SBP
  Banking Surveillance Department data)

### 5.2 Baseline expected loss

**Baseline annual expected loss: PKR 1.5 billion (~USD 5.4 million)**

### 5.3 CreditLens impact scenarios

Based on published literature on AI credit scoring improvements (typically
2–8% relative reduction in default rate at comparable accuracy levels),
three scenarios:

| Scenario | Relative reduction | Absolute reduction | Annual savings (PKR) |
|---|---|---|---|
| **Conservative** | 2% | 200 defaults | **30,000,000** |
| **Moderate** | 4% | 400 defaults | **60,000,000** |
| **Optimistic** | 8% | 800 defaults | **120,000,000** |

*Literature caveat: AI credit scoring typically improves the *cost-weighted*
accuracy of decisions rather than the raw default rate. A bank using the
tool would catch some defaults earlier (reducing exposure) and price risk
more accurately (higher revenue on safe customers). The exact savings
depend on how the bank uses the tool. These figures represent the upper
bound of pure default-avoidance savings.*

### 5.4 ROI for a single bank

Conservative scenario:
- **Annual savings**: PKR 30 million
- **CreditLens cost** (Growth tier): PKR 1.5 million
- **Net annual benefit**: PKR 28.5 million
- **ROI**: **1,900%** (19× return on investment)

Even in the most pessimistic case (bank saves only PKR 5 million per year),
ROI is **233%** — still a compelling purchase.

---

## 6. Break-Even Analysis

### 6.1 For the vendor (CreditLens as a business)

**Interpretation:** break-even requires signing **~2–3 customers in Year 1**
to recover the build cost plus first-year operating cost, and **~2 customers
per year thereafter** to cover ongoing operations.

Given the target market of 15–20 qualifying banks, capturing 2–3
customers is a realistic Year-1 goal.

### 6.2 For a single customer (bank)

Even conservative savings (PKR 5M/year) exceed the price (PKR 1.5M/year).
The deal pays for itself within the first year.

---

## 7. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Regulatory** — SBP does not approve AI-based credit scoring for automated decisions | Medium | High | Position CreditLens as **decision-support**, not decision-making. Human analyst remains in the loop. All predictions have explanation and disclaimer. |
| **Data privacy** — PII exposure during model training or inference | Medium | High | Local deployment (bank's servers). No customer data leaves the bank. Documented in technical spec. |
| **Model drift** — Customer behaviour changes over time; model degrades | High | Medium | Retraining pipeline exists. Recommend quarterly retraining. Model monitoring is a Phase 2 feature. |
| **Explainability trust** — Analysts don't trust SHAP explanations | Medium | Medium | Customer validation will test this. Report feedback honestly. |
| **Competition** — Established vendors (FICO, Experian, local credit bureaus) dominate | High | High | CreditLens targets **explainability-first** positioning; FICO is a black box. Also targets institutions that can't afford FICO pricing. |
| **Adoption resistance** — Banks prefer existing processes | High | Medium | Position as augmentation, not replacement. Free trial period. |
| **Model bias** — Fairness audit reveals concerning group-level disparities | Medium | Medium | Fairness audit is **built in**. This is a differentiator, not a weakness — most commercial tools hide these metrics. |

---

## 8. Pakistan-Specific Considerations

### 8.1 Regulatory environment

- **State Bank of Pakistan (SBP)** regulates all scheduled banks
- **SBP's "Enterprise-wide Risk Management" guidelines** require banks to
  have systematic credit risk assessment
- **SBP's "Digital Prudential Regulations" (2018 onwards)** encourage
  digital tools but require explainability for regulatory review
- **CreditLens directly supports** these requirements via SHAP-based
  explanations
- **Not yet addressed:** formal SBP approval for use in actual lending
  decisions. Would require a **Regulatory Sandbox** or **Innovation
  Challenge** participation (SBP has run these programs — verify current
  availability)

### 8.2 Data localization

- SBP requires certain banking data to remain within Pakistan
- CreditLens is **locally deployable** — no cloud transmission required
- Aligns with SBP data localization guidance

### 8.3 Local competitive landscape

- **FICO** — not directly active in Pakistan at scale
- **Local credit bureaus** (eCIB under SBP, private bureaus like
  DataCheck) — provide raw data, not AI-based scoring
- **Local fintechs** — focus on lending infrastructure rather than risk
  scoring (verify current landscape)
- **Gap:** no Pakistani provider offers **explainable** AI credit scoring
  as a distinct product

### 8.4 Local talent

- Pakistan produces ~25,000+ IT graduates per year (HEC data — verify)
- Data science talent available but competitive
- Local talent acquisition feasible at the salaries stated in Section 3

---

## 9. Comparison with Alternatives

| Solution | Strengths | Weaknesses relative to CreditLens |
|---|---|---|
| **Traditional scorecard (regression-based)** | Simple, regulator-friendly | Limited nonlinear pattern capture; lower accuracy on complex behavior |
| **FICO / commercial black-box** | Proven track record | **No explainability**. Expensive. |
| **Open-source models (no wrapper)** | Free | No UI, no explanation, no fairness audit |
| **Manual analyst review** | Human judgment | Slow, inconsistent, expensive per decision |
| **CreditLens** | Explainability + accuracy + fairness + local deployability | Still a prototype; needs production hardening |

**Positioning statement:** "CreditLens fills the gap between black-box
commercial tools and traditional rule-based scorecards, delivering AI
accuracy with analyst-grade transparency."

---

## 10. Summary Table

| Dimension | Estimate |
|---|---|
| Cost to build (commercial) | PKR 2.3M (~USD 8.2K) |
| Cost to build (student version) | ~PKR 0 (student's own time) |
| Annual operating cost | PKR 1.7M (~USD 6K) |
| Target market (SAM) | 15–20 banks in Pakistan |
| Realistic Year-1 target | 2–4 institutions |
| Price per customer (mid-tier) | PKR 1.5M/year |
| Value to mid-size bank (conservative) | PKR 30M/year saved |
| Value to mid-size bank (moderate) | PKR 60M/year saved |
| Customer ROI | 1,900% (conservative) |
| Vendor break-even | 2–3 customers |

---

## 11. Honest Limitations

1. **No customer willingness-to-pay validation** — pricing tiers are
   proposed, not tested. Real validation would require discussions with
   2–3 bank procurement teams.

2. **Pakistan credit card market data** — SBP reports credit card numbers
   quarterly; the figures used here are estimates. **Verify from the latest
   SBP Payment Systems Review** before the pitch.

3. **AI credit scoring improvements** are typically measured as
   cost-weighted accuracy, not raw default-rate reduction. The 2–8%
   improvement range is from published literature on similar datasets;
   real Pakistani bank results could differ.

4. **Regulatory pathway** — getting SBP approval for use in actual lending
   decisions is a multi-year process. CreditLens is positioned as
   decision-support, which can be deployed today (aided by explainability).

5. **Commercial build cost assumes hiring** — a solo student-built version
   is essentially free, but production-grade deployment (security audits,
   monitoring, 24/7 support) requires a team.

6. **Market size estimates** — TAM/SAM/SOM frameworks are inherently
   approximate. The most defensible number is **"15–20 banks with active
   credit card portfolios in Pakistan"** (SBP, verifiable).

---

## 12. For the Pitch Deck

The pitch should highlight:

- **Problem:** PKR 1.5B+ annual credit card defaults at a single mid-size bank
- **Solution:** AI risk scoring that explains itself
- **Value:** PKR 30M+ conservative annual savings per bank
- **Market:** 15–20 banks in Pakistan + microfinance + fintech
- **Cost:** PKR 2.3M to build, PKR 1.5M per customer annual revenue
- **Break-even:** 2–3 customers
- **Risk mitigation:** Decision-support positioning, local deployability,
  built-in fairness audit

**The single most important sentence for judges:** *"If CreditLens saves a
mid-size Pakistani bank even 2% of its credit card defaults, it saves PKR
30 million per year. The tool costs PKR 1.5 million. The math is not close."*