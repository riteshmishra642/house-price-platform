# Business Understanding

## 1. Business Problem

Buying, selling, financing, and insuring residential property all depend on one
number: **what is this house actually worth?**

Today that number is produced in ways that are slow, inconsistent, or biased:

- **Real estate agents** rely on "comparable sales" (comps) — manual, subjective,
  and slow to update when a market shifts.
- **Banks and mortgage lenders** order formal appraisals that take days and cost
  the buyer/seller money, and can vary significantly between appraisers for the
  same house.
- **Homeowners** have almost no way to sanity-check an agent's or bank's number
  without paying for a second opinion.
- **Online listing platforms** (Zillow, Redfin, Realtor.com) have automated
  valuation models (AVMs), but many mid-size proptech companies and regional
  brokerages do not — this is a real, fundable gap.

The core problem: **turn a property's raw attributes (size, age, location,
quality, layout) into an accurate, explainable, instantly-available price
estimate.**

## 2. Target Users

| User | What they need from this system |
|---|---|
| **Home buyers** | A fast sanity check before making an offer |
| **Home sellers** | Data-backed listing price guidance |
| **Real estate agents/brokerages** | A tool to defend a suggested price to clients |
| **Mortgage lenders/underwriters** | A secondary check against formal appraisals |
| **Real estate investors** | Rapid screening of many properties for undervalued deals |
| **Proptech companies** | A core valuation engine to build products on top of |
| **Insurance companies** | Replacement-cost and risk-pricing inputs |

## 3. Real Estate Industry Challenges This Addresses

- **Appraisal subjectivity** — two human appraisers can disagree by 10%+ on the
  same house.
- **Speed** — a formal appraisal takes days; a model can respond in
  milliseconds.
- **Cost** — appraisals cost $300–$700 each; an automated first-pass estimate
  is nearly free at scale.
- **Scale** — a human cannot price 10,000 homes overnight before a portfolio
  acquisition decision; a model can.
- **Explainability gap** — most cheap AVMs online are black boxes ("Zestimate
  says $412,000") with no reasoning. Buyers and agents don't trust numbers they
  can't interrogate — this is why our platform makes **Explainable AI (Step 10)**
  a first-class feature, not an afterthought.

## 4. Business Goals

1. Predict sale price within an acceptable error margin (defined by KPIs below).
2. Provide **transparent, per-prediction reasoning** (which features pushed the
   price up/down and by how much).
3. Serve predictions through a **production API** usable by other software
   (a lender's internal tool, a brokerage's website, etc.), not just a notebook.
4. Provide a **self-serve dashboard** non-technical users (agents, buyers) can
   use directly.
5. Be **monitorable and maintainable** after deployment — models decay as
   markets shift, so the system must support retraining and drift detection.

## 5. Key Performance Indicators (KPIs)

These are the numbers a business stakeholder would actually ask about —
notice most are *business* framings of *statistical* metrics we'll compute in
Step 9:

| KPI | Business meaning | Statistical proxy |
|---|---|---|
| Pricing accuracy | "How close is the estimate to the real sale price, typically?" | MAE, RMSE (Step 9) |
| % of predictions within 10% of actual price | "How often can a user trust this number outright?" | Custom threshold accuracy |
| Explainability coverage | "Can we justify every single prediction if a client asks?" | SHAP available for 100% of predictions (Step 10) |
| API response time | "Is this fast enough to embed in a live website?" | p95 latency (Step 12) |
| Model staleness | "Do we know when the model needs retraining?" | Drift/monitoring metrics (Step 14) |

## 6. Expected Business Benefits

- **Faster transactions**: instant estimates shorten the time between listing
  and offer.
- **Lower cost per valuation**: near-zero marginal cost vs. a paid appraisal.
- **Trust and adoption**: explainability builds user trust, which drives
  repeat usage — a key proptech growth lever.
- **Scalable due diligence**: investors/lenders can screen large property
  portfolios automatically.

## 7. Potential Revenue Impact (illustrative, not guaranteed)

This is the kind of framing that shows business awareness in an interview —
you don't need real revenue, you need to show you *thought about it*:

- **B2B licensing**: charging brokerages/lenders a per-query or subscription
  fee to embed the valuation API into their own products.
- **Freemium consumer tool**: free basic estimate, paid detailed report
  (comparable to how real AVM products monetize).
- **Reduced appraisal costs** for a lending partner who uses the model as a
  first-pass filter before ordering a small number of full appraisals.
- **Lead generation**: a free valuation tool on a brokerage's site captures
  buyer/seller contact info, valuable for later services.

---

**Why this document matters for you as a candidate**: when someone in an
interview asks "walk me through this project," starting with *this* page
instead of "I loaded a CSV and trained an XGBoost" is what signals product
thinking over notebook thinking.
