# Predicting Retail Sales Channels

A comparative machine-learning study on multi-channel footwear retail data, benchmarking KNN and Naïve Bayes to predict transaction fulfilment channel.

**Author:** Suvankar Das

---

## Problem Statement

Given a retail transaction's price, volume, retailer, and region, can we accurately predict whether it was fulfilled **Online, In-Store, or at an Outlet** — and does this predictive relationship hold consistently across competing brands (Nike vs. Adidas)?

## Dataset

18,997 transactional records across 9 attributes, combining `Nike_Sales.csv` (9,360 rows) and `Adidas_Sales.csv` (9,637 rows) into one brand-unbiased schema.

| Attribute | Type | Description |
|---|---|---|
| Sales Method | Target (categorical) | Fulfilment channel: Online, Outlet, or In-Store |
| Total Sales | Continuous | Revenue per transaction (USD) |
| Price per Unit | Continuous | $7–$110 |
| Units Sold | Continuous | 0–1,275 |
| Product | Categorical (6 levels) | Men's/Women's Street & Athletic Footwear, Apparel |
| Retailer | Categorical (6 levels) | Foot Locker, West Gear, Sports Direct, Kohl's, Amazon, Walmart |
| Region / State | Categorical | 5 US regions |
| Invoice Date | Identifier | Purchase timestamp |

**Data cleaning:** normalised headers to Title Case, dropped Adidas-only columns to match Nike's schema, recomputed `Total Sales = Price per Unit × Units Sold`, standardised text casing, and merged both brands into one complete dataset with zero missing values.

## Method

Built and compared two classifiers in **Orange Data Mining**:

- **K-Nearest Neighbors (KNN)** — instance-based, using Euclidean distance across normalized numeric predictors (Price per Unit, Units Sold, Retailer, Region). Tuned k = 3, 5, 7, 10 via cross-validation; k = 10 gave the best result.
- **Naïve Bayes** — probabilistic baseline for comparison.

Both were trained/tested on the same Adidas split (Test & Score widget), then the same KNN configuration was applied independently to the Nike dataset to test cross-brand generalisation.

## Results

| Metric | KNN | Naïve Bayes |
|---|---|---|
| Accuracy | 81.0% | 67.1% |
| F1-Score | 0.810 | 0.669 |
| AUC | 0.933 | 0.850 |
| MCC | 0.688 | 0.465 |

- KNN correctly classified 7,807/9,637 Adidas transactions vs. Naïve Bayes's 6,465/9,637.
- Both models confuse Online ↔ Outlet most often; Naïve Bayes's weakest class is Outlet (53% recall, near chance level).
- Pairwise test: P(KNN > Naïve Bayes) = 1.000 — the performance gap is statistically certain.

**Cross-brand generalisation:** the same KNN model scored 80.7% accuracy on Nike (vs. 81.0% on Adidas) — a gap under 0.4 on every metric, well within normal fold-to-fold variation. This suggests channel-choice patterns are structural to footwear retail generally, not brand-specific.

**Key visual finding:** Online dominates most US regions, but the South is an anomaly — In-Store sales are nearly negligible there, while Online and Outlet run neck-and-neck.

## Future Scope

- Add product-category and state-level granularity as predictors
- Extend the benchmark to Random Forest / Gradient Boosting
- Use Invoice Date to model seasonal demand trends
- Personalised channel recommendations at checkout (e.g. "cheaper at your nearest Outlet")

## Tools

`Orange Data Mining` `KNN` `Naïve Bayes`

## References

- [Orange Data Mining Toolkit](https://orangedatamining.com)
