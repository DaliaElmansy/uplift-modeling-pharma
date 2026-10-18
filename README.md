# Uplift Modeling for HCP Educational Outreach

A demonstration of **uplift modeling and heterogeneous treatment effect estimation** in a pharmaceutical commercial setting using fully synthetic data.

## Business Question

A commercial analytics team has limited capacity for an HCP educational outreach program.

A conventional response model asks:

> Who is most likely to engage?

An uplift model asks:

> Who is more likely to engage **because of the intervention**?

The goal is to estimate the conditional average treatment effect:

`tau(x) = E[Y(1) - Y(0) | X=x]`

This distinction is important because an HCP with a high baseline probability of engagement may have engaged regardless of whether outreach occurred. Uplift modeling instead focuses on **incremental impact**.

## Why Uplift Modeling?

Traditional predictive modeling estimates the probability of an outcome:

`P(Y=1 | X)`

Uplift modeling instead estimates the change in outcome probability attributable to an intervention:

`P(Y=1 | T=1, X) - P(Y=1 | T=0, X)`

This enables targeting based on expected incremental response rather than baseline propensity.

Conceptually, individuals may fall into groups such as:

- **Persuadables** — more likely to respond because of the intervention
- **Sure things** — likely to respond with or without intervention
- **Lost causes** — unlikely to respond under either condition
- **Negative responders** — intervention may reduce the desired outcome

## Dataset

The repository uses a **fully synthetic randomized experiment**. No patient, HCP, proprietary, or company data are used.

Synthetic HCP-level variables include:

- baseline engagement
- patient volume
- digital engagement
- prior educational interactions
- formulary access
- specialty indicators

Treatment represents an educational outreach intervention and is randomized 50/50.

The data-generating process includes a known heterogeneous treatment effect, which allows estimated uplift to be compared with the true simulated treatment effect.

## Methods

Two meta-learning approaches are implemented.

### T-Learner

Separate outcome models are trained for the treatment and control populations:

`tau_hat(x) = mu1_hat(x) - mu0_hat(x)`

### S-Learner

A single model is trained using treatment assignment as an additional feature.

Each observation is then scored twice:

`tau_hat(x) = mu_hat(x, T=1) - mu_hat(x, T=0)`

Gradient-boosted decision trees are used to capture nonlinear effects and interactions.

## Evaluation

The project evaluates model performance using:

- uplift by predicted decile
- cumulative incremental gain
- Qini-style curves
- Qini coefficient
- correlation with the known true treatment effect

Because the dataset is synthetic, the true individual uplift is available for validation.

## Results

Example results:

- **T-Learner Qini coefficient:** 53.52
- **S-Learner Qini coefficient:** 54.86
- **Correlation between T-Learner estimated uplift and true uplift:** 0.669

Higher-ranked uplift groups demonstrate materially larger treatment-control response differences than lower-ranked groups.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis:

```bash
python src/uplift_modeling.py
```

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── uplift_modeling.py
├── notebooks/
│   └── uplift_modeling_walkthrough.ipynb
├── data/
│   ├── synthetic_hcp_outreach.csv
│   └── uplift_deciles.csv
└── figures/
    ├── qini_curve.png
    └── uplift_by_decile.png
```

## Production Considerations

This repository focuses on modeling and evaluation rather than production deployment.

A production implementation would additionally require:

- validated and reproducible data pipelines
- model and experiment versioning
- model monitoring and drift detection
- automated testing
- governance and auditability
- integration with enterprise data and AI platforms
- privacy and regulatory review
- appropriate human oversight

## Key Takeaway

A conventional response model identifies individuals who are likely to respond.

An uplift model identifies individuals whose outcomes are most likely to **change because of the intervention**.

For intervention targeting and resource allocation, that distinction can materially improve decision quality.

## Disclaimer

This project is for educational and technical demonstration purposes. All data are synthetic and do not represent real patients, healthcare professionals, products, or commercial activities.
