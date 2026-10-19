
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"

FEATURES = [
    "baseline_engagement",
    "patient_volume",
    "digital_engagement",
    "prior_education",
    "formulary_access",
    "specialty_1",
    "specialty_2",
]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_data(n=12000, seed=SEED):
    rng = np.random.default_rng(seed)

    baseline = rng.beta(2.2, 3.5, n)
    volume = np.clip(rng.normal(0.55, 0.20, n), 0, 1)
    digital = rng.beta(2.0, 2.3, n)
    prior = np.clip(rng.poisson(2.0, n) / 6.0, 0, 1)
    access = rng.binomial(1, 0.68, n)

    specialty = rng.integers(0, 3, n)
    sp1 = (specialty == 1).astype(int)
    sp2 = (specialty == 2).astype(int)

    treatment = rng.binomial(1, 0.5, n)

    # Baseline response without treatment.
    mu0_logit = (
        -2.4
        + 2.0 * baseline
        + 0.9 * digital
        + 0.55 * volume
        + 0.35 * prior
        + 0.45 * access
        + 0.20 * sp1
        - 0.15 * sp2
    )

    # Heterogeneous treatment effect on the log-odds scale.
    tau_logit = (
        0.15
        + 1.15 * digital
        + 0.75 * access
        + 0.55 * sp1
        - 0.95 * prior
        - 0.55 * (baseline > 0.75)
        - 0.45 * sp2
    )

    p0 = sigmoid(mu0_logit)
    p1 = sigmoid(mu0_logit + tau_logit)

    # Probability-scale individual treatment effect.
    true_uplift = p1 - p0

    p_observed = np.where(treatment == 1, p1, p0)
    outcome = rng.binomial(1, p_observed)

    return pd.DataFrame(
        {
            "baseline_engagement": baseline,
            "patient_volume": volume,
            "digital_engagement": digital,
            "prior_education": prior,
            "formulary_access": access,
            "specialty_1": sp1,
            "specialty_2": sp2,
            "treatment": treatment,
            "outcome": outcome,
            "true_uplift": true_uplift,
        }
    )


def classifier(seed):
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.06,
        max_iter=220,
        l2_regularization=1.0,
        random_state=seed,
    )


def fit_response_model(train, test):
    """Conventional response model that ignores treatment."""
    model = classifier(SEED + 10)
    model.fit(train[FEATURES], train["outcome"])
    return model.predict_proba(test[FEATURES])[:, 1]


def fit_t_learner(train, test):
    treated = train["treatment"] == 1
    control = train["treatment"] == 0

    model_t = classifier(SEED)
    model_c = classifier(SEED + 1)

    model_t.fit(train.loc[treated, FEATURES], train.loc[treated, "outcome"])
    model_c.fit(train.loc[control, FEATURES], train.loc[control, "outcome"])

    p1 = model_t.predict_proba(test[FEATURES])[:, 1]
    p0 = model_c.predict_proba(test[FEATURES])[:, 1]

    return p1 - p0, p1, p0, model_t, model_c


def fit_s_learner(train, test):
    s_features = FEATURES + ["treatment"]

    model = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.05,
        max_iter=260,
        l2_regularization=1.0,
        random_state=SEED + 2,
    )
    model.fit(train[s_features], train["outcome"])

    x1 = test[FEATURES].copy()
    x1["treatment"] = 1

    x0 = test[FEATURES].copy()
    x0["treatment"] = 0

    p1 = model.predict_proba(x1[s_features])[:, 1]
    p0 = model.predict_proba(x0[s_features])[:, 1]

    return p1 - p0


def fit_dr_learner(train, test):
    """
    Simple doubly robust learner for a randomized binary treatment.

    We fit treatment/control outcome models, construct a DR pseudo-outcome
    using e(x)=0.5, then regress that pseudo-outcome on X.
    """
    treated = train["treatment"] == 1
    control = train["treatment"] == 0

    m1 = classifier(SEED + 20)
    m0 = classifier(SEED + 21)

    m1.fit(train.loc[treated, FEATURES], train.loc[treated, "outcome"])
    m0.fit(train.loc[control, FEATURES], train.loc[control, "outcome"])

    mu1 = m1.predict_proba(train[FEATURES])[:, 1]
    mu0 = m0.predict_proba(train[FEATURES])[:, 1]

    t = train["treatment"].to_numpy()
    y = train["outcome"].to_numpy()
    e = 0.5

    pseudo = (
        mu1
        - mu0
        + t * (y - mu1) / e
        - (1 - t) * (y - mu0) / (1 - e)
    )

    tau_model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=220,
        l2_regularization=2.0,
        random_state=SEED + 22,
    )
    tau_model.fit(train[FEATURES], pseudo)

    return tau_model.predict(test[FEATURES])


def incremental_gain_curve(df, score_col):
    d = df.sort_values(score_col, ascending=False).reset_index(drop=True).copy()

    t = d["treatment"].to_numpy()
    y = d["outcome"].to_numpy()

    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)
    cum_y_t = np.cumsum(y * t)
    cum_y_c = np.cumsum(y * (1 - t))

    gain = cum_y_t - cum_y_c * np.divide(
        cum_t,
        np.maximum(cum_c, 1),
        out=np.zeros_like(cum_t, dtype=float),
        where=np.maximum(cum_c, 1) != 0,
    )

    fraction = (np.arange(len(d)) + 1) / len(d)
    return fraction, gain


def qini_style_area(df, score_col):
    x, gain = incremental_gain_curve(df, score_col)
    random_gain = x * gain[-1]
    return float(np.trapz(gain - random_gain, x))


def observed_uplift(group):
    treated = group[group["treatment"] == 1]["outcome"]
    control = group[group["treatment"] == 0]["outcome"]

    if len(treated) == 0 or len(control) == 0:
        return np.nan

    return treated.mean() - control.mean()


def bootstrap_uplift_ci(group, n_boot=300, seed=SEED):
    rng = np.random.default_rng(seed)

    treated = group[group["treatment"] == 1]["outcome"].to_numpy()
    control = group[group["treatment"] == 0]["outcome"].to_numpy()

    if len(treated) < 2 or len(control) < 2:
        return np.nan, np.nan

    diffs = np.empty(n_boot)

    for i in range(n_boot):
        bt = rng.choice(treated, size=len(treated), replace=True)
        bc = rng.choice(control, size=len(control), replace=True)
        diffs[i] = bt.mean() - bc.mean()

    return np.quantile(diffs, 0.025), np.quantile(diffs, 0.975)


def uplift_by_decile(df, score_col, n_bins=10):
    d = df.copy()

    d["decile"] = pd.qcut(
        d[score_col].rank(method="first", ascending=False),
        q=n_bins,
        labels=np.arange(1, n_bins + 1),
    )

    rows = []

    for decile, group in d.groupby("decile", observed=True):
        yt = group.loc[group["treatment"] == 1, "outcome"].mean()
        yc = group.loc[group["treatment"] == 0, "outcome"].mean()
        lo, hi = bootstrap_uplift_ci(group, seed=SEED + int(decile))

        rows.append(
            {
                "decile": int(decile),
                "n": len(group),
                "treated_response_rate": yt,
                "control_response_rate": yc,
                "observed_uplift": yt - yc,
                "uplift_ci_low": lo,
                "uplift_ci_high": hi,
                "mean_predicted_uplift": group[score_col].mean(),
                "mean_true_uplift": group["true_uplift"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values("decile")


def top_fraction_uplift(df, score_col, fraction=0.20):
    n = max(1, int(len(df) * fraction))
    top = df.sort_values(score_col, ascending=False).head(n)
    return observed_uplift(top)


def main():
    DATA.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    df = make_data()
    df.to_csv(DATA / "synthetic_hcp_outreach.csv", index=False)

    train, test = train_test_split(
        df,
        test_size=0.35,
        random_state=SEED,
        stratify=df["treatment"],
    )

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    test["response_score"] = fit_response_model(train, test)

    (
        test["uplift_t"],
        _,
        _,
        _,
        _,
    ) = fit_t_learner(train, test)

    test["uplift_s"] = fit_s_learner(train, test)
    test["uplift_dr"] = fit_dr_learner(train, test)

    deciles = uplift_by_decile(test, "uplift_t")
    deciles.to_csv(DATA / "uplift_deciles.csv", index=False)

    rows = []

    for label, score in [
        ("Response model", "response_score"),
        ("T-learner", "uplift_t"),
        ("S-learner", "uplift_s"),
        ("DR-learner", "uplift_dr"),
    ]:
        area = qini_style_area(test, score)
        corr = np.corrcoef(test[score], test["true_uplift"])[0, 1]
        top20 = top_fraction_uplift(test, score, 0.20)

        rows.append(
            {
                "model": label,
                "qini_style_area": area,
                "corr_with_true_uplift": corr,
                "observed_uplift_top_20pct": top20,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(DATA / "model_summary.csv", index=False)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nT-learner uplift by decile:")
    print(deciles.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Use one model-to-color mapping across comparison figures.
    model_specs = [
        ("Response model", "response_score"),
        ("T-learner", "uplift_t"),
        ("S-learner", "uplift_s"),
        ("DR-learner", "uplift_dr"),
    ]
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_colors = {
        label: cycle_colors[i % len(cycle_colors)]
        for i, (label, _) in enumerate(model_specs)
    }

    # Figure 1: Qini-style cumulative gain curves
    plt.figure(figsize=(8, 5))

    for label, score in model_specs:
        x, gain = incremental_gain_curve(test, score)
        plt.plot(x, gain, label=label, color=model_colors[label])

    x_ref, gain_ref = incremental_gain_curve(test, "uplift_t")
    plt.plot(
        x_ref,
        x_ref * gain_ref[-1],
        linestyle="--",
        label="Random targeting",
    )

    plt.xlabel("Fraction of population targeted")
    plt.ylabel("Estimated cumulative incremental responders")
    plt.title("Qini-style cumulative gain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "qini_curve.png", dpi=160)
    plt.close()

    # Figure 2: Uplift by decile with bootstrap CI
    err_low = deciles["observed_uplift"] - deciles["uplift_ci_low"]
    err_high = deciles["uplift_ci_high"] - deciles["observed_uplift"]

    plt.figure(figsize=(8, 5))
    plt.bar(deciles["decile"].astype(str), deciles["observed_uplift"])
    plt.errorbar(
        np.arange(len(deciles)),
        deciles["observed_uplift"],
        yerr=[err_low, err_high],
        fmt="none",
        capsize=3,
    )
    plt.xlabel("Predicted uplift decile (1 = highest)")
    plt.ylabel("Observed treated-control response difference")
    plt.title("Observed uplift by predicted decile with 95% bootstrap CI")
    plt.tight_layout()
    plt.savefig(FIG / "uplift_by_decile.png", dpi=160)
    plt.close()

    # Figure 3: Top-20% targeting comparison
    labels = summary["model"].tolist()
    values = summary["observed_uplift_top_20pct"].tolist()

    plt.figure(figsize=(8, 5))
    plt.bar(
        labels,
        values,
        color=[model_colors[label] for label in labels],
    )
    plt.ylabel("Observed uplift in top 20% targeted")
    plt.title("Incremental impact of alternative targeting strategies")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "targeting_comparison.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
