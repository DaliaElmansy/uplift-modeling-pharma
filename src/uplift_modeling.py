
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
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
    return 1 / (1 + np.exp(-z))

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

    mu0 = (-2.4 + 2.0*baseline + 0.9*digital + 0.55*volume +
           0.35*prior + 0.45*access + 0.20*sp1 - 0.15*sp2)

    tau_logit = (0.15 + 1.15*digital + 0.75*access + 0.55*sp1 -
                 0.95*prior - 0.55*(baseline > 0.75) - 0.45*sp2)

    p0 = sigmoid(mu0)
    p1 = sigmoid(mu0 + tau_logit)
    true_uplift = p1 - p0
    p_obs = np.where(treatment == 1, p1, p0)
    outcome = rng.binomial(1, p_obs)

    return pd.DataFrame({
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
    })

def fit_t_learner(train, test):
    mt = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=220, random_state=SEED)
    mc = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=220, random_state=SEED+1)
    mt.fit(train.loc[train.treatment==1, FEATURES], train.loc[train.treatment==1, "outcome"])
    mc.fit(train.loc[train.treatment==0, FEATURES], train.loc[train.treatment==0, "outcome"])
    p1 = mt.predict_proba(test[FEATURES])[:,1]
    p0 = mc.predict_proba(test[FEATURES])[:,1]
    return p1-p0

def fit_s_learner(train, test):
    sf = FEATURES + ["treatment"]
    m = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=260, random_state=SEED+2)
    m.fit(train[sf], train["outcome"])
    x1 = test[FEATURES].copy()
    x0 = test[FEATURES].copy()
    x1["treatment"] = 1
    x0["treatment"] = 0
    return m.predict_proba(x1[sf])[:,1] - m.predict_proba(x0[sf])[:,1]

def gain_curve(df, score):
    d = df.sort_values(score, ascending=False).reset_index(drop=True)
    t = d.treatment.to_numpy()
    y = d.outcome.to_numpy()
    nt = np.cumsum(t)
    nc = np.cumsum(1-t)
    yt = np.cumsum(y*t)
    yc = np.cumsum(y*(1-t))
    gain = yt - yc * np.divide(nt, np.maximum(nc, 1))
    frac = (np.arange(len(d))+1)/len(d)
    return frac, gain

def qini(df, score):
    x, g = gain_curve(df, score)
    return float(np.trapz(g - x*g[-1], x))

def uplift_deciles(df, score):
    d = df.copy()
    d["decile"] = pd.qcut(d[score].rank(method="first", ascending=False), 10, labels=range(1,11))
    rows = []
    for decile, g in d.groupby("decile", observed=True):
        yt = g.loc[g.treatment==1, "outcome"].mean()
        yc = g.loc[g.treatment==0, "outcome"].mean()
        rows.append({
            "decile": int(decile),
            "n": len(g),
            "treated_response_rate": yt,
            "control_response_rate": yc,
            "observed_uplift": yt-yc,
            "mean_predicted_uplift": g[score].mean(),
            "mean_true_uplift": g.true_uplift.mean(),
        })
    return pd.DataFrame(rows).sort_values("decile")

def main():
    df = make_data()
    df.to_csv(DATA/"synthetic_hcp_outreach.csv", index=False)

    train, test = train_test_split(df, test_size=0.35, random_state=SEED, stratify=df["treatment"])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    test["uplift_t"] = fit_t_learner(train, test)
    test["uplift_s"] = fit_s_learner(train, test)

    dec = uplift_deciles(test, "uplift_t")
    dec.to_csv(DATA/"uplift_deciles.csv", index=False)

    print("T-learner Qini:", round(qini(test, "uplift_t"), 2))
    print("S-learner Qini:", round(qini(test, "uplift_s"), 2))
    print("T-learner corr(true uplift):", round(np.corrcoef(test["uplift_t"], test["true_uplift"])[0,1], 3))
    print(dec.to_string(index=False))

    xt, gt = gain_curve(test, "uplift_t")
    xs, gs = gain_curve(test, "uplift_s")

    plt.figure(figsize=(8,5))
    plt.plot(xt, gt, label="T-learner")
    plt.plot(xs, gs, label="S-learner")
    plt.plot(xt, xt*gt[-1], "--", label="Random targeting")
    plt.xlabel("Fraction targeted")
    plt.ylabel("Estimated incremental responders")
    plt.title("Qini-style cumulative gain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG/"qini_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.bar(dec["decile"].astype(str), dec["observed_uplift"])
    plt.xlabel("Predicted uplift decile (1 = highest)")
    plt.ylabel("Observed treated-control response difference")
    plt.title("Observed uplift by predicted decile")
    plt.tight_layout()
    plt.savefig(FIG/"uplift_by_decile.png", dpi=160)
    plt.close()

if __name__ == "__main__":
    main()
