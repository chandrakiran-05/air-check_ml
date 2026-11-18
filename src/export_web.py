"""Export a compact gradient-boosting model + aggregated data as JSON for the static web app."""
import json
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.pipeline import FEATURES, POLLUTANTS, ROOT, aqi_bucket, load_data

OUT = ROOT / "public"


def r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def export_model(df):
    data = df.dropna(subset=["AQI"]).copy()
    for col in POLLUTANTS:
        data[col] = data[col].fillna(data.groupby("City")[col].transform("median"))
        data[col] = data[col].fillna(data[col].median())
    X, y = data[FEATURES], data["AQI"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                                   subsample=0.9, random_state=42).fit(X_tr, y_tr)
    pred = gb.predict(X_te)
    trees = []
    for est in gb.estimators_[:, 0]:
        t = est.tree_
        trees.append({"f": t.feature.tolist(), "t": [r(v, 4) for v in t.threshold],
                      "l": t.children_left.tolist(), "r": t.children_right.tolist(),
                      "v": [r(v, 4) for v in t.value[:, 0, 0]]})
    model = {"features": FEATURES, "init": float(gb.init_.constant_[0][0]),
             "lr": gb.learning_rate, "trees": trees}
    idx = np.random.default_rng(0).choice(len(y_te), 1200, replace=False)
    metrics = {
        "algorithm": "Gradient Boosting (300 trees, depth 4)",
        "r2": r(r2_score(y_te, pred), 4), "mae": r(mean_absolute_error(y_te, pred)),
        "rmse": r(np.sqrt(mean_squared_error(y_te, pred))),
        "bucket_accuracy": r(np.mean([aqi_bucket(a) == aqi_bucket(b) for a, b in zip(y_te, pred)]), 4),
        "n_train": len(X_tr), "n_test": len(X_te),
        "importance": dict(sorted(zip(FEATURES, gb.feature_importances_.round(4).tolist()),
                                  key=lambda kv: -kv[1])),
        "defaults": X.median().round(2).to_dict(),
        "sample": {"actual": [r(v, 1) for v in y_te.values[idx]], "predicted": [r(v, 1) for v in pred[idx]]},
    }
    return model, metrics


def export_data(df):
    df = df.copy()
    df["ym"] = df["Date"].dt.strftime("%Y-%m")
    g = df.groupby(["City", "ym"])
    agg = g[POLLUTANTS + ["AQI"]].mean()
    counts = g.size().rename("n")
    buckets = df.pivot_table(index=["City", "ym"], columns="AQI_Bucket", values="Date",
                             aggfunc="count", fill_value=0)
    rows = []
    for key in agg.index:
        rec = {"c": key[0], "m": key[1], "n": int(counts[key])}
        rec.update({p: r(agg.at[key, p]) for p in POLLUTANTS + ["AQI"]})
        rec["b"] = {k: int(v) for k, v in buckets.loc[key].items() if v} if key in buckets.index else {}
        rows.append(rec)
    corr = df[POLLUTANTS + ["AQI"]].corr().round(2)
    return {"rows": rows, "corr": {"labels": corr.columns.tolist(), "z": corr.values.tolist()}}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    df = load_data()
    model, metrics = export_model(df)
    (OUT / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    (OUT / "metrics.json").write_text(json.dumps(metrics, separators=(",", ":")))
    (OUT / "data.json").write_text(json.dumps(export_data(df), separators=(",", ":")))
    print({k: metrics[k] for k in ("r2", "mae", "rmse", "bucket_accuracy")})
