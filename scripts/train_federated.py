"""Usage: python scripts/train_federated.py [--config configs/default.yaml]
          [--mode federated|local|centralized] [--strategy fedavg|fedprox] [--dp]"""
import argparse, copy, os, sys
import numpy as np, torch, yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedkg.data import load_factory, FEATURES
from fedkg.models import LSTMRegressor
from fedkg.federated import run_federated, predict, local_train, make_loader
from fedkg.metrics import rmse, nasa_score


def fit_single(model, X, y, f, device):
    loader = make_loader(X, y, f["batch_size"])
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for _ in range(f["rounds"]):
        delta = local_train(model, state, loader, 1, f["lr"], 0.0, device)
        state = {k: state[k] + delta[k] for k in state}
    model.load_state_dict(state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--strategy", choices=["fedavg", "fedprox"])
    ap.add_argument("--mode", choices=["federated", "local", "centralized"], default="federated")
    ap.add_argument("--dp", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.strategy: cfg["federated"]["strategy"] = a.strategy
    if a.dp: cfg["privacy"]["enabled"] = True

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, m, f = cfg["data"], cfg["model"], cfg["federated"]
    names = d["clients"]
    data = {n: load_factory(d["root"], n, d["window"], d["rul_cap"]) for n in names}
    train_names = [n for n in names if n != d["holdout"]]
    new_model = lambda: LSTMRegressor(len(FEATURES), m["hidden"], m["layers"], m["dropout"])

    def report(tag, model):
        for n in names:
            p = predict(model, data[n]["X_test"], device)
            held = " (HELD-OUT)" if n == d["holdout"] else ""
            print(f"[{tag}] {n}{held}: RMSE={rmse(p, data[n]['y_test']):.2f}  "
                  f"NASA={nasa_score(p, data[n]['y_test']):.0f}")

    if a.mode == "federated":
        model = new_model()
        cb = lambda rnd, mod: print(f"round {rnd}") if rnd % 5 == 0 else None
        run_federated(model, {n: data[n] for n in train_names}, cfg, device, cb)
        report(f"{f['strategy']}{'+DP' if cfg['privacy']['enabled'] else ''}", model)
    elif a.mode == "centralized":
        X = np.concatenate([data[n]["X_train"] for n in train_names])
        y = np.concatenate([data[n]["y_train"] for n in train_names])
        report("centralized", fit_single(new_model(), X, y, f, device))
    else:  # local-only: one model per factory
        for n in train_names:
            report(f"local-{n}", fit_single(new_model(), data[n]["X_train"], data[n]["y_train"], f, device))


if __name__ == "__main__":
    main()
