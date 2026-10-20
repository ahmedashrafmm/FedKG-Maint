"""FedAvg / FedProx simulated on one machine."""
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .privacy import clip_update, add_noise


def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def local_train(model, global_state, loader, epochs, lr, mu, device):
    """Train locally; mu > 0 adds the FedProx proximal term. Returns the weight delta."""
    model.load_state_dict(global_state)
    model.to(device).train()
    ref = {k: v.detach().clone().to(device) for k, v in global_state.items()}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if mu > 0:
                prox = sum(((p - ref[n]) ** 2).sum() for n, p in model.named_parameters())
                loss = loss + 0.5 * mu * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    new_state = model.state_dict()
    return {k: (new_state[k].detach().cpu() - global_state[k].cpu()).float() for k in global_state}


@torch.no_grad()
def predict(model, X, device, batch_size=512):
    model.to(device).eval()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model(torch.from_numpy(X[i:i + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(out)


def run_federated(model, clients, cfg, device, eval_fn=None):
    """clients: dict name -> data dict. Returns the trained global model."""
    fcfg, pcfg = cfg["federated"], cfg["privacy"]
    mu = fcfg["mu"] if fcfg["strategy"] == "fedprox" else 0.0
    global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    loaders = {n: make_loader(d["X_train"], d["y_train"], fcfg["batch_size"]) for n, d in clients.items()}
    sizes = {n: len(d["y_train"]) for n, d in clients.items()}
    total = sum(sizes.values())

    for rnd in range(1, fcfg["rounds"] + 1):
        deltas = []
        for n in clients:
            d = local_train(copy.deepcopy(model), global_state, loaders[n],
                            fcfg["local_epochs"], fcfg["lr"], mu, device)
            if pcfg["enabled"]:
                d = clip_update(d, pcfg["clip_norm"])
            deltas.append((sizes[n] / total, d))
        avg = {k: sum(w * d[k] for w, d in deltas) for k in global_state}
        if pcfg["enabled"]:
            avg = add_noise(avg, pcfg["clip_norm"], pcfg["noise_multiplier"], len(clients))
        global_state = {k: global_state[k] + avg[k] for k in global_state}
        model.load_state_dict(global_state)
        if eval_fn:
            eval_fn(rnd, model)
    return model
