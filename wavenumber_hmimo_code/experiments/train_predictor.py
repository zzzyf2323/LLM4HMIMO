# experiments/train_predictor.py
from __future__ import annotations

import os
import argparse
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from models.predictor import TinyCausalPredictor, PredictorConfig, predictor_loss, pool_frame_tokens


class TemporalTokenDataset(Dataset):
    def __init__(self, bundle: dict):
        self.dataset = bundle["dataset"]
        self.meta = bundle["meta"]
        self.history = int(self.meta["history"])

        self.samples = []
        for seq in self.dataset:
            T = len(seq["tokens"])
            for t in range(self.history, T):
                x_hist = []
                for tau in range(t - self.history, t):
                    Z_tau = seq["tokens"][tau]      # [Ntok,d]
                    x_hist.append(pool_frame_tokens(Z_tau))
                x_hist = torch.stack(x_hist, dim=0)  # [L,d]

                b_next = seq["support"][t]
                e_next = seq["energy"][t]

                self.samples.append((x_hist, b_next, e_next))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    xs, bs, es = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(bs, dim=0), torch.stack(es, dim=0)


def split_dataset(full_dataset, train_ratio=0.9, seed=2026):
    n = len(full_dataset)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(train_ratio * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)
    return train_subset, val_subset


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_b = 0.0
    total_e = 0.0
    total_n = 0

    for x, b, e in loader:
        x = x.to(device)
        b = b.to(device)
        e = e.to(device)

        support_logits, energy_pred = model(x)
        loss, info = predictor_loss(support_logits, energy_pred, b, e)

        bs = x.shape[0]
        total_loss += info["loss"] * bs
        total_b += info["loss_b"] * bs
        total_e += info["loss_e"] * bs
        total_n += bs

    return {
        "loss": total_loss / max(total_n, 1),
        "loss_b": total_b / max(total_n, 1),
        "loss_e": total_e / max(total_n, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/temporal_dataset.pt")
    parser.add_argument("--out", type=str, default="data/checkpoints/wavenumber_predictor.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = torch.load(args.data, map_location="cpu")
    G = int(bundle["meta"]["G"])

    full_dataset = TemporalTokenDataset(bundle)
    train_set, val_set = split_dataset(full_dataset, train_ratio=0.9, seed=args.seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TinyCausalPredictor(
        PredictorConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_ff=256,
            dropout=0.1,
            G=G,
        )
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        for x, b, e in train_loader:
            x = x.to(device)
            b = b.to(device)
            e = e.to(device)

            optimizer.zero_grad()
            support_logits, energy_pred = model(x)
            loss, info = predictor_loss(support_logits, energy_pred, b, e)
            loss.backward()
            optimizer.step()

        train_info = evaluate(model, train_loader, device)
        val_info = evaluate(model, val_loader, device)

        print(
            f"[Epoch {ep:03d}] "
            f"train loss={train_info['loss']:.4f} "
            f"(b={train_info['loss_b']:.4f}, e={train_info['loss_e']:.4f}) | "
            f"val loss={val_info['loss']:.4f} "
            f"(b={val_info['loss_b']:.4f}, e={val_info['loss_e']:.4f})"
        )

        if val_info["loss"] < best_val:
            best_val = val_info["loss"]
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "num_layers": args.num_layers,
                    "G": G,
                },
                "meta": bundle["meta"],
            }, args.out)
            print(f"[Saved best] {args.out}")


if __name__ == "__main__":
    main()