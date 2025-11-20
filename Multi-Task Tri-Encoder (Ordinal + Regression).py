"""
Tri-Encoder Ordinal + Regression (Multi-Task)
---------------------------------------------
- Tri-encoder features (p, r, m, |p-r|, p*r, |m-r|, m*r, |m-p|, m*p)
- Augmentation at embedding level (shuffle prompts, noise responses, swap metrics)
- Multi-task head:
    * Ordinal head -> 9 sigmoids (1..10 thresholds)
    * Regression head -> single continuous output
- Loss = BCE(ordinal) + lambda_reg * MSE(regression)
- Inference: ensemble average(expected_from_ordinal, regression_head)
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Config
# -------------------------
class CFG:
    seed = 42
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    train_json = "train_data.json"
    test_json = "test_data.json"
    metric_names = "metric_names.json"
    metric_emb_npy = "metric_name_embeddings.npy"
    submission_path = "submission_multitask.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_size = 0.25
    batch_size = 64
    lr = 2e-4
    epochs = 12
    hidden = 1024
    dropout = 0.2
    aug_noise_scale = 0.6
    lambda_reg = 1.0  # weight for regression MSE in total loss
    save_path = "best_multitask.pth"

torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)

# -------------------------
# Utility
# -------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# -------------------------
# Data Loaders (extended)
# -------------------------
def load_metric_map(names_file, emb_file):
    with open(names_file, "r", encoding="utf-8") as f:
        metric_names = json.load(f)
    metric_embs = np.load(emb_file).astype(np.float32)
    if len(metric_names) != len(metric_embs):
        raise ValueError("❌ Metric names and embeddings length mismatch")
    metric_map = {name: metric_embs[i] for i, name in enumerate(metric_names)}
    return metric_map, metric_names, metric_embs


def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def clean_dataframe(df, is_train=False):
    for col in ["system_prompt", "user_prompt", "response", "metric_name"]:
        if col not in df.columns:
            df[col] = ""
    df.fillna("", inplace=True)
    if is_train:
        if "score" not in df.columns:
            raise ValueError("❌ 'score' column missing in training data")
        df["score"] = df["score"].astype(float)
        df = df[df["response"].astype(str).str.strip() != ""].reset_index(drop=True)
    return df


def compute_embeddings(df, model_name):
    print("🔹 Computing embeddings...")
    encoder = SentenceTransformer(model_name, device=CFG.device)
    prompts = (df["system_prompt"].astype(str) + " [SEP] " + df["user_prompt"].astype(str)).tolist()
    responses = df["response"].astype(str).tolist()
    p_emb = encoder.encode(prompts, batch_size=CFG.batch_size, show_progress_bar=True, convert_to_numpy=True)
    r_emb = encoder.encode(responses, batch_size=CFG.batch_size, show_progress_bar=True, convert_to_numpy=True)
    df["prompt_emb"] = list(p_emb.astype(np.float32))
    df["response_emb"] = list(r_emb.astype(np.float32))
    return df


def build_features(df, metric_map):
    metric_embs = []
    for name in df["metric_name"]:
        emb = metric_map.get(name, np.zeros(768, dtype=np.float32))
        metric_embs.append(emb)
    metric_embs = np.stack(metric_embs).astype(np.float32)

    p = np.stack(df["prompt_emb"].values).astype(np.float32)
    r = np.stack(df["response_emb"].values).astype(np.float32)

    # Align metric dimension to 384
    if metric_embs.shape[1] > 384:
        metric_embs = metric_embs[:, :384]
    elif metric_embs.shape[1] < 384:
        pad = np.zeros((metric_embs.shape[0], 384 - metric_embs.shape[1]), dtype=np.float32)
        metric_embs = np.hstack([metric_embs, pad])

    m = metric_embs
    feats = np.hstack([p, r, m, np.abs(p - r), p * r, np.abs(m - r), m * r, np.abs(m - p), m * p])
    print(f"✅ Feature shape: {feats.shape}")
    return feats

# -------------------------
# Dataset
# -------------------------
class OrdinalDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        else:
            return self.X[idx], self.y[idx]

# -------------------------
# Multi-Task Model
# -------------------------
class MultiTaskTriRegressor(nn.Module):
    def __init__(self, in_dim=3456, hidden=1024, dropout=0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            ResidualBlock(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(dropout)
        )
        # Ordinal head: 9 thresholds (sigmoid)
        self.ordinal_head = nn.Linear(hidden // 2, 9)
        # Regression head: single continuous output
        self.reg_head = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.shared(x)
        ord_logits = self.ordinal_head(h)
        ord_probs = torch.sigmoid(ord_logits)         # (B,9)
        reg_out = self.reg_head(h).squeeze(-1)       # (B,)
        return ord_probs, reg_out


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return self.norm(x + out)

# -------------------------
# Train / Eval
# -------------------------
def train_epoch(model, loader, optimizer, lambda_reg):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X = X.to(CFG.device)
        y = y.to(CFG.device)

        optimizer.zero_grad()
        ord_probs, reg_out = model(X)

        # ordinal target binary matrix (B,9)
        y_int = y.long()
        y_ordinal = (torch.arange(1, 10).unsqueeze(0).to(CFG.device) <= (y_int.unsqueeze(1) - 1)).float()

        loss_ord = F.binary_cross_entropy(ord_probs, y_ordinal)
        loss_reg = F.mse_loss(reg_out, y)
        loss = loss_ord + lambda_reg * loss_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(X)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_true = [], []
    for X, y in loader:
        X = X.to(CFG.device)
        y = y.to(CFG.device)
        ord_probs, reg_out = model(X)
        expected_from_ord = 1.0 + ord_probs.sum(dim=1)      # (B,)
        # ensemble: average ordinal-expectation and regression head
        final_pred = 0.5 * expected_from_ord + 0.5 * reg_out
        all_preds.append(final_pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)
    return rmse(true, preds)


def create_augmented_dataframe(df_train, metric_names, metric_embs, rng):
    N = len(df_train)
    K = N // 2   # equal samples per class (1-3) and (4-7)

    p = np.stack(df_train["prompt_emb"].values).astype(np.float32)
    r = np.stack(df_train["response_emb"].values).astype(np.float32)

    name_to_idx = {name: i for i, name in enumerate(metric_names)}
    metric_idx = np.array([name_to_idx.get(n, 0) for n in df_train["metric_name"].values], dtype=int)

    real_df = df_train.copy().reset_index(drop=True)

    # ---------------------------------------------------------
    # Helper: slice first K rows for augmentation
    # ---------------------------------------------------------
    def slice_df(df, K):
        return {
            "system_prompt": df["system_prompt"].values[:K],
            "user_prompt": df["user_prompt"].values[:K],
            "response": df["response"].values[:K],
            "metric_name": df["metric_name"].values[:K]
        }

    base = slice_df(real_df, K)
    pK = p[:K]
    rK = r[:K]

    # -------------------------------
    # (A) Low-score augmentations 1–3
    # -------------------------------
    # 1) shuffle
    perm1_low = rng.permutation(N)[:K]
    neg1_low = pd.DataFrame({
        **base,
        "prompt_emb": list(p[perm1_low]),
        "response_emb": list(rK),
        "score": rng.integers(1, 4, size=K).astype(np.float32)
    })

    # 2) noise
    noise_low = rng.normal(scale=CFG.aug_noise_scale, size=(K, r.shape[1])).astype(np.float32)
    neg2_low = pd.DataFrame({
        **base,
        "prompt_emb": list(pK),
        "response_emb": list((rK + noise_low).astype(np.float32)),
        "score": rng.integers(1, 4, size=K).astype(np.float32)
    })

    # 3) metric swap
    perm2_low = rng.permutation(N)[:K]
    swapped_low = [metric_names[i] for i in metric_idx[perm2_low]]
    neg3_low = pd.DataFrame({
        **base,
        "prompt_emb": list(pK),
        "response_emb": list(rK),
        "metric_name": swapped_low,
        "score": rng.integers(1, 4, size=K).astype(np.float32)
    })

    # -------------------------------
    # (B) Mid-score augmentations 4–7
    # -------------------------------
    # 1) shuffle
    perm1_mid = rng.permutation(N)[:K]
    neg1_mid = pd.DataFrame({
        **base,
        "prompt_emb": list(p[perm1_mid]),
        "response_emb": list(rK),
        "score": rng.integers(4, 8, size=K).astype(np.float32)
    })

    # 2) noise
    noise_mid = rng.normal(scale=CFG.aug_noise_scale, size=(K, r.shape[1])).astype(np.float32)
    neg2_mid = pd.DataFrame({
        **base,
        "prompt_emb": list(pK),
        "response_emb": list((rK + noise_mid).astype(np.float32)),
        "score": rng.integers(4, 8, size=K).astype(np.float32)
    })

    # 3) metric swap
    perm2_mid = rng.permutation(N)[:K]
    swapped_mid = [metric_names[i] for i in metric_idx[perm2_mid]]
    neg3_mid = pd.DataFrame({
        **base,
        "prompt_emb": list(pK),
        "response_emb": list(rK),
        "metric_name": swapped_mid,
        "score": rng.integers(4, 8, size=K).astype(np.float32)
    })

    aug_df = pd.concat(
        [real_df, neg1_low, neg2_low, neg3_low, neg1_mid, neg2_mid, neg3_mid],
        ignore_index=True
    )

    print("Augmented counts:",
          "real =", len(real_df),
          "| low =", len(neg1_low), len(neg2_low), len(neg3_low),
          "| mid =", len(neg1_mid), len(neg2_mid), len(neg3_mid),
          "-> total =", len(aug_df))

    return aug_df

# -------------------------
# Main
# -------------------------
def main():
    print("🔧 Loading data and metric maps...")
    metric_map, metric_names, metric_embs = load_metric_map(CFG.metric_names, CFG.metric_emb_npy)

    df_train = clean_dataframe(load_json_data(CFG.train_json), is_train=True)
    df_test = clean_dataframe(load_json_data(CFG.test_json), is_train=False)

    df_train = compute_embeddings(df_train, CFG.model_name)
    df_test = compute_embeddings(df_test, CFG.model_name)

    rng = np.random.default_rng(CFG.seed)
    df_train_aug = create_augmented_dataframe(df_train, metric_names, metric_embs, rng)

    X_aug = build_features(df_train_aug, metric_map)
    y_aug = df_train_aug["score"].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X_aug, y_aug, test_size=CFG.val_size, random_state=CFG.seed)

    train_ds = OrdinalDataset(X_train, y_train)
    val_ds = OrdinalDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)

    model = MultiTaskTriRegressor(in_dim=X_aug.shape[1], hidden=CFG.hidden, dropout=CFG.dropout).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    best_rmse = float("inf")
    for epoch in range(1, CFG.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, CFG.lambda_reg)
        val_rmse = evaluate(model, val_loader)
        print(f"Epoch {epoch:02d} | Train Loss {tr_loss:.4f} | Val RMSE {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), CFG.save_path)
            print(f"✅ Saved best model (RMSE={val_rmse:.4f})")

    # -------------------------
    # Inference on test
    # -------------------------
    model.load_state_dict(torch.load(CFG.save_path, map_location=CFG.device))
    model.eval()
    X_test = build_features(df_test, metric_map)
    test_loader = DataLoader(torch.tensor(X_test, dtype=torch.float32), batch_size=CFG.batch_size)

    preds = []
    with torch.no_grad():
        for Xb in test_loader:
            Xb = Xb.to(CFG.device)
            ord_probs, reg_out = model(Xb)
            expected_from_ord = 1.0 + ord_probs.sum(dim=1)
            final_pred = 0.5 * expected_from_ord + 0.5 * reg_out
            preds.extend(final_pred.cpu().numpy())

    preds = np.clip(preds, 1.0, 10.0)
    sub = pd.DataFrame({"id": np.arange(1, len(preds) + 1), "score": preds})
    sub.to_csv(CFG.submission_path, index=False)
    print(f"✅ Saved submission to {CFG.submission_path}")
    print(sub.head())

if __name__ == "__main__":
    main()
