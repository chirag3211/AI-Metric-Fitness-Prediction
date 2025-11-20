"""
Tri-Encoder: Improved Deep Residual SwiGLU Regressor (merged with augmentation)
- Augmentation at embedding-level (shuffle prompts, noisy responses, metric-swap)
- Tri-encoder features (p, r, m, interactions) -> 3456 dim
- New model: Deep Residual MLP with SwiGLU + LayerScale
- Training: standardize inputs, AdamW + weight decay, CosineAnnealingLR, AMP, early stopping
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    submission_path = "submission_improved.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_size = 0.20
    batch_size = 128
    lr = 3e-4
    epochs = 30
    hidden = 1536        # base hidden width for MLP blocks
    dropout = 0.25
    aug_noise_scale = 0.6
    weight_decay = 1e-3
    grad_clip = 1.0
    patience = 6         # early stopping patience (in epochs)
    num_blocks = 6       # number of residual blocks
    scaler_path = "feature_scaler.npy"
    seed_torch = True
    use_amp = True       # automatic mixed precision

torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
if CFG.seed_torch and torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.seed)

# -------------------------
# Utils
# -------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# -------------------------
# Data Loaders / Embedding / Features (unchanged)
# -------------------------
def load_metric_map(names_file, emb_file):
    with open(names_file, "r", encoding="utf-8") as f:
        metric_names = json.load(f)
    metric_embs = np.load(emb_file).astype(np.float32)
    if len(metric_names) != len(metric_embs):
        raise ValueError("Metric names and embeddings length mismatch")
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
            raise ValueError("'score' column missing in training data")
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
    feats = np.hstack([
        p, r, m,
        np.abs(p - r), p * r,
        np.abs(m - r), m * r,
        np.abs(m - p), m * p
    ])
    return feats

# -------------------------
# Dataset
# -------------------------
class RegDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
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
# Improved Model: Residual SwiGLU MLP + LayerScale
# -------------------------
class SwiGLU(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.proj = nn.Linear(d_in, d_hidden * 2)
    def forward(self, x):
        x = self.proj(x)
        a, b = x.chunk(2, dim=-1)
        return a * F.silu(b)  # SwiGLU: a * SiLU(b)

class ResidualSwiGLUBlock(nn.Module):
    def __init__(self, dim, mlp_hidden, dropout=0.0, layerscale_init=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = SwiGLU(dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, dim)
        self.dropout = nn.Dropout(dropout)
        # LayerScale: small learned scaling applied to residual
        self.layerscale = nn.Parameter(torch.ones(dim) * layerscale_init)
    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + self.layerscale * h

class ImprovedRegressor(nn.Module):
    def __init__(self, input_dim, hidden=1536, num_blocks=6, dropout=0.2):
        super().__init__()
        # initial projection to hidden
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden)
        self.act = nn.GELU()
        self.blocks = nn.ModuleList([
            ResidualSwiGLUBlock(hidden, mlp_hidden=hidden*2, dropout=dropout) for _ in range(num_blocks)
        ])
        # aggregation head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        x = self.input_norm(x)
        x = self.proj(x)
        x = self.act(x)
        for b in self.blocks:
            x = b(x)
        out = self.head(x)
        return out.squeeze(-1)  # return (B,)

# -------------------------
# Training / Eval (with AMP + Cosine LR + EarlyStopping)
# -------------------------
def train_epoch(model, loader, optimizer, scaler_amp):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for X, y in loader:
        X = X.to(CFG.device)
        y = y.to(CFG.device).squeeze(1)
        optimizer.zero_grad()
        if CFG.use_amp and scaler_amp is not None:
            with torch.cuda.amp.autocast():
                preds = model(X)
                loss = criterion(preds, y)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_true = [], []
    for X, y in loader:
        X = X.to(CFG.device)
        y = y.to(CFG.device).squeeze(1)
        preds = model(X)
        all_preds.append(preds.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())
    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)
    return rmse(true, preds), preds, true

# -------------------------
# Augmentation (embed-level) — same as before
# -------------------------
def create_augmented_dataframe(df_train, metric_names, metric_embs, rng):
    N = len(df_train)
    p = np.stack(df_train["prompt_emb"].values).astype(np.float32)
    r = np.stack(df_train["response_emb"].values).astype(np.float32)

    name_to_idx = {name: i for i, name in enumerate(metric_names)}
    metric_idx = np.array([name_to_idx.get(n, 0) for n in df_train["metric_name"].values], dtype=int)

    real_df = df_train.copy().reset_index(drop=True)

    # 1) shuffle-based negatives (mismatch prompts)
    perm1 = rng.permutation(N)
    neg1 = pd.DataFrame({
        "system_prompt": real_df["system_prompt"].values,
        "user_prompt": real_df["user_prompt"].values,
        "response": real_df["response"].values,
        "prompt_emb": list(p[perm1]),
        "response_emb": list(r),
        "metric_name": real_df["metric_name"].values,
        "score": (rng.integers(0, 3, size=N).astype(np.float32) + 1.0)
    })

    # 2) noise-corrupted negatives
    noise = rng.normal(scale=CFG.aug_noise_scale, size=r.shape).astype(np.float32)
    neg2 = pd.DataFrame({
        "system_prompt": real_df["system_prompt"].values,
        "user_prompt": real_df["user_prompt"].values,
        "response": real_df["response"].values,
        "prompt_emb": list(p),
        "response_emb": list((r + noise).astype(np.float32)),
        "metric_name": real_df["metric_name"].values,
        "score": (rng.integers(0, 3, size=N).astype(np.float32) + 1.0)
    })

    # 3) metric-swapped negatives
    perm2 = rng.permutation(N)
    swapped_metric_names = [metric_names[i] for i in metric_idx[perm2]]
    neg3 = pd.DataFrame({
        "system_prompt": real_df["system_prompt"].values,
        "user_prompt": real_df["user_prompt"].values,
        "response": real_df["response"].values,
        "prompt_emb": list(p),
        "response_emb": list(r),
        "metric_name": swapped_metric_names,
        "score": (rng.integers(0, 4, size=N).astype(np.float32) + 1.0)
    })

    neg1 = neg1.reset_index(drop=True)
    neg2 = neg2.reset_index(drop=True)
    neg3 = neg3.reset_index(drop=True)

    aug_df = pd.concat([real_df, neg1, neg2, neg3], ignore_index=True)
    print("Augmented shapes (rows):", len(real_df), len(neg1), len(neg2), len(neg3), "-> total", len(aug_df))
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

    # Build tri-features
    X_aug = build_features(df_train_aug, metric_map)
    y_aug = df_train_aug["score"].values.astype(np.float32)

    # Standardize features (fit on augmented training only)
    scaler = StandardScaler()
    X_aug = scaler.fit_transform(X_aug)
    # save scaler for later consistency in inference
    np.save(CFG.scaler_path, {"mean": scaler.mean_, "scale": scaler.scale_})

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(X_aug, y_aug, test_size=CFG.val_size, random_state=CFG.seed)

    train_ds = RegDataset(X_train, y_train)
    val_ds = RegDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)

    # Model
    input_dim = X_aug.shape[1]
    model = ImprovedRegressor(input_dim=input_dim, hidden=CFG.hidden, num_blocks=CFG.num_blocks, dropout=CFG.dropout).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6)

    scaler_amp = torch.cuda.amp.GradScaler(enabled=(CFG.use_amp and torch.cuda.is_available()))

    best_rmse = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, CFG.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, scaler_amp)
        val_rmse, _, _ = evaluate(model, val_loader)
        scheduler.step()
        print(f"Epoch {epoch:02d} | Train MSE {tr_loss:.6f} | Val RMSE {val_rmse:.4f} | LR {scheduler.get_last_lr()[0]:.6g}")

        # checkpoint & early stopping
        if val_rmse < best_rmse - 1e-5:
            best_rmse = val_rmse
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_improved_regressor.pth")
            print(f"✅ Saved best model (RMSE={val_rmse:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CFG.patience:
                print(f"🔻 Early stopping: no improvement in {CFG.patience} epochs")
                break

    # -------------------------
    # Inference on test
    # -------------------------
    # load scaler to be robust
    sc = np.load(CFG.scaler_path, allow_pickle=True).item()
    mean = sc["mean"]
    scale = sc["scale"]

    model.load_state_dict(torch.load("best_improved_regressor.pth", map_location=CFG.device))
    model.eval()

    X_test = build_features(df_test, metric_map)
    # standardize with saved scaler
    X_test = (X_test - mean) / (scale + 1e-12)

    test_ds = DataLoader(torch.tensor(X_test, dtype=torch.float32), batch_size=CFG.batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for Xb in test_ds:
            Xb = Xb.to(CFG.device)
            if CFG.use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    out = model(Xb).cpu().numpy()
            else:
                out = model(Xb).cpu().numpy()
            preds.extend(out)

    preds = np.array(preds, dtype=np.float32).squeeze()
    preds = np.clip(preds, 1.0, 10.0)
    sub = pd.DataFrame({"id": np.arange(1, len(preds) + 1), "score": preds})
    sub.to_csv(CFG.submission_path, index=False)
    print(f"✅ Saved submission to {CFG.submission_path}")
    print("Best validation RMSE:", best_rmse)
    print(sub.head())

if __name__ == "__main__":
    main()
