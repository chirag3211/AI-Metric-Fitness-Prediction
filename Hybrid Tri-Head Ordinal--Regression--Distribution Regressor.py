"""
Tri-Encoder: Hybrid Tri-Head Ordinal–Regression–Distribution Regressor (AMP-safe)
- Augmentation at embedding-level (shuffle prompts, noisy responses, metric-swap)
- Tri-encoder features (p, r, m, interactions) -> features
- Single model with three coordinated heads:
    * ordinal head (9 logits; use BCEWithLogits)
    * regression head (scalar)
    * distribution head (10 logits; CE)
  Combined via learned convex weights (softmax-constrained) into final score.
- Training uses combined loss: BCEWithLogits(ordinal) + MSE(reg) + CE(dist)
- AMP-safe, saves scaler & best model, produces submission CSV.
"""
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
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
    submission_path = "submission_hybrid_final.csv"
    scaler_path = "feature_scaler_hybrid.npy"
    best_ckpt = "best_hybrid_final.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_size = 0.20
    batch_size = 128
    lr = 3e-4
    epochs = 40
    weight_decay = 1e-3
    grad_clip = 1.0
    patience = 8
    aug_noise_scale = 0.6
    dropout = 0.2
    hidden = 1536
    num_blocks = 6
    use_amp = True
    # Loss weights (tuneable)
    lambda_ordinal = 1.0
    lambda_reg = 1.0
    lambda_dist = 0.6
    dist_round_method = "round"  # "round" or "floor"
    torch_seed = True

torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
if CFG.torch_seed and torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.seed)

# -------------------------
# Utilities
# -------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# -------------------------
# Data / Embedding / Features
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

    # align metric dimension to 384
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
    print(f"✅ Built features: {feats.shape}")
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
# Augmentation (embed-level)
# -------------------------
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
# Model: Shared Residual SwiGLU Backbone + 3 Heads (AMP-safe)
# -------------------------
class SwiGLU(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.proj = nn.Linear(d_in, d_hidden * 2)
    def forward(self, x):
        x = self.proj(x)
        a, b = x.chunk(2, dim=-1)
        return a * F.silu(b)

class ResidualSwiGLUBlock(nn.Module):
    def __init__(self, dim, mlp_hidden, dropout=0.0, layerscale_init=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = SwiGLU(dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, dim)
        self.dropout = nn.Dropout(dropout)
        self.layerscale = nn.Parameter(torch.ones(dim) * layerscale_init)
    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + self.layerscale * h

class HybridRegressor(nn.Module):
    def __init__(self, input_dim, hidden=1536, num_blocks=6, dropout=0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden)
        self.act = nn.GELU()
        self.blocks = nn.ModuleList([ResidualSwiGLUBlock(hidden, mlp_hidden=hidden*2, dropout=dropout) for _ in range(num_blocks)])

        # Ordinal head: 9 logits (no sigmoid here; use BCEWithLogits)
        self.ordinal_head = nn.Linear(hidden, 9)

        # Regression head: single scalar
        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )

        # Distribution head: 10 logits
        self.dist_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 10)
        )

        # fusion logits (learnable) -> softmax used during combine
        self.fusion_logits = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))

    def forward(self, x):
        x = self.input_norm(x)
        x = self.proj(x)
        x = self.act(x)
        for b in self.blocks:
            x = b(x)

        ord_logits = self.ordinal_head(x)          # (B,9) - raw logits! (no sigmoid)
        reg_val = self.reg_head(x).squeeze(-1)    # (B,)
        dist_logits = self.dist_head(x)           # (B,10)
        fusion_weights = torch.softmax(self.fusion_logits, dim=0)  # (3,)
        return {
            "ordinal_logits": ord_logits,
            "regression": reg_val,
            "dist_logits": dist_logits,
            "fusion_weights": fusion_weights
        }

# -------------------------
# Training / Evaluation (AMP-safe)
# -------------------------
def ordinal_target_matrix(y_int, device):
    # y_int expected in 1..10 (LongTensor)
    th = torch.arange(1, 10, device=device).unsqueeze(0)  # (1,9)
    y_expand = y_int.unsqueeze(1)                        # (B,1)
    target = (th <= (y_expand - 1)).float()              # (B,9)
    return target

def compute_combined_loss(out, y_true, device):
    """
    out: dict from model.forward
    y_true: (B,) float scores in [1,10]
    returns scalar loss (torch)
    """
    # ORDINAL BCE WITH LOGITS
    ord_logits = out["ordinal_logits"]   # (B,9)
    y_int = y_true.round().clamp(1, 10).long()  # integer 1..10
    ord_targets = ordinal_target_matrix(y_int, device)   # (B,9)
    bce_loss = F.binary_cross_entropy_with_logits(ord_logits, ord_targets)

    # REGRESSION MSE (direct head)
    reg_pred = out["regression"]  # (B,)
    mse_loss = F.mse_loss(reg_pred, y_true)

    # DISTRIBUTION CE
    dist_logits = out["dist_logits"]  # (B,10)
    if CFG.dist_round_method == "round":
        dist_labels = y_true.round().clamp(1,10).long() - 1
    else:
        dist_labels = y_true.floor().clamp(1,10).long() - 1
    ce_loss = F.cross_entropy(dist_logits, dist_labels)

    total = (CFG.lambda_ordinal * bce_loss) + (CFG.lambda_reg * mse_loss) + (CFG.lambda_dist * ce_loss)
    return total, {"bce": bce_loss.item(), "mse": mse_loss.item(), "ce": ce_loss.item()}

def combine_heads_to_score(out):
    """
    Combine predictions using learned convex weights:
      score_ord = 1 + sum(sigmoid(ord_logits))
      score_reg = reg_pred (already scalar)
      score_dist = expectation over softmax(dist_logits)
    returns tensor (B,) of combined scores
    """
    ord_logits = out["ordinal_logits"]         # (B,9)
    ord_probs = torch.sigmoid(ord_logits)
    score_ord = 1.0 + ord_probs.sum(dim=1)     # (B,)

    score_reg = out["regression"]              # (B,)

    dist_logits = out["dist_logits"]           # (B,10)
    dist_probs = F.softmax(dist_logits, dim=1)
    classes = torch.arange(1, 11, device=dist_probs.device, dtype=torch.float32).unsqueeze(0)
    score_dist = (dist_probs * classes).sum(dim=1)  # (B,)

    weights = out["fusion_weights"]            # (3,)
    combined = weights[0] * score_ord + weights[1] * score_reg + weights[2] * score_dist
    return combined

def train_one_epoch(model, loader, optimizer, scaler_amp, device):
    model.train()
    total_loss = 0.0
    stats = {"bce":0.0, "mse":0.0, "ce":0.0}
    for X, y in loader:
        X = X.to(device)
        y = y.to(device).squeeze(1)
        optimizer.zero_grad()

        if CFG.use_amp and scaler_amp is not None:
            with torch.cuda.amp.autocast():
                out = model(X)
                loss, comp = compute_combined_loss(out, y, device)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            out = model(X)
            loss, comp = compute_combined_loss(out, y, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()

        total_loss += loss.item() * X.size(0)
        stats["bce"] += comp["bce"] * X.size(0)
        stats["mse"] += comp["mse"] * X.size(0)
        stats["ce"]  += comp["ce"]  * X.size(0)

    n = len(loader.dataset)
    return total_loss / n, {"bce": stats["bce"]/n, "mse": stats["mse"]/n, "ce": stats["ce"]/n}

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device).squeeze(1)
        out = model(X)
        preds = combine_heads_to_score(out)
        all_preds.append(preds.cpu().numpy())
        all_true.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)
    return rmse(true, preds), preds, true

# -------------------------
# Main
# -------------------------
def main():
    device = CFG.device
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

    # standardize features
    scaler = StandardScaler()
    X_aug = scaler.fit_transform(X_aug)
    np.save(CFG.scaler_path, {"mean": scaler.mean_, "scale": scaler.scale_})

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_aug, y_aug, test_size=CFG.val_size, random_state=CFG.seed)

    train_ds = RegDataset(X_train, y_train)
    val_ds = RegDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)

    input_dim = X_aug.shape[1]
    print("Model input dim:", input_dim)

    model = HybridRegressor(input_dim=input_dim, hidden=CFG.hidden, num_blocks=CFG.num_blocks, dropout=CFG.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6)

    scaler_amp = torch.cuda.amp.GradScaler(enabled=(CFG.use_amp and torch.cuda.is_available()))

    best_rmse = float("inf")
    no_improve = 0

    for epoch in range(1, CFG.epochs + 1):
        tr_loss, tr_stats = train_one_epoch(model, train_loader, optimizer, scaler_amp, device)
        val_rmse, val_preds, val_true = evaluate_model(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:02d} | TrainLoss {tr_loss:.6f} | Val RMSE {val_rmse:.4f} | BCE {tr_stats['bce']:.6f} | MSE {tr_stats['mse']:.6f} | CE {tr_stats['ce']:.6f} | LR {scheduler.get_last_lr()[0]:.6g}")

        if val_rmse + 1e-6 < best_rmse:
            best_rmse = val_rmse
            no_improve = 0
            torch.save(model.state_dict(), CFG.best_ckpt)
            print(f"✅ Saved best model (RMSE={val_rmse:.4f})")
        else:
            no_improve += 1
            if no_improve >= CFG.patience:
                print(f"🔻 Early stopping (no improvement in {CFG.patience} epochs).")
                break

    # -------------------------
    # Inference on test
    # -------------------------
    # load scaler
    sc = np.load(CFG.scaler_path, allow_pickle=True).item()
    mean = sc["mean"]
    scale = sc["scale"]

    model.load_state_dict(torch.load(CFG.best_ckpt, map_location=device))
    model.eval()

    X_test = build_features(df_test, metric_map)
    X_test = (X_test - mean) / (scale + 1e-12)

    test_loader = DataLoader(torch.tensor(X_test, dtype=torch.float32), batch_size=CFG.batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for Xb in test_loader:
            Xb = Xb.to(device)
            if CFG.use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    out = model(Xb)
                    score = combine_heads_to_score(out)
            else:
                out = model(Xb)
                score = combine_heads_to_score(out)
            preds.extend(score.cpu().numpy())

    preds = np.array(preds, dtype=np.float32)
    preds = np.clip(preds, 1.0, 10.0)
    sub = pd.DataFrame({"id": np.arange(1, len(preds) + 1), "score": preds})
    sub.to_csv(CFG.submission_path, index=False)
    print(f"✅ Saved submission to {CFG.submission_path}")
    print("Best validation RMSE:", best_rmse)
    print(sub.head())

if __name__ == "__main__":
    main()
