"""
Combined pipeline:
 - trains AttentionTriEncoder (from script A) -> saves best_attention_tri.pth
 - trains HybridRegressor (from script B)   -> saves best_hybrid_final.pth
 - learns a 2-way softmax combiner on validation predictions to produce convex combination
 - writes submission_combined.csv
Notes:
 - Minimal edits only: config renames and a small build_features() helper added.
 - Assumes GPU available if torch.cuda.is_available().
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# -------------------------
# Utility functions (shared)
# -------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

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

def load_metric_map(names_file, emb_file):
    with open(names_file, "r", encoding="utf-8") as f:
        metric_names = json.load(f)
    metric_embs = np.load(emb_file).astype(np.float32)
    if len(metric_names) != len(metric_embs):
        raise ValueError("Metric names and embeddings length mismatch")
    metric_map = {name: metric_embs[i] for i, name in enumerate(metric_names)}
    return metric_map, metric_names, metric_embs

def compute_embeddings(df, model_name, device, batch_size=64):
    print("🔹 Computing embeddings...")
    encoder = SentenceTransformer(model_name, device=device)
    prompts = (df["system_prompt"].astype(str) + " [SEP] " + df["user_prompt"].astype(str)).tolist()
    responses = df["response"].astype(str).tolist()
    p_emb = encoder.encode(prompts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    r_emb = encoder.encode(responses, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    df["prompt_emb"] = list(p_emb.astype(np.float32))
    df["response_emb"] = list(r_emb.astype(np.float32))
    return df

# -------------------------
# Small helper: build_features for Hybrid (minimal, plausible)
# -------------------------
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
# Attention script (adapted names)
# -------------------------
class CFG_ATT:
    seed = 42
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    train_json = "train_data.json"
    test_json = "test_data.json"
    metric_names = "metric_names.json"
    metric_emb_npy = "metric_name_embeddings.npy"
    submission_path = "submission_attention.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_size = 0.25
    batch_size = 64
    lr = 2e-4
    epochs = 10
    proj_dim = 256
    trans_nhead = 8
    trans_layers = 3
    trans_ff = 512
    dropout = 0.2
    aug_noise_scale = 0.6

torch.manual_seed(CFG_ATT.seed)
np.random.seed(CFG_ATT.seed)

def create_augmented_dataframe_att(df_train, metric_names, metric_embs, rng):
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
        "score": rng.integers(0, 4, size=K).astype(np.float32)
    })

    # 2) noise
    noise_low = rng.normal(scale=CFG.aug_noise_scale, size=(K, r.shape[1])).astype(np.float32)
    neg2_low = pd.DataFrame({
        **base,
        "prompt_emb": list(pK),
        "response_emb": list((rK + noise_low).astype(np.float32)),
        "score": rng.integers(0, 4, size=K).astype(np.float32)
    })

    # 3) metric swap
    perm2_low = rng.permutation(N)[:K]
    swapped_low = [metric_names[i] for i in metric_idx[perm2_low]]
    neg3_low = pd.DataFrame({
        **base,
        "prompt_emb": list(pK),
        "response_emb": list(rK),
        "metric_name": swapped_low,
        "score": rng.integers(0, 4, size=K).astype(np.float32)
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

# Dataset & Attention model (kept same)
class TriModalOrdinalDataset(Dataset):
    def __init__(self, prompt_embs, response_embs, metric_embs, scores=None):
        self.p = prompt_embs.astype(np.float32)
        self.r = response_embs.astype(np.float32)
        self.m = metric_embs.astype(np.float32)
        if scores is not None:
            self.y = scores.astype(np.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.p)

    def __getitem__(self, idx):
        if self.y is None:
            return self.p[idx], self.r[idx], self.m[idx]
        else:
            return self.p[idx], self.r[idx], self.m[idx], self.y[idx]

class AttentionTriEncoder(nn.Module):
    def __init__(self, p_dim, r_dim, m_dim, proj_dim=256, nhead=8, nlayers=3,
                 dim_feedforward=512, dropout=0.2, hidden_head=256):
        super().__init__()
        self.p_proj = nn.Linear(p_dim, proj_dim)
        self.r_proj = nn.Linear(r_dim, proj_dim)
        self.m_proj = nn.Linear(m_dim, proj_dim)
        self.token_embed = nn.Parameter(torch.randn(3, proj_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.head = nn.Sequential(
            nn.Linear(proj_dim, hidden_head),
            nn.GELU(),
            nn.LayerNorm(hidden_head),
            nn.Dropout(dropout),
            nn.Linear(hidden_head, hidden_head // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_head // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_head // 2, 9)
        )

    def forward(self, p, r, m):
        B = p.shape[0]
        p_t = self.p_proj(p)
        r_t = self.r_proj(r)
        m_t = self.m_proj(m)
        tokens = torch.stack([p_t, r_t, m_t], dim=0)
        tokens = tokens + self.token_embed.unsqueeze(1)
        out = self.transformer(tokens)
        pooled = out.mean(dim=0)
        logits = self.head(pooled)
        probs = torch.sigmoid(logits)
        return probs

def ordinal_loss_from_probs(probs, y, device):
    y_int = y.long()
    ordinal = (torch.arange(1, 10).unsqueeze(0).to(device) <= (y_int.unsqueeze(1) - 1)).float()
    loss = F.binary_cross_entropy(probs, ordinal)
    return loss

def train_epoch_attention(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        p, r, m, y = batch
        p = p.to(CFG_ATT.device); r = r.to(CFG_ATT.device); m = m.to(CFG_ATT.device); y = y.to(CFG_ATT.device)
        optimizer.zero_grad()
        probs = model(p, r, m)
        loss = ordinal_loss_from_probs(probs, y, CFG_ATT.device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * p.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_attention(model, loader):
    model.eval()
    all_preds = []
    all_true = []
    for batch in loader:
        p, r, m, y = batch
        p = p.to(CFG_ATT.device); r = r.to(CFG_ATT.device); m = m.to(CFG_ATT.device)
        probs = model(p, r, m)
        expected = 1 + probs.sum(dim=1)
        all_preds.append(expected.cpu().numpy())
        all_true.append(y.numpy())
    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)
    return rmse(true, preds)

# -------------------------
# Hybrid script (adapted names)
# -------------------------
class CFG_HYB:
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
    lambda_ordinal = 1.0
    lambda_reg = 1.0
    lambda_dist = 0.6
    dist_round_method = "round"
    torch_seed = True

torch.manual_seed(CFG_HYB.seed)
np.random.seed(CFG_HYB.seed)
if CFG_HYB.torch_seed and torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG_HYB.seed)

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

        self.ordinal_head = nn.Linear(hidden, 9)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
        self.dist_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 10)
        )
        self.fusion_logits = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))

    def forward(self, x):
        x = self.input_norm(x)
        x = self.proj(x)
        x = self.act(x)
        for b in self.blocks:
            x = b(x)
        ord_logits = self.ordinal_head(x)
        reg_val = self.reg_head(x).squeeze(-1)
        dist_logits = self.dist_head(x)
        fusion_weights = torch.softmax(self.fusion_logits, dim=0)
        return {
            "ordinal_logits": ord_logits,
            "regression": reg_val,
            "dist_logits": dist_logits,
            "fusion_weights": fusion_weights
        }

def ordinal_target_matrix(y_int, device):
    th = torch.arange(1, 10, device=device).unsqueeze(0)
    y_expand = y_int.unsqueeze(1)
    target = (th <= (y_expand - 1)).float()
    return target

def compute_combined_loss(out, y_true, device):
    ord_logits = out["ordinal_logits"]
    y_int = y_true.round().clamp(1, 10).long()
    ord_targets = ordinal_target_matrix(y_int, device)
    bce_loss = F.binary_cross_entropy_with_logits(ord_logits, ord_targets)
    reg_pred = out["regression"]
    mse_loss = F.mse_loss(reg_pred, y_true)
    dist_logits = out["dist_logits"]
    if CFG_HYB.dist_round_method == "round":
        dist_labels = y_true.round().clamp(1,10).long() - 1
    else:
        dist_labels = y_true.floor().clamp(1,10).long() - 1
    ce_loss = F.cross_entropy(dist_logits, dist_labels)
    total = (CFG_HYB.lambda_ordinal * bce_loss) + (CFG_HYB.lambda_reg * mse_loss) + (CFG_HYB.lambda_dist * ce_loss)
    return total, {"bce": bce_loss.item(), "mse": mse_loss.item(), "ce": ce_loss.item()}

def combine_heads_to_score(out):
    ord_logits = out["ordinal_logits"]
    ord_probs = torch.sigmoid(ord_logits)
    score_ord = 1.0 + ord_probs.sum(dim=1)
    score_reg = out["regression"]
    dist_logits = out["dist_logits"]
    dist_probs = F.softmax(dist_logits, dim=1)
    classes = torch.arange(1, 11, device=dist_probs.device, dtype=torch.float32).unsqueeze(0)
    score_dist = (dist_probs * classes).sum(dim=1)
    weights = out["fusion_weights"]
    combined = weights[0] * score_ord + weights[1] * score_reg + weights[2] * score_dist
    return combined

def train_one_epoch_hybrid(model, loader, optimizer, scaler_amp, device):
    model.train()
    total_loss = 0.0
    stats = {"bce":0.0, "mse":0.0, "ce":0.0}
    for X, y in loader:
        X = X.to(device)
        y = y.to(device).squeeze(1)
        optimizer.zero_grad()
        if CFG_HYB.use_amp and scaler_amp is not None:
            with torch.cuda.amp.autocast():
                out = model(X)
                loss, comp = compute_combined_loss(out, y, device)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG_HYB.grad_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            out = model(X)
            loss, comp = compute_combined_loss(out, y, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG_HYB.grad_clip)
            optimizer.step()
        total_loss += loss.item() * X.size(0)
        stats["bce"] += comp["bce"] * X.size(0)
        stats["mse"] += comp["mse"] * X.size(0)
        stats["ce"]  += comp["ce"]  * X.size(0)
    n = len(loader.dataset)
    return total_loss / n, {"bce": stats["bce"]/n, "mse": stats["mse"]/n, "ce": stats["ce"]/n}

@torch.no_grad()
def evaluate_model_hybrid(model, loader, device):
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
# Combiner module (learnable 2-way softmax)
# -------------------------
class Combiner2(nn.Module):
    def __init__(self, init_logits=None):
        super().__init__()
        if init_logits is None:
            init_logits = torch.tensor([1.0, 1.0], dtype=torch.float32)
        self.logits = nn.Parameter(init_logits)
    def forward(self, yA, yB):
        # yA, yB: tensors (N,) or (N,1)
        w = torch.softmax(self.logits, dim=0)
        return w[0] * yA + w[1] * yB

# -------------------------
# Full pipeline: train both models and combiner
# -------------------------
def main():
    device = CFG_ATT.device
    print("🔧 Loading data and metric maps...")
    metric_map, metric_names, metric_embs = load_metric_map(CFG_ATT.metric_names, CFG_ATT.metric_emb_npy)

    # Load & clean
    df_train = clean_dataframe(load_json_data(CFG_ATT.train_json), is_train=True)
    df_test = clean_dataframe(load_json_data(CFG_ATT.test_json), is_train=False)

    # Compute embeddings (shared)
    df_train = compute_embeddings(df_train, CFG_ATT.model_name, CFG_ATT.device, batch_size=CFG_ATT.batch_size)
    df_test = compute_embeddings(df_test, CFG_ATT.model_name, CFG_ATT.device, batch_size=CFG_ATT.batch_size)

    rng = np.random.default_rng(CFG_ATT.seed)
    # Use attention-style augmentation to get augmented train for both models
    df_train_aug = create_augmented_dataframe_att(df_train, metric_names, metric_embs, rng)

    # -------------------------
    # Prepare arrays for Attention model
    # -------------------------
    p_arr = np.stack(df_train_aug["prompt_emb"].values).astype(np.float32)
    r_arr = np.stack(df_train_aug["response_emb"].values).astype(np.float32)
    metric_arr = []
    for nm in df_train_aug["metric_name"].values:
        emb = metric_map.get(nm)
        if emb is None:
            emb = np.zeros_like(metric_embs[0], dtype=np.float32)
        metric_arr.append(emb.astype(np.float32))
    metric_arr = np.stack(metric_arr)
    y_arr = df_train_aug["score"].values.astype(np.float32)

    print("Attention training arrays shapes (p, r, m, y):", p_arr.shape, r_arr.shape, metric_arr.shape, y_arr.shape)

    # train/val split for attention model (we'll reuse these val sets as combiner data later)
    Xp_tr, Xp_val, Xr_tr, Xr_val, Xm_tr, Xm_val, y_tr_att, y_val_att = train_test_split(
        p_arr, r_arr, metric_arr, y_arr, test_size=CFG_ATT.val_size, random_state=CFG_ATT.seed
    )

    train_ds_att = TriModalOrdinalDataset(Xp_tr, Xr_tr, Xm_tr, y_tr_att)
    val_ds_att = TriModalOrdinalDataset(Xp_val, Xr_val, Xm_val, y_val_att)
    train_loader_att = DataLoader(train_ds_att, batch_size=CFG_ATT.batch_size, shuffle=True, drop_last=False)
    val_loader_att = DataLoader(val_ds_att, batch_size=CFG_ATT.batch_size, shuffle=False)

    # Attention model instantiate & train
    p_dim = p_arr.shape[1]
    r_dim = r_arr.shape[1]
    m_dim = metric_arr.shape[1]
    model_att = AttentionTriEncoder(p_dim, r_dim, m_dim,
                                proj_dim=CFG_ATT.proj_dim,
                                nhead=CFG_ATT.trans_nhead,
                                nlayers=CFG_ATT.trans_layers,
                                dim_feedforward=CFG_ATT.trans_ff,
                                dropout=CFG_ATT.dropout).to(device)
    optimizer_att = torch.optim.AdamW(model_att.parameters(), lr=CFG_ATT.lr)

    best_rmse_att = float("inf")
    for epoch in range(1, CFG_ATT.epochs + 1):
        tr_loss = train_epoch_attention(model_att, train_loader_att, optimizer_att)
        val_rmse = evaluate_attention(model_att, val_loader_att)
        print(f"[ATT] Epoch {epoch:02d} | Train Loss {tr_loss:.4f} | Val RMSE {val_rmse:.4f}")
        if val_rmse < best_rmse_att:
            best_rmse_att = val_rmse
            torch.save(model_att.state_dict(), "best_attention_tri.pth")
            print(f"✅ Saved best attention model (RMSE={val_rmse:.4f})")

    # -------------------------
    # Prepare features for Hybrid model (use build_features)
    # -------------------------
    X_aug = build_features(df_train_aug, metric_map)
    y_aug = df_train_aug["score"].values.astype(np.float32)

    scaler = StandardScaler()
    X_aug_scaled = scaler.fit_transform(X_aug)
    np.save(CFG_HYB.scaler_path, {"mean": scaler.mean_, "scale": scaler.scale_})

    X_train_h, X_val_h, y_train_h, y_val_h = train_test_split(X_aug_scaled, y_aug, test_size=CFG_HYB.val_size, random_state=CFG_HYB.seed)

    train_ds_h = RegDataset(X_train_h, y_train_h)
    val_ds_h = RegDataset(X_val_h, y_val_h)
    train_loader_h = DataLoader(train_ds_h, batch_size=CFG_HYB.batch_size, shuffle=True, drop_last=False)
    val_loader_h = DataLoader(val_ds_h, batch_size=CFG_HYB.batch_size, shuffle=False)

    input_dim = X_aug.shape[1]
    print("Hybrid model input dim:", input_dim)

    model_h = HybridRegressor(input_dim=input_dim, hidden=CFG_HYB.hidden, num_blocks=CFG_HYB.num_blocks, dropout=CFG_HYB.dropout).to(CFG_HYB.device)
    optimizer_h = torch.optim.AdamW(model_h.parameters(), lr=CFG_HYB.lr, weight_decay=CFG_HYB.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_h, T_max=CFG_HYB.epochs, eta_min=1e-6)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(CFG_HYB.use_amp and torch.cuda.is_available()))

    best_rmse_h = float("inf")
    no_improve = 0
    for epoch in range(1, CFG_HYB.epochs + 1):
        tr_loss, tr_stats = train_one_epoch_hybrid(model_h, train_loader_h, optimizer_h, scaler_amp, CFG_HYB.device)
        val_rmse, val_preds, val_true = evaluate_model_hybrid(model_h, val_loader_h, CFG_HYB.device)
        scheduler.step()
        print(f"[HYB] Epoch {epoch:02d} | TrainLoss {tr_loss:.6f} | Val RMSE {val_rmse:.4f} | BCE {tr_stats['bce']:.6f} | MSE {tr_stats['mse']:.6f} | CE {tr_stats['ce']:.6f} | LR {scheduler.get_last_lr()[0]:.6g}")
        if val_rmse + 1e-6 < best_rmse_h:
            best_rmse_h = val_rmse
            no_improve = 0
            torch.save(model_h.state_dict(), CFG_HYB.best_ckpt)
            print(f"✅ Saved best hybrid model (RMSE={val_rmse:.4f})")
        else:
            no_improve += 1
            if no_improve >= CFG_HYB.patience:
                print(f"🔻 Early stopping hybrid (no improvement in {CFG_HYB.patience} epochs).")
                break

    # -------------------------
    # Train 2-way softmax combiner on a validation split
    # Use the attention val (Xp_val, Xr_val, Xm_val, y_val_att) and a matching hybrid val slice.
    # We'll build a combiner validation set by taking the same indices used earlier for ATT val if possible;
    # simpler: create a dedicated combiner split from the original df_train (20%).
    # -------------------------
    print("🔧 Preparing combiner training set (20% random split from augmented training data)...")
    df_comb_train, df_comb_val = train_test_split(df_train_aug, test_size=0.20, random_state=CFG_ATT.seed)
    # Build attention inputs for combiner val
    p_comb = np.stack(df_comb_val["prompt_emb"].values).astype(np.float32)
    r_comb = np.stack(df_comb_val["response_emb"].values).astype(np.float32)
    m_comb = []
    for nm in df_comb_val["metric_name"].values:
        emb = metric_map.get(nm)
        if emb is None:
            emb = np.zeros_like(metric_embs[0], dtype=np.float32)
        m_comb.append(emb)
    m_comb = np.stack(m_comb)
    y_comb = df_comb_val["score"].values.astype(np.float32)

    # Load saved best attention & hybrid models (if available)
    if os.path.exists("best_attention_tri.pth"):
        model_att.load_state_dict(torch.load("best_attention_tri.pth", map_location=device))
        model_att.to(device)
    else:
        print("⚠️ Warning: best_attention_tri.pth not found — using the last-attention model in memory.")

    if os.path.exists(CFG_HYB.best_ckpt):
        model_h.load_state_dict(torch.load(CFG_HYB.best_ckpt, map_location=CFG_HYB.device))
        model_h.to(CFG_HYB.device)
    else:
        print("⚠️ Warning: best_hybrid_final.pth not found — using the last-hybrid model in memory.")

    # Produce per-model predictions on combiner validation set
    model_att.eval()
    model_h.eval()

    with torch.no_grad():
        # attention preds (expected)
        idx = 0
        att_preds = []
        bs = 128
        while idx < len(p_comb):
            p_b = torch.tensor(p_comb[idx:idx+bs], dtype=torch.float32).to(device)
            r_b = torch.tensor(r_comb[idx:idx+bs], dtype=torch.float32).to(device)
            m_b = torch.tensor(m_comb[idx:idx+bs], dtype=torch.float32).to(device)
            probs = model_att(p_b, r_b, m_b)
            expected = 1 + probs.sum(dim=1)
            att_preds.append(expected.cpu().numpy())
            idx += bs
        att_preds = np.concatenate(att_preds).astype(np.float32)

        # hybrid preds: need to build features and scale with saved scaler
        X_comb = build_features(df_comb_val, metric_map)
        sc = np.load(CFG_HYB.scaler_path, allow_pickle=True).item()
        mean = sc["mean"]
        scale = sc["scale"]
        X_comb_scaled = (X_comb - mean) / (scale + 1e-12)

        hyb_preds = []
        idx = 0
        while idx < len(X_comb_scaled):
            Xb = torch.tensor(X_comb_scaled[idx:idx+bs], dtype=torch.float32).to(CFG_HYB.device)
            out = model_h(Xb)
            score = combine_heads_to_score(out)
            hyb_preds.append(score.cpu().numpy())
            idx += bs
        hyb_preds = np.concatenate(hyb_preds).astype(np.float32)

    # Convert to tensors and train Combiner2 on (att_preds, hyb_preds) -> y_comb
    combiner = Combiner2().to(device)
    opt_comb = torch.optim.Adam(combiner.parameters(), lr=1e-2)
    y_t = torch.tensor(y_comb, dtype=torch.float32).to(device)
    A_t = torch.tensor(att_preds, dtype=torch.float32).to(device)
    B_t = torch.tensor(hyb_preds, dtype=torch.float32).to(device)

    # Train simple MSE objective for combiner for a few epochs (small)
    n_epochs_comb = 200
    best_comb_loss = float("inf")
    best_logits = None
    for ep in range(1, n_epochs_comb + 1):
        combiner.train()
        opt_comb.zero_grad()
        pred_c = combiner(A_t, B_t)
        loss = F.mse_loss(pred_c, y_t)
        loss.backward()
        opt_comb.step()
        if loss.item() < best_comb_loss:
            best_comb_loss = loss.item()
            best_logits = combiner.logits.detach().cpu().numpy().copy()
        if ep % 50 == 0 or ep == 1:
            print(f"[Combiner] Epoch {ep:03d} | MSE {loss.item():.6f}")

    # Final learned weights
    learned_logits = torch.tensor(best_logits)
    learned_w = F.softmax(learned_logits, dim=0).numpy()
    print("✅ Learned combiner logits:", best_logits, "-> weights:", learned_w)

    # -------------------------
    # Inference on test: compute both model predictions then convex combine
    # -------------------------
    # Attention test inputs
    p_test = np.stack(df_test["prompt_emb"].values).astype(np.float32)
    r_test = np.stack(df_test["response_emb"].values).astype(np.float32)
    metric_test = []
    for nm in df_test["metric_name"].values:
        emb = metric_map.get(nm)
        if emb is None:
            emb = np.zeros_like(metric_embs[0], dtype=np.float32)
        metric_test.append(emb.astype(np.float32))
    metric_test = np.stack(metric_test)

    # attention preds on test
    att_test_preds = []
    idx = 0
    bs = 128
    model_att.eval()
    with torch.no_grad():
        while idx < len(p_test):
            p_b = torch.tensor(p_test[idx:idx+bs], dtype=torch.float32).to(device)
            r_b = torch.tensor(r_test[idx:idx+bs], dtype=torch.float32).to(device)
            m_b = torch.tensor(metric_test[idx:idx+bs], dtype=torch.float32).to(device)
            probs = model_att(p_b, r_b, m_b)
            expected = 1 + probs.sum(dim=1)
            att_test_preds.append(expected.cpu().numpy())
            idx += bs
    att_test_preds = np.concatenate(att_test_preds).astype(np.float32)

    # hybrid preds on test
    X_test_h = build_features(df_test, metric_map)
    sc = np.load(CFG_HYB.scaler_path, allow_pickle=True).item()
    mean = sc["mean"]; scale = sc["scale"]
    X_test_scaled = (X_test_h - mean) / (scale + 1e-12)
    hyb_test_preds = []
    idx = 0
    model_h.eval()
    with torch.no_grad():
        while idx < len(X_test_scaled):
            Xb = torch.tensor(X_test_scaled[idx:idx+bs], dtype=torch.float32).to(CFG_HYB.device)
            out = model_h(Xb)
            score = combine_heads_to_score(out)
            hyb_test_preds.append(score.cpu().numpy())
            idx += bs
    hyb_test_preds = np.concatenate(hyb_test_preds).astype(np.float32)

    # apply learned convex combination
    w0, w1 = learned_w[0], learned_w[1]
    final_preds = w0 * att_test_preds + w1 * hyb_test_preds
    final_preds = np.clip(final_preds, 1.0, 10.0)

    sub = pd.DataFrame({"id": np.arange(1, len(final_preds) + 1), "score": final_preds})
    sub.to_csv("submission_combined_corrected_2.csv", index=False)
    print("✅ Saved combined submission to submission_combined_corrected_2.csv")
    print(sub.head())

if __name__ == "__main__":
    main()
