"""
Tri-Encoder Ordinal Classi-Regressor
------------------------------------
Attention-based Tri-Encoder (Option C)
- Augmentation at embedding level (shuffle, noise, metric-swap)
- Uses cross-modal attention over [prompt, response, metric] tokens
- Ordinal classification via 9-sigmoid thresholds -> expected-value regression
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
    submission_path = "submission_attention.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_size = 0.25
    batch_size = 64
    lr = 2e-4
    epochs = 10
    proj_dim = 256          # projection dim for each modality token
    trans_nhead = 8
    trans_layers = 3
    trans_ff = 512
    dropout = 0.2
    aug_noise_scale = 0.6

torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)

# -------------------------
# Utilities
# -------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# -------------------------
# Data and embedding utilities
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

# -------------------------
# Augmentation (embedding level)
# -------------------------
def create_augmented_dataframe(df_train, metric_names, metric_embs, rng):
    """
    Returns augmented dataframe where prompt_emb / response_emb / metric_name may be altered.
    The df rows contain prompt_emb, response_emb, metric_name, score.
    """
    N = len(df_train)
    p = np.stack(df_train["prompt_emb"].values).astype(np.float32)
    r = np.stack(df_train["response_emb"].values).astype(np.float32)

    name_to_idx = {name: i for i, name in enumerate(metric_names)}
    metric_idx = np.array([name_to_idx.get(n, 0) for n in df_train["metric_name"].values], dtype=int)
    m_for_rows = metric_embs[metric_idx]  # (N, Dm) but we will use metric names for swapped semantics

    real_df = df_train.reset_index(drop=True)

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

    # 2) noise-corrupted negatives (add noise to responses)
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
        "score": (rng.integers(0, 3, size=N).astype(np.float32) + 1.0)
    })

    neg1 = neg1.reset_index(drop=True)
    neg2 = neg2.reset_index(drop=True)
    neg3 = neg3.reset_index(drop=True)

    aug_df = pd.concat([real_df, neg1, neg2, neg3], ignore_index=True)
    print("Augmented rows:", len(real_df), "+", len(neg1), "+", len(neg2), "+", len(neg3), "=", len(aug_df))
    return aug_df

# -------------------------
# Dataset for attention model
# -------------------------
class TriModalOrdinalDataset(Dataset):
    def __init__(self, prompt_embs, response_embs, metric_embs, scores=None):
        """
        All inputs are numpy arrays:
         - prompt_embs: (N, Dp)
         - response_embs: (N, Dr)
         - metric_embs: (N, Dm)
         - scores: (N,) or None
        """
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

# -------------------------
# Attention-based Model
# -------------------------
class AttentionTriEncoder(nn.Module):
    def __init__(self, p_dim, r_dim, m_dim, proj_dim=256, nhead=8, nlayers=3,
                 dim_feedforward=512, dropout=0.2, hidden_head=256):
        super().__init__()
        # Projections to common token dim
        self.p_proj = nn.Linear(p_dim, proj_dim)
        self.r_proj = nn.Linear(r_dim, proj_dim)
        self.m_proj = nn.Linear(m_dim, proj_dim)

        # Positional / token embeddings (learnable) — small, but helps transformer
        self.token_embed = nn.Parameter(torch.randn(3, proj_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=False)  # transformer expects (S,B,E) unless batch_first True
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # Pooling head
        self.head = nn.Sequential(
            nn.Linear(proj_dim, hidden_head),
            nn.GELU(),
            nn.LayerNorm(hidden_head),
            nn.Dropout(dropout),
            nn.Linear(hidden_head, hidden_head // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_head // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_head // 2, 9)  # 9 ordinal thresholds
        )

    def forward(self, p, r, m):
        """
        p: (B, Dp), r: (B, Dr), m: (B, Dm)
        returns: probs (B,9) in (0,1) via sigmoid
        """
        B = p.shape[0]
        # project
        p_t = self.p_proj(p)  # (B, proj_dim)
        r_t = self.r_proj(r)
        m_t = self.m_proj(m)

        # stack tokens: seq_len=3
        # transformer default expects (S, B, E) if batch_first=False
        tokens = torch.stack([p_t, r_t, m_t], dim=0)  # (3, B, E)
        # add token embeddings (broadcast over batch)
        tokens = tokens + self.token_embed.unsqueeze(1)  # (3, B, E)

        # pass through transformer
        out = self.transformer(tokens)  # (3, B, E)

        # pool: mean across tokens
        pooled = out.mean(dim=0)  # (B, E)

        logits = self.head(pooled)  # (B, 9)
        probs = torch.sigmoid(logits)
        return probs

# -------------------------
# Training & evaluation helpers
# -------------------------
def ordinal_loss_from_probs(probs, y, device):
    # y: float scores (1..10)
    # build ordinal matrix (B,9)
    y_int = y.long()
    ordinal = (torch.arange(1, 10).unsqueeze(0).to(device) <= (y_int.unsqueeze(1) - 1)).float()
    loss = F.binary_cross_entropy(probs, ordinal)
    return loss

def train_epoch_attention(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        p, r, m, y = batch  # tensors
        p = p.to(CFG.device); r = r.to(CFG.device); m = m.to(CFG.device); y = y.to(CFG.device)
        optimizer.zero_grad()
        probs = model(p, r, m)
        loss = ordinal_loss_from_probs(probs, y, CFG.device)
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
        p = p.to(CFG.device); r = r.to(CFG.device); m = m.to(CFG.device)
        probs = model(p, r, m)
        expected = 1 + probs.sum(dim=1)
        all_preds.append(expected.cpu().numpy())
        all_true.append(y.numpy())
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

    # compute prompt/response embeddings
    df_train = compute_embeddings(df_train, CFG.model_name)
    df_test = compute_embeddings(df_test, CFG.model_name)

    # Augment training dataframe (embedding-level)
    rng = np.random.default_rng(CFG.seed)
    df_train_aug = create_augmented_dataframe(df_train, metric_names, metric_embs, rng)

    # Build numpy arrays of p, r, m for training (do not flatten into handcrafted features)
    p_arr = np.stack(df_train_aug["prompt_emb"].values).astype(np.float32)
    r_arr = np.stack(df_train_aug["response_emb"].values).astype(np.float32)

    # Build metric embedding array per row by looking up metric_map
    metric_arr = []
    for nm in df_train_aug["metric_name"].values:
        emb = metric_map.get(nm)
        if emb is None:
            emb = np.zeros_like(metric_embs[0], dtype=np.float32)
        metric_arr.append(emb.astype(np.float32))
    metric_arr = np.stack(metric_arr)

    y_arr = df_train_aug["score"].values.astype(np.float32)

    print("Training arrays shapes (p, r, m, y):", p_arr.shape, r_arr.shape, metric_arr.shape, y_arr.shape)

    # Train/val split
    Xp_tr, Xp_val, Xr_tr, Xr_val, Xm_tr, Xm_val, y_tr, y_val = train_test_split(
        p_arr, r_arr, metric_arr, y_arr, test_size=CFG.val_size, random_state=CFG.seed
    )

    train_ds = TriModalOrdinalDataset(Xp_tr, Xr_tr, Xm_tr, y_tr)
    val_ds = TriModalOrdinalDataset(Xp_val, Xr_val, Xm_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)

    # Model instantiate
    p_dim = p_arr.shape[1]
    r_dim = r_arr.shape[1]
    m_dim = metric_arr.shape[1]
    model = AttentionTriEncoder(p_dim, r_dim, m_dim,
                                proj_dim=CFG.proj_dim,
                                nhead=CFG.trans_nhead,
                                nlayers=CFG.trans_layers,
                                dim_feedforward=CFG.trans_ff,
                                dropout=CFG.dropout).to(CFG.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    best_rmse = float("inf")
    for epoch in range(1, CFG.epochs + 1):
        tr_loss = train_epoch_attention(model, train_loader, optimizer)
        val_rmse = evaluate_attention(model, val_loader)
        print(f"Epoch {epoch:02d} | Train Loss {tr_loss:.4f} | Val RMSE {val_rmse:.4f}")
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), "best_attention_tri.pth")
            print(f"✅ Saved best attention model (RMSE={val_rmse:.4f})")

    # -------------------------
    # Inference on test
    # -------------------------
    # prepare test arrays
    p_test = np.stack(df_test["prompt_emb"].values).astype(np.float32)
    r_test = np.stack(df_test["response_emb"].values).astype(np.float32)
    metric_test = []
    for nm in df_test["metric_name"].values:
        emb = metric_map.get(nm)
        if emb is None:
            emb = np.zeros_like(metric_embs[0], dtype=np.float32)
        metric_test.append(emb.astype(np.float32))
    metric_test = np.stack(metric_test)

    test_ds = TriModalOrdinalDataset(p_test, r_test, metric_test, scores=None)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False)

    model.load_state_dict(torch.load("best_attention_tri.pth", map_location=CFG.device))
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            p_b, r_b, m_b = batch
            p_b = p_b.to(CFG.device); r_b = r_b.to(CFG.device); m_b = m_b.to(CFG.device)
            probs = model(p_b, r_b, m_b)
            expected = 1 + probs.sum(dim=1)
            preds.extend(expected.cpu().numpy())

    preds = np.clip(preds, 1, 10)
    sub = pd.DataFrame({"id": np.arange(1, len(preds) + 1), "score": preds})
    sub.to_csv(CFG.submission_path, index=False)
    print(f"✅ Saved submission to {CFG.submission_path}")
    print(sub.head())

if __name__ == "__main__":
    main()
