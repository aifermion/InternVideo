"""Fine-tune InternVideo-Next for multi-task classification on MammalPS benchmark_1.

Predicts species, activity, and actions simultaneously using a frozen
InternVideo-Next backbone with per-task attention pooling and 2-layer
MLP classification heads.

Usage:
    python train_benchmark1_internvideo.py train   [OPTIONS]  # training + validation
    python train_benchmark1_internvideo.py test    [OPTIONS]  # test-set evaluation (single-pass)
    python train_benchmark1_internvideo.py test_ms [OPTIONS]  # test-set evaluation (multi-sample)

See --help for all flags.
"""

import argparse
import csv
import datetime
import glob
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torchvision.io import read_video
from tqdm.auto import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.InternVideo_next import InternVideo2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_FRAMES = 16
FRAME_SIZE = 224  # InternVideo-Next expects 224x224

SPECIES_CLASSES = [
    "fox", "hare", "red_deer", "roe_deer", "wolf",
]
SPECIES_TO_IDX = {name: i for i, name in enumerate(SPECIES_CLASSES)}
NUM_SPECIES = len(SPECIES_CLASSES)

ACTIVITY_CLASSES = [
    "camera_reaction", "chasing", "courtship", "escaping", "foraging",
    "grooming", "marking", "playing", "resting", "unknown", "vigilance",
]
ACTIVITY_TO_IDX = {name: i for i, name in enumerate(ACTIVITY_CLASSES)}
NUM_ACTIVITIES = len(ACTIVITY_CLASSES)

ACTION_CLASSES = [
    "bathing", "defecating", "drinking", "grazing", "jumping",
    "laying", "looking_at_camera", "running",
    "scratching_antlers", "scratching_body", "scratching_hoof",
    "shaking_fur", "sniffing", "standing_head_down", "standing_head_up",
    "unknown", "urinating", "vocalizing", "walking",
]
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_CLASSES)}
NUM_ACTIONS = len(ACTION_CLASSES)


# ===================================================================
# Per-task attention pooling head
# ===================================================================

class AttentionPooler(nn.Module):
    """Learned single-query cross-attention pooling over token features."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b = x.shape[0]
        q = self.query.expand(b, -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class TaskHead(nn.Module):
    """Per-task attention pooler + 2-layer MLP (hidden + GELU + dropout + proj)."""

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int,
                 num_classes: int, dropout_rate: float):
        super().__init__()
        self.pooler = AttentionPooler(embed_dim, num_heads)
        self.hidden = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        emb = self.pooler(features)
        h = self.act(self.hidden(emb))
        h = self.drop(h)
        return self.out_proj(h)


# ===================================================================
# Multi-task classifier module
# ===================================================================

class MultiTaskClassifier(nn.Module):
    """
    Frozen InternVideo-Next backbone with per-task attention pooling
    and 2-layer MLP classification heads.
    """

    def __init__(self, backbone: InternVideo2, embed_dim: int, num_heads: int,
                 num_species: int, num_activities: int, num_actions: int,
                 head_hidden_dim: int = 256, dropout_rate: float = 0.1):
        super().__init__()
        self.backbone = backbone

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.species_head = TaskHead(embed_dim, num_heads, head_hidden_dim,
                                     num_species, dropout_rate)
        self.activity_head = TaskHead(embed_dim, num_heads, head_hidden_dim,
                                      num_activities, dropout_rate)
        self.action_head = TaskHead(embed_dim, num_heads, head_hidden_dim,
                                    num_actions, dropout_rate)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad(), torch.amp.autocast("cuda"):
            features = self.backbone(x, projected=False)  # [B, T*H*W, D]

        features = features.float()
        sp_logits = self.species_head(features)
        act_logits = self.activity_head(features)
        action_logits = self.action_head(features)
        return sp_logits, act_logits, action_logits


# ===================================================================
# Data loading
# ===================================================================

def load_csv_samples(csv_path: str, video_dir: str):
    """Read a benchmark_1 metadata CSV and return per-clip label tuples.

    Returns a list of (video_path, species_idx, activity_idx, action_multi_hot).
    Rows with unrecognised species or activity labels are skipped with a warning.
    """
    samples = []
    skipped = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            species = row["species"].strip()
            activity = row["activity"].strip()
            if species not in SPECIES_TO_IDX or activity not in ACTIVITY_TO_IDX:
                skipped += 1
                continue

            video_path = os.path.join(video_dir, row["video_path"].strip())
            species_idx = SPECIES_TO_IDX[species]
            activity_idx = ACTIVITY_TO_IDX[activity]

            action_vec = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for a in row["actions"].strip().split(";"):
                a = a.strip()
                if a in ACTION_TO_IDX:
                    action_vec[ACTION_TO_IDX[a]] = 1.0
            samples.append((video_path, species_idx, activity_idx, action_vec))

    if skipped:
        print(f"  Warning: skipped {skipped} rows with unrecognised labels")
    return samples


def read_and_preprocess_video(
    path: str,
    target_num_frames: int = NUM_FRAMES,
    target_size: int = FRAME_SIZE,
    augment: bool = False,
) -> torch.Tensor:
    """Load an MP4 clip and return float32 [C, T, H, W] in [0, 1].

    When *augment* is True (training), applies temporal jitter, random crop,
    random horizontal flip, and brightness/contrast jitter.  When False
    (evaluation), uses uniform temporal sampling and center crop.
    """
    vframes, _, _ = read_video(path, pts_unit="sec")  # [T, H, W, C] uint8
    n = vframes.shape[0]
    if n == 0:
        raise ValueError(f"Empty video: {path}")

    indices = np.linspace(0, n, num=target_num_frames, endpoint=False, dtype=np.float64)
    if augment and n > target_num_frames:
        stride = n / target_num_frames
        jitter = np.random.uniform(-0.4 * stride, 0.4 * stride, size=target_num_frames)
        indices = np.clip(indices + jitter, 0, n - 1)
    indices = indices.astype(np.int32)

    frames = vframes[indices]  # [T, H, W, C]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]

    _, c, h, w = frames.shape

    if augment:
        scale = max(target_size / h, target_size / w) * np.random.uniform(1.0, 1.25)
    else:
        scale = max(target_size / h, target_size / w)

    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    if (new_h, new_w) != (h, w):
        frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)

    if augment:
        top = np.random.randint(0, max(new_h - target_size, 0) + 1)
        left = np.random.randint(0, max(new_w - target_size, 0) + 1)
    else:
        top = (new_h - target_size) // 2
        left = (new_w - target_size) // 2
    frames = frames[:, :, top:top + target_size, left:left + target_size]

    if augment:
        if np.random.random() < 0.5:
            frames = frames.flip(-1)
        brightness = np.random.uniform(-0.1, 0.1)
        contrast = np.random.uniform(0.9, 1.1)
        frames = (contrast * frames + brightness).clamp(0.0, 1.0)

    return frames.permute(1, 0, 2, 3)  # [C, T, H, W]


def random_sample_and_preprocess(
    path: str,
    target_num_frames: int = NUM_FRAMES,
    target_size: int = FRAME_SIZE,
) -> torch.Tensor:
    """Randomly sample *target_num_frames* from a video and preprocess.

    Returns float32 [C, T, H, W] in [0, 1].  Sampling is with replacement
    when the video has fewer frames than requested.  Uses center crop.
    """
    vframes, _, _ = read_video(path, pts_unit="sec")
    n = vframes.shape[0]
    if n == 0:
        raise ValueError(f"Empty video: {path}")

    indices = sorted(np.random.choice(n, size=target_num_frames, replace=(n < target_num_frames)))
    frames = vframes[indices].permute(0, 3, 1, 2).float() / 255.0

    _, c, h, w = frames.shape
    scale = max(target_size / h, target_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    if (new_h, new_w) != (h, w):
        frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top = (new_h - target_size) // 2
    left = (new_w - target_size) // 2
    frames = frames[:, :, top:top + target_size, left:left + target_size]
    return frames.permute(1, 0, 2, 3)


def compute_sample_weights(samples):
    """Compute per-sample weights from inverse label frequencies."""
    n = len(samples)
    species_counts = np.zeros(NUM_SPECIES, dtype=np.float64)
    activity_counts = np.zeros(NUM_ACTIVITIES, dtype=np.float64)
    action_counts = np.zeros(NUM_ACTIONS, dtype=np.float64)

    for _, sp, act, action_vec in samples:
        species_counts[sp] += 1
        activity_counts[act] += 1
        action_counts += action_vec.astype(np.float64)

    species_counts = np.maximum(species_counts, 1.0)
    activity_counts = np.maximum(activity_counts, 1.0)
    action_counts = np.maximum(action_counts, 1.0)

    inv_sp = 1.0 / species_counts
    inv_act = 1.0 / activity_counts
    inv_action = 1.0 / action_counts

    weights = np.empty(n, dtype=np.float64)
    for i, (_, sp, act, action_vec) in enumerate(samples):
        weights[i] = inv_sp[sp] + inv_act[act] + np.dot(action_vec, inv_action)

    weights /= weights.sum()
    return weights


def make_batches(
    samples,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = False,
    drop_remainder: bool = False,
    sample_weights=None,
    augment: bool = False,
):
    """Yield (videos, species, activity, actions) batches with parallel I/O.

    videos:   float32 [B, C, T, H, W]
    species:  int64   [B]
    activity: int64   [B]
    actions:  float32 [B, NUM_ACTIONS]
    """
    if sample_weights is not None:
        indices = np.random.choice(len(samples), size=len(samples), replace=True, p=sample_weights)
        items = [samples[i] for i in indices]
    else:
        items = list(samples)
        if shuffle:
            random.shuffle(items)

    def _load(item):
        path, sp, act, action_vec = item
        try:
            vid = read_and_preprocess_video(path, augment=augment)
            return vid, sp, act, action_vec
        except Exception as e:
            print(f"  [WARNING] Skipping {path}: {e}")
            return None

    buf_videos, buf_species, buf_activity, buf_actions = [], [], [], []
    chunk_size = num_workers * 2
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for start in range(0, len(items), chunk_size):
            chunk = items[start:start + chunk_size]
            for result in pool.map(_load, chunk):
                if result is None:
                    continue
                vid, sp, act, action_vec = result
                buf_videos.append(vid)
                buf_species.append(sp)
                buf_activity.append(act)
                buf_actions.append(action_vec)
                if len(buf_videos) == batch_size:
                    yield (
                        torch.stack(buf_videos),
                        torch.tensor(buf_species, dtype=torch.long),
                        torch.tensor(buf_activity, dtype=torch.long),
                        torch.from_numpy(np.stack(buf_actions)),
                    )
                    buf_videos, buf_species, buf_activity, buf_actions = [], [], [], []
    if buf_videos and not drop_remainder:
        yield (
            torch.stack(buf_videos),
            torch.tensor(buf_species, dtype=torch.long),
            torch.tensor(buf_activity, dtype=torch.long),
            torch.from_numpy(np.stack(buf_actions)),
        )


# ===================================================================
# Model builder
# ===================================================================

def build_model(model_size: str = "base", pretrained_path: str = None,
                dropout: float = 0.1, head_hidden_dim: int = 256,
                device: torch.device = None):
    """Create MultiTaskClassifier and load pretrained InternVideo-Next weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common_kwargs = dict(
        num_frames=NUM_FRAMES,
        tubelet_size=1,
        sep_pos_embed=False,
        use_checkpoint=False,
        use_flash_attn=True,
        use_fused_rmsnorm=True,
        use_fused_mlp=True,
        qkv_bias=False,
        qk_normalization=True,
        drop_path_rate=0.0,
    )

    if model_size == "base":
        backbone = InternVideo2(
            img_size=224, patch_size=14, embed_dim=768,
            depth=12, num_heads=12, mlp_ratio=4,
            attn_pool_num_heads=16, clip_embed_dim=768,
            cls_token_num=4,
            **common_kwargs,
        )
        embed_dim = 768
        num_heads = 12
        hf_repo = "revliter/internvideo_next_base_p14_res224_f16"
        hf_filename = "model.safetensors"
        ckpt_num_frames = NUM_FRAMES
    elif model_size == "large":
        backbone = InternVideo2(
            img_size=224, patch_size=14, embed_dim=1024,
            depth=24, num_heads=16, mlp_ratio=4,
            attn_pool_num_heads=16, clip_embed_dim=768,
            cls_token_num=4,
            **common_kwargs,
        )
        embed_dim = 1024
        num_heads = 16
        hf_repo = "revliter/internvideo_next_large_p14_res224_f16"
        hf_filename = "model.safetensors"
        ckpt_num_frames = NUM_FRAMES
    else:
        raise ValueError(f"Unknown model_size: {model_size}")

    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"Loading pretrained weights from local: {pretrained_path}")
        if pretrained_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(pretrained_path)
        else:
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    else:
        print(f"Loading pretrained weights from HuggingFace: {hf_repo}/{hf_filename}")
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
        if hf_filename.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(ckpt_path)
        else:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "model" in state:
        state = state["model"]

    # Strip "model." prefix (HF InternVideo-Next checkpoint format)
    if any(k.startswith("model.") for k in state):
        state = {k[len("model."):] if k.startswith("model.") else k: v
                 for k, v in state.items()}

    # Rename lr_scale -> gamma for LayerScale compatibility
    state = {k.replace(".lr_scale", ".gamma"): v for k, v in state.items()}

    if NUM_FRAMES != ckpt_num_frames and "pos_embed" in state:
        old_pos = state["pos_embed"]
        cls_token_num = backbone.cls_token_num
        spatial_size = (224 // 14) ** 2
        cls_pos = old_pos[:, :cls_token_num, :]
        patch_pos = old_pos[:, cls_token_num:, :]
        T_old = patch_pos.shape[1] // spatial_size
        patch_pos = patch_pos.reshape(1, T_old, spatial_size, embed_dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(NUM_FRAMES, spatial_size),
            mode="bilinear", align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        state["pos_embed"] = torch.cat([cls_pos, patch_pos], dim=1)
        print(f"  Interpolated pos_embed: {old_pos.shape} -> {state['pos_embed'].shape} "
              f"(frames {T_old} -> {NUM_FRAMES})")

    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    model = MultiTaskClassifier(
        backbone=backbone,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_species=NUM_SPECIES,
        num_activities=NUM_ACTIVITIES,
        num_actions=NUM_ACTIONS,
        head_hidden_dim=head_hidden_dim,
        dropout_rate=dropout,
    )

    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total:,} total, {trainable:,} trainable")

    return model, embed_dim


def build_optimizer(model, learning_rate: float = 1e-5,
                    min_learning_rate: float = 1e-7,
                    total_steps: int = 1000,
                    weight_decay: float = 0.01):
    """AdamW optimizer with cosine LR schedule over only trainable (head) params."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=min_learning_rate,
    )
    print(
        f"Optimizer ready \u2014 AdamW (lr={learning_rate}\u2192{min_learning_rate} cosine, "
        f"wd={weight_decay}, {total_steps} steps) | encoder: frozen"
    )
    return optimizer, scheduler


# ===================================================================
# Metrics
# ===================================================================

def average_precision_at_k(y_true, y_scores, k):
    n_relevant = int(y_true.sum())
    if n_relevant == 0:
        return 0.0
    sorted_indices = np.argsort(-y_scores)[:k]
    y_true_sorted = y_true[sorted_indices].astype(np.float64)
    cumsum = np.cumsum(y_true_sorted)
    precisions = cumsum / np.arange(1, len(y_true_sorted) + 1)
    return float(np.sum(precisions * y_true_sorted) / min(k, n_relevant))


def compute_map_metrics(logits, labels, class_names, multi_label=False,
                        from_probs=False, pred_threshold=None):
    """Compute mAP, mAP@1, mAP@5, Macro F1 per class and overall."""
    num_classes = len(class_names)

    if multi_label:
        probs = logits.astype(np.float64) if from_probs else 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        binary_labels = labels.astype(np.float64)
        threshold = pred_threshold if pred_threshold is not None else 0.5
        binary_preds = (probs >= threshold).astype(np.float64)
    else:
        if from_probs:
            probs = logits.astype(np.float64)
        else:
            logits_f = logits.astype(np.float64)
            exp_l = np.exp(logits_f - logits_f.max(axis=-1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
        binary_labels = np.zeros((len(labels), num_classes), dtype=np.float64)
        for i, lbl in enumerate(labels):
            binary_labels[i, lbl] = 1.0
        if pred_threshold is not None:
            binary_preds = (probs >= pred_threshold).astype(np.float64)
        else:
            hard_preds = np.argmax(probs, axis=-1)
            binary_preds = np.zeros_like(binary_labels)
            for i, pred in enumerate(hard_preds):
                binary_preds[i, pred] = 1.0

    per_class = {}
    ap_list, ap1_list, ap5_list, f1_list = [], [], [], []

    for c, name in enumerate(class_names):
        y_true = binary_labels[:, c]
        y_scores = probs[:, c]
        y_pred = binary_preds[:, c]
        n_positive = int(y_true.sum())

        if n_positive > 0:
            ap = float(average_precision_score(y_true, y_scores))
            ap1 = average_precision_at_k(y_true, y_scores, k=1)
            ap5 = average_precision_at_k(y_true, y_scores, k=5)
        else:
            ap, ap1, ap5 = 0.0, 0.0, 0.0

        tp = float(np.sum(y_true * y_pred))
        fp = float(np.sum((1 - y_true) * y_pred))
        fn = float(np.sum(y_true * (1 - y_pred)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        class_f1 = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        ap_list.append(ap)
        ap1_list.append(ap1)
        ap5_list.append(ap5)
        f1_list.append(class_f1)

        per_class[name] = {
            "n_positive": n_positive, "ap": ap, "ap_rank1": ap1,
            "ap_rank5": ap5, "f1": class_f1,
        }

    present = [c for c in range(num_classes) if binary_labels[:, c].sum() > 0]
    mean_ap = float(np.mean([ap_list[c] for c in present])) if present else 0.0
    mean_ap_r1 = float(np.mean([ap1_list[c] for c in present])) if present else 0.0
    mean_ap_r5 = float(np.mean([ap5_list[c] for c in present])) if present else 0.0
    macro_f1 = float(np.mean([f1_list[c] for c in present])) if present else 0.0

    return {
        "mAP": mean_ap, "mAP_rank1": mean_ap_r1, "mAP_rank5": mean_ap_r5,
        "macro_f1": macro_f1, "per_class": per_class,
    }


def _print_task_metrics(task_name, metrics, col_width=20, multi_label=False):
    print(f"  --- {task_name} ---")
    print(f"    mAP        : {metrics['mAP'] * 100:.2f}%")
    print(f"    mAP Rank-1 : {metrics['mAP_rank1'] * 100:.2f}%")
    print(f"    mAP Rank-5 : {metrics['mAP_rank5'] * 100:.2f}%")
    print(f"    Macro F1   : {metrics['macro_f1'] * 100:.2f}%")
    count_label = "n+" if multi_label else "n"
    for name, info in metrics["per_class"].items():
        print(
            f"      {name:{col_width}s}  {count_label}={info['n_positive']:5d}  "
            f"AP={info['ap'] * 100:.1f}%  "
            f"AP@1={info['ap_rank1'] * 100:.1f}%  "
            f"AP@5={info['ap_rank5'] * 100:.1f}%  "
            f"F1={info['f1'] * 100:.1f}%"
        )


@torch.no_grad()
def evaluate(model, batches, n_batches, device):
    """Full evaluation pass returning mAP metrics per task."""
    model.eval()
    all_sp, all_act, all_action = [], [], []
    all_sp_labels, all_act_labels, all_action_labels = [], [], []

    pbar = tqdm(batches, total=n_batches, desc="Evaluating", unit="batch")
    for videos, species, activity, actions in pbar:
        videos = videos.to(device)
        sp_logits, act_logits, action_logits = model(videos)

        all_sp.append(sp_logits.cpu().numpy())
        all_act.append(act_logits.cpu().numpy())
        all_action.append(action_logits.cpu().numpy())
        all_sp_labels.append(species.numpy())
        all_act_labels.append(activity.numpy())
        all_action_labels.append(actions.numpy())

    all_sp = np.concatenate(all_sp)
    all_act = np.concatenate(all_act)
    all_action = np.concatenate(all_action)
    all_sp_labels = np.concatenate(all_sp_labels)
    all_act_labels = np.concatenate(all_act_labels)
    all_action_labels = np.concatenate(all_action_labels)

    sp_m = compute_map_metrics(all_sp, all_sp_labels, SPECIES_CLASSES)
    act_m = compute_map_metrics(all_act, all_act_labels, ACTIVITY_CLASSES)
    action_m = compute_map_metrics(all_action, all_action_labels, ACTION_CLASSES, multi_label=True)

    _print_task_metrics("Species", sp_m, col_width=15)
    _print_task_metrics("Activity", act_m, col_width=20)
    _print_task_metrics("Actions (multi-label)", action_m, col_width=20, multi_label=True)

    return {"species": sp_m, "activity": act_m, "actions": action_m}


@torch.no_grad()
def evaluate_multi_sample(model, samples, num_samples=10, min_duration=0.5, device=None):
    """Multi-sample test evaluation with duration filtering.

    For each valid clip (>= *min_duration* seconds):
      1. Randomly sample 16 frames and run inference to get a prediction vector.
      2. Repeat *num_samples* times.
      3. Average the probability vectors across the samples.
      4. Threshold at 0.5 (softmax for species/activity, sigmoid for actions).
    """
    model.eval()
    valid_samples = []
    skipped = 0

    print(f"  Filtering clips shorter than {min_duration}s ...")
    for sample in samples:
        path = sample[0]
        try:
            vframes, _, info = read_video(path, pts_unit="sec")
            fps = info.get("video_fps", 25)
            if isinstance(fps, torch.Tensor):
                fps = fps.item()
            dur = vframes.shape[0] / max(fps, 1)
        except Exception:
            dur = 0.0
        if dur < min_duration:
            skipped += 1
            continue
        valid_samples.append(sample)

    print(f"  Skipped {skipped} clips < {min_duration}s, "
          f"{len(valid_samples)} valid clips remaining")

    all_sp_probs, all_act_probs, all_action_probs = [], [], []
    all_sp_labels, all_act_labels, all_action_labels = [], [], []

    pbar = tqdm(valid_samples, desc="Multi-sample eval", unit="clip")
    for path, sp, act, action_vec in pbar:
        frame_sets = []
        for _ in range(num_samples):
            frame_sets.append(random_sample_and_preprocess(path))
        batch = torch.stack(frame_sets).to(device)

        sp_logits, act_logits, action_logits = model(batch)

        sp_probs = torch.softmax(sp_logits, dim=-1).cpu().numpy()
        act_probs = torch.softmax(act_logits, dim=-1).cpu().numpy()
        action_probs = torch.sigmoid(action_logits).cpu().numpy()

        all_sp_probs.append(sp_probs.mean(axis=0))
        all_act_probs.append(act_probs.mean(axis=0))
        all_action_probs.append(action_probs.mean(axis=0))
        all_sp_labels.append(sp)
        all_act_labels.append(act)
        all_action_labels.append(action_vec)

    all_sp_probs = np.stack(all_sp_probs)
    all_act_probs = np.stack(all_act_probs)
    all_action_probs = np.stack(all_action_probs)
    all_sp_labels = np.array(all_sp_labels, dtype=np.int32)
    all_act_labels = np.array(all_act_labels, dtype=np.int32)
    all_action_labels = np.stack(all_action_labels)

    sp_m = compute_map_metrics(all_sp_probs, all_sp_labels, SPECIES_CLASSES, from_probs=True, pred_threshold=0.5)
    act_m = compute_map_metrics(all_act_probs, all_act_labels, ACTIVITY_CLASSES, from_probs=True, pred_threshold=0.5)
    action_m = compute_map_metrics(all_action_probs, all_action_labels, ACTION_CLASSES, multi_label=True, from_probs=True, pred_threshold=0.5)

    _print_task_metrics("Species", sp_m, col_width=15)
    _print_task_metrics("Activity", act_m, col_width=20)
    _print_task_metrics("Actions (multi-label)", action_m, col_width=20, multi_label=True)

    return {"species": sp_m, "activity": act_m, "actions": action_m}


# ===================================================================
# Checkpointing
# ===================================================================

def save_checkpoint(ckpt_dir, model, optimizer, epoch, global_step):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"checkpoint_step{global_step:07d}.pt")
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()
                             if "backbone." not in k},
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, path)
    return path


def load_checkpoint(path, model, optimizer=None, device=None):
    state = torch.load(path, map_location=device or "cpu", weights_only=False)
    head_state = state["model_state_dict"]
    current = model.state_dict()
    current.update(head_state)
    model.load_state_dict(current)
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state.get("epoch", 0), state.get("global_step", 0)


# ===================================================================
# Training loop
# ===================================================================

def train(model, optimizer, scheduler, train_samples, val_samples, args, device):
    """Full training loop with per-epoch validation and checkpointing."""
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ckpt_dir = os.path.join(args.ckpt_dir, run_id)
    start_epoch = 1
    global_step = 0
    recent_ckpts = []

    w_sp = args.loss_weight_species
    w_act = args.loss_weight_activity
    w_action = args.loss_weight_actions

    print("Computing weighted batch sampling probabilities ...")
    train_weights = compute_sample_weights(train_samples)
    print(f"  weight range: [{train_weights.min():.6e}, {train_weights.max():.6e}]  "
          f"effective ratio: {train_weights.max() / train_weights.min():.1f}x")

    if args.resume_ckpt_dir:
        ckpt_files = sorted(glob.glob(os.path.join(args.resume_ckpt_dir, "checkpoint_step*.pt")))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints in {args.resume_ckpt_dir}")
        epoch_loaded, step_loaded = load_checkpoint(ckpt_files[-1], model, optimizer, device)
        start_epoch = epoch_loaded + 1
        global_step = step_loaded
        run_ckpt_dir = args.resume_ckpt_dir
        recent_ckpts = list(ckpt_files[-args.keep_recent:])
        print(f"Resuming from epoch {epoch_loaded} (step {global_step})")

    print(f"Run ID: {run_id}  |  Checkpoints \u2192 {run_ckpt_dir}")

    scaler = torch.amp.GradScaler("cuda")

    history = {
        "loss": [], "sp_loss": [], "act_loss": [], "action_loss": [],
        "sp_acc": [], "act_acc": [],
        "val_sp_map": [], "val_act_map": [], "val_action_map": [],
    }
    best_val_metric = -1.0
    best_ckpt_path = None

    n_train_batches = math.ceil(len(train_samples) / args.batch_size)
    n_val_batches = math.ceil(len(val_samples) / args.batch_size) if val_samples else 0

    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'=' * 60}")

        model.train()
        train_batches = make_batches(
            train_samples, batch_size=args.batch_size,
            num_workers=args.num_workers, sample_weights=train_weights,
            augment=True,
        )

        acc_loss = 0.0
        acc_sp_loss = 0.0
        acc_act_loss = 0.0
        acc_action_loss = 0.0
        acc_sp_acc = 0.0
        acc_act_acc = 0.0
        n_steps = 0

        batch_bar = tqdm(train_batches, total=n_train_batches, desc="  Training",
                         unit="batch", leave=False)
        for videos, species, activity, actions in batch_bar:
            videos = videos.to(device)
            species = species.to(device)
            activity = activity.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                sp_logits, act_logits, action_logits = model(videos)
                sp_loss = F.cross_entropy(sp_logits, species)
                act_loss = F.cross_entropy(act_logits, activity)
                action_loss = F.binary_cross_entropy_with_logits(action_logits, actions)
                loss = w_sp * sp_loss + w_act * act_loss + w_action * action_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            n_steps += 1

            s_loss = loss.item()
            acc_loss += s_loss
            acc_sp_loss += sp_loss.item()
            acc_act_loss += act_loss.item()
            acc_action_loss += action_loss.item()
            acc_sp_acc += (sp_logits.argmax(-1) == species).float().mean().item()
            acc_act_acc += (act_logits.argmax(-1) == activity).float().mean().item()

            batch_bar.set_postfix(
                loss=f"{s_loss:.4f}",
                sp=f"{acc_sp_acc / n_steps:.3f}",
                act=f"{acc_act_acc / n_steps:.3f}",
            )

            if global_step % args.ckpt_every == 0:
                periodic_ckpt = save_checkpoint(
                    run_ckpt_dir, model, optimizer, epoch, global_step)
                recent_ckpts.append(periodic_ckpt)
                if len(recent_ckpts) > args.keep_recent:
                    evicted = recent_ckpts.pop(0)
                    if evicted != best_ckpt_path:
                        os.remove(evicted)

        if n_steps == 0:
            continue

        avg_loss = acc_loss / n_steps
        avg_sp_loss = acc_sp_loss / n_steps
        avg_act_loss = acc_act_loss / n_steps
        avg_action_loss = acc_action_loss / n_steps
        avg_sp_acc = acc_sp_acc / n_steps
        avg_act_acc = acc_act_acc / n_steps
        history["loss"].append(avg_loss)
        history["sp_loss"].append(avg_sp_loss)
        history["act_loss"].append(avg_act_loss)
        history["action_loss"].append(avg_action_loss)
        history["sp_acc"].append(avg_sp_acc)
        history["act_acc"].append(avg_act_acc)

        log = (
            f"  loss={avg_loss:.4f}  "
            f"sp_acc={avg_sp_acc:.4f}  "
            f"act_acc={avg_act_acc:.4f}"
        )

        if val_samples:
            val_batches = make_batches(
                val_samples, batch_size=args.batch_size,
                num_workers=args.num_workers, shuffle=False,
            )
            val_metrics = evaluate(model, val_batches, n_val_batches, device)
            val_sp_map = val_metrics["species"]["mAP"]
            val_act_map = val_metrics["activity"]["mAP"]
            val_action_map = val_metrics["actions"]["mAP"]
            history["val_sp_map"].append(val_sp_map)
            history["val_act_map"].append(val_act_map)
            history["val_action_map"].append(val_action_map)
            log += (
                f"  val_sp_map={val_sp_map:.4f}  "
                f"val_act_map={val_act_map:.4f}  "
                f"val_action_map={val_action_map:.4f}"
            )

            composite = (val_sp_map + val_act_map + val_action_map) / 3.0
            if composite > best_val_metric:
                best_val_metric = composite
                best_ckpt_path = save_checkpoint(
                    run_ckpt_dir, model, optimizer, epoch, global_step)
                recent_ckpts.append(best_ckpt_path)
                if len(recent_ckpts) > args.keep_recent:
                    evicted = recent_ckpts.pop(0)
                    if evicted != best_ckpt_path:
                        os.remove(evicted)
                print(f"  \u2605 New best checkpoint (composite={best_val_metric:.4f}): {best_ckpt_path}")

        print(log)

        history_path = os.path.join(args.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best composite val metric={best_val_metric:.4f}")
    if best_ckpt_path:
        print(f"Best checkpoint: {best_ckpt_path}")

    return history, run_ckpt_dir


# ===================================================================
# Plotting
# ===================================================================

def plot_training_curves(history, output_path):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("MammalPS Benchmark 1 \u2014 Multi-task Training Curves",
                 fontsize=13, fontweight="bold")

    axes[0].plot(epochs, history["loss"], marker="o", linewidth=2, label="Total loss")
    axes[0].plot(epochs, history["sp_loss"], marker="s", linewidth=1.5, linestyle="--", label="Species loss")
    axes[0].plot(epochs, history["act_loss"], marker="^", linewidth=1.5, linestyle="--", label="Activity loss")
    axes[0].plot(epochs, history["action_loss"], marker="d", linewidth=1.5, linestyle="--", label="Action loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Losses")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8)

    axes[1].plot(epochs, [v * 100 for v in history["sp_acc"]], marker="o", linewidth=2, label="Species acc")
    axes[1].plot(epochs, [v * 100 for v in history["act_acc"]], marker="s", linewidth=2, label="Activity acc")
    if history.get("val_sp_map"):
        axes[1].plot(epochs, [v * 100 for v in history["val_sp_map"]], linestyle="--", marker="o", linewidth=2, label="Val species mAP")
    if history.get("val_act_map"):
        axes[1].plot(epochs, [v * 100 for v in history["val_act_map"]], linestyle="--", marker="s", linewidth=2, label="Val activity mAP")
    axes[1].set(xlabel="Epoch", ylabel="(%)", title="Species & Activity mAP")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8)

    if history.get("val_action_map"):
        axes[2].plot(epochs, [v * 100 for v in history["val_action_map"]], marker="^", linewidth=2, color="tab:purple", label="Val action mAP")
    axes[2].set(xlabel="Epoch", ylabel="(%)", title="Action mAP")
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to {output_path}")
    plt.close()


def plot_per_class_accuracy(metrics, output_path):
    """Three-panel bar chart: per-species AP, per-activity AP, per-action AP."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("MammalPS Benchmark 1 \u2014 Per-class AP Breakdown",
                 fontsize=13, fontweight="bold")

    sp_names = list(metrics["species"]["per_class"].keys())
    sp_aps = [metrics["species"]["per_class"][n]["ap"] * 100 for n in sp_names]
    colors_sp = plt.cm.Set2(np.linspace(0, 1, len(sp_names)))
    bars = axes[0].bar(sp_names, sp_aps, color=colors_sp, edgecolor="k", linewidth=0.6)
    overall_sp = metrics["species"]["mAP"] * 100
    axes[0].axhline(overall_sp, color="black", linestyle="--", linewidth=1.5, label=f"mAP = {overall_sp:.1f}%")
    for bar, v in zip(bars, sp_aps):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    axes[0].set(xlabel="Species", ylabel="AP (%)", title="Per-species Average Precision", ylim=(0, 110))
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    plt.sca(axes[0])
    plt.xticks(rotation=25, ha="right")

    act_names = list(metrics["activity"]["per_class"].keys())
    act_aps = [metrics["activity"]["per_class"][n]["ap"] * 100 for n in act_names]
    colors_act = plt.cm.tab10(np.linspace(0, 1, len(act_names)))
    bars = axes[1].bar(act_names, act_aps, color=colors_act, edgecolor="k", linewidth=0.6)
    overall_act = metrics["activity"]["mAP"] * 100
    axes[1].axhline(overall_act, color="black", linestyle="--", linewidth=1.5, label=f"mAP = {overall_act:.1f}%")
    for bar, v in zip(bars, act_aps):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    axes[1].set(xlabel="Activity", ylabel="AP (%)", title="Per-activity Average Precision", ylim=(0, 110))
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    plt.sca(axes[1])
    plt.xticks(rotation=35, ha="right")

    action_names = list(metrics["actions"]["per_class"].keys())
    action_aps = [metrics["actions"]["per_class"][n]["ap"] * 100 for n in action_names]
    colors_action = plt.cm.tab20(np.linspace(0, 1, len(action_names)))
    bars = axes[2].bar(action_names, action_aps, color=colors_action, edgecolor="k", linewidth=0.6)
    overall_map = metrics["actions"]["mAP"] * 100
    axes[2].axhline(overall_map, color="black", linestyle="--", linewidth=1.5, label=f"mAP = {overall_map:.1f}%")
    for bar, v in zip(bars, action_aps):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=6)
    axes[2].set(xlabel="Action", ylabel="AP (%)", title="Per-action Average Precision", ylim=(0, 110))
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    plt.sca(axes[2])
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Per-class breakdown plot saved to {output_path}")
    plt.close()


# ===================================================================
# Common setup
# ===================================================================

def print_env_info():
    print(f"PyTorch version  : {torch.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device      : {torch.cuda.get_device_name(0)}")
    print(f"Species classes   ({NUM_SPECIES}): {SPECIES_CLASSES}")
    print(f"Activity classes  ({NUM_ACTIVITIES}): {ACTIVITY_CLASSES}")
    print(f"Action classes    ({NUM_ACTIONS}): {ACTION_CLASSES}")
    print()


def resolve_checkpoint(args):
    if args.eval_ckpt:
        if not os.path.isfile(args.eval_ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.eval_ckpt}")
        return args.eval_ckpt
    if args.ckpt_dir:
        ckpt_files = sorted(glob.glob(os.path.join(args.ckpt_dir, "**", "checkpoint_step*.pt"), recursive=True))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found under {args.ckpt_dir}")
        return ckpt_files[-1]
    raise ValueError("Provide --eval_ckpt or --ckpt_dir to locate a checkpoint")


# ===================================================================
# CLI: train
# ===================================================================

def run_train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_dir = os.path.join(args.data_dir, "clips")
    train_csv = args.train_csv or os.path.join(args.data_dir, "metadata", "train.csv")
    val_csv = args.val_csv or os.path.join(args.data_dir, "metadata", "val.csv")

    print_env_info()

    print("Loading train samples ...")
    train_samples = load_csv_samples(train_csv, video_dir)
    print(f"  {len(train_samples)} train clips")

    print("Loading val samples ...")
    val_samples = load_csv_samples(val_csv, video_dir)
    print(f"  {len(val_samples)} val clips")
    print()

    n_train_batches = math.ceil(len(train_samples) / args.batch_size)
    total_steps = args.num_epochs * n_train_batches

    print(
        f"Building InternVideo-Next {args.model_size} multi-task classifier "
        f"({NUM_SPECIES} species, {NUM_ACTIVITIES} activities, {NUM_ACTIONS} actions) ..."
    )
    model, _ = build_model(
        model_size=args.model_size,
        pretrained_path=args.pretrained_path,
        dropout=args.dropout,
        head_hidden_dim=args.head_hidden_dim,
        device=device,
    )
    optimizer, scheduler = build_optimizer(
        model, learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        total_steps=total_steps,
        weight_decay=args.weight_decay,
    )
    print()

    print("Starting training ...")
    history, run_ckpt_dir = train(
        model, optimizer, scheduler, train_samples, val_samples, args, device)
    plot_training_curves(history, os.path.join(args.output_dir, "training_curves.png"))

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nCheckpoints saved in {run_ckpt_dir}")
    print(f"History & plots  in {args.output_dir}")


# ===================================================================
# CLI: test (single-pass)
# ===================================================================

def run_test(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_dir = os.path.join(args.data_dir, "clips")
    test_csv = args.test_csv or os.path.join(args.data_dir, "metadata", "test.csv")

    print_env_info()

    print("Loading test samples ...")
    test_samples = load_csv_samples(test_csv, video_dir)
    print(f"  {len(test_samples)} test clips")
    print()

    print(
        f"Building InternVideo-Next {args.model_size} multi-task classifier "
        f"({NUM_SPECIES} species, {NUM_ACTIVITIES} activities, {NUM_ACTIONS} actions) ..."
    )
    model, _ = build_model(
        model_size=args.model_size,
        pretrained_path=args.pretrained_path,
        dropout=0.0,
        head_hidden_dim=args.head_hidden_dim,
        device=device,
    )

    ckpt_path = resolve_checkpoint(args)
    print(f"Loading checkpoint: {ckpt_path}")
    load_checkpoint(ckpt_path, model, device=device)
    print()

    print("=" * 60)
    print("Test set evaluation")
    print("=" * 60)

    n_test_batches = math.ceil(len(test_samples) / args.batch_size)
    test_batches = make_batches(
        test_samples, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
    )
    test_metrics = evaluate(model, test_batches, n_test_batches, device)

    plot_per_class_accuracy(
        test_metrics, os.path.join(args.output_dir, "per_class_breakdown.png")
    )

    results_path = os.path.join(args.output_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


# ===================================================================
# CLI: test_ms (multi-sample evaluation)
# ===================================================================

def run_test_multisample(args):
    """Multi-sample test-set evaluation.

    Each clip is evaluated *num_test_samples* times with randomly sampled
    frames, and the per-sample probability vectors are averaged before
    computing metrics.  Clips shorter than *min_clip_duration* are skipped.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_dir = os.path.join(args.data_dir, "clips")
    test_csv = args.test_csv or os.path.join(args.data_dir, "metadata", "test.csv")

    print_env_info()

    print("Loading test samples ...")
    test_samples = load_csv_samples(test_csv, video_dir)
    print(f"  {len(test_samples)} test clips")
    print()

    print(
        f"Building InternVideo-Next {args.model_size} multi-task classifier "
        f"({NUM_SPECIES} species, {NUM_ACTIVITIES} activities, {NUM_ACTIONS} actions) ..."
    )
    model, _ = build_model(
        model_size=args.model_size,
        pretrained_path=args.pretrained_path,
        dropout=0.0,
        head_hidden_dim=args.head_hidden_dim,
        device=device,
    )

    ckpt_path = resolve_checkpoint(args)
    print(f"Loading checkpoint: {ckpt_path}")
    load_checkpoint(ckpt_path, model, device=device)
    print()

    print("=" * 60)
    print(f"Multi-sample test evaluation  (samples={args.num_test_samples}, "
          f"min_duration={args.min_clip_duration}s)")
    print("=" * 60)

    test_metrics = evaluate_multi_sample(
        model, test_samples, num_samples=args.num_test_samples,
        min_duration=args.min_clip_duration, device=device,
    )

    plot_per_class_accuracy(
        test_metrics, os.path.join(args.output_dir, "per_class_breakdown_ms.png")
    )

    results_path = os.path.join(args.output_dir, "test_results_multisample.json")
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


# ===================================================================
# Argument parsing
# ===================================================================

def _add_common_args(p):
    p.add_argument("--data_dir", type=str, default="../../mammalps-dataset/benchmark_1",
                    help="Root of benchmark_1 dataset")
    p.add_argument("--model_size", type=str, default="base", choices=["base", "large"])
    p.add_argument("--pretrained_path", type=str, default=None,
                    help="Local path to pretrained .bin weights (else downloads from HuggingFace)")
    p.add_argument("--head_hidden_dim", type=int, default=256,
                    help="Hidden dimension for the 2-layer MLP classification heads")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="results/benchmark1_internvideo")
    p.add_argument("--seed", type=int, default=42)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune InternVideo-Next on MammalPS benchmark_1 (multi-task)")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = sub.add_parser("train", help="Run training + validation")
    _add_common_args(p_train)
    p_train.add_argument("--train_csv", type=str, default=None)
    p_train.add_argument("--val_csv", type=str, default=None)
    p_train.add_argument("--num_epochs", type=int, default=150)
    p_train.add_argument("--learning_rate", type=float, default=1e-5)
    p_train.add_argument("--min_learning_rate", type=float, default=1e-7,
                         help="Final LR for cosine decay schedule")
    p_train.add_argument("--weight_decay", type=float, default=0.01,
                         help="AdamW weight decay")
    p_train.add_argument("--dropout", type=float, default=0.1,
                         help="Dropout rate inside MLP heads")
    p_train.add_argument("--loss_weight_species", type=float, default=1.0,
                         help="Weight for species classification loss")
    p_train.add_argument("--loss_weight_activity", type=float, default=1.0,
                         help="Weight for activity classification loss")
    p_train.add_argument("--loss_weight_actions", type=float, default=1.0,
                         help="Weight for multi-label action loss")
    p_train.add_argument("--ckpt_dir", type=str, default="checkpoints/benchmark1_internvideo")
    p_train.add_argument("--ckpt_every", type=int, default=50)
    p_train.add_argument("--keep_recent", type=int, default=5)
    p_train.add_argument("--resume_ckpt_dir", type=str, default=None)

    # ---- test ----
    p_test = sub.add_parser("test", help="Evaluate on the test set (single-pass)")
    _add_common_args(p_test)
    p_test.add_argument("--test_csv", type=str, default=None)
    p_test.add_argument("--eval_ckpt", type=str, default=None)
    p_test.add_argument("--ckpt_dir", type=str, default=None)

    # ---- test_ms (multi-sample evaluation) ----
    p_test_ms = sub.add_parser(
        "test_ms",
        help="Multi-sample test evaluation (random frame sampling, probability averaging)",
    )
    _add_common_args(p_test_ms)
    p_test_ms.add_argument("--test_csv", type=str, default=None)
    p_test_ms.add_argument("--eval_ckpt", type=str, default=None)
    p_test_ms.add_argument("--ckpt_dir", type=str, default=None)
    p_test_ms.add_argument("--num_test_samples", type=int, default=10,
                           help="Number of random frame samples per clip")
    p_test_ms.add_argument("--min_clip_duration", type=float, default=0.5,
                           help="Skip clips shorter than this duration (seconds)")

    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "test":
        run_test(args)
    elif args.command == "test_ms":
        run_test_multisample(args)


if __name__ == "__main__":
    main()
