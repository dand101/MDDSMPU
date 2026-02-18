import json
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from PIL import Image, ImageSequence
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

DATA_DIR = HERE / "data" / "final_dataset_2_ocr_clean"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"

RUN_DIR = HERE / "runs" / "fusion_ocr_final_dataset_2_clean"
RUN_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_MODEL = "distilroberta-base"
BATCH_SIZE = 8
EPOCHS = 6
NUM_WORKERS = 0
MAX_LEN = 256
IMG_SIZE = 224

WARMUP_EPOCHS_FREEZE_IMG = 2
UNFREEZE_IMG_LAYER4_AT = 3

LR_TEXT = 2e-5
LR_FUSION = 1e-4
LR_IMG = 1e-5
WEIGHT_DECAY = 0.01

SEED = 42
torch.manual_seed(SEED)


def macro_f1_from_preds(y_true: List[int], y_pred: List[int]) -> float:
    def f1_for_label(label: int) -> float:
        tp = sum((yt == label and yp == label) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != label and yp == label) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == label and yp != label) for yt, yp in zip(y_true, y_pred))
        denom = (2 * tp + fp + fn)
        if denom == 0:
            return 0.0
        return (2 * tp) / denom

    return 0.5 * (f1_for_label(0) + f1_for_label(1))


def confusion_counts(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    return tn, fp, fn, tp


def roc_curve_points(y_true: List[int], y_score: List[float]) -> Tuple[List[float], List[float]]:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    P = sum(y_true)
    N = len(y_true) - P
    if P == 0 or N == 0:
        return [0.0, 1.0], [0.0, 1.0]

    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]
    prev_score = None

    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            tpr.append(tp / P)
            fpr.append(fp / N)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    tpr.append(tp / P)
    fpr.append(fp / N)
    return fpr, tpr


def pr_curve_points(y_true: List[int], y_score: List[float]) -> Tuple[List[float], List[float]]:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    P = sum(y_true)
    if P == 0:
        return [0.0, 1.0], [1.0, 0.0]

    tp = 0
    fp = 0
    precision = []
    recall = []

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision.append(tp / max(1, tp + fp))
        recall.append(tp / P)

    recall = [0.0] + recall
    precision = [1.0] + precision
    return recall, precision


def auc_trapz(x: List[float], y: List[float]) -> float:
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return float(area)


def average_precision(y_true: List[int], y_score: List[float]) -> float:
    recall, precision = pr_curve_points(y_true, y_score)
    return auc_trapz(recall, precision)


def calibration_bins(y_true: List[int], y_prob: List[float], n_bins: int = 10):
    bins = [[] for _ in range(n_bins)]
    for yt, p in zip(y_true, y_prob):
        b = min(n_bins - 1, int(p * n_bins))
        bins[b].append((yt, p))

    centers, frac_pos, avg_conf, counts = [], [], [], []
    for i, items in enumerate(bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        centers.append((lo + hi) / 2)

        if not items:
            frac_pos.append(float("nan"))
            avg_conf.append(float("nan"))
            counts.append(0)
            continue

        ys = [yt for yt, _ in items]
        ps = [p for _, p in items]
        frac_pos.append(sum(ys) / len(ys))
        avg_conf.append(sum(ps) / len(ps))
        counts.append(len(items))

    return centers, frac_pos, avg_conf, counts


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_learning_curves(history: List[Dict], out_dir: Path):
    epochs = [h["epoch"] for h in history]

    plt.figure()
    plt.plot(epochs, [h["train_loss"] for h in history], label="train")
    plt.plot(epochs, [h["val_loss"] for h in history], label="val")
    plt.xlabel("epoch");
    plt.ylabel("loss");
    plt.title("Loss");
    plt.legend()
    save_fig(out_dir / "curve_loss.png")

    plt.figure()
    plt.plot(epochs, [h["train_acc"] for h in history], label="train")
    plt.plot(epochs, [h["val_acc"] for h in history], label="val")
    plt.xlabel("epoch");
    plt.ylabel("accuracy");
    plt.title("Accuracy");
    plt.legend()
    save_fig(out_dir / "curve_accuracy.png")

    plt.figure()
    plt.plot(epochs, [h["train_f1"] for h in history], label="train")
    plt.plot(epochs, [h["val_f1"] for h in history], label="val")
    plt.xlabel("epoch");
    plt.ylabel("macro-F1");
    plt.title("Macro-F1");
    plt.legend()
    save_fig(out_dir / "curve_macro_f1.png")


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], out_path: Path):
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    cm = [[tn, fp], [fn, tp]]

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (rows=true, cols=pred)")
    plt.xlabel("pred");
    plt.ylabel("true")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")
    save_fig(out_path)


def plot_roc(y_true: List[int], y_prob_pos: List[float], out_path: Path) -> float:
    fpr, tpr = roc_curve_points(y_true, y_prob_pos)
    pairs = sorted(zip(fpr, tpr), key=lambda x: x[0])
    fpr_s = [p[0] for p in pairs]
    tpr_s = [p[1] for p in pairs]
    auc = auc_trapz(fpr_s, tpr_s)

    plt.figure()
    plt.plot(fpr_s, tpr_s, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], label="chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    save_fig(out_path)
    return auc


def plot_pr(y_true: List[int], y_prob_pos: List[float], out_path: Path) -> float:
    recall, precision = pr_curve_points(y_true, y_prob_pos)
    ap = average_precision(y_true, y_prob_pos)

    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall");
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    save_fig(out_path)
    return ap


def plot_prob_hist(y_prob_pos: List[float], out_path: Path):
    plt.figure()
    plt.hist(y_prob_pos, bins=20)
    plt.xlabel("P(y=1)")
    plt.ylabel("count")
    plt.title("Predicted Probability Histogram")
    save_fig(out_path)


def plot_calibration(y_true: List[int], y_prob_pos: List[float], out_path: Path, n_bins: int = 10):
    centers, frac_pos, _, _ = calibration_bins(y_true, y_prob_pos, n_bins=n_bins)

    x, y = [], []
    for c, f in zip(centers, frac_pos):
        if not (isinstance(f, float) and math.isnan(f)):
            x.append(c);
            y.append(f)

    plt.figure()
    plt.plot([0, 1], [0, 1], label="perfect")
    plt.plot(x, y, label="model")
    plt.xlabel("Predicted probability (bin center)")
    plt.ylabel("Empirical fraction positive")
    plt.title("Calibration (Reliability) Plot")
    plt.legend()
    save_fig(out_path)


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_ws = re.compile(r"\s+")


def quick_normalize_text(t: str) -> str:
    t = (t or "").replace("\r", " ").replace("\n", " ")
    return _ws.sub(" ", t).strip()


def load_image_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if getattr(img, "is_animated", False):
        img = next(ImageSequence.Iterator(img))
    return img.convert("RGB")


train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class FusionDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_len: int, img_transform):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len
        self.img_tf = img_transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        r = self.rows[idx]

        post = quick_normalize_text(r.get("text_clean", r.get("text", "")))
        ocr = quick_normalize_text(r.get("ocr_text_clean", ""))
        has_ocr = int(r.get("has_ocr", 0))

        if has_ocr and ocr:
            text = f"{post} {ocr}"
        else:
            text = post

        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        img = self.img_tf(load_image_rgb(r["image_path"]))
        enc["pixel_values"] = img
        enc["labels"] = torch.tensor(int(r["label"]), dtype=torch.long)
        return enc


class FusionModel(nn.Module):
    def __init__(self, text_model_name: str):
        super().__init__()

        self.text_enc = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_enc.config.hidden_size

        img_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        img_dim = img_backbone.fc.in_features
        img_backbone.fc = nn.Identity()
        self.img_enc = img_backbone

        fused_dim = 512
        self.text_proj = nn.Linear(text_dim, fused_dim)
        self.img_proj = nn.Linear(img_dim, fused_dim)

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        t_out = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)
        t_cls = t_out.last_hidden_state[:, 0, :]
        t = self.text_proj(t_cls)

        v = self.img_enc(pixel_values)
        v = self.img_proj(v)

        h = torch.cat([t, v], dim=1)
        h = self.dropout(h)
        return self.classifier(h)


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def unfreeze_resnet_layer4_only(resnet: nn.Module):
    set_requires_grad(resnet, False)
    for p in resnet.layer4.parameters():
        p.requires_grad = True


def build_optimizer(model: FusionModel, stage: str) -> torch.optim.Optimizer:
    params = []
    params.append({"params": model.text_enc.parameters(), "lr": LR_TEXT})
    params.append({"params": list(model.text_proj.parameters()) +
                             list(model.img_proj.parameters()) +
                             list(model.classifier.parameters()),
                   "lr": LR_FUSION})
    if stage == "unfreeze_img":
        img_params = [p for p in model.img_enc.parameters() if p.requires_grad]
        if img_params:
            params.append({"params": img_params, "lr": LR_IMG})
    return torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)


@torch.no_grad()
def evaluate(model, loader, criterion) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    y_true, y_pred = [], []

    for batch in tqdm(loader, leave=False):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        imgs = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(ids, mask, imgs)
        loss = criterion(logits, labels)
        preds = logits.argmax(1)

        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    acc = correct / max(1, total)
    f1 = macro_f1_from_preds(y_true, y_pred) if total else 0.0
    return total_loss / max(1, total), acc, f1


def train_one_epoch(model, loader, optimizer, criterion) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    y_true, y_pred = [], []

    for batch in tqdm(loader, leave=False):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        imgs = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(ids, mask, imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    acc = correct / max(1, total)
    f1 = macro_f1_from_preds(y_true, y_pred) if total else 0.0
    return total_loss / max(1, total), acc, f1


@torch.no_grad()
def predict_probs(model, loader) -> Tuple[List[int], List[int], List[float]]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for batch in tqdm(loader, leave=False):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        imgs = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(ids, mask, imgs)
        probs = torch.softmax(logits, dim=1)
        prob_pos = probs[:, 1]
        preds = probs.argmax(dim=1)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_prob.extend(prob_pos.detach().cpu().tolist())

    return y_true, y_pred, y_prob


def main():
    assert TRAIN_PATH.exists() and VAL_PATH.exists() and TEST_PATH.exists(), "Run make_ocr_dataset_clean.py first."

    train_rows = read_jsonl(TRAIN_PATH)
    val_rows = read_jsonl(VAL_PATH)
    test_rows = read_jsonl(TEST_PATH)

    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)

    train_ds = FusionDataset(train_rows, tok, MAX_LEN, train_tf)
    val_ds = FusionDataset(val_rows, tok, MAX_LEN, eval_tf)
    test_ds = FusionDataset(test_rows, tok, MAX_LEN, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = FusionModel(TEXT_MODEL).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    print(f"[INFO] device={DEVICE} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"[INFO] data_dir={DATA_DIR}")
    print(f"[INFO] run_dir={RUN_DIR}")

    best_val_f1 = -1.0
    best_path = RUN_DIR / "best.pt"
    history = []

    set_requires_grad(model.img_enc, False)
    optimizer = build_optimizer(model, stage="freeze_img")

    for epoch in range(1, EPOCHS + 1):
        if epoch == UNFREEZE_IMG_LAYER4_AT:
            unfreeze_resnet_layer4_only(model.img_enc)
            optimizer = build_optimizer(model, stage="unfreeze_img")
            print(f"\n[INFO] Unfroze ResNet layer4 at epoch {epoch} (LR_IMG={LR_IMG})")

        stage = "freeze_img" if epoch <= WARMUP_EPOCHS_FREEZE_IMG else "unfreeze_img"
        print(f"\nEpoch {epoch}/{EPOCHS} (stage={stage})")

        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion)

        history.append({
            "epoch": epoch,
            "stage": stage,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_f1": tr_f1,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_f1": va_f1,
        })

        print(f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} macroF1={tr_f1:.4f}")
        print(f"Val  : loss={va_loss:.4f} acc={va_acc:.4f} macroF1={va_f1:.4f}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), best_path)
            print("✓ Saved best model (by val macroF1)")

    (RUN_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_learning_curves(history, RUN_DIR)

    print("\n=== TEST (best by val macroF1) ===")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    te_loss, te_acc, te_f1 = evaluate(model, test_loader, criterion)
    print(f"Test: loss={te_loss:.4f} acc={te_acc:.4f} macroF1={te_f1:.4f}")

    y_true, y_pred, y_prob = predict_probs(model, test_loader)

    roc_auc = plot_roc(y_true, y_prob, RUN_DIR / "roc_curve.png")
    ap = plot_pr(y_true, y_prob, RUN_DIR / "pr_curve.png")
    plot_confusion_matrix(y_true, y_pred, RUN_DIR / "confusion_matrix.png")
    plot_prob_hist(y_prob, RUN_DIR / "prob_hist.png")
    plot_calibration(y_true, y_prob, RUN_DIR / "calibration.png", n_bins=10)

    results = {
        "text_model": TEXT_MODEL,
        "image_model": "resnet50",
        "data_dir": str(DATA_DIR),
        "max_len": MAX_LEN,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "seed": SEED,
        "best_ckpt": str(best_path),
        "best_val_macro_f1": best_val_f1,
        "test": {
            "loss": te_loss,
            "acc": te_acc,
            "macro_f1": te_f1,
            "roc_auc": roc_auc,
            "avg_precision": ap,
            "confusion": {
                "tn": confusion_counts(y_true, y_pred)[0],
                "fp": confusion_counts(y_true, y_pred)[1],
                "fn": confusion_counts(y_true, y_pred)[2],
                "tp": confusion_counts(y_true, y_pred)[3],
            },
        },
        "artifacts": {
            "history_json": str(RUN_DIR / "history.json"),
            "results_json": str(RUN_DIR / "results.json"),
            "loss_curve": str(RUN_DIR / "curve_loss.png"),
            "acc_curve": str(RUN_DIR / "curve_accuracy.png"),
            "f1_curve": str(RUN_DIR / "curve_macro_f1.png"),
            "roc_curve": str(RUN_DIR / "roc_curve.png"),
            "pr_curve": str(RUN_DIR / "pr_curve.png"),
            "confusion_matrix": str(RUN_DIR / "confusion_matrix.png"),
            "prob_hist": str(RUN_DIR / "prob_hist.png"),
            "calibration": str(RUN_DIR / "calibration.png"),
        },
    }
    (RUN_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nSaved plots + results to:", RUN_DIR)
    print("Model:", best_path)
    print(f"ROC AUC: {roc_auc:.4f} | AP: {ap:.4f}")


if __name__ == "__main__":
    main()
