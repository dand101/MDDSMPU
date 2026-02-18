import json
import math
import re
import csv
import statistics
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0

assert TRAIN_PATH.exists() and VAL_PATH.exists() and TEST_PATH.exists(), "Missing split files"

RUN_ROOT = HERE / "runs" / "run_3seeds_all_final_dataset_2"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 2, 3]

IMG_EPOCHS = 3
IMG_BATCH = 32
IMG_LR = 1e-4
IMG_WD = 1e-4
IMG_SIZE = 224

TEXT_MODEL = "distilroberta-base"
TXT_EPOCHS = 4
TXT_BATCH = 16
TXT_LR = 2e-5
TXT_WD = 0.01
MAX_LEN = 256

FUS_EPOCHS = 6
FUS_BATCH = 8
FUS_LR_TEXT = 2e-5
FUS_LR_FUSION = 1e-4
FUS_LR_IMG = 1e-5
FUS_WD = 0.01
WARMUP_EPOCHS_FREEZE_IMG = 2
UNFREEZE_IMG_LAYER4_AT = 3


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


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
    prev = None
    for s, y in pairs:
        if prev is not None and s != prev:
            tpr.append(tp / P)
            fpr.append(fp / N)
        if y == 1:
            tp += 1
        else:
            fp += 1
        prev = s
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
    for s, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        precision.append(tp / max(1, tp + fp))
        recall.append(tp / P)
    return [0.0] + recall, [1.0] + precision


def auc_trapz(x: List[float], y: List[float]) -> float:
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return float(area)


def average_precision(y_true: List[int], y_score: List[float]) -> float:
    r, p = pr_curve_points(y_true, y_score)
    return auc_trapz(r, p)


def calibration_bins(y_true: List[int], y_prob: List[float], n_bins: int = 10):
    bins = [[] for _ in range(n_bins)]
    for yt, pr in zip(y_true, y_prob):
        b = min(n_bins - 1, int(pr * n_bins))
        bins[b].append((yt, pr))

    centers, frac_pos, counts = [], [], []
    for i, items in enumerate(bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        centers.append((lo + hi) / 2)

        if not items:
            frac_pos.append(float("nan"))
            counts.append(0)
            continue
        ys = [yt for yt, _ in items]
        frac_pos.append(sum(ys) / len(ys))
        counts.append(len(items))

    return centers, frac_pos, counts


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
    plt.xlabel("False Positive Rate");
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve");
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
    plt.title("Precision-Recall Curve");
    plt.legend()
    save_fig(out_path)
    return ap


def plot_prob_hist(y_prob_pos: List[float], out_path: Path):
    plt.figure()
    plt.hist(y_prob_pos, bins=20)
    plt.xlabel("P(y=1)");
    plt.ylabel("count")
    plt.title("Predicted Probability Histogram")
    save_fig(out_path)


def plot_calibration(y_true: List[int], y_prob_pos: List[float], out_path: Path, n_bins: int = 10):
    centers, frac_pos, _ = calibration_bins(y_true, y_prob_pos, n_bins=n_bins)
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


def plot_aggregate_bar(means: List[float], stds: List[float], labels: List[str], title: str, ylabel: str,
                       out_path: Path):
    x = list(range(len(labels)))
    plt.figure()
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(out_path)


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


class ImageDataset(Dataset):
    def __init__(self, rows, tfm):
        self.rows = rows
        self.tfm = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = self.tfm(load_image_rgb(r["image_path"]))
        label = torch.tensor(int(r["label"]), dtype=torch.long)
        return img, label


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        text = r.get("text_clean", r.get("text", ""))
        text = quick_normalize_text(text)
        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(r["label"]), dtype=torch.long)
        return item


class FusionDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len: int, img_tfm, use_ocr: bool):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len
        self.img_tfm = img_tfm
        self.use_ocr = use_ocr

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]

        post = quick_normalize_text(r.get("text_clean", r.get("text", "")))

        if self.use_ocr:
            ocr = quick_normalize_text(r.get("ocr_text_clean", ""))
            has_ocr = int(r.get("has_ocr", 0))
            text = f"{post} {ocr}".strip() if (has_ocr and ocr) else post
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
        enc["pixel_values"] = self.img_tfm(load_image_rgb(r["image_path"]))
        enc["labels"] = torch.tensor(int(r["label"]), dtype=torch.long)
        return enc


class TextClassifier(nn.Module):
    def __init__(self, base_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.classifier(cls)


class FusionModel(nn.Module):
    def __init__(self, text_model_name: str):
        super().__init__()
        self.text_enc = AutoModel.from_pretrained(text_model_name)
        tdim = self.text_enc.config.hidden_size

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        vdim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.img_enc = resnet

        fused_dim = 512
        self.text_proj = nn.Linear(tdim, fused_dim)
        self.img_proj = nn.Linear(vdim, fused_dim)

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


@torch.no_grad()
def predict_probs_image(model, loader) -> Tuple[List[int], List[int], List[float]]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        prob_pos = probs[:, 1]
        preds = probs.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_prob.extend(prob_pos.detach().cpu().tolist())
    return y_true, y_pred, y_prob


@torch.no_grad()
def predict_probs_text(model, loader) -> Tuple[List[int], List[int], List[float]]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        logits = model(ids, mask)
        probs = torch.softmax(logits, dim=1)
        prob_pos = probs[:, 1]
        preds = probs.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_prob.extend(prob_pos.detach().cpu().tolist())
    return y_true, y_pred, y_prob


@torch.no_grad()
def predict_probs_fusion(model, loader) -> Tuple[List[int], List[int], List[float]]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for batch in loader:
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


def train_image_seed(seed: int, train_rows, val_rows, test_rows, out_dir: Path) -> Dict:
    set_seed(seed)

    train_loader = DataLoader(ImageDataset(train_rows, train_tf), batch_size=IMG_BATCH, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(ImageDataset(val_rows, eval_tf), batch_size=IMG_BATCH, shuffle=False,
                            num_workers=NUM_WORKERS)
    test_loader = DataLoader(ImageDataset(test_rows, eval_tf), batch_size=IMG_BATCH, shuffle=False,
                             num_workers=NUM_WORKERS)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=IMG_LR, weight_decay=IMG_WD)

    best_val_f1 = -1.0
    best_state = None
    history = []

    def eval_loader(loader):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, labels)
                preds = logits.argmax(1)
                total_loss += loss.item() * labels.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                y_true += labels.detach().cpu().tolist()
                y_pred += preds.detach().cpu().tolist()
        return total_loss / max(1, total), correct / max(1, total), macro_f1_from_preds(y_true, y_pred)

    for epoch in range(1, IMG_EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []
        for imgs, labels in tqdm(train_loader, leave=False, desc=f"img seed{seed} ep{epoch}"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            optim.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()

            preds = logits.argmax(1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true += labels.detach().cpu().tolist()
            y_pred += preds.detach().cpu().tolist()

        tr_loss = total_loss / max(1, total)
        tr_acc = correct / max(1, total)
        tr_f1 = macro_f1_from_preds(y_true, y_pred) if total else 0.0

        va_loss, va_acc, va_f1 = eval_loader(val_loader)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
                        "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1})

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_learning_curves(history, out_dir)

    model.load_state_dict(best_state)
    y_true, y_pred, y_prob = predict_probs_image(model, test_loader)

    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    f1 = macro_f1_from_preds(y_true, y_pred)
    roc_auc = plot_roc(y_true, y_prob, out_dir / "roc_curve.png")
    ap = plot_pr(y_true, y_prob, out_dir / "pr_curve.png")
    plot_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")
    plot_prob_hist(y_prob, out_dir / "prob_hist.png")
    plot_calibration(y_true, y_prob, out_dir / "calibration.png")

    torch.save(best_state, out_dir / "best_state_dict.pt")

    return {"acc": acc, "macro_f1": f1, "roc_auc": roc_auc, "avg_precision": ap}


def train_text_seed(seed: int, train_rows, val_rows, test_rows, out_dir: Path) -> Dict:
    set_seed(seed)

    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)

    train_loader = DataLoader(TextDataset(train_rows, tok, MAX_LEN), batch_size=TXT_BATCH, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(TextDataset(val_rows, tok, MAX_LEN), batch_size=TXT_BATCH, shuffle=False,
                            num_workers=NUM_WORKERS)
    test_loader = DataLoader(TextDataset(test_rows, tok, MAX_LEN), batch_size=TXT_BATCH, shuffle=False,
                             num_workers=NUM_WORKERS)

    model = TextClassifier(TEXT_MODEL).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=TXT_LR, weight_decay=TXT_WD)

    best_val_f1 = -1.0
    best_state = None
    history = []

    def eval_loader(loader):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                logits = model(ids, mask)
                loss = criterion(logits, labels)
                preds = logits.argmax(1)

                total_loss += loss.item() * labels.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                y_true += labels.detach().cpu().tolist()
                y_pred += preds.detach().cpu().tolist()
        return total_loss / max(1, total), correct / max(1, total), macro_f1_from_preds(y_true, y_pred)

    for epoch in range(1, TXT_EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []
        for batch in tqdm(train_loader, leave=False, desc=f"txt seed{seed} ep{epoch}"):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optim.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()

            preds = logits.argmax(1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true += labels.detach().cpu().tolist()
            y_pred += preds.detach().cpu().tolist()

        tr_loss = total_loss / max(1, total)
        tr_acc = correct / max(1, total)
        tr_f1 = macro_f1_from_preds(y_true, y_pred) if total else 0.0

        va_loss, va_acc, va_f1 = eval_loader(val_loader)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
                        "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1})

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_learning_curves(history, out_dir)

    model.load_state_dict(best_state)
    y_true, y_pred, y_prob = predict_probs_text(model, test_loader)

    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    f1 = macro_f1_from_preds(y_true, y_pred)
    roc_auc = plot_roc(y_true, y_prob, out_dir / "roc_curve.png")
    ap = plot_pr(y_true, y_prob, out_dir / "pr_curve.png")
    plot_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")
    plot_prob_hist(y_prob, out_dir / "prob_hist.png")
    plot_calibration(y_true, y_prob, out_dir / "calibration.png")

    torch.save(best_state, out_dir / "best_state_dict.pt")

    return {"acc": acc, "macro_f1": f1, "roc_auc": roc_auc, "avg_precision": ap}


def build_fusion_optimizer(model: FusionModel, stage: str):
    params = []
    params.append({"params": model.text_enc.parameters(), "lr": FUS_LR_TEXT})
    params.append({"params": list(model.text_proj.parameters()) +
                             list(model.img_proj.parameters()) +
                             list(model.classifier.parameters()),
                   "lr": FUS_LR_FUSION})
    if stage == "unfreeze_img":
        img_params = [p for p in model.img_enc.parameters() if p.requires_grad]
        if img_params:
            params.append({"params": img_params, "lr": FUS_LR_IMG})
    return torch.optim.AdamW(params, weight_decay=FUS_WD)


def train_fusion_seed(seed: int, train_rows, val_rows, test_rows, out_dir: Path, use_ocr: bool) -> Dict:
    set_seed(seed)

    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    train_loader = DataLoader(FusionDataset(train_rows, tok, MAX_LEN, train_tf, use_ocr=use_ocr),
                              batch_size=FUS_BATCH, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(FusionDataset(val_rows, tok, MAX_LEN, eval_tf, use_ocr=use_ocr),
                            batch_size=FUS_BATCH, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(FusionDataset(test_rows, tok, MAX_LEN, eval_tf, use_ocr=use_ocr),
                             batch_size=FUS_BATCH, shuffle=False, num_workers=NUM_WORKERS)

    model = FusionModel(TEXT_MODEL).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    set_requires_grad(model.img_enc, False)
    optimizer = build_fusion_optimizer(model, stage="freeze_img")

    best_val_f1 = -1.0
    best_state = None
    history = []

    def eval_loader(loader):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in loader:
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
                y_true += labels.detach().cpu().tolist()
                y_pred += preds.detach().cpu().tolist()
        return total_loss / max(1, total), correct / max(1, total), macro_f1_from_preds(y_true, y_pred)

    for epoch in range(1, FUS_EPOCHS + 1):
        if epoch == UNFREEZE_IMG_LAYER4_AT:
            unfreeze_resnet_layer4_only(model.img_enc)
            optimizer = build_fusion_optimizer(model, stage="unfreeze_img")

        stage = "freeze_img" if epoch <= WARMUP_EPOCHS_FREEZE_IMG else "unfreeze_img"

        model.train()
        total_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []
        for batch in tqdm(train_loader, leave=False, desc=f"fus{'_ocr' if use_ocr else ''} seed{seed} ep{epoch}"):
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
            y_true += labels.detach().cpu().tolist()
            y_pred += preds.detach().cpu().tolist()

        tr_loss = total_loss / max(1, total)
        tr_acc = correct / max(1, total)
        tr_f1 = macro_f1_from_preds(y_true, y_pred) if total else 0.0

        va_loss, va_acc, va_f1 = eval_loader(val_loader)
        history.append({"epoch": epoch, "stage": stage,
                        "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
                        "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1})

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_learning_curves(history, out_dir)

    model.load_state_dict(best_state)
    y_true, y_pred, y_prob = predict_probs_fusion(model, test_loader)

    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    f1 = macro_f1_from_preds(y_true, y_pred)
    roc_auc = plot_roc(y_true, y_prob, out_dir / "roc_curve.png")
    ap = plot_pr(y_true, y_prob, out_dir / "pr_curve.png")
    plot_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")
    plot_prob_hist(y_prob, out_dir / "prob_hist.png")
    plot_calibration(y_true, y_prob, out_dir / "calibration.png")

    torch.save(best_state, out_dir / "best_state_dict.pt")

    return {"acc": acc, "macro_f1": f1, "roc_auc": roc_auc, "avg_precision": ap}


def main():
    train_rows = read_jsonl(TRAIN_PATH)
    val_rows = read_jsonl(VAL_PATH)
    test_rows = read_jsonl(TEST_PATH)

    print(f"[INFO] device={DEVICE} | seeds={SEEDS}")
    print(f"[INFO] data_dir={DATA_DIR}")
    print(f"[INFO] sizes: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(f"[INFO] run_root={RUN_ROOT}")

    models_to_run = [
        ("image", None),
        ("text", None),
        ("fusion", False),
        ("fusion_ocr", True),
    ]

    all_results = {name: {"acc": [], "macro_f1": [], "roc_auc": [], "avg_precision": []} for name, _ in models_to_run}
    per_seed_detail = {}

    for seed in SEEDS:
        print(f"\n================= SEED {seed} =================")
        seed_dir = RUN_ROOT / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        per_seed_detail[str(seed)] = {}

        out_dir = seed_dir / "image"
        out_dir.mkdir(parents=True, exist_ok=True)
        r = train_image_seed(seed, train_rows, val_rows, test_rows, out_dir)
        per_seed_detail[str(seed)]["image"] = r
        for k in all_results["image"]:
            all_results["image"][k].append(r[k])
        print(
            f"[SEED {seed}] image-only      acc={r['acc']:.4f} f1={r['macro_f1']:.4f} auc={r['roc_auc']:.4f} ap={r['avg_precision']:.4f}")

        out_dir = seed_dir / "text"
        out_dir.mkdir(parents=True, exist_ok=True)
        r = train_text_seed(seed, train_rows, val_rows, test_rows, out_dir)
        per_seed_detail[str(seed)]["text"] = r
        for k in all_results["text"]:
            all_results["text"][k].append(r[k])
        print(
            f"[SEED {seed}] text-only       acc={r['acc']:.4f} f1={r['macro_f1']:.4f} auc={r['roc_auc']:.4f} ap={r['avg_precision']:.4f}")

        out_dir = seed_dir / "fusion"
        out_dir.mkdir(parents=True, exist_ok=True)
        r = train_fusion_seed(seed, train_rows, val_rows, test_rows, out_dir, use_ocr=False)
        per_seed_detail[str(seed)]["fusion"] = r
        for k in all_results["fusion"]:
            all_results["fusion"][k].append(r[k])
        print(
            f"[SEED {seed}] fusion          acc={r['acc']:.4f} f1={r['macro_f1']:.4f} auc={r['roc_auc']:.4f} ap={r['avg_precision']:.4f}")

        out_dir = seed_dir / "fusion_ocr"
        out_dir.mkdir(parents=True, exist_ok=True)
        r = train_fusion_seed(seed, train_rows, val_rows, test_rows, out_dir, use_ocr=True)
        per_seed_detail[str(seed)]["fusion_ocr"] = r
        for k in all_results["fusion_ocr"]:
            all_results["fusion_ocr"][k].append(r[k])
        print(
            f"[SEED {seed}] fusion + OCR    acc={r['acc']:.4f} f1={r['macro_f1']:.4f} auc={r['roc_auc']:.4f} ap={r['avg_precision']:.4f}")

    summary = {"means": {}, "stds": {}, "raw": all_results, "per_seed": per_seed_detail}
    print("\n================= SUMMARY (mean ± std) =================")
    for model_name in all_results:
        summary["means"][model_name] = {}
        summary["stds"][model_name] = {}
        for metric in ["acc", "macro_f1", "roc_auc", "avg_precision"]:
            m, s = mean_std(all_results[model_name][metric])
            summary["means"][model_name][metric] = m
            summary["stds"][model_name][metric] = s
        mm = summary["means"][model_name]
        ss = summary["stds"][model_name]
        print(f"{model_name:10s} | "
              f"acc={mm['acc']:.4f} ± {ss['acc']:.4f} | "
              f"macroF1={mm['macro_f1']:.4f} ± {ss['macro_f1']:.4f} | "
              f"AUC={mm['roc_auc']:.4f} ± {ss['roc_auc']:.4f} | "
              f"AP={mm['avg_precision']:.4f} ± {ss['avg_precision']:.4f}")

    (RUN_ROOT / "results_all.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = RUN_ROOT / "results_all.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "seed", "value"])
        for model_name, metrics in all_results.items():
            for metric_name, values in metrics.items():
                for seed, val in zip(SEEDS, values):
                    w.writerow([model_name, metric_name, seed, val])

    model_order = ["image", "text", "fusion", "fusion_ocr"]
    labels = ["img", "txt", "fus", "fus+ocr"]

    for metric, title, ylabel, fname in [
        ("acc", "Test Accuracy (mean ± std across seeds)", "accuracy", "agg_acc.png"),
        ("macro_f1", "Test Macro-F1 (mean ± std across seeds)", "macro-F1", "agg_macro_f1.png"),
        ("roc_auc", "Test ROC AUC (mean ± std across seeds)", "ROC AUC", "agg_roc_auc.png"),
        ("avg_precision", "Test Average Precision (mean ± std across seeds)", "avg precision", "agg_avg_precision.png"),
    ]:
        means = [summary["means"][m][metric] for m in model_order]
        stds = [summary["stds"][m][metric] for m in model_order]
        plot_aggregate_bar(means, stds, labels, title, ylabel, RUN_ROOT / fname)

    print("\nSaved everything to:", RUN_ROOT)
    print("Aggregate files:")
    print(" -", RUN_ROOT / "results_all.json")
    print(" -", RUN_ROOT / "results_all.csv")
    print(" -", RUN_ROOT / "agg_acc.png")
    print(" -", RUN_ROOT / "agg_macro_f1.png")
    print(" -", RUN_ROOT / "agg_roc_auc.png")
    print(" -", RUN_ROOT / "agg_avg_precision.png")


if __name__ == "__main__":
    main()
