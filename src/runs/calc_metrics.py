import json
from pathlib import Path
from fractions import Fraction


def safe_frac(num: int, den: int) -> Fraction:
    return Fraction(num, den) if den != 0 else Fraction(0, 1)


def per_class_from_binary_confusion(conf: dict, positive_class: int) -> dict:
    tn, fp, fn, tp = conf["tn"], conf["fp"], conf["fn"], conf["tp"]

    if positive_class == 1:
        TP, FP, FN, TN = tp, fp, fn, tn
    elif positive_class == 0:
        TP, FP, FN, TN = tn, fn, fp, tp
    else:
        raise ValueError("positive_class must be 0 or 1")

    precision = safe_frac(TP, TP + FP)
    recall = safe_frac(TP, TP + FN)
    f1 = safe_frac(2 * TP, 2 * TP + FP + FN)

    support = TP + FN

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    }


def main():
    results_path = Path("fusion_3way_gate_final_dataset_2/results.json")

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    conf = data["test"]["confusion"]

    c0 = per_class_from_binary_confusion(conf, positive_class=0)
    c1 = per_class_from_binary_confusion(conf, positive_class=1)

    macro_precision = (c0["precision"] + c1["precision"]) / 2
    macro_recall = (c0["recall"] + c1["recall"]) / 2
    macro_f1 = (c0["f1"] + c1["f1"]) / 2

    def show_frac(name: str, x: Fraction):
        print(f"{name:<14} = {x} = {float(x):.16f}")

    print("=== Per-class metrics (exact) ===")
    print("\nClass 0 (treated as positive)")
    show_frac("Precision", c0["precision"])
    show_frac("Recall", c0["recall"])
    show_frac("F1", c0["f1"])
    print(f"Support        = {c0['support']}")

    print("\nClass 1 (treated as positive)")
    show_frac("Precision", c1["precision"])
    show_frac("Recall", c1["recall"])
    show_frac("F1", c1["f1"])
    print(f"Support        = {c1['support']}")

    print("\n=== Macro averages (exact) ===")
    show_frac("Macro-Precision", macro_precision)
    show_frac("Macro-Recall", macro_recall)
    show_frac("Macro-F1", macro_f1)


if __name__ == "__main__":
    main()
