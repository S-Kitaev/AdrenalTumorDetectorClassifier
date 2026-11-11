"""Run CTModel on 10 random patients from CT_mephi_newdata/data/npy."""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
NPY_DIR = Path(r"C:\Users\kkukh\CursorProjects\CT_mephi_newdata\data\npy")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from nnmodel.CTModel import CTModel  # noqa: E402

PHENO_MAP = {
    "benign": "Benign",
    "indeterminate": "Indeterminate",
    "malignant": "Malignant",
}
DIAG_MAP = {
    "carcinoma": "Adrenocortical Carcinoma",
    "pheochromocytoma": "Pheochromocytoma",
    "adenoma": "Adenoma",
    "nodular_hyperplasia": "Nodular Hyperplasia",
    "cyst": "Cyst",
    "myelolipoma": "Myelolipoma",
}


def parse_folder(name: str) -> tuple[str, str, str]:
    """Primary (most severe) phenotype/diagnosis from folder name."""
    rest = name.split("_", 1)[1]
    primary = rest.split("+")[0]
    m = re.match(r"(benign|indeterminate|malignant)_(.+)", primary)
    if not m:
        return name, "unknown", "unknown"
    return name, PHENO_MAP.get(m.group(1), m.group(1)), DIAG_MAP.get(m.group(2), m.group(2))


def pick_scan(folder: Path) -> Path | None:
    files = sorted(folder.glob("*.npy"))
    scans = [p for p in files if "_MASK" not in p.name.upper()]
    if not scans:
        return None
    native = [p for p in scans if "NATIVE" in p.name.upper()]
    return native[0] if native else scans[0]


def norm(s) -> str:
    return (s or "").strip().casefold()


def main():
    folders = [p for p in NPY_DIR.iterdir() if p.is_dir()]
    rng = random.Random(42)
    sample = rng.sample(folders, k=min(10, len(folders)))
    sample.sort(key=lambda p: p.name)

    print(f"Sampled {len(sample)} / {len(folders)} folders\n")
    model = CTModel()
    model.load()
    print(f"diagnosis_loaded={model.model_classification.diagnosis_loaded}\n")

    rows = []
    for folder in sample:
        gt_name, gt_pheno, gt_diag = parse_folder(folder.name)
        scan = pick_scan(folder)
        rec = {
            "folder": folder.name,
            "scan": None if scan is None else scan.name,
            "gt_phenotype": gt_pheno,
            "gt_diagnosis": gt_diag,
            "pred_phenotype": None,
            "pred_diagnosis": None,
            "pheno_ok": False,
            "diag_ok": None,
            "error": None,
            "timings": None,
        }
        print("=" * 80)
        print(f"{folder.name}")
        print(f"  GT: {gt_pheno} / {gt_diag}")
        print(f"  scan: {scan}")
        if scan is None:
            rec["error"] = "no scan npy"
            rows.append(rec)
            continue
        try:
            model.predict(str(scan))
            rec["pred_phenotype"] = model.label_name
            rec["pred_diagnosis"] = model.diagnosis_name
            rec["timings"] = model.timings
            rec["pheno_ok"] = norm(model.label_name) == norm(gt_pheno)
            if gt_pheno == "Indeterminate":
                rec["diag_ok"] = model.diagnosis_name is None
            elif gt_diag not in DIAG_MAP.values():
                rec["diag_ok"] = None  # not in model taxonomy (neurogenic, etc.)
            else:
                rec["diag_ok"] = norm(model.diagnosis_name) == norm(gt_diag)
            print(
                f"  PRED: {model.label_name} / {model.diagnosis_name}  "
                f"pheno={'OK' if rec['pheno_ok'] else 'MISS'}  "
                f"diag={rec['diag_ok']}  t={model.timings.get('total')}"
            )
        except Exception as e:
            rec["error"] = str(e)
            print(f"  ERROR: {e}")
        rows.append(rec)

    n = len([r for r in rows if r["error"] is None])
    pheno_hit = sum(1 for r in rows if r["pheno_ok"])
    diag_eval = [r for r in rows if r["diag_ok"] is not None]
    diag_hit = sum(1 for r in diag_eval if r["diag_ok"])

    print("\n" + "=" * 80)
    print(f"{'folder':<55} {'GT pheno':<16} {'PRED pheno':<16} {'ok':<4} {'GT diag':<28} {'PRED diag':<28} {'ok'}")
    for r in rows:
        print(
            f"{r['folder']:<55} {str(r['gt_phenotype']):<16} {str(r['pred_phenotype']):<16} "
            f"{str(r['pheno_ok']):<4} {str(r['gt_diagnosis']):<28} {str(r['pred_diagnosis']):<28} {r['diag_ok']}"
        )
    print()
    print(f"Phenotype: {pheno_hit}/{n}  Diagnosis: {diag_hit}/{len(diag_eval)}")
    out = ROOT / "eval_random10_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
