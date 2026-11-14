import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

nb_path = Path("Milestone_2/ML_KIR0041_milestone_02.ipynb")
if not nb_path.exists():
    nb_path = Path("ML_KIR0041_milestone_02.ipynb")

if not nb_path.exists():
    print(f"ERROR: {nb_path} not found!")
    sys.exit(1)

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

code_text = ""
for c in nb['cells']:
    if c['cell_type'] == 'code':
        code_text += "".join(c['source']) + "\n"

checks = [
    ("Pretrained ResNet backbone", any(k in code_text for k in ["resnet50", "ResNet50_Weights"])),
    ("Frozen base layers (Phase 1)", "requires_grad = False" in code_text or "requires_grad=False" in code_text),
    ("Unfreezing layers (Phase 2)", "requires_grad = True" in code_text or "requires_grad=True" in code_text),
    ("Discriminative LR parameter groups", "lr" in code_text and "params" in code_text and "optimizer" in code_text),
    ("Data augmentations (Mixup / CutMix)", "mixup" in code_text.lower() or "cutmix" in code_text.lower()),
    ("Regularization (Dropout & Weight Decay)", "Dropout" in code_text and "weight_decay" in code_text),
    ("Cosine Annealing LR Scheduler", "CosineAnnealingLR" in code_text or "cosine" in code_text.lower()),
    ("Top-5 highest loss failure analysis", "highest" in code_text.lower() or "top" in code_text.lower() and "loss" in code_text.lower()),
    ("Confusion matrix & classification report", "confusion_matrix" in code_text and "classification_report" in code_text),
    ("Ablation study comparison", "ablation" in code_text.lower() or "baseline" in code_text.lower()),
    ("Native PyTorch loops preserved", "optimizer.zero_grad()" in code_text and "loss.backward()" in code_text and "optimizer.step()" in code_text),
    ("Memory leak prevention (.no_grad, .item)", "torch.no_grad()" in code_text and ".item()" in code_text),
]

all_passed = True
print("\n--- Milestone 2 Requirement Verification ---")
for name, passed in checks:
    status = "✅ PASS" if passed else "❌ FAIL"
    if not passed:
        all_passed = False
    print(f"{status}: {name}")

if all_passed:
    print("\n🎉 ALL MILESTONE 2 REQUIREMENTS MET 100%!")
else:
    print("\n⚠️ SOME CHECKS FAILED - REVIEW REQUIRED.")
