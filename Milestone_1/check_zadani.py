import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('ML_project_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

# Check imports
code_text = ""
for c in nb['cells']:
    if c['cell_type'] == 'code':
        code_text += "".join(c['source']) + "\n"

print("\n--- Check 1: Forbidden frameworks ---")
for bad in ['lightning', 'pytorch_lightning', 'fastai', 'transformers', 'huggingface', 'torchmetrics', 'keras']:
    if bad in code_text.lower():
        print(f"WARNING: found forbidden keyword '{bad}'!")
    else:
        print(f"Passed: no '{bad}' found.")

print("\n--- Check 2: Dataset subclass ---")
print("Dataset subclass found:", "class EuroSATDataset(Dataset)" in code_text or "class EuroSATDataset(torch.utils.data.Dataset)" in code_text)
print("Has __len__:", "def __len__" in code_text)
print("Has __getitem__:", "def __getitem__" in code_text)

print("\n--- Check 3: DataLoader with batching, shuffling, workers ---")
print("DataLoader called with batch_size:", "batch_size=" in code_text)
print("DataLoader called with shuffle=True:", "shuffle=True" in code_text)
print("DataLoader called with num_workers:", "num_workers=" in code_text)
print("DataLoader called with pin_memory:", "pin_memory=" in code_text)

print("\n--- Check 4: Visualize batch post-transformation ---")
print("Batch visual inspection:", "sample_images, sample_labels" in code_text)

print("\n--- Check 5: Baseline CNN from scratch ---")
print("BaselineCNN inherits nn.Module:", "class BaselineCNN(nn.Module)" in code_text)
print("Pretrained used:", "pretrained" in code_text.lower() or "weights=" in code_text.lower())

print("\n--- Check 6: Explicit training loop ---")
print("optimizer.zero_grad():", "optimizer.zero_grad()" in code_text)
print("loss.backward():", "loss.backward()" in code_text)
print("optimizer.step():", "optimizer.step()" in code_text)
print("model.train():", "model.train()" in code_text)
print("model.eval():", "model.eval()" in code_text)
print("torch.no_grad():", "torch.no_grad()" in code_text)

print("\n--- Check 7: Overfit sanity check ---")
print("Overfit slice 16 samples:", "16" in code_text)
