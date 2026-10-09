# 📐 Pravidla a principy vývoje v PyTorch (založeno na cvičeních a zadání)

Tento dokument shrnuje osvědčené postupy, architektonické zásady a pravidla kódování odvozená ze cvičení kurzu (`CV/ML_01` až `CV/ML_06`) a ze specifikace projektu (`Zadani.txt`). Tyto principy slouží jako závazný standard pro veškerou práci na projektu.

---

## 1. Filozofie projektu a AI Policy ("Native PyTorch First")

1. **Čistý PyTorch jako základ (Milestone 1)**:
   - Veškeré komponenty v Milestone 1 (datasety, modely, trénovací smyčky, metriky) musí být postaveny na nativních primitivech PyTorch (`torch.Tensor`, `torch.autograd`, `torch.nn.Module`, `torch.optim`, `torch.utils.data.Dataset/DataLoader`).
   - Je **zakázáno** používat high-level wrappery jako PyTorch Lightning, Hugging Face Trainer nebo FastAI v Milestone 1.
2. **Připravenost na "Whiteboard / Shape" obhajobu**:
   - U každého modelu musíme být schopni okamžitě napsat a obhájit přesný tvar tenzorů `(B, C, H, W)` na každém rozhraní vrstev.
   - Musíme rozumět toku gradientů: proč a kde hrozí vanishing/exploding gradients a jak jim předcházet (volba aktivace, inicializace vah, normalizace).
3. **Transparentnost a porozumění kódu**:
   - Žádný "černá skříňka" kód. Každý krok pipeline musí mít jasné zdůvodnění a explicitní implementaci.

---

## 2. Zpracování dat a pipeline (`torch.utils.data`)

### 2.1 Vlastní třída Dataset (`torch.utils.data.Dataset`)
*Pravidlo z CV_06 a zadání:*
- Vždy vytvořit vlastní třídu dědící z `torch.utils.data.Dataset`.
- Třída musí explicitně implementovat metody:
  - `__init__(self, ...)`: Uložení cest k souborům, metadat, labelů a transformační pipeline. Obrázky nenačítat všechny najednou do paměti, pokud je dataset velký – preferovat **lazy loading** (načtení souboru z disku v `__getitem__`).
  - `__len__(self)`: Vrátí celkový počet vzorků (`len(self.samples)`).
  - `__getitem__(self, idx)`: Načte konkrétní vzorek (např. pomocí `PIL.Image.open`), aplikuje příslušné transformace a vrátí dvojici `(image_tensor, label_tensor)`.

### 2.2 Datové typy a dimenze tenzorů
- **Obrazy**: Vždy formát `(B, C, H, W)` s datovým typem `torch.float32`.
  - Hodnoty standardizované (z-score normalizace pomocí `mean` a `std`) nebo škálované do rozsahu `[0.0, 1.0]`.
- **Klasifikační labely (Multiclass)**:
  - Vždy celočíselný tenzor `torch.long` (int64) tvaru `(B,)` s indexy tříd $0, 1, \dots, C-1$.
  - Nikdy nepoužívat one-hot kódování pro `nn.CrossEntropyLoss` (PyTorch očekává přímo indexy tříd).

### 2.3 Data Augmentation a oddělení splitů
*Pravidlo z CV_03 a CV_04:*
- **Augmentace pouze pro trénovací sadu**:
  - Trénovací transformace (`train_transform`): např. `RandomHorizontalFlip`, `RandomRotation(15)`, `RandomResizedCrop`, následované `ToTensor()` a `Normalize()`.
  - Validační/Testovací transformace (`eval_transform`): výhradně deterministické úpravy – pouze `ToTensor()` a `Normalize()`. Nikdy neaplikovat náhodné augmentace na validační data!
- Transformace se provádějí **on-the-fly** během čtení z DataLoaderu, což šetří RAM a v každé epoše poskytuje modelu mírně odlišnou variaci vzorku.

### 2.4 Nastavení `DataLoader`
- `batch_size`: Obvykle mocnina dvou (32, 64, 128) s ohledem na velikost VRAM.
- `shuffle`: **Vždy `True`** pro trénovací sadu, **vždy `False`** pro validační a testovací sadu.
- `num_workers`: Nastavit na $> 0$ pro asynchronní načítání v produkci (pozor na Windows kompatibilitu `num_workers=0` nebo `2`).
- `pin_memory=True`: Při trénování na GPU zrychluje přenos z RAM do VRAM.

### 2.5 Vizuální inspekce dávky (Sanity Check dat)
*Pravidlo z CV_01, CV_03 a zadání:*
- Před spuštěním trénování vždy provést manuální kontrolu první dávky:
  ```python
  images, labels = next(iter(train_loader))
  print("Batch images shape:", images.shape)   # očekáváno např. (B, 3, 64, 64)
  print("Batch labels shape:", labels.shape)   # očekáváno např. (B,)
  print("Pixel range:", images.min().item(), "to", images.max().item())
  ```
- Vykreslit mřížku vzorků z dávky včetně názvů tříd pro ověření správného mapování štítků.

---

## 3. Architektura modelů (`torch.nn.Module`)

### 3.1 Konstrukce sítě
*Pravidlo z CV_01, CV_02, CV_03:*
- Třída modelu dědí z `nn.Module`.
- První řádek v `__init__` musí být vždy `super().__init__()`.
- Všechny vrstvy s parametry (`nn.Conv2d`, `nn.BatchNorm2d`, `nn.Linear`) se definují v `__init__`.
- Výpočetní graf se skládá výhradně v metodě `forward(self, x)`.

### 3.2 Analytický výpočet prostorových rozměrů (CNN)
*Pravidlo z CV_03:*
- Výstupní rozměr konvoluční vrstvy se řídí vzorcem:
  $$W_{out} = \left\lfloor \frac{W_{in} - K + 2P}{S} \right\rfloor + 1$$
  kde $K$ je kernel size, $P$ je padding, $S$ je stride.
- Pro pooling se stejným způsobem redukuje prostorový rozměr.
- Při přechodu z 2D konvoluce na plně propojenou vrstvu (`Linear`) použít `nn.Flatten()` nebo `x.view(x.size(0), -1)` s explicitně ověřeným rozměrem vstupních příznaků.

### 3.3 Pravidlo výstupní vrstvy: "Logits first"
*Kritické pravidlo z CV_02 a CV_03:*
- Model v poslední vrstvě vrací **čisté logits** (žádný `nn.Softmax()` ani `F.softmax()`).
- Ztrátová funkce `nn.CrossEntropyLoss` v sobě interně kombinuje `LogSoftmax` a `NLLLoss` pomocí log-sum-exp triku, což:
  1. Zajišťuje numerickou stabilitu (zamezuje podtečení / přetečení v plovoucí řádové čárce).
  2. Zabraňuje saturaci gradientů (vanishing gradients) při špatných predikcích.

### 3.4 Regularizace a stabilita
- **Batch Normalization (`nn.BatchNorm2d`)**: Zařazovat mezi konvoluci a nelinearitu (nebo těsně za ni), urychluje konvergenci a stabilizuje gradienty.
- **Dropout (`nn.Dropout`)**: Používat v plně propojených vrstvách (např. $p=0.2$ až $0.5$) proti přetrénování.
- **Počítadlo parametrů**: Implementovat pomocnou funkci pro výpis parametrů:
  ```python
  def count_params(model):
      total = sum(p.numel() for p in model.parameters())
      trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
      return total, trainable
  ```

---

## 4. Explicitní trénovací a validační smyčka

### 4.1 Režimy modelu (`train` vs. `eval`)
*Pravidlo z CV_02, CV_03, CV_04:*
- Před trénovací smyčkou **vždy** zavolat `model.train()` (aktivuje Dropout, zapíná počítání running statistik v BatchNorm).
- Před validační smyčkou **vždy** zavolat `model.eval()` (deaktivuje Dropout, zmrazí statistiky BatchNorm na evaluační režim).

### 4.2 Validační fáze bez gradientů
- Validační a testovací kód musí běžet uvnitř kontextu `with torch.no_grad():`.
- Důvod: Vypíná sledování autograd grafu, což dramaticky šetří VRAM a zrychluje evaluaci.

### 4.3 Pět kanonických kroků trénovací iterace
V každém kroku trénovacího batche dodržet přesné pořadí:
```python
optimizer.zero_grad()          # 1. Vynulování nastřádaných gradientů
logits = model(xb)             # 2. Forward pass
loss = criterion(logits, yb)   # 3. Výpočet ztrátové funkce
loss.backward()                # 4. Backward pass (autograd spočítá dLoss/dW)
optimizer.step()               # 5. Aktualizace vah optimalizátorem
```

### 4.4 Správná agregace ztráty a metrik
*Pravidlo z CV_02 a CV_03:*
- Nikdy neprůměrovat průměry batche přímo (`sum(batch_loss) / len(batches)` může být zkreslené, pokud poslední batch nemá plnou velikost).
- Správný postup:
  ```python
  running_loss += loss.item() * xb.size(0)
  running_correct += (logits.argmax(dim=1) == yb).sum().item()
  # Na konci epochy:
  epoch_loss = running_loss / len(loader.dataset)
  epoch_acc  = running_correct / len(loader.dataset)
  ```

### 4.5 Checkpointování nejlepšího modelu
*Pravidlo z CV_06:*
- Průběžně sledovat validační ztrátu (`val_loss`) nebo přesnost (`val_acc`).
- Při dosažení nejlepší hodnoty uložit váhy pomocí:
  ```python
  torch.save(model.state_dict(), "best_baseline.pt")
  ```
- Po skončení trénování načíst nejlepší váhy (`model.load_state_dict(...)`) před finálním vyhodnocením na testovací sadě.

---

## 5. Diagnostika a ověřovací mechanismy (Sanity Checks)

### 5.1 The Overfit Sanity Check (Klíčový požadavek zadání)
- **Před jakýmkoliv dlouhým trénováním** vybrat mikrodávku 8–16 vzorků.
- Spustit na ní trénovací smyčku na cca 30–50 epoch bez regulace.
- Model **musí** dosáhnout ztráty blízké 0 a přesnosti 100 %.
- **Co to dokazuje**:
  - Tok gradientů je neporušený.
  - Tenzory nejsou tiše broadcastovány do chybných dimenzí.
  - Ztrátová funkce a výpočet přesnosti správně párují predikce se štítky.

### 5.2 Baseline jako referenční bod
*Pravidlo z CV_02, CV_04 a zadání:*
- V Milestone 1 postavit jednoduchý model od nuly (např. 2–3 vrstvá CNN) bez předtrénovaných vah.
- Změřit jeho surový (unassisted) výkon jako laťku pro pokročilejší modely v Milestone 2.

### 5.3 Diagnostika chyb a evaluace
*Pravidlo z CV_03:*
- Nespokojit se pouze s celkovou přesností (Accuracy).
- Vždy vygenerovat:
  1. **Konfuzní matici (Confusion Matrix)** s normalizací po řádcích – odhalí, které konkrétní třídy se modelu pletou.
  2. **Per-class statistiky** (Precision, Recall, F1-Score).
  3. **Vizuální galerii nejistých a chybných predikcí** (analýza vzorků, kde model nejvíce selhal).
