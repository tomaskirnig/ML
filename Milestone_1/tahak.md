# 🧠 Tahák k projektu ML: EuroSAT Satelitní Klasifikace

> [!NOTE]
> **Účel tohoto dokumentu:** Srozumitelný přehled kódu z [`ML_project_1.ipynb`](file:///C:/Users/tomki/Desktop/Ing_wannabe/ML_project/ML_project_1.ipynb). Vysvětluje lidskou řečí, **co** se v kódu děje, **proč** je to tak udělané a **jak to obhájit** u tabule.
> Text je formátován pro rychlé skenování očima a snadné zapamatování klíčových pojmů.

---

## 📌 Rychlý rozcestník kapitol

* 🖥️ [1. Příprava prostředí a načtení dat](#1-příprava-prostředí-a-načtení-dat)
* 📊 [2. Průzkumná analýza dat (EDA)](#2-průzkumná-analýza-dat-eda)
* 🚚 [3. Zakázkový datový pipeline (Dataset & DataLoader)](#3-zakázkový-datový-pipeline-dataset--dataloader)
* 🔍 [4. Sanity Check první dávky](#4-sanity-check-první-dávky)
* 🏗️ [5. Architektura BaselineCNN & Whiteboard rozpad](#5-architektura-baselinecnn--whiteboard-rozpad)
* ⚡ [6. Forward Pass Sanity Check](#6-forward-pass-sanity-check)
* 🧪 [7. The Overfit Sanity Check (Krok 1.4)](#7-the-overfit-sanity-check-krok-14)
* 🏋️ [8. Explicitní trénovací smyčka (Krok 1.5)](#8-explicitní-trénovací-smyčka-krok-15)
* 🎯 [9. Evaluace a analýza chyb (Krok 1.6)](#9-evaluace-a-analýza-chyb-krok-16)
* 🎓 [10. Otázky a odpovědi k obhajobě (Whiteboard Defense)](#10-otázky-a-odpovědi-k-obhajobě-whiteboard-defense)

---

## 1. Příprava prostředí a načtení dat

### ⚙️ Co se v kódu děje:

* 🌐 **Detekce prostředí (Colab vs. lokální PC):**
  > Skript sám pozná, kde běží.
  * **V Google Colabu:** Připojí Google Disk $\to$ bleskově rozbalí `EuroSAT.zip` na rychlé virtuální SSD (`/content/EuroSAT`).
  * **Na PC ve VS Code:** Rovnou použije lokální složku `EuroSAT/` na pevném disku.

* 🚀 **Volba výpočetního zařízení (`device`):**
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
  > Pokud je k dispozici grafická karta NVIDIA, výpočty běží na **`cuda` (GPU)**. Je to **10× až 50× rychlejší** než na procesoru (CPU).

* 📦 **Rozdělení datasetu (Train / Validation / Test):**
  > Celkem máme **27 000 snímků** rozdělených podle CSV souborů:
  * 🟢 **Trénovací (Train):** `18 900 snímků (70 %)` $\to$ slouží k učení vah sítě.
  * 🟡 **Validační (Validation):** `5 400 snímků (20 %)` $\to$ kontrola přetrénování a ukládání nejlepších vah.
  * 🔴 **Testovací (Test):** `2 700 snímků (10 %)` $\to$ finální zkouška, síť je uvidí až po dokončení tréninku.

---

## 2. Průzkumná analýza dat (EDA)

> [!TIP]
> **Klíčový argument pro učitele:** Data jsou vyvážená a barvy nesou obrovský signál, ale pro jemné detaily jsou nutné konvoluce!

### 🔎 2 hlavní poznatky pro obhajobu:

1. ⚖️ **Vyváženost tříd (Class Balance):**
   > Všech 10 kategorií má podobný počet vzorků (cca 2 000 až 3 000 celkem, cca 1 400 až 2 100 v trénovací sadě).
   * **Důsledek:** Nemusíme řešit penalizace ani složité vážení ztrátové funkce (*weighted loss*).

2. 🎨 **Barevné signatury (RGB profily):**
   * 🌊 **Voda (`SeaLake`, `River`):** Masivní převaha modrého kanálu (B), červená (R) je téměř nulová.
   * 🌲 **Vegetace (`Forest`, `Pasture`, `HerbaceousVegetation`):** Jasná dominance zeleného kanálu (G).
   * 🏙️ **Zástavba a infrastruktura (`Highway`, `Industrial`, `Residential`):** Vyrovnané RGB hodnoty (šedé a zemité odstíny).
   * 💡 **Závěr pro architekturu:** Samotná průměrná barva prozradí hodně, ale k odlišení podobných věcí (např. *Highway* vs. *River* nebo *Forest* vs. *PermanentCrop*) nutně potřebujeme **prostorové tvary a textury** z konvolučních vrstev (CNN).

---

## 3. Zakázkový datový pipeline (Dataset & DataLoader)

### ⚠️ Proč nepoužíváme eager loading (načtení všeho do RAM)?
> Kdybychom načetli všech 27 000 obrázků naráz do pole v RAM, spotřebovali bychom zbytečně gigabajty paměti a u větších projektů by počítač okamžitě zkolaboval na chybě *Out Of Memory* (OOM). Náš projekt proto využívá **100% čistý Lazy Loading** hned od začátku: i výpočet kanálových statistik (`mean_rgb`, `std_rgb`) pro z-score standardizaci provádíme streamovaně po dávkách přes `DataLoader` s konstantní paměťovou náročností $\mathcal{O}(\text{batch}) \approx 3\text{ MB}$. Zadání navíc striktně vyžaduje vytvořit vlastní třídu dědící z `torch.utils.data.Dataset`.

---

### 💡 Jak funguje třída `EuroSATDataset` (tzv. Lazy Loading):
> Místo gigabajtů pixelů v paměti držíme jen lehkou textovou tabulku s cestami k souborům a štítky.

* 📏 `__len__(self)` $\to$ vrátí celkový počet řádků v tabulce.
* 🖼️ `__getitem__(self, idx)` $\to$ **až ve chvíli, kdy si síť řekne o konkrétní obrázek**:
  1. Otevře soubor z disku přes `Image.open()`.
  2. Aplikuje on-the-fly transformace.
  3. Vrátí dvojici: `(obrazový_tenzor, label)`.

---

### 🔄 Transformační pravidla (Augmentace vs. Čistá data):

* 🏋️ **Pro trénovací sadu (`train_transform`):**
  > `RandomHorizontalFlip(p=0.5)` + `RandomVerticalFlip(p=0.5)` + `ToTensor()` + `Normalize(mean, std)`
  * **Proč náhodné zrcadlení?** Satelit fotí kolmo shora. Pole, les nebo řeka neztrácejí význam, ať se na ně díváte z jakéhokoliv úhlu. Model tak v každé epoše vidí mírně jiný obrázek a **nepřetrénuje se na přesnou pozici pixelů**.
  * **Proč Z-score normalizace?** Od každého pixelu odečteme průměr kanálu a vydělíme odchylkou. Čísla se dostanou do rozmezí zhruba $[-2.0, +2.0]$. Výpočty jsou pak numericky stabilní a síť konverguje podstatně rychleji.

* 🎯 **Pro validační a testovací sadu (`eval_transform`):**
  > Pouze `ToTensor()` + `Normalize(mean, std)`
  * **Pravidlo:** Zde **nikdy nepoužíváme náhodné rotace ani ořezy**! Validace musí být 100% deterministická, abychom měřili skutečný nezkreslený výkon.

---

### 🚚 Co dělá `DataLoader`:
> Funguje jako **dopravníkový pás** mezi diskem a grafickou kartou.

* 📦 **Balíčkování (`batch_size=64`):** Skládá obrázky do dávek po 64 kusech.
* 🔀 **Míchání (`shuffle=True` pouze pro Train):** V každé epoše promíchá pořadí dávek, aby se síť neučila pořadí snímků nazpaměť.
* ⚡ **`pin_memory=True`:** Rezervuje paměť v RAM pro bleskový přenos dat do grafické karty (GPU VRAM).

---

## 4. Sanity Check první dávky

> [!IMPORTANT]
> **Kontrola před trénováním:** Zatáhli jsme za záchrannou brzdu a před spuštěním smyčky jsme vyndali úplně první balíček z `DataLoaderu`, abychom ověřili formát.

```python
sample_images, sample_labels = next(iter(train_loader))
```

### ✅ Výsledky ověření:
* 📐 **Tvar obrazového tenzoru:** `torch.Size([64, 3, 64, 64])`
  * $B = 64$ (počet snímků v dávce)
  * $C = 3$ (RGB kanály)
  * $H = 64$ (výška v px)
  * $W = 64$ (šířka v px)
* 🏷️ **Tvar a typ štítků:** `torch.Size([64])` s datovým typem `torch.int64` (celá čísla 0 až 9).
* 👁️ **Vizuální galerie:** Vykreslili jsme 8 snímků po inverzní denormalizaci a lidským okem ověřili, že popisky (např. *Forest*, *River*) přesně odpovídají zobrazenému satelitnímu snímku.

---

## 5. Architektura BaselineCNN & Whiteboard rozpad

### 🎯 Proč vlastní architektura od nuly?
> Zadání v Milestone 1 **zakazuje** stáhnout hotový model (např. ResNet). Musíme postavit vlastní jednoduchou síť ze škrábance, změřit výchozí laťku a detailně rozumět každému rozměru.

### 📐 Vzorec pro prostorový rozměr konvoluce:
$$W_{out} = \left\lfloor \frac{W_{in} - K + 2P}{S} \right\rfloor + 1$$
> Při našem nastavení $K=3$ (kernel), $P=1$ (padding) a $S=1$ (stride) zůstává prostorový rozměr po konvoluci **stejný** ($64 \to 64$).
> Teprve `MaxPool2d(2, 2)` rozměr **půlí** ($64 \to 32 \to 16 \to 8$).

---

### 📋 Whiteboard tabulka rozměrů tenzorů:

| Blok / Vrstva | Vstupní tvar `(B, C, H, W)` | Výstupní tvar `(B, C, H, W)` | Význam a role operace |
| :--- | :--- | :--- | :--- |
| **Vstup** | `(B, 3, 64, 64)` | `(B, 3, 64, 64)` | Standardizovaný satelitní snímek |
| **🧱 Blok 1** | | | |
| `Conv2d(3, 32, k=3, p=1)` | `(B, 3, 64, 64)` | `(B, 32, 64, 64)` | Hledá základní hrany, čáry a barevné přechody. |
| `BatchNorm2d(32)` | `(B, 32, 64, 64)` | `(B, 32, 64, 64)` | Udržuje rozptyl aktivací $\to$ stabilizuje gradienty. |
| `ReLU()` | `(B, 32, 64, 64)` | `(B, 32, 64, 64)` | Nelinearita: záporná čísla $\to 0$, kladná projdou. |
| `MaxPool2d(2, 2)` | `(B, 32, 64, 64)` | `(B, 32, 32, 32)` | Půlí výšku a šířku ($64 \to 32$). |
| **🧱 Blok 2** | | | |
| `Conv2d(32, 64, k=3, p=1)` | `(B, 32, 32, 32)` | `(B, 64, 32, 32)` | Hledá složitější textury (pole, cesty, zástavbu). |
| `BatchNorm2d(64)` + `ReLU`| `(B, 64, 32, 32)` | `(B, 64, 32, 32)` | Normalizace a nelinearita. |
| `MaxPool2d(2, 2)` | `(B, 64, 32, 32)` | `(B, 64, 16, 16)` | Půlí výšku a šířku ($32 \to 16$). |
| **🧱 Blok 3** | | | |
| `Conv2d(64, 128, k=3, p=1)`| `(B, 64, 16, 16)` | `(B, 128, 16, 16)` | Hledá globální sémantické tvary krajiny. |
| `BatchNorm2d(128)` + `ReLU`| `(B, 128, 16, 16)` | `(B, 128, 16, 16)` | Normalizace a nelinearita. |
| `MaxPool2d(2, 2)` | `(B, 128, 16, 16)` | `(B, 128, 8, 8)` | Půlí výšku a šířku ($16 \to 8$). |
| **🎯 Hlava modelu** | | | |
| `Flatten()` | `(B, 128, 8, 8)` | `(B, 8192)` | Rozbalí 2D mapy do 1D: $128 \times 8 \times 8 = \mathbf{8\,192}$ hodnot. |
| `Linear(8192, 256)` + `ReLU` | `(B, 8192)` | `(B, 256)` | Propojí všech 8 192 příznaků do kompaktního vektoru. |
| `Dropout(p=0.3)` | `(B, 256)` | `(B, 256)` | Vypne náhodně 30 % spojení $\to$ prevence přetrénování. |
| `Linear(256, 10)` | `(B, 256)` | `(B, 10)` | **Raw Logits** — skóre pro 10 tříd EuroSAT. |

* 📊 **Počet parametrů:** **2 193 674** (všechny trénovatelné).
* 💡 **Kde leží většina vah?** V první plně propojené vrstvě: $8\,192 \times 256 + 256 = \mathbf{2\,097\,408}$ parametrů (cca 95 % sítě).

---

## 6. Forward Pass Sanity Check

> [!TIP]
> **Zkušební jízda nanečisto:** Před spuštěním dlouhého trénování pošleme do sítě jednu testovací dávku 64 obrázků.

```python
baseline_model.eval()
with torch.no_grad():
    test_logits = baseline_model(test_batch_imgs)
```

### 🏁 Výsledek testu:
* 📥 **Vstup:** `torch.Size([64, 3, 64, 64])` na zařízení `cuda:0`
* 📤 **Výstup:** `torch.Size([64, 10])` na zařízení `cuda:0`
* ✨ **Závěr:** `ÚSPĚCH - tvar odpovídá (B, 10)` — síť se nekousne, tenzory do sebe rozměrově pasují a model vrací 10 nezkreslených raw logits připravených pro `nn.CrossEntropyLoss`.

---

## 7. The Overfit Sanity Check (Krok 1.4)

> [!IMPORTANT]
> **Proč to děláme?** Pokud síť nedokáže dokonale přetrénovat směšných 16 obrázků, nemá smysl pouštět trénink na celém datasetu. Znamenalo by to zásadní bug v pipeline!

### 🔬 Jak test probíhá:
1. Vyřízneme mikrodávku **16 vzorků** (`tiny_imgs`, `tiny_lbls`).
2. Spustíme na nich rychlou smyčku na **40 epoch** s optimalizátorem `Adam(lr=1e-3)`.
3. Sledujeme, zda ztráta spadne k nule a přesnost vyletí na 100 %.

### 🎯 Co to dokazuje pro obhajobu:
* ✅ Tok gradientů v autogradu není nikde přerušený.
* ✅ Štítky správně odpovídají predikcím (žádný off-by-one posun indexů).
* ✅ Žádné tenzory se v tichosti nerozpínají chybným broadcastingem.
* 📊 **Naměřený výsledek v notebooku:** Ztráta klesla na **`0.0014`**, přesnost dosáhla **`100.0 %`** již v 15. epoše.

---

## 8. Explicitní trénovací smyčka (Krok 1.5)

> [!NOTE]
> **Pravidlo Native PyTorch First:** Celá funkce `train_model()` je napsána ručně řádek po řádku bez jakýchkoliv knihoven třetích stran (žádný Lightning ani FastAI).

### 🔄 Pět kanonických kroků trénovací iterace:
V každé dávce trénovací fáze striktně dodržujeme toto pořadí:
```python
optimizer.zero_grad()          # 1. Vynulování nastřádaných gradientů z minulého kroku
logits = model(xb)             # 2. Forward pass (dopředný průchod sítí)
loss = criterion(logits, yb)   # 3. Výpočet ztrátové funkce (CrossEntropyLoss)
loss.backward()                # 4. Backward pass (autograd spočítá derivace dLoss/dW)
optimizer.step()               # 5. Aktualizace vah sítě
```

### 🛡️ Dvě zlatá pravidla smyčky:
1. 🔀 **Střídání režimů:**
   * `model.train()` před trénováním $\to$ zapíná Dropout a počítání statistik v BatchNorm.
   * `model.eval()` před validací $\to$ vypíná Dropout a zafixuje BatchNorm na naučené průměry.
2. 🚫 **`with torch.no_grad():`**
   * Při validaci vypíná ukládání výpočetního grafu autogradu. **Šetří obrovské množství VRAM** a zrychluje evaluaci.

### 💾 Checkpointing (`best_baseline.pt`):
* Po každé epoše porovnáme validační ztrátu (`val_loss`).
* Jakmile dosáhneme nového minima, uložíme stav vah:
  ```python
  torch.save(model.state_dict(), "best_baseline.pt")
  ```
* Po skončení všech 12 epoch **automaticky načteme tyto nejlepší váhy**, takže finální model netrpí přetrénováním z posledních epoch.
* 📊 **Naměřený výsledek v notebooku:** Nejlepší model uložen v **10. epoše** (Val Loss: **`0.2680`**, Val Accuracy: **`90.56 %`**).

---

## 9. Evaluace a analýza chyb (Krok 1.6)

### 🧪 Finální test na neviděných datech:
* Model vyhodnotíme na **2 700 testovacích snímcích**, které síť během tréninku ani validace neviděla.
* 🏆 **Naměřená testovací přesnost (Test Accuracy):** **`91.11 %`** (Ztráta: `0.2709`, Makro F1: `0.9087`).

### 🧩 Normalizovaná konfuzní matice (Confusion Matrix):
* Diagonála ukazuje procento správně trefených vzorků v rámci dané třídy:
  * 🏡 **`Residential` (98.0 % F1)** a 🌲 **`Forest` (96.2 % F1)** — takřka dokonalá klasifikace.
* Nejčastější záměny (důležité pro obhajobu):
  * 🌾 **Pasture (Pastviny)** vs. 🌿 **HerbaceousVegetation (Bylinná vegetace)**: obě kategorie jsou zelené travnaté plochy.
  * 🛣️ **Highway (Dálnice)** vs. 🏞️ **River (Řeka)**: obě jsou úzké klikaté linie táhnoucí se krajinou v rozlišení 64×64 px.
  * 🍇 **PermanentCrop** vs. 🌽 **AnnualCrop**: zemědělské plochy s podobnou strukturou brázd.

### 🖼️ Galerie chyb:
* Kód automaticky vybere snímky, kde se model spletl, a vykreslí je:
  * Vypíše skutečnou třídu (*True*), co si model myslel (*Pred*) a jeho sebevědomí v procentech (*Confidence*).
  * Skvělý podklad pro obhajobu – uvidíte, že některé snímky jsou nejednoznačné i pro lidské oko!

---

## 10. Otázky a odpovědi k obhajobě (Whiteboard Defense)

### ❓ Otázka 1: Proč v síti nemáte na konci `nn.Softmax()`?
> **Odpověď:** V PyTorch se pro multiclass klasifikaci používá ztrátová funkce `nn.CrossEntropyLoss`. Ta v sobě interně kombinuje `LogSoftmax` a `NLLLoss` pomocí matematického triku *log-sum-exp*. Kdybychom Softmax přidali sami, funkce by byla numericky nestabilní (hrozilo by podtečení/přetečení plovoucí řádové čárky) a gradienty by při špatných predikcích zbytečně mizely (vanishing gradients). Proto striktně dodržujeme pravidlo **"Logits First"**.

---

### ❓ Otázka 2: Proč rotujete a zrcadlíte obrázky, ale nepřidáváte např. černobílý filtr?
> **Odpověď:** Z průzkumné analýzy (EDA) víme, že satelitní snímky mají rotační symetrii (pohled shora), takže rotace a zrcadlení zachovávají fyzikální realitu dat. Naopak barvy nesou klíčový klasifikační signál (voda je modrá, lesy a vegetace sytě zelené). Kdybychom snímky převedli na černobílé, zahodili bychom podstatnou část rozlišovací schopnosti.

---

### ❓ Otázka 3: K čemu slouží `BatchNorm2d` a proč je v každém bloku?
> **Odpověď:** `BatchNorm` normalizuje výstupy konvoluce přes celou dávku (udržuje nulový průměr a jednotkový rozptyl). Tím stabilizuje vnitřní kovarianci sítě (tzv. internal covariate shift), umožňuje použít vyšší learning rate a zásadně zabraňuje „výbuchu“ nebo „vymizení“ gradientů (exploding / vanishing gradients).

---

### ❓ Otázka 4: Proč nepoužíváte augmentace na validačních a testovacích datech?
> **Odpověď:** Validační a testovací sada slouží k objektivnímu měření schopnosti zobecnění modelu. Pokud bychom validační data náhodně otáčeli nebo měnili, metrika by nebyla v jednotlivých epochách stabilní a srovnatelná.

---

### ❓ Otázka 5: Odkud se vzalo číslo 8 192 u vrstvy `Linear`?
> **Odpověď:** Ze 3 konvolučních bloků. Vstup $64 \times 64$ byl po třech MaxPoolech zmenšen postupně na $32 \times 32$, $16 \times 16$ a nakonec na $8 \times 8$. Třetí konvoluce má 128 výstupních kanálů. Po zploštění (`Flatten`) máme tedy $128 \times 8 \times 8 = \mathbf{8\,192}$ hodnot.

---

### ❓ Otázka 6: Proč děláme "Overfit Sanity Check" na 16 vzorcích?
> **Odpověď:** Je to základní inženýrská kontrola správnosti implementace. Pokud se síť nedokáže naučit nazpaměť malou hrstku vzorků na 100 % přesnost, znamená to, že v kódu je fatální chyba (např. odpojený autograd graf, špatně mapované indexy štítků nebo tichý broadcasting tenzorů).

---

### ❓ Otázka 7: Proč musíme v trénovací smyčce volat `optimizer.zero_grad()`?
> **Odpověď:** PyTorch ve výchozím nastavení gradienty v tenzorech **akumuluje (sčítá)** při každém volání `loss.backward()`. Kdybychom nezavolali `zero_grad()`, gradienty z nového batche by se přičetly ke starým gradientům a váhy by se aktualizovaly úplně špatným směrem.

---

### ❓ Otázka 8: Proč při počítání průměrné ztráty násobíme `loss.item() * xb.size(0)`?
> **Odpověď:** Protože poslední batch v datasetu nemusí mít plnou velikost 64 vzorků (např. má jen 28 vzorků). Kdybychom jen průměrovali průměry batche, menší dávka by měla stejnou váhu jako plná dávka. Násobením skutečnou velikostí batche a následným vydělením celkovým počtem vzorků získáme přesný vážený průměr ztráty.

---

### ❓ Otázka 9: K čemu slouží `model.train()` a `model.eval()`?
> **Odpověď:** Přepínají chování vrstev závislých na režimu: `Dropout` v `eval` režimu přestane náhodně vypínat neurony a propustí všechny aktivace. `BatchNorm2d` v `eval` režimu přestane aktualizovat klouzavé průměry a rozptyly a použije zafixované statistiky naučené během trénování.

---

### ❓ Otázka 10: Jak se liší validační sada od testovací sady?
> **Odpověď:** Validační sadu používáme **během trénování** k detekci přetrénování a k ukládání nejlepšího modelu (`checkpointing`). Testovací sadu držíme striktně stranou a použijeme ji **až jednou na úplném konci**, abychom získali nezaujaté (unbiased) zhodnocení výkonu modelu na reálných neznámých datech.
