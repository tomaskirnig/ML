# 📽️ Podklady pro Prezentaci Milestone 2 (7–10 minut)

Tento dokument obsahuje strukturu slajdů, textový obsah, diagramy a **mluvnické poznámky (Speaker Notes)** pro ústní prezentaci před komisí/cvičícím.

---

## 🕒 Časový harmonogram prezentace (Target: 8 minut)
* **0:00 – 1:00:** Slide 1 — Úvod, motivace a doména problému (EuroSAT)
* **1:00 – 2:15:** Slide 2 — Shrnutí M1 Baseline & Důvody přechodu na Transfer Learning
* **2:15 – 3:45:** Slide 3 — Architektura ResNet-50 & Přizpůsobená hlava
* **3:45 – 5:15:** Slide 4 — Dvoufázový trénink, Discriminative LR & Mixup/CutMix
* **5:15 – 6:30:** Slide 5 — Ablační studie & Výsledky na testovací sadě
* **6:30 – 7:30:** Slide 6 — Diagnostika selhání (Failure Analysis & Top-5 Loss)
* **7:30 – 8:00:** Slide 7 — Závěr a doporučení pro praxi

---

## 📑 Slide 1: Úvod a doména problému
### Obsah slajdu:
* **Projekt:** Klasifikace satelitních snímků zemského pokryvu (Land Cover / Land Use).
* **Dataset:** EuroSAT (Sentinel-2 satelity, ESA).
  - 27 000 RGB snímků ve 10 třídách (Forest, River, Highway, Industrial, Residential, Pasture, ...).
  - Nízké prostorové rozlišení: $64 \times 64$ pixelů na kanál (3 kanály).
* **Vědecké rozdělení:** Deterministický seed 42 $\to$ 70 % Train (18 900), 20 % Val (5 400), 10 % Test (2 700).
* **Cíl M2:** Překonat limitaci baseline modelu (~81.5 %) a dosáhnout stavu techniky (>97 %) pomocí transfer learningu a moderních optimalizací.

> 🎙️ **Speaker Note:**  
> *"Dobrý den. V Milestone 2 řeším úlohu automatické klasifikace satelitních snímků EuroSAT. Sentinel-2 pořizuje snímky s rozlišením 10 metrů na pixel. Na ploše 64×64 pixelů to znamená, že jeden snímek pokrývá oblast zhruba 640×640 metrů. V Milestone 1 jsme ověřili funkčnost čistého PyTorch pipeline na vlastní 3-vrstvé CNN. V Milestone 2 posouváme model do produkční kvality pomocí Transfer Learningu na modelu ResNet-50."*

---

## 📑 Slide 2: Limitace Baseline (M1) vs. Motivace pro Option A
### Obsah slajdu:
* **Výsledky M1 Baseline CNN:**
  - Test Accuracy: **81.48 %**, Test Loss: **0.6512**, Macro F1: **0.8035**.
  - Znaky přetrénování: Rozestup mezi train a val loss již po 8. epoše.
* **Proč mělká síť ze scratch nestačí?**
  1. *Nízké pole vnímání (Receptive Field):* 3 konvoluce nestačí zachytit globální kontext (např. meandrující řeku vs. dálnici).
  2. *Nedostatek dat:* 18 900 snímků nestačí na naučení bohatých sémantických filtrů od nuly.
* **Volba pro M2: Option A (Transfer Learning s ResNet-50):**
  - Využití vah předtrénovaných na ImageNet-1K (1.28 milionu snímků).
  - Přenos nízkoúrovňových a středněúrovňových filtrů (hrany, rohy, textury).

> 🎙️ **Speaker Note:**  
> *"V prvním milníku jsme narazili na strop mělké architektury. Konvoluční síť se třemi vrstvami dosáhla 81.5 %. Důvodem je malé pole vnímání – filtr nevidí celkový kontext satelitní scény. Místo budování sítě od nuly jsme zvolili Option A ze zadání: přenos znalostí z modelu ResNet-50 předtrénovaného na ImageNetu."*

---

## 📑 Slide 3: Architektura ResNet-50 & Tvary Tenzorů
### Obsah slajdu:
```
Vstup: (B, 3, 64, 64)
  │
  ├─ STEM: Conv1 (7×7, s=2) + BN + MaxPool (s=2) ──► (B, 64, 16, 16)
  ├─ Layer 1: 3× Bottleneck bloky (stride 1)      ──► (B, 256, 16, 16)
  ├─ Layer 2: 4× Bottleneck bloky (stride 2)      ──► (B, 512, 8, 8)
  ├─ Layer 3: 6× Bottleneck bloky (stride 2)      ──► (B, 1024, 4, 4)
  ├─ Layer 4: 3× Bottleneck bloky (stride 2)      ──► (B, 2048, 2, 2)
  │
  ├─ AdaptiveAvgPool2d((1, 1))                    ──► (B, 2048, 1, 1)
  ├─ Flatten                                      ──► (B, 2048)
  │
  └─ Custom Head:
       ├─ Dropout (p = 0.3)
       └─ Linear (2048 -> 10)                     ──► (B, 10) Logity
```
* **Klíčový detail pro EuroSAT ($64 \times 64$):** Výstup po Layer 4 je přesně `(B, 2048, 2, 2)`.
* **Reziduální spoje:** $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x} \implies$ gradient $\frac{\partial \mathcal{E}}{\partial \mathbf{y}}$ protéká identitou přímo zpět, řeší problém mizejícího gradientu.

> 🎙️ **Speaker Note:**  
> *"Zde vidíte architekturu a rozměry tenzorů. U vstupu 64×64 je zásadní si uvědomit, že prostorová dimenze klesá přesně na 2×2 na konci čtvrté vrstvy, nikoliv na 7×7 jako u standardního ImageNetu. Původní plně propojenou vrstvu s tisíci výstupy jsme nahradili sekvencí Dropoutu s pravděpodobností 0.3 a novou lineární vrstvou s deseti logity."*

---

## 📑 Slide 4: Dvoufázový Trénink, Discriminative LR & Dynamické Augmentace
### Obsah slajdu:
* **Dvoufázová strategie:**
  - **Fáze 1: Linear Probing (4 epochy):** Celý backbone zmrazen (`requires_grad=False`), učí se pouze hlava `fc` ($\text{LR}=10^{-3}$, AdamW). Brání *Catastrophic Forgetting*.
  - **Fáze 2: Fine-Tuning (10 epoch):** Rozmrazení `layer4` a sítě s **diskriminativním LR**:
    - Backbone: $\text{LR} = 5 \times 10^{-5}$
    - Klasifikační hlava: $\text{LR} = 10^{-3}$
  - **Plánovač:** `CosineAnnealingLR` s plynulým rozpadem až k $10^{-6}$.
* **Dynamické augmentace v čistém PyTorch:**
  - **Mixup:** Lineární interpolace pixelů a cílů $\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda)\mathbf{x}_j$.
  - **CutMix:** Vložení obdélníkového výřezu jiného snímku s úměrným podílem ztráty.
  - Zabraňují overconfidence modelu a působí jako adaptivní regularizace.

> 🎙️ **Speaker Note:**  
> *"Pokud bychom celou síť trénovali hned, náhodně inicializovaná hlava by obřími gradienty zničila natrénované váhy backbone. Proto v první fázi učíme pouze novou hlavu. V druhé fázi rozmrazíme celou síť a použijeme diskriminativní learning rate – spodní vrstvy ladíme velmi jemně pomalým LR, zatímco hlavu učíme rychleji. Zároveň aplikujeme Mixup a CutMix přímo v trénovacím cyklu."*

---

## 📑 Slide 5: Ablační studie & Výsledky na testovací sadě
### Obsah slajdu:
| Model / Fáze | Architektura | Augmentace | Optimalizace | Test Loss | Test Acc | Macro F1 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **M1 Baseline** | 3-vrstvá CNN | Flipy | Adam, fixní LR | 0.6512 | 81.48 % | 0.8035 |
| **M2 Fáze 1** | ResNet-50 (Frozen) | Flipy + Crop | AdamW, fixní LR | 0.4628 | 85.78 % | ~0.8550 |
| **M2 Fáze 2** | ResNet-50 (Unfrozen) | Mixup + CutMix | Cosine Annealing + Disc. LR | **0.0967** | **97.44 %** | **0.9740** |

* **Zrychlení konvergence:** Zmrazený model překonal baseline už v první epoše.
* **Celkový zisk:** Zvýšení přesnosti o téměř **16 procentních bodů** při sedminové ztrátě.

> 🎙️ **Speaker Note:**  
> *"Zde vidíte výsledky ablační studie. Pouhé zmrazené příznaky z ImageNetu s novou lineární hlavou okamžitě dosáhly 85.78 %. Jakmile jsme v druhé fázi zapojili postupné rozmrazení, diskriminativní LR, Cosine Annealing a Mixup/CutMix s reflection paddingem, testovací přesnost vyskočila na 97.44 % a Macro F1 dosáhl 0.9740. Tím jsme eliminovali jakýkoliv bias vůči méně zastoupeným třídám."*

---

## 📑 Slide 6: Failure Analysis (Top-5 vzorků s nejvyšší ztrátou)
### Obsah slajdu:
* **Analýza vzorků s nejvyšší Cross-Entropy ztrátou na testovacím splitu:**
  1. *Herbaceous Vegetation vs. Pasture:* Spektrální odrazivost trávy je identická; bez hospodářského kontextu (např. ohrada, pasoucí se dobytek) nerozlišitelné.
  2. *Highway vs. River:* Nízké rozlišení $64 \times 64$ zkresluje úzké klikatící se linie v lesním porostu; mokrý tmavý asfalt vykazuje stejný kontrast jako vodní hladina.
  3. *Industrial vs. Residential:* Ploché betonové střechy a sklady mají identickou texturu jako moderní bytové domy.
* **Závěr z konfuzní matice:** 8 z 10 tříd dosahuje přesnosti >98 %. Záměny nastávají pouze na fyzikálních a doménových hranicích, nikoliv kvůli strukturální chybě modelu.

> 🎙️ **Speaker Note:**  
> *"Pro diagnostiku selhání jsme seřadili všech 2 700 testovacích vzorků podle jejich individuální ztráty. Top-5 nejhorších chyb jasně ukazuje fyzikální limity rozlišení 64×64. Model chybuje výhradně v hraničních případech: například u travnatých ploch Pasture vs. Herbaceous Vegetation, kde je spektrální křivka vegetace stejná, nebo u úzkých úseků silnic v lese připomínajících řeku. Nejde tedy o chybu modelu, ale o přirozený šum v popisku satelitních dat."*

---

## 📑 Slide 7: Závěr a Checklist Zadání
### Obsah slajdu:
* ✅ **Option A Splněna:** Předtrénovaný ResNet-50 s adaptovanou hlavou pro 10 tříd.
* ✅ **Frozen Base & Gradual Unfreezing:** Striktní dvoufázový trénink.
* ✅ **Discriminative Learning Rates:** Rozdělené skupiny parametrů v optimalizátoru AdamW.
* ✅ **Dynamic Augmentation:** Vlastní implementace Mixup & CutMix v čistém PyTorch.
* ✅ **Strukturální regularizace & Schedulery:** Dropout (0.3), Weight Decay ($10^{-4}$), Cosine Annealing.
* ✅ **Whiteboard & Memory Leak Prevention:** Doloženy tvary tenzorů `(B, C, H, W)`, `torch.no_grad()`, `.item()`.
* **Kód je 100% reprodukovatelný** a připravený k okamžitému spuštění v Google Colab s GPU akcelerací.

> 🎙️ **Speaker Note:**  
> *"Na závěr shrnuji, že všechny požadavky Milestone 2 ze zadání byly stoprocentně splněny v čistém PyTorch. Kód notebooku je plně reprodukovatelný se seedem 42 a připravený k okamžitému spuštění na GPU v Google Colab. Děkuji za pozornost a těším se na vaše dotazy u tabule."*
