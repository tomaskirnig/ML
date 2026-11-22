# 📽️ Podklady pro Prezentaci Milestone 2 (7–10 minut)

Tento dokument obsahuje strukturu slajdů, textový obsah a mluvnické poznámky (Speaker Notes) pro prezentaci.

---

## 🕒 Časový harmonogram prezentace (Cíl: ~7–8 minut)
* **0:00 – 0:45:** Slide 1 — Úvodní slide (Titul, autor Tomáš Kirnig, cíl)
* **0:45 – 2:00:** Slide 2 — Doména problému & Dataset EuroSAT (Základní info o datech, třídy, rozdělení)
* **2:00 – 3:15:** Slide 3 — Porovnání přístupů: M1 Baseline vs. M2 ResNet-50 (Stručné info k modelům)
* **3:15 – 4:45:** Slide 4 — Srovnání výsledků (Ablační studie: tabulka, klíčový skok o +16 p.b.)
* **4:45 – 5:45:** Slide 5 — Průběh trénování (Křivky ztráty a přesnosti, stabilita)
* **5:45 – 6:45:** Slide 6 — Úspěšnost podle tříd (Konfuzní matice, 100% obytné zóny)
* **6:45 – 7:45:** Slide 7 — Diagnostika selhání (Analýza Top-5 vzorků s nejvyšší ztrátou, závěr)

---

## 📑 Slide 1: Titulní snímek
* **Název:** Klasifikace satelitních snímků EuroSAT
* **Podnázev:** Milestone 2: Transfer Learning a srovnání výsledků
* **Autor:** Tomáš Kirnig (`KIR0041`)
* **Předmět:** Strojové učení (ML)
* **Model:** ResNet-50 (Transfer Learning) vs Baseline CNN

> 🎙️ **Speaker Note:**  
> *"Dobrý den. V Milestone 2 řeším úlohu automatické klasifikace satelitních snímků EuroSAT. V prvním milníku jsme ověřili funkčnost trénovacího pipeline na vlastní jednoduché 3-vrstvé CNN. Dnes se zaměříme na přechod k Transfer Learningu s modelem ResNet-50 a detailní porovnání výsledků."*

---

## 📑 Slide 2: Doména problému & Dataset EuroSAT
* **O datasetu:**
  - Satelitní snímky Sentinel-2 (Evropská vesmírná agentura ESA).
  - 27 000 barevných RGB snímků v 10 třídách zemského pokryvu.
  - Rozlišení: 64×64 pixelů (1 pixel ≈ 10 metrů v terénu).
* **Rozdělení dat & Cíl:**
  - Train: 70 % (18 900 snímků).
  - Validation: 20 % (5 400 snímků).
  - Test: 10 % (2 700 snímků – nedotčená sada pro finální evaluaci).
  - Cíl: Spolehlivá automatická klasifikace zemského povrchu.
* **Vizuál pod textem:** Pás 10 tříd datasetu (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake).

> 🎙️ **Speaker Note:**  
> *"Na úvod stručně k datasetu: EuroSAT obsahuje 27 000 snímků z družic Sentinel-2 rozdělených do deseti tříd. Každý snímek má rozlišení 64x64 pixelů a pokrývá plochu cca 640x640 metrů. Data máme rozdělená v poměru 70:20:10. Testovací sada zůstala po celou dobu nedotčená pro finální ověření."*

---

## 📑 Slide 3: Porovnání přístupů: M1 Baseline vs. M2 ResNet-50
* **M1: Mělká CNN od nuly:**
  - 3 konvoluční vrstvy + lineární vrstva.
  - Limitace: malé pole vnímání, nedostatek dat na naučení hlubokých filtrů, rychlé přetrénování po 8. epoše.
  - Strop přesnosti: 81.5 %.
* **M2: ResNet-50 & Transfer Learning:**
  - Hluboká síť (50 vrstev), předtrénováno na ImageNet-1K (1,28 milionu obrázků).
  - Dvoufázový trénink: nejprve adaptace nové hlavy, poté jemné doladění celé sítě.
  - Dynamické augmentace: Mixup a CutMix proti přetrénování.
* **Klíčová myšlenka:** Místo učení od nuly využíváme hotové obecné vidění a ladíme jen doménovou abstrakci.

> 🎙️ **Speaker Note:**  
> *"V prvním milníku jsme dosáhli 81.5 %, což byl strop mělkého modelu. Důvodem bylo malé pole vnímání a nedostatek dat pro učení od nuly. V Milestone 2 jsme sáhli po ResNet-50 předtrénovaném na ImageNetu. Zvolili jsme dvoufázovou strategii: nejprve adaptujeme novou hlavu, potom jemně doladíme celou síť spolu s dynamickými augmentacemi."*

---

## 📑 Slide 4: Srovnání výsledků (Ablační studie)
| Model / Vývojová fáze | Architektura | Augmentace | Optimalizace | Test Loss | Test Accuracy | Macro F1 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **M1 Baseline** | 3-vrstvá CNN (od nuly) | Základní flipy | Fixní rychlost učení | 0.6512 | 81.48 % | 0.8035 |
| **M2 Fáze 1 (Linear Probe)** | ResNet-50 (zmrazený) | Flipy + Crop | Trénink pouze hlavy | 0.4628 | 85.78 % | 0.8550 |
| **M2 Fáze 2 (Finální model)** | ResNet-50 (celá síť) | Mixup + CutMix | Jemné doladění + Kosinový rozpad | **0.0967** | **97.44 %** | **0.9740** |

* **Závěry:**
  - Pouhý přenos vah překonal baseline o +4.3 %.
  - Dvoufázové doladění přidalo dalších +11.7 %.
  - Celkový skok: téměř **+16 %** při sedminové chybě.

> 🎙️ **Speaker Note:**  
> *"Ablační studie jasně dokládá přínos jednotlivých kroků. Pouhé zmrazené příznaky z ImageNetu překonaly baseline hned v první fázi. Kompletní doladění s Mixupem a kosinovým rozpadem pak dosáhlo špičkových 97.44 % na testovacích datech s vyváženým Macro F1 0.9740."*

---

## 📑 Slide 5: Průběh trénování a stabilita konvergence
* **Pozorování z křivek:**
  - Okamžitý skok přesnosti z 85.8 % na >92 % během 1. epochy po rozmrazení.
  - Validační přesnost vyšší než trénovací (trénink běží se silnou regularizací Mixup/CutMix/Dropout, validační data jsou čistá).
  - Kosinové snižování rychlosti učení zajistilo hladké dosednutí do minima.
* **Vizuál:** Grafy trénovacího a validačního průběhu (Loss, Accuracy, Learning Rate).

> 🎙️ **Speaker Note:**  
> *"Zde vidíme křivky tréninku. Všimněte si okamžitého zlomu v přesnosti po rozmrazení sítě mezi fázemi. Validační přesnost je vyšší než tréninková, což je přímý důsledek použití Mixupu a CutMixu v tréninku. Model se nepřetrénovává a plynule konverguje k minimální ztrátě."*

---

## 📑 Slide 6: Úspěšnost podle tříd (Konfuzní matice)
* **Výsledky na 2 700 testovacích snímcích:**
  - 8 z 10 tříd dosahuje přesnosti > 95 %.
  - `Residential` (Zástavba): **100.0 %** bez chyby.
  - `Forest` (Les) & `SeaLake` (Voda): **99.7 %**.
  - `Industrial` (Průmysl) & `Highway` (Dálnice): **> 98.8 %**.
* **Drobné doménové překryvy (cca 2 %):**
  - Bylinná vegetace ↔ Pastviny (spektrálně téměř identická tráva).
  - Trvalé kultury ↔ Jednoleté plodiny (sady vs pole).

> 🎙️ **Speaker Note:**  
> *"Na konfuzní matici vidíme naprosto dominantní diagonálu. Obytné zóny dosáhly stoprocentní přesnosti. Lesy, vodní plochy a průmyslové zóny jsou nad 99 procenty. Drobné záměny nastávají pouze mezi pastvinami a divokou bylinnou vegetací, což jsou i pro člověka na 64 pixelech téměř nerozeznatelné plochy."*

---

## 📑 Slide 7: Diagnostika selhání (Analýza nejistých vzorků)
* **Top-5 vzorků s nejvyšší ztrátou:**
  - *Pasture → Highway (Loss 7.12, Jistota 93.9 %):* Asfaltová přístupová cesta protíná louku a dominuje 64×64 pixelům.
  - *Herbaceous → Forest (Loss 6.68, Jistota 95.5 %):* Tmavý hustý křovinatý porost spektrálně splývá s jehličnatým lesem.
  - *Industrial → Highway (Loss 4.43, Jistota 94.5 %):* Manipulační plocha a vlečka u skladu s lineárním kontrastem.
* **Závěr:** Model neselhal strukturálně – záměny plynou z fyzikálního rozlišení 64×64 pixelů a reálného vizuálního překryvu v terénu.

> 🎙️ **Speaker Note:**  
> *"Zanalyzovali jsme vzorky s nejvyšší ztrátou na testovací sadě. Vidíme, že model chyboval naprosto pochopitelně – například polní asfaltka uprostřed pastviny nebo manipulační plocha u skladu s kolejemi. Model se tedy naučil správné vizuální vzorce a naráží jen na fyzikální limit rozlišení. Tímto děkuji za pozornost a těším se na dotazy."*
