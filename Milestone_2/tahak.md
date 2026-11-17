# ⚡ Tahák k Ústní Obhajobě Milestone 2 (Rapid Defense)

Rychlé a přesné odpovědi na nejčastější záludné otázky zkoušejícího.

---

### Q1: Jaký je rozdíl mezi `model.train()` a `model.eval()`?
* **model.train():**
  - **BatchNorm:** Počítá průměr $\mu_B$ a rozptyl $\sigma^2_B$ z **aktuální dávky**. Aktualizuje klouzavé odhady: $\mu_{run} = (1 - m)\mu_{run} + m\mu_B$.
  - **Dropout:** Náhodně nuluje neurony s pravděpodobností $p$ a škáluje zbývající faktorem $\frac{1}{1-p}$ (inverted dropout).
* **model.eval():**
  - **BatchNorm:** Zmrazí statistiky. Používá fixní $\mu_{run}$ a $\sigma^2_{run}$.
  - **Dropout:** Vypnutý (chová se jako identita $y = x$).

---

### Q2: Proč používáme diskriminativní learning rate při fine-tuningu?
* **Odpověď:** Spodní vrstvy (`conv1`, `layer1`) detekují nízkoúrovňové univerzální příznaky (Gaborovy filtry, hrany, barevné přechody), které jsou v ImageNetu i satelitech stejné. Vysoký LR by je přepsal (**catastrophic forgetting**).
* Horní vrstvy (`layer4`, `fc`) reprezentují sémantiku specifickou pro třídy (např. pole vs. letiště). Proto jim dáváme řádově vyšší LR ($10^{-3}$ vs $10^{-4}$ nebo $10^{-5}$).

---

### Q3: Jak předcházíme únikům paměti (Memory Leaks) v testovacím/validačním cyklu?
1. `with torch.no_grad():` — zabrání vytváření autograd výpočetního grafu a ukládání mezilehlých aktivací do RAM/VRAM.
2. `loss.item()` — extrahuje skalární hodnotu typu Python `float`. Pokud bychom sčítali `total_loss += loss`, udržujeme v paměti celý graf tenzoru pro každou dávku!
3. Explicitní mazání či `.detach()` při ukládání metrik do polí.

---

### Q4: Proč počítáme CrossEntropyLoss přímo z logitů bez Softmaxu?
* **Odpověď:** Numerická stabilita (**Log-Sum-Exp trick**).
  $$\text{CrossEntropy}(z, y) = -\log \frac{e^{z_y}}{\sum_j e^{z_j}} = -z_y + \log \sum_j e^{z_j}$$
* Oddělený Softmax + Log vede k podtečení (underflow) pro velká záporná čísla nebo přetečení (overflow) pro velká kladná čísla v $e^z$.

---

### Q5: Jak funguje Cosine Annealing LR a proč je lepší než skokové snížení (StepLR)?
* **Odpověď:** Snižuje učící rychlost podle kosinové funkce:
  $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)$$
* Zabraňuje prudkým skokům v optimalizační trajektorii. Umožňuje síti plynule "doplout" do hlubokého lokálního minima beze ztráty stability.

---

### Q6: Jaké jsou rozměry tenzoru na vstupu a výstupu každé fáze ResNet-50 při EuroSAT?
* Vstup: `(B, 3, 64, 64)`
* Stem (Conv1 + BN + MaxPool): `(B, 64, 16, 16)`
* Layer 1: `(B, 256, 16, 16)`
* Layer 2: `(B, 512, 8, 8)`
* Layer 3: `(B, 1024, 4, 4)`
* Layer 4: `(B, 2048, 2, 2)`
* AvgPool: `(B, 2048, 1, 1)`
* Flatten: `(B, 2048)`
* Linear (Head): `(B, 10)`

---

### Q7: Jaké jsou přesné naměřené výsledky a co dokazuje ablační studie?
* **M1 Baseline CNN:** Test Acc **81.48 %**, Test Loss **0.6512**, Macro F1 **0.8035**.
* **M2 Fáze 1 (Linear Probe):** Val Acc **83.63 %**, Val Loss **0.5155** (zmrazený ImageNet backbone, trénink jen hlavy).
* **M2 Fáze 2 (Full Transfer Learning):** Test Acc **97.07 %**, Test Loss **0.1018**, Macro F1 **0.9698** (unfrozen backbone, diskriminativní LR, Cosine Annealing, Mixup/CutMix).
* **Závěr pro zkoušejícího:** Samotné předtrénované váhy nestačí (zisk jen +2.15 % v Fázi 1). Zásadní průlom přináší až plná doménová adaptace konvolučních filtrů s pokročilou regularizací (skok o dalších **+13.44 %** až na **97.07 %**).

---

### Q8: Co odhalila Failure Analysis (Top-5 nejhorších chyb modelu)?
1. **Pasture $\to$ Highway (Rank 1, Loss 5.92, Conf 85.5 %):** Lineární hospodářské ohraničení, úzké polní cesty nebo suché travnaté pruhy v rozlišení $64\times 64$ tvoří geometrické přímky připomínající dálnici.
2. **PermanentCrop $\to$ HerbaceousVegetation (Rank 2 & 3, Loss 5.67 a 5.18, Conf 94.9 % a 78.0 %):** Trvalé plodiny (sady, vinice) a bylinná vegetace sdílejí téměř identické NDVI / chlorofylové odrazivosti spektra.
3. **Pasture $\to$ River (Rank 4, Loss 4.57, Conf 94.9 %):** Zavodněné odvodňovací strouhy nebo lesklé mokřady na pastvině odrážejí světlo stejně jako vodní tok.
4. **Pasture $\to$ Forest (Rank 5, Loss 4.50, Conf 68.2 %):** Hranice pastvin lemovaná hustým stromořadím.
* **Per-Class rekordy:** `Residential` dosáhl **100 % Recall** (žádná obytná zóna nebyla přehlédnuta), `Industrial` a `SeaLake` dosáhly **100 % Precision** (žádné falešně pozitivní detekce).

---

### Q9: Jak přesně funguje Mixup a CutMix a proč pomáhají?
* **Mixup:** Lineární interpolace pixelů i cílových one-hot vektorů:
  $$\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$
* **CutMix:** Náhodně vyřízne obdélník z jednoho snímku a vloží ho do druhého, přičemž $\lambda$ odpovídá přesnému poměru plochy.
* **Význam:** Nutí model predikovat proporcionální pravděpodobnosti místo binárních 0/1. Zabraňuje přehnané jistotě (overconfidence) a vytváří hladké rozhodovací hranice mezi třídami.

---

### Q10: Proč trval lokální trénink na CPU ~50 minut a jak se liší od GPU?
* **Příčina doby na CPU:** V Fázi 2 se počítal plný zpětný průchod (backpropagation) pro **všech 23,5 milionu parametrů** přes všech 50 hlubokých vrstev pro 21 600 snímků. Na Windows navíc kvůli fork-omezením běžel DataLoader v jediném vlákně (`num_workers=0`), takže CPU sekvenčně načítalo a augmentovalo každý JPEG obrázek.
* **Srovnání s GPU (Colab T4):** GPU má tisíce paralelních jader a tensor cores. S nastavením `num_workers=4` a `pin_memory=True` běží stejná epocha za **15–25 sekund** (celý trénink za cca 6–8 minut). Matematické výsledky jsou však totožné.
