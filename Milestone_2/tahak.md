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
