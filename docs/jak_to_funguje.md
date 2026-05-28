# Jak funguje SketchMind

---

## 1. Celkový přehled

Aplikace má tři hlavní části:
- **Kreslící canvas** — uživatel kreslí myší nebo prstem přes webcam
- **Neuronová síť** — dostane obrázek 28x28px a vrátí co si myslí že je na něm
- **Panel s výsledkem** — zobrazí predikci a pravděpodobnosti pro každou cifru

```mermaid
graph LR
    A[Uživatel kreslí] --> B[Canvas 560x560px]
    B --> C[Zmenšení na 28x28]
    C --> D[Neuronová síť]
    D --> E[Predikce + pravděpodobnosti]
    E --> F[UI panel]
```

---

## 2. Sekvenční diagram — od tahu po výsledek

```mermaid
sequenceDiagram
    actor U as Uživatel
    participant C as Canvas (app.py)
    participant P as PIL Image
    participant N as NeuralNetwork
    participant UI as Panel

    U->>C: nakreslí tah myší / prstem
    C->>P: nakresli elipsu na PIL image (maska tahů)
    C->>C: schedule_clear() — nastav timer 2s

    C->>P: center_image() — ořízni prázdné okraje
    P->>P: vystředí cifru do 200x200
    P->>P: zmenší na 28x28 px
    P->>N: předej jako pole 784 čísel (0–1)

    N->>N: forward pass — vrstva po vrstvě
    N->>N: ReLU na skrytých vrstvách
    N->>N: Softmax na výstupu
    N-->>C: vrátí predikci + 10 pravděpodobností

    C->>UI: zobraz číslo + confidence
    C->>UI: aktualizuj probability bary

    Note over C: pokud 2s bez kreslení...
    C->>C: clear() — vymaž canvas i PIL image
    C->>UI: reset na výchozí stav
```

---

## 3. Neuronová síť

### Architektura

```mermaid
graph LR
    A["Vstup\n784 neuronů\n(28x28 pixelů)"]
    B["Skrytá vrstva 1\n128 neuronů\nReLU"]
    C["Skrytá vrstva 2\n64 neuronů\nReLU"]
    D["Výstup\n10 neuronů\nSoftmax"]

    A --> B --> C --> D
```

- **784 vstupů** — každý pixel obrázku 28x28 je jeden vstup (hodnota 0–1)
- **128 a 64** — skryté vrstvy, kde se učí rozpoznávat tvary a vzory
- **10 výstupů** — jeden pro každou cifru (0–9), hodnoty jsou pravděpodobnosti

### Aktivační funkce

**ReLU** (skryté vrstvy) — záporné hodnoty zahodí, kladné nechá projít:
```
f(x) = max(0, x)
```

**Softmax** (výstupní vrstva) — převede čísla na pravděpodobnosti, součet = 1:
```
f(x_i) = e^x_i / součet(e^x_j)
```

### Inicializace vah

Váhy se inicializují náhodně podle He inicializace:
```
w = náhodné číslo * sqrt(2 / počet_vstupů)
```
Bez toho by síť konvergovala velmi pomalu nebo vůbec.

---

## 4. Trénování (train.py)

### Dataset — MNIST

- 60 000 trénovacích obrázků, 10 000 testovacích
- Každý obrázek je 28x28 pixelů, černobílý
- Stahuje se automaticky při prvním spuštění

### Průběh trénování

```mermaid
flowchart TD
    A[Načti MNIST dataset] --> B[Zamíchej data]
    B --> C[Vezmi batch 64 obrázků]
    C --> D[Forward pass - síť udělá predikci]
    D --> E[Spočítej chybu - cross-entropy loss]
    E --> F[Backward pass - spočítej gradienty]
    F --> G[Uprav váhy - lr=0.01]
    G --> H{Další batch?}
    H -- ano --> C
    H -- ne --> I[Konec epochy - vyhodnoť přesnost]
    I --> J{Další epocha? max 20}
    J -- ano --> B
    J -- ne --> K[Ulož model do model.npz]
```

### Loss funkce — Cross-entropy

Měří jak moc se predikce liší od správné odpovědi:
```
loss = -log(pravděpodobnost správné třídy)
```
Čím větší chyba, tím větší číslo. Cíl je dostat loss co nejníže.

### Backpropagation

Algoritmus který projde síť pozpátku a spočítá o kolik je zodpovědná každá váha za chybu.
Pak každou váhu o trochu posune správným směrem (gradient descent).

```
nová_váha = stará_váha - learning_rate * gradient
```

---

## 5. Předpověď v reálném čase (app.py)

Každý tah štětcem spustí predikci:

```mermaid
flowchart LR
    A[Uživatel nakreslí tah] --> B[center_image]
    B --> C[Ořízni prázdné okraje]
    C --> D[Vystředí cifru do 200x200]
    D --> E[Zmenší na 28x28]
    E --> F[Normalizace 0-1]
    F --> G[Neuronová síť]
    G --> H[Zobraz výsledek]
```

**Proč center_image?**
MNIST dataset má cifry vždy vystředěné. Pokud bychom poslali obrázek bez centrování, síť by ho nepoznala protože by vypadal jinak než data na kterých se učila.

---

## 6. Kamera (camera.py)

```mermaid
flowchart TD
    A[Webcam frame] --> B[MediaPipe HandLandmarker]
    B --> C{Detekována ruka?}
    C -- ne --> A
    C -- ano --> D[Zjisti pozici prstu]
    D --> E[Průměr posledních 5 pozic - smoothing]
    E --> F{Index nahoru AND prostředník dolů?}
    F -- ano --> G[Kreslící mód - draw]
    F -- ne --> H[Pouze sledování - move]
```

**Jak se detekuje gesto:**
- index nahoru = špička prstu výše než druhý kloub
- prostředník dolů = špička níže než druhý kloub
- oboje splněno = kreslíme

**Smoothing:**
Pozice prstu se průměruje přes posledních 5 snímků aby se eliminovalo třesení ruky.

---

## 7. Jak se zobrazuje kamera v okně

Problém: canvas je černý, ale chceme vidět kameru i nakreslené tahy zároveň.

Řešení:
```
kamera frame (RGB)
    +
maska nakreslených tahů (grayscale)
    =
výsledný obraz s bílými tahy přes kameru
```

V kódu:
```python
colored_strokes = Image.new("RGB", ..., (255, 255, 255))  # bílý obraz
img.paste(colored_strokes, mask=self.image)               # nalep přes kameru tam kde jsou tahy
```

---

## 8. Přesnost sítě

| Dataset       | Počet obrázků | Přesnost |
|---------------|---------------|----------|
| Trénovací     | 60 000        | ~99%     |
| Testovací     | 10 000        | ~97%     |

Rozdíl mezi trénovací a testovací přesností je malý — síť se nepřeučila (no overfitting).

---

## 9. Struktura souborů

```
SketchMind/
├── src/
│   ├── network.py     neuronová síť (čistý NumPy)
│   ├── train.py       trénování na MNIST
│   ├── app.py         GUI aplikace (Tkinter)
│   └── camera.py      sledování prstu (MediaPipe)
├── tests/
│   ├── test_network.py    testy sítě
│   └── test_train.py      testy načítání dat
├── docs/
│   └── jak_to_funguje.md  tento dokument
├── model/             uložené váhy (ignorováno gitem)
├── data/              MNIST soubory (ignorováno gitem)
├── requirements.txt
└── README.md
```
