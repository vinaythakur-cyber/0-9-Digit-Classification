# 0–9 Digit Classification — with an 11th class for "this isn't a digit"

A PyTorch CNN that classifies handwritten digits **and knows when to refuse.**

Standard MNIST models have 10 output neurons. Show one a photograph of the night
sky and it will tell you, with 99% confidence, that it is an 8. This project adds
an 11th neuron — `unclassifiable` — and, more importantly, the training data and
rejection logic that make that neuron actually work.

![negative families](reports/figures/negative_families.png)

*Row 1 is class 0–9. Every other row is class 10.*

---

## The demo

`make serve` → a drawing canvas at `localhost:8000`. Draw a digit, or press a
button to feed it something that isn't one.

| A drawn digit | A star field |
|---|---|
| ![digit](reports/figures/app-digit.png) | ![star field](reports/figures/app-starfield.png) |

Both panels show all eleven probabilities, because the point of the project is
visible there: softmax forces those numbers to sum to 100%, so a ten-output
model has nowhere to put the mass when the answer is "none of these".

---

## The problem: softmax cannot say "none of the above"

This is not a bug you can fix by adding a neuron and hoping. It is structural.

A softmax layer computes

```
P(class i) = exp(z_i) / Σ_j exp(z_j)
```

The denominator sums over **only the classes you defined**, so the outputs are
forced to sum to 1. The layer answers *"which of these is most likely?"* — it is
mathematically incapable of answering *"is it any of them?"*. Feed it pure noise
and it still elects a winner. ([Nguyen et al., 2015](https://arxiv.org/abs/1412.1897)
showed nets giving >99% confidence to images that look like television static.)

So the obvious fix fails:

> **Adding an 11th output to a model trained only on MNIST does nothing.**
> That neuron sees zero training examples, its weights stay near their random
> initialisation, and it never wins an argmax. The star field still comes back
> as an 8.

The 11th class only becomes real when you give it data. That is most of what
this repository is.

---

## Two independent defenses

```
                    ┌──────────────────────────────────┐
   any image  ────► │  preprocess                      │
   (canvas,         │  → grayscale, auto-fix polarity  │
    upload,         │  → 20×20 box, centred by mass    │
    100×100 …)      │  → MNIST normalisation           │
                    └────────────────┬─────────────────┘
                                     ▼
                    ┌──────────────────────────────────┐
                    │  CNN → 11 logits                 │
                    │  [0, 1, … 9, unclassifiable]     │
                    └────────────────┬─────────────────┘
                                     ▼
       ┌─────────────────────────────┴─────────────────────────────┐
       │  DEFENSE 1 — the learned 11th class                       │
       │  argmax == 10 → reject.                                   │
       │  Strong on anything resembling its training negatives.    │
       │  Blind, by construction, to families it never saw.        │
       ├───────────────────────────────────────────────────────────┤
       │  DEFENSE 2 — energy score over the digit logits           │
       │  E(x) = −logsumexp(z_0 … z_9);  reject if E(x) > τ        │
       │  Unnormalised, so unlike softmax it CAN express           │
       │  "no digit fits well". Catches novel junk.                │
       └───────────────────────────────────────────────────────────┘
                          accept only if both pass
```

**Why energy and not just softmax confidence?** Softmax is scale-invariant:
logits of `[1, 0, 0]` and `[100, 0, 0]` give wildly different confidences but
`[10, 9, 9]` and `[1, 0, 0]` give the *same* one, despite the first having far
more absolute evidence. `logsumexp` is a smooth maximum over the raw logits, so
it measures how much digit evidence exists in absolute terms — exactly the
question softmax normalises away. ([Liu et al., 2020](https://arxiv.org/abs/2010.03759))

The sum runs over the **10 digit logits only**. Including the unknown logit
would let a confidently-unknown input score low energy and destroy the signal.

---

## What the 11th class is trained on

No EMNIST download, no external OOD dataset. Every negative is generated from a
seed, so the whole pipeline is reproducible offline and CI runs it in seconds.

| Family | What it is | Why it's needed |
|---|---|---|
| `star_field` | sparse bright points, blur, occasional diffraction spikes | mostly-black like a digit's border; the case this project was built for |
| `glyph` | A–Z, a–z, Greek, Cyrillic, symbols rendered from 39 fonts, then sheared, rotated and elastically warped | **the hard negatives** — near-identical stroke statistics to digits |
| `patch_shuffle` | a real MNIST digit cut into tiles and shuffled | *identical* pixel histogram and ink coverage to a digit; only layout differs, so no low-level statistic can separate it |
| `scribble` | random multi-point strokes at digit-like stroke width | what users actually draw to break a demo |
| `shapes` | circles, boxes, triangles, crosses, arcs | a circle is genuinely close to a 0; sharpens the boundary |
| `smooth_blobs` | low-frequency noise | stops the shortcut "high spatial frequency ⇒ reject" |
| `gaussian_noise`, `uniform_noise`, `salt_pepper` | classic fooling images | the canonical failure case |
| `texture` | stripes, checkerboards, gradients | strong periodic structure |
| `near_blank` | empty or near-empty canvas | the single most common real input |

Two details that matter more than they look:

- **Glyphs go through the identical framing pipeline as digits** — same 20×20
  box, same centre-of-mass alignment. The model cannot separate them by size or
  position, only by shape.
- **Digit-lookalike characters are deliberately excluded** (`O`, `o`, `I`, `l`,
  `S`, `s`, `Z`, `z`). Labelling a capital O "unclassifiable" while 0 is a digit
  asks the model to learn a distinction that does not exist in the pixels — it
  can only get noisier on real zeros.

Training negatives are **regenerated every epoch** (seeded by `(seed, epoch,
index)`), so the model effectively never sees the same negative twice and cannot
memorise a fixed set. Validation and test negatives are fixed, so metrics are
comparable between runs.

---

## Results

Trained for 20 epochs on CPU (~35 minutes). Thresholds calibrated on the
validation split; every number below is measured on the **held-out test split**
that calibration never touched.

| | |
|---|---|
| Digit accuracy, on inputs it accepted | **99.86%** |
| Coverage — real digits it accepted | **98.96%** |
| False-reject rate — real digits wrongly refused | **1.04%** |
| OOD recall — non-digits correctly refused | **99.48%** |
| AUROC, P(unclassifiable) | 0.9997 |
| AUROC, digit energy | 0.9997 |
| FPR @ 95% TPR (energy) | 0.0008 |

288,299 parameters. 28×28 grayscale in, 11 logits out. A prediction takes a few
milliseconds on a CPU.

End to end through the HTTP API, feeding it digits rendered at canvas
resolution: **15/15 classified correctly**, and every one of the seven negative
families refused. Energy separates cleanly — real digits land at −3.0 to −4.2,
non-digits at −1.7 to −1.8, with the calibrated threshold at −2.36 between them.

### Rejection rate by kind of input

![rejection by family](reports/figures/family_rejection.png)

| family | rejected | | family | rejected |
|---|---|---|---|---|
| `star_field` | 100.0% | | `glyph` (letters) | 99.3% |
| `gaussian_noise` | 100.0% | | `patch_shuffle` | 99.4% |
| `scribble` | 98.8% | | `shapes` | 99.3% |
| `near_blank` | 100.0% | | **`fashion_mnist`** | **97.1%** |

**The row that matters is the last one.** Fashion-MNIST is never used in
training — the model has seen noise, letters and shuffled digits, but never a
shoe. It rejects 97% of them anyway. That is the difference between
learning "not a digit" as a concept and memorising my generators.

It is also, correctly, the **worst** row in the table. A genuinely novel family
should be harder than the ones trained against, and any report where it isn't
deserves suspicion.

### How the two scores separate

![score separation](reports/figures/score_separation.png)

Note the log scale — the overlap is a handful of samples out of thousands.

### Training

![training curves](reports/figures/training_curves.png)

The middle panel is the one to watch. The one-cycle learning rate makes the
model swing between over- and under-rejecting while the rate is high, and only
settles as it anneals. Selecting on accuracy alone would happily have picked one
of those trigger-happy checkpoints — negatives are a third of the data, so a
model that refuses everything scores well. Selecting on
`digit_accuracy + unknown_recall` rejects them.

This is also why the epoch count is 20 rather than 12: once blur augmentation
and framing were added the task got harder, and at 12 epochs the schedule had
not finished settling.

### Confusion matrix

![confusion matrix](reports/figures/confusion_matrix.png)

## Quickstart

```bash
git clone https://github.com/vinaythakur-cyber/0-9-Digit-Classification
cd 0-9-Digit-Classification

pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt

make serve          # http://localhost:8000 — trained weights are committed
```

The web app opens a drawing canvas. Draw a digit, or press one of the sample
buttons to feed it a star field, a letter or pure noise and watch it refuse.

```bash
make test           # 35 tests, no downloads, ~0.3 s
make train          # retrain from scratch (~20 min on CPU, downloads MNIST)
make evaluate       # recalibrate thresholds, regenerate figures + metrics.json
make docker         # build and run the container
```

Classify files from the terminal:

```bash
python scripts/predict_cli.py my_digit.png night_sky.jpg
```

---

## Project structure

```
src/digitood/
├── constants.py            label space in one place — 11 classes defined once
├── config.py               typed config; a typo in the YAML fails at load time
├── imageops.py             the MNIST framing transform, shared by train + serve
├── data/
│   ├── negatives.py        procedural OOD generators (star field, noise, …)
│   ├── glyphs.py           hard negatives rendered from fonts
│   └── datamodule.py       stitches positives + negatives into train/val/test
├── models/cnn.py           the 11-way CNN
├── training/
│   ├── engine.py           train/eval loops, four separate metrics
│   └── train.py            entrypoint; saves weights + config + thresholds
├── inference/
│   ├── preprocess.py       polarity, transparency, framing — any image in
│   └── predictor.py        both defenses, one predict() call
└── evaluation/
    ├── metrics.py          AUROC / FPR@95TPR from first principles
    └── evaluate.py         calibrate on val, measure on test, write the report

app/                        FastAPI service + canvas UI
tests/                      35 tests, fully offline
```

**Why `imageops.py` is its own module.** MNIST digits are size-normalised into a
20×20 box and centred by *centre of mass*, not geometric centre. If training
data goes through that transform and a live canvas drawing does not, the model
sees a distribution it never trained on — accuracy collapses, and it looks
exactly like a broken model even though the weights are fine. One transform,
imported by both sides, makes that class of bug impossible.

---

## API

```bash
curl -X POST localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"image": "data:image/png;base64,iVBOR..."}'
```

```json
{
  "label": "7",
  "digit": 7,
  "confidence": 0.998,
  "is_digit": true,
  "reason": "accepted by both the 11-class head and the confidence/energy check",
  "unknown_probability": 0.0004,
  "energy": -9.42,
  "probabilities": { "0": 0.0001, "...": "...", "unclassifiable": 0.0004 }
}
```

| endpoint | purpose |
|---|---|
| `GET /api/health` | model status, parameter count, calibrated thresholds |
| `POST /api/predict` | classify a base64 data URL from the canvas |
| `POST /api/predict-file` | classify an uploaded image file |
| `GET /api/sample/{kind}` | generate an example negative, for demos |

---

## Design decisions worth defending

**Global average pooling instead of flatten.** A flatten layer lets the network
learn "bright pixel at (3,4) ⇒ class 7" — position memorisation, which is
exactly what makes nets confident on noise. GAP forces the decision to come from
*which features fired*, not where.

**Model selection on `digit_accuracy + unknown_recall`, not accuracy.** With
negatives at 33% of the data, a model that rejects everything scores 33%
"accuracy" while being useless. Optimising the sum forces both halves of the job.

**Class-weighted loss.** The unknown class holds ~33% of samples while each digit
holds ~6.7%. Untouched, cross-entropy finds it profitable to reject aggressively.
Inverse-frequency weighting removes that incentive.

**Training data goes through the same framing transform as served input.**
This one cost me three retrains and is the most useful thing in the repo.
`fit_to_mnist_frame` crops to the ink, rescales, and re-centres — and it is
*not* perfectly idempotent, because it resamples. Training on raw MNIST while
re-framing every served input meant the model was fitted to one distribution
and queried on another. Acceptance of genuine digits on the serving path sat
at **45%** while the offline test reported a **0.77%** false-reject rate and
looked flawless, because the offline test only ever saw raw arrays. Framing
both sides makes the two identical by construction.

**`ink_bbox` thresholds relative to the image's own maximum, at 0.30.**
The partner bug. A browser canvas antialiases its strokes, and that soft edge
sits well above an absolute 0.05 cut — so the measured bounding box came out
~6% too large per side and framing scaled the digit smaller than anything the
model had seen. Relative, because a lightly-drawn digit peaking at 0.5 would
be eaten by an absolute 0.30.

**Thresholds calibrated on validation, never test.** Tuning a threshold on the
test set and then reporting test numbers is the standard way a model report ends
up flattering a model that disappoints in the demo. The two splits also use
different RNG seeds, so they don't even share generated images.

**Calibrated to a false-reject budget, not to maximise recall.** We fix the
tolerable rate of wrongly rejecting real digits at 1% and take whatever OOD
recall that allows. Rejecting a real digit is a visible, user-facing failure;
accepting one odd star field is not.

---

## Limitations — honestly

- **`b` vs `6`, `q` vs `9`.** Some rendered glyphs are genuinely ambiguous to a
  human too. Training them as negatives puts mild pressure on real 6s and 9s.
  Keeping them makes the model stricter; dropping them would make it more
  permissive. The current build keeps them.
- **Rejection generalises, but not perfectly.** Fashion-MNIST is the honest test
  (see Results) — it is never trained against. A genuinely novel OOD family will
  always do worse than the trained-against families.
- **Adversarial inputs are out of scope.** Nothing here defends against an
  attacker deliberately optimising an image to be misclassified.
- **28×28 grayscale only.** Larger or colour inputs are downsampled; fine detail
  is lost by design, since that is the distribution MNIST defines.

### If I took this further

- Outlier Exposure with a large natural-image corpus (Tiny-ImageNet) as
  additional negatives
- Temperature scaling / ODIN input preprocessing for sharper energy separation
- A proper open-set method (OpenMax) as a third defense, for comparison
- ONNX export so the model runs client-side in the browser with no server

---

## References

- Nguyen, Yosinski, Clune (2015), [*Deep Neural Networks are Easily Fooled*](https://arxiv.org/abs/1412.1897)
- Hendrycks & Gimpel (2017), [*A Baseline for Detecting Misclassified and Out-of-Distribution Examples*](https://arxiv.org/abs/1610.02136)
- Liu et al. (2020), [*Energy-based Out-of-distribution Detection*](https://arxiv.org/abs/2010.03759)
- Hendrycks, Mazeika, Dietterich (2019), [*Deep Anomaly Detection with Outlier Exposure*](https://arxiv.org/abs/1812.04606)

## License

MIT — see [LICENSE](LICENSE).
