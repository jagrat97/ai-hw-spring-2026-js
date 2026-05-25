# ai-hw-spring-2026-js

Adversarial attacks on a CNN classifier trained on MNIST.

## Problem

Train a model on MNIST, then attack it with three gradient-based adversarial methods and
measure the attack success rate (ASR).

- **Dataset:** MNIST (loaded via `torchvision.datasets.MNIST`, train split for training,
  test split for evaluation and attacks).
- **Model:** small CNN (2 conv + 2 FC, dropout), trained 5 epochs with Adam.
- **Attacks (untargeted, L∞):**
  - **FGSM** — Fast Gradient Sign Method (Goodfellow et al., 2014), 1 step.
  - **I-FGSM / PGD** — Iterative FGSM with L∞ projection, 10 iterations.
  - **MI-FGSM** — Momentum I-FGSM (Dong et al., 2018), 10 iterations, μ=1.0.

## Layout

```
model.py        # CNN definition
train.py        # trains model, saves model.pt
attacks.py      # FGSM, I-FGSM, MI-FGSM implementations
test.py         # clean eval + attack eval, writes results/
model.pt        # trained weights (committed so you can skip training)
results/
  results.md          # accuracy + ASR table
  samples_fgsm.png    # original/adversarial pairs, one per digit class
  samples_ifgsm.png
  samples_mifgsm.png
```

## How to run

```bash
pip install -r requirements.txt
python train.py    # ~2 min on CPU, saves model.pt
python test.py     # ~1 min on CPU, writes results/
```

## Results

**Clean recognition rate (10 000 test samples): 0.9899**

ASR = fraction of originally-correctly-classified samples flipped to a wrong label by
the attack.

| Attack   | ε   | Accuracy on adv. examples | ASR    |
| -------- | --- | -------------------------- | ------ |
| FGSM     | 0.1 | 0.9062                     | 0.0846 |
| FGSM     | 0.2 | 0.5722                     | 0.4220 |
| FGSM     | 0.3 | 0.2094                     | 0.7885 |
| I-FGSM   | 0.1 | 0.8557                     | 0.1356 |
| I-FGSM   | 0.2 | 0.1307                     | 0.8680 |
| I-FGSM   | 0.3 | 0.0005                     | 0.9995 |
| MI-FGSM  | 0.1 | 0.8765                     | 0.1146 |
| MI-FGSM  | 0.2 | 0.2931                     | 0.7039 |
| MI-FGSM  | 0.3 | 0.0088                     | 0.9911 |

Observations:

- ASR increases monotonically with ε for all three attacks, as expected.
- Iterative attacks (I-FGSM, MI-FGSM) dominate single-step FGSM at every ε. At ε=0.3,
  I-FGSM drives accuracy essentially to zero (0.05%).
- MI-FGSM is comparable to I-FGSM here; its strength is transferability across models
  (not measured in this single-model setup) rather than higher ASR on the same model.

See `results/samples_*.png` for original vs adversarial digit pairs at ε=0.2 — the
adversarial digits are still clearly recognizable to a human, but the model
misclassifies them.
