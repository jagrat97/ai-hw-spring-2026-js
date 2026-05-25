# MNIST Adversarial Attack Results

**Clean recognition rate (test set, 10000 samples):** 0.9899

ASR = fraction of originally-correctly-classified samples flipped to a wrong label.

| Attack | epsilon | Accuracy on adv. examples | ASR |
| --- | --- | --- | --- |
| FGSM | 0.1 | 0.9062 | 0.0846 |
| FGSM | 0.2 | 0.5722 | 0.4220 |
| FGSM | 0.3 | 0.2094 | 0.7885 |
| I-FGSM | 0.1 | 0.8557 | 0.1356 |
| I-FGSM | 0.2 | 0.1307 | 0.8680 |
| I-FGSM | 0.3 | 0.0005 | 0.9995 |
| MI-FGSM | 0.1 | 0.8765 | 0.1146 |
| MI-FGSM | 0.2 | 0.2931 | 0.7039 |
| MI-FGSM | 0.3 | 0.0088 | 0.9911 |
