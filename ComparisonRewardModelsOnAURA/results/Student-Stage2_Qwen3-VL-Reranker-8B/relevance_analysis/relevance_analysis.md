### Relevance of the column's own metric (threshold 0.5)

| Column | Pairs | Mean r_own | % r_own >= threshold | % own metric most relevant (ties count) |
|---|---|---|---|---|
| Aesthetic | 262 | 1.000 | 100.00 | 35.50 |
| Realism | 253 | 0.878 | 96.05 | 0.00 |
| Fidelity | 279 | 1.000 | 100.00 | 54.84 |
| BC | 254 | 0.838 | 89.37 | 0.00 |
| FC | 266 | 0.785 | 99.62 | 0.00 |
| SC | 247 | 0.434 | 39.68 | 0.00 |

### Head of the column's metric, pairs split by r_own (Acc \| Margin AUC on non-tie pairs, Tie Head AUC on all pairs; %)

| Column | Group | Pairs (non-tie / tie) | Acc | Margin AUC | Tie Head AUC |
|---|---|---|---|---|---|
| Aesthetic | all | 262 (209 / 53) | 58.37 | 60.21 | 68.11 |
| Aesthetic | >= 0.5 | 262 (209 / 53) | 58.37 | 60.21 | 68.11 |
| Aesthetic | < 0.5 | 0 (0 / 0) | - | - | - |
| Realism | all | 253 (185 / 68) | 58.92 | 66.13 | 53.47 |
| Realism | >= 0.5 | 243 (177 / 66) | 59.32 | 67.05 | 53.86 |
| Realism | < 0.5 | 10 (8 / 2) | 50.00 | 46.88 | 37.50 |
| Fidelity | all | 279 (239 / 40) | 60.67 | 67.84 | 76.59 |
| Fidelity | >= 0.5 | 279 (239 / 40) | 60.67 | 67.84 | 76.59 |
| Fidelity | < 0.5 | 0 (0 / 0) | - | - | - |
| BC | all | 254 (217 / 37) | 59.91 | 65.03 | 65.10 |
| BC | >= 0.5 | 227 (191 / 36) | 58.64 | 63.98 | 63.19 |
| BC | < 0.5 | 27 (26 / 1) | 69.23 | 70.71 | 96.15 |
| FC | all | 266 (235 / 31) | 61.28 | 67.60 | 63.87 |
| FC | >= 0.5 | 265 (234 / 31) | 61.11 | 67.54 | 63.91 |
| FC | < 0.5 | 1 (1 / 0) | 100.00 | 100.00 | - |
| SC | all | 247 (180 / 67) | 57.78 | 64.85 | 55.90 |
| SC | >= 0.5 | 98 (77 / 21) | 63.64 | 76.12 | 54.79 |
| SC | < 0.5 | 149 (103 / 46) | 53.40 | 56.02 | 54.98 |

### Mean relevance of every Student head per column

| Column | aesthetic | fidelity | realism | background_consistency | foreground_consistency | structure_consistency |
|---|---|---|---|---|---|---|
| Aesthetic | 1.000 | 1.000 | 0.809 | 0.725 | 0.766 | 0.308 |
| Realism | 1.000 | 1.000 | 0.878 | 0.785 | 0.734 | 0.236 |
| Fidelity | 1.000 | 1.000 | 0.797 | 0.728 | 0.754 | 0.312 |
| BC | 1.000 | 1.000 | 0.840 | 0.838 | 0.660 | 0.139 |
| FC | 1.000 | 1.000 | 0.858 | 0.788 | 0.785 | 0.252 |
| SC | 1.000 | 1.000 | 0.726 | 0.638 | 0.840 | 0.434 |
