### Relevance of the column's own metric (threshold 0.5)

| Column | Pairs | Mean r_own | % r_own >= threshold | % own metric most relevant (ties count) |
|---|---|---|---|---|
| Aesthetic | 262 | 1.000 | 100.00 | 77.86 |
| Realism | 253 | 0.899 | 99.21 | 0.00 |
| Fidelity | 279 | 1.000 | 100.00 | 21.86 |
| BC | 254 | 0.800 | 89.37 | 0.00 |
| FC | 266 | 0.689 | 100.00 | 0.00 |
| SC | 247 | 0.315 | 24.29 | 0.00 |

### Head of the column's metric, pairs split by r_own (Acc \| Margin AUC on non-tie pairs, Tie Head AUC on all pairs; %)

| Column | Group | Pairs (non-tie / tie) | Acc | Margin AUC | Tie Head AUC |
|---|---|---|---|---|---|
| Aesthetic | all | 262 (209 / 53) | 54.55 | 58.34 | 71.23 |
| Aesthetic | >= 0.5 | 262 (209 / 53) | 54.55 | 58.34 | 71.23 |
| Aesthetic | < 0.5 | 0 (0 / 0) | - | - | - |
| Realism | all | 253 (185 / 68) | 62.70 | 65.91 | 59.13 |
| Realism | >= 0.5 | 251 (183 / 68) | 62.30 | 65.63 | 59.15 |
| Realism | < 0.5 | 2 (2 / 0) | 100.00 | 100.00 | - |
| Fidelity | all | 279 (239 / 40) | 57.32 | 63.32 | 73.35 |
| Fidelity | >= 0.5 | 279 (239 / 40) | 57.32 | 63.32 | 73.35 |
| Fidelity | < 0.5 | 0 (0 / 0) | - | - | - |
| BC | all | 254 (217 / 37) | 58.06 | 63.77 | 68.09 |
| BC | >= 0.5 | 227 (192 / 35) | 57.29 | 63.39 | 68.02 |
| BC | < 0.5 | 27 (25 / 2) | 64.00 | 65.28 | 64.00 |
| FC | all | 266 (235 / 31) | 62.55 | 71.12 | 62.48 |
| FC | >= 0.5 | 266 (235 / 31) | 62.55 | 71.12 | 62.48 |
| FC | < 0.5 | 0 (0 / 0) | - | - | - |
| SC | all | 247 (180 / 67) | 68.89 | 72.33 | 54.32 |
| SC | >= 0.5 | 60 (48 / 12) | 62.50 | 71.61 | 58.68 |
| SC | < 0.5 | 187 (132 / 55) | 71.21 | 72.45 | 53.44 |

### Mean relevance of every Student head per column

| Column | aesthetic | fidelity | realism | background_consistency | foreground_consistency | structure_consistency |
|---|---|---|---|---|---|---|
| Aesthetic | 1.000 | 1.000 | 0.849 | 0.740 | 0.689 | 0.252 |
| Realism | 1.000 | 1.000 | 0.899 | 0.791 | 0.687 | 0.213 |
| Fidelity | 1.000 | 1.000 | 0.849 | 0.737 | 0.692 | 0.256 |
| BC | 1.000 | 1.000 | 0.885 | 0.800 | 0.671 | 0.185 |
| FC | 1.000 | 1.000 | 0.893 | 0.795 | 0.689 | 0.209 |
| SC | 1.000 | 1.000 | 0.802 | 0.678 | 0.711 | 0.315 |
