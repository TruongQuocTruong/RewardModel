Table 1. Comparison of dataset statistics between AURA and existing benchmarks. AURA-Benchmark contains 10636 instructions over 677 images (15.7 per image). AURA-Test provides 1561 human-labelled pairs over 387 of these instructions. Both are evaluated along 6 metrics, the most among the compared benchmarks (tied with ICE-Bench). Editing objects are larger (median size 29.8% and 26.4%) than in HATIE (6.6%) and CompBench (3.1%). [Keys: (min-max) values, Med. = median]

| Benchmark | #Images | #Instructions | Avg. #Words | Med. #Pixels (K) | Editing Object: #Categories | Editing Object: Med. Size (%) | #Metrics |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ICE-Bench | 5139 | 5139 | 27.6 (3-233) | 1023 (38-1049) | - | - | 6 |
| ImgEdit | 221 | 787 | 12.0 (3-44) | 1049 (786-8294) | - | - | 3 |
| GEdit-Bench | 606 | 606 | 7.6 (2-26) | 270 (245-279) | - | - | 2 |
| MagicBrush | 1053 | 1053 | 6.6 (2-16) | 250 (250-1049) | 80 | - | 5$^{*}$ |
| HATIE | 18226 | 49840 | 7.0 (4-13) | 262 | 76 | 6.6 (0.5-92) | 5 |
| ImagenWorld | 611 | 611 | 16.3 (4-51) | 620 (37-14746) | - | - | 4 |
| Omni IIE Bench | 1240 | 1723 | 23.72 (7-105) | 284 (38-12193) | - | - | 3 |
| CompBench | 2801 | 2905 | 9.97 (1-33) | 410 | - | 3.1 (0.1-58) | 2 |
| **AURA-Benchmark** | 677 | 10636 | 11.2 (3-34) | 786 (352-4194) | 143 | 29.8 (0.2-100) | 6 |
| **AURA-Test** | 197 | 387 | 11.4 (6-28) | 786 (426-4194) | 77 | 26.4 (0.2-99) | 6 |

LaTeX rows:

```latex
ICE-Bench~\cite{pan2025ice} & 5139 & 5139 & 27.6 (3-233) & 1023 (38-1049) & - & - & 6 \\
ImgEdit~\cite{ye2025imgedit} & 221 & 787 & 12.0 (3-44) & 1049 (786-8294) & - & - & 3 \\
GEdit-Bench~\cite{liu2025step1x} & 606 & 606 & 7.6 (2-26) & 270 (245-279) & - & - & 2 \\
MagicBrush~\cite{zhang2023magicbrush} & 1053 & 1053 & 6.6 (2-16) & 250 (250-1049) & 80 & - & 5$^{*}$ \\
HATIE~\cite{ryu2025towards} & 18226 & 49840 & 7.0 (4-13) & 262 & 76 & 6.6 (0.5-92) & 5 \\
ImagenWorld~\cite{mahdizadeh2026imagenworld} & 611 & 611 & 16.3 (4-51) & 620 (37-14746) & - & - & 4 \\
Omni IIE Bench~\cite{yang2026omni} & 1240 & 1723 & 23.72 (7-105) & 284 (38-12193) & - & - & 3 \\
CompBench~\cite{jia2026compbench} & 2801 & 2905 & 9.97 (1-33) & 410 & - & 3.1 (0.1-58) & 2 \\
\benchmarkName & 677 & 10636 & 11.2 (3-34) & 786 (352-4194) & 143 & 29.8 (0.2-100) & 6 \\
\testName & 197 & 387 & 11.4 (6-28) & 786 (426-4194) & 77 & 26.4 (0.2-99) & 6 \\
```
