### Table 1 — Annotation-protocol asymmetry between the two classes

| Quantity | `weed` | `sugarcane` |
|---|---|---|
| Boxes | 998 | 384 |
| Images containing the class | 222 | 180 |
| Boxes per image (median) | 4.0 | 2.0 |
| Box area, % of frame (median) | 4.0 | 30.7 |
| Box area, % of frame (p90) | 21.7 | 94.3 |
| Frame coverage per image (median, %) | 33.6 | 80.3 |

Median box-area ratio (cane/weed): **7.8×**. Images carrying both classes: **180/222**.


### Table 2 — Union-mask IoU by prompt group (best prompt in each group)

Evaluation set: 222 images (180 with annotated sugarcane, 42 without). Threshold swept per run; 95 % image-level bootstrap CI.

| Group | Description | gdino-base | gdino-tiny | owlv2-base | owlv2-large | sam3 | sam3agent | yoloworld-s |
|---|---|---|---|---|---|---|---|---|
| **A** | abstract noun | 0.406 [0.373–0.438] | 0.402 [0.370–0.434] | 0.380 [0.351–0.412] | 0.380 [0.350–0.411] | 0.402 [0.369–0.434] | — | 0.014 [0.005–0.027] |
| **B** | morphological noun | 0.406 [0.373–0.438] | 0.403 [0.370–0.434] | 0.380 [0.351–0.412] | 0.380 [0.350–0.411] | 0.403 [0.370–0.435] | — | 0.241 [0.208–0.277] |
| **E** | long, non-relational | 0.403 [0.371–0.436] | 0.399 [0.368–0.431] | 0.380 [0.351–0.412] | 0.380 [0.350–0.411] | 0.402 [0.370–0.434] | — | 0.394 [0.361–0.427] |
| **C1** | relational, negated | 0.406 [0.373–0.439] | 0.404 [0.371–0.436] | 0.380 [0.351–0.412] | 0.380 [0.351–0.412] | 0.402 [0.370–0.434] | — | 0.386 [0.352–0.421] |
| **C2** | relational, affirmative | 0.406 [0.373–0.438] | 0.404 [0.372–0.437] | 0.380 [0.351–0.412] | 0.381 [0.351–0.412] | 0.404 [0.372–0.437] | — | 0.252 [0.223–0.288] |
| **D** | nominal control (sugarcane) | 0.637 [0.586–0.687] | 0.643 [0.594–0.690] | 0.541 [0.497–0.584] | 0.601 [0.557–0.641] | 0.624 [0.578–0.669] | — | 0.197 [0.144–0.252] |


### Table 3 — Best $F_1$ at IoU 0.50, against the measured label-noise ceiling

| Group | gdino-base | gdino-tiny | owlv2-base | owlv2-large | sam3 | sam3agent | yoloworld-s |
|---|---|---|---|---|---|---|---|
| **A** abstract noun | 0.170 [0.144–0.195] | 0.129 [0.108–0.153] | 0.150 [0.127–0.177] | 0.187 [0.163–0.212] | 0.176 [0.148–0.208] | — | 0.004 [0.000–0.011] |
| **B** morphological noun | 0.182 [0.156–0.210] | 0.119 [0.098–0.142] | 0.138 [0.117–0.163] | 0.155 [0.132–0.180] | 0.182 [0.153–0.213] | — | 0.047 [0.031–0.066] |
| **E** long, non-relational | 0.088 [0.071–0.107] | 0.078 [0.062–0.098] | 0.128 [0.110–0.149] | 0.126 [0.104–0.151] | 0.143 [0.120–0.172] | — | 0.071 [0.054–0.090] |
| **C1** relational, negated | 0.097 [0.081–0.117] | 0.086 [0.069–0.108] | 0.118 [0.098–0.142] | 0.155 [0.132–0.179] | 0.163 [0.135–0.193] | — | 0.068 [0.050–0.090] |
| **C2** relational, affirmative | 0.131 [0.110–0.156] | 0.103 [0.083–0.128] | 0.095 [0.075–0.117] | 0.113 [0.094–0.133] | 0.188 [0.159–0.217] | — | 0.042 [0.027–0.061] |
| **D** nominal control (sugarcane) | 0.341 [0.290–0.395] | 0.336 [0.285–0.391] | 0.286 [0.246–0.328] | 0.273 [0.236–0.314] | 0.052 [0.039–0.067] | — | 0.005 [0.000–0.016] |
| *expert–expert agreement* | 0.391 [0.295–0.466] | 0.391 [0.295–0.466] | 0.391 [0.295–0.466] | 0.391 [0.295–0.466] | 0.391 [0.295–0.466] | 0.391 [0.295–0.466] | 0.391 [0.295–0.466] |
| *simulated label-noise ceiling* | 0.399 [0.342–0.454] | 0.399 [0.342–0.454] | 0.399 [0.342–0.454] | 0.399 [0.342–0.454] | 0.399 [0.342–0.454] | 0.399 [0.342–0.454] | 0.399 [0.342–0.454] |


### Table 4 — Paired contrasts (union-mask IoU difference, 5 000 bootstrap replicates)

| Contrast | Isolates | gdino-base | gdino-tiny | owlv2-base | owlv2-large | sam3 | sam3agent | yoloworld-s |
|---|---|---|---|---|---|---|---|---|
| **C2-E** | relationality | +0.017 [+0.009, +0.026] \* | +0.012 [+0.003, +0.022] \* | -0.056 [-0.069, -0.045] \* | -0.033 [-0.046, -0.022] \* | +0.071 [+0.056, +0.087] \* | — | -0.031 [-0.043, -0.021] \* |
| **C1-C2** | negation | -0.013 [-0.019, -0.007] \* | -0.005 [-0.011, +0.001] | +0.023 [+0.017, +0.030] \* | +0.018 [+0.012, +0.025] \* | -0.037 [-0.050, -0.025] \* | — | +0.015 [+0.007, +0.022] \* |
| **E-A** | prompt length | -0.085 [-0.106, -0.064] \* | -0.054 [-0.070, -0.038] \* | -0.019 [-0.030, -0.008] \* | -0.053 [-0.067, -0.041] \* | -0.069 [-0.093, -0.047] \* | — | +0.062 [+0.047, +0.080] \* |
| **D-A** | category type | +0.008 [-0.003, +0.020] | +0.015 [+0.003, +0.028] \* | +0.045 [-0.002, +0.091] | +0.079 [+0.033, +0.122] \* | +0.037 [+0.022, +0.052] \* | — | +0.002 [-0.000, +0.004] |
| **B-A** | morphology | -0.031 [-0.044, -0.020] \* | -0.027 [-0.039, -0.014] \* | -0.043 [-0.055, -0.032] \* | -0.058 [-0.072, -0.045] \* | -0.034 [-0.047, -0.022] \* | — | +0.013 [+0.008, +0.019] \* |

\* 95 % CI excludes zero.


### Table 5 — False alarms on 869 expert-confirmed weed-free images

Threshold fixed at each run's $F_1$-optimal operating point on the annotated set.

| Model | Group | Prompt | Operating thr. | Images firing | Detections/image |
|---|---|---|---|---|---|
| gdino-base | A | `weed` | 0.14 | 100.0 % | 2.08 |
| gdino-base | A | `weeds` | 0.13 | 100.0 % | 2.55 |
| gdino-base | C2 | `plant growing between cane rows` | 0.09 | 100.0 % | 4.15 |
| gdino-base | C2 | `unwanted plant among sugarcane` | 0.21 | 100.0 % | 2.27 |
| gdino-base | C2 | `volunteer plant in a cane field` | 0.07 | 100.0 % | 3.48 |
| gdino-base | D | `sugarcane` | 0.32 | 100.0 % | 1.05 |
| gdino-base | D | `sugarcane leaf` | 0.36 | 99.7 % | 1.02 |
| gdino-tiny | A | `weed` | 0.07 | 100.0 % | 4.16 |
| gdino-tiny | A | `weeds` | 0.06 | 100.0 % | 5.90 |
| gdino-tiny | C2 | `plant growing between cane rows` | 0.11 | 100.0 % | 3.64 |
| gdino-tiny | C2 | `unwanted plant among sugarcane` | 0.16 | 100.0 % | 2.63 |
| gdino-tiny | C2 | `volunteer plant in a cane field` | 0.10 | 100.0 % | 3.04 |
| gdino-tiny | D | `sugarcane` | 0.41 | 99.4 % | 1.00 |
| gdino-tiny | D | `sugarcane leaf` | 0.51 | 93.8 % | 0.94 |
| owlv2-base | A | `weed` | 0.08 | 100.0 % | 2.79 |
| owlv2-base | A | `weeds` | 0.12 | 100.0 % | 2.59 |
| owlv2-base | C2 | `plant growing between cane rows` | 0.21 | 99.9 % | 4.62 |
| owlv2-base | C2 | `unwanted plant among sugarcane` | 0.11 | 99.9 % | 6.82 |
| owlv2-base | C2 | `volunteer plant in a cane field` | 0.12 | 100.0 % | 4.24 |
| owlv2-base | D | `sugarcane` | 0.12 | 77.1 % | 2.15 |
| owlv2-base | D | `sugarcane leaf` | 0.16 | 88.5 % | 5.30 |
| owlv2-large | A | `weed` | 0.12 | 100.0 % | 3.33 |
| owlv2-large | A | `weeds` | 0.19 | 100.0 % | 2.37 |
| owlv2-large | C2 | `plant growing between cane rows` | 0.24 | 100.0 % | 3.58 |
| owlv2-large | C2 | `unwanted plant among sugarcane` | 0.19 | 100.0 % | 7.43 |
| owlv2-large | C2 | `volunteer plant in a cane field` | 0.18 | 100.0 % | 5.99 |
| owlv2-large | D | `sugarcane` | 0.21 | 88.3 % | 2.79 |
| owlv2-large | D | `sugarcane leaf` | 0.18 | 100.0 % | 21.90 |
| sam3 | A | `weed` | 0.22 | 72.3 % | 1.44 |
| sam3 | A | `weeds` | 0.28 | 99.3 % | 1.71 |
| sam3 | C2 | `plant growing between cane rows` | 0.27 | 100.0 % | 1.56 |
| sam3 | C2 | `unwanted plant among sugarcane` | 0.19 | 100.0 % | 2.79 |
| sam3 | C2 | `volunteer plant in a cane field` | 0.14 | 99.9 % | 1.80 |
| sam3 | D | `sugarcane` | 0.02 | 74.8 % | 11.85 |
| sam3 | D | `sugarcane leaf` | 0.18 | 96.2 % | 74.20 |
| yoloworld-s | A | `weed` | 0.02 | 0.3 % | 0.00 |
| yoloworld-s | A | `weeds` | 0.02 | 34.6 % | 0.35 |
| yoloworld-s | C2 | `plant growing between cane rows` | 0.02 | 99.2 % | 1.68 |
| yoloworld-s | C2 | `unwanted plant among sugarcane` | 0.02 | 99.8 % | 1.92 |
| yoloworld-s | C2 | `volunteer plant in a cane field` | 0.02 | 78.6 % | 0.84 |
| yoloworld-s | D | `sugarcane` | 0.02 | 5.6 % | 0.06 |
| yoloworld-s | D | `sugarcane leaf` | 0.02 | 16.7 % | 0.18 |


### Table 6 — Best open-vocabulary configuration vs supervised RTMDet

Clean test split only (n = 54 images; minimum detectable paired difference 8.3 AP points).

Metric key in `vs_supervised.json`: `microF1@IoU0.50`. This is the same run as the best `sam3` cell in Table 3, scored on a different subset: Table 3 reports the 222-image set (0.188), this table the 54-image Clean test split (0.194), which is the only split on which the supervised comparison is fair.

| System | Micro-$F_1$ @ IoU 0.50 |
|---|---|
| Best OV: `sam3|C2|unwanted plant among sugarcane` | 0.194 |
| Supervised RTMDet (AP50 36.2 % [28.4–42.6]) | 0.417 |
| **Paired difference** | **-0.223 [-0.302, -0.147]** — distinguishable |
