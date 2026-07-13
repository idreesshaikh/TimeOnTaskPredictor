# VLM train card — Qwen3-VL-4B QLoRA SFT (Path A)

_Generated 2026-07-13T21:36:10+00:00 · config `configs/sweeps/lam050.yaml` · seed 42 · mode **FULL**_

- Model: `unsloth/Qwen3-VL-4B-Instruct` · 4-bit QLoRA · vision layers LoRA-tuned · LoRA r=16 α=16 dropout 0.05 on language attn+MLP + vision
- Target: `dwell_seconds: X.X` (winsor cap 24.954 s, train-split p95 — see artifacts/dataset_card.md)
- Task title in prompt: **False** (False = screen only, True = screen+task)
- Features (train-time only): targets blended toward the privileged-feature teacher (λ=0.5) — **87921/87921** rows covered (100.0%), mean |shift| 2.192 s; val targets untouched, inference stays screenshot-only
- Scaffold (v3): target emits the visible axTree stats as a `ui:` line before the dwell — supervised coverage train 100.0% · val 100.0%; at inference the model generates its own estimates (input stays image-only)
- Examples: train **87921** · val **4269** (img_resolved only)

## Memory footprint

- per-device batch **4** × grad-accum **4** = effective **16**
- image bounds: max_side 1024, pixels [100352, 602112] (≤768 visual tokens) · max_seq_length 1536
- peak VRAM: **6.62 GB** allocated / 6.83 GB reserved of 79.25 GB (NVIDIA A100-SXM4-80GB)

## Loss

- train loss: first 3.8456 → last 0.1423

| eval pass | epoch | step | val loss | parse rate | MAE (s) | MAE (log) |
|---|---|---|---|---|---|---|
| 0 | 0.25 | 1374 | 0.32209521532058716 | 100.00% | 3.42 | 0.479 |
| 1 | 0.5 | 2748 | 0.337056964635849 | 100.00% | 3.74 | 0.507 |
| 2 | 0.75 | 4122 | 0.3444441854953766 | 100.00% | 4.57 | 0.633 |
| 3 | 1.0 | 5496 | 0.35548219084739685 | 100.00% | 3.22 | 0.462 |
| 4 | 1.25 | 6870 | 0.3794596195220947 | 100.00% | 2.98 | 0.425 |
| 5 | 1.5 | 8244 | 0.37560734152793884 | 100.00% | 3.62 | 0.536 |
| 6 | 1.75 | 9618 | 0.3776862323284149 | 100.00% | 3.31 | 0.473 |
| 7 | 2.0 | 10992 | 0.3796551823616028 | 100.00% | 3.67 | 0.517 |

## Sample decodes (last eval pass)

- gold 3.6 s → `ui: nodes=1154 interactive=138 links=100 buttons=32 inputs=6 text=115
dwell_seconds: 3.1`
- gold 2.9 s → `ui: nodes=1488 interactive=173 links=117 buttons=48 inputs=8 text=6772
dwell_seconds: 4.6`
- gold 3.9 s → `ui: nodes=481 interactive=69 links=40 buttons=28 inputs=1 text=1049
dwell_seconds: 5.0`

Checkpoints: `artifacts/sweeps/vlm_lam050` (per epoch; final adapters in `artifacts/sweeps/vlm_lam050/final`).

## Full log (JSON)

```json
{
  "train_losses": [
    {
      "step": 5,
      "loss": 3.8456
    },
    {
      "step": 10,
      "loss": 3.8126
    },
    {
      "step": 15,
      "loss": 3.7638
    },
    {
      "step": 20,
      "loss": 3.6389
    },
    {
      "step": 25,
      "loss": 3.5087
    },
    {
      "step": 30,
      "loss": 3.3116
    },
    {
      "step": 35,
      "loss": 3.1178
    },
    {
      "step": 40,
      "loss": 2.8662
    },
    {
      "step": 45,
      "loss": 2.5807
    },
    {
      "step": 50,
      "loss": 2.3391
    },
    {
      "step": 55,
      "loss": 2.0936
    },
    {
      "step": 60,
      "loss": 1.8666
    },
    {
      "step": 65,
      "loss": 1.6099
    },
    {
      "step": 70,
      "loss": 1.301
    },
    {
      "step": 75,
      "loss": 0.992
    },
    {
      "step": 80,
      "loss": 0.6722
    },
    {
      "step": 85,
      "loss": 0.4696
    },
    {
      "step": 90,
      "loss": 0.4175
    },
    {
      "step": 95,
      "loss": 0.398
    },
    {
      "step": 100,
      "loss": 0.3631
    },
    {
      "step": 105,
      "loss": 0.35
    },
    {
      "step": 110,
      "loss": 0.3558
    },
    {
      "step": 115,
      "loss": 0.3421
    },
    {
      "step": 120,
      "loss": 0.3411
    },
    {
      "step": 125,
      "loss": 0.3343
    },
    {
      "step": 130,
      "loss": 0.3343
    },
    {
      "step": 135,
      "loss": 0.3157
    },
    {
      "step": 140,
      "loss": 0.3064
    },
    {
      "step": 145,
      "loss": 0.2999
    },
    {
      "step": 150,
      "loss": 0.2992
    },
    {
      "step": 155,
      "loss": 0.2973
    },
    {
      "step": 160,
      "loss": 0.3099
    },
    {
      "step": 165,
      "loss": 0.295
    },
    {
      "step": 170,
      "loss": 0.3026
    },
    {
      "step": 175,
      "loss": 0.294
    },
    {
      "step": 180,
      "loss": 0.3011
    },
    {
      "step": 185,
      "loss": 0.2982
    },
    {
      "step": 190,
      "loss": 0.294
    },
    {
      "step": 195,
      "loss": 0.2892
    },
    {
      "step": 200,
      "loss": 0.2997
    },
    {
      "step": 205,
      "loss": 0.2925
    },
    {
      "step": 210,
      "loss": 0.2988
    },
    {
      "step": 215,
      "loss": 0.2904
    },
    {
      "step": 220,
      "loss": 0.3022
    },
    {
      "step": 225,
      "loss": 0.301
    },
    {
      "step": 230,
      "loss": 0.2998
    },
    {
      "step": 235,
      "loss": 0.2986
    },
    {
      "step": 240,
      "loss": 0.2894
    },
    {
      "step": 245,
      "loss": 0.289
    },
    {
      "step": 250,
      "loss": 0.2908
    },
    {
      "step": 255,
      "loss": 0.2906
    },
    {
      "step": 260,
      "loss": 0.2957
    },
    {
      "step": 265,
      "loss": 0.288
    },
    {
      "step": 270,
      "loss": 0.2894
    },
    {
      "step": 275,
      "loss": 0.296
    },
    {
      "step": 280,
      "loss": 0.2903
    },
    {
      "step": 285,
      "loss": 0.2945
    },
    {
      "step": 290,
      "loss": 0.2912
    },
    {
      "step": 295,
      "loss": 0.2974
    },
    {
      "step": 300,
      "loss": 0.2853
    },
    {
      "step": 305,
      "loss": 0.2863
    },
    {
      "step": 310,
      "loss": 0.2954
    },
    {
      "step": 315,
      "loss": 0.2913
    },
    {
      "step": 320,
      "loss": 0.2867
    },
    {
      "step": 325,
      "loss": 0.2915
    },
    {
      "step": 330,
      "loss": 0.2848
    },
    {
      "step": 335,
      "loss": 0.2785
    },
    {
      "step": 340,
      "loss": 0.282
    },
    {
      "step": 345,
      "loss": 0.2918
    },
    {
      "step": 350,
      "loss": 0.2776
    },
    {
      "step": 355,
      "loss": 0.2834
    },
    {
      "step": 360,
      "loss": 0.2887
    },
    {
      "step": 365,
      "loss": 0.2829
    },
    {
      "step": 370,
      "loss": 0.2797
    },
    {
      "step": 375,
      "loss": 0.2879
    },
    {
      "step": 380,
      "loss": 0.285
    },
    {
      "step": 385,
      "loss": 0.2813
    },
    {
      "step": 390,
      "loss": 0.2914
    },
    {
      "step": 395,
      "loss": 0.2869
    },
    {
      "step": 400,
      "loss": 0.2783
    },
    {
      "step": 405,
      "loss": 0.2831
    },
    {
      "step": 410,
      "loss": 0.2835
    },
    {
      "step": 415,
      "loss": 0.2735
    },
    {
      "step": 420,
      "loss": 0.2828
    },
    {
      "step": 425,
      "loss": 0.2795
    },
    {
      "step": 430,
      "loss": 0.2767
    },
    {
      "step": 435,
      "loss": 0.2792
    },
    {
      "step": 440,
      "loss": 0.2846
    },
    {
      "step": 445,
      "loss": 0.2855
    },
    {
      "step": 450,
      "loss": 0.2733
    },
    {
      "step": 455,
      "loss": 0.2812
    },
    {
      "step": 460,
      "loss": 0.2804
    },
    {
      "step": 465,
      "loss": 0.2792
    },
    {
      "step": 470,
      "loss": 0.2804
    },
    {
      "step": 475,
      "loss": 0.2728
    },
    {
      "step": 480,
      "loss": 0.2915
    },
    {
      "step": 485,
      "loss": 0.2809
    },
    {
      "step": 490,
      "loss": 0.2827
    },
    {
      "step": 495,
      "loss": 0.2777
    },
    {
      "step": 500,
      "loss": 0.2754
    },
    {
      "step": 505,
      "loss": 0.2801
    },
    {
      "step": 510,
      "loss": 0.273
    },
    {
      "step": 515,
      "loss": 0.2828
    },
    {
      "step": 520,
      "loss": 0.2715
    },
    {
      "step": 525,
      "loss": 0.2724
    },
    {
      "step": 530,
      "loss": 0.2785
    },
    {
      "step": 535,
      "loss": 0.2799
    },
    {
      "step": 540,
      "loss": 0.2818
    },
    {
      "step": 545,
      "loss": 0.2782
    },
    {
      "step": 550,
      "loss": 0.2826
    },
    {
      "step": 555,
      "loss": 0.28
    },
    {
      "step": 560,
      "loss": 0.2755
    },
    {
      "step": 565,
      "loss": 0.275
    },
    {
      "step": 570,
      "loss": 0.2786
    },
    {
      "step": 575,
      "loss": 0.277
    },
    {
      "step": 580,
      "loss": 0.2773
    },
    {
      "step": 585,
      "loss": 0.2748
    },
    {
      "step": 590,
      "loss": 0.2751
    },
    {
      "step": 595,
      "loss": 0.2787
    },
    {
      "step": 600,
      "loss": 0.2756
    },
    {
      "step": 605,
      "loss": 0.2737
    },
    {
      "step": 610,
      "loss": 0.2755
    },
    {
      "step": 615,
      "loss": 0.2717
    },
    {
      "step": 620,
      "loss": 0.2774
    },
    {
      "step": 625,
      "loss": 0.2737
    },
    {
      "step": 630,
      "loss": 0.2752
    },
    {
      "step": 635,
      "loss": 0.2746
    },
    {
      "step": 640,
      "loss": 0.2724
    },
    {
      "step": 645,
      "loss": 0.2736
    },
    {
      "step": 650,
      "loss": 0.278
    },
    {
      "step": 655,
      "loss": 0.2828
    },
    {
      "step": 660,
      "loss": 0.2737
    },
    {
      "step": 665,
      "loss": 0.2652
    },
    {
      "step": 670,
      "loss": 0.2708
    },
    {
      "step": 675,
      "loss": 0.2674
    },
    {
      "step": 680,
      "loss": 0.268
    },
    {
      "step": 685,
      "loss": 0.2694
    },
    {
      "step": 690,
      "loss": 0.2741
    },
    {
      "step": 695,
      "loss": 0.2643
    },
    {
      "step": 700,
      "loss": 0.264
    },
    {
      "step": 705,
      "loss": 0.2653
    },
    {
      "step": 710,
      "loss": 0.2672
    },
    {
      "step": 715,
      "loss": 0.2649
    },
    {
      "step": 720,
      "loss": 0.2678
    },
    {
      "step": 725,
      "loss": 0.2694
    },
    {
      "step": 730,
      "loss": 0.267
    },
    {
      "step": 735,
      "loss": 0.2676
    },
    {
      "step": 740,
      "loss": 0.2679
    },
    {
      "step": 745,
      "loss": 0.2626
    },
    {
      "step": 750,
      "loss": 0.2632
    },
    {
      "step": 755,
      "loss": 0.2633
    },
    {
      "step": 760,
      "loss": 0.2663
    },
    {
      "step": 765,
      "loss": 0.2694
    },
    {
      "step": 770,
      "loss": 0.2626
    },
    {
      "step": 775,
      "loss": 0.2533
    },
    {
      "step": 780,
      "loss": 0.2654
    },
    {
      "step": 785,
      "loss": 0.265
    },
    {
      "step": 790,
      "loss": 0.2617
    },
    {
      "step": 795,
      "loss": 0.2619
    },
    {
      "step": 800,
      "loss": 0.2646
    },
    {
      "step": 805,
      "loss": 0.2666
    },
    {
      "step": 810,
      "loss": 0.2624
    },
    {
      "step": 815,
      "loss": 0.2595
    },
    {
      "step": 820,
      "loss": 0.2576
    },
    {
      "step": 825,
      "loss": 0.2638
    },
    {
      "step": 830,
      "loss": 0.2521
    },
    {
      "step": 835,
      "loss": 0.2698
    },
    {
      "step": 840,
      "loss": 0.2586
    },
    {
      "step": 845,
      "loss": 0.2593
    },
    {
      "step": 850,
      "loss": 0.2604
    },
    {
      "step": 855,
      "loss": 0.2592
    },
    {
      "step": 860,
      "loss": 0.2568
    },
    {
      "step": 865,
      "loss": 0.2686
    },
    {
      "step": 870,
      "loss": 0.2547
    },
    {
      "step": 875,
      "loss": 0.2442
    },
    {
      "step": 880,
      "loss": 0.251
    },
    {
      "step": 885,
      "loss": 0.2515
    },
    {
      "step": 890,
      "loss": 0.2612
    },
    {
      "step": 895,
      "loss": 0.2572
    },
    {
      "step": 900,
      "loss": 0.2504
    },
    {
      "step": 905,
      "loss": 0.2528
    },
    {
      "step": 910,
      "loss": 0.2505
    },
    {
      "step": 915,
      "loss": 0.2541
    },
    {
      "step": 920,
      "loss": 0.2502
    },
    {
      "step": 925,
      "loss": 0.2527
    },
    {
      "step": 930,
      "loss": 0.2601
    },
    {
      "step": 935,
      "loss": 0.2515
    },
    {
      "step": 940,
      "loss": 0.2472
    },
    {
      "step": 945,
      "loss": 0.2554
    },
    {
      "step": 950,
      "loss": 0.2595
    },
    {
      "step": 955,
      "loss": 0.2476
    },
    {
      "step": 960,
      "loss": 0.2529
    },
    {
      "step": 965,
      "loss": 0.2509
    },
    {
      "step": 970,
      "loss": 0.2461
    },
    {
      "step": 975,
      "loss": 0.2452
    },
    {
      "step": 980,
      "loss": 0.2439
    },
    {
      "step": 985,
      "loss": 0.2522
    },
    {
      "step": 990,
      "loss": 0.2556
    },
    {
      "step": 995,
      "loss": 0.2471
    },
    {
      "step": 1000,
      "loss": 0.2559
    },
    {
      "step": 1005,
      "loss": 0.2542
    },
    {
      "step": 1010,
      "loss": 0.2527
    },
    {
      "step": 1015,
      "loss": 0.2446
    },
    {
      "step": 1020,
      "loss": 0.2518
    },
    {
      "step": 1025,
      "loss": 0.248
    },
    {
      "step": 1030,
      "loss": 0.2395
    },
    {
      "step": 1035,
      "loss": 0.2444
    },
    {
      "step": 1040,
      "loss": 0.2526
    },
    {
      "step": 1045,
      "loss": 0.2409
    },
    {
      "step": 1050,
      "loss": 0.238
    },
    {
      "step": 1055,
      "loss": 0.2637
    },
    {
      "step": 1060,
      "loss": 0.2487
    },
    {
      "step": 1065,
      "loss": 0.2561
    },
    {
      "step": 1070,
      "loss": 0.2397
    },
    {
      "step": 1075,
      "loss": 0.242
    },
    {
      "step": 1080,
      "loss": 0.2569
    },
    {
      "step": 1085,
      "loss": 0.248
    },
    {
      "step": 1090,
      "loss": 0.2475
    },
    {
      "step": 1095,
      "loss": 0.2417
    },
    {
      "step": 1100,
      "loss": 0.2511
    },
    {
      "step": 1105,
      "loss": 0.2465
    },
    {
      "step": 1110,
      "loss": 0.2395
    },
    {
      "step": 1115,
      "loss": 0.2421
    },
    {
      "step": 1120,
      "loss": 0.2384
    },
    {
      "step": 1125,
      "loss": 0.2459
    },
    {
      "step": 1130,
      "loss": 0.2396
    },
    {
      "step": 1135,
      "loss": 0.242
    },
    {
      "step": 1140,
      "loss": 0.2409
    },
    {
      "step": 1145,
      "loss": 0.2412
    },
    {
      "step": 1150,
      "loss": 0.2435
    },
    {
      "step": 1155,
      "loss": 0.2346
    },
    {
      "step": 1160,
      "loss": 0.2471
    },
    {
      "step": 1165,
      "loss": 0.243
    },
    {
      "step": 1170,
      "loss": 0.2326
    },
    {
      "step": 1175,
      "loss": 0.2352
    },
    {
      "step": 1180,
      "loss": 0.2332
    },
    {
      "step": 1185,
      "loss": 0.2419
    },
    {
      "step": 1190,
      "loss": 0.2365
    },
    {
      "step": 1195,
      "loss": 0.2417
    },
    {
      "step": 1200,
      "loss": 0.2453
    },
    {
      "step": 1205,
      "loss": 0.2329
    },
    {
      "step": 1210,
      "loss": 0.25
    },
    {
      "step": 1215,
      "loss": 0.2389
    },
    {
      "step": 1220,
      "loss": 0.2459
    },
    {
      "step": 1225,
      "loss": 0.2296
    },
    {
      "step": 1230,
      "loss": 0.2315
    },
    {
      "step": 1235,
      "loss": 0.236
    },
    {
      "step": 1240,
      "loss": 0.2358
    },
    {
      "step": 1245,
      "loss": 0.2411
    },
    {
      "step": 1250,
      "loss": 0.243
    },
    {
      "step": 1255,
      "loss": 0.2357
    },
    {
      "step": 1260,
      "loss": 0.2476
    },
    {
      "step": 1265,
      "loss": 0.2374
    },
    {
      "step": 1270,
      "loss": 0.2325
    },
    {
      "step": 1275,
      "loss": 0.2349
    },
    {
      "step": 1280,
      "loss": 0.2392
    },
    {
      "step": 1285,
      "loss": 0.2333
    },
    {
      "step": 1290,
      "loss": 0.2213
    },
    {
      "step": 1295,
      "loss": 0.2157
    },
    {
      "step": 1300,
      "loss": 0.2405
    },
    {
      "step": 1305,
      "loss": 0.2357
    },
    {
      "step": 1310,
      "loss": 0.2344
    },
    {
      "step": 1315,
      "loss": 0.2382
    },
    {
      "step": 1320,
      "loss": 0.2398
    },
    {
      "step": 1325,
      "loss": 0.2264
    },
    {
      "step": 1330,
      "loss": 0.2521
    },
    {
      "step": 1335,
      "loss": 0.2252
    },
    {
      "step": 1340,
      "loss": 0.235
    },
    {
      "step": 1345,
      "loss": 0.225
    },
    {
      "step": 1350,
      "loss": 0.2339
    },
    {
      "step": 1355,
      "loss": 0.2352
    },
    {
      "step": 1360,
      "loss": 0.2278
    },
    {
      "step": 1365,
      "loss": 0.2325
    },
    {
      "step": 1370,
      "loss": 0.236
    },
    {
      "step": 1375,
      "loss": 0.2239
    },
    {
      "step": 1380,
      "loss": 0.2356
    },
    {
      "step": 1385,
      "loss": 0.2252
    },
    {
      "step": 1390,
      "loss": 0.2225
    },
    {
      "step": 1395,
      "loss": 0.2378
    },
    {
      "step": 1400,
      "loss": 0.2261
    },
    {
      "step": 1405,
      "loss": 0.2328
    },
    {
      "step": 1410,
      "loss": 0.2378
    },
    {
      "step": 1415,
      "loss": 0.2267
    },
    {
      "step": 1420,
      "loss": 0.2313
    },
    {
      "step": 1425,
      "loss": 0.217
    },
    {
      "step": 1430,
      "loss": 0.2263
    },
    {
      "step": 1435,
      "loss": 0.2331
    },
    {
      "step": 1440,
      "loss": 0.2362
    },
    {
      "step": 1445,
      "loss": 0.2245
    },
    {
      "step": 1450,
      "loss": 0.2355
    },
    {
      "step": 1455,
      "loss": 0.2374
    },
    {
      "step": 1460,
      "loss": 0.236
    },
    {
      "step": 1465,
      "loss": 0.2206
    },
    {
      "step": 1470,
      "loss": 0.2282
    },
    {
      "step": 1475,
      "loss": 0.2251
    },
    {
      "step": 1480,
      "loss": 0.2405
    },
    {
      "step": 1485,
      "loss": 0.2253
    },
    {
      "step": 1490,
      "loss": 0.2429
    },
    {
      "step": 1495,
      "loss": 0.2216
    },
    {
      "step": 1500,
      "loss": 0.213
    },
    {
      "step": 1505,
      "loss": 0.2127
    },
    {
      "step": 1510,
      "loss": 0.2239
    },
    {
      "step": 1515,
      "loss": 0.2316
    },
    {
      "step": 1520,
      "loss": 0.2303
    },
    {
      "step": 1525,
      "loss": 0.2112
    },
    {
      "step": 1530,
      "loss": 0.2257
    },
    {
      "step": 1535,
      "loss": 0.2179
    },
    {
      "step": 1540,
      "loss": 0.2246
    },
    {
      "step": 1545,
      "loss": 0.2262
    },
    {
      "step": 1550,
      "loss": 0.2246
    },
    {
      "step": 1555,
      "loss": 0.2217
    },
    {
      "step": 1560,
      "loss": 0.2131
    },
    {
      "step": 1565,
      "loss": 0.221
    },
    {
      "step": 1570,
      "loss": 0.2307
    },
    {
      "step": 1575,
      "loss": 0.2296
    },
    {
      "step": 1580,
      "loss": 0.2307
    },
    {
      "step": 1585,
      "loss": 0.2223
    },
    {
      "step": 1590,
      "loss": 0.2187
    },
    {
      "step": 1595,
      "loss": 0.2014
    },
    {
      "step": 1600,
      "loss": 0.2245
    },
    {
      "step": 1605,
      "loss": 0.2205
    },
    {
      "step": 1610,
      "loss": 0.2119
    },
    {
      "step": 1615,
      "loss": 0.2302
    },
    {
      "step": 1620,
      "loss": 0.2261
    },
    {
      "step": 1625,
      "loss": 0.2164
    },
    {
      "step": 1630,
      "loss": 0.215
    },
    {
      "step": 1635,
      "loss": 0.2185
    },
    {
      "step": 1640,
      "loss": 0.2158
    },
    {
      "step": 1645,
      "loss": 0.2249
    },
    {
      "step": 1650,
      "loss": 0.2155
    },
    {
      "step": 1655,
      "loss": 0.2164
    },
    {
      "step": 1660,
      "loss": 0.2189
    },
    {
      "step": 1665,
      "loss": 0.209
    },
    {
      "step": 1670,
      "loss": 0.2242
    },
    {
      "step": 1675,
      "loss": 0.2135
    },
    {
      "step": 1680,
      "loss": 0.2183
    },
    {
      "step": 1685,
      "loss": 0.2282
    },
    {
      "step": 1690,
      "loss": 0.2187
    },
    {
      "step": 1695,
      "loss": 0.2288
    },
    {
      "step": 1700,
      "loss": 0.2039
    },
    {
      "step": 1705,
      "loss": 0.2239
    },
    {
      "step": 1710,
      "loss": 0.2133
    },
    {
      "step": 1715,
      "loss": 0.2199
    },
    {
      "step": 1720,
      "loss": 0.2214
    },
    {
      "step": 1725,
      "loss": 0.2172
    },
    {
      "step": 1730,
      "loss": 0.2267
    },
    {
      "step": 1735,
      "loss": 0.2165
    },
    {
      "step": 1740,
      "loss": 0.2292
    },
    {
      "step": 1745,
      "loss": 0.2228
    },
    {
      "step": 1750,
      "loss": 0.2098
    },
    {
      "step": 1755,
      "loss": 0.2211
    },
    {
      "step": 1760,
      "loss": 0.2174
    },
    {
      "step": 1765,
      "loss": 0.2121
    },
    {
      "step": 1770,
      "loss": 0.2245
    },
    {
      "step": 1775,
      "loss": 0.2187
    },
    {
      "step": 1780,
      "loss": 0.2074
    },
    {
      "step": 1785,
      "loss": 0.2239
    },
    {
      "step": 1790,
      "loss": 0.2166
    },
    {
      "step": 1795,
      "loss": 0.2116
    },
    {
      "step": 1800,
      "loss": 0.2126
    },
    {
      "step": 1805,
      "loss": 0.2123
    },
    {
      "step": 1810,
      "loss": 0.2202
    },
    {
      "step": 1815,
      "loss": 0.2259
    },
    {
      "step": 1820,
      "loss": 0.2017
    },
    {
      "step": 1825,
      "loss": 0.223
    },
    {
      "step": 1830,
      "loss": 0.2104
    },
    {
      "step": 1835,
      "loss": 0.2209
    },
    {
      "step": 1840,
      "loss": 0.2245
    },
    {
      "step": 1845,
      "loss": 0.2166
    },
    {
      "step": 1850,
      "loss": 0.2162
    },
    {
      "step": 1855,
      "loss": 0.2139
    },
    {
      "step": 1860,
      "loss": 0.2237
    },
    {
      "step": 1865,
      "loss": 0.2178
    },
    {
      "step": 1870,
      "loss": 0.2122
    },
    {
      "step": 1875,
      "loss": 0.2234
    },
    {
      "step": 1880,
      "loss": 0.2192
    },
    {
      "step": 1885,
      "loss": 0.207
    },
    {
      "step": 1890,
      "loss": 0.2034
    },
    {
      "step": 1895,
      "loss": 0.2158
    },
    {
      "step": 1900,
      "loss": 0.2131
    },
    {
      "step": 1905,
      "loss": 0.2116
    },
    {
      "step": 1910,
      "loss": 0.219
    },
    {
      "step": 1915,
      "loss": 0.2176
    },
    {
      "step": 1920,
      "loss": 0.2031
    },
    {
      "step": 1925,
      "loss": 0.2048
    },
    {
      "step": 1930,
      "loss": 0.2088
    },
    {
      "step": 1935,
      "loss": 0.2009
    },
    {
      "step": 1940,
      "loss": 0.2131
    },
    {
      "step": 1945,
      "loss": 0.2046
    },
    {
      "step": 1950,
      "loss": 0.2115
    },
    {
      "step": 1955,
      "loss": 0.2015
    },
    {
      "step": 1960,
      "loss": 0.2298
    },
    {
      "step": 1965,
      "loss": 0.1971
    },
    {
      "step": 1970,
      "loss": 0.2065
    },
    {
      "step": 1975,
      "loss": 0.2209
    },
    {
      "step": 1980,
      "loss": 0.2147
    },
    {
      "step": 1985,
      "loss": 0.2035
    },
    {
      "step": 1990,
      "loss": 0.22
    },
    {
      "step": 1995,
      "loss": 0.2081
    },
    {
      "step": 2000,
      "loss": 0.2078
    },
    {
      "step": 2005,
      "loss": 0.2066
    },
    {
      "step": 2010,
      "loss": 0.2096
    },
    {
      "step": 2015,
      "loss": 0.2007
    },
    {
      "step": 2020,
      "loss": 0.2196
    },
    {
      "step": 2025,
      "loss": 0.209
    },
    {
      "step": 2030,
      "loss": 0.2122
    },
    {
      "step": 2035,
      "loss": 0.2126
    },
    {
      "step": 2040,
      "loss": 0.192
    },
    {
      "step": 2045,
      "loss": 0.2138
    },
    {
      "step": 2050,
      "loss": 0.2077
    },
    {
      "step": 2055,
      "loss": 0.2214
    },
    {
      "step": 2060,
      "loss": 0.2052
    },
    {
      "step": 2065,
      "loss": 0.2115
    },
    {
      "step": 2070,
      "loss": 0.2128
    },
    {
      "step": 2075,
      "loss": 0.2057
    },
    {
      "step": 2080,
      "loss": 0.2196
    },
    {
      "step": 2085,
      "loss": 0.2112
    },
    {
      "step": 2090,
      "loss": 0.1876
    },
    {
      "step": 2095,
      "loss": 0.2067
    },
    {
      "step": 2100,
      "loss": 0.2051
    },
    {
      "step": 2105,
      "loss": 0.2052
    },
    {
      "step": 2110,
      "loss": 0.2113
    },
    {
      "step": 2115,
      "loss": 0.2178
    },
    {
      "step": 2120,
      "loss": 0.222
    },
    {
      "step": 2125,
      "loss": 0.2025
    },
    {
      "step": 2130,
      "loss": 0.1964
    },
    {
      "step": 2135,
      "loss": 0.2009
    },
    {
      "step": 2140,
      "loss": 0.2157
    },
    {
      "step": 2145,
      "loss": 0.2102
    },
    {
      "step": 2150,
      "loss": 0.2052
    },
    {
      "step": 2155,
      "loss": 0.2048
    },
    {
      "step": 2160,
      "loss": 0.2039
    },
    {
      "step": 2165,
      "loss": 0.2189
    },
    {
      "step": 2170,
      "loss": 0.2038
    },
    {
      "step": 2175,
      "loss": 0.1979
    },
    {
      "step": 2180,
      "loss": 0.2086
    },
    {
      "step": 2185,
      "loss": 0.1953
    },
    {
      "step": 2190,
      "loss": 0.2176
    },
    {
      "step": 2195,
      "loss": 0.2073
    },
    {
      "step": 2200,
      "loss": 0.2115
    },
    {
      "step": 2205,
      "loss": 0.217
    },
    {
      "step": 2210,
      "loss": 0.1957
    },
    {
      "step": 2215,
      "loss": 0.2211
    },
    {
      "step": 2220,
      "loss": 0.2125
    },
    {
      "step": 2225,
      "loss": 0.2047
    },
    {
      "step": 2230,
      "loss": 0.2092
    },
    {
      "step": 2235,
      "loss": 0.2144
    },
    {
      "step": 2240,
      "loss": 0.2041
    },
    {
      "step": 2245,
      "loss": 0.1984
    },
    {
      "step": 2250,
      "loss": 0.2033
    },
    {
      "step": 2255,
      "loss": 0.1931
    },
    {
      "step": 2260,
      "loss": 0.2071
    },
    {
      "step": 2265,
      "loss": 0.2065
    },
    {
      "step": 2270,
      "loss": 0.2315
    },
    {
      "step": 2275,
      "loss": 0.216
    },
    {
      "step": 2280,
      "loss": 0.2217
    },
    {
      "step": 2285,
      "loss": 0.2055
    },
    {
      "step": 2290,
      "loss": 0.1979
    },
    {
      "step": 2295,
      "loss": 0.2064
    },
    {
      "step": 2300,
      "loss": 0.2088
    },
    {
      "step": 2305,
      "loss": 0.2013
    },
    {
      "step": 2310,
      "loss": 0.2089
    },
    {
      "step": 2315,
      "loss": 0.202
    },
    {
      "step": 2320,
      "loss": 0.2032
    },
    {
      "step": 2325,
      "loss": 0.2022
    },
    {
      "step": 2330,
      "loss": 0.2136
    },
    {
      "step": 2335,
      "loss": 0.1958
    },
    {
      "step": 2340,
      "loss": 0.2038
    },
    {
      "step": 2345,
      "loss": 0.202
    },
    {
      "step": 2350,
      "loss": 0.2081
    },
    {
      "step": 2355,
      "loss": 0.2009
    },
    {
      "step": 2360,
      "loss": 0.1952
    },
    {
      "step": 2365,
      "loss": 0.2088
    },
    {
      "step": 2370,
      "loss": 0.2005
    },
    {
      "step": 2375,
      "loss": 0.2011
    },
    {
      "step": 2380,
      "loss": 0.1877
    },
    {
      "step": 2385,
      "loss": 0.1931
    },
    {
      "step": 2390,
      "loss": 0.1993
    },
    {
      "step": 2395,
      "loss": 0.2057
    },
    {
      "step": 2400,
      "loss": 0.1891
    },
    {
      "step": 2405,
      "loss": 0.208
    },
    {
      "step": 2410,
      "loss": 0.2019
    },
    {
      "step": 2415,
      "loss": 0.1925
    },
    {
      "step": 2420,
      "loss": 0.2042
    },
    {
      "step": 2425,
      "loss": 0.2081
    },
    {
      "step": 2430,
      "loss": 0.198
    },
    {
      "step": 2435,
      "loss": 0.199
    },
    {
      "step": 2440,
      "loss": 0.2099
    },
    {
      "step": 2445,
      "loss": 0.1991
    },
    {
      "step": 2450,
      "loss": 0.2116
    },
    {
      "step": 2455,
      "loss": 0.2099
    },
    {
      "step": 2460,
      "loss": 0.1997
    },
    {
      "step": 2465,
      "loss": 0.2025
    },
    {
      "step": 2470,
      "loss": 0.1997
    },
    {
      "step": 2475,
      "loss": 0.1962
    },
    {
      "step": 2480,
      "loss": 0.2041
    },
    {
      "step": 2485,
      "loss": 0.2116
    },
    {
      "step": 2490,
      "loss": 0.2096
    },
    {
      "step": 2495,
      "loss": 0.2025
    },
    {
      "step": 2500,
      "loss": 0.1868
    },
    {
      "step": 2505,
      "loss": 0.1965
    },
    {
      "step": 2510,
      "loss": 0.2065
    },
    {
      "step": 2515,
      "loss": 0.202
    },
    {
      "step": 2520,
      "loss": 0.2057
    },
    {
      "step": 2525,
      "loss": 0.1957
    },
    {
      "step": 2530,
      "loss": 0.2004
    },
    {
      "step": 2535,
      "loss": 0.1955
    },
    {
      "step": 2540,
      "loss": 0.1937
    },
    {
      "step": 2545,
      "loss": 0.2086
    },
    {
      "step": 2550,
      "loss": 0.1955
    },
    {
      "step": 2555,
      "loss": 0.1814
    },
    {
      "step": 2560,
      "loss": 0.19
    },
    {
      "step": 2565,
      "loss": 0.1936
    },
    {
      "step": 2570,
      "loss": 0.2085
    },
    {
      "step": 2575,
      "loss": 0.2019
    },
    {
      "step": 2580,
      "loss": 0.1996
    },
    {
      "step": 2585,
      "loss": 0.1968
    },
    {
      "step": 2590,
      "loss": 0.2132
    },
    {
      "step": 2595,
      "loss": 0.1957
    },
    {
      "step": 2600,
      "loss": 0.2063
    },
    {
      "step": 2605,
      "loss": 0.1966
    },
    {
      "step": 2610,
      "loss": 0.1993
    },
    {
      "step": 2615,
      "loss": 0.2003
    },
    {
      "step": 2620,
      "loss": 0.1916
    },
    {
      "step": 2625,
      "loss": 0.2006
    },
    {
      "step": 2630,
      "loss": 0.2053
    },
    {
      "step": 2635,
      "loss": 0.2181
    },
    {
      "step": 2640,
      "loss": 0.2034
    },
    {
      "step": 2645,
      "loss": 0.2092
    },
    {
      "step": 2650,
      "loss": 0.2024
    },
    {
      "step": 2655,
      "loss": 0.207
    },
    {
      "step": 2660,
      "loss": 0.2096
    },
    {
      "step": 2665,
      "loss": 0.1987
    },
    {
      "step": 2670,
      "loss": 0.1939
    },
    {
      "step": 2675,
      "loss": 0.1877
    },
    {
      "step": 2680,
      "loss": 0.2032
    },
    {
      "step": 2685,
      "loss": 0.1929
    },
    {
      "step": 2690,
      "loss": 0.1974
    },
    {
      "step": 2695,
      "loss": 0.1977
    },
    {
      "step": 2700,
      "loss": 0.1911
    },
    {
      "step": 2705,
      "loss": 0.1964
    },
    {
      "step": 2710,
      "loss": 0.2111
    },
    {
      "step": 2715,
      "loss": 0.208
    },
    {
      "step": 2720,
      "loss": 0.1975
    },
    {
      "step": 2725,
      "loss": 0.2006
    },
    {
      "step": 2730,
      "loss": 0.1961
    },
    {
      "step": 2735,
      "loss": 0.2022
    },
    {
      "step": 2740,
      "loss": 0.1933
    },
    {
      "step": 2745,
      "loss": 0.1996
    },
    {
      "step": 2750,
      "loss": 0.1873
    },
    {
      "step": 2755,
      "loss": 0.1939
    },
    {
      "step": 2760,
      "loss": 0.1868
    },
    {
      "step": 2765,
      "loss": 0.1853
    },
    {
      "step": 2770,
      "loss": 0.2028
    },
    {
      "step": 2775,
      "loss": 0.1882
    },
    {
      "step": 2780,
      "loss": 0.1964
    },
    {
      "step": 2785,
      "loss": 0.2033
    },
    {
      "step": 2790,
      "loss": 0.1957
    },
    {
      "step": 2795,
      "loss": 0.1935
    },
    {
      "step": 2800,
      "loss": 0.2083
    },
    {
      "step": 2805,
      "loss": 0.2232
    },
    {
      "step": 2810,
      "loss": 0.2016
    },
    {
      "step": 2815,
      "loss": 0.2006
    },
    {
      "step": 2820,
      "loss": 0.2084
    },
    {
      "step": 2825,
      "loss": 0.1883
    },
    {
      "step": 2830,
      "loss": 0.1777
    },
    {
      "step": 2835,
      "loss": 0.1983
    },
    {
      "step": 2840,
      "loss": 0.1984
    },
    {
      "step": 2845,
      "loss": 0.1948
    },
    {
      "step": 2850,
      "loss": 0.191
    },
    {
      "step": 2855,
      "loss": 0.2026
    },
    {
      "step": 2860,
      "loss": 0.1924
    },
    {
      "step": 2865,
      "loss": 0.1967
    },
    {
      "step": 2870,
      "loss": 0.1911
    },
    {
      "step": 2875,
      "loss": 0.1909
    },
    {
      "step": 2880,
      "loss": 0.1947
    },
    {
      "step": 2885,
      "loss": 0.1848
    },
    {
      "step": 2890,
      "loss": 0.1947
    },
    {
      "step": 2895,
      "loss": 0.192
    },
    {
      "step": 2900,
      "loss": 0.2035
    },
    {
      "step": 2905,
      "loss": 0.1847
    },
    {
      "step": 2910,
      "loss": 0.204
    },
    {
      "step": 2915,
      "loss": 0.2012
    },
    {
      "step": 2920,
      "loss": 0.1908
    },
    {
      "step": 2925,
      "loss": 0.1982
    },
    {
      "step": 2930,
      "loss": 0.1905
    },
    {
      "step": 2935,
      "loss": 0.1871
    },
    {
      "step": 2940,
      "loss": 0.2013
    },
    {
      "step": 2945,
      "loss": 0.1924
    },
    {
      "step": 2950,
      "loss": 0.1912
    },
    {
      "step": 2955,
      "loss": 0.1813
    },
    {
      "step": 2960,
      "loss": 0.1987
    },
    {
      "step": 2965,
      "loss": 0.2031
    },
    {
      "step": 2970,
      "loss": 0.2012
    },
    {
      "step": 2975,
      "loss": 0.191
    },
    {
      "step": 2980,
      "loss": 0.1931
    },
    {
      "step": 2985,
      "loss": 0.178
    },
    {
      "step": 2990,
      "loss": 0.1889
    },
    {
      "step": 2995,
      "loss": 0.1911
    },
    {
      "step": 3000,
      "loss": 0.2019
    },
    {
      "step": 3005,
      "loss": 0.1979
    },
    {
      "step": 3010,
      "loss": 0.1932
    },
    {
      "step": 3015,
      "loss": 0.185
    },
    {
      "step": 3020,
      "loss": 0.1896
    },
    {
      "step": 3025,
      "loss": 0.2014
    },
    {
      "step": 3030,
      "loss": 0.1911
    },
    {
      "step": 3035,
      "loss": 0.2014
    },
    {
      "step": 3040,
      "loss": 0.2039
    },
    {
      "step": 3045,
      "loss": 0.2023
    },
    {
      "step": 3050,
      "loss": 0.2008
    },
    {
      "step": 3055,
      "loss": 0.1804
    },
    {
      "step": 3060,
      "loss": 0.1947
    },
    {
      "step": 3065,
      "loss": 0.1789
    },
    {
      "step": 3070,
      "loss": 0.1911
    },
    {
      "step": 3075,
      "loss": 0.1911
    },
    {
      "step": 3080,
      "loss": 0.1812
    },
    {
      "step": 3085,
      "loss": 0.1753
    },
    {
      "step": 3090,
      "loss": 0.1872
    },
    {
      "step": 3095,
      "loss": 0.2085
    },
    {
      "step": 3100,
      "loss": 0.185
    },
    {
      "step": 3105,
      "loss": 0.2038
    },
    {
      "step": 3110,
      "loss": 0.1755
    },
    {
      "step": 3115,
      "loss": 0.2032
    },
    {
      "step": 3120,
      "loss": 0.1915
    },
    {
      "step": 3125,
      "loss": 0.1936
    },
    {
      "step": 3130,
      "loss": 0.2074
    },
    {
      "step": 3135,
      "loss": 0.1949
    },
    {
      "step": 3140,
      "loss": 0.2012
    },
    {
      "step": 3145,
      "loss": 0.192
    },
    {
      "step": 3150,
      "loss": 0.1919
    },
    {
      "step": 3155,
      "loss": 0.2096
    },
    {
      "step": 3160,
      "loss": 0.207
    },
    {
      "step": 3165,
      "loss": 0.1948
    },
    {
      "step": 3170,
      "loss": 0.1855
    },
    {
      "step": 3175,
      "loss": 0.1911
    },
    {
      "step": 3180,
      "loss": 0.2033
    },
    {
      "step": 3185,
      "loss": 0.2024
    },
    {
      "step": 3190,
      "loss": 0.1799
    },
    {
      "step": 3195,
      "loss": 0.1988
    },
    {
      "step": 3200,
      "loss": 0.1875
    },
    {
      "step": 3205,
      "loss": 0.1863
    },
    {
      "step": 3210,
      "loss": 0.1924
    },
    {
      "step": 3215,
      "loss": 0.1847
    },
    {
      "step": 3220,
      "loss": 0.1971
    },
    {
      "step": 3225,
      "loss": 0.1895
    },
    {
      "step": 3230,
      "loss": 0.1859
    },
    {
      "step": 3235,
      "loss": 0.1941
    },
    {
      "step": 3240,
      "loss": 0.1944
    },
    {
      "step": 3245,
      "loss": 0.1789
    },
    {
      "step": 3250,
      "loss": 0.1925
    },
    {
      "step": 3255,
      "loss": 0.2008
    },
    {
      "step": 3260,
      "loss": 0.1972
    },
    {
      "step": 3265,
      "loss": 0.2017
    },
    {
      "step": 3270,
      "loss": 0.1905
    },
    {
      "step": 3275,
      "loss": 0.2071
    },
    {
      "step": 3280,
      "loss": 0.1978
    },
    {
      "step": 3285,
      "loss": 0.1886
    },
    {
      "step": 3290,
      "loss": 0.1883
    },
    {
      "step": 3295,
      "loss": 0.1837
    },
    {
      "step": 3300,
      "loss": 0.1897
    },
    {
      "step": 3305,
      "loss": 0.1955
    },
    {
      "step": 3310,
      "loss": 0.1793
    },
    {
      "step": 3315,
      "loss": 0.1851
    },
    {
      "step": 3320,
      "loss": 0.1982
    },
    {
      "step": 3325,
      "loss": 0.1732
    },
    {
      "step": 3330,
      "loss": 0.1941
    },
    {
      "step": 3335,
      "loss": 0.1893
    },
    {
      "step": 3340,
      "loss": 0.1876
    },
    {
      "step": 3345,
      "loss": 0.1803
    },
    {
      "step": 3350,
      "loss": 0.1759
    },
    {
      "step": 3355,
      "loss": 0.1926
    },
    {
      "step": 3360,
      "loss": 0.2069
    },
    {
      "step": 3365,
      "loss": 0.1744
    },
    {
      "step": 3370,
      "loss": 0.1831
    },
    {
      "step": 3375,
      "loss": 0.1962
    },
    {
      "step": 3380,
      "loss": 0.1893
    },
    {
      "step": 3385,
      "loss": 0.1837
    },
    {
      "step": 3390,
      "loss": 0.2006
    },
    {
      "step": 3395,
      "loss": 0.2053
    },
    {
      "step": 3400,
      "loss": 0.1963
    },
    {
      "step": 3405,
      "loss": 0.193
    },
    {
      "step": 3410,
      "loss": 0.1903
    },
    {
      "step": 3415,
      "loss": 0.1967
    },
    {
      "step": 3420,
      "loss": 0.1929
    },
    {
      "step": 3425,
      "loss": 0.174
    },
    {
      "step": 3430,
      "loss": 0.1893
    },
    {
      "step": 3435,
      "loss": 0.177
    },
    {
      "step": 3440,
      "loss": 0.1963
    },
    {
      "step": 3445,
      "loss": 0.1994
    },
    {
      "step": 3450,
      "loss": 0.1895
    },
    {
      "step": 3455,
      "loss": 0.1999
    },
    {
      "step": 3460,
      "loss": 0.1964
    },
    {
      "step": 3465,
      "loss": 0.1996
    },
    {
      "step": 3470,
      "loss": 0.2067
    },
    {
      "step": 3475,
      "loss": 0.1764
    },
    {
      "step": 3480,
      "loss": 0.1932
    },
    {
      "step": 3485,
      "loss": 0.1816
    },
    {
      "step": 3490,
      "loss": 0.1981
    },
    {
      "step": 3495,
      "loss": 0.1899
    },
    {
      "step": 3500,
      "loss": 0.187
    },
    {
      "step": 3505,
      "loss": 0.179
    },
    {
      "step": 3510,
      "loss": 0.1985
    },
    {
      "step": 3515,
      "loss": 0.1874
    },
    {
      "step": 3520,
      "loss": 0.1786
    },
    {
      "step": 3525,
      "loss": 0.1866
    },
    {
      "step": 3530,
      "loss": 0.2008
    },
    {
      "step": 3535,
      "loss": 0.1907
    },
    {
      "step": 3540,
      "loss": 0.1748
    },
    {
      "step": 3545,
      "loss": 0.1803
    },
    {
      "step": 3550,
      "loss": 0.1756
    },
    {
      "step": 3555,
      "loss": 0.1801
    },
    {
      "step": 3560,
      "loss": 0.1835
    },
    {
      "step": 3565,
      "loss": 0.1683
    },
    {
      "step": 3570,
      "loss": 0.2043
    },
    {
      "step": 3575,
      "loss": 0.1845
    },
    {
      "step": 3580,
      "loss": 0.1819
    },
    {
      "step": 3585,
      "loss": 0.193
    },
    {
      "step": 3590,
      "loss": 0.1983
    },
    {
      "step": 3595,
      "loss": 0.1956
    },
    {
      "step": 3600,
      "loss": 0.1866
    },
    {
      "step": 3605,
      "loss": 0.1871
    },
    {
      "step": 3610,
      "loss": 0.1789
    },
    {
      "step": 3615,
      "loss": 0.168
    },
    {
      "step": 3620,
      "loss": 0.1884
    },
    {
      "step": 3625,
      "loss": 0.186
    },
    {
      "step": 3630,
      "loss": 0.1809
    },
    {
      "step": 3635,
      "loss": 0.1862
    },
    {
      "step": 3640,
      "loss": 0.1798
    },
    {
      "step": 3645,
      "loss": 0.1845
    },
    {
      "step": 3650,
      "loss": 0.1866
    },
    {
      "step": 3655,
      "loss": 0.1897
    },
    {
      "step": 3660,
      "loss": 0.1782
    },
    {
      "step": 3665,
      "loss": 0.1942
    },
    {
      "step": 3670,
      "loss": 0.1908
    },
    {
      "step": 3675,
      "loss": 0.1897
    },
    {
      "step": 3680,
      "loss": 0.1987
    },
    {
      "step": 3685,
      "loss": 0.1833
    },
    {
      "step": 3690,
      "loss": 0.1844
    },
    {
      "step": 3695,
      "loss": 0.1808
    },
    {
      "step": 3700,
      "loss": 0.1839
    },
    {
      "step": 3705,
      "loss": 0.1928
    },
    {
      "step": 3710,
      "loss": 0.1842
    },
    {
      "step": 3715,
      "loss": 0.1898
    },
    {
      "step": 3720,
      "loss": 0.1866
    },
    {
      "step": 3725,
      "loss": 0.1888
    },
    {
      "step": 3730,
      "loss": 0.1951
    },
    {
      "step": 3735,
      "loss": 0.1927
    },
    {
      "step": 3740,
      "loss": 0.1884
    },
    {
      "step": 3745,
      "loss": 0.1914
    },
    {
      "step": 3750,
      "loss": 0.1865
    },
    {
      "step": 3755,
      "loss": 0.1844
    },
    {
      "step": 3760,
      "loss": 0.1781
    },
    {
      "step": 3765,
      "loss": 0.1685
    },
    {
      "step": 3770,
      "loss": 0.1925
    },
    {
      "step": 3775,
      "loss": 0.1822
    },
    {
      "step": 3780,
      "loss": 0.1862
    },
    {
      "step": 3785,
      "loss": 0.207
    },
    {
      "step": 3790,
      "loss": 0.1865
    },
    {
      "step": 3795,
      "loss": 0.1816
    },
    {
      "step": 3800,
      "loss": 0.1909
    },
    {
      "step": 3805,
      "loss": 0.1829
    },
    {
      "step": 3810,
      "loss": 0.1801
    },
    {
      "step": 3815,
      "loss": 0.1967
    },
    {
      "step": 3820,
      "loss": 0.19
    },
    {
      "step": 3825,
      "loss": 0.1889
    },
    {
      "step": 3830,
      "loss": 0.1698
    },
    {
      "step": 3835,
      "loss": 0.1724
    },
    {
      "step": 3840,
      "loss": 0.1832
    },
    {
      "step": 3845,
      "loss": 0.1917
    },
    {
      "step": 3850,
      "loss": 0.1879
    },
    {
      "step": 3855,
      "loss": 0.1933
    },
    {
      "step": 3860,
      "loss": 0.1888
    },
    {
      "step": 3865,
      "loss": 0.1872
    },
    {
      "step": 3870,
      "loss": 0.187
    },
    {
      "step": 3875,
      "loss": 0.1884
    },
    {
      "step": 3880,
      "loss": 0.1817
    },
    {
      "step": 3885,
      "loss": 0.1956
    },
    {
      "step": 3890,
      "loss": 0.1745
    },
    {
      "step": 3895,
      "loss": 0.179
    },
    {
      "step": 3900,
      "loss": 0.1831
    },
    {
      "step": 3905,
      "loss": 0.1934
    },
    {
      "step": 3910,
      "loss": 0.1767
    },
    {
      "step": 3915,
      "loss": 0.1862
    },
    {
      "step": 3920,
      "loss": 0.1748
    },
    {
      "step": 3925,
      "loss": 0.1769
    },
    {
      "step": 3930,
      "loss": 0.1879
    },
    {
      "step": 3935,
      "loss": 0.1771
    },
    {
      "step": 3940,
      "loss": 0.1891
    },
    {
      "step": 3945,
      "loss": 0.1901
    },
    {
      "step": 3950,
      "loss": 0.1909
    },
    {
      "step": 3955,
      "loss": 0.1859
    },
    {
      "step": 3960,
      "loss": 0.1777
    },
    {
      "step": 3965,
      "loss": 0.1759
    },
    {
      "step": 3970,
      "loss": 0.1845
    },
    {
      "step": 3975,
      "loss": 0.1966
    },
    {
      "step": 3980,
      "loss": 0.1936
    },
    {
      "step": 3985,
      "loss": 0.1821
    },
    {
      "step": 3990,
      "loss": 0.1966
    },
    {
      "step": 3995,
      "loss": 0.1891
    },
    {
      "step": 4000,
      "loss": 0.1954
    },
    {
      "step": 4005,
      "loss": 0.1925
    },
    {
      "step": 4010,
      "loss": 0.1824
    },
    {
      "step": 4015,
      "loss": 0.1929
    },
    {
      "step": 4020,
      "loss": 0.177
    },
    {
      "step": 4025,
      "loss": 0.1845
    },
    {
      "step": 4030,
      "loss": 0.1703
    },
    {
      "step": 4035,
      "loss": 0.1758
    },
    {
      "step": 4040,
      "loss": 0.1901
    },
    {
      "step": 4045,
      "loss": 0.1878
    },
    {
      "step": 4050,
      "loss": 0.2005
    },
    {
      "step": 4055,
      "loss": 0.1823
    },
    {
      "step": 4060,
      "loss": 0.1743
    },
    {
      "step": 4065,
      "loss": 0.1822
    },
    {
      "step": 4070,
      "loss": 0.184
    },
    {
      "step": 4075,
      "loss": 0.1703
    },
    {
      "step": 4080,
      "loss": 0.1675
    },
    {
      "step": 4085,
      "loss": 0.1789
    },
    {
      "step": 4090,
      "loss": 0.1795
    },
    {
      "step": 4095,
      "loss": 0.1633
    },
    {
      "step": 4100,
      "loss": 0.1894
    },
    {
      "step": 4105,
      "loss": 0.1871
    },
    {
      "step": 4110,
      "loss": 0.187
    },
    {
      "step": 4115,
      "loss": 0.1791
    },
    {
      "step": 4120,
      "loss": 0.1759
    },
    {
      "step": 4125,
      "loss": 0.1732
    },
    {
      "step": 4130,
      "loss": 0.1811
    },
    {
      "step": 4135,
      "loss": 0.1878
    },
    {
      "step": 4140,
      "loss": 0.1949
    },
    {
      "step": 4145,
      "loss": 0.1824
    },
    {
      "step": 4150,
      "loss": 0.2009
    },
    {
      "step": 4155,
      "loss": 0.1976
    },
    {
      "step": 4160,
      "loss": 0.1845
    },
    {
      "step": 4165,
      "loss": 0.1884
    },
    {
      "step": 4170,
      "loss": 0.1792
    },
    {
      "step": 4175,
      "loss": 0.1778
    },
    {
      "step": 4180,
      "loss": 0.1869
    },
    {
      "step": 4185,
      "loss": 0.1836
    },
    {
      "step": 4190,
      "loss": 0.1699
    },
    {
      "step": 4195,
      "loss": 0.1962
    },
    {
      "step": 4200,
      "loss": 0.1913
    },
    {
      "step": 4205,
      "loss": 0.1707
    },
    {
      "step": 4210,
      "loss": 0.1696
    },
    {
      "step": 4215,
      "loss": 0.1914
    },
    {
      "step": 4220,
      "loss": 0.1714
    },
    {
      "step": 4225,
      "loss": 0.1913
    },
    {
      "step": 4230,
      "loss": 0.1638
    },
    {
      "step": 4235,
      "loss": 0.1772
    },
    {
      "step": 4240,
      "loss": 0.1847
    },
    {
      "step": 4245,
      "loss": 0.1806
    },
    {
      "step": 4250,
      "loss": 0.1843
    },
    {
      "step": 4255,
      "loss": 0.1777
    },
    {
      "step": 4260,
      "loss": 0.1824
    },
    {
      "step": 4265,
      "loss": 0.1835
    },
    {
      "step": 4270,
      "loss": 0.1823
    },
    {
      "step": 4275,
      "loss": 0.1703
    },
    {
      "step": 4280,
      "loss": 0.1779
    },
    {
      "step": 4285,
      "loss": 0.1684
    },
    {
      "step": 4290,
      "loss": 0.2021
    },
    {
      "step": 4295,
      "loss": 0.1711
    },
    {
      "step": 4300,
      "loss": 0.1723
    },
    {
      "step": 4305,
      "loss": 0.1714
    },
    {
      "step": 4310,
      "loss": 0.1723
    },
    {
      "step": 4315,
      "loss": 0.1858
    },
    {
      "step": 4320,
      "loss": 0.1795
    },
    {
      "step": 4325,
      "loss": 0.1695
    },
    {
      "step": 4330,
      "loss": 0.1773
    },
    {
      "step": 4335,
      "loss": 0.1809
    },
    {
      "step": 4340,
      "loss": 0.1677
    },
    {
      "step": 4345,
      "loss": 0.1772
    },
    {
      "step": 4350,
      "loss": 0.1753
    },
    {
      "step": 4355,
      "loss": 0.1693
    },
    {
      "step": 4360,
      "loss": 0.1804
    },
    {
      "step": 4365,
      "loss": 0.175
    },
    {
      "step": 4370,
      "loss": 0.1621
    },
    {
      "step": 4375,
      "loss": 0.164
    },
    {
      "step": 4380,
      "loss": 0.1723
    },
    {
      "step": 4385,
      "loss": 0.1698
    },
    {
      "step": 4390,
      "loss": 0.1829
    },
    {
      "step": 4395,
      "loss": 0.1651
    },
    {
      "step": 4400,
      "loss": 0.182
    },
    {
      "step": 4405,
      "loss": 0.1692
    },
    {
      "step": 4410,
      "loss": 0.1673
    },
    {
      "step": 4415,
      "loss": 0.1753
    },
    {
      "step": 4420,
      "loss": 0.1981
    },
    {
      "step": 4425,
      "loss": 0.1883
    },
    {
      "step": 4430,
      "loss": 0.1845
    },
    {
      "step": 4435,
      "loss": 0.1749
    },
    {
      "step": 4440,
      "loss": 0.1818
    },
    {
      "step": 4445,
      "loss": 0.1911
    },
    {
      "step": 4450,
      "loss": 0.1767
    },
    {
      "step": 4455,
      "loss": 0.1802
    },
    {
      "step": 4460,
      "loss": 0.1896
    },
    {
      "step": 4465,
      "loss": 0.1868
    },
    {
      "step": 4470,
      "loss": 0.1785
    },
    {
      "step": 4475,
      "loss": 0.1729
    },
    {
      "step": 4480,
      "loss": 0.1893
    },
    {
      "step": 4485,
      "loss": 0.174
    },
    {
      "step": 4490,
      "loss": 0.1739
    },
    {
      "step": 4495,
      "loss": 0.1776
    },
    {
      "step": 4500,
      "loss": 0.1691
    },
    {
      "step": 4505,
      "loss": 0.1596
    },
    {
      "step": 4510,
      "loss": 0.1835
    },
    {
      "step": 4515,
      "loss": 0.1695
    },
    {
      "step": 4520,
      "loss": 0.1854
    },
    {
      "step": 4525,
      "loss": 0.1638
    },
    {
      "step": 4530,
      "loss": 0.1831
    },
    {
      "step": 4535,
      "loss": 0.1745
    },
    {
      "step": 4540,
      "loss": 0.1844
    },
    {
      "step": 4545,
      "loss": 0.1763
    },
    {
      "step": 4550,
      "loss": 0.1759
    },
    {
      "step": 4555,
      "loss": 0.172
    },
    {
      "step": 4560,
      "loss": 0.1856
    },
    {
      "step": 4565,
      "loss": 0.174
    },
    {
      "step": 4570,
      "loss": 0.1664
    },
    {
      "step": 4575,
      "loss": 0.1701
    },
    {
      "step": 4580,
      "loss": 0.1879
    },
    {
      "step": 4585,
      "loss": 0.1707
    },
    {
      "step": 4590,
      "loss": 0.1633
    },
    {
      "step": 4595,
      "loss": 0.1597
    },
    {
      "step": 4600,
      "loss": 0.1849
    },
    {
      "step": 4605,
      "loss": 0.1782
    },
    {
      "step": 4610,
      "loss": 0.1877
    },
    {
      "step": 4615,
      "loss": 0.1657
    },
    {
      "step": 4620,
      "loss": 0.1793
    },
    {
      "step": 4625,
      "loss": 0.183
    },
    {
      "step": 4630,
      "loss": 0.1792
    },
    {
      "step": 4635,
      "loss": 0.1776
    },
    {
      "step": 4640,
      "loss": 0.1815
    },
    {
      "step": 4645,
      "loss": 0.1914
    },
    {
      "step": 4650,
      "loss": 0.1657
    },
    {
      "step": 4655,
      "loss": 0.175
    },
    {
      "step": 4660,
      "loss": 0.1915
    },
    {
      "step": 4665,
      "loss": 0.1608
    },
    {
      "step": 4670,
      "loss": 0.1712
    },
    {
      "step": 4675,
      "loss": 0.1762
    },
    {
      "step": 4680,
      "loss": 0.1772
    },
    {
      "step": 4685,
      "loss": 0.177
    },
    {
      "step": 4690,
      "loss": 0.176
    },
    {
      "step": 4695,
      "loss": 0.1743
    },
    {
      "step": 4700,
      "loss": 0.1769
    },
    {
      "step": 4705,
      "loss": 0.1665
    },
    {
      "step": 4710,
      "loss": 0.1722
    },
    {
      "step": 4715,
      "loss": 0.1698
    },
    {
      "step": 4720,
      "loss": 0.1759
    },
    {
      "step": 4725,
      "loss": 0.166
    },
    {
      "step": 4730,
      "loss": 0.1732
    },
    {
      "step": 4735,
      "loss": 0.168
    },
    {
      "step": 4740,
      "loss": 0.1737
    },
    {
      "step": 4745,
      "loss": 0.176
    },
    {
      "step": 4750,
      "loss": 0.1764
    },
    {
      "step": 4755,
      "loss": 0.1662
    },
    {
      "step": 4760,
      "loss": 0.1782
    },
    {
      "step": 4765,
      "loss": 0.176
    },
    {
      "step": 4770,
      "loss": 0.1777
    },
    {
      "step": 4775,
      "loss": 0.1718
    },
    {
      "step": 4780,
      "loss": 0.1778
    },
    {
      "step": 4785,
      "loss": 0.1832
    },
    {
      "step": 4790,
      "loss": 0.1715
    },
    {
      "step": 4795,
      "loss": 0.1832
    },
    {
      "step": 4800,
      "loss": 0.1692
    },
    {
      "step": 4805,
      "loss": 0.1716
    },
    {
      "step": 4810,
      "loss": 0.1815
    },
    {
      "step": 4815,
      "loss": 0.1733
    },
    {
      "step": 4820,
      "loss": 0.1815
    },
    {
      "step": 4825,
      "loss": 0.1917
    },
    {
      "step": 4830,
      "loss": 0.1724
    },
    {
      "step": 4835,
      "loss": 0.1683
    },
    {
      "step": 4840,
      "loss": 0.1786
    },
    {
      "step": 4845,
      "loss": 0.1823
    },
    {
      "step": 4850,
      "loss": 0.1808
    },
    {
      "step": 4855,
      "loss": 0.1766
    },
    {
      "step": 4860,
      "loss": 0.1759
    },
    {
      "step": 4865,
      "loss": 0.1637
    },
    {
      "step": 4870,
      "loss": 0.1876
    },
    {
      "step": 4875,
      "loss": 0.1865
    },
    {
      "step": 4880,
      "loss": 0.1744
    },
    {
      "step": 4885,
      "loss": 0.1737
    },
    {
      "step": 4890,
      "loss": 0.1786
    },
    {
      "step": 4895,
      "loss": 0.1819
    },
    {
      "step": 4900,
      "loss": 0.1733
    },
    {
      "step": 4905,
      "loss": 0.1818
    },
    {
      "step": 4910,
      "loss": 0.175
    },
    {
      "step": 4915,
      "loss": 0.1776
    },
    {
      "step": 4920,
      "loss": 0.1611
    },
    {
      "step": 4925,
      "loss": 0.1683
    },
    {
      "step": 4930,
      "loss": 0.1757
    },
    {
      "step": 4935,
      "loss": 0.172
    },
    {
      "step": 4940,
      "loss": 0.1721
    },
    {
      "step": 4945,
      "loss": 0.1701
    },
    {
      "step": 4950,
      "loss": 0.1567
    },
    {
      "step": 4955,
      "loss": 0.1849
    },
    {
      "step": 4960,
      "loss": 0.1752
    },
    {
      "step": 4965,
      "loss": 0.1701
    },
    {
      "step": 4970,
      "loss": 0.1809
    },
    {
      "step": 4975,
      "loss": 0.1698
    },
    {
      "step": 4980,
      "loss": 0.1845
    },
    {
      "step": 4985,
      "loss": 0.1697
    },
    {
      "step": 4990,
      "loss": 0.1746
    },
    {
      "step": 4995,
      "loss": 0.1648
    },
    {
      "step": 5000,
      "loss": 0.1735
    },
    {
      "step": 5005,
      "loss": 0.1791
    },
    {
      "step": 5010,
      "loss": 0.1601
    },
    {
      "step": 5015,
      "loss": 0.1786
    },
    {
      "step": 5020,
      "loss": 0.1666
    },
    {
      "step": 5025,
      "loss": 0.1806
    },
    {
      "step": 5030,
      "loss": 0.1787
    },
    {
      "step": 5035,
      "loss": 0.171
    },
    {
      "step": 5040,
      "loss": 0.1732
    },
    {
      "step": 5045,
      "loss": 0.1596
    },
    {
      "step": 5050,
      "loss": 0.18
    },
    {
      "step": 5055,
      "loss": 0.1718
    },
    {
      "step": 5060,
      "loss": 0.1728
    },
    {
      "step": 5065,
      "loss": 0.173
    },
    {
      "step": 5070,
      "loss": 0.1809
    },
    {
      "step": 5075,
      "loss": 0.1675
    },
    {
      "step": 5080,
      "loss": 0.1677
    },
    {
      "step": 5085,
      "loss": 0.183
    },
    {
      "step": 5090,
      "loss": 0.1709
    },
    {
      "step": 5095,
      "loss": 0.1749
    },
    {
      "step": 5100,
      "loss": 0.1838
    },
    {
      "step": 5105,
      "loss": 0.1689
    },
    {
      "step": 5110,
      "loss": 0.1645
    },
    {
      "step": 5115,
      "loss": 0.1686
    },
    {
      "step": 5120,
      "loss": 0.153
    },
    {
      "step": 5125,
      "loss": 0.1842
    },
    {
      "step": 5130,
      "loss": 0.1777
    },
    {
      "step": 5135,
      "loss": 0.1565
    },
    {
      "step": 5140,
      "loss": 0.1934
    },
    {
      "step": 5145,
      "loss": 0.1848
    },
    {
      "step": 5150,
      "loss": 0.1684
    },
    {
      "step": 5155,
      "loss": 0.1668
    },
    {
      "step": 5160,
      "loss": 0.1798
    },
    {
      "step": 5165,
      "loss": 0.1621
    },
    {
      "step": 5170,
      "loss": 0.1674
    },
    {
      "step": 5175,
      "loss": 0.1847
    },
    {
      "step": 5180,
      "loss": 0.1642
    },
    {
      "step": 5185,
      "loss": 0.1638
    },
    {
      "step": 5190,
      "loss": 0.1638
    },
    {
      "step": 5195,
      "loss": 0.1758
    },
    {
      "step": 5200,
      "loss": 0.1739
    },
    {
      "step": 5205,
      "loss": 0.1682
    },
    {
      "step": 5210,
      "loss": 0.176
    },
    {
      "step": 5215,
      "loss": 0.1624
    },
    {
      "step": 5220,
      "loss": 0.1695
    },
    {
      "step": 5225,
      "loss": 0.1632
    },
    {
      "step": 5230,
      "loss": 0.1676
    },
    {
      "step": 5235,
      "loss": 0.1775
    },
    {
      "step": 5240,
      "loss": 0.1819
    },
    {
      "step": 5245,
      "loss": 0.1702
    },
    {
      "step": 5250,
      "loss": 0.1606
    },
    {
      "step": 5255,
      "loss": 0.1725
    },
    {
      "step": 5260,
      "loss": 0.1536
    },
    {
      "step": 5265,
      "loss": 0.1759
    },
    {
      "step": 5270,
      "loss": 0.1682
    },
    {
      "step": 5275,
      "loss": 0.1818
    },
    {
      "step": 5280,
      "loss": 0.1643
    },
    {
      "step": 5285,
      "loss": 0.1832
    },
    {
      "step": 5290,
      "loss": 0.1624
    },
    {
      "step": 5295,
      "loss": 0.173
    },
    {
      "step": 5300,
      "loss": 0.1691
    },
    {
      "step": 5305,
      "loss": 0.1821
    },
    {
      "step": 5310,
      "loss": 0.1816
    },
    {
      "step": 5315,
      "loss": 0.1738
    },
    {
      "step": 5320,
      "loss": 0.1833
    },
    {
      "step": 5325,
      "loss": 0.1886
    },
    {
      "step": 5330,
      "loss": 0.163
    },
    {
      "step": 5335,
      "loss": 0.1668
    },
    {
      "step": 5340,
      "loss": 0.1824
    },
    {
      "step": 5345,
      "loss": 0.1555
    },
    {
      "step": 5350,
      "loss": 0.1816
    },
    {
      "step": 5355,
      "loss": 0.1571
    },
    {
      "step": 5360,
      "loss": 0.171
    },
    {
      "step": 5365,
      "loss": 0.1663
    },
    {
      "step": 5370,
      "loss": 0.1781
    },
    {
      "step": 5375,
      "loss": 0.1769
    },
    {
      "step": 5380,
      "loss": 0.15
    },
    {
      "step": 5385,
      "loss": 0.1655
    },
    {
      "step": 5390,
      "loss": 0.1771
    },
    {
      "step": 5395,
      "loss": 0.1726
    },
    {
      "step": 5400,
      "loss": 0.1637
    },
    {
      "step": 5405,
      "loss": 0.18
    },
    {
      "step": 5410,
      "loss": 0.1613
    },
    {
      "step": 5415,
      "loss": 0.1703
    },
    {
      "step": 5420,
      "loss": 0.1692
    },
    {
      "step": 5425,
      "loss": 0.1832
    },
    {
      "step": 5430,
      "loss": 0.1725
    },
    {
      "step": 5435,
      "loss": 0.167
    },
    {
      "step": 5440,
      "loss": 0.1776
    },
    {
      "step": 5445,
      "loss": 0.1831
    },
    {
      "step": 5450,
      "loss": 0.1813
    },
    {
      "step": 5455,
      "loss": 0.1797
    },
    {
      "step": 5460,
      "loss": 0.1748
    },
    {
      "step": 5465,
      "loss": 0.1715
    },
    {
      "step": 5470,
      "loss": 0.1653
    },
    {
      "step": 5475,
      "loss": 0.1728
    },
    {
      "step": 5480,
      "loss": 0.1739
    },
    {
      "step": 5485,
      "loss": 0.1725
    },
    {
      "step": 5490,
      "loss": 0.1558
    },
    {
      "step": 5495,
      "loss": 0.1751
    },
    {
      "step": 5500,
      "loss": 0.1376
    },
    {
      "step": 5505,
      "loss": 0.165
    },
    {
      "step": 5510,
      "loss": 0.1715
    },
    {
      "step": 5515,
      "loss": 0.1597
    },
    {
      "step": 5520,
      "loss": 0.156
    },
    {
      "step": 5525,
      "loss": 0.1625
    },
    {
      "step": 5530,
      "loss": 0.1664
    },
    {
      "step": 5535,
      "loss": 0.1785
    },
    {
      "step": 5540,
      "loss": 0.167
    },
    {
      "step": 5545,
      "loss": 0.1705
    },
    {
      "step": 5550,
      "loss": 0.1617
    },
    {
      "step": 5555,
      "loss": 0.1586
    },
    {
      "step": 5560,
      "loss": 0.1737
    },
    {
      "step": 5565,
      "loss": 0.1595
    },
    {
      "step": 5570,
      "loss": 0.1808
    },
    {
      "step": 5575,
      "loss": 0.1664
    },
    {
      "step": 5580,
      "loss": 0.154
    },
    {
      "step": 5585,
      "loss": 0.1481
    },
    {
      "step": 5590,
      "loss": 0.1804
    },
    {
      "step": 5595,
      "loss": 0.1647
    },
    {
      "step": 5600,
      "loss": 0.1554
    },
    {
      "step": 5605,
      "loss": 0.1666
    },
    {
      "step": 5610,
      "loss": 0.1684
    },
    {
      "step": 5615,
      "loss": 0.1637
    },
    {
      "step": 5620,
      "loss": 0.1836
    },
    {
      "step": 5625,
      "loss": 0.1672
    },
    {
      "step": 5630,
      "loss": 0.1618
    },
    {
      "step": 5635,
      "loss": 0.1597
    },
    {
      "step": 5640,
      "loss": 0.1669
    },
    {
      "step": 5645,
      "loss": 0.1715
    },
    {
      "step": 5650,
      "loss": 0.1545
    },
    {
      "step": 5655,
      "loss": 0.1496
    },
    {
      "step": 5660,
      "loss": 0.1798
    },
    {
      "step": 5665,
      "loss": 0.167
    },
    {
      "step": 5670,
      "loss": 0.1517
    },
    {
      "step": 5675,
      "loss": 0.1497
    },
    {
      "step": 5680,
      "loss": 0.17
    },
    {
      "step": 5685,
      "loss": 0.1731
    },
    {
      "step": 5690,
      "loss": 0.1642
    },
    {
      "step": 5695,
      "loss": 0.1599
    },
    {
      "step": 5700,
      "loss": 0.177
    },
    {
      "step": 5705,
      "loss": 0.1738
    },
    {
      "step": 5710,
      "loss": 0.1729
    },
    {
      "step": 5715,
      "loss": 0.1623
    },
    {
      "step": 5720,
      "loss": 0.1605
    },
    {
      "step": 5725,
      "loss": 0.1637
    },
    {
      "step": 5730,
      "loss": 0.1712
    },
    {
      "step": 5735,
      "loss": 0.16
    },
    {
      "step": 5740,
      "loss": 0.1554
    },
    {
      "step": 5745,
      "loss": 0.1559
    },
    {
      "step": 5750,
      "loss": 0.1723
    },
    {
      "step": 5755,
      "loss": 0.1595
    },
    {
      "step": 5760,
      "loss": 0.1479
    },
    {
      "step": 5765,
      "loss": 0.1656
    },
    {
      "step": 5770,
      "loss": 0.1623
    },
    {
      "step": 5775,
      "loss": 0.1646
    },
    {
      "step": 5780,
      "loss": 0.1656
    },
    {
      "step": 5785,
      "loss": 0.1594
    },
    {
      "step": 5790,
      "loss": 0.1692
    },
    {
      "step": 5795,
      "loss": 0.1604
    },
    {
      "step": 5800,
      "loss": 0.1675
    },
    {
      "step": 5805,
      "loss": 0.1752
    },
    {
      "step": 5810,
      "loss": 0.1663
    },
    {
      "step": 5815,
      "loss": 0.1494
    },
    {
      "step": 5820,
      "loss": 0.1459
    },
    {
      "step": 5825,
      "loss": 0.1717
    },
    {
      "step": 5830,
      "loss": 0.1666
    },
    {
      "step": 5835,
      "loss": 0.174
    },
    {
      "step": 5840,
      "loss": 0.1646
    },
    {
      "step": 5845,
      "loss": 0.1679
    },
    {
      "step": 5850,
      "loss": 0.1528
    },
    {
      "step": 5855,
      "loss": 0.1499
    },
    {
      "step": 5860,
      "loss": 0.1731
    },
    {
      "step": 5865,
      "loss": 0.1567
    },
    {
      "step": 5870,
      "loss": 0.1809
    },
    {
      "step": 5875,
      "loss": 0.1577
    },
    {
      "step": 5880,
      "loss": 0.1735
    },
    {
      "step": 5885,
      "loss": 0.1596
    },
    {
      "step": 5890,
      "loss": 0.1642
    },
    {
      "step": 5895,
      "loss": 0.1617
    },
    {
      "step": 5900,
      "loss": 0.1732
    },
    {
      "step": 5905,
      "loss": 0.1669
    },
    {
      "step": 5910,
      "loss": 0.1593
    },
    {
      "step": 5915,
      "loss": 0.1762
    },
    {
      "step": 5920,
      "loss": 0.1723
    },
    {
      "step": 5925,
      "loss": 0.161
    },
    {
      "step": 5930,
      "loss": 0.1519
    },
    {
      "step": 5935,
      "loss": 0.1535
    },
    {
      "step": 5940,
      "loss": 0.1581
    },
    {
      "step": 5945,
      "loss": 0.1645
    },
    {
      "step": 5950,
      "loss": 0.1662
    },
    {
      "step": 5955,
      "loss": 0.1669
    },
    {
      "step": 5960,
      "loss": 0.1681
    },
    {
      "step": 5965,
      "loss": 0.1531
    },
    {
      "step": 5970,
      "loss": 0.1506
    },
    {
      "step": 5975,
      "loss": 0.1521
    },
    {
      "step": 5980,
      "loss": 0.1462
    },
    {
      "step": 5985,
      "loss": 0.1617
    },
    {
      "step": 5990,
      "loss": 0.1625
    },
    {
      "step": 5995,
      "loss": 0.1665
    },
    {
      "step": 6000,
      "loss": 0.1767
    },
    {
      "step": 6005,
      "loss": 0.1614
    },
    {
      "step": 6010,
      "loss": 0.1581
    },
    {
      "step": 6015,
      "loss": 0.1681
    },
    {
      "step": 6020,
      "loss": 0.1666
    },
    {
      "step": 6025,
      "loss": 0.1606
    },
    {
      "step": 6030,
      "loss": 0.1755
    },
    {
      "step": 6035,
      "loss": 0.1567
    },
    {
      "step": 6040,
      "loss": 0.1591
    },
    {
      "step": 6045,
      "loss": 0.1591
    },
    {
      "step": 6050,
      "loss": 0.1617
    },
    {
      "step": 6055,
      "loss": 0.1573
    },
    {
      "step": 6060,
      "loss": 0.1626
    },
    {
      "step": 6065,
      "loss": 0.1597
    },
    {
      "step": 6070,
      "loss": 0.1621
    },
    {
      "step": 6075,
      "loss": 0.158
    },
    {
      "step": 6080,
      "loss": 0.1703
    },
    {
      "step": 6085,
      "loss": 0.1738
    },
    {
      "step": 6090,
      "loss": 0.1741
    },
    {
      "step": 6095,
      "loss": 0.1644
    },
    {
      "step": 6100,
      "loss": 0.162
    },
    {
      "step": 6105,
      "loss": 0.1428
    },
    {
      "step": 6110,
      "loss": 0.1632
    },
    {
      "step": 6115,
      "loss": 0.17
    },
    {
      "step": 6120,
      "loss": 0.1549
    },
    {
      "step": 6125,
      "loss": 0.165
    },
    {
      "step": 6130,
      "loss": 0.1473
    },
    {
      "step": 6135,
      "loss": 0.1563
    },
    {
      "step": 6140,
      "loss": 0.1608
    },
    {
      "step": 6145,
      "loss": 0.1526
    },
    {
      "step": 6150,
      "loss": 0.1579
    },
    {
      "step": 6155,
      "loss": 0.1616
    },
    {
      "step": 6160,
      "loss": 0.1616
    },
    {
      "step": 6165,
      "loss": 0.1514
    },
    {
      "step": 6170,
      "loss": 0.1482
    },
    {
      "step": 6175,
      "loss": 0.1535
    },
    {
      "step": 6180,
      "loss": 0.166
    },
    {
      "step": 6185,
      "loss": 0.1473
    },
    {
      "step": 6190,
      "loss": 0.1448
    },
    {
      "step": 6195,
      "loss": 0.1633
    },
    {
      "step": 6200,
      "loss": 0.1613
    },
    {
      "step": 6205,
      "loss": 0.1538
    },
    {
      "step": 6210,
      "loss": 0.1609
    },
    {
      "step": 6215,
      "loss": 0.1772
    },
    {
      "step": 6220,
      "loss": 0.1686
    },
    {
      "step": 6225,
      "loss": 0.1624
    },
    {
      "step": 6230,
      "loss": 0.1582
    },
    {
      "step": 6235,
      "loss": 0.1445
    },
    {
      "step": 6240,
      "loss": 0.1452
    },
    {
      "step": 6245,
      "loss": 0.152
    },
    {
      "step": 6250,
      "loss": 0.163
    },
    {
      "step": 6255,
      "loss": 0.155
    },
    {
      "step": 6260,
      "loss": 0.1546
    },
    {
      "step": 6265,
      "loss": 0.1721
    },
    {
      "step": 6270,
      "loss": 0.1644
    },
    {
      "step": 6275,
      "loss": 0.1618
    },
    {
      "step": 6280,
      "loss": 0.1686
    },
    {
      "step": 6285,
      "loss": 0.1603
    },
    {
      "step": 6290,
      "loss": 0.159
    },
    {
      "step": 6295,
      "loss": 0.1511
    },
    {
      "step": 6300,
      "loss": 0.16
    },
    {
      "step": 6305,
      "loss": 0.1645
    },
    {
      "step": 6310,
      "loss": 0.1622
    },
    {
      "step": 6315,
      "loss": 0.154
    },
    {
      "step": 6320,
      "loss": 0.163
    },
    {
      "step": 6325,
      "loss": 0.1653
    },
    {
      "step": 6330,
      "loss": 0.1728
    },
    {
      "step": 6335,
      "loss": 0.1603
    },
    {
      "step": 6340,
      "loss": 0.1491
    },
    {
      "step": 6345,
      "loss": 0.1664
    },
    {
      "step": 6350,
      "loss": 0.1439
    },
    {
      "step": 6355,
      "loss": 0.1655
    },
    {
      "step": 6360,
      "loss": 0.1545
    },
    {
      "step": 6365,
      "loss": 0.158
    },
    {
      "step": 6370,
      "loss": 0.1697
    },
    {
      "step": 6375,
      "loss": 0.1758
    },
    {
      "step": 6380,
      "loss": 0.1723
    },
    {
      "step": 6385,
      "loss": 0.1585
    },
    {
      "step": 6390,
      "loss": 0.1576
    },
    {
      "step": 6395,
      "loss": 0.1681
    },
    {
      "step": 6400,
      "loss": 0.1703
    },
    {
      "step": 6405,
      "loss": 0.1587
    },
    {
      "step": 6410,
      "loss": 0.1528
    },
    {
      "step": 6415,
      "loss": 0.1577
    },
    {
      "step": 6420,
      "loss": 0.1447
    },
    {
      "step": 6425,
      "loss": 0.1687
    },
    {
      "step": 6430,
      "loss": 0.1411
    },
    {
      "step": 6435,
      "loss": 0.1597
    },
    {
      "step": 6440,
      "loss": 0.1627
    },
    {
      "step": 6445,
      "loss": 0.1665
    },
    {
      "step": 6450,
      "loss": 0.1701
    },
    {
      "step": 6455,
      "loss": 0.158
    },
    {
      "step": 6460,
      "loss": 0.1481
    },
    {
      "step": 6465,
      "loss": 0.1603
    },
    {
      "step": 6470,
      "loss": 0.1583
    },
    {
      "step": 6475,
      "loss": 0.1659
    },
    {
      "step": 6480,
      "loss": 0.1514
    },
    {
      "step": 6485,
      "loss": 0.1561
    },
    {
      "step": 6490,
      "loss": 0.1551
    },
    {
      "step": 6495,
      "loss": 0.1553
    },
    {
      "step": 6500,
      "loss": 0.1426
    },
    {
      "step": 6505,
      "loss": 0.1663
    },
    {
      "step": 6510,
      "loss": 0.1658
    },
    {
      "step": 6515,
      "loss": 0.1464
    },
    {
      "step": 6520,
      "loss": 0.1667
    },
    {
      "step": 6525,
      "loss": 0.1631
    },
    {
      "step": 6530,
      "loss": 0.1693
    },
    {
      "step": 6535,
      "loss": 0.1456
    },
    {
      "step": 6540,
      "loss": 0.1653
    },
    {
      "step": 6545,
      "loss": 0.1562
    },
    {
      "step": 6550,
      "loss": 0.1599
    },
    {
      "step": 6555,
      "loss": 0.1686
    },
    {
      "step": 6560,
      "loss": 0.1378
    },
    {
      "step": 6565,
      "loss": 0.1605
    },
    {
      "step": 6570,
      "loss": 0.1514
    },
    {
      "step": 6575,
      "loss": 0.157
    },
    {
      "step": 6580,
      "loss": 0.1503
    },
    {
      "step": 6585,
      "loss": 0.1461
    },
    {
      "step": 6590,
      "loss": 0.1578
    },
    {
      "step": 6595,
      "loss": 0.16
    },
    {
      "step": 6600,
      "loss": 0.1545
    },
    {
      "step": 6605,
      "loss": 0.1451
    },
    {
      "step": 6610,
      "loss": 0.1565
    },
    {
      "step": 6615,
      "loss": 0.1567
    },
    {
      "step": 6620,
      "loss": 0.1552
    },
    {
      "step": 6625,
      "loss": 0.1613
    },
    {
      "step": 6630,
      "loss": 0.1644
    },
    {
      "step": 6635,
      "loss": 0.1522
    },
    {
      "step": 6640,
      "loss": 0.1606
    },
    {
      "step": 6645,
      "loss": 0.1475
    },
    {
      "step": 6650,
      "loss": 0.1629
    },
    {
      "step": 6655,
      "loss": 0.1685
    },
    {
      "step": 6660,
      "loss": 0.1572
    },
    {
      "step": 6665,
      "loss": 0.1615
    },
    {
      "step": 6670,
      "loss": 0.1641
    },
    {
      "step": 6675,
      "loss": 0.1723
    },
    {
      "step": 6680,
      "loss": 0.1633
    },
    {
      "step": 6685,
      "loss": 0.1583
    },
    {
      "step": 6690,
      "loss": 0.1545
    },
    {
      "step": 6695,
      "loss": 0.159
    },
    {
      "step": 6700,
      "loss": 0.1712
    },
    {
      "step": 6705,
      "loss": 0.1707
    },
    {
      "step": 6710,
      "loss": 0.1646
    },
    {
      "step": 6715,
      "loss": 0.1476
    },
    {
      "step": 6720,
      "loss": 0.1609
    },
    {
      "step": 6725,
      "loss": 0.1718
    },
    {
      "step": 6730,
      "loss": 0.163
    },
    {
      "step": 6735,
      "loss": 0.1569
    },
    {
      "step": 6740,
      "loss": 0.151
    },
    {
      "step": 6745,
      "loss": 0.16
    },
    {
      "step": 6750,
      "loss": 0.1514
    },
    {
      "step": 6755,
      "loss": 0.158
    },
    {
      "step": 6760,
      "loss": 0.1669
    },
    {
      "step": 6765,
      "loss": 0.1592
    },
    {
      "step": 6770,
      "loss": 0.154
    },
    {
      "step": 6775,
      "loss": 0.1556
    },
    {
      "step": 6780,
      "loss": 0.1758
    },
    {
      "step": 6785,
      "loss": 0.1665
    },
    {
      "step": 6790,
      "loss": 0.158
    },
    {
      "step": 6795,
      "loss": 0.1448
    },
    {
      "step": 6800,
      "loss": 0.1665
    },
    {
      "step": 6805,
      "loss": 0.1616
    },
    {
      "step": 6810,
      "loss": 0.1727
    },
    {
      "step": 6815,
      "loss": 0.1383
    },
    {
      "step": 6820,
      "loss": 0.1381
    },
    {
      "step": 6825,
      "loss": 0.1563
    },
    {
      "step": 6830,
      "loss": 0.1672
    },
    {
      "step": 6835,
      "loss": 0.1714
    },
    {
      "step": 6840,
      "loss": 0.1673
    },
    {
      "step": 6845,
      "loss": 0.1523
    },
    {
      "step": 6850,
      "loss": 0.1542
    },
    {
      "step": 6855,
      "loss": 0.1587
    },
    {
      "step": 6860,
      "loss": 0.159
    },
    {
      "step": 6865,
      "loss": 0.1499
    },
    {
      "step": 6870,
      "loss": 0.1608
    },
    {
      "step": 6875,
      "loss": 0.1636
    },
    {
      "step": 6880,
      "loss": 0.1724
    },
    {
      "step": 6885,
      "loss": 0.1581
    },
    {
      "step": 6890,
      "loss": 0.1589
    },
    {
      "step": 6895,
      "loss": 0.156
    },
    {
      "step": 6900,
      "loss": 0.1497
    },
    {
      "step": 6905,
      "loss": 0.1566
    },
    {
      "step": 6910,
      "loss": 0.1605
    },
    {
      "step": 6915,
      "loss": 0.1594
    },
    {
      "step": 6920,
      "loss": 0.1446
    },
    {
      "step": 6925,
      "loss": 0.1585
    },
    {
      "step": 6930,
      "loss": 0.166
    },
    {
      "step": 6935,
      "loss": 0.1479
    },
    {
      "step": 6940,
      "loss": 0.175
    },
    {
      "step": 6945,
      "loss": 0.1541
    },
    {
      "step": 6950,
      "loss": 0.1512
    },
    {
      "step": 6955,
      "loss": 0.1587
    },
    {
      "step": 6960,
      "loss": 0.159
    },
    {
      "step": 6965,
      "loss": 0.1658
    },
    {
      "step": 6970,
      "loss": 0.1551
    },
    {
      "step": 6975,
      "loss": 0.1518
    },
    {
      "step": 6980,
      "loss": 0.1628
    },
    {
      "step": 6985,
      "loss": 0.1449
    },
    {
      "step": 6990,
      "loss": 0.1664
    },
    {
      "step": 6995,
      "loss": 0.1584
    },
    {
      "step": 7000,
      "loss": 0.1459
    },
    {
      "step": 7005,
      "loss": 0.1569
    },
    {
      "step": 7010,
      "loss": 0.1517
    },
    {
      "step": 7015,
      "loss": 0.1704
    },
    {
      "step": 7020,
      "loss": 0.1449
    },
    {
      "step": 7025,
      "loss": 0.1555
    },
    {
      "step": 7030,
      "loss": 0.1594
    },
    {
      "step": 7035,
      "loss": 0.1594
    },
    {
      "step": 7040,
      "loss": 0.1619
    },
    {
      "step": 7045,
      "loss": 0.1565
    },
    {
      "step": 7050,
      "loss": 0.1722
    },
    {
      "step": 7055,
      "loss": 0.1507
    },
    {
      "step": 7060,
      "loss": 0.1479
    },
    {
      "step": 7065,
      "loss": 0.1624
    },
    {
      "step": 7070,
      "loss": 0.1544
    },
    {
      "step": 7075,
      "loss": 0.1548
    },
    {
      "step": 7080,
      "loss": 0.1424
    },
    {
      "step": 7085,
      "loss": 0.1392
    },
    {
      "step": 7090,
      "loss": 0.1545
    },
    {
      "step": 7095,
      "loss": 0.1572
    },
    {
      "step": 7100,
      "loss": 0.1634
    },
    {
      "step": 7105,
      "loss": 0.1609
    },
    {
      "step": 7110,
      "loss": 0.1583
    },
    {
      "step": 7115,
      "loss": 0.1492
    },
    {
      "step": 7120,
      "loss": 0.1585
    },
    {
      "step": 7125,
      "loss": 0.1517
    },
    {
      "step": 7130,
      "loss": 0.1437
    },
    {
      "step": 7135,
      "loss": 0.1615
    },
    {
      "step": 7140,
      "loss": 0.1412
    },
    {
      "step": 7145,
      "loss": 0.1459
    },
    {
      "step": 7150,
      "loss": 0.167
    },
    {
      "step": 7155,
      "loss": 0.157
    },
    {
      "step": 7160,
      "loss": 0.1506
    },
    {
      "step": 7165,
      "loss": 0.1383
    },
    {
      "step": 7170,
      "loss": 0.1531
    },
    {
      "step": 7175,
      "loss": 0.1504
    },
    {
      "step": 7180,
      "loss": 0.1514
    },
    {
      "step": 7185,
      "loss": 0.1469
    },
    {
      "step": 7190,
      "loss": 0.1521
    },
    {
      "step": 7195,
      "loss": 0.1699
    },
    {
      "step": 7200,
      "loss": 0.1544
    },
    {
      "step": 7205,
      "loss": 0.1609
    },
    {
      "step": 7210,
      "loss": 0.1563
    },
    {
      "step": 7215,
      "loss": 0.167
    },
    {
      "step": 7220,
      "loss": 0.1533
    },
    {
      "step": 7225,
      "loss": 0.1662
    },
    {
      "step": 7230,
      "loss": 0.1541
    },
    {
      "step": 7235,
      "loss": 0.1615
    },
    {
      "step": 7240,
      "loss": 0.1565
    },
    {
      "step": 7245,
      "loss": 0.1557
    },
    {
      "step": 7250,
      "loss": 0.1562
    },
    {
      "step": 7255,
      "loss": 0.1382
    },
    {
      "step": 7260,
      "loss": 0.1479
    },
    {
      "step": 7265,
      "loss": 0.1637
    },
    {
      "step": 7270,
      "loss": 0.1478
    },
    {
      "step": 7275,
      "loss": 0.1464
    },
    {
      "step": 7280,
      "loss": 0.1526
    },
    {
      "step": 7285,
      "loss": 0.1471
    },
    {
      "step": 7290,
      "loss": 0.1552
    },
    {
      "step": 7295,
      "loss": 0.1654
    },
    {
      "step": 7300,
      "loss": 0.1589
    },
    {
      "step": 7305,
      "loss": 0.1639
    },
    {
      "step": 7310,
      "loss": 0.1563
    },
    {
      "step": 7315,
      "loss": 0.1441
    },
    {
      "step": 7320,
      "loss": 0.1595
    },
    {
      "step": 7325,
      "loss": 0.1633
    },
    {
      "step": 7330,
      "loss": 0.1686
    },
    {
      "step": 7335,
      "loss": 0.1517
    },
    {
      "step": 7340,
      "loss": 0.1576
    },
    {
      "step": 7345,
      "loss": 0.1629
    },
    {
      "step": 7350,
      "loss": 0.1497
    },
    {
      "step": 7355,
      "loss": 0.1472
    },
    {
      "step": 7360,
      "loss": 0.1514
    },
    {
      "step": 7365,
      "loss": 0.1507
    },
    {
      "step": 7370,
      "loss": 0.162
    },
    {
      "step": 7375,
      "loss": 0.1577
    },
    {
      "step": 7380,
      "loss": 0.156
    },
    {
      "step": 7385,
      "loss": 0.1619
    },
    {
      "step": 7390,
      "loss": 0.1532
    },
    {
      "step": 7395,
      "loss": 0.1623
    },
    {
      "step": 7400,
      "loss": 0.1629
    },
    {
      "step": 7405,
      "loss": 0.168
    },
    {
      "step": 7410,
      "loss": 0.1617
    },
    {
      "step": 7415,
      "loss": 0.1659
    },
    {
      "step": 7420,
      "loss": 0.1577
    },
    {
      "step": 7425,
      "loss": 0.1589
    },
    {
      "step": 7430,
      "loss": 0.1596
    },
    {
      "step": 7435,
      "loss": 0.1521
    },
    {
      "step": 7440,
      "loss": 0.149
    },
    {
      "step": 7445,
      "loss": 0.1557
    },
    {
      "step": 7450,
      "loss": 0.1532
    },
    {
      "step": 7455,
      "loss": 0.1591
    },
    {
      "step": 7460,
      "loss": 0.1724
    },
    {
      "step": 7465,
      "loss": 0.1523
    },
    {
      "step": 7470,
      "loss": 0.1519
    },
    {
      "step": 7475,
      "loss": 0.1458
    },
    {
      "step": 7480,
      "loss": 0.1612
    },
    {
      "step": 7485,
      "loss": 0.1611
    },
    {
      "step": 7490,
      "loss": 0.1677
    },
    {
      "step": 7495,
      "loss": 0.151
    },
    {
      "step": 7500,
      "loss": 0.1619
    },
    {
      "step": 7505,
      "loss": 0.1474
    },
    {
      "step": 7510,
      "loss": 0.1583
    },
    {
      "step": 7515,
      "loss": 0.1592
    },
    {
      "step": 7520,
      "loss": 0.1473
    },
    {
      "step": 7525,
      "loss": 0.1453
    },
    {
      "step": 7530,
      "loss": 0.1548
    },
    {
      "step": 7535,
      "loss": 0.1533
    },
    {
      "step": 7540,
      "loss": 0.1556
    },
    {
      "step": 7545,
      "loss": 0.154
    },
    {
      "step": 7550,
      "loss": 0.1508
    },
    {
      "step": 7555,
      "loss": 0.1511
    },
    {
      "step": 7560,
      "loss": 0.1584
    },
    {
      "step": 7565,
      "loss": 0.1421
    },
    {
      "step": 7570,
      "loss": 0.1514
    },
    {
      "step": 7575,
      "loss": 0.1525
    },
    {
      "step": 7580,
      "loss": 0.1665
    },
    {
      "step": 7585,
      "loss": 0.1625
    },
    {
      "step": 7590,
      "loss": 0.1493
    },
    {
      "step": 7595,
      "loss": 0.1404
    },
    {
      "step": 7600,
      "loss": 0.1526
    },
    {
      "step": 7605,
      "loss": 0.1407
    },
    {
      "step": 7610,
      "loss": 0.1545
    },
    {
      "step": 7615,
      "loss": 0.1587
    },
    {
      "step": 7620,
      "loss": 0.1573
    },
    {
      "step": 7625,
      "loss": 0.1583
    },
    {
      "step": 7630,
      "loss": 0.1504
    },
    {
      "step": 7635,
      "loss": 0.1474
    },
    {
      "step": 7640,
      "loss": 0.1616
    },
    {
      "step": 7645,
      "loss": 0.1448
    },
    {
      "step": 7650,
      "loss": 0.1538
    },
    {
      "step": 7655,
      "loss": 0.149
    },
    {
      "step": 7660,
      "loss": 0.1525
    },
    {
      "step": 7665,
      "loss": 0.149
    },
    {
      "step": 7670,
      "loss": 0.154
    },
    {
      "step": 7675,
      "loss": 0.1632
    },
    {
      "step": 7680,
      "loss": 0.1556
    },
    {
      "step": 7685,
      "loss": 0.1558
    },
    {
      "step": 7690,
      "loss": 0.1474
    },
    {
      "step": 7695,
      "loss": 0.1541
    },
    {
      "step": 7700,
      "loss": 0.1591
    },
    {
      "step": 7705,
      "loss": 0.1483
    },
    {
      "step": 7710,
      "loss": 0.1445
    },
    {
      "step": 7715,
      "loss": 0.1578
    },
    {
      "step": 7720,
      "loss": 0.1611
    },
    {
      "step": 7725,
      "loss": 0.1571
    },
    {
      "step": 7730,
      "loss": 0.1411
    },
    {
      "step": 7735,
      "loss": 0.146
    },
    {
      "step": 7740,
      "loss": 0.1684
    },
    {
      "step": 7745,
      "loss": 0.1535
    },
    {
      "step": 7750,
      "loss": 0.1503
    },
    {
      "step": 7755,
      "loss": 0.1573
    },
    {
      "step": 7760,
      "loss": 0.1464
    },
    {
      "step": 7765,
      "loss": 0.1581
    },
    {
      "step": 7770,
      "loss": 0.1535
    },
    {
      "step": 7775,
      "loss": 0.1476
    },
    {
      "step": 7780,
      "loss": 0.1562
    },
    {
      "step": 7785,
      "loss": 0.1522
    },
    {
      "step": 7790,
      "loss": 0.1547
    },
    {
      "step": 7795,
      "loss": 0.1562
    },
    {
      "step": 7800,
      "loss": 0.1653
    },
    {
      "step": 7805,
      "loss": 0.1617
    },
    {
      "step": 7810,
      "loss": 0.1532
    },
    {
      "step": 7815,
      "loss": 0.1608
    },
    {
      "step": 7820,
      "loss": 0.1664
    },
    {
      "step": 7825,
      "loss": 0.1506
    },
    {
      "step": 7830,
      "loss": 0.1441
    },
    {
      "step": 7835,
      "loss": 0.1619
    },
    {
      "step": 7840,
      "loss": 0.1609
    },
    {
      "step": 7845,
      "loss": 0.1615
    },
    {
      "step": 7850,
      "loss": 0.1383
    },
    {
      "step": 7855,
      "loss": 0.1483
    },
    {
      "step": 7860,
      "loss": 0.152
    },
    {
      "step": 7865,
      "loss": 0.1551
    },
    {
      "step": 7870,
      "loss": 0.1534
    },
    {
      "step": 7875,
      "loss": 0.1554
    },
    {
      "step": 7880,
      "loss": 0.1555
    },
    {
      "step": 7885,
      "loss": 0.1468
    },
    {
      "step": 7890,
      "loss": 0.1428
    },
    {
      "step": 7895,
      "loss": 0.1387
    },
    {
      "step": 7900,
      "loss": 0.1497
    },
    {
      "step": 7905,
      "loss": 0.1544
    },
    {
      "step": 7910,
      "loss": 0.1485
    },
    {
      "step": 7915,
      "loss": 0.1573
    },
    {
      "step": 7920,
      "loss": 0.1562
    },
    {
      "step": 7925,
      "loss": 0.1342
    },
    {
      "step": 7930,
      "loss": 0.1549
    },
    {
      "step": 7935,
      "loss": 0.1525
    },
    {
      "step": 7940,
      "loss": 0.1575
    },
    {
      "step": 7945,
      "loss": 0.1512
    },
    {
      "step": 7950,
      "loss": 0.1396
    },
    {
      "step": 7955,
      "loss": 0.1625
    },
    {
      "step": 7960,
      "loss": 0.1496
    },
    {
      "step": 7965,
      "loss": 0.1532
    },
    {
      "step": 7970,
      "loss": 0.1499
    },
    {
      "step": 7975,
      "loss": 0.1474
    },
    {
      "step": 7980,
      "loss": 0.1503
    },
    {
      "step": 7985,
      "loss": 0.1469
    },
    {
      "step": 7990,
      "loss": 0.154
    },
    {
      "step": 7995,
      "loss": 0.1526
    },
    {
      "step": 8000,
      "loss": 0.1526
    },
    {
      "step": 8005,
      "loss": 0.1531
    },
    {
      "step": 8010,
      "loss": 0.1525
    },
    {
      "step": 8015,
      "loss": 0.1466
    },
    {
      "step": 8020,
      "loss": 0.1531
    },
    {
      "step": 8025,
      "loss": 0.1642
    },
    {
      "step": 8030,
      "loss": 0.1458
    },
    {
      "step": 8035,
      "loss": 0.1463
    },
    {
      "step": 8040,
      "loss": 0.149
    },
    {
      "step": 8045,
      "loss": 0.1519
    },
    {
      "step": 8050,
      "loss": 0.1488
    },
    {
      "step": 8055,
      "loss": 0.1611
    },
    {
      "step": 8060,
      "loss": 0.1535
    },
    {
      "step": 8065,
      "loss": 0.1422
    },
    {
      "step": 8070,
      "loss": 0.1541
    },
    {
      "step": 8075,
      "loss": 0.1444
    },
    {
      "step": 8080,
      "loss": 0.1496
    },
    {
      "step": 8085,
      "loss": 0.1479
    },
    {
      "step": 8090,
      "loss": 0.1524
    },
    {
      "step": 8095,
      "loss": 0.1497
    },
    {
      "step": 8100,
      "loss": 0.1464
    },
    {
      "step": 8105,
      "loss": 0.1558
    },
    {
      "step": 8110,
      "loss": 0.1567
    },
    {
      "step": 8115,
      "loss": 0.1638
    },
    {
      "step": 8120,
      "loss": 0.1506
    },
    {
      "step": 8125,
      "loss": 0.1543
    },
    {
      "step": 8130,
      "loss": 0.1397
    },
    {
      "step": 8135,
      "loss": 0.1486
    },
    {
      "step": 8140,
      "loss": 0.1661
    },
    {
      "step": 8145,
      "loss": 0.1495
    },
    {
      "step": 8150,
      "loss": 0.1481
    },
    {
      "step": 8155,
      "loss": 0.1497
    },
    {
      "step": 8160,
      "loss": 0.1711
    },
    {
      "step": 8165,
      "loss": 0.1493
    },
    {
      "step": 8170,
      "loss": 0.1536
    },
    {
      "step": 8175,
      "loss": 0.1435
    },
    {
      "step": 8180,
      "loss": 0.1528
    },
    {
      "step": 8185,
      "loss": 0.1582
    },
    {
      "step": 8190,
      "loss": 0.1463
    },
    {
      "step": 8195,
      "loss": 0.157
    },
    {
      "step": 8200,
      "loss": 0.1471
    },
    {
      "step": 8205,
      "loss": 0.1521
    },
    {
      "step": 8210,
      "loss": 0.1573
    },
    {
      "step": 8215,
      "loss": 0.1669
    },
    {
      "step": 8220,
      "loss": 0.1528
    },
    {
      "step": 8225,
      "loss": 0.1528
    },
    {
      "step": 8230,
      "loss": 0.1492
    },
    {
      "step": 8235,
      "loss": 0.1376
    },
    {
      "step": 8240,
      "loss": 0.1325
    },
    {
      "step": 8245,
      "loss": 0.1555
    },
    {
      "step": 8250,
      "loss": 0.1406
    },
    {
      "step": 8255,
      "loss": 0.156
    },
    {
      "step": 8260,
      "loss": 0.1468
    },
    {
      "step": 8265,
      "loss": 0.1514
    },
    {
      "step": 8270,
      "loss": 0.1496
    },
    {
      "step": 8275,
      "loss": 0.1606
    },
    {
      "step": 8280,
      "loss": 0.1521
    },
    {
      "step": 8285,
      "loss": 0.1526
    },
    {
      "step": 8290,
      "loss": 0.1449
    },
    {
      "step": 8295,
      "loss": 0.1595
    },
    {
      "step": 8300,
      "loss": 0.1515
    },
    {
      "step": 8305,
      "loss": 0.1486
    },
    {
      "step": 8310,
      "loss": 0.1328
    },
    {
      "step": 8315,
      "loss": 0.1573
    },
    {
      "step": 8320,
      "loss": 0.1701
    },
    {
      "step": 8325,
      "loss": 0.1537
    },
    {
      "step": 8330,
      "loss": 0.1601
    },
    {
      "step": 8335,
      "loss": 0.156
    },
    {
      "step": 8340,
      "loss": 0.1425
    },
    {
      "step": 8345,
      "loss": 0.1454
    },
    {
      "step": 8350,
      "loss": 0.1553
    },
    {
      "step": 8355,
      "loss": 0.161
    },
    {
      "step": 8360,
      "loss": 0.1541
    },
    {
      "step": 8365,
      "loss": 0.1649
    },
    {
      "step": 8370,
      "loss": 0.1428
    },
    {
      "step": 8375,
      "loss": 0.1512
    },
    {
      "step": 8380,
      "loss": 0.1524
    },
    {
      "step": 8385,
      "loss": 0.163
    },
    {
      "step": 8390,
      "loss": 0.1511
    },
    {
      "step": 8395,
      "loss": 0.1441
    },
    {
      "step": 8400,
      "loss": 0.1401
    },
    {
      "step": 8405,
      "loss": 0.1583
    },
    {
      "step": 8410,
      "loss": 0.1397
    },
    {
      "step": 8415,
      "loss": 0.1472
    },
    {
      "step": 8420,
      "loss": 0.1526
    },
    {
      "step": 8425,
      "loss": 0.1493
    },
    {
      "step": 8430,
      "loss": 0.1538
    },
    {
      "step": 8435,
      "loss": 0.1428
    },
    {
      "step": 8440,
      "loss": 0.1515
    },
    {
      "step": 8445,
      "loss": 0.1505
    },
    {
      "step": 8450,
      "loss": 0.1483
    },
    {
      "step": 8455,
      "loss": 0.1542
    },
    {
      "step": 8460,
      "loss": 0.1427
    },
    {
      "step": 8465,
      "loss": 0.1512
    },
    {
      "step": 8470,
      "loss": 0.1421
    },
    {
      "step": 8475,
      "loss": 0.1454
    },
    {
      "step": 8480,
      "loss": 0.1423
    },
    {
      "step": 8485,
      "loss": 0.1555
    },
    {
      "step": 8490,
      "loss": 0.1553
    },
    {
      "step": 8495,
      "loss": 0.1463
    },
    {
      "step": 8500,
      "loss": 0.1444
    },
    {
      "step": 8505,
      "loss": 0.1647
    },
    {
      "step": 8510,
      "loss": 0.1489
    },
    {
      "step": 8515,
      "loss": 0.1447
    },
    {
      "step": 8520,
      "loss": 0.152
    },
    {
      "step": 8525,
      "loss": 0.1546
    },
    {
      "step": 8530,
      "loss": 0.1325
    },
    {
      "step": 8535,
      "loss": 0.1488
    },
    {
      "step": 8540,
      "loss": 0.1454
    },
    {
      "step": 8545,
      "loss": 0.1451
    },
    {
      "step": 8550,
      "loss": 0.1453
    },
    {
      "step": 8555,
      "loss": 0.1518
    },
    {
      "step": 8560,
      "loss": 0.144
    },
    {
      "step": 8565,
      "loss": 0.1582
    },
    {
      "step": 8570,
      "loss": 0.1477
    },
    {
      "step": 8575,
      "loss": 0.1607
    },
    {
      "step": 8580,
      "loss": 0.1355
    },
    {
      "step": 8585,
      "loss": 0.148
    },
    {
      "step": 8590,
      "loss": 0.1538
    },
    {
      "step": 8595,
      "loss": 0.1564
    },
    {
      "step": 8600,
      "loss": 0.1502
    },
    {
      "step": 8605,
      "loss": 0.1558
    },
    {
      "step": 8610,
      "loss": 0.1668
    },
    {
      "step": 8615,
      "loss": 0.1503
    },
    {
      "step": 8620,
      "loss": 0.1498
    },
    {
      "step": 8625,
      "loss": 0.1445
    },
    {
      "step": 8630,
      "loss": 0.1535
    },
    {
      "step": 8635,
      "loss": 0.1533
    },
    {
      "step": 8640,
      "loss": 0.1485
    },
    {
      "step": 8645,
      "loss": 0.1605
    },
    {
      "step": 8650,
      "loss": 0.1528
    },
    {
      "step": 8655,
      "loss": 0.1377
    },
    {
      "step": 8660,
      "loss": 0.1474
    },
    {
      "step": 8665,
      "loss": 0.1457
    },
    {
      "step": 8670,
      "loss": 0.132
    },
    {
      "step": 8675,
      "loss": 0.1442
    },
    {
      "step": 8680,
      "loss": 0.1458
    },
    {
      "step": 8685,
      "loss": 0.1418
    },
    {
      "step": 8690,
      "loss": 0.1538
    },
    {
      "step": 8695,
      "loss": 0.144
    },
    {
      "step": 8700,
      "loss": 0.1549
    },
    {
      "step": 8705,
      "loss": 0.1639
    },
    {
      "step": 8710,
      "loss": 0.1412
    },
    {
      "step": 8715,
      "loss": 0.1607
    },
    {
      "step": 8720,
      "loss": 0.156
    },
    {
      "step": 8725,
      "loss": 0.1581
    },
    {
      "step": 8730,
      "loss": 0.1601
    },
    {
      "step": 8735,
      "loss": 0.1595
    },
    {
      "step": 8740,
      "loss": 0.1456
    },
    {
      "step": 8745,
      "loss": 0.1467
    },
    {
      "step": 8750,
      "loss": 0.156
    },
    {
      "step": 8755,
      "loss": 0.1605
    },
    {
      "step": 8760,
      "loss": 0.1617
    },
    {
      "step": 8765,
      "loss": 0.1544
    },
    {
      "step": 8770,
      "loss": 0.1445
    },
    {
      "step": 8775,
      "loss": 0.1444
    },
    {
      "step": 8780,
      "loss": 0.1564
    },
    {
      "step": 8785,
      "loss": 0.1576
    },
    {
      "step": 8790,
      "loss": 0.1508
    },
    {
      "step": 8795,
      "loss": 0.1611
    },
    {
      "step": 8800,
      "loss": 0.1535
    },
    {
      "step": 8805,
      "loss": 0.1467
    },
    {
      "step": 8810,
      "loss": 0.1632
    },
    {
      "step": 8815,
      "loss": 0.1362
    },
    {
      "step": 8820,
      "loss": 0.1425
    },
    {
      "step": 8825,
      "loss": 0.1532
    },
    {
      "step": 8830,
      "loss": 0.1471
    },
    {
      "step": 8835,
      "loss": 0.1371
    },
    {
      "step": 8840,
      "loss": 0.1461
    },
    {
      "step": 8845,
      "loss": 0.1527
    },
    {
      "step": 8850,
      "loss": 0.153
    },
    {
      "step": 8855,
      "loss": 0.1407
    },
    {
      "step": 8860,
      "loss": 0.1324
    },
    {
      "step": 8865,
      "loss": 0.1474
    },
    {
      "step": 8870,
      "loss": 0.1399
    },
    {
      "step": 8875,
      "loss": 0.1529
    },
    {
      "step": 8880,
      "loss": 0.1485
    },
    {
      "step": 8885,
      "loss": 0.1556
    },
    {
      "step": 8890,
      "loss": 0.1506
    },
    {
      "step": 8895,
      "loss": 0.1556
    },
    {
      "step": 8900,
      "loss": 0.1562
    },
    {
      "step": 8905,
      "loss": 0.1478
    },
    {
      "step": 8910,
      "loss": 0.1644
    },
    {
      "step": 8915,
      "loss": 0.1558
    },
    {
      "step": 8920,
      "loss": 0.1501
    },
    {
      "step": 8925,
      "loss": 0.145
    },
    {
      "step": 8930,
      "loss": 0.157
    },
    {
      "step": 8935,
      "loss": 0.1541
    },
    {
      "step": 8940,
      "loss": 0.1625
    },
    {
      "step": 8945,
      "loss": 0.1527
    },
    {
      "step": 8950,
      "loss": 0.1462
    },
    {
      "step": 8955,
      "loss": 0.1382
    },
    {
      "step": 8960,
      "loss": 0.1587
    },
    {
      "step": 8965,
      "loss": 0.1714
    },
    {
      "step": 8970,
      "loss": 0.1449
    },
    {
      "step": 8975,
      "loss": 0.1315
    },
    {
      "step": 8980,
      "loss": 0.1579
    },
    {
      "step": 8985,
      "loss": 0.1576
    },
    {
      "step": 8990,
      "loss": 0.156
    },
    {
      "step": 8995,
      "loss": 0.1527
    },
    {
      "step": 9000,
      "loss": 0.1399
    },
    {
      "step": 9005,
      "loss": 0.1482
    },
    {
      "step": 9010,
      "loss": 0.1354
    },
    {
      "step": 9015,
      "loss": 0.1542
    },
    {
      "step": 9020,
      "loss": 0.1536
    },
    {
      "step": 9025,
      "loss": 0.1566
    },
    {
      "step": 9030,
      "loss": 0.158
    },
    {
      "step": 9035,
      "loss": 0.1559
    },
    {
      "step": 9040,
      "loss": 0.1587
    },
    {
      "step": 9045,
      "loss": 0.1556
    },
    {
      "step": 9050,
      "loss": 0.1539
    },
    {
      "step": 9055,
      "loss": 0.1534
    },
    {
      "step": 9060,
      "loss": 0.1518
    },
    {
      "step": 9065,
      "loss": 0.1231
    },
    {
      "step": 9070,
      "loss": 0.1471
    },
    {
      "step": 9075,
      "loss": 0.1496
    },
    {
      "step": 9080,
      "loss": 0.1553
    },
    {
      "step": 9085,
      "loss": 0.1533
    },
    {
      "step": 9090,
      "loss": 0.1521
    },
    {
      "step": 9095,
      "loss": 0.1579
    },
    {
      "step": 9100,
      "loss": 0.1518
    },
    {
      "step": 9105,
      "loss": 0.1533
    },
    {
      "step": 9110,
      "loss": 0.1588
    },
    {
      "step": 9115,
      "loss": 0.1494
    },
    {
      "step": 9120,
      "loss": 0.1519
    },
    {
      "step": 9125,
      "loss": 0.154
    },
    {
      "step": 9130,
      "loss": 0.1611
    },
    {
      "step": 9135,
      "loss": 0.1524
    },
    {
      "step": 9140,
      "loss": 0.1633
    },
    {
      "step": 9145,
      "loss": 0.1448
    },
    {
      "step": 9150,
      "loss": 0.1455
    },
    {
      "step": 9155,
      "loss": 0.1486
    },
    {
      "step": 9160,
      "loss": 0.1536
    },
    {
      "step": 9165,
      "loss": 0.1486
    },
    {
      "step": 9170,
      "loss": 0.1476
    },
    {
      "step": 9175,
      "loss": 0.1531
    },
    {
      "step": 9180,
      "loss": 0.1445
    },
    {
      "step": 9185,
      "loss": 0.1502
    },
    {
      "step": 9190,
      "loss": 0.1444
    },
    {
      "step": 9195,
      "loss": 0.143
    },
    {
      "step": 9200,
      "loss": 0.1588
    },
    {
      "step": 9205,
      "loss": 0.15
    },
    {
      "step": 9210,
      "loss": 0.1493
    },
    {
      "step": 9215,
      "loss": 0.1619
    },
    {
      "step": 9220,
      "loss": 0.1424
    },
    {
      "step": 9225,
      "loss": 0.1486
    },
    {
      "step": 9230,
      "loss": 0.1555
    },
    {
      "step": 9235,
      "loss": 0.1598
    },
    {
      "step": 9240,
      "loss": 0.1554
    },
    {
      "step": 9245,
      "loss": 0.1552
    },
    {
      "step": 9250,
      "loss": 0.1371
    },
    {
      "step": 9255,
      "loss": 0.1604
    },
    {
      "step": 9260,
      "loss": 0.1658
    },
    {
      "step": 9265,
      "loss": 0.1525
    },
    {
      "step": 9270,
      "loss": 0.1715
    },
    {
      "step": 9275,
      "loss": 0.1555
    },
    {
      "step": 9280,
      "loss": 0.1642
    },
    {
      "step": 9285,
      "loss": 0.1476
    },
    {
      "step": 9290,
      "loss": 0.1518
    },
    {
      "step": 9295,
      "loss": 0.15
    },
    {
      "step": 9300,
      "loss": 0.1694
    },
    {
      "step": 9305,
      "loss": 0.1477
    },
    {
      "step": 9310,
      "loss": 0.1455
    },
    {
      "step": 9315,
      "loss": 0.1563
    },
    {
      "step": 9320,
      "loss": 0.1554
    },
    {
      "step": 9325,
      "loss": 0.1535
    },
    {
      "step": 9330,
      "loss": 0.14
    },
    {
      "step": 9335,
      "loss": 0.1521
    },
    {
      "step": 9340,
      "loss": 0.1516
    },
    {
      "step": 9345,
      "loss": 0.1617
    },
    {
      "step": 9350,
      "loss": 0.1539
    },
    {
      "step": 9355,
      "loss": 0.1541
    },
    {
      "step": 9360,
      "loss": 0.1356
    },
    {
      "step": 9365,
      "loss": 0.1368
    },
    {
      "step": 9370,
      "loss": 0.1432
    },
    {
      "step": 9375,
      "loss": 0.1485
    },
    {
      "step": 9380,
      "loss": 0.1512
    },
    {
      "step": 9385,
      "loss": 0.1638
    },
    {
      "step": 9390,
      "loss": 0.1445
    },
    {
      "step": 9395,
      "loss": 0.1418
    },
    {
      "step": 9400,
      "loss": 0.1444
    },
    {
      "step": 9405,
      "loss": 0.1584
    },
    {
      "step": 9410,
      "loss": 0.148
    },
    {
      "step": 9415,
      "loss": 0.1482
    },
    {
      "step": 9420,
      "loss": 0.1521
    },
    {
      "step": 9425,
      "loss": 0.137
    },
    {
      "step": 9430,
      "loss": 0.149
    },
    {
      "step": 9435,
      "loss": 0.156
    },
    {
      "step": 9440,
      "loss": 0.1474
    },
    {
      "step": 9445,
      "loss": 0.154
    },
    {
      "step": 9450,
      "loss": 0.1547
    },
    {
      "step": 9455,
      "loss": 0.1509
    },
    {
      "step": 9460,
      "loss": 0.1448
    },
    {
      "step": 9465,
      "loss": 0.157
    },
    {
      "step": 9470,
      "loss": 0.1475
    },
    {
      "step": 9475,
      "loss": 0.1337
    },
    {
      "step": 9480,
      "loss": 0.153
    },
    {
      "step": 9485,
      "loss": 0.1486
    },
    {
      "step": 9490,
      "loss": 0.1439
    },
    {
      "step": 9495,
      "loss": 0.145
    },
    {
      "step": 9500,
      "loss": 0.1506
    },
    {
      "step": 9505,
      "loss": 0.1482
    },
    {
      "step": 9510,
      "loss": 0.1544
    },
    {
      "step": 9515,
      "loss": 0.1423
    },
    {
      "step": 9520,
      "loss": 0.1428
    },
    {
      "step": 9525,
      "loss": 0.1508
    },
    {
      "step": 9530,
      "loss": 0.1517
    },
    {
      "step": 9535,
      "loss": 0.1314
    },
    {
      "step": 9540,
      "loss": 0.14
    },
    {
      "step": 9545,
      "loss": 0.1372
    },
    {
      "step": 9550,
      "loss": 0.1403
    },
    {
      "step": 9555,
      "loss": 0.1588
    },
    {
      "step": 9560,
      "loss": 0.1443
    },
    {
      "step": 9565,
      "loss": 0.1577
    },
    {
      "step": 9570,
      "loss": 0.1603
    },
    {
      "step": 9575,
      "loss": 0.1577
    },
    {
      "step": 9580,
      "loss": 0.1556
    },
    {
      "step": 9585,
      "loss": 0.1477
    },
    {
      "step": 9590,
      "loss": 0.1375
    },
    {
      "step": 9595,
      "loss": 0.1662
    },
    {
      "step": 9600,
      "loss": 0.1493
    },
    {
      "step": 9605,
      "loss": 0.1515
    },
    {
      "step": 9610,
      "loss": 0.1555
    },
    {
      "step": 9615,
      "loss": 0.1446
    },
    {
      "step": 9620,
      "loss": 0.1621
    },
    {
      "step": 9625,
      "loss": 0.142
    },
    {
      "step": 9630,
      "loss": 0.1389
    },
    {
      "step": 9635,
      "loss": 0.1334
    },
    {
      "step": 9640,
      "loss": 0.1536
    },
    {
      "step": 9645,
      "loss": 0.1613
    },
    {
      "step": 9650,
      "loss": 0.1576
    },
    {
      "step": 9655,
      "loss": 0.1524
    },
    {
      "step": 9660,
      "loss": 0.1424
    },
    {
      "step": 9665,
      "loss": 0.1365
    },
    {
      "step": 9670,
      "loss": 0.1431
    },
    {
      "step": 9675,
      "loss": 0.157
    },
    {
      "step": 9680,
      "loss": 0.1439
    },
    {
      "step": 9685,
      "loss": 0.1442
    },
    {
      "step": 9690,
      "loss": 0.1514
    },
    {
      "step": 9695,
      "loss": 0.147
    },
    {
      "step": 9700,
      "loss": 0.1544
    },
    {
      "step": 9705,
      "loss": 0.1542
    },
    {
      "step": 9710,
      "loss": 0.1491
    },
    {
      "step": 9715,
      "loss": 0.1558
    },
    {
      "step": 9720,
      "loss": 0.1439
    },
    {
      "step": 9725,
      "loss": 0.1445
    },
    {
      "step": 9730,
      "loss": 0.1548
    },
    {
      "step": 9735,
      "loss": 0.143
    },
    {
      "step": 9740,
      "loss": 0.1628
    },
    {
      "step": 9745,
      "loss": 0.1437
    },
    {
      "step": 9750,
      "loss": 0.1457
    },
    {
      "step": 9755,
      "loss": 0.1506
    },
    {
      "step": 9760,
      "loss": 0.1563
    },
    {
      "step": 9765,
      "loss": 0.1588
    },
    {
      "step": 9770,
      "loss": 0.1482
    },
    {
      "step": 9775,
      "loss": 0.1597
    },
    {
      "step": 9780,
      "loss": 0.1651
    },
    {
      "step": 9785,
      "loss": 0.1299
    },
    {
      "step": 9790,
      "loss": 0.1403
    },
    {
      "step": 9795,
      "loss": 0.1498
    },
    {
      "step": 9800,
      "loss": 0.1608
    },
    {
      "step": 9805,
      "loss": 0.1478
    },
    {
      "step": 9810,
      "loss": 0.1483
    },
    {
      "step": 9815,
      "loss": 0.1398
    },
    {
      "step": 9820,
      "loss": 0.1514
    },
    {
      "step": 9825,
      "loss": 0.1414
    },
    {
      "step": 9830,
      "loss": 0.1352
    },
    {
      "step": 9835,
      "loss": 0.1467
    },
    {
      "step": 9840,
      "loss": 0.149
    },
    {
      "step": 9845,
      "loss": 0.1386
    },
    {
      "step": 9850,
      "loss": 0.1497
    },
    {
      "step": 9855,
      "loss": 0.1526
    },
    {
      "step": 9860,
      "loss": 0.1628
    },
    {
      "step": 9865,
      "loss": 0.1557
    },
    {
      "step": 9870,
      "loss": 0.1481
    },
    {
      "step": 9875,
      "loss": 0.1458
    },
    {
      "step": 9880,
      "loss": 0.156
    },
    {
      "step": 9885,
      "loss": 0.1365
    },
    {
      "step": 9890,
      "loss": 0.1466
    },
    {
      "step": 9895,
      "loss": 0.1485
    },
    {
      "step": 9900,
      "loss": 0.1423
    },
    {
      "step": 9905,
      "loss": 0.1444
    },
    {
      "step": 9910,
      "loss": 0.1474
    },
    {
      "step": 9915,
      "loss": 0.1447
    },
    {
      "step": 9920,
      "loss": 0.1424
    },
    {
      "step": 9925,
      "loss": 0.1426
    },
    {
      "step": 9930,
      "loss": 0.1555
    },
    {
      "step": 9935,
      "loss": 0.1396
    },
    {
      "step": 9940,
      "loss": 0.1467
    },
    {
      "step": 9945,
      "loss": 0.1504
    },
    {
      "step": 9950,
      "loss": 0.1448
    },
    {
      "step": 9955,
      "loss": 0.1374
    },
    {
      "step": 9960,
      "loss": 0.1473
    },
    {
      "step": 9965,
      "loss": 0.1469
    },
    {
      "step": 9970,
      "loss": 0.1509
    },
    {
      "step": 9975,
      "loss": 0.142
    },
    {
      "step": 9980,
      "loss": 0.1564
    },
    {
      "step": 9985,
      "loss": 0.1495
    },
    {
      "step": 9990,
      "loss": 0.1469
    },
    {
      "step": 9995,
      "loss": 0.1559
    },
    {
      "step": 10000,
      "loss": 0.1466
    },
    {
      "step": 10005,
      "loss": 0.1394
    },
    {
      "step": 10010,
      "loss": 0.1487
    },
    {
      "step": 10015,
      "loss": 0.1498
    },
    {
      "step": 10020,
      "loss": 0.1512
    },
    {
      "step": 10025,
      "loss": 0.1603
    },
    {
      "step": 10030,
      "loss": 0.1506
    },
    {
      "step": 10035,
      "loss": 0.1464
    },
    {
      "step": 10040,
      "loss": 0.143
    },
    {
      "step": 10045,
      "loss": 0.1467
    },
    {
      "step": 10050,
      "loss": 0.142
    },
    {
      "step": 10055,
      "loss": 0.1498
    },
    {
      "step": 10060,
      "loss": 0.1518
    },
    {
      "step": 10065,
      "loss": 0.1612
    },
    {
      "step": 10070,
      "loss": 0.1369
    },
    {
      "step": 10075,
      "loss": 0.156
    },
    {
      "step": 10080,
      "loss": 0.1551
    },
    {
      "step": 10085,
      "loss": 0.1512
    },
    {
      "step": 10090,
      "loss": 0.1437
    },
    {
      "step": 10095,
      "loss": 0.1441
    },
    {
      "step": 10100,
      "loss": 0.1418
    },
    {
      "step": 10105,
      "loss": 0.1401
    },
    {
      "step": 10110,
      "loss": 0.1426
    },
    {
      "step": 10115,
      "loss": 0.1495
    },
    {
      "step": 10120,
      "loss": 0.147
    },
    {
      "step": 10125,
      "loss": 0.1477
    },
    {
      "step": 10130,
      "loss": 0.1392
    },
    {
      "step": 10135,
      "loss": 0.1428
    },
    {
      "step": 10140,
      "loss": 0.139
    },
    {
      "step": 10145,
      "loss": 0.1456
    },
    {
      "step": 10150,
      "loss": 0.1506
    },
    {
      "step": 10155,
      "loss": 0.14
    },
    {
      "step": 10160,
      "loss": 0.1449
    },
    {
      "step": 10165,
      "loss": 0.1509
    },
    {
      "step": 10170,
      "loss": 0.1428
    },
    {
      "step": 10175,
      "loss": 0.1395
    },
    {
      "step": 10180,
      "loss": 0.1504
    },
    {
      "step": 10185,
      "loss": 0.1352
    },
    {
      "step": 10190,
      "loss": 0.1429
    },
    {
      "step": 10195,
      "loss": 0.1451
    },
    {
      "step": 10200,
      "loss": 0.1487
    },
    {
      "step": 10205,
      "loss": 0.1534
    },
    {
      "step": 10210,
      "loss": 0.1406
    },
    {
      "step": 10215,
      "loss": 0.1411
    },
    {
      "step": 10220,
      "loss": 0.1349
    },
    {
      "step": 10225,
      "loss": 0.1407
    },
    {
      "step": 10230,
      "loss": 0.1487
    },
    {
      "step": 10235,
      "loss": 0.1465
    },
    {
      "step": 10240,
      "loss": 0.151
    },
    {
      "step": 10245,
      "loss": 0.1415
    },
    {
      "step": 10250,
      "loss": 0.1487
    },
    {
      "step": 10255,
      "loss": 0.1341
    },
    {
      "step": 10260,
      "loss": 0.1495
    },
    {
      "step": 10265,
      "loss": 0.161
    },
    {
      "step": 10270,
      "loss": 0.1325
    },
    {
      "step": 10275,
      "loss": 0.1279
    },
    {
      "step": 10280,
      "loss": 0.1407
    },
    {
      "step": 10285,
      "loss": 0.1563
    },
    {
      "step": 10290,
      "loss": 0.1554
    },
    {
      "step": 10295,
      "loss": 0.1635
    },
    {
      "step": 10300,
      "loss": 0.1489
    },
    {
      "step": 10305,
      "loss": 0.1573
    },
    {
      "step": 10310,
      "loss": 0.1559
    },
    {
      "step": 10315,
      "loss": 0.1347
    },
    {
      "step": 10320,
      "loss": 0.1555
    },
    {
      "step": 10325,
      "loss": 0.1629
    },
    {
      "step": 10330,
      "loss": 0.1623
    },
    {
      "step": 10335,
      "loss": 0.1483
    },
    {
      "step": 10340,
      "loss": 0.1493
    },
    {
      "step": 10345,
      "loss": 0.1435
    },
    {
      "step": 10350,
      "loss": 0.1422
    },
    {
      "step": 10355,
      "loss": 0.1468
    },
    {
      "step": 10360,
      "loss": 0.1399
    },
    {
      "step": 10365,
      "loss": 0.1459
    },
    {
      "step": 10370,
      "loss": 0.1514
    },
    {
      "step": 10375,
      "loss": 0.1384
    },
    {
      "step": 10380,
      "loss": 0.1254
    },
    {
      "step": 10385,
      "loss": 0.1456
    },
    {
      "step": 10390,
      "loss": 0.1493
    },
    {
      "step": 10395,
      "loss": 0.1609
    },
    {
      "step": 10400,
      "loss": 0.151
    },
    {
      "step": 10405,
      "loss": 0.1432
    },
    {
      "step": 10410,
      "loss": 0.1367
    },
    {
      "step": 10415,
      "loss": 0.1439
    },
    {
      "step": 10420,
      "loss": 0.1408
    },
    {
      "step": 10425,
      "loss": 0.1503
    },
    {
      "step": 10430,
      "loss": 0.1592
    },
    {
      "step": 10435,
      "loss": 0.1369
    },
    {
      "step": 10440,
      "loss": 0.1357
    },
    {
      "step": 10445,
      "loss": 0.1602
    },
    {
      "step": 10450,
      "loss": 0.1523
    },
    {
      "step": 10455,
      "loss": 0.1482
    },
    {
      "step": 10460,
      "loss": 0.1543
    },
    {
      "step": 10465,
      "loss": 0.1484
    },
    {
      "step": 10470,
      "loss": 0.143
    },
    {
      "step": 10475,
      "loss": 0.1509
    },
    {
      "step": 10480,
      "loss": 0.1533
    },
    {
      "step": 10485,
      "loss": 0.1408
    },
    {
      "step": 10490,
      "loss": 0.1443
    },
    {
      "step": 10495,
      "loss": 0.1444
    },
    {
      "step": 10500,
      "loss": 0.149
    },
    {
      "step": 10505,
      "loss": 0.1407
    },
    {
      "step": 10510,
      "loss": 0.1436
    },
    {
      "step": 10515,
      "loss": 0.1644
    },
    {
      "step": 10520,
      "loss": 0.1376
    },
    {
      "step": 10525,
      "loss": 0.1368
    },
    {
      "step": 10530,
      "loss": 0.14
    },
    {
      "step": 10535,
      "loss": 0.1572
    },
    {
      "step": 10540,
      "loss": 0.147
    },
    {
      "step": 10545,
      "loss": 0.1794
    },
    {
      "step": 10550,
      "loss": 0.1586
    },
    {
      "step": 10555,
      "loss": 0.146
    },
    {
      "step": 10560,
      "loss": 0.1352
    },
    {
      "step": 10565,
      "loss": 0.1385
    },
    {
      "step": 10570,
      "loss": 0.1384
    },
    {
      "step": 10575,
      "loss": 0.1615
    },
    {
      "step": 10580,
      "loss": 0.1299
    },
    {
      "step": 10585,
      "loss": 0.1582
    },
    {
      "step": 10590,
      "loss": 0.1448
    },
    {
      "step": 10595,
      "loss": 0.145
    },
    {
      "step": 10600,
      "loss": 0.1566
    },
    {
      "step": 10605,
      "loss": 0.1454
    },
    {
      "step": 10610,
      "loss": 0.1477
    },
    {
      "step": 10615,
      "loss": 0.1441
    },
    {
      "step": 10620,
      "loss": 0.1236
    },
    {
      "step": 10625,
      "loss": 0.1388
    },
    {
      "step": 10630,
      "loss": 0.1505
    },
    {
      "step": 10635,
      "loss": 0.149
    },
    {
      "step": 10640,
      "loss": 0.143
    },
    {
      "step": 10645,
      "loss": 0.1425
    },
    {
      "step": 10650,
      "loss": 0.1515
    },
    {
      "step": 10655,
      "loss": 0.1446
    },
    {
      "step": 10660,
      "loss": 0.1451
    },
    {
      "step": 10665,
      "loss": 0.1406
    },
    {
      "step": 10670,
      "loss": 0.1595
    },
    {
      "step": 10675,
      "loss": 0.1418
    },
    {
      "step": 10680,
      "loss": 0.1491
    },
    {
      "step": 10685,
      "loss": 0.1395
    },
    {
      "step": 10690,
      "loss": 0.1516
    },
    {
      "step": 10695,
      "loss": 0.1531
    },
    {
      "step": 10700,
      "loss": 0.1342
    },
    {
      "step": 10705,
      "loss": 0.1436
    },
    {
      "step": 10710,
      "loss": 0.1463
    },
    {
      "step": 10715,
      "loss": 0.1361
    },
    {
      "step": 10720,
      "loss": 0.1472
    },
    {
      "step": 10725,
      "loss": 0.1582
    },
    {
      "step": 10730,
      "loss": 0.1497
    },
    {
      "step": 10735,
      "loss": 0.1751
    },
    {
      "step": 10740,
      "loss": 0.1443
    },
    {
      "step": 10745,
      "loss": 0.1468
    },
    {
      "step": 10750,
      "loss": 0.1435
    },
    {
      "step": 10755,
      "loss": 0.1672
    },
    {
      "step": 10760,
      "loss": 0.1372
    },
    {
      "step": 10765,
      "loss": 0.1404
    },
    {
      "step": 10770,
      "loss": 0.1567
    },
    {
      "step": 10775,
      "loss": 0.153
    },
    {
      "step": 10780,
      "loss": 0.1471
    },
    {
      "step": 10785,
      "loss": 0.1443
    },
    {
      "step": 10790,
      "loss": 0.1703
    },
    {
      "step": 10795,
      "loss": 0.1454
    },
    {
      "step": 10800,
      "loss": 0.1559
    },
    {
      "step": 10805,
      "loss": 0.1488
    },
    {
      "step": 10810,
      "loss": 0.1483
    },
    {
      "step": 10815,
      "loss": 0.1564
    },
    {
      "step": 10820,
      "loss": 0.15
    },
    {
      "step": 10825,
      "loss": 0.1491
    },
    {
      "step": 10830,
      "loss": 0.1409
    },
    {
      "step": 10835,
      "loss": 0.1433
    },
    {
      "step": 10840,
      "loss": 0.1439
    },
    {
      "step": 10845,
      "loss": 0.1521
    },
    {
      "step": 10850,
      "loss": 0.1467
    },
    {
      "step": 10855,
      "loss": 0.1469
    },
    {
      "step": 10860,
      "loss": 0.1465
    },
    {
      "step": 10865,
      "loss": 0.1495
    },
    {
      "step": 10870,
      "loss": 0.158
    },
    {
      "step": 10875,
      "loss": 0.1538
    },
    {
      "step": 10880,
      "loss": 0.1503
    },
    {
      "step": 10885,
      "loss": 0.1513
    },
    {
      "step": 10890,
      "loss": 0.1432
    },
    {
      "step": 10895,
      "loss": 0.1463
    },
    {
      "step": 10900,
      "loss": 0.1609
    },
    {
      "step": 10905,
      "loss": 0.154
    },
    {
      "step": 10910,
      "loss": 0.1339
    },
    {
      "step": 10915,
      "loss": 0.1484
    },
    {
      "step": 10920,
      "loss": 0.129
    },
    {
      "step": 10925,
      "loss": 0.1608
    },
    {
      "step": 10930,
      "loss": 0.1356
    },
    {
      "step": 10935,
      "loss": 0.1461
    },
    {
      "step": 10940,
      "loss": 0.16
    },
    {
      "step": 10945,
      "loss": 0.1543
    },
    {
      "step": 10950,
      "loss": 0.1483
    },
    {
      "step": 10955,
      "loss": 0.1483
    },
    {
      "step": 10960,
      "loss": 0.1631
    },
    {
      "step": 10965,
      "loss": 0.1648
    },
    {
      "step": 10970,
      "loss": 0.1568
    },
    {
      "step": 10975,
      "loss": 0.1378
    },
    {
      "step": 10980,
      "loss": 0.1502
    },
    {
      "step": 10985,
      "loss": 0.146
    },
    {
      "step": 10990,
      "loss": 0.1423
    }
  ],
  "eval_losses": [
    {
      "step": 1374,
      "eval_loss": 0.3221
    },
    {
      "step": 2748,
      "eval_loss": 0.3371
    },
    {
      "step": 4122,
      "eval_loss": 0.3444
    },
    {
      "step": 5496,
      "eval_loss": 0.3555
    },
    {
      "step": 6870,
      "eval_loss": 0.3795
    },
    {
      "step": 8244,
      "eval_loss": 0.3756
    },
    {
      "step": 9618,
      "eval_loss": 0.3777
    },
    {
      "step": 10992,
      "eval_loss": 0.3797
    }
  ],
  "decode_history": [
    {
      "epoch": 0.25,
      "step": 1374,
      "eval_loss": 0.32209521532058716,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 32,
        "mae_log": 0.4790344416834271,
        "rmse_log": 0.6463075902887407,
        "mae_s": 3.42164375,
        "rmse_s": 5.577826547926172,
        "spearman_rho": 0.10860883198037898,
        "mean_actual_s": 6.12451875,
        "mean_pred_s": 4.659375
      },
      "samples": [
        {
          "output": "ui: nodes=1450 interactive=100 links=70 buttons=27 inputs=3 text=560\ndwell_seconds: 2.9",
          "gold_s": 3.621,
          "pred_s": 2.9
        },
        {
          "output": "ui: nodes=1303 interactive=119 links=50 buttons=59 inputs=10 text=3700\ndwell_seconds: 4.3",
          "gold_s": 2.878,
          "pred_s": 4.3
        },
        {
          "output": "ui: nodes=1000 interactive=100 links=79 buttons=18 inputs=3 text=1999\ndwell_seconds: 3.7",
          "gold_s": 3.851,
          "pred_s": 3.7
        }
      ]
    },
    {
      "epoch": 0.5,
      "step": 2748,
      "eval_loss": 0.337056964635849,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 32,
        "mae_log": 0.5073012850876231,
        "rmse_log": 0.6716397879675641,
        "mae_s": 3.73770625,
        "rmse_s": 5.7048781975384895,
        "spearman_rho": -0.01895313387892055,
        "mean_actual_s": 6.12451875,
        "mean_pred_s": 5.790625
      },
      "samples": [
        {
          "output": "ui: nodes=1990 interactive=16 links=7 buttons=3 inputs=6 text=116\ndwell_seconds: 3.1",
          "gold_s": 3.621,
          "pred_s": 3.1
        },
        {
          "output": "ui: nodes=1406 interactive=53 links=40 buttons=10 inputs=3 text=10076\ndwell_seconds: 4.6",
          "gold_s": 2.878,
          "pred_s": 4.6
        },
        {
          "output": "ui: nodes=1009 interactive=100 links=73 buttons=26 inputs=1 text=1910\ndwell_seconds: 4.1",
          "gold_s": 3.851,
          "pred_s": 4.1
        }
      ]
    },
    {
      "epoch": 0.75,
      "step": 4122,
      "eval_loss": 0.3444441854953766,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 32,
        "mae_log": 0.6327093719392785,
        "rmse_log": 0.7618331137663568,
        "mae_s": 4.57051875,
        "rmse_s": 6.145969532852404,
        "spearman_rho": -0.19758473272526308,
        "mean_actual_s": 6.12451875,
        "mean_pred_s": 5.875
      },
      "samples": [
        {
          "output": "ui: nodes=1152 interactive=123 links=105 buttons=15 inputs=3 text=1901\ndwell_seconds: 3.1",
          "gold_s": 3.621,
          "pred_s": 3.1
        },
        {
          "output": "ui: nodes=1462 interactive=145 links=70 buttons=70 inputs=5 text=7130\ndwell_seconds: 4.5",
          "gold_s": 2.878,
          "pred_s": 4.5
        },
        {
          "output": "ui: nodes=480 interactive=100 links=98 buttons=1 inputs=1 text=1000\ndwell_seconds: 10.1",
          "gold_s": 3.851,
          "pred_s": 10.1
        }
      ]
    },
    {
      "epoch": 1.0,
      "step": 5496,
      "eval_loss": 0.35548219084739685,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 32,
        "mae_log": 0.4619839958482485,
        "rmse_log": 0.6041057896608554,
        "mae_s": 3.21583125,
        "rmse_s": 5.100407361059506,
        "spearman_rho": 0.04408533854783509,
        "mean_actual_s": 6.12451875,
        "mean_pred_s": 4.9
      },
      "samples": [
        {
          "output": "ui: nodes=1156 interactive=138 links=100 buttons=32 inputs=6 text=1164\ndwell_seconds: 3.1",
          "gold_s": 3.621,
          "pred_s": 3.1
        },
        {
          "output": "ui: nodes=1409 interactive=163 links=100 buttons=35 inputs=28 text=5000\ndwell_seconds: 4.3",
          "gold_s": 2.878,
          "pred_s": 4.3
        },
        {
          "output": "ui: nodes=488 interactive=69 links=53 buttons=15 inputs=1 text=1029\ndwell_seconds: 4.0",
          "gold_s": 3.851,
          "pred_s": 4.0
        }
      ]
    },
    {
      "epoch": 1.25,
      "step": 6870,
      "eval_loss": 0.3794596195220947,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 32,
        "mae_log": 0.4247657703446106,
        "rmse_log": 0.5583804846040833,
        "mae_s": 2.97520625,
        "rmse_s": 4.984483824705422,
        "spearman_rho": 0.10334738484087104,
        "mean_actual_s": 6.12451875,
        "mean_pred_s": 4.653124999999999
      },
      "samples": [
        {
          "output": "ui: nodes=1158 interactive=117 links=70 buttons=39 inputs=8 text=1049\ndwell_seconds: 3.0",
          "gold_s": 3.621,
          "pred_s": 3.0
        },
        {
          "output": "ui: nodes=1402 interactive=171 links=100 buttons=47 inputs=24 text=7180\ndwell_seconds: 4.6",
          "gold_s": 2.878,
          "pred_s": 4.6
        },
        {
          "output": "ui: nodes=488 interactive=71 links=64 buttons=4 inputs=3 text=597\ndwell_seconds: 5.0",
          "gold_s": 3.851,
          "pred_s": 5.0
        }
      ]
    },
    {
      "epoch": 1.5,
      "step": 8244,
      "eval_loss": 0.37560734152793884,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 32,
        "mae_log": 0.5356236864231196,
        "rmse_log": 0.6743325725991396,
        "mae_s": 3.6249562499999994,
        "rmse_s": 5.227703989205012,
        "spearman_rho": -0.11800433885014122,
        "mean_actual_s": 6.12451875,
        "mean_pred_s": 5.63125
      },
      "samples": [
        {
          "output": "ui: nodes=1153 interactive=138 links=100 buttons=32 inputs=6 text=115\ndwell_seconds: 3.1",
          "gold_s": 3.621,
          "pred_s": 3.1
        },
        {
          "output": "ui: nodes=1482 interactive=173 links=111 buttons=47 inputs=15 text=7100\ndwell_seconds: 4.6",
          "gold_s": 2.878,
          "pred_s": 4.6
        },
        {
          "output": "ui: nodes=481 interactive=72 links=40 buttons=31 inputs=1 text=1047\ndwell_seconds: 4.1",
          "gold_s": 3.851,
          "pred_s": 4.1
        }
      ]
    },
    {
      "epoch": 1.75,
      "step": 9618,
      "eval_loss": 0.3776862323284149,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 32,
        "mae_log": 0.47344202573287575,
        "rmse_log": 0.6372963001955727,
        "mae_s": 3.30645625,
        "rmse_s": 5.153788557823264,
        "spearman_rho": -0.023875163504107465,
        "mean_actual_s": 6.12451875,
        "mean_pred_s": 5.378125000000001
      },
      "samples": [
        {
          "output": "ui: nodes=1154 interactive=138 links=100 buttons=32 inputs=6 text=115\ndwell_seconds: 3.1",
          "gold_s": 3.621,
          "pred_s": 3.1
        },
        {
          "output": "ui: nodes=1488 interactive=173 links=110 buttons=46 inputs=17 text=7100\ndwell_seconds: 4.7",
          "gold_s": 2.878,
          "pred_s": 4.7
        },
        {
          "output": "ui: nodes=481 interactive=71 links=64 buttons=4 inputs=3 text=598\ndwell_seconds: 5.0",
          "gold_s": 3.851,
          "pred_s": 5.0
        }
      ]
    },
    {
      "epoch": 2.0,
      "step": 10992,
      "eval_loss": 0.3796551823616028,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 32,
        "mae_log": 0.5167521658072645,
        "rmse_log": 0.6767919092816373,
        "mae_s": 3.66895625,
        "rmse_s": 5.44877901449031,
        "spearman_rho": -0.11160194244877168,
        "mean_actual_s": 6.12451875,
        "mean_pred_s": 5.6156250000000005
      },
      "samples": [
        {
          "output": "ui: nodes=1154 interactive=138 links=100 buttons=32 inputs=6 text=115\ndwell_seconds: 3.1",
          "gold_s": 3.621,
          "pred_s": 3.1
        },
        {
          "output": "ui: nodes=1488 interactive=173 links=117 buttons=48 inputs=8 text=6772\ndwell_seconds: 4.6",
          "gold_s": 2.878,
          "pred_s": 4.6
        },
        {
          "output": "ui: nodes=481 interactive=69 links=40 buttons=28 inputs=1 text=1049\ndwell_seconds: 5.0",
          "gold_s": 3.851,
          "pred_s": 5.0
        }
      ]
    }
  ],
  "vram": {
    "device": "NVIDIA A100-SXM4-80GB",
    "total_gb": 79.25,
    "peak_allocated_gb": 6.62,
    "peak_reserved_gb": 6.83
  },
  "effective_config": {
    "train": {
      "learning_rate": 0.0001,
      "num_epochs": 2,
      "evals_per_epoch": 4,
      "batch_size": 4,
      "grad_accum": 4,
      "eval_batch_size": 4,
      "warmup_ratio": 0.03,
      "weight_decay": 0.01,
      "lr_scheduler": "cosine",
      "optim": "adamw_8bit",
      "max_grad_norm": 1.0,
      "logging_steps": 5,
      "save_total_limit": 2,
      "report_to": "wandb",
      "wandb_project": "tot-vlm",
      "run_name": "qwen3vl4b-v3-lam050"
    },
    "model": {
      "checkpoint": "unsloth/Qwen3-VL-4B-Instruct",
      "load_in_4bit": true,
      "gradient_checkpointing": true,
      "max_seq_length": 1536,
      "finetune_vision_layers": true,
      "lora_r": 16,
      "lora_alpha": 16,
      "lora_dropout": 0.05
    },
    "image": {
      "max_side": 1024,
      "min_pixels": 100352,
      "max_pixels": 602112
    },
    "data": {
      "include_task_title": false,
      "scaffold": true,
      "scaffold_stats": "artifacts/scaffold_stats.parquet"
    },
    "dry_run": false,
    "lupi": {
      "lambda": 0.5,
      "n_rows": 87921,
      "n_blended": 87921,
      "coverage": 1.0,
      "mean_abs_shift_s": 2.192
    },
    "scaffold": {
      "coverage_train": 1.0,
      "coverage_val": 1.0
    }
  }
}
```
