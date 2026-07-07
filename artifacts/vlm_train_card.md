# VLM train card — Qwen3-VL-4B QLoRA SFT (Path A)

_Generated 2026-07-06T18:07:54+00:00 · config `configs/vlm.yaml` · seed 42 · mode **FULL**_

- Model: `unsloth/Qwen3-VL-4B-Instruct` · 4-bit QLoRA · vision frozen · LoRA r=16 α=16 on language attn+MLP
- Target: `dwell_seconds: X.X` (winsor cap 24.954 s, train-split p95 — see artifacts/dataset_card.md)
- Task title in prompt: **False** (False = image+features, True = image+features+task)
- Features (train-time only): targets blended toward the privileged-feature teacher (λ=0.5) — **87921/87921** rows covered (100.0%), mean |shift| 2.192 s; val targets untouched, inference stays screenshot-only
- Examples: train **87921** · val **4269** (img_resolved only)

## Memory footprint

- per-device batch **4** × grad-accum **4** = effective **16**
- image bounds: max_side 1024, pixels [100352, 602112] (≤768 visual tokens) · max_seq_length 1536
- peak VRAM: **6.29 GB** allocated / 6.49 GB reserved of 39.49 GB (NVIDIA A100-SXM4-40GB)

## Loss

- train loss: first 3.8955 → last 0.0769

| eval pass | epoch | step | val loss | parse rate | MAE (s) | MAE (log) |
|---|---|---|---|---|---|---|
| 0 | 1.0 | 5496 | 0.12439540773630142 | 100.00% | 3.25 | 0.450 |
| 1 | 2.0 | 10992 | 0.13700838387012482 | 100.00% | 3.74 | 0.518 |

## Sample decodes (last eval pass)

- gold 3.6 s → `dwell_seconds: 2.6`
- gold 2.9 s → `dwell_seconds: 4.1`
- gold 3.9 s → `dwell_seconds: 4.3`

Checkpoints: `artifacts/vlm_ckpt` (per epoch; final adapters in `artifacts/vlm_ckpt/final`).

## Full log (JSON)

```json
{
  "train_losses": [
    {
      "step": 5,
      "loss": 3.8955
    },
    {
      "step": 10,
      "loss": 3.7992
    },
    {
      "step": 15,
      "loss": 3.678
    },
    {
      "step": 20,
      "loss": 3.4457
    },
    {
      "step": 25,
      "loss": 3.1081
    },
    {
      "step": 30,
      "loss": 2.7594
    },
    {
      "step": 35,
      "loss": 2.4063
    },
    {
      "step": 40,
      "loss": 2.027
    },
    {
      "step": 45,
      "loss": 1.5721
    },
    {
      "step": 50,
      "loss": 0.9527
    },
    {
      "step": 55,
      "loss": 0.3992
    },
    {
      "step": 60,
      "loss": 0.2587
    },
    {
      "step": 65,
      "loss": 0.2141
    },
    {
      "step": 70,
      "loss": 0.1193
    },
    {
      "step": 75,
      "loss": 0.0932
    },
    {
      "step": 80,
      "loss": 0.0916
    },
    {
      "step": 85,
      "loss": 0.0914
    },
    {
      "step": 90,
      "loss": 0.0899
    },
    {
      "step": 95,
      "loss": 0.0898
    },
    {
      "step": 100,
      "loss": 0.0906
    },
    {
      "step": 105,
      "loss": 0.088
    },
    {
      "step": 110,
      "loss": 0.0913
    },
    {
      "step": 115,
      "loss": 0.0891
    },
    {
      "step": 120,
      "loss": 0.0908
    },
    {
      "step": 125,
      "loss": 0.0912
    },
    {
      "step": 130,
      "loss": 0.0913
    },
    {
      "step": 135,
      "loss": 0.0897
    },
    {
      "step": 140,
      "loss": 0.0856
    },
    {
      "step": 145,
      "loss": 0.0897
    },
    {
      "step": 150,
      "loss": 0.0876
    },
    {
      "step": 155,
      "loss": 0.0865
    },
    {
      "step": 160,
      "loss": 0.0904
    },
    {
      "step": 165,
      "loss": 0.0894
    },
    {
      "step": 170,
      "loss": 0.0866
    },
    {
      "step": 175,
      "loss": 0.0879
    },
    {
      "step": 180,
      "loss": 0.0873
    },
    {
      "step": 185,
      "loss": 0.0881
    },
    {
      "step": 190,
      "loss": 0.0885
    },
    {
      "step": 195,
      "loss": 0.0861
    },
    {
      "step": 200,
      "loss": 0.0895
    },
    {
      "step": 205,
      "loss": 0.0886
    },
    {
      "step": 210,
      "loss": 0.0884
    },
    {
      "step": 215,
      "loss": 0.0868
    },
    {
      "step": 220,
      "loss": 0.0914
    },
    {
      "step": 225,
      "loss": 0.0932
    },
    {
      "step": 230,
      "loss": 0.0897
    },
    {
      "step": 235,
      "loss": 0.0907
    },
    {
      "step": 240,
      "loss": 0.0864
    },
    {
      "step": 245,
      "loss": 0.0848
    },
    {
      "step": 250,
      "loss": 0.0872
    },
    {
      "step": 255,
      "loss": 0.0883
    },
    {
      "step": 260,
      "loss": 0.086
    },
    {
      "step": 265,
      "loss": 0.0886
    },
    {
      "step": 270,
      "loss": 0.0876
    },
    {
      "step": 275,
      "loss": 0.0866
    },
    {
      "step": 280,
      "loss": 0.0891
    },
    {
      "step": 285,
      "loss": 0.0865
    },
    {
      "step": 290,
      "loss": 0.0884
    },
    {
      "step": 295,
      "loss": 0.0925
    },
    {
      "step": 300,
      "loss": 0.0881
    },
    {
      "step": 305,
      "loss": 0.0895
    },
    {
      "step": 310,
      "loss": 0.0915
    },
    {
      "step": 315,
      "loss": 0.0874
    },
    {
      "step": 320,
      "loss": 0.0882
    },
    {
      "step": 325,
      "loss": 0.0903
    },
    {
      "step": 330,
      "loss": 0.0895
    },
    {
      "step": 335,
      "loss": 0.0841
    },
    {
      "step": 340,
      "loss": 0.0886
    },
    {
      "step": 345,
      "loss": 0.0851
    },
    {
      "step": 350,
      "loss": 0.0875
    },
    {
      "step": 355,
      "loss": 0.0867
    },
    {
      "step": 360,
      "loss": 0.0886
    },
    {
      "step": 365,
      "loss": 0.0868
    },
    {
      "step": 370,
      "loss": 0.0876
    },
    {
      "step": 375,
      "loss": 0.0872
    },
    {
      "step": 380,
      "loss": 0.0871
    },
    {
      "step": 385,
      "loss": 0.086
    },
    {
      "step": 390,
      "loss": 0.0868
    },
    {
      "step": 395,
      "loss": 0.0867
    },
    {
      "step": 400,
      "loss": 0.087
    },
    {
      "step": 405,
      "loss": 0.0875
    },
    {
      "step": 410,
      "loss": 0.0883
    },
    {
      "step": 415,
      "loss": 0.0851
    },
    {
      "step": 420,
      "loss": 0.0823
    },
    {
      "step": 425,
      "loss": 0.0881
    },
    {
      "step": 430,
      "loss": 0.0887
    },
    {
      "step": 435,
      "loss": 0.0887
    },
    {
      "step": 440,
      "loss": 0.0876
    },
    {
      "step": 445,
      "loss": 0.0874
    },
    {
      "step": 450,
      "loss": 0.0831
    },
    {
      "step": 455,
      "loss": 0.0859
    },
    {
      "step": 460,
      "loss": 0.0869
    },
    {
      "step": 465,
      "loss": 0.089
    },
    {
      "step": 470,
      "loss": 0.0853
    },
    {
      "step": 475,
      "loss": 0.0851
    },
    {
      "step": 480,
      "loss": 0.0871
    },
    {
      "step": 485,
      "loss": 0.0873
    },
    {
      "step": 490,
      "loss": 0.0873
    },
    {
      "step": 495,
      "loss": 0.0884
    },
    {
      "step": 500,
      "loss": 0.0863
    },
    {
      "step": 505,
      "loss": 0.0874
    },
    {
      "step": 510,
      "loss": 0.0876
    },
    {
      "step": 515,
      "loss": 0.0884
    },
    {
      "step": 520,
      "loss": 0.0862
    },
    {
      "step": 525,
      "loss": 0.0875
    },
    {
      "step": 530,
      "loss": 0.0885
    },
    {
      "step": 535,
      "loss": 0.0873
    },
    {
      "step": 540,
      "loss": 0.0861
    },
    {
      "step": 545,
      "loss": 0.0888
    },
    {
      "step": 550,
      "loss": 0.0842
    },
    {
      "step": 555,
      "loss": 0.0868
    },
    {
      "step": 560,
      "loss": 0.0857
    },
    {
      "step": 565,
      "loss": 0.0889
    },
    {
      "step": 570,
      "loss": 0.0849
    },
    {
      "step": 575,
      "loss": 0.0865
    },
    {
      "step": 580,
      "loss": 0.0895
    },
    {
      "step": 585,
      "loss": 0.0867
    },
    {
      "step": 590,
      "loss": 0.0861
    },
    {
      "step": 595,
      "loss": 0.0855
    },
    {
      "step": 600,
      "loss": 0.0867
    },
    {
      "step": 605,
      "loss": 0.0854
    },
    {
      "step": 610,
      "loss": 0.0824
    },
    {
      "step": 615,
      "loss": 0.0861
    },
    {
      "step": 620,
      "loss": 0.0903
    },
    {
      "step": 625,
      "loss": 0.0836
    },
    {
      "step": 630,
      "loss": 0.0877
    },
    {
      "step": 635,
      "loss": 0.0834
    },
    {
      "step": 640,
      "loss": 0.0872
    },
    {
      "step": 645,
      "loss": 0.0861
    },
    {
      "step": 650,
      "loss": 0.085
    },
    {
      "step": 655,
      "loss": 0.0878
    },
    {
      "step": 660,
      "loss": 0.0849
    },
    {
      "step": 665,
      "loss": 0.0857
    },
    {
      "step": 670,
      "loss": 0.0882
    },
    {
      "step": 675,
      "loss": 0.0818
    },
    {
      "step": 680,
      "loss": 0.0869
    },
    {
      "step": 685,
      "loss": 0.0868
    },
    {
      "step": 690,
      "loss": 0.087
    },
    {
      "step": 695,
      "loss": 0.0846
    },
    {
      "step": 700,
      "loss": 0.0842
    },
    {
      "step": 705,
      "loss": 0.08
    },
    {
      "step": 710,
      "loss": 0.0895
    },
    {
      "step": 715,
      "loss": 0.088
    },
    {
      "step": 720,
      "loss": 0.0869
    },
    {
      "step": 725,
      "loss": 0.0848
    },
    {
      "step": 730,
      "loss": 0.0847
    },
    {
      "step": 735,
      "loss": 0.0846
    },
    {
      "step": 740,
      "loss": 0.0869
    },
    {
      "step": 745,
      "loss": 0.0867
    },
    {
      "step": 750,
      "loss": 0.0832
    },
    {
      "step": 755,
      "loss": 0.0885
    },
    {
      "step": 760,
      "loss": 0.0867
    },
    {
      "step": 765,
      "loss": 0.0879
    },
    {
      "step": 770,
      "loss": 0.085
    },
    {
      "step": 775,
      "loss": 0.0846
    },
    {
      "step": 780,
      "loss": 0.0826
    },
    {
      "step": 785,
      "loss": 0.0882
    },
    {
      "step": 790,
      "loss": 0.0851
    },
    {
      "step": 795,
      "loss": 0.0829
    },
    {
      "step": 800,
      "loss": 0.0853
    },
    {
      "step": 805,
      "loss": 0.0859
    },
    {
      "step": 810,
      "loss": 0.0854
    },
    {
      "step": 815,
      "loss": 0.0856
    },
    {
      "step": 820,
      "loss": 0.0869
    },
    {
      "step": 825,
      "loss": 0.0883
    },
    {
      "step": 830,
      "loss": 0.0878
    },
    {
      "step": 835,
      "loss": 0.0849
    },
    {
      "step": 840,
      "loss": 0.0852
    },
    {
      "step": 845,
      "loss": 0.086
    },
    {
      "step": 850,
      "loss": 0.0847
    },
    {
      "step": 855,
      "loss": 0.0846
    },
    {
      "step": 860,
      "loss": 0.086
    },
    {
      "step": 865,
      "loss": 0.087
    },
    {
      "step": 870,
      "loss": 0.0826
    },
    {
      "step": 875,
      "loss": 0.0864
    },
    {
      "step": 880,
      "loss": 0.0846
    },
    {
      "step": 885,
      "loss": 0.0829
    },
    {
      "step": 890,
      "loss": 0.0863
    },
    {
      "step": 895,
      "loss": 0.084
    },
    {
      "step": 900,
      "loss": 0.0834
    },
    {
      "step": 905,
      "loss": 0.0857
    },
    {
      "step": 910,
      "loss": 0.0841
    },
    {
      "step": 915,
      "loss": 0.0887
    },
    {
      "step": 920,
      "loss": 0.0852
    },
    {
      "step": 925,
      "loss": 0.0873
    },
    {
      "step": 930,
      "loss": 0.0849
    },
    {
      "step": 935,
      "loss": 0.0869
    },
    {
      "step": 940,
      "loss": 0.0832
    },
    {
      "step": 945,
      "loss": 0.0829
    },
    {
      "step": 950,
      "loss": 0.0865
    },
    {
      "step": 955,
      "loss": 0.0849
    },
    {
      "step": 960,
      "loss": 0.0895
    },
    {
      "step": 965,
      "loss": 0.0885
    },
    {
      "step": 970,
      "loss": 0.0863
    },
    {
      "step": 975,
      "loss": 0.0857
    },
    {
      "step": 980,
      "loss": 0.0845
    },
    {
      "step": 985,
      "loss": 0.0854
    },
    {
      "step": 990,
      "loss": 0.0854
    },
    {
      "step": 995,
      "loss": 0.0816
    },
    {
      "step": 1000,
      "loss": 0.0897
    },
    {
      "step": 1005,
      "loss": 0.088
    },
    {
      "step": 1010,
      "loss": 0.0861
    },
    {
      "step": 1015,
      "loss": 0.0848
    },
    {
      "step": 1020,
      "loss": 0.0862
    },
    {
      "step": 1025,
      "loss": 0.0859
    },
    {
      "step": 1030,
      "loss": 0.0839
    },
    {
      "step": 1035,
      "loss": 0.0842
    },
    {
      "step": 1040,
      "loss": 0.0846
    },
    {
      "step": 1045,
      "loss": 0.0865
    },
    {
      "step": 1050,
      "loss": 0.0829
    },
    {
      "step": 1055,
      "loss": 0.0875
    },
    {
      "step": 1060,
      "loss": 0.0853
    },
    {
      "step": 1065,
      "loss": 0.0858
    },
    {
      "step": 1070,
      "loss": 0.0821
    },
    {
      "step": 1075,
      "loss": 0.0839
    },
    {
      "step": 1080,
      "loss": 0.0858
    },
    {
      "step": 1085,
      "loss": 0.0822
    },
    {
      "step": 1090,
      "loss": 0.0858
    },
    {
      "step": 1095,
      "loss": 0.0826
    },
    {
      "step": 1100,
      "loss": 0.0856
    },
    {
      "step": 1105,
      "loss": 0.0855
    },
    {
      "step": 1110,
      "loss": 0.0825
    },
    {
      "step": 1115,
      "loss": 0.0817
    },
    {
      "step": 1120,
      "loss": 0.0827
    },
    {
      "step": 1125,
      "loss": 0.0887
    },
    {
      "step": 1130,
      "loss": 0.0868
    },
    {
      "step": 1135,
      "loss": 0.0879
    },
    {
      "step": 1140,
      "loss": 0.0803
    },
    {
      "step": 1145,
      "loss": 0.0819
    },
    {
      "step": 1150,
      "loss": 0.0831
    },
    {
      "step": 1155,
      "loss": 0.0836
    },
    {
      "step": 1160,
      "loss": 0.0874
    },
    {
      "step": 1165,
      "loss": 0.0853
    },
    {
      "step": 1170,
      "loss": 0.0839
    },
    {
      "step": 1175,
      "loss": 0.0829
    },
    {
      "step": 1180,
      "loss": 0.0809
    },
    {
      "step": 1185,
      "loss": 0.0853
    },
    {
      "step": 1190,
      "loss": 0.0836
    },
    {
      "step": 1195,
      "loss": 0.0843
    },
    {
      "step": 1200,
      "loss": 0.0869
    },
    {
      "step": 1205,
      "loss": 0.086
    },
    {
      "step": 1210,
      "loss": 0.0857
    },
    {
      "step": 1215,
      "loss": 0.0863
    },
    {
      "step": 1220,
      "loss": 0.082
    },
    {
      "step": 1225,
      "loss": 0.086
    },
    {
      "step": 1230,
      "loss": 0.0831
    },
    {
      "step": 1235,
      "loss": 0.0841
    },
    {
      "step": 1240,
      "loss": 0.0857
    },
    {
      "step": 1245,
      "loss": 0.0857
    },
    {
      "step": 1250,
      "loss": 0.0822
    },
    {
      "step": 1255,
      "loss": 0.0823
    },
    {
      "step": 1260,
      "loss": 0.0844
    },
    {
      "step": 1265,
      "loss": 0.0871
    },
    {
      "step": 1270,
      "loss": 0.0833
    },
    {
      "step": 1275,
      "loss": 0.0852
    },
    {
      "step": 1280,
      "loss": 0.0864
    },
    {
      "step": 1285,
      "loss": 0.086
    },
    {
      "step": 1290,
      "loss": 0.0823
    },
    {
      "step": 1295,
      "loss": 0.0806
    },
    {
      "step": 1300,
      "loss": 0.0843
    },
    {
      "step": 1305,
      "loss": 0.0868
    },
    {
      "step": 1310,
      "loss": 0.0849
    },
    {
      "step": 1315,
      "loss": 0.0836
    },
    {
      "step": 1320,
      "loss": 0.0814
    },
    {
      "step": 1325,
      "loss": 0.083
    },
    {
      "step": 1330,
      "loss": 0.0833
    },
    {
      "step": 1335,
      "loss": 0.0845
    },
    {
      "step": 1340,
      "loss": 0.0835
    },
    {
      "step": 1345,
      "loss": 0.0842
    },
    {
      "step": 1350,
      "loss": 0.085
    },
    {
      "step": 1355,
      "loss": 0.087
    },
    {
      "step": 1360,
      "loss": 0.0854
    },
    {
      "step": 1365,
      "loss": 0.083
    },
    {
      "step": 1370,
      "loss": 0.0846
    },
    {
      "step": 1375,
      "loss": 0.0814
    },
    {
      "step": 1380,
      "loss": 0.083
    },
    {
      "step": 1385,
      "loss": 0.0797
    },
    {
      "step": 1390,
      "loss": 0.0855
    },
    {
      "step": 1395,
      "loss": 0.0841
    },
    {
      "step": 1400,
      "loss": 0.0852
    },
    {
      "step": 1405,
      "loss": 0.0851
    },
    {
      "step": 1410,
      "loss": 0.084
    },
    {
      "step": 1415,
      "loss": 0.086
    },
    {
      "step": 1420,
      "loss": 0.0841
    },
    {
      "step": 1425,
      "loss": 0.084
    },
    {
      "step": 1430,
      "loss": 0.0823
    },
    {
      "step": 1435,
      "loss": 0.0839
    },
    {
      "step": 1440,
      "loss": 0.0854
    },
    {
      "step": 1445,
      "loss": 0.0831
    },
    {
      "step": 1450,
      "loss": 0.0849
    },
    {
      "step": 1455,
      "loss": 0.0846
    },
    {
      "step": 1460,
      "loss": 0.0866
    },
    {
      "step": 1465,
      "loss": 0.0838
    },
    {
      "step": 1470,
      "loss": 0.0848
    },
    {
      "step": 1475,
      "loss": 0.0885
    },
    {
      "step": 1480,
      "loss": 0.0879
    },
    {
      "step": 1485,
      "loss": 0.0842
    },
    {
      "step": 1490,
      "loss": 0.0862
    },
    {
      "step": 1495,
      "loss": 0.0845
    },
    {
      "step": 1500,
      "loss": 0.0825
    },
    {
      "step": 1505,
      "loss": 0.0817
    },
    {
      "step": 1510,
      "loss": 0.0821
    },
    {
      "step": 1515,
      "loss": 0.0876
    },
    {
      "step": 1520,
      "loss": 0.0846
    },
    {
      "step": 1525,
      "loss": 0.0836
    },
    {
      "step": 1530,
      "loss": 0.0863
    },
    {
      "step": 1535,
      "loss": 0.0806
    },
    {
      "step": 1540,
      "loss": 0.0846
    },
    {
      "step": 1545,
      "loss": 0.0842
    },
    {
      "step": 1550,
      "loss": 0.0826
    },
    {
      "step": 1555,
      "loss": 0.0833
    },
    {
      "step": 1560,
      "loss": 0.0799
    },
    {
      "step": 1565,
      "loss": 0.0818
    },
    {
      "step": 1570,
      "loss": 0.0879
    },
    {
      "step": 1575,
      "loss": 0.0848
    },
    {
      "step": 1580,
      "loss": 0.0863
    },
    {
      "step": 1585,
      "loss": 0.0876
    },
    {
      "step": 1590,
      "loss": 0.0842
    },
    {
      "step": 1595,
      "loss": 0.0823
    },
    {
      "step": 1600,
      "loss": 0.0854
    },
    {
      "step": 1605,
      "loss": 0.0802
    },
    {
      "step": 1610,
      "loss": 0.0826
    },
    {
      "step": 1615,
      "loss": 0.0829
    },
    {
      "step": 1620,
      "loss": 0.0814
    },
    {
      "step": 1625,
      "loss": 0.0863
    },
    {
      "step": 1630,
      "loss": 0.0828
    },
    {
      "step": 1635,
      "loss": 0.0806
    },
    {
      "step": 1640,
      "loss": 0.0816
    },
    {
      "step": 1645,
      "loss": 0.0893
    },
    {
      "step": 1650,
      "loss": 0.0827
    },
    {
      "step": 1655,
      "loss": 0.0845
    },
    {
      "step": 1660,
      "loss": 0.0842
    },
    {
      "step": 1665,
      "loss": 0.0816
    },
    {
      "step": 1670,
      "loss": 0.0839
    },
    {
      "step": 1675,
      "loss": 0.0837
    },
    {
      "step": 1680,
      "loss": 0.0799
    },
    {
      "step": 1685,
      "loss": 0.0804
    },
    {
      "step": 1690,
      "loss": 0.0806
    },
    {
      "step": 1695,
      "loss": 0.0804
    },
    {
      "step": 1700,
      "loss": 0.0808
    },
    {
      "step": 1705,
      "loss": 0.0842
    },
    {
      "step": 1710,
      "loss": 0.0835
    },
    {
      "step": 1715,
      "loss": 0.0833
    },
    {
      "step": 1720,
      "loss": 0.0817
    },
    {
      "step": 1725,
      "loss": 0.0823
    },
    {
      "step": 1730,
      "loss": 0.0841
    },
    {
      "step": 1735,
      "loss": 0.0821
    },
    {
      "step": 1740,
      "loss": 0.0836
    },
    {
      "step": 1745,
      "loss": 0.084
    },
    {
      "step": 1750,
      "loss": 0.0818
    },
    {
      "step": 1755,
      "loss": 0.085
    },
    {
      "step": 1760,
      "loss": 0.0836
    },
    {
      "step": 1765,
      "loss": 0.0845
    },
    {
      "step": 1770,
      "loss": 0.085
    },
    {
      "step": 1775,
      "loss": 0.0843
    },
    {
      "step": 1780,
      "loss": 0.0802
    },
    {
      "step": 1785,
      "loss": 0.0866
    },
    {
      "step": 1790,
      "loss": 0.0837
    },
    {
      "step": 1795,
      "loss": 0.0832
    },
    {
      "step": 1800,
      "loss": 0.0812
    },
    {
      "step": 1805,
      "loss": 0.0814
    },
    {
      "step": 1810,
      "loss": 0.0815
    },
    {
      "step": 1815,
      "loss": 0.0854
    },
    {
      "step": 1820,
      "loss": 0.0799
    },
    {
      "step": 1825,
      "loss": 0.084
    },
    {
      "step": 1830,
      "loss": 0.0838
    },
    {
      "step": 1835,
      "loss": 0.0842
    },
    {
      "step": 1840,
      "loss": 0.0868
    },
    {
      "step": 1845,
      "loss": 0.0836
    },
    {
      "step": 1850,
      "loss": 0.0818
    },
    {
      "step": 1855,
      "loss": 0.0818
    },
    {
      "step": 1860,
      "loss": 0.0856
    },
    {
      "step": 1865,
      "loss": 0.0851
    },
    {
      "step": 1870,
      "loss": 0.087
    },
    {
      "step": 1875,
      "loss": 0.0842
    },
    {
      "step": 1880,
      "loss": 0.0847
    },
    {
      "step": 1885,
      "loss": 0.0836
    },
    {
      "step": 1890,
      "loss": 0.0826
    },
    {
      "step": 1895,
      "loss": 0.0853
    },
    {
      "step": 1900,
      "loss": 0.0833
    },
    {
      "step": 1905,
      "loss": 0.0847
    },
    {
      "step": 1910,
      "loss": 0.083
    },
    {
      "step": 1915,
      "loss": 0.0832
    },
    {
      "step": 1920,
      "loss": 0.0792
    },
    {
      "step": 1925,
      "loss": 0.0819
    },
    {
      "step": 1930,
      "loss": 0.0831
    },
    {
      "step": 1935,
      "loss": 0.0795
    },
    {
      "step": 1940,
      "loss": 0.0879
    },
    {
      "step": 1945,
      "loss": 0.0827
    },
    {
      "step": 1950,
      "loss": 0.0799
    },
    {
      "step": 1955,
      "loss": 0.08
    },
    {
      "step": 1960,
      "loss": 0.0865
    },
    {
      "step": 1965,
      "loss": 0.0841
    },
    {
      "step": 1970,
      "loss": 0.085
    },
    {
      "step": 1975,
      "loss": 0.0829
    },
    {
      "step": 1980,
      "loss": 0.0835
    },
    {
      "step": 1985,
      "loss": 0.0804
    },
    {
      "step": 1990,
      "loss": 0.0845
    },
    {
      "step": 1995,
      "loss": 0.0832
    },
    {
      "step": 2000,
      "loss": 0.0834
    },
    {
      "step": 2005,
      "loss": 0.078
    },
    {
      "step": 2010,
      "loss": 0.0816
    },
    {
      "step": 2015,
      "loss": 0.0811
    },
    {
      "step": 2020,
      "loss": 0.085
    },
    {
      "step": 2025,
      "loss": 0.0807
    },
    {
      "step": 2030,
      "loss": 0.0829
    },
    {
      "step": 2035,
      "loss": 0.084
    },
    {
      "step": 2040,
      "loss": 0.0811
    },
    {
      "step": 2045,
      "loss": 0.0848
    },
    {
      "step": 2050,
      "loss": 0.0817
    },
    {
      "step": 2055,
      "loss": 0.0838
    },
    {
      "step": 2060,
      "loss": 0.0812
    },
    {
      "step": 2065,
      "loss": 0.0839
    },
    {
      "step": 2070,
      "loss": 0.0847
    },
    {
      "step": 2075,
      "loss": 0.078
    },
    {
      "step": 2080,
      "loss": 0.0855
    },
    {
      "step": 2085,
      "loss": 0.0847
    },
    {
      "step": 2090,
      "loss": 0.0783
    },
    {
      "step": 2095,
      "loss": 0.0838
    },
    {
      "step": 2100,
      "loss": 0.0826
    },
    {
      "step": 2105,
      "loss": 0.0842
    },
    {
      "step": 2110,
      "loss": 0.0799
    },
    {
      "step": 2115,
      "loss": 0.085
    },
    {
      "step": 2120,
      "loss": 0.0834
    },
    {
      "step": 2125,
      "loss": 0.0834
    },
    {
      "step": 2130,
      "loss": 0.0808
    },
    {
      "step": 2135,
      "loss": 0.0835
    },
    {
      "step": 2140,
      "loss": 0.0791
    },
    {
      "step": 2145,
      "loss": 0.0823
    },
    {
      "step": 2150,
      "loss": 0.0801
    },
    {
      "step": 2155,
      "loss": 0.0815
    },
    {
      "step": 2160,
      "loss": 0.0815
    },
    {
      "step": 2165,
      "loss": 0.0831
    },
    {
      "step": 2170,
      "loss": 0.0818
    },
    {
      "step": 2175,
      "loss": 0.0816
    },
    {
      "step": 2180,
      "loss": 0.0835
    },
    {
      "step": 2185,
      "loss": 0.0801
    },
    {
      "step": 2190,
      "loss": 0.0868
    },
    {
      "step": 2195,
      "loss": 0.0821
    },
    {
      "step": 2200,
      "loss": 0.0804
    },
    {
      "step": 2205,
      "loss": 0.0841
    },
    {
      "step": 2210,
      "loss": 0.0834
    },
    {
      "step": 2215,
      "loss": 0.0834
    },
    {
      "step": 2220,
      "loss": 0.0812
    },
    {
      "step": 2225,
      "loss": 0.0809
    },
    {
      "step": 2230,
      "loss": 0.0832
    },
    {
      "step": 2235,
      "loss": 0.0836
    },
    {
      "step": 2240,
      "loss": 0.0834
    },
    {
      "step": 2245,
      "loss": 0.0835
    },
    {
      "step": 2250,
      "loss": 0.0819
    },
    {
      "step": 2255,
      "loss": 0.0804
    },
    {
      "step": 2260,
      "loss": 0.0832
    },
    {
      "step": 2265,
      "loss": 0.0824
    },
    {
      "step": 2270,
      "loss": 0.0857
    },
    {
      "step": 2275,
      "loss": 0.083
    },
    {
      "step": 2280,
      "loss": 0.0869
    },
    {
      "step": 2285,
      "loss": 0.0805
    },
    {
      "step": 2290,
      "loss": 0.0819
    },
    {
      "step": 2295,
      "loss": 0.0805
    },
    {
      "step": 2300,
      "loss": 0.0852
    },
    {
      "step": 2305,
      "loss": 0.0796
    },
    {
      "step": 2310,
      "loss": 0.0872
    },
    {
      "step": 2315,
      "loss": 0.0808
    },
    {
      "step": 2320,
      "loss": 0.0839
    },
    {
      "step": 2325,
      "loss": 0.0843
    },
    {
      "step": 2330,
      "loss": 0.0832
    },
    {
      "step": 2335,
      "loss": 0.0815
    },
    {
      "step": 2340,
      "loss": 0.0779
    },
    {
      "step": 2345,
      "loss": 0.0842
    },
    {
      "step": 2350,
      "loss": 0.084
    },
    {
      "step": 2355,
      "loss": 0.0821
    },
    {
      "step": 2360,
      "loss": 0.0806
    },
    {
      "step": 2365,
      "loss": 0.0845
    },
    {
      "step": 2370,
      "loss": 0.0818
    },
    {
      "step": 2375,
      "loss": 0.0867
    },
    {
      "step": 2380,
      "loss": 0.0823
    },
    {
      "step": 2385,
      "loss": 0.0815
    },
    {
      "step": 2390,
      "loss": 0.0839
    },
    {
      "step": 2395,
      "loss": 0.0811
    },
    {
      "step": 2400,
      "loss": 0.0834
    },
    {
      "step": 2405,
      "loss": 0.0825
    },
    {
      "step": 2410,
      "loss": 0.0801
    },
    {
      "step": 2415,
      "loss": 0.0832
    },
    {
      "step": 2420,
      "loss": 0.0878
    },
    {
      "step": 2425,
      "loss": 0.0868
    },
    {
      "step": 2430,
      "loss": 0.0835
    },
    {
      "step": 2435,
      "loss": 0.0812
    },
    {
      "step": 2440,
      "loss": 0.083
    },
    {
      "step": 2445,
      "loss": 0.0833
    },
    {
      "step": 2450,
      "loss": 0.0859
    },
    {
      "step": 2455,
      "loss": 0.0829
    },
    {
      "step": 2460,
      "loss": 0.0826
    },
    {
      "step": 2465,
      "loss": 0.0795
    },
    {
      "step": 2470,
      "loss": 0.0832
    },
    {
      "step": 2475,
      "loss": 0.0817
    },
    {
      "step": 2480,
      "loss": 0.0835
    },
    {
      "step": 2485,
      "loss": 0.0839
    },
    {
      "step": 2490,
      "loss": 0.0852
    },
    {
      "step": 2495,
      "loss": 0.082
    },
    {
      "step": 2500,
      "loss": 0.0777
    },
    {
      "step": 2505,
      "loss": 0.0819
    },
    {
      "step": 2510,
      "loss": 0.0868
    },
    {
      "step": 2515,
      "loss": 0.0812
    },
    {
      "step": 2520,
      "loss": 0.0801
    },
    {
      "step": 2525,
      "loss": 0.0817
    },
    {
      "step": 2530,
      "loss": 0.0841
    },
    {
      "step": 2535,
      "loss": 0.0807
    },
    {
      "step": 2540,
      "loss": 0.0814
    },
    {
      "step": 2545,
      "loss": 0.082
    },
    {
      "step": 2550,
      "loss": 0.0808
    },
    {
      "step": 2555,
      "loss": 0.0784
    },
    {
      "step": 2560,
      "loss": 0.082
    },
    {
      "step": 2565,
      "loss": 0.0814
    },
    {
      "step": 2570,
      "loss": 0.0809
    },
    {
      "step": 2575,
      "loss": 0.0819
    },
    {
      "step": 2580,
      "loss": 0.0801
    },
    {
      "step": 2585,
      "loss": 0.0821
    },
    {
      "step": 2590,
      "loss": 0.0824
    },
    {
      "step": 2595,
      "loss": 0.0835
    },
    {
      "step": 2600,
      "loss": 0.0817
    },
    {
      "step": 2605,
      "loss": 0.0827
    },
    {
      "step": 2610,
      "loss": 0.0825
    },
    {
      "step": 2615,
      "loss": 0.0829
    },
    {
      "step": 2620,
      "loss": 0.0824
    },
    {
      "step": 2625,
      "loss": 0.0828
    },
    {
      "step": 2630,
      "loss": 0.0822
    },
    {
      "step": 2635,
      "loss": 0.0833
    },
    {
      "step": 2640,
      "loss": 0.0803
    },
    {
      "step": 2645,
      "loss": 0.0867
    },
    {
      "step": 2650,
      "loss": 0.0832
    },
    {
      "step": 2655,
      "loss": 0.0841
    },
    {
      "step": 2660,
      "loss": 0.0838
    },
    {
      "step": 2665,
      "loss": 0.0819
    },
    {
      "step": 2670,
      "loss": 0.0788
    },
    {
      "step": 2675,
      "loss": 0.0797
    },
    {
      "step": 2680,
      "loss": 0.0796
    },
    {
      "step": 2685,
      "loss": 0.0819
    },
    {
      "step": 2690,
      "loss": 0.0836
    },
    {
      "step": 2695,
      "loss": 0.0788
    },
    {
      "step": 2700,
      "loss": 0.0811
    },
    {
      "step": 2705,
      "loss": 0.0808
    },
    {
      "step": 2710,
      "loss": 0.0868
    },
    {
      "step": 2715,
      "loss": 0.0846
    },
    {
      "step": 2720,
      "loss": 0.082
    },
    {
      "step": 2725,
      "loss": 0.0795
    },
    {
      "step": 2730,
      "loss": 0.0803
    },
    {
      "step": 2735,
      "loss": 0.0824
    },
    {
      "step": 2740,
      "loss": 0.0811
    },
    {
      "step": 2745,
      "loss": 0.0839
    },
    {
      "step": 2750,
      "loss": 0.0807
    },
    {
      "step": 2755,
      "loss": 0.0868
    },
    {
      "step": 2760,
      "loss": 0.0808
    },
    {
      "step": 2765,
      "loss": 0.0795
    },
    {
      "step": 2770,
      "loss": 0.0807
    },
    {
      "step": 2775,
      "loss": 0.0789
    },
    {
      "step": 2780,
      "loss": 0.0836
    },
    {
      "step": 2785,
      "loss": 0.0839
    },
    {
      "step": 2790,
      "loss": 0.0831
    },
    {
      "step": 2795,
      "loss": 0.0801
    },
    {
      "step": 2800,
      "loss": 0.0866
    },
    {
      "step": 2805,
      "loss": 0.0828
    },
    {
      "step": 2810,
      "loss": 0.0814
    },
    {
      "step": 2815,
      "loss": 0.0799
    },
    {
      "step": 2820,
      "loss": 0.0831
    },
    {
      "step": 2825,
      "loss": 0.0767
    },
    {
      "step": 2830,
      "loss": 0.0799
    },
    {
      "step": 2835,
      "loss": 0.0839
    },
    {
      "step": 2840,
      "loss": 0.0822
    },
    {
      "step": 2845,
      "loss": 0.0823
    },
    {
      "step": 2850,
      "loss": 0.0831
    },
    {
      "step": 2855,
      "loss": 0.0827
    },
    {
      "step": 2860,
      "loss": 0.0828
    },
    {
      "step": 2865,
      "loss": 0.0784
    },
    {
      "step": 2870,
      "loss": 0.0833
    },
    {
      "step": 2875,
      "loss": 0.0804
    },
    {
      "step": 2880,
      "loss": 0.0774
    },
    {
      "step": 2885,
      "loss": 0.0789
    },
    {
      "step": 2890,
      "loss": 0.0805
    },
    {
      "step": 2895,
      "loss": 0.0845
    },
    {
      "step": 2900,
      "loss": 0.0847
    },
    {
      "step": 2905,
      "loss": 0.0812
    },
    {
      "step": 2910,
      "loss": 0.0796
    },
    {
      "step": 2915,
      "loss": 0.0819
    },
    {
      "step": 2920,
      "loss": 0.0805
    },
    {
      "step": 2925,
      "loss": 0.0812
    },
    {
      "step": 2930,
      "loss": 0.0834
    },
    {
      "step": 2935,
      "loss": 0.0832
    },
    {
      "step": 2940,
      "loss": 0.0848
    },
    {
      "step": 2945,
      "loss": 0.0829
    },
    {
      "step": 2950,
      "loss": 0.0843
    },
    {
      "step": 2955,
      "loss": 0.0827
    },
    {
      "step": 2960,
      "loss": 0.0789
    },
    {
      "step": 2965,
      "loss": 0.0791
    },
    {
      "step": 2970,
      "loss": 0.0854
    },
    {
      "step": 2975,
      "loss": 0.0807
    },
    {
      "step": 2980,
      "loss": 0.0813
    },
    {
      "step": 2985,
      "loss": 0.0799
    },
    {
      "step": 2990,
      "loss": 0.079
    },
    {
      "step": 2995,
      "loss": 0.0793
    },
    {
      "step": 3000,
      "loss": 0.0863
    },
    {
      "step": 3005,
      "loss": 0.0854
    },
    {
      "step": 3010,
      "loss": 0.081
    },
    {
      "step": 3015,
      "loss": 0.0806
    },
    {
      "step": 3020,
      "loss": 0.0795
    },
    {
      "step": 3025,
      "loss": 0.0819
    },
    {
      "step": 3030,
      "loss": 0.0821
    },
    {
      "step": 3035,
      "loss": 0.0861
    },
    {
      "step": 3040,
      "loss": 0.0823
    },
    {
      "step": 3045,
      "loss": 0.0812
    },
    {
      "step": 3050,
      "loss": 0.0802
    },
    {
      "step": 3055,
      "loss": 0.0797
    },
    {
      "step": 3060,
      "loss": 0.0838
    },
    {
      "step": 3065,
      "loss": 0.0824
    },
    {
      "step": 3070,
      "loss": 0.0872
    },
    {
      "step": 3075,
      "loss": 0.0816
    },
    {
      "step": 3080,
      "loss": 0.0805
    },
    {
      "step": 3085,
      "loss": 0.0777
    },
    {
      "step": 3090,
      "loss": 0.0812
    },
    {
      "step": 3095,
      "loss": 0.0794
    },
    {
      "step": 3100,
      "loss": 0.079
    },
    {
      "step": 3105,
      "loss": 0.0803
    },
    {
      "step": 3110,
      "loss": 0.0771
    },
    {
      "step": 3115,
      "loss": 0.0805
    },
    {
      "step": 3120,
      "loss": 0.0872
    },
    {
      "step": 3125,
      "loss": 0.0831
    },
    {
      "step": 3130,
      "loss": 0.0826
    },
    {
      "step": 3135,
      "loss": 0.0823
    },
    {
      "step": 3140,
      "loss": 0.0814
    },
    {
      "step": 3145,
      "loss": 0.0798
    },
    {
      "step": 3150,
      "loss": 0.0796
    },
    {
      "step": 3155,
      "loss": 0.0863
    },
    {
      "step": 3160,
      "loss": 0.0841
    },
    {
      "step": 3165,
      "loss": 0.081
    },
    {
      "step": 3170,
      "loss": 0.0793
    },
    {
      "step": 3175,
      "loss": 0.08
    },
    {
      "step": 3180,
      "loss": 0.0833
    },
    {
      "step": 3185,
      "loss": 0.0817
    },
    {
      "step": 3190,
      "loss": 0.0799
    },
    {
      "step": 3195,
      "loss": 0.0831
    },
    {
      "step": 3200,
      "loss": 0.0781
    },
    {
      "step": 3205,
      "loss": 0.0795
    },
    {
      "step": 3210,
      "loss": 0.0812
    },
    {
      "step": 3215,
      "loss": 0.0816
    },
    {
      "step": 3220,
      "loss": 0.0804
    },
    {
      "step": 3225,
      "loss": 0.0828
    },
    {
      "step": 3230,
      "loss": 0.0868
    },
    {
      "step": 3235,
      "loss": 0.0806
    },
    {
      "step": 3240,
      "loss": 0.0793
    },
    {
      "step": 3245,
      "loss": 0.0804
    },
    {
      "step": 3250,
      "loss": 0.0838
    },
    {
      "step": 3255,
      "loss": 0.0838
    },
    {
      "step": 3260,
      "loss": 0.0839
    },
    {
      "step": 3265,
      "loss": 0.0851
    },
    {
      "step": 3270,
      "loss": 0.0825
    },
    {
      "step": 3275,
      "loss": 0.0824
    },
    {
      "step": 3280,
      "loss": 0.0839
    },
    {
      "step": 3285,
      "loss": 0.08
    },
    {
      "step": 3290,
      "loss": 0.0818
    },
    {
      "step": 3295,
      "loss": 0.0778
    },
    {
      "step": 3300,
      "loss": 0.0823
    },
    {
      "step": 3305,
      "loss": 0.0814
    },
    {
      "step": 3310,
      "loss": 0.0801
    },
    {
      "step": 3315,
      "loss": 0.0798
    },
    {
      "step": 3320,
      "loss": 0.0839
    },
    {
      "step": 3325,
      "loss": 0.0789
    },
    {
      "step": 3330,
      "loss": 0.0802
    },
    {
      "step": 3335,
      "loss": 0.0822
    },
    {
      "step": 3340,
      "loss": 0.0794
    },
    {
      "step": 3345,
      "loss": 0.0821
    },
    {
      "step": 3350,
      "loss": 0.0804
    },
    {
      "step": 3355,
      "loss": 0.087
    },
    {
      "step": 3360,
      "loss": 0.0848
    },
    {
      "step": 3365,
      "loss": 0.084
    },
    {
      "step": 3370,
      "loss": 0.081
    },
    {
      "step": 3375,
      "loss": 0.0843
    },
    {
      "step": 3380,
      "loss": 0.0779
    },
    {
      "step": 3385,
      "loss": 0.0831
    },
    {
      "step": 3390,
      "loss": 0.0817
    },
    {
      "step": 3395,
      "loss": 0.0787
    },
    {
      "step": 3400,
      "loss": 0.0841
    },
    {
      "step": 3405,
      "loss": 0.0828
    },
    {
      "step": 3410,
      "loss": 0.0809
    },
    {
      "step": 3415,
      "loss": 0.0811
    },
    {
      "step": 3420,
      "loss": 0.0835
    },
    {
      "step": 3425,
      "loss": 0.0753
    },
    {
      "step": 3430,
      "loss": 0.0833
    },
    {
      "step": 3435,
      "loss": 0.0797
    },
    {
      "step": 3440,
      "loss": 0.0775
    },
    {
      "step": 3445,
      "loss": 0.0827
    },
    {
      "step": 3450,
      "loss": 0.0825
    },
    {
      "step": 3455,
      "loss": 0.0802
    },
    {
      "step": 3460,
      "loss": 0.0828
    },
    {
      "step": 3465,
      "loss": 0.0847
    },
    {
      "step": 3470,
      "loss": 0.0791
    },
    {
      "step": 3475,
      "loss": 0.0782
    },
    {
      "step": 3480,
      "loss": 0.0815
    },
    {
      "step": 3485,
      "loss": 0.0835
    },
    {
      "step": 3490,
      "loss": 0.0795
    },
    {
      "step": 3495,
      "loss": 0.0775
    },
    {
      "step": 3500,
      "loss": 0.0817
    },
    {
      "step": 3505,
      "loss": 0.0829
    },
    {
      "step": 3510,
      "loss": 0.0782
    },
    {
      "step": 3515,
      "loss": 0.0793
    },
    {
      "step": 3520,
      "loss": 0.0772
    },
    {
      "step": 3525,
      "loss": 0.0824
    },
    {
      "step": 3530,
      "loss": 0.0834
    },
    {
      "step": 3535,
      "loss": 0.0815
    },
    {
      "step": 3540,
      "loss": 0.0774
    },
    {
      "step": 3545,
      "loss": 0.0809
    },
    {
      "step": 3550,
      "loss": 0.0762
    },
    {
      "step": 3555,
      "loss": 0.0824
    },
    {
      "step": 3560,
      "loss": 0.0793
    },
    {
      "step": 3565,
      "loss": 0.0782
    },
    {
      "step": 3570,
      "loss": 0.0833
    },
    {
      "step": 3575,
      "loss": 0.0794
    },
    {
      "step": 3580,
      "loss": 0.0802
    },
    {
      "step": 3585,
      "loss": 0.0807
    },
    {
      "step": 3590,
      "loss": 0.0807
    },
    {
      "step": 3595,
      "loss": 0.0835
    },
    {
      "step": 3600,
      "loss": 0.0817
    },
    {
      "step": 3605,
      "loss": 0.0824
    },
    {
      "step": 3610,
      "loss": 0.0792
    },
    {
      "step": 3615,
      "loss": 0.0778
    },
    {
      "step": 3620,
      "loss": 0.0764
    },
    {
      "step": 3625,
      "loss": 0.0822
    },
    {
      "step": 3630,
      "loss": 0.0771
    },
    {
      "step": 3635,
      "loss": 0.0803
    },
    {
      "step": 3640,
      "loss": 0.0802
    },
    {
      "step": 3645,
      "loss": 0.0804
    },
    {
      "step": 3650,
      "loss": 0.0817
    },
    {
      "step": 3655,
      "loss": 0.0813
    },
    {
      "step": 3660,
      "loss": 0.0849
    },
    {
      "step": 3665,
      "loss": 0.0821
    },
    {
      "step": 3670,
      "loss": 0.0794
    },
    {
      "step": 3675,
      "loss": 0.079
    },
    {
      "step": 3680,
      "loss": 0.0794
    },
    {
      "step": 3685,
      "loss": 0.0818
    },
    {
      "step": 3690,
      "loss": 0.0836
    },
    {
      "step": 3695,
      "loss": 0.0787
    },
    {
      "step": 3700,
      "loss": 0.0791
    },
    {
      "step": 3705,
      "loss": 0.0848
    },
    {
      "step": 3710,
      "loss": 0.081
    },
    {
      "step": 3715,
      "loss": 0.0838
    },
    {
      "step": 3720,
      "loss": 0.0801
    },
    {
      "step": 3725,
      "loss": 0.0783
    },
    {
      "step": 3730,
      "loss": 0.0854
    },
    {
      "step": 3735,
      "loss": 0.0791
    },
    {
      "step": 3740,
      "loss": 0.0876
    },
    {
      "step": 3745,
      "loss": 0.0826
    },
    {
      "step": 3750,
      "loss": 0.082
    },
    {
      "step": 3755,
      "loss": 0.075
    },
    {
      "step": 3760,
      "loss": 0.0799
    },
    {
      "step": 3765,
      "loss": 0.0777
    },
    {
      "step": 3770,
      "loss": 0.0793
    },
    {
      "step": 3775,
      "loss": 0.0802
    },
    {
      "step": 3780,
      "loss": 0.0811
    },
    {
      "step": 3785,
      "loss": 0.0831
    },
    {
      "step": 3790,
      "loss": 0.083
    },
    {
      "step": 3795,
      "loss": 0.0807
    },
    {
      "step": 3800,
      "loss": 0.0795
    },
    {
      "step": 3805,
      "loss": 0.0793
    },
    {
      "step": 3810,
      "loss": 0.0814
    },
    {
      "step": 3815,
      "loss": 0.0806
    },
    {
      "step": 3820,
      "loss": 0.0801
    },
    {
      "step": 3825,
      "loss": 0.0821
    },
    {
      "step": 3830,
      "loss": 0.081
    },
    {
      "step": 3835,
      "loss": 0.0781
    },
    {
      "step": 3840,
      "loss": 0.0793
    },
    {
      "step": 3845,
      "loss": 0.0803
    },
    {
      "step": 3850,
      "loss": 0.0793
    },
    {
      "step": 3855,
      "loss": 0.0809
    },
    {
      "step": 3860,
      "loss": 0.0856
    },
    {
      "step": 3865,
      "loss": 0.08
    },
    {
      "step": 3870,
      "loss": 0.0815
    },
    {
      "step": 3875,
      "loss": 0.0823
    },
    {
      "step": 3880,
      "loss": 0.0806
    },
    {
      "step": 3885,
      "loss": 0.081
    },
    {
      "step": 3890,
      "loss": 0.0795
    },
    {
      "step": 3895,
      "loss": 0.0805
    },
    {
      "step": 3900,
      "loss": 0.0809
    },
    {
      "step": 3905,
      "loss": 0.081
    },
    {
      "step": 3910,
      "loss": 0.0774
    },
    {
      "step": 3915,
      "loss": 0.079
    },
    {
      "step": 3920,
      "loss": 0.0784
    },
    {
      "step": 3925,
      "loss": 0.0798
    },
    {
      "step": 3930,
      "loss": 0.0826
    },
    {
      "step": 3935,
      "loss": 0.0782
    },
    {
      "step": 3940,
      "loss": 0.0816
    },
    {
      "step": 3945,
      "loss": 0.0814
    },
    {
      "step": 3950,
      "loss": 0.077
    },
    {
      "step": 3955,
      "loss": 0.0783
    },
    {
      "step": 3960,
      "loss": 0.0815
    },
    {
      "step": 3965,
      "loss": 0.0829
    },
    {
      "step": 3970,
      "loss": 0.0819
    },
    {
      "step": 3975,
      "loss": 0.0829
    },
    {
      "step": 3980,
      "loss": 0.0777
    },
    {
      "step": 3985,
      "loss": 0.0774
    },
    {
      "step": 3990,
      "loss": 0.0809
    },
    {
      "step": 3995,
      "loss": 0.0799
    },
    {
      "step": 4000,
      "loss": 0.0815
    },
    {
      "step": 4005,
      "loss": 0.0785
    },
    {
      "step": 4010,
      "loss": 0.0846
    },
    {
      "step": 4015,
      "loss": 0.0805
    },
    {
      "step": 4020,
      "loss": 0.0806
    },
    {
      "step": 4025,
      "loss": 0.0794
    },
    {
      "step": 4030,
      "loss": 0.0825
    },
    {
      "step": 4035,
      "loss": 0.0803
    },
    {
      "step": 4040,
      "loss": 0.085
    },
    {
      "step": 4045,
      "loss": 0.0803
    },
    {
      "step": 4050,
      "loss": 0.0796
    },
    {
      "step": 4055,
      "loss": 0.082
    },
    {
      "step": 4060,
      "loss": 0.0824
    },
    {
      "step": 4065,
      "loss": 0.0825
    },
    {
      "step": 4070,
      "loss": 0.0816
    },
    {
      "step": 4075,
      "loss": 0.0776
    },
    {
      "step": 4080,
      "loss": 0.0786
    },
    {
      "step": 4085,
      "loss": 0.081
    },
    {
      "step": 4090,
      "loss": 0.0821
    },
    {
      "step": 4095,
      "loss": 0.0771
    },
    {
      "step": 4100,
      "loss": 0.0823
    },
    {
      "step": 4105,
      "loss": 0.0805
    },
    {
      "step": 4110,
      "loss": 0.0802
    },
    {
      "step": 4115,
      "loss": 0.0797
    },
    {
      "step": 4120,
      "loss": 0.0803
    },
    {
      "step": 4125,
      "loss": 0.0797
    },
    {
      "step": 4130,
      "loss": 0.0782
    },
    {
      "step": 4135,
      "loss": 0.0758
    },
    {
      "step": 4140,
      "loss": 0.0811
    },
    {
      "step": 4145,
      "loss": 0.0817
    },
    {
      "step": 4150,
      "loss": 0.0832
    },
    {
      "step": 4155,
      "loss": 0.0819
    },
    {
      "step": 4160,
      "loss": 0.0815
    },
    {
      "step": 4165,
      "loss": 0.0808
    },
    {
      "step": 4170,
      "loss": 0.0814
    },
    {
      "step": 4175,
      "loss": 0.0776
    },
    {
      "step": 4180,
      "loss": 0.0835
    },
    {
      "step": 4185,
      "loss": 0.0804
    },
    {
      "step": 4190,
      "loss": 0.077
    },
    {
      "step": 4195,
      "loss": 0.0822
    },
    {
      "step": 4200,
      "loss": 0.0821
    },
    {
      "step": 4205,
      "loss": 0.0802
    },
    {
      "step": 4210,
      "loss": 0.081
    },
    {
      "step": 4215,
      "loss": 0.0795
    },
    {
      "step": 4220,
      "loss": 0.0789
    },
    {
      "step": 4225,
      "loss": 0.0818
    },
    {
      "step": 4230,
      "loss": 0.0775
    },
    {
      "step": 4235,
      "loss": 0.0816
    },
    {
      "step": 4240,
      "loss": 0.0806
    },
    {
      "step": 4245,
      "loss": 0.0818
    },
    {
      "step": 4250,
      "loss": 0.0803
    },
    {
      "step": 4255,
      "loss": 0.0778
    },
    {
      "step": 4260,
      "loss": 0.0824
    },
    {
      "step": 4265,
      "loss": 0.0784
    },
    {
      "step": 4270,
      "loss": 0.0764
    },
    {
      "step": 4275,
      "loss": 0.0751
    },
    {
      "step": 4280,
      "loss": 0.0798
    },
    {
      "step": 4285,
      "loss": 0.0783
    },
    {
      "step": 4290,
      "loss": 0.0846
    },
    {
      "step": 4295,
      "loss": 0.0778
    },
    {
      "step": 4300,
      "loss": 0.0785
    },
    {
      "step": 4305,
      "loss": 0.0782
    },
    {
      "step": 4310,
      "loss": 0.0772
    },
    {
      "step": 4315,
      "loss": 0.0834
    },
    {
      "step": 4320,
      "loss": 0.0814
    },
    {
      "step": 4325,
      "loss": 0.0773
    },
    {
      "step": 4330,
      "loss": 0.0787
    },
    {
      "step": 4335,
      "loss": 0.0779
    },
    {
      "step": 4340,
      "loss": 0.0837
    },
    {
      "step": 4345,
      "loss": 0.0807
    },
    {
      "step": 4350,
      "loss": 0.0811
    },
    {
      "step": 4355,
      "loss": 0.0804
    },
    {
      "step": 4360,
      "loss": 0.0828
    },
    {
      "step": 4365,
      "loss": 0.0763
    },
    {
      "step": 4370,
      "loss": 0.0809
    },
    {
      "step": 4375,
      "loss": 0.079
    },
    {
      "step": 4380,
      "loss": 0.0824
    },
    {
      "step": 4385,
      "loss": 0.0813
    },
    {
      "step": 4390,
      "loss": 0.0817
    },
    {
      "step": 4395,
      "loss": 0.078
    },
    {
      "step": 4400,
      "loss": 0.0792
    },
    {
      "step": 4405,
      "loss": 0.0792
    },
    {
      "step": 4410,
      "loss": 0.0786
    },
    {
      "step": 4415,
      "loss": 0.0797
    },
    {
      "step": 4420,
      "loss": 0.0825
    },
    {
      "step": 4425,
      "loss": 0.0772
    },
    {
      "step": 4430,
      "loss": 0.0813
    },
    {
      "step": 4435,
      "loss": 0.083
    },
    {
      "step": 4440,
      "loss": 0.0824
    },
    {
      "step": 4445,
      "loss": 0.0799
    },
    {
      "step": 4450,
      "loss": 0.0794
    },
    {
      "step": 4455,
      "loss": 0.0827
    },
    {
      "step": 4460,
      "loss": 0.0808
    },
    {
      "step": 4465,
      "loss": 0.078
    },
    {
      "step": 4470,
      "loss": 0.0812
    },
    {
      "step": 4475,
      "loss": 0.0786
    },
    {
      "step": 4480,
      "loss": 0.0792
    },
    {
      "step": 4485,
      "loss": 0.0798
    },
    {
      "step": 4490,
      "loss": 0.0799
    },
    {
      "step": 4495,
      "loss": 0.0789
    },
    {
      "step": 4500,
      "loss": 0.0779
    },
    {
      "step": 4505,
      "loss": 0.0752
    },
    {
      "step": 4510,
      "loss": 0.0795
    },
    {
      "step": 4515,
      "loss": 0.0797
    },
    {
      "step": 4520,
      "loss": 0.0821
    },
    {
      "step": 4525,
      "loss": 0.0799
    },
    {
      "step": 4530,
      "loss": 0.0815
    },
    {
      "step": 4535,
      "loss": 0.0769
    },
    {
      "step": 4540,
      "loss": 0.079
    },
    {
      "step": 4545,
      "loss": 0.0808
    },
    {
      "step": 4550,
      "loss": 0.0776
    },
    {
      "step": 4555,
      "loss": 0.0777
    },
    {
      "step": 4560,
      "loss": 0.0858
    },
    {
      "step": 4565,
      "loss": 0.0789
    },
    {
      "step": 4570,
      "loss": 0.08
    },
    {
      "step": 4575,
      "loss": 0.0773
    },
    {
      "step": 4580,
      "loss": 0.0834
    },
    {
      "step": 4585,
      "loss": 0.0795
    },
    {
      "step": 4590,
      "loss": 0.0793
    },
    {
      "step": 4595,
      "loss": 0.0778
    },
    {
      "step": 4600,
      "loss": 0.0807
    },
    {
      "step": 4605,
      "loss": 0.0798
    },
    {
      "step": 4610,
      "loss": 0.0757
    },
    {
      "step": 4615,
      "loss": 0.077
    },
    {
      "step": 4620,
      "loss": 0.0815
    },
    {
      "step": 4625,
      "loss": 0.0786
    },
    {
      "step": 4630,
      "loss": 0.0808
    },
    {
      "step": 4635,
      "loss": 0.0801
    },
    {
      "step": 4640,
      "loss": 0.0835
    },
    {
      "step": 4645,
      "loss": 0.0831
    },
    {
      "step": 4650,
      "loss": 0.0791
    },
    {
      "step": 4655,
      "loss": 0.0753
    },
    {
      "step": 4660,
      "loss": 0.0853
    },
    {
      "step": 4665,
      "loss": 0.0799
    },
    {
      "step": 4670,
      "loss": 0.083
    },
    {
      "step": 4675,
      "loss": 0.0767
    },
    {
      "step": 4680,
      "loss": 0.0799
    },
    {
      "step": 4685,
      "loss": 0.0807
    },
    {
      "step": 4690,
      "loss": 0.0795
    },
    {
      "step": 4695,
      "loss": 0.083
    },
    {
      "step": 4700,
      "loss": 0.0804
    },
    {
      "step": 4705,
      "loss": 0.0816
    },
    {
      "step": 4710,
      "loss": 0.0761
    },
    {
      "step": 4715,
      "loss": 0.0797
    },
    {
      "step": 4720,
      "loss": 0.0777
    },
    {
      "step": 4725,
      "loss": 0.081
    },
    {
      "step": 4730,
      "loss": 0.0787
    },
    {
      "step": 4735,
      "loss": 0.0813
    },
    {
      "step": 4740,
      "loss": 0.0816
    },
    {
      "step": 4745,
      "loss": 0.0813
    },
    {
      "step": 4750,
      "loss": 0.0807
    },
    {
      "step": 4755,
      "loss": 0.0799
    },
    {
      "step": 4760,
      "loss": 0.0838
    },
    {
      "step": 4765,
      "loss": 0.0783
    },
    {
      "step": 4770,
      "loss": 0.0818
    },
    {
      "step": 4775,
      "loss": 0.0805
    },
    {
      "step": 4780,
      "loss": 0.083
    },
    {
      "step": 4785,
      "loss": 0.0807
    },
    {
      "step": 4790,
      "loss": 0.0784
    },
    {
      "step": 4795,
      "loss": 0.0742
    },
    {
      "step": 4800,
      "loss": 0.0767
    },
    {
      "step": 4805,
      "loss": 0.0813
    },
    {
      "step": 4810,
      "loss": 0.079
    },
    {
      "step": 4815,
      "loss": 0.0793
    },
    {
      "step": 4820,
      "loss": 0.0858
    },
    {
      "step": 4825,
      "loss": 0.0834
    },
    {
      "step": 4830,
      "loss": 0.0794
    },
    {
      "step": 4835,
      "loss": 0.0811
    },
    {
      "step": 4840,
      "loss": 0.0815
    },
    {
      "step": 4845,
      "loss": 0.0791
    },
    {
      "step": 4850,
      "loss": 0.0797
    },
    {
      "step": 4855,
      "loss": 0.0767
    },
    {
      "step": 4860,
      "loss": 0.0821
    },
    {
      "step": 4865,
      "loss": 0.0752
    },
    {
      "step": 4870,
      "loss": 0.082
    },
    {
      "step": 4875,
      "loss": 0.0786
    },
    {
      "step": 4880,
      "loss": 0.08
    },
    {
      "step": 4885,
      "loss": 0.0784
    },
    {
      "step": 4890,
      "loss": 0.0811
    },
    {
      "step": 4895,
      "loss": 0.0784
    },
    {
      "step": 4900,
      "loss": 0.0788
    },
    {
      "step": 4905,
      "loss": 0.0802
    },
    {
      "step": 4910,
      "loss": 0.075
    },
    {
      "step": 4915,
      "loss": 0.0742
    },
    {
      "step": 4920,
      "loss": 0.0797
    },
    {
      "step": 4925,
      "loss": 0.079
    },
    {
      "step": 4930,
      "loss": 0.0774
    },
    {
      "step": 4935,
      "loss": 0.0795
    },
    {
      "step": 4940,
      "loss": 0.0801
    },
    {
      "step": 4945,
      "loss": 0.0794
    },
    {
      "step": 4950,
      "loss": 0.0748
    },
    {
      "step": 4955,
      "loss": 0.0843
    },
    {
      "step": 4960,
      "loss": 0.0774
    },
    {
      "step": 4965,
      "loss": 0.0815
    },
    {
      "step": 4970,
      "loss": 0.0812
    },
    {
      "step": 4975,
      "loss": 0.0791
    },
    {
      "step": 4980,
      "loss": 0.0851
    },
    {
      "step": 4985,
      "loss": 0.079
    },
    {
      "step": 4990,
      "loss": 0.0806
    },
    {
      "step": 4995,
      "loss": 0.0812
    },
    {
      "step": 5000,
      "loss": 0.0786
    },
    {
      "step": 5005,
      "loss": 0.0811
    },
    {
      "step": 5010,
      "loss": 0.077
    },
    {
      "step": 5015,
      "loss": 0.0817
    },
    {
      "step": 5020,
      "loss": 0.0748
    },
    {
      "step": 5025,
      "loss": 0.0778
    },
    {
      "step": 5030,
      "loss": 0.0805
    },
    {
      "step": 5035,
      "loss": 0.0804
    },
    {
      "step": 5040,
      "loss": 0.0788
    },
    {
      "step": 5045,
      "loss": 0.0778
    },
    {
      "step": 5050,
      "loss": 0.0821
    },
    {
      "step": 5055,
      "loss": 0.0831
    },
    {
      "step": 5060,
      "loss": 0.08
    },
    {
      "step": 5065,
      "loss": 0.0803
    },
    {
      "step": 5070,
      "loss": 0.0811
    },
    {
      "step": 5075,
      "loss": 0.0803
    },
    {
      "step": 5080,
      "loss": 0.0805
    },
    {
      "step": 5085,
      "loss": 0.0798
    },
    {
      "step": 5090,
      "loss": 0.0787
    },
    {
      "step": 5095,
      "loss": 0.081
    },
    {
      "step": 5100,
      "loss": 0.0843
    },
    {
      "step": 5105,
      "loss": 0.0787
    },
    {
      "step": 5110,
      "loss": 0.0803
    },
    {
      "step": 5115,
      "loss": 0.0798
    },
    {
      "step": 5120,
      "loss": 0.0792
    },
    {
      "step": 5125,
      "loss": 0.0819
    },
    {
      "step": 5130,
      "loss": 0.0843
    },
    {
      "step": 5135,
      "loss": 0.0763
    },
    {
      "step": 5140,
      "loss": 0.0812
    },
    {
      "step": 5145,
      "loss": 0.0798
    },
    {
      "step": 5150,
      "loss": 0.0803
    },
    {
      "step": 5155,
      "loss": 0.0775
    },
    {
      "step": 5160,
      "loss": 0.0824
    },
    {
      "step": 5165,
      "loss": 0.0779
    },
    {
      "step": 5170,
      "loss": 0.0775
    },
    {
      "step": 5175,
      "loss": 0.0825
    },
    {
      "step": 5180,
      "loss": 0.0787
    },
    {
      "step": 5185,
      "loss": 0.0811
    },
    {
      "step": 5190,
      "loss": 0.0816
    },
    {
      "step": 5195,
      "loss": 0.0771
    },
    {
      "step": 5200,
      "loss": 0.0813
    },
    {
      "step": 5205,
      "loss": 0.0778
    },
    {
      "step": 5210,
      "loss": 0.0855
    },
    {
      "step": 5215,
      "loss": 0.0754
    },
    {
      "step": 5220,
      "loss": 0.0767
    },
    {
      "step": 5225,
      "loss": 0.0783
    },
    {
      "step": 5230,
      "loss": 0.0793
    },
    {
      "step": 5235,
      "loss": 0.0785
    },
    {
      "step": 5240,
      "loss": 0.0797
    },
    {
      "step": 5245,
      "loss": 0.0748
    },
    {
      "step": 5250,
      "loss": 0.0735
    },
    {
      "step": 5255,
      "loss": 0.0841
    },
    {
      "step": 5260,
      "loss": 0.0783
    },
    {
      "step": 5265,
      "loss": 0.0784
    },
    {
      "step": 5270,
      "loss": 0.0792
    },
    {
      "step": 5275,
      "loss": 0.084
    },
    {
      "step": 5280,
      "loss": 0.0806
    },
    {
      "step": 5285,
      "loss": 0.0788
    },
    {
      "step": 5290,
      "loss": 0.0772
    },
    {
      "step": 5295,
      "loss": 0.0798
    },
    {
      "step": 5300,
      "loss": 0.0747
    },
    {
      "step": 5305,
      "loss": 0.0824
    },
    {
      "step": 5310,
      "loss": 0.0785
    },
    {
      "step": 5315,
      "loss": 0.08
    },
    {
      "step": 5320,
      "loss": 0.078
    },
    {
      "step": 5325,
      "loss": 0.0785
    },
    {
      "step": 5330,
      "loss": 0.0754
    },
    {
      "step": 5335,
      "loss": 0.0778
    },
    {
      "step": 5340,
      "loss": 0.0785
    },
    {
      "step": 5345,
      "loss": 0.0804
    },
    {
      "step": 5350,
      "loss": 0.0791
    },
    {
      "step": 5355,
      "loss": 0.0795
    },
    {
      "step": 5360,
      "loss": 0.0797
    },
    {
      "step": 5365,
      "loss": 0.0786
    },
    {
      "step": 5370,
      "loss": 0.0807
    },
    {
      "step": 5375,
      "loss": 0.0792
    },
    {
      "step": 5380,
      "loss": 0.0739
    },
    {
      "step": 5385,
      "loss": 0.0791
    },
    {
      "step": 5390,
      "loss": 0.0815
    },
    {
      "step": 5395,
      "loss": 0.0834
    },
    {
      "step": 5400,
      "loss": 0.0796
    },
    {
      "step": 5405,
      "loss": 0.0819
    },
    {
      "step": 5410,
      "loss": 0.0758
    },
    {
      "step": 5415,
      "loss": 0.08
    },
    {
      "step": 5420,
      "loss": 0.0802
    },
    {
      "step": 5425,
      "loss": 0.0758
    },
    {
      "step": 5430,
      "loss": 0.0809
    },
    {
      "step": 5435,
      "loss": 0.0799
    },
    {
      "step": 5440,
      "loss": 0.0792
    },
    {
      "step": 5445,
      "loss": 0.0771
    },
    {
      "step": 5450,
      "loss": 0.0842
    },
    {
      "step": 5455,
      "loss": 0.0831
    },
    {
      "step": 5460,
      "loss": 0.0778
    },
    {
      "step": 5465,
      "loss": 0.077
    },
    {
      "step": 5470,
      "loss": 0.0762
    },
    {
      "step": 5475,
      "loss": 0.0802
    },
    {
      "step": 5480,
      "loss": 0.0806
    },
    {
      "step": 5485,
      "loss": 0.079
    },
    {
      "step": 5490,
      "loss": 0.0777
    },
    {
      "step": 5495,
      "loss": 0.079
    },
    {
      "step": 5500,
      "loss": 0.0744
    },
    {
      "step": 5505,
      "loss": 0.0792
    },
    {
      "step": 5510,
      "loss": 0.0783
    },
    {
      "step": 5515,
      "loss": 0.0744
    },
    {
      "step": 5520,
      "loss": 0.0731
    },
    {
      "step": 5525,
      "loss": 0.0781
    },
    {
      "step": 5530,
      "loss": 0.0776
    },
    {
      "step": 5535,
      "loss": 0.079
    },
    {
      "step": 5540,
      "loss": 0.0796
    },
    {
      "step": 5545,
      "loss": 0.0763
    },
    {
      "step": 5550,
      "loss": 0.0779
    },
    {
      "step": 5555,
      "loss": 0.0739
    },
    {
      "step": 5560,
      "loss": 0.08
    },
    {
      "step": 5565,
      "loss": 0.0794
    },
    {
      "step": 5570,
      "loss": 0.0825
    },
    {
      "step": 5575,
      "loss": 0.0758
    },
    {
      "step": 5580,
      "loss": 0.0697
    },
    {
      "step": 5585,
      "loss": 0.075
    },
    {
      "step": 5590,
      "loss": 0.0808
    },
    {
      "step": 5595,
      "loss": 0.0786
    },
    {
      "step": 5600,
      "loss": 0.0767
    },
    {
      "step": 5605,
      "loss": 0.0777
    },
    {
      "step": 5610,
      "loss": 0.0768
    },
    {
      "step": 5615,
      "loss": 0.0793
    },
    {
      "step": 5620,
      "loss": 0.077
    },
    {
      "step": 5625,
      "loss": 0.0826
    },
    {
      "step": 5630,
      "loss": 0.0764
    },
    {
      "step": 5635,
      "loss": 0.077
    },
    {
      "step": 5640,
      "loss": 0.0769
    },
    {
      "step": 5645,
      "loss": 0.0775
    },
    {
      "step": 5650,
      "loss": 0.0791
    },
    {
      "step": 5655,
      "loss": 0.0767
    },
    {
      "step": 5660,
      "loss": 0.0806
    },
    {
      "step": 5665,
      "loss": 0.0793
    },
    {
      "step": 5670,
      "loss": 0.0768
    },
    {
      "step": 5675,
      "loss": 0.0759
    },
    {
      "step": 5680,
      "loss": 0.0767
    },
    {
      "step": 5685,
      "loss": 0.0815
    },
    {
      "step": 5690,
      "loss": 0.0829
    },
    {
      "step": 5695,
      "loss": 0.0735
    },
    {
      "step": 5700,
      "loss": 0.0772
    },
    {
      "step": 5705,
      "loss": 0.0781
    },
    {
      "step": 5710,
      "loss": 0.0838
    },
    {
      "step": 5715,
      "loss": 0.077
    },
    {
      "step": 5720,
      "loss": 0.0771
    },
    {
      "step": 5725,
      "loss": 0.0731
    },
    {
      "step": 5730,
      "loss": 0.0802
    },
    {
      "step": 5735,
      "loss": 0.0701
    },
    {
      "step": 5740,
      "loss": 0.0755
    },
    {
      "step": 5745,
      "loss": 0.0767
    },
    {
      "step": 5750,
      "loss": 0.0749
    },
    {
      "step": 5755,
      "loss": 0.0809
    },
    {
      "step": 5760,
      "loss": 0.0725
    },
    {
      "step": 5765,
      "loss": 0.0769
    },
    {
      "step": 5770,
      "loss": 0.0776
    },
    {
      "step": 5775,
      "loss": 0.076
    },
    {
      "step": 5780,
      "loss": 0.0797
    },
    {
      "step": 5785,
      "loss": 0.0767
    },
    {
      "step": 5790,
      "loss": 0.0772
    },
    {
      "step": 5795,
      "loss": 0.0842
    },
    {
      "step": 5800,
      "loss": 0.075
    },
    {
      "step": 5805,
      "loss": 0.0748
    },
    {
      "step": 5810,
      "loss": 0.078
    },
    {
      "step": 5815,
      "loss": 0.076
    },
    {
      "step": 5820,
      "loss": 0.0743
    },
    {
      "step": 5825,
      "loss": 0.0798
    },
    {
      "step": 5830,
      "loss": 0.0769
    },
    {
      "step": 5835,
      "loss": 0.0816
    },
    {
      "step": 5840,
      "loss": 0.0793
    },
    {
      "step": 5845,
      "loss": 0.0738
    },
    {
      "step": 5850,
      "loss": 0.0741
    },
    {
      "step": 5855,
      "loss": 0.0746
    },
    {
      "step": 5860,
      "loss": 0.0771
    },
    {
      "step": 5865,
      "loss": 0.0761
    },
    {
      "step": 5870,
      "loss": 0.0816
    },
    {
      "step": 5875,
      "loss": 0.0751
    },
    {
      "step": 5880,
      "loss": 0.0768
    },
    {
      "step": 5885,
      "loss": 0.0792
    },
    {
      "step": 5890,
      "loss": 0.0805
    },
    {
      "step": 5895,
      "loss": 0.0775
    },
    {
      "step": 5900,
      "loss": 0.0787
    },
    {
      "step": 5905,
      "loss": 0.0777
    },
    {
      "step": 5910,
      "loss": 0.0769
    },
    {
      "step": 5915,
      "loss": 0.0792
    },
    {
      "step": 5920,
      "loss": 0.0782
    },
    {
      "step": 5925,
      "loss": 0.0793
    },
    {
      "step": 5930,
      "loss": 0.0776
    },
    {
      "step": 5935,
      "loss": 0.0773
    },
    {
      "step": 5940,
      "loss": 0.0824
    },
    {
      "step": 5945,
      "loss": 0.0734
    },
    {
      "step": 5950,
      "loss": 0.0787
    },
    {
      "step": 5955,
      "loss": 0.0771
    },
    {
      "step": 5960,
      "loss": 0.0762
    },
    {
      "step": 5965,
      "loss": 0.0755
    },
    {
      "step": 5970,
      "loss": 0.0766
    },
    {
      "step": 5975,
      "loss": 0.0722
    },
    {
      "step": 5980,
      "loss": 0.0772
    },
    {
      "step": 5985,
      "loss": 0.0765
    },
    {
      "step": 5990,
      "loss": 0.0759
    },
    {
      "step": 5995,
      "loss": 0.076
    },
    {
      "step": 6000,
      "loss": 0.0767
    },
    {
      "step": 6005,
      "loss": 0.0782
    },
    {
      "step": 6010,
      "loss": 0.0774
    },
    {
      "step": 6015,
      "loss": 0.0795
    },
    {
      "step": 6020,
      "loss": 0.077
    },
    {
      "step": 6025,
      "loss": 0.0809
    },
    {
      "step": 6030,
      "loss": 0.0778
    },
    {
      "step": 6035,
      "loss": 0.0749
    },
    {
      "step": 6040,
      "loss": 0.0769
    },
    {
      "step": 6045,
      "loss": 0.0784
    },
    {
      "step": 6050,
      "loss": 0.0758
    },
    {
      "step": 6055,
      "loss": 0.0757
    },
    {
      "step": 6060,
      "loss": 0.0778
    },
    {
      "step": 6065,
      "loss": 0.0753
    },
    {
      "step": 6070,
      "loss": 0.0788
    },
    {
      "step": 6075,
      "loss": 0.0778
    },
    {
      "step": 6080,
      "loss": 0.0794
    },
    {
      "step": 6085,
      "loss": 0.0827
    },
    {
      "step": 6090,
      "loss": 0.0781
    },
    {
      "step": 6095,
      "loss": 0.077
    },
    {
      "step": 6100,
      "loss": 0.0801
    },
    {
      "step": 6105,
      "loss": 0.0757
    },
    {
      "step": 6110,
      "loss": 0.0756
    },
    {
      "step": 6115,
      "loss": 0.0794
    },
    {
      "step": 6120,
      "loss": 0.0737
    },
    {
      "step": 6125,
      "loss": 0.0779
    },
    {
      "step": 6130,
      "loss": 0.0792
    },
    {
      "step": 6135,
      "loss": 0.0759
    },
    {
      "step": 6140,
      "loss": 0.0765
    },
    {
      "step": 6145,
      "loss": 0.0754
    },
    {
      "step": 6150,
      "loss": 0.0775
    },
    {
      "step": 6155,
      "loss": 0.0712
    },
    {
      "step": 6160,
      "loss": 0.0785
    },
    {
      "step": 6165,
      "loss": 0.0754
    },
    {
      "step": 6170,
      "loss": 0.075
    },
    {
      "step": 6175,
      "loss": 0.0783
    },
    {
      "step": 6180,
      "loss": 0.0763
    },
    {
      "step": 6185,
      "loss": 0.078
    },
    {
      "step": 6190,
      "loss": 0.0777
    },
    {
      "step": 6195,
      "loss": 0.0748
    },
    {
      "step": 6200,
      "loss": 0.079
    },
    {
      "step": 6205,
      "loss": 0.0731
    },
    {
      "step": 6210,
      "loss": 0.0775
    },
    {
      "step": 6215,
      "loss": 0.0751
    },
    {
      "step": 6220,
      "loss": 0.0783
    },
    {
      "step": 6225,
      "loss": 0.0745
    },
    {
      "step": 6230,
      "loss": 0.0791
    },
    {
      "step": 6235,
      "loss": 0.079
    },
    {
      "step": 6240,
      "loss": 0.0752
    },
    {
      "step": 6245,
      "loss": 0.077
    },
    {
      "step": 6250,
      "loss": 0.0763
    },
    {
      "step": 6255,
      "loss": 0.0778
    },
    {
      "step": 6260,
      "loss": 0.0799
    },
    {
      "step": 6265,
      "loss": 0.08
    },
    {
      "step": 6270,
      "loss": 0.0763
    },
    {
      "step": 6275,
      "loss": 0.082
    },
    {
      "step": 6280,
      "loss": 0.0796
    },
    {
      "step": 6285,
      "loss": 0.0813
    },
    {
      "step": 6290,
      "loss": 0.0769
    },
    {
      "step": 6295,
      "loss": 0.078
    },
    {
      "step": 6300,
      "loss": 0.0788
    },
    {
      "step": 6305,
      "loss": 0.0754
    },
    {
      "step": 6310,
      "loss": 0.0744
    },
    {
      "step": 6315,
      "loss": 0.0772
    },
    {
      "step": 6320,
      "loss": 0.074
    },
    {
      "step": 6325,
      "loss": 0.08
    },
    {
      "step": 6330,
      "loss": 0.0825
    },
    {
      "step": 6335,
      "loss": 0.0799
    },
    {
      "step": 6340,
      "loss": 0.0776
    },
    {
      "step": 6345,
      "loss": 0.0766
    },
    {
      "step": 6350,
      "loss": 0.0772
    },
    {
      "step": 6355,
      "loss": 0.0755
    },
    {
      "step": 6360,
      "loss": 0.0783
    },
    {
      "step": 6365,
      "loss": 0.0754
    },
    {
      "step": 6370,
      "loss": 0.073
    },
    {
      "step": 6375,
      "loss": 0.0811
    },
    {
      "step": 6380,
      "loss": 0.0831
    },
    {
      "step": 6385,
      "loss": 0.0762
    },
    {
      "step": 6390,
      "loss": 0.0747
    },
    {
      "step": 6395,
      "loss": 0.0758
    },
    {
      "step": 6400,
      "loss": 0.08
    },
    {
      "step": 6405,
      "loss": 0.0762
    },
    {
      "step": 6410,
      "loss": 0.0745
    },
    {
      "step": 6415,
      "loss": 0.0762
    },
    {
      "step": 6420,
      "loss": 0.0758
    },
    {
      "step": 6425,
      "loss": 0.0768
    },
    {
      "step": 6430,
      "loss": 0.0731
    },
    {
      "step": 6435,
      "loss": 0.0745
    },
    {
      "step": 6440,
      "loss": 0.0727
    },
    {
      "step": 6445,
      "loss": 0.0764
    },
    {
      "step": 6450,
      "loss": 0.077
    },
    {
      "step": 6455,
      "loss": 0.0746
    },
    {
      "step": 6460,
      "loss": 0.0743
    },
    {
      "step": 6465,
      "loss": 0.0764
    },
    {
      "step": 6470,
      "loss": 0.0747
    },
    {
      "step": 6475,
      "loss": 0.0744
    },
    {
      "step": 6480,
      "loss": 0.0771
    },
    {
      "step": 6485,
      "loss": 0.074
    },
    {
      "step": 6490,
      "loss": 0.0737
    },
    {
      "step": 6495,
      "loss": 0.0746
    },
    {
      "step": 6500,
      "loss": 0.0781
    },
    {
      "step": 6505,
      "loss": 0.0793
    },
    {
      "step": 6510,
      "loss": 0.0799
    },
    {
      "step": 6515,
      "loss": 0.0761
    },
    {
      "step": 6520,
      "loss": 0.0833
    },
    {
      "step": 6525,
      "loss": 0.0763
    },
    {
      "step": 6530,
      "loss": 0.075
    },
    {
      "step": 6535,
      "loss": 0.0742
    },
    {
      "step": 6540,
      "loss": 0.0791
    },
    {
      "step": 6545,
      "loss": 0.0758
    },
    {
      "step": 6550,
      "loss": 0.0805
    },
    {
      "step": 6555,
      "loss": 0.0774
    },
    {
      "step": 6560,
      "loss": 0.0768
    },
    {
      "step": 6565,
      "loss": 0.0747
    },
    {
      "step": 6570,
      "loss": 0.076
    },
    {
      "step": 6575,
      "loss": 0.0774
    },
    {
      "step": 6580,
      "loss": 0.0792
    },
    {
      "step": 6585,
      "loss": 0.0753
    },
    {
      "step": 6590,
      "loss": 0.0748
    },
    {
      "step": 6595,
      "loss": 0.0778
    },
    {
      "step": 6600,
      "loss": 0.0764
    },
    {
      "step": 6605,
      "loss": 0.0705
    },
    {
      "step": 6610,
      "loss": 0.0747
    },
    {
      "step": 6615,
      "loss": 0.0773
    },
    {
      "step": 6620,
      "loss": 0.078
    },
    {
      "step": 6625,
      "loss": 0.0776
    },
    {
      "step": 6630,
      "loss": 0.0758
    },
    {
      "step": 6635,
      "loss": 0.0763
    },
    {
      "step": 6640,
      "loss": 0.0749
    },
    {
      "step": 6645,
      "loss": 0.0735
    },
    {
      "step": 6650,
      "loss": 0.0797
    },
    {
      "step": 6655,
      "loss": 0.0762
    },
    {
      "step": 6660,
      "loss": 0.0766
    },
    {
      "step": 6665,
      "loss": 0.0781
    },
    {
      "step": 6670,
      "loss": 0.0752
    },
    {
      "step": 6675,
      "loss": 0.0762
    },
    {
      "step": 6680,
      "loss": 0.0709
    },
    {
      "step": 6685,
      "loss": 0.0793
    },
    {
      "step": 6690,
      "loss": 0.0758
    },
    {
      "step": 6695,
      "loss": 0.0745
    },
    {
      "step": 6700,
      "loss": 0.0766
    },
    {
      "step": 6705,
      "loss": 0.0774
    },
    {
      "step": 6710,
      "loss": 0.0785
    },
    {
      "step": 6715,
      "loss": 0.0788
    },
    {
      "step": 6720,
      "loss": 0.0791
    },
    {
      "step": 6725,
      "loss": 0.0792
    },
    {
      "step": 6730,
      "loss": 0.0781
    },
    {
      "step": 6735,
      "loss": 0.0738
    },
    {
      "step": 6740,
      "loss": 0.0772
    },
    {
      "step": 6745,
      "loss": 0.0737
    },
    {
      "step": 6750,
      "loss": 0.0768
    },
    {
      "step": 6755,
      "loss": 0.08
    },
    {
      "step": 6760,
      "loss": 0.0763
    },
    {
      "step": 6765,
      "loss": 0.0794
    },
    {
      "step": 6770,
      "loss": 0.0749
    },
    {
      "step": 6775,
      "loss": 0.0737
    },
    {
      "step": 6780,
      "loss": 0.0781
    },
    {
      "step": 6785,
      "loss": 0.0761
    },
    {
      "step": 6790,
      "loss": 0.0784
    },
    {
      "step": 6795,
      "loss": 0.0759
    },
    {
      "step": 6800,
      "loss": 0.0754
    },
    {
      "step": 6805,
      "loss": 0.0758
    },
    {
      "step": 6810,
      "loss": 0.0741
    },
    {
      "step": 6815,
      "loss": 0.0741
    },
    {
      "step": 6820,
      "loss": 0.0785
    },
    {
      "step": 6825,
      "loss": 0.0781
    },
    {
      "step": 6830,
      "loss": 0.0764
    },
    {
      "step": 6835,
      "loss": 0.077
    },
    {
      "step": 6840,
      "loss": 0.0794
    },
    {
      "step": 6845,
      "loss": 0.0784
    },
    {
      "step": 6850,
      "loss": 0.0795
    },
    {
      "step": 6855,
      "loss": 0.0769
    },
    {
      "step": 6860,
      "loss": 0.0744
    },
    {
      "step": 6865,
      "loss": 0.0744
    },
    {
      "step": 6870,
      "loss": 0.0759
    },
    {
      "step": 6875,
      "loss": 0.0756
    },
    {
      "step": 6880,
      "loss": 0.0811
    },
    {
      "step": 6885,
      "loss": 0.0705
    },
    {
      "step": 6890,
      "loss": 0.0736
    },
    {
      "step": 6895,
      "loss": 0.0793
    },
    {
      "step": 6900,
      "loss": 0.0765
    },
    {
      "step": 6905,
      "loss": 0.0777
    },
    {
      "step": 6910,
      "loss": 0.0756
    },
    {
      "step": 6915,
      "loss": 0.0787
    },
    {
      "step": 6920,
      "loss": 0.0792
    },
    {
      "step": 6925,
      "loss": 0.0788
    },
    {
      "step": 6930,
      "loss": 0.0769
    },
    {
      "step": 6935,
      "loss": 0.0715
    },
    {
      "step": 6940,
      "loss": 0.0777
    },
    {
      "step": 6945,
      "loss": 0.0753
    },
    {
      "step": 6950,
      "loss": 0.0761
    },
    {
      "step": 6955,
      "loss": 0.0774
    },
    {
      "step": 6960,
      "loss": 0.0765
    },
    {
      "step": 6965,
      "loss": 0.0767
    },
    {
      "step": 6970,
      "loss": 0.0731
    },
    {
      "step": 6975,
      "loss": 0.077
    },
    {
      "step": 6980,
      "loss": 0.0731
    },
    {
      "step": 6985,
      "loss": 0.0749
    },
    {
      "step": 6990,
      "loss": 0.0773
    },
    {
      "step": 6995,
      "loss": 0.0765
    },
    {
      "step": 7000,
      "loss": 0.0714
    },
    {
      "step": 7005,
      "loss": 0.0771
    },
    {
      "step": 7010,
      "loss": 0.0755
    },
    {
      "step": 7015,
      "loss": 0.0793
    },
    {
      "step": 7020,
      "loss": 0.0766
    },
    {
      "step": 7025,
      "loss": 0.0754
    },
    {
      "step": 7030,
      "loss": 0.0733
    },
    {
      "step": 7035,
      "loss": 0.0741
    },
    {
      "step": 7040,
      "loss": 0.0788
    },
    {
      "step": 7045,
      "loss": 0.0789
    },
    {
      "step": 7050,
      "loss": 0.0789
    },
    {
      "step": 7055,
      "loss": 0.075
    },
    {
      "step": 7060,
      "loss": 0.0735
    },
    {
      "step": 7065,
      "loss": 0.0747
    },
    {
      "step": 7070,
      "loss": 0.0764
    },
    {
      "step": 7075,
      "loss": 0.0707
    },
    {
      "step": 7080,
      "loss": 0.0746
    },
    {
      "step": 7085,
      "loss": 0.075
    },
    {
      "step": 7090,
      "loss": 0.0705
    },
    {
      "step": 7095,
      "loss": 0.0773
    },
    {
      "step": 7100,
      "loss": 0.074
    },
    {
      "step": 7105,
      "loss": 0.0776
    },
    {
      "step": 7110,
      "loss": 0.0767
    },
    {
      "step": 7115,
      "loss": 0.0723
    },
    {
      "step": 7120,
      "loss": 0.0767
    },
    {
      "step": 7125,
      "loss": 0.0793
    },
    {
      "step": 7130,
      "loss": 0.0724
    },
    {
      "step": 7135,
      "loss": 0.0805
    },
    {
      "step": 7140,
      "loss": 0.0782
    },
    {
      "step": 7145,
      "loss": 0.0759
    },
    {
      "step": 7150,
      "loss": 0.0762
    },
    {
      "step": 7155,
      "loss": 0.0756
    },
    {
      "step": 7160,
      "loss": 0.074
    },
    {
      "step": 7165,
      "loss": 0.0749
    },
    {
      "step": 7170,
      "loss": 0.0759
    },
    {
      "step": 7175,
      "loss": 0.0729
    },
    {
      "step": 7180,
      "loss": 0.0774
    },
    {
      "step": 7185,
      "loss": 0.0734
    },
    {
      "step": 7190,
      "loss": 0.0786
    },
    {
      "step": 7195,
      "loss": 0.0748
    },
    {
      "step": 7200,
      "loss": 0.0748
    },
    {
      "step": 7205,
      "loss": 0.08
    },
    {
      "step": 7210,
      "loss": 0.078
    },
    {
      "step": 7215,
      "loss": 0.0776
    },
    {
      "step": 7220,
      "loss": 0.0743
    },
    {
      "step": 7225,
      "loss": 0.0784
    },
    {
      "step": 7230,
      "loss": 0.0783
    },
    {
      "step": 7235,
      "loss": 0.0737
    },
    {
      "step": 7240,
      "loss": 0.075
    },
    {
      "step": 7245,
      "loss": 0.0755
    },
    {
      "step": 7250,
      "loss": 0.0782
    },
    {
      "step": 7255,
      "loss": 0.0784
    },
    {
      "step": 7260,
      "loss": 0.072
    },
    {
      "step": 7265,
      "loss": 0.0744
    },
    {
      "step": 7270,
      "loss": 0.074
    },
    {
      "step": 7275,
      "loss": 0.072
    },
    {
      "step": 7280,
      "loss": 0.0772
    },
    {
      "step": 7285,
      "loss": 0.0758
    },
    {
      "step": 7290,
      "loss": 0.0743
    },
    {
      "step": 7295,
      "loss": 0.0769
    },
    {
      "step": 7300,
      "loss": 0.0782
    },
    {
      "step": 7305,
      "loss": 0.0754
    },
    {
      "step": 7310,
      "loss": 0.0767
    },
    {
      "step": 7315,
      "loss": 0.0748
    },
    {
      "step": 7320,
      "loss": 0.0772
    },
    {
      "step": 7325,
      "loss": 0.076
    },
    {
      "step": 7330,
      "loss": 0.0783
    },
    {
      "step": 7335,
      "loss": 0.0809
    },
    {
      "step": 7340,
      "loss": 0.0766
    },
    {
      "step": 7345,
      "loss": 0.0753
    },
    {
      "step": 7350,
      "loss": 0.0741
    },
    {
      "step": 7355,
      "loss": 0.0757
    },
    {
      "step": 7360,
      "loss": 0.0712
    },
    {
      "step": 7365,
      "loss": 0.077
    },
    {
      "step": 7370,
      "loss": 0.0759
    },
    {
      "step": 7375,
      "loss": 0.0751
    },
    {
      "step": 7380,
      "loss": 0.074
    },
    {
      "step": 7385,
      "loss": 0.0777
    },
    {
      "step": 7390,
      "loss": 0.077
    },
    {
      "step": 7395,
      "loss": 0.0751
    },
    {
      "step": 7400,
      "loss": 0.0778
    },
    {
      "step": 7405,
      "loss": 0.0774
    },
    {
      "step": 7410,
      "loss": 0.0776
    },
    {
      "step": 7415,
      "loss": 0.0789
    },
    {
      "step": 7420,
      "loss": 0.074
    },
    {
      "step": 7425,
      "loss": 0.0779
    },
    {
      "step": 7430,
      "loss": 0.0763
    },
    {
      "step": 7435,
      "loss": 0.0751
    },
    {
      "step": 7440,
      "loss": 0.0771
    },
    {
      "step": 7445,
      "loss": 0.0757
    },
    {
      "step": 7450,
      "loss": 0.073
    },
    {
      "step": 7455,
      "loss": 0.0724
    },
    {
      "step": 7460,
      "loss": 0.0781
    },
    {
      "step": 7465,
      "loss": 0.0756
    },
    {
      "step": 7470,
      "loss": 0.0705
    },
    {
      "step": 7475,
      "loss": 0.0753
    },
    {
      "step": 7480,
      "loss": 0.0788
    },
    {
      "step": 7485,
      "loss": 0.0737
    },
    {
      "step": 7490,
      "loss": 0.0782
    },
    {
      "step": 7495,
      "loss": 0.0791
    },
    {
      "step": 7500,
      "loss": 0.0745
    },
    {
      "step": 7505,
      "loss": 0.0803
    },
    {
      "step": 7510,
      "loss": 0.0735
    },
    {
      "step": 7515,
      "loss": 0.0774
    },
    {
      "step": 7520,
      "loss": 0.0749
    },
    {
      "step": 7525,
      "loss": 0.0747
    },
    {
      "step": 7530,
      "loss": 0.0797
    },
    {
      "step": 7535,
      "loss": 0.0774
    },
    {
      "step": 7540,
      "loss": 0.0778
    },
    {
      "step": 7545,
      "loss": 0.0729
    },
    {
      "step": 7550,
      "loss": 0.0774
    },
    {
      "step": 7555,
      "loss": 0.0744
    },
    {
      "step": 7560,
      "loss": 0.0759
    },
    {
      "step": 7565,
      "loss": 0.074
    },
    {
      "step": 7570,
      "loss": 0.0752
    },
    {
      "step": 7575,
      "loss": 0.072
    },
    {
      "step": 7580,
      "loss": 0.0768
    },
    {
      "step": 7585,
      "loss": 0.0766
    },
    {
      "step": 7590,
      "loss": 0.0689
    },
    {
      "step": 7595,
      "loss": 0.0736
    },
    {
      "step": 7600,
      "loss": 0.0714
    },
    {
      "step": 7605,
      "loss": 0.0732
    },
    {
      "step": 7610,
      "loss": 0.0765
    },
    {
      "step": 7615,
      "loss": 0.0768
    },
    {
      "step": 7620,
      "loss": 0.0753
    },
    {
      "step": 7625,
      "loss": 0.0742
    },
    {
      "step": 7630,
      "loss": 0.0761
    },
    {
      "step": 7635,
      "loss": 0.0745
    },
    {
      "step": 7640,
      "loss": 0.0784
    },
    {
      "step": 7645,
      "loss": 0.0728
    },
    {
      "step": 7650,
      "loss": 0.0765
    },
    {
      "step": 7655,
      "loss": 0.0734
    },
    {
      "step": 7660,
      "loss": 0.072
    },
    {
      "step": 7665,
      "loss": 0.076
    },
    {
      "step": 7670,
      "loss": 0.0717
    },
    {
      "step": 7675,
      "loss": 0.0744
    },
    {
      "step": 7680,
      "loss": 0.0755
    },
    {
      "step": 7685,
      "loss": 0.074
    },
    {
      "step": 7690,
      "loss": 0.0737
    },
    {
      "step": 7695,
      "loss": 0.0798
    },
    {
      "step": 7700,
      "loss": 0.074
    },
    {
      "step": 7705,
      "loss": 0.0774
    },
    {
      "step": 7710,
      "loss": 0.0774
    },
    {
      "step": 7715,
      "loss": 0.0734
    },
    {
      "step": 7720,
      "loss": 0.0758
    },
    {
      "step": 7725,
      "loss": 0.0759
    },
    {
      "step": 7730,
      "loss": 0.0773
    },
    {
      "step": 7735,
      "loss": 0.0748
    },
    {
      "step": 7740,
      "loss": 0.0762
    },
    {
      "step": 7745,
      "loss": 0.0767
    },
    {
      "step": 7750,
      "loss": 0.0735
    },
    {
      "step": 7755,
      "loss": 0.074
    },
    {
      "step": 7760,
      "loss": 0.0745
    },
    {
      "step": 7765,
      "loss": 0.0797
    },
    {
      "step": 7770,
      "loss": 0.0712
    },
    {
      "step": 7775,
      "loss": 0.0707
    },
    {
      "step": 7780,
      "loss": 0.0762
    },
    {
      "step": 7785,
      "loss": 0.0739
    },
    {
      "step": 7790,
      "loss": 0.0741
    },
    {
      "step": 7795,
      "loss": 0.0777
    },
    {
      "step": 7800,
      "loss": 0.0734
    },
    {
      "step": 7805,
      "loss": 0.0751
    },
    {
      "step": 7810,
      "loss": 0.0766
    },
    {
      "step": 7815,
      "loss": 0.0781
    },
    {
      "step": 7820,
      "loss": 0.0741
    },
    {
      "step": 7825,
      "loss": 0.0734
    },
    {
      "step": 7830,
      "loss": 0.074
    },
    {
      "step": 7835,
      "loss": 0.077
    },
    {
      "step": 7840,
      "loss": 0.0774
    },
    {
      "step": 7845,
      "loss": 0.0781
    },
    {
      "step": 7850,
      "loss": 0.0718
    },
    {
      "step": 7855,
      "loss": 0.0756
    },
    {
      "step": 7860,
      "loss": 0.0755
    },
    {
      "step": 7865,
      "loss": 0.0764
    },
    {
      "step": 7870,
      "loss": 0.0757
    },
    {
      "step": 7875,
      "loss": 0.0784
    },
    {
      "step": 7880,
      "loss": 0.0768
    },
    {
      "step": 7885,
      "loss": 0.0788
    },
    {
      "step": 7890,
      "loss": 0.0776
    },
    {
      "step": 7895,
      "loss": 0.0716
    },
    {
      "step": 7900,
      "loss": 0.0756
    },
    {
      "step": 7905,
      "loss": 0.0746
    },
    {
      "step": 7910,
      "loss": 0.0738
    },
    {
      "step": 7915,
      "loss": 0.0781
    },
    {
      "step": 7920,
      "loss": 0.0733
    },
    {
      "step": 7925,
      "loss": 0.074
    },
    {
      "step": 7930,
      "loss": 0.0776
    },
    {
      "step": 7935,
      "loss": 0.077
    },
    {
      "step": 7940,
      "loss": 0.0737
    },
    {
      "step": 7945,
      "loss": 0.0735
    },
    {
      "step": 7950,
      "loss": 0.0718
    },
    {
      "step": 7955,
      "loss": 0.0793
    },
    {
      "step": 7960,
      "loss": 0.0723
    },
    {
      "step": 7965,
      "loss": 0.0764
    },
    {
      "step": 7970,
      "loss": 0.0747
    },
    {
      "step": 7975,
      "loss": 0.0771
    },
    {
      "step": 7980,
      "loss": 0.0769
    },
    {
      "step": 7985,
      "loss": 0.0768
    },
    {
      "step": 7990,
      "loss": 0.0755
    },
    {
      "step": 7995,
      "loss": 0.0781
    },
    {
      "step": 8000,
      "loss": 0.0754
    },
    {
      "step": 8005,
      "loss": 0.0731
    },
    {
      "step": 8010,
      "loss": 0.0787
    },
    {
      "step": 8015,
      "loss": 0.0755
    },
    {
      "step": 8020,
      "loss": 0.0735
    },
    {
      "step": 8025,
      "loss": 0.0794
    },
    {
      "step": 8030,
      "loss": 0.0761
    },
    {
      "step": 8035,
      "loss": 0.0775
    },
    {
      "step": 8040,
      "loss": 0.0727
    },
    {
      "step": 8045,
      "loss": 0.0769
    },
    {
      "step": 8050,
      "loss": 0.0832
    },
    {
      "step": 8055,
      "loss": 0.0768
    },
    {
      "step": 8060,
      "loss": 0.0743
    },
    {
      "step": 8065,
      "loss": 0.0729
    },
    {
      "step": 8070,
      "loss": 0.077
    },
    {
      "step": 8075,
      "loss": 0.074
    },
    {
      "step": 8080,
      "loss": 0.0701
    },
    {
      "step": 8085,
      "loss": 0.0766
    },
    {
      "step": 8090,
      "loss": 0.0749
    },
    {
      "step": 8095,
      "loss": 0.0723
    },
    {
      "step": 8100,
      "loss": 0.071
    },
    {
      "step": 8105,
      "loss": 0.0728
    },
    {
      "step": 8110,
      "loss": 0.0756
    },
    {
      "step": 8115,
      "loss": 0.0783
    },
    {
      "step": 8120,
      "loss": 0.0778
    },
    {
      "step": 8125,
      "loss": 0.0776
    },
    {
      "step": 8130,
      "loss": 0.0733
    },
    {
      "step": 8135,
      "loss": 0.0739
    },
    {
      "step": 8140,
      "loss": 0.0774
    },
    {
      "step": 8145,
      "loss": 0.0759
    },
    {
      "step": 8150,
      "loss": 0.0785
    },
    {
      "step": 8155,
      "loss": 0.0747
    },
    {
      "step": 8160,
      "loss": 0.074
    },
    {
      "step": 8165,
      "loss": 0.0738
    },
    {
      "step": 8170,
      "loss": 0.0731
    },
    {
      "step": 8175,
      "loss": 0.0745
    },
    {
      "step": 8180,
      "loss": 0.0744
    },
    {
      "step": 8185,
      "loss": 0.0813
    },
    {
      "step": 8190,
      "loss": 0.0734
    },
    {
      "step": 8195,
      "loss": 0.0772
    },
    {
      "step": 8200,
      "loss": 0.0757
    },
    {
      "step": 8205,
      "loss": 0.0732
    },
    {
      "step": 8210,
      "loss": 0.077
    },
    {
      "step": 8215,
      "loss": 0.0773
    },
    {
      "step": 8220,
      "loss": 0.0776
    },
    {
      "step": 8225,
      "loss": 0.0778
    },
    {
      "step": 8230,
      "loss": 0.0719
    },
    {
      "step": 8235,
      "loss": 0.0727
    },
    {
      "step": 8240,
      "loss": 0.0738
    },
    {
      "step": 8245,
      "loss": 0.0747
    },
    {
      "step": 8250,
      "loss": 0.0758
    },
    {
      "step": 8255,
      "loss": 0.0743
    },
    {
      "step": 8260,
      "loss": 0.0709
    },
    {
      "step": 8265,
      "loss": 0.0735
    },
    {
      "step": 8270,
      "loss": 0.0752
    },
    {
      "step": 8275,
      "loss": 0.0759
    },
    {
      "step": 8280,
      "loss": 0.071
    },
    {
      "step": 8285,
      "loss": 0.0744
    },
    {
      "step": 8290,
      "loss": 0.0725
    },
    {
      "step": 8295,
      "loss": 0.077
    },
    {
      "step": 8300,
      "loss": 0.0752
    },
    {
      "step": 8305,
      "loss": 0.0735
    },
    {
      "step": 8310,
      "loss": 0.0725
    },
    {
      "step": 8315,
      "loss": 0.0793
    },
    {
      "step": 8320,
      "loss": 0.0795
    },
    {
      "step": 8325,
      "loss": 0.0757
    },
    {
      "step": 8330,
      "loss": 0.0781
    },
    {
      "step": 8335,
      "loss": 0.0772
    },
    {
      "step": 8340,
      "loss": 0.0744
    },
    {
      "step": 8345,
      "loss": 0.0773
    },
    {
      "step": 8350,
      "loss": 0.0783
    },
    {
      "step": 8355,
      "loss": 0.0747
    },
    {
      "step": 8360,
      "loss": 0.0788
    },
    {
      "step": 8365,
      "loss": 0.0755
    },
    {
      "step": 8370,
      "loss": 0.0744
    },
    {
      "step": 8375,
      "loss": 0.073
    },
    {
      "step": 8380,
      "loss": 0.0772
    },
    {
      "step": 8385,
      "loss": 0.0721
    },
    {
      "step": 8390,
      "loss": 0.0746
    },
    {
      "step": 8395,
      "loss": 0.0762
    },
    {
      "step": 8400,
      "loss": 0.0715
    },
    {
      "step": 8405,
      "loss": 0.0773
    },
    {
      "step": 8410,
      "loss": 0.0718
    },
    {
      "step": 8415,
      "loss": 0.0765
    },
    {
      "step": 8420,
      "loss": 0.0762
    },
    {
      "step": 8425,
      "loss": 0.0756
    },
    {
      "step": 8430,
      "loss": 0.0728
    },
    {
      "step": 8435,
      "loss": 0.0729
    },
    {
      "step": 8440,
      "loss": 0.0722
    },
    {
      "step": 8445,
      "loss": 0.0762
    },
    {
      "step": 8450,
      "loss": 0.0712
    },
    {
      "step": 8455,
      "loss": 0.0754
    },
    {
      "step": 8460,
      "loss": 0.0756
    },
    {
      "step": 8465,
      "loss": 0.0737
    },
    {
      "step": 8470,
      "loss": 0.0763
    },
    {
      "step": 8475,
      "loss": 0.0711
    },
    {
      "step": 8480,
      "loss": 0.0736
    },
    {
      "step": 8485,
      "loss": 0.0766
    },
    {
      "step": 8490,
      "loss": 0.0696
    },
    {
      "step": 8495,
      "loss": 0.0736
    },
    {
      "step": 8500,
      "loss": 0.0761
    },
    {
      "step": 8505,
      "loss": 0.0768
    },
    {
      "step": 8510,
      "loss": 0.0719
    },
    {
      "step": 8515,
      "loss": 0.0715
    },
    {
      "step": 8520,
      "loss": 0.0795
    },
    {
      "step": 8525,
      "loss": 0.0714
    },
    {
      "step": 8530,
      "loss": 0.0731
    },
    {
      "step": 8535,
      "loss": 0.0794
    },
    {
      "step": 8540,
      "loss": 0.0754
    },
    {
      "step": 8545,
      "loss": 0.0783
    },
    {
      "step": 8550,
      "loss": 0.0718
    },
    {
      "step": 8555,
      "loss": 0.0749
    },
    {
      "step": 8560,
      "loss": 0.078
    },
    {
      "step": 8565,
      "loss": 0.0754
    },
    {
      "step": 8570,
      "loss": 0.0799
    },
    {
      "step": 8575,
      "loss": 0.0762
    },
    {
      "step": 8580,
      "loss": 0.0738
    },
    {
      "step": 8585,
      "loss": 0.076
    },
    {
      "step": 8590,
      "loss": 0.073
    },
    {
      "step": 8595,
      "loss": 0.0769
    },
    {
      "step": 8600,
      "loss": 0.0769
    },
    {
      "step": 8605,
      "loss": 0.0775
    },
    {
      "step": 8610,
      "loss": 0.0767
    },
    {
      "step": 8615,
      "loss": 0.0753
    },
    {
      "step": 8620,
      "loss": 0.0747
    },
    {
      "step": 8625,
      "loss": 0.0691
    },
    {
      "step": 8630,
      "loss": 0.076
    },
    {
      "step": 8635,
      "loss": 0.08
    },
    {
      "step": 8640,
      "loss": 0.0745
    },
    {
      "step": 8645,
      "loss": 0.0751
    },
    {
      "step": 8650,
      "loss": 0.0794
    },
    {
      "step": 8655,
      "loss": 0.0741
    },
    {
      "step": 8660,
      "loss": 0.0784
    },
    {
      "step": 8665,
      "loss": 0.0707
    },
    {
      "step": 8670,
      "loss": 0.0756
    },
    {
      "step": 8675,
      "loss": 0.072
    },
    {
      "step": 8680,
      "loss": 0.077
    },
    {
      "step": 8685,
      "loss": 0.0712
    },
    {
      "step": 8690,
      "loss": 0.0764
    },
    {
      "step": 8695,
      "loss": 0.0709
    },
    {
      "step": 8700,
      "loss": 0.0761
    },
    {
      "step": 8705,
      "loss": 0.0749
    },
    {
      "step": 8710,
      "loss": 0.0714
    },
    {
      "step": 8715,
      "loss": 0.0752
    },
    {
      "step": 8720,
      "loss": 0.0741
    },
    {
      "step": 8725,
      "loss": 0.0792
    },
    {
      "step": 8730,
      "loss": 0.0765
    },
    {
      "step": 8735,
      "loss": 0.0737
    },
    {
      "step": 8740,
      "loss": 0.0736
    },
    {
      "step": 8745,
      "loss": 0.0779
    },
    {
      "step": 8750,
      "loss": 0.078
    },
    {
      "step": 8755,
      "loss": 0.0808
    },
    {
      "step": 8760,
      "loss": 0.0764
    },
    {
      "step": 8765,
      "loss": 0.0782
    },
    {
      "step": 8770,
      "loss": 0.0771
    },
    {
      "step": 8775,
      "loss": 0.0724
    },
    {
      "step": 8780,
      "loss": 0.0782
    },
    {
      "step": 8785,
      "loss": 0.0792
    },
    {
      "step": 8790,
      "loss": 0.0763
    },
    {
      "step": 8795,
      "loss": 0.0739
    },
    {
      "step": 8800,
      "loss": 0.0762
    },
    {
      "step": 8805,
      "loss": 0.0748
    },
    {
      "step": 8810,
      "loss": 0.0731
    },
    {
      "step": 8815,
      "loss": 0.0736
    },
    {
      "step": 8820,
      "loss": 0.0756
    },
    {
      "step": 8825,
      "loss": 0.0777
    },
    {
      "step": 8830,
      "loss": 0.0736
    },
    {
      "step": 8835,
      "loss": 0.0708
    },
    {
      "step": 8840,
      "loss": 0.074
    },
    {
      "step": 8845,
      "loss": 0.0761
    },
    {
      "step": 8850,
      "loss": 0.0755
    },
    {
      "step": 8855,
      "loss": 0.0731
    },
    {
      "step": 8860,
      "loss": 0.0732
    },
    {
      "step": 8865,
      "loss": 0.0764
    },
    {
      "step": 8870,
      "loss": 0.0743
    },
    {
      "step": 8875,
      "loss": 0.0747
    },
    {
      "step": 8880,
      "loss": 0.0736
    },
    {
      "step": 8885,
      "loss": 0.074
    },
    {
      "step": 8890,
      "loss": 0.0752
    },
    {
      "step": 8895,
      "loss": 0.0729
    },
    {
      "step": 8900,
      "loss": 0.0753
    },
    {
      "step": 8905,
      "loss": 0.0716
    },
    {
      "step": 8910,
      "loss": 0.0736
    },
    {
      "step": 8915,
      "loss": 0.076
    },
    {
      "step": 8920,
      "loss": 0.0724
    },
    {
      "step": 8925,
      "loss": 0.0726
    },
    {
      "step": 8930,
      "loss": 0.0755
    },
    {
      "step": 8935,
      "loss": 0.0732
    },
    {
      "step": 8940,
      "loss": 0.0765
    },
    {
      "step": 8945,
      "loss": 0.0761
    },
    {
      "step": 8950,
      "loss": 0.0845
    },
    {
      "step": 8955,
      "loss": 0.0697
    },
    {
      "step": 8960,
      "loss": 0.0769
    },
    {
      "step": 8965,
      "loss": 0.0785
    },
    {
      "step": 8970,
      "loss": 0.0725
    },
    {
      "step": 8975,
      "loss": 0.0722
    },
    {
      "step": 8980,
      "loss": 0.0761
    },
    {
      "step": 8985,
      "loss": 0.0766
    },
    {
      "step": 8990,
      "loss": 0.0764
    },
    {
      "step": 8995,
      "loss": 0.0738
    },
    {
      "step": 9000,
      "loss": 0.0764
    },
    {
      "step": 9005,
      "loss": 0.074
    },
    {
      "step": 9010,
      "loss": 0.0761
    },
    {
      "step": 9015,
      "loss": 0.0779
    },
    {
      "step": 9020,
      "loss": 0.0755
    },
    {
      "step": 9025,
      "loss": 0.0764
    },
    {
      "step": 9030,
      "loss": 0.0722
    },
    {
      "step": 9035,
      "loss": 0.0759
    },
    {
      "step": 9040,
      "loss": 0.0768
    },
    {
      "step": 9045,
      "loss": 0.0718
    },
    {
      "step": 9050,
      "loss": 0.0732
    },
    {
      "step": 9055,
      "loss": 0.0772
    },
    {
      "step": 9060,
      "loss": 0.0723
    },
    {
      "step": 9065,
      "loss": 0.0749
    },
    {
      "step": 9070,
      "loss": 0.0734
    },
    {
      "step": 9075,
      "loss": 0.0718
    },
    {
      "step": 9080,
      "loss": 0.0742
    },
    {
      "step": 9085,
      "loss": 0.0779
    },
    {
      "step": 9090,
      "loss": 0.0721
    },
    {
      "step": 9095,
      "loss": 0.0818
    },
    {
      "step": 9100,
      "loss": 0.0755
    },
    {
      "step": 9105,
      "loss": 0.0753
    },
    {
      "step": 9110,
      "loss": 0.0766
    },
    {
      "step": 9115,
      "loss": 0.0746
    },
    {
      "step": 9120,
      "loss": 0.0713
    },
    {
      "step": 9125,
      "loss": 0.0767
    },
    {
      "step": 9130,
      "loss": 0.0796
    },
    {
      "step": 9135,
      "loss": 0.0743
    },
    {
      "step": 9140,
      "loss": 0.0818
    },
    {
      "step": 9145,
      "loss": 0.0741
    },
    {
      "step": 9150,
      "loss": 0.0777
    },
    {
      "step": 9155,
      "loss": 0.0802
    },
    {
      "step": 9160,
      "loss": 0.078
    },
    {
      "step": 9165,
      "loss": 0.0743
    },
    {
      "step": 9170,
      "loss": 0.0728
    },
    {
      "step": 9175,
      "loss": 0.0758
    },
    {
      "step": 9180,
      "loss": 0.0756
    },
    {
      "step": 9185,
      "loss": 0.074
    },
    {
      "step": 9190,
      "loss": 0.0755
    },
    {
      "step": 9195,
      "loss": 0.0712
    },
    {
      "step": 9200,
      "loss": 0.0748
    },
    {
      "step": 9205,
      "loss": 0.0741
    },
    {
      "step": 9210,
      "loss": 0.0719
    },
    {
      "step": 9215,
      "loss": 0.0751
    },
    {
      "step": 9220,
      "loss": 0.0739
    },
    {
      "step": 9225,
      "loss": 0.0738
    },
    {
      "step": 9230,
      "loss": 0.0749
    },
    {
      "step": 9235,
      "loss": 0.0802
    },
    {
      "step": 9240,
      "loss": 0.0738
    },
    {
      "step": 9245,
      "loss": 0.0763
    },
    {
      "step": 9250,
      "loss": 0.0742
    },
    {
      "step": 9255,
      "loss": 0.0768
    },
    {
      "step": 9260,
      "loss": 0.0766
    },
    {
      "step": 9265,
      "loss": 0.0745
    },
    {
      "step": 9270,
      "loss": 0.0769
    },
    {
      "step": 9275,
      "loss": 0.0792
    },
    {
      "step": 9280,
      "loss": 0.0803
    },
    {
      "step": 9285,
      "loss": 0.08
    },
    {
      "step": 9290,
      "loss": 0.0751
    },
    {
      "step": 9295,
      "loss": 0.0794
    },
    {
      "step": 9300,
      "loss": 0.0742
    },
    {
      "step": 9305,
      "loss": 0.0792
    },
    {
      "step": 9310,
      "loss": 0.0784
    },
    {
      "step": 9315,
      "loss": 0.0781
    },
    {
      "step": 9320,
      "loss": 0.0741
    },
    {
      "step": 9325,
      "loss": 0.0733
    },
    {
      "step": 9330,
      "loss": 0.0716
    },
    {
      "step": 9335,
      "loss": 0.0728
    },
    {
      "step": 9340,
      "loss": 0.0767
    },
    {
      "step": 9345,
      "loss": 0.0759
    },
    {
      "step": 9350,
      "loss": 0.0723
    },
    {
      "step": 9355,
      "loss": 0.0767
    },
    {
      "step": 9360,
      "loss": 0.0774
    },
    {
      "step": 9365,
      "loss": 0.077
    },
    {
      "step": 9370,
      "loss": 0.073
    },
    {
      "step": 9375,
      "loss": 0.0763
    },
    {
      "step": 9380,
      "loss": 0.0775
    },
    {
      "step": 9385,
      "loss": 0.0774
    },
    {
      "step": 9390,
      "loss": 0.0697
    },
    {
      "step": 9395,
      "loss": 0.0714
    },
    {
      "step": 9400,
      "loss": 0.0758
    },
    {
      "step": 9405,
      "loss": 0.0732
    },
    {
      "step": 9410,
      "loss": 0.0781
    },
    {
      "step": 9415,
      "loss": 0.0744
    },
    {
      "step": 9420,
      "loss": 0.0757
    },
    {
      "step": 9425,
      "loss": 0.0747
    },
    {
      "step": 9430,
      "loss": 0.0771
    },
    {
      "step": 9435,
      "loss": 0.0787
    },
    {
      "step": 9440,
      "loss": 0.0726
    },
    {
      "step": 9445,
      "loss": 0.0769
    },
    {
      "step": 9450,
      "loss": 0.0722
    },
    {
      "step": 9455,
      "loss": 0.0731
    },
    {
      "step": 9460,
      "loss": 0.0716
    },
    {
      "step": 9465,
      "loss": 0.0772
    },
    {
      "step": 9470,
      "loss": 0.0759
    },
    {
      "step": 9475,
      "loss": 0.0728
    },
    {
      "step": 9480,
      "loss": 0.0716
    },
    {
      "step": 9485,
      "loss": 0.0759
    },
    {
      "step": 9490,
      "loss": 0.0779
    },
    {
      "step": 9495,
      "loss": 0.0735
    },
    {
      "step": 9500,
      "loss": 0.0721
    },
    {
      "step": 9505,
      "loss": 0.076
    },
    {
      "step": 9510,
      "loss": 0.0752
    },
    {
      "step": 9515,
      "loss": 0.0739
    },
    {
      "step": 9520,
      "loss": 0.0744
    },
    {
      "step": 9525,
      "loss": 0.0725
    },
    {
      "step": 9530,
      "loss": 0.0758
    },
    {
      "step": 9535,
      "loss": 0.0725
    },
    {
      "step": 9540,
      "loss": 0.0744
    },
    {
      "step": 9545,
      "loss": 0.0729
    },
    {
      "step": 9550,
      "loss": 0.0696
    },
    {
      "step": 9555,
      "loss": 0.0752
    },
    {
      "step": 9560,
      "loss": 0.0754
    },
    {
      "step": 9565,
      "loss": 0.0758
    },
    {
      "step": 9570,
      "loss": 0.0761
    },
    {
      "step": 9575,
      "loss": 0.0743
    },
    {
      "step": 9580,
      "loss": 0.0721
    },
    {
      "step": 9585,
      "loss": 0.0734
    },
    {
      "step": 9590,
      "loss": 0.0751
    },
    {
      "step": 9595,
      "loss": 0.0774
    },
    {
      "step": 9600,
      "loss": 0.0765
    },
    {
      "step": 9605,
      "loss": 0.0738
    },
    {
      "step": 9610,
      "loss": 0.0768
    },
    {
      "step": 9615,
      "loss": 0.078
    },
    {
      "step": 9620,
      "loss": 0.0759
    },
    {
      "step": 9625,
      "loss": 0.0772
    },
    {
      "step": 9630,
      "loss": 0.0727
    },
    {
      "step": 9635,
      "loss": 0.0754
    },
    {
      "step": 9640,
      "loss": 0.0769
    },
    {
      "step": 9645,
      "loss": 0.0769
    },
    {
      "step": 9650,
      "loss": 0.0774
    },
    {
      "step": 9655,
      "loss": 0.076
    },
    {
      "step": 9660,
      "loss": 0.0766
    },
    {
      "step": 9665,
      "loss": 0.0683
    },
    {
      "step": 9670,
      "loss": 0.0764
    },
    {
      "step": 9675,
      "loss": 0.078
    },
    {
      "step": 9680,
      "loss": 0.0758
    },
    {
      "step": 9685,
      "loss": 0.0766
    },
    {
      "step": 9690,
      "loss": 0.0723
    },
    {
      "step": 9695,
      "loss": 0.0741
    },
    {
      "step": 9700,
      "loss": 0.0759
    },
    {
      "step": 9705,
      "loss": 0.0786
    },
    {
      "step": 9710,
      "loss": 0.0726
    },
    {
      "step": 9715,
      "loss": 0.0744
    },
    {
      "step": 9720,
      "loss": 0.0744
    },
    {
      "step": 9725,
      "loss": 0.0738
    },
    {
      "step": 9730,
      "loss": 0.0722
    },
    {
      "step": 9735,
      "loss": 0.0798
    },
    {
      "step": 9740,
      "loss": 0.0799
    },
    {
      "step": 9745,
      "loss": 0.0742
    },
    {
      "step": 9750,
      "loss": 0.0695
    },
    {
      "step": 9755,
      "loss": 0.0748
    },
    {
      "step": 9760,
      "loss": 0.076
    },
    {
      "step": 9765,
      "loss": 0.0721
    },
    {
      "step": 9770,
      "loss": 0.0742
    },
    {
      "step": 9775,
      "loss": 0.0721
    },
    {
      "step": 9780,
      "loss": 0.0796
    },
    {
      "step": 9785,
      "loss": 0.0714
    },
    {
      "step": 9790,
      "loss": 0.0756
    },
    {
      "step": 9795,
      "loss": 0.0786
    },
    {
      "step": 9800,
      "loss": 0.0732
    },
    {
      "step": 9805,
      "loss": 0.0741
    },
    {
      "step": 9810,
      "loss": 0.0738
    },
    {
      "step": 9815,
      "loss": 0.074
    },
    {
      "step": 9820,
      "loss": 0.0723
    },
    {
      "step": 9825,
      "loss": 0.0764
    },
    {
      "step": 9830,
      "loss": 0.0714
    },
    {
      "step": 9835,
      "loss": 0.0745
    },
    {
      "step": 9840,
      "loss": 0.0754
    },
    {
      "step": 9845,
      "loss": 0.0791
    },
    {
      "step": 9850,
      "loss": 0.0751
    },
    {
      "step": 9855,
      "loss": 0.0725
    },
    {
      "step": 9860,
      "loss": 0.0736
    },
    {
      "step": 9865,
      "loss": 0.0779
    },
    {
      "step": 9870,
      "loss": 0.0767
    },
    {
      "step": 9875,
      "loss": 0.0792
    },
    {
      "step": 9880,
      "loss": 0.0753
    },
    {
      "step": 9885,
      "loss": 0.0701
    },
    {
      "step": 9890,
      "loss": 0.0695
    },
    {
      "step": 9895,
      "loss": 0.0729
    },
    {
      "step": 9900,
      "loss": 0.0713
    },
    {
      "step": 9905,
      "loss": 0.0711
    },
    {
      "step": 9910,
      "loss": 0.0738
    },
    {
      "step": 9915,
      "loss": 0.0717
    },
    {
      "step": 9920,
      "loss": 0.073
    },
    {
      "step": 9925,
      "loss": 0.0749
    },
    {
      "step": 9930,
      "loss": 0.0782
    },
    {
      "step": 9935,
      "loss": 0.078
    },
    {
      "step": 9940,
      "loss": 0.0715
    },
    {
      "step": 9945,
      "loss": 0.077
    },
    {
      "step": 9950,
      "loss": 0.0766
    },
    {
      "step": 9955,
      "loss": 0.0719
    },
    {
      "step": 9960,
      "loss": 0.071
    },
    {
      "step": 9965,
      "loss": 0.0688
    },
    {
      "step": 9970,
      "loss": 0.0762
    },
    {
      "step": 9975,
      "loss": 0.0713
    },
    {
      "step": 9980,
      "loss": 0.0731
    },
    {
      "step": 9985,
      "loss": 0.0733
    },
    {
      "step": 9990,
      "loss": 0.0713
    },
    {
      "step": 9995,
      "loss": 0.0748
    },
    {
      "step": 10000,
      "loss": 0.0764
    },
    {
      "step": 10005,
      "loss": 0.0746
    },
    {
      "step": 10010,
      "loss": 0.0713
    },
    {
      "step": 10015,
      "loss": 0.0759
    },
    {
      "step": 10020,
      "loss": 0.076
    },
    {
      "step": 10025,
      "loss": 0.0763
    },
    {
      "step": 10030,
      "loss": 0.0754
    },
    {
      "step": 10035,
      "loss": 0.0743
    },
    {
      "step": 10040,
      "loss": 0.0786
    },
    {
      "step": 10045,
      "loss": 0.0714
    },
    {
      "step": 10050,
      "loss": 0.0735
    },
    {
      "step": 10055,
      "loss": 0.0745
    },
    {
      "step": 10060,
      "loss": 0.0748
    },
    {
      "step": 10065,
      "loss": 0.075
    },
    {
      "step": 10070,
      "loss": 0.0729
    },
    {
      "step": 10075,
      "loss": 0.0775
    },
    {
      "step": 10080,
      "loss": 0.0768
    },
    {
      "step": 10085,
      "loss": 0.0748
    },
    {
      "step": 10090,
      "loss": 0.0769
    },
    {
      "step": 10095,
      "loss": 0.0746
    },
    {
      "step": 10100,
      "loss": 0.0681
    },
    {
      "step": 10105,
      "loss": 0.0721
    },
    {
      "step": 10110,
      "loss": 0.0779
    },
    {
      "step": 10115,
      "loss": 0.0754
    },
    {
      "step": 10120,
      "loss": 0.0734
    },
    {
      "step": 10125,
      "loss": 0.0725
    },
    {
      "step": 10130,
      "loss": 0.0736
    },
    {
      "step": 10135,
      "loss": 0.0778
    },
    {
      "step": 10140,
      "loss": 0.0769
    },
    {
      "step": 10145,
      "loss": 0.075
    },
    {
      "step": 10150,
      "loss": 0.0792
    },
    {
      "step": 10155,
      "loss": 0.0716
    },
    {
      "step": 10160,
      "loss": 0.0705
    },
    {
      "step": 10165,
      "loss": 0.074
    },
    {
      "step": 10170,
      "loss": 0.0746
    },
    {
      "step": 10175,
      "loss": 0.0724
    },
    {
      "step": 10180,
      "loss": 0.0756
    },
    {
      "step": 10185,
      "loss": 0.0712
    },
    {
      "step": 10190,
      "loss": 0.0693
    },
    {
      "step": 10195,
      "loss": 0.0749
    },
    {
      "step": 10200,
      "loss": 0.073
    },
    {
      "step": 10205,
      "loss": 0.0746
    },
    {
      "step": 10210,
      "loss": 0.0776
    },
    {
      "step": 10215,
      "loss": 0.0734
    },
    {
      "step": 10220,
      "loss": 0.0706
    },
    {
      "step": 10225,
      "loss": 0.0752
    },
    {
      "step": 10230,
      "loss": 0.0728
    },
    {
      "step": 10235,
      "loss": 0.074
    },
    {
      "step": 10240,
      "loss": 0.0722
    },
    {
      "step": 10245,
      "loss": 0.0714
    },
    {
      "step": 10250,
      "loss": 0.0766
    },
    {
      "step": 10255,
      "loss": 0.0709
    },
    {
      "step": 10260,
      "loss": 0.0757
    },
    {
      "step": 10265,
      "loss": 0.0733
    },
    {
      "step": 10270,
      "loss": 0.071
    },
    {
      "step": 10275,
      "loss": 0.0769
    },
    {
      "step": 10280,
      "loss": 0.0729
    },
    {
      "step": 10285,
      "loss": 0.0725
    },
    {
      "step": 10290,
      "loss": 0.0752
    },
    {
      "step": 10295,
      "loss": 0.0735
    },
    {
      "step": 10300,
      "loss": 0.0743
    },
    {
      "step": 10305,
      "loss": 0.0767
    },
    {
      "step": 10310,
      "loss": 0.0727
    },
    {
      "step": 10315,
      "loss": 0.0754
    },
    {
      "step": 10320,
      "loss": 0.0749
    },
    {
      "step": 10325,
      "loss": 0.0782
    },
    {
      "step": 10330,
      "loss": 0.0737
    },
    {
      "step": 10335,
      "loss": 0.0742
    },
    {
      "step": 10340,
      "loss": 0.0762
    },
    {
      "step": 10345,
      "loss": 0.0743
    },
    {
      "step": 10350,
      "loss": 0.0793
    },
    {
      "step": 10355,
      "loss": 0.0755
    },
    {
      "step": 10360,
      "loss": 0.0739
    },
    {
      "step": 10365,
      "loss": 0.0744
    },
    {
      "step": 10370,
      "loss": 0.0755
    },
    {
      "step": 10375,
      "loss": 0.0749
    },
    {
      "step": 10380,
      "loss": 0.0706
    },
    {
      "step": 10385,
      "loss": 0.0736
    },
    {
      "step": 10390,
      "loss": 0.0708
    },
    {
      "step": 10395,
      "loss": 0.0731
    },
    {
      "step": 10400,
      "loss": 0.0767
    },
    {
      "step": 10405,
      "loss": 0.0717
    },
    {
      "step": 10410,
      "loss": 0.0742
    },
    {
      "step": 10415,
      "loss": 0.075
    },
    {
      "step": 10420,
      "loss": 0.0741
    },
    {
      "step": 10425,
      "loss": 0.074
    },
    {
      "step": 10430,
      "loss": 0.077
    },
    {
      "step": 10435,
      "loss": 0.0772
    },
    {
      "step": 10440,
      "loss": 0.0712
    },
    {
      "step": 10445,
      "loss": 0.076
    },
    {
      "step": 10450,
      "loss": 0.0783
    },
    {
      "step": 10455,
      "loss": 0.0784
    },
    {
      "step": 10460,
      "loss": 0.0765
    },
    {
      "step": 10465,
      "loss": 0.0757
    },
    {
      "step": 10470,
      "loss": 0.079
    },
    {
      "step": 10475,
      "loss": 0.0757
    },
    {
      "step": 10480,
      "loss": 0.0749
    },
    {
      "step": 10485,
      "loss": 0.0748
    },
    {
      "step": 10490,
      "loss": 0.0771
    },
    {
      "step": 10495,
      "loss": 0.076
    },
    {
      "step": 10500,
      "loss": 0.0762
    },
    {
      "step": 10505,
      "loss": 0.0723
    },
    {
      "step": 10510,
      "loss": 0.0699
    },
    {
      "step": 10515,
      "loss": 0.0748
    },
    {
      "step": 10520,
      "loss": 0.0727
    },
    {
      "step": 10525,
      "loss": 0.0715
    },
    {
      "step": 10530,
      "loss": 0.0736
    },
    {
      "step": 10535,
      "loss": 0.0773
    },
    {
      "step": 10540,
      "loss": 0.0699
    },
    {
      "step": 10545,
      "loss": 0.0782
    },
    {
      "step": 10550,
      "loss": 0.0752
    },
    {
      "step": 10555,
      "loss": 0.0761
    },
    {
      "step": 10560,
      "loss": 0.0769
    },
    {
      "step": 10565,
      "loss": 0.0748
    },
    {
      "step": 10570,
      "loss": 0.072
    },
    {
      "step": 10575,
      "loss": 0.0701
    },
    {
      "step": 10580,
      "loss": 0.0684
    },
    {
      "step": 10585,
      "loss": 0.079
    },
    {
      "step": 10590,
      "loss": 0.0755
    },
    {
      "step": 10595,
      "loss": 0.0716
    },
    {
      "step": 10600,
      "loss": 0.0784
    },
    {
      "step": 10605,
      "loss": 0.0673
    },
    {
      "step": 10610,
      "loss": 0.0781
    },
    {
      "step": 10615,
      "loss": 0.0752
    },
    {
      "step": 10620,
      "loss": 0.0721
    },
    {
      "step": 10625,
      "loss": 0.073
    },
    {
      "step": 10630,
      "loss": 0.0758
    },
    {
      "step": 10635,
      "loss": 0.0748
    },
    {
      "step": 10640,
      "loss": 0.0773
    },
    {
      "step": 10645,
      "loss": 0.0735
    },
    {
      "step": 10650,
      "loss": 0.0757
    },
    {
      "step": 10655,
      "loss": 0.0735
    },
    {
      "step": 10660,
      "loss": 0.0787
    },
    {
      "step": 10665,
      "loss": 0.0737
    },
    {
      "step": 10670,
      "loss": 0.0735
    },
    {
      "step": 10675,
      "loss": 0.0721
    },
    {
      "step": 10680,
      "loss": 0.0707
    },
    {
      "step": 10685,
      "loss": 0.0748
    },
    {
      "step": 10690,
      "loss": 0.0753
    },
    {
      "step": 10695,
      "loss": 0.0717
    },
    {
      "step": 10700,
      "loss": 0.0721
    },
    {
      "step": 10705,
      "loss": 0.0704
    },
    {
      "step": 10710,
      "loss": 0.0745
    },
    {
      "step": 10715,
      "loss": 0.0724
    },
    {
      "step": 10720,
      "loss": 0.0759
    },
    {
      "step": 10725,
      "loss": 0.0753
    },
    {
      "step": 10730,
      "loss": 0.0734
    },
    {
      "step": 10735,
      "loss": 0.0777
    },
    {
      "step": 10740,
      "loss": 0.0728
    },
    {
      "step": 10745,
      "loss": 0.0757
    },
    {
      "step": 10750,
      "loss": 0.0782
    },
    {
      "step": 10755,
      "loss": 0.0757
    },
    {
      "step": 10760,
      "loss": 0.0757
    },
    {
      "step": 10765,
      "loss": 0.0759
    },
    {
      "step": 10770,
      "loss": 0.0747
    },
    {
      "step": 10775,
      "loss": 0.0745
    },
    {
      "step": 10780,
      "loss": 0.0774
    },
    {
      "step": 10785,
      "loss": 0.0718
    },
    {
      "step": 10790,
      "loss": 0.0743
    },
    {
      "step": 10795,
      "loss": 0.0687
    },
    {
      "step": 10800,
      "loss": 0.0766
    },
    {
      "step": 10805,
      "loss": 0.0756
    },
    {
      "step": 10810,
      "loss": 0.0741
    },
    {
      "step": 10815,
      "loss": 0.0748
    },
    {
      "step": 10820,
      "loss": 0.0776
    },
    {
      "step": 10825,
      "loss": 0.0719
    },
    {
      "step": 10830,
      "loss": 0.0772
    },
    {
      "step": 10835,
      "loss": 0.0759
    },
    {
      "step": 10840,
      "loss": 0.0711
    },
    {
      "step": 10845,
      "loss": 0.0744
    },
    {
      "step": 10850,
      "loss": 0.0758
    },
    {
      "step": 10855,
      "loss": 0.0785
    },
    {
      "step": 10860,
      "loss": 0.0762
    },
    {
      "step": 10865,
      "loss": 0.0762
    },
    {
      "step": 10870,
      "loss": 0.0754
    },
    {
      "step": 10875,
      "loss": 0.0752
    },
    {
      "step": 10880,
      "loss": 0.0751
    },
    {
      "step": 10885,
      "loss": 0.0765
    },
    {
      "step": 10890,
      "loss": 0.0746
    },
    {
      "step": 10895,
      "loss": 0.0768
    },
    {
      "step": 10900,
      "loss": 0.0713
    },
    {
      "step": 10905,
      "loss": 0.0721
    },
    {
      "step": 10910,
      "loss": 0.0686
    },
    {
      "step": 10915,
      "loss": 0.0779
    },
    {
      "step": 10920,
      "loss": 0.077
    },
    {
      "step": 10925,
      "loss": 0.0714
    },
    {
      "step": 10930,
      "loss": 0.0759
    },
    {
      "step": 10935,
      "loss": 0.0721
    },
    {
      "step": 10940,
      "loss": 0.0782
    },
    {
      "step": 10945,
      "loss": 0.0773
    },
    {
      "step": 10950,
      "loss": 0.0746
    },
    {
      "step": 10955,
      "loss": 0.0729
    },
    {
      "step": 10960,
      "loss": 0.0798
    },
    {
      "step": 10965,
      "loss": 0.078
    },
    {
      "step": 10970,
      "loss": 0.0786
    },
    {
      "step": 10975,
      "loss": 0.0729
    },
    {
      "step": 10980,
      "loss": 0.0743
    },
    {
      "step": 10985,
      "loss": 0.0763
    },
    {
      "step": 10990,
      "loss": 0.0769
    }
  ],
  "eval_losses": [
    {
      "step": 5496,
      "eval_loss": 0.1244
    },
    {
      "step": 10992,
      "eval_loss": 0.137
    }
  ],
  "decode_history": [
    {
      "epoch": 1.0,
      "step": 5496,
      "eval_loss": 0.12439540773630142,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 32,
        "mae_log": 0.45023086600200213,
        "rmse_log": 0.5992315639390868,
        "mae_s": 3.2518312499999995,
        "rmse_s": 5.237985204136224,
        "spearman_rho": 0.19825508855022245,
        "mean_actual_s": 6.12451875,
        "mean_pred_s": 5.2406250000000005
      },
      "samples": [
        {
          "output": "dwell_seconds: 3.5",
          "gold_s": 3.621,
          "pred_s": 3.5
        },
        {
          "output": "dwell_seconds: 4.1",
          "gold_s": 2.878,
          "pred_s": 4.1
        },
        {
          "output": "dwell_seconds: 4.6",
          "gold_s": 3.851,
          "pred_s": 4.6
        }
      ]
    },
    {
      "epoch": 2.0,
      "step": 10992,
      "eval_loss": 0.13700838387012482,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 32,
        "mae_log": 0.5175066550280347,
        "rmse_log": 0.6379020530915637,
        "mae_s": 3.74339375,
        "rmse_s": 5.468910220395833,
        "spearman_rho": 0.16071928189928747,
        "mean_actual_s": 6.12451875,
        "mean_pred_s": 5.637499999999999
      },
      "samples": [
        {
          "output": "dwell_seconds: 2.6",
          "gold_s": 3.621,
          "pred_s": 2.6
        },
        {
          "output": "dwell_seconds: 4.1",
          "gold_s": 2.878,
          "pred_s": 4.1
        },
        {
          "output": "dwell_seconds: 4.3",
          "gold_s": 3.851,
          "pred_s": 4.3
        }
      ]
    }
  ],
  "vram": {
    "device": "NVIDIA A100-SXM4-40GB",
    "total_gb": 39.49,
    "peak_allocated_gb": 6.29,
    "peak_reserved_gb": 6.49
  },
  "effective_config": {
    "train": {
      "learning_rate": 0.0002,
      "num_epochs": 2,
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
      "run_name": "qwen3vl4b-qlora-pathA"
    },
    "model": {
      "checkpoint": "unsloth/Qwen3-VL-4B-Instruct",
      "load_in_4bit": true,
      "gradient_checkpointing": true,
      "max_seq_length": 1536,
      "finetune_vision_layers": false,
      "lora_r": 16,
      "lora_alpha": 16,
      "lora_dropout": 0.0
    },
    "image": {
      "max_side": 1024,
      "min_pixels": 100352,
      "max_pixels": 602112
    },
    "data": {
      "include_task_title": false
    },
    "dry_run": false,
    "lupi": {
      "lambda": 0.5,
      "n_rows": 87921,
      "n_blended": 87921,
      "coverage": 1.0,
      "mean_abs_shift_s": 2.192
    }
  }
}
```
