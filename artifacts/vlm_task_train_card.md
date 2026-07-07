# VLM train card — Qwen3-VL-4B QLoRA SFT (Path A)

_Generated 2026-07-07T02:29:02+00:00 · config `configs/vlm_task.yaml` · seed 42 · mode **FULL**_

- Model: `unsloth/Qwen3-VL-4B-Instruct` · 4-bit QLoRA · vision frozen · LoRA r=16 α=16 on language attn+MLP
- Target: `dwell_seconds: X.X` (winsor cap 24.954 s, train-split p95 — see artifacts/dataset_card.md)
- Task title in prompt: **True** (False = image+features, True = image+features+task)
- Features (train-time only): targets blended toward the privileged-feature teacher (λ=0.5) — **87921/87921** rows covered (100.0%), mean |shift| 2.192 s; val targets untouched, inference stays screenshot-only
- Examples: train **87921** · val **4269** (img_resolved only)

## Memory footprint

- per-device batch **4** × grad-accum **4** = effective **16**
- image bounds: max_side 1024, pixels [100352, 602112] (≤768 visual tokens) · max_seq_length 1536
- peak VRAM: **6.4 GB** allocated / 6.59 GB reserved of 39.49 GB (NVIDIA A100-SXM4-40GB)

## Loss

- train loss: first 4.5111 → last 0.1346

| eval pass | epoch | step | val loss | parse rate | MAE (s) | MAE (log) |
|---|---|---|---|---|---|---|
| 0 | 1.0 | 5496 | 0.8223534822463989 | 100.00% | 3.59 | 0.504 |
| 1 | 2.0 | 10992 | 0.9746091365814209 | 100.00% | 3.74 | 0.526 |

## Sample decodes (last eval pass)

- gold 3.6 s → `dwell_seconds: 3.0`
- gold 2.9 s → `dwell_seconds: 4.1`
- gold 3.9 s → `dwell_seconds: 3.2`

Checkpoints: `artifacts/vlm_task_ckpt` (per epoch; final adapters in `artifacts/vlm_task_ckpt/final`).

## Full log (JSON)

```json
{
  "train_losses": [
    {
      "step": 5,
      "loss": 4.5111
    },
    {
      "step": 10,
      "loss": 4.3691
    },
    {
      "step": 15,
      "loss": 4.248
    },
    {
      "step": 20,
      "loss": 3.9865
    },
    {
      "step": 25,
      "loss": 3.6792
    },
    {
      "step": 30,
      "loss": 3.3695
    },
    {
      "step": 35,
      "loss": 3.0783
    },
    {
      "step": 40,
      "loss": 2.7063
    },
    {
      "step": 45,
      "loss": 2.3173
    },
    {
      "step": 50,
      "loss": 1.8101
    },
    {
      "step": 55,
      "loss": 1.4003
    },
    {
      "step": 60,
      "loss": 1.0152
    },
    {
      "step": 65,
      "loss": 0.8609
    },
    {
      "step": 70,
      "loss": 0.8359
    },
    {
      "step": 75,
      "loss": 0.7301
    },
    {
      "step": 80,
      "loss": 0.7289
    },
    {
      "step": 85,
      "loss": 0.7429
    },
    {
      "step": 90,
      "loss": 0.7
    },
    {
      "step": 95,
      "loss": 0.6987
    },
    {
      "step": 100,
      "loss": 0.6664
    },
    {
      "step": 105,
      "loss": 0.6758
    },
    {
      "step": 110,
      "loss": 0.6413
    },
    {
      "step": 115,
      "loss": 0.6512
    },
    {
      "step": 120,
      "loss": 0.6589
    },
    {
      "step": 125,
      "loss": 0.6679
    },
    {
      "step": 130,
      "loss": 0.6478
    },
    {
      "step": 135,
      "loss": 0.6885
    },
    {
      "step": 140,
      "loss": 0.6325
    },
    {
      "step": 145,
      "loss": 0.6201
    },
    {
      "step": 150,
      "loss": 0.6166
    },
    {
      "step": 155,
      "loss": 0.606
    },
    {
      "step": 160,
      "loss": 0.6097
    },
    {
      "step": 165,
      "loss": 0.6209
    },
    {
      "step": 170,
      "loss": 0.6455
    },
    {
      "step": 175,
      "loss": 0.5716
    },
    {
      "step": 180,
      "loss": 0.6065
    },
    {
      "step": 185,
      "loss": 0.5976
    },
    {
      "step": 190,
      "loss": 0.6092
    },
    {
      "step": 195,
      "loss": 0.6304
    },
    {
      "step": 200,
      "loss": 0.6058
    },
    {
      "step": 205,
      "loss": 0.6294
    },
    {
      "step": 210,
      "loss": 0.6088
    },
    {
      "step": 215,
      "loss": 0.6182
    },
    {
      "step": 220,
      "loss": 0.5786
    },
    {
      "step": 225,
      "loss": 0.5903
    },
    {
      "step": 230,
      "loss": 0.5691
    },
    {
      "step": 235,
      "loss": 0.579
    },
    {
      "step": 240,
      "loss": 0.589
    },
    {
      "step": 245,
      "loss": 0.5808
    },
    {
      "step": 250,
      "loss": 0.5653
    },
    {
      "step": 255,
      "loss": 0.5862
    },
    {
      "step": 260,
      "loss": 0.5646
    },
    {
      "step": 265,
      "loss": 0.6282
    },
    {
      "step": 270,
      "loss": 0.635
    },
    {
      "step": 275,
      "loss": 0.5535
    },
    {
      "step": 280,
      "loss": 0.6331
    },
    {
      "step": 285,
      "loss": 0.5679
    },
    {
      "step": 290,
      "loss": 0.5611
    },
    {
      "step": 295,
      "loss": 0.5724
    },
    {
      "step": 300,
      "loss": 0.5307
    },
    {
      "step": 305,
      "loss": 0.558
    },
    {
      "step": 310,
      "loss": 0.5426
    },
    {
      "step": 315,
      "loss": 0.6064
    },
    {
      "step": 320,
      "loss": 0.5561
    },
    {
      "step": 325,
      "loss": 0.5614
    },
    {
      "step": 330,
      "loss": 0.5555
    },
    {
      "step": 335,
      "loss": 0.5488
    },
    {
      "step": 340,
      "loss": 0.5602
    },
    {
      "step": 345,
      "loss": 0.5557
    },
    {
      "step": 350,
      "loss": 0.5501
    },
    {
      "step": 355,
      "loss": 0.5798
    },
    {
      "step": 360,
      "loss": 0.5777
    },
    {
      "step": 365,
      "loss": 0.5464
    },
    {
      "step": 370,
      "loss": 0.5545
    },
    {
      "step": 375,
      "loss": 0.5304
    },
    {
      "step": 380,
      "loss": 0.5763
    },
    {
      "step": 385,
      "loss": 0.5608
    },
    {
      "step": 390,
      "loss": 0.5541
    },
    {
      "step": 395,
      "loss": 0.5411
    },
    {
      "step": 400,
      "loss": 0.559
    },
    {
      "step": 405,
      "loss": 0.5414
    },
    {
      "step": 410,
      "loss": 0.5676
    },
    {
      "step": 415,
      "loss": 0.5684
    },
    {
      "step": 420,
      "loss": 0.535
    },
    {
      "step": 425,
      "loss": 0.561
    },
    {
      "step": 430,
      "loss": 0.5329
    },
    {
      "step": 435,
      "loss": 0.5477
    },
    {
      "step": 440,
      "loss": 0.5528
    },
    {
      "step": 445,
      "loss": 0.5841
    },
    {
      "step": 450,
      "loss": 0.5131
    },
    {
      "step": 455,
      "loss": 0.5528
    },
    {
      "step": 460,
      "loss": 0.5423
    },
    {
      "step": 465,
      "loss": 0.5253
    },
    {
      "step": 470,
      "loss": 0.5713
    },
    {
      "step": 475,
      "loss": 0.525
    },
    {
      "step": 480,
      "loss": 0.5421
    },
    {
      "step": 485,
      "loss": 0.5323
    },
    {
      "step": 490,
      "loss": 0.5322
    },
    {
      "step": 495,
      "loss": 0.506
    },
    {
      "step": 500,
      "loss": 0.5568
    },
    {
      "step": 505,
      "loss": 0.5354
    },
    {
      "step": 510,
      "loss": 0.5122
    },
    {
      "step": 515,
      "loss": 0.5001
    },
    {
      "step": 520,
      "loss": 0.5384
    },
    {
      "step": 525,
      "loss": 0.5287
    },
    {
      "step": 530,
      "loss": 0.5419
    },
    {
      "step": 535,
      "loss": 0.5033
    },
    {
      "step": 540,
      "loss": 0.5267
    },
    {
      "step": 545,
      "loss": 0.5706
    },
    {
      "step": 550,
      "loss": 0.5166
    },
    {
      "step": 555,
      "loss": 0.5403
    },
    {
      "step": 560,
      "loss": 0.5052
    },
    {
      "step": 565,
      "loss": 0.5283
    },
    {
      "step": 570,
      "loss": 0.4934
    },
    {
      "step": 575,
      "loss": 0.5288
    },
    {
      "step": 580,
      "loss": 0.5361
    },
    {
      "step": 585,
      "loss": 0.5065
    },
    {
      "step": 590,
      "loss": 0.5057
    },
    {
      "step": 595,
      "loss": 0.5242
    },
    {
      "step": 600,
      "loss": 0.5378
    },
    {
      "step": 605,
      "loss": 0.5194
    },
    {
      "step": 610,
      "loss": 0.5457
    },
    {
      "step": 615,
      "loss": 0.5305
    },
    {
      "step": 620,
      "loss": 0.5014
    },
    {
      "step": 625,
      "loss": 0.5068
    },
    {
      "step": 630,
      "loss": 0.5045
    },
    {
      "step": 635,
      "loss": 0.5113
    },
    {
      "step": 640,
      "loss": 0.5293
    },
    {
      "step": 645,
      "loss": 0.4987
    },
    {
      "step": 650,
      "loss": 0.5077
    },
    {
      "step": 655,
      "loss": 0.5102
    },
    {
      "step": 660,
      "loss": 0.5073
    },
    {
      "step": 665,
      "loss": 0.5354
    },
    {
      "step": 670,
      "loss": 0.5315
    },
    {
      "step": 675,
      "loss": 0.5236
    },
    {
      "step": 680,
      "loss": 0.5136
    },
    {
      "step": 685,
      "loss": 0.5036
    },
    {
      "step": 690,
      "loss": 0.484
    },
    {
      "step": 695,
      "loss": 0.4687
    },
    {
      "step": 700,
      "loss": 0.5009
    },
    {
      "step": 705,
      "loss": 0.5096
    },
    {
      "step": 710,
      "loss": 0.4954
    },
    {
      "step": 715,
      "loss": 0.5404
    },
    {
      "step": 720,
      "loss": 0.4927
    },
    {
      "step": 725,
      "loss": 0.5316
    },
    {
      "step": 730,
      "loss": 0.4812
    },
    {
      "step": 735,
      "loss": 0.5091
    },
    {
      "step": 740,
      "loss": 0.4754
    },
    {
      "step": 745,
      "loss": 0.488
    },
    {
      "step": 750,
      "loss": 0.5252
    },
    {
      "step": 755,
      "loss": 0.4818
    },
    {
      "step": 760,
      "loss": 0.4876
    },
    {
      "step": 765,
      "loss": 0.5234
    },
    {
      "step": 770,
      "loss": 0.4807
    },
    {
      "step": 775,
      "loss": 0.5169
    },
    {
      "step": 780,
      "loss": 0.4666
    },
    {
      "step": 785,
      "loss": 0.5059
    },
    {
      "step": 790,
      "loss": 0.4767
    },
    {
      "step": 795,
      "loss": 0.4985
    },
    {
      "step": 800,
      "loss": 0.4483
    },
    {
      "step": 805,
      "loss": 0.4898
    },
    {
      "step": 810,
      "loss": 0.5041
    },
    {
      "step": 815,
      "loss": 0.4995
    },
    {
      "step": 820,
      "loss": 0.4975
    },
    {
      "step": 825,
      "loss": 0.5075
    },
    {
      "step": 830,
      "loss": 0.4814
    },
    {
      "step": 835,
      "loss": 0.4852
    },
    {
      "step": 840,
      "loss": 0.4758
    },
    {
      "step": 845,
      "loss": 0.4833
    },
    {
      "step": 850,
      "loss": 0.4588
    },
    {
      "step": 855,
      "loss": 0.4814
    },
    {
      "step": 860,
      "loss": 0.4688
    },
    {
      "step": 865,
      "loss": 0.5191
    },
    {
      "step": 870,
      "loss": 0.4459
    },
    {
      "step": 875,
      "loss": 0.502
    },
    {
      "step": 880,
      "loss": 0.4782
    },
    {
      "step": 885,
      "loss": 0.4727
    },
    {
      "step": 890,
      "loss": 0.5161
    },
    {
      "step": 895,
      "loss": 0.462
    },
    {
      "step": 900,
      "loss": 0.5014
    },
    {
      "step": 905,
      "loss": 0.5319
    },
    {
      "step": 910,
      "loss": 0.4861
    },
    {
      "step": 915,
      "loss": 0.4787
    },
    {
      "step": 920,
      "loss": 0.4511
    },
    {
      "step": 925,
      "loss": 0.4783
    },
    {
      "step": 930,
      "loss": 0.4488
    },
    {
      "step": 935,
      "loss": 0.4821
    },
    {
      "step": 940,
      "loss": 0.4894
    },
    {
      "step": 945,
      "loss": 0.4781
    },
    {
      "step": 950,
      "loss": 0.4711
    },
    {
      "step": 955,
      "loss": 0.4778
    },
    {
      "step": 960,
      "loss": 0.4894
    },
    {
      "step": 965,
      "loss": 0.467
    },
    {
      "step": 970,
      "loss": 0.5099
    },
    {
      "step": 975,
      "loss": 0.4502
    },
    {
      "step": 980,
      "loss": 0.4627
    },
    {
      "step": 985,
      "loss": 0.4724
    },
    {
      "step": 990,
      "loss": 0.4821
    },
    {
      "step": 995,
      "loss": 0.4755
    },
    {
      "step": 1000,
      "loss": 0.4603
    },
    {
      "step": 1005,
      "loss": 0.4752
    },
    {
      "step": 1010,
      "loss": 0.4755
    },
    {
      "step": 1015,
      "loss": 0.46
    },
    {
      "step": 1020,
      "loss": 0.439
    },
    {
      "step": 1025,
      "loss": 0.464
    },
    {
      "step": 1030,
      "loss": 0.4571
    },
    {
      "step": 1035,
      "loss": 0.4547
    },
    {
      "step": 1040,
      "loss": 0.4844
    },
    {
      "step": 1045,
      "loss": 0.4772
    },
    {
      "step": 1050,
      "loss": 0.4509
    },
    {
      "step": 1055,
      "loss": 0.4807
    },
    {
      "step": 1060,
      "loss": 0.4768
    },
    {
      "step": 1065,
      "loss": 0.4786
    },
    {
      "step": 1070,
      "loss": 0.4485
    },
    {
      "step": 1075,
      "loss": 0.427
    },
    {
      "step": 1080,
      "loss": 0.4612
    },
    {
      "step": 1085,
      "loss": 0.4463
    },
    {
      "step": 1090,
      "loss": 0.4982
    },
    {
      "step": 1095,
      "loss": 0.4757
    },
    {
      "step": 1100,
      "loss": 0.4864
    },
    {
      "step": 1105,
      "loss": 0.4623
    },
    {
      "step": 1110,
      "loss": 0.4548
    },
    {
      "step": 1115,
      "loss": 0.4717
    },
    {
      "step": 1120,
      "loss": 0.4491
    },
    {
      "step": 1125,
      "loss": 0.4716
    },
    {
      "step": 1130,
      "loss": 0.4376
    },
    {
      "step": 1135,
      "loss": 0.4592
    },
    {
      "step": 1140,
      "loss": 0.4457
    },
    {
      "step": 1145,
      "loss": 0.4733
    },
    {
      "step": 1150,
      "loss": 0.4692
    },
    {
      "step": 1155,
      "loss": 0.4481
    },
    {
      "step": 1160,
      "loss": 0.4917
    },
    {
      "step": 1165,
      "loss": 0.4657
    },
    {
      "step": 1170,
      "loss": 0.4585
    },
    {
      "step": 1175,
      "loss": 0.4102
    },
    {
      "step": 1180,
      "loss": 0.437
    },
    {
      "step": 1185,
      "loss": 0.4454
    },
    {
      "step": 1190,
      "loss": 0.4349
    },
    {
      "step": 1195,
      "loss": 0.4725
    },
    {
      "step": 1200,
      "loss": 0.4589
    },
    {
      "step": 1205,
      "loss": 0.4662
    },
    {
      "step": 1210,
      "loss": 0.4572
    },
    {
      "step": 1215,
      "loss": 0.4332
    },
    {
      "step": 1220,
      "loss": 0.4287
    },
    {
      "step": 1225,
      "loss": 0.4502
    },
    {
      "step": 1230,
      "loss": 0.4637
    },
    {
      "step": 1235,
      "loss": 0.4438
    },
    {
      "step": 1240,
      "loss": 0.447
    },
    {
      "step": 1245,
      "loss": 0.4767
    },
    {
      "step": 1250,
      "loss": 0.4445
    },
    {
      "step": 1255,
      "loss": 0.4372
    },
    {
      "step": 1260,
      "loss": 0.4678
    },
    {
      "step": 1265,
      "loss": 0.4925
    },
    {
      "step": 1270,
      "loss": 0.4747
    },
    {
      "step": 1275,
      "loss": 0.4572
    },
    {
      "step": 1280,
      "loss": 0.4317
    },
    {
      "step": 1285,
      "loss": 0.4573
    },
    {
      "step": 1290,
      "loss": 0.453
    },
    {
      "step": 1295,
      "loss": 0.4491
    },
    {
      "step": 1300,
      "loss": 0.4683
    },
    {
      "step": 1305,
      "loss": 0.4415
    },
    {
      "step": 1310,
      "loss": 0.4437
    },
    {
      "step": 1315,
      "loss": 0.4412
    },
    {
      "step": 1320,
      "loss": 0.4354
    },
    {
      "step": 1325,
      "loss": 0.4299
    },
    {
      "step": 1330,
      "loss": 0.4501
    },
    {
      "step": 1335,
      "loss": 0.445
    },
    {
      "step": 1340,
      "loss": 0.4196
    },
    {
      "step": 1345,
      "loss": 0.4405
    },
    {
      "step": 1350,
      "loss": 0.4675
    },
    {
      "step": 1355,
      "loss": 0.437
    },
    {
      "step": 1360,
      "loss": 0.4061
    },
    {
      "step": 1365,
      "loss": 0.4188
    },
    {
      "step": 1370,
      "loss": 0.4351
    },
    {
      "step": 1375,
      "loss": 0.4328
    },
    {
      "step": 1380,
      "loss": 0.4149
    },
    {
      "step": 1385,
      "loss": 0.417
    },
    {
      "step": 1390,
      "loss": 0.4525
    },
    {
      "step": 1395,
      "loss": 0.4191
    },
    {
      "step": 1400,
      "loss": 0.4371
    },
    {
      "step": 1405,
      "loss": 0.4221
    },
    {
      "step": 1410,
      "loss": 0.4448
    },
    {
      "step": 1415,
      "loss": 0.4341
    },
    {
      "step": 1420,
      "loss": 0.4345
    },
    {
      "step": 1425,
      "loss": 0.416
    },
    {
      "step": 1430,
      "loss": 0.4237
    },
    {
      "step": 1435,
      "loss": 0.3979
    },
    {
      "step": 1440,
      "loss": 0.445
    },
    {
      "step": 1445,
      "loss": 0.4287
    },
    {
      "step": 1450,
      "loss": 0.4315
    },
    {
      "step": 1455,
      "loss": 0.4405
    },
    {
      "step": 1460,
      "loss": 0.4425
    },
    {
      "step": 1465,
      "loss": 0.4373
    },
    {
      "step": 1470,
      "loss": 0.4333
    },
    {
      "step": 1475,
      "loss": 0.39
    },
    {
      "step": 1480,
      "loss": 0.4468
    },
    {
      "step": 1485,
      "loss": 0.4361
    },
    {
      "step": 1490,
      "loss": 0.4259
    },
    {
      "step": 1495,
      "loss": 0.4198
    },
    {
      "step": 1500,
      "loss": 0.4082
    },
    {
      "step": 1505,
      "loss": 0.4238
    },
    {
      "step": 1510,
      "loss": 0.4334
    },
    {
      "step": 1515,
      "loss": 0.4446
    },
    {
      "step": 1520,
      "loss": 0.4446
    },
    {
      "step": 1525,
      "loss": 0.426
    },
    {
      "step": 1530,
      "loss": 0.4336
    },
    {
      "step": 1535,
      "loss": 0.4355
    },
    {
      "step": 1540,
      "loss": 0.4392
    },
    {
      "step": 1545,
      "loss": 0.4595
    },
    {
      "step": 1550,
      "loss": 0.4521
    },
    {
      "step": 1555,
      "loss": 0.4318
    },
    {
      "step": 1560,
      "loss": 0.3978
    },
    {
      "step": 1565,
      "loss": 0.4204
    },
    {
      "step": 1570,
      "loss": 0.4296
    },
    {
      "step": 1575,
      "loss": 0.4115
    },
    {
      "step": 1580,
      "loss": 0.4219
    },
    {
      "step": 1585,
      "loss": 0.4291
    },
    {
      "step": 1590,
      "loss": 0.3846
    },
    {
      "step": 1595,
      "loss": 0.4042
    },
    {
      "step": 1600,
      "loss": 0.4082
    },
    {
      "step": 1605,
      "loss": 0.3834
    },
    {
      "step": 1610,
      "loss": 0.4258
    },
    {
      "step": 1615,
      "loss": 0.4181
    },
    {
      "step": 1620,
      "loss": 0.4286
    },
    {
      "step": 1625,
      "loss": 0.4221
    },
    {
      "step": 1630,
      "loss": 0.445
    },
    {
      "step": 1635,
      "loss": 0.4009
    },
    {
      "step": 1640,
      "loss": 0.4122
    },
    {
      "step": 1645,
      "loss": 0.4194
    },
    {
      "step": 1650,
      "loss": 0.4132
    },
    {
      "step": 1655,
      "loss": 0.4174
    },
    {
      "step": 1660,
      "loss": 0.4296
    },
    {
      "step": 1665,
      "loss": 0.4285
    },
    {
      "step": 1670,
      "loss": 0.4067
    },
    {
      "step": 1675,
      "loss": 0.4221
    },
    {
      "step": 1680,
      "loss": 0.4117
    },
    {
      "step": 1685,
      "loss": 0.421
    },
    {
      "step": 1690,
      "loss": 0.4309
    },
    {
      "step": 1695,
      "loss": 0.3974
    },
    {
      "step": 1700,
      "loss": 0.4359
    },
    {
      "step": 1705,
      "loss": 0.4046
    },
    {
      "step": 1710,
      "loss": 0.4293
    },
    {
      "step": 1715,
      "loss": 0.3826
    },
    {
      "step": 1720,
      "loss": 0.4161
    },
    {
      "step": 1725,
      "loss": 0.4332
    },
    {
      "step": 1730,
      "loss": 0.4288
    },
    {
      "step": 1735,
      "loss": 0.3944
    },
    {
      "step": 1740,
      "loss": 0.3871
    },
    {
      "step": 1745,
      "loss": 0.4333
    },
    {
      "step": 1750,
      "loss": 0.4181
    },
    {
      "step": 1755,
      "loss": 0.418
    },
    {
      "step": 1760,
      "loss": 0.4403
    },
    {
      "step": 1765,
      "loss": 0.4323
    },
    {
      "step": 1770,
      "loss": 0.4183
    },
    {
      "step": 1775,
      "loss": 0.4442
    },
    {
      "step": 1780,
      "loss": 0.4072
    },
    {
      "step": 1785,
      "loss": 0.409
    },
    {
      "step": 1790,
      "loss": 0.4287
    },
    {
      "step": 1795,
      "loss": 0.4018
    },
    {
      "step": 1800,
      "loss": 0.4228
    },
    {
      "step": 1805,
      "loss": 0.4102
    },
    {
      "step": 1810,
      "loss": 0.3979
    },
    {
      "step": 1815,
      "loss": 0.4074
    },
    {
      "step": 1820,
      "loss": 0.3804
    },
    {
      "step": 1825,
      "loss": 0.4039
    },
    {
      "step": 1830,
      "loss": 0.4009
    },
    {
      "step": 1835,
      "loss": 0.3718
    },
    {
      "step": 1840,
      "loss": 0.4026
    },
    {
      "step": 1845,
      "loss": 0.4111
    },
    {
      "step": 1850,
      "loss": 0.4103
    },
    {
      "step": 1855,
      "loss": 0.4101
    },
    {
      "step": 1860,
      "loss": 0.4343
    },
    {
      "step": 1865,
      "loss": 0.4358
    },
    {
      "step": 1870,
      "loss": 0.3824
    },
    {
      "step": 1875,
      "loss": 0.4056
    },
    {
      "step": 1880,
      "loss": 0.3843
    },
    {
      "step": 1885,
      "loss": 0.4059
    },
    {
      "step": 1890,
      "loss": 0.4248
    },
    {
      "step": 1895,
      "loss": 0.3891
    },
    {
      "step": 1900,
      "loss": 0.3782
    },
    {
      "step": 1905,
      "loss": 0.4034
    },
    {
      "step": 1910,
      "loss": 0.408
    },
    {
      "step": 1915,
      "loss": 0.4068
    },
    {
      "step": 1920,
      "loss": 0.3493
    },
    {
      "step": 1925,
      "loss": 0.3999
    },
    {
      "step": 1930,
      "loss": 0.3907
    },
    {
      "step": 1935,
      "loss": 0.3971
    },
    {
      "step": 1940,
      "loss": 0.376
    },
    {
      "step": 1945,
      "loss": 0.4075
    },
    {
      "step": 1950,
      "loss": 0.3916
    },
    {
      "step": 1955,
      "loss": 0.3988
    },
    {
      "step": 1960,
      "loss": 0.3884
    },
    {
      "step": 1965,
      "loss": 0.3913
    },
    {
      "step": 1970,
      "loss": 0.4101
    },
    {
      "step": 1975,
      "loss": 0.3895
    },
    {
      "step": 1980,
      "loss": 0.3779
    },
    {
      "step": 1985,
      "loss": 0.3836
    },
    {
      "step": 1990,
      "loss": 0.3595
    },
    {
      "step": 1995,
      "loss": 0.3782
    },
    {
      "step": 2000,
      "loss": 0.416
    },
    {
      "step": 2005,
      "loss": 0.3878
    },
    {
      "step": 2010,
      "loss": 0.3863
    },
    {
      "step": 2015,
      "loss": 0.3466
    },
    {
      "step": 2020,
      "loss": 0.4043
    },
    {
      "step": 2025,
      "loss": 0.3976
    },
    {
      "step": 2030,
      "loss": 0.3975
    },
    {
      "step": 2035,
      "loss": 0.3597
    },
    {
      "step": 2040,
      "loss": 0.3523
    },
    {
      "step": 2045,
      "loss": 0.3996
    },
    {
      "step": 2050,
      "loss": 0.3749
    },
    {
      "step": 2055,
      "loss": 0.3821
    },
    {
      "step": 2060,
      "loss": 0.3968
    },
    {
      "step": 2065,
      "loss": 0.3695
    },
    {
      "step": 2070,
      "loss": 0.3694
    },
    {
      "step": 2075,
      "loss": 0.3841
    },
    {
      "step": 2080,
      "loss": 0.4365
    },
    {
      "step": 2085,
      "loss": 0.3554
    },
    {
      "step": 2090,
      "loss": 0.3515
    },
    {
      "step": 2095,
      "loss": 0.355
    },
    {
      "step": 2100,
      "loss": 0.411
    },
    {
      "step": 2105,
      "loss": 0.3858
    },
    {
      "step": 2110,
      "loss": 0.3782
    },
    {
      "step": 2115,
      "loss": 0.3774
    },
    {
      "step": 2120,
      "loss": 0.4002
    },
    {
      "step": 2125,
      "loss": 0.4025
    },
    {
      "step": 2130,
      "loss": 0.3555
    },
    {
      "step": 2135,
      "loss": 0.3576
    },
    {
      "step": 2140,
      "loss": 0.3964
    },
    {
      "step": 2145,
      "loss": 0.3768
    },
    {
      "step": 2150,
      "loss": 0.3713
    },
    {
      "step": 2155,
      "loss": 0.3514
    },
    {
      "step": 2160,
      "loss": 0.363
    },
    {
      "step": 2165,
      "loss": 0.3808
    },
    {
      "step": 2170,
      "loss": 0.3598
    },
    {
      "step": 2175,
      "loss": 0.3837
    },
    {
      "step": 2180,
      "loss": 0.3859
    },
    {
      "step": 2185,
      "loss": 0.3861
    },
    {
      "step": 2190,
      "loss": 0.4148
    },
    {
      "step": 2195,
      "loss": 0.3917
    },
    {
      "step": 2200,
      "loss": 0.3643
    },
    {
      "step": 2205,
      "loss": 0.3793
    },
    {
      "step": 2210,
      "loss": 0.365
    },
    {
      "step": 2215,
      "loss": 0.3953
    },
    {
      "step": 2220,
      "loss": 0.4097
    },
    {
      "step": 2225,
      "loss": 0.41
    },
    {
      "step": 2230,
      "loss": 0.3667
    },
    {
      "step": 2235,
      "loss": 0.3893
    },
    {
      "step": 2240,
      "loss": 0.3658
    },
    {
      "step": 2245,
      "loss": 0.3555
    },
    {
      "step": 2250,
      "loss": 0.3676
    },
    {
      "step": 2255,
      "loss": 0.3832
    },
    {
      "step": 2260,
      "loss": 0.3813
    },
    {
      "step": 2265,
      "loss": 0.4003
    },
    {
      "step": 2270,
      "loss": 0.3575
    },
    {
      "step": 2275,
      "loss": 0.3763
    },
    {
      "step": 2280,
      "loss": 0.3873
    },
    {
      "step": 2285,
      "loss": 0.3578
    },
    {
      "step": 2290,
      "loss": 0.378
    },
    {
      "step": 2295,
      "loss": 0.3671
    },
    {
      "step": 2300,
      "loss": 0.385
    },
    {
      "step": 2305,
      "loss": 0.3671
    },
    {
      "step": 2310,
      "loss": 0.3722
    },
    {
      "step": 2315,
      "loss": 0.4093
    },
    {
      "step": 2320,
      "loss": 0.3804
    },
    {
      "step": 2325,
      "loss": 0.3904
    },
    {
      "step": 2330,
      "loss": 0.3668
    },
    {
      "step": 2335,
      "loss": 0.371
    },
    {
      "step": 2340,
      "loss": 0.3608
    },
    {
      "step": 2345,
      "loss": 0.3589
    },
    {
      "step": 2350,
      "loss": 0.3813
    },
    {
      "step": 2355,
      "loss": 0.3633
    },
    {
      "step": 2360,
      "loss": 0.3684
    },
    {
      "step": 2365,
      "loss": 0.37
    },
    {
      "step": 2370,
      "loss": 0.3637
    },
    {
      "step": 2375,
      "loss": 0.3566
    },
    {
      "step": 2380,
      "loss": 0.3581
    },
    {
      "step": 2385,
      "loss": 0.3455
    },
    {
      "step": 2390,
      "loss": 0.3897
    },
    {
      "step": 2395,
      "loss": 0.3745
    },
    {
      "step": 2400,
      "loss": 0.3955
    },
    {
      "step": 2405,
      "loss": 0.3706
    },
    {
      "step": 2410,
      "loss": 0.3515
    },
    {
      "step": 2415,
      "loss": 0.3449
    },
    {
      "step": 2420,
      "loss": 0.3958
    },
    {
      "step": 2425,
      "loss": 0.398
    },
    {
      "step": 2430,
      "loss": 0.3567
    },
    {
      "step": 2435,
      "loss": 0.3645
    },
    {
      "step": 2440,
      "loss": 0.357
    },
    {
      "step": 2445,
      "loss": 0.3654
    },
    {
      "step": 2450,
      "loss": 0.3463
    },
    {
      "step": 2455,
      "loss": 0.3595
    },
    {
      "step": 2460,
      "loss": 0.3842
    },
    {
      "step": 2465,
      "loss": 0.3745
    },
    {
      "step": 2470,
      "loss": 0.3344
    },
    {
      "step": 2475,
      "loss": 0.3593
    },
    {
      "step": 2480,
      "loss": 0.3478
    },
    {
      "step": 2485,
      "loss": 0.3997
    },
    {
      "step": 2490,
      "loss": 0.3765
    },
    {
      "step": 2495,
      "loss": 0.3646
    },
    {
      "step": 2500,
      "loss": 0.3629
    },
    {
      "step": 2505,
      "loss": 0.3685
    },
    {
      "step": 2510,
      "loss": 0.3749
    },
    {
      "step": 2515,
      "loss": 0.3304
    },
    {
      "step": 2520,
      "loss": 0.3726
    },
    {
      "step": 2525,
      "loss": 0.3669
    },
    {
      "step": 2530,
      "loss": 0.3599
    },
    {
      "step": 2535,
      "loss": 0.3598
    },
    {
      "step": 2540,
      "loss": 0.33
    },
    {
      "step": 2545,
      "loss": 0.3511
    },
    {
      "step": 2550,
      "loss": 0.3813
    },
    {
      "step": 2555,
      "loss": 0.3375
    },
    {
      "step": 2560,
      "loss": 0.3463
    },
    {
      "step": 2565,
      "loss": 0.3353
    },
    {
      "step": 2570,
      "loss": 0.357
    },
    {
      "step": 2575,
      "loss": 0.34
    },
    {
      "step": 2580,
      "loss": 0.3506
    },
    {
      "step": 2585,
      "loss": 0.3491
    },
    {
      "step": 2590,
      "loss": 0.3501
    },
    {
      "step": 2595,
      "loss": 0.3505
    },
    {
      "step": 2600,
      "loss": 0.3522
    },
    {
      "step": 2605,
      "loss": 0.3624
    },
    {
      "step": 2610,
      "loss": 0.3639
    },
    {
      "step": 2615,
      "loss": 0.368
    },
    {
      "step": 2620,
      "loss": 0.3618
    },
    {
      "step": 2625,
      "loss": 0.3555
    },
    {
      "step": 2630,
      "loss": 0.3596
    },
    {
      "step": 2635,
      "loss": 0.38
    },
    {
      "step": 2640,
      "loss": 0.3378
    },
    {
      "step": 2645,
      "loss": 0.3559
    },
    {
      "step": 2650,
      "loss": 0.3554
    },
    {
      "step": 2655,
      "loss": 0.3692
    },
    {
      "step": 2660,
      "loss": 0.3445
    },
    {
      "step": 2665,
      "loss": 0.3682
    },
    {
      "step": 2670,
      "loss": 0.3513
    },
    {
      "step": 2675,
      "loss": 0.3496
    },
    {
      "step": 2680,
      "loss": 0.3192
    },
    {
      "step": 2685,
      "loss": 0.3304
    },
    {
      "step": 2690,
      "loss": 0.3082
    },
    {
      "step": 2695,
      "loss": 0.3281
    },
    {
      "step": 2700,
      "loss": 0.3363
    },
    {
      "step": 2705,
      "loss": 0.3541
    },
    {
      "step": 2710,
      "loss": 0.3567
    },
    {
      "step": 2715,
      "loss": 0.3731
    },
    {
      "step": 2720,
      "loss": 0.3257
    },
    {
      "step": 2725,
      "loss": 0.3654
    },
    {
      "step": 2730,
      "loss": 0.336
    },
    {
      "step": 2735,
      "loss": 0.3697
    },
    {
      "step": 2740,
      "loss": 0.3451
    },
    {
      "step": 2745,
      "loss": 0.3641
    },
    {
      "step": 2750,
      "loss": 0.3474
    },
    {
      "step": 2755,
      "loss": 0.3407
    },
    {
      "step": 2760,
      "loss": 0.324
    },
    {
      "step": 2765,
      "loss": 0.3387
    },
    {
      "step": 2770,
      "loss": 0.3349
    },
    {
      "step": 2775,
      "loss": 0.3448
    },
    {
      "step": 2780,
      "loss": 0.3513
    },
    {
      "step": 2785,
      "loss": 0.3513
    },
    {
      "step": 2790,
      "loss": 0.2999
    },
    {
      "step": 2795,
      "loss": 0.3062
    },
    {
      "step": 2800,
      "loss": 0.3572
    },
    {
      "step": 2805,
      "loss": 0.3434
    },
    {
      "step": 2810,
      "loss": 0.3438
    },
    {
      "step": 2815,
      "loss": 0.3428
    },
    {
      "step": 2820,
      "loss": 0.34
    },
    {
      "step": 2825,
      "loss": 0.3326
    },
    {
      "step": 2830,
      "loss": 0.3285
    },
    {
      "step": 2835,
      "loss": 0.2982
    },
    {
      "step": 2840,
      "loss": 0.348
    },
    {
      "step": 2845,
      "loss": 0.319
    },
    {
      "step": 2850,
      "loss": 0.3331
    },
    {
      "step": 2855,
      "loss": 0.325
    },
    {
      "step": 2860,
      "loss": 0.2869
    },
    {
      "step": 2865,
      "loss": 0.3575
    },
    {
      "step": 2870,
      "loss": 0.3223
    },
    {
      "step": 2875,
      "loss": 0.3232
    },
    {
      "step": 2880,
      "loss": 0.3368
    },
    {
      "step": 2885,
      "loss": 0.325
    },
    {
      "step": 2890,
      "loss": 0.3315
    },
    {
      "step": 2895,
      "loss": 0.3756
    },
    {
      "step": 2900,
      "loss": 0.3557
    },
    {
      "step": 2905,
      "loss": 0.3362
    },
    {
      "step": 2910,
      "loss": 0.3315
    },
    {
      "step": 2915,
      "loss": 0.3465
    },
    {
      "step": 2920,
      "loss": 0.3387
    },
    {
      "step": 2925,
      "loss": 0.3134
    },
    {
      "step": 2930,
      "loss": 0.3127
    },
    {
      "step": 2935,
      "loss": 0.3555
    },
    {
      "step": 2940,
      "loss": 0.3444
    },
    {
      "step": 2945,
      "loss": 0.3373
    },
    {
      "step": 2950,
      "loss": 0.3255
    },
    {
      "step": 2955,
      "loss": 0.3168
    },
    {
      "step": 2960,
      "loss": 0.3031
    },
    {
      "step": 2965,
      "loss": 0.3375
    },
    {
      "step": 2970,
      "loss": 0.3232
    },
    {
      "step": 2975,
      "loss": 0.3203
    },
    {
      "step": 2980,
      "loss": 0.3505
    },
    {
      "step": 2985,
      "loss": 0.3206
    },
    {
      "step": 2990,
      "loss": 0.328
    },
    {
      "step": 2995,
      "loss": 0.3122
    },
    {
      "step": 3000,
      "loss": 0.3287
    },
    {
      "step": 3005,
      "loss": 0.3293
    },
    {
      "step": 3010,
      "loss": 0.3366
    },
    {
      "step": 3015,
      "loss": 0.3048
    },
    {
      "step": 3020,
      "loss": 0.3414
    },
    {
      "step": 3025,
      "loss": 0.3533
    },
    {
      "step": 3030,
      "loss": 0.3784
    },
    {
      "step": 3035,
      "loss": 0.3364
    },
    {
      "step": 3040,
      "loss": 0.31
    },
    {
      "step": 3045,
      "loss": 0.3278
    },
    {
      "step": 3050,
      "loss": 0.3304
    },
    {
      "step": 3055,
      "loss": 0.3385
    },
    {
      "step": 3060,
      "loss": 0.3438
    },
    {
      "step": 3065,
      "loss": 0.3198
    },
    {
      "step": 3070,
      "loss": 0.3317
    },
    {
      "step": 3075,
      "loss": 0.3253
    },
    {
      "step": 3080,
      "loss": 0.3086
    },
    {
      "step": 3085,
      "loss": 0.3139
    },
    {
      "step": 3090,
      "loss": 0.3123
    },
    {
      "step": 3095,
      "loss": 0.3574
    },
    {
      "step": 3100,
      "loss": 0.3003
    },
    {
      "step": 3105,
      "loss": 0.3456
    },
    {
      "step": 3110,
      "loss": 0.3126
    },
    {
      "step": 3115,
      "loss": 0.3377
    },
    {
      "step": 3120,
      "loss": 0.3152
    },
    {
      "step": 3125,
      "loss": 0.3028
    },
    {
      "step": 3130,
      "loss": 0.3088
    },
    {
      "step": 3135,
      "loss": 0.3118
    },
    {
      "step": 3140,
      "loss": 0.3023
    },
    {
      "step": 3145,
      "loss": 0.329
    },
    {
      "step": 3150,
      "loss": 0.3579
    },
    {
      "step": 3155,
      "loss": 0.3385
    },
    {
      "step": 3160,
      "loss": 0.3397
    },
    {
      "step": 3165,
      "loss": 0.3112
    },
    {
      "step": 3170,
      "loss": 0.306
    },
    {
      "step": 3175,
      "loss": 0.3152
    },
    {
      "step": 3180,
      "loss": 0.3117
    },
    {
      "step": 3185,
      "loss": 0.3247
    },
    {
      "step": 3190,
      "loss": 0.3077
    },
    {
      "step": 3195,
      "loss": 0.2973
    },
    {
      "step": 3200,
      "loss": 0.3048
    },
    {
      "step": 3205,
      "loss": 0.3599
    },
    {
      "step": 3210,
      "loss": 0.2888
    },
    {
      "step": 3215,
      "loss": 0.2966
    },
    {
      "step": 3220,
      "loss": 0.2949
    },
    {
      "step": 3225,
      "loss": 0.3408
    },
    {
      "step": 3230,
      "loss": 0.3486
    },
    {
      "step": 3235,
      "loss": 0.3374
    },
    {
      "step": 3240,
      "loss": 0.3088
    },
    {
      "step": 3245,
      "loss": 0.2913
    },
    {
      "step": 3250,
      "loss": 0.3144
    },
    {
      "step": 3255,
      "loss": 0.336
    },
    {
      "step": 3260,
      "loss": 0.327
    },
    {
      "step": 3265,
      "loss": 0.3113
    },
    {
      "step": 3270,
      "loss": 0.3216
    },
    {
      "step": 3275,
      "loss": 0.3237
    },
    {
      "step": 3280,
      "loss": 0.3236
    },
    {
      "step": 3285,
      "loss": 0.3408
    },
    {
      "step": 3290,
      "loss": 0.3625
    },
    {
      "step": 3295,
      "loss": 0.2913
    },
    {
      "step": 3300,
      "loss": 0.3758
    },
    {
      "step": 3305,
      "loss": 0.2885
    },
    {
      "step": 3310,
      "loss": 0.3063
    },
    {
      "step": 3315,
      "loss": 0.304
    },
    {
      "step": 3320,
      "loss": 0.3245
    },
    {
      "step": 3325,
      "loss": 0.325
    },
    {
      "step": 3330,
      "loss": 0.3147
    },
    {
      "step": 3335,
      "loss": 0.3392
    },
    {
      "step": 3340,
      "loss": 0.3058
    },
    {
      "step": 3345,
      "loss": 0.3149
    },
    {
      "step": 3350,
      "loss": 0.2951
    },
    {
      "step": 3355,
      "loss": 0.3137
    },
    {
      "step": 3360,
      "loss": 0.3087
    },
    {
      "step": 3365,
      "loss": 0.3015
    },
    {
      "step": 3370,
      "loss": 0.3109
    },
    {
      "step": 3375,
      "loss": 0.2946
    },
    {
      "step": 3380,
      "loss": 0.2976
    },
    {
      "step": 3385,
      "loss": 0.3218
    },
    {
      "step": 3390,
      "loss": 0.3309
    },
    {
      "step": 3395,
      "loss": 0.3116
    },
    {
      "step": 3400,
      "loss": 0.3143
    },
    {
      "step": 3405,
      "loss": 0.2955
    },
    {
      "step": 3410,
      "loss": 0.296
    },
    {
      "step": 3415,
      "loss": 0.3291
    },
    {
      "step": 3420,
      "loss": 0.3217
    },
    {
      "step": 3425,
      "loss": 0.2736
    },
    {
      "step": 3430,
      "loss": 0.3082
    },
    {
      "step": 3435,
      "loss": 0.2763
    },
    {
      "step": 3440,
      "loss": 0.316
    },
    {
      "step": 3445,
      "loss": 0.3361
    },
    {
      "step": 3450,
      "loss": 0.2904
    },
    {
      "step": 3455,
      "loss": 0.3149
    },
    {
      "step": 3460,
      "loss": 0.2925
    },
    {
      "step": 3465,
      "loss": 0.3301
    },
    {
      "step": 3470,
      "loss": 0.3131
    },
    {
      "step": 3475,
      "loss": 0.3097
    },
    {
      "step": 3480,
      "loss": 0.2974
    },
    {
      "step": 3485,
      "loss": 0.3269
    },
    {
      "step": 3490,
      "loss": 0.2977
    },
    {
      "step": 3495,
      "loss": 0.2975
    },
    {
      "step": 3500,
      "loss": 0.3159
    },
    {
      "step": 3505,
      "loss": 0.3156
    },
    {
      "step": 3510,
      "loss": 0.3364
    },
    {
      "step": 3515,
      "loss": 0.3111
    },
    {
      "step": 3520,
      "loss": 0.2965
    },
    {
      "step": 3525,
      "loss": 0.3061
    },
    {
      "step": 3530,
      "loss": 0.3018
    },
    {
      "step": 3535,
      "loss": 0.2942
    },
    {
      "step": 3540,
      "loss": 0.3082
    },
    {
      "step": 3545,
      "loss": 0.2902
    },
    {
      "step": 3550,
      "loss": 0.3153
    },
    {
      "step": 3555,
      "loss": 0.3006
    },
    {
      "step": 3560,
      "loss": 0.2915
    },
    {
      "step": 3565,
      "loss": 0.2922
    },
    {
      "step": 3570,
      "loss": 0.3073
    },
    {
      "step": 3575,
      "loss": 0.2778
    },
    {
      "step": 3580,
      "loss": 0.3026
    },
    {
      "step": 3585,
      "loss": 0.3163
    },
    {
      "step": 3590,
      "loss": 0.3092
    },
    {
      "step": 3595,
      "loss": 0.2749
    },
    {
      "step": 3600,
      "loss": 0.3101
    },
    {
      "step": 3605,
      "loss": 0.3091
    },
    {
      "step": 3610,
      "loss": 0.2993
    },
    {
      "step": 3615,
      "loss": 0.2767
    },
    {
      "step": 3620,
      "loss": 0.3065
    },
    {
      "step": 3625,
      "loss": 0.3176
    },
    {
      "step": 3630,
      "loss": 0.2963
    },
    {
      "step": 3635,
      "loss": 0.2936
    },
    {
      "step": 3640,
      "loss": 0.306
    },
    {
      "step": 3645,
      "loss": 0.3024
    },
    {
      "step": 3650,
      "loss": 0.2975
    },
    {
      "step": 3655,
      "loss": 0.3145
    },
    {
      "step": 3660,
      "loss": 0.3038
    },
    {
      "step": 3665,
      "loss": 0.3157
    },
    {
      "step": 3670,
      "loss": 0.2579
    },
    {
      "step": 3675,
      "loss": 0.3137
    },
    {
      "step": 3680,
      "loss": 0.3122
    },
    {
      "step": 3685,
      "loss": 0.3328
    },
    {
      "step": 3690,
      "loss": 0.2924
    },
    {
      "step": 3695,
      "loss": 0.3235
    },
    {
      "step": 3700,
      "loss": 0.2691
    },
    {
      "step": 3705,
      "loss": 0.2837
    },
    {
      "step": 3710,
      "loss": 0.2776
    },
    {
      "step": 3715,
      "loss": 0.3062
    },
    {
      "step": 3720,
      "loss": 0.3112
    },
    {
      "step": 3725,
      "loss": 0.2999
    },
    {
      "step": 3730,
      "loss": 0.3055
    },
    {
      "step": 3735,
      "loss": 0.2872
    },
    {
      "step": 3740,
      "loss": 0.3015
    },
    {
      "step": 3745,
      "loss": 0.3134
    },
    {
      "step": 3750,
      "loss": 0.2822
    },
    {
      "step": 3755,
      "loss": 0.2877
    },
    {
      "step": 3760,
      "loss": 0.287
    },
    {
      "step": 3765,
      "loss": 0.2751
    },
    {
      "step": 3770,
      "loss": 0.2755
    },
    {
      "step": 3775,
      "loss": 0.2829
    },
    {
      "step": 3780,
      "loss": 0.2978
    },
    {
      "step": 3785,
      "loss": 0.3083
    },
    {
      "step": 3790,
      "loss": 0.3063
    },
    {
      "step": 3795,
      "loss": 0.3082
    },
    {
      "step": 3800,
      "loss": 0.2667
    },
    {
      "step": 3805,
      "loss": 0.2869
    },
    {
      "step": 3810,
      "loss": 0.2792
    },
    {
      "step": 3815,
      "loss": 0.29
    },
    {
      "step": 3820,
      "loss": 0.2911
    },
    {
      "step": 3825,
      "loss": 0.2849
    },
    {
      "step": 3830,
      "loss": 0.272
    },
    {
      "step": 3835,
      "loss": 0.2616
    },
    {
      "step": 3840,
      "loss": 0.265
    },
    {
      "step": 3845,
      "loss": 0.2712
    },
    {
      "step": 3850,
      "loss": 0.2927
    },
    {
      "step": 3855,
      "loss": 0.2853
    },
    {
      "step": 3860,
      "loss": 0.2811
    },
    {
      "step": 3865,
      "loss": 0.2657
    },
    {
      "step": 3870,
      "loss": 0.2814
    },
    {
      "step": 3875,
      "loss": 0.2778
    },
    {
      "step": 3880,
      "loss": 0.2877
    },
    {
      "step": 3885,
      "loss": 0.3021
    },
    {
      "step": 3890,
      "loss": 0.2759
    },
    {
      "step": 3895,
      "loss": 0.2774
    },
    {
      "step": 3900,
      "loss": 0.2826
    },
    {
      "step": 3905,
      "loss": 0.3174
    },
    {
      "step": 3910,
      "loss": 0.2828
    },
    {
      "step": 3915,
      "loss": 0.2523
    },
    {
      "step": 3920,
      "loss": 0.3073
    },
    {
      "step": 3925,
      "loss": 0.2686
    },
    {
      "step": 3930,
      "loss": 0.3022
    },
    {
      "step": 3935,
      "loss": 0.2769
    },
    {
      "step": 3940,
      "loss": 0.2816
    },
    {
      "step": 3945,
      "loss": 0.2802
    },
    {
      "step": 3950,
      "loss": 0.272
    },
    {
      "step": 3955,
      "loss": 0.2837
    },
    {
      "step": 3960,
      "loss": 0.2758
    },
    {
      "step": 3965,
      "loss": 0.2804
    },
    {
      "step": 3970,
      "loss": 0.2921
    },
    {
      "step": 3975,
      "loss": 0.2714
    },
    {
      "step": 3980,
      "loss": 0.2814
    },
    {
      "step": 3985,
      "loss": 0.2958
    },
    {
      "step": 3990,
      "loss": 0.3062
    },
    {
      "step": 3995,
      "loss": 0.2874
    },
    {
      "step": 4000,
      "loss": 0.3012
    },
    {
      "step": 4005,
      "loss": 0.2856
    },
    {
      "step": 4010,
      "loss": 0.2812
    },
    {
      "step": 4015,
      "loss": 0.2921
    },
    {
      "step": 4020,
      "loss": 0.3323
    },
    {
      "step": 4025,
      "loss": 0.2694
    },
    {
      "step": 4030,
      "loss": 0.2561
    },
    {
      "step": 4035,
      "loss": 0.2976
    },
    {
      "step": 4040,
      "loss": 0.2807
    },
    {
      "step": 4045,
      "loss": 0.2922
    },
    {
      "step": 4050,
      "loss": 0.2688
    },
    {
      "step": 4055,
      "loss": 0.2906
    },
    {
      "step": 4060,
      "loss": 0.26
    },
    {
      "step": 4065,
      "loss": 0.284
    },
    {
      "step": 4070,
      "loss": 0.2808
    },
    {
      "step": 4075,
      "loss": 0.2754
    },
    {
      "step": 4080,
      "loss": 0.2547
    },
    {
      "step": 4085,
      "loss": 0.2566
    },
    {
      "step": 4090,
      "loss": 0.2735
    },
    {
      "step": 4095,
      "loss": 0.2585
    },
    {
      "step": 4100,
      "loss": 0.284
    },
    {
      "step": 4105,
      "loss": 0.2776
    },
    {
      "step": 4110,
      "loss": 0.28
    },
    {
      "step": 4115,
      "loss": 0.2756
    },
    {
      "step": 4120,
      "loss": 0.2673
    },
    {
      "step": 4125,
      "loss": 0.2547
    },
    {
      "step": 4130,
      "loss": 0.2502
    },
    {
      "step": 4135,
      "loss": 0.2583
    },
    {
      "step": 4140,
      "loss": 0.2789
    },
    {
      "step": 4145,
      "loss": 0.2591
    },
    {
      "step": 4150,
      "loss": 0.2611
    },
    {
      "step": 4155,
      "loss": 0.2746
    },
    {
      "step": 4160,
      "loss": 0.2882
    },
    {
      "step": 4165,
      "loss": 0.2663
    },
    {
      "step": 4170,
      "loss": 0.284
    },
    {
      "step": 4175,
      "loss": 0.2446
    },
    {
      "step": 4180,
      "loss": 0.2807
    },
    {
      "step": 4185,
      "loss": 0.2806
    },
    {
      "step": 4190,
      "loss": 0.2723
    },
    {
      "step": 4195,
      "loss": 0.2632
    },
    {
      "step": 4200,
      "loss": 0.2786
    },
    {
      "step": 4205,
      "loss": 0.2687
    },
    {
      "step": 4210,
      "loss": 0.2435
    },
    {
      "step": 4215,
      "loss": 0.2486
    },
    {
      "step": 4220,
      "loss": 0.2556
    },
    {
      "step": 4225,
      "loss": 0.2673
    },
    {
      "step": 4230,
      "loss": 0.2521
    },
    {
      "step": 4235,
      "loss": 0.2524
    },
    {
      "step": 4240,
      "loss": 0.28
    },
    {
      "step": 4245,
      "loss": 0.2963
    },
    {
      "step": 4250,
      "loss": 0.2592
    },
    {
      "step": 4255,
      "loss": 0.269
    },
    {
      "step": 4260,
      "loss": 0.271
    },
    {
      "step": 4265,
      "loss": 0.2697
    },
    {
      "step": 4270,
      "loss": 0.2426
    },
    {
      "step": 4275,
      "loss": 0.2753
    },
    {
      "step": 4280,
      "loss": 0.2283
    },
    {
      "step": 4285,
      "loss": 0.242
    },
    {
      "step": 4290,
      "loss": 0.2715
    },
    {
      "step": 4295,
      "loss": 0.2724
    },
    {
      "step": 4300,
      "loss": 0.2861
    },
    {
      "step": 4305,
      "loss": 0.2563
    },
    {
      "step": 4310,
      "loss": 0.2541
    },
    {
      "step": 4315,
      "loss": 0.2534
    },
    {
      "step": 4320,
      "loss": 0.2825
    },
    {
      "step": 4325,
      "loss": 0.2574
    },
    {
      "step": 4330,
      "loss": 0.2594
    },
    {
      "step": 4335,
      "loss": 0.2779
    },
    {
      "step": 4340,
      "loss": 0.2565
    },
    {
      "step": 4345,
      "loss": 0.2491
    },
    {
      "step": 4350,
      "loss": 0.2623
    },
    {
      "step": 4355,
      "loss": 0.2686
    },
    {
      "step": 4360,
      "loss": 0.2798
    },
    {
      "step": 4365,
      "loss": 0.2538
    },
    {
      "step": 4370,
      "loss": 0.2963
    },
    {
      "step": 4375,
      "loss": 0.2564
    },
    {
      "step": 4380,
      "loss": 0.2385
    },
    {
      "step": 4385,
      "loss": 0.265
    },
    {
      "step": 4390,
      "loss": 0.2708
    },
    {
      "step": 4395,
      "loss": 0.2808
    },
    {
      "step": 4400,
      "loss": 0.2743
    },
    {
      "step": 4405,
      "loss": 0.303
    },
    {
      "step": 4410,
      "loss": 0.254
    },
    {
      "step": 4415,
      "loss": 0.2536
    },
    {
      "step": 4420,
      "loss": 0.2558
    },
    {
      "step": 4425,
      "loss": 0.2717
    },
    {
      "step": 4430,
      "loss": 0.2746
    },
    {
      "step": 4435,
      "loss": 0.2879
    },
    {
      "step": 4440,
      "loss": 0.2377
    },
    {
      "step": 4445,
      "loss": 0.257
    },
    {
      "step": 4450,
      "loss": 0.2559
    },
    {
      "step": 4455,
      "loss": 0.2674
    },
    {
      "step": 4460,
      "loss": 0.2822
    },
    {
      "step": 4465,
      "loss": 0.2721
    },
    {
      "step": 4470,
      "loss": 0.248
    },
    {
      "step": 4475,
      "loss": 0.2554
    },
    {
      "step": 4480,
      "loss": 0.2562
    },
    {
      "step": 4485,
      "loss": 0.2864
    },
    {
      "step": 4490,
      "loss": 0.2548
    },
    {
      "step": 4495,
      "loss": 0.2513
    },
    {
      "step": 4500,
      "loss": 0.2546
    },
    {
      "step": 4505,
      "loss": 0.2255
    },
    {
      "step": 4510,
      "loss": 0.2511
    },
    {
      "step": 4515,
      "loss": 0.2712
    },
    {
      "step": 4520,
      "loss": 0.272
    },
    {
      "step": 4525,
      "loss": 0.2427
    },
    {
      "step": 4530,
      "loss": 0.2562
    },
    {
      "step": 4535,
      "loss": 0.2604
    },
    {
      "step": 4540,
      "loss": 0.2467
    },
    {
      "step": 4545,
      "loss": 0.2587
    },
    {
      "step": 4550,
      "loss": 0.2422
    },
    {
      "step": 4555,
      "loss": 0.24
    },
    {
      "step": 4560,
      "loss": 0.2669
    },
    {
      "step": 4565,
      "loss": 0.2438
    },
    {
      "step": 4570,
      "loss": 0.2557
    },
    {
      "step": 4575,
      "loss": 0.2469
    },
    {
      "step": 4580,
      "loss": 0.2496
    },
    {
      "step": 4585,
      "loss": 0.2636
    },
    {
      "step": 4590,
      "loss": 0.2473
    },
    {
      "step": 4595,
      "loss": 0.2633
    },
    {
      "step": 4600,
      "loss": 0.2379
    },
    {
      "step": 4605,
      "loss": 0.2591
    },
    {
      "step": 4610,
      "loss": 0.2359
    },
    {
      "step": 4615,
      "loss": 0.2598
    },
    {
      "step": 4620,
      "loss": 0.2422
    },
    {
      "step": 4625,
      "loss": 0.2568
    },
    {
      "step": 4630,
      "loss": 0.2415
    },
    {
      "step": 4635,
      "loss": 0.2579
    },
    {
      "step": 4640,
      "loss": 0.2425
    },
    {
      "step": 4645,
      "loss": 0.2787
    },
    {
      "step": 4650,
      "loss": 0.268
    },
    {
      "step": 4655,
      "loss": 0.2359
    },
    {
      "step": 4660,
      "loss": 0.2365
    },
    {
      "step": 4665,
      "loss": 0.2466
    },
    {
      "step": 4670,
      "loss": 0.2215
    },
    {
      "step": 4675,
      "loss": 0.2262
    },
    {
      "step": 4680,
      "loss": 0.2313
    },
    {
      "step": 4685,
      "loss": 0.2612
    },
    {
      "step": 4690,
      "loss": 0.2426
    },
    {
      "step": 4695,
      "loss": 0.2363
    },
    {
      "step": 4700,
      "loss": 0.2386
    },
    {
      "step": 4705,
      "loss": 0.2452
    },
    {
      "step": 4710,
      "loss": 0.2442
    },
    {
      "step": 4715,
      "loss": 0.246
    },
    {
      "step": 4720,
      "loss": 0.2467
    },
    {
      "step": 4725,
      "loss": 0.2341
    },
    {
      "step": 4730,
      "loss": 0.2394
    },
    {
      "step": 4735,
      "loss": 0.29
    },
    {
      "step": 4740,
      "loss": 0.2444
    },
    {
      "step": 4745,
      "loss": 0.2526
    },
    {
      "step": 4750,
      "loss": 0.2696
    },
    {
      "step": 4755,
      "loss": 0.2317
    },
    {
      "step": 4760,
      "loss": 0.2329
    },
    {
      "step": 4765,
      "loss": 0.2703
    },
    {
      "step": 4770,
      "loss": 0.2434
    },
    {
      "step": 4775,
      "loss": 0.2609
    },
    {
      "step": 4780,
      "loss": 0.2272
    },
    {
      "step": 4785,
      "loss": 0.2627
    },
    {
      "step": 4790,
      "loss": 0.2309
    },
    {
      "step": 4795,
      "loss": 0.2524
    },
    {
      "step": 4800,
      "loss": 0.2587
    },
    {
      "step": 4805,
      "loss": 0.2601
    },
    {
      "step": 4810,
      "loss": 0.2362
    },
    {
      "step": 4815,
      "loss": 0.2412
    },
    {
      "step": 4820,
      "loss": 0.259
    },
    {
      "step": 4825,
      "loss": 0.2554
    },
    {
      "step": 4830,
      "loss": 0.2337
    },
    {
      "step": 4835,
      "loss": 0.2342
    },
    {
      "step": 4840,
      "loss": 0.2465
    },
    {
      "step": 4845,
      "loss": 0.2432
    },
    {
      "step": 4850,
      "loss": 0.244
    },
    {
      "step": 4855,
      "loss": 0.2396
    },
    {
      "step": 4860,
      "loss": 0.2203
    },
    {
      "step": 4865,
      "loss": 0.2558
    },
    {
      "step": 4870,
      "loss": 0.229
    },
    {
      "step": 4875,
      "loss": 0.2347
    },
    {
      "step": 4880,
      "loss": 0.2293
    },
    {
      "step": 4885,
      "loss": 0.224
    },
    {
      "step": 4890,
      "loss": 0.2416
    },
    {
      "step": 4895,
      "loss": 0.2618
    },
    {
      "step": 4900,
      "loss": 0.2147
    },
    {
      "step": 4905,
      "loss": 0.2436
    },
    {
      "step": 4910,
      "loss": 0.2124
    },
    {
      "step": 4915,
      "loss": 0.2387
    },
    {
      "step": 4920,
      "loss": 0.2117
    },
    {
      "step": 4925,
      "loss": 0.239
    },
    {
      "step": 4930,
      "loss": 0.2321
    },
    {
      "step": 4935,
      "loss": 0.2476
    },
    {
      "step": 4940,
      "loss": 0.2528
    },
    {
      "step": 4945,
      "loss": 0.2432
    },
    {
      "step": 4950,
      "loss": 0.2042
    },
    {
      "step": 4955,
      "loss": 0.2496
    },
    {
      "step": 4960,
      "loss": 0.2538
    },
    {
      "step": 4965,
      "loss": 0.2314
    },
    {
      "step": 4970,
      "loss": 0.2382
    },
    {
      "step": 4975,
      "loss": 0.2516
    },
    {
      "step": 4980,
      "loss": 0.2532
    },
    {
      "step": 4985,
      "loss": 0.2521
    },
    {
      "step": 4990,
      "loss": 0.234
    },
    {
      "step": 4995,
      "loss": 0.2101
    },
    {
      "step": 5000,
      "loss": 0.2594
    },
    {
      "step": 5005,
      "loss": 0.217
    },
    {
      "step": 5010,
      "loss": 0.2203
    },
    {
      "step": 5015,
      "loss": 0.2346
    },
    {
      "step": 5020,
      "loss": 0.2241
    },
    {
      "step": 5025,
      "loss": 0.2123
    },
    {
      "step": 5030,
      "loss": 0.2046
    },
    {
      "step": 5035,
      "loss": 0.2616
    },
    {
      "step": 5040,
      "loss": 0.2405
    },
    {
      "step": 5045,
      "loss": 0.2303
    },
    {
      "step": 5050,
      "loss": 0.2395
    },
    {
      "step": 5055,
      "loss": 0.2358
    },
    {
      "step": 5060,
      "loss": 0.2453
    },
    {
      "step": 5065,
      "loss": 0.2502
    },
    {
      "step": 5070,
      "loss": 0.2246
    },
    {
      "step": 5075,
      "loss": 0.2509
    },
    {
      "step": 5080,
      "loss": 0.2577
    },
    {
      "step": 5085,
      "loss": 0.2434
    },
    {
      "step": 5090,
      "loss": 0.2508
    },
    {
      "step": 5095,
      "loss": 0.2462
    },
    {
      "step": 5100,
      "loss": 0.2382
    },
    {
      "step": 5105,
      "loss": 0.227
    },
    {
      "step": 5110,
      "loss": 0.2311
    },
    {
      "step": 5115,
      "loss": 0.2233
    },
    {
      "step": 5120,
      "loss": 0.2372
    },
    {
      "step": 5125,
      "loss": 0.247
    },
    {
      "step": 5130,
      "loss": 0.2266
    },
    {
      "step": 5135,
      "loss": 0.2514
    },
    {
      "step": 5140,
      "loss": 0.2335
    },
    {
      "step": 5145,
      "loss": 0.2498
    },
    {
      "step": 5150,
      "loss": 0.2284
    },
    {
      "step": 5155,
      "loss": 0.2694
    },
    {
      "step": 5160,
      "loss": 0.2542
    },
    {
      "step": 5165,
      "loss": 0.2677
    },
    {
      "step": 5170,
      "loss": 0.22
    },
    {
      "step": 5175,
      "loss": 0.2418
    },
    {
      "step": 5180,
      "loss": 0.2157
    },
    {
      "step": 5185,
      "loss": 0.2281
    },
    {
      "step": 5190,
      "loss": 0.244
    },
    {
      "step": 5195,
      "loss": 0.2254
    },
    {
      "step": 5200,
      "loss": 0.2256
    },
    {
      "step": 5205,
      "loss": 0.2644
    },
    {
      "step": 5210,
      "loss": 0.2261
    },
    {
      "step": 5215,
      "loss": 0.2422
    },
    {
      "step": 5220,
      "loss": 0.2472
    },
    {
      "step": 5225,
      "loss": 0.2194
    },
    {
      "step": 5230,
      "loss": 0.2321
    },
    {
      "step": 5235,
      "loss": 0.2561
    },
    {
      "step": 5240,
      "loss": 0.247
    },
    {
      "step": 5245,
      "loss": 0.2245
    },
    {
      "step": 5250,
      "loss": 0.2115
    },
    {
      "step": 5255,
      "loss": 0.2273
    },
    {
      "step": 5260,
      "loss": 0.2228
    },
    {
      "step": 5265,
      "loss": 0.2515
    },
    {
      "step": 5270,
      "loss": 0.2199
    },
    {
      "step": 5275,
      "loss": 0.2251
    },
    {
      "step": 5280,
      "loss": 0.2286
    },
    {
      "step": 5285,
      "loss": 0.2399
    },
    {
      "step": 5290,
      "loss": 0.2538
    },
    {
      "step": 5295,
      "loss": 0.2637
    },
    {
      "step": 5300,
      "loss": 0.2208
    },
    {
      "step": 5305,
      "loss": 0.2161
    },
    {
      "step": 5310,
      "loss": 0.2314
    },
    {
      "step": 5315,
      "loss": 0.2452
    },
    {
      "step": 5320,
      "loss": 0.245
    },
    {
      "step": 5325,
      "loss": 0.2596
    },
    {
      "step": 5330,
      "loss": 0.2287
    },
    {
      "step": 5335,
      "loss": 0.2019
    },
    {
      "step": 5340,
      "loss": 0.2377
    },
    {
      "step": 5345,
      "loss": 0.2201
    },
    {
      "step": 5350,
      "loss": 0.25
    },
    {
      "step": 5355,
      "loss": 0.2346
    },
    {
      "step": 5360,
      "loss": 0.2309
    },
    {
      "step": 5365,
      "loss": 0.2134
    },
    {
      "step": 5370,
      "loss": 0.2206
    },
    {
      "step": 5375,
      "loss": 0.2177
    },
    {
      "step": 5380,
      "loss": 0.2239
    },
    {
      "step": 5385,
      "loss": 0.2174
    },
    {
      "step": 5390,
      "loss": 0.249
    },
    {
      "step": 5395,
      "loss": 0.2322
    },
    {
      "step": 5400,
      "loss": 0.2028
    },
    {
      "step": 5405,
      "loss": 0.2203
    },
    {
      "step": 5410,
      "loss": 0.2322
    },
    {
      "step": 5415,
      "loss": 0.2215
    },
    {
      "step": 5420,
      "loss": 0.2323
    },
    {
      "step": 5425,
      "loss": 0.2205
    },
    {
      "step": 5430,
      "loss": 0.2118
    },
    {
      "step": 5435,
      "loss": 0.2135
    },
    {
      "step": 5440,
      "loss": 0.1953
    },
    {
      "step": 5445,
      "loss": 0.2179
    },
    {
      "step": 5450,
      "loss": 0.2151
    },
    {
      "step": 5455,
      "loss": 0.2146
    },
    {
      "step": 5460,
      "loss": 0.2345
    },
    {
      "step": 5465,
      "loss": 0.2243
    },
    {
      "step": 5470,
      "loss": 0.2412
    },
    {
      "step": 5475,
      "loss": 0.2079
    },
    {
      "step": 5480,
      "loss": 0.2051
    },
    {
      "step": 5485,
      "loss": 0.2134
    },
    {
      "step": 5490,
      "loss": 0.1936
    },
    {
      "step": 5495,
      "loss": 0.2319
    },
    {
      "step": 5500,
      "loss": 0.1847
    },
    {
      "step": 5505,
      "loss": 0.1957
    },
    {
      "step": 5510,
      "loss": 0.1995
    },
    {
      "step": 5515,
      "loss": 0.1967
    },
    {
      "step": 5520,
      "loss": 0.1944
    },
    {
      "step": 5525,
      "loss": 0.1852
    },
    {
      "step": 5530,
      "loss": 0.2035
    },
    {
      "step": 5535,
      "loss": 0.2071
    },
    {
      "step": 5540,
      "loss": 0.1934
    },
    {
      "step": 5545,
      "loss": 0.1902
    },
    {
      "step": 5550,
      "loss": 0.1863
    },
    {
      "step": 5555,
      "loss": 0.2104
    },
    {
      "step": 5560,
      "loss": 0.206
    },
    {
      "step": 5565,
      "loss": 0.1859
    },
    {
      "step": 5570,
      "loss": 0.1945
    },
    {
      "step": 5575,
      "loss": 0.2014
    },
    {
      "step": 5580,
      "loss": 0.2041
    },
    {
      "step": 5585,
      "loss": 0.1976
    },
    {
      "step": 5590,
      "loss": 0.1998
    },
    {
      "step": 5595,
      "loss": 0.1895
    },
    {
      "step": 5600,
      "loss": 0.1734
    },
    {
      "step": 5605,
      "loss": 0.1918
    },
    {
      "step": 5610,
      "loss": 0.1891
    },
    {
      "step": 5615,
      "loss": 0.1774
    },
    {
      "step": 5620,
      "loss": 0.1885
    },
    {
      "step": 5625,
      "loss": 0.2103
    },
    {
      "step": 5630,
      "loss": 0.1958
    },
    {
      "step": 5635,
      "loss": 0.1912
    },
    {
      "step": 5640,
      "loss": 0.1901
    },
    {
      "step": 5645,
      "loss": 0.1818
    },
    {
      "step": 5650,
      "loss": 0.181
    },
    {
      "step": 5655,
      "loss": 0.1829
    },
    {
      "step": 5660,
      "loss": 0.1813
    },
    {
      "step": 5665,
      "loss": 0.1894
    },
    {
      "step": 5670,
      "loss": 0.215
    },
    {
      "step": 5675,
      "loss": 0.1927
    },
    {
      "step": 5680,
      "loss": 0.2064
    },
    {
      "step": 5685,
      "loss": 0.1835
    },
    {
      "step": 5690,
      "loss": 0.2042
    },
    {
      "step": 5695,
      "loss": 0.1915
    },
    {
      "step": 5700,
      "loss": 0.2099
    },
    {
      "step": 5705,
      "loss": 0.2016
    },
    {
      "step": 5710,
      "loss": 0.2139
    },
    {
      "step": 5715,
      "loss": 0.1733
    },
    {
      "step": 5720,
      "loss": 0.1853
    },
    {
      "step": 5725,
      "loss": 0.1773
    },
    {
      "step": 5730,
      "loss": 0.1976
    },
    {
      "step": 5735,
      "loss": 0.1729
    },
    {
      "step": 5740,
      "loss": 0.1803
    },
    {
      "step": 5745,
      "loss": 0.1771
    },
    {
      "step": 5750,
      "loss": 0.1772
    },
    {
      "step": 5755,
      "loss": 0.2064
    },
    {
      "step": 5760,
      "loss": 0.1915
    },
    {
      "step": 5765,
      "loss": 0.213
    },
    {
      "step": 5770,
      "loss": 0.2087
    },
    {
      "step": 5775,
      "loss": 0.1894
    },
    {
      "step": 5780,
      "loss": 0.1959
    },
    {
      "step": 5785,
      "loss": 0.2037
    },
    {
      "step": 5790,
      "loss": 0.1664
    },
    {
      "step": 5795,
      "loss": 0.1767
    },
    {
      "step": 5800,
      "loss": 0.1942
    },
    {
      "step": 5805,
      "loss": 0.172
    },
    {
      "step": 5810,
      "loss": 0.2017
    },
    {
      "step": 5815,
      "loss": 0.1926
    },
    {
      "step": 5820,
      "loss": 0.1644
    },
    {
      "step": 5825,
      "loss": 0.1841
    },
    {
      "step": 5830,
      "loss": 0.1763
    },
    {
      "step": 5835,
      "loss": 0.203
    },
    {
      "step": 5840,
      "loss": 0.19
    },
    {
      "step": 5845,
      "loss": 0.1989
    },
    {
      "step": 5850,
      "loss": 0.1974
    },
    {
      "step": 5855,
      "loss": 0.1663
    },
    {
      "step": 5860,
      "loss": 0.1687
    },
    {
      "step": 5865,
      "loss": 0.1856
    },
    {
      "step": 5870,
      "loss": 0.1754
    },
    {
      "step": 5875,
      "loss": 0.1681
    },
    {
      "step": 5880,
      "loss": 0.1931
    },
    {
      "step": 5885,
      "loss": 0.1888
    },
    {
      "step": 5890,
      "loss": 0.1836
    },
    {
      "step": 5895,
      "loss": 0.1855
    },
    {
      "step": 5900,
      "loss": 0.1849
    },
    {
      "step": 5905,
      "loss": 0.1883
    },
    {
      "step": 5910,
      "loss": 0.1884
    },
    {
      "step": 5915,
      "loss": 0.1882
    },
    {
      "step": 5920,
      "loss": 0.1679
    },
    {
      "step": 5925,
      "loss": 0.1857
    },
    {
      "step": 5930,
      "loss": 0.1693
    },
    {
      "step": 5935,
      "loss": 0.193
    },
    {
      "step": 5940,
      "loss": 0.204
    },
    {
      "step": 5945,
      "loss": 0.1913
    },
    {
      "step": 5950,
      "loss": 0.1692
    },
    {
      "step": 5955,
      "loss": 0.179
    },
    {
      "step": 5960,
      "loss": 0.1959
    },
    {
      "step": 5965,
      "loss": 0.1913
    },
    {
      "step": 5970,
      "loss": 0.186
    },
    {
      "step": 5975,
      "loss": 0.175
    },
    {
      "step": 5980,
      "loss": 0.1712
    },
    {
      "step": 5985,
      "loss": 0.1991
    },
    {
      "step": 5990,
      "loss": 0.1865
    },
    {
      "step": 5995,
      "loss": 0.1813
    },
    {
      "step": 6000,
      "loss": 0.2059
    },
    {
      "step": 6005,
      "loss": 0.1846
    },
    {
      "step": 6010,
      "loss": 0.1853
    },
    {
      "step": 6015,
      "loss": 0.186
    },
    {
      "step": 6020,
      "loss": 0.1807
    },
    {
      "step": 6025,
      "loss": 0.1827
    },
    {
      "step": 6030,
      "loss": 0.1892
    },
    {
      "step": 6035,
      "loss": 0.1704
    },
    {
      "step": 6040,
      "loss": 0.1961
    },
    {
      "step": 6045,
      "loss": 0.1996
    },
    {
      "step": 6050,
      "loss": 0.2108
    },
    {
      "step": 6055,
      "loss": 0.1841
    },
    {
      "step": 6060,
      "loss": 0.1866
    },
    {
      "step": 6065,
      "loss": 0.1601
    },
    {
      "step": 6070,
      "loss": 0.1739
    },
    {
      "step": 6075,
      "loss": 0.17
    },
    {
      "step": 6080,
      "loss": 0.1847
    },
    {
      "step": 6085,
      "loss": 0.2051
    },
    {
      "step": 6090,
      "loss": 0.191
    },
    {
      "step": 6095,
      "loss": 0.1803
    },
    {
      "step": 6100,
      "loss": 0.1916
    },
    {
      "step": 6105,
      "loss": 0.1856
    },
    {
      "step": 6110,
      "loss": 0.1707
    },
    {
      "step": 6115,
      "loss": 0.1584
    },
    {
      "step": 6120,
      "loss": 0.2121
    },
    {
      "step": 6125,
      "loss": 0.1903
    },
    {
      "step": 6130,
      "loss": 0.1829
    },
    {
      "step": 6135,
      "loss": 0.1755
    },
    {
      "step": 6140,
      "loss": 0.1721
    },
    {
      "step": 6145,
      "loss": 0.1687
    },
    {
      "step": 6150,
      "loss": 0.1839
    },
    {
      "step": 6155,
      "loss": 0.1642
    },
    {
      "step": 6160,
      "loss": 0.1786
    },
    {
      "step": 6165,
      "loss": 0.162
    },
    {
      "step": 6170,
      "loss": 0.1831
    },
    {
      "step": 6175,
      "loss": 0.1727
    },
    {
      "step": 6180,
      "loss": 0.1774
    },
    {
      "step": 6185,
      "loss": 0.1674
    },
    {
      "step": 6190,
      "loss": 0.1819
    },
    {
      "step": 6195,
      "loss": 0.1653
    },
    {
      "step": 6200,
      "loss": 0.1712
    },
    {
      "step": 6205,
      "loss": 0.1747
    },
    {
      "step": 6210,
      "loss": 0.1856
    },
    {
      "step": 6215,
      "loss": 0.1684
    },
    {
      "step": 6220,
      "loss": 0.1741
    },
    {
      "step": 6225,
      "loss": 0.1912
    },
    {
      "step": 6230,
      "loss": 0.1671
    },
    {
      "step": 6235,
      "loss": 0.1805
    },
    {
      "step": 6240,
      "loss": 0.1896
    },
    {
      "step": 6245,
      "loss": 0.1834
    },
    {
      "step": 6250,
      "loss": 0.2089
    },
    {
      "step": 6255,
      "loss": 0.1723
    },
    {
      "step": 6260,
      "loss": 0.1634
    },
    {
      "step": 6265,
      "loss": 0.1818
    },
    {
      "step": 6270,
      "loss": 0.1796
    },
    {
      "step": 6275,
      "loss": 0.1692
    },
    {
      "step": 6280,
      "loss": 0.186
    },
    {
      "step": 6285,
      "loss": 0.2065
    },
    {
      "step": 6290,
      "loss": 0.1864
    },
    {
      "step": 6295,
      "loss": 0.1744
    },
    {
      "step": 6300,
      "loss": 0.1807
    },
    {
      "step": 6305,
      "loss": 0.1828
    },
    {
      "step": 6310,
      "loss": 0.1704
    },
    {
      "step": 6315,
      "loss": 0.1645
    },
    {
      "step": 6320,
      "loss": 0.1893
    },
    {
      "step": 6325,
      "loss": 0.1929
    },
    {
      "step": 6330,
      "loss": 0.168
    },
    {
      "step": 6335,
      "loss": 0.1964
    },
    {
      "step": 6340,
      "loss": 0.1864
    },
    {
      "step": 6345,
      "loss": 0.1652
    },
    {
      "step": 6350,
      "loss": 0.1798
    },
    {
      "step": 6355,
      "loss": 0.1863
    },
    {
      "step": 6360,
      "loss": 0.1724
    },
    {
      "step": 6365,
      "loss": 0.196
    },
    {
      "step": 6370,
      "loss": 0.1775
    },
    {
      "step": 6375,
      "loss": 0.2047
    },
    {
      "step": 6380,
      "loss": 0.1875
    },
    {
      "step": 6385,
      "loss": 0.1846
    },
    {
      "step": 6390,
      "loss": 0.1539
    },
    {
      "step": 6395,
      "loss": 0.1836
    },
    {
      "step": 6400,
      "loss": 0.1846
    },
    {
      "step": 6405,
      "loss": 0.1747
    },
    {
      "step": 6410,
      "loss": 0.17
    },
    {
      "step": 6415,
      "loss": 0.1753
    },
    {
      "step": 6420,
      "loss": 0.1838
    },
    {
      "step": 6425,
      "loss": 0.1813
    },
    {
      "step": 6430,
      "loss": 0.164
    },
    {
      "step": 6435,
      "loss": 0.1657
    },
    {
      "step": 6440,
      "loss": 0.1669
    },
    {
      "step": 6445,
      "loss": 0.1697
    },
    {
      "step": 6450,
      "loss": 0.1889
    },
    {
      "step": 6455,
      "loss": 0.1996
    },
    {
      "step": 6460,
      "loss": 0.1717
    },
    {
      "step": 6465,
      "loss": 0.1823
    },
    {
      "step": 6470,
      "loss": 0.1581
    },
    {
      "step": 6475,
      "loss": 0.1696
    },
    {
      "step": 6480,
      "loss": 0.2003
    },
    {
      "step": 6485,
      "loss": 0.1524
    },
    {
      "step": 6490,
      "loss": 0.17
    },
    {
      "step": 6495,
      "loss": 0.1796
    },
    {
      "step": 6500,
      "loss": 0.1637
    },
    {
      "step": 6505,
      "loss": 0.169
    },
    {
      "step": 6510,
      "loss": 0.1831
    },
    {
      "step": 6515,
      "loss": 0.1723
    },
    {
      "step": 6520,
      "loss": 0.1697
    },
    {
      "step": 6525,
      "loss": 0.1789
    },
    {
      "step": 6530,
      "loss": 0.1755
    },
    {
      "step": 6535,
      "loss": 0.1837
    },
    {
      "step": 6540,
      "loss": 0.1842
    },
    {
      "step": 6545,
      "loss": 0.1637
    },
    {
      "step": 6550,
      "loss": 0.1635
    },
    {
      "step": 6555,
      "loss": 0.1756
    },
    {
      "step": 6560,
      "loss": 0.1738
    },
    {
      "step": 6565,
      "loss": 0.1783
    },
    {
      "step": 6570,
      "loss": 0.1631
    },
    {
      "step": 6575,
      "loss": 0.175
    },
    {
      "step": 6580,
      "loss": 0.1564
    },
    {
      "step": 6585,
      "loss": 0.1648
    },
    {
      "step": 6590,
      "loss": 0.1763
    },
    {
      "step": 6595,
      "loss": 0.1809
    },
    {
      "step": 6600,
      "loss": 0.1774
    },
    {
      "step": 6605,
      "loss": 0.1553
    },
    {
      "step": 6610,
      "loss": 0.1701
    },
    {
      "step": 6615,
      "loss": 0.1663
    },
    {
      "step": 6620,
      "loss": 0.1676
    },
    {
      "step": 6625,
      "loss": 0.1844
    },
    {
      "step": 6630,
      "loss": 0.158
    },
    {
      "step": 6635,
      "loss": 0.1636
    },
    {
      "step": 6640,
      "loss": 0.1534
    },
    {
      "step": 6645,
      "loss": 0.1777
    },
    {
      "step": 6650,
      "loss": 0.1714
    },
    {
      "step": 6655,
      "loss": 0.1788
    },
    {
      "step": 6660,
      "loss": 0.1749
    },
    {
      "step": 6665,
      "loss": 0.1863
    },
    {
      "step": 6670,
      "loss": 0.1718
    },
    {
      "step": 6675,
      "loss": 0.1794
    },
    {
      "step": 6680,
      "loss": 0.1726
    },
    {
      "step": 6685,
      "loss": 0.1816
    },
    {
      "step": 6690,
      "loss": 0.1654
    },
    {
      "step": 6695,
      "loss": 0.1837
    },
    {
      "step": 6700,
      "loss": 0.1787
    },
    {
      "step": 6705,
      "loss": 0.1675
    },
    {
      "step": 6710,
      "loss": 0.1738
    },
    {
      "step": 6715,
      "loss": 0.1806
    },
    {
      "step": 6720,
      "loss": 0.1617
    },
    {
      "step": 6725,
      "loss": 0.1697
    },
    {
      "step": 6730,
      "loss": 0.1821
    },
    {
      "step": 6735,
      "loss": 0.1743
    },
    {
      "step": 6740,
      "loss": 0.1715
    },
    {
      "step": 6745,
      "loss": 0.1704
    },
    {
      "step": 6750,
      "loss": 0.1614
    },
    {
      "step": 6755,
      "loss": 0.2054
    },
    {
      "step": 6760,
      "loss": 0.1631
    },
    {
      "step": 6765,
      "loss": 0.177
    },
    {
      "step": 6770,
      "loss": 0.1659
    },
    {
      "step": 6775,
      "loss": 0.1909
    },
    {
      "step": 6780,
      "loss": 0.1685
    },
    {
      "step": 6785,
      "loss": 0.1786
    },
    {
      "step": 6790,
      "loss": 0.177
    },
    {
      "step": 6795,
      "loss": 0.1775
    },
    {
      "step": 6800,
      "loss": 0.1506
    },
    {
      "step": 6805,
      "loss": 0.1634
    },
    {
      "step": 6810,
      "loss": 0.1623
    },
    {
      "step": 6815,
      "loss": 0.1637
    },
    {
      "step": 6820,
      "loss": 0.1557
    },
    {
      "step": 6825,
      "loss": 0.1741
    },
    {
      "step": 6830,
      "loss": 0.173
    },
    {
      "step": 6835,
      "loss": 0.1695
    },
    {
      "step": 6840,
      "loss": 0.1704
    },
    {
      "step": 6845,
      "loss": 0.1553
    },
    {
      "step": 6850,
      "loss": 0.1525
    },
    {
      "step": 6855,
      "loss": 0.1822
    },
    {
      "step": 6860,
      "loss": 0.1559
    },
    {
      "step": 6865,
      "loss": 0.1788
    },
    {
      "step": 6870,
      "loss": 0.1825
    },
    {
      "step": 6875,
      "loss": 0.1606
    },
    {
      "step": 6880,
      "loss": 0.1867
    },
    {
      "step": 6885,
      "loss": 0.1638
    },
    {
      "step": 6890,
      "loss": 0.1745
    },
    {
      "step": 6895,
      "loss": 0.1677
    },
    {
      "step": 6900,
      "loss": 0.1524
    },
    {
      "step": 6905,
      "loss": 0.1637
    },
    {
      "step": 6910,
      "loss": 0.1555
    },
    {
      "step": 6915,
      "loss": 0.1689
    },
    {
      "step": 6920,
      "loss": 0.1539
    },
    {
      "step": 6925,
      "loss": 0.1655
    },
    {
      "step": 6930,
      "loss": 0.1699
    },
    {
      "step": 6935,
      "loss": 0.1493
    },
    {
      "step": 6940,
      "loss": 0.1721
    },
    {
      "step": 6945,
      "loss": 0.1784
    },
    {
      "step": 6950,
      "loss": 0.1659
    },
    {
      "step": 6955,
      "loss": 0.1786
    },
    {
      "step": 6960,
      "loss": 0.1693
    },
    {
      "step": 6965,
      "loss": 0.1576
    },
    {
      "step": 6970,
      "loss": 0.1745
    },
    {
      "step": 6975,
      "loss": 0.1681
    },
    {
      "step": 6980,
      "loss": 0.1668
    },
    {
      "step": 6985,
      "loss": 0.1664
    },
    {
      "step": 6990,
      "loss": 0.1753
    },
    {
      "step": 6995,
      "loss": 0.1674
    },
    {
      "step": 7000,
      "loss": 0.1595
    },
    {
      "step": 7005,
      "loss": 0.1841
    },
    {
      "step": 7010,
      "loss": 0.1752
    },
    {
      "step": 7015,
      "loss": 0.1557
    },
    {
      "step": 7020,
      "loss": 0.1764
    },
    {
      "step": 7025,
      "loss": 0.166
    },
    {
      "step": 7030,
      "loss": 0.1598
    },
    {
      "step": 7035,
      "loss": 0.1592
    },
    {
      "step": 7040,
      "loss": 0.1729
    },
    {
      "step": 7045,
      "loss": 0.1523
    },
    {
      "step": 7050,
      "loss": 0.1822
    },
    {
      "step": 7055,
      "loss": 0.1461
    },
    {
      "step": 7060,
      "loss": 0.1667
    },
    {
      "step": 7065,
      "loss": 0.1667
    },
    {
      "step": 7070,
      "loss": 0.1755
    },
    {
      "step": 7075,
      "loss": 0.1585
    },
    {
      "step": 7080,
      "loss": 0.1571
    },
    {
      "step": 7085,
      "loss": 0.1568
    },
    {
      "step": 7090,
      "loss": 0.1679
    },
    {
      "step": 7095,
      "loss": 0.1879
    },
    {
      "step": 7100,
      "loss": 0.1568
    },
    {
      "step": 7105,
      "loss": 0.155
    },
    {
      "step": 7110,
      "loss": 0.151
    },
    {
      "step": 7115,
      "loss": 0.1783
    },
    {
      "step": 7120,
      "loss": 0.1588
    },
    {
      "step": 7125,
      "loss": 0.176
    },
    {
      "step": 7130,
      "loss": 0.1614
    },
    {
      "step": 7135,
      "loss": 0.1699
    },
    {
      "step": 7140,
      "loss": 0.1705
    },
    {
      "step": 7145,
      "loss": 0.1672
    },
    {
      "step": 7150,
      "loss": 0.1485
    },
    {
      "step": 7155,
      "loss": 0.1538
    },
    {
      "step": 7160,
      "loss": 0.1596
    },
    {
      "step": 7165,
      "loss": 0.1613
    },
    {
      "step": 7170,
      "loss": 0.177
    },
    {
      "step": 7175,
      "loss": 0.1649
    },
    {
      "step": 7180,
      "loss": 0.1653
    },
    {
      "step": 7185,
      "loss": 0.1791
    },
    {
      "step": 7190,
      "loss": 0.1731
    },
    {
      "step": 7195,
      "loss": 0.1698
    },
    {
      "step": 7200,
      "loss": 0.176
    },
    {
      "step": 7205,
      "loss": 0.1551
    },
    {
      "step": 7210,
      "loss": 0.1736
    },
    {
      "step": 7215,
      "loss": 0.1709
    },
    {
      "step": 7220,
      "loss": 0.1473
    },
    {
      "step": 7225,
      "loss": 0.1631
    },
    {
      "step": 7230,
      "loss": 0.1651
    },
    {
      "step": 7235,
      "loss": 0.1691
    },
    {
      "step": 7240,
      "loss": 0.1679
    },
    {
      "step": 7245,
      "loss": 0.1481
    },
    {
      "step": 7250,
      "loss": 0.1779
    },
    {
      "step": 7255,
      "loss": 0.1587
    },
    {
      "step": 7260,
      "loss": 0.1507
    },
    {
      "step": 7265,
      "loss": 0.1649
    },
    {
      "step": 7270,
      "loss": 0.1561
    },
    {
      "step": 7275,
      "loss": 0.1546
    },
    {
      "step": 7280,
      "loss": 0.1723
    },
    {
      "step": 7285,
      "loss": 0.1528
    },
    {
      "step": 7290,
      "loss": 0.1603
    },
    {
      "step": 7295,
      "loss": 0.1621
    },
    {
      "step": 7300,
      "loss": 0.1584
    },
    {
      "step": 7305,
      "loss": 0.1655
    },
    {
      "step": 7310,
      "loss": 0.1649
    },
    {
      "step": 7315,
      "loss": 0.1711
    },
    {
      "step": 7320,
      "loss": 0.1602
    },
    {
      "step": 7325,
      "loss": 0.1614
    },
    {
      "step": 7330,
      "loss": 0.1645
    },
    {
      "step": 7335,
      "loss": 0.178
    },
    {
      "step": 7340,
      "loss": 0.1568
    },
    {
      "step": 7345,
      "loss": 0.1657
    },
    {
      "step": 7350,
      "loss": 0.1758
    },
    {
      "step": 7355,
      "loss": 0.1613
    },
    {
      "step": 7360,
      "loss": 0.1495
    },
    {
      "step": 7365,
      "loss": 0.1535
    },
    {
      "step": 7370,
      "loss": 0.1703
    },
    {
      "step": 7375,
      "loss": 0.1722
    },
    {
      "step": 7380,
      "loss": 0.1706
    },
    {
      "step": 7385,
      "loss": 0.1617
    },
    {
      "step": 7390,
      "loss": 0.1552
    },
    {
      "step": 7395,
      "loss": 0.1842
    },
    {
      "step": 7400,
      "loss": 0.1577
    },
    {
      "step": 7405,
      "loss": 0.1573
    },
    {
      "step": 7410,
      "loss": 0.1755
    },
    {
      "step": 7415,
      "loss": 0.1605
    },
    {
      "step": 7420,
      "loss": 0.1547
    },
    {
      "step": 7425,
      "loss": 0.1519
    },
    {
      "step": 7430,
      "loss": 0.1911
    },
    {
      "step": 7435,
      "loss": 0.1511
    },
    {
      "step": 7440,
      "loss": 0.1556
    },
    {
      "step": 7445,
      "loss": 0.1523
    },
    {
      "step": 7450,
      "loss": 0.1461
    },
    {
      "step": 7455,
      "loss": 0.1713
    },
    {
      "step": 7460,
      "loss": 0.1611
    },
    {
      "step": 7465,
      "loss": 0.1581
    },
    {
      "step": 7470,
      "loss": 0.1586
    },
    {
      "step": 7475,
      "loss": 0.1702
    },
    {
      "step": 7480,
      "loss": 0.1546
    },
    {
      "step": 7485,
      "loss": 0.167
    },
    {
      "step": 7490,
      "loss": 0.159
    },
    {
      "step": 7495,
      "loss": 0.1557
    },
    {
      "step": 7500,
      "loss": 0.1575
    },
    {
      "step": 7505,
      "loss": 0.1787
    },
    {
      "step": 7510,
      "loss": 0.1561
    },
    {
      "step": 7515,
      "loss": 0.1604
    },
    {
      "step": 7520,
      "loss": 0.1536
    },
    {
      "step": 7525,
      "loss": 0.1668
    },
    {
      "step": 7530,
      "loss": 0.1689
    },
    {
      "step": 7535,
      "loss": 0.1489
    },
    {
      "step": 7540,
      "loss": 0.149
    },
    {
      "step": 7545,
      "loss": 0.1636
    },
    {
      "step": 7550,
      "loss": 0.1606
    },
    {
      "step": 7555,
      "loss": 0.1568
    },
    {
      "step": 7560,
      "loss": 0.1724
    },
    {
      "step": 7565,
      "loss": 0.1559
    },
    {
      "step": 7570,
      "loss": 0.1653
    },
    {
      "step": 7575,
      "loss": 0.1414
    },
    {
      "step": 7580,
      "loss": 0.1474
    },
    {
      "step": 7585,
      "loss": 0.1541
    },
    {
      "step": 7590,
      "loss": 0.1636
    },
    {
      "step": 7595,
      "loss": 0.1504
    },
    {
      "step": 7600,
      "loss": 0.1669
    },
    {
      "step": 7605,
      "loss": 0.1481
    },
    {
      "step": 7610,
      "loss": 0.157
    },
    {
      "step": 7615,
      "loss": 0.1603
    },
    {
      "step": 7620,
      "loss": 0.1579
    },
    {
      "step": 7625,
      "loss": 0.1454
    },
    {
      "step": 7630,
      "loss": 0.1497
    },
    {
      "step": 7635,
      "loss": 0.1577
    },
    {
      "step": 7640,
      "loss": 0.1676
    },
    {
      "step": 7645,
      "loss": 0.1526
    },
    {
      "step": 7650,
      "loss": 0.1513
    },
    {
      "step": 7655,
      "loss": 0.1555
    },
    {
      "step": 7660,
      "loss": 0.159
    },
    {
      "step": 7665,
      "loss": 0.1471
    },
    {
      "step": 7670,
      "loss": 0.1425
    },
    {
      "step": 7675,
      "loss": 0.1512
    },
    {
      "step": 7680,
      "loss": 0.1658
    },
    {
      "step": 7685,
      "loss": 0.1568
    },
    {
      "step": 7690,
      "loss": 0.1433
    },
    {
      "step": 7695,
      "loss": 0.1583
    },
    {
      "step": 7700,
      "loss": 0.1458
    },
    {
      "step": 7705,
      "loss": 0.1711
    },
    {
      "step": 7710,
      "loss": 0.166
    },
    {
      "step": 7715,
      "loss": 0.1576
    },
    {
      "step": 7720,
      "loss": 0.1644
    },
    {
      "step": 7725,
      "loss": 0.1551
    },
    {
      "step": 7730,
      "loss": 0.1636
    },
    {
      "step": 7735,
      "loss": 0.1731
    },
    {
      "step": 7740,
      "loss": 0.1789
    },
    {
      "step": 7745,
      "loss": 0.1796
    },
    {
      "step": 7750,
      "loss": 0.1638
    },
    {
      "step": 7755,
      "loss": 0.1719
    },
    {
      "step": 7760,
      "loss": 0.1537
    },
    {
      "step": 7765,
      "loss": 0.1512
    },
    {
      "step": 7770,
      "loss": 0.1499
    },
    {
      "step": 7775,
      "loss": 0.145
    },
    {
      "step": 7780,
      "loss": 0.152
    },
    {
      "step": 7785,
      "loss": 0.1609
    },
    {
      "step": 7790,
      "loss": 0.1706
    },
    {
      "step": 7795,
      "loss": 0.1547
    },
    {
      "step": 7800,
      "loss": 0.1512
    },
    {
      "step": 7805,
      "loss": 0.1624
    },
    {
      "step": 7810,
      "loss": 0.1434
    },
    {
      "step": 7815,
      "loss": 0.1618
    },
    {
      "step": 7820,
      "loss": 0.1532
    },
    {
      "step": 7825,
      "loss": 0.1534
    },
    {
      "step": 7830,
      "loss": 0.1426
    },
    {
      "step": 7835,
      "loss": 0.1426
    },
    {
      "step": 7840,
      "loss": 0.1397
    },
    {
      "step": 7845,
      "loss": 0.1608
    },
    {
      "step": 7850,
      "loss": 0.1708
    },
    {
      "step": 7855,
      "loss": 0.1587
    },
    {
      "step": 7860,
      "loss": 0.1539
    },
    {
      "step": 7865,
      "loss": 0.1551
    },
    {
      "step": 7870,
      "loss": 0.1478
    },
    {
      "step": 7875,
      "loss": 0.1693
    },
    {
      "step": 7880,
      "loss": 0.1593
    },
    {
      "step": 7885,
      "loss": 0.136
    },
    {
      "step": 7890,
      "loss": 0.1524
    },
    {
      "step": 7895,
      "loss": 0.151
    },
    {
      "step": 7900,
      "loss": 0.1596
    },
    {
      "step": 7905,
      "loss": 0.1374
    },
    {
      "step": 7910,
      "loss": 0.161
    },
    {
      "step": 7915,
      "loss": 0.1456
    },
    {
      "step": 7920,
      "loss": 0.1545
    },
    {
      "step": 7925,
      "loss": 0.1361
    },
    {
      "step": 7930,
      "loss": 0.1668
    },
    {
      "step": 7935,
      "loss": 0.1517
    },
    {
      "step": 7940,
      "loss": 0.1497
    },
    {
      "step": 7945,
      "loss": 0.1611
    },
    {
      "step": 7950,
      "loss": 0.1502
    },
    {
      "step": 7955,
      "loss": 0.1468
    },
    {
      "step": 7960,
      "loss": 0.153
    },
    {
      "step": 7965,
      "loss": 0.1569
    },
    {
      "step": 7970,
      "loss": 0.1424
    },
    {
      "step": 7975,
      "loss": 0.1538
    },
    {
      "step": 7980,
      "loss": 0.1549
    },
    {
      "step": 7985,
      "loss": 0.1713
    },
    {
      "step": 7990,
      "loss": 0.1538
    },
    {
      "step": 7995,
      "loss": 0.1603
    },
    {
      "step": 8000,
      "loss": 0.1554
    },
    {
      "step": 8005,
      "loss": 0.1553
    },
    {
      "step": 8010,
      "loss": 0.16
    },
    {
      "step": 8015,
      "loss": 0.1486
    },
    {
      "step": 8020,
      "loss": 0.1455
    },
    {
      "step": 8025,
      "loss": 0.1486
    },
    {
      "step": 8030,
      "loss": 0.1594
    },
    {
      "step": 8035,
      "loss": 0.1448
    },
    {
      "step": 8040,
      "loss": 0.1508
    },
    {
      "step": 8045,
      "loss": 0.1563
    },
    {
      "step": 8050,
      "loss": 0.1538
    },
    {
      "step": 8055,
      "loss": 0.1601
    },
    {
      "step": 8060,
      "loss": 0.1411
    },
    {
      "step": 8065,
      "loss": 0.1665
    },
    {
      "step": 8070,
      "loss": 0.1469
    },
    {
      "step": 8075,
      "loss": 0.1386
    },
    {
      "step": 8080,
      "loss": 0.1319
    },
    {
      "step": 8085,
      "loss": 0.1572
    },
    {
      "step": 8090,
      "loss": 0.1417
    },
    {
      "step": 8095,
      "loss": 0.1471
    },
    {
      "step": 8100,
      "loss": 0.1446
    },
    {
      "step": 8105,
      "loss": 0.1396
    },
    {
      "step": 8110,
      "loss": 0.1645
    },
    {
      "step": 8115,
      "loss": 0.1655
    },
    {
      "step": 8120,
      "loss": 0.1414
    },
    {
      "step": 8125,
      "loss": 0.1481
    },
    {
      "step": 8130,
      "loss": 0.1375
    },
    {
      "step": 8135,
      "loss": 0.1445
    },
    {
      "step": 8140,
      "loss": 0.1495
    },
    {
      "step": 8145,
      "loss": 0.1442
    },
    {
      "step": 8150,
      "loss": 0.1475
    },
    {
      "step": 8155,
      "loss": 0.1712
    },
    {
      "step": 8160,
      "loss": 0.1544
    },
    {
      "step": 8165,
      "loss": 0.1498
    },
    {
      "step": 8170,
      "loss": 0.1517
    },
    {
      "step": 8175,
      "loss": 0.1441
    },
    {
      "step": 8180,
      "loss": 0.1371
    },
    {
      "step": 8185,
      "loss": 0.1592
    },
    {
      "step": 8190,
      "loss": 0.1484
    },
    {
      "step": 8195,
      "loss": 0.16
    },
    {
      "step": 8200,
      "loss": 0.1572
    },
    {
      "step": 8205,
      "loss": 0.1512
    },
    {
      "step": 8210,
      "loss": 0.1428
    },
    {
      "step": 8215,
      "loss": 0.1425
    },
    {
      "step": 8220,
      "loss": 0.1631
    },
    {
      "step": 8225,
      "loss": 0.1511
    },
    {
      "step": 8230,
      "loss": 0.1384
    },
    {
      "step": 8235,
      "loss": 0.1563
    },
    {
      "step": 8240,
      "loss": 0.1561
    },
    {
      "step": 8245,
      "loss": 0.1417
    },
    {
      "step": 8250,
      "loss": 0.1371
    },
    {
      "step": 8255,
      "loss": 0.1371
    },
    {
      "step": 8260,
      "loss": 0.1528
    },
    {
      "step": 8265,
      "loss": 0.1637
    },
    {
      "step": 8270,
      "loss": 0.1699
    },
    {
      "step": 8275,
      "loss": 0.1541
    },
    {
      "step": 8280,
      "loss": 0.1399
    },
    {
      "step": 8285,
      "loss": 0.1426
    },
    {
      "step": 8290,
      "loss": 0.1591
    },
    {
      "step": 8295,
      "loss": 0.1599
    },
    {
      "step": 8300,
      "loss": 0.153
    },
    {
      "step": 8305,
      "loss": 0.1485
    },
    {
      "step": 8310,
      "loss": 0.1322
    },
    {
      "step": 8315,
      "loss": 0.1659
    },
    {
      "step": 8320,
      "loss": 0.1509
    },
    {
      "step": 8325,
      "loss": 0.1532
    },
    {
      "step": 8330,
      "loss": 0.1489
    },
    {
      "step": 8335,
      "loss": 0.1374
    },
    {
      "step": 8340,
      "loss": 0.1407
    },
    {
      "step": 8345,
      "loss": 0.1561
    },
    {
      "step": 8350,
      "loss": 0.1379
    },
    {
      "step": 8355,
      "loss": 0.1612
    },
    {
      "step": 8360,
      "loss": 0.146
    },
    {
      "step": 8365,
      "loss": 0.1816
    },
    {
      "step": 8370,
      "loss": 0.1436
    },
    {
      "step": 8375,
      "loss": 0.1596
    },
    {
      "step": 8380,
      "loss": 0.1377
    },
    {
      "step": 8385,
      "loss": 0.1619
    },
    {
      "step": 8390,
      "loss": 0.1484
    },
    {
      "step": 8395,
      "loss": 0.1497
    },
    {
      "step": 8400,
      "loss": 0.1375
    },
    {
      "step": 8405,
      "loss": 0.1422
    },
    {
      "step": 8410,
      "loss": 0.1369
    },
    {
      "step": 8415,
      "loss": 0.1465
    },
    {
      "step": 8420,
      "loss": 0.1487
    },
    {
      "step": 8425,
      "loss": 0.1423
    },
    {
      "step": 8430,
      "loss": 0.1451
    },
    {
      "step": 8435,
      "loss": 0.1638
    },
    {
      "step": 8440,
      "loss": 0.1428
    },
    {
      "step": 8445,
      "loss": 0.147
    },
    {
      "step": 8450,
      "loss": 0.1435
    },
    {
      "step": 8455,
      "loss": 0.1572
    },
    {
      "step": 8460,
      "loss": 0.156
    },
    {
      "step": 8465,
      "loss": 0.1365
    },
    {
      "step": 8470,
      "loss": 0.1434
    },
    {
      "step": 8475,
      "loss": 0.1435
    },
    {
      "step": 8480,
      "loss": 0.158
    },
    {
      "step": 8485,
      "loss": 0.169
    },
    {
      "step": 8490,
      "loss": 0.1452
    },
    {
      "step": 8495,
      "loss": 0.1425
    },
    {
      "step": 8500,
      "loss": 0.1455
    },
    {
      "step": 8505,
      "loss": 0.1502
    },
    {
      "step": 8510,
      "loss": 0.1584
    },
    {
      "step": 8515,
      "loss": 0.1457
    },
    {
      "step": 8520,
      "loss": 0.1536
    },
    {
      "step": 8525,
      "loss": 0.1403
    },
    {
      "step": 8530,
      "loss": 0.1534
    },
    {
      "step": 8535,
      "loss": 0.1398
    },
    {
      "step": 8540,
      "loss": 0.1566
    },
    {
      "step": 8545,
      "loss": 0.1416
    },
    {
      "step": 8550,
      "loss": 0.1429
    },
    {
      "step": 8555,
      "loss": 0.1428
    },
    {
      "step": 8560,
      "loss": 0.1367
    },
    {
      "step": 8565,
      "loss": 0.1604
    },
    {
      "step": 8570,
      "loss": 0.1567
    },
    {
      "step": 8575,
      "loss": 0.1521
    },
    {
      "step": 8580,
      "loss": 0.1438
    },
    {
      "step": 8585,
      "loss": 0.1353
    },
    {
      "step": 8590,
      "loss": 0.1553
    },
    {
      "step": 8595,
      "loss": 0.1533
    },
    {
      "step": 8600,
      "loss": 0.1566
    },
    {
      "step": 8605,
      "loss": 0.1476
    },
    {
      "step": 8610,
      "loss": 0.1613
    },
    {
      "step": 8615,
      "loss": 0.152
    },
    {
      "step": 8620,
      "loss": 0.1407
    },
    {
      "step": 8625,
      "loss": 0.1289
    },
    {
      "step": 8630,
      "loss": 0.157
    },
    {
      "step": 8635,
      "loss": 0.1633
    },
    {
      "step": 8640,
      "loss": 0.1377
    },
    {
      "step": 8645,
      "loss": 0.1384
    },
    {
      "step": 8650,
      "loss": 0.1636
    },
    {
      "step": 8655,
      "loss": 0.1532
    },
    {
      "step": 8660,
      "loss": 0.1572
    },
    {
      "step": 8665,
      "loss": 0.1523
    },
    {
      "step": 8670,
      "loss": 0.1375
    },
    {
      "step": 8675,
      "loss": 0.1426
    },
    {
      "step": 8680,
      "loss": 0.1588
    },
    {
      "step": 8685,
      "loss": 0.1305
    },
    {
      "step": 8690,
      "loss": 0.1325
    },
    {
      "step": 8695,
      "loss": 0.1544
    },
    {
      "step": 8700,
      "loss": 0.1451
    },
    {
      "step": 8705,
      "loss": 0.1576
    },
    {
      "step": 8710,
      "loss": 0.1401
    },
    {
      "step": 8715,
      "loss": 0.1509
    },
    {
      "step": 8720,
      "loss": 0.1317
    },
    {
      "step": 8725,
      "loss": 0.1419
    },
    {
      "step": 8730,
      "loss": 0.1316
    },
    {
      "step": 8735,
      "loss": 0.1379
    },
    {
      "step": 8740,
      "loss": 0.1371
    },
    {
      "step": 8745,
      "loss": 0.1334
    },
    {
      "step": 8750,
      "loss": 0.1582
    },
    {
      "step": 8755,
      "loss": 0.1701
    },
    {
      "step": 8760,
      "loss": 0.144
    },
    {
      "step": 8765,
      "loss": 0.1308
    },
    {
      "step": 8770,
      "loss": 0.1484
    },
    {
      "step": 8775,
      "loss": 0.152
    },
    {
      "step": 8780,
      "loss": 0.1488
    },
    {
      "step": 8785,
      "loss": 0.1432
    },
    {
      "step": 8790,
      "loss": 0.1559
    },
    {
      "step": 8795,
      "loss": 0.1354
    },
    {
      "step": 8800,
      "loss": 0.1375
    },
    {
      "step": 8805,
      "loss": 0.1458
    },
    {
      "step": 8810,
      "loss": 0.1403
    },
    {
      "step": 8815,
      "loss": 0.14
    },
    {
      "step": 8820,
      "loss": 0.1458
    },
    {
      "step": 8825,
      "loss": 0.1377
    },
    {
      "step": 8830,
      "loss": 0.1376
    },
    {
      "step": 8835,
      "loss": 0.1267
    },
    {
      "step": 8840,
      "loss": 0.1419
    },
    {
      "step": 8845,
      "loss": 0.1505
    },
    {
      "step": 8850,
      "loss": 0.1501
    },
    {
      "step": 8855,
      "loss": 0.1291
    },
    {
      "step": 8860,
      "loss": 0.1394
    },
    {
      "step": 8865,
      "loss": 0.1551
    },
    {
      "step": 8870,
      "loss": 0.1389
    },
    {
      "step": 8875,
      "loss": 0.1444
    },
    {
      "step": 8880,
      "loss": 0.1546
    },
    {
      "step": 8885,
      "loss": 0.1407
    },
    {
      "step": 8890,
      "loss": 0.1501
    },
    {
      "step": 8895,
      "loss": 0.1519
    },
    {
      "step": 8900,
      "loss": 0.1651
    },
    {
      "step": 8905,
      "loss": 0.1447
    },
    {
      "step": 8910,
      "loss": 0.1493
    },
    {
      "step": 8915,
      "loss": 0.1347
    },
    {
      "step": 8920,
      "loss": 0.1359
    },
    {
      "step": 8925,
      "loss": 0.1424
    },
    {
      "step": 8930,
      "loss": 0.1476
    },
    {
      "step": 8935,
      "loss": 0.1552
    },
    {
      "step": 8940,
      "loss": 0.1499
    },
    {
      "step": 8945,
      "loss": 0.1498
    },
    {
      "step": 8950,
      "loss": 0.1634
    },
    {
      "step": 8955,
      "loss": 0.1339
    },
    {
      "step": 8960,
      "loss": 0.1472
    },
    {
      "step": 8965,
      "loss": 0.1373
    },
    {
      "step": 8970,
      "loss": 0.1296
    },
    {
      "step": 8975,
      "loss": 0.1375
    },
    {
      "step": 8980,
      "loss": 0.1467
    },
    {
      "step": 8985,
      "loss": 0.1485
    },
    {
      "step": 8990,
      "loss": 0.1374
    },
    {
      "step": 8995,
      "loss": 0.1378
    },
    {
      "step": 9000,
      "loss": 0.1263
    },
    {
      "step": 9005,
      "loss": 0.1513
    },
    {
      "step": 9010,
      "loss": 0.1426
    },
    {
      "step": 9015,
      "loss": 0.1462
    },
    {
      "step": 9020,
      "loss": 0.1505
    },
    {
      "step": 9025,
      "loss": 0.1357
    },
    {
      "step": 9030,
      "loss": 0.1402
    },
    {
      "step": 9035,
      "loss": 0.1539
    },
    {
      "step": 9040,
      "loss": 0.1468
    },
    {
      "step": 9045,
      "loss": 0.1335
    },
    {
      "step": 9050,
      "loss": 0.131
    },
    {
      "step": 9055,
      "loss": 0.1468
    },
    {
      "step": 9060,
      "loss": 0.1409
    },
    {
      "step": 9065,
      "loss": 0.1429
    },
    {
      "step": 9070,
      "loss": 0.145
    },
    {
      "step": 9075,
      "loss": 0.1465
    },
    {
      "step": 9080,
      "loss": 0.1453
    },
    {
      "step": 9085,
      "loss": 0.1572
    },
    {
      "step": 9090,
      "loss": 0.1241
    },
    {
      "step": 9095,
      "loss": 0.1459
    },
    {
      "step": 9100,
      "loss": 0.1728
    },
    {
      "step": 9105,
      "loss": 0.1442
    },
    {
      "step": 9110,
      "loss": 0.1436
    },
    {
      "step": 9115,
      "loss": 0.1212
    },
    {
      "step": 9120,
      "loss": 0.1337
    },
    {
      "step": 9125,
      "loss": 0.1446
    },
    {
      "step": 9130,
      "loss": 0.1525
    },
    {
      "step": 9135,
      "loss": 0.1304
    },
    {
      "step": 9140,
      "loss": 0.148
    },
    {
      "step": 9145,
      "loss": 0.1394
    },
    {
      "step": 9150,
      "loss": 0.1406
    },
    {
      "step": 9155,
      "loss": 0.1388
    },
    {
      "step": 9160,
      "loss": 0.149
    },
    {
      "step": 9165,
      "loss": 0.1297
    },
    {
      "step": 9170,
      "loss": 0.1381
    },
    {
      "step": 9175,
      "loss": 0.1525
    },
    {
      "step": 9180,
      "loss": 0.1427
    },
    {
      "step": 9185,
      "loss": 0.1514
    },
    {
      "step": 9190,
      "loss": 0.1458
    },
    {
      "step": 9195,
      "loss": 0.1361
    },
    {
      "step": 9200,
      "loss": 0.1581
    },
    {
      "step": 9205,
      "loss": 0.1618
    },
    {
      "step": 9210,
      "loss": 0.1439
    },
    {
      "step": 9215,
      "loss": 0.1399
    },
    {
      "step": 9220,
      "loss": 0.1371
    },
    {
      "step": 9225,
      "loss": 0.1276
    },
    {
      "step": 9230,
      "loss": 0.1529
    },
    {
      "step": 9235,
      "loss": 0.1536
    },
    {
      "step": 9240,
      "loss": 0.1413
    },
    {
      "step": 9245,
      "loss": 0.1499
    },
    {
      "step": 9250,
      "loss": 0.1492
    },
    {
      "step": 9255,
      "loss": 0.1409
    },
    {
      "step": 9260,
      "loss": 0.157
    },
    {
      "step": 9265,
      "loss": 0.1432
    },
    {
      "step": 9270,
      "loss": 0.1633
    },
    {
      "step": 9275,
      "loss": 0.157
    },
    {
      "step": 9280,
      "loss": 0.1518
    },
    {
      "step": 9285,
      "loss": 0.133
    },
    {
      "step": 9290,
      "loss": 0.151
    },
    {
      "step": 9295,
      "loss": 0.152
    },
    {
      "step": 9300,
      "loss": 0.1392
    },
    {
      "step": 9305,
      "loss": 0.144
    },
    {
      "step": 9310,
      "loss": 0.1446
    },
    {
      "step": 9315,
      "loss": 0.1455
    },
    {
      "step": 9320,
      "loss": 0.1355
    },
    {
      "step": 9325,
      "loss": 0.1436
    },
    {
      "step": 9330,
      "loss": 0.1531
    },
    {
      "step": 9335,
      "loss": 0.1426
    },
    {
      "step": 9340,
      "loss": 0.1395
    },
    {
      "step": 9345,
      "loss": 0.1531
    },
    {
      "step": 9350,
      "loss": 0.1529
    },
    {
      "step": 9355,
      "loss": 0.1474
    },
    {
      "step": 9360,
      "loss": 0.1444
    },
    {
      "step": 9365,
      "loss": 0.1428
    },
    {
      "step": 9370,
      "loss": 0.1336
    },
    {
      "step": 9375,
      "loss": 0.1368
    },
    {
      "step": 9380,
      "loss": 0.1469
    },
    {
      "step": 9385,
      "loss": 0.1335
    },
    {
      "step": 9390,
      "loss": 0.1516
    },
    {
      "step": 9395,
      "loss": 0.144
    },
    {
      "step": 9400,
      "loss": 0.1422
    },
    {
      "step": 9405,
      "loss": 0.1348
    },
    {
      "step": 9410,
      "loss": 0.1386
    },
    {
      "step": 9415,
      "loss": 0.1403
    },
    {
      "step": 9420,
      "loss": 0.1532
    },
    {
      "step": 9425,
      "loss": 0.1427
    },
    {
      "step": 9430,
      "loss": 0.1359
    },
    {
      "step": 9435,
      "loss": 0.1351
    },
    {
      "step": 9440,
      "loss": 0.1472
    },
    {
      "step": 9445,
      "loss": 0.1507
    },
    {
      "step": 9450,
      "loss": 0.132
    },
    {
      "step": 9455,
      "loss": 0.1332
    },
    {
      "step": 9460,
      "loss": 0.1301
    },
    {
      "step": 9465,
      "loss": 0.1428
    },
    {
      "step": 9470,
      "loss": 0.1328
    },
    {
      "step": 9475,
      "loss": 0.1185
    },
    {
      "step": 9480,
      "loss": 0.1453
    },
    {
      "step": 9485,
      "loss": 0.1476
    },
    {
      "step": 9490,
      "loss": 0.1411
    },
    {
      "step": 9495,
      "loss": 0.1478
    },
    {
      "step": 9500,
      "loss": 0.152
    },
    {
      "step": 9505,
      "loss": 0.1247
    },
    {
      "step": 9510,
      "loss": 0.1321
    },
    {
      "step": 9515,
      "loss": 0.1468
    },
    {
      "step": 9520,
      "loss": 0.1326
    },
    {
      "step": 9525,
      "loss": 0.1464
    },
    {
      "step": 9530,
      "loss": 0.1386
    },
    {
      "step": 9535,
      "loss": 0.1397
    },
    {
      "step": 9540,
      "loss": 0.1449
    },
    {
      "step": 9545,
      "loss": 0.1191
    },
    {
      "step": 9550,
      "loss": 0.1372
    },
    {
      "step": 9555,
      "loss": 0.1433
    },
    {
      "step": 9560,
      "loss": 0.1414
    },
    {
      "step": 9565,
      "loss": 0.1419
    },
    {
      "step": 9570,
      "loss": 0.1429
    },
    {
      "step": 9575,
      "loss": 0.1433
    },
    {
      "step": 9580,
      "loss": 0.1423
    },
    {
      "step": 9585,
      "loss": 0.148
    },
    {
      "step": 9590,
      "loss": 0.1453
    },
    {
      "step": 9595,
      "loss": 0.1485
    },
    {
      "step": 9600,
      "loss": 0.1407
    },
    {
      "step": 9605,
      "loss": 0.1529
    },
    {
      "step": 9610,
      "loss": 0.1415
    },
    {
      "step": 9615,
      "loss": 0.1402
    },
    {
      "step": 9620,
      "loss": 0.143
    },
    {
      "step": 9625,
      "loss": 0.1562
    },
    {
      "step": 9630,
      "loss": 0.1331
    },
    {
      "step": 9635,
      "loss": 0.1344
    },
    {
      "step": 9640,
      "loss": 0.1439
    },
    {
      "step": 9645,
      "loss": 0.1409
    },
    {
      "step": 9650,
      "loss": 0.1439
    },
    {
      "step": 9655,
      "loss": 0.1304
    },
    {
      "step": 9660,
      "loss": 0.1297
    },
    {
      "step": 9665,
      "loss": 0.1206
    },
    {
      "step": 9670,
      "loss": 0.1307
    },
    {
      "step": 9675,
      "loss": 0.1452
    },
    {
      "step": 9680,
      "loss": 0.1527
    },
    {
      "step": 9685,
      "loss": 0.1399
    },
    {
      "step": 9690,
      "loss": 0.1474
    },
    {
      "step": 9695,
      "loss": 0.1539
    },
    {
      "step": 9700,
      "loss": 0.1491
    },
    {
      "step": 9705,
      "loss": 0.1246
    },
    {
      "step": 9710,
      "loss": 0.1212
    },
    {
      "step": 9715,
      "loss": 0.1483
    },
    {
      "step": 9720,
      "loss": 0.1348
    },
    {
      "step": 9725,
      "loss": 0.1405
    },
    {
      "step": 9730,
      "loss": 0.1404
    },
    {
      "step": 9735,
      "loss": 0.1481
    },
    {
      "step": 9740,
      "loss": 0.1498
    },
    {
      "step": 9745,
      "loss": 0.135
    },
    {
      "step": 9750,
      "loss": 0.1327
    },
    {
      "step": 9755,
      "loss": 0.1366
    },
    {
      "step": 9760,
      "loss": 0.1409
    },
    {
      "step": 9765,
      "loss": 0.1435
    },
    {
      "step": 9770,
      "loss": 0.1336
    },
    {
      "step": 9775,
      "loss": 0.1372
    },
    {
      "step": 9780,
      "loss": 0.1501
    },
    {
      "step": 9785,
      "loss": 0.1203
    },
    {
      "step": 9790,
      "loss": 0.137
    },
    {
      "step": 9795,
      "loss": 0.1369
    },
    {
      "step": 9800,
      "loss": 0.1328
    },
    {
      "step": 9805,
      "loss": 0.1439
    },
    {
      "step": 9810,
      "loss": 0.1289
    },
    {
      "step": 9815,
      "loss": 0.137
    },
    {
      "step": 9820,
      "loss": 0.1402
    },
    {
      "step": 9825,
      "loss": 0.1394
    },
    {
      "step": 9830,
      "loss": 0.1475
    },
    {
      "step": 9835,
      "loss": 0.1359
    },
    {
      "step": 9840,
      "loss": 0.1371
    },
    {
      "step": 9845,
      "loss": 0.1458
    },
    {
      "step": 9850,
      "loss": 0.147
    },
    {
      "step": 9855,
      "loss": 0.1248
    },
    {
      "step": 9860,
      "loss": 0.1386
    },
    {
      "step": 9865,
      "loss": 0.1456
    },
    {
      "step": 9870,
      "loss": 0.1323
    },
    {
      "step": 9875,
      "loss": 0.1645
    },
    {
      "step": 9880,
      "loss": 0.1401
    },
    {
      "step": 9885,
      "loss": 0.1347
    },
    {
      "step": 9890,
      "loss": 0.1388
    },
    {
      "step": 9895,
      "loss": 0.14
    },
    {
      "step": 9900,
      "loss": 0.1387
    },
    {
      "step": 9905,
      "loss": 0.1371
    },
    {
      "step": 9910,
      "loss": 0.1485
    },
    {
      "step": 9915,
      "loss": 0.1453
    },
    {
      "step": 9920,
      "loss": 0.1312
    },
    {
      "step": 9925,
      "loss": 0.1376
    },
    {
      "step": 9930,
      "loss": 0.1424
    },
    {
      "step": 9935,
      "loss": 0.1435
    },
    {
      "step": 9940,
      "loss": 0.1321
    },
    {
      "step": 9945,
      "loss": 0.1411
    },
    {
      "step": 9950,
      "loss": 0.1383
    },
    {
      "step": 9955,
      "loss": 0.1472
    },
    {
      "step": 9960,
      "loss": 0.1352
    },
    {
      "step": 9965,
      "loss": 0.1362
    },
    {
      "step": 9970,
      "loss": 0.1371
    },
    {
      "step": 9975,
      "loss": 0.1319
    },
    {
      "step": 9980,
      "loss": 0.1388
    },
    {
      "step": 9985,
      "loss": 0.1432
    },
    {
      "step": 9990,
      "loss": 0.1345
    },
    {
      "step": 9995,
      "loss": 0.1271
    },
    {
      "step": 10000,
      "loss": 0.1302
    },
    {
      "step": 10005,
      "loss": 0.1356
    },
    {
      "step": 10010,
      "loss": 0.1498
    },
    {
      "step": 10015,
      "loss": 0.1207
    },
    {
      "step": 10020,
      "loss": 0.1578
    },
    {
      "step": 10025,
      "loss": 0.1497
    },
    {
      "step": 10030,
      "loss": 0.1503
    },
    {
      "step": 10035,
      "loss": 0.1406
    },
    {
      "step": 10040,
      "loss": 0.126
    },
    {
      "step": 10045,
      "loss": 0.1253
    },
    {
      "step": 10050,
      "loss": 0.1144
    },
    {
      "step": 10055,
      "loss": 0.1362
    },
    {
      "step": 10060,
      "loss": 0.1464
    },
    {
      "step": 10065,
      "loss": 0.1431
    },
    {
      "step": 10070,
      "loss": 0.1393
    },
    {
      "step": 10075,
      "loss": 0.1505
    },
    {
      "step": 10080,
      "loss": 0.1559
    },
    {
      "step": 10085,
      "loss": 0.1536
    },
    {
      "step": 10090,
      "loss": 0.1351
    },
    {
      "step": 10095,
      "loss": 0.1334
    },
    {
      "step": 10100,
      "loss": 0.1375
    },
    {
      "step": 10105,
      "loss": 0.122
    },
    {
      "step": 10110,
      "loss": 0.1406
    },
    {
      "step": 10115,
      "loss": 0.1495
    },
    {
      "step": 10120,
      "loss": 0.1217
    },
    {
      "step": 10125,
      "loss": 0.1387
    },
    {
      "step": 10130,
      "loss": 0.1322
    },
    {
      "step": 10135,
      "loss": 0.1421
    },
    {
      "step": 10140,
      "loss": 0.1221
    },
    {
      "step": 10145,
      "loss": 0.1238
    },
    {
      "step": 10150,
      "loss": 0.1444
    },
    {
      "step": 10155,
      "loss": 0.125
    },
    {
      "step": 10160,
      "loss": 0.126
    },
    {
      "step": 10165,
      "loss": 0.1463
    },
    {
      "step": 10170,
      "loss": 0.1261
    },
    {
      "step": 10175,
      "loss": 0.1445
    },
    {
      "step": 10180,
      "loss": 0.1397
    },
    {
      "step": 10185,
      "loss": 0.1307
    },
    {
      "step": 10190,
      "loss": 0.1463
    },
    {
      "step": 10195,
      "loss": 0.1472
    },
    {
      "step": 10200,
      "loss": 0.1312
    },
    {
      "step": 10205,
      "loss": 0.1336
    },
    {
      "step": 10210,
      "loss": 0.1167
    },
    {
      "step": 10215,
      "loss": 0.1438
    },
    {
      "step": 10220,
      "loss": 0.1469
    },
    {
      "step": 10225,
      "loss": 0.1216
    },
    {
      "step": 10230,
      "loss": 0.1493
    },
    {
      "step": 10235,
      "loss": 0.1426
    },
    {
      "step": 10240,
      "loss": 0.1323
    },
    {
      "step": 10245,
      "loss": 0.1402
    },
    {
      "step": 10250,
      "loss": 0.1533
    },
    {
      "step": 10255,
      "loss": 0.1182
    },
    {
      "step": 10260,
      "loss": 0.1461
    },
    {
      "step": 10265,
      "loss": 0.1499
    },
    {
      "step": 10270,
      "loss": 0.1266
    },
    {
      "step": 10275,
      "loss": 0.1303
    },
    {
      "step": 10280,
      "loss": 0.1426
    },
    {
      "step": 10285,
      "loss": 0.1459
    },
    {
      "step": 10290,
      "loss": 0.1491
    },
    {
      "step": 10295,
      "loss": 0.1542
    },
    {
      "step": 10300,
      "loss": 0.1414
    },
    {
      "step": 10305,
      "loss": 0.1499
    },
    {
      "step": 10310,
      "loss": 0.1421
    },
    {
      "step": 10315,
      "loss": 0.154
    },
    {
      "step": 10320,
      "loss": 0.135
    },
    {
      "step": 10325,
      "loss": 0.1261
    },
    {
      "step": 10330,
      "loss": 0.1529
    },
    {
      "step": 10335,
      "loss": 0.1343
    },
    {
      "step": 10340,
      "loss": 0.1337
    },
    {
      "step": 10345,
      "loss": 0.1207
    },
    {
      "step": 10350,
      "loss": 0.1328
    },
    {
      "step": 10355,
      "loss": 0.1246
    },
    {
      "step": 10360,
      "loss": 0.1342
    },
    {
      "step": 10365,
      "loss": 0.1278
    },
    {
      "step": 10370,
      "loss": 0.139
    },
    {
      "step": 10375,
      "loss": 0.1274
    },
    {
      "step": 10380,
      "loss": 0.1305
    },
    {
      "step": 10385,
      "loss": 0.139
    },
    {
      "step": 10390,
      "loss": 0.1419
    },
    {
      "step": 10395,
      "loss": 0.146
    },
    {
      "step": 10400,
      "loss": 0.1355
    },
    {
      "step": 10405,
      "loss": 0.1299
    },
    {
      "step": 10410,
      "loss": 0.1537
    },
    {
      "step": 10415,
      "loss": 0.148
    },
    {
      "step": 10420,
      "loss": 0.1282
    },
    {
      "step": 10425,
      "loss": 0.1455
    },
    {
      "step": 10430,
      "loss": 0.1301
    },
    {
      "step": 10435,
      "loss": 0.1373
    },
    {
      "step": 10440,
      "loss": 0.1353
    },
    {
      "step": 10445,
      "loss": 0.1391
    },
    {
      "step": 10450,
      "loss": 0.1308
    },
    {
      "step": 10455,
      "loss": 0.1449
    },
    {
      "step": 10460,
      "loss": 0.1702
    },
    {
      "step": 10465,
      "loss": 0.1185
    },
    {
      "step": 10470,
      "loss": 0.1497
    },
    {
      "step": 10475,
      "loss": 0.134
    },
    {
      "step": 10480,
      "loss": 0.1388
    },
    {
      "step": 10485,
      "loss": 0.1477
    },
    {
      "step": 10490,
      "loss": 0.1372
    },
    {
      "step": 10495,
      "loss": 0.1498
    },
    {
      "step": 10500,
      "loss": 0.1405
    },
    {
      "step": 10505,
      "loss": 0.1425
    },
    {
      "step": 10510,
      "loss": 0.1373
    },
    {
      "step": 10515,
      "loss": 0.1369
    },
    {
      "step": 10520,
      "loss": 0.1438
    },
    {
      "step": 10525,
      "loss": 0.1186
    },
    {
      "step": 10530,
      "loss": 0.1391
    },
    {
      "step": 10535,
      "loss": 0.1453
    },
    {
      "step": 10540,
      "loss": 0.1475
    },
    {
      "step": 10545,
      "loss": 0.1418
    },
    {
      "step": 10550,
      "loss": 0.1468
    },
    {
      "step": 10555,
      "loss": 0.1365
    },
    {
      "step": 10560,
      "loss": 0.1322
    },
    {
      "step": 10565,
      "loss": 0.133
    },
    {
      "step": 10570,
      "loss": 0.122
    },
    {
      "step": 10575,
      "loss": 0.1433
    },
    {
      "step": 10580,
      "loss": 0.1445
    },
    {
      "step": 10585,
      "loss": 0.1338
    },
    {
      "step": 10590,
      "loss": 0.1407
    },
    {
      "step": 10595,
      "loss": 0.1247
    },
    {
      "step": 10600,
      "loss": 0.1588
    },
    {
      "step": 10605,
      "loss": 0.1372
    },
    {
      "step": 10610,
      "loss": 0.1428
    },
    {
      "step": 10615,
      "loss": 0.1391
    },
    {
      "step": 10620,
      "loss": 0.1333
    },
    {
      "step": 10625,
      "loss": 0.1255
    },
    {
      "step": 10630,
      "loss": 0.1377
    },
    {
      "step": 10635,
      "loss": 0.1449
    },
    {
      "step": 10640,
      "loss": 0.139
    },
    {
      "step": 10645,
      "loss": 0.1314
    },
    {
      "step": 10650,
      "loss": 0.1355
    },
    {
      "step": 10655,
      "loss": 0.1252
    },
    {
      "step": 10660,
      "loss": 0.1239
    },
    {
      "step": 10665,
      "loss": 0.1378
    },
    {
      "step": 10670,
      "loss": 0.1428
    },
    {
      "step": 10675,
      "loss": 0.14
    },
    {
      "step": 10680,
      "loss": 0.1341
    },
    {
      "step": 10685,
      "loss": 0.1281
    },
    {
      "step": 10690,
      "loss": 0.1468
    },
    {
      "step": 10695,
      "loss": 0.1294
    },
    {
      "step": 10700,
      "loss": 0.1392
    },
    {
      "step": 10705,
      "loss": 0.1295
    },
    {
      "step": 10710,
      "loss": 0.1443
    },
    {
      "step": 10715,
      "loss": 0.1382
    },
    {
      "step": 10720,
      "loss": 0.1472
    },
    {
      "step": 10725,
      "loss": 0.1484
    },
    {
      "step": 10730,
      "loss": 0.1411
    },
    {
      "step": 10735,
      "loss": 0.1377
    },
    {
      "step": 10740,
      "loss": 0.1332
    },
    {
      "step": 10745,
      "loss": 0.1486
    },
    {
      "step": 10750,
      "loss": 0.1447
    },
    {
      "step": 10755,
      "loss": 0.1419
    },
    {
      "step": 10760,
      "loss": 0.1242
    },
    {
      "step": 10765,
      "loss": 0.1342
    },
    {
      "step": 10770,
      "loss": 0.1322
    },
    {
      "step": 10775,
      "loss": 0.1442
    },
    {
      "step": 10780,
      "loss": 0.1305
    },
    {
      "step": 10785,
      "loss": 0.1456
    },
    {
      "step": 10790,
      "loss": 0.138
    },
    {
      "step": 10795,
      "loss": 0.1221
    },
    {
      "step": 10800,
      "loss": 0.1438
    },
    {
      "step": 10805,
      "loss": 0.1492
    },
    {
      "step": 10810,
      "loss": 0.1486
    },
    {
      "step": 10815,
      "loss": 0.1495
    },
    {
      "step": 10820,
      "loss": 0.1314
    },
    {
      "step": 10825,
      "loss": 0.1332
    },
    {
      "step": 10830,
      "loss": 0.1515
    },
    {
      "step": 10835,
      "loss": 0.129
    },
    {
      "step": 10840,
      "loss": 0.1341
    },
    {
      "step": 10845,
      "loss": 0.1438
    },
    {
      "step": 10850,
      "loss": 0.1381
    },
    {
      "step": 10855,
      "loss": 0.1596
    },
    {
      "step": 10860,
      "loss": 0.1497
    },
    {
      "step": 10865,
      "loss": 0.158
    },
    {
      "step": 10870,
      "loss": 0.1514
    },
    {
      "step": 10875,
      "loss": 0.1427
    },
    {
      "step": 10880,
      "loss": 0.1464
    },
    {
      "step": 10885,
      "loss": 0.1349
    },
    {
      "step": 10890,
      "loss": 0.1368
    },
    {
      "step": 10895,
      "loss": 0.1575
    },
    {
      "step": 10900,
      "loss": 0.1412
    },
    {
      "step": 10905,
      "loss": 0.1247
    },
    {
      "step": 10910,
      "loss": 0.1236
    },
    {
      "step": 10915,
      "loss": 0.1521
    },
    {
      "step": 10920,
      "loss": 0.1246
    },
    {
      "step": 10925,
      "loss": 0.1473
    },
    {
      "step": 10930,
      "loss": 0.1409
    },
    {
      "step": 10935,
      "loss": 0.1489
    },
    {
      "step": 10940,
      "loss": 0.1431
    },
    {
      "step": 10945,
      "loss": 0.1548
    },
    {
      "step": 10950,
      "loss": 0.1483
    },
    {
      "step": 10955,
      "loss": 0.1449
    },
    {
      "step": 10960,
      "loss": 0.1412
    },
    {
      "step": 10965,
      "loss": 0.1338
    },
    {
      "step": 10970,
      "loss": 0.1316
    },
    {
      "step": 10975,
      "loss": 0.1393
    },
    {
      "step": 10980,
      "loss": 0.1455
    },
    {
      "step": 10985,
      "loss": 0.1469
    },
    {
      "step": 10990,
      "loss": 0.1346
    }
  ],
  "eval_losses": [
    {
      "step": 5496,
      "eval_loss": 0.8224
    },
    {
      "step": 10992,
      "eval_loss": 0.9746
    }
  ],
  "decode_history": [
    {
      "epoch": 1.0,
      "step": 5496,
      "eval_loss": 0.8223534822463989,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 32,
        "mae_log": 0.5041231686581205,
        "rmse_log": 0.6834608700079152,
        "mae_s": 3.586956249999999,
        "rmse_s": 5.547898160452298,
        "spearman_rho": 0.052936525839315685,
        "mean_actual_s": 6.12451875,
        "mean_pred_s": 5.584375
      },
      "samples": [
        {
          "output": "dwell_seconds: 3.2",
          "gold_s": 3.621,
          "pred_s": 3.2
        },
        {
          "output": "dwell_seconds: 4.2",
          "gold_s": 2.878,
          "pred_s": 4.2
        },
        {
          "output": "dwell_seconds: 3.0",
          "gold_s": 3.851,
          "pred_s": 3.0
        }
      ]
    },
    {
      "epoch": 2.0,
      "step": 10992,
      "eval_loss": 0.9746091365814209,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 32,
        "mae_log": 0.5257592568091476,
        "rmse_log": 0.7120057139475431,
        "mae_s": 3.7361437499999997,
        "rmse_s": 5.7269171679665485,
        "spearman_rho": -0.014147953492462584,
        "mean_actual_s": 6.12451875,
        "mean_pred_s": 5.528125
      },
      "samples": [
        {
          "output": "dwell_seconds: 3.0",
          "gold_s": 3.621,
          "pred_s": 3.0
        },
        {
          "output": "dwell_seconds: 4.1",
          "gold_s": 2.878,
          "pred_s": 4.1
        },
        {
          "output": "dwell_seconds: 3.2",
          "gold_s": 3.851,
          "pred_s": 3.2
        }
      ]
    }
  ],
  "vram": {
    "device": "NVIDIA A100-SXM4-40GB",
    "total_gb": 39.49,
    "peak_allocated_gb": 6.4,
    "peak_reserved_gb": 6.59
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
      "run_name": "qwen3vl4b-qlora-pathA-task"
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
      "include_task_title": true
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
