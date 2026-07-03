"""
totvlm/model.py
===============
The two models of this project:

1. VLM (primary, Path A): Qwen3-VL-4B-Instruct via Unsloth FastVisionModel,
   4-bit QLoRA — vision layers FROZEN, LoRA on language attention + MLP
   (CLAUDE.md backbone rules). The model is SFT'd to EMIT the dwell as text
   (`dwell_seconds: X.X`); there are no regression heads.
   `load_vlm()` returns (model, processor) ready for TRL's SFTTrainer with
   UnslothVisionDataCollator — see totvlm/train.py.

2. No-image LightGBM baseline: the bar the VLM must beat.

All knobs come from configs/*.yaml — nothing tunable lives here.
"""
from __future__ import annotations

# ── VLM: 4-bit Qwen3-VL + LoRA (Unsloth) ──────────────────────────────────────

def load_vlm(cfg: dict):
    """
    Load the 4-bit backbone and wrap it with LoRA adapters.

    Args:
        cfg: full configs/vlm.yaml mapping (uses cfg["model"] + cfg["seed"]).

    Returns:
        (model, processor) — processor doubles as the tokenizer for
        SFTTrainer / UnslothVisionDataCollator.
    """
    # Lazy import: unsloth is CUDA-only (installed via the `vlm` extra).
    # Keeping it out of module scope lets train_baseline.py import the
    # LightGBM helpers below on any machine.
    import torch
    from unsloth import FastVisionModel

    # TF32 matmuls: faster on GPUs that support them, no-op elsewhere,
    # negligible precision cost.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    m = cfg["model"]
    model, processor = FastVisionModel.from_pretrained(
        m["checkpoint"],
        load_in_4bit=m["load_in_4bit"],
        max_seq_length=m["max_seq_length"],
        # "unsloth" = Unsloth's offloaded gradient checkpointing (fits long
        # vision sequences in less VRAM; ~30% slower than off).
        use_gradient_checkpointing="unsloth" if m["gradient_checkpointing"]
        else False,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=m["finetune_vision_layers"],   # False: frozen
        finetune_language_layers=True,
        finetune_attention_modules=True,                      # attn + MLP
        finetune_mlp_modules=True,
        r=m["lora_r"],
        lora_alpha=m["lora_alpha"],
        lora_dropout=m["lora_dropout"],
        bias="none",
        random_state=cfg["seed"],
    )
    return model, processor


def load_vlm_for_inference(
    checkpoint: str, *, max_seq_length: int, load_in_4bit: bool = True
):
    """
    Load a trained adapter dir (artifacts/vlm_ckpt/final) — or a base model
    id — in inference mode. FastVisionModel detects the LoRA adapter_config
    and loads base + adapters automatically.
    """
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        checkpoint,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
    )
    FastVisionModel.for_inference(model)
    return model, processor


def predict_dwell_batch(
    model,
    processor,
    examples: list[dict],
    *,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    """
    Batched greedy decoding over chat examples — build_vlm_examples output
    (its gold assistant turn is dropped here) or build_inference_examples
    output (already prompt-only). Returns one raw output string per example,
    in order — parsing/fallback is the caller's job so failures stay visible.

    Left padding: with batched generate, prompts are padded to a common
    length; padding on the left keeps every prompt's end aligned so slicing
    off the prompt tokens is uniform across the batch.
    """
    import torch

    tokenizer = getattr(processor, "tokenizer", processor)
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    try:
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            prompts = [
                ex["messages"][:-1]
                if ex["messages"][-1]["role"] == "assistant"
                else ex["messages"]
                for ex in chunk
            ]
            inputs = processor.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,     # greedy — deterministic eval
                )
            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            outputs += [
                t.strip() for t in processor.batch_decode(
                    new_tokens, skip_special_tokens=True
                )
            ]
            done = start + len(chunk)
            if done % (batch_size * 10) < batch_size or done == len(examples):
                print(f"  decoded {done}/{len(examples)}", flush=True)
    finally:
        tokenizer.padding_side = prev_side
    return outputs


# ── No-image LightGBM baseline ────────────────────────────────────────────────
# The bar the VLM must beat: interpretable axTree/geometry features → y
# (log1p winsorized dwell). Trained on TRAIN, early-stopped on VAL l1,
# evaluated once on TEST through totvlm.scoring. All knobs come from
# configs/baseline.yaml — nothing tunable lives here.

def train_lgbm_baseline(
    x_train,
    y_train,
    x_val,
    y_val,
    params: dict,
    early_stopping_rounds: int,
):
    """Fit LightGBM (native API — no scikit-learn dependency) with early
    stopping on the val split. Returns the Booster; its predict() uses the
    best iteration automatically."""
    import lightgbm as lgb

    params = dict(params)
    num_rounds = params.pop("n_estimators")
    params["metric"] = "l1"   # MAE in log space — matches the primary metric

    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return booster


def lgbm_feature_importances(booster, feature_names: list[str]) -> list[dict]:
    """Gain importances, sorted descending, JSON-friendly."""
    gains = booster.feature_importance(importance_type="gain")
    pairs = sorted(zip(feature_names, gains), key=lambda p: -p[1])
    total = float(sum(gains)) or 1.0
    return [
        {"feature": f, "gain": round(float(g), 1),
         "pct": round(100 * float(g) / total, 2)}
        for f, g in pairs
    ]
