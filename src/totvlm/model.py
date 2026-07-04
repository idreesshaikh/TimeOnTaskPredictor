"""The two models: Qwen3-VL 4-bit QLoRA (vision frozen, SFT'd to emit
`dwell_seconds: X.X` as text — no regression head) and the no-image LightGBM
baseline. All knobs come from configs/*.yaml."""
from __future__ import annotations


def load_vlm(cfg: dict):
    """Load the 4-bit backbone + LoRA adapters (configs/vlm*.yaml). Returns
    (model, processor); processor doubles as the tokenizer for SFTTrainer."""
    # lazy import: unsloth is CUDA-only, and this module's LightGBM helpers
    # must stay importable on any machine
    import torch
    from unsloth import FastVisionModel

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    m = cfg["model"]
    model, processor = FastVisionModel.from_pretrained(
        m["checkpoint"],
        load_in_4bit=m["load_in_4bit"],
        max_seq_length=m["max_seq_length"],
        use_gradient_checkpointing="unsloth" if m["gradient_checkpointing"]
        else False,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=m["finetune_vision_layers"],
        finetune_language_layers=True,
        finetune_attention_modules=True,
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
    """Load a trained adapter dir — or a base model id (zero-shot) — in
    inference mode."""
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
    """Batched greedy decoding over chat examples (gold assistant turns are
    dropped). Returns raw output strings in order — parsing is the caller's
    job so failures stay visible. Left padding keeps prompt ends aligned so
    slicing off prompt tokens is uniform across the batch."""
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
                    do_sample=False,
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


def train_lgbm_baseline(
    x_train,
    y_train,
    x_val,
    y_val,
    params: dict,
    early_stopping_rounds: int,
    extra_callbacks: list | None = None,
):
    """Fit LightGBM with early stopping on val. The Booster's predict() uses
    the best iteration automatically."""
    import lightgbm as lgb

    params = dict(params)
    num_rounds = params.pop("n_estimators")
    params["metric"] = "l1"   # MAE in log space — the primary metric

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
            *(extra_callbacks or []),
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
