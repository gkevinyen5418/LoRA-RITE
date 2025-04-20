# LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization

This is a pytorch reimplementation of the original LoRA-RITE in Jax.

## Usage

Please copy `lora_rite.py` to your directory or install it as a module.

Then you can do the following to create a normal pytorch optimizer object.

```
from lora_rite import LoRARite

lora_params = [p for n, p in model.named_parameters() if "lora" in n]
optimizer = LoRARite(lora_params, lr=learning_rate, betas=(0.9,0.999))
```

Here we assume the lora parameters will be in an alternating order `lora_a_1, lora_b_1, lora_a_2, lora_b_2, ...` as in the huggingface peft LoRA implementation.
In the rare case where this assumption is not satisfied, one can manually reorder it so that the assumption is met.

## Commonsense Reasoning Evaluation

This setting is significantly different from what is used in the paper due to the potentially high amount of effort needed to align the environments of pytorch and JAX.
We adopt the recipe from the LLM-adapter paper, where the datasets are highly overlapped with our original experiments.

Gemma-2B Result

|Optimizer| BOOLQ | PIQA | SIQA | HellaSwag | Winogrande | ARC-E | ARC-C | OBQA | Average |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | 
| LoRARite | 62.91 | 74.86 | 67.50 | 69.30 | 62.04 | 78.45 | 62.29 | 68.80 | 68.27 |
| Adam |  62.20 | 75.46 | 65.35 | 67.38 | 55.80 |76.60 |58.70| 68.00| 66.19|


