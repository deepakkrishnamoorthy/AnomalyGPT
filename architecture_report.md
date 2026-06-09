# Architecture Report

Status: updated after inspecting `WalkGPT-main`.

The current WalkGPT codebase is a grounded LLaVA-style causal LM with SAM-based mask decoding. For AnomalyGPT, the safest path is to preserve this grounding pipeline and add temporal/physics features around it instead of replacing the architecture.

## Backbone

Primary files:

- `WalkGPT-main/model/walkgpt.py`
- `WalkGPT-main/model/llava_walkgpt/model/language_model/llava_llama.py`
- `WalkGPT-main/model/llava_walkgpt/model/llava_arch.py`
- `WalkGPT-main/model/segment_anything/build_sam.py`

Main components:

- Language model: `LlavaLlamaForCausalLM` / `LlavaLlamaModel`.
- Image encoder for LLM tokens: SAM ViT-H image encoder through `build_sam_vit_h`.
- Optional LLaVA CLIP vision tower is still initialized for compatibility, but this branch projects SAM tokens into the language model path.
- Visual token projector: `MultiScaleQFormerProjector`, exposed on the model as `out_mm_projector`.
- Token schema currently includes `[SEG]`, `[p]`, `[/p]`, `[distance]`, `[/distance]`, `[assessment]`, and `[/assessment]`.

Current single-frame flow:

```text
image
  -> SAM image encoder
  -> SAM grid tokens
  -> MultiScaleQFormerProjector
  -> LLaVA/LLaMA language model
  -> hidden state at [SEG]
  -> CalibratedTextProjector
  -> SAM prompt encoder + mask decoder
  -> segmentation mask
```

AnomalyGPT adaptation:

- Replace image samples with short clips, initially by sampling representative frames and later by adding temporal pooling.
- Add `<physics>` or `[physics]` tokens alongside `[assessment]` and `[SEG]`.
- Inject trajectory features before or beside the language hidden states used for anomaly explanations.

## Segmentation Branch

Primary file:

- `WalkGPT-main/model/walkgpt.py`

Key modules:

- `self.model.visual_model.prompt_encoder`
- `self.model.visual_model.mask_decoder`
- `self.model.visual_model.postprocess_masks`
- `self.model.text_hidden_fcs[0]`

The model finds `[SEG]` tokens in `input_ids`, projects their final hidden states into SAM's prompt embedding space, and uses those embeddings as text prompts for SAM mask decoding.

Relevant tensor flow:

```text
output_hidden_states[-1]                  # language hidden states
  -> CalibratedTextProjector              # hidden_size -> 256
  -> pred_embeddings at [SEG] positions   # [num_seg_tokens, 256]
  -> SAM prompt_encoder(text_embeds=...)
  -> SAM mask_decoder(...)
  -> pred_masks
```

For VAD:

- `[SEG]` should correspond to anomalous object/region masks.
- Normal/background masks should remain implicit unless a contrastive normal-region objective is added.
- Weak masks can come from anomaly boxes, tracked objects, SAM2/Grounded SAM proposals, or pixel masks where available.

## MSQP

Implementation:

- `WalkGPT-main/utils/utils_walkgpt.py`
- Class: `MultiScaleQFormerProjector`

Purpose:

- Converts dense SAM grid tokens into a compact set of language-model visual tokens.
- Uses learned query groups across multiple spatial scales.

Input:

```text
sam_feats: [B, L, 256]
```

where `L` is a square grid length from SAM image embeddings.

Internal flow:

```text
SAM tokens
  -> Linear(256 -> 1024)
  -> x1 grid tokens
  -> x2 pooled tokens
  -> x4 pooled tokens
  -> global token
  -> segmentation-aware gate
  -> cross-attention with learned query groups
  -> concat 12 + 8 + 8 + 4 queries
  -> optional padding to 6x6
  -> Linear(1024 -> LLaMA hidden size)
```

Output:

```text
[B, 36, llama_hidden_dim]
```

The comment says 32 tokens, but `target_square_side=6` pads the 32 learned query outputs to 36 tokens in the current WalkGPT construction.

VAD use:

- Reuse as the frame-level visual grounding projector.
- Add temporal aggregation either before MSQP over frame SAM tokens or after MSQP over per-frame visual tokens.

## CTP

Implementation:

- `WalkGPT-main/utils/utils_walkgpt.py`
- Class: `CalibratedTextProjector`

Purpose:

- Maps LLM hidden states into the 256-dimensional SAM prompt embedding space.
- Adds a learned text type vector, L2 normalization, and learned temperature scaling.

Input:

```text
LLM hidden states: [rows, sequence_length, hidden_size]
```

Output:

```text
segmentation-aware text embeddings: [rows, sequence_length, 256]
```

Only hidden states at `[SEG]` token positions are selected for mask decoding and region alignment.

VAD use:

- Keep CTP for anomaly localization.
- Add a sibling physics projector for `<physics>` hidden states or fuse physics embeddings into `[SEG]`/assessment token hidden states.

## Region Alignment

Implementation:

- `WalkGPT-main/utils/utils_walkgpt.py`
- Function: `infonce_loss`
- Called in `WalkGPT-main/model/walkgpt.py`

Purpose:

- Aligns text-derived segmentation embeddings with row-aligned SAM grid tokens.

Input:

```text
pred_embeddings: [M, 256]
sam_tokens_256:  [rows, N, 256]
seg_row_ids:     [M]
```

Flow:

```text
[SEG] embedding
  -> TinyCrossAttn over same-row SAM tokens
  -> positive visual feature
  -> InfoNCE against all other row/grid tokens
```

VAD use:

- Keep this as the visual grounding loss.
- Add physics alignment loss between anomaly explanation hidden states and physics embeddings.

## Losses

Current losses in `walkgptForCausalLM.model_forward`:

- Language modeling loss: `model_output.loss`, weighted by `ce_loss_weight`.
- Mask BCE loss: `sigmoid_ce_loss`, weighted by `bce_loss_weight`.
- Mask Dice loss: `dice_loss`, weighted by `dice_loss_weight`.
- Region alignment loss: `infonce_loss`, currently multiplied by `0.2` inside `walkgpt.py`.

Important note:

- `nce_loss_weight` exists in argument parsing, but the current model code hardcodes `nce_loss = 0.2 * loss_nce`. This should be cleaned up during AnomalyGPT training integration.

Planned VAD losses:

- Explanation consistency loss: text should agree with measured velocity, acceleration, direction deviation, and flow deviation.
- Physics alignment loss: align physics embeddings with anomaly explanation or `[SEG]` embeddings.
- Detection losses: frame-level AUC-compatible anomaly scores when frame labels are available.

## Dataset Pipeline

Primary files:

- `WalkGPT-main/utils/dataset.py`
- `WalkGPT-main/utils/PAVE_dataset.py`
- `WalkGPT-main/train_walkgpt.py`
- `WalkGPT-main/evaluation_walkgpt.py`

PAVE samples emit:

```text
image_path
image
image_clip
conversations
masks
label
resize
clip_resize
questions
sampled_classes
```

The training collator expands conversations, tokenizes prompts, pads image/text batches, creates labels with user/instruction tokens ignored, and tracks `offset` so each image can map to multiple text rows.

VAD dataset should mirror this interface at first:

- Represent one clip as a sample.
- Use one or more selected keyframes for compatibility with the existing SAM path.
- Return anomaly masks in the same `masks` slot.
- Put anomaly questions and structured answers in `conversations`.
- Add `tracks` and `physics` metadata after the initial compatibility pass.

## Immediate Adaptation Plan

1. Add `AnomalyGPTDataset` modeled after `PAVEDataset`.
2. Register `AnomalyGPT` in `HybridDataset` and validation dataset selection.
3. Replace `[distance]` registration with `[physics]` / `[/physics]` in AnomalyGPT train/eval paths.
4. Keep `[assessment]` and `[SEG]` unchanged for now.
5. Add physics feature extraction from `tracks.json`.
6. Add a light physics projector/fusion module after the dataset path works.
7. Extend evaluation with frame AUC, pixel AUC, IoU/Dice, and text/physics consistency metrics.
