# Architecture Report

Status: scaffold created before the WalkGPT codebase is present in this folder.

The target adaptation keeps WalkGPT's grounded LVLM architecture and replaces accessibility reasoning with anomaly reasoning over short video clips.

## Backbone

To document after WalkGPT is added:

- Vision encoder
- Language model
- Projector layers
- Video feature adaptation path for 16-32 frame clips

## Segmentation Branch

To document after WalkGPT is added:

- Segmentation head
- Mask decoder
- Alignment between anomaly text tokens and predicted masks

## MSQP

To document after WalkGPT is added:

- Implementation file
- Input tensor shapes
- Output tensor shapes
- Role in grounded visual-language querying

## CTP

To document after WalkGPT is added:

- Language embedding flow
- Segmentation alignment flow
- Interaction with special tokens such as `<SEG>`

## Losses

To document after WalkGPT is added:

- Language modeling loss
- Segmentation loss
- Region alignment loss
- New physics alignment and explanation consistency losses
