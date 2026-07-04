# EarthFormer -> Perceiver IO Readout Report

## Scope

This stage leaves the migrated EarthFormer backbone unchanged. The official
EarthFormer path still runs through the initial encoder, positional embeddings,
Cuboid Attention encoder, decoder, skip paths, and final decoder exactly as
before.

The only replacement is the downstream use of `dec_final_proj`. For the CSI
readout path, the wrapper calls `forward_latent()` and consumes the tensor
immediately before the official projection head:

```text
pre_head_latent: (B,T,H,W,16)
```

The official `dec_final_proj` module remains present in the migrated backbone
for checkpoint compatibility and for the original frame-forecasting verifier.

## Tensor Flow

```text
SEVIRI images
  (B,T_in,7,200,200)
    |
    v
Official EarthFormer backbone
  pretrained SEVIR weights, unchanged
    |
    v
Pre-head latent
  (B,T,H,W,16)
    |
    v
Flatten spatial grid
  (B,T,HW,16)
    |
    v
Learnable output queries
  (B,T,D)
    |
    v
Perceiver IO output-query cross attention
  independent batch item per forecast timestep: (B*T,1,D) attends to (B*T,HW,16)
    |
    v
Output embeddings
  (B,T,D)
    |
    v
Lightweight regression
  Linear -> GELU -> Linear
    |
    v
CSI-shaped sequence
  (B,T)
```

## Perceiver IO Compatibility

The readout follows the Perceiver IO decoder principle: output queries retrieve
task-specific outputs by cross-attending to encoded representations. This
implementation intentionally omits all Perceiver components that would duplicate
EarthFormer's role:

- no Perceiver encoder
- no latent bottleneck
- no Fourier positional encoding
- no iterative latent attention
- no temporal attention in the readout
- no auxiliary feature conditioning

Each query represents one forecast hour. For timestep `t`, query `q_t` attends
only to the spatial tokens from EarthFormer latent timestep `t`. The reshape to
`(B*T, HW, 16)` makes timesteps independent during readout and prevents temporal
mixing after EarthFormer.

## Parameters

With the default config:

```text
query_dimension = 64
num_output_queries = 12
num_attention_heads = 4
readout_dropout = 0.1
regression_hidden_dim = 32
```

The verifier reports:

```text
EarthFormer parameters: 8,652,525
Perceiver readout parameters: 13,537
```

This preserves transfer learning: the pretrained EarthFormer parameters are
separate from the newly initialized readout parameters and can be frozen with
`--freeze-earthformer`.

## Verification Command

From the directory containing `EarthFormer/`:

```bash
python -m EarthFormer.scripts.verify_perceiver_readout \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu
```

Successful validation should report:

```text
pre_head_latent:    [1, 12, 200, 200, 16]
flattened_tokens:  [1, 12, 40000, 16]
queries:           [1, 12, 64]
attention_output:  [1, 12, 64]
regression_output: [1, 12]
prediction:        [1, 12]
earthformer_grad_ok: true
readout_grad_ok: true
checkpoint_roundtrip.strict: true
```

## Diagram

The thesis-style architecture figure is available in both editable Mermaid and
SVG form:

```text
figures/earthformer_perceiver_readout.mmd
figures/earthformer_perceiver_readout.svg
```
