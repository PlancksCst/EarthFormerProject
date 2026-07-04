# EarthFormer Migration Architectural Report

## Scope

This migration abandons the DualET forecasting architecture and uses the official
EarthFormer implementation as the backbone. The new code does not import DualET
model modules, does not add a CSI regression head, and does not alter Cuboid
Attention, encoder blocks, decoder blocks, skip/memory wiring, or the official
prediction projection.

The implemented compatibility layer is intentionally thin:

- Official source: `Codes/prep+models/earthformer_src/src/earthformer`
- New migration wrapper: `Codes/prep+models/earthformer_migration`
- Official checkpoint: `earthformer_sevir.pt`
- Local checkpoint path:
  `Codes/prep+models/earthformer_src/pretrained_checkpoints/earthformer_sevir.pt`

## Sources Studied

- Paper: Earthformer: Exploring Space-Time Transformers for Earth System
  Forecasting, arXiv:2207.05833.
- Official repository: `amazon-science/earth-forecasting-transformer`.
- Official SEVIR config:
  `Codes/prep+models/earthformer_src/scripts/cuboid_transformer/sevir/earthformer_sevir_v1.yaml`.
- Official model implementation:
  `Codes/prep+models/earthformer_src/src/earthformer/cuboid_transformer/cuboid_transformer.py`.
- Official checkpoint utility:
  `Codes/prep+models/earthformer_src/src/earthformer/utils/checkpoint.py`.

## Official EarthFormer SEVIR Configuration

The official SEVIR pretrained model is a non-autoregressive encoder-decoder
Cuboid Transformer for sequence-to-sequence forecasting.

Official SEVIR tensor contract:

- Input: `(B, 13, 384, 384, 1)`
- Target/output: `(B, 12, 384, 384, 1)`
- Layout: `NTHWC`
- Base units: `128`
- Encoder depth: `[1, 1]`
- Decoder depth: `[1, 1]`
- Attention pattern: axial Cuboid self-attention
- Cross-attention pattern: `cross_1x1`
- Initial downsample type: `stack_conv`
- Initial downsample scales: `[3, 2, 2]`
- Initial stack dims: `[16, 64, 128]`
- Encoder hierarchy downsample: `2`
- Global vectors: `8`
- Positional embedding: learned `t+h+w`
- Relative position bias: enabled
- Decoder input initialization: zero query passed through decoder positional
  embedding and `z_proj`

## Official Data Flow

For SEVIR, the official model receives past VIL frames and forecasts future VIL
frames. The code path is in `CuboidTransformerModel.forward()`:

1. `x = self.initial_encoder(x)`
2. `x = self.enc_pos_embed(x)`
3. `mem_l, mem_global_vector_l = self.encoder(x, init_global_vectors)`
4. `initial_z = self.get_initial_z(final_mem=mem_l[-1], T_out=T_out)`
5. `dec_out = self.decoder(initial_z, mem_l, mem_global_vector_l)`
6. `dec_out = self.final_decoder(dec_out)`
7. `out = self.dec_final_proj(dec_out)`

The official prediction head begins at `self.dec_final_proj`. The tensor
immediately before this projection is the correct future attachment point for a
later CSI/GHI head investigation.

## Patch Embedding / Initial Encoder

EarthFormer does not use a ViT-style flat patch embedding in the SEVIR config.
The official SEVIR model uses `InitialStackPatchMergingEncoder`:

- Each stage applies several 2D convolutions independently per time step.
- Each stage then applies `PatchMerging3D`.
- The temporal dimension is preserved because downsampling is `(1, s, s)`.
- Spatial size uses padding when not divisible by the downsample scale.

Official SEVIR initial encoder shapes:

| Stage | Shape |
| --- | --- |
| Input | `(B, 13, 384, 384, 1)` |
| Stack stage 0, scale 3 | `(B, 13, 128, 128, 16)` |
| Stack stage 1, scale 2 | `(B, 13, 64, 64, 64)` |
| Stack stage 2, scale 2 | `(B, 13, 32, 32, 128)` |

Adapted SEVIRI initial encoder shapes:

| Stage | Shape |
| --- | --- |
| Input | `(B, 13, 200, 200, 7)` |
| Stack stage 0, scale 3 | `(B, 13, 67, 67, 16)` |
| Stack stage 1, scale 2 | `(B, 13, 34, 34, 64)` |
| Stack stage 2, scale 2 | `(B, 13, 17, 17, 128)` |

Only the first convolution changes shape because the input has 7 SEVIRI
channels instead of one SEVIR VIL channel.

## Cuboid Attention

The SEVIR pretrained config uses axial Cuboid self-attention. For a tensor
`(T, H, W, C)`, the registered `axial` pattern decomposes attention into:

- temporal cuboids: `(T, 1, 1)`
- height-axis cuboids: `(1, H, 1)`
- width-axis cuboids: `(1, 1, W)`

This keeps joint spatiotemporal modeling inside every Transformer stage without
separating the model into independent spatial and temporal networks.

The decoder cross-attention pattern is `cross_1x1`, which uses local spatial
cross-attention between decoder states and encoder memories at each hierarchy.

## Encoder Outputs

Official SEVIR encoder memory shapes:

| Memory | Shape |
| --- | --- |
| `mem_l[0]` | `(B, 13, 32, 32, 128)` |
| `mem_l[1]` | `(B, 13, 16, 16, 256)` |
| global `0` | `(B, 8, 128)` |
| global `1` | `(B, 8, 256)` |

Adapted SEVIRI encoder memory shapes:

| Memory | Shape |
| --- | --- |
| `mem_l[0]` | `(B, 13, 17, 17, 128)` |
| `mem_l[1]` | `(B, 13, 9, 9, 256)` |
| global `0` | `(B, 8, 128)` |
| global `1` | `(B, 8, 256)` |

## Decoder Outputs

The decoder is top-down over the encoder memories. With
`dec_use_first_self_attn=false`, the top decoder block first cross-attends to
the top encoder memory.

Adapted SEVIRI decoder flow:

| Stage | Shape |
| --- | --- |
| Initial decoder query | `(B, 12, 9, 9, 256)` |
| After Cuboid decoder | `(B, 12, 17, 17, 128)` |
| Final upsampling stage 0 | `(B, 12, 34, 34, 128)` |
| Final upsampling stage 1 | `(B, 12, 67, 67, 64)` |
| Final upsampling stage 2 | `(B, 12, 200, 200, 16)` |
| Official projection output | `(B, 12, 200, 200, 1)` |

## Prediction Head Boundary

The official prediction head is:

```text
dec_final_proj: Linear(16 -> 1)
```

The latent tensor immediately before it is:

```text
pre_head_latent = final_decoder(decoder_output)
shape = (B, 12, 200, 200, 16)
```

The migration wrapper exposes this tensor through:

```python
result = wrapped_model(x, return_latent=True)
latent = result["pre_head_latent"]
prediction = result["prediction"]
```

For deeper inspection, `forward_trace()` returns both tensors plus shape traces.
No CSI head is attached at this stage.

## Compatibility Changes

### 1. Input channels: `1 -> 7`

The official first convolution has shape `(16, 1, 3, 3)`. The adapted model
requires `(16, 7, 3, 3)`.

Initialization strategy:

```text
new_weight[:, c, :, :] = old_weight[:, 0, :, :] / 7
```

This repeats the pretrained VIL spatial filters across the seven SEVIRI
channels while preserving the expected activation scale when channels are
combined. No embedding redesign is introduced.

### 2. Image size: `384x384 -> 200x200`

The model config changes only `input_shape` and `target_shape` spatial sizes.
All attention blocks, encoder/decoder depths, global vectors, and projection
layers stay unchanged.

Shape-dependent learned tensors are adapted as follows:

- Learned `H_embed` / `W_embed`: linear interpolation from official spatial
  size to the SEVIRI spatial size at the corresponding hierarchy.
- Relative position bias tables: 1D interpolation along relative-position
  length when the number of heads is unchanged.
- Relative position index buffers: skipped from checkpoint loading and
  regenerated by the newly instantiated official model for the 200x200 geometry.

### 3. Sequence length: `T <= 13`

The official SEVIR input length remains `13`. The wrapper accepts shorter
sequences and right-pads missing timesteps with zeros. This keeps the official
architecture fixed.

### 4. Dataset loader

`SEVIRIImageSequenceDataset` reads:

- `dualet_metadata.parquet` as the existing sample manifest.
- `normalization.json` for channel normalization.
- zarr `X` arrays with shape `(N, 7, 200, 200)`.

It returns only:

```text
satellite: (T, 7, 200, 200)
image_mask
metadata strings
```

It intentionally does not return CSI, GHI, clear-sky GHI, solar elevation,
latitude, longitude, or cyclical time encodings.

## Checkpoint Loading Result

Verification command:

```powershell
& "C:\Users\admin\Downloads\FYP\Codes\torch_env\Scripts\python.exe" `
  "Codes\prep+models\earthformer_migration\verify_forward.py" `
  --split train --batch-size 1 --device cpu
```

Result summary:

- Checkpoint downloaded from the official SEVIR pretrained checkpoint URL.
- Loaded keys: `280`
- Adapted keys: `11`
- Unexpected checkpoint keys: `0`
- Missing keys after load: only regenerated `relative_position_index` buffers.
- Forward pass succeeded.
- Prediction finite: `true`
- Latent finite: `true`

Forward trace on local sample `sample_id=0`, location `LBRAS`:

| Tensor | Shape |
| --- | --- |
| Dataset batch | `(1, 13, 7, 200, 200)` |
| EarthFormer input | `(1, 13, 200, 200, 7)` |
| After initial encoder | `(1, 13, 17, 17, 128)` |
| Encoder memory 0 | `(1, 13, 17, 17, 128)` |
| Encoder memory 1 | `(1, 13, 9, 9, 256)` |
| Initial decoder query | `(1, 12, 9, 9, 256)` |
| After Cuboid decoder | `(1, 12, 17, 17, 128)` |
| Pre-head latent | `(1, 12, 200, 200, 16)` |
| Official prediction | `(1, 12, 200, 200, 1)` |

## Faithfulness Assessment

Preserved unchanged:

- `PatchMerging3D`
- `InitialStackPatchMergingEncoder` structure
- learned positional embedding mechanism
- Cuboid self-attention and cross-attention
- encoder hierarchy
- decoder hierarchy
- skip/memory pathway through `mem_l`
- global vector mechanism
- final upsampling decoder
- official `dec_final_proj` prediction head

Changed only for compatibility:

- Model input shape changed to `(13, 200, 200, 7)`.
- Model target spatial shape changed to `(12, 200, 200, 1)` so the official
  SEVIR projection head remains one-channel.
- First input convolution initialized from the pretrained single-channel
  filters.
- Spatial learned embeddings and relative position bias tables interpolated for
  the smaller crop.
- Dataset loader replaced with an image-only SEVIRI loader.

No CSI prediction head, custom regression layer, DualET encoder, DualET decoder,
or custom forecasting architecture was introduced.

## References

- Paper: https://arxiv.org/abs/2207.05833
- Official repository: https://github.com/amazon-science/earth-forecasting-transformer
- Official SEVIR pretrained checkpoint:
  https://earthformer.s3.amazonaws.com/pretrained_checkpoints/earthformer_sevir.pt
