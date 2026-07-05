# EarthFormer SEVIRI CSI Forecasting

Research-grade PyTorch training scaffold for fine-tuning the official
EarthFormer backbone on Meteosat SEVIRI image sequences with a Perceiver IO
output-query readout for next-day CSI forecasting.

The current training path keeps the verified EarthFormer migration intact and
uses the already-validated Perceiver readout after `pre_head_latent`. The model
predicts only a 13-element CSI sequence. GHI is reconstructed externally as
`CSI * clear_sky_ghi` during validation and inference.

## Project Structure

```text
EarthFormer/
  configs/
    config.py
  models/
    model.py
    perceiver_model.py
    __init__.py
  readout/
    perceiver_readout.py
    __init__.py
  datasets/
    seviri_dataset.py
    __init__.py
  training/
    train.py
    validate.py
    checkpoint.py
    debugging.py
    losses.py
  scripts/
    diagnostic_utils.py
    verify_forward.py
    verify_perceiver_readout.py
    verify_perceiver_pipeline.py
    inspect_perceiver.py
    check_attention.py
    test_one_batch.py
    test_real_batch.py
    test_overfit.py
    test_resume.py
    run_sanity_suite.py
    inference.py
  figures/
    earthformer_perceiver_readout.mmd
    earthformer_perceiver_readout.svg
  utils/
    logger.py
    metrics.py
    plotting.py
    precision.py
    seed.py
  outputs/
  checkpoints/
  requirements.txt
  README.md
```

The working migration code is vendored in this repository:

```text
earthformer_migration/
earthformer_src/
```

The old sibling layout remains supported as a fallback for existing local
workspaces, but package entry points now prefer the vendored copies.

## Dataset Contract

The dataset root must contain:

```text
metadata.parquet
normalization.json
```

For backward compatibility with the earlier local dataset, the loader also
accepts:

```text
dualet_metadata.parquet
```

Each sample uses:

```text
satellite input: (13, 7, 200, 200)
target CSI:      (13,)
clear-sky GHI:   (13,)
```

The dataset may also provide `target_ghi`; otherwise validation reconstructs
ground-truth GHI from `target_csi * clear_sky_ghi`. Auxiliary features are not
passed to the model.

## Installation

```bash
pip install -r requirements.txt
```

The official EarthFormer source is vendored at:

```text
earthformer_src/
```

The pretrained SEVIR checkpoint is not committed. If it is absent, the migration
loader downloads the official SEVIR checkpoint from the EarthFormer S3 URL into:

```text
earthformer_src/pretrained_checkpoints/earthformer_sevir.pt
```

## Training

From inside `EarthFormer/`:

```bash
python training/train.py --dataset-root /path/to/dataset
```

When `--dataset-root` is omitted, the config tries, in order:

- `EARTHFORMER_DATASET_ROOT`
- a single matching Kaggle dataset under `/kaggle/input`
- Colab local SSD `/content/datasets`
- Google Drive `/content/drive/MyDrive/EarthFormer/datasets`
- local `data/`
- local `../verification_datasets/BEST_7_3months`
- local `../verification_datasets/BEST_7_full_year`

Useful options:

```bash
python training/train.py \
  --dataset-root /path/to/dataset \
  --batch-size 2 \
  --backbone-learning-rate 1e-5 \
  --head-learning-rate 1e-4 \
  --warmup-epochs 5 \
  --early-stopping-patience 5 \
  --clear-sky-threshold 20.0 \
  --low-csi-weight 2.0 \
  --low-csi-threshold 0.7 \
  --ghi-loss-weight 0.1 \
  --use-hour-query-embedding \
  --query-diversity-weight 0.0 \
  --epochs 20 \
  --input-length 13 \
  --output-length 13 \
  --num-workers 2 \
  --device auto
```

The training loop includes:

- `DataLoader`
- AdamW optimizer with differential learning rates for EarthFormer and readout
- linear warmup followed by cosine scheduling
- early stopping on validation loss
- full-precision CUDA/CPU training by default
- optional CUDA mixed precision with `--amp` or `EARTHFORMER_MIXED_PRECISION=1`
- BF16 autocast by default when AMP is enabled; FP16 requires `--amp-dtype fp16`
- gradient clipping
- validation every epoch
- physical valid-hour masking with `target_mask == 0` and
  `clear_sky_ghi > --clear-sky-threshold`
- weighted CSI loss for low-CSI cloudy hours
- optional reconstructed-GHI loss term controlled by `--ghi-loss-weight`
- hour-aware Perceiver output queries via learnable forecast-hour embeddings
- optional output-query diversity regularization via
  `--use-query-diversity-loss --query-diversity-weight <value>`
- query cosine-similarity diagnostics and heatmaps after validation
- `tqdm` progress bar
- CSV logging with CSI and reconstructed-GHI metrics
- validation prediction CSVs under `outputs/predictions/`
- distribution, scatter, and residual diagnostics under `outputs/plots/`
- best-epoch plot copies under `outputs/best_epoch/`
- latest and best checkpoint saving

On Colab, primary artifacts are written to `/content/checkpoints` and
`/content/outputs`. If Google Drive is mounted at
`/content/drive/MyDrive/EarthFormer`, the same checkpoints, logs, plots,
predictions, and reports are mirrored to the matching Drive directories.

## Perceiver IO Readout Validation

The Perceiver readout path keeps the official EarthFormer backbone unchanged:

```text
SEVIRI images
  -> official EarthFormer backbone
  -> pre_head_latent (B,T,H,W,16)
  -> spatial tokens (B,T,HW,16)
  -> learnable output queries (T,D)
  -> per-timestep cross attention
  -> output embeddings (B,T,D)
  -> Linear -> GELU -> Linear
  -> CSI-shaped output (B,T)
```

Run the verifier from the parent directory that contains `EarthFormer/`:

```bash
python -m EarthFormer.scripts.verify_perceiver_readout \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu
```

The verifier prints tensor shapes for `pre_head_latent`, flattened tokens,
queries, attention output, and regression output. It also checks gradient flow,
strict checkpoint save/load compatibility, and estimated tensor memory usage.

Configurable readout options:

```bash
--query-dimension 64
--num-output-queries 13
--num-attention-heads 4
--readout-dropout 0.1
--regression-hidden-dim 32
--freeze-earthformer
```

The architecture diagram is stored at:

```text
figures/earthformer_perceiver_readout.svg
```

## Forecasting Sanity Suite

The forecasting diagnostics validate the complete current architecture:

```text
SEVIRI images
  -> EarthFormer backbone
  -> pre_head_latent
  -> Perceiver IO readout
  -> CSI-shaped forecast output
```

Run the full suite in Colab:

```bash
python EarthFormer/scripts/run_sanity_suite.py \
  --dataset-root /content/datasets \
  --checkpoint-dir /content/checkpoints \
  --output-dir /content/outputs \
  --batch-size 1 \
  --num-workers 2 \
  --device auto
```

Individual checks:

```bash
python EarthFormer/scripts/verify_perceiver_pipeline.py --dataset-root /content/datasets
python EarthFormer/scripts/inspect_perceiver.py --dataset-root /content/datasets
python EarthFormer/scripts/check_attention.py --dataset-root /content/datasets
python EarthFormer/scripts/test_one_batch.py --dataset-root /content/datasets
python EarthFormer/scripts/test_overfit.py --dataset-root /content/datasets --samples 8 --max-epochs 50
python EarthFormer/scripts/test_resume.py --dataset-root /content/datasets
```

All diagnostic outputs are written to:

```text
outputs/diagnostics/
```

The optimization sanity tests use a deterministic satellite-only target by
default. This is only to verify forward/backward/optimizer/checkpoint behavior;
it does not introduce CSI, GHI, location, time, or auxiliary metadata into the
model.

## Checkpointing

Checkpoints are saved to:

```text
checkpoints/last.pt
checkpoints/best.pt
```

Resume training with:

```bash
python training/train.py \
  --dataset-root /path/to/dataset \
  --resume-checkpoint checkpoints/last.pt
```

Each checkpoint contains:

- epoch
- model state
- optimizer state
- scheduler state
- AMP scaler state
- best validation loss
- serialized configuration
- monitored best metric
- early-stopping state

## Validation

Validation is called automatically after every epoch. It reports MSE loss plus
MAE, RMSE, nRMSE, and R2 for both CSI and reconstructed GHI.
Each validation epoch also saves per-hour prediction CSVs and diagnostic plots
for prediction distributions, prediction-vs-target scatter, residual
histograms, and residual-vs-prediction behavior.

## Inference

```bash
python scripts/inference.py \
  --dataset-root /path/to/dataset \
  --split test \
  --model-checkpoint checkpoints/best.pt
```

The script saves:

```text
outputs/inference_sample.pt
```

with:

- CSI prediction tensor
- reconstructed GHI prediction tensor
- optional CSI/GHI target tensors
- clear-sky GHI tensor
- pre-head latent tensor
- sample metadata

## Kaggle Usage

Kaggle datasets are mounted as:

```text
/kaggle/input/<dataset_name>/
```

Run:

```bash
python training/train.py \
  --dataset-root /kaggle/input/<dataset_name> \
  --checkpoint-dir /kaggle/working/checkpoints \
  --output-dir /kaggle/working/outputs
```

You can also set:

```bash
export EARTHFORMER_DATASET_ROOT=/kaggle/input/<dataset_name>
```

and then run:

```bash
python training/train.py
```

## Coding Requirements

- PyTorch only.
- Modular files with small functions.
- Type hints and docstrings.
- No notebook-specific code.
- No hardcoded Windows paths.
- No auxiliary feature conditioning, latitude, longitude, solar elevation, or
  feature fusion at this stage. Clear-sky GHI is used only after prediction for
  external GHI reconstruction.
