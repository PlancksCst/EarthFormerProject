# EarthFormer SEVIRI Backbone Training

Research-grade PyTorch training scaffold for fine-tuning the official
EarthFormer backbone on Meteosat SEVIRI image sequences, with an optional
Perceiver IO output-query readout for CSI-stage architectural validation.

The original backbone training path still uses the official EarthFormer
frame-forecasting output already validated in `earthformer_migration`. The new
Perceiver readout path is separate: it replaces only the use of
`dec_final_proj` after `pre_head_latent` and produces a CSI-shaped `(B,T)`
sequence for architecture validation. It does not use auxiliary features.

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
    losses.py
  scripts/
    verify_forward.py
    verify_perceiver_readout.py
    inference.py
  figures/
    earthformer_perceiver_readout.mmd
    earthformer_perceiver_readout.svg
  utils/
    logger.py
    metrics.py
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
satellite input: (T, 7, 200, 200), T <= 13
temporary image target: (12, 1, 200, 200)
```

The target is a temporary one-channel SEVIRI image target used only to validate
that the pretrained EarthFormer backbone can be optimized. It is not CSI.

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
- local `data/`
- local `../verification_datasets/BEST_7_3months`
- local `../verification_datasets/BEST_7_full_year`

Useful options:

```bash
python training/train.py \
  --dataset-root /path/to/dataset \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --epochs 20 \
  --num-workers 2 \
  --device auto
```

The training loop includes:

- `DataLoader`
- AdamW optimizer
- cosine scheduler
- automatic mixed precision on CUDA
- gradient clipping
- validation every epoch
- `tqdm` progress bar
- CSV logging
- latest and best checkpoint saving

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
--num-output-queries 12
--num-attention-heads 4
--readout-dropout 0.1
--regression-hidden-dim 32
--freeze-earthformer
```

The architecture diagram is stored at:

```text
figures/earthformer_perceiver_readout.svg
```

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

## Validation

Validation is called automatically after every epoch. The validation function
returns only average MSE loss.

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

- official prediction tensor
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
- No auxiliary features, latitude, longitude, solar elevation, clear-sky GHI,
  or feature fusion at this stage.
