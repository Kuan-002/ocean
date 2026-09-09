# Ocean: Slot Attention with AC and GRPO

This repository contains the minimal training and evaluation pipeline for the
Ocean CLEVR-Hans experiments. It supports the following sequence:

1. train a Slot Attention (SA) encoder;
2. freeze the SA encoder and train either an actor-critic (AC) or group-relative
   policy optimization (GRPO) slot selector;
3. evaluate the selected model on the CLEVR-Hans test split.

Datasets, environment files, checkpoints, run directories, logs, figures, and
notebooks are intentionally excluded from version control.

## 1. Install the environment

Python 3.10 is recommended.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All commands below assume that they are executed from the repository root with
the virtual environment activated.

## 2. Prepare CLEVR-Hans outside the repository

Store the dataset outside this Git checkout. The configured dataset directory
must have this layout:

```text
/external/path/CLEVR-Hans7/
+-- train/
|   +-- images/<image files>
|   +-- CLEVR_HANS_scenes_train.json
+-- val/
|   +-- images/<image files>
|   +-- CLEVR_HANS_scenes_val.json
+-- test/
    +-- images/<image files>
    +-- CLEVR_HANS_scenes_test.json
```

Each scenes JSON file must contain a top-level `scenes` array. Every scene must
provide `image_filename` (string), `class_id` (integer), and `objects` (array).

## 3. Create an external environment file

Create the experiment `.env` outside the repository. The following fields
define the supported CLEVR-Hans pipeline:

```dotenv
DEVICE=cuda

DATASET=ch
DATASET_PATH=/external/path/CLEVR-Hans7
DATASET_MAX_NUM_OBJ=10
DATASET_IMAGE_RESOLUTION=64
DATASET_CACHE=False
DATASET_BATCH_SIZE=64
DATASET_EVAL_BATCH_SIZE=64
DATASET_NUM_WORKERS=8
LABELS=[0,1,2,3,4,5,6]

SA_WIDTH=64
SA_NUM_SLOTS=11
SA_SLOT_DIM=64
SA_ROUTING_ITERS=3
SA_SEED=8
SA_LEARNING_RATE=0.0005
SA_WEIGHT_DECAY=5e-7
SA_DETERMINISTIC=True

EPOCHS=500
MAX_TRAIN_BATCHES=0
MAX_VAL_BATCHES=0
MAX_NORM=-1

RNN_SEL_SA_DETERMINISTIC_SLOTS=True
RNN_SEL_SA_NOISE_SEED=0
```

`DEVICE=cuda` uses CUDA when available and otherwise falls back to CPU unless
`--require_cuda` is supplied. A batch limit of `0` means no limit. The SA
architecture fields must remain unchanged when its checkpoint is used by a
selector.

## 4. Train Slot Attention

```bash
python scripts/train_slot_attention.py \
  --env_path /external/path/ocean.env \
  --out_subpath /external/path/runs/sa_seed8 \
  --epochs 500
```

The SA trainer writes its artifacts under the resolved output directory:

```text
sa_seed8/
+-- .env
+-- sa_history.json
+-- checkpoints/sa/
    +-- best_ckpt.pt
    +-- last_ckpt.pt
```

The checkpoint required by selector training is
`checkpoints/sa/best_ckpt.pt`. Resume SA training with `--resume PATH`.

SA-specific command-line parameters:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--env_path` | path | required | External `.env` file |
| `--out_subpath` | path | required | External output directory |
| `--epochs` | integer | `EPOCHS` | Number of training epochs |
| `--max_train_batches` | integer | `.env` value | Training batches per epoch; `0` is unlimited |
| `--max_val_batches` | integer | `.env` value | Validation batches per epoch; `0` is unlimited |
| `--num_workers` | integer | `.env` value | DataLoader workers |
| `--resume` | path | none | Existing SA checkpoint |
| `--require_cuda` | flag | false | Fail instead of falling back to CPU |

## 5. Train a selector

The SA model is frozen during selector training. Use the same `.env` file and
the best SA checkpoint from the previous stage. Output directories must remain
outside the repository.

### Actor-critic

```bash
python scripts/train_selector_v1_ac.py \
  --env_path /external/path/ocean.env \
  --sa_checkpoint /external/path/runs/sa_seed8/checkpoints/sa/best_ckpt.pt \
  --out_subpath /external/path/runs/ac_seed8 \
  --stage1_epochs 100 \
  --embed_dim 512 \
  --pos_dim 0 \
  --min_steps 3 \
  --target_min_slots 3 \
  --target_max_slots 0 \
  --early_exit_conf 0.8
```

### GRPO

```bash
python scripts/train_selector_v1_grpo.py \
  --env_path /external/path/ocean.env \
  --sa_checkpoint /external/path/runs/sa_seed8/checkpoints/sa/best_ckpt.pt \
  --out_subpath /external/path/runs/grpo_seed8 \
  --epochs 100 \
  --embed_dim 512 \
  --pos_dim 0 \
  --min_steps 3 \
  --target_min_slots 3 \
  --target_max_slots 0 \
  --early_exit_conf 0.8 \
  --grpo_group_size 4 \
  --grpo_advantage return
```

Common selector parameters:

| Parameter | Meaning |
| --- | --- |
| `--lr`, `--weight_decay` | Selector optimizer settings |
| `--embed_dim`, `--pos_dim`, `--dropout` | Selector architecture |
| `--max_steps`, `--min_steps`, `--class_min_slots` | Selection horizon and stop constraints |
| `--target_min_slots`, `--target_max_slots` | Reward target range |
| `--lambda_slot`, `--lambda_over`, `--lambda_under` | Slot-count reward penalties |
| `--class_coef`, `--full_order_class_coef`, `--entropy_coef` | Training loss weights |
| `--early_exit_conf` | Greedy evaluation confidence threshold |
| `--eval_every`, `--eval_batches` | Validation frequency and optional batch limit |
| `--max_train_batches`, `--num_workers` | Local/debugging overrides |
| `--early_stop_patience`, `--early_stop_min_delta` | Early stopping settings |
| `--skip_test` | Skip the automatic final test pass |

AC additionally defines `--gamma`, `--value_coef`, and `--max_grad_norm`.
GRPO additionally defines `--grpo_group_size`, `--grpo_advantage`,
`--stop_reward_mode`, `--stop_reward_scale`, `--premature_stop_coef`, and
`--future_gain_margin`.

AC produces `selector_v1_ac_best.pt` and `selector_v1_ac_meta.json`. GRPO
produces `selector_v1_grpo_best.pt` and `selector_v1_grpo_meta.json`. Both also
write a last checkpoint, `history.json`, and (unless `--skip_test` is used)
`test_metrics.json`.

## 6. Run final evaluation

AC and GRPO use the same evaluator. Select the checkpoint naming convention
with `--method`:

```bash
python scripts/eval_selector_v1_test.py \
  --method ac \
  --run_dir /external/path/runs/ac_seed8 \
  --output /external/path/results/ac_seed8.json
```

```bash
python scripts/eval_selector_v1_test.py \
  --method grpo \
  --run_dir /external/path/runs/grpo_seed8 \
  --output /external/path/results/grpo_seed8.json
```

Evaluation parameters:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--method` | `ac` or `grpo` | required | Metadata/checkpoint naming convention |
| `--run_dir` | path | required | External selector run directory |
| `--env_path` | path | metadata value | Override the environment file |
| `--sa_checkpoint` | path | metadata value | Override the SA checkpoint |
| `--checkpoint` | path | method-specific best checkpoint | Override selector checkpoint |
| `--output` | path | `<run_dir>/test_metrics.json` | Result JSON path |
| `--config_out_dir` | path | `<output parent>/config` | Temporary Config output |
| `--eval_batches` | integer | `0` | Test batch limit; `0` is unlimited |
| `--num_workers` | integer | `.env` value | DataLoader workers |
| `--early_exit_conf` | float | metadata value | Override early-exit confidence |
| `--require_cuda` | flag | false | Require CUDA |

The output JSON contains aggregate classification metrics, per-class accuracy,
slot-count distributions, average and median selected slots, and full-order
accuracy.

## Repository boundary

The repository tracks source code and documentation only. `.gitignore` blocks
all Shell scripts, `.env` files, datasets, run directories, checkpoints, and
common serialized data formats. Do not place external artifacts in this Git
checkout even if they are ignored.
