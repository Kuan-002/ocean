# Ocean AC/GRPO 测试接口

本仓库只定义 CLEVR-Hans 上的 AC/GRPO checkpoint 测试流程，不包含训练、
数据集、checkpoint、实验环境、分析、作图或报告文件。AC 与 GRPO 使用相同的
`SlotSelectorAC` 推理结构，方法参数只决定 metadata 与 selector checkpoint
的文件名。

## Shell 接口

唯一 Shell 入口的调用格式为：

`scripts/run_test.sh METHOD RUN_DIR [OPTIONS...]`

| 参数 | 格式 | 说明 |
| --- | --- | --- |
| `METHOD` | `ac` 或 `grpo` | 必填；选择模型文件命名规则 |
| `RUN_DIR` | 目录路径 | 必填；外部实验 run 目录 |
| `OPTIONS` | Python CLI 参数 | 可选；原样传递给评估器 |

Shell 入口使用环境变量 `PYTHON_BIN` 指定 Python 可执行文件，默认值为
`python`。

## Python 参数

Python 入口为 `scripts/eval_selector_v1_test.py`。

| 参数 | 类型/取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--method` | `ac`、`grpo` | 无 | 必填；实验方法 |
| `--run_dir` | 路径 | 无 | 必填；外部 run 目录 |
| `--env_path` | 文件路径 | metadata 中的值 | 覆盖实验 `.env` 路径 |
| `--sa_checkpoint` | 文件路径 | metadata 中的值 | 覆盖 Slot Attention checkpoint |
| `--checkpoint` | 文件路径 | 见下表 | 覆盖 selector checkpoint |
| `--output` | JSON 文件路径 | `<run_dir>/test_metrics.json` | 指标输出位置 |
| `--config_out_dir` | 目录路径 | `<output目录>/config` | Config 临时输出目录 |
| `--eval_batches` | 非负整数 | `0` | `0` 表示完整测试集，否则限制 batch 数 |
| `--num_workers` | 非负整数 | `.env` 中的值 | DataLoader worker 数；受限环境可设为 `0` |
| `--early_exit_conf` | 浮点数 | `ac_config.early_exit_conf` | 覆盖提前退出置信度 |
| `--require_cuda` | flag | 关闭 | CUDA 不可用时直接报错 |

## Run 目录格式

`RUN_DIR` 必须位于本仓库之外。方法与默认文件名映射如下：

| 方法 | metadata | selector checkpoint |
| --- | --- | --- |
| AC | `selector_v1_ac_meta.json` | `selector_v1_ac_best.pt` |
| GRPO | `selector_v1_grpo_meta.json` | `selector_v1_grpo_best.pt` |

当 `--checkpoint` 未指定时，selector checkpoint 必须位于 `RUN_DIR`。当
metadata 中的 `args.env_path` 不存在时，评估器回退到 `RUN_DIR/.env`。

## Metadata 格式

Metadata 是 UTF-8 JSON 对象。评估器读取以下字段：

```text
{
  "args": {
    "env_path": string,
    "sa_checkpoint": string
  },
  "ac_config": object
}
```

- `args.env_path`：外部 `.env` 文件；传入 `--env_path` 时可省略。
- `args.sa_checkpoint`：外部 Slot Attention checkpoint；传入
  `--sa_checkpoint` 时可省略。
- `ac_config`：`ACConfig` 的字段集合，定义网络尺寸、slot 数、分类数、停止
  策略与分类器变体。未知字段会导致加载失败，结构字段必须与权重一致。

## Checkpoint 格式

所有 checkpoint 都必须位于仓库之外。

| 类型 | PyTorch 字典字段 |
| --- | --- |
| AC/GRPO selector | `model_state_dict` |
| Slot Attention | `model_state_dict`、`optimizer_state_dict`、`best_loss` |

Selector 权重必须与 metadata 的 `ac_config` 完全匹配。Slot Attention 权重
必须与 `.env` 中的 `SA_WIDTH`、`SA_NUM_SLOTS`、`SA_SLOT_DIM`、
`SA_ROUTING_ITERS` 和 `DATASET_IMAGE_RESOLUTION` 匹配。

## `.env` 参数

测试路径使用以下配置：

| 分类 | 参数 |
| --- | --- |
| 设备 | `DEVICE` |
| 数据 | `DATASET=ch`、`DATASET_PATH`、`DATASET_MAX_NUM_OBJ`、`DATASET_IMAGE_RESOLUTION`、`DATASET_CACHE`、`DATASET_EVAL_BATCH_SIZE`、`DATASET_NUM_WORKERS`、`LABELS` |
| Slot Attention | `SA_WIDTH`、`SA_NUM_SLOTS`、`SA_SLOT_DIM`、`SA_ROUTING_ITERS`、`SA_LEARNING_RATE`、`SA_WEIGHT_DECAY` |
| 可复现性 | `SA_SEED`、`SA_DETERMINISTIC`、`RNN_SEL_SA_DETERMINISTIC_SLOTS`、`RNN_SEL_SA_NOISE_SEED` |

`DATASET` 只接受 `ch`。`LABELS` 的格式为逗号分隔的整数列表，例如
`[0,1,2,3,4,5,6]`。

## CLEVR-Hans 数据格式

`DATASET_PATH` 必须位于仓库之外，并具有以下结构：

```text
DATASET_PATH/
├── val/
│   ├── images/<image files>
│   └── CLEVR_HANS_scenes_val.json
└── test/
    ├── images/<image files>
    └── CLEVR_HANS_scenes_test.json
```

每个 scenes JSON 的根对象包含 `scenes` 数组；每个元素至少包含：

| 字段 | 类型 | 用途 |
| --- | --- | --- |
| `image_filename` | string | `images/` 下的文件名 |
| `class_id` | integer | 分类标签 |
| `objects` | array | 对象列表；长度用于对象数量过滤 |

## 输出 JSON 格式

输出为单个 UTF-8 JSON 对象，数值保留四位小数。固定指标包括：

- `loss`、`total`
- `accuracy`/`acc`、`balanced_accuracy`
- `precision`、`recall`、`f1`
- `auc`、`macro_auc`、`micro_auc`
- `macro_specificity`、`full_order_accuracy`
- `avg_selected`、`median_selected`

动态字段包括：

- `class_<id>_accuracy`
- `class_<id>_avg_selected`
- `class_<id>_count`
- `selected_count_<n>`

## 上传排除规则

`.gitignore` 采用代码白名单，并额外屏蔽 `data/`、`dataset/`、`datasets/`、
`checkpoints/`、`runs/`、`.env*` 以及常见 checkpoint/数据扩展名。仓库唯一
允许跟踪的 Shell 文件是 `scripts/run_test.sh`。
