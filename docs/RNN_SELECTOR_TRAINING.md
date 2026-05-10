# RNN 槽选择器训练说明（新版）

## 流程

1. **冻结**：Slot Attention、`DeepSetsClassifier` 仅前向。
2. **模仿**：`action_head` 对齐 §4.1 专家轨迹；每步选槽并 `step_hidden` 后，对 **`class_head(h)`** 与 **`y_hat = clf(slots,None).argmax`** 做交叉熵，权重 **`RNN_SEL_IMITATION_ALPHA_CLASS`**（默认 2.0）。
3. **参考策略**：`deepcopy(policy)` 冻结，用于 GRPO KL。
4. **GRPO**：`slots` 按组 `G` 复制；**训练 rollout 仅**以 STOP 或 `max_steps` 结束（**无** `p≥τ` 早停）。损失 `L_grpo + alpha_class * L_class`，含梯度裁剪与优势裁剪。
5. **验证**：`eval_rnn_selector` 使用贪心；若 **`RNN_SEL_EVAL_DISABLE_CONF_EARLY_EXIT=True`**，则与训练一致（关掉选槽后的置信早停）。

## 产出文件（`--out_subpath`）

| 文件 | 说明 |
|------|------|
| `.env` | 训练用 env 拷贝 |
| `rnn_selector_pi_ref.pt` | 模仿结束后的权重 |
| `rnn_selector_best.pt` | 验证 `success_rate` 最高时 |
| `rnn_selector_last.pt` | GRPO 全部轮次结束 |

## 主要环境变量

| 变量 | 含义 |
|------|------|
| `RNN_SEL_IMITATION_ALPHA_CLASS` | 模仿阶段 `class_head` 辅助 CE 权重（0 关闭） |
| `RNN_SEL_EVAL_DISABLE_CONF_EARLY_EXIT` | 验证/inspect 时关闭 `p≥τ` 槽后早停 |
| `RNN_SEL_EVAL_REQUIRE_TAU_TO_STOP` | 低置信时掩 STOP（仅 eval） |
| `RNN_SEL_ALPHA_CLASS` | GRPO 中 `L_class` 系数 |
| 其余 `RNN_SEL_*` | 见 `src/config.py` `_load_deepsets_pipeline_config` |

## 操作命令

在仓库根目录执行（将 Python 换成本机带 PyTorch 的解释器）。

### 训练

```bash
cd /homes/kw1025/ocean
python scripts/run_train_rnn_selector.py \
  --env_path scripts/envs/.envA_ch3_rnn \
  --out_subpath ./out/rnn_selector_run \
  --sa_checkpoint out/sa_ch3_64_1/checkpoints/sa/999_ckpt.pt \
  --cls_checkpoint out/deepsets_classification_1/deepsets_classifier_best.pt
```

或使用一键脚本（输出目录带时间戳）：

```bash
cd /homes/kw1025/ocean
./scripts/train_rnn_selector_ch3.sh
```

### 评估

```bash
cd /homes/kw1025/ocean
python scripts/eval_rnn_selector.py \
  --env_path scripts/envs/.envA_ch3_rnn \
  --sa_checkpoint out/sa_ch3_64_1/checkpoints/sa/999_ckpt.pt \
  --selector_checkpoint ./out/rnn_selector_run/rnn_selector_best.pt \
  --split val
```

### 打印贪心轨迹（与训练 rollout 对齐可加 `--no_conf_early_exit`）

```bash
cd /homes/kw1025/ocean
python scripts/inspect_rnn_selector_rollout.py \
  --env_path scripts/envs/.envA_ch3_rnn \
  --sa_checkpoint out/sa_ch3_64_1/checkpoints/sa/999_ckpt.pt \
  --selector_checkpoint ./out/rnn_selector_run/rnn_selector_best.pt \
  --no_conf_early_exit \
  --n_batches 3
```

### 与 GRPO 训练语义对齐的验证集指标

在 `scripts/envs/.envA_ch3_rnn` 中设置：

```bash
RNN_SEL_EVAL_DISABLE_CONF_EARLY_EXIT=True
```

再运行 `eval_rnn_selector.py`（或训练脚本中的 val），则 **不再**因 `p≥τ` 在选槽后提前结束。

---

## 一键流水线：SA → DeepSets → RNN

脚本依次训练并 **自动把上一步的 checkpoint 路径传给下一步**（见 [scripts/run_pipeline_ch3.sh](file:///homes/kw1025/ocean/scripts/run_pipeline_ch3.sh)）。

- 环境模板：[scripts/envs/.env_pipeline_ch3](file:///homes/kw1025/ocean/scripts/envs/.env_pipeline_ch3)（请先改 **`DATASET_PATH`**）。
- SA：`main.py --type only_sa`，**`EPOCHS=500`**，`CHECKPOINT=10`。
- CLS：`run_train_classifier.py`，**`DS_CLS_EPOCHS=500`**，冻结 SA。
- RNN：`run_train_rnn_selector.py`，**模仿 15 epoch + GRPO 80 epoch**（可在 env 中改 `RNN_SEL_*`）。
- 结束后在 **`$PIPELINE_ROOT/MANIFEST.txt`** 写入 `sa_checkpoint`、`cls_checkpoint`、`rnn_*` 及一条 eval 示例命令。

```bash
cd /homes/kw1025/ocean
chmod +x scripts/run_pipeline_ch3.sh
./scripts/run_pipeline_ch3.sh
```

指定 Python 与 env：

```bash
PYTHON=/vol/bitbucket/kw1025/ocean/bin/python \
ENV_FILE=scripts/envs/.env_pipeline_ch3 \
./scripts/run_pipeline_ch3.sh
```

---

实现入口：[scripts/run_train_rnn_selector.py](file:///homes/kw1025/ocean/scripts/run_train_rnn_selector.py)；核心逻辑：[src/explanation/rnn_selector_training.py](file:///homes/kw1025/ocean/src/explanation/rnn_selector_training.py)。
