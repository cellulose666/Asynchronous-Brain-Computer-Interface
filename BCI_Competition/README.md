# BCI Competition Async Decoding

本项目当前主流程已经简化为一条 BNCI2014001 的 OOF 二阶段训练流程：

```text
Stage 1: idle vs task
Stage 2: left_hand / right_hand / feet / tongue
Final: idle / left_hand / right_hand / feet / tongue
```

## 目录结构

```text
BCI_Competition/
  code/
    datasets/
      download_bnci2014001.py
    preprocessing/
      build_oof_windows.py          # 主预处理入口
      build_async_windows.py        # 兼容入口，转到 build_oof_windows.py
      build_protocol_index.py       # 兼容入口
      build_signal_store.py         # 兼容入口
      build_causal_filter_store.py  # 兼容入口
      build_zero_phase_filter_store.py
      build_offline_view.py
      build_validation_folds.py
      build_fold_normalization.py
      build_oof_training_bundle.py
    train/
      train_hierarchical_oof.py     # 主训练入口
      train_eegnet_async.py         # 兼容入口
      train_eegnet_oof.py           # 兼容入口
    eval/
      evaluate_async.py
    models/
      model_factory.py
      models/
  data/
    public/BNCI2014001/
    processed/
  results/
    checkpoints/
    tables/
```

## 下载 BNCI2014001

```powershell
conda activate BCI2026
$env:PYTHONNOUSERSITE=1
python BCI_Competition\code\datasets\download_bnci2014001.py
```

数据会缓存到：

```text
BCI_Competition\data\public\BNCI2014001
```

完整数据应包含 18 个 MAT 文件：

```text
A01T.mat A01E.mat ... A09T.mat A09E.mat
```

## 简化 OOF 预处理

主入口：

```powershell
python BCI_Competition\code\preprocessing\build_oof_windows.py --subjects 1
```

处理全部 subject：

```powershell
python BCI_Competition\code\preprocessing\build_oof_windows.py --subjects all
```

预处理规则：

- 读取原始 BNCI2014001 MAT；
- 只保留 22 个 EEG 通道；
- 保留原生 250 Hz；
- 根据 artifact/BAD annotation 划分 clean segment；
- 不把 artifact trial 两边的信号拼接在一起；
- 对每个 clean segment 单独做 causal 8-30 Hz 滤波；
- 每个 segment 重置滤波状态；
- 2 秒窗口，即 500 个采样点；
- 0.5 秒步长，即 125 个采样点 stride；
- 窗口完全落入一个 MI event 才标为任务态；
- 窗口完全不重叠 MI event 才标为 idle；
- 跨 idle/task 边界的窗口直接丢弃；
- train session 使用 leave-one-run-out 生成 OOF fold。

输出：

```text
BCI_Competition\data\processed\bnci2014001_oof_windows.npz
```

`.npz` 包含：

```text
X        float32, (n_windows, 22, 500)
y        int64, 0 idle / 1 left_hand / 2 right_hand / 3 feet / 4 tongue
subject  int64
session  int64, 0 train / 1 test
run      int64
fold     int64, train session 中 fold id = run id；test session 为 -1
split    int64, 0 train-session / 2 test-session
```

## OOF 二阶段训练

主入口：

```powershell
python BCI_Competition\code\train\train_hierarchical_oof.py --subjects 1 --model eegnet
```

可选模型来自 `code/models/model_factory.py`：

```text
eegnet
shallowconvnet
deepcnn
conformer
deformer
dbconformer
```

示例：

```powershell
python BCI_Competition\code\train\train_hierarchical_oof.py `
  --subjects 1 `
  --model eegnet `
  --binary-epochs 30 `
  --mi-epochs 30 `
  --batch-size 32
```

训练逻辑：

```text
对每个 subject：
  对 train session 的每个 run 做一个 fold：
    validation = 当前 run
    train = 其他 train runs
    只用 train runs 计算 mean/std
    训练 Stage 1 idle/task
    训练 Stage 2 MI 四分类
    对 validation run 生成 OOF logits/predictions

  拼接全部 validation run 的预测，得到完整 OOF predictions

  再使用全部 train session runs：
    重新计算 mean/std
    重新训练 Stage 1
    重新训练 Stage 2
    保存最终模型
    保存综合训练集 metrics
```

输出：

```text
BCI_Competition\results\checkpoints\simple_oof\s01_eegnet_oof_final.pt
BCI_Competition\results\tables\simple_oof\s01_eegnet_oof_predictions.npz
BCI_Competition\results\tables\simple_oof\s01_eegnet_oof_metrics.json
BCI_Competition\results\tables\simple_oof\eegnet_oof_summary.json
```

## 当前说明

- Zhou/Zhou2016 下载和预处理代码已删除。
- 当前训练阶段不再使用 bundle。
- `code/preprocessing` 中旧文件名保留为兼容入口，实际都转到 `build_oof_windows.py`。
- `code/train` 中旧训练入口保留为兼容入口，实际都转到 `train_hierarchical_oof.py`。
