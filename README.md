# Two-List CRC Model Evaluation

本项目用于评估一个新的医学术语抽取模型。核心思想是：当只有两份不完整人工标注列表时，先把两份列表合并成可观察状态 `10`、`01`、`11`，再训练 q-function 估计漏标偏差，最后给出朴素指标和 CRC 校正后的 precision / recall。

## 当前运行结论

我在当前目录运行过代码，环境为 `Python 3.11.9`。

### 成功运行的命令

```bash
python3 run_list_state_simulation.py
python3 simulation_bootstrap_validation.py
python3 build_two_lists_and_validate.py
python3 CRCModel_evaluation_full_review/build_two_lists_and_validate.py
```

`run_list_state_simulation.py` 和 `simulation_bootstrap_validation.py` 是同一条完整模拟流程，运行结果一致：

| 指标 | 结果 |
| --- | ---: |
| full terms | 1737 |
| train reconstructed visible | 909 |
| test reconstructed visible | 597 |
| train state counts | `11=452`, `10=296`, `01=161` |
| test state counts | `11=289`, `10=183`, `01=125` |
| true precision | 0.9455 |
| true recall | 0.6565 |
| naive precision mean | 0.8285 |
| naive recall mean | 0.6608 |
| CRC corrected precision mean | 0.9232 |
| CRC corrected recall mean | 0.6563 |
| bootstrap resamples | 1000 |

解释：朴素 precision 明显低估真实 precision，CRC 校正后 precision 从 `0.8285` 提高到 `0.9232`，更接近 full truth 的 `0.9455`。recall 本来偏差较小，校正后 `0.6563` 基本贴近真实值 `0.6565`。

完整输出写入：

```text
simulation_outputs/
simulation_outputs/data/
simulation_outputs/models/pubmedbert/
simulation_outputs/plots/
```

图像结果在：

```text
simulation_outputs/plots/precision_hist.png
simulation_outputs/plots/recall_hist.png
simulation_outputs/plots/summary_barplot.png
```

### 独立运行修复说明

最初根目录脚本不能完全独立运行，原因有两个：

1. `run_list_state_simulation.py`、`build_two_lists_and_validate.py` 和 `model_term_matching.py` 把上一级目录加入了 `sys.path` 最前面，所以会优先导入上一级的 `crc_functions.py` 和 `synthetic_pipeline.py`。
2. 当前目录缺少独立运行需要的 `crc_functions.py` 和 `synthetic_pipeline.py`，并且默认 Excel 文件名和当前真实文件名不一致。

现在已经修复：

1. 根目录已补充 `crc_functions.py` 和 `synthetic_pipeline.py`。
2. 三个入口文件已改为当前目录优先导入。
3. `synthetic_pipeline.py` 已兼容当前存在的 `mimic_iii_synthetic_term_extraction_50_long_full_context-2 2.xlsx`。

因此根目录现在可以直接运行：

```bash
python3 build_two_lists_and_validate.py
```

该命令已成功，结果如下：

```json
{
  "n_list1": 1222,
  "n_list2": 1002,
  "n_reconstructed_visible": 1493,
  "n_expected_visible": 1493,
  "keys_equal": true,
  "state_equal": true,
  "matched_equal": true,
  "state_mismatch_count": 0,
  "reconstructed_state_counts": {
    "11": 731,
    "10": 491,
    "01": 271
  }
}
```

## 安装依赖

推荐先进入项目根目录：

```bash
cd /Users/kylewang/Desktop/GENIE/CRCmodel/CRCevaluation_20260422/list_state_extension
```

安装依赖：

```bash
python3 -m pip install pandas numpy torch transformers matplotlib openpyxl
```

完整 q-function 训练需要本机已经有 PubMedBERT 模型缓存。当前代码默认路径是：

```text
~/.cache/huggingface/hub/models--microsoft--BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/snapshots/e1354b7a3a09615f6aba48dfad4b7a613eef7062
```

代码使用 `local_files_only=True`，所以新机器上如果没有这个缓存，需要先下载模型或把 `PUBMEDBERT_PATH` 改成你的本地模型目录。

## 文件说明

| 文件或目录 | 功能 |
| --- | --- |
| `README.md` | 项目说明和运行教程。 |
| `run_list_state_simulation.py` | 完整模拟入口：读取 Excel，模拟 train/test 两份人工列表，重建可观察状态，训练 q-function，bootstrap 评估并画图。 |
| `simulation_bootstrap_validation.py` | 轻量入口文件，内部直接调用 `run_list_state_simulation.run()`，因此结果与完整模拟入口一致。 |
| `build_two_lists_and_validate.py` | 两列表重建验证脚本；当前已可在根目录独立运行。 |
| `evaluate_two_lists_model.py` | 核心 API 文件，提供两列表合并、模型匹配、q-function 训练、precision/recall 估计和一站式 `evaluate_two_lists_with_model()`。 |
| `model_term_matching.py` | 单独的术语匹配工具，支持字符匹配和 AI 匹配，把模型预测与人工可见表对齐并生成 `matched`。 |
| `crc_functions.py` | CRC 相关辅助函数：状态表构造、q-function 训练包装、指标图绘制、表格读写。 |
| `synthetic_pipeline.py` | 合成数据读取、采样概率、bootstrap、真实指标计算和直方图绘制。 |
| `CRCModel_evaluation_full_review/` | 更自包含的一份完整代码副本，包含 `crc_functions.py`、`synthetic_pipeline.py`、`requirements.txt` 和同名运行脚本。 |
| `data/` | 根目录两列表验证输出数据，包括模拟列表、期望可见状态、重建状态和验证报告。 |
| `simulation_outputs/` | 完整模拟输出，包括 train/test CSV、q-function 模型、训练摘要、bootstrap 摘要和图。 |
| `mimic_iii_synthetic_term_extraction_50_long_full_context-2 2.xlsx` | 根目录完整模拟使用的 Excel 输入。 |

## 核心概念

两份人工列表按以下键合并：

```text
doc_id, phrase, type, context
```

生成的状态：

| state | 含义 |
| --- | --- |
| `10` | 只出现在 list 1 |
| `01` | 只出现在 list 2 |
| `11` | 同时出现在 list 1 和 list 2 |
| `00` | 两份列表都没标到，不可观察，不会出现在可见表中 |

q-function 训练三个二分类头：

| head | 预测目标 |
| --- | --- |
| `q1` | `state` 属于 `10` 或 `11` |
| `q2` | `state` 属于 `01` 或 `11` |
| `q12` | `state == 11` |

`matched` 表示某个人工可见 term 是否被新模型预测命中。q-function 训练不使用 `matched`，`matched` 只用于最终 precision / recall 估计。

## 运行教程

### 1. 验证两列表能否正确重建状态

在根目录直接运行：

```bash
python3 build_two_lists_and_validate.py
```

输出文件：

```text
data/simulated_list1.csv
data/simulated_list2.csv
data/expected_visible_from_simulation.csv
data/reconstructed_visible_states.csv
data/validation_report.json
```

如果 `keys_equal=true`、`state_equal=true`、`matched_equal=true`，说明两列表重建逻辑与模拟真值一致。

### 2. 运行完整模拟和 CRC 校正评估

```bash
python3 run_list_state_simulation.py
```

或：

```bash
python3 simulation_bootstrap_validation.py
```

主要输出：

```text
simulation_outputs/simulation_summary.json
simulation_outputs/data/train_list1.csv
simulation_outputs/data/train_list2.csv
simulation_outputs/data/test_list1.csv
simulation_outputs/data/test_list2.csv
simulation_outputs/data/train_reconstructed_visible.csv
simulation_outputs/data/test_reconstructed_visible.csv
simulation_outputs/models/pubmedbert/q_function.pt
simulation_outputs/models/pubmedbert/train_summary.json
simulation_outputs/plots/precision_hist.png
simulation_outputs/plots/recall_hist.png
simulation_outputs/plots/summary_barplot.png
```

看结果时优先打开：

```text
simulation_outputs/simulation_summary.json
simulation_outputs/plots/summary_barplot.png
```

三个图的含义：

| 图像 | 含义 |
| --- | --- |
| `precision_hist.png` | bootstrap 下 CRC 校正 precision 与朴素 precision 的分布，并用虚线标出 full truth。 |
| `recall_hist.png` | bootstrap 下 CRC 校正 recall 与朴素 recall 的分布，并用虚线标出 full truth。 |
| `summary_barplot.png` | full truth、naive visible mean、CRC corrected mean 的 precision / recall 柱状对比。 |

### 3. 用自己的两份人工列表和模型预测评估

输入格式：

`list1_df` 和 `list2_df` 必须包含：

```text
doc_id, phrase, type, context
```

`model_df` 必须包含：

```text
doc_id, phrase, type
```

最小示例：

```python
import pandas as pd
from evaluate_two_lists_model import evaluate_two_lists_with_model

list1_df = pd.DataFrame([
    {
        "doc_id": 1,
        "phrase": "hypertension",
        "type": "Disease or Syndrome",
        "context": "The patient has hypertension.",
    }
])

list2_df = pd.DataFrame([
    {
        "doc_id": 1,
        "phrase": "hypertension",
        "type": "T047",
        "context": "The patient has hypertension.",
    }
])

model_df = pd.DataFrame([
    {
        "doc_id": 1,
        "phrase": "hypertension",
        "type": "Disease or Syndrome",
    }
])

summary = evaluate_two_lists_with_model(
    list1_df=list1_df,
    list2_df=list2_df,
    model_df=model_df,
    output_dir="evaluation_outputs",
    method="character",
)

print(summary)
```

输出目录：

```text
evaluation_outputs/data/q_training_visible_terms.csv
evaluation_outputs/data/evaluation_visible_terms.csv
evaluation_outputs/data/model_human_matches.csv
evaluation_outputs/models/q_function/q_function.pt
evaluation_outputs/models/q_function/train_summary.json
evaluation_outputs/estimate/estimate_two_list_pubmedbert.json
evaluation_outputs/evaluation_summary.json
```

如果已有训练好的 q-function，可以跳过重新训练：

```python
from evaluate_two_lists_model import QFunction, evaluate_two_lists_with_model

q_function = QFunction.load("simulation_outputs/models/pubmedbert/q_function.pt")

summary = evaluate_two_lists_with_model(
    list1_df=list1_df,
    list2_df=list2_df,
    model_df=model_df,
    q_function=q_function,
    pred_total=len(model_df),
)
```

## 匹配方式

### 字符匹配

默认方式是 `method="character"`。它会做大小写、空格、简单标点和部分 UMLS 语义类型规范化。

例如在 `evaluate_two_lists_model.py` 中：

```text
hypertension + T047
hypertension + Disease or Syndrome
```

会被认为是同一类型并可合并为 `state=11`。

### AI 匹配

当需要识别同义词、缩写、改写时，可以使用 `method="ai"`，并传入一个 `ai_matcher(prompt)` 函数。

```python
def ai_matcher(prompt: str) -> str:
    # 调用你的大模型服务，返回 JSON 字符串
    return '{"matches": [{"h_idx": 0, "g_idx": 0, "phrase_match": true, "type_match": true}]}'

summary = evaluate_two_lists_with_model(
    list1_df=list1_df,
    list2_df=list2_df,
    model_df=model_df,
    method="ai",
    ai_matcher=ai_matcher,
)
```

AI 返回格式必须是 JSON：

```json
{
  "matches": [
    {
      "h_idx": 0,
      "g_idx": 0,
      "phrase_match": true,
      "type_match": true
    }
  ]
}
```

只有 `phrase_match=true` 的配对会被使用。

## 注意事项

1. 根目录和 `CRCModel_evaluation_full_review/` 目录仍有重复代码；根目录现在已经可以独立运行，后续建议以根目录为主。
2. `evaluate_two_lists_model.py` 中的两列表合并会规范化 `T047` 与 `Disease or Syndrome` 这类类型；`model_term_matching.py` 当前走到的重建路径可能不会把这两个类型合并，因此小样例会出现 `10` 和 `01` 两行。这一点在实际评估时建议优先使用 `evaluate_two_lists_model.py` 的一站式 API。
3. 完整训练依赖本地 PubMedBERT 缓存；如果模型路径不存在，训练会失败，需要先准备本地模型。
4. 如果模型真实预测总数不等于 `len(model_df)`，请手动传入 `pred_total`，否则 precision 的分母会不准确。
