# CRC 估计器比较实验

## 1. 实验目的

这个实验回答两个问题：

1. 完整 CRC 中的 inclusion–exclusion 是否带来过多方差？
2. 误差来自估计器结构，还是来自 q-function 对捕获概率的估计？

两个 annotation 先合并成 merged ground truth。随后，每个术语

\[
z_i=(doc\_id_i, phrase_i, type_i, context_i)
\]

按 `two_annotations_pseudo_truth.py` 顶部定义的 \(p_1(z_i),p_2(z_i)\) 独立进入两个模拟列表。模型预测保持不变，默认重复 1000 次。

## 2. 四种比较方法

令 \(R_{1i},R_{2i}\in\{0,1\}\) 表示术语是否进入两个列表，\(R_{12i}=R_{1i}R_{2i}\)。q-function 在至少出现一次的术语上估计：

\[
q_{1i}=P(R_{1i}=1\mid R_{1i}+R_{2i}>0,z_i),
\]

\[
q_{2i}=P(R_{2i}=1\mid R_{1i}+R_{2i}>0,z_i),
\]

\[
q_{12i}=P(R_{1i}=R_{2i}=1\mid R_{1i}+R_{2i}>0,z_i).
\]

### A. Naive

只使用两个列表的可见并集：

\[
w_i^{naive}=1.
\]

它的方差通常较小，但遗漏两个列表都没有发现的术语，因此 precision 可能被低估。

### B. 完整 CRC

现有算法使用：

\[
\hat\pi_i=\frac{q_{12i}}{q_{1i}q_{2i}},
\]

\[
w_i^{CRC}=
\frac{R_{1i}/q_{1i}+R_{2i}/q_{2i}-R_{12i}/q_{12i}}
{\hat\pi_i}.
\]

括号中的 inclusion–exclusion 项对状态 `10`、`01`、`11` 给出不同权重。它利用的信息更多，但也会增加方差。

### C. 仅 q-ratio 的稳定化修正

去掉 inclusion–exclusion，只保留对“两个列表都遗漏”的修正：

\[
w_i^{ratio}=\frac{1}{\hat\pi_i}
=\frac{q_{1i}q_{2i}}{q_{12i}}.
\]

在给定 \(z\) 后两个列表独立时，\(\hat\pi_i\) 可解释为术语至少出现一次的概率。这个方法预期方差较低，但 q-function 有系统误差或列表条件独立假设不成立时，可能产生偏差。

### D. Oracle union-HT

模拟中已知真实的 \(p_1(z),p_2(z)\)，因此可以计算：

\[
\pi_i^{oracle}=1-(1-p_1(z_i))(1-p_2(z_i)),
\]

\[
w_i^{oracle}=\frac{1}{\pi_i^{oracle}}.
\]

Oracle 不能直接用于真实数据，但可以作为模拟基准。如果 Oracle 明显优于 q-ratio，主要问题在 q-function；如果两者相近，q-function 已较好地恢复并集捕获概率。

## 3. 三个指标

令 \(Y_i=1\) 表示术语与模型预测在 phrase 和 type 上都匹配，\(P_i=1\) 表示 phrase 匹配，\(M\) 是模型预测总数。对任意权重 \(w_i\)：

\[
\widehat{Precision}=\frac{\sum_i w_iY_i}{M},
\]

\[
\widehat{Recall}=\frac{\sum_i w_iY_i}{\sum_i w_i},
\]

\[
\widehat{TypeAccuracy}=\frac{\sum_i w_iY_i}{\sum_i w_iP_i}.
\]

## 4. 比较标准与图片

对每个方法和指标计算：

\[
Bias=E(\hat\theta)-\theta,
\qquad
SD=\sqrt{Var(\hat\theta)},
\]

\[
RMSE=\sqrt{Bias^2+Var(\hat\theta)}.
\]

生成三类新图片：

- `estimator_comparison.png`：四种方法的均值和 95% Monte Carlo 分位区间。
- `interval_coverage.png`：用每种方法的 bootstrap SD 构造正态区间，比较 nominal coverage 和 observed coverage。这是固定模拟数据上的 Monte Carlo 校准诊断，不等同于真实研究中的嵌套 bootstrap 置信区间。
- `type_capture_coverage.png`：按术语类别显示 List 1、List 2 和并集的平均捕获率及 95% 区间；黑色 `x` 是 \(p(z)\) 给出的理论期望。

为了避免小样本类别主导图片，类别 coverage 图默认只显示 ground truth 中至少 10 个术语的类别。

## 5. 运行

```bash
python3 two_annotations_pseudo_truth.py --bootstrap 1000
```

主要输出：

```text
simulation_outputs/two_annotations_pseudo_truth_p_z/CRC_metrics_summary.json
simulation_outputs/two_annotations_pseudo_truth_p_z/plots/estimator_comparison.png
simulation_outputs/two_annotations_pseudo_truth_p_z/plots/interval_coverage.png
simulation_outputs/two_annotations_pseudo_truth_p_z/plots/type_capture_coverage.png
```

代码不会输出逐次 bootstrap CSV，也不会生成逐行删除审计记录。

## 6. 本次 1000 次 bootstrap 结果

Merged ground truth 的真实值为：precision `0.71520`、recall `0.55946`、type accuracy `0.85422`。

### Precision

| 方法 | Mean | Bias | SD | RMSE | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full CRC | 0.71225 | -0.00296 | 0.00991 | 0.01033 | 0.932 |
| q-ratio only | 0.71173 | -0.00347 | 0.00795 | 0.00867 | 0.925 |
| Oracle union-HT | 0.71481 | -0.00040 | 0.00815 | 0.00816 | 0.946 |
| Naive | 0.68763 | -0.02757 | 0.00772 | 0.02863 | 0.061 |

q-ratio only 相对 Full CRC 把 precision 的 SD 降低约 20%，但绝对偏差略有增加；总体 RMSE 从 `0.01033` 降到 `0.00867`。Oracle 的偏差最小，说明 q-function 对并集捕获概率仍有少量校准误差。Naive 虽然 SD 最小，但 precision 严重低估，95% coverage 只有 `0.061`。

### Recall

| 方法 | Mean | Bias | SD | RMSE | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full CRC | 0.55963 | +0.00017 | 0.00545 | 0.00545 | 0.950 |
| q-ratio only | 0.56016 | +0.00070 | 0.00428 | 0.00433 | 0.950 |
| Oracle union-HT | 0.55934 | -0.00013 | 0.00439 | 0.00439 | 0.952 |
| Naive | 0.56022 | +0.00075 | 0.00430 | 0.00437 | 0.949 |

四种 recall 都接近真实值。原因是缺失术语同时影响 recall 的加权分子和分母，当前数据中的误差发生了较强抵消。q-ratio only 比 Full CRC 的 SD 低约 22%。

### Type accuracy

| 方法 | Mean | Bias | SD | RMSE | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full CRC | 0.85417 | -0.00005 | 0.00475 | 0.00475 | 0.950 |
| q-ratio only | 0.85532 | +0.00110 | 0.00378 | 0.00393 | 0.940 |
| Oracle union-HT | 0.85398 | -0.00024 | 0.00390 | 0.00391 | 0.953 |
| Naive | 0.85449 | +0.00027 | 0.00382 | 0.00383 | 0.950 |

Full CRC 的 type accuracy 偏差最小，但方差最大。Naive 在这个指标上表现很好，表示当前模拟中“是否被捕获”与“phrase 匹配后 type 是否正确”的关系较弱；这不是所有数据都必然成立。

### Coverage 图解释

- `estimator_comparison.png` 中，除 naive precision 外，其余 95% Monte Carlo 区间都覆盖真实值。
- `interval_coverage.png` 中，Full CRC、q-ratio only 和 Oracle 的 precision 曲线接近理想对角线；naive precision 因系统偏差明显欠覆盖。Recall 和 type accuracy 的四条曲线基本接近理想线。
- `type_capture_coverage.png` 中，medication、lab、vital sign 的理论并集捕获率约为 `0.9875`；risk factor 和 anatomy 约为 `0.8875`。1000 次模拟均值与黑色 `x` 基本重合，说明代码正确实现了类别难度不同的 \(p(z)\)。小类别的区间更宽，这是样本量导致的正常随机波动。

本次结果支持把 q-ratio only 作为低方差候选方案，但不能据此直接替换 Full CRC：当前模拟满足给定 \(z\) 后独立捕获，而真实的两个 annotator 可能相关。正式选择前应优先运行下一节的相关捕获情景。

## 7. 后续更严格的比较

当前主实验假定给定 \(z\) 后两个列表独立。建议后续增加以下实验：

1. **相关捕获情景**：加入共享的术语难度 \(U_i\)，例如

   \[
   P(R_{ji}=1\mid z_i,U_i)=logit^{-1}(\alpha_j(z_i)+\lambda_jU_i).
   \]

   改变 \(\lambda_j\) 可以比较独立、弱相关和强相关列表，检验 \(q_{12}/(q_1q_2)\) 假设失效时的稳健性。

2. **q-function 校准**：按 type 绘制预测捕获概率与实际捕获频率的 reliability plot，并报告 Brier score。

3. **删除率和样本量网格**：改变两个基础删除率和文档数，绘制 bias–variance–RMSE 曲面。

4. **权重截断敏感性**：比较不同最大权重，观察偏差增加与方差下降的权衡。

5. **传统 capture–recapture**：加入总体 Chapman 和按 type 分层 Chapman。它们实现简单，但同质捕获假设较强，可作为传统基线而非默认方法。

6. **文档级 bootstrap**：真实数据报告置信区间时，应按 `doc_id` 聚类重采样，避免把同一文档中的术语错误地当作独立样本。
