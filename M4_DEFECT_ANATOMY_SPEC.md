# M4 缺损约束与解剖先验实现规格

## 0. 文档状态与适用范围

- **阶段**：M4 design freeze
- **基线**：已由 tag `m3-baseline-complete` 封存的 M3 Point–CT baseline
- **文档性质**：后续 M4 分阶段实现、测试、审查、审计与验收的规范性依据
- **规范词**：本文中的“必须”“不得”“禁止”“仅允许”均为强制要求；“建议”“推荐”表示实现优先选择，若偏离必须在对应子阶段先更新并冻结规格

本文件只冻结 M4 的边界、接口语义、组合关系、退化行为和实施顺序，不实现 Python 功能，也不替尚未冻结的子阶段设计作决定。

---

## 1. M4 总目标

M4 从已封存的 M3 Point–CT baseline 出发，按以下三个增量实验层级引入缺损信息与解剖先验：

| 层级 | 新增能力 |
| --- | --- |
| B1 | defect hard constraint |
| B2 | defect hard constraint + defect soft modulation |
| B3 | defect hard constraint + defect soft modulation + anatomical prior |

M4 必须继续复用 M3 已有的以下主链路与既有语义：

- Point encoder
- CT encoder
- similarity
- rectangular Sinkhorn + dustbin
- mutual matching filter
- weighted Procrustes/SVD

M4 的新增能力必须位于明确、独立、可关闭的模块边界内，不得以引入缺损约束或解剖先验为由改写 M3 encoder、匹配后处理、配准或正式评估语义。

本阶段明确不实现：

- entropy reliability
- residual feedback
- registration–fusion iterative loop
- M5/M6 内容
- RANSAC
- ICP
- GeometricTransformer
- LGR

## 2. 掩码统一语义

### 2.1 缺损掩码的唯一编码

M4 全链路统一采用：

- `1` / `True`：intact / valid anatomical region
- `0` / `False`：defect region

Point 原始输入必须使用：

```text
point_defect_mask: [N_point] bool
```

CT 原始输入或支持域输入必须使用 `ct_defect_mask`，并且必须通过显式、可验证的映射，最终得到与 CT matching support 一一对齐的布尔 mask。原始 `ct_defect_mask` 的最终 shape 由其数据来源与支持域决定，留待对应子阶段冻结；不得在本规格阶段假设其原始分辨率或存储布局。

### 2.2 padding validity 与 anatomical defect 必须独立

以下两类 mask 的语义严格分离：

| mask | 唯一职责 |
| --- | --- |
| `point_valid_mask` / `ct_valid_mask` | 描述张量元素是否有效，或是否属于 padding |
| `point_defect_mask` / `ct_defect_mask` | 描述解剖区域是 intact 还是 defect |

不得用 defect mask 改写 padding valid mask，也不得通过 padding 行为间接表达 defect 状态。两类 mask 必须使用独立变量、独立接口参数和独立验证逻辑；任何实现均不得复用同一张量承载两种语义。

## 3. Coarsest-level mask contract

M3 实际执行匹配的 coarsest-level 表示为：

```text
Q: [Np, d]
K: [Nv, d]
```

因此，M4 在进入 matching 前必须显式构造：

```text
point_intact_coarse: [Np] bool
ct_intact_coarse:    [Nv] bool
```

二者必须分别与以下物理坐标严格一一对应：

```text
point_physical: [Np, 3]
ct_physical:    [Nv, 3]
```

最低验收条件如下：

1. dtype 必须为 `bool`，shape 必须精确等于对应 matching support 的 token 数。
2. token 顺序、采样身份和物理坐标索引必须来自同一份显式映射关系。
3. 映射必须具备可测试的来源与确定性；不得根据 shape 或长度推测索引对应关系。
4. 禁止自动截断、隐式广播、循环补齐、默认全 intact，或任何“最接近可用 shape”的容错。
5. 映射缺失、长度不等、索引越界、dtype 错误、来源不明确或对齐验证失败时，必须 fail closed：立即停止该样本/批次进入 M4 matching，并报告可定位的错误。

dataset-level 的 point_defect_mask / ct_defect_mask 来源、命名、原始 shape 与基础数据契约由 M4-1A 冻结；从原始 defect mask 到 point_intact_coarse / ct_intact_coarse 的 coarsest-level 映射算法、索引关系与对齐验证均由 M4-1B 冻结。

## 4. B1：defect hard constraint

### 4.1 模块边界

B1 只实现缺损硬约束。它必须是 encoder 之后、Sinkhorn 接口之前的独立模块，不得修改 Point encoder 或 CT encoder 本身。

模块输入至少包括：

```text
similarity_or_logits: [Np, Nv]
point_valid_mask:     [Np] bool
ct_valid_mask:        [Nv] bool
point_intact_coarse:  [Np] bool
ct_intact_coarse:     [Nv] bool
```

若 M3 的正式接口包含 batch 维，上述 shape 在最前方增加一致的 batch 维，但逐样本契约不变。

模块必须输出供既有 rectangular Sinkhorn 消费的 hard-constrained logits 和/或显式 matching masks。输出至少应能无歧义地区分：

- M3 原有的 padding-valid token；
- 可进入普通 real-to-real matching 的 intact pair；
- 因 defect 状态而禁止成为普通 correspondence 的 pair；
- 仍可进入 dustbin/unmatched 路径的有效 token。

### 4.2 硬约束定义

定义有效 real-to-real pair：

```text
V_ij = point_valid_mask[i] AND ct_valid_mask[j]
```

定义 intact real-to-real pair：

```text
H_ij = point_intact_coarse[i] AND ct_intact_coarse[j]
```

B1 中普通 correspondence 的准入条件必须为：

```text
E_ij = V_ij AND H_ij
```

因此，下列三类配对均不得作为普通、可靠的 real-to-real correspondence：

- defect-to-intact
- intact-to-defect
- defect-to-defect

实现时必须通过 Sinkhorn 所支持的显式 mask 或等价的安全负无穷语义禁止这些 real-to-real pair。若采用有限哨兵值，其数值范围、dtype 行为和归一化后零质量性质必须在 M4-2 先通过数值测试冻结，不得在本规格阶段任意选定。

### 4.3 与 Sinkhorn dustbin 的关系

defect token 不是 padding token。对于在 `point_valid_mask` / `ct_valid_mask` 中仍然有效的 defect token：

1. 它与所有 real token 的普通 matching edge 必须被硬禁止。
2. 它必须保留既有 Sinkhorn 设计中的 dustbin/unmatched 可达路径，使不可匹配质量能够进入 dustbin。
3. 不得通过把 defect token 标为 padding 来“实现”上述效果。
4. 不得改变 M3 dustbin 的参数语义、归一化契约或后续 mutual filtering 语义。
5. hard constraint 必须在 dustbin 扩展与有效性处理的明确位置执行；具体接入点由 M4-2 基于 M3 当前接口冻结并用单元测试证明。

硬约束具有最高优先级：任何后续 soft modulation 或 anatomical prior 都不得重新启用被 B1 禁止的 real-to-real pair。

### 4.4 关闭行为

当 `use_defect_hard_constraint=False` 时，该模块必须严格旁路，向后续链路提供与封存 M3 相同的 logits、valid-mask 和 dustbin 输入语义。不得仅做到“数值接近”；除 M3 本身允许的确定性/浮点容差外，不得新增裁剪、填充值替换、重新归一化或 mask 变换。

## 5. B2：soft defect modulation

### 5.1 目的与独立性

B2 在 B1 已生效的前提下，对仍被允许的 intact-to-intact real pair 增加 soft modulation。soft modulation 与 hard constraint 必须是两个独立模块、两个独立开关，以支持 B1/B2 消融；soft 模块不得承担 hard exclusion 职责。

定义与 coarsest support 一一对应的 defect-aware representation / reliability gate：

```text
D_p: Point-side defect-aware representation, aligned with [Np]
D_v: CT-side defect-aware representation, aligned with [Nv]
```

`D_p`、`D_v` 的具体维度、网络结构、监督方式和参数共享策略属于 M4-3 的待冻结内容，本文件不作决定。无论采用何种结构，模块都必须产生：

```text
r_p: [Np], each value in (0, 1]
r_v: [Nv], each value in (0, 1]
C:   [Np, Nv], each value in (0, 1]
```

### 5.2 推荐的数值稳定公式

推荐采用可分解的 pair-wise reliability：

```text
r_p[i] = clamp(sigmoid(g_p(D_p[i])), epsilon, 1)
r_v[j] = clamp(sigmoid(g_v(D_v[j])), epsilon, 1)
C_ij   = clamp(r_p[i] * r_v[j], epsilon, 1)
```

其中 `epsilon`、`g_p`、`g_v` 和调制强度均须仅使用 train/validation 在 M4-3 冻结。对 B1 允许的 real-to-real pair，推荐在 logit/log-probability 域执行加性调制：

```text
L_B2[i,j] = L_B1[i,j] + lambda_D * log(C_ij)
```

其中 `lambda_D >= 0`。由于 `C_ij in (0,1]`，该项只会保持或降低普通 pair 的相对可信度，并避免对已经归一化的 Sinkhorn 概率进行直接乘法。不得在 Sinkhorn 输出概率上追加未经重新推导的乘法调制。

硬约束必须优先且最终生效：上述公式仅在 `E_ij=True` 的 real-to-real pair 上计算/写入；`E_ij=False` 的 pair 始终保持禁止状态。默认不得调制 dustbin edge 或 dustbin 参数；若未来需要改变，必须先另行更新规格并证明不破坏 M3 dustbin 语义。

### 5.3 fail-closed 与退化行为

- `D_p`、`D_v`、`r_p`、`r_v` 或 `C` 与 coarsest support 不对齐时必须报错并停止，不得广播或截断。
- 出现 NaN、Inf、越界 reliability 或非正 `C_ij` 时必须报错并停止，不得静默替换。
- `use_defect_soft_modulation=False` 时不得构造会改变结果的 soft 项，且必须精确退化到 B1。
- `use_defect_soft_modulation=True` 而 `use_defect_hard_constraint=False` 属于非法配置，必须在模型/配置初始化阶段 fail closed。

## 6. B3：CT-side anatomical prior

### 6.1 定义与允许用途

B3 在 B2 基础上增加只与 CT coarsest support 一一对应的 anatomical prior：

```text
A_v / W_A: [Nv]
```

`A_v` 可表示生成 prior 所需的 CT-side anatomical information，`W_A` 表示最终用于 matching 的权重；若实现同时保留二者，则每个输出都必须声明 shape，并保证最终权重与 `[Nv]` support 严格对齐。

anatomical prior 唯一允许的作用是增强稳定、可信的 CT 解剖区域对 matching 的贡献。它不得：

- 直接或间接读取 GT transform；
- 使用 test performance 生成、选择或调节 prior；
- 根据 test subject 身份生成 subject-specific prior；
- 将任何 test 信息泄漏到训练、验证、模型选择或超参数冻结过程。

### 6.2 输入、输出和组合契约

允许输入仅限于在推理时合法可获得、且不包含上述泄漏信息的 CT-side 数据或由其确定性导出的特征/元数据。具体输入集合和 stable-region 生成方式必须在 M4-4 使用 train/validation 单独冻结；本文件不指定具体解剖区域或阈值。

最低输出契约为：

```text
W_A: [Nv], finite, W_A[j] >= 1, aligned one-to-one with ct_physical and K
```

其中 `W_A[j]=1` 表示不施加 anatomy 增强，只有经冻结规则认定为稳定、可信的 CT 区域才允许取 `W_A[j]>1`。推荐将 `W_A` 参数化/规范化为稳定的正权重，并在 logit 域仅对 B1 允许的 real-to-real pair 进行 CT-column modulation：

```text
L_B3[i,j] = L_B2[i,j] + lambda_A * log(W_A[j])
```

`W_A` 的上界、规范化方式和 `lambda_A >= 0` 的具体取值由 M4-4 冻结。所选参数化必须保证数值有限、适合 Sinkhorn，且不能使 hard-forbidden pair 恢复。若最终采用与推荐式不同但数学等价或更稳定的组合形式，必须先在 M4-4 规格中记录理由、退化证明与数值测试。

### 6.3 fail-closed 与无 prior 退化

- prior 缺失、shape/dtype 不符、support 对齐无法证明，或包含 NaN、Inf、非正权重时，B3 必须 fail closed。
- 禁止通过截断、广播、按 shape 猜测索引或默认生成 subject-specific 权重来修复 prior。
- 当 `use_anatomical_prior=False` 时，prior 分支必须完全旁路并严格退化到 B2。
- 当 `use_anatomical_prior=True` 但 soft modulation 未启用时，配置非法，必须在模型/配置初始化阶段 fail closed。
- anatomical prior 必须具有独立开关，确保 B2 与 B3 可直接比较。

## 7. 模块开关与合法组合

M4 必须提供三个显式布尔配置：

```text
use_defect_hard_constraint
use_defect_soft_modulation
use_anatomical_prior
```

冻结的合法组合至少包括：

| 实验层级 | hard | soft | anatomy |
| --- | --- | --- | --- |
| B0 | `False` | `False` | `False` |
| B1 | `True` | `False` | `False` |
| B2 | `True` | `True` | `False` |
| B3 | `True` | `True` | `True` |

当前规格下，以下组合非法：

- `soft=True` 且 `hard=False`
- `anatomy=True` 且 `soft=False`

由依赖关系可知，任何 `anatomy=True` 且 `hard=False` 的组合同样非法。非法组合必须在初始化或配置验证的最早阶段报错，不得静默改写开关，也不得自动补开依赖项。除非未来先正式修改并重新冻结本规格，否则不得扩展这些组合。

## 8. M3 backward-compatibility contract

M4 必须增加 regression tests，证明三个 M4 开关全部关闭时（B0）与封存的 M3 行为一致。回归范围至少覆盖：

- similarity shape 与值传递契约；
- Sinkhorn input/output contract；
- dustbin semantics；
- mutual filtering；
- correspondence physical coordinates；
- weighted registration input/output。

B0 路径不得要求 defect 数据存在，不得实例化会改变随机数状态或数值路径的 M4 分支，也不得改变 M3 的正常输入输出结构。具体逐项容差、fixture 和 golden reference 在相应实现子阶段冻结，但其参考对象必须是 tag `m3-baseline-complete` 所封存的 M3 行为。

禁止为了 M4 修改：

- M3 formal evaluation thresholds
- M3 checkpoint
- M3 test results
- M3 protocol hash
- M3 evaluation protocol hash

同样禁止修改 M3 baseline 的既有训练、matching、Sinkhorn、mutual filtering、weighted registration 和正式 evaluation 语义。

## 9. 数据泄漏限制

M3 的正式 test 已经执行并封存。M4 的开发、训练决策、模型选择、阈值选择与全部超参数冻结只能使用 train / validation。

禁止针对以下正式 test subjects 的表现进行任何 subject-specific 调整：

```text
Pat12, Pat6, Pat5,
Pat7, Pat4,
Pat11, Pat3,
Pat8, Pat9,
Pat1, Pat2
```

限制包括但不限于：缺损生成参数、mask 映射规则、reliability gate、anatomical region、prior 权重、阈值、checkpoint 选择和停止条件。

已有 M3 test 结果只允许作为 B0 的最终报告结果，不得作为 M4 模型选择、模块设计、超参数调节或错误分析后定向修改的依据。M4 正式 evaluation 前必须能够审计证明所有设计冻结均只依赖 train/validation。

## 10. 冻结的实现顺序与阶段门

M4 必须严格按以下顺序推进：

1. **M4-1A**：dataset-level defect mask contract
2. **M4-1B**：coarsest Point / CT mask mapping contract
3. **M4-2**：B1 hard constraint
4. **M4-3**：B2 soft modulation
5. **M4-4**：B3 anatomical prior
6. **M4-5**：training / validation / ablation protocol
7. **M4-6**：formal evaluation and sealing

每一步都必须独立完成以下阶段门：

```text
implementation
→ unit tests
→ review diff
→ audit
→ commit
→ push
→ server validation
```

前一步未完成全部阶段门时，不得进入下一步；不得跨步骤一次性实现全部 M4。每个子阶段只允许冻结和实现其职责范围内的内容，必须保留可审查的独立 diff、测试证据和审计记录。

## 11. 当前未解决问题

以下问题在当前设计冻结阶段明确保持未解决，必须在对应子阶段分别研究、评审并冻结，不得在实现中隐式决定或自行猜测：

### A. defect data 来源与命名

M4 使用的 defect data 的最终来源及文件命名尚未确定，应在 M4-1A 冻结。

### B. Point raw-to-coarsest 映射

raw defect mask 如何从原始分辨率映射到 KPConv coarsest points 尚未确定，应在 M4-1B 冻结。

### C. CT mask-to-support 映射

CT defect mask 如何映射到 20 mm CT support 尚未确定，应在 M4-1B 冻结。

### D. soft modulation 结构与权重

soft modulation 的具体网络结构与权重尚未确定，应在 M4-3 仅基于 train/validation 冻结。

### E. anatomical prior 生成方式

anatomical stable-region prior 的具体生成方式尚未确定，应在 M4-4 仅基于 train/validation 冻结。

### F. train/validation defect generation protocol

M4 train/val defect generation protocol 尚未确定，应在 M4-5 冻结。

在上述问题完成各自子阶段的正式冻结前，任何代码不得以默认值、启发式、subject identity、test 结果或隐式数据约定替代明确决策。

## 12. 后续实现的最低验收原则

后续各子阶段除满足自身新增测试外，还必须共同满足：

1. **对齐可证**：所有 defect/anatomy 信息与 matching support 的对应关系显式、确定且可测试。
2. **语义隔离**：padding validity、defect state、soft reliability、anatomical prior 各自独立。
3. **硬约束优先**：soft modulation 与 anatomy 不得恢复 hard-forbidden correspondence。
4. **逐级严格退化**：B3 关闭 anatomy 后退化到 B2；B2 关闭 soft 后退化到 B1；全部关闭后退化到封存 M3。
5. **失败关闭**：缺失、错位、非法配置或非有限数值不得静默通过。
6. **无 test 泄漏**：所有 M4 开发与冻结决策只依赖 train/validation。
7. **M3 不变**：M3 baseline、协议、结果和正式 evaluation 语义保持封存状态。
