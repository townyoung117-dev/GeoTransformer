# Point Cloud–CT Baseline V1 Specification

## 1. 目标

基于 GeoTransformer 官方工程构建一个用于三维人脸点云与 CT 体数据刚性配准的基础模型 Baseline V1。

Baseline V1 的目的仅为建立稳定、可训练、可评价的 Point Cloud–CT 跨模态刚性配准基础网络，为后续缺损解剖双约束、熵权残差反馈和置信度融合等创新模块提供统一实验基线。

Baseline V1 不实现最终专利完整模型。

---

## 2. Baseline V1 中禁止加入的模块

本阶段禁止提前实现或以占位逻辑接入：

* point defect mask（点云缺损掩码）；
* CT defect mask（CT 缺损掩码）；
* defect hard constraint（缺损硬约束）；
* defect-aware modulation（缺损感知软调制）；
* anatomical prior（解剖先验）；
* entropy reliability（熵权匹配可靠性）；
* residual feedback（残差特征反馈）；
* registration-fusion closed loop（配准—融合闭环迭代）；
* confidence-guided fusion（置信度引导融合）；
* 多维置信度融合；
* multimodal final fusion representation（最终多模态融合表示）。

这些模块只能在 Baseline V1 的 M1–M3 主链稳定后，从 M4 开始按开发顺序逐步加入。

---

## 3. 输入数据

每个样本必须包含两种模态及其 GT 刚性变换。

### 3.1 人脸点云

```text
point_xyz      : [N, 3]
point_normal   : [N, 3]
```

其中：

`point_xyz`：

* 表示点云三维物理坐标；
* 使用全流程统一的物理长度单位；
* 目标颌面数据预计统一为 mm，但最终单位仍须由真实数据元信息确认。

Dataset 原始字段可继续命名为 `point_xyz`，但其契约必须固定为 physical coordinate；一旦构造网络内部坐标，必须使用 `point_xyz_net`、`Xp_net` 或同等明确的名称，禁止覆盖 `point_xyz` 后改变其单位或坐标系。

`point_normal`：

* 表示对应表面的无量纲单位方向向量；
* 必须满足 `||n|| ≈ 1`，并在数据契约验证中检查有限性和归一化误差；
* 刚体变换时只进行旋转：`n' = R n`；
* 不施加平移。

点云初始 feature 是否以及如何使用 `point_normal`、是否包含绝对坐标，属于真实数据统计和 M2 接口验证后才能确定的参数；实现时必须与 KPConv 的几何坐标输入机制区分，禁止重复或错误编码。

### 3.2 CT

必须保留：

```text
ct_volume
ct_spacing
ct_origin
ct_direction
```

CT 坐标契约必须严格区分以下三个概念：

#### 3.2.1 `array_index`

`array_index` 表示程序中 CT array 的存储下标。某些读取结果可能按 `[z, y, x]` 排列，但具体顺序不得预设，必须依据真实数据格式和实际读取库验证。

#### 3.2.2 `image_index`

`image_index` 表示医学影像定义下的三维 voxel index：

```text
image_index = [i, j, k]
```

它与 `spacing`、`origin`、`direction` 共同定义物理空间位置。禁止默认：

```text
array[z, y, x] == image_index[x, y, z]
```

M1 必须建立明确且经过测试的 `array_index ↔ image_index` 映射。

#### 3.2.3 `physical_coordinate`

`physical_coordinate` 表示真实三维物理空间坐标。数学关系统一为：

```text
x_phys = origin + direction @ (image_index * spacing)
```

其中 `image_index`、`spacing`、`origin`、`direction` 和 `x_phys` 的向量/矩阵轴约定必须显式记录并保持一致；不得把 `array_index` 或 `image_index` 直接当作物理坐标。

`origin` 对应医学影像 `image_index=(0,0,0)` 定义的物理位置。若实际读取库对 voxel center / voxel corner 有特定定义，必须以该库的 metadata 契约为准并在 M1 验证。禁止在没有数据格式或读取库依据时自行加入或减去 `0.5 * spacing`，也禁止人为引入其他 half-voxel offset。

M1 必须建立人工可验证的 coordinate tests，至少覆盖：

```text
image_index = (0,0,0)
image_index = (1,0,0)
image_index = (0,1,0)
image_index = (0,0,1)
```

每个测试都必须验证：

```text
image_index → physical_coordinate
physical_coordinate → image_index
round-trip
```

测试集必须包含 `direction` 非 identity 的情况。

### 3.3 GT rigid transform

```text
gt_transform : [4, 4]
```

Baseline V1 统一规定：

```text
Point Cloud → CT physical coordinate system
```

即：

```text
x_ct_phys = R_gt @ x_point_phys + t_gt
```

任何训练、评价、SVD 和可视化代码都必须遵守同一方向。

### 3.4 Physical Coordinate 与 Network Coordinate

Baseline V1 强制区分物理坐标与网络内部坐标。

#### Physical coordinates

```text
Xp_phys ∈ R^(Np × 3)
Xv_phys ∈ R^(Nv × 3)
```

`Xp_phys` 表示与 `P_raw` 一一对应的 Point matching token 在 Point Cloud 物理坐标系中的坐标；`Xv_phys` 表示与 `V_raw` 一一对应的 CT matching token 在 CT 物理坐标系中的坐标。二者必须分别用于：

* GT transform；
* GT correspondence geometry；
* physical distance 和 positive radius；
* correspondence inlier evaluation；
* weighted Procrustes / weighted SVD；
* RRE / RTE；
* 最终可视化对齐。

最终 weighted SVD 必须使用：

```text
Xp_phys_corr
Xv_phys_corr
```

禁止直接使用经过归一化、中心化或尺度变换的网络坐标计算最终医学物理空间 `R,t`。

#### Network coordinates

如果 KPConv、Sparse 3D CNN、数据增强或数值稳定需要 centering、normalization、scaling、coordinate shifting 或 voxel quantization，可以构造：

```text
Xp_net
Xv_net
```

`Xp_net/Xv_net` 只用于网络内部计算，并必须满足：

1. 所有 physical → network 变换均显式保存参数和应用顺序；
2. 能够从 token 追踪或恢复对应的 physical coordinate；
3. 不改变 Point→CT GT transform 的医学物理含义；
4. correspondence token 始终能够索引回对应的 `Xp_phys/Xv_phys`；
5. 最终 weighted SVD、RTE 和物理距离评价必须回到 physical coordinate；
6. 不允许在任何接口中混用 `*_net` 与 `*_phys`。

未来坐标变量命名应使用 `*_phys`、`*_net` 或同等明确的后缀。禁止使用含糊的 `points`、`coords` 后在不同阶段改变坐标系或单位却不记录。

---

## 4. 数据接口

未来 Dataset 至少返回：

```text
subject_id
case_id
derived_sample_id
split

point_xyz
point_normal

ct_volume
ct_spacing
ct_origin
ct_direction

gt_transform
```

`subject_id`、`case_id` 和 `derived_sample_id` 必须保持概念区分；若一个 subject 只有一个原始 case，允许 `subject_id == case_id`。

数据集划分必须在原始 subject/case 级别完成，并以 subject 作为防泄漏的最高分组边界。同一 subject 的所有原始 case，以及同一原始病例生成的全部 rigid transform variants、defect variants、augmentation variants、derived point clouds 和 derived CT samples，必须属于同一个 split。禁止同一 subject 的任何派生样本跨越 train、validation 和 test。

M1 Dataset Contract 至少必须追踪 `subject_id`、`case_id`、`derived_sample_id` 和 `split`，并提供 subject/case-level leakage 检查。如果真实数据尚未定义正式 split，M1 只建立 metadata contract 和检查逻辑，不得自行随机创建最终论文 split。最终正式 split 必须固定、保存且可复现；未来应记录 `split_manifest`，若采用随机病例级划分还应记录 `split_seed`。本轮不创建 manifest 文件。

后续 CT active voxel / sparse tensor 可以在 Dataset、collate 或预处理模块中构造，但必须保存 `array_index ↔ image_index ↔ physical_coordinate` 的可追踪映射。

Point 和 CT 必须作为两个独立分支组织，禁止复用官方 `registration_collate_fn_stack_mode` 将二者伪装成两组点云后堆叠。未来的 Point–CT collate 至少应分别保存 point branch 输入、CT branch 输入、CT metadata 和 `gt_transform`。

Baseline V1 第一阶段保持：

```text
batch_size = 1
```

不得在模型稳定前修改 GeoTransformer 原工程的 batch 假设。

---

## 5. Point Cloud Encoder

点云支路基于 GeoTransformer 官方 KPConv 实现。

结构概念：

```text
Point Cloud
     ↓
KPConv / KPConvFPN
     ↓
multi-scale point feature
     ↓
select coarsest level
     ↓
P_raw [Np, 1024]
     ↓
point_proj
     ↓
Q [Np, 256]
```

允许复用 GeoTransformer：

```text
geotransformer/modules/kpconv/
```

以及必要的：

```text
grid subsampling
radius neighbor search
```

KPConv encoder 在匹配层级输出的原始点特征统一记为：

```text
P_raw ∈ R^(Np × Cp)
```

其中 `P_raw` 尚未进入跨模态公共特征空间。Baseline V1 初始优先使用官方 3DMatch `KPConvFPN` 的 coarsest level：

```text
P_raw shape ≈ [Np, 1024]
Cp = 1024
point_proj = Linear(1024, 256)
```

保留 FPN level-2 中间特征（约 `[Np_level2, 512]`）作为未来可配置备选，但当前 Baseline V1 不同时使用两个层级，也不在第一版中进行多层融合。

必须保存与 `P_raw` 每个 token 一一对应的三维物理坐标：

```text
Xp_phys ∈ R^(Np × 3)
```

如果 KPConv 使用 `Xp_net`，则 `Xp_net` 必须与 `P_raw`、`Xp_phys` 保持相同 token 索引，并满足第 3.4 节的可追踪契约。

---

## 6. CT Encoder

GeoTransformer 原工程不存在 CT encoder，因此必须新增独立 CT 支路。

概念结构：

```text
Full CT volume / ROI
        ↓
Sparse 3D CNN encoding
        ↓
context-aware CT features
        ↓
matching-token selection / support-domain definition
        ↓
CT matching tokens: Xv_phys + V_raw + valid_mask
```

Sparse 3D CNN 可以使用整个 CT volume / ROI 的内部体素信息进行 feature encoding，使 CT feature 能够表征 surface geometry、bone/internal structure、local density/intensity context 和 anatomical context。这里的 anatomical context 仅表示普通 CT encoder 从输入学习到的上下文，不引入第 2 节禁止的显式 anatomical prior。但是，进入 Point–CT correspondence matching 的 CT token 必须具有明确的三维物理坐标和跨模态几何意义；不得默认所有 CT 内部 voxel 都与人脸表面 point 存在一一 correspondence。

`matching-token selection / support-domain definition` 必须在 M2 根据真实 CT 数据和医学结构确定。当前规格不预先选择 surface voxel、bone boundary voxel、soft-tissue boundary voxel、all active voxel 或 hybrid token 中的任何方案。

Sparse 3D CNN encoder 在匹配层级输出的原始 CT token 特征统一记为：

```text
V_raw ∈ R^(Nv × Cv)
```

其中 `V_raw` 尚未进入跨模态公共特征空间，`Cv` 由 M2 选定的 Sparse 3D CNN 决定。

每个进入 matching 的有效 CT token 都必须包含并保持一一对应：

```text
Xv_phys ∈ R^(Nv × 3)
V_raw   ∈ R^(Nv × Cv)
valid_mask ∈ {False, True}^Nv
```

`Xv_phys` 必须来自 CT spacing、origin、direction 和经过验证的 `array_index ↔ image_index` 映射，不能直接使用 array/image index 替代，并且必须能够从 sparse feature token 追踪回原 CT physical coordinate。如果网络内部另建 `Xv_net`，必须遵守第 3.4 节契约。

具体 Sparse 3D CNN 库及网络深度在 M2 阶段确定，本规格阶段不由 Codex自行选择。

---

## 7. 公共特征空间

两侧 encoder 原始特征分别进行独立投影：

```text
P_raw ∈ R^(Np × Cp)
V_raw ∈ R^(Nv × Cv)

Q = point_proj(P_raw)
K = ct_proj(V_raw)

Q ∈ R^(Np × 256)
K ∈ R^(Nv × 256)
```

定义如下：

* `P_raw` 是 KPConv encoder 输出的原始点特征；
* `V_raw` 是 Sparse 3D CNN encoder 输出的原始 CT token 特征；
* `point_proj` 和 `ct_proj` 使用独立参数；
* `Q`、`K` 才是跨模态公共特征空间表示；
* Point encoder 与 CT encoder 不共享参数；
* `d=256` 作为 Baseline V1 的固定初始设计。

公共特征只允许由 `point_proj/ct_proj` 各执行一次独立 projection；不得把已经投影到 256 维的表示再次当作 encoder raw feature 重复投影。

---

## 8. Baseline Cross-modal Matching

Baseline V1 采用普通跨模态特征相似度，不加入缺损、解剖或熵权机制。

默认先进行特征归一化，再使用 temperature-scaled cosine similarity：

```text
Q_hat = L2Normalize(Q)
K_hat = L2Normalize(K)

S = (Q_hat K_hat^T) / tau
```

其中：

```text
S ∈ R^(Np × Nv)
```

考虑到显存限制，实际匹配应在 encoder 的 coarse/downsampled feature level 上进行，不允许直接对原始数万个点和所有 CT voxel 构造超大完整矩阵。

具体 coarse point/token 数量应根据数据统计后确定。

`tau` 必须为正数且配置化，具体值通过验证集确定。未来允许尝试 learnable temperature，但 Baseline V1 规格阶段不确定其具体数值。禁止在 L2 normalization 后再次固定除以 `sqrt(d)`；该组合会显著压缩 similarity logits，不作为 Baseline 默认定义。

---

## 9. Sinkhorn / Optimal Transport

Baseline V1 复用 GeoTransformer：

```text
LearnableLogOptimalTransport
```

Sinkhorn 契约统一定义为：

1. 输入为跨模态 similarity matrix `S` 和两侧有效位置 mask；
2. 输入 shape 为 `[B, Np, Nv]`，允许 `Np != Nv`；
3. Point 和 CT 两侧必须各自至少存在一个有效 token，否则直接报告失败；
4. 保留 dustbin row 和 dustbin column；
5. 输出为 `[B, Np+1, Nv+1]` 的 log matching scores；
6. 普通 Point–CT correspondence 的 transport confidence 定义为 `exp(log_matching_score)` 后的 OT assignment score；
7. mutual matching 时允许任一普通 token 的最优目标为 dustbin；
8. dustbin correspondence 不进入 weighted SVD；
9. 如果筛选后的有效 correspondence 不足或几何退化，registration 必须返回显式 failure flag，不得伪造 `R,t`。

该 OT assignment score 用于 correspondence ranking、confidence threshold 和 weighted Procrustes weight，但不得在没有专门 probability calibration 的情况下解释为 statistically calibrated probability。其绝对数值可能受 `Np`、`Nv`、dustbin、temperature、Sinkhorn normalization 和 token density 影响。

因此 confidence threshold、Sinkhorn iteration 和 dustbin `alpha` 初始值必须配置化，并通过 validation set 确定，不能直接复制 3DMatch 的固定值。`alpha` 按 `LearnableLogOptimalTransport` 的现有契约作为可学习参数参与训练。

Baseline 阶段不加入：

```text
defect mask
anatomical prior
entropy reliability
```

---

## 10. Correspondence

Baseline V1 推理主链固定为：

```text
global coarse similarity
      ↓
Sinkhorn
      ↓
dustbin competition
      ↓
mutual correspondence
      ↓
confidence filtering
      ↓
weighted SVD
```

从匹配矩阵获得候选 Point–CT correspondence：

```text
Point physical coordinate Xp_phys
↔
CT physical coordinate Xv_phys
```

每组 correspondence 至少包含：

```text
Xp_phys_corr
Xv_phys_corr
transport_confidence
```

Baseline V1 可采用：

```text
confidence threshold
+
mutual matching
```

进行基础筛选。

具体 threshold 不应在编码时任意固定，必须配置化并通过验证集确定。

必须先让普通 token 与 dustbin 竞争，再排除 dustbin row/column；不得先删除 dustbin 再做 mutual matching。若 correspondence 数量不足或含非有限值，流程进入 registration failure，不得继续构造正常预测。

---

## 11. Rigid Transform Estimation

允许直接复用 GeoTransformer：

```text
weighted_procrustes
```

根据：

```text
Xp_phys_corr
Xv_phys_corr
transport_confidence
```

求解：

```text
R_pred
t_pred
T_pred
```

变换方向必须始终为：

```text
Point Cloud → CT
```

即：

```text
Xv_phys_corr ≈ R_pred Xp_phys_corr + t_pred
```

现有接口的方向约定为 `src_points → ref_points`。因此 Point→CT 必须按以下参数顺序调用：

```text
weighted_procrustes(
    src_points = Xp_phys_corr,
    ref_points = Xv_phys_corr,
    weights = transport_confidence,
    return_transform = True,
)
```

现有源码只负责根据传入坐标执行 `src_points → ref_points` 求解，不识别 physical/network 语义。因此调用方必须保证传入的是同一物理长度单位下的 `Xp_phys_corr/Xv_phys_corr`；禁止直接传入 `Xp_net/Xv_net` 后把输出当作医学物理空间变换。

Baseline V1 第一版采用 direct weighted Procrustes / weighted SVD，但不得无检查地把原函数输出视为成功结果。进入求解前和求解后至少检查：

* 有效 correspondence 数不少于 3，且至少包含 3 个非共线对应点；
* correspondence 坐标和 transport confidence 均为 finite，不含 NaN/Inf；
* transport confidence 非负，且有效权重和大于数值稳定阈值；
* source/target 加权中心化后的 correspondence geometry effective rank 至少为 2，或通过等价非共线判定；不得要求 centered coordinate rank 必须等于 3；
* 近似平面本身不视为退化；但所有点几乎重合、所有点近似共线、有效基线距离过小、权重几乎全部集中于单点或极少数点、cross-covariance 数值退化均必须判为失败；
* SVD 输出、`R`、`t` 和 `T` 均为 finite；
* `R^T R ≈ I` 且 `det(R) ≈ 1`。

effective rank、最小基线距离、有效权重分布和 cross-covariance 数值退化的判定阈值必须配置化，并等待真实数据尺度确定。

统一输出状态：

```text
registration_success : bool
registration_failure_reason : optional string
```

只有 `registration_success=True` 时，`R_pred`、`t_pred`、`T_pred` 才是有效预测。失败时必须返回 `registration_success=False`，禁止自动返回 identity transform 或任意 `R,t` 冒充正常预测。

当前官方 `weighted_procrustes` 只返回变换或 `R,t`，不提供上述输入退化检查和 failure flag。因此这些检查及 `registration_success` 属于未来 M3 registration orchestration 的强制外层契约，不得误称为原函数已经提供的能力。

Baseline V1 第一版明确不使用 GeoTransformer 的完整 Local-to-Global Registration。第一版固定采用：

```text
Global coarse Point–CT correspondence
      ↓
confidence filtering
      ↓
direct weighted Procrustes / weighted SVD
      ↓
R, t
```

原因是 GeoTransformer LGR 依赖 paired point patches 和 `P×K×K` 局部匹配结构，而 Baseline V1 当前采用单个全局矩形 Point–CT coarse matching matrix。LGR 只保留为未来可选增强模块，在跨模态 patch 定义和基础 correspondence 稳定后再评估。

---

## 12. GT Correspondence

GT correspondence 必须依据 `gt_transform` 将 `Xp_phys` 从 Point Cloud 物理坐标系变换到 CT physical coordinate system 后，再与 `Xv_phys` 建立，并始终使用真实物理距离而不是 array/image index distance。

GT correspondence 必须按阶段定义：

### 12.1 M1 坐标验证阶段

M1 只验证：

* CT array index convention；
* `array_index ↔ image_index` 映射；
* `image_index ↔ physical_coordinate` 正反变换；
* spacing、origin、direction；
* Point→CT GT transform direction；
* physical coordinate round-trip；
* GT transform inverse / round-trip；
* point normal 只旋转、不平移；
* CT physical bounding box。

M1 coordinate tests 必须覆盖第 3.2 节列出的四个基础 `image_index`，验证 `index → physical`、`physical → index` 和 round-trip，并包含 `direction` 非 identity 的情况。GT transform round-trip、normal rotation 和 CT physical bounding box 必须分别设置人工可验证样例。

M1 不定义最终训练使用的 CT token correspondence，也不把 dense voxel index 临时冒充 feature token。

### 12.2 M2 token coordinate 阶段

M2 在 Sparse 3D CNN、CT tokenization 和 matching support domain 明确后，为每个 CT matching token 定义对应的真实 physical coordinate `Xv_phys`。该坐标必须包含 sparse quantization、stride、token center 和坐标轴顺序带来的映射，并可追踪回原 CT physical coordinate。

### 12.3 M2/M3 连接阶段

只有在 M2 完成后，才能最终定义：

```text
Point transformed by GT
      ↓
CT token correspondence
      ↓
positive radius / unmatched / dustbin label
```

原概念中的“nearest valid CT token”属于 M2/M3 连接阶段的候选标签生成策略，不属于 M1。具体正样本半径必须依据颌面 CT spacing、CT token density 和点云精度确定，不允许直接照搬 GeoTransformer 3DMatch 的 `0.05/0.1` 参数。

### 12.4 Many-to-one 注意事项

Nearest CT token 可能造成多个 Point 对应同一个 CT token，而 Sinkhorn + mutual matching 更接近部分一对一匹配。最终 `L_match` 监督必须在 M2 token density 明确后决定：

* 是否对 GT correspondence 去重；
* 是否采用 radius-based soft correspondence；
* 是否允许 many-to-one；
* unmatched Point/CT token 如何进入 dustbin。

Baseline V1 Specification 当前不强制确定上述具体方案，禁止在真实 token 统计前猜测。

---

## 13. Training Loss

Baseline V1 的第一目标是学习可靠 Point–CT correspondence。

第一阶段损失至少包含：

```text
L_match
```

用于监督跨模态对应关系。

`L_match` 必须与第 12 节最终选定的 GT 去重、soft correspondence、many-to-one 和 dustbin 策略一致，不能一边使用 many-to-one GT，另一边无说明地强制 mutual one-to-one 监督。

是否增加：

```text
L_registration
```

必须在 weighted SVD 是否保持可微、correspondence 是否存在离散筛选等问题确认后再决定。

不得直接照搬 GeoTransformer 的 3DMatch loss 半径和阈值。

---

## 14. Evaluation

Baseline 必须至少计算：

```text
RRE
RTE
```

其中：

* RRE：相对旋转误差，单位 degree；
* RTE：相对平移误差，使用数据真实物理单位，预期为 mm。

同时建议记录：

```text
Correspondence Inlier Ratio
Registration Recall
Inference Time
```

还必须统计 `registration_success` / failure rate，RRE、RTE 和 Registration Recall 不得把 registration failure 伪装成 identity transform 后当作正常预测计算。

RTE、correspondence inlier distance 和最终对齐可视化必须使用 physical coordinate；不得用 `Xp_net/Xv_net` 的归一化单位替代。

所有评价均基于：

```text
Point Cloud → CT
```

方向。

3DMatch benchmark 专用指标不能直接作为颌面医学数据最终评价协议。

---

## 15. Baseline V1 总结构

```text
Face Point Cloud                           CT Volume
      ↓                                        ↓
    KPConv                              Sparse 3D CNN
      ↓                                        ↓
P_raw [Np,Cp]                           V_raw [Nv,Cv]
      ↓                                        ↓
point_proj                               ct_proj
      ↓                                        ↓
Q [Np,256]                              K [Nv,256]
      \                                      /
       \                                    /
        L2 normalize + cosine / tau
                      ↓
        global rectangular similarity
                      ↓
        Sinkhorn + dustbin competition
                      ↓
        mutual + confidence filtering
                      ↓
         direct weighted Procrustes
                      ↓
   registration_success, R, t, T (if success)
```

GeometricTransformer、完整 LGR、patch fine matching 和 RANSAC 均不是 Baseline V1 必选模块，只能作为未来 comparison / optional enhancement；不得改变上述基础主链。

---

## 16. 开发顺序

必须按照以下顺序逐阶段完成：

```text
M1  数据与坐标契约阶段

M2  KPConv point encoder + Sparse3D CT encoder + CT matching support domain + CT token physical coordinate

M2/M3 connection  GT token correspondence 与 L_match 标签契约

M3  vanilla cross-modal matching + Sinkhorn + correspondence + direct weighted SVD + Baseline training/evaluation

M3 validation
Baseline V1 必须首先在完整/健康配对数据上稳定工作

M4  defect-anatomy dual constraint

M5  entropy reliability + residual feedback

M6  confidence-guided fusion / Full Model
```

### 16.1 M1 最小范围

M1 只包括：

* Point + CT 原始加载；
* CT metadata 的完整保留；
* subject/case-level split contract 和 leakage check；
* `array_index ↔ image_index` validation；
* `image_index ↔ physical_coordinate` conversion；
* physical ↔ network coordinate separation contract；
* physical coordinate round-trip tests；
* Point→CT GT transform convention；
* GT transform round-trip；
* normal rotation test；
* physical bounding-box validation；
* Dataset contract；
* `batch_size=1` 的 dual-branch collate skeleton；
* unit tests。

M1 禁止实现：

* Sparse 3D CNN；
* CT token matching；
* Cross-modal matching；
* Sinkhorn；
* weighted SVD pipeline；
* 所有 defect、anatomical、entropy、residual feedback 和 fusion 创新模块。

任何创新模块不得提前进入 M1–M3。

---

## 17. Data-dependent / M2-dependent parameters

以下参数不得直接照搬 3DMatch，也不得在没有数据依据时任意固定：

* mm 作为统一 physical unit 的最终确认；
* CT array axis order；
* medical image index convention；
* voxel/token center、stride 和 physical coordinate 定义；
* CT matching support domain；
* KPConv voxel size、radius、sigma 和 neighbor limits；
* point normal 作为输入 feature 的具体使用方式；
* CT intensity window 和归一化方式；
* CT ROI 范围；
* active voxel 定义；
* Sparse 3D CNN framework、网络深度和输出 channel `Cv`；
* coarse `Np/Nv` 和显存预算；
* matching temperature `tau`；
* Sinkhorn parameters，包括 iteration 和数值稳定相关配置；
* dustbin `alpha` 初始化及学习策略；
* GT positive radius；
* confidence threshold；
* geometric degeneracy threshold；
* correspondence / registration inlier radius；
* Registration Recall 的成功阈值；
* point/CT augmentation 范围。

`d=256` 可以作为 Baseline V1 固定初始设计，但不代表其已被当前颌面数据证明为最优。

---

## 18. 独立实验目录原则

不得直接把 Point–CT 逻辑塞入现有：

```text
experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/
```

未来 Point–CT 工作使用独立实验目录：

```text
experiments/geotransformer.pointct.baseline_v1/
```

该实验必须使用独立的：

* config；
* dataset assembly；
* model；
* loss；
* train；
* test；
* evaluation。

官方 3DMatch experiment 尽量保持原样，作为上游参考和回归对照。公共数学模块只在接口语义确实通用时复用；禁止为 Point–CT 方便而破坏官方双点云实验契约。

---

## 19. 实现原则

1. 优先复用 GeoTransformer 已验证的基础模块，不重复实现成熟数学功能。
2. 不允许为了兼容 CT 而把 CT 简单转换成普通表面点云作为主网络输入。
3. CT 必须保留体素模态及其内部信息。
4. 所有坐标变换必须有明确方向、单位和测试。
5. 每个阶段必须提供独立 smoke test。
6. 每次重大修改必须保持 Git 可回滚。
7. Codex 遇到无法从数据或规格确定的问题时必须停止猜测并明确报告。
