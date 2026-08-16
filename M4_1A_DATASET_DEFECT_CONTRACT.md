# M4-1A Dataset-Level Defect Data Contract

## 0. 文档状态与适用范围

本文档冻结 M4-1A 的 dataset-level defect data contract。其职责边界仅为：

```text
disk defect artifacts
    -> raw dataset fields
```

本文档不实现 loader，不修改 M3 数据路径，也不冻结任何 raw-to-coarsest 或 CT support 聚合算法。

本文档中的“必须”“不得”和“fail closed”均为强制 contract。M4 启用时，任何不满足 contract 的 variant 都不得通过默认值、猜测、截断、填充、重排或重采样继续训练或评估。

## 1. 正式 defect variant 目录结构

每个 subject 的正式组织方式冻结为：

```text
local_data/
└── <subject_id>/
    ├── ct.nrrd
    ├── pointcloud.npz
    ├── pair_metadata.json
    ├── ct_metadata.json
    ├── qc.json
    └── defects/
        └── <defect_id>/
            ├── ct_defect.nrrd
            ├── face_pointcloud_defect.npz
            ├── Mp_gt.npy
            ├── Mv_gt.nrrd
            ├── metadata_defect.json
            ├── defect_config.json
            └── qc_report.json
```

M4 runtime 的核心 artifact 冻结为：

- `face_pointcloud_defect.npz`
- `ct_defect.nrrd`
- `Mp_gt.npy`
- `Mv_gt.nrrd`
- `metadata_defect.json`

`defect_config.json` 与 `qc_report.json` 是 provenance / QA artifact，不得作为 encoder 数学输入。

PLY、triangle ids、bone/soft 细分 mask 及其他辅助产物不得成为当前 M4-1A 的必需 runtime 输入。

## 2. Point defective input contract

正式 Point defective input 为：

```text
face_pointcloud_defect.npz
```

该 NPZ 必须至少包含以下字段：

| key | dtype / shape | contract |
|---|---|---|
| `xyz` | numeric floating `[N,3]` | 所有值 finite；坐标为 physical LPS；单位 mm |
| `normal` | numeric floating `[N,3]` | 所有值 finite；第一维必须与 `xyz` 相同；每一行必须是与对应点关联的单位法向量 |
| `defect_id` | scalar string | 必须与 variant directory 和 metadata 中的 `defect_id` 一致 |

若 NPZ 还包含以下可选字段，则必须与 dataset/metadata 声明一致：

- `patient_id`
- `coordinate_system`
- `unit`
- `sampling_method`

runtime 不得依赖 NPZ 中的机器绝对文件路径。`N` 是该 defect variant 的 raw defective point 数量，不得由其他 artifact 猜测。

## 3. Point defect mask contract

正式 Point defect mask 文件为：

```text
Mp_gt.npy
```

冻结的磁盘契约为：

| 属性 | 要求 |
|---|---|
| on-disk dtype | `uint8` |
| shape | `[N]` |
| 允许值 | 仅 `{0,1}` |
| `1` | intact / stable |
| `0` | defect / defect-influenced / unreliable |

必须满足：

```text
len(Mp_gt) == face_pointcloud_defect.npz["xyz"].shape[0]
```

并且 row ordering contract 冻结为：

```text
Mp_gt[k] <-> face_pointcloud_defect.npz["xyz"][k]
```

任何未来 defect generator 都必须保持并提供可审计的这一对应关系。ordering 不一致或无法证明时，dataset 必须 fail closed；不得通过下列方式补救：

- nearest neighbor
- shape guessing
- truncation
- padding
- reorder

dataset 加载后的正式 runtime field 冻结为：

```text
point_defect_mask: [N] bool
True  = intact / stable
False = defect / defect-influenced / unreliable
```

`uint8 -> bool` 只允许逐元素、语义保持的 dtype 转换，不得反转，不得改变 row ordering。M4-1A 对外提供的 `[N]` 即 `[N_raw]`。

### 3.1 Point ordering proof 的职责边界

`Mp_gt[k] <-> xyz[k]` 是 dataset-generation / provenance contract，不是 runtime loader 能仅凭两个数组 shape 或长度相等自行推导出的事实。

runtime loader 必须验证其能够直接验证的项目：

- `xyz`、`normal` 和 `Mp_gt` 的 dtype；
- `xyz`、`normal` 和 `Mp_gt` 的 shape；
- `xyz`、`normal` 与 `Mp_gt` 的长度关系；
- `Mp_gt` 仅含二值 `{0,1}`；
- subject / defect identity 一致性；
- metadata 与 dataset acceptance 所要求的 provenance declarations 存在且一致。

但是：

```text
len(Mp_gt) == N
```

只证明长度一致，不证明第 `k` 个 mask element 与第 `k` 个 point 的身份一致。runtime loader 不得据此宣称 ordering 已被证明。

当前 Pat1 代表样本的 ordering 已通过实际生成源码数据流审计证明：生成 `Mp_gt` 时使用的 `xyz` 与写入 `face_pointcloud_defect.npz["xyz"]` 的数组保持同一 row ordering。因此，该样本允许用于 M4-1A / M4-1B interface development。

在正式批量 defect generation 前，所有正式 variant 的 row-order contract 必须由冻结、版本化且具有可审计 provenance 的 generator 保证。若数据来源没有可信的 generation / provenance contract，必须在 dataset acceptance 阶段 fail closed；不得由 runtime loader 使用 nearest neighbor、shape guessing 或其他几何匹配方式补救。

## 4. CT defective input contract

正式 CT defective input 为：

```text
ct_defect.nrrd
```

它必须提供：

| 属性 | contract |
|---|---|
| `volume` | array `[Z,Y,X]` |
| `spacing` | `[x,y,z]`，单位 mm |
| `origin` | `[x,y,z]`，单位 mm |
| `direction` | `[3,3]` |
| coordinate system | physical LPS |
| physical unit | mm |

所有 geometry 值必须 finite。`ct_defect.nrrd` 的 geometry 必须与同一 defect variant 的 `Mv_gt.nrrd` 严格一致。

## 5. CT defect mask contract

正式 CT defect mask 文件为：

```text
Mv_gt.nrrd
```

冻结的磁盘契约为：

| 属性 | 要求 |
|---|---|
| array shape | 与 `ct_defect.nrrd` 完全一致的 `[Z,Y,X]` |
| on-disk dtype | `uint8` |
| 允许值 | 仅 `{0,1}` |
| `1` | intact / valid |
| `0` | defect / invalid |

dataset 必须逐项验证 `ct_defect.nrrd` 与 `Mv_gt.nrrd` 的：

- shape
- spacing
- origin
- direction
- coordinate system

任何不一致必须 fail closed。不得通过 resampling、padding、cropping、axis guessing、header ignoring 或默认 mask 修复不一致。

dataset 加载后的正式 runtime field 冻结为：

```text
ct_defect_mask: [Z,Y,X] bool
True  = intact / valid
False = defect / invalid
```

`uint8 -> bool` 只允许逐体素、语义保持的 dtype 转换，不得反转。

M4-1A 不得生成 20 mm CT support mask。raw CT mask 到 20 mm matching support 的映射与聚合属于 M4-1B。

## 6. Variant identity contract

每一个 defect sample 必须具有：

- `subject_id`
- `defect_id`

`defect_id` 必须在同一 subject 内唯一。variant directory name 必须与 `defect_id` 完全一致。

至少下列三处 defect identity 必须交叉验证一致：

1. variant directory name
2. `face_pointcloud_defect.npz` 中的 `defect_id`
3. `metadata_defect.json` 中的 `defect_id`

若 NPZ 或 metadata 声明 `patient_id` / `subject_id`，它也必须与承载该 variant 的 `<subject_id>` 一致。

禁止通过目录枚举顺序、variant 数字后缀或任何隐式数字索引猜测 defect identity。

### 6.1 Defect variant selection / gating contract

B0 时，即三个 M4 开关全部为 `False` 时：

- 不选择 defect variant；
- 不要求 `defect_id`；
- 不扫描或枚举 `defects/`。

M4-enabled 时，每个 dataset sample 必须显式且唯一地解析到：

```text
(subject_id, defect_id)
```

以下任一情况必须 fail closed：

- `defect_id` 缺失；
- sample 的 subject identity 与 variant 声明不一致；
- 指定 variant 不存在；
- 同一选择解析到多个 variant；
- 无法唯一确定 `(subject_id, defect_id)`。

禁止：

- 自动选择 `defects/` 下第一个目录；
- 依赖文件系统目录枚举顺序；
- 默认选择 `defect_001`；
- 根据数字后缀猜测 variant；
- 因当前 subject 下只有一个 defect 就隐式选择它。

本规格只冻结“必须显式唯一选择”的语义。具体 Python constructor / config API 留待实现阶段确定；M4-1A contract 不在此发明额外的选择接口。

## 7. metadata_defect.json runtime 原则

`metadata_defect.json` 用于：

- subject / variant identity
- artifact provenance
- mask semantics
- CT geometry QA
- generation provenance

旧 metadata 中的机器绝对路径只能作为历史 provenance，不得控制 M4 runtime 文件加载，也不得成为 artifact 存在性或 identity 的真实依据。

正式 dataset 必须仅通过以下一种或多种受控方式定位 artifact：

- dataset manifest
- subject-relative path
- defect-relative path

runtime 不得回退到 metadata 中记录的旧绝对路径。

### 7.1 Minimum runtime metadata schema

当前代表样本的 `metadata_defect.json` 使用现有顶层字段，而不是新建的嵌套 runtime schema。M4-enabled dataset 至少必须读取并验证下列真实字段：

| 当前顶层字段 | 当前结构及必须验证的职责 |
|---|---|
| `patient_id` | scalar string；当前 metadata 中没有另设顶层 `subject_id`，因此必须将 `patient_id` 与 dataset 的 `subject_id` 交叉验证 |
| `defect_id` | scalar string；必须与显式选择的 `defect_id`、variant directory 和 NPZ 声明一致 |
| `formal_transform` | 当前为 scalar string；必须明确且无歧义地解析为该 defect pair 自身的 rigid transform |
| `ct_geometry` | object；当前子字段为 `size_xyz`、`spacing_xyz_mm`、`origin_xyz_mm`、`direction`，必须与实际 defective CT / Mv geometry 交叉验证 |
| `Mp` | object；当前包含 `shape`、`zero_count`、`one_count`、`convention`、surface/boundary counts 及两个 distance threshold，用于 Point mask 语义与生成 provenance 验证 |
| `Mv` | object；当前包含 `shape_zyx`、`zero_count`、`one_count`、`convention`，用于 CT mask 语义与生成 provenance 验证 |

当前还存在 `defect_pointcloud`、`sampling_method`、`open3d_version`、`complete_inputs`、`complete_input_sha256` 和 `formal_outputs` 等 provenance 字段。它们可参与 dataset acceptance 和审计，但 `formal_outputs` 中的历史绝对路径仍不得用于 runtime artifact 定位。

当前 metadata 本身没有 generator version、generator script SHA 或 Git commit，也没有一个仅凭 JSON 即可重新证明 Point row ordering 的字段。因此，当前 Pat1 的 ordering acceptance 依据是已完成的生成源码审计；不能把缺失的 ordering proof 虚构为新的 metadata nested key。

### 7.2 Defect pair GT transform contract

每个 M4 defect sample 必须具有自己的 runtime field：

```text
gt_transform: [4,4]
```

其唯一语义为：

```text
Point defective physical coordinates
    -> CT defective physical coordinates

x_ct = R @ x_point + t
```

两侧 physical coordinate system 均为 LPS，translation `t` 的单位为 mm。`gt_transform` 必须 finite，并满足合法 rigid transform contract：齐次矩阵最后一行为 `[0,0,0,1]`，旋转块 `R` 必须正交且 `det(R)=+1`，translation 必须为 finite 三维向量。

当前代表样本的 `metadata_defect.json` 以真实的顶层 JSON 字段表达 Identity：

```json
"formal_transform": "Identity"
```

该显式字符串在当前代表样本中解析为：

```text
gt_transform = identity [4,4]
```

当前 metadata 同时包含顶层 `R_gt_generated: false` 与 `t_gt_generated: false`；它们是生成 provenance 标记，不替代 `formal_transform`，也不单独定义 transform 方向。

defect sample 的 raw `gt_transform` 必须来自当前 defect variant 明确声明的 `formal_transform`。不得静默使用 parent complete case 的 `pair_metadata.json` 中的 `gt_transform` 替代。即使 defect metadata 明确声明其 transform 与 parent transform 等价，也必须显式解析并验证两者的矩阵、方向、LPS 坐标语义和单位一致；不得因为当前 M3 complete GT 恰好为 Identity 就假设相同。

dataset 加载后的 defect sample 必须继续提供现有 M3 所要求的 `gt_transform`，以保持后续 perturbation、GT correspondence、training 和 evaluation 接口兼容。

若 `formal_transform` 缺失、方向不明确、无法无歧义解析为 `[4,4]`、含 non-finite 值、不是合法 rigid transform，或 Point-to-CT / LPS / mm 语义不明确，M4-enabled 必须 fail closed。当前已冻结的磁盘表达明确支持上述顶层字符串 `"Identity"`；其他非 Identity 的磁盘表达必须先有明确且版本化的 serialization contract，dataset 不得猜测其矩阵布局或方向。

## 8. B0 backward compatibility

当以下三个开关全部为 `False`：

```text
use_defect_hard_constraint = False
use_defect_soft_modulation = False
use_anatomical_prior       = False
```

即 B0 时，必须满足：

- 不选择 defect variant，不要求 `defect_id`，不扫描 `defects/`；
- 不要求 `defects/` 目录存在；
- 不要求任何 defect artifact 存在；
- 不调用或实例化 defect loader；
- dataset sample 不要求 `point_defect_mask`；
- dataset sample 不要求 `ct_defect_mask`；
- 原有 M3 sample contract、随机状态和数值路径保持不变；
- 不得自动创建全 `True` defect mask。

## 9. M4-enabled fail-closed contract

当任一需要 defect data 的 M4 功能启用时，当前选定 defect variant 的完整 runtime core artifacts 必须存在并通过验证。

至少下列情况必须 fail closed：

- defect directory 缺失；
- 任一核心 artifact 缺失；
- sample 没有显式且唯一地解析到 `(subject_id, defect_id)`；
- `defect_id` 缺失、指定 variant 不存在或同一选择对应多个 variant；
- subject / `defect_id` 不一致；
- `formal_transform` 缺失、非法或语义不明确；
- defect `gt_transform` 无法验证为 finite `[4,4]` Point-to-CT LPS/mm rigid transform；
- NPZ `xyz` 或 `normal` 的 dtype、shape、finite 性或语义非法；
- `xyz` 与 `normal` 长度不一致；
- normal 不是与对应 point 关联的单位法向量；
- `Mp_gt` 的 on-disk dtype 不是 `uint8`，或不能进行语义保持的 bool 转换；
- `Mp_gt` 含 `{0,1}` 以外的值；
- `Mp_gt` 长度不等于 `N`；
- Point mask ordering contract 不成立或数据来源无法证明该 contract；
- `Mv_gt` 的 on-disk dtype 不是 `uint8`；
- `Mv_gt` 含 `{0,1}` 以外的值；
- CT/Mv shape 不一致；
- spacing 不一致；
- origin 不一致；
- direction 不一致；
- coordinate system 不一致；
- CT/Mv geometry 包含 non-finite 值。

失败时不得使用默认全 intact mask、零 mask 或其他替代值继续训练或评估。

## 10. M4-1A 与 M4-1B 边界

M4-1A 负责 defect variant 的 dataset-level raw contract，包括将选定 defect variant 的磁盘 artifact 加载并验证为：

- defective Point input，沿用现有 M3 Point sample 字段语义；
- defective CT input，沿用现有 M3 CT sample 字段语义；
- `point_defect_mask: [N_raw] bool`；
- `ct_defect_mask: [Z,Y,X] bool`；
- `gt_transform: [4,4]`，语义为 Point defective physical coordinates -> CT defective physical coordinates。

`gt_transform` 属于 M4-1A dataset-level contract，不属于 M4-1B 的 coarsest mask mapping 职责。

M4-1A 不负责：

- KPConv coarsest mask
- raw Point 到 KPConv coarsest 的索引或聚合
- CT 20 mm support mask
- raw CT mask 到 20 mm support 的映射或聚合
- mask aggregation
- hard constraint
- soft modulation
- anatomical prior

上述内容全部留给后续冻结阶段，其中 raw-to-coarsest 与 CT support 对齐属于 M4-1B。

## 11. 当前代表样本与生成器 provenance

当前代表样本：

```text
local_data/Pat1/defects/defect_001_left_maxilla_cheek_medium/
```

其已审计结论为：

```text
Point ordering:       PROVEN
CT geometry:          PASS
Point M4 semantics:   COMPATIBLE
CT M4 semantics:      COMPATIBLE
```

因此，该代表样本可以用于 M4-1A / M4-1B 接口开发。

当前生成器仍有以下非阻断性 provenance 改进项：

- metadata 使用历史机器绝对路径；
- 没有 generator version；
- 没有 generator script SHA；
- 没有 Git commit provenance。

这些问题不否定当前代表样本已证明的 ordering、geometry 和语义 contract，但必须在 M4 正式批量 defect generation protocol 冻结前解决 generator provenance / versioning。

## 12. 本阶段禁止提前决定

M4-1A 不得决定：

- raw Point 到 KPConv coarsest 的映射；
- raw CT mask 到 20 mm support 的聚合；
- 缺损位置或尺寸分布；
- 每个 subject 生成多少 variant；
- train / validation defect generation protocol；
- hard constraint 实现；
- soft modulation；
- anatomical prior。

## 13. M4-1A 验收边界

本阶段的唯一规格产物是本文档。后续 M4-1A loader 实现必须遵守本文档，但不属于本阶段交付。

本阶段不创建或修改 Python loader、JSON、dataset manifest、`local_data`、M3 文件或 defect artifact，也不执行正式 test。
