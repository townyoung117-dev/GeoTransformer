# M4-1B Coarse Mask Mapping Contract

## 0. 文档状态与适用边界

本文档冻结 M4-1B 的 dataset-level raw defect mask 到实际 coarsest matching token mask 的映射契约。本文档以以下已冻结文档和已审查源码审计为前提：

- `M4_DEFECT_ANATOMY_SPEC.md`
- `M4_1A_DATASET_DEFECT_CONTRACT.md`
- `M4_1B_COARSE_MASK_MAPPING_AUDIT.md`

本文档中的“必须”“不得”和“fail closed”均为强制 contract。M4 mapping enabled 时，任何来源、membership、row identity、shape、ordering 或统计无法证明的映射都不得进入 matching。

本阶段只冻结 mapping 与 alignment，不实现：

- B1 matching hard constraint；
- B2 soft modulation；
- B3 anatomical prior；
- entropy reliability；
- residual feedback；
- registration-fusion loop 或其他 fusion。

## 1. 输入、输出与统一语义

M4-1A 提供：

```text
point_defect_mask: [N_raw] bool
ct_defect_mask:    [Z,Y,X] bool
```

M4-1B 必须输出：

```text
point_intact_coarse: [Np] bool
ct_intact_coarse:    [Nv] bool
```

统一语义冻结为：

```text
True  = intact / valid anatomical token
False = defect / invalid anatomical token
```

token count 必须满足：

```text
Np == points[3].shape[0]
Nv == ct_support_indices_20mm.shape[0]
```

`point_intact_coarse` / `ct_intact_coarse` 是 anatomical defect state，不是 padding validity。不得用它们替换、覆盖或隐式表达 `point_valid_mask` / `ct_valid_mask`。

## 2. Point mapping 冻结决策

```text
POINT_MAPPING = EXPLICIT_SAME_PASS_HIERARCHICAL_PROVENANCE
```

当前真实 Point hierarchy 冻结为：

```text
points[0]: raw Point in network coordinates
    -> 5 mm grid pass  -> points[1]
    -> 10 mm grid pass -> points[2]
    -> 20 mm grid pass -> points[3]
```

`points[3]` 是上述三次逐级 grid subsampling 的真实输出。它不是 raw Point 的子集，也不得被替换为一次性 raw-to-20-mm subsampling 的输出。

### 2.1 Same-pass parent maps

实际产生 `points[i+1]` 的同一次 grid pass 必须同时产生一个整数 parent map：

```text
parent_i: [points[i].shape[0]] integer
i in {0,1,2}
```

定义：

```text
parent_i[k] =
    points[i][k] 在该次真实 grid pass 中所属的
    points[i+1] output row id
```

每个 `parent_i` 必须满足：

1. `len(parent_i) == points[i].shape[0]`；
2. 每个 id 都满足 `0 <= parent_i[k] < points[i+1].shape[0]`；
3. 每个 input row 恰好具有一个 parent；
4. 每个 `points[i+1]` output row 至少具有一个 contributor；
5. parent id 直接引用该 invocation 实际返回的 `subsampled_points` row；
6. coordinates 与 parent map 必须来自同一次 invocation，不能事后配对。

多个 input rows 具有相同 parent id 是正常且必要的 many-to-one membership，不得把 repeated/duplicate parent id 当作错误。禁止的是一个 input row 无 parent、具有歧义 parent，或 parent 指向错误 invocation/row space。

### 2.2 Parent index space 与 stacked clouds

若 grid extension 输入 stacked clouds 及 `lengths`，`parent_i` 必须使用与 concatenated `points[i+1]` 返回 tensor 完全相同的 **global output row index space**。

令 level `i` 的第 `b` 个 cloud input segment 为：

```text
I_i,b = [sum_{r<b} L_i[r], sum_{r<=b} L_i[r])
```

其对应 level `i+1` output segment 为：

```text
O_i+1,b = [sum_{r<b} L_i+1[r], sum_{r<=b} L_i+1[r])
```

则必须验证：

```text
k in I_i,b  =>  parent_i[k] in O_i+1,b
```

不同 cloud 的 parent assignment 不得跨越 length segment。当前 PointCT 使用 batch size 1 不能成为省略该验证或把 parent id 定义为未声明 local index 的理由；extension contract 必须适用于其真实 stacked API。

### 2.3 明确禁止的 Point mapping

以下方法全部禁止：

- nearest raw point；
- coordinate matching；
- coordinate sorting 或按 coordinate 强行对齐；
- 第二次独立调用 `grid_subsample`；
- 先由旧 grid path 产生 encoder points，再额外调用 provenance path 事后配 parent；
- 使用 `neighbors[]` 作为 voxel membership；
- 使用 `subsampling[]` 作为 voxel membership；
- 根据 shape、length 或 token count 猜 mapping；
- 根据重新构造的 voxel key/order 推断既有 output row id。

## 3. Point B0 extension isolation

现有接口及其执行路径必须保持不变：

```text
grid_subsampling(points, lengths, voxel_size)
```

实现阶段应优先新增独立 M4 provenance entry point；概念上可以是 `grid_subsampling_with_parent(...)`，但最终函数名由实现阶段决定，本 contract 不冻结名称。

### 3.1 M4 mapping disabled / B0

B0 或 M4 mapping disabled 时必须：

- 只调用现有 `grid_subsampling`；
- 不构造或返回 parent map；
- 不改变 coordinate accumulation；
- 不改变 `unordered_map` iteration；
- 不改变返回值、返回 row order 或 dtype；
- 不改变 grid 调用次数；
- 不改变 neighbor/subsampling/upsampling preprocessing；
- 不因新增 provenance path 改变随机状态或任何 encoder 数值。

### 3.2 M4 mapping enabled

M4 mapping enabled 时，三级 Point hierarchy 必须使用 provenance entry point **实际返回的 coordinates** 作为 encoder 的 `points[1]`、`points[2]`、`points[3]`，并使用每次 same pass 返回的 `parent_i`。

provenance path 必须保持现有 grid 的 voxel membership、origin、coordinate accumulation/average 与 output-row生成语义；不得为了方便 mask mapping 排序或替换 coarsest coordinates。

禁止以下双路径：

```text
old grid_subsampling -> encoder points
second provenance call -> parent maps
```

encoder points 与 parent maps 必须来自同一条 provenance hierarchy。

## 4. Point hard-mask aggregation

初始化：

```text
intact_0 = point_defect_mask
```

逐级聚合：

```text
intact_{i+1}[j] =
    AND(
        intact_i[k]
        for every k where parent_i[k] == j
    )

i in {0,1,2}
```

最终：

```text
point_intact_coarse = intact_3
```

严格语义是：

```text
point_intact_coarse[j] == True
iff
points[3][j] 的全部 hierarchical raw contributors 均为 intact
```

任意一个 hierarchical raw contributor 为 defect，都必须得到：

```text
point_intact_coarse[j] = False
```

不得设置 defect proportion threshold，不得用 majority vote、nearest label、mean 后阈值或其他近似替代 logical AND。

## 5. Point 无损 provenance statistics

M4-1B 必须同时保留：

```text
point_raw_total_count_coarse:  [Np] integer
point_raw_defect_count_coarse: [Np] integer
```

raw 初始化为：

```text
raw_total_count_0[k]  = 1
raw_defect_count_0[k] = 1 if point_defect_mask[k] == False else 0
```

每级必须使用同一个 `parent_i` 做整数 SUM：

```text
raw_total_count_{i+1}[j] =
    SUM(raw_total_count_i[k] for k where parent_i[k] == j)

raw_defect_count_{i+1}[j] =
    SUM(raw_defect_count_i[k] for k where parent_i[k] == j)
```

最终必须满足：

```text
point_raw_total_count_coarse[j] > 0
0 <= point_raw_defect_count_coarse[j] <= point_raw_total_count_coarse[j]
point_intact_coarse[j]
    == (point_raw_defect_count_coarse[j] == 0)
sum(point_raw_total_count_coarse) == N_raw
```

禁止对 child defect fractions 做无权平均。未来可以无损计算：

```text
point_defect_fraction =
    point_raw_defect_count_coarse / point_raw_total_count_coarse
```

这些 counts 在 M4-1B/B1 中只承担 provenance/statistics 职责；本阶段不得用它们实现 B2 gate 或任何 soft modulation。

## 6. CT relevant raw voxel scope 冻结决策

```text
CT_RELEVANT_RAW_VOXEL_SCOPE = ENTIRE_EXISTING_20MM_IMAGE_AXIS_CELL
```

CT array 轴顺序是 `[Z,Y,X]`，spacing 是 image-axis `[spacing_x,spacing_y,spacing_z]` mm。对 array 内 raw voxel index `[z,y,x]`，20 mm membership key 冻结为：

```text
q_x = floor(x * spacing_x / 20 mm)
q_y = floor(y * spacing_y / 20 mm)
q_z = floor(z * spacing_z / 20 mm)
```

对应 z-major、x-fastest linear key 为：

```text
linear = q_z * size_y * size_x + q_y * size_x + q_x
```

`direction` 不改变 membership key。`direction` 只参与 support physical coordinate：

```text
support_phys_lps =
    origin_xyz_mm + direction @ (([q_x,q_y,q_z] + 0.5) * 20 mm)
```

### 6.1 Related raw voxels 的唯一正式定义

对 support row `j`，其 grid index 按现有 array-axis storage 表达为：

```text
ct_support_indices_20mm[j] == [q_z,q_y,q_x]
```

`related raw voxels(j)` 定义为：

```text
所有位于实际 ct_volume / ct_defect_mask array [Z,Y,X] 内，
并满足 floor(index_xyz * spacing_xyz / 20 mm) == [q_x,q_y,q_z]
的 raw voxels
```

该集合包含 cell 内全部 existing raw voxels，不受 HU、foreground 或 external-surface status 限制。

boundary support cell 只统计 array 内实际存在的 raw voxels。即使当前 coarse cell center 位于原 array physical extent 之外，也不得为 array 外区域 pad、外推或猜测 mask。

### 6.2 明确排除的 CT contributor definitions

禁止把 related raw voxels 限制为：

- external-surface voxels；
- `HU > -500` foreground voxels；
- 已存在的 5 mm context voxels；
- sparse-convolution receptive field contributors。

B1 defect mask 描述的是 20 mm anatomical/spatial cell validity，不是对 `V_raw` 完整 neural receptive field 的逐贡献追踪。raw-cell membership 与 sparse feature receptive field 是不同 contract，不得混用。

## 7. CT support ordering contract

当前 preprocessing 已有的以下 arrays 是 M4-1B 的 authoritative support identity：

```text
ct_support_indices_20mm
ct_support_linear_20mm
ct_support_spatial_shape_20mm
ct_support_phys_20mm
```

M4-1B 必须复用这些实际 arrays。不得为了 mask mapping 重新生成、替换、排序或重排 support arrays。

coarse mask 与 statistics 必须通过 exact linear key aggregation/gather，严格采用当前 `ct_support_linear_20mm` 的唯一升序 row order：

```text
ct_intact_coarse[j]
ct_raw_total_count_coarse[j]
ct_raw_defect_count_coarse[j]
```

都必须对应同一个：

```text
ct_support_linear_20mm[j]
ct_support_indices_20mm[j]
ct_support_phys_20mm[j]
V_raw[j]
K[j]
Xv_phys_coarse[j]
```

不得依赖或继承 spconv active-row order。当前 CT encoder 对 `V_raw` 的 linear-key recovery 是 support alignment 的正式依据。

## 8. CT hard-mask aggregation 与无损统计

M4-1B 必须输出：

```text
ct_raw_total_count_coarse:  [Nv] integer
ct_raw_defect_count_coarse: [Nv] integer
```

对每个 existing support row `j`：

```text
ct_raw_total_count_coarse[j] =
    count(related raw voxels(j))

ct_raw_defect_count_coarse[j] =
    count(v in related raw voxels(j) where ct_defect_mask[v] == False)
```

必须满足：

```text
ct_raw_total_count_coarse[j] > 0
0 <= ct_raw_defect_count_coarse[j] <= ct_raw_total_count_coarse[j]
```

hard-mask 冻结为：

```text
ct_intact_coarse[j] =
    (ct_raw_defect_count_coarse[j] == 0)
```

即只有 cell 内全部 existing raw voxels 都为 intact 时，`ct_intact_coarse[j]` 才为 `True`。任意一个 related raw voxel 为 defect，结果必须为 `False`。不得设置 defect fraction threshold。

count-sum 必须与以下集合的基数完全一致：

```text
{array 内 raw voxels whose 20 mm linear key belongs to
 the existing ct_support_linear_20mm set}
```

未来可以无损计算：

```text
ct_defect_fraction =
    ct_raw_defect_count_coarse / ct_raw_total_count_coarse
```

这些 counts 在 M4-1B/B1 中只作为 provenance/statistics；本阶段不得实现 soft modulation。

实现可以使用 exact cell counting、streamed aggregation 或 keyed reduction。不得为了获得 counts 构造不必要的全 volume `raw voxel -> support row` 巨大 membership tensor；但任何等价实现都必须与本节的数学 membership 完全相同并可验证。

## 9. Alignment invariants

### 9.1 Point

必须同时满足：

```text
point_defect_mask.dtype == bool
point_defect_mask.shape == (points[0].shape[0],)

point_intact_coarse.dtype == bool
point_intact_coarse.shape == (points[3].shape[0],)

point_raw_total_count_coarse.shape == (points[3].shape[0],)
point_raw_defect_count_coarse.shape == (points[3].shape[0],)
```

并保持逐 row identity：

```text
point_intact_coarse[i]
<-> point_raw_*_count_coarse[i]
<-> points[3][i]
<-> P_raw[i]
<-> Q[i]
<-> Xp_phys_coarse[i]
```

### 9.2 CT

必须同时满足：

```text
ct_defect_mask.dtype == bool
ct_defect_mask.shape == ct_volume.shape

ct_intact_coarse.dtype == bool
ct_intact_coarse.shape == (ct_support_indices_20mm.shape[0],)

ct_raw_total_count_coarse.shape == (ct_support_indices_20mm.shape[0],)
ct_raw_defect_count_coarse.shape == (ct_support_indices_20mm.shape[0],)
```

并保持逐 row identity：

```text
ct_intact_coarse[j]
<-> ct_raw_*_count_coarse[j]
<-> ct_support_linear_20mm[j]
<-> ct_support_indices_20mm[j]
<-> ct_support_phys_20mm[j]
<-> V_raw[j]
<-> K[j]
<-> Xv_phys_coarse[j]
```

shape equality 只是必要条件，绝不能单独作为 alignment proof。Point 必须由 same-pass parent provenance 证明；CT 必须由 authoritative support linear keys 证明。

## 10. Fail-closed contract

M4 mapping enabled 时，下列情况至少必须 fail closed。

### 10.1 Point

- raw mask dtype 或 shape mismatch；
- parent map 缺失；
- parent map 非整数或 length 错误；
- negative/out-of-range parent id；
- input row 未得到恰好一个 parent；
- output row 没有 contributor；
- stacked length segment cross-assignment；
- composed raw-to-coarse provenance coverage 不完整；
- coarse mask/count shape 不等于 `points[3]` count；
- total/defect counts 非法或总数不守恒；
- coordinates 与 provenance 不是 same pass；
- 无法证明 `point_intact_coarse` 与实际 encoder `points[3]` row order一致。

多个 inputs 合法指向同一 parent 不属于 fail condition。

### 10.2 CT

- raw mask dtype 或 `ct_volume` shape mismatch；
- support linear key 非唯一；
- support linear key 非严格升序；
- support index -> linear 重算与 stored linear 不一致；
- support index/key 超出 `ct_support_spatial_shape_20mm`；
- support row 没有 existing related raw voxel；
- coarse mask/count shape 不等于 support count；
- total/defect counts 非法或 count sum 与 exact keyed raw-voxel集合不一致；
- mask/statistics ordering 无法由 support linear key证明；
- `V_raw` / `K` / `Xv_phys_coarse` count 与 support count不一致；
- x20 active key alignment validation 失败。

### 10.3 禁止的容错

任何一侧都不得使用：

- truncate；
- pad；
- broadcast；
- nearest-neighbor repair；
- coordinate sorting repair；
- default all-intact；
- missing-mask fallback；
- token-count/shape guessing；
- 任意“尽量继续运行”的近似映射。

## 11. B0 contract

B0 定义为三个 M4 开关全部为 `False`。M4 mapping disabled/B0 时：

- 不要求 raw defect mask；
- 不生成 Point parent provenance；
- 不生成 `point_intact_coarse`；
- 不生成 `ct_intact_coarse`；
- 不生成任何 Point/CT defect count statistics；
- 原 `grid_subsampling` interface 与 execution path 完全保持；
- 原 Point neighbor preprocessing 完全保持；
- 原 CT context/support path 完全保持；
- 不改变 encoder coordinates 或 features；
- 不改变随机状态；
- 不改变调用次数或 numeric path；
- 不以默认 mask 或空 mask模拟关闭状态。

B0 不进入 M4 mapping 是显式 gating，不是 missing-mask fallback。

## 12. x20 sparse ordering residual validation

当前 `CTEncoder` 已经通过以下过程把 `V_raw` 恢复到 support order：

```text
x20 active indices
    -> x20 linear keys
    -> sort + searchsorted(existing support linear keys)
    -> gather V_raw in support order
```

当前 residual issue 是：x20 active linear keys 自身尚未被显式验证为 unique + in-bounds。

M4-1B alignment validation 必须要求：

1. x20 active `[z,y,x]` indices 全部位于 x20/support spatial shape 内；
2. x20 linear keys 唯一；
3. 每个 authoritative support key 在 x20 active keys 中恰好存在一次；
4. gather 后 `V_raw.shape[0] == Nv`，row order 等于 support order。

不得因为该 validation 提前重构 CT encoder。实现阶段应优先用最小 validation/test 解决；若确实必须修改 `CTEncoder`，只允许增加 fail-closed validation，不得改变 sparse features、indice generation、gather identity 或其他数值路径。

## 13. 实现阶段拆分与顺序

M4-1B 后续实现必须拆分为以下顺序：

### M4-1B-P1

```text
Point provenance extension
-> parent index-space/segment validation
-> B0 regression
-> review diff / audit / commit
```

### M4-1B-P2

```text
Point hierarchical mask aggregation
-> raw total/defect count aggregation
-> alignment/fail-closed tests
-> review diff / audit / commit
```

### M4-1B-C1

```text
CT entire-existing-cell keyed count aggregation
-> support-order mask/statistics
-> geometry/boundary/fail-closed tests
-> review diff / audit / commit
```

### M4-1B-I

```text
Point/CT integration
-> end-to-end alignment tests
-> B0 regression
-> Pat1 read-only smoke
-> review diff / audit / commit
```

每个阶段必须单独执行：

```text
implementation
-> tests
-> review diff
-> audit
-> commit
```

不得一次性实现全部阶段。任何 commit/push 仍需对应阶段的明确授权。

## 14. M4-1B 验收边界

M4-1B 只有在以下条件全部满足后才可进入 M4-2：

1. Point mapping 使用 actual same-pass hierarchical parent provenance；
2. Point mask/counts 与 `points[3]`、`P_raw`、`Q`、`Xp_phys_coarse` 逐 row 对齐可证；
3. CT mapping 使用 entire existing 20 mm image-axis cell；
4. CT mask/counts 严格按 authoritative support linear order生成；
5. CT mask/counts 与 `V_raw`、`K`、`Xv_phys_coarse` 逐 row 对齐可证；
6. 所有 mask 为 bool，所有 counts 为合法整数并满足守恒；
7. x20 unique/in-bounds/support-presence validation 已由最小实现和测试证明；
8. 所有 invalid provenance、shape、key、ordering 与 count case 都 fail closed；
9. B0 不生成任何 M4 mapping artifact，且封存 Point/CT encoder numeric path不变；
10. 没有实现或提前接入 B1/B2/B3。

只有满足本 contract 后，`point_intact_coarse` / `ct_intact_coarse` 才允许作为 M4-2 B1 hard-constraint 的正式输入。
