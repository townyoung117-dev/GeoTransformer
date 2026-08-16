# M4-1B Raw Defect Mask to Coarsest Token Mapping Audit

## 0. 审计范围与结论摘要

本报告针对分支 `m4_defect_anatomy`、HEAD `ef33aef` 的当前源码进行只读审计。审计入口是 M4-1A 已提供的：

```text
point_xyz_phys:     [N_raw, 3]
point_defect_mask:  [N_raw] bool
ct_volume:          [Z, Y, X]
ct_defect_mask:     [Z, Y, X] bool
```

本报告不实现任何映射、hard constraint、soft modulation 或 anatomy。核心结论如下：

- Point 的 `points[3]` 是 `5 mm -> 10 mm -> 20 mm` 三次逐级 grid subsampling 的结果，不是 raw Point 的子集，也不是一次性对 raw Point 做 20 mm voxel average 的结果。
- 当前 Point grid-subsampling extension 只接受 coordinates/lengths，只返回 coordinates/lengths；不接受 features 或 labels，也不返回 voxel key、parent index 或 membership。
- 当前 `subsampling[]` 是有半径与数量上限的 geometric radius-neighbor table，不是 voxel membership，不能作为 raw-to-coarse provenance。
- Point grid 输出由 `std::unordered_map` 迭代写出，没有 key sort；当前源码不提供跨 build/platform/runtime 的稳定 row-order contract。一次 preprocessing 内部，`points[3][i]` 与 `P_raw[i]` 严格保持同一 query-row ordering，但不能靠重新调用 grid subsampling 重建第 `i` 行身份。
- CT 20 mm support 的 linear key 由 image-axis 网格量化得到，`np.unique` 产生升序唯一 key，因此 support ordering 是源码可证明的确定性顺序。
- spconv 可以按自己的 active-index ordering 输出；当前 CT encoder 不依赖该顺序，而是按 20 mm linear key 排序、`searchsorted`、存在性检查并 gather，恢复出与输入 support 严格一致的 `V_raw` ordering。
- 当前两侧均没有 raw contributor membership。因而当前源码本身不足以生成可证明的 `point_intact_coarse` / `ct_intact_coarse`。

```text
POINT_MAPPING_STATUS = UNRESOLVED
CT_MAPPING_STATUS    = UNRESOLVED
```

这里的 `UNRESOLVED` 表示“当前实现缺少所需显式映射”，不表示没有推荐实现方向。

## 1. 审计到的真实调用链

### 1.1 Dataset-level raw fields

`geotransformer/datasets/registration/pointct/dataset.py` 的 `PointCTDataset._get_defect_item()` 返回 defective Point/CT 及两个 raw bool mask。`pointct_collate_fn()` 在 batch size 1 下把它们分别原样放入：

```text
sample["point_defect_mask"] -> batch["point"]["point_defect_mask"]
sample["ct_defect_mask"]    -> batch["ct"]["ct_defect_mask"]
```

该层不生成 coarsest mask，也不进行空间重采样。

### 1.2 Point branch

真实 Point 调用链为：

```text
PointCTDataset._get_defect_item
  -> pointct_collate_fn
  -> experiments/geotransformer.pointct.baseline_v1/dataset.py::m2_point_collate_fn
  -> geotransformer/utils/data.py::single_collate_fn_stack_mode
  -> geotransformer/utils/data.py::precompute_data_stack_mode
  -> geotransformer/modules/ops/grid_subsample.py::grid_subsample  (三次)
  -> geotransformer.ext.grid_subsampling                         (CPU C++ extension)
  -> experiments/.../backbone.py::KPConvFPN
  -> experiments/.../point_encoder.py::PointEncoder
  -> P_raw / Q / Xp_net_coarse / Xp_phys_coarse
```

`m2_point_collate_fn()` 先执行：

```text
point_xyz_net = float32(point_xyz_phys * 0.001)
```

因此 raw dataset coordinates 是 LPS/mm，KPConv preprocessing 的所有 `points[level]` 都是 float32/m。

### 1.3 CT branch

真实 CT 调用链为：

```text
PointCTDataset._get_defect_item
  -> pointct_collate_fn
  -> experiments/geotransformer.pointct.baseline_v1/dataset.py::m2_ct_collate_fn
       -> build_ct_context_5mm
       -> extract_external_surface
            -> find_boundary_connected_outside_air
       -> build_ct_support_20mm
  -> experiments/.../ct_encoder.py::CTEncoder
       -> SparseConvTensor at 5 mm
       -> encoder_5mm -> encoder_10mm -> encoder_20mm
       -> linear-key recovery to input support order
  -> V_raw / K / Xv_phys_coarse
```

## 2. Point raw -> KPConv coarsest

### 2.1 实际 level、单位与 sampling size

冻结配置是：

```text
point_network_scale_mm_to_m = 0.001
num_stages                  = 4
initial voxel_size argument = 0.0025 m
initial radius              = 0.00625 m
neighbor_limits             = [177, 32, 33, 34]
```

`precompute_data_stack_mode()` 在每个 loop 末尾先把 `voxel_size *= 2`，而 `i == 0` 不执行 subsampling。因此 `0.0025 m` 本身没有用于产生新 level；实际第一轮 subsampling 是 5 mm：

| level | 产生方式 | 实际单位 | 本级实际 grid size | row 的含义 |
| --- | --- | --- | --- | --- |
| `points[0]` | `point_xyz_phys * 0.001` | m, float32 | 无 subsampling | raw defective Point，row order 沿用 NPZ/dataset |
| `points[1]` | 对 `points[0]` 调用 `grid_subsample` | m, float32 | `0.005 m = 5 mm` | 每个非空 5 mm voxel 的 coordinate mean |
| `points[2]` | 对 `points[1]` 调用 `grid_subsample` | m, float32 | `0.010 m = 10 mm` | 每个非空 10 mm voxel 内 **5 mm centroid rows** 的 mean |
| `points[3]` | 对 `points[2]` 调用 `grid_subsample` | m, float32 | `0.020 m = 20 mm` | 每个非空 20 mm voxel 内 **10 mm centroid rows** 的 mean |

`PointEncoder` 最终直接取：

```text
Xp_net_coarse  = point_dict["points"][3]       # m
Xp_phys_coarse = Xp_net_coarse / 0.001         # mm
```

### 2.2 grid coordinate 与 centroid 公式

对一次 voxel size 为 `v` 的调用，C++ 对每个单独 cloud 计算：

```text
origin = floor(min(points) / v) * v
q_x    = floor((p_x - origin_x) / v)
q_y    = floor((p_y - origin_y) / v)
q_z    = floor((p_z - origin_z) / v)
key    = q_x + N_x * q_y + N_x * N_y * q_z
```

每个 key 的输出 coordinate 是：

```text
s(key) = sum(p for p in key) / count(key)
```

所以：

1. `points[1]` 是 raw rows 的 voxel mean；
2. `points[2]` 是 `points[1]` rows 的 voxel mean；
3. `points[3]` 是 `points[2]` rows 的 voxel mean。

`points[3]` 不是原始 Point 的子集。它通常也不是所有 raw contributors 的加权 mean：每个 child centroid 在下一层只贡献一次，child 中包含多少 raw rows 不参与下一层 coordinate averaging。一个 `points[3]` token 的真实 raw contributor set 只能定义为其 hierarchical child sets 的并集。

### 2.3 ordering 是否 deterministic

C++ 用 `std::unordered_map<std::size_t, SampledData>` 收集 voxel，并通过：

```text
for (auto& v : data) { push_back(voxel_mean); }
```

写出 rows，没有对 voxel key 排序，也没有随输出返回 key。

因此必须区分两种结论：

- **单次 preprocessing 内的内部一致性：PROVEN。** `points[3]` 的既有 row order 被后续 radius tables 和 KPConv query rows原样消费；`P_raw[i]` 与 `points[3][i]` / `Xp_phys_coarse[i]` 对应。
- **可移植的稳定 row-order contract：NOT PROVEN。** C++ 标准不把 `unordered_map` iteration order 冻结为按 key 排序的语义；当前仓库也没有额外 sort 或 persisted key。不能把某次独立重跑的第 `i` 行当作既有 `points[level][i]` 的身份凭据。

对“同一 exact input + 同一 voxel size 重复调用”的逐项回答：

| 项目 | 源码能否严格保证与既有 ordered output 完全一致 | 结论 |
| --- | --- | --- |
| 输出数量 | 同一有限输入的 distinct voxel-key cardinality 相同 | 数量可由算法确定 |
| 每个 voxel 的 coordinate value | 同一输入遍历顺序下仍执行相同 float32 accumulation/mean | coordinate multiset 在当前实现中可重现；没有 row identity key |
| 输出 ordering | `unordered_map` iteration，未排序、未声明稳定 contract | **不能严格保证** |
| ordered coordinate tensor 逐行完全一致 | 依赖 ordering | **不能作为 contract 保证** |

结论：不得通过“第二次调用同样的 `grid_subsample`”来给第一次调用产生的 `points[level]` 补配 mask。

### 2.4 当前 provenance 与 neighbor tables

当前 `grid_subsample` 只返回 `s_points, s_lengths`。源码没有保存：

- input row -> output voxel row；
- output row -> contributor input rows；
- voxel key -> output row；
- composed raw row -> `points[3]` row；
- contributor count。

`precompute_data_stack_mode()` 随后生成的 `neighbors[]` 是同 level radius neighbors；`subsampling[i]` 是：

```text
query   = points[i + 1]
support = points[i]
radius  = 6.25, 12.5, 25 mm（逐级翻倍）
limit   = 177, 32, 33
```

它不能作为 voxel membership，理由包括：

1. 查询规则是 Euclidean radius，不是上面的 voxel-key equality；
2. 一个真实 voxel contributor 到 centroid 的距离可能超过 radius；
3. radius 也可能收进相邻 voxel 的 rows；
4. 每行还受 `neighbor_limits[i]` 截断；
5. table 不携带 raw contributor union，也不证明每个 contributor 恰好属于一个 parent。

`neighbors[]` 同样是卷积 neighborhood，不是 raw-to-coarse provenance。

### 2.5 Point encoder row alignment

`KPConvFPN` 的三个 strided blocks分别使用：

```text
q_points = points[1], points[2], points[3]
neighbor_indices = subsampling[0], [1], [2]
```

KPConv 输出第一维就是 query row 数 `M`，其运算没有对 query rows 排序。最终：

```text
feats_s4 row i <-> points[3] row i
P_raw = feats_s4
Q     = Linear(P_raw)  # row-preserving
```

所以在同一次 preprocessing/forward 内：

```text
P_raw[i] <-> Q[i] <-> points[3][i] <-> Xp_phys_coarse[i]
```

这是严格成立的。M4-1B 的缺口只在于 raw mask 如何被放到这同一个 `i` 上。

## 3. 当前 grid_subsample extension 的真实能力

审计文件：

- `geotransformer/modules/ops/grid_subsample.py`
- `geotransformer/extensions/pybind.cpp`
- `geotransformer/extensions/cpu/grid_subsampling/grid_subsampling.h/.cpp`
- `geotransformer/extensions/cpu/grid_subsampling/grid_subsampling_cpu.h/.cpp`

结果：

| 能力 | 当前是否支持 | 实际行为 |
| --- | --- | --- |
| coordinates | 是 | 每 voxel 对 coordinate 求 arithmetic mean |
| batch lengths | 是 | 每个 cloud 单独量化，输出后 concatenate |
| features input | **否** | Python、binding 和 C++ signature 都没有 features 参数 |
| feature aggregation | 不适用 | 没有 mean/sum/first 等实现 |
| labels input | **否** | 没有 labels 参数 |
| label aggregation | 不适用 | 没有 majority/random/first 等实现 |
| input index membership | **否** | 不返回 parent/member indices |
| voxel key | **否** | key 只在 C++ local `unordered_map` 中存在 |
| contributor count | **否** | `SampledData.count` 只用于求 mean，不写回 Python |

不得根据其他 KPConv repository 的常见 grid-subsampling API 推断本仓库支持 feature/label propagation。

## 4. Point mask mapping 候选方案

目标 shape：

```text
point_intact_coarse.shape == (points[3].shape[0],)
```

### 4.1 候选比较

| 方案 | 严格对应 `points[3]` | ordering 风险 | 需要 NN | 是否改变 KPConv 数值路径 | fail closed 能力 | B1 hard constraint 适用性 |
| --- | --- | --- | --- | --- | --- | --- |
| A. nearest raw point mask | 否；centroid 可能没有同一 raw point，且一个近邻不能代表 contributor set | tie、距离阈值和重复坐标均有歧义 | 是 | 可在旁路计算，不必改 encoder；但语义是近似 | 只能对距离/tie 报错，不能证明 contributor 完整性 | **不适合** |
| B. voxel membership aggregation | 若 membership 来自产生各 level 的同一次调用，则可以 | 独立重建 membership 或独立重跑有风险 | 否 | 可只增加旁路 provenance；coordinates 不应改变 | 可以验证 parent range、覆盖率、非空与 shape | **适合**，但当前 membership 不存在 |
| C. grid_subsample feature/label propagation | 当前接口不可用 | 若未来另一次调用则仍有风险 | 否 | 扩展当前 API/实现；必须证明 coordinate path 不变 | 可在同一输出 row 聚合时 fail closed | 概念上可用，但当前不存在；单纯 bool mean 也不足以保留 raw counts |
| D. 利用现有 `subsampling` index | 否；它是截断 radius neighborhood，不是 voxel membership | table row 与 coarse row一致，但 contributor identity错误 | 否 | 不改变 encoder | 无法验证它等于真实 voxel contributors | **不适合** |
| E. 新增显式 provenance mapping | 是；前提是由产生 `points[i+1]` 的同一 pass 返回 parent row identity | 可消除独立调用的 ordering 风险 | 否 | 可保持原 coordinates/neighbor/KPConv 路径 | 可以全面 fail closed | **推荐** |

### 4.2 保守 all-contributors-intact 语义

若一个 parent map 定义为：

```text
parent_i[k] = points[i + 1] 中接收 points[i][k] 的 row id
```

则保守 bool 可逐级定义：

```text
intact_0 = point_defect_mask
intact_{i+1}[j] = AND(intact_i[k] for all k with parent_i[k] == j)
```

最终：

```text
point_intact_coarse[j] = True
iff points[3][j] 的全部 hierarchical raw contributors 均为 intact
```

只要任一 raw contributor 为 defect，逐级 AND/`min` 就传播为 `False`。这适合 B1 hard constraint。

为 B2 保留信息时，不应对各 child 的 defect fraction 做无权 mean；应同步传播：

```text
raw_total_count
raw_defect_count
defect_fraction = raw_defect_count / raw_total_count
```

这样 fraction 按 raw contributor 数量加权，同时不提前实现 B2。

### 4.3 Point 推荐

```text
POINT_MAPPING_RECOMMENDATION = EXPLICIT_SAME_PASS_HIERARCHICAL_PROVENANCE
```

推荐原则：让**实际产生** `points[1]`、`points[2]`、`points[3]` 的每次 grid pass 同时返回 input-row -> output-row parent map（或等价的完整 member CSR），再组合成 raw-to-`points[3]` provenance。不得另行重跑 grid subsampling，不得用 coordinate/NN 事后配对。

强制验证至少包括：

- `len(parent_i) == points[i].shape[0]`；
- 所有 parent id 都在 `[0, points[i+1].shape[0])`；
- 每个 output row 至少有一个 contributor；
- composed raw-to-coarse map 覆盖每个 raw row 且每个 raw row 恰好一个 coarse parent；
- `len(point_intact_coarse) == points[3].shape[0]`；
- mask/provenance 与 coordinates 来自同一次 preprocessing invocation；
- M4-disabled 时仍走封存 B0 的原始调用与返回值路径。

当前 extension 尚不提供这些数据，所以当前 Point mapping 仍是 `UNRESOLVED`。

## 5. CT raw -> 20 mm support

### 5.1 轴、geometry 与 grid shape

CT array 是 `[z,y,x]`，spacing 是 image-axis `[s_x,s_y,s_z]` mm。对 grid size `g`，代码定义：

```text
max_displacement_xyz = ([X,Y,Z] - 1) * spacing_xyz
shape_xyz(g)          = floor(max_displacement_xyz / g) + 1
shape_zyx(g)          = reverse(shape_xyz(g))
```

量化公式是：

```text
q_x = floor(x * s_x / g)
q_y = floor(y * s_y / g)
q_z = floor(z * s_z / g)
```

grid 是随 CT image axes 旋转的 image-axis grid；direction 不参与 cell identity，但参与从 cell coordinate 到 LPS physical coordinate 的变换。

### 5.2 5 mm context

`build_ct_context_5mm()`：

1. 只选 `ct_volume > -500 HU` 的 foreground raw voxels；
2. 用上式在 `g = 5 mm` 下量化；
3. linear key 采用 x-fastest 的 z-major 展平：

   ```text
   linear = q_z * size_y * size_x + q_y * size_x + q_x
   ```

4. 每个 cell 聚合 foreground HU 的 `sum/count`，再 clip 到 `[-500,2000]` 并归一化到 `[0,1]`；
5. 最终 `occupied_linear = np.flatnonzero(counts)`，严格升序；
6. `_linear_to_indices_zyx()` 生成 `[q_z,q_y,q_x]`。

输出 `ct_context_features`、`ct_context_indices`、`ct_context_linear` 具有相同的升序-linear ordering。

### 5.3 external surface

`extract_external_surface()` 定义：

```text
foreground = ct_volume > -500 HU
outside_air = 与任一 volume face background seed 通过 6-neighbour 连通的 background
external_surface = foreground AND dilate_6(outside_air)
```

位于 volume 六个 face 上的 foreground 被显式并入 surface，以表达其向 volume 外侧的邻接面。

该 surface 是 support selection 的来源，但不是 CT encoder 5 mm context 的完整 contributor membership；context 包含全部 foreground voxel。

### 5.4 20 mm support

`build_ct_support_20mm()` 对 `external_surface == True` 的 raw voxel indices 执行：

```text
q_xyz = floor(index_xyz * spacing_xyz / 20 mm)
linear = q_z * size_y * size_x + q_y * size_x + q_x
support_linear = np.unique(linear)
support_indices = inverse_linear_as_[q_z,q_y,q_x]
```

`np.unique(linear)` 默认排序，因此：

- `ct_support_linear_20mm` 严格升序且唯一；
- `ct_support_indices_20mm[j]` 与 linear row `j` 严格对应；
- support ordering 由 linear key 决定，是 deterministic；
- `ct_support_spatial_shape_20mm` 是 `[size_z,size_y,size_x]`。

physical coordinate 使用完整 20 mm cell center，而不是 raw voxel center：

```text
displacement_xyz_mm = ([q_x,q_y,q_z] + 0.5) * 20 mm
support_phys_lps     = origin_xyz_mm + direction @ displacement_xyz_mm
```

源码用 row-vector 等价式 `origin + displacement @ direction.T`。最后一个 boundary cell 的 center 可以落在原始 array physical extent 之外；这是当前冻结的 coarse-token coordinate 行为。

### 5.5 当前 raw membership

当前 support builder 只保留 external-surface cell 的 unique linear keys，不保留：

- 每个 external-surface raw voxel -> support row；
- 每个 support row -> surface raw voxels；
- 每个 support row -> cell 内全部 raw voxels；
- 每个 support row -> 5 mm context cells 或其 raw foreground contributors。

所以当前没有可直接复用的 raw CT mask membership。

## 6. CT encoder ordering 与 sparse reorder

### 6.1 输入 validation

`CTEncoder.forward()` 明确验证：

- 5 mm context indices 的 computed linear 必须严格升序且唯一；
- support indices 重算出的 linear 必须等于传入 `ct_support_linear_20mm`；
- support linear 必须严格升序且唯一；
- support index、linear、physical coordinate 数量必须一致；
- 两次 stride-2 后的 `x20.spatial_shape` 必须等于 support shape。

### 6.2 spconv reorder 的恢复

spconv 的 `x20.indices` 不被假设与 input support 同顺序。当前代码执行：

```text
x20_linear = linear(x20.indices[:, 1:])
sorted_linear, sorted_order = sort(x20_linear)
positions = searchsorted(sorted_linear, support_linear)
found = sorted_linear[positions] == support_linear
feature_rows = sorted_order[positions]
V_raw = x20.features[feature_rows]
```

若某个 support key 不存在于 x20，立即报错。随后：

```text
Xv_phys_coarse = ct_support_phys_20mm  # input support order
K = Linear(V_raw)                     # row-preserving
```

因此，尽管 spconv active rows 可能 reorder，当前显式 key lookup 会把结果恢复成 input support ordering：

```text
V_raw[j] <-> K[j] <-> ct_support_linear_20mm[j]
         <-> ct_support_indices_20mm[j] <-> ct_support_phys_20mm[j]
```

残余 validation 缺口：当前代码没有显式断言 `x20_linear` 自身唯一且 in-bounds。spconv 正常 sparse-tensor contract 应产生唯一 active coordinates，但若 M4-1B 以“全面 fail closed”为目标，应把该条件作为需要显式验证/测试的 invariant，而不是仅依赖隐含 library behavior。

## 7. CT raw mask -> support 候选方案

目标 shape：

```text
ct_intact_coarse.shape == (ct_support_indices_20mm.shape[0],)
```

### 7.1 候选比较

| 方案 | 严格对应 support | ordering 风险 | 近似 | 是否改变 CT encoder 数值路径 | fail closed | B1 适用性 |
| --- | --- | --- | --- | --- | --- | --- |
| A. support center sample | row 可按 support 顺序计算，但 center 可能不是 raw voxel center，boundary center 甚至可在 array extent 外 | 低；但需定义 rounding/out-of-bounds | 是，point sample | 可旁路，不改 encoder | 可对越界报错，不能证明 cell 全 intact | **不适合保守 hard mask** |
| B. nearest raw voxel | 可为每个 support row给一个值，但不是 contributors 聚合 | tie 与 direction/spacing 距离定义风险 | 是，NN | 可旁路 | 不能证明 cell 中无 defect | **不适合** |
| C. entire 20 mm cell raw aggregation | 若复用完全相同的 q/linear 公式并按 support key gather，可严格对应 | 可用 linear key 消除 row 风险 | 否 | 可纯 preprocessing 旁路 | 可验证 raw shape、key、counts、coverage | **适合保守 cell-level B1 候选** |
| D. only existing CT context voxel aggregation | 可按 5 mm/20 mm keys 对齐，但“context contributor”语义必须另行精确定义 | key 可确定；semantic scope 有风险 | 否 | 可旁路 | 只能验证被选 context，不能覆盖被阈值排除的 raw voxels | 是否适合取决于尚未冻结的 contributor 定义 |
| E. explicit raw-voxel membership mapping | 是；membership 以 support linear key为 join key | 最低 | 否 | 可旁路，不改 sparse features | 最强，可保存 count/fraction/provenance | **推荐的基础设施** |

### 7.2 “相关 raw voxel”的两种语义

#### 方案一：整个 20 mm physical cell

可精确定义为 array 中所有满足下式的 existing raw voxel centers：

```text
floor(index_xyz * spacing_xyz / 20 mm) == support_index_xyz[j]
```

其性质：

- 包含 foreground/background 及任何 HU 的 raw voxels；
- 与 `ct_defect_mask` 的完整 `[Z,Y,X]` domain 一致；
- boundary cell 只聚合 array 中实际存在的 raw voxels，不为 array 外区域猜 mask；
- `AND` 语义最保守：cell 内任一 defect raw voxel 都使 token 为 defect；
- 不等同于 sparse feature 的精确 receptive field。

#### 方案二：当前 context/support construction 中实际贡献的 raw voxels

这个短语在当前源码中至少有三种不同集合，不能混用：

1. **support-selection contributors**：量化到该 20 mm key 的 external-surface raw voxels；
2. **5 mm context contributors**：`HU > -500` 且进入相关 5 mm context cells 的 raw voxels；
3. **`V_raw` feature contributors**：经过 5/10/20 mm sparse convolution receptive field影响该 x20 feature 的多个 context cells；可能跨越目标 20 mm cell。

仅聚合 external-surface voxels会忽略同 cell 内非-surface defect；仅聚合 context voxels会忽略 `HU <= -500` 的 mask domain。若要求与最终 learned feature 的全部数学 contributor 一致，则需要追踪 sparse-convolution indice pairs/receptive field，语义远超“raw mask 到 support cell”的简单空间聚合。

本审计不替 M4-1B 最终规格冻结两者。正式实现前必须选择并写明上述集合；不得把它们都称作“actual contributors”。

### 7.3 CT 推荐

```text
CT_MAPPING_RECOMMENDATION = EXPLICIT_SUPPORT_KEYED_RAW_MEMBERSHIP_AGGREGATION
CT_RELEVANT_RAW_VOXEL_SCOPE = UNRESOLVED
```

推荐的共同基础设施是：复用当前 `q_xyz`、spatial shape 和 sorted `ct_support_linear_20mm`，以 linear key 生成/验证显式 raw membership 或等价的 `raw_total_count/raw_defect_count` 聚合，并严格按已有 support-linear order gather。不得通过 physical NN 或 spconv row order 配对。

若后续冻结“entire 20 mm cell”，则可直接对该 key 下全部 existing raw voxels执行 AND，并保存 defect fraction/count 供未来 B2 使用。若后续冻结“context contributors”，必须先进一步精确定义是上面的集合 1、2 还是 3。当前 scope 未冻结，所以当前 CT mapping 状态仍是 `UNRESOLVED`。

## 8. M4-1B 必须验证的 alignment invariants

### 8.1 Point

最终必须逐样本验证：

```text
point_defect_mask.dtype == bool
point_defect_mask.shape == (points[0].shape[0],)
point_intact_coarse.dtype == bool
point_intact_coarse.shape == (points[3].shape[0],)
P_raw.shape[0] == Q.shape[0] == Xp_phys_coarse.shape[0] == points[3].shape[0]
```

并验证每一级 parent map 的 length/range/full-coverage/non-empty-parent 以及 provenance invocation identity。shape 相等只是一项必要条件，不是 row alignment 证明。

### 8.2 CT

最终必须逐样本验证：

```text
ct_defect_mask.dtype == bool
ct_defect_mask.shape == ct_volume.shape
ct_intact_coarse.dtype == bool
ct_intact_coarse.shape == (ct_support_indices_20mm.shape[0],)
Nv == len(support_indices) == len(support_linear) == len(support_phys)
Nv == V_raw.shape[0] == K.shape[0] == Xv_phys_coarse.shape[0]
```

还必须验证：

- mask 与 CT 的 spacing/origin/direction/shape 已通过 M4-1A geometry identity；
- support linear 由 support indices 重算后逐项相等、唯一、严格升序；
- coarse mask 以同一 support linear key 顺序 gather；
- 每个 support row 有冻结语义所要求的至少一个 related raw voxel；
- `V_raw` lookup 后与 support ordering一致；
- non-identity direction 下 physical coordinate 仍满足 `origin + direction @ displacement`，但 membership 仍按 image-axis index量化；
- x20 active linear key 唯一、in-bounds，并包含所有 support keys。

## 9. 明确禁止的映射/容错方式

下列行为在 M4-enabled 路径均应明确禁止并 fail closed：

| 行为 | 判断 | 原因 |
| --- | --- | --- |
| shape 相同就假设 aligned | **禁止** | length 不证明 row identity/provenance |
| truncate | **禁止** | 静默丢失 mask 或 token |
| pad | **禁止** | 发明无 provenance 的 mask rows |
| broadcast | **禁止** | 把 sample/cell 语义伪装成 token alignment |
| 任意 nearest-neighbor 修复 | **禁止** | centroid/support center 不等于 contributor identity；存在 tie/阈值歧义 |
| 根据坐标排序后强行对齐 | **禁止** | duplicate/float/tie 风险，且 Point 原 grid row 没有稳定 key输出 |
| 默认全 intact | **禁止** | 会重新启用应被 hard constraint 排除的 token |
| mask 缺失时 fallback | **禁止** | M4-enabled 必须由 M4-1A/M4-1B fail closed |
| 根据 token 数猜 mapping | **禁止** | count 相等不证明 membership 或 ordering |

B0 例外不是 fallback：三个 M4 开关全 False 时本来就不选择 defect variant、不要求 mask，也不进入 M4 mapping。这是显式 gating contract，不是 M4-enabled 的容错。

## 10. M4-1B 测试设计建议（仅设计）

### 10.1 Point tests

1. **all intact**：全部 raw `True`，所有有 contributor 的 coarse token 均为 `True`。
2. **all defect**：全部 raw `False`，所有 coarse token 均为 `False`。
3. **mixed raw mask**：用显式 parent map 手算每级 AND 与 raw defect counts。
4. **mixed contributors in one coarse cell**：至少一个 `True` 与一个 `False` 进入同一最终 token，结果必须 `False`，fraction/count 正确。
5. **hierarchical weighting**：child raw counts 不同，验证 B1 AND 与为未来保留的 raw-count-weighted fraction；禁止对 child fractions无权平均。
6. **same-pass row identity**：打乱/模拟 unordered output rows，parent ids 仍必须直接指向实际 output rows；不得按坐标重配。
7. **parent validation fail closed**：wrong length、negative/out-of-range id、empty parent、duplicate assignment contract、uncovered raw row分别报错。
8. **shape/order mismatch fail closed**：mask length等于 N 但故意 provenance id错位，必须失败或由可审计 expected mapping 检出。
9. **encoder alignment**：逐行验证 coarse mask、`points[3]`、`P_raw`、`Xp_phys_coarse` count/order contract。
10. **B0 unchanged**：无 raw defect mask 时不构造 provenance/mask；既有 `points[]`、neighbor tables和 encoder outputs 按封存 tolerances/identity要求不变。
11. **repeat/build determinism guard**：测试不得把独立 grid重跑当 oracle；同时记录当前 unordered-map ordering风险。

### 10.2 CT tests

1. **all intact**：冻结的 related-voxel set 全 `True`，全部 support token 为 `True`。
2. **all defect**：全部 related raw voxels `False`，全部 support token 为 `False`。
3. **one defect voxel inside support cell**：同 cell 其他 voxels intact，保守 AND 结果必须 `False`。
4. **boundary support cell**：cell center 可在 array extent外，只聚合 array 中实际存在且量化到该 key 的 voxels，不 pad array外 mask。
5. **non-identity direction**：support membership key 不受 direction改变；physical coordinate正确应用 direction，mask row仍按 support linear 对齐。
6. **anisotropic spacing**：手算 `floor(index_xyz * spacing_xyz / 20)` 与 boundary key。
7. **sorted unique support**：输入 surface voxel遍历顺序改变时，support linear、indices、coarse mask顺序仍按升序 key一致。
8. **spconv reorder recovery**：构造 reordered x20 indices/features，验证 `V_raw` 与 mask 都恢复到 support ordering；missing/duplicate/out-of-bounds x20 key fail closed。
9. **shape/order mismatch fail closed**：相同长度但交换 mask key对应关系，必须由 key/provenance validation 检出。
10. **scope-specific fixture**：分别构造“同 cell 非-surface defect”和“低于 foreground threshold 的 defect”，用于冻结 entire-cell 与 context-only 语义时显式展示差异。
11. **B0 unchanged**：无 raw defect mask时 CT context/support/encoder数值与当前 B0 完全一致，不生成默认 coarse mask。

不得使用正式 evaluation subject 作为这些单元测试的开发 oracle；应使用可手算的 synthetic fixtures。

## 11. 预计 M4-1B 实现涉及的文件（本轮未修改）

在采用上述推荐方案时，最小预计范围为：

| 文件 | 预计职责 | B0 保护要求 |
| --- | --- | --- |
| `geotransformer/extensions/cpu/grid_subsampling/grid_subsampling_cpu.h/.cpp` | 在实际 voxel pass 中产生 parent/member provenance 或 count aggregate | coordinate accumulation与输出逻辑不得改变；无 M4 provenance请求时保持原路径 |
| `geotransformer/extensions/cpu/grid_subsampling/grid_subsampling.h/.cpp` | 暴露新增 provenance tensors | 现有 `grid_subsampling(points,lengths,voxel_size)` contract保持兼容 |
| `geotransformer/extensions/pybind.cpp` | 若采用独立 extension entry point，则注册该接口 | 不替换现有 B0 binding |
| `geotransformer/modules/ops/grid_subsample.py` | Python wrapper传递可选 provenance结果 | 默认返回值与 B0完全一致 |
| `geotransformer/utils/data.py` | 在同一三级 subsampling chain中保存/组合 parent maps | 既有 `points/neighbors/subsampling/upsampling` 数值与 ordering不得改变 |
| `experiments/geotransformer.pointct.baseline_v1/dataset.py` | M4-enabled collate中聚合 Point mask；按既有 CT support key聚合 CT mask | raw mask不存在时不扫描、不计算、不产生默认 mask |
| `tests/pointct/` 下新增专用 M4-1B contract tests，或最小扩展相邻 Point/CT preprocessing tests | 验证上节 synthetic cases与 B0 regression | 不使用正式 test subject |

仅完成 M4-1B mapping 时，`geotransformer/datasets/registration/pointct/dataset.py` 的 M4-1A raw loader、`point_encoder.py` 和 `ct_encoder.py` 的 feature数值路径理论上不需要改变。若实现选择让 `training.py` 把 coarse masks移动到 device/交给后续模块，应只增加显式字段转运；真正使用 mask 改 logits属于 M4-2，不应提前进入 M4-1B。

## 12. 未解决问题与 M3/B0 风险

### 12.1 仍未解决

1. Point extension尚未提供 same-pass parent/member provenance；当前不能严格构造 `point_intact_coarse`。
2. Point `unordered_map` output ordering没有 portable deterministic contract。推荐映射能保证同一次运行内 row alignment，但不能把独立重跑当作稳定身份来源。
3. CT 的 related raw voxel scope尚未冻结：entire 20 mm cell、surface-selection contributors、5 mm context contributors或 sparse receptive-field contributors语义不同。
4. CT encoder尚未显式检查 x20 active linear keys唯一且 in-bounds；应在最终 alignment contract中补足验证或通过明确 library contract与测试证明。
5. B2 将来采用何种 defect fraction/reliability仍未冻结；M4-1B只应保留 raw total/defect counts等无损统计，不提前决定 modulation。

### 12.2 对 M3/B0 的风险判断

发现两类需要隔离的风险：

- **Point extension风险较高**：改变 unordered-map data structure、iteration、float accumulation或调用次数都可能改变 `points[level]` row/order/数值，继而改变 neighbor tables和 KPConv输出。实现必须采用 optional/same-pass provenance，并以 B0 exact regression保护原路径。
- **CT风险较低但非零**：mask aggregation可完全旁路既有 context/support/encoder；但若为方便而重建、重排或替换 support arrays，就会破坏 `V_raw`/physical row contract。必须复用既有 support linear keys，仅旁路生成与其同序的 mask。

在不触发 M4 mapping 的 B0 下，没有任何必要修改 encoder coordinates/features、support selection、KPConv/spconv或 matching数值路径。

## 13. 审计依据的关键源码位置

- `geotransformer/datasets/registration/pointct/dataset.py`: `_get_defect_item`, `pointct_collate_fn`
- `experiments/geotransformer.pointct.baseline_v1/config.py`: Point scale/stages/voxel/radius 与 CT 5/20 mm 配置
- `experiments/geotransformer.pointct.baseline_v1/dataset.py`: `m2_point_collate_fn`, `build_ct_context_5mm`, `extract_external_surface`, `build_ct_support_20mm`, `m2_ct_collate_fn`
- `geotransformer/utils/data.py`: `single_collate_fn_stack_mode`, `precompute_data_stack_mode`
- `geotransformer/modules/ops/grid_subsample.py`: 当前 Python grid wrapper
- `geotransformer/extensions/cpu/grid_subsampling/grid_subsampling_cpu.h/.cpp`: centroid、voxel key 与 unordered output
- `geotransformer/extensions/cpu/grid_subsampling/grid_subsampling.h/.cpp`: Torch binding signature/return values
- `experiments/geotransformer.pointct.baseline_v1/backbone.py`: `KPConvFPN` query level 与 coarse feature row flow
- `geotransformer/modules/kpconv/kpconv.py` 及 `modules.py`: KPConv output第一维保持 query-row顺序
- `experiments/geotransformer.pointct.baseline_v1/point_encoder.py`: `P_raw` 与 `points[3]`/physical coordinate alignment
- `experiments/geotransformer.pointct.baseline_v1/ct_encoder.py`: support validation、spconv linear-key recovery 与 `V_raw` ordering
- `experiments/geotransformer.pointct.baseline_v1/training.py`: 当前 encoder input assembly，不含 M4 coarse masks

---

No M4-1B implementation was added.

No defect artifact was modified.

No formal evaluation was executed.
