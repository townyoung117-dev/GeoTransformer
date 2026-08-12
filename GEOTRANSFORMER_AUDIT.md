# GeoTransformer 官方工程代码审查

## 0. 审查范围、依据与结论边界

本文档只审查当前工作树中的代码和随仓库提供的数据/元数据，不运行训练、测试或评估，不安装依赖，不改变 Python、PyTorch、CUDA 或编译环境，也不对源码做任何补丁。

- 工作分支：`point_ct_baseline`
- 审查时 HEAD：`e7a135af4c318ff3b8d7f6c963df094d7e4ea540`
- `upstream` URL：`https://github.com/qinzheng93/GeoTransformer.git`
- 审查开始时 `git status --short` 为空。
- 本地没有 `upstream/*` remote-tracking ref；在不执行网络 fetch 的限制下，当前 HEAD 与远端官方仓库某一具体分支/标签是否逐字节一致，**无法从当前仓库确定**。
- 仓库没有 `data/3DMatch/data/` 点云载荷，也没有 `weights/`。因此，各样本的实际点数、运行时校准出的邻居上限、模型学习后的特征/匹配分布、Sinkhorn 参数 `alpha` 的训练值以及复现实验指标，**无法从当前仓库确定**。
- 当前仓库完全没有 CT volume、医学影像读取、体素物理空间、Sparse 3D CNN 或 Point Cloud–CT 标注定义。所有这部分的具体设计均**无法从当前仓库确定**，本文只做模块边界分类，不设计或实现目标网络。

下面所有 shape 均来自静态代码追踪。记第 `i` 个点层级中 reference/source 点数为 `R_i`/`S_i`，总数为 `T_i = R_i + S_i`；`P` 是进入细匹配的 patch pair 数，`K=64` 是每个 patch 的点上限。

## 一、工程结构

### 1. 根目录

| 路径 | 作用 |
|---|---|
| `README.md` | 论文、安装说明、3DMatch/KITTI/ModelNet 数据组织、训练测试命令和公开结果说明。 |
| `setup.py` | 以 `torch.utils.cpp_extension.CUDAExtension` 编译 `geotransformer.ext`，源文件实际是 CPU grid subsampling/radius neighbors 及 pybind 包装；构建入口仍使用 `CUDAExtension`。 |
| `requirements.txt` | Python 依赖声明。此次审查未安装或变更任何依赖。 |
| `assets/` | README 使用的 `teaser.png`。 |
| `.idea/` | IDE 工程配置，与算法运行链无关。 |
| `experiments/` | 3DMatch、KITTI、ModelNet 三套自包含实验配置、数据装配、backbone、model、loss、训练/测试/评估入口。 |
| `geotransformer/` | 公共训练框架、数据集、算子、KPConv、Transformer、matching、Sinkhorn、registration、loss 和工具代码。 |
| `data/` | 随仓库提供的元数据、benchmark GT、demo 数组和少量数据准备脚本；不含完整 3DMatch 点云载荷。 |

仓库根目录不存在独立的 `configs/` 和 `scripts/` 目录。配置采用每个实验目录内的 `config.py`；shell 工具也放在实验目录内，数据准备工具放在 `data/<dataset>/` 内。

### 2. `experiments/`

包含三个实验目录：

1. `experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/`
   - 本文主审查对象。
   - `config.py`：全部 3DMatch 配置；模块导入时会调用 `ensure_dir` 创建 `output/...` 子目录。
   - `dataset.py`：构造 train/val/test dataset、邻居校准和 DataLoader。
   - `backbone.py`：四阶段 `KPConvFPN`。
   - `model.py`：端到端 `GeoTransformer` 模型及完整匹配/估计链。
   - `loss.py`：coarse/fine loss 与在线 evaluator。
   - `trainval.py`、`test.py`、`eval.py`：训练、推理、离线 benchmark 评估。
   - `eval_dgr.py`：另一套按 RRE/RTE 阈值汇总的评估脚本。
   - `eval.sh`、`eval_all.sh`：shell 调度脚本。
   - `demo.py`：两个 `.npy` 点云的演示入口。
2. `experiments/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn/`
   - KITTI 五阶段变体，文件组织与 3DMatch 基本平行。
3. `experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/`
   - ModelNet/RPMNet 设置，使用 iteration-based trainer，另有对应数据、模型和损失。

实验目录通过在该目录中直接运行脚本并使用 `from config import ...`、`from model import ...` 这类局部导入工作，不是标准 Python package。

### 3. `geotransformer/`

| 子目录 | 作用 |
|---|---|
| `engine/` | `BaseTrainer`、`EpochBasedTrainer`、`IterBasedTrainer`、`BaseTester`、`SingleTester`、日志器；负责参数解析、CUDA/DDP、训练循环、验证循环、snapshot、TensorBoard 和汇总。 |
| `datasets/registration/` | `threedmatch/`、`kitti/`、`modelnet/` 三类 pair dataset。3DMatch 还提供 benchmark log/info 解析与 transform error。 |
| `extensions/` | pybind C++ 入口、CPU grid subsampling 和 radius neighbor search 实现、第三方头文件。 |
| `modules/kpconv/` | 非 deformable `KPConv`、残差块、pool/upsample/interpolation、kernel point 加载。 |
| `modules/geotransformer/` | geometric structure embedding、Geometric Transformer、superpoint matching/target、point matching、Local-to-Global Registration。 |
| `modules/transformer/` | vanilla、PE、RPE、LRPE attention/transformer 层、conditional block 编排、位置编码和 FFN 输出层。3DMatch 主路径使用 RPE self-attention + vanilla cross-attention。 |
| `modules/sinkhorn/` | `LearnableLogOptimalTransport`。 |
| `modules/registration/` | correspondence 提取/GT node correspondence、weighted Procrustes、registration metrics。 |
| `modules/ops/` | pairwise distance、index select、point partition、grid subsample、radius search、刚体变换和角度工具。 |
| `modules/loss/` | Circle Loss / Weighted Circle Loss。 |
| `modules/layers/` | 通用 conv/norm/activation/dropout factory。 |
| `transforms/` | 点云归一化、采样、旋转、尺度、抖动、裁剪等 NumPy 数据增强函数；3DMatch 主 dataset 没有通过该目录调用增强，而是直接使用 `utils.pointcloud`。 |
| `utils/` | DataLoader/collate/precompute、NumPy 点云与 registration 工具、Open3D、日志格式、计时、统计和可视化。 |

### 4. `data/`

- `data/3DMatch/metadata/`
  - `train.pkl`：20642 个 pair；`val.pkl`：1331 个 pair；`3DMatch.pkl`：1623 个 pair；`3DLoMatch.pkl`：1781 个 pair。
  - 每项代码可见字段为 `scene_name`、`frag_id0`、`frag_id1`、`overlap`、`pcd0`、`pcd1`、`rotation(3,3)`、`translation(3,)`。
  - `benchmarks/{3DMatch,3DLoMatch}/<scene>/gt.log|gt.info|gt_overlap.log` 提供官方 benchmark GT。
  - `split/` 提供 train/val 场景列表。
  - 元数据引用的实际点云应位于 `data/3DMatch/data/`，当前仓库没有该目录。
- `data/Kitti/metadata/`：train/val/test 元数据；`downsample_pcd.py` 是 KITTI 数据降采样工具。
- `data/ModelNet/split_data.py`：生成 ModelNet split pickle 的工具；完整数据不在仓库中。
- `data/demo/`：`ref.npy`、`src.npy`、`gt.npy`，用于 `demo.py`。

### 5. 配置与脚本

- 无统一配置系统；3DMatch 配置准确路径为 `experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/config.py`，使用 `EasyDict`。
- 3DMatch shell 工具是同目录下 `eval.sh` 和 `eval_all.sh`。
- 数据工具只有 `data/Kitti/downsample_pcd.py` 和 `data/ModelNet/split_data.py`。
- 未发现单元测试目录、CI 配置或自动化 shape test。

## 二、3DMatch 训练、测试和评估入口

统一实验目录缩写为：

`EXP = experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn`

| 项目 | 准确路径 | 入口/符号 |
|---|---|---|
| 训练入口 | `EXP/trainval.py` | `main()` → `Trainer(cfg)` → `EpochBasedTrainer.run()`；每步为 `Trainer.train_step()`。 |
| 验证入口 | `EXP/trainval.py` | 每个 epoch 后由 `EpochBasedTrainer.inference_epoch()` 调用 `Trainer.val_step()`。 |
| 测试/特征导出入口 | `EXP/test.py` | `main()` → `Tester(SingleTester)` → `SingleTester.run()`；`test_step()` 前向，`after_test_step()` 保存 `.npz`。 |
| 主 evaluation 入口 | `EXP/eval.py` | `main()` → `eval_one_epoch()`；读取测试导出的 `.npz`，支持 `lgr`、`ransac`、`svd`。 |
| shell evaluation | `EXP/eval.sh` | 第三个参数为字面值 `test` 时先运行 `test.py`，然后总是运行 `eval.py --method=lgr`。 |
| 批量 evaluation | `EXP/eval_all.sh` | epoch 20–40 逐个执行 test 和 LGR eval。 |
| 备用 evaluation | `EXP/eval_dgr.py` | `main()` → `eval_one_epoch()`；使用不同的目录和注册接受准则。 |
| config | `EXP/config.py` | `make_cfg()`；包含 data/train/test/eval/ransac/optim/backbone/model/coarse_matching/geotransformer/fine_matching/loss。 |
| model 定义 | `EXP/model.py` | `GeoTransformer(nn.Module)`、`create_model()`。 |
| backbone 定义 | `EXP/backbone.py` | `KPConvFPN(nn.Module)`。 |
| dataset 装配 | `EXP/dataset.py` | `train_valid_data_loader()`、`test_data_loader()`。 |
| dataset 实体 | `geotransformer/datasets/registration/threedmatch/dataset.py` | `ThreeDMatchPairDataset`、`__getitem__()`。 |
| loss 定义 | `EXP/loss.py` | `CoarseMatchingLoss`、`FineMatchingLoss`、`OverallLoss`。 |
| 在线 evaluator | `EXP/loss.py` | `Evaluator`。 |

README 给出的训练命令要求在 `EXP` 中运行 `python trainval.py`。测试参数 `--snapshot`、`--test_epoch` 并非在 `test.py` 本身声明，而是由 `geotransformer/engine/base_tester.py::inject_default_parser()` 注入。

## 三、数据调用链：从一个 sample 到 loss/final output

### 1. Dataset sample

`geotransformer/datasets/registration/threedmatch/dataset.py::ThreeDMatchPairDataset.__getitem__()`：

1. 从 `<subset>.pkl` 读取 pair 元数据。
2. `_load_point_cloud()` 通过 `torch.load(data_root/file_name)` 读 reference/source 点数组；若超过 `point_limit=30000`，训练时随机截取。
3. `_augment_point_cloud()` 在训练时随机旋转其中一云，并同步更新 GT `rotation`/`translation`，再给两云加入幅值受 `augmentation_noise=0.005` 控制的均匀噪声。
4. `get_transform_from_rotation_translation()` 组成 `src→ref` 的 `(4,4)` 变换。
5. 返回：
   - `ref_points: (R_0,3)`、`src_points: (S_0,3)`，`float32`；
   - `ref_feats: (R_0,1)`、`src_feats: (S_0,1)`，内容全 1；
   - `transform: (4,4)`；
   - scene/frame/overlap 元信息。

3DMatch 实验未启用 dataset 的 `return_corr_indices`，dense GT correspondence 不是在 dataset 中预先生成。

### 2. DataLoader、collate 和 preprocessing

`EXP/dataset.py::train_valid_data_loader()` / `test_data_loader()`：

1. 构造 `ThreeDMatchPairDataset`。
2. `geotransformer/utils/data.py::calibrate_neighbors_stack_mode()` 扫描训练样本，使用邻域数量直方图的 80% 分位式规则得到四层 `neighbor_limits`。
3. `build_dataloader_stack_mode()` → `geotransformer/utils/torch.py::build_dataloader()` → PyTorch `DataLoader`。
4. collate 为 `registration_collate_fn_stack_mode()`：按 `[ref_1,...,ref_B,src_1,...,src_B]` 堆叠点和特征。
5. `precompute_data_stack_mode()`：逐层 `grid_subsample()`，再计算 within-level `neighbors`、encoder `subsampling` 和 decoder `upsampling` 索引。
6. `grid_subsample()` 和 `radius_search()` 分别调用编译扩展 `geotransformer.ext.grid_subsampling` / `radius_neighbors`；这些预计算发生在 CPU collate 侧。

batch size 1 时，模型收到的核心结构为：

- `features: (T_0,1)`；
- `points[i]: (T_i,3)`，`i=0..3`；
- `lengths[i]: (2,) = [R_i,S_i]`；
- `neighbors[i]: (T_i,H_i)`；不足邻居的位置以 support point 数量作为 shadow index；
- `subsampling[i]: (T_{i+1},H_i)`，用于 strided KPConv；
- `upsampling[i]: (T_i,H_{i+1})`，decoder 实际只取第一列做 nearest upsample；
- `transform: (4,4)` 及元数据。

`EpochBasedTrainer.train_epoch()` / `SingleTester.run()` 调用 `geotransformer/utils/torch.py::to_cuda()`，递归把上述 tensor 移到 CUDA。

重要硬约束：collate 理论上支持 `B>1`，但 `EXP/model.py::GeoTransformer.forward()` 只读取 `lengths[level][0]` 作为 reference 长度，并把其余全部当 source。因此当前模型逻辑只对每 GPU batch size 1 成立；配置也明确设置 train/test batch size 为 1，README 用 DDP 扩展总 batch。

### 3. 模型内部总链

`EXP/model.py::GeoTransformer.forward()` 的实际顺序为：

1. 从堆叠张量按 `lengths[*][0]` 切出原始层、fine 层和 coarse 层两云。
2. `point_to_node_partition()`：分别把 reference/source fine points 分配给本云 coarse nodes，形成每个 coarse node 最多 64 点的 patch。
3. `get_node_correspondences()`：用 GT transform 生成 coarse/node correspondence 及 overlap；这是训练 coarse loss 与训练时 patch 采样的监督。
4. `KPConvFPN.forward()`：共享参数、堆叠模式提取多尺度特征；length-aware 邻域保证 ref/src 之间在 backbone 内不会互相卷积。
5. `GeometricTransformer.forward()`：对两云 coarse features 执行六个 self/cross blocks，输出 L2-normalized coarse descriptors。
6. `SuperPointMatching.forward()`：预测 coarse superpoint pairs；训练时仍把预测 indices 写入输出供 evaluator 使用，但细匹配 patch 改由 `SuperPointTargetGenerator.forward()` 从 GT pairs 随机采样。
7. 根据选中的 node pair gather `(P,64,3)` patch points、mask 和 `(P,64,256)` fine features。
8. 点积得到 `(P,64,64)` score logits，除以 `sqrt(256)`。
9. `LearnableLogOptimalTransport.forward()` 做 Sinkhorn，得到 `(P,65,65)` log matching scores。
10. 主配置 `use_dustbin=False`，推理 correspondence 前删去最后一行/列，送入 `LocalGlobalRegistration.forward()`。
11. LGR 产生 `ref_corr_points (C,3)`、`src_corr_points (C,3)`、`corr_scores (C,)` 和 `estimated_transform (4,4)`。
12. 训练时 `OverallLoss.forward()` 只对 coarse/fine matching 计算损失；最终 correspondence 和 transform 在模型的 `torch.no_grad()` 区域生成，不参与反向传播。

对应简化调用图：

```text
ThreeDMatchPairDataset.__getitem__
  -> registration_collate_fn_stack_mode
    -> precompute_data_stack_mode
      -> grid_subsample + radius_search
  -> EpochBasedTrainer/SingleTester -> to_cuda
  -> GeoTransformer.forward
    -> point_to_node_partition (ref/src)
    -> get_node_correspondences (GT)
    -> KPConvFPN
      -> KPConv / ResidualBlock / nearest_upsample
    -> GeometricTransformer
      -> GeometricStructureEmbedding
      -> RPEConditionalTransformer
         [RPE self, vanilla cross] x 3
    -> SuperPointMatching 或训练时 SuperPointTargetGenerator
    -> patch point/feature gather
    -> feature dot-product score
    -> LearnableLogOptimalTransport
    -> LocalGlobalRegistration
      -> top-k mutual correspondence
      -> local weighted Procrustes hypotheses
      -> global inlier verification/refinement
      -> weighted Procrustes -> 4x4 src-to-ref transform
    -> OverallLoss / Evaluator 或 test final output
```

## 四、KPConv Backbone

### 1. 实现位置

- 实验 FPN：`EXP/backbone.py::KPConvFPN`
- 核心卷积：`geotransformer/modules/kpconv/kpconv.py::KPConv`
- block：`geotransformer/modules/kpconv/modules.py::{ConvBlock,ResidualBlock,UnaryBlock,LastUnaryBlock}`
- pooling/upsample：`geotransformer/modules/kpconv/functional.py`
- kernel points：`geotransformer/modules/kpconv/kernel_points.py::load_kernels()`
- 固定 disposition：`geotransformer/modules/kpconv/dispositions/k_015_center_3D.ply`

这是固定 kernel point、非 deformable 的 KPConv。`KPConv.forward(s_feats,q_points,s_points,neighbor_indices)` 接收：

- support feature `(N,C_in)`；
- query point `(M,3)`；
- support point `(N,3)`；
- neighbor index `(M,H)`；
- 输出 query feature `(M,C_out)`。

15 个 kernel points 的影响按 `clamp(1-distance/sigma, min=0)` 计算，再汇聚邻居特征并乘每个 kernel 的 `(C_in,C_out)` 权重。

### 2. 四层点结构和空间尺度

配置：`num_stages=4`、`init_voxel_size=0.025`、`init_radius=0.0625`、`init_sigma=0.05`。

`precompute_data_stack_mode()` 在 level 0 不做 grid subsampling，并在每轮末尾先把 voxel size 乘 2；所以代码实际对后续层调用的 grid voxel size 和 within-level radius 为：

| level | `points[i]` | 相对上一层 | 代码中的 grid voxel size | within-level radius |
|---|---:|---|---:|---:|
| 0 | `(T_0,3)` | dataset 原始输入 | 不调用 grid subsample | 0.0625 |
| 1 | `(T_1,3)` | level 0 下采样 | 0.05 | 0.125 |
| 2 | `(T_2,3)` | level 1 下采样 | 0.10 | 0.25 |
| 3 | `(T_3,3)` | level 2 下采样 | 0.20 | 0.50 |

README 要求用户数据与 3DMatch 的 2.5 cm voxel scale 对齐，但当前仓库没有实际预处理点云，dataset 输入是否已事先按 0.025 m 下采样，**无法从当前仓库确定**。各次 grid subsampling 后的精确 `T_i` 取决于点分布，不存在代码规定的固定倍率，故精确点数量**无法从当前仓库确定**。

### 3. encoder/decoder feature shape

3DMatch 配置 `input_dim=1`、`init_dim=64`、`output_dim=256`：

| 阶段 | 主要 block | 点数 | 输出维度 |
|---|---|---:|---:|
| 输入 | all-one feature | `T_0` | 1 |
| encoder stage 1 | `ConvBlock` + `ResidualBlock` | `T_0` | 128 |
| encoder stage 2 | strided `ResidualBlock` + 2 residual | `T_1` | 256 |
| encoder stage 3 | strided `ResidualBlock` + 2 residual | `T_2` | 512 |
| encoder stage 4 | strided `ResidualBlock` + 2 residual | `T_3` | 1024 |
| decoder stage 3 | upsample 1024，与 stage-3 512 concat → 1536，再 unary | `T_2` | 512 |
| decoder stage 2 | upsample 512，与 stage-2 256 concat → 768，再 last unary | `T_1` | 256 |

`KPConvFPN.forward()` 最终返回顺序为：

```text
feats_list = [level-1 (T_1,256), level-2 (T_2,512), level-3 (T_3,1024)]
```

主模型取：

- `feats_f = feats_list[0]`：fine features，reference/source 分别 `(R_1,256)` / `(S_1,256)`；
- `feats_c = feats_list[-1]`：coarse features，分别 `(R_3,1024)` / `(S_3,1024)`。

level 0 的 full-resolution feature 不由 FPN decoder 输出；level 2 feature只作为中间结果保留在 list 中，主模型不直接使用。

## 五、Geometric Transformer

### 1. 代码位置和输入输出

- wrapper：`geotransformer/modules/geotransformer/geotransformer.py::GeometricTransformer`
- geometric embedding：同文件 `GeometricStructureEmbedding`
- block 编排：`geotransformer/modules/transformer/conditional_transformer.py::RPEConditionalTransformer`
- RPE self-attention：`geotransformer/modules/transformer/rpe_transformer.py`
- vanilla cross-attention：`geotransformer/modules/transformer/vanilla_transformer.py`
- sinusoidal encoding：`geotransformer/modules/transformer/positional_embedding.py::SinusoidalPositionalEmbedding`
- attention FFN：`geotransformer/modules/transformer/output_layer.py::AttentionOutput`

输入：

- `ref_points_c: (1,R_3,3)`、`src_points_c: (1,S_3,3)`；
- `ref_feats_c: (1,R_3,1024)`、`src_feats_c: (1,S_3,1024)`。

线性投影到 hidden dimension 256；经过六个 block：

```python
['self', 'cross', 'self', 'cross', 'self', 'cross']
```

最后投影到 256，输出 `(1,R_3,256)` / `(1,S_3,256)`，主模型 squeeze 并 L2 normalize 为 coarse descriptors。

### 2. Geometric Structure Embedding

`GeometricStructureEmbedding.get_embedding_indices(points)` 对每一云独立计算：

1. pairwise 欧氏距离：`dist_map (B,N,N)`；`d_indices = dist_map / sigma_d`，其中 `sigma_d=0.2`。
2. 每个 anchor 找 `angle_k=3` 个最近邻（先取 4 个再排除自身）。
3. 对 anchor 到各点的向量和 anchor 到 3 个局部近邻的向量计算夹角，得到 `a_indices (B,N,N,3)`；角度经 `180/(sigma_a*pi)` 缩放，`sigma_a=15`。
4. distance/angle indices 分别进入 256 维 sinusoidal embedding 和线性投影。
5. angle 的 3 个邻居维按配置 `max` reduction，得到 `(B,N,N,256)`。
6. distance embedding 与 angle embedding 相加，作为 RPE self-attention 的 pairwise position state。

几何 index 计算包在 `torch.no_grad()` 内；投影层仍可训练。该实现显式展开 `N×N×k` 张量，内存/计算量随 coarse point 数近似二次增长。

### 3. Self-attention 与 Cross-attention

- `self` block 使用 `RPETransformerLayer`：attention logit 是 feature `q·k` 与 `q·p_geometric` 的和，再除以每头维度平方根。四头时每头 64 维。
- `cross` block 使用普通 `TransformerLayer`，只依赖两云当前 feature，不向 cross-attention 直接传入两云之间的坐标差或几何 embedding。
- 每层均有 residual + LayerNorm，再接 256→512→256 FFN、residual + LayerNorm。
- `parallel=False` 为默认值；cross block 先更新 `feats0`，再用已经更新的 `feats0` 更新 `feats1`，不是两个方向完全并行更新。
- 主模型没有给 Transformer 传 `ref_masks/src_masks`；empty patch mask 只在之后的 coarse matching 中使用。

### 4. 默认假设

代码可直接确认的假设是：

1. 两个输入都是具有显式 `(x,y,z)` 坐标的 3D point set。
2. 两边 coarse feature 具有相同 input dimension，并通过同一个 `in_proj`、Transformer 和 `out_proj`；当前 feature 又来自同一套共享 KPConvFPN。
3. 两云内部的距离与角度结构在正确刚体对应下应可比较；embedding 对平移/旋转保持不变，但不对尺度变化保持不变。
4. `sigma_d`、matching radius、voxel size 等共同要求两云坐标单位/尺度一致。
5. 每云至少需要足够 coarse points 供 `topk(k=angle_k+1)`；当前 `angle_k=3` 意味着至少 4 个 coarse points。
6. 目标是无尺度、无形变的刚体变换；最终 Procrustes 强制旋转矩阵行列式为正，不估计缩放。

CT 特征 token 是否应具有与面部点云同义的 3D 坐标、角度结构是否可直接比较、CT spacing/origin/direction 如何进入坐标，**无法从当前仓库确定**。

## 六、Matching

### 1. Coarse-level / superpoint matching

代码：`geotransformer/modules/geotransformer/superpoint_matching.py::SuperPointMatching.forward()`。

输入：L2-normalized `ref_feats (R_3,256)`、`src_feats (S_3,256)` 以及 non-empty node masks。

步骤：

1. 删除 mask=False 的空 patch node。
2. `pairwise_distance(..., normalized=True)` 得到 descriptor squared distance。
3. `matching_scores = exp(-distance)`。
4. `dual_normalization=True` 时，分别做 row normalization 和 column normalization，再逐元素相乘；这不是 Sinkhorn。
5. 展平后取全局 top `min(256,R_3*S_3)`。

输出：`ref_node_corr_indices (P,)`、`src_node_corr_indices (P,)`、`node_corr_scores (P,)`。该阶段不要求 mutual one-to-one；同一个 node 可以进入多个 pair。

### 2. GT superpoint matching / training target

相关代码：

- patch 构建：`geotransformer/modules/ops/pointcloud_partition.py::point_to_node_partition()`；
- GT pair：`geotransformer/modules/registration/matching.py::get_node_correspondences()`；
- target 采样：`geotransformer/modules/geotransformer/superpoint_target.py::SuperPointTargetGenerator.forward()`。

`point_to_node_partition(fine_points,coarse_nodes,64)` 把每个 fine point 唯一分到最近 coarse node，并为每个 node保留最多 64 个属于该 node 的点。无效位置用 `points.shape[0]` sentinel，并由 mask 标记。

`get_node_correspondences()` 先用 GT transform 把 source nodes/patch points 变到 reference 坐标，通过 enclosing sphere 粗筛潜在 patch pairs，再以 point distance `<0.05` 统计两侧 patch overlap，取两侧 overlap 平均值。

训练 fine matching 时，`SuperPointTargetGenerator` 保留 overlap `>0.1` 的 GT node pairs，最多无放回随机选 128 个。此时 `node_corr_scores` 是 GT overlap；但配置 `use_global_score=False`，所以它不乘入细匹配 score。

### 3. Fine-level matching score

对选定 node pairs gather：

- point：`ref/src_node_corr_knn_points (P,64,3)`；
- mask：`(P,64)`；
- feature：`ref/src_node_corr_knn_feats (P,64,256)`。

`EXP/model.py` 计算：

```python
matching_scores = einsum('bnd,bmd->bnm', ref_feats, src_feats) / sqrt(256)
```

得到 `(P,64,64)` raw similarity，然后进入 Sinkhorn。

### 4. Correspondence generation

3DMatch 主路径使用 `geotransformer/modules/geotransformer/local_global_registration.py::LocalGlobalRegistration`，不是同目录的独立 `PointMatching` 类。

`LocalGlobalRegistration.compute_correspondence_matrix()`：

1. 对每个 ref patch point 取 source top-3；对每个 src patch point 取 reference top-3。
2. 只保留概率 `>0.05` 的位置。
3. `mutual=True`，取两方向 top-k mask 的交集。
4. 与 ref/src valid point mask 的外积相交。

随后 `torch.nonzero` 抽取 point coordinates 和对应概率。主配置：

- `topk=3`
- `mutual=True`
- `confidence_threshold=0.05`
- `use_dustbin=False`
- `use_global_score=False`
- `correspondence_limit=None`

因此最终输出的 correspondence 是所有通过 patch mask、双向 top-3 和置信度阈值的 pair；没有全局数量截断，也没有再按最终 transform 的 inlier mask裁掉后才输出。最终 transform 的估计会在内部使用 inlier-gated 权重。

## 七、Sinkhorn / Optimal Transport

具体实现：`geotransformer/modules/sinkhorn/learnable_sinkhorn.py::LearnableLogOptimalTransport`。

### 1. 输入和 dustbin

- 原始 input score：一般形式 `(B,M,N)`；3DMatch 中为 `(P,64,64)`。
- row/col mask：`(B,M)` / `(B,N)`；3DMatch 中是 patch valid masks。
- learnable dustbin score：标量参数 `alpha`，初始化为 1.0。
- 始终添加 extra column 和 extra row，输出 `(B,M+1,N+1)`；3DMatch 为 `(P,65,65)`。

regular row/column 的目标 log mass 是 `-log(num_valid_row+num_valid_col)`；dustbin row 质量与有效列数成比例，dustbin column 质量与有效行数成比例。invalid row/column 用 `-1e12` 屏蔽。代码在 log domain 交替更新 `u`、`v` 100 次，最后返回 rescaled log assignment scores。

### 2. 输出如何使用

- 训练 fine loss 使用完整 `(P,65,65)`：最后一列监督无匹配的 ref point，最后一行监督无匹配的 src point。
- 推理配置 `fine_matching.use_dustbin=False`：`EXP/model.py` 在进入 LGR 前显式执行 `matching_scores[:, :-1, :-1]`，只保留 `(P,64,64)`。
- LGR 先 `exp()` 把 log score 变为概率，再做 top-k、mutual 和阈值筛选。

`LearnableLogOptimalTransport` 本身支持 `M != N`，数学实现并不要求两侧 patch 点数相等；当前实验是因为两边都使用 `num_points_in_patch=64` 才相等。

主配置没有执行 LGR 的 dustbin 分支。源码中 `LocalGlobalRegistration.compute_correspondence_matrix()` 在 `use_dustbin=True` 时写的是 `corr_mat = corr_mat[:, -1:, -1]`，其结果 shape/语义并不是常见的“删除最后一行列”；同样表达式也存在于 `PointMatching`。由于当前配置为 False，这一分支不在 3DMatch 主调用链中；若未来启用，必须先单独核验，不能据当前主路径认定可用。

## 八、Rigid Registration

### 1. 最终模块

- orchestration：`geotransformer/modules/geotransformer/local_global_registration.py::LocalGlobalRegistration.local_to_global_registration()`
- weighted SVD：`geotransformer/modules/registration/procrustes.py::weighted_procrustes()` / `WeightedProcrustes`
- transform 应用：`geotransformer/modules/ops/transformation.py::apply_transform()`

### 2. 输入和 correspondence 筛选

输入为 patch-pair batch 的：

- `ref_knn_points (P,64,3)`、`src_knn_points (P,64,3)`；
- Sinkhorn 概率矩阵 `(P,64,64)`；
- boolean correspondence matrix `(P,64,64)`。

前置筛选已经包括 valid mask、双向 top-3 和概率阈值 0.05。`correspondence_threshold=3` 只决定一个 patch pair 是否有资格生成 local transform hypothesis；返回的 global correspondences 仍包含所有前置筛选结果。`correspondence_limit=None`，验证集不截断。

### 3. Local-to-Global Registration

1. 抽取全部 patch 的 dense correspondences 及 score。
2. 按 patch pair 分 chunk，只保留 correspondence 数至少 3 的 chunk 生成 local hypothesis。
3. 把不同长度 chunk pad 成 batch，针对每个 patch 用 weighted Procrustes 计算一个 `src→ref` 变换。
4. 用每个 local transform 对全局验证 correspondence 计算残差，以 `<acceptance_radius=0.1` 的 inlier 数选择最佳 hypothesis。
5. 用最佳 hypothesis 的 inlier mask 乘原 score，做 global weighted Procrustes。
6. 再按当前变换重算 inlier-gated score并迭代 refinement；配置 `num_refinement_steps=5`，正常有 local hypothesis 时共执行 5 次 global Procrustes（初次一次，循环四次）。
7. 如果没有任何合格 local chunk，代码先用全部 correspondence 初始化一次，再进入 global/refinement 路径。

因此官方 3DMatch 主模型明确使用 Local-to-Global Registration，不需要 RANSAC；离线 `eval.py` 仍提供 RANSAC 和全局 SVD 作为可选比较方法。

### 4. Weighted SVD 和 4×4 输出

`weighted_procrustes(src_points,ref_points,weights,return_transform=True)`：

1. 小于 `weight_thresh` 的权重清零，按 correspondence 维归一化。
2. 计算 weighted source/reference centroid。
3. 构造 cross-covariance `H`。
4. `torch.svd(H.cpu())` 在 CPU 做 SVD，再把 `U/V` 移回 CUDA。
5. 用最后一个对角元素为 `sign(det(V U^T))` 的对角矩阵修正 reflection，得到 `R∈SO(3)`。
6. `t = ref_centroid - R * src_centroid`。
7. 从 identity 初始化 homogeneous matrix，填入 `R` 和 `t`，返回 `(4,4)` 或 batched `(B,4,4)`；最后一行为 `[0,0,0,1]`。

最终 `estimated_transform` 的方向与 dataset GT 相同，都是 source 到 reference。

## 九、官方训练 Loss

实现：`EXP/loss.py`。

### 1. Coarse matching loss

`CoarseMatchingLoss.forward(output_dict)`：

- 输入 descriptors：`ref_feats_c (R_3,256)`、`src_feats_c (S_3,256)`，均已 L2 normalize。
- 监督：`gt_node_corr_indices (G,2)` 和 `gt_node_corr_overlaps (G,)`，由 GT transform、fine patch points 和 `matching_radius=0.05` 在线生成。
- descriptor distance matrix：`(R_3,S_3)`。
- positive：GT overlap `>0.1`。
- negative：overlap 恰为 0；未出现在 GT correspondence map 的 pair 也保持 0。
- positive scale：`sqrt(overlap)`。
- 损失：`WeightedCircleLoss`，参数为 positive margin 0.1、negative margin 1.4、positive optimal 0.1、negative optimal 1.4、log scale 24。

它监督 coarse descriptor space，不直接监督 top-k pair selection 的离散操作。

### 2. Fine matching loss

`FineMatchingLoss.forward(output_dict,data_dict)`：

1. 输入 selected patch points/masks，以及完整 Sinkhorn `matching_scores (P,65,65)`。
2. 用 GT `(4,4)` 把 source patch points 变到 reference。
3. 计算 `(P,64,64)` squared distance；有效且实际距离 `<positive_radius=0.05` 的位置为 GT match。
4. 某个有效 ref point 没有任何 GT match时，标记最后一列 dustbin；某个有效 src point 没有任何 GT match时，标记最后一行 dustbin。
5. bottom-right dustbin/dustbin 没有 label。
6. 损失是所有 true label 位置的负平均 log matching score：`-matching_scores[labels].mean()`。

训练时这些 patch pairs 来自 `SuperPointTargetGenerator` 采样的 GT node correspondences，而不是预测 coarse pairs。

### 3. Overall loss 与 registration loss

`OverallLoss`：

```text
loss = 1.0 * coarse_loss + 1.0 * fine_loss
```

返回 key：`loss`、`c_loss`、`f_loss`。

当前 3DMatch 官方训练代码**没有 registration loss**：没有对 `estimated_transform`、R、t、RRE、RTE 或 correspondence residual 添加可微损失。LGR 和最终 transform 生成位于 `torch.no_grad()`，`Evaluator` 只记录指标，不参与反向传播。

## 十、Evaluation

### 1. 训练/验证/测试过程中的在线指标

`EXP/loss.py::Evaluator`：

| 输出 key | 函数 | 含义 |
|---|---|---|
| `PIR` | `evaluate_coarse()` | 预测 coarse node pairs 中落入 GT node correspondence map 的比例；配置接受 overlap `>0.0`。 |
| `IR` | `evaluate_fine()` | GT 对齐后 correspondence 距离 `<0.1` 的比例。 |
| `RRE` | `evaluate_registration()` | `isotropic_transform_error()` 的相对旋转误差，单位度。 |
| `RTE` | 同上 | 相对平移 L2 误差。 |
| `RMSE` | 同上 | 代码实际计算 source points 在 `inv(GT) @ EST` 下的**平均 L2 位移**；名称为 RMSE，但实现不是平方后均值再开方。 |
| `RR` | 同上 | 上述 `RMSE < 0.2` 的 0/1 recall。 |

### 2. 测试导出

`EXP/test.py::Tester.after_test_step()` 把每个 pair 存为：

`<cfg.feature_dir>/<benchmark>/<scene>/<ref_id>_<src_id>.npz`

内容含全/fine/coarse points、coarse descriptors、预测/GT node correspondences、fine correspondences/scores、GT/估计 transform 和 overlap。

### 3. 主离线 benchmark evaluation

`EXP/eval.py::eval_one_epoch()`：

1. **Coarse matching**
   - `geotransformer/utils/registration.py::evaluate_sparse_correspondences()` 计算 precision、recall、hit ratio。
   - `eval.py` 实际汇总 coarse precision/PIR，以及 precision `>0`、`>=0.1`、`>=0.3`、`>=0.5` 的 PMR。
2. **Fine matching**
   - `evaluate_correspondences()` 返回 inlier ratio、overlap、mean residual、correspondence 数。
   - radius 为 0.1。
   - FMR 定义为 pair 的 inlier ratio `>=0.05`；汇总 FMR、IR、overlap。
3. **Registration**
   - `lgr`：直接用模型输出 `estimated_transform`。
   - `ransac`：`geotransformer/utils/open3d.py::registration_with_ransac_from_correspondences()`。
   - `svd`：对全部输出 correspondence score 使用 `weighted_procrustes()`。
   - benchmark acceptance 使用 `geotransformer/datasets/registration/threedmatch/utils.py::compute_transform_error()` 的 covariance-weighted transform error，条件为 `<0.2²`。
   - 汇总 RR，以及 accepted pairs 的 mean/median RRE、RTE。
   - 写 `<cfg.registration_dir>/<benchmark>/<scene>/est.log`。

`cfg.eval.rre_threshold=15.0` 和 `rte_threshold=0.3` 不被主 `eval.py` 用作接受条件；它们由 `eval_dgr.py` 的替代评估路径使用。

### 4. 静态审查发现的 evaluation 注意事项

以下是源码可直接确认的事实，本文未打补丁：

- `eval.py` 在 `args.num_corr is not None` 且 correspondence 超限时引用 `engine.args.num_corr`，但文件中没有 `engine` 定义；该分支按当前源码会触发名称解析错误。默认 `num_corr=None` 不进入该分支。
- `eval.sh` 只有第三参数等于 `test` 才运行 `test.py`；README 示例 `./eval.sh EPOCH 3DMatch` 没有第三参数，因此脚本实际只做 eval，并假定 `.npz` 已存在。
- 当前 `test.py` 把结果直接写在 `<feature_dir>/<benchmark>/<scene>`；`eval.py` 与其一致。`eval_dgr.py` 则读取 `<feature_dir>/<benchmark>/epoch-<test_epoch>/<scene>`，与当前 `test.py` 输出布局不一致。该脚本是否为旧版/外部 DGR 流程，**无法从当前仓库确定**。
- `eval.py` 的 `--test_epoch` 只影响日志中的 epoch 标识，不参与 feature root 路径选择。

## 十一、面向未来 Point Cloud–CT 改造的模块分类

本节只做“保留边界”分析，不实现目标网络。分类以目标草图

```text
Face Point Cloud -> KPConv -> Fp
CT Volume        -> Sparse 3D CNN -> Fv
Fp/Fv -> Cross-modal Matching -> correspondence -> weighted SVD -> R,t
```

为前提。CT 数据定义、网络结构、监督和空间坐标约定仍**无法从当前仓库确定**。

### A. 可以直接复用

这些模块不依赖输入模态，或只要求最终数据已经转换为明确的 3D correspondence：

1. **训练基础设施**
   - `geotransformer/engine/` 的训练/验证/tester 骨架、snapshot、DDP、日志和 TensorBoard。
   - `geotransformer/utils/{common,logger相关engine,summary_board,timer,average_meter,torch}.py` 的通用能力。
   - 注意 engine 当前强制 CUDA 且大量代码硬编码 `.cuda()`；若未来仍是单 CUDA 训练可直接用，若要求 CPU/多设备透明则转入 B 类。
2. **point cloud 分支的低层算子**
   - `geotransformer/modules/kpconv/kpconv.py` 和 block/pooling 基础实现，可作为 Face Point Cloud encoder 的底层组件。
   - grid subsampling、radius search、pairwise distance、index select 等点侧算子。
3. **刚体数学模块**
   - `weighted_procrustes` / `WeightedProcrustes`：只要跨模态 matcher 已给出同一物理坐标系单位下的 `(src,ref,weight)` correspondence，即可直接估计 R、t。
   - `apply_transform`、inverse/compose/decompose 相关刚体工具。
4. **通用误差公式**
   - RRE、RTE、transform composition 等 registration metrics。
   - 3DMatch benchmark 专用 covariance/log 评估不能直接当医学数据指标，但底层 RRE/RTE 可复用。
5. **Optimal Transport 核心**
   - `LearnableLogOptimalTransport` 只消费 score matrix 和两侧 mask，支持不等长 `M/N`，在新的 cross-modal score 已定义后可直接复用其数学核心。

### B. 可以保留但需要修改

1. **`EXP/backbone.py::KPConvFPN`**
   - encoder 可作为 point branch 基础，但输出层级、维度、接口要与 Sparse 3D CNN 的 CT feature level 对齐；当前 FPN 只输出 level 1/2/3 点特征。
2. **通用 Transformer 层**
   - `RPE/vanilla MultiHeadAttention`、FFN、conditional block 编排可保留。
   - `GeometricTransformer` wrapper 目前共享输入投影且两侧都是 point coordinates；未来至少需要 modality-specific projection、mask、可能不同的位置/物理空间编码。
3. **`SuperPointMatching`**
   - descriptor distance、dual normalization、top-k 机制在跨模态共同 embedding space 中仍可用；但 score calibration、双边候选数量、one-to-one 约束和 mask 语义需重新验证。
4. **fine score + Sinkhorn 接口**
   - 点积/温度/Sinkhorn 的框架可保留；CT token patch 数量可能与 point patch 不同，score 的物理含义、dustbin、阈值和监督必须修改。
5. **Local-to-Global Registration**
   - 在跨模态模块最终输出两组三维坐标后，local hypothesis + global refinement 思想可保留。
   - 当前 LGR 输入是“成对 point patches + score matrix”，并依赖 patch chunk、top-3 和固定距离阈值；如果 CT 端不是 point patch 表示，前半接口和筛选逻辑需要修改。底层 weighted SVD 属 A 类。
6. **loss 框架**
   - `OverallLoss` 的组合方式和 Circle Loss 原语可保留；positive/negative 定义、GT correspondence、dustbin 标签和跨模态特征监督必须按新数据重定义。
7. **实验 config、test 导出和 evaluation orchestration**
   - 配置/记录/导出模式可沿用，但字段、文件内容、医学影像坐标元信息及指标必须修改。

### C. 必须重写或新增

1. **目标域 dataset**
   - `ThreeDMatchPairDataset` 只读取两个点云文件，返回两组 point arrays 和 all-one point feature；不能读取 CT volume、spacing、origin、direction 或 segmentation，必须为目标数据重写 dataset。
2. **双点云 collate/preprocessing 主接口**
   - `registration_collate_fn_stack_mode()` 把 `[ref points,src points]` 合并，并对两侧都做同一套 grid/radius point preprocessing。
   - Point Cloud–CT 需要 point branch 与 sparse-volume branch 各自的数据结构和 batching，现接口必须重写；可在 point 侧内部继续调用现有预计算。
3. **CT Volume encoder**
   - 当前仓库没有 Sparse 3D CNN、dense/sparse voxel tensor 构造或医学影像加载。该分支必须新增；具体框架、输入 channel、分辨率和下采样层级**无法从当前仓库确定**。
4. **`EXP/model.py::GeoTransformer.forward()` 的整体编排**
   - 当前把两云先堆叠后送同一个 KPConvFPN，再以一个 reference length 切分；CT 不可能直接进入该共享 point backbone。
   - `lengths[*][0]`、`points[1]/points[-1]`、两侧同层 fine/coarse points、两侧相同 feature dim等全部是双点云硬契约，整体 forward 必须重写。
5. **CT 侧 point-to-node patch 构造**
   - 当前对 ref/src 都调用 `point_to_node_partition(fine_points,coarse_nodes,64)`，CT 端若是体素/token feature，不能原样调用。
6. **双点云 GT node correspondence 生成**
   - `get_node_correspondences()` 假设两侧均有 coarse node 和 point patch，并可用 GT rigid transform 后的欧氏点距计算 patch overlap。目标域的跨模态 GT 构造必须重写。
7. **GeometricStructureEmbedding 的跨模态语义层**
   - 当前两边都用 point-set 内部 pair distance/triplet angle。CT token 的几何结构若不是同分布 point sampling，不能默认可比；跨模态 geometric encoding 需重新定义或至少重写 wrapper/输入语义。
8. **FineMatchingLoss 的 GT map**
   - 当前直接对两组 patch points做 GT 变换后半径判断。CT 端的正样本可能涉及 voxel center、surface point、插值或可见性，必须按新标注重写。
9. **demo/test 文件契约**
   - `demo.py`、`test.py` 和 `.npz` 输出均假定两个 Nx3 点云；Point Cloud–CT 的输入、预处理和输出元数据必须重写。
10. **目标任务 evaluation protocol**
   - 3DMatch/3DLoMatch 的 FMR、benchmark covariance RMSE 和 scene log 不是医学跨模态配准协议。目标指标与坐标单位必须重新定义；RRE/RTE 数学函数可从 A 类复用。

### 所有默认“两侧都是 point cloud”的主路径模块清单

为避免后续误复用，集中列出：

- `ThreeDMatchPairDataset.__getitem__()`；
- `registration_collate_fn_stack_mode()`、`precompute_data_stack_mode()` 在 registration pair 上的用法；
- `EXP/backbone.py::KPConvFPN` 被同一次 forward 同时用于 ref/src 的方式；
- `EXP/model.py::GeoTransformer.forward()` 的 length 切分与两侧 point hierarchy；
- 两次 `point_to_node_partition()`；
- `get_node_correspondences()`；
- `GeometricTransformer.forward(ref_points,src_points,ref_feats,src_feats)` 当前 wrapper；
- `SuperPointMatching` 当前共同 descriptor space 假设；
- patch gather、point-feature dot product、`FineMatchingLoss`；
- `LocalGlobalRegistration` 的 patch-point 输入接口；
- `Evaluator.evaluate_fine/evaluate_registration()` 的 point correspondence/point realignment；
- `demo.py`、`test.py`、`eval.py` 的双点云数据契约。

## 十二、审查结论

GeoTransformer 3DMatch 代码的核心不是单一 Transformer，而是一条严格耦合的“共享 KPConv point hierarchy → coarse geometric Transformer → superpoint proposal → GT/预测 patch selection → fine feature OT → Local-to-Global weighted SVD”链路。

对未来 Point Cloud–CT 基础工程，最稳固的可复用边界是：训练基础设施、point branch 的 KPConv 原语、通用 attention/OT 原语、刚体变换工具和 weighted SVD。必须替换的边界是：目标 dataset、双模态 collate、CT encoder、整体 model orchestration、CT token/坐标定义、跨模态监督和目标任务评估。现有 3DMatch 的两个 point cloud 在采样结构、坐标尺度、feature dimension 和 patch 定义上的对称性，不能未经验证地迁移到 CT volume。

本次审查没有实现任何 Point Cloud–CT 模型，也没有更改任何现有 `.py` 文件。
