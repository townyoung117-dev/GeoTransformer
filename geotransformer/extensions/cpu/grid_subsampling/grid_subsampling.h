#pragma once

#include <vector>
#include "../../common/torch_helper.h"

std::vector<at::Tensor> grid_subsampling(
  at::Tensor points,
  at::Tensor lengths,
  float voxel_size
);

std::vector<at::Tensor> grid_subsampling_with_parent(
  at::Tensor points,
  at::Tensor lengths,
  float voxel_size
);
