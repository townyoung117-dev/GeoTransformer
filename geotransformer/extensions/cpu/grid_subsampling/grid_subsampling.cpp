#include <cmath>
#include <cstring>
#include "grid_subsampling.h"
#include "grid_subsampling_cpu.h"

std::vector<at::Tensor> grid_subsampling(
  at::Tensor points,
  at::Tensor lengths,
  float voxel_size
) {
  CHECK_CPU(points);
  CHECK_CPU(lengths);
  CHECK_IS_FLOAT(points);
  CHECK_IS_LONG(lengths);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(lengths);

  std::size_t batch_size = lengths.size(0);
  std::size_t total_points = points.size(0);

  std::vector<PointXYZ> vec_points = std::vector<PointXYZ>(
    reinterpret_cast<PointXYZ*>(points.data_ptr<float>()),
    reinterpret_cast<PointXYZ*>(points.data_ptr<float>()) + total_points
  );
  std::vector<PointXYZ> vec_s_points;

  std::vector<long> vec_lengths = std::vector<long>(
    lengths.data_ptr<long>(),
    lengths.data_ptr<long>() + batch_size
  );
  std::vector<long> vec_s_lengths;

  grid_subsampling_cpu(
    vec_points,
    vec_s_points,
    vec_lengths,
    vec_s_lengths,
    voxel_size
  );

  std::size_t total_s_points = vec_s_points.size();
  at::Tensor s_points = torch::zeros(
    {total_s_points, 3},
    at::device(points.device()).dtype(at::ScalarType::Float)
  );
  at::Tensor s_lengths = torch::zeros(
    {batch_size},
    at::device(lengths.device()).dtype(at::ScalarType::Long)
  );

  std::memcpy(
    s_points.data_ptr<float>(),
    reinterpret_cast<float*>(vec_s_points.data()),
    sizeof(float) * total_s_points * 3
  );
  std::memcpy(
    s_lengths.data_ptr<long>(),
    vec_s_lengths.data(),
    sizeof(long) * batch_size
  );

  return {s_points, s_lengths};
}

std::vector<at::Tensor> grid_subsampling_with_parent(
  at::Tensor points,
  at::Tensor lengths,
  float voxel_size
) {
  CHECK_CPU(points);
  CHECK_CPU(lengths);
  CHECK_IS_FLOAT(points);
  CHECK_IS_LONG(lengths);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(lengths);
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
              "points must have shape [N,3]");
  TORCH_CHECK(lengths.dim() == 1 && lengths.size(0) > 0,
              "lengths must have shape [B] with B > 0");
  TORCH_CHECK(std::isfinite(voxel_size) && voxel_size > 0,
              "voxel_size must be finite and positive");

  std::size_t batch_size = lengths.size(0);
  std::size_t total_points = points.size(0);

  std::vector<PointXYZ> vec_points = std::vector<PointXYZ>(
    reinterpret_cast<PointXYZ*>(points.data_ptr<float>()),
    reinterpret_cast<PointXYZ*>(points.data_ptr<float>()) + total_points
  );
  std::vector<PointXYZ> vec_s_points;
  std::vector<long> vec_parent_indices;

  std::vector<long> vec_lengths = std::vector<long>(
    lengths.data_ptr<long>(),
    lengths.data_ptr<long>() + batch_size
  );
  std::vector<long> vec_s_lengths;
  std::size_t declared_total_points = 0;
  for (auto length : vec_lengths) {
    TORCH_CHECK(length > 0, "every stacked cloud length must be positive");
    declared_total_points += static_cast<std::size_t>(length);
  }
  TORCH_CHECK(declared_total_points == total_points,
              "lengths sum must equal the input point count");

  grid_subsampling_with_parent_cpu(
    vec_points,
    vec_s_points,
    vec_parent_indices,
    vec_lengths,
    vec_s_lengths,
    voxel_size
  );
  TORCH_CHECK(vec_parent_indices.size() == total_points,
              "same-pass parent count must equal the input point count");
  TORCH_CHECK(vec_s_lengths.size() == batch_size,
              "subsampled lengths must preserve the stacked cloud count");

  std::size_t total_s_points = vec_s_points.size();
  at::Tensor s_points = torch::zeros(
    {total_s_points, 3},
    at::device(points.device()).dtype(at::ScalarType::Float)
  );
  at::Tensor s_lengths = torch::zeros(
    {batch_size},
    at::device(lengths.device()).dtype(at::ScalarType::Long)
  );
  at::Tensor parent_indices = torch::zeros(
    {total_points},
    at::device(lengths.device()).dtype(at::ScalarType::Long)
  );

  std::memcpy(
    s_points.data_ptr<float>(),
    reinterpret_cast<float*>(vec_s_points.data()),
    sizeof(float) * total_s_points * 3
  );
  std::memcpy(
    s_lengths.data_ptr<long>(),
    vec_s_lengths.data(),
    sizeof(long) * batch_size
  );
  std::memcpy(
    parent_indices.data_ptr<long>(),
    vec_parent_indices.data(),
    sizeof(long) * total_points
  );

  return {s_points, s_lengths, parent_indices};
}
