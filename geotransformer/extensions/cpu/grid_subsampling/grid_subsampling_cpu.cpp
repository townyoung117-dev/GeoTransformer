#include "grid_subsampling_cpu.h"

void single_grid_subsampling_cpu(
  std::vector<PointXYZ>& points,
  std::vector<PointXYZ>& s_points,
  float voxel_size
) {
//  float sub_scale = 1. / voxel_size;
  PointXYZ minCorner = min_point(points);
  PointXYZ maxCorner = max_point(points);
  PointXYZ originCorner = floor(minCorner * (1. / voxel_size)) * voxel_size;

  std::size_t sampleNX = static_cast<std::size_t>(
//    floor((maxCorner.x - originCorner.x) * sub_scale) + 1
    floor((maxCorner.x - originCorner.x) / voxel_size) + 1
  );
  std::size_t sampleNY = static_cast<std::size_t>(
//    floor((maxCorner.y - originCorner.y) * sub_scale) + 1
    floor((maxCorner.y - originCorner.y) / voxel_size) + 1
  );

  std::size_t iX = 0;
  std::size_t iY = 0;
  std::size_t iZ = 0;
  std::size_t mapIdx = 0;
  std::unordered_map<std::size_t, SampledData> data;

  for (auto& p : points) {
//    iX = static_cast<std::size_t>(floor((p.x - originCorner.x) * sub_scale));
//    iY = static_cast<std::size_t>(floor((p.y - originCorner.y) * sub_scale));
//    iZ = static_cast<std::size_t>(floor((p.z - originCorner.z) * sub_scale));
    iX = static_cast<std::size_t>(floor((p.x - originCorner.x) / voxel_size));
    iY = static_cast<std::size_t>(floor((p.y - originCorner.y) / voxel_size));
    iZ = static_cast<std::size_t>(floor((p.z - originCorner.z) / voxel_size));
    mapIdx = iX + sampleNX * iY + sampleNX * sampleNY * iZ;

    if (!data.count(mapIdx)) {
      data.emplace(mapIdx, SampledData());
    }

    data[mapIdx].update(p);
  }

  s_points.reserve(data.size());
  for (auto& v : data) {
    s_points.push_back(v.second.point * (1.0 / v.second.count));
  }
}

void grid_subsampling_cpu(
  std::vector<PointXYZ>& points,
  std::vector<PointXYZ>& s_points,
  std::vector<long>& lengths,
  std::vector<long>& s_lengths,
  float voxel_size
) {
  std::size_t start_index = 0;
  std::size_t batch_size = lengths.size();
  for (std::size_t b = 0; b < batch_size; b++) {
    std::vector<PointXYZ> cur_points = std::vector<PointXYZ>(
      points.begin() + start_index,
      points.begin() + start_index + lengths[b]
    );
    std::vector<PointXYZ> cur_s_points;

    single_grid_subsampling_cpu(cur_points, cur_s_points, voxel_size);

    s_points.insert(s_points.end(), cur_s_points.begin(), cur_s_points.end());
    s_lengths.push_back(cur_s_points.size());

    start_index += lengths[b];
  }

  return;
}

void single_grid_subsampling_with_parent_cpu(
  std::vector<PointXYZ>& points,
  std::vector<PointXYZ>& s_points,
  std::vector<long>& parent_indices,
  float voxel_size
) {
  PointXYZ minCorner = min_point(points);
  PointXYZ maxCorner = max_point(points);
  PointXYZ originCorner = floor(minCorner * (1. / voxel_size)) * voxel_size;

  std::size_t sampleNX = static_cast<std::size_t>(
    floor((maxCorner.x - originCorner.x) / voxel_size) + 1
  );
  std::size_t sampleNY = static_cast<std::size_t>(
    floor((maxCorner.y - originCorner.y) / voxel_size) + 1
  );

  std::size_t iX = 0;
  std::size_t iY = 0;
  std::size_t iZ = 0;
  std::size_t mapIdx = 0;
  std::unordered_map<std::size_t, SampledData> data;
  std::vector<std::size_t> input_voxel_indices;
  input_voxel_indices.reserve(points.size());

  for (auto& p : points) {
    iX = static_cast<std::size_t>(floor((p.x - originCorner.x) / voxel_size));
    iY = static_cast<std::size_t>(floor((p.y - originCorner.y) / voxel_size));
    iZ = static_cast<std::size_t>(floor((p.z - originCorner.z) / voxel_size));
    mapIdx = iX + sampleNX * iY + sampleNX * sampleNY * iZ;
    input_voxel_indices.push_back(mapIdx);

    if (!data.count(mapIdx)) {
      data.emplace(mapIdx, SampledData());
    }

    data[mapIdx].update(p);
  }

  std::unordered_map<std::size_t, long> voxel_output_rows;
  voxel_output_rows.reserve(data.size());
  s_points.reserve(data.size());
  long output_row = 0;
  for (auto& v : data) {
    s_points.push_back(v.second.point * (1.0 / v.second.count));
    voxel_output_rows.emplace(v.first, output_row);
    output_row += 1;
  }

  parent_indices.reserve(input_voxel_indices.size());
  for (auto voxel_index : input_voxel_indices) {
    parent_indices.push_back(voxel_output_rows.at(voxel_index));
  }
}

void grid_subsampling_with_parent_cpu(
  std::vector<PointXYZ>& points,
  std::vector<PointXYZ>& s_points,
  std::vector<long>& parent_indices,
  std::vector<long>& lengths,
  std::vector<long>& s_lengths,
  float voxel_size
) {
  std::size_t start_index = 0;
  std::size_t output_offset = 0;
  std::size_t batch_size = lengths.size();
  for (std::size_t b = 0; b < batch_size; b++) {
    std::vector<PointXYZ> cur_points = std::vector<PointXYZ>(
      points.begin() + start_index,
      points.begin() + start_index + lengths[b]
    );
    std::vector<PointXYZ> cur_s_points;
    std::vector<long> cur_parent_indices;

    single_grid_subsampling_with_parent_cpu(
      cur_points,
      cur_s_points,
      cur_parent_indices,
      voxel_size
    );

    s_points.insert(s_points.end(), cur_s_points.begin(), cur_s_points.end());
    for (auto parent_index : cur_parent_indices) {
      parent_indices.push_back(static_cast<long>(output_offset) + parent_index);
    }
    s_lengths.push_back(cur_s_points.size());

    start_index += lengths[b];
    output_offset += cur_s_points.size();
  }

  return;
}
