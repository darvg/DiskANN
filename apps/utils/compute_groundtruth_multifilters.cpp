// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstdint>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <filesystem>
#include <random>
#include <limits>
#include <cstring>
#include <queue>
#include <omp.h>
#include <mkl.h>
#include <boost/program_options.hpp>
#include <unordered_map>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#ifdef _WINDOWS
#include <malloc.h>
#else
#include <stdlib.h>
#endif
#include "filter_utils.h"
#include "utils.h"

#define PARTSIZE 10000000
#define FPARTSIZE 100000000000
#define ALIGNMENT 512

namespace po = boost::program_options;

using pairIF = std::pair<size_t, float>;
struct cmpmaxstruct
{
    bool operator()(const pairIF &l, const pairIF &r)
    {
        return l.second < r.second;
    };
};
using maxPQIFCS = std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;

template <class T> T *aligned_malloc(const size_t n, const size_t alignment)
{
#ifdef _WINDOWS
    return (T *)_aligned_malloc(sizeof(T) * n, alignment);
#else
    return static_cast<T *>(aligned_alloc(alignment, sizeof(T) * n));
#endif
}

void save_groundtruth_as_one_file(const std::string filename, uint32_t *result_ids, float *result_distances,
                                  size_t num_pts, size_t dimension)
{
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    int int_num_pts = (int)num_pts, int_dimension = (int)dimension;
    writer.write((char *)&int_num_pts, sizeof(int));
    writer.write((char *)&int_dimension, sizeof(int));
    std::cout << "Saving truthset in one file (npts, dim, npts*dim id-matrix, "
                 "npts*dim dist-matrix) with npts = "
              << num_pts << ", dim = " << dimension
              << ", size = " << 2 * num_pts * dimension * sizeof(uint32_t) + 2 * sizeof(int) << "B" << std::endl;

    writer.write((char *)result_ids, num_pts * dimension * sizeof(uint32_t));
    writer.write((char *)result_distances, num_pts * dimension * sizeof(float));
    writer.close();
    std::cout << "Finished writing truthset" << std::endl;
}

template <class T> T div_round_up(const T numerator, const T denominator)
{
    return (numerator % denominator == 0) ? (numerator / denominator) : 1 + (numerator / denominator);
}

inline bool custom_dist(const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b)
{
    return a.second < b.second;
}

void compute_l2sq(float *const points_l2sq, const float *const matrix, const int64_t num_points, const uint64_t dim)
{
    assert(points_l2sq != NULL);
#pragma omp parallel for schedule(static, 65536)
    for (int64_t d = 0; d < num_points; ++d)
        points_l2sq[d] = cblas_sdot((int64_t)dim, matrix + (ptrdiff_t)d * (ptrdiff_t)dim, 1,
                                    matrix + (ptrdiff_t)d * (ptrdiff_t)dim, 1);
}

void distsq_to_points(const size_t dim,
                      float *dist_matrix, // Col Major, cols are queries, rows are points
                      size_t npoints, const float *const points,
                      const float *const points_l2sq, // points in Col major
                      size_t nqueries, const float *const queries,
                      const float *const queries_l2sq, // queries in Col major
                      float *ones_vec = NULL)          // Scratchspace of num_data size and init to 1.0
{
    bool ones_vec_alloc = false;
    if (ones_vec == NULL)
    {
        ones_vec = new float[nqueries > npoints ? nqueries : npoints];
        std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float)1.0);
        ones_vec_alloc = true;
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float)-2.0, points, dim, queries, dim,
                (float)0.0, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, points_l2sq, npoints,
                ones_vec, nqueries, (float)1.0, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, ones_vec, npoints,
                queries_l2sq, nqueries, (float)1.0, dist_matrix, npoints);
    if (ones_vec_alloc)
        delete[] ones_vec;
}

template <typename T> int get_num_parts(uint32_t npts, uint32_t ndims)
{
    int npts_i32 = (int)npts;
    int ndims_i32 = (int)ndims;
    int num_parts = (npts_i32 % PARTSIZE) == 0 ? npts_i32 / PARTSIZE : (uint32_t)std::floor(npts_i32 / PARTSIZE) + 1;
    return num_parts;
}

size_t get_num_fparts(size_t nqueries)
{
    // each part should have FPARTSIZE BYTES
    return FPARTSIZE / (nqueries * sizeof(uint32_t));
}

char *get_relevant_vectors(char *base_data, std::vector<uint32_t> relevant_ids, const uint32_t VECTOR_SIZE)
{
    size_t num_pts = relevant_ids.size();

    char *relevant_vectors = (char *)malloc(VECTOR_SIZE * num_pts);
    for (size_t i = 0; i < num_pts; i++)
    {
        uint32_t id = relevant_ids[i];
        char *curr_point = base_data + (VECTOR_SIZE * id);
        memcpy(relevant_vectors + (i * VECTOR_SIZE), curr_point, VECTOR_SIZE);
    }
    return relevant_vectors;
}

inline void unmap_bin(char *data, size_t fsize)
{
    if (munmap(data, fsize) == -1)
    {
        throw;
    }
}

inline std::tuple<char *, size_t, size_t, size_t> load_bin_as_mmap(const char *fname)
{
    int fd = open(fname, O_RDONLY);
    if (fd <= 0)
        throw;

    struct stat sb;
    if (fstat(fd, &sb) != 0)
        throw;

    size_t fsize = sb.st_size;
    diskann::cout << "File Size: " << fsize << std::endl;
    char *buf = (char *)mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);

    uint32_t number_of_points, dimension;
    std::memcpy(&number_of_points, buf, sizeof(uint32_t));
    std::memcpy(&dimension, buf + sizeof(uint32_t), sizeof(uint32_t));
    return std::make_tuple(buf, fsize, (size_t)number_of_points, (size_t)dimension);
}

template <typename T>
inline std::tuple<float *, size_t, size_t> load_bin_as_float(const char *filename, uint32_t part_num)
{
    size_t num_pts, dimension;
    float *data;
    int num_pts_i32, dimension_i32;

    std::ifstream reader;
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    reader.open(filename, std::ios::binary);
    std::cout << "Reading bin file " << filename << " ...\n";

    reader.read((char *)&num_pts_i32, sizeof(int));
    reader.read((char *)&dimension_i32, sizeof(int));

    uint64_t start_id = part_num * PARTSIZE;
    uint64_t end_id = (std::min)(start_id + PARTSIZE, (uint64_t)num_pts_i32);
    num_pts = end_id - start_id;
    dimension = (size_t)dimension_i32;
    std::cout << "#pts in part = " << num_pts << ", #dims = " << dimension
              << ", size = " << num_pts * dimension * sizeof(T) << "B" << std::endl;

    reader.seekg(start_id * dimension * sizeof(T) + 2 * sizeof(uint32_t), std::ios::beg);
    T *data_T = new T[num_pts * dimension];
    reader.read((char *)data_T, sizeof(T) * num_pts * dimension);
    std::cout << "Finished reading part of the bin file." << std::endl;
    reader.close();
    data = aligned_malloc<float>(num_pts * dimension, ALIGNMENT);

#pragma omp parallel for schedule(dynamic, 32768)
    for (int64_t i = 0; i < (int64_t)num_pts; i++)
    {
        for (int64_t j = 0; j < (int64_t)dimension; j++)
        {
            float cur_val_float = (float)data_T[i * dimension + j];
            std::memcpy((char *)(data + i * dimension + j), (char *)&cur_val_float, sizeof(float));
        }
    }
    delete[] data_T;
    std::cout << "Finished converting part data to float." << std::endl;

    return std::make_tuple(data, num_pts, dimension);
}

template <typename T>
inline std::tuple<float *, size_t, size_t> data_to_float(T *orig_data, uint32_t npts, uint32_t ndims, uint32_t part_num)
{
    size_t num_pts, dimension;
    float *data;

    uint64_t start_id = part_num * PARTSIZE;
    uint64_t end_id = (std::min)(start_id + PARTSIZE, (uint64_t)npts);
    num_pts = end_id - start_id;
    dimension = (size_t)ndims;

    T *data_T = &orig_data[start_id];
    data = aligned_malloc<float>(num_pts * dimension, ALIGNMENT);

#pragma omp parallel for schedule(dynamic, 32768)
    for (int64_t i = 0; i < (int64_t)num_pts; i++)
    {
        for (int64_t j = 0; j < (int64_t)dimension; j++)
        {
            float cur_val_float = (float)data_T[i * dimension + j];
            std::memcpy((char *)(data + i * dimension + j), (char *)&cur_val_float, sizeof(float));
        }
    }

    return std::make_tuple(data, num_pts, dimension);
}

void exact_knn(const size_t dimension, const size_t k, uint32_t *const closest_points, float *const closest_distances,
               size_t num_pts, float *pts_input, size_t num_queries, float *queries_input,
               diskann::Metric metric = diskann::Metric::L2)
{
    float *pts_l2sq = new float[num_pts];
    float *queries_l2sq = new float[num_queries];
    compute_l2sq(pts_l2sq, pts_input, num_pts, dimension);
    compute_l2sq(queries_l2sq, queries_input, num_queries, dimension);
    float *pts = pts_input;
    float *queries = queries_input;

    size_t q_batch_size = (1 << 9);
    float *dist_matrix = new float[(size_t)q_batch_size * (size_t)num_pts];

    for (uint64_t b = 0; b < div_round_up(num_queries, q_batch_size); ++b)
    {
        int64_t q_b = b * q_batch_size;
        int64_t q_e = ((b + 1) * q_batch_size > num_queries) ? num_queries : (b + 1) * q_batch_size;

        distsq_to_points(dimension, dist_matrix, num_pts, pts, pts_l2sq, q_e - q_b,
                         queries + (ptrdiff_t)q_b * (ptrdiff_t)dimension, queries_l2sq + q_b);

#pragma omp parallel for schedule(dynamic, 16)
        for (long long q = q_b; q < q_e; q++)
        {
            maxPQIFCS point_dist;
            for (size_t p = 0; p < k; p++)
                point_dist.emplace(p, dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)num_pts]);
            for (size_t p = k; p < num_pts; p++)
            {
                if (point_dist.top().second > dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)num_pts])
                    point_dist.emplace(p, dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)num_pts]);
                if (point_dist.size() > k)
                    point_dist.pop();
            }
            for (ptrdiff_t l = 0; l < (ptrdiff_t)k; ++l)
            {
                closest_points[(ptrdiff_t)(k - 1 - l) + (ptrdiff_t)q * (ptrdiff_t)k] = point_dist.top().first;
                closest_distances[(ptrdiff_t)(k - 1 - l) + (ptrdiff_t)q * (ptrdiff_t)k] = point_dist.top().second;
                point_dist.pop();
            }
            assert(std::is_sorted(closest_distances + (ptrdiff_t)q * (ptrdiff_t)k,
                                  closest_distances + (ptrdiff_t)(q + 1) * (ptrdiff_t)k));
        }
    }

    delete[] dist_matrix;
    delete[] pts_l2sq;
    delete[] queries_l2sq;
}

template <typename T>
std::vector<std::pair<uint32_t, float>> process_query(char *orig_data, uint32_t num_pts, uint32_t num_dims, size_t k,
                                                      float *query_vector)
{
    size_t num_parts = get_num_parts<T>(num_pts, num_dims);
    std::vector<std::pair<uint32_t, float>> results_id_dist_map;
    for (size_t part_num = 0; part_num < num_parts; part_num++)
    {
        size_t start_id = part_num * PARTSIZE;
        auto [base_data, num_base_pts, dimension] = data_to_float<T>((T *)orig_data, num_pts, num_dims, part_num);

        uint32_t *closest_points_part = new uint32_t[k];
        float *dist_closest_points_part = new float[k];

        auto part_k = k < num_base_pts ? k : num_base_pts;
        exact_knn(dimension, part_k, closest_points_part, dist_closest_points_part, num_base_pts, base_data, 1,
                  query_vector);

        for (uint64_t j = 0; j < part_k; j++)
        {
            // NOTE: indexing with just j instead of curr_query * part_k + j could lead to bugs
            results_id_dist_map.push_back(
                std::make_pair((uint32_t)(closest_points_part[j] + start_id), dist_closest_points_part[j]));
        }

        delete[] closest_points_part;
        delete[] dist_closest_points_part;
        diskann::aligned_free(base_data);
    }

    return results_id_dist_map;
}

template <typename T>
int aux_main(const path &base_file, const path &gt_file, const path &query_file, size_t k,
             std::vector<std::vector<uint32_t>> query_to_base_pts)
{
    auto [query_data, num_queries, dimension] = load_bin_as_float<T>(query_file.c_str(), 0);
    [[maybe_unused]] auto [base_data, fsize, num_base_pts, dim_unused] = load_bin_as_mmap(base_file.c_str());
    const uint32_t VECTOR_SIZE = dimension * sizeof(T);
    char *base_data_no_metadata = base_data + (2 * sizeof(uint32_t));
    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(2);

    std::vector<std::vector<std::pair<uint32_t, float>>> results_per_query(num_queries);
    for (size_t query_id = 0; query_id < 10; query_id++)
    {
        float *curr_query_vector = query_data + (dimension * query_id);
        char *curr_query_base_data =
            get_relevant_vectors(base_data_no_metadata, query_to_base_pts[query_id], VECTOR_SIZE);
        results_per_query[query_id] = process_query<T>(
            curr_query_base_data, (uint32_t)query_to_base_pts[query_id].size(), dimension, k, curr_query_vector);
        free(curr_query_base_data);
        float pct = (float)query_id / num_queries;
        std::cout << pct << "\r" << std::flush;
    }

    uint32_t *closest_points = new uint32_t[num_queries * k];
    float *closest_distances = new float[num_queries * k];
    for (size_t query_id = 0; query_id < 10; query_id++)
    {
        std::vector<std::pair<uint32_t, float>> &curr_query_results = results_per_query[query_id];
        std::sort(curr_query_results.begin(), curr_query_results.end(), custom_dist);

        size_t i = 0;
        std::cout << "query " << query_id << " has " << query_to_base_pts[query_id].size() << "points" << std::endl;
        for (auto result : curr_query_results)
        {
            if (i == k)
                break;

            uint32_t new_id = result.first;
            uint32_t true_id = query_to_base_pts[query_id][new_id];
            float distance = result.second;
            std::cout << new_id << " " << distance << " " << true_id << std::endl;
            uint32_t curr_index = query_id * k + i;
            closest_points[curr_index] = true_id;
            closest_distances[curr_index] = distance;

            i++;
        }
    }

    save_groundtruth_as_one_file(gt_file, closest_points, closest_distances, 10, k);
    delete[] closest_points;
    delete[] closest_distances;
    unmap_bin(base_data, fsize);
    diskann::aligned_free(query_data);

    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, universal_label;
    path base_file, query_file, gt_file, base_label_file, query_label_file;
    uint64_t K;

    try
    {
        po::options_description desc{"Arguments"};

        desc.add_options()("help,h", "Print information on arguments");

        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(), "distance function <l2/mips>");
        desc.add_options()("base_file", po::value<path>(&base_file)->required(),
                           "File containing the base vectors in binary format");
        desc.add_options()("base_label_file", po::value<path>(&base_label_file)->default_value(""),
                           "Input labels file in txt format if present");
        desc.add_options()("query_file", po::value<path>(&query_file)->required(),
                           "File containing the query vectors in binary format");
        desc.add_options()("query_label_file", po::value<path>(&query_label_file)->default_value(std::string("")),
                           "Filter file for Queries for Filtered Search ");
        desc.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                           "Universal label, if using it, only in conjunction with label_file");
        desc.add_options()("gt_file", po::value<path>(&gt_file)->required(),
                           "File name for the writing ground truth in binary "
                           "format, please don' append .bin at end if "
                           "no filter_label or filter_label_file is provided it "
                           "will save the file with '.bin' at end."
                           "else it will save the file as filename_label.bin");
        desc.add_options()("K", po::value<uint64_t>(&K)->required(),
                           "Number of ground truth nearest neighbors to compute");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    if (data_type != std::string("float") && data_type != std::string("int8") && data_type != std::string("uint8"))
        throw diskann::ANNException(
            data_type + std::string(" is unsupported. Only float, int8, and uint8 are supported."), -1);

    // 1. parse query and base labels file
    std::vector<std::vector<label_set>> query_pts_to_labels;

    query_pts_to_labels = diskann::parse_query_label_file(query_label_file);
    auto [base_pts_to_labels, labels_to_num_base_pts, all_labels] =
        diskann::parse_label_file(base_label_file, universal_label);

    // 2. for each set of queries, split the base points into different sets
    std::vector<std::vector<uint32_t>> queries_to_base_pts =
        diskann::compute_base_points_per_query(query_pts_to_labels, base_pts_to_labels, universal_label);

    // 3. compute groundtruths per query, and save to a single file
    if (data_type == std::string("float"))
        aux_main<float>(base_file, gt_file, query_file, K, queries_to_base_pts);
    else if (data_type == std::string("int8"))
        aux_main<int8_t>(base_file, gt_file, query_file, K, queries_to_base_pts);
    else if (data_type == std::string("uint8"))
        aux_main<uint8_t>(base_file, gt_file, query_file, K, queries_to_base_pts);
    else
        throw;
}
