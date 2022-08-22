#include <index.h>
#include <random>
#include <omp.h>
#include <string.h>
#include <boost/program_options.hpp>

#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"
#include "ann_exception.h"


namespace po = boost::program_options;

size_t random(size_t range_from, size_t range_to) {
  std::random_device                    rand_dev;
  std::mt19937                          generator(rand_dev());
  std::uniform_int_distribution<size_t> distr(range_from, range_to);
  return distr(generator);
}


std::vector<tsl::robin_set<std::string>> parse_label_file(const std::string &map_file, tsl::robin_set<std::string>& labels, 
										std::map<std::string,_u64> &labels_to_num_points, bool use_universal_label, std::string universal_label) {
	// TODO: Add description of function
	std::ifstream infile(map_file);
	std::string   line, token;
	unsigned      line_cnt = 0;

	while (std::getline(infile, line)) {
		line_cnt++;
	}
	std::vector<tsl::robin_set<std::string>> points_to_labels;
	points_to_labels.resize(line_cnt, tsl::robin_set<std::string>());

	// std::map<std::string,tsl::robin_set<uint32_t>> labels_to_points;


	infile.clear();
	infile.seekg(0, std::ios::beg);
	while (std::getline(infile, line)) {
		std::istringstream       iss(line);
		tsl::robin_set<std::string> lbls;
		// long int              val;
		getline(iss, token, '\t');
		_u64 i = (_u64) std::stoul(token);
		getline(iss, token, '\t');
		std::istringstream new_iss(token);
		while (getline(new_iss, token, ',')) {
			token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
			token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
			lbls.insert(token);
			labels.insert(token);
			// labels_to_points[token].insert(i);
			labels_to_num_points[token] += 1;
		}
		if (lbls.size() <= 0) {
			std::cout << "No label found";
			exit(-1);
		}
		//std::sort(lbls.begin(), lbls.end());
		points_to_labels[i] = lbls;
		line_cnt++;
	}
	if ((use_universal_label) && (labels_to_num_points.find(universal_label) != labels_to_num_points.end())){
		_u64 universal_label_count = labels_to_num_points[universal_label];
		for (auto &label : labels){
			labels_to_num_points[label] += universal_label_count;
		}
		labels_to_num_points.erase(universal_label);
	}
	std::cout << "Identified " << labels.size() << " distinct label(s)"
						<< std::endl;
	// return labels_to_points;
	return points_to_labels;
}


template <typename T>
void convert_base_ids_to_label_vecs(
		const std::string data_type, const std::string &data_path, 
		const tsl::robin_set<std::string>& labels, bool use_universal_label,
		std::string universal_label, std::vector<tsl::robin_set<std::string>> points_to_labels,
		tsl::robin_map<std::string, tsl::robin_map<_u64,_u64>> &data_id_to_label_id,
		tsl::robin_map<std::string, tsl::robin_map<_u64,_u64>> &label_id_to_data_id,
		std::map<std::string,_u64> labels_to_num_points,
		_u64 &number_of_points, _u64 &dimension) {
	// TODO: Add description of function

  	tsl::robin_map<std::string, tsl::robin_map<_u64, _u64>> rev_map;
	std::cout << "Loading base file " << data_path << "..." << std::endl;

	std::ifstream data_path_stream, label_file_stream;
	data_path_stream.exceptions(std::ios::badbit | std::ios::failbit);
  	data_path_stream.open(data_path, std::ios::binary);

	data_path_stream.read((char *) &number_of_points, sizeof(number_of_points));
	data_path_stream.read((char *) &dimension, sizeof(dimension));

	for (const auto &label : labels){
		if ((use_universal_label) && (label == universal_label)) continue;
		std::string label_file = data_path + label;
		std::ofstream label_file_stream;
		label_file_stream.exceptions(std::ios::badbit | std::ios::failbit);
		label_file_stream.open(label_file, std::ios::binary);
		unsigned number_of_label_points = labels_to_num_points[label];
		label_file_stream.write((char *) &number_of_label_points, sizeof(number_of_label_points));
		label_file_stream.write((char *) &dimension, sizeof(dimension));
		label_file_stream.close();
	}

	tsl::robin_map<std::string,_u64> current_label_positions;
	for (auto& label : labels){
		current_label_positions[label] = 0;
	}

	for (unsigned current_point_id = 0; current_point_id < number_of_points; current_point_id++){
		tsl::robin_set<std::string> current_point_labels = points_to_labels[current_point_id];
		std::vector<T> current_point_vector(dimension);
		data_path_stream.read((char *) current_point_vector.data(), sizeof(T)*dimension);
		
		if ((current_point_labels.size() == 1) && use_universal_label && (current_point_labels.count(universal_label))){
			for (const auto &label : labels){
				std::ofstream label_file_stream;
				label_file_stream.exceptions(std::ios::badbit | std::ios::failbit);
				std::string label_file = data_path + label;
				label_file_stream.open(label_file, std::ios::binary);
				data_id_to_label_id[label][current_point_id] = current_label_positions[label];
				label_id_to_data_id[label][current_label_positions[label]] = current_point_id;
				current_label_positions[label]++;
				label_file_stream.write((char *) current_point_vector.data(), sizeof(T)*dimension);
				label_file_stream.close();
			}
		}
		for (const auto &label : current_point_labels){
			std::ofstream label_file_stream;
			label_file_stream.exceptions(std::ios::badbit | std::ios::failbit);
			std::string label_file = data_path + label;
			label_file_stream.open(label_file, std::ios::binary);
			data_id_to_label_id[label][current_point_id] = current_label_positions[label];
			current_label_positions[label]++;
			label_file_stream.write((char *) current_point_vector.data(), sizeof(T)*dimension);
			label_file_stream.close();
		}
		// NOTE: Should we close file streams only when we have say 200 files open?
	}
}


template<typename T>
int label_build_in_memory_index(const std::string& data_path, const unsigned R,
                           const unsigned L,
                           const float alpha, const std::string& save_path,
                           const unsigned num_threads, const tsl::robin_set<std::string>& labels, const bool use_universal_label, const std::string& universal_label) {
  diskann::Parameters paras;
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>(
      "C", 750);  // maximum candidate set size during pruning procedure
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", 0);
  paras.Set<unsigned>("num_threads", num_threads);

  for (const auto &label : labels){
	if ((use_universal_label) && (label == universal_label)) continue;

	std::string label_data_path = data_path + label;
	_u64 num_label_points, dimension;
	diskann::get_bin_metadata(label_data_path,num_label_points,dimension);
	diskann::Index<T> index(diskann::L2, dimension, num_label_points, false, false);
	auto              s = std::chrono::high_resolution_clock::now();

	index.build(label_data_path.c_str(), num_label_points, paras);
	std::chrono::duration<double> diff =
		std::chrono::high_resolution_clock::now() - s;

	std::cout << "Indexing time: " << diff.count() << "\n";
	std::string label_save_path = save_path + label;
	index.save(label_save_path.c_str());
  }

  return 0;
}

std::vector<std::vector<unsigned>> load_in_memory_index(const char* filename,
                                                        _u64&       file_size,
                                                        _u64&       ep) {
  std::ifstream in(filename, std::ios::binary);
  size_t        expected_file_size, width;
  in.read((char*) &expected_file_size, sizeof(_u64));
  file_size += expected_file_size;
  in.read((char*) &width, sizeof(unsigned));
  in.read((char*) &ep, sizeof(unsigned));
  std::cout << "Loading vamana index " << filename << "..." << std::flush;
  std::vector<std::vector<unsigned>> filt_graph;

  size_t   cc = 0;
  unsigned nodes = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char*) &k, sizeof(unsigned));
    if (in.eof())
      break;
    cc += k;
    ++nodes;
    std::vector<unsigned> tmp(k);
    in.read((char*) tmp.data(), k * sizeof(unsigned));

    filt_graph.emplace_back(tmp);
    if (nodes % 10000000 == 0)
      std::cout << "." << std::flush;
  }
  /*if (_final_graph.size() != _nd) {
    std::cout << "ERROR. mismatch in number of points. Graph has "
              << _final_graph.size() << " points and loaded dataset has " << _nd
              << " points. " << std::endl;
    return;
  }*/

  std::cout << "..done. Index has " << nodes << " nodes and " << cc
            << " out-edges" << std::endl;
  return (filt_graph);
}


template<typename T>
std::vector<std::vector<_u64>> stitch_label_index(_u64 number_of_points, _u64 dimension, tsl::robin_set<std::string>& labels, 
														const bool use_universal_label, const std::string universal_label, 
														tsl::robin_map<std::string, tsl::robin_map<_u64, _u64>> label_id_to_data_id,
														const std::string data_type, const std::string& data_path, const std::string& save_path){
	std::vector<tsl::robin_set<unsigned>> full_graph(number_of_points);
	tsl::robin_map<std::string, _u64> entry_points;
	_u64 file_size = 0;
	for (auto& label : labels){
		std::string label_save_path = label + save_path;
		_u64 entry_point = 0;
		std::vector<std::vector<unsigned>> filt_graph =
        load_in_memory_index(label_save_path.c_str(), file_size, entry_point);
    	entry_point = random(0, filt_graph.size());
    	entry_points[label] = label_id_to_data_id[label][entry_point];
		for (_u64 i = 0; i < filt_graph.size(); i++){
			_u64 point_id = label_id_to_data_id[label][i];
			for (_u64 j = 0; j < filt_graph[i].size(); j++){
				_u64 neighbor_id = label_id_to_data_id[label][filt_graph[i][j]];
				if (full_graph[point_id].find(neighbor_id) != full_graph[point_id].end()){
					file_size -= 1;
				}else{
					full_graph[point_id].insert(neighbor_id);
				}
			}
		}
	}
	std::vector<std::vector<_u64>> full_graph_sorted(number_of_points);
	for (_u64 i = 0; i < number_of_points; i++){
		full_graph_sorted[i].insert(full_graph_sorted[i].end(), full_graph[i].begin(), full_graph[i].end());
		std::sort(full_graph_sorted[i].begin(),full_graph_sorted[i].end());
	}
	return full_graph_sorted;
}

template<typename T>
void prune_and_save(std::string data_path, std::string stitched_index_path,
                    std::string pruned_index_path, _u32 R,
                    std::map<std::string, size_t> labels_dist,
                    std::mt19937                  labels_rng_gen) {
  diskann::Index<T> index(diskann::L2, data_path.c_str());
  index.load(stitched_index_path.c_str());  // to load NSG

  diskann::Parameters paras;
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>(
      "C", 750);  // maximum candidate set size during pruning procedure
  paras.Set<float>("alpha", 1.2);
  paras.Set<bool>("saturate_graph", 1);
  paras.Set<std::map<std::string, size_t>>("labels_counts", labels_dist);

  index.prune_all_nodes(paras);

  index.save(pruned_index_path.c_str());
}

int main (int argc, char *argv[]) {

	// 1. setup command line arguments
  std::string data_type, dist_fn, data_path, index_path_prefix, label_file, universal_label;
  unsigned num_threads, R, L, stitched_R;
  float alpha;
  bool use_universal_label = universal_label.size();

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips>");
    desc.add_options()("data_path",
                       po::value<std::string>(&data_path)->required(),
                       "Input data file in bin format");
    desc.add_options()("index_path_prefix",
                       po::value<std::string>(&index_path_prefix)->required(),
                       "Path prefix for saving index file components");
    desc.add_options()("max_degree,R",
                       po::value<uint32_t>(&R)->default_value(64),
                       "Maximum graph degree");
    desc.add_options()(
        "Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
        "Build complexity, higher value results in better graphs");
    desc.add_options()(
        "alpha", po::value<float>(&alpha)->default_value(1.2f),
        "alpha controls density and diameter of graph, set 1 for sparse graph, "
        "1.2 or 1.4 for denser graphs with lower diameter");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()("label_file",
                       po::value<std::string>(&label_file)->required(),
                       "Input label file in txt format if present");
    desc.add_options()(
        "stitched_R",
        po::value<uint32_t>(&stitched_R)->required(),
        "Universal label, if using it, only in conjunction with labels_file");
    desc.add_options()(
        "universal_label",
        po::value<std::string>(&universal_label)->default_value(""),
        "Universal label, if using it, only in conjunction with labels_file");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }


	// 2. parse label file
	tsl::robin_set<std::string> labels;
	std::map<std::string,_u64> labels_to_num_points;
	std::vector<tsl::robin_set<std::string>> points_to_labels = parse_label_file(label_file,labels,labels_to_num_points,use_universal_label,universal_label);
	_u64 number_of_points = 0;
	_u64 dimension = 0;
  
	// TODO: 3. for every label, collect points into map
	tsl::robin_map<std::string, tsl::robin_map<_u64,_u64>> data_id_to_label_id;
	tsl::robin_map<std::string, tsl::robin_map<_u64,_u64>> label_id_to_data_id;

	if (data_type == "uint8")
		convert_base_ids_to_label_vecs<uint8_t>(
		data_type, data_path, labels, use_universal_label, universal_label, points_to_labels,
		data_id_to_label_id, label_id_to_data_id, labels_to_num_points, number_of_points, dimension);
	else if (data_type == "int8")
		convert_base_ids_to_label_vecs<int8_t>(
		data_type, data_path, labels, use_universal_label, universal_label, points_to_labels,
		data_id_to_label_id, label_id_to_data_id, labels_to_num_points, number_of_points, dimension);
	else if (data_type == "float")
		convert_base_ids_to_label_vecs<float>(
		data_type, data_path, labels, use_universal_label, universal_label, points_to_labels,
		data_id_to_label_id, label_id_to_data_id, labels_to_num_points, number_of_points, dimension);
	else
		std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;

	if (data_type == "uint8")
		label_build_in_memory_index<uint8_t>(data_path, R, L, alpha, 
									index_path_prefix,num_threads, 
									labels, use_universal_label, universal_label);
	else if (data_type == "int8")
		label_build_in_memory_index<int8_t>(data_path, R, L, alpha, 
									index_path_prefix,num_threads, 
									labels, use_universal_label, universal_label);
	else if (data_type == "float")
		label_build_in_memory_index<float>(data_path, R, L, alpha, 
									index_path_prefix,num_threads, 
									labels, use_universal_label, universal_label);
	else
		std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;


	// TODO: 3a. for every label, build unfiltered index

	// TODO: 4. load the indices into memory and combine

	std::vector<std::vector<_u64>> stitched_index; 

	if (data_type == "uint8")
		stitched_index = stitch_label_index<uint8_t>(number_of_points, dimension, labels, 
						use_universal_label, universal_label, 
						label_id_to_data_id, data_type, data_path, index_path_prefix);

	else if (data_type == "int8")
		stitched_index = stitch_label_index<int8_t>(number_of_points, dimension, labels, 
						use_universal_label, universal_label, 
						label_id_to_data_id, data_type, data_path, index_path_prefix);

	else if (data_type == "float")
		stitched_index = stitch_label_index<float>(number_of_points, dimension, labels, 
						use_universal_label, universal_label, 
						label_id_to_data_id, data_type, data_path, index_path_prefix);


	// // TODO: 6. prune combined graph, then save
	// if (data_type == "uint8")
	// 	prune_and_save<uint8_t>(data_path, stitched_index_path, pruned_index_path, 
	// 							R, labels_dist, labels_rng_gen);

	// else if (data_type == "int8")
	// 	prune_and_save<int8_t>(data_path, stitched_index_path, pruned_index_path, 
	// 							R, labels_dist, labels_rng_gen);

	// else if (data_type == "float")
	// 	prune_and_save<float>(data_path, stitched_index_path, pruned_index_path, 
	// 							R, labels_dist, labels_rng_gen);

}

