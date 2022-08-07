#include <random>
#include "index.h"

size_t random(size_t range_from, size_t range_to) {
  std::random_device                    rand_dev;
  std::mt19937                          generator(rand_dev());
  std::uniform_int_distribution<size_t> distr(range_from, range_to);
  return distr(generator);
}



  tsl::robin_set<std::string> parse_label_file(const std::string &map_file) {
    // TODO: Add description of function
    std::ifstream infile(map_file);
    std::string   line, token;
    unsigned      line_cnt = 0;

    while (std::getline(infile, line)) {
      line_cnt++;
    }
    std::vector<std::vector<std::string>> pts_to_labels;
    tsl::robin_set<std::string> labels;
    pts_to_labels.resize(line_cnt, std::vector<std::string>());

    infile.clear();
    infile.seekg(0, std::ios::beg);
    while (std::getline(infile, line)) {
      std::istringstream       iss(line);
      std::vector<std::string> lbls(0);
      // long int              val;
      getline(iss, token, '\t');
      _u32 i = (_u32) std::stoul(token);
      getline(iss, token, '\t');
      std::istringstream new_iss(token);
      while (getline(new_iss, token, ',')) {
        token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
        token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
        lbls.push_back(token);
        labels.insert(token);
      }
      if (lbls.size() <= 0) {
        std::cout << "No label found";
        exit(-1);
      }
      std::sort(lbls.begin(), lbls.end());
      pts_to_labels[i] = lbls;
      line_cnt++;
    }
    std::cout << "Identified " << labels.size() << " distinct label(s)"
              << std::endl;
    return labels;
  }


int main (int argc, char *argv[]) {

	// 1. setup command line arguments
  std::string data_type, dist_fn, data_path, index_path_prefix, label_file, universal_label;
  unsigned num_threads, R, L, stitched_R;
  float alpha;

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
  tsl::robin_set<std::string> labels = parse_label_file(label_file);
  
	// TODO: 3. for every label, collect points into map
		// TODO: 3a. for every label, build unfiltered index
	// TODO: 4. load the indices into memory and combine
	// TODO: 5. adjust filesize
	// TODO: 6. prune combined graph, then save
}

