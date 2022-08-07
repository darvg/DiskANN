#include <random>
#include "index.h"

size_t random(size_t range_from, size_t range_to) {
  std::random_device                    rand_dev;
  std::mt19937                          generator(rand_dev());
  std::uniform_int_distribution<size_t> distr(range_from, range_to);
  return distr(generator);
}

int main (int argc, char *argv[]) {
	// TODO: 1. setup command line arguments (copy from old)
	// TODO: 2. parse label file
	// TODO: 3. for every label, collect points into map
		// TODO: 3a. for every label, build unfiltered index
	// TODO: 4. load the indices into memory and combine
	// TODO: 5. adjust filesize
	// TODO: 6. prune combined graph, then save
}

