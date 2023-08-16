**Usage for filtered indices**
================================
Example commands for multi-filtered Vamana

```
./build_memory_index --data_type <> --dist_fn <> --data_path <> --index_path_prefix <> --label_file <> --universal_label <> --FilteredLbuild <> --filter_penalty_hp <>
```
```
./compute_groundtruth_multifilters --data_type <> --dist_fn <> --base_file <> --base_label_file <> --query_file <> --query_label_file <> --universal_label <> --gt_file <> --K <>
```
```
./search_memory_index --data_type <> --dist_fn <> --index_path_prefix <> --result_path <> --query_file <> --query_filters_file <> --gt_file <> -K <> -L <> --filter_penalty_hp <>
```

Almost everything is identical to what is in the filtered-vamana workflow, but there are a couple of new parameters here:
- For both `build_memory_index` and `search_memory_index`, there is a `filter_penalty_hp` parameter. This is a hyperparameter that controls the scaling of the penalty applied to distances for missing filters.
- Both `search_memory_index` and `compute_groundtruth_multifilters` require a `query_label_file`. This refers to the filters passed in for each query, with each line having the following AND of ORs format:
  ```
  a|b|c&d|e|f&g|h|i& ...
  ```
