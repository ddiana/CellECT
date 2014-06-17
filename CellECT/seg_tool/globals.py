# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara


########## globals:

DEFAULT_PARAMETER = {}

_mean_per_dim = []

_std_per_dim = []

should_load_last_save = False

task_index = 0

path_to_workspace = ""

generic_features = ["border_to_interior_intensity_ratio"]
specific_features =  ["inner_point", "size", "shape", "border_to_nucleus_distance_mean", "border_to_nucleus_distance_hist", "distance_from_margin"]

expected_workspace_directories = set(["config_files", "init_watershed_all_time_stamps", "input_slices", "segs_all_time_stamps", "tracker_config", "training_data"])

default_parameter_dictionary_keys_bq_only =  ("bq_token",\
   "bq_mex_url",\
   "bq_t")

default_parameter_dictionary_keys_cellness_metric_only = ( "training_vol_mat_path",\
   "training_vol_mat_var",\
   "training_vol_nuclei_mat_path", \
   "training_vol_nuclei_mat_var",\
   "training_positive_seg_mat_path",\
   "training_positive_seg_mat_var", \
   "training_positive_labels_mat_path",\
   "training_positive_labels_mat_var",\
   "training_negative_seg_mat_path",\
   "training_negative_seg_mat_var",\
   "training_negative_labels_mat_path",\
   "training_negative_labels_mat_var")

default_parameter_dictionary_keys = ("volume_mat_path",
   "volume_mat_var",\
   "volume_nuclei_mat_path",\
   "volume_nuclei_mat_var",\
   "first_seg_mat_path",\
   "first_seg_mat_var", \
   "nuclei_mat_path",\
   "nuclei_mat_var",\
   "bg_seeds_var",\
   "bg_seeds_path",\
   "save_location_prefix",\
   "has_bg", \
   "use_size", \
   "use_border_intensity", \
   "use_border_distance", \
   "use_dist_from_margin", \
   "x_res", \
   "y_res", \
   "z_res", \
   "ap_axis_file")

