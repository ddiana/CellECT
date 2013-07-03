import sys


## to run, move to main directory (where interactive_seg.py is)

def prepare_config_file(timestamp):


	output_file_name = "timestamp_"+ str(timestamp) + ".cnf"

	f = open(output_file_name, "w")

	f.write("volume_mat_path = init_watershed_all_time_stamps/watershed_input_timestamp_" + str(timestamp) + ".mat\n")
	f.write("volume_mat_var = vol\n")
	f.write("first_seg_mat_path =  init_watershed_all_time_stamps/init_watershed_output_timestamp_" + str(timestamp) + ".mat\n")
	f.write("first_seg_mat_var = ws\n")
	f.write("nuclei_mat_path =  init_watershed_all_time_stamps/watershed_input_timestamp_" + str(timestamp) +".mat\n")
	f.write("nuclei_mat_var = seeds\n")
	f.write("training_vol_mat_path =  training_data/positive_example.mat\n")
	f.write("training_vol_mat_var = vol\n")
	f.write("training_vol_nuclei_mat_path = training_data/positive_example.mat\n")
	f.write("training_vol_nuclei_mat_var = seeds\n")
	f.write("training_positive_seg_mat_path = training_data/positive_example.mat\n")
	f.write("training_positive_seg_mat_var = label_map\n")
	f.write("training_positive_labels_mat_path = training_data/positive_example.mat\n")
	f.write("training_positive_labels_mat_var = labels\n")
	f.write("training_negative_seg_mat_path = training_data/negative_example.mat\n")
	f.write("training_negative_seg_mat_var = L\n")
	f.write("training_negative_labels_mat_path = training_data/negative_example.mat\n")
	f.write("training_negative_labels_mat_var = labels\n")
	f.write("save_location_prefix = segs_all_time_stamps/timestamp_" + str(timestamp) + "_\n")
	f.write("has_bg = 1\n")
	f.write("use_size = 1\n")
	f.write("use_border_intensity = 1\n")
	f.write("use_border_distance = 0\n")

	

if __name__ == "__main__":

	if len(sys.argv) != 2:
		print "Usage: python prepare_config_file_for_timestamp.py TIMESTAMP"
		print "Make sure you adjust values accordingly in the source file."
		exit()
	else:
		ts = sys.argv[1]
		prepare_config_file(ts)
