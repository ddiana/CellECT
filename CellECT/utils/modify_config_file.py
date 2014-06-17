

if __name__ == "__main__":

	time_points = range(192)

	#prefix = "/home/diana/WORK/REPOSITORY/CellECT/old_ascidian_workspace/config_files/"
	prefix = "/home/diana/WORK/DATASETS/ascidian/spim/temp_config_files_from_lab/new/"

	for t in time_points:

		with open("%s/timestamp_%d.cnf" % (prefix, t),"a") as cnf_file:

			cnf_file.write("ap_axis_file = init_watershed_all_time_stamps/time_stamp_%d_APaxis.csv" % t)


			

