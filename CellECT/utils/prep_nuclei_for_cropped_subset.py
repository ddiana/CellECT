import pdb
import sys


if __name__ == "__main__":


	input_file = sys.argv[1]
	output_file = sys.argv[2]

	x_offset = 200
	y_offset = 200
	x_limit = 800
	y_limit = 800
	downsample_ratio = 2



	with open(input_file,"r") as f:
		with open(output_file,"w") as g:

			vals = f.readline().split(",")
			g.write("x, y, z, t, confidence\n")

			counter = 0

			while vals:

				if counter > 0:

					vals = f.readline().split(",")
					if len(vals) >1:
						vals = [float(val.strip()) for val in vals]


						if vals[4]>1:
	   
							if vals[0]> x_offset and vals[0] < x_limit and vals[1] > y_offset and vals[1] < y_limit:
	  
								vals [0] = (vals[0] - x_offset) / downsample_ratio
								vals [1] = (vals[1] - y_offset) / downsample_ratio
								vals [3] = 11
		
								g.write("%.2f, %.2f, %.2f, %.2f, %.2f\n" % (vals[0], vals[1], vals[2], vals[3], vals[4]) )
					else:
						break


				counter +=1 

