from libtiff import TIFF
import pdb


def downsample_tif(filename_in, filename_out):

	input_tiff = TIFF.open(filename_in, mode = "r")
	output_tiff = TIFF.open(filename_out, mode = "w")

	counter = 0
	for img in input_tiff.iter_images():
		img = img[::2, ::2]
		output_tiff.write_image(img)
		counter += 1

		if counter /100 == counter / 100.:

			print counter
		



if __name__ == "__main__":

	filename_in = "twenty_10.tif"
	filename_out = "twenty_10_half.tif"
	downsample_tif(filename_in, filename_out )
