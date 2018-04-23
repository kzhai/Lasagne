import os
import re

import numpy
import numpy.random

from PlotRetainRateHistogramVsEpoch import noise_file_name_pattern


def plot_retain_rates_for_image_features(model_directory, feature_map_size, layer_index=0,
                                         snapshot_interval=[-1, -1, 1],
                                         plot_directory=None,
                                         fps=10):
	# retain_rates_file_name_pattern = re.compile(r'layer\.(?P<layer>[\d]+?)\.epoch\.(?P<epoch>[\d]+?)\.npy')
	epoch_to_retain_rate_array = {}
	epoch_to_retain_rate_file = {}
	for file_name in os.listdir(model_directory):
		matcher = re.match(noise_file_name_pattern, file_name)
		if matcher is None:
			continue

		temp_layer_index = int(matcher.group("layer"))
		if temp_layer_index != layer_index:
			continue

		temp_epoch_index = int(matcher.group("epoch"))
		if snapshot_interval[0] >= 0 and temp_epoch_index < snapshot_interval[0]:
			continue
		if snapshot_interval[1] >= 0 and temp_epoch_index > snapshot_interval[1]:
			continue
		if temp_epoch_index % snapshot_interval[2] != 0:
			continue

		retain_rates = numpy.load(os.path.join(model_directory, file_name))
		# print numpy.mean(retain_rates[:512]), numpy.max(retain_rates[:512]), numpy.min(retain_rates[:512])
		# print numpy.mean(retain_rates[512:]), numpy.max(retain_rates[512:]), numpy.min(retain_rates[512:])
		retain_rates = numpy.reshape(retain_rates, feature_map_size)
		retain_rates = numpy.clip(retain_rates, 0, 1.0)

		# output_file_path = None if plot_directory is None else os.path.join(plot_directory, "noise.%d.epoch.%d.pdf" % (
		# temp_layer_index, temp_epoch_index))
		output_file_path = None if plot_directory is None else os.path.join(plot_directory, "noise.%d.epoch.%d.jpeg" % (
			temp_layer_index, temp_epoch_index))
		if len(feature_map_size) == 2:
			'''
			retain_rates -= numpy.min(retain_rates)
			if numpy.max(retain_rates) == 0:
				retain_rates += 1
			else:
				retain_rates /= numpy.max(retain_rates)
			'''
			#plot_image(retain_rates, output_file_path, title="epoch %d" % temp_epoch_index)
			epoch_to_retain_rate_array[temp_epoch_index] = retain_rates * 255
			epoch_to_retain_rate_file[temp_epoch_index] = output_file_path
		elif len(feature_map_size) == 3:
			# retain_rates = retain_rates[:, 13:17, 14:16]
			'''
			for x in xrange(3):
				retain_rates[x, :, :] -= numpy.min(retain_rates[x, :, :])
				if numpy.max(retain_rates[x, :, :]) == 0:
					retain_rates[x, :, :] += 1
				else:
					retain_rates[x, :, :] /= numpy.max(retain_rates[x, :, :])
			'''
			plot_image_rgb(retain_rates, output_file_path)

	output_file_path = None if plot_directory is None else os.path.join(plot_directory, "noise.%d.gif" % layer_index)
	if output_file_path is not None:
		retain_rate_array_list = [value for (key, value) in sorted(epoch_to_retain_rate_array.items())]
		# plot_gif_from_arrays(numpy.array(retain_rate_array_list), output_file_path, 60, scale=50)
		retain_rate_file_list = [value for (key, value) in sorted(epoch_to_retain_rate_file.items())]
		plot_gif_from_files(retain_rate_file_list, output_file_path, spf=1e-3)


def plot_gif_from_arrays(retain_rate_arrays, output_file_path, fps=10, scale=1.0):
	"""creates a gif given a stack of ndarray using moviepy
	filename : string
	The filename of the gif to write to
	array : array_like
	A numpy array that contains a sequence of images
	fps : int
	frames per second (default: 10)
	scale : float
	how much to rescale each image by (default: 1.0)
	"""
	from moviepy.editor import ImageSequenceClip

	if retain_rate_arrays.ndim == 3:  # If number of dimensions are 3,
		# copy into the color dimension if images are black and white
		retain_rate_arrays = retain_rate_arrays[..., numpy.newaxis] * numpy.ones(3)
	clip = ImageSequenceClip(list(retain_rate_arrays), fps=fps).resize(scale)
	clip.write_gif(output_file_path, fps=fps)

	return clip


def plot_gif_from_files(retain_rate_files, output_file_path, spf=1e-3):
	import imageio
	'''
	images = []
	for retain_rate_file in retain_rate_files:
		images.append(imageio.imread(retain_rate_file))
	imageio.mimsave(output_file_path, images, duration=spf)
	'''

	with imageio.get_writer(output_file_path, mode='I', duration=spf) as writer:
		for retain_rate_file in retain_rate_files:
			image = imageio.imread(retain_rate_file)
			writer.append_data(image)

	return


'''
randomimage = numpy.random.randn(100, 64, 64)
create_gif('test.gif', randomimage)                 #example 1

myimage = numpy.ones(shape=(300, 200))
myimage[:] = 25
myimage2 = numpy.ones(shape=(300, 200))
myimage2[:] = 85

arrayOfNdarray = numpy.array([myimage, myimage2])

create_gif(filename="grey_then_black.gif",          #example 2
           array=arrayOfNdarray,
           fps=5,
           scale=1.3)
'''


def plot_image(matrix, output_file_path=None, interpolation='bilinear', title=None):
	import matplotlib.pyplot as plt

	fig = plt.figure(figsize=(10, 10))
	plt.style.use('classic')
	plt.imshow(matrix, interpolation=interpolation)
	plt.grid(False)
	plt.tight_layout()
	if title:
		fig.axes[-1].set_title(title, size=15)

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')


def make_cube(r, g, b):
	ny, nx = r.shape
	# R = numpy.zeros([ny, nx, 3], dtype="d")
	R = numpy.zeros([ny, nx, 3])
	R[:, :, 0] = r
	G = numpy.zeros_like(R)
	G[:, :, 1] = g
	B = numpy.zeros_like(R)
	B[:, :, 2] = b

	RGB = R + G + B

	return R, G, B, RGB


def plot_image_rgb(matrix, output_file_path=None, interpolation='nearest'):
	import matplotlib.pyplot as plt
	from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes

	fig, ax = plt.subplots()
	ax_r, ax_g, ax_b = make_rgb_axes(ax, pad=0.02)
	# fig.add_axes(ax_r)
	# fig.add_axes(ax_g)
	# fig.add_axes(ax_b)

	# matrix = matrix[:, 13:18, 13:18]
	# print matrix[0, :, :]
	# matrix = matrix.astype(numpy.int)
	# print matrix[0, :, :]
	r = matrix[0, :, :]
	g = matrix[1, :, :]
	b = matrix[2, :, :]

	im_r, im_g, im_b, im_rgb = make_cube(r, g, b)
	kwargs = dict(origin="lower", interpolation=interpolation)
	ax.imshow(im_rgb, **kwargs)
	ax_r.imshow(im_r, **kwargs)
	ax_g.imshow(im_g, **kwargs)
	ax_b.imshow(im_b, **kwargs)

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')
		plt.close()


if __name__ == '__main__':
	import argparse

	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("--model_directory", dest="model_directory", action='store', default=None,
	                             help="model directory [None]")
	argument_parser.add_argument("--plot_directory", dest="plot_directory", action='store', default=None,
	                             help="plot directory [None]")
	argument_parser.add_argument("--snapshot_interval", dest="snapshot_interval", action='store', default="1",
	                             help="snapshot interval [1]")

	argument_parser.add_argument("--feature_map_size", dest="feature_map_size", action='store', default=None,
	                             help="feature map dimensions [None]")
	argument_parser.add_argument("--layer_index", dest="layer_index", action='store', type=int, default=0,
	                             help="layer index [0 - input layer]")

	arguments, additionals = argument_parser.parse_known_args()

	print("========== ========== ========== ========== ==========")
	for key, value in vars(arguments).items():
		print("%s=%s" % (key, value))
	print("========== ========== ========== ========== ==========")

	model_directory = arguments.model_directory
	plot_directory = arguments.plot_directory

	snapshot_interval = arguments.snapshot_interval
	snapshot_interval_tokens = [int(x) for x in snapshot_interval.split(",")]
	if len(snapshot_interval_tokens) == 1:
		snapshot_interval = [-1, -1, snapshot_interval_tokens[0]]
	elif len(snapshot_interval_tokens) == 2:
		snapshot_interval = [snapshot_interval_tokens[0], snapshot_interval_tokens[1], 1]
	elif len(snapshot_interval_tokens) == 3:
		snapshot_interval = [snapshot_interval_tokens[0], snapshot_interval_tokens[1], snapshot_interval_tokens[2]]

	feature_map_size = tuple([int(dimension) for dimension in arguments.feature_map_size.split(",")])
	layer_index = arguments.layer_index
	plot_retain_rates_for_image_features(model_directory, feature_map_size, layer_index, snapshot_interval,
	                                     plot_directory)
