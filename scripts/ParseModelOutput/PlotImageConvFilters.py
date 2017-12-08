import os
import re

import numpy
import numpy.random

conv_filters_file_name_pattern = re.compile(r'conv\.(?P<layer>[\d]+?)\.epoch\.(?P<epoch>[\d]+?)\.npy')


def plot_feature_map(model_directory, feature_map_size, layer_index=0, snapshot_interval=100, plot_directory=None):
	for file_name in os.listdir(model_directory):
		matcher = re.match(conv_filters_file_name_pattern, file_name)
		if matcher is None:
			continue

		temp_layer_index = int(matcher.group("layer"))
		if temp_layer_index != layer_index:
			continue

		temp_epoch_index = int(matcher.group("epoch"))
		if temp_epoch_index % snapshot_interval != 0:
			continue

		conv_filters = numpy.load(os.path.join(model_directory, file_name))

		'''
		methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
		           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
		           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
		'''

		output_file_path = None if plot_directory is None else os.path.join(plot_directory, "conv.%d.epoch.%d.pdf" % (
			temp_layer_index, temp_epoch_index))
		plot_images(conv_filters, feature_map_size, output_file_path)
	# grid = np.random.rand(4, 4)


def plot_images(conv_filters, feature_map_size, output_file_path=None, interpolation='bilinear'):
	import matplotlib.pyplot as plt

	assert conv_filters.shape[0] * conv_filters.shape[1] == feature_map_size[0] * feature_map_size[1];
	conv_filters = numpy.reshape(conv_filters, (
		feature_map_size[0], feature_map_size[1], conv_filters.shape[2], conv_filters.shape[3]))

	fig, axes = plt.subplots(feature_map_size[0], feature_map_size[1],
	                         figsize=(feature_map_size[1] * 2, feature_map_size[0] * 2),
	                         subplot_kw={'xticks': [], 'yticks': []})

	fig.subplots_adjust(hspace=0.05, wspace=0.05)

	for x in range(feature_map_size[0]):
		for y in range(feature_map_size[1]):
			ax = axes[x, y];
			ax.imshow(conv_filters[x, y, :, :], interpolation=interpolation)
	# ax.set_title(interp_method)

	plt.grid(False)
	plt.tight_layout()

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')


if __name__ == '__main__':
	import argparse

	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("--model_directory", dest="model_directory", action='store', default=None,
	                             help="model directory [None]")
	argument_parser.add_argument("--plot_directory", dest="plot_directory", action='store', default=None,
	                             help="plot directory [None]")
	argument_parser.add_argument("--snapshot_interval", dest="snapshot_interval", action='store', type=int, default=1,
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

	feature_map_size = tuple([int(dimension) for dimension in arguments.feature_map_size.split(",")])
	layer_index = arguments.layer_index
	plot_feature_map(model_directory, feature_map_size, layer_index, snapshot_interval, plot_directory)
