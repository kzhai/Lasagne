import os
import re

import numpy
import numpy.random

retain_rates_file_name_pattern = re.compile(r'noise\.(?P<layer>[\d]+?)\.epoch\.(?P<epoch>[\d]+?)\.npy')


def plot_feature_map(model_directory, feature_map_size, layer_index=0, snapshot_interval=[-1, -1, 1],
                     plot_directory=None):
	# retain_rates_file_name_pattern = re.compile(r'layer\.(?P<layer>[\d]+?)\.epoch\.(?P<epoch>[\d]+?)\.npy')

	for file_name in os.listdir(model_directory):
		matcher = re.match(retain_rates_file_name_pattern, file_name)
		if matcher is None:
			continue

		temp_layer_index = int(matcher.group("layer"))
		if temp_layer_index != layer_index:
			continue

		temp_epoch_index = int(matcher.group("epoch"))
		if temp_epoch_index % snapshot_interval != 0:
			continue

		retain_rates = numpy.load(os.path.join(model_directory, file_name))
		#print numpy.mean(retain_rates[:512]), numpy.max(retain_rates[:512]), numpy.min(retain_rates[:512])
		#print numpy.mean(retain_rates[512:]), numpy.max(retain_rates[512:]), numpy.min(retain_rates[512:])
		retain_rates = numpy.reshape(retain_rates, feature_map_size)
		retain_rates = numpy.clip(retain_rates, 0, 1.0)

		output_file_path = None if plot_directory is None else os.path.join(plot_directory, "noise.%d.epoch.%d.pdf" % (
			temp_layer_index, temp_epoch_index))
		if len(feature_map_size) == 2:
			'''
			retain_rates -= numpy.min(retain_rates)
			if numpy.max(retain_rates) == 0:
				retain_rates += 1
			else:
				retain_rates /= numpy.max(retain_rates)
			'''
			plot_image(retain_rates, output_file_path)
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


def plot_image(matrix, output_file_path=None, interpolation='bilinear'):
	import matplotlib.pyplot as plt

	plt.figure()
	plt.imshow(matrix, interpolation=interpolation)
	plt.grid(False)
	plt.tight_layout()

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
