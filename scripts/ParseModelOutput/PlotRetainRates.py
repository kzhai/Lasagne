import os
import re

import numpy
import numpy.random

retain_rates_file_name_pattern = re.compile(r'noise\.(?P<layer>[\d]+?)\.epoch\.(?P<epoch>[\d]+?)\.npy')


def plot_retain_rates(model_directory, snapshot_interval=[-1, -1, 1], plot_directory=None):
	epoch_indices = set()
	layer_dimensions = {}
	for file_name in os.listdir(model_directory):
		matcher = re.match(retain_rates_file_name_pattern, file_name)
		if matcher is None:
			continue

		epoch_index = int(matcher.group("epoch"))
		layer_index = int(matcher.group("layer"))

		if snapshot_interval[0] >= 0 and epoch_index < snapshot_interval[0]:
			continue
		if snapshot_interval[1] >= 0 and epoch_index > snapshot_interval[1]:
			continue
		if epoch_index % snapshot_interval[2] != 0:
			continue

		if layer_index not in layer_dimensions:
			dimension = numpy.prod(numpy.load(os.path.join(model_directory, file_name)).shape)
			layer_dimensions[layer_index] = dimension
		epoch_indices.add(epoch_index)

	layer_epoch_retain_rates = []
	for layer_index in layer_dimensions:
		layer_epoch_retain_rates.append(numpy.zeros((len(epoch_indices), layer_dimensions[layer_index])))

	for file_name in os.listdir(model_directory):
		matcher = re.match(retain_rates_file_name_pattern, file_name)
		if matcher is None:
			continue

		epoch_index = int(matcher.group("epoch"))
		layer_index = int(matcher.group("layer"))

		if snapshot_interval[0] >= 0 and epoch_index < snapshot_interval[0]:
			continue
		if snapshot_interval[1] >= 0 and epoch_index > snapshot_interval[1]:
			continue
		if epoch_index % snapshot_interval[2] != 0:
			continue

		retain_rates = numpy.load(os.path.join(model_directory, file_name))
		layer_epoch_retain_rates[layer_index][epoch_index / snapshot_interval[2], :] = retain_rates.flatten()

	for layer_index in layer_dimensions:
		layer_epoch_retain_rates[layer_index] = numpy.clip(layer_epoch_retain_rates[layer_index], 0, 1)
		print(layer_epoch_retain_rates[layer_index])

	output_file_path = None if plot_directory is None else os.path.join(plot_directory, "noise.pdf")
	plot_2D_violins([layer_epoch_retain_rates[layer_index][-1] for layer_index in layer_dimensions], output_file_path)

	for layer_index in layer_dimensions:
		output_file_path = None if plot_directory is None else os.path.join(plot_directory,
		                                                                    "noise.%d.pdf" % layer_index)
		# plot_3D_hist(layer_epoch_retain_rates[layer_index], snapshot_interval)
		plot_3D_wires(layer_epoch_retain_rates[layer_index], snapshot_interval[2], output_file_path)


def plot_2D_violins(matrix=None, output_file_path=None):
	import matplotlib.pyplot as plt

	# demonstrate how to toggle the display of different elements:
	# fig, axes = plt.subplots(111)
	fig = plt.figure()
	axe = fig.add_subplot(111)

	labels = ["layer %d" % x for x in range(len(matrix))]
	# axe.boxplot(matrix, labels=labels, showmeans=True, meanline=True, notch=1, sym='o', vert=1, whis=1.5)
	axe.violinplot(matrix, showmeans=True, showextrema=True, showmedians=False, vert=1)

	axe.get_xaxis().set_tick_params(direction='out')
	axe.xaxis.set_ticks_position('bottom')
	axe.set_xticks(numpy.arange(1, len(labels) + 1))
	axe.set_xticklabels(labels)
	axe.set_xlim(0.25, len(labels) + 0.75)

	# axe.set_title(setting, fontsize=10)
	axe.grid("on")

	# ax.set_xlim(0, 10)
	# ax.set_ylim(0, 10)
	# ax.set_zlim(0, 10)
	# axe.set_xlabel('layer index')
	axe.set_ylabel('retain rate')
	# axe.set_zlabel('# of neurons')
	# axe.set_title("Column (x) stride set to 0")

	plt.tight_layout()
	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')


def plot_3D_wires(matrix=None, rescale_x_interval=1, output_file_path=None):
	from mpl_toolkits.mplot3d import axes3d
	import matplotlib.pyplot as plt

	# fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	if matrix is None:
		# Get the test data
		X, Y, Z = axes3d.get_test_data(0.05)
	else:
		number_of_epochs = matrix.shape[0]
		number_of_neurons = matrix.shape[1]
		x_y_coordinates = numpy.zeros((2, number_of_epochs * number_of_neurons))
		coordinate_index = 0
		for epoch_index in range(number_of_epochs):
			for neuron_index in range(number_of_neurons):
				x_y_coordinates[0, coordinate_index] = epoch_index
				x_y_coordinates[1, coordinate_index] = matrix[epoch_index, neuron_index]
				coordinate_index += 1
		x, y = x_y_coordinates

		bin_x_size = 1
		bin_y_size = 0.01
		# bin_y_size = 1

		hist, xedges, yedges = numpy.histogram2d(x, y, bins=[number_of_epochs / bin_x_size, 1. / bin_y_size],
		                                         range=[[0, number_of_epochs], [0, 1]])

		# xpos, ypos = numpy.meshgrid(xedges[:-1] + 0.25 * bin_x_size, yedges[:-1] + 0.25 * bin_y_size)
		xpos, ypos = numpy.meshgrid(xedges[:-1], yedges[:-1])

		Z = hist.T
		X = xpos * rescale_x_interval
		Y = ypos

	'''
	# Give the first plot only wireframes of the type y = c
	ax1.plot_wireframe(X, Y, Z, rstride=10, cstride=1)
	ax1.set_title("Column (x) stride set to 0")

	# Give the second plot only wireframes of the type x = c
	ax2.plot_wireframe(X, Y, Z, rstride=1, cstride=10)
	ax2.set_title("Row (y) stride set to 0")
	'''

	# Give the second plot only wireframes of the type x = c
	ax.plot_wireframe(X, Y, Z, rstride=10, cstride=1)
	# ax.set_title("Column (x) stride set to 0")
	# ax.plot_wireframe(X, Y, Z, rstride=1, cstride=10)
	# ax.set_title("Row (y) stride set to 0")
	# ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

	# ax.set_xlim(0, 600)
	# ax.set_ylim(0, 1)
	# ax.set_zlim(0, 10)
	ax.set_xlabel('epoch')
	ax.set_ylabel('retain rate')
	ax.set_zlabel('# of neurons')

	plt.tight_layout()
	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')


def plot_3D_bars(matrix=None, rescale_x_interval=1):
	import matplotlib.pyplot as plt

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	if matrix is None:
		x, y = numpy.random.rand(2, 100) * 4

		bin_x_size = 1
		bin_y_size = 1
		hist, xedges, yedges = numpy.histogram2d(x, y, bins=[5, 4], range=[[0, 4], [0, 4]])
	else:
		number_of_epochs = matrix.shape[0]
		number_of_neurons = matrix.shape[1]
		x_y_coordinates = numpy.zeros((2, number_of_epochs * number_of_neurons))
		coordinate_index = 0
		for epoch_index in range(number_of_epochs):
			for neuron_index in range(number_of_neurons):
				x_y_coordinates[0, coordinate_index] = epoch_index
				x_y_coordinates[1, coordinate_index] = matrix[epoch_index, neuron_index]
				coordinate_index += 1
		x, y = x_y_coordinates

		bin_x_size = 1
		bin_y_size = 0.01

		hist, xedges, yedges = numpy.histogram2d(x, y, bins=[number_of_epochs / bin_x_size, 1. / bin_y_size],
		                                         range=[[0, number_of_epochs], [0, 1]])

	# Construct arrays for the anchor positions of the 16 bars.
	# Note: numpy.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
	# ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
	# with indexing='ij'.
	xpos, ypos = numpy.meshgrid(xedges[:-1] + 0.25 * bin_x_size, yedges[:-1] + 0.25 * bin_y_size)
	xpos = xpos.flatten('F') * rescale_x_interval
	ypos = ypos.flatten('F')
	zpos = numpy.zeros_like(xpos)

	# Construct arrays with the dimensions for the 16 bars.
	dx = 0.5 * bin_y_size * numpy.ones_like(zpos)
	dy = dx.copy()
	dz = hist.flatten()

	ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

	plt.show()


if __name__ == '__main__':
	import argparse

	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("--model_directory", dest="model_directory", action='store', default=None,
	                             help="model directory [None]")
	argument_parser.add_argument("--plot_directory", dest="plot_directory", action='store', default=None,
	                             help="plot directory [None]")
	argument_parser.add_argument("--snapshot_interval", dest="snapshot_interval", action='store', default="1",
	                             help="snapshot interval [1]")

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

	plot_retain_rates(model_directory, snapshot_interval, plot_directory)
