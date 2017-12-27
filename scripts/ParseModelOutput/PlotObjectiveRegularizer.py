import os
import re
import sys

import numpy
import scipy
import scipy.interpolate

from ParseModelOutputs import model_setting_pattern

debug_output_pattern = re.compile(
	r'^(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| DEBUG \|\s+(?P<output>[\w]+?): loss (?P<loss>([-\w.]+?|nan)), objective (?P<objective>([-\w.]+?|nan)), regularizer (?P<regularizer>([-\w.]+?|nan)), accuracy (?P<accuracy>[\w.]+?)%$')

debug_rademacher_pattern = re.compile(
	r'^(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| DEBUG \|\s+Rademacher \(p=(?P<p_value>[\w]+?), q=(?P<q_value>[\w]+?)\) complexity: regularizer=(?P<regularizer>[-\w.]+?)$')


def parse_model(model_log_file):
	model_settings = {}

	debug_output_logs_on_train = numpy.zeros((0, 4))
	debug_output_logs_on_test = numpy.zeros((0, 4))
	debug_output_on_train = True

	debug_rademacher_logs_on_train = []  # numpy.zeros((0, 3))
	debug_rademacher_logs_on_test = []  # numpy.zeros((0, 3))
	debug_rademacher_on_train = True

	model_log_stream = open(model_log_file, "r")
	for line in model_log_stream:
		line = line.strip()
		if len(line) == 0:
			continue

		matcher_settings = re.match(model_setting_pattern, line)
		if matcher_settings is not None:
			setting = matcher_settings.group("setting")
			value = matcher_settings.group("value")
			if setting in model_settings:
				if model_settings[setting] != value:
					sys.stderr.write(
						'different setting definition: %s = "%s" | "%s"\n' % (setting, model_settings[setting], value))
				continue
			model_settings[setting] = value

		matcher_debug_output = re.match(debug_output_pattern, line)
		if matcher_debug_output is not None:
			output = matcher_debug_output.group("output")
			if output == "deterministic":
				loss = float(matcher_debug_output.group("loss"))
				objective = float(matcher_debug_output.group("objective"))
				regularizer = float(matcher_debug_output.group("regularizer"))
				accuracy = float(matcher_debug_output.group("accuracy"))

				temp_minibatch = numpy.asarray([loss, objective, regularizer, accuracy])

				if debug_output_on_train:
					debug_output_logs_on_train = numpy.vstack((debug_output_logs_on_train, temp_minibatch))
				else:
					debug_output_logs_on_test = numpy.vstack((debug_output_logs_on_test, temp_minibatch))
				debug_output_on_train = not debug_output_on_train

		matcher_debug_rademacher = re.match(debug_rademacher_pattern, line)
		if matcher_debug_rademacher is not None:
			p_value = matcher_debug_rademacher.group("p_value")
			q_value = matcher_debug_rademacher.group("q_value")
			regularizer = matcher_debug_rademacher.group("regularizer")
			if p_value == "2" and q_value == "2":
				if debug_rademacher_on_train:
					debug_rademacher_logs_on_train.append(float(regularizer))
				else:
					debug_rademacher_logs_on_test.append(float(regularizer))
				debug_rademacher_on_train = not debug_rademacher_on_train

	return model_settings, debug_output_logs_on_train, debug_output_logs_on_test, debug_rademacher_logs_on_train, debug_rademacher_logs_on_test


def parse_models_2d(model_directories, number_of_points=80, output_file_path=None):
	trace_to_plot = numpy.zeros((0, 3))
	for model_name in os.listdir(model_directories):
		model_directory = os.path.join(model_directories, model_name)
		if os.path.isfile(model_directory):
			continue

		model_log_file = os.path.join(model_directory, "model.log")
		if not os.path.exists(model_log_file):
			continue

		model_settings, debug_output_logs_on_train, debug_output_logs_on_test, debug_rademacher_logs_on_train, debug_rademacher_logs_on_test = parse_model(
			model_log_file)
		if len(model_settings) == 0:
			continue

		layer_activation_parameters_tokens = model_settings["layer_activation_parameters"].strip("[]").split(",")
		layer_activation_parameter = float(layer_activation_parameters_tokens[-1])

		'''
		for x in xrange(1, number_of_points + 1):
			train_debug_output_string = "\t".join(
				["%g" % token for token in debug_output_logs_on_train[-x, :].tolist()])
			train_debug_rademacher_string = debug_rademacher_logs_on_train[-x]
			print("%s\t%s\t%s" % (layer_activation_parameters_string, train_debug_output_string, train_debug_rademacher_string))
		'''

		temp_trace_to_plot = numpy.hstack(
			(numpy.full((len(debug_rademacher_logs_on_train), 1), layer_activation_parameter),
			 numpy.asarray(debug_rademacher_logs_on_train)[:, numpy.newaxis]))
		# layer_activation_parameter, rademacher_complexity
		assert temp_trace_to_plot.shape[1] == 2
		temp_trace_to_plot = numpy.hstack((temp_trace_to_plot, debug_output_logs_on_train))
		# layer_activation_parameter, rademacher_complexity, loss, objective, regularizer, accuracy
		assert temp_trace_to_plot.shape[1] == 6
		temp_trace_to_plot = temp_trace_to_plot[:, [0, 3, 1]]

		if number_of_points > 0:
			temp_trace_to_plot = temp_trace_to_plot[-number_of_points:, :]
		trace_to_plot = numpy.vstack((trace_to_plot, temp_trace_to_plot))

	plot_multiple_yaxis(trace_to_plot, output_file_path)


def plot_multiple_yaxis(debug_output_logs_on_train, output_file_path=None):
	import matplotlib.pyplot as plt

	# assert len(train_logs) == len(valid_logs)
	# assert len(train_logs) == len(test_logs)

	def make_patch_spines_invisible(ax):
		ax.set_frame_on(True)
		ax.patch.set_visible(False)
		for sp in ax.spines.values():
			sp.set_visible(False)

	fig, primary_panel = plt.subplots()
	# fig.subplots_adjust(right=0.75)

	secondary_panel = primary_panel.twinx()
	# par2 = primary_panel.twinx()

	# Offset the right spine of par2.  The ticks and label have already been
	# placed on the right by twinx above.
	# par2.spines["right"].set_position(("axes", 1.2))
	# Having been created by twinx, par2 has its frame off, so the line of its
	# detached spine is invisible.  First, activate the frame but make the patch
	# and spines invisible.
	# make_patch_spines_invisible(par2)
	# Second, show the right spine.
	# par2.spines["right"].set_visible(True)

	p1, = primary_panel.plot(debug_output_logs_on_train[:, 0], debug_output_logs_on_train[:, 1], "r+",
	                         label="Loss (left axis)")
	p2, = secondary_panel.plot(debug_output_logs_on_train[:, 0], debug_output_logs_on_train[:, 2], "b+",
	                           label="Regularizer (right axis)")

	min_xlim = numpy.min(debug_output_logs_on_train[:, 0])
	max_xlim = numpy.max(debug_output_logs_on_train[:, 0])

	# min_xlim = 0
	# max_xlim = 1

	min_primary_ylim = numpy.min(debug_output_logs_on_train[:, 1])
	max_primary_ylim = numpy.max(debug_output_logs_on_train[:, 1])
	min_secondary_ylim = numpy.min(debug_output_logs_on_train[:, 2])
	max_secondary_ylim = numpy.max(debug_output_logs_on_train[:, 2])

	lines = [p1, p2]

	primary_panel.set_xlim(min_xlim, max_xlim)
	primary_panel.set_ylim(min_primary_ylim, max_primary_ylim)
	secondary_panel.set_ylim(min_secondary_ylim, max_secondary_ylim)
	# secondary_panel.set_ylim(numpy.min(train_logs[:, 3]), numpy.max(train_logs[:, 3]))
	# par2.set_ylim(1, 65)

	primary_panel.set_xlabel("Retain Rates")
	primary_panel.set_ylabel("Loss")
	secondary_panel.set_ylabel("Regularizer")
	# par2.set_ylabel("Velocity")

	'''
	primary_panel.yaxis.label.set_color(p1.get_color())
	secondary_panel.yaxis.label.set_color(p2.get_color())
	# par2.yaxis.label.set_color(p3.get_color())

	tkw = dict(size=4, width=1.5)
	primary_panel.tick_params(axis='y', colors=p1.get_color(), **tkw)
	secondary_panel.tick_params(axis='y', colors=p2.get_color(), **tkw)
	# par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
	primary_panel.tick_params(axis='x', **tkw)
	'''

	primary_panel.legend(lines, [l.get_label() for l in lines], loc='upper center', shadow=True, fontsize='large')

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')


def parse_models_3d(model_directories, number_of_points=10, plot_directory=None):
	trace_to_plot_coordinate = None
	for model_name in os.listdir(model_directories):
		model_directory = os.path.join(model_directories, model_name)
		if os.path.isfile(model_directory):
			continue

		model_log_file = os.path.join(model_directory, "model.log")
		if not os.path.exists(model_log_file):
			continue

		model_settings, debug_output_logs_on_train, debug_output_logs_on_test, debug_rademacher_logs_on_train, debug_rademacher_logs_on_test = parse_model(
			model_log_file)
		if len(model_settings) == 0:
			continue

		layer_activation_parameters_tokens = model_settings["layer_activation_parameters"].strip("[]").split(",")
		layer_activation_parameter = float(layer_activation_parameters_tokens[-1])

		dense_dimensions_tokens = model_settings["dense_dimensions"].strip("[]").split(",")
		dense_dimension = int(dense_dimensions_tokens[0])

		print(model_name, layer_activation_parameter, dense_dimension)
		if layer_activation_parameter < 0.2:
			continue

		debug_output_logs_on_train = debug_output_logs_on_train[-number_of_points:]
		debug_output_logs_on_test = debug_output_logs_on_test[-number_of_points:]
		debug_rademacher_logs_on_train = debug_rademacher_logs_on_train[-number_of_points:]

		temp_trace_to_plot = numpy.zeros((len(debug_rademacher_logs_on_train), 2))
		temp_trace_to_plot[:, 0] = dense_dimension
		temp_trace_to_plot[:, 1] = layer_activation_parameter

		temp_trace_to_plot = numpy.hstack(
			(temp_trace_to_plot, numpy.asarray(debug_rademacher_logs_on_train)[:, numpy.newaxis]))
		# dense_dimension, layer_activation_parameter, rademacher_complexity,
		assert temp_trace_to_plot.shape[1] == 3

		temp_trace_to_plot = numpy.hstack((temp_trace_to_plot, debug_output_logs_on_train))
		# dense_dimension, layer_activation_parameter, rademacher_complexity,
		# train_loss, train_objective, train_regularizer, train_accuracy,
		assert temp_trace_to_plot.shape[1] == 7
		temp_trace_to_plot = numpy.hstack((temp_trace_to_plot, debug_output_logs_on_test))
		# dense_dimension, layer_activation_parameter, rademacher_complexity,
		# train_loss, train_objective, train_regularizer, train_accuracy,
		# test_loss, test_objective, test_regularizer, test_accuracy
		assert temp_trace_to_plot.shape[1] == 11

		temp_trace_to_plot = numpy.mean(temp_trace_to_plot, axis=0)[numpy.newaxis, :]

		# Warning: all following rescales are necessary to the output before 171110
		# rademacher_scale *= numpy.sqrt((784 + dense_dimension) / (numpy.log(dense_dimension)) * dense_dimension)
		# rademacher_scale *= numpy.sqrt((dense_dimension + 10) / (numpy.log(10)) * 10)
		# Warning: all following rescales are necessary to the output between 171110 to 171117
		rademacher_scale = 1. / numpy.sqrt(dense_dimension)
		temp_trace_to_plot[:, 2] /= rademacher_scale

		# if number_of_points > 0:
		# temp_trace_to_plot = temp_trace_to_plot[-number_of_points:, :]
		if trace_to_plot_coordinate is None:
			trace_to_plot_coordinate = temp_trace_to_plot
		else:
			trace_to_plot_coordinate = numpy.vstack((trace_to_plot_coordinate, temp_trace_to_plot))

	# dense_dimension, layer_activation_parameter, rademacher_complexity,
	# train_loss, train_objective, train_regularizer, train_accuracy,
	# test_loss, test_objective, test_regularizer, test_accuracy
	assert trace_to_plot_coordinate.shape[1] == 11

	trace_to_plot_coordinate = trace_to_plot_coordinate[:, [1, 0, 10, 4, 2]]
	# layer_activation_parameter, dense_dimension, test_accuracy, train_objective, rademacher_complexity
	assert trace_to_plot_coordinate.shape[1] == 5

	zlabels = ["accuracy % (test)", "loss (train)", "regularizer"]
	zlims = [(95.5, 98.5), (0, 0.2), None]

	lambda_reg = 5e-2
	trace_to_plot_coordinate = numpy.hstack((trace_to_plot_coordinate,
	                                         (trace_to_plot_coordinate[:, 3] + lambda_reg * trace_to_plot_coordinate[:, 4])[:,
	                                         numpy.newaxis]))
	# layer_activation_parameter, dense_dimension, test_accuracy, train_objective, rademacher_complexity, train_loss
	assert trace_to_plot_coordinate.shape[1] == 6
	zlabels.append("objective (train)")
	#zlims.append((0, 0.4))
	zlims.append(None)

	# output_file_path = None if plot_directory is None else os.path.join(plot_directory, "triangular.pdf")
	# plot_3D_triangular(trace_to_plot_coordinate, output_file_path)

	trace_to_plot_surface = transform_coordinate_to_surface(trace_to_plot_coordinate)
	for smooth in [1e-3, 1e-2]:
		# output_file_path = None if plot_directory is None else os.path.join(plot_directory, "wires_%f.pdf" % (smooth))
		# plot_3D_wires(trace_to_plot_surface, output_file_path, smooth=smooth)

		output_file_path = None if plot_directory is None else os.path.join(plot_directory,
		                                                                    "hillshaded_lambda=%f_smooth=%f.pdf" % (lambda_reg, smooth))
		plot_3D_hillshaded(trace_to_plot_surface, output_file_path, smooth=smooth, zlabels=zlabels, zlims=zlims)


def transform_coordinate_to_surface(matrix):
	x_sorted = numpy.sort(numpy.unique(matrix[:, 0]))
	y_sorted = numpy.sort(numpy.unique(matrix[:, 1]))
	# print x_sorted
	# print y_sorted
	x_to_index = {x: index for index, x in enumerate(x_sorted)}
	y_to_index = {y: index for index, y in enumerate(y_sorted)}

	surface_matrix = numpy.zeros((matrix.shape[1], len(x_sorted), len(y_sorted)))
	for i, x in enumerate(x_sorted):
		for j, y in enumerate(y_sorted):
			surface_matrix[0, i, :] = x
			surface_matrix[1, :, j] = y
	for i in range(matrix.shape[0]):
		x_index = x_to_index[matrix[i, 0]]
		y_index = y_to_index[matrix[i, 1]]
		for j in range(matrix.shape[1] - 2):
			surface_matrix[2 + j, x_index, y_index] = matrix[i, 2 + j]
	return surface_matrix


def plot_3D_wires(matrix, output_file_path=None, smooth=0):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.tri as mtri

	x = matrix[0, :, :]
	y = matrix[1, :, :]

	fig, axes = plt.subplots(1, matrix.shape[0] - 2, subplot_kw={'projection': '3d'})
	for i in range(matrix.shape[0] - 2):
		z = matrix[2 + i, :, :]
		if smooth:
			tck = scipy.interpolate.bisplrep(x, y, z, s=smooth)
			z = scipy.interpolate.bisplev(x[:, 0], y[0, :], tck)

		axes[i].plot_wireframe(x, y, z, rstride=1, cstride=1)
		# ax.plot_wireframe(X, Y, Z, rstride=1, cstride=10)
		# ax.set_xlim(0, 10)
		# ax.set_ylim(0, 10)
		# ax.set_zlim(0, 10)
		# axes[i].set_title("Loss")
		axes[i].set_xlabel('retain rates')
		axes[i].set_ylabel('dimension')

	plt.tight_layout()
	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')


def plot_3D_hillshaded(matrix, output_file_path=None, smooth=0, zlabels=None, zlims=None):
	"""
	Demonstrates using custom hillshading in a 3D surface plot.
	"""
	from matplotlib import cm
	from matplotlib.colors import LightSource
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.tri as mtri

	'''
	filename = cbook.get_sample_data('jacksboro_fault_dem.npz', asfileobj=False)
	with np.load(filename) as dem:
		z = dem['elevation']
		nrows, ncols = z.shape
		x = np.linspace(dem['xmin'], dem['xmax'], ncols)
		y = np.linspace(dem['ymin'], dem['ymax'], nrows)
		x, y = np.meshgrid(x, y)

	region = np.s_[5:50, 5:50]
	x, y, z = x[region], y[region], z[region]
	'''

	ls = LightSource(270, 45)

	x = matrix[0, :, :]
	y = matrix[1, :, :]

	assert zlabels is None or len(zlabels) == matrix.shape[0] - 2
	assert zlims is None or len(zlims) == matrix.shape[0] - 2

	number_of_rows = int(numpy.floor(numpy.sqrt(matrix.shape[0] - 2)))
	number_of_cols = int(numpy.ceil((matrix.shape[0] - 2) / number_of_rows))
	fig, axes = plt.subplots(number_of_rows, number_of_cols, subplot_kw={'projection': '3d'})
	for i in range(matrix.shape[0] - 2):
		z = matrix[2 + i, :, :]
		if smooth:
			tck = scipy.interpolate.bisplrep(x, y, z, s=smooth)
			z = scipy.interpolate.bisplev(x[:, 0], y[0, :], tck)

		# To use a custom hillshading mode, override the built-in shading and pass
		# in the rgb colors of the shaded surface calculated from "shade".
		rgb = ls.shade(z, cmap=cm.nipy_spectral, vert_exag=0.1, blend_mode='soft')
		if number_of_rows == 1:
			axe = axes[i % number_of_cols]
		else:
			axe = axes[i / number_of_rows, i % number_of_cols]
		surf = axe.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=True,
		                        alpha=0.75, shade=False)
		# axes[i].set_title("Loss")
		axe.set_xlabel('retain rates')
		axe.set_xlim(0.2, 1.0)
		axe.set_ylabel('dimension')
		if zlabels is not None:
			axe.set_zlabel(zlabels[i])
		if zlims is not None and zlims[i] is not None:
			axe.set_zlim(zlims[i][0], zlims[i][1])

	'''
	axes[0].set_zlabel('loss')
	axes[0].set_zlim(0,0.2)
	axes[1].set_zlabel('regularizer')
	axes[2].set_zlabel('objective')
	axes[2].set_zlim(0, 0.5)
	'''

	plt.tight_layout()
	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')


def plot_3D_triangular(matrix, output_file_path=None):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.tri as mtri

	x = matrix[:, 0]
	y = matrix[:, 1]

	# figsize = (20, 10)
	fig, axes = plt.subplots(1, matrix.shape[1] - 2, subplot_kw={"projection": '3d'})
	# fig, axes = plt.subplots(1, matrix.shape[1] - 2, projection='3d')
	# fig = plt.figure()

	for i in range(matrix.shape[1] - 2):
		z = matrix[:, 2 + i]

		# axes[i] = fig.gca(projection='3d')
		axes[i].plot_trisurf(x, y, z, linewidth=0.01, alpha=0.75, antialiased=True, cmap=plt.cm.nipy_spectral)
		# axes[i].set_title("Loss")
		axes[i].set_xlabel('retain rates')
		axes[i].set_ylabel('dimension')

	# axes[0].set_zlabel('loss')
	# axes[1].set_zlabel('regularizer')

	plt.tight_layout()
	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')


if __name__ == '__main__':
	import argparse

	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("--model_directories", dest="model_directories", action='store', default=None,
	                             help="model directory [None]")
	argument_parser.add_argument("--number_of_samples", dest="number_of_samples", action='store', type=int, default=0,
	                             help="number of samples [0]")
	argument_parser.add_argument("--plot_directory", dest="plot_directory", action='store', default=None,
	                             help="output file path [None]")
	# argument_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	# help="output directory [None]")

	arguments, additionals = argument_parser.parse_known_args()

	print("========== ========== ========== ========== ==========")
	for key, value in vars(arguments).items():
		print("%s=%s" % (key, value))
	print("========== ========== ========== ========== ==========")

	model_directories = arguments.model_directories
	plot_directory = arguments.plot_directory
	number_of_samples = arguments.number_of_samples

	# parse_models_2d(input_directory, number_of_samples, output_file_path=output_file)
	parse_models_3d(model_directories, number_of_samples, plot_directory=plot_directory)
