import os
import re
import sys

import numpy

from ParseModelOutputs import model_setting_pattern

debug_output_pattern = re.compile(
	r'^(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| DEBUG \|\s+(?P<output>[\w]+?): loss (?P<loss>([-\d.]+?|nan)), objective (?P<objective>([-\d.]+?|nan)), regularizer (?P<regularizer>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%$')

debug_rademacher_pattern = re.compile(
	r'^(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| DEBUG \|\s+Rademacher \(p=(?P<p_value>[\w]+?), q=(?P<q_value>[\w]+?)\) complexity: regularizer=(?P<regularizer>[\w.-]+?)$')


def parse_models(model_directories, number_of_points=80, output_file_path=None):
	trace_to_plot = numpy.zeros((0, 3));
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
		layer_activation_parameters = float(layer_activation_parameters_tokens[-1])
		'''
		for x in xrange(1, number_of_points + 1):
			train_debug_output_string = "\t".join(
				["%g" % token for token in debug_output_logs_on_train[-x, :].tolist()])
			train_debug_rademacher_string = debug_rademacher_logs_on_train[-x]
			print("%s\t%s\t%s" % (layer_activation_parameters_string, train_debug_output_string, train_debug_rademacher_string))
		'''

		temp_trace_to_plot = numpy.hstack(
			(numpy.full((len(debug_rademacher_logs_on_train), 1), layer_activation_parameters),
			 numpy.asarray(debug_rademacher_logs_on_train)[:, numpy.newaxis]))
		temp_trace_to_plot = numpy.hstack((temp_trace_to_plot, debug_output_logs_on_train))
		temp_trace_to_plot = temp_trace_to_plot[:, [0, 3, 1]]

		if number_of_points > 0:
			temp_trace_to_plot = temp_trace_to_plot[-number_of_points:, :]
		trace_to_plot = numpy.vstack((trace_to_plot, temp_trace_to_plot))

	print trace_to_plot.shape
	print numpy.min(trace_to_plot, axis=0), numpy.max(trace_to_plot, axis=0)
	plot_multiple_yaxis(trace_to_plot, output_file_path)


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


if __name__ == '__main__':
	import argparse

	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("--input_directory", dest="input_directory", action='store', default=None,
	                             help="input directory [None]")
	argument_parser.add_argument("--number_of_samples", dest="number_of_samples", action='store', type=int, default=0,
	                             help="number of samples [0]")
	argument_parser.add_argument("--output_file", dest="output_file", action='store', default=None,
	                             help="output file path [None]")
	# argument_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	# help="output directory [None]")

	arguments, additionals = argument_parser.parse_known_args()

	print("========== ========== ========== ========== ==========")
	for key, value in vars(arguments).items():
		print("%s=%s" % (key, value))
	print("========== ========== ========== ========== ==========")

	input_directory = arguments.input_directory
	output_file = arguments.output_file
	number_of_samples = arguments.number_of_samples

	parse_models(input_directory, number_of_samples, output_file_path=output_file)
