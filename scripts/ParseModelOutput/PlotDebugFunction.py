import os
import re
import sys

import numpy

from ParseModelOutputs import model_setting_pattern

debug_function_pattern = re.compile(
	r'^(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| DEBUG \|\s+(?P<output>[\w]+?): loss (?P<loss>([-\d.]+?|nan)), objective (?P<objective>([-\d.]+?|nan)), regularizer (?P<regularizer>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')

#2017-10-20 16:24:25,813 | lasagne.experiments.debugger | DEBUG | Rademacher (p=2, q=2) complexity: regularizer=5.91717e-05

def parse_models(model_directories, number_of_points=10):
	for model_name in os.listdir(model_directories):
		model_directory = os.path.join(model_directories, model_name)
		if os.path.isfile(model_directory):
			continue

		model_log_file = os.path.join(model_directory, "model.log")
		if not os.path.exists(model_log_file):
			continue

		model_settings, train_logs, test_logs = parse_model(model_log_file)
		if len(model_settings) == 0:
			continue

		layer_activation_parameters_string = "\t".join(
			model_settings["layer_activation_parameters"].strip("[]").split(","))
		for x in xrange(1, number_of_points + 1):
			train_debug_function_string = "\t".join(["%g" % token for token in train_logs[-x, :].tolist()])
			print("%s\t%s" % (layer_activation_parameters_string, train_debug_function_string))


def parse_model(model_log_file):
	model_settings = {}

	train_debug_function_logs = numpy.zeros((0, 4))
	test_debug_function_logs = numpy.zeros((0, 4))
	debug_function_train = True

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
			matcher_found = True

		matcher_debug_function = re.match(debug_function_pattern, line)
		if matcher_debug_function is not None:
			output = matcher_debug_function.group("output")
			if output == "deterministic":
				loss = float(matcher_debug_function.group("loss"))
				objective = float(matcher_debug_function.group("objective"))
				regularizer = float(matcher_debug_function.group("regularizer"))
				accuracy = float(matcher_debug_function.group("accuracy"))

				temp_minibatch = numpy.asarray([loss, objective, regularizer, accuracy])

				if debug_function_train:
					train_debug_function_logs = numpy.vstack((train_debug_function_logs, temp_minibatch))
				else:
					test_debug_function_logs = numpy.vstack((test_debug_function_logs, temp_minibatch))
				debug_function_train = not debug_function_train

	return model_settings, train_debug_function_logs, test_debug_function_logs


if __name__ == '__main__':
	import argparse

	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("--input_directory", dest="input_directory", action='store', default=None,
	                             help="input directory [None]")
	# argument_parser.add_argument("--output", dest="select_settings", action='store', default="None",
	# help="select settings to display [None]")
	# argument_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	# help="output directory [None]")

	arguments, additionals = argument_parser.parse_known_args()

	print("========== ========== ========== ========== ==========")
	for key, value in vars(arguments).items():
		print("%s=%s" % (key, value))
	print("========== ========== ========== ========== ==========")

	input_directory = arguments.input_directory

	parse_models(input_directory)
