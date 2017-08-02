import os
import re
import sys

import numpy
import numpy.random

valid_log_pattern = re.compile(
	r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+validate: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), duration (?P<duration>[\d.]+?)s, loss (?P<loss>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')
test_log_pattern = re.compile(
	r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+test: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), duration (?P<duration>[\d.]+?)s, loss (?P<loss>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')
train_log_pattern = re.compile(
	r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+train: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), duration (?P<duration>[\d.]+?)s, loss (?P<loss>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')

train_progress_pattern = re.compile(
	r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+train: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), loss (?P<loss>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')
best_validate_minibatch_pattern = re.compile(
	r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+best model found: epoch (?P<epoch>\d+?), minibatch (?P<minibatch>\d+?), loss (?P<loss>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')

model_setting_pattern = re.compile(
	r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+(?P<setting>[\w]+?)=(?P<value>.+)')


def parse_models(dataset_directory, select_settings=None, output_file=None):
	if output_file != None:
		output_stream = open(output_file, 'w')

	for model_name in os.listdir(dataset_directory):
		model_directory = os.path.join(dataset_directory, model_name)
		if os.path.isfile(model_directory):
			continue

		model_log_file = os.path.join(model_directory, "model.log")
		if not os.path.exists(model_log_file):
			continue

		model_settings, train_logs, valid_logs, test_logs, best_model_logs = parse_model(model_log_file)
		if len(model_settings) == 0:
			continue;

		'''
		if test_logs.shape[0]==0:
			print model_directory #model_setting.layer_activation_parameters, model_setting.layer_activation_styles
			continue
		'''

		header_fields = []
		output_fields = []
		header_fields.append("model_name")
		output_fields.append(model_name)
		if select_settings != None:
			if len(select_settings) == 0:
				select_settings = list(model_settings.keys())
			for select_setting in select_settings:
				header_fields.append(select_setting)
				output_fields.append(model_settings[select_setting])

		header_fields.append("final_best_accuracy")
		output_fields.append(best_model_logs[-1, -1] if best_model_logs.shape[0] > 0 else -1)
		header_fields.append("highest_best_accuracy")
		output_fields.append(numpy.max(best_model_logs[:, -1]) if best_model_logs.shape[0] > 0 else -1)
		header_fields.append("final_test_accuracy")
		output_fields.append(test_logs[-1, -1] if test_logs.shape[0] > 0 else -1)
		header_fields.append("highest_test_accuracy")
		output_fields.append(numpy.max(test_logs[:, -1]) if test_logs.shape[0] > 0 else -1)
		header_fields.append("final_train_accuracy")
		output_fields.append(train_logs[-1, -1] if train_logs.shape[0] > 0 else -1)
		header_fields.append("highest_train_accuracy")
		output_fields.append(numpy.max(train_logs[:, -1]) if train_logs.shape[0] > 0 else -1)
		header_fields.append("final_validate_accuracy")
		output_fields.append(valid_logs[-1, -1] if valid_logs.shape[0] > 0 else -1)
		header_fields.append("highest_validate_accuracy")
		output_fields.append(numpy.max(valid_logs[:, -1]) if valid_logs.shape[0] > 0 else -1)

		if output_file == None:
			print("\t".join("%s" % field for field in header_fields))
			print("\t".join("%s" % field for field in output_fields))
		else:
			output_stream.write("\t".join("%s" % field for field in header_fields))
			output_stream.write("\n")
			output_stream.write("\t".join("%s" % field for field in output_fields))
			output_stream.write("\n")

	if output_file != None:
		output_stream.close()


def parse_model(model_log_file):
	model_settings = {}

	model_log_stream = open(model_log_file, "r")

	train_logs = numpy.zeros((0, 4))
	valid_logs = numpy.zeros((0, 4))
	test_logs = numpy.zeros((0, 4))

	best_model_logs = numpy.zeros((0, 4))
	train_progress_logs = numpy.zeros((0, 4))

	best_found = False
	for line in model_log_stream:
		line = line.strip()
		if len(line) == 0:
			continue

		matcher_found = False

		matcher = re.match(model_setting_pattern, line)
		if matcher is not None:
			setting = matcher.group("setting")
			value = matcher.group("value")
			if setting in model_settings:
				if model_settings[setting] != value:
					sys.stderr.write(
						'different setting definition: %s = "%s" | "%s"\n' % (setting, model_settings[setting], value))
				continue
			model_settings[setting] = value
			matcher_found = True

		train_minibatch = minibatch_pattern_match(train_log_pattern, line)
		if train_minibatch is not None:
			train_logs = numpy.vstack((train_logs, train_minibatch))
			matcher_found = True

		valid_minibatch = minibatch_pattern_match(valid_log_pattern, line)
		if valid_minibatch is not None:
			valid_logs = numpy.vstack((valid_logs, valid_minibatch))
			matcher_found = True

		test_minibatch = minibatch_pattern_match(test_log_pattern, line)
		if test_minibatch is not None:
			test_logs = numpy.vstack((test_logs, test_minibatch))
			matcher_found = True
			if best_found:
				best_model_logs = numpy.vstack((best_model_logs, test_minibatch))
				best_found = False

		best_validate_minibatch = minibatch_pattern_match(best_validate_minibatch_pattern, line)
		if best_validate_minibatch is not None:
			best_found = True
			matcher_found = True

		train_progress = minibatch_pattern_match(train_progress_pattern, line)
		if train_progress is not None:
			train_progress_logs = numpy.vstack((train_progress_logs, train_progress))
			matcher_found = True

		if matcher_found:
			continue
		else:
			# print line
			pass

	return model_settings, train_logs, valid_logs, test_logs, best_model_logs


def minibatch_pattern_match(pattern, line):
	matcher = re.match(pattern, line)
	if matcher is not None:
		epoch = int(matcher.group("epoch"))
		minibatch = int(matcher.group("minibatch"))
		loss = float(matcher.group("loss"))
		accuracy = float(matcher.group("accuracy"))

		temp_minibatch = numpy.asarray([epoch, minibatch, loss, accuracy])
		return temp_minibatch

	return None


if __name__ == '__main__':
	import argparse

	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("--model_directory", dest="model_directory", action='store', default=None,
	                             help="directory contains model outputs [None]")
	argument_parser.add_argument("--select_settings", dest="select_settings", action='store', default="None",
	                             help="select settings to display [None]")
	argument_parser.add_argument("--output_result_file", dest="output_result_file", action='store', default=None,
	                             help="output result file [None]")

	arguments, additionals = argument_parser.parse_known_args()

	print("========== ========== ========== ========== ==========")
	for key, value in vars(arguments).items():
		print("%s=%s" % (key, value))
	print("========== ========== ========== ========== ==========")

	model_directory = arguments.model_directory
	select_settings = arguments.select_settings

	if select_settings.lower() == "none":
		select_settings = None
	elif select_settings.lower() == "all":
		select_settings = []
	else:
		select_settings = select_settings.split(",")
	output_result_file = arguments.output_result_file

	parse_models(model_directory, select_settings, output_result_file)
# parse_model(model_directory)
