import numpy
import numpy.random
import os
import re

__all__ = [
	"valid_log_pattern",
	"test_log_pattern",
	"train_log_pattern",
	"output_field_names",
	#
	"parse_file_to_matrix",
	"parse_file_to_lists",
	#
	# "train_progress_pattern",
	# "best_validate_minibatch_pattern",
	#
	"model_setting_pattern",
	"parse_model_settings",
	#
	"parse_model_output",
]

#'''
valid_log_pattern = re.compile(
	r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+validate: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), duration (?P<duration>[\d.]+?)s, loss (?P<loss>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')
test_log_pattern = re.compile(
	r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+test: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), duration (?P<duration>[\d.]+?)s, loss (?P<loss>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')
train_log_pattern = re.compile(
	r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+train: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), duration (?P<duration>[\d.]+?)s, loss (?P<loss>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')
output_field_names = ["epoch", "minibatch", "loss", "accuracy"]
#'''

'''
train_log_pattern = re.compile(
	r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+train: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), duration (?P<duration>[\d.]+?)s, loss (?P<loss>([-\d.]+?|nan)), regularizer (?P<regularizer>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')
valid_log_pattern = re.compile(
	r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+validate: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), duration (?P<duration>[\d.]+?)s, loss (?P<loss>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')
test_log_pattern = re.compile(
	r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+test: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), duration (?P<duration>[\d.]+?)s, loss (?P<loss>([-\d.]+?|nan)), regularizer (?P<regularizer>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')
output_field_names = ["epoch", "minibatch", "loss", "regularizer", "accuracy"]
'''

# train_progress_pattern = re.compile(r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+train: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), loss (?P<loss>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')
# best_validate_minibatch_pattern = re.compile(r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+best model found: epoch (?P<epoch>\d+?), minibatch (?P<minibatch>\d+?), loss (?P<loss>([-\d.]+?|nan)), accuracy (?P<accuracy>[\d.]+?)%')

model_setting_pattern = re.compile(
	r'(?P<date>[\d-]+?) (?P<time>[\d:,]+?) \| (?P<name>[\w\.]+?) \| INFO \|\s+(?P<setting>[\w]+?)=(?P<value>.+)')


def parse_models(model_directories, select_settings=None, output_file=None):
	if output_file != None:
		output_stream = open(output_file, 'w')

	accuracy_field_index = output_field_names.index("accuracy")

	for model_name in os.listdir(model_directories):
		model_directory = os.path.join(model_directories, model_name)
		if os.path.isfile(model_directory):
			continue

		model_log_file = os.path.join(model_directory, "model.log")
		if not os.path.exists(model_log_file):
			continue

		model_settings, train_logs, valid_logs, test_logs = parse_model_output(model_log_file)

		if len(model_settings) == 0:
			continue

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

		'''
		header_fields.append("final_best_accuracy")
		output_fields.append(best_model_logs[-1, accuracy_field_index] if best_model_logs.shape[0] > 0 else -1)
		header_fields.append("highest_best_accuracy")
		output_fields.append(numpy.max(best_model_logs[:, accuracy_field_index]) if best_model_logs.shape[0] > 0 else -1)
		'''

		header_fields.append("final_test_accuracy")
		output_fields.append(test_logs[-1, accuracy_field_index] if test_logs.shape[0] > 0 else -1)
		header_fields.append("highest_test_accuracy")
		output_fields.append(numpy.max(test_logs[:, accuracy_field_index]) if test_logs.shape[0] > 0 else -1)

		header_fields.append("final_train_accuracy")
		output_fields.append(train_logs[-1, accuracy_field_index] if train_logs.shape[0] > 0 else -1)
		header_fields.append("highest_train_accuracy")
		output_fields.append(numpy.max(train_logs[:, accuracy_field_index]) if train_logs.shape[0] > 0 else -1)

		'''
		header_fields.append("final_validate_accuracy")
		output_fields.append(valid_logs[-1, accuracy_field_index] if valid_logs.shape[0] > 0 else -1)
		header_fields.append("highest_validate_accuracy")
		output_fields.append(numpy.max(valid_logs[:, accuracy_field_index]) if valid_logs.shape[0] > 0 else -1)
		'''

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


def parse_model_output(model_log_file):
	model_settings = parse_model_settings(model_log_file)
	train_logs = parse_file_to_matrix(model_log_file, train_log_pattern, output_field_names)
	valid_logs = parse_file_to_matrix(model_log_file, valid_log_pattern, output_field_names)
	test_logs = parse_file_to_matrix(model_log_file, test_log_pattern, output_field_names)

	#assert len(test_logs) == len(train_logs), (len(test_logs), len(train_logs))
	#assert len(valid_logs) == 0 or len(valid_logs) == len(train_logs)
	# model_settings, train_logs, valid_logs, test_logs = parse_model_output(model_log_file, output_log_pattern_field_names)

	return model_settings, train_logs, valid_logs, test_logs


def parse_file_to_matrix(input_file, line_pattern, field_names, index_field_name="epoch"):
	output_matrix = numpy.zeros((0, len(field_names)))
	index_field_index = field_names.index(index_field_name)

	input_stream = open(input_file, "r")
	for input_line in input_stream:
		input_line = input_line.strip()
		if len(input_line) == 0:
			continue

		# line_fields = extract_fields_of_pattern_from_line(input_line, line_pattern, field_names)
		matcher = re.match(line_pattern, input_line)
		if matcher is None:
			continue

		line_fields = numpy.zeros(len(field_names))
		for field_index, field_name in enumerate(field_names):
			line_fields[field_index] = float(matcher.group(field_name))

		if len(output_matrix) > 0 and line_fields[index_field_index] == output_matrix[-1, index_field_index]:
			continue
		output_matrix = numpy.vstack((output_matrix, line_fields))

	return output_matrix


def parse_file_to_lists(input_file, line_pattern, field_names):
	# output_matrix = numpy.zeros((0, len(field_names)))
	# output_lists = [[] for x in len(field_names)]
	output_lists = []
	# index_field_index = field_names.index(index_field_name)

	input_stream = open(input_file, "r")
	for input_line in input_stream:
		input_line = input_line.strip()
		if len(input_line) == 0:
			continue

		# line_fields = extract_fields_of_pattern_from_line(input_line, line_pattern, field_names)
		matcher = re.match(line_pattern, input_line)
		if matcher is None:
			continue

		line_fields = []
		for field_index, field_name in enumerate(field_names):
			line_fields.append(matcher.group(field_name))
		# output_lists[field_index].append(matcher.group(field_name))
		# line_fields[field_index] = float(matcher.group(field_name))
		output_lists.append(line_fields)

	return output_lists


def parse_model_settings(log_file):
	model_settings = {}
	log_stream = open(log_file, "r")
	for line in log_stream:
		line = line.strip()
		if len(line) == 0:
			continue

		matcher = re.match(model_setting_pattern, line)
		if matcher is not None:
			setting = matcher.group("setting")
			value = matcher.group("value")
			assert setting not in model_settings
			model_settings[setting] = value

	return model_settings


def __extract_fields_of_pattern_from_line(input_line, line_pattern, field_names):
	matcher = re.match(line_pattern, input_line)
	if matcher is None:
		return None

	temp_minibatch = numpy.zeros(len(field_names))
	for field_index, field_name in enumerate(field_names):
		temp_minibatch[field_index] = float(matcher.group(field_name))
		'''
		epoch = int(matcher.group("epoch"))
		minibatch = int(matcher.group("minibatch"))
		loss = float(matcher.group("loss"))
		regularizer = float(matcher.group("regularizer"))
		accuracy = float(matcher.group("accuracy"))

		temp_minibatch = numpy.asarray([epoch, minibatch, loss, regularizer, accuracy])
		'''
	return temp_minibatch


def __parse_model_output(log_file, log_field_names=output_field_names):
	model_settings = {}

	log_stream = open(log_file, "r")

	train_logs = numpy.zeros((0, len(log_field_names)))
	valid_logs = numpy.zeros((0, len(log_field_names)))
	test_logs = numpy.zeros((0, len(log_field_names)))

	# best_model_logs = numpy.zeros((0, 5))
	# train_progress_logs = numpy.zeros((0, 5))

	best_found = False
	for line in log_stream:
		line = line.strip()
		if len(line) == 0:
			continue

		matcher_found = False

		matcher = re.match(model_setting_pattern, line)
		if matcher is not None:
			setting = matcher.group("setting")
			value = matcher.group("value")
			assert setting not in model_settings
			'''
			if setting in model_settings:
				if model_settings[setting] != value:
					sys.stderr.write(
						'different setting definition: %s = "%s" | "%s"\n' % (setting, model_settings[setting], value))
				continue
			'''
			model_settings[setting] = value
			matcher_found = True

		train_log_line = __extract_fields_of_pattern_from_line(line, train_log_pattern, log_field_names)
		if train_log_line is not None:
			train_logs = numpy.vstack((train_logs, train_log_line))
			matcher_found = True

		valid_log_line = __extract_fields_of_pattern_from_line(line, valid_log_pattern, log_field_names)
		if valid_log_line is not None:
			valid_logs = numpy.vstack((valid_logs, valid_log_line))
			matcher_found = True

		test_log_line = __extract_fields_of_pattern_from_line(line, test_log_pattern, log_field_names)
		if test_log_line is not None:
			if len(test_logs) > 0 and test_log_line[0] == test_logs[-1, 0]:
				continue
			test_logs = numpy.vstack((test_logs, test_log_line))
			matcher_found = True
			'''
			if best_found:
				best_model_logs = numpy.vstack((best_model_logs, test_minibatch))
				best_found = False
			'''

		'''
		best_validate_minibatch = minibatch_pattern_match(best_validate_minibatch_pattern, line)
		if best_validate_minibatch is not None:
			best_found = True
			matcher_found = True

		train_progress = minibatch_pattern_match(train_progress_pattern, line)
		if train_progress is not None:
			train_progress_logs = numpy.vstack((train_progress_logs, train_progress))
			matcher_found = True
		'''

		if matcher_found:
			continue
		else:
			# print line
			pass

	assert len(valid_logs) == 0 or len(train_logs) == len(valid_logs)
	assert len(train_logs) == len(test_logs)
	# train_logs = pandas.DataFrame(train_logs, columns=["epoch", "minibatch", "loss", "regularizer", "accuracy"])
	# valid_logs = pandas.DataFrame(valid_logs, columns=["epoch", "minibatch", "loss", "regularizer", "accuracy"])
	# test_logs = pandas.DataFrame(test_logs, columns=["epoch", "minibatch", "loss", "regularizer", "accuracy"])

	return model_settings, train_logs, valid_logs, test_logs


if __name__ == '__main__':
	import argparse

	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("--model_directories", dest="model_directories", action='store', default=None,
	                             help="directory contains model outputs [None]")
	argument_parser.add_argument("--select_settings", dest="select_settings", action='store', default="all",
	                             help="select settings to display [all]")
	argument_parser.add_argument("--output_result_file", dest="output_result_file", action='store', default=None,
	                             help="output result file [None]")

	arguments, additionals = argument_parser.parse_known_args()

	print("========== ========== ========== ========== ==========")
	for key, value in vars(arguments).items():
		print("%s=%s" % (key, value))
	print("========== ========== ========== ========== ==========")

	model_directories = arguments.model_directories
	select_settings = arguments.select_settings

	if select_settings.lower() == "none":
		select_settings = None
	elif select_settings.lower() == "all":
		select_settings = []
	else:
		select_settings = select_settings.split(",")
	output_result_file = arguments.output_result_file

	parse_models(model_directories, select_settings, output_result_file)
