import os
import re

import numpy
import numpy.random

from PlotRetainRates import retain_rates_file_name_pattern
from ParseModelOutputs import parse_model


def plot_retain_rates(model_directories, layer_index=0, snapshot_index=-1, plot_directory=None):
	model_retain_rates = []
	model_names = []
	for model_name in os.listdir(model_directories):
		model_directory = os.path.join(model_directories, model_name)
		if os.path.isfile(model_directory):
			continue

		if snapshot_index >= 0:
			epoch_index = snapshot_index
		else:
			epoch_indices = set()

			for file_name in os.listdir(model_directory):
				matcher = re.match(retain_rates_file_name_pattern, file_name)
				if matcher is None:
					continue

				if layer_index != int(matcher.group("layer")):
					continue

				epoch_index = int(matcher.group("epoch"))
				epoch_indices.add(epoch_index)

			epoch_index = max(epoch_indices)

		model_log_file = os.path.join(model_directory, "model.log")
		model_settings, train_logs, valid_logs, test_logs, best_model_logs = parse_model(model_log_file)

		model_setting_tokens = [model_name]
		#["regularizer", "learning_rate_decay", "learning_rate"]
		if "1.0" not in model_settings["regularizer"]:
			continue;
		#for setting_name in ["dropout_rate_update_interval"]:
			#model_setting_tokens.append(model_settings[setting_name])

		retain_rates = numpy.load(os.path.join(model_directory, "layer.%d.epoch.%d.npy" % (layer_index, epoch_index)))
		retain_rates = numpy.clip(retain_rates, 0, 1)
		if len(model_retain_rates) > 0:
			assert len(model_retain_rates[-1]) == numpy.prod(retain_rates.shape)

		model_retain_rates.append(retain_rates.flatten())
		model_names.append("-".join(model_setting_tokens))

	output_file_path = None if plot_directory is None else os.path.join(plot_directory, "noise.pdf")
	plot_bmh(model_retain_rates, model_names, output_file_path)


def plot_bmh(matrix, labels=None, output_file_path=None):
	import matplotlib.pyplot as plt

	if labels != None:
		assert len(matrix) == len(labels)

	plt.style.use('bmh')

	fig, axe = plt.subplots()
	for vector, label in zip(matrix, labels):
		axe.hist(vector, label=label, histtype="stepfilled", bins=100, alpha=0.8, normed=True)
	# axe.set_title("'bmh' style sheet")

	legend = axe.legend(loc='upper right', shadow=True, fontsize='medium')
	# Put a nicer background color on the legend.
	# legend.get_frame().set_facecolor('#00FFCC')

	plt.show()


if __name__ == '__main__':
	import argparse

	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("--model_directories", dest="model_directories", action='store', default=None,
	                             help="model directories [None]")
	argument_parser.add_argument("--plot_directory", dest="plot_directory", action='store', default=None,
	                             help="plot directory [None]")
	argument_parser.add_argument("--snapshot_index", dest="snapshot_index", action='store', default=-1, type=int,
	                             help="snapshot index [-1]")
	argument_parser.add_argument("--layer_index", dest="layer_index", action='store', default=0, type=int,
	                             help="layer index [-1]")

	arguments, additionals = argument_parser.parse_known_args()

	print("========== ========== ========== ========== ==========")
	for key, value in vars(arguments).items():
		print("%s=%s" % (key, value))
	print("========== ========== ========== ========== ==========")

	model_directories = arguments.model_directories
	plot_directory = arguments.plot_directory
	snapshot_index = arguments.snapshot_index
	layer_index = arguments.layer_index

	plot_retain_rates(model_directories, layer_index, snapshot_index, plot_directory)
