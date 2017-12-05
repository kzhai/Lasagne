import os
import re

import numpy
import numpy.random

retain_rates_file_name_pattern = re.compile(r'function_outputs_train\.epoch_(?P<epoch>[\d]+?)\.npy')


def plot_loss_objective_regularizer(model_directory, snapshot_interval=[-1, -1, 1], plot_directory=None):
	epoch_indices = set()
	for file_name in os.listdir(model_directory):
		matcher = re.match(retain_rates_file_name_pattern, file_name)
		if matcher is None:
			continue

		epoch_index = int(matcher.group("epoch"))

		if snapshot_interval[0] >= 0 and epoch_index < snapshot_interval[0]:
			continue
		if snapshot_interval[1] >= 0 and epoch_index > snapshot_interval[1]:
			continue
		if epoch_index % snapshot_interval[2] != 0:
			continue

		epoch_indices.add(epoch_index)

	epoch_stochastic_loss = [x for x in epoch_indices]
	epoch_deterministic_loss = [x for x in epoch_indices]
	for file_name in os.listdir(model_directory):
		matcher = re.match(retain_rates_file_name_pattern, file_name)
		if matcher is None:
			continue

		epoch_index = int(matcher.group("epoch"))

		if snapshot_interval[0] >= 0 and epoch_index < snapshot_interval[0]:
			continue
		if snapshot_interval[1] >= 0 and epoch_index > snapshot_interval[1]:
			continue
		if epoch_index % snapshot_interval[2] != 0:
			continue

		epoch_minibatch_output = numpy.load(os.path.join(model_directory, file_name))
		epoch_stochastic_loss[epoch_index / snapshot_interval[2]] = epoch_minibatch_output[:, 2]
		epoch_deterministic_loss[epoch_index / snapshot_interval[2]] = epoch_minibatch_output[:, 6]

	epoch_indices = list(epoch_indices)
	epoch_indices.sort()
	plot_errorbars([epoch_stochastic_loss, epoch_deterministic_loss], epoch_indices)
	'''
	output_file_path = None if plot_directory is None else os.path.join(plot_directory,
	                                                                    "noise.%d.pdf" % layer_index)
	# plot_3D_hist(layer_epoch_retain_rates[layer_index], snapshot_interval)
	plot_3D_wires(layer_epoch_retain_rates[layer_index], snapshot_interval[2], output_file_path)
	'''


def plot_errorbars(stochastic_deterministic_snapshots, labels):
	import matplotlib.pyplot as plt

	fig = plt.figure()
	axe = fig.add_subplot(111)

	labels = numpy.asarray(labels)
	stochastic_snapshots = stochastic_deterministic_snapshots[0]
	deterministic_snapshots = stochastic_deterministic_snapshots[1]

	# stochastic_boxes = axe.boxplot(stochastic_snapshots, labels=labels, showmeans=True)
	# deterministic_boxes = axe.boxplot(deterministic_snapshots, labels=labels, showmeans=True)

	axe.yaxis.grid(True)
	# axe.set_xticks([y + 1 for y in range(len(all_data))])
	# axe.set_title('box plot')
	# axe.set_xlabel('xlabel')
	# axe.set_ylabel('ylabel')

	# plt.show()
	# plt.clf()

	stochastic_50_percentile = numpy.asarray([numpy.percentile(stochastic_snapshot, 50, interpolation='linear') for
	                                          stochastic_snapshot in stochastic_snapshots])
	stochastic_25_percentile = numpy.asarray([numpy.percentile(stochastic_snapshot, 25, interpolation='linear') for
	                                          stochastic_snapshot in stochastic_snapshots])
	stochastic_yerrbars = stochastic_50_percentile - stochastic_25_percentile
	stochastic_errbar = axe.errorbar(labels, stochastic_50_percentile, yerr=stochastic_yerrbars, linewidth=1,
	                                 marker='o', markersize=5, linestyle='-.', color="r")

	deterministic_50_percentile = numpy.asarray(
		[numpy.percentile(deterministic_snapshot, 50, interpolation='linear') for deterministic_snapshot in
		 deterministic_snapshots])
	deterministic_25_percentile = numpy.asarray(
		[numpy.percentile(deterministic_snapshot, 25, interpolation='linear') for deterministic_snapshot in
		 deterministic_snapshots])
	deterministic_yerrbars = deterministic_50_percentile - deterministic_25_percentile
	deterministic_errbar = axe.errorbar(labels, deterministic_50_percentile, yerr=deterministic_yerrbars, linewidth=1,
	                                    marker='x', markersize=5, linestyle='-.', color="b")

	axe.yaxis.grid(True)
	axe.set_xlabel('epoch')
	axe.set_ylabel('objective')

	plt.legend((stochastic_errbar, deterministic_errbar), ('stochstic', 'deterministic'), loc='upper right',
	           shadow=True)

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

	plot_loss_objective_regularizer(model_directory, snapshot_interval, plot_directory)
