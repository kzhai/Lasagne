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
	epoch_stochastic_objective = [x for x in epoch_indices]
	epoch_deterministic_objective = [x for x in epoch_indices]
	epoch_stochastic_regularizer = [x for x in epoch_indices]
	epoch_deterministic_regularizer = [x for x in epoch_indices]
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

		# minibatcb_index
		# stochastic_accuracy, stochastic_loss, stochastic_objective, stochastic_regularizer
		# deterministic_accuracy, deterministic_loss, deterministic_ objective, deterministic_regularizer
		epoch_minibatch_output = numpy.load(os.path.join(model_directory, file_name))
		epoch_stochastic_objective[
			(epoch_index - snapshot_interval[0]) / snapshot_interval[2]] = epoch_minibatch_output[:, 2]
		epoch_deterministic_objective[
			(epoch_index - snapshot_interval[0]) / snapshot_interval[2]] = epoch_minibatch_output[:, 6]

		epoch_stochastic_loss[
			(epoch_index - snapshot_interval[0]) / snapshot_interval[2]] = epoch_minibatch_output[:, 3]
		epoch_deterministic_loss[
			(epoch_index - snapshot_interval[0]) / snapshot_interval[2]] = epoch_minibatch_output[:, 7]

		epoch_stochastic_regularizer[
			(epoch_index - snapshot_interval[0]) / snapshot_interval[2]] = epoch_minibatch_output[:, 4]
		epoch_deterministic_regularizer[
			(epoch_index - snapshot_interval[0]) / snapshot_interval[2]] = epoch_minibatch_output[:, 8]

	epoch_indices = list(epoch_indices)
	epoch_indices.sort()
	epoch_indices = numpy.asarray(epoch_indices) * 2
	#output_file_path = None if plot_directory is None else os.path.join(plot_directory, "loss,stochastic+deterministic.pdf")
	#plot_errorbars([epoch_stochastic_loss, epoch_deterministic_loss], epoch_indices, output_file_path)
	output_file_path = None if plot_directory is None else os.path.join(plot_directory,
	                                                                    "objective,stochastic+deterministic.pdf")
	plot_errorbars([epoch_stochastic_objective, epoch_deterministic_objective], epoch_indices, output_file_path)

	'''
	output_file_path = None if plot_directory is None else os.path.join(plot_directory,
	                                                                    "loss+regularizer,stochastic+deterministic.pdf")
	plot_errorbars_multiple_yaxis([epoch_stochastic_loss, epoch_deterministic_loss],
	                              [epoch_stochastic_objective, epoch_deterministic_objective], epoch_indices,
	                              output_file_path)
	'''

def plot_errorbars_multiple_yaxis(primary_snapshots, secondary_snapshots, labels, output_file_path=None):
	import matplotlib.pyplot as plt

	# assert len(train_logs) == len(valid_logs)
	# assert len(train_logs) == len(test_logs)

	def make_patch_spines_invisible(ax):
		ax.set_frame_on(True)
		ax.patch.set_visible(False)
		for sp in ax.spines.values():
			sp.set_visible(False)

	def compute_mean_and_yerrbar(snapshots):
		percentile_50 = numpy.asarray([numpy.percentile(snapshot, 50, interpolation='linear') for
		                               snapshot in snapshots])
		percentile_25 = numpy.asarray([numpy.percentile(snapshot, 25, interpolation='linear') for
		                               snapshot in snapshots])
		yerrbars = percentile_50 - percentile_25

		return percentile_50, yerrbars

	fig, primary_panel = plt.subplots()
	# fig.subplots_adjust(right=0.75)

	secondary_panel = primary_panel.twinx()

	legends = []
	annotations = []

	means, yerrbars = compute_mean_and_yerrbar(primary_snapshots[0])
	errbar = primary_panel.errorbar(labels, means, yerr=yerrbars, linewidth=1, marker='o', markersize=5, linestyle='-.',
	                                color="r")
	legends.append(errbar)
	annotations.append('loss in stochastic mode (left axis)')

	means, yerrbars = compute_mean_and_yerrbar(primary_snapshots[1])
	errbar = primary_panel.errorbar(labels, means, yerr=yerrbars, linewidth=1, marker='x', markersize=5, linestyle='-.',
	                                color="b")
	legends.append(errbar)
	annotations.append('loss in deterministic mode (left axis)')

	means, yerrbars = compute_mean_and_yerrbar(secondary_snapshots[0])
	errbar = secondary_panel.errorbar(labels, means, yerr=yerrbars, linewidth=1, marker='o', markersize=5,
	                                  linestyle='--', color="r")
	legends.append(errbar)
	annotations.append('regularizer in stochastic mode (right axis)')

	means, yerrbars = compute_mean_and_yerrbar(secondary_snapshots[1])
	errbar = secondary_panel.errorbar(labels, means, yerr=yerrbars, linewidth=1, marker='x', markersize=5,
	                                  linestyle='--', color="b")
	legends.append(errbar)
	annotations.append('regularizer in deterministic mode (right axis)')

	plt.legend(legends, annotations, loc='upper center', shadow=True)

	primary_panel.yaxis.grid(True)
	secondary_panel.yaxis.grid(True)

	# primary_panel.set_xlim(min_xlim, max_xlim)
	#primary_panel.set_ylim(0.00, 0.04)
	#secondary_panel.set_ylim(0, 0.02)
	# secondary_panel.set_ylim(numpy.min(train_logs[:, 3]), numpy.max(train_logs[:, 3]))
	# par2.set_ylim(1, 65)

	primary_panel.set_xlabel("Epoch")
	primary_panel.set_ylabel("Loss")
	secondary_panel.set_ylabel("Regularizer")

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

	# primary_panel.legend(lines, [l.get_label() for l in lines])

	if output_file_path is not None:
		plt.tight_layout()
		plt.savefig(output_file_path, bbox_inches='tight')

	plt.show()


def plot_errorbars(stochastic_deterministic_snapshots, labels, output_file_path=None):
	import matplotlib.pyplot as plt

	fig = plt.figure()
	axe = fig.add_subplot(111)

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
	axe.set_ylim(0.002, 0.047)

	plt.legend((stochastic_errbar, deterministic_errbar), ('stochastic model', 'deterministic mode'), loc='upper right',
	           shadow=True)

	plt.tight_layout()
	if output_file_path is not None:
		plt.savefig(output_file_path, bbox_inches='tight')

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
