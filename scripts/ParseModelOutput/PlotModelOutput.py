import os

import numpy
import numpy.random

from ParseModelOutput import parse_file_to_matrix, train_log_pattern, valid_log_pattern, test_log_pattern, \
	output_field_names

__all__ = [
	"plot_multiple_subplots",
	"plot_multiple_yaxis",
]

colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
markers = ['+', 'x', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'd']
linestyles = ['-', '--', '-.', ':']


def plot_model_output(model_directory, snapshot_interval=[-1, -1, 1], plot_directory=None):
	model_log_file = os.path.join(model_directory, "model.log")

	# model_settings, train_logs, valid_logs, test_logs = parse_model_output(model_log_file)

	# model_settings = parse_model_settings(model_log_file)
	train_logs = parse_file_to_matrix(model_log_file, train_log_pattern, output_field_names)
	valid_logs = parse_file_to_matrix(model_log_file, valid_log_pattern, output_field_names)
	test_logs = parse_file_to_matrix(model_log_file, test_log_pattern, output_field_names)

	snapshot_indices = range(len(train_logs))
	if snapshot_interval[0] >= 0 and snapshot_interval[1] > 0:
		snapshot_indices = range(snapshot_interval[0], snapshot_interval[1], snapshot_interval[2])

	train_logs = train_logs[snapshot_indices]
	valid_logs = valid_logs[snapshot_indices] if len(valid_logs) > 0 else valid_logs
	test_logs = test_logs[snapshot_indices]

	field_names = ["epoch", "loss", "regularizer", "accuracy"]
	field_indices = []
	for field_name in field_names:
		field_indices.append(output_field_names.index(field_name))

	output_file_path = None if plot_directory is None else os.path.join(plot_directory, "objective.pdf")

	matrix = numpy.hstack((train_logs[:, field_indices], test_logs[:, field_indices[1:]]))
	'''
	plot_multiple_subplots(matrix, col_names=["epoch", "train", "train", "train", "test", "test", "test"],
	                       y_col_groups=[[3, 6], [1, 4], [2, 5]],
	                       y_col_group_labels=["accuracy", "loss", "regularizer"], x_col=0,
	                       output_file_path=output_file_path)
	'''
	plot_multiple_yaxis(matrix, col_names=["epoch", "train", "train", "train", "test", "test", "test"],
	                    y_col_groups=[[3, 6], [1, 4], [2, 5]],
	                    y_col_group_labels=["accuracy", "loss", "regularizer"], x_col=0,
	                    output_file_path=output_file_path)

	'''
	plot_multiple_subplots_backup(train_logs[:, field_indices], valid_logs[:, field_indices],
	                              test_logs[:, field_indices],
	                              field_names, output_file_path)
	plot_multiple_yaxis_backup(train_logs, valid_logs, test_logs, output_file_path)
	'''

def plot_multiple_subplots(matrix, col_names, y_col_groups=None, y_col_group_labels=None, x_col=0,
                           output_file_path=None):
	import matplotlib.pyplot as plt

	if y_col_groups is None:
		assert y_col_group_labels is None
		y_col_groups = [[y_col] for y_col in range(len(col_names)) if y_col != x_col]
		y_col_group_labels = [col_names[y_col] for y_col in range(len(col_names)) if y_col != x_col]
	assert y_col_group_labels is None or len(y_col_group_labels) == len(y_col_groups)

	min_xlim = numpy.min(matrix[:, x_col])
	max_xlim = numpy.max(matrix[:, x_col])

	fig, axes = plt.subplots(len(y_col_groups), 1)
	for plot_index, y_cols in enumerate(y_col_groups):
		for line_index, y_col in enumerate(y_cols):
			axes[plot_index].plot(matrix[:, x_col], matrix[:, y_col],
			                      colors[line_index % len(colors)] + markers[line_index % len(markers)],
			                      linestyle=linestyles[line_index % (len(linestyles))], linewidth=1,
			                      label=col_names[y_col])

		min_ylim = numpy.min(matrix[:, y_cols])
		max_ylim = numpy.max(matrix[:, y_cols])

		axes[plot_index].set_xlim(min_xlim, max_xlim)
		axes[plot_index].set_ylim(min_ylim, max_ylim)
		axes[plot_index].legend(loc="right")

		if plot_index < len(y_col_groups) - 1:
			axes[plot_index].set_xticklabels([])
		if y_col_group_labels is not None:
			axes[plot_index].set_ylabel(y_col_group_labels[plot_index])

	axes[-1].set_xlabel(col_names[x_col])

	plt.subplots_adjust(hspace=0.7)
	plt.tight_layout()
	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')


def plot_multiple_yaxis(matrix, col_names, y_col_groups=None, y_col_group_labels=None, x_col=0, output_file_path=None):
	import matplotlib.pyplot as plt

	def make_patch_spines_invisible(ax):
		ax.set_frame_on(True)
		ax.patch.set_visible(False)
		for sp in ax.spines.values():
			sp.set_visible(False)

	if y_col_groups is None:
		assert y_col_group_labels is None
		y_col_groups = [[y_col] for y_col in range(len(col_names)) if y_col != x_col]
		y_col_group_labels = [col_names[y_col] for y_col in range(len(col_names)) if y_col != x_col]
	assert y_col_group_labels is None or len(y_col_group_labels) == len(y_col_groups)

	min_xlim = numpy.min(matrix[:, x_col])
	max_xlim = numpy.max(matrix[:, x_col])

	#fig, host = plt.subplots()
	fig, host = plt.subplots(figsize=(12,8))
	# fig.subplots_adjust(right=0.75)
	pars = [host]
	for x in range(len(y_col_groups) - 1):
		pars.append(host.twinx())

		# Offset the right spine of par2.  The ticks and label have already been placed on the right by twinx above.
		pars[-1].spines["right"].set_position(("axes", 1 + .1 * x))
		# Having been created by twinx, par2 has its frame off, so the line of its detached spine is invisible.  First, activate the frame but make the patch and spines invisible.
		make_patch_spines_invisible(pars[-1])
		# Second, show the right spine.
		pars[-1].spines["right"].set_visible(True)

	# accuracy_field_index = output_field_names.index("accuracy")
	# loss_field_index = output_field_names.index("loss")

	lines = []
	for plot_index, y_cols in enumerate(y_col_groups):
		par_color = colors[plot_index]
		group_label = "" if y_col_group_labels is None else y_col_group_labels[plot_index] + ": "

		for line_index, y_col in enumerate(y_cols):
			line, = pars[plot_index].plot(matrix[:, x_col], matrix[:, y_col],
			                              par_color + markers[line_index % len(markers)],
			                              linestyle=linestyles[line_index % (len(linestyles))], linewidth=1,
			                              label=("%s%s" % (group_label, col_names[y_col])))
			lines.append(line)

		min_ylim = numpy.min(matrix[:, y_cols])
		max_ylim = numpy.max(matrix[:, y_cols])
		pars[plot_index].set_ylim(min_ylim, max_ylim)
		if y_col_group_labels is not None:
			pars[plot_index].set_ylabel(y_col_group_labels[plot_index])

		pars[plot_index].yaxis.label.set_color(par_color)
		pars[plot_index].tick_params(axis='y', colors=par_color)
	# tkw = dict(size=4, width=1.5)
	# pars[plot_index].tick_params(axis='y', colors=par_color, **tkw)
	# primary_panel.tick_params(axis='x', **tkw)

	host.set_xlim(min_xlim, max_xlim)
	host.set_xlabel(col_names[x_col])

	# secondary_panel.set_ylim(numpy.min(train_logs[:, 3]), numpy.max(train_logs[:, 3]))
	# par2.set_ylim(1, 65)
	host.legend(lines, [l.get_label() for l in lines], loc="center right")

	plt.tight_layout()
	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')


def plot_multiple_subplots_backup(train_logs, valid_logs, test_logs, field_names, output_file_path=None):
	import matplotlib.pyplot as plt

	epoch_index = 0
	min_xlim = min(numpy.min(train_logs[:, 0]), numpy.min(test_logs[:, 0]))
	max_xlim = max(numpy.max(train_logs[:, 0]), numpy.max(test_logs[:, 0]))

	fig, axes = plt.subplots(len(field_names[1:]), 1)
	for index, field in enumerate(field_names[1:]):
		axes[index].plot(train_logs[:, 0], train_logs[:, index + 1], 'bo', linestyle="-", linewidth=1,
		                 label="train")
		axes[index].plot(test_logs[:, 0], test_logs[:, index + 1], 'rx', linestyle="-", linewidth=1, label="test")

		min_ylim = min(numpy.min(train_logs[:, index + 1]), numpy.min(test_logs[:, index + 1]))
		max_ylim = max(numpy.max(train_logs[:, index + 1]), numpy.max(test_logs[:, index + 1]))

		axes[index].set_xlim(min_xlim, max_xlim)
		axes[index].set_ylim(min_ylim, max_ylim)
		axes[index].set_ylabel(field)
		axes[index].legend(loc="right")

		if index < len(field_names):
			axes[index].set_xticklabels([])

	axes[-1].set_xlabel(field_names[0])
	plt.subplots_adjust(hspace=0.7)

	plt.tight_layout()
	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')


def plot_multiple_yaxis_backup(train_logs, valid_logs, test_logs, output_file_path=None):
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

	p1, = primary_panel.plot(train_logs[:, 0], train_logs[:, 3], "r+", label="Train-Accuracy")
	p2, = secondary_panel.plot(train_logs[:, 0], train_logs[:, 2], "rx", label="Train-Loss")
	p5, = primary_panel.plot(test_logs[:, 0], test_logs[:, 3], "b+", label="Test-Accuracy")
	p6, = secondary_panel.plot(test_logs[:, 0], test_logs[:, 2], "bx", label="Test-Loss")
	if len(valid_logs) > 0:
		p3, = primary_panel.plot(valid_logs[:, 0], valid_logs[:, 3], "g+", label="Validate-Accuracy")
		p4, = secondary_panel.plot(valid_logs[:, 0], valid_logs[:, 2], "gx", label="Validate-Loss")

		min_xlim = min(numpy.min(train_logs[:, 0]), numpy.min(test_logs[:, 0]), numpy.min(valid_logs[:, 0]))
		max_xlim = max(numpy.max(train_logs[:, 0]), numpy.max(test_logs[:, 0]), numpy.max(valid_logs[:, 0]))
		min_primary_ylim = min(numpy.min(train_logs[:, 3]), numpy.min(test_logs[:, 3]), numpy.min(valid_logs[:, 3]))
		max_primary_ylim = max(numpy.max(train_logs[:, 3]), numpy.max(test_logs[:, 3]), numpy.max(valid_logs[:, 3]))
		min_secondary_ylim = min(numpy.min(train_logs[:, 2]), numpy.min(test_logs[:, 2]), numpy.min(valid_logs[:, 2]))
		max_secondary_ylim = max(numpy.max(train_logs[:, 2]), numpy.max(test_logs[:, 2]), numpy.max(valid_logs[:, 2]))

		lines = [p1, p2, p3, p4, p5, p6]
	else:
		min_xlim = min(numpy.min(train_logs[:, 0]), numpy.min(test_logs[:, 0]))
		max_xlim = max(numpy.max(train_logs[:, 0]), numpy.max(test_logs[:, 0]))
		min_primary_ylim = min(numpy.min(train_logs[:, 3]), numpy.min(test_logs[:, 3]))
		max_primary_ylim = max(numpy.max(train_logs[:, 3]), numpy.max(test_logs[:, 3]))
		min_secondary_ylim = min(numpy.min(train_logs[:, 2]), numpy.min(test_logs[:, 2]))
		max_secondary_ylim = max(numpy.max(train_logs[:, 2]), numpy.max(test_logs[:, 2]))

		lines = [p1, p2, p5, p6]

	primary_panel.set_xlim(min_xlim, max_xlim)
	primary_panel.set_ylim(min_primary_ylim, max_primary_ylim)
	secondary_panel.set_ylim(min_secondary_ylim, max_secondary_ylim)
	# secondary_panel.set_ylim(numpy.min(train_logs[:, 3]), numpy.max(train_logs[:, 3]))
	# par2.set_ylim(1, 65)

	primary_panel.set_xlabel("Epoch")
	primary_panel.set_ylabel("Accuracy (%)")
	secondary_panel.set_ylabel("Loss")
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

	primary_panel.legend(lines, [l.get_label() for l in lines])

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')


if __name__ == '__main__':
	import argparse

	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("--model_directory", dest="model_directory", action='store', default=None,
	                             help="model directory [None]")
	# argument_parser.add_argument("--output", dest="select_settings", action='store', default="None",
	# help="select settings to display [None]")
	argument_parser.add_argument("--plot_directory", dest="plot_directory", action='store', default=None,
	                             help="plot directory [None]")
	# argument_parser.add_argument("--maximum_iteration", dest="maximum_iteration", action='store', type=int, default=0,
	# help="maximum iteration [0]")
	argument_parser.add_argument("--snapshot_interval", dest="snapshot_interval", action='store', default="1",
	                             help="snapshot interval [1]")

	arguments, additionals = argument_parser.parse_known_args()

	print("========== ========== ========== ========== ==========")
	for key, value in vars(arguments).items():
		print("%s=%s" % (key, value))
	print("========== ========== ========== ========== ==========")

	model_directory = arguments.model_directory
	# maximum_iteration = arguments.maximum_iteration
	plot_directory = arguments.plot_directory

	snapshot_interval = arguments.snapshot_interval
	snapshot_interval_tokens = [int(x) for x in snapshot_interval.split(",")]
	if len(snapshot_interval_tokens) == 1:
		snapshot_interval = [-1, -1, snapshot_interval_tokens[0]]
	elif len(snapshot_interval_tokens) == 2:
		snapshot_interval = [snapshot_interval_tokens[0], snapshot_interval_tokens[1], 1]
	elif len(snapshot_interval_tokens) == 3:
		snapshot_interval = [snapshot_interval_tokens[0], snapshot_interval_tokens[1], snapshot_interval_tokens[2]]

	plot_model_output(model_directory, snapshot_interval, plot_directory)
