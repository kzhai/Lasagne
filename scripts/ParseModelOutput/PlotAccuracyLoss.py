import os

import numpy
import numpy.random


def plot_accuracy_loss(model_directory, maximum_iteration=0, plot_directory=None):
	model_log_file = os.path.join(model_directory, "model.log")
	from ParseModelOutputs import parse_model
	model_settings, train_logs, valid_logs, test_logs, best_model_logs = parse_model(model_log_file)
	if maximum_iteration > 0:
		train_logs = train_logs[:maximum_iteration]
		test_logs = test_logs[:maximum_iteration]

	output_file_path = None if plot_directory is None else os.path.join(plot_directory,
	                                                                    "objective.%d.pdf" % maximum_iteration)
	plot_multiple_yaxis(train_logs, valid_logs, test_logs, output_file_path)


def plot_multiple_yaxis(train_logs, valid_logs, test_logs, output_file_path=None):
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
	argument_parser.add_argument("--maximum_iteration", dest="maximum_iteration", action='store', type=int, default=0,
	                             help="maximum iteration [0]")

	arguments, additionals = argument_parser.parse_known_args()

	print("========== ========== ========== ========== ==========")
	for key, value in vars(arguments).items():
		print("%s=%s" % (key, value))
	print("========== ========== ========== ========== ==========")

	model_directory = arguments.model_directory
	maximum_iteration = arguments.maximum_iteration
	plot_directory = arguments.plot_directory

	plot_accuracy_loss(model_directory, maximum_iteration, plot_directory)
