import os
import random
import re
import time
from operator import itemgetter

import numpy
import numpy.random

from PlotRetainRates import retain_rates_file_name_pattern

random.seed(time.time())


def plot_retain_rates_for_word_features(model_directory, feature_mapping, layer_index=0, snapshot_interval=[0, 1000, 1],
                                        thresholds=20, plot_directory=None):
	# retain_rates_file_name_pattern = re.compile(r'layer\.(?P<layer>[\d]+?)\.epoch\.(?P<epoch>[\d]+?)\.npy')
	word_features = []
	for line in open(feature_mapping, 'r'):
		line = line.strip()
		tokens = line.split("\t")
		word_features.append(tokens[0])

	retain_rates = numpy.ones((len(word_features), snapshot_interval[1] + 1))

	for file_name in os.listdir(model_directory):
		matcher = re.match(retain_rates_file_name_pattern, file_name)
		if matcher is None:
			continue

		temp_layer_index = int(matcher.group("layer"))
		if temp_layer_index != layer_index:
			continue

		temp_epoch_index = int(matcher.group("epoch"))
		if temp_epoch_index > snapshot_interval[1]:
			continue

		if snapshot_interval[0] >= 0 and temp_epoch_index < snapshot_interval[0]:
			continue;
		if snapshot_interval[1] >= 0 and temp_epoch_index > snapshot_interval[1]:
			continue

		temp_retain_rates = numpy.load(os.path.join(model_directory, file_name))
		assert temp_retain_rates.shape == (len(word_features),)

		# print retain_rates.shape, temp_retain_rates.shape, temp_epoch_index
		retain_rates[:, temp_epoch_index] = temp_retain_rates
		if temp_epoch_index % snapshot_interval[2] != 0:
			continue

		temp_retain_rates_argsort_descending = numpy.argsort(temp_retain_rates)[::-1]
		output_file = os.path.join(plot_directory, "noise.%d.epoch.%d.tsv" % (temp_layer_index, temp_epoch_index))
		output_stream = open(output_file, 'w')
		for x in temp_retain_rates_argsort_descending:
			output_stream.write("%s\t%g\n" % (word_features[x], temp_retain_rates[x]))
		# output_stream.write("%g\t%s\n" % (max(1, int(retain_rates[x]*1000)), word_features[x]))

	# plot_image_rgb(retain_rates, output_file_path)

	if len(thresholds) == 1:
		retain_rates_argsort = numpy.argsort(retain_rates[:, -1]).tolist()
		interesting_feature_indices_1 = retain_rates_argsort[:thresholds[0]]
		interesting_feature_indices_2 = retain_rates_argsort[-thresholds[0]:]
		print(sorted(itemgetter(*interesting_feature_indices_1)(word_features)))
		print(sorted(itemgetter(*interesting_feature_indices_2)(word_features)))
		interesting_feature_indices = list(set(interesting_feature_indices_1 + interesting_feature_indices_2))
	elif len(thresholds) == 2:
		# interesting_feature_indices = numpy.argwhere(numpy.min(retain_rates, axis=1) <= 0)[:, 0].tolist()
		# interesting_feature_indices_1 = numpy.argwhere(numpy.min(retain_rates, axis=1) <= threshold)[:, 0].tolist()
		# interesting_feature_indices_2 = numpy.argwhere(numpy.max(retain_rates, axis=1) > 2)[:, 0].tolist()

		interesting_feature_indices_1 = numpy.argwhere(retain_rates[:, -1] <= thresholds[0])[:, 0].tolist()
		interesting_feature_indices_2 = numpy.argwhere(retain_rates[:, -1] >= thresholds[1])[:, 0].tolist()
		print(sorted(itemgetter(*interesting_feature_indices_1)(word_features)))
		print(sorted(itemgetter(*interesting_feature_indices_2)(word_features)))
		interesting_feature_indices = list(set(interesting_feature_indices_1 + interesting_feature_indices_2))

	# print interesting_feature_indices
	# print itemgetter(*interesting_feature_indices)(word_features)
	# print range(snapshot_interval[0], snapshot_interval[1]),
	# print retain_rates[interesting_feature_indices, :]

	retain_rates = numpy.clip(retain_rates, 0, 1)
	output_file_path = None if plot_directory is None else os.path.join(plot_directory, "noise.%d.epoch.%d.pdf" % (
		layer_index, snapshot_interval[1]))
	plot_lines(itemgetter(*interesting_feature_indices)(word_features),
	           range(snapshot_interval[0], snapshot_interval[1] + 1),
	           retain_rates[interesting_feature_indices, :],
	           output_file_path)


def plot_lines(features, indices, matrix, output_file_path=None, exclusive_radius=[4.5, 0.5]):
	import matplotlib.pyplot as plt

	# assert len(features) == len(indices)
	assert matrix.shape == (len(features), len(indices))

	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

	unavailable_coordinates = set()

	fig, ax = plt.subplots()
	for index, feature in enumerate(features):
		# line1, = ax.plot(x, matrix[index, :], '-', linewidth=1, label=feature)
		line1, = ax.plot(indices, matrix[index, :], colors[index % len(colors)] + "-", linewidth=1, alpha=.75)

		possible_text_x_pos = list(set(indices))

		temp_possible_text_x_pos = numpy.argwhere(matrix[index, :] < 1)[:, 0].tolist()
		possible_text_x_pos = list(set(possible_text_x_pos) & set(temp_possible_text_x_pos))

		temp_possible_text_x_pos = numpy.argwhere(matrix[index, :] >= 0)[:, 0].tolist()
		possible_text_x_pos = list(set(possible_text_x_pos) & set(temp_possible_text_x_pos))

		# print "indices before:", len(possible_text_x_pos)
		# temp_possible_text_x_pos = numpy.arange(0, 200).tolist()
		# possible_text_x_pos = list(set(possible_text_x_pos) & set(temp_possible_text_x_pos))
		# print "indices after:", len(possible_text_x_pos)

		# text_x_pos = random.sample(possible_text_x_pos, 1)[0]
		random.shuffle(possible_text_x_pos)
		text_x_pos = possible_text_x_pos.pop()
		# numpy.random.randint(indices[0], len(indices))
		text_y_pos = matrix[index, text_x_pos]

		new_coordinates = False
		while len(possible_text_x_pos) > 0:
			for (x, y) in unavailable_coordinates:
				# print x, y
				if (text_x_pos < x + exclusive_radius[0] and text_x_pos > x - exclusive_radius[0]) and (
								text_y_pos < y + exclusive_radius[1] and text_y_pos > y - exclusive_radius[1]):
					# text_x_pos = random.sample(possible_text_x_pos, 1)[0]
					text_x_pos = possible_text_x_pos.pop()
					# numpy.random.randint(indices[0], len(indices))
					text_y_pos = matrix[index, text_x_pos]
					new_coordinates = True
					break
				new_coordinates = False
			# print "exclusive radius is still too large..."
			if not new_coordinates:
				break

		if len(possible_text_x_pos) == 0:
			print("out of coordinates for text '%s'" % feature)
		unavailable_coordinates.add((text_x_pos, text_y_pos))
		# print unavailable_coordinates

		# print text_x_pos, text_y_pos, feature
		plt.text(text_x_pos, text_y_pos, feature, color=colors[index % len(colors)], va="bottom", ha="left",
		         rotation=0, alpha=.75)

	# line1.set_dashes(dashes)
	ax.grid(True, which='minor')
	ax.set_xlabel('epoch')
	ax.set_ylabel('retain rates')

	# line2, = ax.plot(x, -1 * np.sin(x), dashes=[30, 5, 10, 5],
	# label='Dashes set proactively')

	# ax.legend(loc='lower right')

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

	argument_parser.add_argument("--display_thresholds", dest="display_thresholds", action='store', default="20",
	                             help="threshold [int or list of floats]")

	argument_parser.add_argument("--feature_mapping", dest="feature_mapping", action='store', default=None,
	                             help="feature mapping file [None]")
	argument_parser.add_argument("--layer_index", dest="layer_index", action='store', type=int, default=0,
	                             help="layer index [0 - input layer]")

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
		snapshot_interval = snapshot_interval_tokens

	display_thresholds = arguments.display_thresholds
	threshold_tokens = [float(x) for x in display_thresholds.split(",")]
	if len(threshold_tokens) == 1:
		display_thresholds = [int(threshold_tokens[0])]
	elif len(threshold_tokens) == 2:
		display_thresholds = threshold_tokens

	feature_mapping = arguments.feature_mapping
	layer_index = arguments.layer_index

	plot_retain_rates_for_word_features(model_directory, feature_mapping, layer_index, snapshot_interval,
	                                    display_thresholds,
	                                    plot_directory)
