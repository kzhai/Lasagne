import os
import random
import re
import time
from operator import itemgetter

import numpy
import numpy.random

from PlotRetainRateHistogramVsEpoch import noise_file_name_pattern


def plot_retain_rates_for_word_features(model_directory, feature_mapping, layer_index=0, snapshot_interval=[0, 1000, 1],
                                        thresholds=[], targets=[], plot_directory=None):
	# retain_rates_file_name_pattern = re.compile(r'layer\.(?P<layer>[\d]+?)\.epoch\.(?P<epoch>[\d]+?)\.npy')
	index_to_word = {}
	word_to_index = {}
	for line in open(feature_mapping, 'r'):
		line = line.strip()
		tokens = line.split("\t")
		word_to_index[tokens[0]] = len(word_to_index)
		index_to_word[len(index_to_word)] = tokens[0]

	retain_rates = numpy.ones((len(index_to_word), snapshot_interval[1] + 1))

	for file_name in os.listdir(model_directory):
		matcher = re.match(noise_file_name_pattern, file_name)
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
		assert temp_retain_rates.shape == (len(index_to_word),)

		# print retain_rates.shape, temp_retain_rates.shape, temp_epoch_index
		retain_rates[:, temp_epoch_index] = temp_retain_rates
		if temp_epoch_index % snapshot_interval[2] != 0:
			continue

		temp_retain_rates_argsort_descending = numpy.argsort(temp_retain_rates)[::-1]
		output_file = os.path.join(plot_directory, "noise.%d.epoch.%d.tsv" % (temp_layer_index, temp_epoch_index))
		output_stream = open(output_file, 'w')
		for x in temp_retain_rates_argsort_descending:
			output_stream.write("%s\t%g\n" % (index_to_word[x], temp_retain_rates[x]))
		# output_stream.write("%g\t%s\n" % (max(1, int(retain_rates[x]*1000)), word_features[x]))

	# plot_image_rgb(retain_rates, output_file_path)
	interesting_feature_indices = set()

	if len(thresholds) == 1:
		retain_rates_argsort = numpy.argsort(retain_rates[:, -1]).tolist()
		interesting_feature_indices_1 = retain_rates_argsort[:thresholds[0]]
		interesting_feature_indices_2 = retain_rates_argsort[-thresholds[0]:]
		print(sorted(itemgetter(*interesting_feature_indices_1)(index_to_word)))
		print(sorted(itemgetter(*interesting_feature_indices_2)(index_to_word)))
		interesting_feature_indices |= set(interesting_feature_indices_1 + interesting_feature_indices_2)
	elif len(thresholds) == 2:
		# interesting_feature_indices = numpy.argwhere(numpy.min(retain_rates, axis=1) <= 0)[:, 0].tolist()
		# interesting_feature_indices_1 = numpy.argwhere(numpy.min(retain_rates, axis=1) <= threshold)[:, 0].tolist()
		# interesting_feature_indices_2 = numpy.argwhere(numpy.max(retain_rates, axis=1) > 2)[:, 0].tolist()

		interesting_feature_indices_1 = numpy.argwhere(retain_rates[:, -1] <= thresholds[0])[:, 0].tolist()
		interesting_feature_indices_2 = numpy.argwhere(retain_rates[:, -1] >= thresholds[1])[:, 0].tolist()
		print(len(sorted(itemgetter(*interesting_feature_indices_1)(index_to_word))),
		      sorted(itemgetter(*interesting_feature_indices_1)(index_to_word)))
		print(len(sorted(itemgetter(*interesting_feature_indices_2)(index_to_word))),
		      sorted(itemgetter(*interesting_feature_indices_2)(index_to_word)))
		interesting_feature_indices |= set(interesting_feature_indices_1 + interesting_feature_indices_2)
	else:
		pass

	interesting_feature_indices |= set([word_to_index[w] for w in targets])
	interesting_feature_indices = list(interesting_feature_indices)

	# print interesting_feature_indices
	# print itemgetter(*interesting_feature_indices)(word_features)
	# print range(snapshot_interval[0], snapshot_interval[1]),
	# print retain_rates[interesting_feature_indices, :]

	retain_rates = numpy.clip(retain_rates, 0, 1)
	output_file_path = None if plot_directory is None else os.path.join(plot_directory,
	                                                                    "noise=%d,epoch=%d,counts=%d.pdf" % (
		                                                                    layer_index, snapshot_interval[1],
		                                                                    len(interesting_feature_indices)))
	plot_lines(itemgetter(*interesting_feature_indices)(index_to_word),
	           range(snapshot_interval[0], snapshot_interval[1] + 1),
	           retain_rates[interesting_feature_indices, :],
	           output_file_path)


def plot_lines(features, indices, matrix, output_file_path=None, exclusive_radius=[10, 5]):
	import matplotlib.pyplot as plt

	numpy.random.seed(int(time.time()))

	# assert len(features) == len(indices)
	assert matrix.shape == (len(features), len(indices))

	from matplotlib.colors import ColorConverter
	colors = list(ColorConverter.colors)
	random.shuffle(colors)

	# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
	print(len(colors))

	unavailable_coordinates = set()
	annotation_counts = 0

	fig, ax = plt.subplots(figsize=(5, 4))
	for index, feature in enumerate(features):
		# line1, = ax.plot(x, matrix[index, :], '-', linewidth=1, label=feature)
		line1, = ax.plot(indices, matrix[index, :], color=colors[index % len(colors)], linestyle="-", linewidth=2,
		                 alpha=1)

		possible_text_x_pos = list(set(indices))
		numpy.random.shuffle(possible_text_x_pos)

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

			new_text_x_pos = annotation_counts * 80
			new_text_y_pos = 1.01 + numpy.random.random() * 0.05
			ax.annotate(feature,
			            xy=(text_x_pos, text_y_pos),
			            xycoords='data',
			            xytext=(new_text_x_pos, new_text_y_pos),
			            # textcoords='offset points',
			            textcoords='data',
			            # bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"),
			            bbox=dict(boxstyle="round", fc="w"),
			            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3,rad=.2", alpha=0.5),
			            color=colors[index % len(colors)],
			            rotation=15, alpha=1, weight="medium",
			            )

			annotation_counts += 1
			continue

		unavailable_coordinates.add((text_x_pos, text_y_pos))
		# print unavailable_coordinates

		# print text_x_pos, text_y_pos, feature
		plt.text(text_x_pos, text_y_pos, feature, color=colors[index % len(colors)], va="bottom", ha="left",
		         rotation=15, alpha=1, weight="medium")

	# line1.set_dashes(dashes)
	ax.grid(True, which='minor')
	ax.set_xlabel('epoch')
	ax.set_ylabel('retain rates')
	ax.set_ylim(0, 1)
	ax.set_xlim(indices[0], indices[-1])

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

	argument_parser.add_argument("--display_thresholds", dest="display_thresholds", action='store', default=None,
	                             help="threshold [int or list of floats]")
	argument_parser.add_argument("--display_targets", dest="display_targets", action='store', default=None,
	                             help="targets [list of words]")

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
	if display_thresholds is None:
		display_thresholds = []
	else:
		threshold_tokens = [float(x) for x in display_thresholds.split(",")]
		if len(threshold_tokens) == 1:
			display_thresholds = [int(threshold_tokens[0])]
		elif len(threshold_tokens) == 2:
			display_thresholds = threshold_tokens

	display_targets = arguments.display_targets
	if display_targets is None:
		display_targets = []
	else:
		display_targets_stream = open(display_targets, 'r')
		display_targets = []
		for line in display_targets_stream:
			line = line.strip()
			if len(line) == 0:
				continue
			display_targets.append(line);
		# display_targets = display_targets.split(",")

	feature_mapping = arguments.feature_mapping
	layer_index = arguments.layer_index

	plot_retain_rates_for_word_features(model_directory, feature_mapping, layer_index, snapshot_interval,
	                                    display_thresholds, display_targets, plot_directory)
