import os
import random
import re
import time
from operator import itemgetter

import numpy
import numpy.random

retain_rates_file_name_pattern = re.compile(r'noise\.(?P<layer>[\d]+?)\.epoch\.(?P<epoch>[\d]+?)\.npy')

random.seed(time.time())


def plot_feature_map(model_directory, feature_mapping, layer_index=0, snapshot_interval=[0, 1000, 1],
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
		print sorted(itemgetter(*interesting_feature_indices_1)(word_features))
		print sorted(itemgetter(*interesting_feature_indices_2)(word_features))
		interesting_feature_indices = list(set(interesting_feature_indices_1 + interesting_feature_indices_2))
	elif len(thresholds) == 2:
		# interesting_feature_indices = numpy.argwhere(numpy.min(retain_rates, axis=1) <= 0)[:, 0].tolist()
		# interesting_feature_indices_1 = numpy.argwhere(numpy.min(retain_rates, axis=1) <= threshold)[:, 0].tolist()
		# interesting_feature_indices_2 = numpy.argwhere(numpy.max(retain_rates, axis=1) > 2)[:, 0].tolist()

		interesting_feature_indices_1 = numpy.argwhere(retain_rates[:, -1] <= thresholds[0])[:, 0].tolist()
		interesting_feature_indices_2 = numpy.argwhere(retain_rates[:, -1] >= thresholds[1])[:, 0].tolist()
		print sorted(itemgetter(*interesting_feature_indices_1)(word_features))
		print sorted(itemgetter(*interesting_feature_indices_2)(word_features))
		interesting_feature_indices = list(set(interesting_feature_indices_1 + interesting_feature_indices_2))

	#print interesting_feature_indices
	#print itemgetter(*interesting_feature_indices)(word_features)
	#print range(snapshot_interval[0], snapshot_interval[1]),
	#print retain_rates[interesting_feature_indices, :]

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
		#temp_possible_text_x_pos = numpy.arange(0, 200).tolist()
		#possible_text_x_pos = list(set(possible_text_x_pos) & set(temp_possible_text_x_pos))
		# print "indices after:", len(possible_text_x_pos)

		#text_x_pos = random.sample(possible_text_x_pos, 1)[0]
		random.shuffle(possible_text_x_pos)
		text_x_pos = possible_text_x_pos.pop()
		# numpy.random.randint(indices[0], len(indices))
		text_y_pos = matrix[index, text_x_pos]

		new_coordinates = False
		while len(possible_text_x_pos)>0:
			for (x, y) in unavailable_coordinates:
				#print x, y
				if (text_x_pos < x + exclusive_radius[0] and text_x_pos > x - exclusive_radius[0]) and (
						text_y_pos < y + exclusive_radius[1] and text_y_pos > y - exclusive_radius[1]):
					#text_x_pos = random.sample(possible_text_x_pos, 1)[0]
					text_x_pos = possible_text_x_pos.pop()
					# numpy.random.randint(indices[0], len(indices))
					text_y_pos = matrix[index, text_x_pos]
					new_coordinates = True
					break
				new_coordinates = False
			#print "exclusive radius is still too large..."
			if not new_coordinates:
				break

		if len(possible_text_x_pos)==0:
			print "out of coordinates for text '%s'" % feature
		unavailable_coordinates.add((text_x_pos, text_y_pos))
		#print unavailable_coordinates

		#print text_x_pos, text_y_pos, feature
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


def test(matrix=None, rescale_x_interval=1, output_file_path=None):
	import matplotlib.pyplot as plt

	# fname = get_sample_data('/mnt/c/User/kezhai/Download/percent_bachelors_degrees_women_usa.csv')
	# gender_degree_data = csv2rec(fname)

	x_ticks = numpy.arange(1970, 2012)

	# These are the colors that will be used in the plot
	color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
	                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
	                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
	                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

	# You typically want your plot to be ~1.33x wider than tall. This plot
	# is a rare exception because of the number of lines being plotted on it.
	# Common sizes: (10, 7.5) and (12, 9)
	fig, ax = plt.subplots(1, 1, figsize=(12, 14))

	# Remove the plot frame lines. They are unnecessary here.
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)

	# Ensure that the axis ticks only show up on the bottom and left of the plot.
	# Ticks on the right and top of the plot are generally unnecessary.
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	fig.subplots_adjust(left=.06, right=.75, bottom=.02, top=.94)
	# Limit the range of the plot to only where the data is.
	# Avoid unnecessary whitespace.
	ax.set_xlim(1969.5, 2011.1)
	ax.set_ylim(-0.25, 90)

	# Make sure your axis ticks are large enough to be easily read.
	# You don't want your viewers squinting to read your plot.
	plt.xticks(range(1970, 2011, 10), fontsize=14)
	plt.yticks(range(0, 91, 10), fontsize=14)
	ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0f}'.format))
	ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))

	# Provide tick lines across the plot to help your viewers trace along
	# the axis ticks. Make sure that the lines are light and small so they
	# don't obscure the primary data lines.
	plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

	# Remove the tick marks; they are unnecessary with the tick lines we just
	# plotted.
	plt.tick_params(axis='both', which='both', bottom='off', top='off',
	                labelbottom='on', left='off', right='off', labelleft='on')

	# Now that the plot is prepared, it's time to actually plot the data!
	# Note that I plotted the majors in order of the highest % in the final year.
	majors = ['Health Professions', 'Public Administration', 'Education',
	          'Psychology', 'Foreign Languages', 'English',
	          'Communications\nand Journalism', 'Art and Performance', 'Biology',
	          'Agriculture', 'Social Sciences and History', 'Business',
	          'Math and Statistics', 'Architecture', 'Physical Sciences',
	          'Computer Science', 'Engineering']

	y_offsets = {'Foreign Languages': 0.5, 'English': -0.5,
	             'Communications\nand Journalism': 0.75,
	             'Art and Performance': -0.25, 'Agriculture': 1.25,
	             'Social Sciences and History': 0.25, 'Business': -0.75,
	             'Math and Statistics': 0.75, 'Architecture': -0.75,
	             'Computer Science': 0.75, 'Engineering': -0.25}

	for rank, column in enumerate(majors):
		# Plot each line separately with its own color.
		column_rec_name = column.replace('\n', '_').replace(' ', '_').lower()

		line = plt.plot(x_ticks,
		                numpy.random.random(42) * 100,
		                lw=2.5,
		                color=color_sequence[rank])

		# Add a text label to the right end of every line. Most of the code below
		# is adding specific offsets y position because some labels overlapped.
		# y_pos = gender_degree_data[column_rec_name][-1] - 0.5
		y_pos = 0.5

		if column in y_offsets:
			y_pos += y_offsets[column]

		# Again, make sure that all labels are large enough to be easily read
		# by the viewer.
		plt.text(2011.5, y_pos, column, fontsize=14, color=color_sequence[rank])

	# Make the title big enough so it spans the entire plot, but don't make it
	# so big that it requires two lines to show.

	# Note that if the title is descriptive enough, it is unnecessary to include
	# axis labels; they are self-evident, in this plot's case.
	fig.suptitle('Percentage of Bachelor\'s degrees conferred to women in '
	             'the U.S.A. by major (1970-2011)\n', fontsize=18, ha='center')

	# Finally, save the figure as a PNG.
	# You can also save it as a PDF, JPEG, etc.
	# Just change the file extension in this call.
	# plt.savefig('percent-bachelors-degrees-women-usa.png', bbox_inches='tight')
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

	plot_feature_map(model_directory, feature_mapping, layer_index, snapshot_interval, display_thresholds, plot_directory)
