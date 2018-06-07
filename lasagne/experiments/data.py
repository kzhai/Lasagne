import logging
import os

import numpy

logger = logging.getLogger(__name__)

__all__ = [
	"load_features_labels",
]


def load_features_labels(input_directory, dataset="test"):
	data_set_x = numpy.load(os.path.join(input_directory, "%s.feature.npy" % dataset))
	data_set_y = numpy.load(os.path.join(input_directory, "%s.label.npy" % dataset))
	assert len(data_set_x) == len(data_set_y)
	logger.info("successfully load %d %s data from %s..." % (len(data_set_x), dataset, input_directory))
	return (data_set_x, data_set_y)


def load_features(input_directory, dataset="test"):
	data_set_x = numpy.load(os.path.join(input_directory, "%s.feature.npy" % dataset))
	logger.info("successfully load %d %s data from %s..." % (len(data_set_x), dataset, input_directory))
	return data_set_x


#
#
#
#
#

def generate_feature_label_sequences(dataset, **kwargs):
	# sequence_length, window_size = 1, position_offset = -1
	if dataset is None:
		return None

	sequence_set_x = dataset

	sequence_length = kwargs.get("sequence_length", 10)
	window_size = kwargs.get("window_size", 1)
	offset_position = kwargs.get("window_size", 1)
	join_sequences = kwargs.get("join_sequences", False)

	# dataset_x = -numpy.ones((0, sequence_length, window_size), dtype=numpy.int)
	dataset_x = -numpy.ones((0, sequence_length), dtype=numpy.int)
	# sequence_m = numpy.zeros((0, sequence_length), dtype=numpy.int8)
	dataset_y = numpy.zeros((0, sequence_length), dtype=numpy.int)

	print(sequence_set_x.shape)
	sequence = numpy.concatenate((sequence_set_x, sequence_set_x[:sequence_length]))
	print(sequence_set_x.shape)
	for index in range(len(sequence) - sequence_length):
		dataset_x = numpy.vstack((dataset_x, sequence[index:index + sequence_length]))
		dataset_y = numpy.vstack((dataset_y, sequence[index + 1:index + sequence_length + 1]))
	print(dataset_x.shape)
	print(dataset_y.shape)

	return dataset_x, dataset_y


def parse_sequence(dataset, sequence_length, window_size=1, position_offset=-1):
	if dataset is None:
		return None

	sequence_set_x, sequence_set_y = dataset
	# Parse data into sequences
	sequence_x = -numpy.ones((0, sequence_length, window_size), dtype=numpy.int32)
	sequence_m = numpy.zeros((0, sequence_length), dtype=numpy.int8)
	sequence_y = numpy.zeros(0, dtype=numpy.int32)

	sequence_indices_by_instance = [0]
	for instance_x, instance_y in zip(sequence_set_x, sequence_set_y):
		# context_windows = get_context_windows(train_sequence_x, window_size)
		# train_minibatch, train_minibatch_masks = get_mini_batches(context_windows, backprop_step)
		instance_sequence_x, instance_sequence_m = get_context_sequences(instance_x, sequence_length, window_size,
		                                                                 position_offset)
		assert len(instance_sequence_x) == len(instance_sequence_m)
		assert len(instance_sequence_x) == len(instance_y)

		sequence_x = numpy.concatenate((sequence_x, instance_sequence_x), axis=0)
		sequence_m = numpy.concatenate((sequence_m, instance_sequence_m), axis=0)
		sequence_y = numpy.concatenate((sequence_y, instance_y), axis=0)

		sequence_indices_by_instance.append(len(sequence_y))

	return sequence_x, sequence_y, sequence_m, sequence_indices_by_instance


def get_context_sequences(instance, sequence_length, window_size, position_offset=-1):
	'''
	context_windows :: list of word idxs
	return a list of minibatches of indexes
	which size is equal to backprop_step
	border cases are treated as follow:
	eg: [0,1,2,3] and backprop_step = 3
	will output:
	[[0],[0,1],[0,1,2],[1,2,3]]
	'''

	context_windows = get_context(instance, window_size, position_offset)
	sequences_x, sequences_m = get_sequences(context_windows, sequence_length)
	return sequences_x, sequences_m


def get_context(instance, window_size, position_offset=-1, vocab_size=None):
	'''
	window_size :: int corresponding to the size of the window
	given a list of indexes composing a sentence
	it will return a list of list of indexes corresponding
	to context windows surrounding each word in the sentence
	'''

	assert window_size >= 1
	if position_offset < 0:
		assert window_size % 2 == 1
		position_offset = window_size / 2
	assert position_offset < window_size

	instance = list(instance)

	if vocab_size is None:
		context_windows = -numpy.ones((len(instance), window_size), dtype=numpy.int32)
		# padded_sequence = window_size / 2 * [-1] + instance + window_size / 2 * [-1]
		padded_sequence = position_offset * [-1] + instance + (window_size - position_offset) * [-1]
		for i in range(len(instance)):
			context_windows[i, :] = padded_sequence[i:i + window_size]
	else:
		context_windows = numpy.zeros((len(instance), vocab_size), dtype=numpy.int32)
		# padded_sequence = window_size / 2 * [-1] + instance + window_size / 2 * [-1]
		padded_sequence = position_offset * [-1] + instance + (window_size - position_offset) * [-1]
		for i in range(len(instance)):
			for j in padded_sequence[i:i + window_size]:
				context_windows[i, j] += 1

	# assert len(context_windows) == len(sequence)
	return context_windows


def get_sequences(context_windows, sequence_length):
	'''
	context_windows :: list of word idxs
	return a list of minibatches of indexes
	which size is equal to backprop_step
	border cases are treated as follow:
	eg: [0,1,2,3] and backprop_step = 3
	will output:
	[[0],[0,1],[0,1,2],[1,2,3]]
	'''

	number_of_tokens, window_size = context_windows.shape
	sequences_x = -numpy.ones((number_of_tokens, sequence_length, window_size), dtype=numpy.int32)
	sequences_m = numpy.zeros((number_of_tokens, sequence_length), dtype=numpy.int32)
	for i in range(min(number_of_tokens, sequence_length)):
		sequences_x[i, 0:i + 1, :] = context_windows[0:i + 1, :]
		sequences_m[i, 0:i + 1] = 1
	for i in range(min(number_of_tokens, sequence_length), number_of_tokens):
		sequences_x[i, :, :] = context_windows[i - sequence_length + 1:i + 1, :]
		sequences_m[i, :] = 1
	return sequences_x, sequences_m
