import collections
import operator
import os
import sys

import numpy
import numpy.random


def parse_sequence_data(input_directory, output_directory):
	input_all_file = os.path.join(input_directory, "all.dat")
	label_information, type_information = collect_statistics(input_all_file)

	(label_to_index, index_to_label, label_document_frequency, label_term_frequency) = label_information
	output_label_file = os.path.join(output_directory, "label.info")
	output_label_stream = open(output_label_file, 'w')
	for label_index in range(len(index_to_label)):
		output_label_stream.write("%s\t%d\t%d\n" % (
			index_to_label[label_index], label_document_frequency[index_to_label[label_index]],
			label_term_frequency[index_to_label[label_index]]))
	output_label_stream.close()
	print("successfully parsed label file...")

	(type_to_index, index_to_type, type_document_frequency, type_term_frequency) = type_information
	output_type_file = os.path.join(output_directory, "type.info")
	output_type_stream = open(output_type_file, 'w')
	for type_index in range(len(index_to_type)):
		output_type_stream.write("%s\t%d\t%d\n" % (
			index_to_type[type_index], type_document_frequency[index_to_type[type_index]],
			type_term_frequency[index_to_type[type_index]]))
	output_type_stream.close()
	print("successfully parsed type file...")

	input_all_file = os.path.join(input_directory, "all.dat")
	output_all_feature_file = os.path.join(output_directory, "all.feature.npy")
	output_all_label_file = os.path.join(output_directory, "all.label.npy")
	parse_data(label_information, type_information, input_all_file, output_all_feature_file,
	           output_all_label_file)
	print("successfully parsed training file...")

	input_train_file = os.path.join(input_directory, "train.dat")
	output_train_feature_file = os.path.join(output_directory, "train.feature.npy")
	output_train_label_file = os.path.join(output_directory, "train.label.npy")
	parse_data(label_information, type_information, input_train_file, output_train_feature_file,
	           output_train_label_file)
	print("successfully parsed training file...")

	input_test_file = os.path.join(input_directory, "test.dat")
	output_test_feature_file = os.path.join(output_directory, "test.feature.npy")
	output_test_label_file = os.path.join(output_directory, "test.label.npy")
	parse_data(label_information, type_information, input_test_file, output_test_feature_file,
	           output_test_label_file)
	print("successfully parsed testing file...")

	input_valid_file = os.path.join(input_directory, "validate.dat")
	if os.path.exists(input_valid_file):
		output_valid_feature_file = os.path.join(output_directory, "validate.feature.npy")
		output_valid_label_file = os.path.join(output_directory, "validate.label.npy")
		parse_data(label_information, type_information, input_valid_file, output_valid_feature_file,
		           output_valid_label_file)
		print("successfully parsed validating file...")

def parse_data(label_information, type_information, input_file, output_feature_file, output_label_file):
	(label_to_index, index_to_label, label_document_frequency, label_term_frequency) = label_information
	(type_to_index, index_to_type, type_document_frequency, type_term_frequency) = type_information

	print(len(label_to_index))

	input_stream = open(input_file, 'r')
	data_features = []
	data_labels = []
	for line in input_stream:
		line = line.strip()
		fields = line.split("\t")
		if len(fields) != 2:
			sys.stderr.serialize("document collapsed: %s\n" % line)
			continue

		tokens = [type_to_index[token] for token in fields[0].split()]
		data_features.append(tokens)
		labels = [label_to_index[label] for label in fields[1].split()]
		data_labels.append(labels)

	data_features = numpy.array([numpy.array(data_feature, dtype=numpy.int16) for data_feature in data_features])
	data_labels = numpy.array([numpy.array(data_label, dtype=numpy.int16) for data_label in data_labels])

	print(type(data_features), len(data_features), type(data_labels), data_labels.shape)
	print(data_features.dtype, data_labels.dtype)
	print(type(data_features[0]), data_features[0].shape, type(data_labels[0]), data_labels[0].shape)
	print(data_features[0].dtype, data_labels[0].dtype)

	# print data_features.dtype, data_features.shape, numpy.max(data_features), numpy.min(data_features)
	# data_features = data_features.astype(numpy.float32)
	# print data_features.dtype, data_features.shape, numpy.max(data_features), numpy.min(data_features)
	# data_labels = data_labels.astype(numpy.uint8)
	# print data_labels.dtype, data_labels.shape, numpy.max(data_labels), numpy.min(data_labels)

	# assert data_features.shape[0] == data_labels.shape[0]
	# if data_features.shape[0] <= 0:
	# sys.stderr.write("no feature/label extracted...\n")
	# return
	# else:
	# print "successfully generated %d data instances..." % data_labels.shape[0]
	numpy.save(output_feature_file, data_features)
	numpy.save(output_label_file, data_labels)


def collect_statistics(input_file):
	label_term_frequency = collections.defaultdict()
	label_document_frequency = collections.defaultdict()

	type_document_frequency = collections.defaultdict()
	type_term_frequency = collections.defaultdict()

	input_stream = open(input_file, 'r')
	for line in input_stream:
		line = line.strip()
		fields = line.split("\t")
		if len(fields) != 2:
			sys.stderr.serialize("document collapsed: %s\n" % line)
			continue

		token_list = fields[0].split()
		token_set = set(token_list)
		for token in token_list:
			type_term_frequency[token] = type_term_frequency.get(token, 0) + 1
		for token in token_set:
			type_document_frequency[token] = type_document_frequency.get(token, 0) + 1

		label_list = fields[1].split()
		label_set = set(label_list)
		for label in label_list:
			label_term_frequency[label] = label_term_frequency.get(label, 0) + 1
		for label in label_set:
			label_document_frequency[label] = label_document_frequency.get(label, 0) + 1

	label_to_index = {}
	index_to_label = {}
	for label, document_frequency in sorted(list(label_document_frequency.items()), key=operator.itemgetter(1),
	                                        reverse=True):
		assert label not in label_to_index
		label_to_index[label] = len(label_to_index)
		index_to_label[len(index_to_label)] = label

	type_to_index = {}
	index_to_type = {}
	for token, document_frequency in sorted(list(type_document_frequency.items()), key=operator.itemgetter(1),
	                                        reverse=True):
		assert token not in type_to_index
		type_to_index[token] = len(type_to_index)
		index_to_type[len(index_to_type)] = token

	return (label_to_index, index_to_label, label_document_frequency, label_term_frequency), (
		type_to_index, index_to_type, type_document_frequency, type_term_frequency)


if __name__ == '__main__':
	import argparse

	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("--input_directory", dest="input_directory", action='store', default=None,
	                             help="input directory [None]")
	# argument_parser.add_argument("--output", dest="select_settings", action='store', default="None",
	# help="select settings to display [None]")
	argument_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	                             help="output directory [None]")

	arguments, additionals = argument_parser.parse_known_args()

	print("========== ========== ========== ========== ==========")
	for key, value in vars(arguments).items():
		print("%s=%s" % (key, value))
	print("========== ========== ========== ========== ==========")

	input_directory = arguments.input_directory
	output_directory = arguments.output_directory

	parse_sequence_data(input_directory, output_directory)
