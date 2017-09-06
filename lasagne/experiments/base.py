import argparse
import datetime
import logging
import os
import random
import sys
import timeit

import numpy

from lasagne.experiments import debugger
from .. import objectives, updates, Xpolicy, Xregularization

# logging.basicConfig()
logger = logging.getLogger(__name__)

__all__ = [
	"layer_deliminator",
	"param_deliminator",
	#
	"parse_parameter_policy",
	#
	"construct_discriminative_parser",
	"validate_discriminative_arguments",
	#
	"config_model",
	"validate_config",
	"train_model",
	#
	"load_and_split_data",
	"load_data",
	"load_mnist",
]

layer_deliminator = ","
param_deliminator = ","


def construct_generic_parser():
	generic_parser = argparse.ArgumentParser(description="generic neural network arguments", add_help=True)

	# generic argument set 1
	generic_parser.add_argument("--input_directory", dest="input_directory", action='store', default=None,
	                            help="input directory [None]")
	generic_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	                            help="output directory [None]")
	# generic_parser.add_argument("--logging_file", dest="logging_file", action='store', default=None,
	# help="logging file [None]")

	# generic argument set 2
	generic_parser.add_argument("--objective", dest="objective", action='store', default="categorical_crossentropy",
	                            help="objective function [categorical_crossentropy] defined in objectives.py")
	generic_parser.add_argument("--update", dest="update", action='store', default="nesterov_momentum",
	                            help="update function [nesterov_momentum] defined updates.py")
	generic_parser.add_argument("--regularizer", dest='regularizer', action='append', default=[],
	                            help="regularizer function [None] defined in regularization.py")
	# "'l2:0.1'=l2-regularizer with lambda 0.1 applied over all layers, " +
	# "'l1:0.1;0.2;0.3'=l1-regularizer with lambda 0.1, 0.2, 0.3 applied over three layers"

	# generic argument set 3
	generic_parser.add_argument("--minibatch_size", dest="minibatch_size", type=int, action='store', default=-1,
	                            help="mini-batch size [-1]")
	generic_parser.add_argument("--number_of_epochs", dest="number_of_epochs", type=int, action='store', default=-1,
	                            help="number of epochs [-1]")
	# generic_parser.add_argument("--snapshot_interval", dest="snapshot_interval", type=int, action='store', default=0,
	# help="snapshot interval in number of epochs [0 - no snapshot]")

	# generic argument set 4
	'''
	generic_parser.add_argument("--learning_rate", dest="learning_rate", type=float, action='store', default=1e-2,
	                            help="learning rate [1e-2]")
	generic_parser.add_argument("--learning_rate_decay", dest="learning_rate_decay", action='store', default=None,
	                            help="learning rate decay [None], example, 'iteration,inverse_t,0.2,0.1', 'epoch,exponential,1.7,0.1', 'epoch,step,0.2,100'")
	'''
	generic_parser.add_argument("--learning_rate", dest="learning_rate", action='store', default="1e-2",
	                            help="learning policy [1e-2,constant] defined in policy.py with parameters")

	generic_parser.add_argument("--max_norm_constraint", dest="max_norm_constraint", type=float, action='store',
	                            default=0, help="max norm constraint [0 - None]")

	# generic_parser.add_argument('--debug', dest="debug", action='store_true', default=False, help="debug mode [False]")

	generic_parser.add_argument("--snapshot", dest='snapshot', action='append', default=[],
	                            help="snapshot function [None]")
	generic_parser.add_argument("--debug", dest='debug', action='append', default=[], help="debug function [None]")

	return generic_parser


def parse_parameter_policy(policy_string):
	policy_tokens = policy_string.split(param_deliminator)

	policy_tokens[0] = float(policy_tokens[0])
	assert policy_tokens[0] >= 0
	if len(policy_tokens) == 1:
		policy_tokens.append(Xpolicy.constant)
		return policy_tokens

	policy_tokens[1] = getattr(Xpolicy, policy_tokens[1])
	if policy_tokens[1] is Xpolicy.constant:
		assert len(policy_tokens) == 2
		return policy_tokens

	if policy_tokens[1] is Xpolicy.piecewise_constant:
		assert len(policy_tokens) == 4

		policy_tokens[2] = [float(boundary_token) for boundary_token in policy_tokens[2].split("-")]
		previous_boundary = 0
		for next_boundary in policy_tokens[2]:
			assert next_boundary > previous_boundary
			previous_boundary = next_boundary
		policy_tokens[3] = [float(value_token) for value_token in policy_tokens[3].split("-")]
		assert len(policy_tokens[2]) == len(policy_tokens[3])
		return policy_tokens

	assert policy_tokens[1] is Xpolicy.inverse_time_decay \
	       or policy_tokens[1] is Xpolicy.natural_exp_decay \
	       or policy_tokens[1] is Xpolicy.exponential_decay

	for x in xrange(2, 4):
		policy_tokens[x] = float(policy_tokens[x])
		assert policy_tokens[x] > 0

	if len(policy_tokens) == 4:
		policy_tokens.append(0)
	elif len(policy_tokens) == 5:
		policy_tokens[4] = float(policy_tokens[4])
		assert policy_tokens[4] > 0
	else:
		logger.error("unrecognized parameter decay policy %s..." % (policy_tokens))

	return policy_tokens


def validate_generic_arguments(arguments):
	# generic argument set 4
	'''
	assert arguments.learning_rate > 0
	if arguments.learning_rate_decay is not None:
		learning_rate_decay_tokens = arguments.learning_rate_decay.split(",")
		assert len(learning_rate_decay_tokens) == 4
		assert learning_rate_decay_tokens[0] in ["iteration", "epoch"]
		assert learning_rate_decay_tokens[1] in ["inverse_t", "exponential", "step"]
		learning_rate_decay_tokens[2] = float(learning_rate_decay_tokens[2])
		learning_rate_decay_tokens[3] = float(learning_rate_decay_tokens[3])
		arguments.learning_rate_decay = learning_rate_decay_tokens
	'''
	arguments.learning_rate = parse_parameter_policy(arguments.learning_rate)
	assert arguments.max_norm_constraint >= 0

	# generic argument set snapshots
	snapshots = {}
	for snapshot_interval_mapping in arguments.snapshot:
		fields = snapshot_interval_mapping.split(":")
		snapshot_function = getattr(debugger, fields[0])
		if len(fields) == 1:
			interval = 1
		elif len(fields) == 2:
			interval = int(fields[1])
		else:
			logger.error("unrecognized snapshot function setting %s..." % (snapshot_interval_mapping))
		snapshots[snapshot_function] = interval
	arguments.snapshot = snapshots
	debugs = set()
	for debug in arguments.debug:
		debug = getattr(debugger, debug)
		debugs.add(debug)
	arguments.debug = debugs

	# generic argument set 3
	assert arguments.minibatch_size > 0
	assert arguments.number_of_epochs > 0
	# assert arguments.snapshot_interval >= 0

	# generic argument set 2
	arguments.objective = getattr(objectives, arguments.objective)
	arguments.update = getattr(updates, arguments.update)

	regularizers = {}
	for regularizer_weight_mapping in arguments.regularizer:
		fields = regularizer_weight_mapping.split(":")
		# regularizer_function = getattr(regularization, fields[0])
		regularizer_function = getattr(Xregularization, fields[0])
		if len(fields) == 1:
			regularizers[regularizer_function] = [Xpolicy.constant, 1.0]
		elif len(fields) == 2:
			regularizers[regularizer_function] = parse_parameter_policy(fields[1])
			'''
			tokens = fields[1].split(layer_deliminator)
			if len(tokens) == 1:
				weight = float(tokens[0])
			else:
				weight = [float(token) for token in tokens]
			regularizers[regularizer_function] = weight
			'''
		else:
			logger.error("unrecognized regularizer function setting %s..." % (regularizer_weight_mapping))
	arguments.regularizer = regularizers

	# generic argument set 1
	# self.input_directory = arguments.input_directory
	assert os.path.exists(arguments.input_directory)

	output_directory = arguments.output_directory
	assert (output_directory is not None)
	if not os.path.exists(output_directory):
		os.mkdir(os.path.abspath(output_directory))
	# adjusting output directory
	now = datetime.datetime.now()
	suffix = now.strftime("%y%m%d-%H%M%S-%f") + ""
	# suffix += "-%s" % ("mlp")
	output_directory = os.path.join(output_directory, suffix)
	assert not os.path.exists(output_directory)
	# os.mkdir(os.path.abspath(output_directory))
	arguments.output_directory = output_directory

	return arguments


def construct_discriminative_parser():
	model_parser = construct_generic_parser()

	# model argument set
	model_parser.add_argument("--validation_data", dest="validation_data", type=int, action='store', default=0,
	                          help="validation data [0 - no validation data used], -1 - load validate.(feature|label).npy for validation]")
	model_parser.add_argument("--validation_interval", dest="validation_interval", type=int, action='store',
	                          default=1000,
	                          help="validation interval in number of mini-batches [1000]")

	return model_parser


def validate_discriminative_arguments(arguments):
	arguments = validate_generic_arguments(arguments)

	# model argument set
	assert arguments.validation_data >= -1
	assert (arguments.validation_interval > 0)

	return arguments


def load_data(input_directory, dataset="test"):
	data_set_x = numpy.load(os.path.join(input_directory, "%s.feature.npy" % dataset))
	data_set_y = numpy.load(os.path.join(input_directory, "%s.label.npy" % dataset))
	assert len(data_set_x) == len(data_set_y)
	logger.info("successfully load data %s with %d to %s..." % (input_directory, len(data_set_x), dataset))
	return (data_set_x, data_set_y)


def load_and_split_data(input_directory, number_of_validate_data=0):
	data_x = numpy.load(os.path.join(input_directory, "train.feature.npy"))
	data_y = numpy.load(os.path.join(input_directory, "train.label.npy"))
	assert len(data_x) == len(data_y)

	assert number_of_validate_data >= 0 and number_of_validate_data < len(data_y)
	indices = numpy.random.permutation(len(data_y))
	train_indices = indices[number_of_validate_data:]
	validate_indices = indices[:number_of_validate_data]

	train_set_x = data_x[train_indices]
	train_set_y = data_y[train_indices]
	train_dataset = (train_set_x, train_set_y)
	# numpy.save(os.path.join(output_directory, "train.index.npy"), train_indices)
	logger.info("successfully load data %s with %d to train..." % (input_directory, len(train_set_x)))

	if len(validate_indices) > 0:
		validate_set_x = data_x[validate_indices]
		validate_set_y = data_y[validate_indices]
		validate_dataset = (validate_set_x, validate_set_y)
		# numpy.save(os.path.join(output_directory, "validate.index.npy"), validate_indices)
		logger.info("successfully load data %s with %d to validate..." % (input_directory, len(validate_set_x)))
	else:
		validate_dataset = None

	return (train_dataset, train_indices), (validate_dataset, validate_indices)


def split_train_data_to_cross_validate(input_directory, number_of_folds=5, output_directory=None):
	data_x = numpy.load(os.path.join(input_directory, "train.feature.npy"))
	data_y = numpy.load(os.path.join(input_directory, "train.label.npy"))
	assert len(data_x) == len(data_y)
	number_of_data = len(data_y)

	assert number_of_folds >= 0 and number_of_folds < len(data_y)
	split_indices = list(range(0, number_of_data, number_of_data / number_of_folds))
	if len(split_indices) == number_of_folds + 1:
		split_indices[-1] = number_of_data
	elif len(split_indices) == number_of_folds:
		split_indices.append(number_of_data)
	else:
		logger.error("something went wrong...")
		sys.exit()
	assert len(split_indices) == number_of_folds + 1
	indices = list(range(len(data_y)))
	random.shuffle(indices)

	if output_directory is None:
		output_directory = input_directory

	fold_index = 0
	for start_index, end_index in zip(split_indices[:-1], split_indices[1:]):
		fold_output_directory = os.path.join(output_directory, "folds.%d.index.%d" % (number_of_folds, fold_index))
		if not os.path.exists(fold_output_directory):
			os.mkdir(fold_output_directory)

		train_indices = indices[:start_index] + indices[end_index:]
		test_indices = indices[start_index:end_index]
		assert len(train_indices) > 0
		assert len(test_indices) > 0

		train_set_x = data_x[train_indices]
		train_set_y = data_y[train_indices]
		numpy.save(os.path.join(fold_output_directory, "train.feature.npy"), train_set_x)
		numpy.save(os.path.join(fold_output_directory, "train.label.npy"), train_set_y)
		numpy.save(os.path.join(fold_output_directory, "train.index.npy"), train_indices)

		test_set_x = data_x[test_indices]
		test_set_y = data_y[test_indices]
		numpy.save(os.path.join(fold_output_directory, "test.feature.npy"), test_set_x)
		numpy.save(os.path.join(fold_output_directory, "test.label.npy"), test_set_y)
		numpy.save(os.path.join(fold_output_directory, "test.index.npy"), test_indices)

		logger.info(
			"successfully split data to %d for train and %d for test..." % (len(train_indices), len(test_indices)))
		logger.info("successfully generate fold %d to %s..." % (fold_index, fold_output_directory))

		print("successfully split data to %d for train and %d for test..." % (len(train_indices), len(test_indices)))
		print("successfully generate fold %d to %s..." % (fold_index, fold_output_directory))

		fold_index += 1

	return


def config_model(construct_parser_function, validate_arguments_function):
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	arguments, additionals = construct_parser_function().parse_known_args()
	# arguments, additionals = model_parser.parse_known_args()

	settings = validate_arguments_function(arguments)

	if len(additionals) > 0:
		print("========== ==========", "additional", "========== ==========")
		for addition in additionals:
			print("%s" % (addition))
		print("========== ==========", "additional", "========== ==========")

	return settings


def validate_config(settings):
	test_dataset = load_data(settings.input_directory, dataset="test")
	test_set_x, test_set_y = test_dataset
	input_shape = list(test_set_x.shape[1:])
	input_shape.insert(0, None)
	input_shape = tuple(input_shape)
	settings.input_shape = input_shape

	return settings


def train_model(network, settings, dataset_preprocessing_function=None):
	input_directory = settings.input_directory
	output_directory = settings.output_directory
	assert not os.path.exists(output_directory)
	os.mkdir(output_directory)

	logging.basicConfig(filename=os.path.join(settings.output_directory, "model.log"), level=logging.DEBUG,
	                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

	validation_data = settings.validation_data
	test_dataset = load_data(input_directory, dataset="test")
	if validation_data >= 0:
		train_dataset_info, validate_dataset_info = load_and_split_data(input_directory, validation_data)
		train_dataset, train_indices = train_dataset_info
		validate_dataset, validate_indices = validate_dataset_info
		numpy.save(os.path.join(output_directory, "train.index.npy"), train_indices)
		numpy.save(os.path.join(output_directory, "validate.index.npy"), validate_indices)
	else:
		train_dataset = load_data(input_directory, dataset="train")
		validate_dataset = load_data(input_directory, dataset="validate")

	if debugger.subsample_dataset in settings.debug:
		train_dataset, validate_dataset, test_dataset = debugger.subsample_dataset(train_dataset, validate_dataset,
		                                                                           test_dataset)

	if dataset_preprocessing_function is not None:
		train_dataset = dataset_preprocessing_function(train_dataset)
		validate_dataset = dataset_preprocessing_function(validate_dataset)
		test_dataset = dataset_preprocessing_function(test_dataset)

	print("========== ==========", "parameters", "========== ==========")
	for key, value in list(vars(settings).items()):
		print("%s=%s" % (key, value))
	print("========== ==========", "parameters", "========== ==========")

	logger.info("========== ==========" + "parameters" + "========== ==========")
	for key, value in list(vars(settings).items()):
		logger.info("%s=%s" % (key, value))
	logger.info("========== ==========" + "parameters" + "========== ==========")

	# pickle.dump(settings, open(os.path.join(output_directory, "settings.pkl"), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

	########################
	# START MODEL TRAINING #
	########################

	if debugger.display_architecture in settings.debug:
		debugger.display_architecture(network)
	for snapshot_function in settings.snapshot:
		snapshot_function(network, settings)

	start_train = timeit.default_timer()
	# Finally, launch the training loop.
	# We iterate over epochs:
	for epoch_index in range(settings.number_of_epochs):
		if debugger.debug_function_output in settings.debug:
			debugger.debug_function_output(network, train_dataset)
		if debugger.debug_rademacher_p_inf_q_1 in settings.debug:
			debugger.debug_rademacher_p_inf_q_1(network, train_dataset)

		network.train(train_dataset, settings.minibatch_size, validate_dataset, test_dataset, output_directory)
		network.epoch_index += 1

		# if settings.snapshot_interval > 0 and network.epoch_index % settings.snapshot_interval == 0:
		# model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
		# pickle.dump(network, open(model_file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

		for snapshot_function in settings.snapshot:
			if network.epoch_index % settings.snapshot[snapshot_function] == 0:
				snapshot_function(network, settings)

		print("PROGRESS: %f%%" % (100. * (epoch_index + 1) / settings.number_of_epochs))

	# model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
	# pickle.dump(network, open(model_file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

	end_train = timeit.default_timer()

	print("Optimization complete...")
	logger.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
		network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index))
	print('The code for file %s ran for %.2fm' % (os.path.split(__file__)[1], (end_train - start_train) / 60.))


#
#
#
#
#

def load_mnist():
	# We first define a download function, supporting both Python 2 and 3.
	if sys.version_info[0] == 2:
		from urllib import urlretrieve
	else:
		from urllib.request import urlretrieve

	def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
		print("Downloading %s" % filename)
		urlretrieve(source + filename, filename)

	# We then define functions for loading MNIST images and labels.
	# For convenience, they also download the requested files if needed.
	import gzip

	def load_mnist_images(filename):
		if not os.path.exists(filename):
			download(filename)
		# Read the inputs in Yann LeCun's binary format.
		with gzip.open(filename, 'rb') as f:
			data = numpy.frombuffer(f.read(), numpy.uint8, offset=16)
		# The inputs are vectors now, we reshape them to monochrome 2D images,
		# following the shape convention: (examples, channels, rows, columns)
		data = data.reshape(-1, 1, 28, 28)
		# The inputs come as bytes, we convert them to float32 in range [0,1].
		# (Actually to range [0, 255/256], for compatibility to the version
		# provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
		return data / numpy.float32(256)

	def load_mnist_labels(filename):
		if not os.path.exists(filename):
			download(filename)
		# Read the labels in Yann LeCun's binary format.
		with gzip.open(filename, 'rb') as f:
			data = numpy.frombuffer(f.read(), numpy.uint8, offset=8)
		# The labels are vectors of integers now, that's exactly what we want.
		return data

	# We can now download and read the training and test set images and labels.
	X_train = load_mnist_images('train-images-idx3-ubyte.gz')
	y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
	X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
	y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

	# We reserve the last 10000 training examples for validation.
	X_train, X_val = X_train[:-10000], X_train[-10000:]
	y_train, y_val = y_train[:-10000], y_train[-10000:]

	# We just return all the arrays in order, as expected in main().
	# (It doesn't matter how we do this as long as we can read them again.)
	return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
	input_directory = sys.argv[1]
	number_of_folds = int(sys.argv[2])
	split_train_data_to_cross_validate(input_directory, number_of_folds)
