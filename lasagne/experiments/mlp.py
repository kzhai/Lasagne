import cPickle
import logging
import os
import sys
import timeit

import numpy

from .base import load_data, load_and_split_data
from .. import networks

__all__ = [
	"train_mlp",
]


def construct_mlp_parser():
	from .base import construct_discriminative_parser, add_dense_options, add_dropout_options
	model_parser = construct_discriminative_parser();
	model_parser = add_dense_options(model_parser);
	model_parser = add_dropout_options(model_parser);

	'''
	model_parser.add_argument("--pretrained_model_file", dest="pretrained_model_file",
							  help="pretrained model file [None]");
	model_parser.add_argument("--dae_regularizer_lambdas", dest="dae_regularizer_lambdas", nargs="+", type=float, action='store', default=0,
							  help="dae regularization lambda [0]")
	model_parser.add_argument("--layer_corruption_levels", dest="layer_corruption_levels", nargs="+", type=float, action='store', default=0,
							  help="layer corruption level for pre-training [0], either one number of a list of numbers, example, '0.2' represents 0.2 corruption level for all denoising auto encoders, or '0.2,0.5' represents 0.2 corruption level for first denoising auto encoder layer and 0.5 for second one respectively");
	'''

	return model_parser;


def validate_mlp_arguments(arguments):
	from .base import validate_discriminative_arguments, validate_dense_arguments, validate_dropout_arguments
	arguments = validate_discriminative_arguments(arguments);

	arguments = validate_dense_arguments(arguments);
	number_of_layers = len(arguments.dense_dimensions);
	arguments = validate_dropout_arguments(arguments, number_of_layers);

	'''
	dae_regularizer_lambdas = arguments.dae_regularizer_lambdas
	if isinstance(dae_regularizer_lambdas, int):
		dae_regularizer_lambdas = [dae_regularizer_lambdas] * (number_of_layers - 1)
	assert len(dae_regularizer_lambdas) == number_of_layers - 1;
	assert (dae_regularizer_lambda >= 0 for dae_regularizer_lambda in dae_regularizer_lambdas)
	self.dae_regularizer_lambdas = dae_regularizer_lambdas;

	layer_corruption_levels = arguments.layer_corruption_levels;
	if isinstance(layer_corruption_levels, int):
		layer_corruption_levels = [layer_corruption_levels] * (number_of_layers - 1)
	assert len(layer_corruption_levels) == number_of_layers - 1;
	assert (layer_corruption_level >= 0 for layer_corruption_level in layer_corruption_levels)
	assert (layer_corruption_level <= 1 for layer_corruption_level in layer_corruption_levels)
	self.layer_corruption_levels = layer_corruption_levels;

	pretrained_model_file = arguments.pretrained_model_file;
	pretrained_model = None;
	if pretrained_model_file != None:
		assert os.path.exists(pretrained_model_file)
		pretrained_model = cPickle.load(open(pretrained_model_file, 'rb'));
	'''

	return arguments


def train_mlp():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	arguments, additionals = construct_mlp_parser().parse_known_args();
	# arguments, additionals = model_parser.parse_known_args()

	settings = validate_mlp_arguments(arguments);

	input_directory = settings.input_directory
	output_directory = settings.output_directory
	assert not os.path.exists(output_directory)
	os.mkdir(output_directory);

	logging.basicConfig(filename=os.path.join(output_directory, "model.log"), level=logging.DEBUG,
	                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s');
	validation_data = settings.validation_data
	minibatch_size = settings.minibatch_size

	print "========== ==========", "parameters", "========== =========="
	for key, value in vars(settings).iteritems():
		print "%s=%s" % (key, value);
	print "========== ==========", "additional", "========== =========="
	for addition in additionals:
		print "%s" % (addition);

	logging.info("========== ==========" + "parameters" + "========== ==========")
	for key, value in vars(settings).iteritems():
		logging.info("%s=%s" % (key, value));
	logging.info("========== ==========" + "parameters" + "========== ==========")

	cPickle.dump(settings, open(os.path.join(output_directory, "settings.pkl"), 'wb'),
	             protocol=cPickle.HIGHEST_PROTOCOL);

	#
	#
	#
	#
	#

	test_dataset = load_data(input_directory, dataset="test");
	if validation_data >= 0:
		train_dataset_info, validate_dataset_info = load_and_split_data(input_directory, validation_data);
		train_dataset, train_indices = train_dataset_info;
		validate_dataset, validate_indices = validate_dataset_info;
		numpy.save(os.path.join(output_directory, "train.index.npy"), train_indices);
		numpy.save(os.path.join(output_directory, "validate.index.npy"), validate_indices);
	else:
		train_dataset = load_data(input_directory, dataset="train");
		validate_dataset = load_data(input_directory, dataset="validate");
	train_set_x, train_set_y = train_dataset;
	input_shape = list(train_set_x.shape[1:]);
	input_shape.insert(0, None)
	input_shape = tuple(input_shape)

	#
	#
	#
	#
	#

	network = networks.MultiLayerPerceptron(
		incoming=input_shape,

		layer_dimensions=settings.dense_dimensions,
		layer_nonlinearities=settings.dense_nonlinearities,

		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles,

		objective_functions=settings.objective,
		update_function=settings.update,
		# pretrained_model=pretrained_model

		learning_rate=settings.learning_rate,
		learning_rate_decay=settings.learning_rate_decay,
		max_norm_constraint=settings.max_norm_constraint,
		# learning_rate_decay_style=settings.learning_rate_decay_style,
		# learning_rate_decay_parameter=settings.learning_rate_decay_parameter,
		validation_interval=settings.validation_interval,
	)

	network.set_regularizers(settings.regularizer);
	# network.set_L1_regularizer_lambda(settings.L1_regularizer_lambdas)
	# network.set_L2_regularizer_lambda(settings.L2_regularizer_lambdas)

	########################
	# START MODEL TRAINING #
	########################

	start_train = timeit.default_timer()
	# Finally, launch the training loop.
	# We iterate over epochs:
	for epoch_index in range(settings.number_of_epochs):
		network.train(train_dataset, minibatch_size, validate_dataset, test_dataset, output_directory);

		if settings.snapshot_interval > 0 and network.epoch_index % settings.snapshot_interval == 0:
			model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
			cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

		print "PROGRESS: %f%%" % (100. * epoch_index / settings.number_of_epochs);

	model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
	cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

	end_train = timeit.default_timer()

	print "Optimization complete..."
	logging.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
		network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index));
	print >> sys.stderr, (
		'The code for file %s ran for %.2fm' % (os.path.split(__file__)[1], (end_train - start_train) / 60.))


def resume_mlp():
	pass


def test_mlp():
	pass


if __name__ == '__main__':
	train_mlp()
