import logging
import os

import numpy

from .. import layers
from .. import networks

logger = logging.getLogger(__name__)

__all__ = [
	"train_dlenet",
]


def construct_dlenet_parser():
	from .lenet import construct_lenet_parser
	model_parser = construct_lenet_parser()

	model_parser.description = "convolutional dynamic le net argument"

	# model argument set
	model_parser.add_argument("--dropout_rate_update_interval", dest="dropout_rate_update_interval", type=int,
	                          action='store', default=0,
	                          help="dropout rate update interval [0=no update]")
	model_parser.add_argument('--update_hidden_layer_dropout_only', dest="update_hidden_layer_dropout_only",
	                          action='store_true', default=False,
	                          help="update hidden layer dropout only [False]")

	return model_parser


def validate_dlenet_arguments(arguments):
	from .lenet import validate_lenet_arguments
	arguments = validate_lenet_arguments(arguments)

	# model argument set
	assert (arguments.dropout_rate_update_interval >= 0)

	return arguments


def train_dlenet():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	from . import config_model, validate_config
	settings = config_model(construct_dlenet_parser, validate_dlenet_arguments)
	settings = validate_config(settings)

	network = networks.DynamicLeNet(
		incoming=settings.input_shape,

		convolution_filters=settings.convolution_filters,
		convolution_nonlinearities=settings.convolution_nonlinearities,
		# convolution_filter_sizes=None,
		# maxpooling_sizes=None,
		pool_modes=settings.pool_modes,

		# local_convolution_filters=settings.local_convolution_filters,

		dense_dimensions=settings.dense_dimensions,
		dense_nonlinearities=settings.dense_nonlinearities,

		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles,

		objective_functions=settings.objective,
		update_function=settings.update,

		learning_rate=settings.learning_rate,
		learning_rate_decay=settings.learning_rate_decay,
		max_norm_constraint=settings.max_norm_constraint,
		# learning_rate_decay_style=settings.learning_rate_decay_style,
		# learning_rate_decay_parameter=settings.learning_rate_decay_parameter,

		dropout_rate_update_interval=settings.dropout_rate_update_interval,
		update_hidden_layer_dropout_only=settings.update_hidden_layer_dropout_only,

		validation_interval=settings.validation_interval,
	)

	'''
	convolution_filter_sizes = settings.convolution_filter_sizes,
	convolution_strides = settings.convolution_strides,
	convolution_pads = settings.convolution_pads,

	local_convolution_filter_sizes = settings.local_convolution_filter_sizes,
	local_convolution_strides = settings.local_convolution_strides,
	local_convolution_pads = settings.local_convolution_pads,

	pooling_sizes = settings.pooling_sizes,
	pooling_strides = settings.pooling_strides,
	'''

	network.set_regularizers(settings.regularizer)

	from . import train_model
	train_model(network, settings)

	'''
	########################
	# START MODEL TRAINING #
	########################

	start_train = timeit.default_timer()
	# Finally, launch the training loop.
	# We iterate over epochs:

	if settings.debug:
		snapshot_retain_rates(network, output_directory)

	for epoch_index in range(settings.number_of_epochs):
		network.train(train_dataset, minibatch_size, validate_dataset, test_dataset, output_directory)

		if settings.snapshot_interval>0 and network.epoch_index % settings.snapshot_interval == 0:
			model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
			#cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

		print "PROGRESS: %f%%" % (100. * epoch_index / settings.number_of_epochs)

		if settings.debug:
			snapshot_retain_rates(network, output_directory)

	model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
	#cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

	end_train = timeit.default_timer()

	print "Optimization complete..."
	logger.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
		network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index))
	print >> sys.stderr, ('The code for file %s ran for %.2fm' % (os.path.split(__file__)[1], (end_train - start_train) / 60.))
	'''


def snapshot_retain_rates(network, output_directory):
	dropout_layer_index = 0
	for network_layer in network.get_network_layers():
		if not isinstance(network_layer, layers.AdaptiveDropoutLayer):
			continue

		layer_retain_probability = network_layer.activation_probability.eval()
		logger.info("retain rates stats: epoch %i, shape %s, average %f, minimum %f, maximum %f" % (
			network.epoch_index,
			layer_retain_probability.shape,
			numpy.mean(layer_retain_probability),
			numpy.min(layer_retain_probability),
			numpy.max(layer_retain_probability)))

		retain_rate_file = os.path.join(output_directory,
		                                "layer.%d.epoch.%d.npy" % (dropout_layer_index, network.epoch_index))
		numpy.save(retain_rate_file, layer_retain_probability)
		dropout_layer_index += 1


if __name__ == '__main__':
	train_dlenet()
