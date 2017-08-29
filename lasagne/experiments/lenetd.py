import logging
import os

import numpy

from .. import layers, networks
from . import param_deliminator

logger = logging.getLogger(__name__)

__all__ = [
	"train_dlenet",
]


def construct_dlenet_parser():
	from .lenet import construct_lenet_parser
	model_parser = construct_lenet_parser()

	model_parser.description = "convolutional dynamic le net argument"

	# model argument set 1
	'''
	model_parser.add_argument("--dropout_learning_rate", dest="dropout_learning_rate", type=float, action='store',
	                          default=0, help="dropout learning rate [0 = learning_rate]")
	model_parser.add_argument("--dropout_learning_rate_decay", dest="dropout_learning_rate_decay", action='store',
	                          default=None, help="dropout learning rate decay [None = learning_rate_decay]")
	'''
	model_parser.add_argument("--dropout_learning_rate", dest="dropout_learning_rate", action='store',
	                          default=None, help="dropout learning rate [None = learning_rate]")
	model_parser.add_argument("--dropout_rate_update_interval", dest="dropout_rate_update_interval", type=int,
	                          action='store', default=0, help="dropout rate update interval [1]")
	#model_parser.add_argument('--update_hidden_layer_dropout_only', dest="update_hidden_layer_dropout_only",
	                          #action='store_true', default=False, help="update hidden layer dropout only [False]")

	return model_parser


def validate_dlenet_arguments(arguments):
	from .lenet import validate_lenet_arguments
	arguments = validate_lenet_arguments(arguments)

	'''
	from . import validate_discriminative_arguments, validate_dense_arguments, validate_convpool_arguments
	from .mlpd import validate_dropout_arguments

	arguments = validate_discriminative_arguments(arguments)

	arguments = validate_convpool_arguments(arguments)
	number_of_convolution_layers = len(arguments.convolution_filters)

	arguments = validate_dense_arguments(arguments)
	number_of_dense_layers = len(arguments.dense_dimensions)

	number_of_layers = number_of_convolution_layers + number_of_dense_layers
	arguments = validate_dropout_arguments(arguments, number_of_layers)
	'''

	# model argument set 1
	from . import validate_decay_policy
	if arguments.dropout_learning_rate is None:
		arguments.dropout_learning_rate = arguments.learning_rate;
	else:
		dropout_learning_rate_tokens = arguments.dropout_learning_rate.split(param_deliminator)
		arguments.dropout_learning_rate = validate_decay_policy(dropout_learning_rate_tokens)

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

		layer_activation_types=settings.layer_activation_types,
		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles,

		objective_functions=settings.objective,
		update_function=settings.update,

		learning_rate_policy=settings.learning_rate,
		#learning_rate_decay=settings.learning_rate_decay,

		dropout_learning_rate_policy=settings.dropout_learning_rate,
		#dropout_learning_rate_decay=settings.dropout_learning_rate_decay,
		dropout_rate_update_interval=settings.dropout_rate_update_interval,
		#update_hidden_layer_dropout_only=settings.update_hidden_layer_dropout_only,

		max_norm_constraint=settings.max_norm_constraint,
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


if __name__ == '__main__':
	train_dlenet()
