import logging

from .. import networks

logger = logging.getLogger(__name__)

__all__ = [
	"train_lenetA",
]


def construct_lenetA_parser():
	from .lenet import lenet_parser
	model_parser = lenet_parser()

	model_parser.description = "adaptive convolutional le net argument"

	# model argument set 1
	model_parser.add_argument("--adaptable_learning_rate", dest="adaptable_learning_rate", action='store',
	                          default=None, help="adaptable learning rate [None - learning_rate]")
	model_parser.add_argument("--adaptable_training_mode", dest="adaptable_training_mode", action='store',
	                          default="train_adaptables_networkwise",
	                          help="train adaptables mode [train_adaptables_networkwise]")
	# model_parser.add_argument("--adaptable_update_interval", dest="adaptable_update_interval", type=int,
	# action='store', default=1, help="adatable update interval [1]")

	return model_parser


def validate_lenetA_arguments(arguments):
	from .lenet import lenet_validator
	arguments = lenet_validator(arguments)

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
	from . import parse_parameter_policy
	# arguments.adaptable_learning_rate = parse_parameter_policy(arguments.adaptable_learning_rate)
	if arguments.adaptable_learning_rate is None:
		arguments.adaptable_learning_rate = arguments.learning_rate
	else:
		arguments.adaptable_learning_rate = parse_parameter_policy(arguments.adaptable_learning_rate)

	# assert (arguments.adaptable_update_interval >= 0)
	assert arguments.adaptable_training_mode in {"train_adaptables_networkwise", "train_adaptables_layerwise",
	                                             "train_adaptables_layerwise_in_turn"}

	return arguments


def train_lenetA():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	from . import config_model, validate_config
	settings = config_model(construct_lenetA_parser, validate_lenetA_arguments)
	settings = validate_config(settings)

	'''
	network = networks.AdaptiveLeNet(
		incoming=settings.input_shape,

		conv_filters=settings.convolution_filters,
		conv_nonlinearities=settings.convolution_nonlinearities,
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

		adaptable_learning_rate_policy=settings.adaptable_learning_rate,
		# adaptable_update_interval=settings.adaptable_update_interval,
		adaptable_training_mode=settings.train_adaptables_mode,

		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,

		conv_kernel_sizes=settings.convolution_kernel_sizes,
		conv_strides=settings.convolution_strides,
		conv_pads=settings.convolution_pads,

		pool_kernel_sizes=settings.pool_kernel_sizes,
		pool_strides=settings.pool_strides,
	)
	'''

	network = networks.AdaptiveFeedForwardNetwork(
		incoming=settings.input_shape,
		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,
		adaptable_learning_rate_policy=settings.adaptable_learning_rate,
		# adaptable_update_interval=settings.adaptable_update_interval,
		adaptable_training_mode=settings.adaptable_training_mode,
		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)
	lenet = networks.LeNet(
		network._input_layer,

		conv_filters=settings.convolution_filters,
		conv_nonlinearities=settings.convolution_nonlinearities,
		# convolution_filter_sizes=None,
		# maxpooling_sizes=None,
		pool_modes=settings.pool_modes,

		dense_dimensions=settings.dense_dimensions,
		dense_nonlinearities=settings.dense_nonlinearities,

		layer_activation_types=settings.layer_activation_types,
		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles,

		conv_kernel_sizes=settings.convolution_kernel_sizes,
		conv_strides=settings.convolution_strides,
		conv_pads=settings.convolution_pads,

		pool_kernel_sizes=settings.pool_kernel_sizes,
		pool_strides=settings.pool_strides,
	)
	network.set_network(lenet)

	network.set_regularizers(settings.regularizer)

	from . import start_training
	start_training(network, settings)


if __name__ == '__main__':
	train_lenetA()
