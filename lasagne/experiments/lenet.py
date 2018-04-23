import logging
import pickle

from . import layer_deliminator, param_deliminator
from .. import networks, nonlinearities

logger = logging.getLogger(__name__)

__all__ = [
	"add_convpool_options",
	"validate_convpool_arguments",
	#
	"start_lenet",
]


def add_convpool_options(model_parser):
	# model argument set 1
	model_parser.add_argument("--convolution_filters", dest="convolution_filters", action='store', default=None,
	                          help="convolutional layer filter dimensions [None]")
	model_parser.add_argument("--convolution_nonlinearities", dest="convolution_nonlinearities", action='store',
	                          default=None,
	                          help="convolutional layer activation functions [None]")

	# model argument set 4
	model_parser.add_argument("--convolution_kernel_sizes", dest="convolution_kernel_sizes", action='store',
	                          default="5,5",
	                          help="convolution kernel sizes [5,5], example, '5*5,6*6' represents 5*5 and 6*6 kernel size for convolution layers respectively")
	model_parser.add_argument("--convolution_strides", dest="convolution_strides", action='store', default="1,1",
	                          help="convolution strides [1,1], example, '1*1,2*2' represents 1*1 and 2*2 stride size for convolution layers respectively")
	model_parser.add_argument("--convolution_pads", dest="convolution_pads", action='store', default="2",
	                          help="convolution pads [2], example, '2,3' represents 2 and 3 pads for convolution layers respectively")

	'''
	# model argument set 5
	model_parser.add_argument("--locally_convolution_filter_sizes", dest="locally_convolution_filter_sizes", action='store', default="3*3",
							  help="locally convolution filter sizes [3*3], example, '5*5,6*6' represents 5*5 and 6*6 filter size for locally connected convolution layers respectively")
	model_parser.add_argument("--locally_convolution_strides", dest="locally_convolution_strides", action='store', default="1*1",
							  help="locally convolution strides [1*1], example, '1*1,2*2' represents 1*1 and 2*2 stride size for locally connected convolution layers respectively")
	model_parser.add_argument("--locally_convolution_pads", dest="locally_convolution_pads", action='store', default="1",
							  help="locally convolution pads [1], example, '2,3' represents 2 and 3 pads for locally connected convolution layers respectively")
	'''

	# model argument set 6
	model_parser.add_argument("--pool_modes", dest="pool_modes", action='store', default="max",
	                          help="pool layer modes [max], set to none to omit, example, 'max;max;none;none;max'")
	model_parser.add_argument("--pool_kernel_sizes", dest="pool_kernel_sizes", action='store', default="2,2",
	                          help="pool kernel sizes [3,3], example, '2*2,3*3' represents 2*2 and 3*3 pooling size respectively")
	model_parser.add_argument("--pool_strides", dest="pool_strides", action='store', default="2,2",
	                          help="pool strides [2,2], example, '2*2,3*3' represents 2*2 and 3*3 pooling stride respectively")

	return model_parser


def validate_convpool_arguments(arguments):
	# model argument set 1
	assert arguments.convolution_filters is not None
	conv_filters = arguments.convolution_filters.split(layer_deliminator)
	arguments.convolution_filters = [int(conv_filter) for conv_filter in conv_filters]

	assert arguments.convolution_nonlinearities is not None
	conv_nonlinearities = arguments.convolution_nonlinearities.split(layer_deliminator)
	arguments.convolution_nonlinearities = [getattr(nonlinearities, conv_nonlinearity) for conv_nonlinearity in
	                                        conv_nonlinearities]
	assert len(arguments.convolution_filters) == len(arguments.convolution_nonlinearities)

	assert arguments.convolution_kernel_sizes is not None
	conv_kernel_sizes = arguments.convolution_kernel_sizes.split(param_deliminator)
	arguments.convolution_kernel_sizes = tuple([int(conv_kernel_size) for conv_kernel_size in conv_kernel_sizes])

	assert arguments.convolution_strides is not None
	conv_strides = arguments.convolution_strides.split(param_deliminator)
	arguments.convolution_strides = tuple([int(conv_stride) for conv_stride in conv_strides])

	assert arguments.convolution_pads is not None
	# conv_pads = arguments.convolution_pads.split(param_deliminator)
	arguments.convolution_pads = int(arguments.convolution_pads)

	assert arguments.pool_modes is not None
	pool_modes = arguments.pool_modes.split(layer_deliminator)
	if len(pool_modes) == 1:
		pool_modes *= len(arguments.convolution_filters)
	for pool_mode_index in range(len(pool_modes)):
		if pool_modes[pool_mode_index].lower() == "none":
			pool_modes[pool_mode_index] = None
	arguments.pool_modes = pool_modes
	assert len(arguments.convolution_filters) == len(arguments.pool_modes)

	assert arguments.pool_kernel_sizes is not None
	pool_kernel_sizes = arguments.pool_kernel_sizes.split(param_deliminator)
	arguments.pool_kernel_sizes = tuple([int(pool_kernel_size) for pool_kernel_size in pool_kernel_sizes])

	assert arguments.pool_strides is not None
	pool_strides = arguments.pool_strides.split(param_deliminator)
	arguments.pool_strides = tuple([int(pool_stride) for pool_stride in pool_strides])

	return arguments


def lenet_parser():
	from . import add_discriminative_options, add_dense_options, add_dropout_options

	model_parser = add_discriminative_options()
	model_parser = add_convpool_options(model_parser)
	model_parser = add_dense_options(model_parser)
	model_parser = add_dropout_options(model_parser)

	return model_parser


def lenet_validator(arguments):
	from . import validate_discriminative_options, validate_dense_arguments, validate_dropout_arguments

	arguments = validate_discriminative_options(arguments)

	arguments = validate_convpool_arguments(arguments)
	number_of_convolution_layers = len(arguments.convolution_filters)

	arguments = validate_dense_arguments(arguments)
	number_of_dense_layers = len(arguments.dense_dimensions)

	number_of_layers = number_of_convolution_layers + number_of_dense_layers
	arguments = validate_dropout_arguments(arguments, number_of_layers)

	return arguments


def start_lenet():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	from . import config_model, validate_config
	settings = config_model(lenet_parser, lenet_validator)
	settings = validate_config(settings)

	'''
	network = networks.LeNet(
		incoming=settings.input_shape,

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

		objective_functions=settings.objective,
		update_function=settings.update,

		learning_rate_policy=settings.learning_rate,
		# learning_rate_decay=settings.learning_rate_decay,
		max_norm_constraint=settings.max_norm_constraint,
		# learning_rate_decay_style=settings.learning_rate_decay_style,
		# learning_rate_decay_parameter=settings.learning_rate_decay_parameter,

		validation_interval=settings.validation_interval,

		conv_kernel_sizes=settings.convolution_kernel_sizes,
		conv_strides=settings.convolution_strides,
		conv_pads=settings.convolution_pads,

		pool_kernel_sizes=settings.pool_kernel_sizes,
		pool_strides=settings.pool_strides,
	)
	'''

	network = networks.FeedForwardNetwork(
		incoming=settings.input_shape,
		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,
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


def resume_lenet():
	from . import config_model, validate_config, add_resume_options, validate_resume_options

	settings = config_model(add_resume_options, validate_resume_options)
	settings = validate_config(settings)

	network = networks.FeedForwardNetwork(
		incoming=settings.input_shape,
		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,
		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)

	model = pickle.load(open(settings.model_file, 'rb'))
	lenet = networks.LeNetFromPretrainedModel(
		network._input_layer,
		pretrained_network=model
	)
	network.set_network(lenet)
	network.set_regularizers(settings.regularizer)

	from . import resume_training
	resume_training(network, settings)


if __name__ == '__main__':
	start_lenet()
