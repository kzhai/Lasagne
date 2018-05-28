import logging
import pickle

logger = logging.getLogger(__name__)

from . import param_deliminator

from lasagne import nonlinearities, networks
from . import layer_deliminator, validate_config
from . import start_training, resume_training

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
	                          help="pool layer modes [max], set to none to omit, example, 'max*max*none*none*max'")
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


def start_lenet(settings):
	settings = validate_config(settings)

	network = networks.FeedForwardNetwork(
		incoming=settings.input_shape,
		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,
		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)

	lenet = networks.LeNetFromSpecifications(
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

	start_training(network, settings)


def resume_lenet(settings):
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

	resume_training(network, settings)


def start_lenetA(settings):
	# settings = config_model(mlpA_parser, mlpA_validator)
	settings = validate_config(settings)

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

	lenet = networks.AdaptiveLeNetFromSpecifications(
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

	start_training(network, settings)


def resume_lenetA(settings):
	# settings = config_model(discriminative_adaptive_resume_parser, discriminative_adaptive_resume_validator)
	settings = validate_config(settings)

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

	model = pickle.load(open(settings.model_file, 'rb'))
	mlp = networks.AdaptiveLeNetFromPretrainedModel(
		network._input_layer,
		pretrained_network=model
	)
	network.set_network(mlp)
	network.set_regularizers(settings.regularizer)

	resume_training(network, settings)


def main():
	import argparse
	from . import add_discriminative_options, add_resume_options, add_adaptive_options, add_dense_options, \
		add_dropout_options
	from . import validate_discriminative_options, validate_resume_options, validate_adaptive_options, \
		validate_dense_arguments, validate_dropout_arguments

	model_selector = argparse.ArgumentParser(description="mode selector")

	subparsers = model_selector.add_subparsers(dest="run_model", help='model help')

	start_lenet_parser = subparsers.add_parser('start-lenet', help='start lenet model')
	start_lenet_parser = add_discriminative_options(start_lenet_parser)
	start_lenet_parser = add_convpool_options(start_lenet_parser)
	start_lenet_parser = add_dense_options(start_lenet_parser)
	start_lenet_parser = add_dropout_options(start_lenet_parser)

	resume_lenet_parser = subparsers.add_parser('resume-lenet', help='resume lenet model')
	resume_lenet_parser = add_discriminative_options(resume_lenet_parser)
	resume_lenet_parser = add_resume_options(resume_lenet_parser)

	start_lenetA_parser = subparsers.add_parser('start-lenetA', help='start adaptive lenet model')
	start_lenetA_parser = add_discriminative_options(start_lenetA_parser)
	start_lenetA_parser = add_convpool_options(start_lenetA_parser)
	start_lenetA_parser = add_dense_options(start_lenetA_parser)
	start_lenetA_parser = add_dropout_options(start_lenetA_parser)
	start_lenetA_parser = add_adaptive_options(start_lenetA_parser)

	resume_lenetA_parser = subparsers.add_parser('resume-lenetA', help='resume adaptive lenet model')
	resume_lenetA_parser = add_discriminative_options(resume_lenetA_parser)
	resume_lenetA_parser = add_resume_options(resume_lenetA_parser)
	resume_lenetA_parser = add_adaptive_options(resume_lenetA_parser)

	arguments, additionals = model_selector.parse_known_args()

	if len(additionals) > 0:
		print("========== ==========", "additionals", "========== ==========")
		for addition in additionals:
			print("%s" % (addition))
	# print("========== ==========", "additionals", "========== ==========")

	if arguments.run_model == "start-lenet":
		arguments = validate_discriminative_options(arguments)
		arguments = validate_convpool_arguments(arguments)
		arguments = validate_dense_arguments(arguments)
		number_of_layers = len(arguments.dense_dimensions) + len(arguments.convolution_filters)
		arguments = validate_dropout_arguments(arguments, number_of_layers)

		start_lenet(arguments)
	elif arguments.run_model == "resume-lenet":
		arguments = validate_discriminative_options(arguments)
		arguments = validate_resume_options(arguments)

		resume_lenet(arguments)
	elif arguments.run_model == "start-lenetA":
		arguments = validate_discriminative_options(arguments)
		arguments = validate_convpool_arguments(arguments)
		arguments = validate_dense_arguments(arguments)
		number_of_layers = len(arguments.dense_dimensions) + len(arguments.convolution_filters)
		arguments = validate_dropout_arguments(arguments, number_of_layers)
		arguments = validate_adaptive_options(arguments)

		start_lenetA(arguments)
	elif arguments.run_model == "resume-lenetA":
		arguments = validate_discriminative_options(arguments)
		arguments = validate_resume_options(arguments)
		arguments = validate_adaptive_options(arguments)

		resume_lenetA(arguments)


if __name__ == '__main__':
	main()
