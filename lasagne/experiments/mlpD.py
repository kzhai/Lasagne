import logging

from . import layer_deliminator, param_deliminator
from .. import networks, layers

logger = logging.getLogger(__name__)

__all__ = [
	"train_mlpD",
]


def construct_mlpD_parser():
	from .mlpA import construct_mlpA_parser
	model_parser = construct_mlpA_parser()

	'''
	from . import construct_discriminative_parser, add_dense_options, add_dropout_parameter_options
	model_parser = construct_discriminative_parser()
	model_parser = add_dense_options(model_parser)
	model_parser = add_dropout_parameter_options(model_parser)
	'''

	model_parser.description = "dynamic multi-layer perceptron argument"

	# model argument set 1
	model_parser.add_argument("--prune_thresholds", dest="prune_thresholds", action='store', default=None,
	                          # default="-1e3,piecewise_constant,100,1e-3",
	                          # help="prune thresholds [-1e3,piecewise_constant,100,1e-3]"
	                          help="prune thresholds [None]"
	                          )
	model_parser.add_argument("--split_thresholds", dest="split_thresholds", action='store', default=None,
	                          # default="1e3,piecewise_constant,100,0.999",
	                          # help="split thresholds [1e3,piecewise_constant,100,0.999]"
	                          help="split thresholds [None]"
	                          )

	# model_parser.add_argument("--prune_split_interval", dest="prune_split_interval", action='store', default=1,
	# type=int, help="prune split interval [1]")
	model_parser.add_argument("--prune_split_interval", dest="prune_split_interval", action='store', default="1",
	                          help="prune split interval [1]")

	return model_parser


def validate_mlpD_arguments(arguments):
	from .mlpA import validate_mlpA_arguments
	arguments = validate_mlpA_arguments(arguments)

	'''
	from . import validate_discriminative_arguments, validate_dense_arguments, validate_dropout_parameter_arguments
	arguments = validate_discriminative_arguments(arguments)

	arguments = validate_dense_arguments(arguments)
	number_of_layers = len(arguments.dense_dimensions)
	arguments = validate_dropout_parameter_arguments(arguments, number_of_layers)
	'''

	# model argument set 1
	# arguments.adaptable_learning_rate = parse_parameter_policy(arguments.adaptable_learning_rate)
	from . import parse_parameter_policy
	number_of_layers = sum(layer_activation_type is layers.DynamicDropoutLayer for layer_activation_type in
	                       arguments.layer_activation_types)

	if arguments.prune_thresholds is not None:
		prune_thresholds = arguments.prune_thresholds
		prune_thresholds_tokens = prune_thresholds.split(layer_deliminator)
		prune_thresholds = [parse_parameter_policy(prune_thresholds_token) for prune_thresholds_token in
		                    prune_thresholds_tokens]
		if len(prune_thresholds) == 1:
			prune_thresholds *= number_of_layers
		assert len(prune_thresholds) == number_of_layers
		arguments.prune_thresholds = prune_thresholds

	if arguments.split_thresholds is not None:
		split_thresholds = arguments.split_thresholds
		split_thresholds_tokens = split_thresholds.split(layer_deliminator)
		split_thresholds = [parse_parameter_policy(split_thresholds_token) for split_thresholds_token in
		                    split_thresholds_tokens]
		if len(split_thresholds) == 1:
			split_thresholds *= number_of_layers
		assert len(split_thresholds) == number_of_layers
		arguments.split_thresholds = split_thresholds

	prune_split_interval = arguments.prune_split_interval
	prune_split_interval_tokens = [int(prune_split_interval_token) for prune_split_interval_token in
	                               prune_split_interval.split(param_deliminator)]
	if len(prune_split_interval_tokens) == 1:
		prune_split_interval_tokens.insert(0, 0)
	assert prune_split_interval_tokens[1] >= 0
	arguments.prune_split_interval = prune_split_interval_tokens

	return arguments


def train_mlpD():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	from . import config_model, validate_config
	settings = config_model(construct_mlpD_parser, validate_mlpD_arguments)
	settings = validate_config(settings)

	network = networks.DynamicMultiLayerPerceptron(
		incoming=settings.input_shape,

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
		adaptable_training_mode=settings.adaptable_training_mode,

		prune_threshold_policies=settings.prune_thresholds,
		split_threshold_policies=settings.split_thresholds,
		prune_split_interval=settings.prune_split_interval,

		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)

	network.set_regularizers(settings.regularizer)

	from . import train_model
	train_model(network, settings)


if __name__ == '__main__':
	train_mlpD()
