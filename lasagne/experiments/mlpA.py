import logging

from .. import networks

logger = logging.getLogger(__name__)

__all__ = [
	"train_mlpA",
]


def construct_mlpA_parser():
	from .mlp import construct_mlp_parser
	model_parser = construct_mlp_parser()

	model_parser.description = "adaptive multi-layer perceptron argument"

	# model argument set 1
	model_parser.add_argument("--adaptable_learning_rate", dest="adaptable_learning_rate", action='store',
	                          default=None, help="adaptable learning rate [None - learning_rate]")
	model_parser.add_argument("--train_adaptables_mode", dest="train_adaptables_mode",
	                          action='store', default="train_adaptables_networkwise", help="train adaptables mode [train_adaptables_networkwise]")
	#model_parser.add_argument("--adaptable_update_interval", dest="adaptable_update_interval", type=int,
	                          #action='store', default=1, help="adatable update interval [1]")

	return model_parser


def validate_mlpA_arguments(arguments):
	from .mlp import validate_mlp_arguments
	arguments = validate_mlp_arguments(arguments)

	'''
	from . import validate_discriminative_arguments, validate_dense_arguments, validate_dropout_parameter_arguments
	arguments = validate_discriminative_arguments(arguments)

	arguments = validate_dense_arguments(arguments)
	number_of_layers = len(arguments.dense_dimensions)
	arguments = validate_dropout_parameter_arguments(arguments, number_of_layers)
	'''

	# model argument set 1
	from . import parse_parameter_policy
	#arguments.adaptable_learning_rate = parse_parameter_policy(arguments.adaptable_learning_rate)
	if arguments.adaptable_learning_rate is None:
		arguments.adaptable_learning_rate = arguments.learning_rate
	else:
		arguments.adaptable_learning_rate = parse_parameter_policy(arguments.adaptable_learning_rate)

	assert arguments.train_adaptables_mode in {"train_adaptables_networkwise", "train_adaptables_layerwise", "train_adaptables_layerwise_in_turn"}

	#assert (arguments.adaptable_update_interval >= 0)

	return arguments


def train_mlpA():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	from . import config_model, validate_config
	settings = config_model(construct_mlpA_parser, validate_mlpA_arguments)
	settings = validate_config(settings)

	network = networks.AdaptiveMultiLayerPerceptron(
		incoming=settings.input_shape,

		dense_dimensions=settings.dense_dimensions,
		dense_nonlinearities=settings.dense_nonlinearities,

		layer_activation_types=settings.layer_activation_types,
		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles,

		objective_functions=settings.objective,
		update_function=settings.update,
		# pretrained_model=pretrained_model

		learning_rate_policy=settings.learning_rate,

		adaptable_learning_rate_policy=settings.adaptable_learning_rate,
		#adaptable_update_interval=settings.adaptable_update_interval,
		train_adaptables_mode=settings.train_adaptables_mode,

		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)

	network.set_regularizers(settings.regularizer)

	from . import train_model
	train_model(network, settings)


if __name__ == '__main__':
	train_mlpA()
