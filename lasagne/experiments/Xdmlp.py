import logging

from .. import networks

logger = logging.getLogger(__name__)

__all__ = [
	"train_dmlp",
]


def construct_dmlp_parser():
	from .mlp import construct_mlp_parser
	model_parser = construct_mlp_parser()

	'''
	from . import construct_discriminative_parser, add_dense_options, add_dropout_parameter_options
	model_parser = construct_discriminative_parser()
	model_parser = add_dense_options(model_parser)
	model_parser = add_dropout_parameter_options(model_parser)
	'''

	model_parser.description = "dynamic multi-layer perceptron argument"

	# model argument set 1
	model_parser.add_argument("--dropout_learning_rate", dest="dropout_learning_rate", action='store',
	                          default=None, help="dropout learning rate [None = learning_rate]")
	model_parser.add_argument("--dropout_rate_update_interval", dest="dropout_rate_update_interval", type=int,
	                          action='store', default=0, help="dropout rate update interval [1]")
	# model_parser.add_argument('--update_hidden_layer_dropout_only', dest="update_hidden_layer_dropout_only",
	# action='store_true', default=False, help="update hidden layer dropout only [False]")

	return model_parser


def validate_dmlp_arguments(arguments):
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
	if arguments.dropout_learning_rate is None:
		arguments.dropout_learning_rate = arguments.learning_rate
	else:
		arguments.dropout_learning_rate = parse_parameter_policy(arguments.dropout_learning_rate)

	assert (arguments.dropout_rate_update_interval >= 0)

	return arguments


def train_dmlp():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	from . import config_model, validate_config
	settings = config_model(construct_dmlp_parser, validate_dmlp_arguments)
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
		# pretrained_model=pretrained_model

		learning_rate_policy=settings.learning_rate,
		# learning_rate_decay=settings.learning_rate_decay,

		dropout_learning_rate_policy=settings.dropout_learning_rate,
		# dropout_learning_rate_decay=settings.dropout_learning_rate_decay,
		dropout_rate_update_interval=settings.dropout_rate_update_interval,
		# update_hidden_layer_dropout_only=settings.update_hidden_layer_dropout_only,

		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)

	network.set_regularizers(settings.regularizer)
	# network.set_L1_regularizer_lambda(settings.L1_regularizer_lambdas)
	# network.set_L2_regularizer_lambda(settings.L2_regularizer_lambdas)

	from . import train_model
	train_model(network, settings)


def resume_dmlp():
	pass


def test_dmlp():
	pass


if __name__ == '__main__':
	train_dmlp()
