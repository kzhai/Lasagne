import logging

from . import param_deliminator, parse_parameter_policy
from .. import networks

logger = logging.getLogger(__name__)

__all__ = [
	"train_mlpHan",
]

def construct_mlpHan_parser():
	from . import add_discriminative_options, add_dense_options, add_dropout_init_options
	model_parser = add_discriminative_options()
	model_parser = add_dense_options(model_parser)
	model_parser = add_dropout_init_options(model_parser)

	model_parser.description = "prunable multi-layer perceptron argument"

	# model argument set 1
	model_parser.add_argument("--prune_policy", dest="prune_policy", action='store',
	                          default="1e-3", help="prune policy, [threshold=1e-3, interval=1, delay=0]")

	'''
	model_parser.add_argument("--prune_synapses_after", dest="prune_synapses_after", type=int, action='store',
	                          default=100, help="prune synapses after [100] epochs")
	model_parser.add_argument("--prune_synapses_interval", dest="prune_synapses_interval", type=int, action='store',
	                          default=1, help="prune synapses every [1] epochs")
	model_parser.add_argument("--prune_synapses_threshold", dest="prune_synapses_threshold", type=float,
	                          action='store', default=1e-3, help="prune synapses smaller than [1e-3] absolute weight")
	# model_parser.add_argument('--update_hidden_layer_dropout_only', dest="update_hidden_layer_dropout_only",
	# action='store_true', default=False, help="update hidden layer dropout only [False]")
	'''

	return model_parser


def validate_mlpHan_arguments(arguments):
	from . import validate_discriminative_options, validate_dense_arguments, validate_dropout_init_arguments
	arguments = validate_discriminative_options(arguments)
	arguments = validate_dense_arguments(arguments)
	number_of_layers = len(arguments.dense_dimensions)
	arguments = validate_dropout_init_arguments(arguments, number_of_layers)

	# model argument set 1
	arguments.prune_policy = parse_parameter_policy(arguments.prune_policy)
	'''
	assert arguments.prune_synapses_after >= 0
	assert arguments.prune_synapses_interval > 0
	assert arguments.prune_synapses_threshold > 0
	'''
	return arguments


def train_mlpHan():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	from . import config_model, validate_config
	settings = config_model(construct_mlpHan_parser, validate_mlpHan_arguments)
	settings = validate_config(settings)

	network = networks.MultiLayerPerceptronHan(
		incoming=settings.input_shape,

		dense_dimensions=settings.dense_dimensions,
		dense_nonlinearities=settings.dense_nonlinearities,

		prune_threshold_policy=settings.prune_policy,
		# prune_synapses_after=settings.prune_synapses_after,
		# prune_synapses_interval=settings.prune_synapses_interval,
		# prune_synapses_threshold=settings.prune_synapses_threshold,

		# layer_activation_types=settings.layer_activation_types,
		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles,

		objective_functions=settings.objective,
		update_function=settings.update,
		# pretrained_model=pretrained_model

		learning_rate_policy=settings.learning_rate,
		# learning_rate_decay=settings.learning_rate_decay,

		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)

	network.set_regularizers(settings.regularizer)
	# network.set_L1_regularizer_lambda(settings.L1_regularizer_lambdas)
	# network.set_L2_regularizer_lambda(settings.L2_regularizer_lambdas)

	from . import start_training
	start_training(network, settings)


if __name__ == '__main__':
	train_mlpHan()
