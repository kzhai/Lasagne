import logging

from .. import networks

logger = logging.getLogger(__name__)

__all__ = [
	"train_snn",
]


def construct_snn_parser():
	from . import add_discriminative_options, add_dense_options
	model_parser = add_discriminative_options()
	model_parser = add_dense_options(model_parser)

	# model argument set 1
	model_parser.add_argument("--input_activation_rate", dest="input_activation_rate", type=float, action='store',
	                          default=1.0,
	                          help="activation rate for input layer [1.0]")

	'''
	model_parser.add_argument("--pretrained_model_file", dest="pretrained_model_file",
							  help="pretrained model file [None]")

	model_parser.add_argument("--dae_regularizer_lambdas", dest="dae_regularizer_lambdas", nargs="+", type=float, action='store', default=0,
							  help="dae regularization lambda [0]")
	model_parser.add_argument("--layer_corruption_levels", dest="layer_corruption_levels", nargs="+", type=float, action='store', default=0,
							  help="layer corruption level for pre-training [0], either one number of a list of numbers, example, '0.2' represents 0.2 corruption level for all denoising auto encoders, or '0.2,0.5' represents 0.2 corruption level for first denoising auto encoder layer and 0.5 for second one respectively")
	'''

	return model_parser


def validate_snn_arguments(arguments):
	from . import validate_discriminative_options, validate_dense_arguments
	arguments = validate_discriminative_options(arguments)
	arguments = validate_dense_arguments(arguments)
	number_of_layers = len(arguments.dense_dimensions)

	assert 0 < arguments.input_activation_rate <= 1
	arguments.input_activation_rate = arguments.input_activation_rate

	'''
	dae_regularizer_lambdas = arguments.dae_regularizer_lambdas
	if isinstance(dae_regularizer_lambdas, int):
		dae_regularizer_lambdas = [dae_regularizer_lambdas] * (number_of_layers - 1)
	assert len(dae_regularizer_lambdas) == number_of_layers - 1
	assert (dae_regularizer_lambda >= 0 for dae_regularizer_lambda in dae_regularizer_lambdas)
	arguments.dae_regularizer_lambdas = dae_regularizer_lambdas

	layer_corruption_levels = arguments.layer_corruption_levels
	if isinstance(layer_corruption_levels, int):
		layer_corruption_levels = [layer_corruption_levels] * (number_of_layers - 1)
	assert len(layer_corruption_levels) == number_of_layers - 1
	assert (layer_corruption_level >= 0 for layer_corruption_level in layer_corruption_levels)
	assert (layer_corruption_level <= 1 for layer_corruption_level in layer_corruption_levels)
	arguments.layer_corruption_levels = layer_corruption_levels

	pretrained_model_file = arguments.pretrained_model_file
	pretrained_model = None
	if pretrained_model_file != None:
		assert os.path.exists(pretrained_model_file)
		pretrained_model = cPickle.load(open(pretrained_model_file, 'rb'))
	'''

	return arguments


def train_snn():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	from . import config_model, validate_config
	settings = config_model(construct_snn_parser, validate_snn_arguments)
	settings = validate_config(settings)

	######################
	# BUILD ACTUAL MODEL #
	######################

	network = networks.StandoutNeuralNetworkTypeB(
		incoming=settings.input_shape,

		dense_dimensions=settings.dense_dimensions,
		dense_nonlinearities=settings.dense_nonlinearities,

		input_activation_rate=settings.input_activation_rate,
		# pretrained_model=None,

		objective_functions=settings.objective,
		update_function=settings.update,

		learning_rate_policy=settings.learning_rate,
		#learning_rate_decay=settings.learning_rate_decay,
		max_norm_constraint=settings.max_norm_constraint,

		validation_interval=settings.validation_interval,
	)

	network.set_regularizers(settings.regularizer)
	# network.set_L1_regularizer_lambda(settings.L1_regularizer_lambdas)
	# network.set_L2_regularizer_lambda(settings.L2_regularizer_lambdas)

	from . import start_training
	start_training(network, settings)


if __name__ == '__main__':
	train_snn()
