import logging

logger = logging.getLogger(__name__)

from .. import networks, nonlinearities

__all__ = [
	"add_dense_options",
	"validate_dense_arguments",
	"add_dropout_options",
	"validate_dropout_arguments",
	#
	"train_mlp",
]


def add_dense_options(model_parser):
	# model argument set 1
	model_parser.add_argument("--dense_dimensions", dest="dense_dimensions", action='store', default=None,
	                          help="dimension of different layer [None], example, '100,500,10' represents 3 layers contains 100, 500, and 10 neurons respectively")
	model_parser.add_argument("--dense_nonlinearities", dest="dense_nonlinearities", action='store', default=None,
	                          help="activation functions of different layer [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively")

	return model_parser


def validate_dense_arguments(arguments):
	# model argument set 1
	assert arguments.dense_dimensions is not None
	dense_dimensions = arguments.dense_dimensions.split(",")
	arguments.dense_dimensions = [int(dimensionality) for dimensionality in dense_dimensions]

	assert arguments.dense_nonlinearities is not None
	dense_nonlinearities = arguments.dense_nonlinearities.split(",")
	arguments.dense_nonlinearities = [getattr(nonlinearities, dense_nonlinearity) for dense_nonlinearity in
	                                  dense_nonlinearities]

	assert len(arguments.dense_nonlinearities) == len(arguments.dense_dimensions)

	return arguments


def add_dropout_options(model_parser):
	# model argument set 2
	model_parser.add_argument("--layer_activation_parameters", dest="layer_activation_parameters", action='store',
	                          default="1.0",
	                          help="dropout probability of different layer [1], either one number of a list of numbers, example, '0.2' represents 0.2 dropout rate for all input+hidden layers, or '0.2,0.5' represents 0.2 dropout rate for input layer and 0.5 dropout rate for first hidden layer respectively")
	model_parser.add_argument("--layer_activation_styles", dest="layer_activation_styles", action='store',
	                          default="bernoulli",
	                          help="dropout style different layer [bernoulli], example, 'bernoulli,beta-bernoulli' represents 2 layers with bernoulli and beta-bernoulli dropout respectively")

	return model_parser


def validate_dropout_arguments(arguments, number_of_layers):
	# model argument set
	layer_activation_styles = arguments.layer_activation_styles
	layer_activation_style_tokens = layer_activation_styles.split(",")
	if len(layer_activation_style_tokens) == 1:
		layer_activation_styles = [layer_activation_styles for layer_index in range(number_of_layers)]
	elif len(layer_activation_style_tokens) == number_of_layers:
		layer_activation_styles = layer_activation_style_tokens
	# [float(layer_activation_parameter) for layer_activation_parameter in layer_activation_parameter_tokens]
	assert len(layer_activation_styles) == number_of_layers
	assert (layer_activation_style in set(
		["bernoulli", "beta_bernoulli", "reciprocal_beta_bernoulli", "reverse_reciprocal_beta_bernoulli",
		 "mixed_beta_bernoulli"]) for layer_activation_style in layer_activation_styles)
	arguments.layer_activation_styles = layer_activation_styles

	layer_activation_parameters = arguments.layer_activation_parameters
	layer_activation_parameter_tokens = layer_activation_parameters.split(",")
	if len(layer_activation_parameter_tokens) == 1:
		layer_activation_parameters = [layer_activation_parameters for layer_index in range(number_of_layers)]
	elif len(layer_activation_parameter_tokens) == number_of_layers:
		layer_activation_parameters = layer_activation_parameter_tokens
	assert len(layer_activation_parameters) == number_of_layers

	for layer_index in range(number_of_layers):
		if layer_activation_styles[layer_index] == "bernoulli":
			layer_activation_parameters[layer_index] = float(layer_activation_parameters[layer_index])
			assert layer_activation_parameters[layer_index] <= 1
			assert layer_activation_parameters[layer_index] > 0
		elif layer_activation_styles[layer_index] == "beta_bernoulli" \
				or layer_activation_styles[layer_index] == "reciprocal_beta_bernoulli" \
				or layer_activation_styles[layer_index] == "reverse_reciprocal_beta_bernoulli" \
				or layer_activation_styles[layer_index] == "mixed_beta_bernoulli":
			layer_activation_parameter_tokens = layer_activation_parameters[layer_index].split("+")
			assert len(layer_activation_parameter_tokens) == 2
			layer_activation_parameters[layer_index] = (float(layer_activation_parameter_tokens[0]),
			                                            float(layer_activation_parameter_tokens[1]))
			assert layer_activation_parameters[layer_index][0] > 0
			assert layer_activation_parameters[layer_index][1] > 0
			if layer_activation_styles[layer_index] == "mixed_beta_bernoulli":
				assert layer_activation_parameters[layer_index][0] < 1
	arguments.layer_activation_parameters = layer_activation_parameters

	return arguments


def construct_mlp_parser():
	from . import construct_discriminative_parser, add_dense_options, add_dropout_options
	model_parser = construct_discriminative_parser()
	model_parser = add_dense_options(model_parser)
	model_parser = add_dropout_options(model_parser)

	'''
	model_parser.add_argument("--pretrained_model_file", dest="pretrained_model_file",
							  help="pretrained model file [None]")
	model_parser.add_argument("--dae_regularizer_lambdas", dest="dae_regularizer_lambdas", nargs="+", type=float, action='store', default=0,
							  help="dae regularization lambda [0]")
	model_parser.add_argument("--layer_corruption_levels", dest="layer_corruption_levels", nargs="+", type=float, action='store', default=0,
							  help="layer corruption level for pre-training [0], either one number of a list of numbers, example, '0.2' represents 0.2 corruption level for all denoising auto encoders, or '0.2,0.5' represents 0.2 corruption level for first denoising auto encoder layer and 0.5 for second one respectively")
	'''

	return model_parser


def validate_mlp_arguments(arguments):
	from . import validate_discriminative_arguments, validate_dense_arguments, validate_dropout_arguments
	arguments = validate_discriminative_arguments(arguments)

	arguments = validate_dense_arguments(arguments)
	number_of_layers = len(arguments.dense_dimensions)
	arguments = validate_dropout_arguments(arguments, number_of_layers)

	'''
	dae_regularizer_lambdas = arguments.dae_regularizer_lambdas
	if isinstance(dae_regularizer_lambdas, int):
		dae_regularizer_lambdas = [dae_regularizer_lambdas] * (number_of_layers - 1)
	assert len(dae_regularizer_lambdas) == number_of_layers - 1
	assert (dae_regularizer_lambda >= 0 for dae_regularizer_lambda in dae_regularizer_lambdas)
	self.dae_regularizer_lambdas = dae_regularizer_lambdas

	layer_corruption_levels = arguments.layer_corruption_levels
	if isinstance(layer_corruption_levels, int):
		layer_corruption_levels = [layer_corruption_levels] * (number_of_layers - 1)
	assert len(layer_corruption_levels) == number_of_layers - 1
	assert (layer_corruption_level >= 0 for layer_corruption_level in layer_corruption_levels)
	assert (layer_corruption_level <= 1 for layer_corruption_level in layer_corruption_levels)
	self.layer_corruption_levels = layer_corruption_levels

	pretrained_model_file = arguments.pretrained_model_file
	pretrained_model = None
	if pretrained_model_file != None:
		assert os.path.exists(pretrained_model_file)
		pretrained_model = cPickle.load(open(pretrained_model_file, 'rb'))
	'''

	return arguments


def train_mlp():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""
	from . import config_model, validate_config
	settings = config_model(construct_mlp_parser, validate_mlp_arguments)
	settings = validate_config(settings)

	network = networks.MultiLayerPerceptron(
		incoming=settings.input_shape,

		dense_dimensions=settings.dense_dimensions,
		dense_nonlinearities=settings.dense_nonlinearities,

		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles,

		objective_functions=settings.objective,
		update_function=settings.update,
		# pretrained_model=pretrained_model

		learning_rate=settings.learning_rate,
		learning_rate_decay=settings.learning_rate_decay,
		max_norm_constraint=settings.max_norm_constraint,
		# learning_rate_decay_style=settings.learning_rate_decay_style,
		# learning_rate_decay_parameter=settings.learning_rate_decay_parameter,
		validation_interval=settings.validation_interval,
	)

	network.set_regularizers(settings.regularizer)
	# network.set_L1_regularizer_lambda(settings.L1_regularizer_lambdas)
	# network.set_L2_regularizer_lambda(settings.L2_regularizer_lambdas)

	from . import train_model
	train_model(network, settings)


def resume_mlp():
	pass


def test_mlp():
	pass


if __name__ == '__main__':
	train_mlp()
