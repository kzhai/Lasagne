from .. import networks

__all__ = [
	"train_fdn",
]


def construct_fdn_parser():
	from .base import construct_discriminative_parser, add_dense_options
	model_parser = construct_discriminative_parser();
	model_parser = add_dense_options(model_parser);

	# model argument set 2
	model_parser.add_argument("--layer_activation_parameters", dest="layer_activation_parameters", action='store',
	                          default="1.0",
	                          help="dropout probability of different layer [1], either one number of a list of numbers, example, '0.2' represents 0.2 dropout rate for all input+hidden layers, or '0.2,0.5' represents 0.2 dropout rate for input layer and 0.5 dropout rate for first hidden layer respectively");
	# model_parser.add_argument("--layer_activation_styles", dest="layer_activation_styles", action='store', default="bernoulli",
	# help="dropout style different layer [bernoulli], example, 'bernoulli,beta-bernoulli' represents 2 layers with bernoulli and beta-bernoulli dropout respectively");

	'''
	model_parser.add_argument("--pretrained_model_file", dest="pretrained_model_file",
							  help="pretrained model file [None]");

	model_parser.add_argument("--dae_regularizer_lambdas", dest="dae_regularizer_lambdas", nargs="+", type=float, action='store', default=0,
							  help="dae regularization lambda [0]")
	model_parser.add_argument("--layer_corruption_levels", dest="layer_corruption_levels", nargs="+", type=float, action='store', default=0,
							  help="layer corruption level for pre-training [0], either one number of a list of numbers, example, '0.2' represents 0.2 corruption level for all denoising auto encoders, or '0.2,0.5' represents 0.2 corruption level for first denoising auto encoder layer and 0.5 for second one respectively");
	'''

	return model_parser;


def validate_fdn_arguments(arguments):
	from .base import validate_discriminative_arguments, validate_dense_arguments
	arguments = validate_discriminative_arguments(arguments);
	arguments = validate_dense_arguments(arguments);
	number_of_layers = len(arguments.dense_dimensions);

	# model argument set 2
	layer_activation_parameters = arguments.layer_activation_parameters;
	layer_activation_parameter_tokens = layer_activation_parameters.split(",")
	if len(layer_activation_parameter_tokens) == 1:
		layer_activation_parameters = [layer_activation_parameters for layer_index in range(number_of_layers)]
	elif len(layer_activation_parameter_tokens) == number_of_layers:
		layer_activation_parameters = layer_activation_parameter_tokens
	assert len(layer_activation_parameters) == number_of_layers;
	layer_activation_parameters = [float(layer_activation_parameter_tokens) for layer_activation_parameter_tokens in
	                               layer_activation_parameters]
	arguments.layer_activation_parameters = layer_activation_parameters;

	'''
	dae_regularizer_lambdas = arguments.dae_regularizer_lambdas
	if isinstance(dae_regularizer_lambdas, int):
		dae_regularizer_lambdas = [dae_regularizer_lambdas] * (number_of_layers - 1)
	assert len(dae_regularizer_lambdas) == number_of_layers - 1;
	assert (dae_regularizer_lambda >= 0 for dae_regularizer_lambda in dae_regularizer_lambdas)
	self.dae_regularizer_lambdas = dae_regularizer_lambdas;

	layer_corruption_levels = arguments.layer_corruption_levels;
	if isinstance(layer_corruption_levels, int):
		layer_corruption_levels = [layer_corruption_levels] * (number_of_layers - 1)
	assert len(layer_corruption_levels) == number_of_layers - 1;
	assert (layer_corruption_level >= 0 for layer_corruption_level in layer_corruption_levels)
	assert (layer_corruption_level <= 1 for layer_corruption_level in layer_corruption_levels)
	self.layer_corruption_levels = layer_corruption_levels;

	pretrained_model_file = arguments.pretrained_model_file;
	pretrained_model = None;
	if pretrained_model_file != None:
		assert os.path.exists(pretrained_model_file)
		pretrained_model = cPickle.load(open(pretrained_model_file, 'rb'));
	'''

	return arguments


def train_fdn():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	from .base import config_model
	settings = config_model(construct_fdn_parser, validate_fdn_arguments)

	network = networks.FastDropoutNetwork(
		incoming=settings.input_shape,

		layer_dimensions=settings.dense_dimensions,
		layer_nonlinearities=settings.dense_nonlinearities,

		layer_activation_parameters=settings.layer_activation_parameters,
		# layer_activation_styles=settings.layer_activation_styles,

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

	network.set_regularizers(settings.regularizer);
	# network.set_L1_regularizer_lambda(settings.L1_regularizer_lambdas)
	# network.set_L2_regularizer_lambda(settings.L2_regularizer_lambdas)

	from .base import train_model
	train_model(network, settings)


def resume_fdn():
	pass


def test_fdn():
	pass


if __name__ == '__main__':
	train_fdn()
