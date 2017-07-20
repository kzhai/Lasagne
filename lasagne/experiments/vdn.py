from .. import networks

__all__ = [
	"train_vdn",
]


def construct_vdn_parser():
	from . import construct_discriminative_parser, add_dense_options
	model_parser = construct_discriminative_parser()
	model_parser = add_dense_options(model_parser)

	# model argument set 2
	model_parser.add_argument("--layer_activation_parameters", dest="layer_activation_parameters", action='store',
	                          default="1.0",
	                          help="dropout probability of different layer [1], either one number of a list of numbers, example, '0.2' represents 0.2 dropout rate for all input+hidden layers, or '0.2,0.5' represents 0.2 dropout rate for input layer and 0.5 dropout rate for first hidden layer respectively")

	model_parser.add_argument("--variational_dropout_style", dest="variational_dropout_style", action='store',
	                          default=None,
	                          help="variational dropout style [None], example, 'TypeA' or 'TypeB'")
	model_parser.add_argument("--adaptive_styles", dest="adaptive_styles", action='store', default="layerwise",
	                          help="adaptive styles [layerwise], either one string of a list of string, example, 'layerwise', 'elementwise' and 'weightwise' (only apply to VariationalDropoutTypeB layers")
	model_parser.add_argument("--variational_dropout_regularizer_lambdas",
	                          dest="variational_dropout_regularizer_lambdas", action='store', default="0.1",
	                          help="variational dropout regularizer lambdas [0.1], either one number of a list of numbers")
	# model_parser.add_argument("--layer_activation_styles", dest="layer_activation_styles", action='store', default="bernoulli",
	# help="dropout style different layer [bernoulli], example, 'bernoulli,beta-bernoulli' represents 2 layers with bernoulli and beta-bernoulli dropout respectively")

	'''
	model_parser.add_argument("--pretrained_model_file", dest="pretrained_model_file",
							  help="pretrained model file [None]")

	model_parser.add_argument("--dae_regularizer_lambdas", dest="dae_regularizer_lambdas", nargs="+", type=float, action='store', default=0,
							  help="dae regularization lambda [0]")
	model_parser.add_argument("--layer_corruption_levels", dest="layer_corruption_levels", nargs="+", type=float, action='store', default=0,
							  help="layer corruption level for pre-training [0], either one number of a list of numbers, example, '0.2' represents 0.2 corruption level for all denoising auto encoders, or '0.2,0.5' represents 0.2 corruption level for first denoising auto encoder layer and 0.5 for second one respectively")
	'''

	return model_parser


def validate_vdn_arguments(arguments):
	from . import validate_discriminative_arguments, validate_dense_arguments
	arguments = validate_discriminative_arguments(arguments)
	arguments = validate_dense_arguments(arguments)
	number_of_layers = len(arguments.dense_dimensions)

	# model argument set 2
	assert arguments.variational_dropout_style != None
	arguments.variational_dropout_style = arguments.variational_dropout_style
	assert arguments.variational_dropout_style in set(["TypeA", "TypeB"])

	layer_activation_parameters = arguments.layer_activation_parameters
	layer_activation_parameter_tokens = layer_activation_parameters.split(",")
	if len(layer_activation_parameter_tokens) == 1:
		layer_activation_parameters = [layer_activation_parameters for layer_index in range(number_of_layers)]
	elif len(layer_activation_parameter_tokens) == number_of_layers:
		layer_activation_parameters = layer_activation_parameter_tokens
	assert len(layer_activation_parameters) == number_of_layers
	layer_activation_parameters = [float(layer_activation_parameter_tokens) for layer_activation_parameter_tokens in
	                               layer_activation_parameters]
	arguments.layer_activation_parameters = layer_activation_parameters

	adaptive_styles = arguments.adaptive_styles
	adaptive_styles_tokens = adaptive_styles.split(",")
	if len(adaptive_styles_tokens) == 1:
		adaptive_styles = [adaptive_styles for layer_index in range(number_of_layers)]
	elif len(adaptive_styles_tokens) == number_of_layers:
		adaptive_styles = adaptive_styles_tokens
	assert len(adaptive_styles) == number_of_layers
	arguments.adaptive_styles = adaptive_styles

	if arguments.variational_dropout_style == "TypeB":
		assert (adaptive_style == None or adaptive_style in set(["layerwise", "elementwise", "weightwise"]) for
		        adaptive_style in arguments.adaptive_styles)
	elif arguments.variational_dropout_style == "TypeA":
		assert (adaptive_style == None or adaptive_style in set(["layerwise", "elementwise"]) for adaptive_style in
		        arguments.adaptive_styles)

	variational_dropout_regularizer_lambdas = arguments.variational_dropout_regularizer_lambdas
	variational_dropout_regularizer_lambdas_tokens = variational_dropout_regularizer_lambdas.split(",")
	if len(variational_dropout_regularizer_lambdas_tokens) == 1:
		variational_dropout_regularizer_lambdas = [variational_dropout_regularizer_lambdas for layer_index in
		                                           range(number_of_layers)]
	elif len(variational_dropout_regularizer_lambdas_tokens) == number_of_layers:
		variational_dropout_regularizer_lambdas = variational_dropout_regularizer_lambdas_tokens
	assert len(variational_dropout_regularizer_lambdas) == number_of_layers
	variational_dropout_regularizer_lambdas = [float(variational_dropout_regularizer_lambdas_tokens) for
	                                           variational_dropout_regularizer_lambdas_tokens in
	                                           variational_dropout_regularizer_lambdas]
	arguments.variational_dropout_regularizer_lambdas = variational_dropout_regularizer_lambdas

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


def train_vdn():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	from . import config_model, validate_config
	settings = config_model(construct_vdn_parser, validate_vdn_arguments)
	settings = validate_config(settings)

	if settings.variational_dropout_style == "TypeA":
		network = networks.VariationalDropoutTypeANetwork(
			incoming=settings.input_shape,

			dense_dimensions=settings.dense_dimensions,
			dense_nonlinearities=settings.dense_nonlinearities,

			layer_activation_parameters=settings.layer_activation_parameters,
			adaptive_styles=settings.adaptive_styles,
			variational_dropout_regularizer_lambdas=settings.variational_dropout_regularizer_lambdas,

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
	elif settings.variational_dropout_style == "TypeB":
		network = networks.VariationalDropoutTypeBNetwork(
			incoming=settings.input_shape,

			dense_dimensions=settings.dense_dimensions,
			dense_nonlinearities=settings.dense_nonlinearities,

			layer_activation_parameters=settings.layer_activation_parameters,
			adaptive_styles=settings.adaptive_styles,
			variational_dropout_regularizer_lambdas=settings.variational_dropout_regularizer_lambdas,

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


if __name__ == '__main__':
	train_vdn()
