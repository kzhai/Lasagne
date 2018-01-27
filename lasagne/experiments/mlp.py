import logging

logger = logging.getLogger(__name__)

from .. import layers, networks, nonlinearities, Xpolicy, Xregularization
from . import layer_deliminator

__all__ = [
	"add_dense_options",
	"validate_dense_arguments",
	"add_dropout_init_options",
	"validate_dropout_init_arguments",
	"add_dropout_options",
	"validate_dropout_arguments",
	#
	"train_mlp",
]


def add_dense_options(model_parser):
	# model argument set 1
	model_parser.add_argument("--dense_dimensions", dest="dense_dimensions", action='store', default=None,
	                          help="dense layer dimensionalities [None]")
	model_parser.add_argument("--dense_nonlinearities", dest="dense_nonlinearities", action='store', default=None,
	                          help="dense layer activation functions [None]")

	return model_parser


def validate_dense_arguments(arguments):
	# model argument set 1
	assert arguments.dense_dimensions is not None
	dense_dimensions = arguments.dense_dimensions.split(layer_deliminator)
	arguments.dense_dimensions = [int(dimensionality) for dimensionality in dense_dimensions]

	assert arguments.dense_nonlinearities is not None
	dense_nonlinearities = arguments.dense_nonlinearities.split(layer_deliminator)
	arguments.dense_nonlinearities = [getattr(nonlinearities, dense_nonlinearity) for dense_nonlinearity in
	                                  dense_nonlinearities]

	assert len(arguments.dense_nonlinearities) == len(arguments.dense_dimensions)

	return arguments


def add_dropout_init_options(model_parser):
	# model argument set 2
	model_parser.add_argument("--layer_activation_styles", dest="layer_activation_styles", action='store',
	                          default="bernoulli",
	                          help="layer activation styles [bernoulli]")
	model_parser.add_argument("--layer_activation_parameters", dest="layer_activation_parameters", action='store',
	                          default="1.0",
	                          help="layer activation parameters [1] for layer activation styles respectively")

	return model_parser


def validate_dropout_init_arguments(arguments, number_of_layers):
	layer_activation_styles = arguments.layer_activation_styles
	layer_activation_style_tokens = layer_activation_styles.split(layer_deliminator)
	if len(layer_activation_style_tokens) == 1:
		layer_activation_styles = [layer_activation_styles for layer_index in range(number_of_layers)]
	elif len(layer_activation_style_tokens) == number_of_layers:
		layer_activation_styles = layer_activation_style_tokens
	# [float(layer_activation_parameter) for layer_activation_parameter in layer_activation_parameter_tokens]
	assert len(layer_activation_styles) == number_of_layers
	assert (layer_activation_style in {"uniform", "bernoulli", "beta_bernoulli", "reciprocal_beta_bernoulli",
	                                   "reverse_reciprocal_beta_bernoulli", "mixed_beta_bernoulli"} for
	        layer_activation_style in layer_activation_styles)
	arguments.layer_activation_styles = layer_activation_styles

	layer_activation_parameters = arguments.layer_activation_parameters
	layer_activation_parameter_tokens = layer_activation_parameters.split(layer_deliminator)
	if len(layer_activation_parameter_tokens) == 1:
		layer_activation_parameters = [layer_activation_parameters for layer_index in range(number_of_layers)]
	elif len(layer_activation_parameter_tokens) == number_of_layers:
		layer_activation_parameters = layer_activation_parameter_tokens
	assert len(layer_activation_parameters) == number_of_layers

	for layer_index in range(number_of_layers):
		if layer_activation_styles[layer_index] == "uniform":
			layer_activation_parameters[layer_index] = float(layer_activation_parameters[layer_index])
			assert layer_activation_parameters[layer_index] <= 1
			assert layer_activation_parameters[layer_index] > 0
		elif layer_activation_styles[layer_index] == "bernoulli":
			layer_activation_parameters[layer_index] = float(layer_activation_parameters[layer_index])
			assert layer_activation_parameters[layer_index] <= 1
			assert layer_activation_parameters[layer_index] > 0
		elif layer_activation_styles[layer_index] == "beta_bernoulli" \
				or layer_activation_styles[layer_index] == "reciprocal_beta_bernoulli" \
				or layer_activation_styles[layer_index] == "reverse_reciprocal_beta_bernoulli" \
				or layer_activation_styles[layer_index] == "mixed_beta_bernoulli":
			layer_activation_parameter_tokens = layer_activation_parameters[layer_index].split("+")
			assert len(layer_activation_parameter_tokens) == 2, layer_activation_parameter_tokens
			layer_activation_parameters[layer_index] = (float(layer_activation_parameter_tokens[0]),
			                                            float(layer_activation_parameter_tokens[1]))
			assert layer_activation_parameters[layer_index][0] > 0
			assert layer_activation_parameters[layer_index][1] > 0
			if layer_activation_styles[layer_index] == "mixed_beta_bernoulli":
				assert layer_activation_parameters[layer_index][0] < 1
	arguments.layer_activation_parameters = layer_activation_parameters

	return arguments


def add_dropout_options(model_parser):
	# model argument set 2
	model_parser.add_argument("--layer_activation_types", dest="layer_activation_types", action='store', default=None,
	                          help="dropout type [None]")
	model_parser = add_dropout_init_options(model_parser)

	return model_parser


def validate_dropout_arguments(arguments, number_of_layers):
	# model argument set
	layer_activation_types = arguments.layer_activation_types
	if layer_activation_types is None:
		layer_activation_types = ["BernoulliDropoutLayer"] * number_of_layers
	else:
		layer_activation_type_tokens = layer_activation_types.split(layer_deliminator)
		if len(layer_activation_type_tokens) == 1:
			layer_activation_types = layer_activation_type_tokens * number_of_layers
		else:
			layer_activation_types = layer_activation_type_tokens
		assert len(layer_activation_types) == number_of_layers
	assert layer_activation_types[0] not in {"FastDropoutLayer", "VariationalDropoutTypeBLayer"}
	for layer_activation_type_index in range(len(layer_activation_types)):
		if layer_activation_types[layer_activation_type_index] in {"BernoulliDropoutLayer", "GaussianDropoutLayer",
		                                                           "FastDropoutLayer"}:
			pass
		elif layer_activation_types[layer_activation_type_index] in {"VariationalDropoutLayer",
		                                                             "VariationalDropoutTypeALayer",
		                                                             "VariationalDropoutTypeBLayer"}:
			if Xregularization.kl_divergence_kingma not in arguments.regularizer:
				arguments.regularizer[Xregularization.kl_divergence_kingma] = [1.0, Xpolicy.constant]
			assert Xregularization.kl_divergence_kingma in arguments.regularizer
		elif layer_activation_types[layer_activation_type_index] in {"SparseVariationalDropoutLayer"}:
			if Xregularization.kl_divergence_sparse not in arguments.regularizer:
				arguments.regularizer[Xregularization.kl_divergence_sparse] = [1.0, Xpolicy.constant]
			assert Xregularization.kl_divergence_sparse in arguments.regularizer
		elif layer_activation_types[layer_activation_type_index] in {"AdaptiveDropoutLayer", "DynamicDropoutLayer"}:
			if (Xregularization.rademacher_p_2_q_2 not in arguments.regularizer) and \
					(Xregularization.rademacher_p_inf_q_1 not in arguments.regularizer):
				arguments.regularizer[Xregularization.rademacher] = [1.0, Xpolicy.constant]
			assert (Xregularization.rademacher_p_2_q_2 in arguments.regularizer) or \
			       (Xregularization.rademacher_p_inf_q_1 in arguments.regularizer)
		else:
			logger.error("unrecognized dropout type %s..." % (layer_activation_types[layer_activation_type_index]))
		layer_activation_types[layer_activation_type_index] = getattr(layers, layer_activation_types[
			layer_activation_type_index])
	arguments.layer_activation_types = layer_activation_types

	arguments = validate_dropout_init_arguments(arguments, number_of_layers)
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
	from . import validate_discriminative_arguments, validate_dense_arguments
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

		layer_activation_types=settings.layer_activation_types,
		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles,

		objective_functions=settings.objective,
		update_function=settings.update,
		# pretrained_model=pretrained_model

		learning_rate_policy=settings.learning_rate,
		# learning_rate_decay=settings.learning_rate_decay,
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
