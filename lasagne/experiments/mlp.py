import logging
import pickle
import os

logger = logging.getLogger(__name__)

from .. import layers, networks, nonlinearities, Xpolicy, Xregularization
from . import layer_deliminator

__all__ = [
	#
	"add_dense_options",
	"validate_dense_arguments",
	"add_dropout_options",
	"validate_dropout_arguments",
	"add_dropout_init_options",
	"validate_dropout_init_arguments",
	#
	"mlp_parser",
	"mlp_validator",
	#
	"start_mlp",
	"resume_mlp",
	#
	"mlpA_parser",
	"mlpA_validator",
	#
	"mlpD_parser",
	"mlpD_validator",
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


def mlp_parser():
	from . import discriminative_parser
	model_parser = discriminative_parser()

	# model_parser = add_dense_options(model_parser)
	# model_parser = add_dropout_options(model_parser)

	# subparsers = model_parser.add_subparsers()
	# resume_parser = subparsers.add_parser('resume', help='resume training')
	# resume_parser = add_resume_options(resume_parser)

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


def mlp_validator(arguments):
	from . import discriminative_validator
	arguments = discriminative_validator(arguments)

	arguments = validate_dense_arguments(arguments)
	number_of_layers = len(arguments.dense_dimensions)
	arguments = validate_dropout_arguments(arguments, number_of_layers)

	return arguments


def mlpA_parser():
	from . import discriminative_adaptive_parser
	model_parser = discriminative_adaptive_parser()

	model_parser = add_dense_options(model_parser)
	model_parser = add_dropout_options(model_parser)

	return model_parser


def mlpA_validator(arguments):
	from . import discriminative_adaptive_validator
	arguments = discriminative_adaptive_validator(arguments)

	arguments = validate_dense_arguments(arguments)
	number_of_layers = len(arguments.dense_dimensions)
	arguments = validate_dropout_arguments(arguments, number_of_layers)

	return arguments


def mlpD_parser():
	from . import discriminative_adaptive_dynamic_parser
	model_parser = discriminative_adaptive_dynamic_parser()

	model_parser = add_dense_options(model_parser)
	model_parser = add_dropout_options(model_parser)

	return model_parser


def mlpD_validator(arguments):
	from . import discriminative_adaptive_dynamic_validator
	arguments = discriminative_adaptive_dynamic_validator(arguments)

	arguments = validate_dense_arguments(arguments)
	number_of_layers = len(arguments.dense_dimensions)
	arguments = validate_dropout_arguments(arguments, number_of_layers)

	return arguments


def start_mlp():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""
	from . import config_model, validate_config
	settings = config_model(mlp_parser, mlp_validator)
	settings = validate_config(settings)

	network = networks.FeedForwardNetwork(
		incoming=settings.input_shape,
		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,
		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)
	mlp = networks.MultiLayerPerceptronFromSpecifications(
		network._input_layer,
		dense_dimensions=settings.dense_dimensions,
		dense_nonlinearities=settings.dense_nonlinearities,
		layer_activation_types=settings.layer_activation_types,
		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles
	)
	network.set_network(mlp)
	network.set_regularizers(settings.regularizer)

	from . import start_training
	start_training(network, settings)


def resume_mlp():
	from . import config_model, validate_config, discriminative_resume_parser, discriminative_resume_validator

	settings = config_model(discriminative_resume_parser, discriminative_resume_validator)
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
	mlp = networks.MultiLayerPerceptronFromPretrainedModel(
		network._input_layer,
		pretrained_network=model
	)
	network.set_network(mlp)
	network.set_regularizers(settings.regularizer)

	from . import resume_training
	resume_training(network, settings)


if __name__ == '__main__':
	import argparse

	model_selector = argparse.ArgumentParser(description="mode selector")
	model_selector.add_argument("--resume", dest="resume", action='store_true', default=False,
	                            help="resume [None]")

	arguments, additionals = model_selector.parse_known_args()

	if arguments.resume:
		resume_mlp()
	else:
		start_mlp()
