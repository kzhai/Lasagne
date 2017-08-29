import logging
import os

import numpy

from . import layer_deliminator, param_deliminator
from .. import layers, networks, policy, regularization

logger = logging.getLogger(__name__)

__all__ = [
	"train_dmlp",
]


'''
def validate_dropout_arguments(arguments, number_of_layers):
	# model argument set
	layer_activation_types = arguments.layer_activation_types
	if layer_activation_types is None:
		layer_activation_types = ["AdaptiveDropoutLayer"] * number_of_layers
	else:
		layer_activation_type_tokens = layer_activation_types.split(layer_deliminator)
		if len(layer_activation_type_tokens) == 1:
			layer_activation_types = layer_activation_type_tokens * number_of_layers
		else:
			layer_activation_types = layer_activation_type_tokens
		assert len(layer_activation_types) == number_of_layers
	assert layer_activation_types[0] in set(["AdaptiveDropoutLayer", "BernoulliDropoutLayer"])
	for layer_activation_type_index in xrange(len(layer_activation_types)):
		if layer_activation_types[layer_activation_type_index] in set(
				["BernoulliDropoutLayer", "GaussianDropoutLayer", "FastDropoutLayer", "AdaptiveDropoutLayer"]):
			pass
		elif layer_activation_types[layer_activation_type_index] in set(
				["VariationalDropoutLayer", "VariationalDropoutTypeALayer", "VariationalDropoutTypeBLayer"]):
			if regularization.kl_divergence_kingma not in arguments.regularizer:
				arguments.regularizer[regularization.kl_divergence_kingma] = [1.0, policy.constant]
			assert regularization.kl_divergence_kingma in arguments.regularizer
		elif layer_activation_types[layer_activation_type_index] in set(["SparseVariationalDropoutLayer"]):
			if regularization.kl_divergence_sparse not in arguments.regularizer:
				arguments.regularizer[regularization.kl_divergence_sparse] = [1.0, policy.constant]
			assert regularization.kl_divergence_sparse in arguments.regularizer
		else:
			logger.error("unrecognized dropout type %s..." % (layer_activation_types[layer_activation_type_index]))
		layer_activation_types[layer_activation_type_index] = getattr(layers.noise, layer_activation_types[
			layer_activation_type_index])
	arguments.layer_activation_types = layer_activation_types

	layer_activation_styles = arguments.layer_activation_styles
	layer_activation_style_tokens = layer_activation_styles.split(layer_deliminator)
	if len(layer_activation_style_tokens) == 1:
		layer_activation_styles = [layer_activation_styles for layer_index in range(number_of_layers)]
	elif len(layer_activation_style_tokens) == number_of_layers:
		layer_activation_styles = layer_activation_style_tokens
	# [float(layer_activation_parameter) for layer_activation_parameter in layer_activation_parameter_tokens]
	assert len(layer_activation_styles) == number_of_layers
	assert (layer_activation_style in set(
		["uniform", "bernoulli", "beta_bernoulli", "reciprocal_beta_bernoulli", "reverse_reciprocal_beta_bernoulli",
		 "mixed_beta_bernoulli"]) for layer_activation_style in layer_activation_styles)
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
'''

def construct_dmlp_parser():
	from .mlp import construct_mlp_parser
	model_parser = construct_mlp_parser();

	model_parser.description = "dynamic multi-layer perceptron argument"

	# model argument set 1
	'''
	model_parser.add_argument("--dropout_learning_rate", dest="dropout_learning_rate", type=float, action='store',
	                          default=0, help="dropout learning rate [0 = learning_rate]")
	model_parser.add_argument("--dropout_learning_rate_decay", dest="dropout_learning_rate_decay", action='store',
	                          default=None, help="dropout learning rate decay [None = learning_rate_decay]")
	'''

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
	from . import validate_discriminative_arguments, validate_dense_arguments
	arguments = validate_discriminative_arguments(arguments)

	arguments = validate_dense_arguments(arguments)
	number_of_layers = len(arguments.dense_dimensions)
	arguments = validate_dropout_arguments(arguments, number_of_layers)
	'''

	# model argument set 1
	from . import validate_decay_policy
	if arguments.dropout_learning_rate is None:
		arguments.dropout_learning_rate = arguments.learning_rate;
	else:
		dropout_learning_rate_tokens = arguments.dropout_learning_rate.split(param_deliminator)
		arguments.dropout_learning_rate = validate_decay_policy(dropout_learning_rate_tokens)

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
		#update_hidden_layer_dropout_only=settings.update_hidden_layer_dropout_only,

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
