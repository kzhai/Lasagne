import logging
import pickle

logger = logging.getLogger(__name__)

from lasagne import layers, nonlinearities, networks
from lasagne import Xregularization, Xpolicy
from . import layer_deliminator, validate_config
from . import start_training, resume_training

__all__ = [
	"add_dense_options",
	"validate_dense_arguments",
	"add_dropout_options",
	"validate_dropout_arguments",
	#"add_dropout_init_options",
	#"validate_dropout_init_arguments",
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

	return arguments, len(arguments.dense_dimensions)


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


def start_mlp(settings):
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	# settings = config_model(parser, validator)
	settings = validate_config(settings)

	network = networks.FeedForwardNetwork(
		incoming=settings.input_shape,
		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,
		parameter_local_max_norm=settings.parameter_local_max_norm,
		gradient_global_max_norm=settings.gradient_global_max_norm,
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

	start_training(network, settings)


def resume_mlp(settings):
	# settings = config_model(add_resume_options, validate_resume_options)
	settings = validate_config(settings)

	network = networks.FeedForwardNetwork(
		incoming=settings.input_shape,
		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,
		parameter_local_max_norm=settings.parameter_local_max_norm,
		gradient_global_max_norm=settings.gradient_global_max_norm,
		validation_interval=settings.validation_interval,
	)

	model = pickle.load(open(settings.model_file, 'rb'))
	mlp = networks.MultiLayerPerceptronFromPretrainedModel(
		network._input_layer,
		pretrained_network=model
	)
	network.set_network(mlp)
	network.set_regularizers(settings.regularizer)

	resume_training(network, settings)


def start_mlpA(settings):
	# settings = config_model(mlpA_parser, mlpA_validator)
	settings = validate_config(settings)

	network = networks.AdaptiveFeedForwardNetwork(
		incoming=settings.input_shape,
		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,
		adaptable_learning_rate_policy=settings.adaptable_learning_rate,
		# adaptable_update_interval=settings.adaptable_update_interval,
		adaptable_training_mode=settings.adaptable_training_mode,
		parameter_local_max_norm=settings.parameter_local_max_norm,
		gradient_global_max_norm=settings.gradient_global_max_norm,
		validation_interval=settings.validation_interval,
	)

	mlp = networks.AdaptedMultiLayerPerceptronFromSpecifications(
		network._input_layer,
		dense_dimensions=settings.dense_dimensions,
		dense_nonlinearities=settings.dense_nonlinearities,
		layer_activation_types=settings.layer_activation_types,
		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles
	)
	network.set_network(mlp)
	network.set_regularizers(settings.regularizer)

	start_training(network, settings)


def resume_mlpA(settings):
	# settings = config_model(discriminative_adaptive_resume_parser, discriminative_adaptive_resume_validator)
	settings = validate_config(settings)

	network = networks.AdaptiveFeedForwardNetwork(
		incoming=settings.input_shape,
		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,
		adaptable_learning_rate_policy=settings.adaptable_learning_rate,
		# adaptable_update_interval=settings.adaptable_update_interval,
		adaptable_training_mode=settings.adaptable_training_mode,
		parameter_local_max_norm=settings.parameter_local_max_norm,
		gradient_global_max_norm=settings.gradient_global_max_norm,
		validation_interval=settings.validation_interval,
	)

	model = pickle.load(open(settings.model_file, 'rb'))
	mlp = networks.AdaptedMultiLayerPerceptronFromPretrainedModel(
		network._input_layer,
		pretrained_network=model
	)
	network.set_network(mlp)
	network.set_regularizers(settings.regularizer)

	resume_training(network, settings)


def main():
	import argparse
	from . import add_discriminative_options, add_resume_options, add_adaptive_options
	from . import validate_discriminative_options, validate_resume_options, validate_adaptive_options

	model_selector = argparse.ArgumentParser(description="mode selector")
	# model_selector.add_argument("--resume", dest="resume", action='store_true', default=False,
	# help="resume [None]")
	# model_selector.add_argument("--mode", dest="mode", action='store', default="start", help="mode [start]")
	# model_selector.add_argument("--model", action='store', default="start-mlp", help="model [start mlp]")

	subparsers = model_selector.add_subparsers(dest="run_model", help='model help')

	start_mlp_parser = subparsers.add_parser('start-mlp', help='start mlp model')
	start_mlp_parser = add_discriminative_options(start_mlp_parser)
	start_mlp_parser = add_dense_options(start_mlp_parser)
	start_mlp_parser = add_dropout_options(start_mlp_parser)

	resume_mlp_parser = subparsers.add_parser('resume-mlp', help='resume mlp model')
	resume_mlp_parser = add_discriminative_options(resume_mlp_parser)
	resume_mlp_parser = add_resume_options(resume_mlp_parser)

	start_mlpA_parser = subparsers.add_parser('start-mlpA', help='start adaptive mlp model')
	start_mlpA_parser = add_discriminative_options(start_mlpA_parser)
	start_mlpA_parser = add_dense_options(start_mlpA_parser)
	start_mlpA_parser = add_dropout_options(start_mlpA_parser)
	start_mlpA_parser = add_adaptive_options(start_mlpA_parser)

	resume_mlpA_parser = subparsers.add_parser('resume-mlpA', help='resume adaptive mlp model')
	resume_mlpA_parser = add_discriminative_options(resume_mlpA_parser)
	resume_mlpA_parser = add_resume_options(resume_mlpA_parser)
	resume_mlpA_parser = add_adaptive_options(resume_mlpA_parser)

	arguments, additionals = model_selector.parse_known_args()

	if len(additionals) > 0:
		print("========== ==========", "additionals", "========== ==========")
		for addition in additionals:
			print("%s" % (addition))
	# print("========== ==========", "additionals", "========== ==========")

	if arguments.run_model == "start-mlp":
		arguments = validate_discriminative_options(arguments)
		arguments, number_of_layers = validate_dense_arguments(arguments)
		number_of_layers = len(arguments.dense_dimensions)
		arguments = validate_dropout_arguments(arguments, number_of_layers)

		start_mlp(arguments)
	elif arguments.run_model == "resume-mlp":
		arguments = validate_discriminative_options(arguments)
		arguments = validate_resume_options(arguments)

		resume_mlp(arguments)
	elif arguments.run_model == "start-mlpA":
		arguments = validate_discriminative_options(arguments)
		arguments, number_of_layers = validate_dense_arguments(arguments)
		#number_of_layers = len(arguments.dense_dimensions)
		arguments = validate_dropout_arguments(arguments, number_of_layers)
		arguments = validate_adaptive_options(arguments)

		start_mlpA(arguments)
	elif arguments.run_model == "resume-mlpA":
		arguments = validate_discriminative_options(arguments)
		arguments = validate_resume_options(arguments)
		arguments = validate_adaptive_options(arguments)

		resume_mlpA(arguments)


if __name__ == '__main__':
	main()
