import logging
import os

import numpy

from .. import layers
from .. import networks

logger = logging.getLogger(__name__)

__all__ = [
	"train_dmlp",
]


def construct_dmlp_parser():
	from .mlp import construct_mlp_parser
	model_parser = construct_mlp_parser()

	model_parser.description = "dynamic multi-layer perceptron argument"

	# model argument set 1
	'''
	model_parser.add_argument("--dropout_learning_rate", dest="dropout_learning_rate", type=float, action='store',
	                          default=0, help="dropout learning rate [0 = learning_rate]")
	model_parser.add_argument("--dropout_learning_rate_decay", dest="dropout_learning_rate_decay", action='store',
	                          default=None, help="dropout learning rate decay [None = learning_rate_decay]")
	'''
	model_parser.add_argument("--dropout_learning_rate", dest="dropout_learning_rate", action='store',
	                          default=None, help="dropout learning rate [None = learning_rate]")

	# model argument set 2
	model_parser.add_argument("--dropout_rate_update_interval", dest="dropout_rate_update_interval", type=int,
	                          action='store', default=0, help="dropout rate update interval [1]")
	model_parser.add_argument('--update_hidden_layer_dropout_only', dest="update_hidden_layer_dropout_only",
	                          action='store_true', default=False, help="update hidden layer dropout only [False]")

	return model_parser


def validate_dmlp_arguments(arguments):
	from .mlp import validate_mlp_arguments
	arguments = validate_mlp_arguments(arguments)

	# model argument set 1
	if arguments.dropout_learning_rate is None:
		arguments.dropout_learning_rate = arguments.learning_rate;
	else:
		dropout_learning_rate_tokens = arguments.dropout_learning_rate.split(",")
		dropout_learning_rate_tokens[0] = float(dropout_learning_rate_tokens[0]);
		assert dropout_learning_rate_tokens[0] > 0
		if len(dropout_learning_rate_tokens) == 1:
			pass
		elif len(dropout_learning_rate_tokens) == 5:
			assert dropout_learning_rate_tokens[1] in ["iteration", "epoch"]
			assert dropout_learning_rate_tokens[2] in ["inverse_t", "exponential", "step"]
			dropout_learning_rate_tokens[3] = float(dropout_learning_rate_tokens[3])
			dropout_learning_rate_tokens[4] = float(dropout_learning_rate_tokens[4])
		else:
			logger.error("unrecognized dropout learning rate %s..." % (arguments.dropout_learning_rate))
		arguments.dropout_learning_rate = dropout_learning_rate_tokens

	'''
	if arguments.dropout_learning_rate == 0:
		arguments.dropout_learning_rate = arguments.learning_rate;
	assert arguments.dropout_learning_rate > 0
	if arguments.dropout_learning_rate_decay is None:
		arguments.dropout_learning_rate_decay = arguments.learning_rate_decay;
	else:
		dropout_learning_rate_decay_tokens = arguments.dropout_learning_rate_decay.split(",")
		assert len(dropout_learning_rate_decay_tokens) == 4
		assert dropout_learning_rate_decay_tokens[0] in ["iteration", "epoch"]
		assert dropout_learning_rate_decay_tokens[1] in ["inverse_t", "exponential", "step"]
		dropout_learning_rate_decay_tokens[2] = float(dropout_learning_rate_decay_tokens[2])
		dropout_learning_rate_decay_tokens[3] = float(dropout_learning_rate_decay_tokens[3])
		arguments.dropout_learning_rate_decay = dropout_learning_rate_decay_tokens
	'''

	# model argument set 2
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

		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles,

		objective_functions=settings.objective,
		update_function=settings.update,
		# pretrained_model=pretrained_model

		learning_rate_policy=settings.learning_rate,
		#learning_rate_decay=settings.learning_rate_decay,

		dropout_learning_rate_policy=settings.dropout_learning_rate,
		#dropout_learning_rate_decay=settings.dropout_learning_rate_decay,
		dropout_rate_update_interval=settings.dropout_rate_update_interval,
		update_hidden_layer_dropout_only=settings.update_hidden_layer_dropout_only,

		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)

	network.set_regularizers(settings.regularizer)
	# network.set_L1_regularizer_lambda(settings.L1_regularizer_lambdas)
	# network.set_L2_regularizer_lambda(settings.L2_regularizer_lambdas)

	from . import train_model
	train_model(network, settings)


def snapshot_retain_rates(network, output_directory):
	dropout_layer_index = 0
	for network_layer in network.get_network_layers():
		if not isinstance(network_layer, layers.AdaptiveDropoutLayer):
			continue

		layer_retain_probability = network_layer.activation_probability.eval()
		logger.info("retain rates stats: epoch %i, shape %s, average %f, minimum %f, maximum %f" % (
			network.epoch_index,
			layer_retain_probability.shape,
			numpy.mean(layer_retain_probability),
			numpy.min(layer_retain_probability),
			numpy.max(layer_retain_probability)))

		retain_rate_file = os.path.join(output_directory,
		                                "layer.%d.epoch.%d.npy" % (dropout_layer_index, network.epoch_index))
		numpy.save(retain_rate_file, layer_retain_probability)
		dropout_layer_index += 1


def resume_dmlp():
	pass


def test_dmlp():
	pass


if __name__ == '__main__':
	train_dmlp()
