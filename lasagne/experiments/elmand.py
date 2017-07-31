import logging
import os

import numpy

from .. import layers
from .. import networks

logger = logging.getLogger(__name__)

__all__ = [
	"train_delman",
]


def construct_delman_parser():
	from .elman import construct_elman_parser
	model_parser = construct_elman_parser()

	model_parser.description = "convolutional dynamic elman net argument"

	# model argument set
	model_parser.add_argument("--dropout_rate_update_interval", dest="dropout_rate_update_interval", type=int,
	                          action='store', default=0,
	                          help="dropout rate update interval [0=no update]")
	model_parser.add_argument('--update_hidden_layer_dropout_only', dest="update_hidden_layer_dropout_only",
	                          action='store_true', default=False,
	                          help="update hidden layer dropout only [False]")

	return model_parser


def validate_delman_arguments(arguments):
	from .elman import validate_elman_arguments
	arguments = validate_elman_arguments(arguments)

	# model argument set
	assert (arguments.dropout_rate_update_interval >= 0)

	return arguments


def train_delman():
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer perceptron
	This is demonstrated on MNIST.
	"""

	from . import config_model
	settings = config_model(construct_delman_parser, validate_delman_arguments)

	network = networks.DynamicElmanNetwork(
		window_size=settings.window_size,
		sequence_length=settings.sequence_length,

		layer_dimensions=settings.layer_dimensions,
		layer_nonlinearities=settings.layer_nonlinearities,

		position_offset=settings.position_offset,

		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles,

		vocabulary_dimension=settings.vocabulary_dimension,
		embedding_dimension=settings.embedding_dimension,
		recurrent_type=settings.recurrent_type,

		objective_functions=settings.objective,
		update_function=settings.update,
		# pretrained_model=pretrained_model

		learning_rate=settings.learning_rate,
		learning_rate_decay=settings.learning_rate_decay,
		max_norm_constraint=settings.max_norm_constraint,

		dropout_rate_update_interval=settings.dropout_rate_update_interval,
		update_hidden_layer_dropout_only=settings.update_hidden_layer_dropout_only,

		validation_interval=settings.validation_interval,
	)

	network.set_regularizers(settings.regularizer)

	from . import train_model
	train_model(network, settings, network.parse_sequence)


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


if __name__ == '__main__':
	train_delman()
