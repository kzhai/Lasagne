import logging

from .. import networks

logger = logging.getLogger(__name__)

__all__ = [
	"train_delman",
]


def construct_delman_parser():
	from .elman import construct_elman_parser
	model_parser = construct_elman_parser()

	model_parser.description = "dynamic elman net argument"

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
	                          action='store', default=0,
	                          help="dropout rate update interval [0=no update]")
	model_parser.add_argument('--update_hidden_layer_dropout_only', dest="update_hidden_layer_dropout_only",
	                          action='store_true', default=False,
	                          help="update hidden layer dropout only [False]")

	return model_parser


def validate_delman_arguments(arguments):
	from .elman import validate_elman_arguments
	arguments = validate_elman_arguments(arguments)

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

	# model argument set 2
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
		sequence_length=settings.sequence_length,

		layer_dimensions=settings.layer_dimensions,
		layer_nonlinearities=settings.layer_nonlinearities,

		vocabulary_dimension=settings.vocabulary_dimension,
		embedding_dimension=settings.embedding_dimension,

		recurrent_type=settings.recurrent_type,

		window_size=settings.window_size,
		position_offset=settings.position_offset,
		# gradient_steps=settings.gradient_steps,
		# gradient_clipping=settings.gradient_clipping,

		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles,

		objective_functions=settings.objective,
		update_function=settings.update,
		# pretrained_model=pretrained_model

		learning_rate_policy=settings.learning_rate,
		#learning_rate_decay=settings.learning_rate_decay,

		dropout_learning_rate=settings.dropout_learning_rate,
		#dropout_learning_rate_decay=settings.dropout_learning_rate_decay,
		dropout_rate_update_interval=settings.dropout_rate_update_interval,
		update_hidden_layer_dropout_only=settings.update_hidden_layer_dropout_only,

		total_norm_constraint=settings.total_norm_constraint,
		normalize_embeddings=settings.normalize_embeddings,

		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,
	)

	network.set_regularizers(settings.regularizer)

	from . import train_model
	train_model(network, settings, network.parse_sequence)


if __name__ == '__main__':
	train_delman()
