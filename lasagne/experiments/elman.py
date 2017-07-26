import os

from .. import layers, networks, nonlinearities

__all__ = [
	"train_elman",
]


def construct_elman_parser():
	from . import construct_discriminative_parser, add_dropout_options

	model_parser = construct_discriminative_parser()
	model_parser = add_dropout_options(model_parser)

	# model argument set 1
	model_parser.add_argument("--layer_dimensions", dest="layer_dimensions", action='store', default=None,
	                          help="dimension of different layer [None], example, '100,500,10' represents 3 layers contains 100, 500, and 10 neurons respectively")
	model_parser.add_argument("--layer_nonlinearities", dest="layer_nonlinearities", action='store', default=None,
	                          help="activation functions of different layer [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively")

	# model argument set 2
	model_parser.add_argument("--window_size", dest="window_size", action='store', default=1, type=int,
	                          help="window size [1]")
	model_parser.add_argument("--position_offset", dest="position_offset", action='store', default=-1, type=int,
	                          help="position offset of current word in window [-1=window_size/2]")
	model_parser.add_argument("--sequence_length", dest="sequence_length", action='store', default=100, type=int,
	                          help="longest sequnece length for back propagation steps [100]")

	# model argument set 3
	# parser.add_option("--vocabulary_dimension", type="int", dest="vocabulary_dimension",
	# help="vocabulary size [-1]")
	model_parser.add_argument("--embedding_dimension", dest="embedding_dimension", action='store', default=-1, type=int,
	                          help="dimension of word embedding layer [-1]")
	# model argument set 4
	#model_parser.add_argument("--recurrent_style", dest="recurrent_style", action='store', default='elman',
	                          #help="recurrent network style [default=elman, bi-elman]")
	model_parser.add_argument("--recurrent_type", dest="recurrent_type", action='store', default='LSTMLayer',
	                          help="recurrent layer type [default=RecurrentLayer, LSTMLayer]")

	return model_parser


def validate_elman_arguments(arguments):
	from . import validate_discriminative_arguments, validate_dropout_arguments

	arguments = validate_discriminative_arguments(arguments)

	# model argument set 1
	number_of_recurrent_layers = 0
	number_of_dense_layers = 0

	assert arguments.layer_dimensions != None
	options_layer_dimensions = arguments.layer_dimensions + ","
	layer_dimensions = []
	recurrent_mode = False
	start_index = 0
	for end_index in xrange(len(options_layer_dimensions)):
		if options_layer_dimensions[end_index] == "[":
			assert (not recurrent_mode)
			recurrent_mode = True
			start_index = end_index + 1
		elif options_layer_dimensions[end_index] == "]":
			assert recurrent_mode
			layer_dimensions.append([int(options_layer_dimensions[start_index:end_index])])
			recurrent_mode = False
			start_index = end_index + 1
		elif options_layer_dimensions[end_index] == ",":
			if end_index > start_index:
				if recurrent_mode:
					layer_dimensions.append([int(options_layer_dimensions[start_index:end_index])])
					number_of_recurrent_layers += 1
				else:
					layer_dimensions.append(int(options_layer_dimensions[start_index:end_index]))
					number_of_dense_layers += 1
			start_index = end_index + 1
	arguments.layer_dimensions = layer_dimensions

	assert arguments.layer_nonlinearities != None
	options_layer_nonlinearities = arguments.layer_nonlinearities + ","
	layer_nonlinearities = []
	recurrent_mode = False
	start_index = 0
	for end_index in xrange(len(options_layer_nonlinearities)):
		if options_layer_nonlinearities[end_index] == "[":
			assert (not recurrent_mode)
			recurrent_mode = True
			start_index = end_index + 1
		elif options_layer_nonlinearities[end_index] == "]":
			assert recurrent_mode
			layer_nonlinearities.append(
				[getattr(nonlinearities, options_layer_nonlinearities[start_index:end_index])])
			recurrent_mode = False
			start_index = end_index + 1
		elif options_layer_nonlinearities[end_index] == ",":
			if end_index > start_index:
				if recurrent_mode:
					layer_nonlinearities.append(
						[getattr(nonlinearities, options_layer_nonlinearities[start_index:end_index])])
				else:
					layer_nonlinearities.append(
						getattr(nonlinearities, options_layer_nonlinearities[start_index:end_index]))
			start_index = end_index + 1
	arguments.layer_nonlinearities = layer_nonlinearities

	'''
	assert arguments.dense_dimensions != None
	dense_dimensions = arguments.dense_dimensions.split(",")
	arguments.dense_dimensions = [int(dimensionality) for dimensionality in dense_dimensions]

	assert arguments.dense_nonlinearities != None
	dense_nonlinearities = arguments.dense_nonlinearities.split(",")
	arguments.dense_nonlinearities = [getattr(nonlinearities, dense_nonlinearity) for dense_nonlinearity in
	                                  dense_nonlinearities]

	assert len(arguments.dense_nonlinearities) == len(arguments.dense_dimensions)
	'''

	arguments = validate_dropout_arguments(arguments, number_of_dense_layers)

	# model argument set 2
	assert arguments.window_size > 0
	# assert options.window_size % 2 == 1
	if arguments.position_offset < 0:
		arguments.position_offset = arguments.window_size / 2
	assert arguments.position_offset < arguments.window_size
	assert arguments.sequence_length > 0

	# model argument set 3
	assert arguments.embedding_dimension > 0

	# model argument set 4
	#assert arguments.recurrent_style in ["elman", "bi-elman"]
	arguments.recurrent_type = getattr(layers.recurrent, arguments.recurrent_type)

	vocabulary_dimension = 0
	for line in open(os.path.join(arguments.input_directory, "type.info"), 'r'):
		vocabulary_dimension += 1
	# this is to include a dummy entry for out-of-vocabulary type
	vocabulary_dimension += 1
	arguments.vocabulary_dimension = vocabulary_dimension

	label_dimension = 0
	for line in open(os.path.join(arguments.input_directory, "label.info"), 'r'):
		label_dimension += 1
	arguments.label_dimension = label_dimension
	assert arguments.label_dimension == arguments.layer_dimensions[-1]

	return arguments


def train_elman():
	from . import config_model
	settings = config_model(construct_elman_parser, validate_elman_arguments)
	# settings = validate_config(settings)

	'''
	layer_latent_feature_alphas = options.layer_latent_feature_alphas
	if layer_latent_feature_alphas is not None:
		layer_latent_feature_alpha_tokens = layer_latent_feature_alphas.split(",")
		if len(layer_latent_feature_alpha_tokens) == 1:
			layer_latent_feature_alphas = [float(layer_latent_feature_alphas) for layer_index in xrange(number_of_layers)]
		else:
			assert len(layer_latent_feature_alpha_tokens) == number_of_layers
			layer_latent_feature_alphas = [float(layer_latent_feature_alpha) for layer_latent_feature_alpha in layer_latent_feature_alpha_tokens]
	else:
		layer_latent_feature_alphas = [0 for layer_index in xrange(number_of_layers)]
	assert (layer_latent_feature_alpha >= 0 for layer_latent_feature_alpha in layer_latent_feature_alphas)
	'''

	network = networks.ElmanNetwork(
		# incoming=settings.input_shape,
		# incoming_mask=settings.input_mask_shape,

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
		validation_interval=settings.validation_interval,
	)

	network.set_regularizers(settings.regularizer)

	# network.set_L1_regularizer_lambda(L1_regularizer_lambdas)
	# network.set_L2_regularizer_lambda(L2_regularizer_lambdas)
	# network.set_dae_regularizer_lambda(dae_regularizer_lambdas, layer_corruption_levels)

	from . import train_model
	train_model(network, settings, network.parse_sequence)


if __name__ == '__main__':
	train_elman()
