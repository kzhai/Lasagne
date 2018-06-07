import logging
import os

import numpy
import theano

logger = logging.getLogger(__name__)

from lasagne import nonlinearities, networks, layers
from . import layer_deliminator
from . import start_training

__all__ = [
	"add_elman_options",
	"validate_elman_arguments",
	#
]


def add_elman_options(model_parser):
	# model argument set 1
	model_parser.add_argument("--layer_dimensions", dest="layer_dimensions", action='store', default=None,
	                          help="dimension of different layer [None], example, '100,500,10' represents 3 layers contains 100, 500, and 10 neurons respectively")
	model_parser.add_argument("--layer_nonlinearities", dest="layer_nonlinearities", action='store', default=None,
	                          help="activation functions of different layer [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively")

	# model argument set 2
	model_parser.add_argument("--window_size", dest="window_size", action='store', default=1, type=int,
	                          help="window size [1] for local aggregation or transformation")
	model_parser.add_argument("--position_offset", dest="position_offset", action='store', default=-1, type=int,
	                          help="position offset of current word in window [-1=window_size/2]")
	model_parser.add_argument("--sequence_length", dest="sequence_length", action='store', default=10, type=int,
	                          help="longest sequnece length for back propagation steps [10]")

	# model argument set 3
	model_parser.add_argument("--embedding_dimension", dest="embedding_dimension", action='store', default=-1, type=int,
	                          help="dimension of word embedding layer [-1]")

	# model argument set 4
	model_parser.add_argument("--recurrent_type", dest="recurrent_type", action='store', default='LSTMLayer',
	                          help="recurrent layer type [default=LSTMLayer]")
	model_parser.add_argument("--total_norm_constraint", dest="total_norm_constraint", action='store', default=0,
	                          type=float, help="total norm constraint [0]")

	'''
	It is highly recomended that---in this implementation---to realize backward pass through the concept of input mask (hence to support different input sequence lenght).
	The settings for gradient_steps should be less than the settings for sequence_length, and larger than -1.
	'''
	'''
	model_parser.add_argument("--gradient_steps", dest="gradient_steps", action='store', default=-1, type=int,
							  help="number of timesteps to include in the backpropagated gradient [-1 = backpropagate through the entire sequence]")
	model_parser.add_argument("--gradient_clipping", dest="gradient_clipping", action='store', default=0, type=float,
							  help="if nonzero, the gradient messages are clipped to the given value during the backward pass [0]")
	'''

	model_parser.add_argument('--normalize_embeddings', dest="normalize_embeddings", action='store_true', default=False,
	                          help="normalize embeddings after each mini-batch [False]")

	return model_parser


def validate_elman_arguments(arguments):
	# model argument set 1
	number_of_recurrent_layers = 0
	number_of_dense_layers = 0

	assert arguments.layer_dimensions is not None
	options_layer_dimensions = arguments.layer_dimensions + layer_deliminator
	layer_dimensions = []
	recurrent_mode = False
	start_index = 0
	for end_index in range(len(options_layer_dimensions)):
		if options_layer_dimensions[end_index] == "[":
			assert (not recurrent_mode)
			recurrent_mode = True
			start_index = end_index + 1
		elif options_layer_dimensions[end_index] == "]":
			assert recurrent_mode
			layer_dimensions.append([int(options_layer_dimensions[start_index:end_index])])
			recurrent_mode = False
			start_index = end_index + 1
		elif options_layer_dimensions[end_index] == layer_deliminator:
			if end_index > start_index:
				if recurrent_mode:
					layer_dimensions.append([int(options_layer_dimensions[start_index:end_index])])
					number_of_recurrent_layers += 1
				else:
					layer_dimensions.append(int(options_layer_dimensions[start_index:end_index]))
					number_of_dense_layers += 1
			start_index = end_index + 1
	arguments.layer_dimensions = layer_dimensions

	assert arguments.layer_nonlinearities is not None
	options_layer_nonlinearities = arguments.layer_nonlinearities + layer_deliminator
	layer_nonlinearities = []
	recurrent_mode = False
	start_index = 0
	for end_index in range(len(options_layer_nonlinearities)):
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
		elif options_layer_nonlinearities[end_index] == layer_deliminator:
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
	# assert arguments.recurrent_style in ["elman", "bi-elman"]
	arguments.recurrent_type = getattr(layers.recurrent, arguments.recurrent_type)
	assert arguments.total_norm_constraint >= 0
	# assert arguments.gradient_steps >= -1 and arguments.gradient_steps < arguments.sequence_length
	# assert arguments.gradient_clipping >= 0

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
	assert arguments.label_dimension == arguments.layer_dimensions[-1], (
		arguments.label_dimension, arguments.layer_dimensions)

	return arguments, number_of_dense_layers + 1


def elman_parser():
	from . import add_discriminative_options, add_dropout_options

	model_parser = add_discriminative_options()
	model_parser = add_elman_options(model_parser)
	# model_parser = add_dense_options(model_parser)
	model_parser = add_dropout_options(model_parser)

	return model_parser


def elman_validator(arguments):
	from . import validate_discriminative_options, validate_dropout_arguments

	arguments = validate_discriminative_options(arguments)
	arguments = validate_elman_arguments(arguments)
	number_of_dense_layers = len(arguments.dense_dimensions)
	number_of_layers = number_of_dense_layers + 1
	arguments = validate_dropout_arguments(arguments, number_of_layers)

	return arguments


#
#
#
#
#


def load_data(self, input_file, sequence_length, window_size=1, position_offset=-1):
	data_set = numpy.load(input_file)
	assert len(data_set.shape) == 1
	data_set_x = numpy.zeros((data_set.shape[0] - sequence_length - 1, sequence_length))
	data_set_y = numpy.zeros((data_set.shape[0] - sequence_length - 1, sequence_length))

	for index in range(data_set.shape[0] - sequence_length):
		data_set_x[index, :] = data_set[index:index + sequence_length]
		data_set_y[index, :] = data_set[index + 1:index + sequence_length + 1]

	return (data_set_x, data_set_y)


'''
def validate_config(settings):
	test_dataset = load_data(settings.input_directory, dataset="test")
	test_set_x, test_set_y = test_dataset
	input_shape = list(test_set_x.shape[1:])
	input_shape.insert(0, None)
	input_shape = tuple(input_shape)
	settings.input_shape = input_shape

	return settings
'''


def start_elman(settings):
	# settings = validate_config(settings)

	# input_shape = (None, settings.sequence_length, settings.window_size)
	input_shape = (None, settings.sequence_length)
	# input_layer = layers.InputLayer(shape=input_shape, input_var=theano.tensor.itensor3())
	input_layer = layers.InputLayer(shape=input_shape, input_var=theano.tensor.imatrix())

	# input_mask_shape = (None, settings.sequence_length)
	# input_mask_layer = layers.InputLayer(shape=input_mask_shape, input_var=theano.tensor.imatrix())

	# self._recurrent_type = recurrent_type
	# self._gradient_steps = gradient_steps
	# self._gradient_clipping = gradient_clipping

	# self._sequence_length = sequence_length

	# self._position_offset = position_offset

	network = networks.RecurrentNetwork(
		incoming=input_layer,
		# incoming_mask=input_mask_layer,

		objective_functions=settings.objective,
		update_function=settings.update,
		learning_rate_policy=settings.learning_rate,

		max_norm_constraint=settings.max_norm_constraint,
		validation_interval=settings.validation_interval,

		total_norm_constraint=0,
		normalize_embeddings=False,

		sequence_length=settings.sequence_length,
		window_size=settings.window_size,
		position_offset=settings.position_offset,

		# gradient_steps=-1,
		# gradient_clipping=0,
	)

	elman = networks.ElmanNetworkFromSpecifications(
		input_layer=network._input_layer,

		vocabulary_dimension=settings.vocabulary_dimension,
		embedding_dimension=settings.embedding_dimension,
		sequence_length=settings.sequence_length,

		layer_dimensions=settings.layer_dimensions,
		layer_nonlinearities=settings.layer_nonlinearities,

		recurrent_type=settings.recurrent_type,

		layer_activation_types=settings.layer_activation_types,
		layer_activation_parameters=settings.layer_activation_parameters,
		layer_activation_styles=settings.layer_activation_styles,

		input_mask_layer=None,
		window_size=settings.window_size,

		# gradient_steps=settings.gradient_steps,
		# gradient_clipping=settings.gradient_clipping,
	)

	network.set_network(elman)
	network.set_regularizers(settings.regularizer)

	start_training(network, settings)


#
#
#
#
#


def main():
	import argparse
	from . import add_discriminative_options, add_resume_options, add_adaptive_options, add_dropout_options
	from . import validate_discriminative_options, validate_resume_options, validate_adaptive_options, \
		validate_dropout_arguments

	model_selector = argparse.ArgumentParser(description="mode selector")

	subparsers = model_selector.add_subparsers(dest="run_model", help='model help')

	start_elman_parser = subparsers.add_parser('start-elman', help='start elman model')
	start_elman_parser = add_discriminative_options(start_elman_parser)
	start_elman_parser = add_elman_options(start_elman_parser)
	# start_elman_parser = add_dense_options(start_elman_parser)
	start_elman_parser = add_dropout_options(start_elman_parser)

	resume_elman_parser = subparsers.add_parser('resume-elman', help='resume elman model')
	resume_elman_parser = add_discriminative_options(resume_elman_parser)
	resume_elman_parser = add_resume_options(resume_elman_parser)

	start_elmanA_parser = subparsers.add_parser('start-elmanA', help='start adaptive elman model')
	start_elmanA_parser = add_discriminative_options(start_elmanA_parser)
	start_elmanA_parser = add_elman_options(start_elmanA_parser)
	# start_elmanA_parser = add_dense_options(start_elmanA_parser)
	start_elmanA_parser = add_dropout_options(start_elmanA_parser)
	start_elmanA_parser = add_adaptive_options(start_elmanA_parser)

	resume_elmanA_parser = subparsers.add_parser('resume-elmanA', help='resume adaptive elman model')
	resume_elmanA_parser = add_discriminative_options(resume_elmanA_parser)
	resume_elmanA_parser = add_resume_options(resume_elmanA_parser)
	resume_elmanA_parser = add_adaptive_options(resume_elmanA_parser)

	arguments, additionals = model_selector.parse_known_args()

	if len(additionals) > 0:
		print("========== ==========", "additionals", "========== ==========")
		for addition in additionals:
			print("%s" % (addition))
	# print("========== ==========", "additionals", "========== ==========")

	if arguments.run_model == "start-elman":
		arguments = validate_discriminative_options(arguments)
		arguments, number_of_dense_layers = validate_elman_arguments(arguments)
		arguments = validate_dropout_arguments(arguments, number_of_dense_layers)

		start_elman(arguments)
	elif arguments.run_model == "resume-elman":
		arguments = validate_discriminative_options(arguments)
		arguments = validate_resume_options(arguments)

	# resume_elman(arguments)
	elif arguments.run_model == "start-elmanA":
		arguments = validate_discriminative_options(arguments)
		arguments, number_of_dense_layers = validate_elman_arguments(arguments)
		arguments = validate_dropout_arguments(arguments, number_of_dense_layers)
		arguments = validate_adaptive_options(arguments)

	# start_elmanA(arguments)
	elif arguments.run_model == "resume-elmanA":
		arguments = validate_discriminative_options(arguments)
		arguments = validate_resume_options(arguments)
		arguments = validate_adaptive_options(arguments)


# resume_elmanA(arguments)


if __name__ == '__main__':
	main()
