import logging
import sys
import timeit

import numpy
import theano
import theano.tensor

from . import RecurrentNetwork
from .. import init, nonlinearities, objectives, updates
from .. import layers
from ..layers import noise

__all__ = [
	"ElmanNetwork",
	"DynamicElmanNetwork"
]


class ElmanNetwork(RecurrentNetwork):
	def __init__(self,
	             # incoming,
	             # incoming_mask,

	             window_size,
	             sequence_length,

	             layer_dimensions,
	             layer_nonlinearities,

	             position_offset=-1,

	             layer_activation_parameters=None,
	             layer_activation_styles=None,

	             vocabulary_dimension=None,
	             embedding_dimension=None,
	             recurrent_type=None,

	             objective_functions=objectives.categorical_crossentropy,
	             update_function=updates.nesterov_momentum,
	             learning_rate=1e-3,
	             learning_rate_decay=None,
	             max_norm_constraint=0,

	             validation_interval=-1,
	             ):
		# input_shape = (None, sequence_length, window_size)
		# input_mask_shape = (None, sequence_length)
		super(ElmanNetwork, self).__init__(
			# input_shape,
			# input_mask_shape,

			window_size,
			sequence_length,
			position_offset,

			objective_functions,
			update_function,
			learning_rate,
			learning_rate_decay,
			max_norm_constraint,
			validation_interval,
		)

		# x = theano.tensor.matrix('x')  # the data is presented as rasterized images
		# self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

		neural_network = self._input_layer

		# print_output_dimension("after input merge", neural_network, sequence_length, window_size)

		# batch_size, input_sequence_length, input_window_size = layers.get_output_shape(neural_network)
		# assert sequence_length > 0 and sequence_length == input_sequence_length
		# assert window_size > 0 and window_size == input_window_size

		self._embedding_layer = layers.EmbeddingLayer(neural_network,
		                                              input_size=vocabulary_dimension,
		                                              output_size=embedding_dimension,
		                                              W=init.GlorotNormal())
		neural_network = self._embedding_layer
		# print_output_dimension("after embedding layer", neural_network, sequence_length, window_size)

		neural_network = layers.ReshapeLayer(neural_network,
		                                     (-1, self._sequence_length, self._window_size * embedding_dimension))
		# print_output_dimension("after window merge", neural_network, sequence_length, window_size)

		#
		#
		#
		#
		#

		'''
		input_layer_shape = layers.get_output_shape(neural_network)[1:]
		previous_layer_shape = numpy.prod(input_layer_shape)

		activation_probability = sample_activation_probability(previous_layer_shape,
		                                                       dense_activation_styles[dropout_layer_index],
		                                                       dense_activation_parameters[
			                                                       dropout_layer_index])
		activation_probability = numpy.reshape(activation_probability, input_layer_shape)
		dropout_layer_index += 1

		neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability)
		'''

		last_rnn_layer_index = 0
		for layer_index in xrange(len(layer_dimensions)):
			layer_dimension = layer_dimensions[layer_index]
			if isinstance(layer_dimension, list):
				last_rnn_layer_index = layer_index
		dropout_layer_index = 0

		for layer_index in xrange(len(layer_dimensions)):
			layer_dimension = layer_dimensions[layer_index]
			layer_nonlinearity = layer_nonlinearities[layer_index]

			if isinstance(layer_dimension, int):
				if not isinstance(neural_network, layers.DenseLayer):
					previous_layer_dimension = layers.get_output_shape(neural_network)
					if layer_index > last_rnn_layer_index:
						neural_network = layers.ReshapeLayer(neural_network,
						                                     (-1, numpy.prod(previous_layer_dimension[1:])))
					else:
						neural_network = layers.ReshapeLayer(neural_network,
						                                     (-1, previous_layer_dimension[-1]))

					# print_output_dimension("after reshape (for dense layer)", neural_network, sequence_length, window_size)

				previous_layer_dimension = layers.get_output_shape(neural_network)
				activation_probability = noise.sample_activation_probability(previous_layer_dimension[-1],
				                                                             layer_activation_styles[
					                                                             dropout_layer_index],
				                                                             layer_activation_parameters[
					                                                             dropout_layer_index])
				neural_network = noise.LinearDropoutLayer(neural_network, activation_probability=activation_probability)
				dropout_layer_index += 1

				neural_network = layers.DenseLayer(neural_network,
				                                   layer_dimension,
				                                   W=init.GlorotUniform(
					                                   gain=init.GlorotUniformGain[layer_nonlinearity]),
				                                   nonlinearity=layer_nonlinearity)
			# print_output_dimension("after dense layer %i" % layer_index, neural_network, sequence_length, window_size)

			elif isinstance(layer_dimension, list):
				# if not isinstance(layers.get_all_layers(neural_network)[-1], recurrent_type):
				if not isinstance(neural_network, recurrent_type):
					previous_layer_dimension = layers.get_output_shape(neural_network)
					neural_network = layers.ReshapeLayer(neural_network,
					                                     (-1, self._sequence_length, previous_layer_dimension[-1]))
				# print_output_dimension("after reshape (for recurrent layer)", neural_network, sequence_length, window_size)

				layer_dimension = layer_dimension[0]
				layer_nonlinearity = layer_nonlinearity[0]
				if recurrent_type == layers.recurrent.RecurrentLayer:
					neural_network = layers.RecurrentLayer(neural_network,
					                                       layer_dimension,
					                                       W_in_to_hid=init.GlorotUniform(
						                                       gain=init.GlorotUniformGain[layer_nonlinearity]),
					                                       W_hid_to_hid=init.GlorotUniform(
						                                       gain=init.GlorotUniformGain[layer_nonlinearity]),
					                                       b=init.Constant(0.),
					                                       nonlinearity=layer_nonlinearity,
					                                       hid_init=init.Constant(0.),
					                                       backwards=False,
					                                       learn_init=False,
					                                       gradient_steps=-1,
					                                       grad_clipping=0,
					                                       unroll_scan=False,
					                                       precompute_input=True,
					                                       mask_input=self._input_mask_layer,
					                                       # only_return_final=True
					                                       )
				elif recurrent_type == layers.recurrent.LSTMLayer:
					neural_network = layers.LSTMLayer(neural_network,
					                                  layer_dimension,
					                                  ingate=layers.Gate(),
					                                  forgetgate=layers.Gate(),
					                                  cell=layers.Gate(W_cell=None, nonlinearity=layer_nonlinearity),
					                                  outgate=layers.Gate(),
					                                  nonlinearity=layer_nonlinearity,
					                                  cell_init=init.Constant(0.),
					                                  hid_init=init.Constant(0.),
					                                  backwards=False,
					                                  learn_init=False,
					                                  peepholes=True,
					                                  gradient_steps=-1,
					                                  grad_clipping=0,
					                                  unroll_scan=False,
					                                  precompute_input=True,
					                                  mask_input=self._input_mask_layer,
					                                  # only_return_final=True
					                                  )
				# print_output_dimension("after recurrent layer %i" % layer_index, neural_network, sequence_length, window_size)
			else:
				logging.error("Unrecognized layer specifications...")
				sys.stderr.write("Unrecognized layer specifications...\n")
				sys.exit()

		self._neural_network = neural_network

		self.build_functions()


class DynamicElmanNetwork(RecurrentNetwork):
	def __init__(self,
	             # incoming,
	             # incoming_mask,

	             window_size,
	             sequence_length,

	             layer_dimensions,
	             layer_nonlinearities,

	             position_offset=-1,

	             layer_activation_parameters=None,
	             layer_activation_styles=None,

	             vocabulary_dimension=None,
	             embedding_dimension=None,
	             recurrent_type=None,

	             objective_functions=objectives.categorical_crossentropy,
	             update_function=updates.nesterov_momentum,

	             learning_rate=1e-3,
	             learning_rate_decay=None,
	             max_norm_constraint=0,

	             dropout_rate_update_interval=-1,
	             update_hidden_layer_dropout_only=False,

	             validation_interval=-1,
	             ):
		# input_shape = (None, sequence_length, window_size)
		# input_mask_shape = (None, sequence_length)
		super(DynamicElmanNetwork, self).__init__(
			# input_shape,
			# input_mask_shape,

			window_size,
			sequence_length,
			position_offset,

			objective_functions,
			update_function,
			learning_rate,
			learning_rate_decay,
			max_norm_constraint,
			validation_interval,
		)

		self._dropout_rate_update_interval = dropout_rate_update_interval

		# x = theano.tensor.matrix('x')  # the data is presented as rasterized images
		# self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

		neural_network = self._input_layer

		# print_output_dimension("after input merge", neural_network, sequence_length, window_size)

		# batch_size, input_sequence_length, input_window_size = layers.get_output_shape(neural_network)
		# assert sequence_length > 0 and sequence_length == input_sequence_length
		# assert window_size > 0 and window_size == input_window_size

		self._embedding_layer = layers.EmbeddingLayer(neural_network,
		                                              input_size=vocabulary_dimension,
		                                              output_size=embedding_dimension,
		                                              W=init.GlorotNormal())
		neural_network = self._embedding_layer
		# print_output_dimension("after embedding layer", neural_network, sequence_length, window_size)

		neural_network = layers.ReshapeLayer(neural_network,
		                                     (-1, self._sequence_length, self._window_size * embedding_dimension))
		# print_output_dimension("after window merge", neural_network, sequence_length, window_size)

		#
		#
		#
		#
		#

		'''
		input_layer_shape = layers.get_output_shape(neural_network)[1:]
		previous_layer_shape = numpy.prod(input_layer_shape)

		activation_probability = sample_activation_probability(previous_layer_shape,
		                                                       dense_activation_styles[dropout_layer_index],
		                                                       dense_activation_parameters[
			                                                       dropout_layer_index])
		activation_probability = numpy.reshape(activation_probability, input_layer_shape)
		dropout_layer_index += 1

		neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability)
		'''

		last_rnn_layer_index = 0
		for layer_index in xrange(len(layer_dimensions)):
			layer_dimension = layer_dimensions[layer_index]
			if isinstance(layer_dimension, list):
				last_rnn_layer_index = layer_index
		dropout_layer_index = 0

		for layer_index in xrange(len(layer_dimensions)):
			layer_dimension = layer_dimensions[layer_index]
			layer_nonlinearity = layer_nonlinearities[layer_index]

			if isinstance(layer_dimension, int):
				if not isinstance(neural_network, layers.DenseLayer):
					previous_layer_dimension = layers.get_output_shape(neural_network)
					if layer_index > last_rnn_layer_index:
						neural_network = layers.ReshapeLayer(neural_network,
						                                     (-1, numpy.prod(previous_layer_dimension[1:])))
					else:
						neural_network = layers.ReshapeLayer(neural_network,
						                                     (-1, previous_layer_dimension[-1]))

					# print_output_dimension("after reshape (for dense layer)", neural_network, sequence_length, window_size)

				previous_layer_dimension = layers.get_output_shape(neural_network)
				activation_probability = noise.sample_activation_probability(previous_layer_dimension[-1],
				                                                             layer_activation_styles[
					                                                             dropout_layer_index],
				                                                             layer_activation_parameters[
					                                                             dropout_layer_index])

				if update_hidden_layer_dropout_only and dropout_layer_index == 0:
					neural_network = noise.LinearDropoutLayer(neural_network,
					                                          activation_probability=activation_probability)
				else:
					# neural_network = noise.TrainableDropoutLayer(neural_network, activation_probability=init.Constant(layer_activation_parameters[layer_index]))
					neural_network = noise.AdaptiveDropoutLayer(neural_network,
					                                            activation_probability=activation_probability)

				dropout_layer_index += 1

				neural_network = layers.DenseLayer(neural_network,
				                                   layer_dimension,
				                                   W=init.GlorotUniform(
					                                   gain=init.GlorotUniformGain[layer_nonlinearity]),
				                                   nonlinearity=layer_nonlinearity)
			# print_output_dimension("after dense layer %i" % layer_index, neural_network, sequence_length, window_size)

			elif isinstance(layer_dimension, list):
				# if not isinstance(layers.get_all_layers(neural_network)[-1], recurrent_type):
				if not isinstance(neural_network, recurrent_type):
					previous_layer_dimension = layers.get_output_shape(neural_network)
					neural_network = layers.ReshapeLayer(neural_network,
					                                     (-1, self._sequence_length, previous_layer_dimension[-1]))
				# print_output_dimension("after reshape (for recurrent layer)", neural_network, sequence_length, window_size)

				layer_dimension = layer_dimension[0]
				layer_nonlinearity = layer_nonlinearity[0]
				if recurrent_type == layers.recurrent.RecurrentLayer:
					neural_network = layers.RecurrentLayer(neural_network,
					                                       layer_dimension,
					                                       W_in_to_hid=init.GlorotUniform(
						                                       gain=init.GlorotUniformGain[layer_nonlinearity]),
					                                       W_hid_to_hid=init.GlorotUniform(
						                                       gain=init.GlorotUniformGain[layer_nonlinearity]),
					                                       b=init.Constant(0.),
					                                       nonlinearity=layer_nonlinearity,
					                                       hid_init=init.Constant(0.),
					                                       backwards=False,
					                                       learn_init=False,
					                                       gradient_steps=-1,
					                                       grad_clipping=0,
					                                       unroll_scan=False,
					                                       precompute_input=True,
					                                       mask_input=self._input_mask_layer,
					                                       # only_return_final=True
					                                       )
				elif recurrent_type == layers.recurrent.LSTMLayer:
					neural_network = layers.LSTMLayer(neural_network,
					                                  layer_dimension,
					                                  ingate=layers.Gate(),
					                                  forgetgate=layers.Gate(),
					                                  cell=layers.Gate(W_cell=None,
					                                                   nonlinearity=layer_nonlinearity),
					                                  outgate=layers.Gate(),
					                                  nonlinearity=layer_nonlinearity,
					                                  cell_init=init.Constant(0.),
					                                  hid_init=init.Constant(0.),
					                                  backwards=False,
					                                  learn_init=False,
					                                  peepholes=True,
					                                  gradient_steps=-1,
					                                  grad_clipping=0,
					                                  unroll_scan=False,
					                                  precompute_input=True,
					                                  mask_input=self._input_mask_layer,
					                                  # only_return_final=True
					                                  )
				# print_output_dimension("after recurrent layer %i" % layer_index, neural_network, sequence_length, window_size)
			else:
				logging.error("Unrecognized layer specifications...")
				sys.stderr.write("Unrecognized layer specifications...\n")
				sys.exit()

		self._neural_network = neural_network

		self.build_functions()

	def build_functions(self):
		super(DynamicElmanNetwork, self).build_functions()

		# Create update expressions for training, i.e., how to modify the parameters at each training step. Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		dropout_loss = self.get_loss(self._output_variable, deterministic=True)
		dropout_accuracy = self.get_objectives(self._output_variable,
		                                       objective_functions="categorical_accuracy",
		                                       deterministic=True)

		adaptable_params = self.get_network_params(adaptable=True)
		adaptable_params_updates = self._update_function(dropout_loss, adaptable_params,
		                                                 self._learning_rate_variable, momentum=0.95)

		# Compile a second function computing the validation train_loss and accuracy:
		self._train_dropout_function = theano.function(
			inputs=[self._input_variable, self._output_variable, self._input_mask_variable,
			        self._learning_rate_variable],
			outputs=[dropout_loss, dropout_accuracy],
			updates=adaptable_params_updates
		)

		from debugger import debug_rademacher
		self._debug_function = theano.function(
			inputs=[self._input_variable, self._output_variable, self._input_mask_variable,
			        self._learning_rate_variable],
			outputs=debug_rademacher(self, self._output_variable, deterministic=True),
			# outputs=[self.get_objectives(self._output_variable, determininistic=True), self.get_loss(self._output_variable, deterministic=True)],
			on_unused_input='ignore'
		)

	def train_minibatch(self, minibatch_x, minibatch_y, minibatch_m, learning_rate):
		minibatch_running_time = timeit.default_timer()
		train_function_outputs = self._train_function(minibatch_x, minibatch_y, minibatch_m, learning_rate)
		minibatch_average_train_loss = train_function_outputs[0]
		minibatch_average_train_accuracy = train_function_outputs[1]
		if self._dropout_rate_update_interval > 0 and self.minibatch_index % self._dropout_rate_update_interval == 0:
			train_dropout_function_outputs = self._train_dropout_function(minibatch_x, minibatch_y, minibatch_m,
			                                                              learning_rate)
		minibatch_running_time = timeit.default_timer() - minibatch_running_time

		# print(self._debug_function(minibatch_x, minibatch_y, minibatch_m, learning_rate))

		self._normalize_embedding_function()

		return minibatch_running_time, minibatch_average_train_loss, minibatch_average_train_accuracy

	'''
	def get_output_from_sequence(self, input_sequence):
		context_windows = get_context_windows(input_sequence, self._window_size)
		train_minibatch, train_minibatch_masks = get_mini_batches(context_windows, self._backprop_step)
		# lasagne.layers.get_output(self._neural_network, inputs, **kwargs)
		return self.get_output({self._input_data_layer:train_minibatch, self._input_mask_layer:train_minibatch_masks})
	'''

	'''
	def get_objective_to_minimize(self, label):
		train_loss = theano.tensor.mean(self._objective_to_minimize(self.get_output(), label))

		train_loss += self.L1_regularizer()
		train_loss += self.L2_regularizer()
		#train_loss += self.dae_regularizer()

		return train_loss
	'''


def print_output_dimension(checkpoint_text, neural_network, sequence_length, window_size, batch_size=6):
	reference_to_input_layers = [input_layer for input_layer in layers.get_all_layers(neural_network) if
	                             isinstance(input_layer, layers.InputLayer)]
	if len(reference_to_input_layers) == 1:
		print checkpoint_text, ":", layers.get_output_shape(neural_network, {
			reference_to_input_layers[0]: (batch_size, sequence_length, window_size)})
	elif len(reference_to_input_layers) == 2:
		print checkpoint_text, ":", layers.get_output_shape(neural_network, {
			reference_to_input_layers[0]: (batch_size, sequence_length, window_size),
			reference_to_input_layers[1]: (batch_size, sequence_length)})


def main():
	window_size = 5
	position_offset = 1
	sequence_length = 9

	network = ElmanNetwork(
		input_network=layers.InputLayer(shape=(None, sequence_length, window_size,)),
		input_mask=layers.InputLayer(shape=(None, sequence_length)),
		vocabulary_dimension=100,
		embedding_dimension=50,
		window_size=window_size,
		position_offset=position_offset,
		sequence_length=sequence_length,
		layer_dimensions=[32, [64], 127],
		layer_nonlinearities=[nonlinearities.rectify, [nonlinearities.rectify],
		                      nonlinearities.softmax],
		objective_to_minimize=objectives.categorical_crossentropy,
	)

	regularizer_functions = {}
	# regularizer_functions[regularization.l1] = 0.1
	# regularizer_functions[regularization.l2] = [0.2, 0.5, 0.4]
	# regularizer_functions[regularization.rademacher] = 1e-6
	network.set_regularizers(regularizer_functions)

	data = [554, 23, 241, 534, 358, 136, 193, 11, 208, 251, 104, 502, 413, 256, 104]
	'''
	context_windows = get_context(data, window_size)
	print context_windows
	mini_batches, mini_batch_masks = network.get_instance_sequences(data)
	print mini_batches
	print mini_batch_masks
	'''

	########################
	# START MODEL TRAINING #
	########################

	start_train = timeit.default_timer()
	# Finally, launch the training loop.
	# We iterate over epochs:
	number_of_epochs = 10
	minibatch_size = 1000
	for epoch_index in range(number_of_epochs):
		# network.train(train_dataset, minibatch_size, validate_dataset, test_dataset)
		print("PROGRESS: %f%%" % (100. * epoch_index / number_of_epochs))
	end_train = timeit.default_timer()

	print("Optimization complete...")
	logging.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
		network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index))
	print('The code finishes in %.2fm' % ((end_train - start_train) / 60.))


if __name__ == '__main__':
	main()