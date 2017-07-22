import timeit

import numpy
import theano
import theano.tensor

from . import FeedForwardNetwork
from .. import init, objectives, updates
from .. import layers
from ..layers import noise, local, normalization

__all__ = [
	"AlexNet",
	"DynamicAlexNet",
]


class AlexNet(FeedForwardNetwork):
	def __init__(self,
	             incoming,

	             convolution_filters,
	             convolution_nonlinearities,

	             number_of_layers_to_LRN,
	             pool_modes,

	             locally_connected_filters,
	             # locally_connected_nonlinearities,

	             dense_dimensions,
	             dense_nonlinearities,

	             layer_activation_parameters=None,
	             layer_activation_styles=None,

	             objective_functions=objectives.categorical_crossentropy,
	             update_function=updates.nesterov_momentum,
	             learning_rate=1e-3,
	             learning_rate_decay=None,
	             max_norm_constraint=0,
	             # learning_rate_decay_style=None,
	             # learning_rate_decay_parameter=0,

	             validation_interval=-1,

	             convolution_kernel_sizes=(5, 5),
	             convolution_strides=(1, 1),
	             convolution_pads=2,

	             local_convolution_filter_sizes=(3, 3),
	             local_convolution_strides=(1, 1),
	             local_convolution_pads="same",

	             pooling_kernel_sizes=(3, 3),
	             pooling_strides=(2, 2),
	             ):
		super(AlexNet, self).__init__(incoming,
		                              objective_functions,
		                              update_function,
		                              learning_rate,
		                              learning_rate_decay,
		                              max_norm_constraint,
		                              # learning_rate_decay_style,
		                              # learning_rate_decay_parameter,
		                              validation_interval)

		# x = theano.tensor.matrix('x')  # the data is presented as rasterized images
		#self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

		# self._input_layer = layers.InputLayer(shape=input_shape)
		# self._input_variable = self._input_layer.input_var

		assert len(layer_activation_parameters) == len(dense_nonlinearities)  # + len(convolution_nonlinearities)
		assert len(layer_activation_styles) == len(dense_nonlinearities)  # + len(convolution_nonlinearities)
		assert len(convolution_filters) == len(convolution_nonlinearities)
		assert len(convolution_filters) == len(pool_modes)
		assert len(convolution_filters) >= number_of_layers_to_LRN

		dropout_layer_index = 0
		neural_network = self._input_layer
		for conv_layer_index in range(len(convolution_filters)):
			'''
			input_layer_shape = layers.get_output_shape(neural_network)[1:]
			previous_layer_shape = numpy.prod(input_layer_shape)
			activation_probability = noise.sample_activation_probability(previous_layer_shape,
																		 layer_activation_styles[dropout_layer_index],
																		 layer_activation_parameters[dropout_layer_index])
			activation_probability = numpy.reshape(activation_probability, input_layer_shape)
			# print "before dropout", lasagne.layers.get_output_shape(neural_network)
			neural_network = noise.LinearDropoutLayer(neural_network,
													  activation_probability=activation_probability)
			dropout_layer_index += 1
			'''

			conv_filter_number = convolution_filters[conv_layer_index]
			conv_nonlinearity = convolution_nonlinearities[conv_layer_index]

			conv_filter_size = convolution_kernel_sizes
			conv_stride = convolution_strides
			conv_pad = convolution_pads

			# print "before convolution", lasagne.layers.get_output_shape(neural_network)
			# Convolutional layer with 32 kernels of size 5x5. Strided and padded convolutions are supported as well see the docstring.
			neural_network = layers.Conv2DLayer(neural_network,
			                                    W=init.GlorotUniform(gain=(init.GlorotUniformGain[conv_nonlinearity])),
			                                    b=init.Constant(1.),
			                                    nonlinearity=conv_nonlinearity,
			                                    num_filters=conv_filter_number,
			                                    filter_size=conv_filter_size,
			                                    stride=conv_stride,
			                                    pad=conv_pad,
			                                    )
			# print "convolution", numpy.mean(neural_network.W.eval()), numpy.max(neural_network.W.eval()), numpy.min(neural_network.W.eval())

			if conv_layer_index < number_of_layers_to_LRN:
				neural_network = normalization.LocalResponseNormalization2DLayer(neural_network)

			pool_mode = pool_modes[conv_layer_index]
			if pool_mode != None:
				pool_size = pooling_kernel_sizes
				pool_stride = pooling_strides

				# Max-pooling layer of factor 2 in both dimensions:
				filter_size_for_pooling = layers.get_output_shape(neural_network)[2:]
				if numpy.any(filter_size_for_pooling < pool_size):
					print("warning: filter size %s is smaller than pooling size %s, skip pooling layer" % (
						layers.get_output_shape(neural_network), pool_size))
					continue
				neural_network = layers.Pool2DLayer(neural_network,
				                                    pool_size=pool_size,
				                                    stride=pool_stride,
				                                    mode=pool_mode,
				                                    )

		if locally_connected_filters != None and len(locally_connected_filters) > 0:
			for local_layer_index in range(len(locally_connected_filters)):
				neural_network = local.LocallyConnected2DLayer(neural_network,
				                                               locally_connected_filters[local_layer_index],
				                                               filter_size=local_convolution_filter_sizes,
				                                               stride=local_convolution_strides,
				                                               pad=local_convolution_pads,
				                                               W=init.GlorotUniform(),
				                                               b=init.Constant(0.),
				                                               )
			# print "locally-connected", numpy.mean(neural_network.W.eval()), numpy.max(neural_network.W.eval()), numpy.min(neural_network.W.eval())

		assert len(dense_dimensions) == len(dense_nonlinearities)
		for dense_layer_index in range(len(dense_dimensions)):
			input_layer_shape = layers.get_output_shape(neural_network)[1:]
			previous_layer_shape = numpy.prod(input_layer_shape)
			activation_probability = noise.sample_activation_probability(previous_layer_shape,
			                                                             layer_activation_styles[dropout_layer_index],
			                                                             layer_activation_parameters[
				                                                             dropout_layer_index])
			activation_probability = numpy.reshape(activation_probability, input_layer_shape)
			neural_network = noise.LinearDropoutLayer(neural_network, activation_probability=activation_probability)
			dropout_layer_index += 1

			layer_shape = dense_dimensions[dense_layer_index]
			layer_nonlinearity = dense_nonlinearities[dense_layer_index]

			# print "before dense", lasagne.layers.get_output_shape(neural_network)
			neural_network = layers.DenseLayer(neural_network,
			                                   layer_shape,
			                                   W=init.GlorotUniform(gain=init.GlorotUniformGain[layer_nonlinearity]),
			                                   nonlinearity=layer_nonlinearity)
		# print "dense", numpy.mean(neural_network.W.eval()), numpy.max(neural_network.W.eval()), numpy.min(neural_network.W.eval())

		self._neural_network = neural_network

		self.build_functions()


class DynamicAlexNet(FeedForwardNetwork):
	def __init__(self,
	             incoming,

	             convolution_filters,
	             convolution_nonlinearities,

	             number_of_layers_to_LRN,
	             pool_modes,

	             locally_connected_filters,
	             # locally_connected_nonlinearities,

	             dense_dimensions,
	             dense_nonlinearities,

	             layer_activation_parameters=None,
	             layer_activation_styles=None,

	             objective_functions=objectives.categorical_crossentropy,
	             update_function=updates.nesterov_momentum,
	             learning_rate=1e-3,
	             learning_rate_decay=None,
	             max_norm_constraint=0,
	             # learning_rate_decay_style=None,
	             # learning_rate_decay_parameter=0,

	             dropout_rate_update_interval=-1,
	             update_hidden_layer_dropout_only=False,

	             validation_interval=-1,

	             convolution_kernel_sizes=(5, 5),
	             convolution_strides=(1, 1),
	             convolution_pads=2,

	             local_convolution_filter_sizes=(3, 3),
	             local_convolution_strides=(1, 1),
	             local_convolution_pads="same",

	             pooling_kernel_sizes=(3, 3),
	             pooling_strides=(2, 2),
	             ):
		super(DynamicAlexNet, self).__init__(incoming,
		                                     objective_functions,
		                                     update_function,
		                                     learning_rate,
		                                     learning_rate_decay,
		                                     max_norm_constraint,
		                                     # learning_rate_decay_style,
		                                     # learning_rate_decay_parameter,
		                                     validation_interval)

		self._dropout_rate_update_interval = dropout_rate_update_interval

		# x = theano.tensor.matrix('x')  # the data is presented as rasterized images
		#self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

		# self._input_layer = layers.InputLayer(shape=input_shape)
		# self._input_variable = self._input_layer.input_var

		assert len(layer_activation_parameters) == len(dense_nonlinearities)  # + len(convolution_nonlinearities)
		assert len(layer_activation_styles) == len(dense_nonlinearities)  # + len(convolution_nonlinearities)
		assert len(convolution_filters) == len(convolution_nonlinearities)
		assert len(convolution_filters) == len(pool_modes)
		assert len(convolution_filters) >= number_of_layers_to_LRN

		dropout_layer_index = 0

		neural_network = self._input_layer
		# print "after input", layers.get_output_shape(neural_network)
		for conv_layer_index in range(len(convolution_filters)):
			'''
			input_layer_shape = layers.get_output_shape(neural_network)[1:]
			previous_layer_shape = numpy.prod(input_layer_shape)
			activation_probability = noise.sample_activation_probability(previous_layer_shape,
																		 layer_activation_styles[dropout_layer_index],
																		 layer_activation_parameters[dropout_layer_index])
			activation_probability = numpy.reshape(activation_probability, input_layer_shape)
			# print "before dropout", lasagne.layers.get_output_shape(neural_network)
			if update_hidden_layer_dropout_only:
				neural_network = noise.LinearDropoutLayer(neural_network,
														  activation_probability=activation_probability)
			else:
				neural_network = noise.AdaptiveDropoutLayer(neural_network,
															activation_probability=activation_probability)
			dropout_layer_index += 1
			'''

			conv_filter_number = convolution_filters[conv_layer_index]
			conv_nonlinearity = convolution_nonlinearities[conv_layer_index]

			conv_filter_size = convolution_kernel_sizes
			conv_stride = convolution_strides
			conv_pad = convolution_pads

			# Convolutional layer with 32 kernels of size 5x5. Strided and padded convolutions are supported as well see the docstring.
			neural_network = layers.Conv2DLayer(neural_network,
			                                    W=init.GlorotUniform(gain=(init.GlorotUniformGain[conv_nonlinearity])),
			                                    # This is ONLY for CIFAR-10 dataset.
			                                    # W=init.Uniform(0.1**(1+len(convolution_filters)-conv_layer_index)),
			                                    # W=init.HeNormal(gain=0.1),
			                                    b=init.Constant(1.),
			                                    nonlinearity=conv_nonlinearity,
			                                    num_filters=conv_filter_number,
			                                    filter_size=conv_filter_size,

			                                    stride=conv_stride,
			                                    pad=conv_pad,
			                                    )
			# print "after convolution", layers.get_output_shape(neural_network)

			if conv_layer_index < number_of_layers_to_LRN:
				neural_network = normalization.LocalResponseNormalization2DLayer(neural_network)

			pool_mode = pool_modes[conv_layer_index]
			if pool_mode != None:
				pool_size = pooling_kernel_sizes
				pool_stride = pooling_strides

				# Max-pooling layer of factor 2 in both dimensions:
				filter_size_for_pooling = layers.get_output_shape(neural_network)[2:]
				if numpy.any(filter_size_for_pooling < pool_size):
					print("warning: filter size %s is smaller than pooling size %s, skip pooling layer" % (
						layers.get_output_shape(neural_network), pool_size))
					continue
				neural_network = layers.Pool2DLayer(neural_network,
				                                    pool_size=pool_size,
				                                    stride=pool_stride,
				                                    mode=pool_mode,
				                                    )

		if locally_connected_filters != None and len(locally_connected_filters) > 0:
			for local_layer_index in range(len(locally_connected_filters)):
				neural_network = local.LocallyConnected2DLayer(neural_network,
				                                               locally_connected_filters[local_layer_index],
				                                               filter_size=local_convolution_filter_sizes,
				                                               stride=local_convolution_strides,
				                                               pad=local_convolution_pads,
				                                               W=init.GlorotUniform(),
				                                               b=init.Constant(0.),
				                                               )

		assert len(dense_dimensions) == len(dense_nonlinearities)
		for dense_layer_index in range(len(dense_dimensions)):
			input_layer_shape = layers.get_output_shape(neural_network)[1:]
			previous_layer_shape = numpy.prod(input_layer_shape)
			activation_probability = noise.sample_activation_probability(previous_layer_shape,
			                                                             layer_activation_styles[dropout_layer_index],
			                                                             layer_activation_parameters[
				                                                             dropout_layer_index])
			activation_probability = numpy.reshape(activation_probability, input_layer_shape)
			if update_hidden_layer_dropout_only and dense_layer_index == 0:
				neural_network = noise.LinearDropoutLayer(neural_network,
				                                          activation_probability=activation_probability)
			else:
				# neural_network = noise.TrainableDropoutLayer(neural_network, activation_probability=init.Constant(layer_activation_parameters[layer_index]))
				neural_network = noise.AdaptiveDropoutLayer(neural_network,
				                                            activation_probability=activation_probability)
			dropout_layer_index += 1

			layer_shape = dense_dimensions[dense_layer_index]
			layer_nonlinearity = dense_nonlinearities[dense_layer_index]

			neural_network = layers.DenseLayer(neural_network,
			                                   layer_shape,
			                                   W=init.GlorotUniform(gain=(init.GlorotUniformGain[layer_nonlinearity])),
			                                   nonlinearity=layer_nonlinearity)
		# print "after dense", layers.get_output_shape(neural_network)

		self._neural_network = neural_network

		self.build_functions()

	def build_functions(self):
		super(DynamicAlexNet, self).build_functions()

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
			inputs=[self._input_variable, self._output_variable, self._learning_rate_variable],
			outputs=[dropout_loss, dropout_accuracy],
			updates=adaptable_params_updates
		)

		'''
		from debugger import debug_rademacher
		self._debug_function = theano.function(
			inputs=[self._input_variable, self._output_variable, self._learning_rate_variable],
			outputs=debug_rademacher(self, self._output_variable, deterministic=True),
			# outputs=[self.get_objectives(self._output_variable, determininistic=True), self.get_loss(self._output_variable, deterministic=True)],
			on_unused_input='ignore'
		)
		'''

	def train_minibatch(self, minibatch_x, minibatch_y, learning_rate):
		minibatch_running_time = timeit.default_timer()
		train_function_outputs = self._train_function(minibatch_x, minibatch_y, learning_rate)
		minibatch_average_train_loss = train_function_outputs[0]
		minibatch_average_train_accuracy = train_function_outputs[1]

		if self._dropout_rate_update_interval > 0 and self.minibatch_index % self._dropout_rate_update_interval == 0:
			train_dropout_function_outputs = self._train_dropout_function(minibatch_x, minibatch_y, learning_rate)
		# minibatch_average_train_dropout_loss = train_dropout_function_outputs[0]
		# minibatch_average_train_dropout_accuracy = train_dropout_function_outputs[1]
		minibatch_running_time = timeit.default_timer() - minibatch_running_time

		# print self._debug_function(minibatch_x, minibatch_y, learning_rate)

		return minibatch_running_time, minibatch_average_train_loss, minibatch_average_train_accuracy
