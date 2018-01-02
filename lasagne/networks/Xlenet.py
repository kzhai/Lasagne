import logging
import os

import numpy

from . import AdaptiveFeedForwardNetwork
from .. import init, objectives, updates, Xpolicy
from .. import layers

logger = logging.getLogger(__name__)

__all__ = [
	"AdaptiveLeNet",
]


class AdaptiveLeNet(AdaptiveFeedForwardNetwork):
	def __init__(self,
	             # input_network=None,
	             # input_shape,
	             incoming,

	             conv_filters,
	             conv_nonlinearities,
	             # convolution_filter_sizes=None,
	             pool_modes,

	             dense_dimensions,
	             dense_nonlinearities,

	             layer_activation_types,
	             layer_activation_parameters,
	             layer_activation_styles,

	             objective_functions=objectives.categorical_crossentropy,
	             update_function=updates.nesterov_momentum,

	             learning_rate_policy=[1e-3, Xpolicy.constant],
	             # learning_rate_decay=None,

	             adaptable_learning_rate_policy=[1e-3, Xpolicy.constant],
	             # dropout_learning_rate_decay=None,
	             # adaptable_update_interval=1,
	             # update_hidden_layer_dropout_only=False,
	             # train_adaptables_mode="network",
	             train_adaptables_mode="train_adaptables_networkwise",

	             max_norm_constraint=0,
	             validation_interval=-1,

	             conv_kernel_sizes=(5, 5),
	             conv_strides=(1, 1),
	             conv_pads=2,

	             pool_kernel_sizes=(3, 3),
	             pool_strides=(2, 2),
	             ):
		super(AdaptiveLeNet, self).__init__(incoming=incoming,

		                                    objective_functions=objective_functions,
		                                    update_function=update_function,
		                                    learning_rate_policy=learning_rate_policy,
		                                    # learning_rate_decay,

		                                    adaptable_learning_rate_policy=adaptable_learning_rate_policy,
		                                    # dropout_learning_rate_decay,
		                                    # adaptable_update_interval=adaptable_update_interval,
		                                    train_adaptables_mode=train_adaptables_mode,

		                                    max_norm_constraint=max_norm_constraint,
		                                    validation_interval=validation_interval,
		                                    )

		assert len(layer_activation_types) == len(dense_nonlinearities) + len(conv_nonlinearities)
		assert len(layer_activation_parameters) == len(dense_nonlinearities) + len(conv_nonlinearities)
		assert len(layer_activation_styles) == len(dense_nonlinearities) + len(conv_nonlinearities)
		assert len(conv_filters) == len(conv_nonlinearities)
		assert len(conv_filters) == len(pool_modes)

		dropout_layer_index = 0
		neural_network = self._input_layer
		for conv_layer_index in range(len(conv_filters)):
			input_layer_shape = layers.get_output_shape(neural_network)[1:]
			previous_layer_shape = numpy.prod(input_layer_shape)
			activation_probability = layers.sample_activation_probability(previous_layer_shape,
			                                                              layer_activation_styles[dropout_layer_index],
			                                                              layer_activation_parameters[
				                                                              dropout_layer_index])
			activation_probability = numpy.reshape(activation_probability, input_layer_shape)
			neural_network = layer_activation_types[dropout_layer_index](neural_network,
			                                                             activation_probability=activation_probability)
			dropout_layer_index += 1

			conv_filter = conv_filters[conv_layer_index]
			conv_nonlinearity = conv_nonlinearities[conv_layer_index]
			# conv_filter_size = convolution_filter_sizes[conv_layer_index]
			conv_kernel_size = conv_kernel_sizes
			conv_stride = conv_strides
			conv_pad = conv_pads

			# Convolutional layer with 32 kernels of size 5x5. Strided and padded convolutions are supported as well see the docstring.
			neural_network = layers.Conv2DLayer(neural_network,
			                                    W=init.GlorotUniform(gain=init.GlorotUniformGain[conv_nonlinearity]),
			                                    # This is ONLY for CIFAR-10 dataset.
			                                    # W=init.Uniform(0.1**(1+len(convolution_filters)-conv_layer_index)),
			                                    # W=init.HeNormal(gain=0.1),
			                                    # b=init.Constant(1.0 * (conv_layer_index!=0)),
			                                    b=init.Constant(0.),
			                                    nonlinearity=conv_nonlinearity,
			                                    num_filters=conv_filter,
			                                    filter_size=conv_kernel_size,

			                                    stride=conv_stride,
			                                    pad=conv_pad,
			                                    )

			pool_mode = pool_modes[conv_layer_index]
			if pool_mode is not None:
				pool_kernel_size = pool_kernel_sizes
				pool_stride = pool_strides

				# Max-pooling layer of factor 2 in both dimensions:
				filter_size_for_pooling = layers.get_output_shape(neural_network)[2:]
				if numpy.any(filter_size_for_pooling < pool_kernel_size):
					print("warning: filter size %s is smaller than pooling size %s, skip pooling layer" % (
						layers.get_output_shape(neural_network), pool_kernel_size))
					continue
				neural_network = layers.Pool2DLayer(neural_network, pool_size=pool_kernel_size, stride=pool_stride,
				                                    mode=pool_mode)

		#neural_network = layers.ReshapeLayer(neural_network, (-1, numpy.prod(layers.get_output_shape(neural_network)[1:])))

		assert len(dense_dimensions) == len(dense_nonlinearities)
		for dense_layer_index in range(len(dense_dimensions)):
			input_layer_shape = layers.get_output_shape(neural_network)[1:]
			previous_layer_shape = numpy.prod(input_layer_shape)
			activation_probability = layers.sample_activation_probability(previous_layer_shape,
			                                                              layer_activation_styles[dropout_layer_index],
			                                                              layer_activation_parameters[
				                                                              dropout_layer_index])
			activation_probability = numpy.reshape(activation_probability, input_layer_shape)
			neural_network = layer_activation_types[dropout_layer_index](neural_network,
			                                                             activation_probability=activation_probability)
			dropout_layer_index += 1

			layer_shape = dense_dimensions[dense_layer_index]
			layer_nonlinearity = dense_nonlinearities[dense_layer_index]

			neural_network = layers.DenseLayer(neural_network,
			                                   layer_shape,
			                                   W=init.GlorotUniform(gain=init.GlorotUniformGain[layer_nonlinearity]),
			                                   # This is ONLY for CIFAR-10 dataset.
			                                   # W=init.HeNormal('relu'),
			                                   nonlinearity=layer_nonlinearity)

		self._neural_network = neural_network

		self.build_functions()

	'''
	def build_functions(self):
		super(DynamicLeNet, self).build_functions()

		# Create update expressions for training, i.e., how to modify the parameters at each training step. Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		dropout_loss = self.get_loss(self._output_variable, deterministic=True)
		dropout_objective = self.get_objectives(self._output_variable, deterministic=True)
		dropout_accuracy = self.get_objectives(self._output_variable,
		                                       objective_functions="categorical_accuracy",
		                                       deterministic=True)

		adaptable_params = self.get_network_params(adaptable=True)
		adaptable_params_updates = self._update_function(dropout_loss, adaptable_params,
		                                                 self._learning_rate_variable, momentum=0.95)

		# Compile a second function computing the validation train_loss and accuracy:
		self._train_dropout_function = theano.function(
			inputs=[self._input_variable, self._output_variable, self._learning_rate_variable],
			outputs=[dropout_objective, dropout_accuracy],
			updates=adaptable_params_updates
		)

	def train_minibatch(self, minibatch_x, minibatch_y, learning_rate):
		minibatch_running_time = timeit.default_timer()
		train_function_outputs = self._train_function(minibatch_x, minibatch_y, learning_rate)
		minibatch_average_train_objective = train_function_outputs[0]
		minibatch_average_train_accuracy = train_function_outputs[1]

		if self._dropout_rate_update_interval > 0 and self.minibatch_index % self._dropout_rate_update_interval == 0:
			train_dropout_function_outputs = self._train_dropout_function(minibatch_x, minibatch_y, learning_rate)
		minibatch_running_time = timeit.default_timer() - minibatch_running_time

		# print self._debug_function(minibatch_x, minibatch_y, learning_rate)

		return minibatch_running_time, minibatch_average_train_objective, minibatch_average_train_accuracy
	'''

	def debug(self, settings, **kwargs):
		output_directory = settings.output_directory
		dropout_layer_index = 0
		for network_layer in self.get_network_layers():
			if not isinstance(network_layer, layers.AdaptiveDropoutLayer):
				continue

			layer_retain_probability = network_layer.activation_probability.eval()
			logger.info("retain rates: epoch %i, shape %s, average %f, minimum %f, maximum %f" % (
				self.epoch_index,
				layer_retain_probability.shape,
				numpy.mean(layer_retain_probability),
				numpy.min(layer_retain_probability),
				numpy.max(layer_retain_probability)))

			retain_rate_file = os.path.join(output_directory,
			                                "layer.%d.epoch.%d.npy" % (dropout_layer_index, self.epoch_index))
			numpy.save(retain_rate_file, layer_retain_probability)
			dropout_layer_index += 1

	"""
	def dae_regularizer(self):
		if self._layer_dae_regularizer_lambdas == None:
			return 0
		else:
			dae_regularization = 0
			for dae_layer in self._layer_dae_regularizer_lambdas:
				dae_regularization += self._layer_dae_regularizer_lambdas[dae_layer] * dae_layer.get_objective_to_minimize()
			return dae_regularization

	def set_dae_regularizer_lambda(self,
								   layer_dae_regularizer_lambdas,
								   layer_corruption_levels=None,
								   L1_regularizer_lambdas=None,
								   L2_regularizer_lambdas=None
								   ):
		if layer_dae_regularizer_lambdas == None or all(layer_dae_regularizer_lambda == 0 for layer_dae_regularizer_lambda in layer_dae_regularizer_lambdas):
			self._layer_dae_regularizer_lambdas = None
		else:
			assert len(layer_dae_regularizer_lambdas) == (len(self.get_all_layers()) - 1) / 2 - 1
			dae_regularizer_layers = self.__build_dae_network(layer_corruption_levels, L1_regularizer_lambdas, L2_regularizer_lambdas)
			self._layer_dae_regularizer_lambdas = {temp_layer:layer_dae_regularizer_lambda for temp_layer, layer_dae_regularizer_lambda in zip(dae_regularizer_layers, layer_dae_regularizer_lambdas)}

		return

	def __build_dae_network(self,
							layer_corruption_levels=None,
							L1_regularizer_lambdas=None,
							L2_regularizer_lambdas=None
							):
		layers = self.get_all_layers()

		denoising_auto_encoders = []

		if layer_corruption_levels is None:
			layer_corruption_levels = numpy.zeros((len(layers) - 1) / 2 - 1)
		assert len(layer_corruption_levels) == (len(layers) - 1) / 2 - 1

		for hidden_layer_index in range(2, len(layers) - 1, 2):
			hidden_layer = layers[hidden_layer_index]
			# this is to get around the dropout layer
			# input = hidden_layer.input
			input_layer = layers[hidden_layer_index - 2]
			hidden_layer_shape = hidden_layer.num_units
			hidden_layer_nonlinearity = hidden_layer.nonlinearity

			# this is to get around the dropout layer
			# layer_corruption_level = layer_corruption_levels[hidden_layer_index - 1]
			layer_corruption_level = layer_corruption_levels[hidden_layer_index / 2 - 1]

			denoising_auto_encoder = DenoisingAutoEncoder(
				input=input_layer,
				layer_shape=hidden_layer_shape,
				encoder_nonlinearity=hidden_layer_nonlinearity,
				decoder_nonlinearity=lasagne.nonlinearities.sigmoid,
				# _objective_to_minimize=lasagne.objectives.binary_crossentropy,
				_objective_to_minimize=theano.tensor.nnet.binary_crossentropy,
				# _objective_to_minimize=lasagne.objectives.binary_crossentropy,
				corruption_level=layer_corruption_level,
				# L1_regularizer_lambdas=L1_regularizer_lambdas,
				# L2_regularizer_lambdas=L2_regularizer_lambdas,
				W_encode=hidden_layer.W,
				b_encoder=hidden_layer.b,
				)

			denoising_auto_encoders.append(denoising_auto_encoder)

		return denoising_auto_encoders
	"""

	"""
	def pretrain_with_dae(self, data_x, layer_corruption_levels=None, number_of_epochs=50, minibatch_size=1, learning_rate=1e-3, momentum=0.95):
		denoising_auto_encoders = self.__build_dae_network(layer_corruption_levels)
		pretrain_functions = []
		for denoising_auto_encoder in denoising_auto_encoders:
			'''
			train_prediction = lasagne.layers.get_output(denoising_auto_encoder)
			pretrain_loss = x * theano.tensor.log(train_prediction) + (1 - x) * theano.tensor.log(1 - train_prediction)
			pretrain_loss = theano.tensor.mean(-theano.tensor.sum(pretrain_loss, axis=1))
			'''

			pretrain_loss = denoising_auto_encoder.get_objective_to_minimize()
			# pretrain_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(train_prediction, y))

			'''
			# We could add some weight decay as well here, see lasagne.regularization.
			dae_layers = lasagne.layers.get_all_layers(denoising_auto_encoders)
			L1_regularizer_layer_lambdas = {temp_layer:L1_regularizer_lambda for temp_layer, L1_regularizer_lambda in zip(dae_layers[1:], L1_regularizer_lambdas)}
			L1_regularizer = lasagne.regularization.regularize_layer_params_weighted(L1_regularizer_layer_lambdas, lasagne.regularization.l1)
			L2_regularizer_layer_lambdas = {temp_layer:L2_regularizer_lambda for temp_layer, L2_regularizer_lambda in zip(dae_layers[1:], L2_regularizer_lambdas)}
			L2_regularizer = lasagne.regularization.regularize_layer_params_weighted(L2_regularizer_layer_lambdas, lasagne.regularization.l2)
			pretrain_loss += L1_regularizer + L2_regularizer
			'''

			# Create update expressions for training, i.e., how to modify the
			# parameters at each training step. Here, we'll use Stochastic Gradient
			# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
			all_dae_params = denoising_auto_encoder.get_all_params(trainable=True)
			# all_dae_params = lasagne.layers.get_all_params(denoising_auto_encoder, trainable=True)
			updates = lasagne.updates.nesterov_momentum(pretrain_loss, all_dae_params, learning_rate, momentum)

			'''
			# Create a pretrain_loss expression for validation/testing. The crucial difference
			# here is that we do a deterministic forward pass through the networks,
			# disabling dropout layers.
			validate_prediction = lasagne.layers.get_output(networks, deterministic=True)
			validate_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(validate_prediction, y))
			# As a bonus, also create an expression for the classification accuracy:
			validate_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(validate_prediction, axis=1), y), dtype=theano.config.floatX)
			'''

			# Compile a function performing a training step on a mini-batch (by giving
			# the updates dictionary) and returning the corresponding training pretrain_loss:
			pretrain_function = theano.function(
				inputs=[self.input],
				outputs=[pretrain_loss,
						 self.input,
						 # denoising_auto_encoder._neural_network.get_encoder_output_for(self.input),
						 # denoising_auto_encoder._neural_network.get_decoder_output_for(self.input),
						 # denoising_auto_encoder._neural_network.get_output_for(self.input)
						 lasagne.layers.get_output(denoising_auto_encoder._neural_network, self.input),
						 ],
				updates=updates
			)

			pretrain_functions.append(pretrain_function)

		number_of_minibatches_to_pretrain = data_x.shape[0] / minibatch_size

		# start_time = timeit.default_timer()
		for dae_index in range(len(denoising_auto_encoders)):
			# denoising_auto_encoder = denoising_auto_encoders[dae_index]
			# layer_corruption_level = layer_corruption_levels[dae_index]
			for pretrain_epoch_index in range(number_of_epochs):
				start_time = time.time()

				average_pretrain_loss = []
				for minibatch_index in range(number_of_minibatches_to_pretrain):
					iteration_index = pretrain_epoch_index * number_of_minibatches_to_pretrain + minibatch_index

					minibatch_x = data_x[minibatch_index * minibatch_size:(minibatch_index + 1) * minibatch_size, :]

					function_output = pretrain_functions[dae_index](minibatch_x)
					temp_average_pretrain_loss = function_output[0]
					# print temp_average_pretrain_loss

					average_pretrain_loss.append(temp_average_pretrain_loss)

				end_time = time.time()

				print 'pre-training layer %i, epoch %d, average cost %f, time elapsed %f' % (dae_index + 1, pretrain_epoch_index, numpy.mean(average_pretrain_loss), end_time - start_time)
	"""
