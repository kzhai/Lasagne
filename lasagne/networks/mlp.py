import logging
import timeit

import numpy
import theano
import theano.tensor

from . import FeedForwardNetwork
from .. import init, nonlinearities, objectives, updates
from .. import layers

logger = logging.getLogger(__name__)

__all__ = [
	"MultiLayerPerceptron",
	"DynamicMultiLayerPerceptron",
]


class MultiLayerPerceptron(FeedForwardNetwork):
	def __init__(self,
	             incoming,

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
	             ):
		super(MultiLayerPerceptron, self).__init__(incoming,
		                                           objective_functions,
		                                           update_function,
		                                           learning_rate,
		                                           learning_rate_decay,
		                                           max_norm_constraint,
		                                           # learning_rate_decay_style,
		                                           # learning_rate_decay_parameter,
		                                           validation_interval,
		                                           )

		# x = theano.tensor.matrix('x')  # the data is presented as rasterized images
		# self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

		# self._input_layer = layers.InputLayer(shape=input_shape)
		# self._input_variable = self._input_layer.input_var

		assert len(dense_dimensions) == len(dense_nonlinearities)
		assert len(dense_dimensions) == len(layer_activation_parameters)
		assert len(dense_dimensions) == len(layer_activation_styles)

		'''
		pretrained_network_layers = None
		if pretrained_model != None:
			pretrained_network_layers = lasagne.layers.get_all_layers(pretrained_model._neural_network)
		'''

		# neural_network = input_network
		neural_network = self._input_layer
		for layer_index in range(len(dense_dimensions)):
			previous_layer_dimension = layers.get_output_shape(neural_network)[1:]
			activation_probability = layers.sample_activation_probability(previous_layer_dimension,
			                                                              layer_activation_styles[layer_index],
			                                                              layer_activation_parameters[layer_index])

			neural_network = layers.LinearDropoutLayer(neural_network,
			                                           activation_probability=activation_probability)

			layer_dimension = dense_dimensions[layer_index]
			layer_nonlinearity = dense_nonlinearities[layer_index]

			neural_network = layers.DenseLayer(neural_network, layer_dimension, W=init.GlorotUniform(
				gain=init.GlorotUniformGain[layer_nonlinearity]), nonlinearity=layer_nonlinearity)

			'''
			if pretrained_network_layers == None or len(pretrained_network_layers) <= layer_index:
				_neural_network = lasagne.layers.DenseLayer(_neural_network, layer_dimension, nonlinearity=layer_nonlinearity)
			else:
				pretrained_layer = pretrained_network_layers[layer_index]
				assert isinstance(pretrained_layer, lasagne.layers.DenseLayer)
				assert pretrained_layer.nonlinearity == layer_nonlinearity, (pretrained_layer.nonlinearity, layer_nonlinearity)
				assert pretrained_layer.num_units == layer_dimension

				_neural_network = lasagne.layers.DenseLayer(_neural_network,
													layer_dimension,
													W=pretrained_layer.W,
													b=pretrained_layer.b,
													nonlinearity=layer_nonlinearity)
			'''

		self._neural_network = neural_network

		self.build_functions()


class DynamicMultiLayerPerceptron(FeedForwardNetwork):
	def __init__(self,
	             incoming,

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
	             ):
		super(DynamicMultiLayerPerceptron, self).__init__(incoming,
		                                                  objective_functions,
		                                                  update_function,
		                                                  learning_rate,
		                                                  learning_rate_decay,
		                                                  max_norm_constraint,
		                                                  # learning_rate_decay_style,
		                                                  # learning_rate_decay_parameter,
		                                                  validation_interval,
		                                                  )

		self._dropout_rate_update_interval = dropout_rate_update_interval

		# x = theano.tensor.matrix('x')  # the data is presented as rasterized images
		# self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

		# self._input_layer = layers.InputLayer(shape=input_shape)
		# self._input_variable = self._input_layer.input_var

		assert len(dense_dimensions) == len(dense_nonlinearities)
		assert len(dense_dimensions) == len(layer_activation_parameters)
		assert len(dense_dimensions) == len(layer_activation_styles)

		'''
		pretrained_network_layers = None
		if pretrained_model != None:
			pretrained_network_layers = lasagne.layers.get_all_layers(pretrained_model._neural_network)
		'''

		# neural_network = input_network
		neural_network = self._input_layer
		for layer_index in range(len(dense_dimensions)):
			previous_layer_dimension = layers.get_output_shape(neural_network)[1:]
			activation_probability = layers.sample_activation_probability(previous_layer_dimension,
			                                                              layer_activation_styles[layer_index],
			                                                              layer_activation_parameters[layer_index])

			if update_hidden_layer_dropout_only and layer_index == 0:
				neural_network = layers.LinearDropoutLayer(neural_network,
				                                           activation_probability=activation_probability)
			else:
				neural_network = layers.AdaptiveDropoutLayer(neural_network,
				                                             activation_probability=activation_probability)

			layer_dimension = dense_dimensions[layer_index]
			layer_nonlinearity = dense_nonlinearities[layer_index]

			neural_network = layers.DenseLayer(neural_network, layer_dimension, W=init.GlorotUniform(
				gain=init.GlorotUniformGain[layer_nonlinearity]), nonlinearity=layer_nonlinearity)

			'''
			if pretrained_network_layers == None or len(pretrained_network_layers) <= layer_index:
				_neural_network = lasagne.layers.DenseLayer(_neural_network, layer_dimension, nonlinearity=layer_nonlinearity)
			else:
				pretrained_layer = pretrained_network_layers[layer_index]
				assert isinstance(pretrained_layer, lasagne.layers.DenseLayer)
				assert pretrained_layer.nonlinearity == layer_nonlinearity, (pretrained_layer.nonlinearity, layer_nonlinearity)
				assert pretrained_layer.num_units == layer_dimension

				_neural_network = lasagne.layers.DenseLayer(_neural_network,
													layer_dimension,
													W=pretrained_layer.W,
													b=pretrained_layer.b,
													nonlinearity=layer_nonlinearity)
			'''

		self._neural_network = neural_network

		self.build_functions()

	def build_functions(self):
		super(DynamicMultiLayerPerceptron, self).build_functions()

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
			outputs = debug_rademacher(self, self._output_variable, deterministic=True),
			#outputs=[self.get_objectives(self._output_variable, determininistic=True), self.get_loss(self._output_variable, deterministic=True)],
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
		minibatch_running_time = timeit.default_timer() - minibatch_running_time

		# print self._debug_function(minibatch_x, minibatch_y, learning_rate)

		return minibatch_running_time, minibatch_average_train_loss, minibatch_average_train_accuracy

	#
	#
	#
	#
	#

	'''
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

		self.build_functions()

		return
	'''

	'''
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
				decoder_nonlinearity=nonlinearities.sigmoid,
				# objective_to_minimize=objectives.binary_crossentropy,
				objective_to_minimize=theano.tensor.nnet.binary_crossentropy,
				# objective_to_minimize=objectives.binary_crossentropy,
				corruption_level=layer_corruption_level,
				# L1_regularizer_lambdas=L1_regularizer_lambdas,
				# L2_regularizer_lambdas=L2_regularizer_lambdas,
				W_encode=hidden_layer.W,
				b_encoder=hidden_layer.b,
				)

			denoising_auto_encoders.append(denoising_auto_encoder)

		return denoising_auto_encoders
	'''

	'''
	def pretrain_with_dae(self,
						  pretrain_data,
						  number_of_epochs=50,
						  minibatch_size=1,
						  #layer_corruption=None,
						  objective_function=objectives.squared_error,
						  update_function=updates.nesterov_momentum,
						  learning_rate=1e-3,
						  learning_rate_decay_style=None,
						  learning_rate_decay_parameter=0,
						  ):
		#denoising_auto_encoders = self.__build_dae_network(layer_corruption_levels)
		network_layers = self.get_network_layers()

		denoising_auto_encoders = []
		for input_layer, dropout_layer, dense_layer in zip(network_layers[:-2], network_layers[1:-1], network_layers[2:]):
			if not (isinstance(dropout_layer, noise.LinearDropoutLayer) or isinstance(dropout_layer, noise.DropoutLayer)):
				continue
			if not isinstance(dense_layer, layers.DenseLayer):
				continue

			denoising_auto_encoder = DenoisingAutoEncoder(
				#layers.get_output(input_layer, deterministic=True),
				input_layer,
				dense_layer.num_units,
				dense_layer.nonlinearity,
				layer_corruption=1.-dropout_layer.activation_probability,
				W_encoder=dense_layer.W,
				objective_functions=objective_function,
				update_function=update_function,
				learning_rate=learning_rate,
				learning_rate_decay_style=learning_rate_decay_style,
				learning_rate_decay_parameter=learning_rate_decay_parameter,
			)

			print denoising_auto_encoder.get_network_params(trainable=True)
			denoising_auto_encoders.append(denoising_auto_encoder)

		pretrain_time = timeit.default_timer()
		for dae_index in range(len(denoising_auto_encoders)):
			denoising_auto_encoder = denoising_auto_encoders[dae_index]
			# layer_corruption_level = layer_corruption_levels[dae_index]

			layer_pretrain_time = 0
			for pretrain_epoch_index in range(number_of_epochs):
				layer_epoch_pretrain_time = denoising_auto_encoder.train(pretrain_data, minibatch_size)
				layer_pretrain_time += layer_epoch_pretrain_time
			logger.info('pretrain layer %i with denoising auto-encoder finishes in %fs' % (dae_index + 1, layer_pretrain_time))
			print 'pretrain layer %i with denoising auto-encoder finishes in %fs' % (dae_index + 1, layer_pretrain_time)

		pretrain_time = timeit.default_timer() - pretrain_time
		logger.info('pretrain network denoising auto-encoder finishes in %fs' % pretrain_time)
		print 'pretrain network denoising auto-encoder finishes in %fs' % pretrain_time
	'''


def main():
	from ..experiments import load_mnist

	train_set_x, train_set_y, validate_set_x, validate_set_y, test_set_x, test_set_y = load_mnist()
	train_set_x = numpy.reshape(train_set_x, (train_set_x.shape[0], numpy.prod(train_set_x.shape[1:])))
	validate_set_x = numpy.reshape(validate_set_x, (validate_set_x.shape[0], numpy.prod(validate_set_x.shape[1:])))
	test_set_x = numpy.reshape(test_set_x, (test_set_x.shape[0], numpy.prod(test_set_x.shape[1:])))

	train_dataset = train_set_x, train_set_y
	validate_dataset = validate_set_x, validate_set_y
	test_dataset = test_set_x, test_set_y

	input_shape = list(train_set_x.shape[1:])
	input_shape.insert(0, None)
	input_shape = tuple(input_shape)

	'''
	network=MultiLayerPerceptron(
		incoming = input_shape,

		layer_dimensions = [32, 64, 10],
		layer_nonlinearities = [nonlinearities.rectify, nonlinearities.rectify, nonlinearities.softmax],

		layer_activation_parameters = [0.8, 0.5, 0.5],
		layer_activation_styles = ["bernoulli", "bernoulli", "bernoulli"],

		objective_functions = objectives.categorical_crossentropy,
		update_function = updates.nesterov_momentum,
	)
	'''

	network = DynamicMultiLayerPerceptron(
		incoming=input_shape,

		dense_dimensions=[32, 64, 10],
		dense_nonlinearities=[nonlinearities.rectify, nonlinearities.rectify, nonlinearities.softmax],

		layer_activation_parameters=[0.8, 0.5, 0.5],
		layer_activation_styles=["bernoulli", "bernoulli", "bernoulli"],

		objective_functions=objectives.categorical_crossentropy,
		update_function=updates.nesterov_momentum,

		dropout_rate_update_interval=10,
		validation_interval=1000,
	)

	regularizer_functions = {}
	# regularizer_functions[regularization.l1] = 0.1
	# regularizer_functions[regularization.l2] = [0.2, 0.5, 0.4]
	# regularizer_functions[regularization.rademacher] = 1e-6
	network.set_regularizers(regularizer_functions)

	########################
	# START MODEL TRAINING #
	########################

	start_train = timeit.default_timer()
	# Finally, launch the training loop.
	# We iterate over epochs:
	number_of_epochs = 10
	minibatch_size = 1000
	for epoch_index in range(number_of_epochs):
		network.train(train_dataset, minibatch_size, validate_dataset, test_dataset)
		print("PROGRESS: %f%%" % (100. * epoch_index / number_of_epochs))
	end_train = timeit.default_timer()

	print("Optimization complete...")
	logger.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
		network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index))
	print('The code finishes in %.2fm' % ((end_train - start_train) / 60.))


if __name__ == '__main__':
	main()
