import logging
import timeit

import numpy

from . import DynamicFeedForwardNetwork, ElasticFeedForwardNetwork
from . import FeedForwardNetwork, adjust_parameter_according_to_policy
from .. import init, nonlinearities, objectives, Xpolicy, updates
from .. import layers

logger = logging.getLogger(__name__)

__all__ = [
	"DynamicMultiLayerPerceptron",
	# "VanillaMultiLayerPerceptron",
	"PrunableMultiLayerPerceptron",
]


class DynamicMultiLayerPerceptron(DynamicFeedForwardNetwork):
	def __init__(self,
	             incoming,

	             dense_dimensions,
	             dense_nonlinearities,

	             layer_activation_types,
	             layer_activation_parameters,
	             layer_activation_styles,

	             objective_functions=objectives.categorical_crossentropy,
	             update_function=updates.nesterov_momentum,

	             learning_rate_policy=[1e-3, Xpolicy.constant],
	             # learning_rate_decay=None,

	             dropout_learning_rate_policy=[1e-3, Xpolicy.constant],
	             # dropout_learning_rate_decay=None,
	             dropout_rate_update_interval=1,
	             # update_hidden_layer_dropout_only=False,

	             max_norm_constraint=0,
	             # learning_rate_decay_style=None,
	             # learning_rate_decay_parameter=0,
	             validation_interval=-1,
	             ):
		super(DynamicMultiLayerPerceptron, self).__init__(incoming=incoming,

		                                                  objective_functions=objective_functions,
		                                                  update_function=update_function,
		                                                  learning_rate_policy=learning_rate_policy,
		                                                  # learning_rate_decay,

		                                                  dropout_learning_rate_policy=dropout_learning_rate_policy,
		                                                  # dropout_learning_rate_decay,
		                                                  dropout_rate_update_interval=dropout_rate_update_interval,

		                                                  max_norm_constraint=max_norm_constraint,
		                                                  validation_interval=validation_interval,
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

			'''
			if update_hidden_layer_dropout_only and layer_index == 0:
				neural_network = layers.BernoulliDropoutLayer(neural_network,
				                                              activation_probability=activation_probability)
			else:
				neural_network = layers.AdaptiveDropoutLayer(neural_network,
				                                             activation_probability=activation_probability)
			'''
			# neural_network = layers.AdaptiveDropoutLayer(neural_network, activation_probability=activation_probability)

			neural_network = layer_activation_types[layer_index](neural_network,
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


class ElasticDynamicMultiLayerPerceptron(FeedForwardNetwork):
	def __init__(self,
	             incoming,

	             dense_dimensions,
	             dense_nonlinearities,

	             layer_activation_parameters=None,
	             layer_activation_styles=None,

	             objective_functions=objectives.categorical_crossentropy,
	             update_function=updates.nesterov_momentum,
	             learning_rate_policy=1e-3,
	             learning_rate_decay=None,
	             max_norm_constraint=0,
	             # learning_rate_decay_style=None,
	             # learning_rate_decay_parameter=0,

	             validation_interval=-1,
	             ):
		super(ElasticDynamicMultiLayerPerceptron, self).__init__(incoming,
		                                                         objective_functions,
		                                                         update_function,
		                                                         learning_rate_policy,
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

			neural_network = layers.PrunableDropoutLayer(neural_network,
			                                             activation_probability=activation_probability)

			layer_dimension = dense_dimensions[layer_index]
			layer_nonlinearity = dense_nonlinearities[layer_index]

			neural_network = layers.ElasticDenseLayer(neural_network, layer_dimension, W=init.GlorotUniform(
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

	def alter_network(self):
		print(self.get_network_params())
		for layer in self.get_network_layers():
			print("----------")
			print(layer, layer.get_params())

		for layer_1, layer_2, layer_3 in zip(self.get_network_layers()[:-2], self.get_network_layers()[1:-1],
		                                     self.get_network_layers()[2:]):
			if isinstance(layer_1, layers.ElasticDenseLayer):
				break

		layer_1.prune_output(100)
		# previous_layer_dimension = layers.get_output_shape(layer_1)[1:]
		# activation_probability = layers.sample_activation_probability(previous_layer_dimension, "Bernoulli", 0.5)
		layer_2.prune_activation_probability(layer_1)
		layer_3.prune_activation_probability(layer_2)

		print(self.get_network_params())
		for layer in self.get_network_layers():
			print("----------")
			print(layer, layer.get_params())

		'''
		neural_network = layer_1

		for layer_index in range(len(dense_dimensions)):
			previous_layer_dimension = layers.get_output_shape(neural_network)[1:]
			activation_probability = noise.sample_activation_probability(previous_layer_dimension,
			                                                             layer_activation_styles[layer_index],
			                                                             layer_activation_parameters[layer_index])

			neural_network = noise.LinearDropoutLayer(neural_network,
			                                          activation_probability=activation_probability)

			layer_dimension = dense_dimensions[layer_index]
			layer_nonlinearity = dense_nonlinearities[layer_index]

			neural_network = layers.DenseLayer(neural_network, layer_dimension, W=init.GlorotUniform(
				gain=init.GlorotUniformGain[layer_nonlinearity]), nonlinearity=layer_nonlinearity)

		self._neural_network = neural_network
		'''

		self.build_functions()


class PrunableMultiLayerPerceptron(ElasticFeedForwardNetwork):
	def __init__(self,
	             incoming,

	             dense_dimensions,
	             dense_nonlinearities,

	             prune_policy,
	             # prune_synapses_after,
	             # prune_synapses_interval,
	             # prune_synapses_threshold,

	             # layer_activation_types,
	             layer_activation_parameters,
	             layer_activation_styles,

	             objective_functions=objectives.categorical_crossentropy,
	             update_function=updates.nesterov_momentum,
	             learning_rate_policy=[1e-3, Xpolicy.constant],
	             max_norm_constraint=0,

	             validation_interval=-1,
	             ):
		super(PrunableMultiLayerPerceptron, self).__init__(incoming,
		                                                   objective_functions=objective_functions,
		                                                   update_function=update_function,
		                                                   learning_rate_policy=learning_rate_policy,
		                                                   prune_policy=prune_policy,
		                                                   max_norm_constraint=max_norm_constraint,
		                                                   validation_interval=validation_interval,
		                                                   )
		# x = theano.tensor.matrix('x')  # the data is presented as rasterized images
		# self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

		# self._input_layer = layers.InputLayer(shape=input_shape)
		# self._input_variable = self._input_layer.input_var

		assert len(dense_dimensions) == len(dense_nonlinearities)
		assert len(dense_dimensions) == len(layer_activation_parameters)
		assert len(dense_dimensions) == len(layer_activation_styles)

		# neural_network = input_network
		neural_network = self._input_layer
		for layer_index in range(len(dense_dimensions)):
			previous_layer_dimension = layers.get_output_shape(neural_network)[1:]
			activation_probability = layers.sample_activation_probability(previous_layer_dimension,
			                                                              layer_activation_styles[layer_index],
			                                                              layer_activation_parameters[layer_index])

			neural_network = layers.PrunableDropoutLayer(neural_network,
			                                             activation_probability=activation_probability)

			layer_dimension = dense_dimensions[layer_index]
			layer_nonlinearity = dense_nonlinearities[layer_index]

			neural_network = layers.PrunableDenseLayer(neural_network, layer_dimension, W=init.GlorotUniform(
				gain=init.GlorotUniformGain[layer_nonlinearity]), nonlinearity=layer_nonlinearity)

		self._neural_network = neural_network

		self.build_functions()

	def adjust_network(self, train_dataset=None, validate_dataset=None, test_dataset=None):
		self.prune_synapses(train_dataset, validate_dataset, test_dataset)

	def prune_synapses(self, train_dataset=None, validate_dataset=None, test_dataset=None,
	                   dropout_decay_style="elementwise"):
		connection_threshold = adjust_parameter_according_to_policy(self._prune_policy, self.epoch_index)

		layer_info_list = []
		for layer_0, layer_1 in zip(self.get_network_layers()[:-1], self.get_network_layers()[1:]):
			if (not isinstance(layer_0, layers.PrunableDropoutLayer)) or \
					(not isinstance(layer_1, layers.PrunableDenseLayer)):
				continue

			# input_size = layers.get_output_shape(layer_0)[1:]
			# old_sizes.append(layer_1.num_units)
			if dropout_decay_style == "layerwise":
				C_old = numpy.sum(layer_1.mask)
			elif dropout_decay_style == "elementwise":
				C_old = numpy.sum(layer_1.mask, axis=1)
			neuron_indices_to_keep = layer_1.prune_weight(connection_threshold)
			if dropout_decay_style == "layerwise":
				C_new = numpy.sum(layer_1.mask)
			elif dropout_decay_style == "elementwise":
				C_new = numpy.sum(layer_1.mask, axis=1)

			assert len(neuron_indices_to_keep) <= layer_1.num_units, (len(neuron_indices_to_keep), neuron_indices_to_keep, layer_1.num_units)
			layer_info_list.append(neuron_indices_to_keep)

			if numpy.all(C_old==C_new):
				continue
			print("Adjusting number of connections in layer %s from %d to %d" % (layer_1, numpy.sum(C_old),
			                                                                     numpy.sum(C_new)))
			logger.info("Adjusting number of connections in layer %s from %d to %d" % (layer_1, numpy.sum(C_old),
			                                                                           numpy.sum(C_new)))
			layer_0.decay_activation_probability(numpy.sqrt(C_new / C_old))

		layer_info_index = 0
		for layer_1, layer_2, layer_3 in zip(self.get_network_layers()[:-2], self.get_network_layers()[1:-1],
		                                     self.get_network_layers()[2:]):
			if (not isinstance(layer_1, layers.PrunableDenseLayer)) or \
					(not isinstance(layer_2, layers.PrunableDropoutLayer)) or \
					(not isinstance(layer_3, layers.PrunableDenseLayer)):
				continue

			neuron_indices_to_keep = layer_info_list[layer_info_index]
			layer_info_index += 1

			if len(neuron_indices_to_keep) == layer_1.num_units:
				continue
			old_size = layer_1.num_units
			new_size = len(neuron_indices_to_keep)

			layer_1.prune_output(neuron_indices_to_keep)
			layer_2.prune_activation_probability(neuron_indices_to_keep)
			layer_3.prune_input(neuron_indices_to_keep)
			print("Adjusting number of units in layer %s from %d to %d" % (layer_1, old_size, new_size))
			logger.info("Adjusting number of units in layer %s from %d to %d" % (layer_1, old_size, new_size))

		self.build_functions()


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

	network = PrunableMultiLayerPerceptron(
		incoming=input_shape,

		dense_dimensions=[1024, 10],
		dense_nonlinearities=[nonlinearities.rectify, nonlinearities.softmax],

		layer_activation_parameters=[0.8, 0.5],
		layer_activation_styles=["bernoulli", "bernoulli"],

		objective_functions=objectives.categorical_crossentropy,
		update_function=updates.nesterov_momentum,

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
