import logging
import timeit

import numpy

from . import FeedForwardNetwork
from .. import init, nonlinearities, objectives, updates, Xpolicy
from .. import layers

logger = logging.getLogger(__name__)

__all__ = [
	"MultiLayerPerceptron",
]


class MultiLayerPerceptron(FeedForwardNetwork):
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
	             max_norm_constraint=0,

	             validation_interval=-1,
	             ):
		super(MultiLayerPerceptron, self).__init__(incoming,
		                                           objective_functions,
		                                           update_function,
		                                           learning_rate_policy,
		                                           max_norm_constraint,
		                                           validation_interval,
		                                           )

		# x = theano.tensor.matrix('x')  # the data is presented as rasterized images
		# self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

		# self._input_layer = layers.InputLayer(shape=input_shape)
		# self._input_variable = self._input_layer.input_var

		assert len(dense_dimensions) == len(layer_activation_types)
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

	network = MultiLayerPerceptron(
		incoming=input_shape,

		dense_dimensions=[1024, 10],
		dense_nonlinearities=[nonlinearities.rectify, nonlinearities.softmax],

		layer_activation_types=[layers.AdaptiveDropoutLayer, layers.AdaptiveDropoutLayer],
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
	minibatch_size = 100
	for epoch_index in range(number_of_epochs):
		network.train(train_dataset, validate_dataset, test_dataset, minibatch_size)
		print("PROGRESS: %f%%" % (100. * epoch_index / number_of_epochs))
	end_train = timeit.default_timer()

	print("Optimization complete...")
	logger.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
		network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index))
	print('The code finishes in %.2fm' % ((end_train - start_train) / 60.))


if __name__ == '__main__':
	main()
