import logging
import timeit

import numpy

from lasagne import Xinit
from lasagne import init, nonlinearities, objectives, updates
from lasagne import layers

logger = logging.getLogger(__name__)

__all__ = [
	"MultiLayerPerceptronFromSpecifications",
	"MultiLayerPerceptronFromPretrainedModel",
	#
	"AdaptedMultiLayerPerceptronFromSpecifications",
	"AdaptedMultiLayerPerceptronFromPretrainedModel",
]


def MultiLayerPerceptronFromSpecifications(
		input_layer,
		dense_dimensions,
		dense_nonlinearities,

		layer_activation_types,
		layer_activation_parameters,
		layer_activation_styles
):
	assert len(dense_dimensions) == len(layer_activation_types)
	assert len(dense_dimensions) == len(dense_nonlinearities)
	assert len(dense_dimensions) == len(layer_activation_parameters)
	assert len(dense_dimensions) == len(layer_activation_styles)

	neural_network = input_layer
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
			gain=Xinit.GlorotUniformGain[layer_nonlinearity]), nonlinearity=layer_nonlinearity)

	return neural_network


def MultiLayerPerceptronFromPretrainedModel(
		input_layer,
		pretrained_network,
):
	neural_network = input_layer

	for layer in layers.get_all_layers(pretrained_network):
		if isinstance(layer, layers.DenseLayer):
			# print(neural_network, layer.num_units, layer.W, layer.b, layer.nonlinearity)
			neural_network = layers.DenseLayer(neural_network, layer.num_units, W=layer.W, b=layer.b,
			                                   nonlinearity=layer.nonlinearity)
		if isinstance(layer, layers.BernoulliDropoutLayer):
			# print(neural_network, layer.activation_probability)
			neural_network = layers.BernoulliDropoutLayer(neural_network,
			                                              activation_probability=layer.activation_probability)

	return neural_network


AdaptedMultiLayerPerceptronFromSpecifications = MultiLayerPerceptronFromSpecifications


def AdaptedMultiLayerPerceptronFromPretrainedModel(input_layer, pretrained_network):
	neural_network = input_layer

	for layer in layers.get_all_layers(pretrained_network):
		if isinstance(layer, layers.DenseLayer):
			# print(neural_network, layer.num_units, layer.W, layer.b, layer.nonlinearity)
			neural_network = layers.DenseLayer(neural_network, layer.num_units, W=layer.W, b=layer.b,
			                                   nonlinearity=layer.nonlinearity)
		if isinstance(layer, layers.BernoulliDropoutLayer):
			# print(neural_network, layer.activation_probability)
			neural_network = layers.AdaptiveDropoutLayer(neural_network,
			                                             activation_probability=layer.activation_probability)

	return neural_network


'''
def AdaptiveMultiLayerPerceptronFromSpecifications(input_layer,
                                                   dense_dimensions,
                                                   dense_nonlinearities,
                                                   layer_activation_types,
                                                   layer_activation_parameters,
                                                   layer_activation_styles,
                                                   ):
	assert len(dense_dimensions) == len(layer_activation_types)
	assert len(dense_dimensions) == len(dense_nonlinearities)
	assert len(dense_dimensions) == len(layer_activation_parameters)
	assert len(dense_dimensions) == len(layer_activation_styles)

	neural_network = input_layer
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
			gain=Xinit.GlorotUniformGain[layer_nonlinearity]), nonlinearity=layer_nonlinearity)

	return neural_network
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

	network = MultiLayerPerceptronFromSpecifications(
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
