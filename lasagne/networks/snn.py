import logging

import numpy
import theano
import theano.tensor

from . import FeedForwardNetwork
from .. import init, nonlinearities, objectives, updates
from .. import layers

logger = logging.getLogger(__name__)

__all__ = [
	#"StandoutNeuralNetworkTypeA",
	"StandoutNeuralNetworkTypeB",
]


class StandoutNeuralNetworkTypeB(FeedForwardNetwork):
	def __init__(self,
	             incoming,

	             dense_dimensions,
	             dense_nonlinearities,

	             input_activation_rate=1.0,
	             # layer_activation_parameters=None,
	             # layer_activation_styles=None,
	             # pretrained_model=None,

	             objective_functions=objectives.categorical_crossentropy,
	             update_function=updates.nesterov_momentum,

	             learning_rate_policy=1e-3,
	             # learning_rate_decay=None,
	             max_norm_constraint=0,
	             # learning_rate_decay_style=None,
	             # learning_rate_decay_parameter=0,

	             validation_interval=-1,
	             ):
		super(StandoutNeuralNetworkTypeB, self).__init__(incoming,
		                                                 objective_functions,
		                                                 update_function,
		                                                 learning_rate_policy,
		                                                 # learning_rate_decay,
		                                                 max_norm_constraint,
		                                                 # learning_rate_decay_style,
		                                                 # learning_rate_decay_parameter,
		                                                 validation_interval)

		# self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

		assert len(dense_dimensions) == len(dense_nonlinearities)
		# assert len(layer_dimensions) == len(layer_activation_parameters)
		# assert len(layer_dimensions) == len(layer_activation_styles)

		neural_network = self._input_layer

		previous_layer_dimension = layers.get_output_shape(neural_network)[1:]
		activation_probability = numpy.zeros(previous_layer_dimension) + input_activation_rate
		neural_network = layers.AdaptiveDropoutLayer(neural_network, activation_probability=activation_probability)

		for layer_index in range(len(dense_dimensions)):
			layer_dimension = dense_dimensions[layer_index]
			layer_nonlinearity = dense_nonlinearities[layer_index]

			dense_layer = layers.DenseLayer(neural_network,
			                                layer_dimension,
			                                W=init.GlorotUniform(gain=init.GlorotUniformGain[layer_nonlinearity]),
			                                nonlinearity=layer_nonlinearity)

			if layer_index < len(dense_dimensions) - 1:
				dropout_layer = layers.StandoutLayer(neural_network, layer_dimension, W=dense_layer.W, b=dense_layer.b)
				neural_network = layers.ElemwiseMergeLayer([dense_layer, dropout_layer], theano.tensor.mul)
			else:
				neural_network = dense_layer

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

	network = StandoutNeuralNetworkTypeB(
		incoming=input_shape,

		dense_dimensions=[32, 64, 10],
		dense_nonlinearities=[nonlinearities.rectify, nonlinearities.rectify, nonlinearities.softmax],
		input_activation_rate=0.8,

		objective_functions=objectives.categorical_crossentropy,
		update_function=updates.nesterov_momentum,
	)

	# print layers.get_all_params(network)
	print(network.get_network_params(trainable=True))
	print(network.get_network_params(trainable=True, adaptable=True))
	print(network.get_network_params(trainable=True, adaptable=False))
	print(network.get_network_params(regularizable=False))
	print(network.get_network_params(regularizable=False, adaptable=True))
	print(network.get_network_params(regularizable=False, adaptable=False))
	# network.set_L1_regularizer_lambda([0, 0, 0])
	# network.set_L2_regularizer_lambda([0, 0, 0])

	########################
	# START MODEL TRAINING #
	########################
	import timeit
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
