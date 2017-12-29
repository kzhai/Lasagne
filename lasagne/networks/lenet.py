import logging

import numpy

from . import FeedForwardNetwork
from .. import init, objectives, updates
from .. import layers

logger = logging.getLogger(__name__)

__all__ = [
	"LeNet",
]


class LeNet(FeedForwardNetwork):
	def __init__(self,
	             # input_network=None,
	             # input_shape,
	             incoming,

	             conv_filters,
	             conv_nonlinearities,
	             # convolution_filter_sizes=None,
	             # maxpooling_sizes=None,
	             pool_modes,

	             dense_dimensions,
	             dense_nonlinearities,

	             layer_activation_types,
	             layer_activation_parameters,
	             layer_activation_styles,

	             objective_functions=objectives.categorical_crossentropy,
	             update_function=updates.nesterov_momentum,

	             learning_rate_policy=1e-3,
	             max_norm_constraint=0,

	             validation_interval=-1,

	             conv_kernel_sizes=(5, 5),
	             conv_strides=(1, 1),
	             conv_pads=2,

	             pool_kernel_sizes=(3, 3),
	             pool_strides=(2, 2),
	             ):
		super(LeNet, self).__init__(incoming,
		                            objective_functions,
		                            update_function,
		                            learning_rate_policy,
		                            max_norm_constraint,
		                            validation_interval)

		# x = theano.tensor.matrix('x')  # the data is presented as rasterized images
		# self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

		# self._input_layer = layers.InputLayer(shape=input_shape)
		# self._input_variable = self._input_layer.input_var

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

		neural_network = layers.ReshapeLayer(neural_network,
		                                     (-1, numpy.prod(layers.get_output_shape(neural_network)[1:])))

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
