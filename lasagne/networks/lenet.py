import logging

import numpy

from . import FeedForwardNetwork
from .. import init, objectives, updates
from .. import layers

logger = logging.getLogger(__name__)

__all__ = [
	"LeNet",
	# "AdaptiveLeNet",
	"LeNetFromPretrainedModel",
]


def LeNet(input_layer,

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

          conv_kernel_sizes=(5, 5),
          conv_strides=(1, 1),
          conv_pads=2,

          pool_kernel_sizes=(3, 3),
          pool_strides=(2, 2),
          ):
	assert len(layer_activation_types) == len(dense_nonlinearities) + len(conv_nonlinearities)
	assert len(layer_activation_parameters) == len(dense_nonlinearities) + len(conv_nonlinearities)
	assert len(layer_activation_styles) == len(dense_nonlinearities) + len(conv_nonlinearities)
	assert len(conv_filters) == len(conv_nonlinearities)
	assert len(conv_filters) == len(pool_modes)

	dropout_layer_index = 0
	neural_network = input_layer
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

	# neural_network = layers.ReshapeLayer(neural_network, (-1, numpy.prod(layers.get_output_shape(neural_network)[1:])))

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

	return neural_network


def LeNetFromPretrainedModel(input_layer, pretrained_network):
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
		if isinstance(layer, layers.Conv2DLayer):
			neural_network = layers.Conv2DLayer(neural_network,
			                                    W=init.GlorotUniform(gain=init.GlorotUniformGain[layer.nonlinearity]),
			                                    b=init.Constant(0.),
			                                    nonlinearity=layer.nonlinearity,
			                                    num_filters=layer.num_filters,
			                                    filter_size=layer.filter_size,
			                                    stride=layer.stride,
			                                    pad=layer.pad,
			                                    )
		if isinstance(layer, layers.Pool2DLayer):
			neural_network = layers.Pool2DLayer(neural_network, pool_size=layer.pool_size, stride=layer.stride,
			                                    mode=layer.mode)

	return neural_network


def AdaptiveLeNetFromPretrainedModel(input_layer, pretrained_network):
	neural_network = input_layer

	for layer in layers.get_all_layers(pretrained_network):
		if isinstance(layer, layers.DenseLayer):
			neural_network = layers.DenseLayer(neural_network, layer.num_units, W=layer.W, b=layer.b,
			                                   nonlinearity=layer.nonlinearity)
		if isinstance(layer, layers.BernoulliDropoutLayer):
			neural_network = layers.AdaptiveDropoutLayer(neural_network,
			                                             activation_probability=layer.activation_probability)
		if isinstance(layer, layers.Conv2DLayer):
			neural_network = layers.Conv2DLayer(neural_network,
			                                    W=init.GlorotUniform(gain=init.GlorotUniformGain[layer.nonlinearity]),
			                                    b=init.Constant(0.),
			                                    nonlinearity=layer.nonlinearity,
			                                    num_filters=layer.num_filters,
			                                    filter_size=layer.filter_size,
			                                    stride=layer.stride,
			                                    pad=layer.pad,
			                                    )
		if isinstance(layer, layers.Pool2DLayer):
			neural_network = layers.Pool2DLayer(neural_network, pool_size=layer.pool_size, stride=layer.stride,
			                                    mode=layer.mode)

	return neural_network


'''
def AdaptiveLeNet(input_layer,

	             conv_filters,
	             conv_nonlinearities,
	             # convolution_filter_sizes=None,
	             pool_modes,

	             dense_dimensions,
	             dense_nonlinearities,

	             layer_activation_types,
	             layer_activation_parameters,
	             layer_activation_styles,

	             conv_kernel_sizes=(5, 5),
	             conv_strides=(1, 1),
	             conv_pads=2,

	             pool_kernel_sizes=(3, 3),
	             pool_strides=(2, 2),
	             ):

		assert len(layer_activation_types) == len(dense_nonlinearities) + len(conv_nonlinearities)
		assert len(layer_activation_parameters) == len(dense_nonlinearities) + len(conv_nonlinearities)
		assert len(layer_activation_styles) == len(dense_nonlinearities) + len(conv_nonlinearities)
		assert len(conv_filters) == len(conv_nonlinearities)
		assert len(conv_filters) == len(pool_modes)

		dropout_layer_index = 0
		neural_network = input_layer
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

		return neural_network
'''
