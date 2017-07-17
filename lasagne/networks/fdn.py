import logging
import numpy
import theano
import theano.tensor

from .base import DiscriminativeNetwork
#from .dae import DenoisingAutoEncoder
from .. import layers
#from ..layers import noise
from .. import init, nonlinearities, objectives, updates

__all__ = [
    "FastDropoutNetwork",
]

class FastDropoutNetwork(DiscriminativeNetwork):
    def __init__(self,
                 incoming,

                 layer_dimensions,
                 layer_nonlinearities,

                 layer_activation_parameters=None,

                 objective_functions=objectives.categorical_crossentropy,
                 update_function=updates.nesterov_momentum,
                 learning_rate=1e-3,
                 learning_rate_decay=None,
                 max_norm_constraint=0,
                 #learning_rate_decay_style=None,
                 #learning_rate_decay_parameter=0,

                 validation_interval=-1,
                 ):
        super(FastDropoutNetwork, self).__init__(incoming,
                                                 objective_functions,
                                                 update_function,
                                                 learning_rate,
                                                 learning_rate_decay,
                                                 max_norm_constraint,
                                                 #learning_rate_decay_style,
                                                 #learning_rate_decay_parameter,
                                                 validation_interval,
                                                 );

        #x = theano.tensor.matrix('x')  # the data is presented as rasterized images
        self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

        #self._input_layer = layers.InputLayer(shape=input_shape);
        #self._input_variable = self._input_layer.input_var;

        assert len(layer_dimensions) == len(layer_nonlinearities)
        assert len(layer_dimensions) == len(layer_activation_parameters)
        #assert len(layer_dimensions) == len(layer_activation_styles)

        '''
        pretrained_network_layers = None;
        if pretrained_model != None:
            pretrained_network_layers = lasagne.layers.get_all_layers(pretrained_model._neural_network);
        '''

        #neural_network = input_network;
        neural_network = self._input_layer;
        for layer_index in range(len(layer_dimensions)):
            #previous_layer_dimension = layers.get_output_shape(neural_network)[1:];
            #activation_probability = noise.sample_activation_probability(previous_layer_dimension, layer_activation_styles[layer_index], layer_activation_parameters[layer_index]);

            #neural_network = noise.GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);

            layer_dimension = layer_dimensions[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index];

            neural_network = layers.DenseLayer(neural_network, layer_dimension, W=init.GlorotUniform(gain=init.GlorotUniformGain[layer_nonlinearity]), nonlinearity=layer_nonlinearity)

            neural_network = layers.FastDropoutLayer(neural_network, activation_probability=layer_activation_parameters[layer_index])

            #print_output_dimension("checkpoint", neural_network, 100)
            #print_output("checkpoint", neural_network, numpy.random.random((100, 784)))

            '''
            if pretrained_network_layers == None or len(pretrained_network_layers) <= layer_index:
                _neural_network = lasagne.layers.DenseLayer(_neural_network, layer_dimension, nonlinearity=layer_nonlinearity)
            else:
                pretrained_layer = pretrained_network_layers[layer_index];
                assert isinstance(pretrained_layer, lasagne.layers.DenseLayer)
                assert pretrained_layer.nonlinearity == layer_nonlinearity, (pretrained_layer.nonlinearity, layer_nonlinearity)
                assert pretrained_layer.num_units == layer_dimension

                _neural_network = lasagne.layers.DenseLayer(_neural_network,
                                                    layer_dimension,
                                                    W=pretrained_layer.W,
                                                    b=pretrained_layer.b,
                                                    nonlinearity=layer_nonlinearity)
            '''

        self._neural_network = neural_network;

        self.build_functions();

def print_output_dimension(checkpoint_text, neural_network, batch_size):
    reference_to_input_layers = [input_layer for input_layer in layers.get_all_layers(neural_network) if
                                 isinstance(input_layer, layers.InputLayer)];
    print(checkpoint_text, ":", layers.get_output_shape(neural_network, {reference_to_input_layers[0]: (batch_size, 784)}))

def print_output(checkpoint_text, neural_network, data):
    reference_to_input_layers = [input_layer for input_layer in layers.get_all_layers(neural_network) if
                                 isinstance(input_layer, layers.InputLayer)];
    print(checkpoint_text, ":", layers.get_output(neural_network, {reference_to_input_layers[0]: data}).eval())

def main():
    from ..experiments import load_mnist

    train_set_x, train_set_y, validate_set_x, validate_set_y, test_set_x, test_set_y = load_mnist();
    train_set_x = numpy.reshape(train_set_x, (train_set_x.shape[0], numpy.prod(train_set_x.shape[1:])))
    validate_set_x = numpy.reshape(validate_set_x, (validate_set_x.shape[0], numpy.prod(validate_set_x.shape[1:])))
    test_set_x = numpy.reshape(test_set_x, (test_set_x.shape[0], numpy.prod(test_set_x.shape[1:])))

    train_dataset = train_set_x, train_set_y
    validate_dataset = validate_set_x, validate_set_y
    test_dataset = test_set_x, test_set_y

    input_shape = list(train_set_x.shape[1:]);
    input_shape.insert(0, None)
    input_shape = tuple(input_shape)

    network = FastDropoutNetwork(
        incoming=input_shape,

        layer_dimensions=[32, 64, 10],
        layer_nonlinearities=[nonlinearities.rectify, nonlinearities.rectify, nonlinearities.softmax],
        layer_activation_parameters=[0.8, 0.5, 0.5],

        objective_functions=objectives.categorical_crossentropy,
        update_function=updates.nesterov_momentum,
    )

    #print layers.get_all_params(network)
    #print network.get_network_params(adaptable=True);
    #network.set_L1_regularizer_lambda([0, 0, 0])
    #network.set_L2_regularizer_lambda([0, 0, 0])

    ########################
    # START MODEL TRAINING #
    ########################
    import timeit
    start_train = timeit.default_timer()
    # Finally, launch the training loop.
    # We iterate over epochs:
    number_of_epochs = 10;
    minibatch_size = 1000;
    for epoch_index in range(number_of_epochs):
        network.train(train_dataset, minibatch_size, validate_dataset, test_dataset);
        print("PROGRESS: %f%%" % (100. * epoch_index / number_of_epochs));
    end_train = timeit.default_timer()

    print("Optimization complete...")
    logging.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
        network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index));
    print('The code finishes in %.2fm' % ((end_train - start_train) / 60.))