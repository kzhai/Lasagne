import logging
import numpy
import theano
import theano.tensor
import timeit

from .base import DiscriminativeNetwork
from .. import layers
from .. import init, nonlinearities, objectives, updates, regularization

__all__ = [
    "VariationalDropoutTypeANetwork",
    "VariationalDropoutTypeBNetwork",
]

class VariationalDropoutTypeANetwork(DiscriminativeNetwork):
    def __init__(self,
                 incoming,

                 layer_dimensions,
                 layer_nonlinearities,

                 layer_activation_parameters=None,
                 adaptive_styles = "layerwise",
                 variational_dropout_regularizer_lambdas=None,

                 objective_functions=objectives.categorical_crossentropy,
                 update_function=updates.nesterov_momentum,
                 learning_rate=1e-3,
                 learning_rate_decay_style=None,
                 learning_rate_decay_parameter=0,

                 validation_interval=-1,
                 ):
        super(VariationalDropoutTypeANetwork, self).__init__(incoming,
                                                             objective_functions,
                                                             update_function,
                                                             learning_rate,
                                                             learning_rate_decay_style,
                                                             learning_rate_decay_parameter,
                                                             validation_interval,
                                                             );

        #x = theano.tensor.matrix('x')  # the data is presented as rasterized images
        self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

        #self._input_layer = layers.InputLayer(shape=input_shape);
        #self._input_variable = self._input_layer.input_var;

        assert len(layer_dimensions) == len(layer_nonlinearities)
        assert len(layer_dimensions) == len(layer_activation_parameters)
        assert len(layer_dimensions) == len(variational_dropout_regularizer_lambdas)
        assert len(layer_dimensions) == len(adaptive_styles)
        assert (adaptive_style in set(["layerwise", "elementwise"]) for adaptive_style in adaptive_styles)

        '''
        pretrained_network_layers = None;
        if pretrained_model != None:
            pretrained_network_layers = lasagne.layers.get_all_layers(pretrained_model._neural_network);
        '''

        self.layers_coeff = {};

        #neural_network = input_network;
        neural_network = self._input_layer;
        for layer_index in xrange(len(layer_dimensions)):
            #previous_layer_dimension = layers.get_output_shape(neural_network)[1:];
            #activation_probability = noise.sample_activation_probability(previous_layer_dimension, layer_activation_styles[layer_index], layer_activation_parameters[layer_index]);

            #neural_network = noise.GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);

            layer_dimension = layer_dimensions[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index];

            neural_network = layers.VariationalDropoutTypeALayer(neural_network, activation_probability=layer_activation_parameters[layer_index], adaptive=adaptive_styles[layer_index])

            self.layers_coeff[neural_network] = variational_dropout_regularizer_lambdas[layer_index];

            neural_network = layers.DenseLayer(neural_network, layer_dimension, W=init.GlorotUniform(gain=init.GlorotUniformGain[layer_nonlinearity]), nonlinearity=layer_nonlinearity)

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

    def get_loss(self, label, **kwargs):
        loss = super(VariationalDropoutTypeANetwork, self).get_loss(label, **kwargs);

        '''
        loss += regularization.regularize_layer_params_weighted(self.layers_coeff, regularization.priorKL,
                                                                tags={"trainable":False,
                                                                      "regularizable": False,
                                                                      "adaptable": True}, **kwargs);
        '''

        loss += self.get_prior_KL();

        return loss;

    def get_prior_KL(self):
        return objectives.priorKL(self._neural_network)/self._output_variable.shape[0];

    def build_functions(self):
        super(VariationalDropoutTypeANetwork, self).build_functions();

        self._debug_function = theano.function(
            inputs=[self._input_variable, self._output_variable, self._learning_rate_variable],
            outputs=[super(VariationalDropoutTypeANetwork, self).get_loss(self._output_variable), self.get_prior_KL(), objectives.priorKL(self._neural_network)],
            on_unused_input='ignore'
        )

    def train_minibatch(self, minibatch_x, minibatch_y, learning_rate):
        minibatch_running_time = timeit.default_timer();
        train_function_outputs = self._train_function(minibatch_x, minibatch_y, learning_rate)
        minibatch_average_train_loss = train_function_outputs[0];
        minibatch_average_train_accuracy = train_function_outputs[1];
        minibatch_running_time = timeit.default_timer() - minibatch_running_time;

        #print self._debug_function(minibatch_x, minibatch_y, learning_rate);

        return minibatch_running_time, minibatch_average_train_loss, minibatch_average_train_accuracy

    '''
    def __set_regularizer_functions(self, regularizer_functions=None):
        _regularizer_functions = {};
        if regularizer_functions!=None:
            assert hasattr(self, "_neural_network");
            if type(regularizer_functions) is types.FunctionType:
                _regularizer_functions[regularizer_functions] = 1.0;
            elif type(regularizer_functions) is dict:
                for regularizer_function, layer_weight_mappings in regularizer_functions.iteritems():
                    assert type(regularizer_function) is types.FunctionType;
                    if type(layer_weight_mappings) is dict:
                        for layer, weight in layer_weight_mappings.iteritems():
                            assert isinstance(layer, layers.Layer);
                            assert type(weight)==float;
                    else:
                        assert type(layer_weight_mappings)==float;
                    _regularizer_functions[regularizer_function] = layer_weight_mappings;
            else:
                logging.error('unrecognized regularizer functions: %s' % (regularizer_functions))
        self._regularizer_functions = _regularizer_functions;
        self.regularizer_functions_change_stack.append((self.epoch_index, self._regularizer_functions));
    '''

class VariationalDropoutTypeBNetwork(DiscriminativeNetwork):
    def __init__(self,
                 incoming,

                 layer_dimensions,
                 layer_nonlinearities,

                 layer_activation_parameters=None,
                 adaptive_styles = "layerwise",
                 variational_dropout_regularizer_lambdas=None,

                 objective_functions=objectives.categorical_crossentropy,
                 update_function=updates.nesterov_momentum,
                 learning_rate=1e-3,
                 learning_rate_decay_style=None,
                 learning_rate_decay_parameter=0,

                 validation_interval=-1,
                 ):
        super(VariationalDropoutTypeBNetwork, self).__init__(incoming,
                                                             objective_functions,
                                                             update_function,
                                                             learning_rate,
                                                             learning_rate_decay_style,
                                                             learning_rate_decay_parameter,
                                                             validation_interval,
                                                             );

        #x = theano.tensor.matrix('x')  # the data is presented as rasterized images
        self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

        #self._input_layer = layers.InputLayer(shape=input_shape);
        #self._input_variable = self._input_layer.input_var;

        assert len(layer_dimensions) == len(layer_nonlinearities)
        assert len(layer_dimensions) == len(layer_activation_parameters)
        assert len(layer_dimensions) == len(variational_dropout_regularizer_lambdas)
        assert len(layer_dimensions) == len(adaptive_styles)
        assert (adaptive_style in set(["layerwise", "elementwise", "weightwise"]) for adaptive_style in adaptive_styles)

        '''
        pretrained_network_layers = None;
        if pretrained_model != None:
            pretrained_network_layers = lasagne.layers.get_all_layers(pretrained_model._neural_network);
        '''

        self.layers_coeff = {};

        #neural_network = input_network;
        neural_network = self._input_layer;
        for layer_index in xrange(len(layer_dimensions)):
            #previous_layer_dimension = layers.get_output_shape(neural_network)[1:];
            #activation_probability = noise.sample_activation_probability(previous_layer_dimension, layer_activation_styles[layer_index], layer_activation_parameters[layer_index]);

            #neural_network = noise.GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);

            layer_dimension = layer_dimensions[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index];

            neural_network = layers.DenseLayer(neural_network, layer_dimension, W=init.GlorotUniform(gain=init.GlorotUniformGain[layer_nonlinearity]), nonlinearity=layer_nonlinearity)

            neural_network = layers.VariationalDropoutTypeBLayer(neural_network, activation_probability=layer_activation_parameters[layer_index], adaptive=adaptive_styles[layer_index])

            self.layers_coeff[neural_network] = variational_dropout_regularizer_lambdas[layer_index];

            #print_output_dimension("checkpoint", neural_network, 100)
            #print_output("checkpoint", neural_network, numpy.random.random((100, 784)))

        self._neural_network = neural_network;

        self.build_functions();

    def get_loss(self, label, **kwargs):
        loss = super(VariationalDropoutTypeBNetwork, self).get_loss(label, **kwargs);

        '''
        loss += regularization.regularize_layer_params_weighted(self.layers_coeff, regularization.priorKL,
                                                                tags={"trainable":False,
                                                                      "regularizable": False,
                                                                      "adaptable": True}, **kwargs);
        '''

        loss += self.get_prior_KL();

        return loss;

    def get_prior_KL(self):
        return objectives.priorKL(self._neural_network)/self._output_variable.shape[0];

    def build_functions(self):
        super(VariationalDropoutTypeBNetwork, self).build_functions();

        self._debug_function = theano.function(
            inputs=[self._input_variable, self._output_variable, self._learning_rate_variable],
            outputs=[super(VariationalDropoutTypeBNetwork, self).get_loss(self._output_variable), self.get_prior_KL(),
                     objectives.priorKL(self._neural_network)],
            on_unused_input='ignore'
        )

    def train_minibatch(self, minibatch_x, minibatch_y, learning_rate):
        minibatch_running_time = timeit.default_timer();
        train_function_outputs = self._train_function(minibatch_x, minibatch_y, learning_rate)
        minibatch_average_train_loss = train_function_outputs[0];
        minibatch_average_train_accuracy = train_function_outputs[1];
        minibatch_running_time = timeit.default_timer() - minibatch_running_time;

        #print self._debug_function(minibatch_x, minibatch_y, learning_rate);

        return minibatch_running_time, minibatch_average_train_loss, minibatch_average_train_accuracy


def print_output_dimension(checkpoint_text, neural_network, batch_size):
    reference_to_input_layers = [input_layer for input_layer in layers.get_all_layers(neural_network) if
                                 isinstance(input_layer, layers.InputLayer)];
    print checkpoint_text, ":", layers.get_output_shape(neural_network, {reference_to_input_layers[0]: (batch_size, 784)})

def print_output(checkpoint_text, neural_network, data):
    reference_to_input_layers = [input_layer for input_layer in layers.get_all_layers(neural_network) if
                                 isinstance(input_layer, layers.InputLayer)];
    print checkpoint_text, ":", layers.get_output(neural_network, {reference_to_input_layers[0]: data}).eval()

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

    network = VariationalDropoutTypeANetwork(
        incoming=input_shape,

        layer_dimensions=[32, 64, 10],
        layer_nonlinearities=[nonlinearities.rectify, nonlinearities.rectify, nonlinearities.softmax],

        layer_activation_parameters=[0.8, 0.5, 0.5],
        adaptive_styles=["elementwise", "elementwise", "elementwise"],
        variational_dropout_regularizer_lambdas=[0.1, 0.1, 0.1],
        #variational_dropout_regularizer_lambdas=[1, 1, 1],

        objective_functions=objectives.categorical_crossentropy,
        update_function=updates.nesterov_momentum,

        learning_rate = 0.001,
        learning_rate_decay_style=None,
        learning_rate_decay_parameter=0,
        validation_interval=1000,
    )

    regularizer_functions = {};
    regularizer_functions[regularization.l1] = 0.1
    #regularizer_functions[regularization.l2] = 0.2

    network.set_regularizers(regularizer_functions)

    '''
    network=VariationalDropoutTypeBNetwork(
        incoming=input_shape,

        layer_dimensions=[32, 64, 10],
        layer_nonlinearities=[nonlinearities.rectify, nonlinearities.rectify, nonlinearities.softmax],

        layer_activation_parameters=[0.8, 0.5, 0.5],
        adaptive_styles=["weightwise", "weightwise", "weightwise"],
        variational_dropout_regularizer_lambdas=[2e-5, 2e-5, 2e-5],
        # variational_dropout_regularizer_lambdas=[1, 1, 1],

        objective_functions=objectives.categorical_crossentropy,
        update_function=updates.nesterov_momentum,

        learning_rate=0.001,
        learning_rate_decay_style=None,
        learning_rate_decay_parameter=0,
        validation_interval=1000,
    )
    '''
    print network.get_network_params(regularizable=False,adaptable=True);

    ########################
    # START MODEL TRAINING #
    ########################

    start_train = timeit.default_timer()
    # Finally, launch the training loop.
    # We iterate over epochs:
    number_of_epochs = 5;
    minibatch_size = 1000;

    for epoch_index in range(number_of_epochs):
        network.train(train_dataset, minibatch_size, validate_dataset, test_dataset);
        print "PROGRESS: %f%%" % (100. * epoch_index / number_of_epochs);
    end_train = timeit.default_timer()

    print "Optimization complete..."
    logging.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
        network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index));
    print 'The code finishes in %.2fm' % ((end_train - start_train) / 60.)

if __name__ == '__main__':
    main();