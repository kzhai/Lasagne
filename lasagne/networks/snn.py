import logging
import numpy
import theano
import theano.tensor

from .base import Network, DiscriminativeNetwork, decay_learning_rate
from .. import layers
from ..layers import noise
from .. import init, nonlinearities, objectives, updates

__all__ = [
    "StandoutNeuralNetworkTypeA",
    "StandoutNeuralNetworkTypeB",
]

class StandoutNeuralNetworkTypeA(DiscriminativeNetwork):
    def __init__(self,
                 incoming,

                 layer_dimensions,
                 layer_nonlinearities,

                 input_activation_rate=1.0,
                 # layer_activation_parameters=None,
                 # layer_activation_styles=None,
                 # pretrained_model=None,

                 objective_functions=objectives.categorical_crossentropy,
                 update_function=updates.nesterov_momentum,
                 learning_rate=1e-3,
                 learning_rate_decay_style=None,
                 learning_rate_decay_parameter=0,
                 validation_interval=-1,
                 ):
        super(StandoutNeuralNetworkTypeA, self).__init__(incoming,
                                                         objective_functions,
                                                         update_function,
                                                         learning_rate,
                                                         learning_rate_decay_style,
                                                         learning_rate_decay_parameter,
                                                         validation_interval)

        self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

        assert len(layer_dimensions) == len(layer_nonlinearities)
        # assert len(layer_dimensions) == len(layer_activation_parameters)
        # assert len(layer_dimensions) == len(layer_activation_styles)

        neural_network = self._input_layer;

        for layer_index in xrange(len(layer_dimensions)):
            previous_layer_dimension = layers.get_output_shape(neural_network)[1:];
            # activation_probability = noise.sample_activation_probability(previous_layer_dimension, layer_activation_styles[layer_index], layer_activation_parameters[layer_index]);
            # activation_probability = sample_activation_probability(previous_layer_dimension, layer_activation_styles[layer_index], layer_activation_parameters[layer_index]);

            if layer_index == 0:
                # activation_probability = sample_activation_probability(previous_layer_dimension, layer_activation_styles[0], layer_activation_parameters[0]);
                # neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);
                neural_network = layers.LinearDropoutLayer(neural_network,
                                                           activation_probability=input_activation_rate);

            layer_dimension = layer_dimensions[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index];

            dense_layer = layers.DenseLayer(neural_network,
                                            layer_dimension,
                                            W=init.GlorotUniform(gain=init.GlorotUniformGain[layer_nonlinearity]),
                                            nonlinearity=layer_nonlinearity)

            if layer_index < len(layer_dimensions) - 1:
                dropout_layer = layers.StandoutLayer(neural_network, layer_dimension)
                neural_network = layers.ElemwiseMergeLayer([dense_layer, dropout_layer], theano.tensor.mul);
            else:
                neural_network = dense_layer;

        self._neural_network = neural_network;

        self.build_functions();

    def get_snn_objectives(self, label, **kwargs):
        output = self.get_output(**kwargs);
        #temp_objective_function = getattr(objectives, "categorical_crossentropy");
        objective = nonlinearities.sigmoid(theano.tensor.mean(
            objectives.categorical_crossentropy(output, label), dtype=theano.config.floatX)
            - theano.tensor.log(N));
        return objective

    def build_functions(self):
        # Create a train_loss expression for training, i.e., a scalar objective we want to minimize (for our multi-class problem, it is the cross-entropy train_loss):
        train_loss = self.get_loss(self._output_variable);
        train_accuracy = self.get_objective(self._output_variable, objective_function="categorical_accuracy");
        # train_prediction = self.get_output(**kwargs)
        # train_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(train_prediction, axis=1), self._output_variable), dtype=theano.config.floatX)

        # Create update expressions for training, i.e., how to modify the parameters at each training step. Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        trainable_inadaptable_params = self.get_network_params(trainable=True, adaptable=False)
        trainable_inadaptable_params_updates = self._update_function(train_loss, trainable_inadaptable_params, self._learning_rate_variable, momentum=0.95)

        # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) and returning the corresponding training train_loss:
        self._train_function = theano.function(
            inputs=[self._input_variable, self._output_variable, self._learning_rate_variable],
            outputs=[train_loss, train_accuracy],
            updates=trainable_inadaptable_params_updates
        )

        #
        #
        #
        #
        #

        # Create a train_loss expression for training, i.e., a scalar objective we want to minimize (for our multi-class problem, it is the cross-entropy train_loss):
        snn_train_loss = self.get_snn_objectives(self._output_variable, deterministic=True);

        # Create update expressions for training, i.e., how to modify the parameters at each training step. Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        trainable_adaptable_params = self.get_network_params(trainable=True, adaptable=True)
        trainable_adaptable_params_updates = self._update_function(snn_train_loss, trainable_adaptable_params, self._learning_rate_variable, momentum=0.95)

        #
        #
        #
        #
        #

        # Create a train_loss expression for validation/testing. The crucial difference here is that we do a deterministic forward pass through the networks, disabling dropout layers.
        test_loss = self.get_loss(self._output_variable, deterministic=True);
        test_accuracy = self.get_objective(self._output_variable, objective_function="categorical_accuracy",
                                           deterministic=True);
        # As a bonus, also create an expression for the classification accuracy:
        # test_prediction = self.get_output(deterministic=True)
        # test_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(test_prediction, axis=1), self._output_variable), dtype=theano.config.floatX)

        # Compile a second function computing the validation train_loss and accuracy:
        self._test_function = theano.function(
            inputs=[self._input_variable, self._output_variable],
            outputs=[test_loss, test_accuracy],
        )

    def train(self, train_dataset, minibatch_size, validate_dataset=None, test_dataset=None, output_directory=None):
        # In each epoch_index, we do a full pass over the training data:
        epoch_running_time = 0;

        train_dataset_x, train_dataset_y = train_dataset

        number_of_data = train_dataset_x.shape[0];
        data_indices = numpy.random.permutation(number_of_data);
        minibatch_start_index = 0;

        total_train_loss = 0;
        total_train_accuracy = 0;
        while minibatch_start_index < number_of_data:
            # automatically handles the left-over data
            minibatch_indices = data_indices[minibatch_start_index:minibatch_start_index + minibatch_size];
            minibatch_start_index += minibatch_size;

            minibatch_x = train_dataset_x[minibatch_indices, :]
            minibatch_y = train_dataset_y[minibatch_indices]

            learning_rate = decay_learning_rate(self.minibatch_index, self.learning_rate,
                                                self.learning_rate_decay_style, self.learning_rate_decay_parameter);

            minibatch_running_time = timeit.default_timer();
            # print self._debug_function(minibatch_x, minibatch_y, learning_rate);

            train_function_outputs = self._train_function(minibatch_x, minibatch_y, learning_rate)
            minibatch_average_train_loss = train_function_outputs[0];
            minibatch_average_train_accuracy = train_function_outputs[1];
            minibatch_running_time = timeit.default_timer() - minibatch_running_time;
            epoch_running_time += minibatch_running_time

            current_minibatch_size = len(data_indices[minibatch_start_index:minibatch_start_index + minibatch_size])
            total_train_loss += minibatch_average_train_loss * current_minibatch_size;
            total_train_accuracy += minibatch_average_train_accuracy * current_minibatch_size;

            # average_train_accuracy = total_train_accuracy / number_of_data;
            # average_train_loss = total_train_loss / number_of_data;
            # logging.debug('train: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (
            # self.epoch_index, self.minibatch_index, average_train_loss, average_train_accuracy * 100))

            # And a full pass over the validation data:
            if validate_dataset != None and self.validation_interval > 0 and self.minibatch_index % self.validation_interval == 0:
                average_train_accuracy = total_train_accuracy / number_of_data;
                average_train_loss = total_train_loss / number_of_data;
                logging.info('train: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (
                    self.epoch_index, self.minibatch_index, average_train_loss, average_train_accuracy * 100))

                output_file = None;
                if output_directory != None:
                    output_file = os.path.join(output_directory, 'model.pkl')
                self.validate(validate_dataset, test_dataset, output_file);

            self.minibatch_index += 1;

        if validate_dataset != None:
            output_file = None;
            if output_directory != None:
                output_file = os.path.join(output_directory, 'model.pkl')
            self.validate(validate_dataset, test_dataset, output_file);
        elif test_dataset != None:
            # if output_directory != None:
            # output_file = os.path.join(output_directory, 'model-%d.pkl' % self.epoch_index)
            # cPickle.dump(self, open(output_file, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);
            self.test(test_dataset);

        average_train_accuracy = total_train_accuracy / number_of_data;
        average_train_loss = total_train_loss / number_of_data;
        logging.info('train: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
            self.epoch_index, self.minibatch_index, epoch_running_time, average_train_loss,
            average_train_accuracy * 100))
        print 'train: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
            self.epoch_index, self.minibatch_index, epoch_running_time, average_train_loss,
            average_train_accuracy * 100)

        self.epoch_index += 1;

        return epoch_running_time;

class StandoutNeuralNetworkTypeB(DiscriminativeNetwork):
    def __init__(self,
                 incoming,

                 layer_dimensions,
                 layer_nonlinearities,

                 input_activation_rate=1.0,
                 # layer_activation_parameters=None,
                 # layer_activation_styles=None,
                 # pretrained_model=None,

                 objective_functions=objectives.categorical_crossentropy,
                 update_function=updates.nesterov_momentum,
                 learning_rate=1e-3,
                 learning_rate_decay_style=None,
                 learning_rate_decay_parameter=0,
                 validation_interval=-1,
                 ):
        super(StandoutNeuralNetworkTypeB, self).__init__(incoming,
                                                         objective_functions,
                                                         update_function,
                                                         learning_rate,
                                                         learning_rate_decay_style,
                                                         learning_rate_decay_parameter,
                                                         validation_interval)

        self._output_variable = theano.tensor.ivector()  # the labels are presented as 1D vector of [int] labels

        assert len(layer_dimensions) == len(layer_nonlinearities)
        #assert len(layer_dimensions) == len(layer_activation_parameters)
        #assert len(layer_dimensions) == len(layer_activation_styles)

        neural_network = self._input_layer;

        for layer_index in xrange(len(layer_dimensions)):
            previous_layer_dimension = layers.get_output_shape(neural_network)[1:];
            # activation_probability = noise.sample_activation_probability(previous_layer_dimension, layer_activation_styles[layer_index], layer_activation_parameters[layer_index]);
            # activation_probability = sample_activation_probability(previous_layer_dimension, layer_activation_styles[layer_index], layer_activation_parameters[layer_index]);

            if layer_index == 0:
                # activation_probability = sample_activation_probability(previous_layer_dimension, layer_activation_styles[0], layer_activation_parameters[0]);
                # neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);
                neural_network = layers.LinearDropoutLayer(neural_network, activation_probability=input_activation_rate);

            layer_dimension = layer_dimensions[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index];

            dense_layer = layers.DenseLayer(neural_network,
                                            layer_dimension,
                                            W=init.GlorotUniform(gain=init.GlorotUniformGain[layer_nonlinearity]),
                                            nonlinearity=layer_nonlinearity)

            if layer_index < len(layer_dimensions) - 1:
                dropout_layer = layers.StandoutLayer(neural_network, layer_dimension, W=dense_layer.W, b=dense_layer.b);
                neural_network = layers.ElemwiseMergeLayer([dense_layer, dropout_layer], theano.tensor.mul);
            else:
                neural_network = dense_layer;

        self._neural_network = neural_network;

        self.build_functions();

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

    network = StandoutNeuralNetworkTypeB(
        incoming=input_shape,

        layer_dimensions=[32, 64, 10],
        layer_nonlinearities=[nonlinearities.rectify, nonlinearities.rectify, nonlinearities.softmax],
        input_activation_rate=0.8,

        objective_functions=objectives.categorical_crossentropy,
        update_function=updates.nesterov_momentum,

        learning_rate = 0.001,
        learning_rate_decay_style=None,
        learning_rate_decay_parameter=0,
        validation_interval=1000,
    )

    #print layers.get_all_params(network)
    print network.get_network_params(trainable=True);
    print network.get_network_params(trainable=True, adaptable=True);
    print network.get_network_params(trainable=True, adaptable=False);
    print network.get_network_params(regularizable=False);
    print network.get_network_params(regularizable=False, adaptable=True);
    print network.get_network_params(regularizable=False, adaptable=False);
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
        print "PROGRESS: %f%%" % (100. * epoch_index / number_of_epochs);
    end_train = timeit.default_timer()

    print "Optimization complete..."
    logging.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
        network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index));
    print 'The code finishes in %.2fm' % ((end_train - start_train) / 60.)

if __name__ == '__main__':
    main();