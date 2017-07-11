from itertools import chain

import copy
import cPickle
import logging
import numpy
import os
import sys
import theano
import theano.tensor
import timeit
import types

from .. import layers
from .. import objectives, regularization, utils

__all__ = [
    "decay_learning_rate",
    "DiscriminativeNetwork",
    "GenerativeNetwork",
    "Network",
]

def decay_learning_rate(learning_rate, epoch_or_iteration_index, learning_rate_decay=None):
    #learning_rate = 1e-3, learning_rate_decay_style = None, learning_rate_decay_parameter = 0
    #learning_rate, learning_rate_decay_style, learning_rate_decay_parameter = learning_rate_configuration;
    if learning_rate_decay == None:
        current_learning_rate = learning_rate;
    elif learning_rate_decay[1] == "inverse_t":
        current_learning_rate = learning_rate * learning_rate_decay[2] / (1. + learning_rate_decay[3] * epoch_or_iteration_index)
    elif learning_rate_decay[1] == "exponential":
        current_learning_rate = learning_rate * learning_rate_decay[2] * numpy.exp(- learning_rate_decay[3] * epoch_or_iteration_index);
    elif learning_rate_decay[1] == "step":
        current_learning_rate = learning_rate * numpy.power(learning_rate_decay[2], epoch_or_iteration_index // learning_rate_decay[3]);
    else:
        current_learning_rate = learning_rate;

    return current_learning_rate.astype(theano.config.floatX)
    #return numpy.float32(current_learning_rate)

class Network(object):
    def __init__(self,
                 incoming,
                 objective_functions,
                 update_function,
                 learning_rate=1e-3,
                 learning_rate_decay=None,
                 #learning_rate_decay_style=None,
                 #learning_rate_decay_parameter=0,
                 ):
        if isinstance(incoming, tuple):
            self._input_shape = incoming
            self._input_layer = layers.InputLayer(shape=self._input_shape);
        else:
            self._input_shape = incoming.output_shape
            self._input_layer = incoming

        if any(d is not None and d <= 0 for d in self._input_shape):
            raise ValueError(("Cannot create Layer with a non-positive input_shape dimension. input_shape=%r") % (
            self._input_shape))

        input_layers = [layer for layer in layers.get_all_layers(self._input_layer) if isinstance(layer, layers.InputLayer)]
        assert len(input_layers)==1;
        #self._input_variable = self._input_layer.input_var;
        self._input_variable = input_layers[0].input_var;
        self._learning_rate_variable = theano.tensor.scalar();

        #
        #
        #
        #
        #

        self.epoch_index = 0;
        self.minibatch_index = 0;

        self.objective_functions_change_stack = [];
        self.regularizer_functions_change_stack = [];
        self.update_function_change_stack = [];
        self.learning_rate_decay_change_stack = [];
        self.learning_rate_change_stack = [];
        #self.learning_rate_decay_style_change_stack = []
        #self.learning_rate_decay_parameter_change_stack = []

        self.__set_objective_functions(objective_functions)
        self.__set_regularizer_functions();
        self.__set_update_function(update_function);
        self.set_learning_rate_decay(learning_rate_decay);
        self.set_learning_rate(learning_rate);
        #self.set_learning_rate_decay_style(learning_rate_decay_style);
        #self.set_learning_rate_decay_parameter(learning_rate_decay_parameter);

        # self.learning_change_stack = [];
        # (epoch_index, minibatch_index, learning_rate, learning_rate_decay_style, learning_rate_decay_parameter)
        # self.network_change_stack = [];
        # (epoch_index, minibatch_index, objective_function, update_function, regularizer_information)
        # self.data_change_stack = [];
        # (epoch_index, minibatch_index, train_dataset, validate_dataset, test_dataset)

        #self._layer_L1_regularizer_lambdas = None;
        #self._layer_L2_regularizer_lambdas = None;

    def get_input(self, **kwargs):
        return layers.get_output(self._input_layer, **kwargs);

    def get_output(self, inputs=None, **kwargs):
        return layers.get_output(self._neural_network, inputs, **kwargs)

    def get_output_shape(self, input_shapes=None):
        return layers.get_output_shape(self._neural_network, input_shapes);

    '''
    def get_all_layers(self, treat_as_input=None):
        return layers.get_all_layers(self._neural_network, treat_as_input);

    def get_all_params(self, **tags):
        return layers.get_all_params(self._neural_network, **tags);

    def count_all_params(self, **tags):
        return layers.count_params(self._neural_network, **tags);
    '''

    def get_network_output(self, inputs=None, **kwargs):
        return layers.get_output(self._neural_network, inputs, **kwargs)

    def get_network_output_shape(self, input_shapes=None):
        return layers.get_output_shape(self._neural_network, input_shapes);

    def get_network_layers(self):
        return layers.get_all_layers(self._neural_network, [self._input_layer]);

    def get_network_params(self, **tags):
        params = chain.from_iterable(l.get_params(**tags) for l in self.get_network_layers()[1:])
        return utils.unique(params)

    def count_network_params(self, **tags):
        # return lasagne.layers.count_params(self.get_all_layers()[1:], **tags);
        params = self.get_network_params(**tags)
        shapes = [p.get_value().shape for p in params]
        counts = [numpy.prod(shape) for shape in shapes]
        return sum(counts)

    def get_network_param_values(self, **tags):
        # return lasagne.layers.get_all_param_values(self.get_all_layers()[1:], **tags);
        params = self.get_network_params(**tags)
        return [p.get_value() for p in params]

    def set_input_variable(self, input):
        '''This is to establish the computational graph'''
        # self.get_all_layers()[0].input_var = input
        self._input_variable = input

    #
    #
    #
    #
    #

    @property
    def build_functions(self):
        raise NotImplementedError("Not implemented in successor classes!");

    @property
    def get_objectives(self, **kwargs):
        raise NotImplementedError("Not implemented in successor classes!");

    def get_regularizers(self, **kwargs):
        assert hasattr(self, "_neural_network");
        regularizer = 0;
        for regularizer_function, lambdas in self._regularizer_functions.iteritems():
            assert type(regularizer_function) is types.FunctionType;
            if regularizer_function==regularization.rademacher \
                    or regularizer_function == regularization.rademacher_p_2_q_2 \
                    or regularizer_function==regularization.rademacher_p_1_q_inf \
                    or regularizer_function == regularization.rademacher_p_inf_q_1:
                assert type(lambdas) is float;
                regularizer += lambdas * regularization.rademacher(self, **kwargs);
            elif regularizer_function in set([regularization.l1, regularization.l2, regularization.linf]):
                if type(lambdas) is list:
                    dense_layers = [];
                    for layer in self.get_network_layers():
                        if isinstance(layer, layers.dense.DenseLayer):
                            dense_layers.append(layer);
                    assert len(dense_layers)==len(lambdas), (dense_layers, lambdas);
                    regularizer += regularization.regularize_layer_params_weighted(dict(zip(dense_layers, lambdas)), regularizer_function, **kwargs);
                elif type(lambdas) is float:
                    regularizer += lambdas * regularization.regularize_network_params(self._neural_network, regularizer_function, **kwargs);
                else:
                    logging.error("unrecognized regularizer function settings: %s, %s" % (regularizer_function, lambdas));
            else:
                logging.error("unrecognized regularizer function: %s" % (regularizer_function));

        return regularizer

    @property
    def get_loss(self, **kwargs):
        raise NotImplementedError("Not implemented in successor classes!");

    #
    #
    #
    #
    #

    '''
    def L1_regularizer(self):
        if self._layer_L1_regularizer_lambdas == None:
            return 0;
        else:
            # We could add some weight decay as well here, see lasagne.regularization.
            return regularization.regularize_layer_params_weighted(self._layer_L1_regularizer_lambdas,
                                                                   regularization.l1)

    def set_L1_regularizer_lambda(self, L1_regularizer_lambdas=None):
        if L1_regularizer_lambdas == None or L1_regularizer_lambdas == 0 or all(
                        L1_regularizer_lambda == 0 for L1_regularizer_lambda in L1_regularizer_lambdas):
            self._layer_L1_regularizer_lambdas = None;
        else:
            assert len(L1_regularizer_lambdas) == len(self.get_network_layers()) - 1;
            self._layer_L1_regularizer_lambdas = {temp_layer: L1_regularizer_lambda for
                                                  temp_layer, L1_regularizer_lambda in
                                                  zip(self.get_network_layers(), L1_regularizer_lambdas)};

            # self.build_functions()

    def L2_regularizer(self):
        if self._layer_L2_regularizer_lambdas == None:
            return 0;
        else:
            # We could add some weight decay as well here, see lasagne.regularization.
            return regularization.regularize_layer_params_weighted(self._layer_L2_regularizer_lambdas,
                                                                   regularization.l2)

    def set_L2_regularizer_lambda(self, L2_regularizer_lambdas):
        if L2_regularizer_lambdas == None or L2_regularizer_lambdas == 0 or all(
                        L2_regularizer_lambda == 0 for L2_regularizer_lambda in L2_regularizer_lambdas):
            self._layer_L2_regularizer_lambdas = None;
        else:
            assert len(L2_regularizer_lambdas) == len(self.get_network_layers()) - 1;

            self._layer_L2_regularizer_lambdas = {temp_layer: L2_regularizer_lambda for
                                                  temp_layer, L2_regularizer_lambda in
                                                  zip(self.get_network_layers(), L2_regularizer_lambdas)};

            # self.build_functions()
    '''

    #
    #
    #
    #
    #

    def __set_objective_functions(self, objective_functions):
        assert objective_functions != None;
        if type(objective_functions) is types.FunctionType:
            self._objective_functions = {objective_functions: 1.0};
        elif type(objective_functions) is list:
            self._objective_functions = {objective_function: 1.0 for objective_function in objective_functions};
        else:
            logging.error('unrecognized objective functions: %s' % (objective_functions))
        self.objective_functions_change_stack.append((self.epoch_index, self._objective_functions));

    def set_objectives(self, objectives):
        self.__set_objective_functions(objectives);
        self.build_functions();

    def __set_regularizer_functions(self, regularizer_functions=None):
        _regularizer_functions = {};
        if regularizer_functions!=None:
            assert hasattr(self, "_neural_network");
            if type(regularizer_functions) is types.FunctionType:
                _regularizer_functions[regularizer_functions] = 1.0;
            elif type(regularizer_functions) is dict:
                for regularizer_function, lambdas in regularizer_functions.iteritems():
                    assert type(regularizer_function) is types.FunctionType;
                    if type(lambdas) is list:
                        for weight in lambdas:
                            #assert isinstance(layer, layers.Layer);
                            assert type(weight)==float;
                    else:
                        assert type(lambdas)==float;
                    _regularizer_functions[regularizer_function] = lambdas;
            else:
                logging.error('unrecognized regularizer functions: %s' % (regularizer_functions))
        self._regularizer_functions = _regularizer_functions;
        self.regularizer_functions_change_stack.append((self.epoch_index, self._regularizer_functions));

    def set_regularizers(self, regularizers=None):
        self.__set_regularizer_functions(regularizers);
        self.build_functions();

    def __set_update_function(self, update_function):
        assert update_function != None;
        self._update_function = update_function;
        self.update_function_change_stack.append((self.epoch_index, self._update_function));

    def set_update(self, update):
        self.__set_update_function(update);
        self.build_functions();

    def set_learning_rate_decay(self, learning_rate_decay):
        self.learning_rate_decay = learning_rate_decay;
        self.learning_rate_decay_change_stack.append((self.epoch_index, self.learning_rate_decay));

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate;
        self.learning_rate_change_stack.append((self.epoch_index, self.learning_rate));

    '''
    def set_learning_rate_decay_style(self, learning_rate_decay_style):
        self.learning_rate_decay_style = learning_rate_decay_style;
        self.learning_rate_decay_style_change_stack.append((self.epoch_index, self.learning_rate_decay_style));

    def set_learning_rate_decay_parameter(self, learning_rate_decay_parameter):
        self.learning_rate_decay_parameter = learning_rate_decay_parameter;
        self.learning_rate_decay_parameter_change_stack.append((self.epoch_index, self.learning_rate_decay_parameter));
    '''

    '''
    def build_functions(self):
        # Create a train_loss expression for training, i.e., a scalar objective we want to minimize (for our multi-class problem, it is the cross-entropy train_loss):
        train_loss = self.get_loss(self._output_variable);
        train_accuracy = self.get_objective(self._output_variable, objective_function="categorical_accuracy");
        # train_prediction = self.get_output(**kwargs)
        # train_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(train_prediction, axis=1), self._output_variable), dtype=theano.config.floatX)

        # Create update expressions for training, i.e., how to modify the parameters at each training step. Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        all_params = self.get_network_params(trainable=True)
        all_params_updates = self._update_function(train_loss, all_params, self._learning_rate_variable)

        # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) and returning the corresponding training train_loss:
        self._train_function = theano.function(
            inputs=[self._input_variable, self._output_variable, self._learning_rate_variable],
            outputs=[train_loss, train_accuracy],
            updates=all_params_updates
        )

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
    '''

    '''
    def test(self, test_dataset):
        test_dataset_x, test_dataset_y = test_dataset
        test_running_time = timeit.default_timer();
        average_test_loss, average_test_accuracy = self._test_function(test_dataset_x, test_dataset_y);
        test_running_time = timeit.default_timer() - test_running_time;
        return average_test_loss, average_test_accuracy, test_running_time;

    def train(self, train_dataset, learning_rate):
        train_dataset_x, train_dataset_y = train_dataset
        train_running_time = timeit.default_timer();
        average_train_loss, average_train_accuracy = self._train_function(train_dataset_x, train_dataset_y, learning_rate)
        train_running_time = timeit.default_timer() - train_running_time;
        return average_train_loss, average_train_accuracy, train_running_time;
    '''

class DiscriminativeNetwork(Network):
    def __init__(self, incoming,
                 objective_functions,
                 update_function,
                 learning_rate=1e-3,
                 learning_rate_decay=None,
                 #learning_rate_decay_style=None,
                 #learning_rate_decay_parameter=0,
                 validation_interval=-1,
                 ):

        super(DiscriminativeNetwork, self).__init__(incoming,
                                                    objective_functions,
                                                    update_function,
                                                    learning_rate,
                                                    learning_rate_decay,
                                                    #learning_rate_decay_style,
                                                    #learning_rate_decay_parameter
                                                    );

        self.validation_interval = validation_interval;
        self.best_epoch_index = 0;
        self.best_minibatch_index = 0;
        self.best_validate_accuracy = 0;
        # self.best_validate_model = None;

    '''
    @property
    def build_functions(self):
        raise NotImplementedError("Not implemented in successor classes!");
    '''

    def get_objectives(self, label, objective_functions=None, **kwargs):
        output = self.get_output(**kwargs);
        if objective_functions == None:
            #objective = theano.tensor.mean(self._objective_functions(output, label), dtype=theano.config.floatX);
            objective = 0;
            for objective_function, weight in self._objective_functions.items():
                objective += weight * theano.tensor.mean(objective_function(output, label), dtype=theano.config.floatX);
        else:
            #TODO: expand to multiple objective functions
            temp_objective_function = getattr(objectives, objective_functions);
            objective = theano.tensor.mean(temp_objective_function(output, label), dtype=theano.config.floatX);
        return objective

    def get_loss(self, label, **kwargs):
        loss = self.get_objectives(label, **kwargs) + self.get_regularizers();
        return loss;

    def build_functions(self):
        # Create a train_loss expression for training, i.e., a scalar objective we want to minimize (for our multi-class problem, it is the cross-entropy train_loss):
        train_loss = self.get_loss(self._output_variable);
        train_accuracy = self.get_objectives(self._output_variable, objective_functions="categorical_accuracy");
        # train_prediction = self.get_output(**kwargs)
        # train_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(train_prediction, axis=1), self._output_variable), dtype=theano.config.floatX)

        # Create update expressions for training, i.e., how to modify the parameters at each training step. Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        trainable_params = self.get_network_params(trainable=True)
        trainable_params_updates = self._update_function(train_loss, trainable_params, self._learning_rate_variable, momentum=0.95)

        # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) and returning the corresponding training train_loss:
        self._train_function = theano.function(
            inputs=[self._input_variable, self._output_variable, self._learning_rate_variable],
            outputs=[train_loss, train_accuracy],
            updates=trainable_params_updates
        )

        # Create a train_loss expression for validation/testing. The crucial difference here is that we do a deterministic forward pass through the networks, disabling dropout layers.
        test_loss = self.get_loss(self._output_variable, deterministic=True);
        test_accuracy = self.get_objectives(self._output_variable, objective_functions="categorical_accuracy",
                                            deterministic=True);
        # As a bonus, also create an expression for the classification accuracy:
        # test_prediction = self.get_output(deterministic=True)
        # test_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(test_prediction, axis=1), self._output_variable), dtype=theano.config.floatX)

        # Compile a second function computing the validation train_loss and accuracy:
        self._test_function = theano.function(
            inputs=[self._input_variable, self._output_variable],
            outputs=[test_loss, test_accuracy],
        )

        '''
        debug_loss = self.debug_loss(self._output_variable);
        self._debug_function = theano.function(
            inputs=[self._input_variable, self._output_variable, self._learning_rate_variable],
            outputs=debug_loss,
            on_unused_input='ignore'
        )
        '''

    '''
    def test(self, test_dataset):
        test_dataset_x, test_dataset_y = test_dataset
        test_running_time = timeit.default_timer();
        average_test_loss, average_test_accuracy = self._test_function(test_dataset_x, test_dataset_y);
        test_running_time = timeit.default_timer() - test_running_time;
        return average_test_loss, average_test_accuracy, test_running_time;

    def train(self, train_dataset, learning_rate):
        train_dataset_x, train_dataset_y = train_dataset
        train_running_time = timeit.default_timer();
        average_train_loss, average_train_accuracy = self._train_function(train_dataset_x, train_dataset_y, learning_rate)
        train_running_time = timeit.default_timer() - train_running_time;
        return average_train_loss, average_train_accuracy, train_running_time;
    '''

    def test(self, test_dataset):
        test_dataset_x, test_dataset_y = test_dataset
        test_running_time = timeit.default_timer();
        test_function_outputs = self._test_function(test_dataset_x, test_dataset_y);
        average_test_loss = test_function_outputs[0]
        average_test_accuracy = test_function_outputs[1];
        test_running_time = timeit.default_timer() - test_running_time;
        # average_test_loss, average_test_accuracy, test_running_time = self.network.test(test_dataset);
        logging.info('\t\ttest: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
            self.epoch_index, self.minibatch_index, test_running_time, average_test_loss, average_test_accuracy * 100))
        print('\t\ttest: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
            self.epoch_index, self.minibatch_index, test_running_time, average_test_loss, average_test_accuracy * 100))

    def validate(self, validate_dataset, test_dataset=None, best_model_file_path=None):
        # average_validate_loss, average_validate_accuracy, validate_running_time = self.test(validate_dataset);
        validate_running_time = timeit.default_timer();
        validate_dataset_x, validate_dataset_y = validate_dataset
        validate_function_outputs = self._test_function(validate_dataset_x, validate_dataset_y);
        average_validate_loss = validate_function_outputs[0];
        average_validate_accuracy = validate_function_outputs[1];
        validate_running_time = timeit.default_timer() - validate_running_time;
        logging.info('\tvalidate: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
            self.epoch_index, self.minibatch_index, validate_running_time, average_validate_loss,
            average_validate_accuracy * 100))
        print('\tvalidate: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
            self.epoch_index, self.minibatch_index, validate_running_time, average_validate_loss,
            average_validate_accuracy * 100))

        # if we got the best validation score until now
        if average_validate_accuracy > self.best_validate_accuracy:
            self.best_epoch_index = self.epoch_index
            self.best_minibatch_index = self.minibatch_index
            self.best_validate_accuracy = average_validate_accuracy
            # self.best_validate_model = copy.deepcopy(self)

            if best_model_file_path != None:
                # save the best model
                #cPickle.dump(self, open(best_model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);
                logging.info('\tbest model found: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (
                    self.epoch_index, self.minibatch_index, average_validate_loss, average_validate_accuracy * 100))

        if test_dataset != None:
            self.test(test_dataset);
            '''
            test_running_time = timeit.default_timer();
            test_dataset_x, test_dataset_y = test_dataset
            average_test_loss, average_test_accuracy = self._test_function(test_dataset_x, test_dataset_y);
            test_running_time = timeit.default_timer() - test_running_time;
            logging.info('\t\ttest: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
                self.epoch_index, self.minibatch_index, test_running_time, average_test_loss,
                average_test_accuracy * 100))
            '''

    def train(self, train_dataset, minibatch_size, validate_dataset=None, test_dataset=None, output_directory=None):
        # In each epoch_index, we do a full pass over the training data:
        epoch_running_time = 0;

        train_dataset_x, train_dataset_y = train_dataset

        number_of_data = train_dataset_x.shape[0];
        data_indices = numpy.random.permutation(number_of_data);
        minibatch_start_index = 0;
        if self.learning_rate_decay[0]=="epoch":
            learning_rate = decay_learning_rate(self.learning_rate, self.epoch_index, self.learning_rate_decay);

        total_train_loss = 0;
        total_train_accuracy = 0;
        while minibatch_start_index < number_of_data:
            # automatically handles the left-over data
            minibatch_indices = data_indices[minibatch_start_index:minibatch_start_index + minibatch_size];
            minibatch_start_index += minibatch_size;

            minibatch_x = train_dataset_x[minibatch_indices, :]
            minibatch_y = train_dataset_y[minibatch_indices]

            if self.learning_rate_decay[0] == "iteration":
                learning_rate = decay_learning_rate(self.learning_rate, self.minibatch_index, self.learning_rate_decay);

            minibatch_running_time, minibatch_average_train_loss, minibatch_average_train_accuracy = self.train_minibatch(minibatch_x, minibatch_y, learning_rate);

            '''
            minibatch_running_time = timeit.default_timer();
            #print self._debug_function(minibatch_x, minibatch_y, learning_rate);

            train_function_outputs = self._train_function(minibatch_x, minibatch_y, learning_rate)
            minibatch_average_train_loss = train_function_outputs[0];
            minibatch_average_train_accuracy = train_function_outputs[1];
            minibatch_running_time = timeit.default_timer() - minibatch_running_time;
            '''

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
            #if output_directory != None:
                #output_file = os.path.join(output_directory, 'model-%d.pkl' % self.epoch_index)
                #cPickle.dump(self, open(output_file, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);
            self.test(test_dataset);

        average_train_accuracy = total_train_accuracy / number_of_data;
        average_train_loss = total_train_loss / number_of_data;
        logging.info('train: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
            self.epoch_index, self.minibatch_index, epoch_running_time, average_train_loss,
            average_train_accuracy * 100))
        print 'train: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
            self.epoch_index, self.minibatch_index, epoch_running_time, average_train_loss,
            average_train_accuracy * 100)

        #print 'train: epoch %i, minibatch %i, learning-rate %g' % (self.epoch_index, self.minibatch_index, learning_rate)

        self.epoch_index += 1;

        return epoch_running_time;

    def train_minibatch(self, minibatch_x, minibatch_y, learning_rate):
        minibatch_running_time = timeit.default_timer();
        train_function_outputs = self._train_function(minibatch_x, minibatch_y, learning_rate)
        minibatch_average_train_loss = train_function_outputs[0];
        minibatch_average_train_accuracy = train_function_outputs[1];
        minibatch_running_time = timeit.default_timer() - minibatch_running_time;

        #print self._debug_function(minibatch_x, minibatch_y, learning_rate);

        return minibatch_running_time, minibatch_average_train_loss, minibatch_average_train_accuracy

#
#
#
#
#

class GenerativeNetwork(Network):
    def __init__(self, incoming,
                 objective_functions,
                 update_function,
                 learning_rate=1e-3,
                 learning_rate_decay_style=None,
                 learning_rate_decay_parameter=0,
                 ):
        super(GenerativeNetwork, self).__init__(incoming,
                                                objective_functions,
                                                update_function,
                                                learning_rate,
                                                learning_rate_decay_style,
                                                learning_rate_decay_parameter,
                                                );

    '''
    @property
    def build_functions(self):
        raise NotImplementedError("Not implemented in successor classes!");
    '''

    def get_objectives(self, objective_function=objectives.squared_error, **kwargs):
        objective = theano.tensor.mean(
            theano.tensor.sum(objective_function(self.get_output(**kwargs), self.get_input()), axis=1),
            dtype=theano.config.floatX);
        return objective

    def get_loss(self, **kwargs):
        loss = self.get_objectives(**kwargs)

        #loss += self.L1_regularizer();
        #loss += self.L2_regularizer();

        return loss;

    def build_functions(self):
        # Create a train_loss expression for training, i.e., a scalar objective we want to minimize (for our multi-class problem, it is the cross-entropy train_loss):
        train_loss = self.get_loss();
        #train_accuracy = self.get_objective(self._output_variable, objective_function="categorical_accuracy");
        # train_prediction = self.get_output(**kwargs)
        # train_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(train_prediction, axis=1), self._output_variable), dtype=theano.config.floatX)

        # Create update expressions for training, i.e., how to modify the parameters at each training step. Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        all_params = self.get_network_params(trainable=True)
        all_params_updates = self._update_function(train_loss, all_params, self._learning_rate_variable, momentum=0.95)

        # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) and returning the corresponding training train_loss:
        self._train_function = theano.function(
            inputs=[self._input_variable, self._learning_rate_variable],
            outputs=[train_loss],
            updates=all_params_updates
        )

    '''
    def test(self, test_dataset):
        test_dataset_x, test_dataset_y = test_dataset
        test_running_time = timeit.default_timer();
        average_test_loss, average_test_accuracy = self._test_function(test_dataset_x, test_dataset_y);
        test_running_time = timeit.default_timer() - test_running_time;
        return average_test_loss, average_test_accuracy, test_running_time;

    def train(self, train_dataset, learning_rate):
        train_dataset_x, train_dataset_y = train_dataset
        train_running_time = timeit.default_timer();
        average_train_loss, average_train_accuracy = self._train_function(train_dataset_x, train_dataset_y, learning_rate)
        train_running_time = timeit.default_timer() - train_running_time;
        return average_train_loss, average_train_accuracy, train_running_time;
    '''

    '''
    def test(self, test_dataset):
        test_dataset_x, test_dataset_y = test_dataset
        test_running_time = timeit.default_timer();
        average_test_loss, average_test_accuracy = self._test_function(test_dataset_x, test_dataset_y);
        test_running_time = timeit.default_timer() - test_running_time;
        # average_test_loss, average_test_accuracy, test_running_time = self.network.test(test_dataset);
        logging.info('\t\ttest: epoch %i, minibatch %i, duration %f, loss %f, accuracy %f%%' % (
            self.epoch_index, self.minibatch_index, test_running_time, average_test_loss, average_test_accuracy * 100))
    '''

    '''
    def validate(self, validate_dataset, test_dataset=None, best_model_file_path=None):
        # average_validate_loss, average_validate_accuracy, validate_running_time = self.test(validate_dataset);
        validate_running_time = timeit.default_timer();
        validate_dataset_x, validate_dataset_y = validate_dataset
        average_validate_loss, average_validate_accuracy = self._test_function(validate_dataset_x, validate_dataset_y);
        validate_running_time = timeit.default_timer() - validate_running_time;
        logging.info('\tvalidate: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
            self.epoch_index, self.minibatch_index, validate_running_time, average_validate_loss,
            average_validate_accuracy * 100))

        # if we got the best validation score until now
        if average_validate_accuracy > self.best_validate_accuracy:
            self.best_epoch_index = self.epoch_index
            self.best_minibatch_index = self.minibatch_index
            self.best_validate_accuracy = average_validate_accuracy
            # self.best_validate_model = copy.deepcopy(self)

            if best_model_file_path != None:
                # save the best model
                cPickle.dump(self, open(best_model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);
                logging.info('\tbest model found: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (
                    self.epoch_index, self.minibatch_index, average_validate_loss, average_validate_accuracy * 100))

        if test_dataset != None:
            test_running_time = timeit.default_timer();
            test_dataset_x, test_dataset_y = test_dataset
            average_test_loss, average_test_accuracy = self._test_function(test_dataset_x, test_dataset_y);
            test_running_time = timeit.default_timer() - test_running_time;
            logging.info('\t\ttest: epoch %i, minibatch %i, duration %fs, loss %f, accuracy %f%%' % (
                self.epoch_index, self.minibatch_index, test_running_time, average_test_loss,
                average_test_accuracy * 100))
    '''

    def train(self, train_dataset, minibatch_size, output_directory=None):
        # In each epoch_index, we do a full pass over the training data:
        epoch_running_time = 0;

        number_of_data = train_dataset.shape[0];
        data_indices = numpy.random.permutation(number_of_data);
        minibatch_start_index = 0;

        total_train_loss = 0;
        #total_train_accuracy = 0;
        while minibatch_start_index < number_of_data:
            # automatically handles the left-over data
            minibatch_indices = data_indices[minibatch_start_index:minibatch_start_index + minibatch_size];
            minibatch_start_index += minibatch_size;

            minibatch_x = train_dataset[minibatch_indices, :]
            #minibatch_y = train_dataset_y[minibatch_indices]

            learning_rate = decay_learning_rate(self.minibatch_index, self.learning_rate,
                                                self.learning_rate_decay_style, self.learning_rate_decay_parameter);

            minibatch_running_time = timeit.default_timer();
            minibatch_average_train_loss, = self._train_function(minibatch_x, learning_rate)
            minibatch_running_time = timeit.default_timer() - minibatch_running_time;
            epoch_running_time += minibatch_running_time

            current_minibatch_size = len(data_indices[minibatch_start_index:minibatch_start_index + minibatch_size])
            total_train_loss += minibatch_average_train_loss * current_minibatch_size;
            #total_train_accuracy += minibatch_average_train_accuracy * current_minibatch_size;

            # average_train_accuracy = total_train_accuracy / number_of_data;
            # average_train_loss = total_train_loss / number_of_data;
            # logging.debug('train: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (
            # self.epoch_index, self.minibatch_index, average_train_loss, average_train_accuracy * 100))

            self.minibatch_index += 1;

        #average_train_accuracy = total_train_accuracy / number_of_data;
        average_train_loss = total_train_loss / number_of_data;
        logging.info('train: epoch %i, minibatch %i, duration %fs, loss %f' % (
            self.epoch_index, self.minibatch_index, epoch_running_time, average_train_loss))

        #print 'train: epoch %i, minibatch %i, duration %fs, loss %f' % (
            #self.epoch_index, self.minibatch_index, epoch_running_time, average_train_loss)

        self.epoch_index += 1;

        return epoch_running_time;
