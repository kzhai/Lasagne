import logging
import os
import timeit

import numpy
import theano

from . import FeedForwardNetwork, AdaptiveFeedForwardNetwork, DynamicFeedForwardNetwork
from . import adjust_parameter_according_to_policy
from .. import init, nonlinearities, objectives, Xpolicy, updates
from .. import layers

logger = logging.getLogger(__name__)

__all__ = [
    "AdaptiveMultiLayerPerceptron",
    #
    #
    #
    "DynamicMultiLayerPerceptron",
    #
    #
    #
    "DynamicMultiLayerPerceptronSurgery",
]


class AdaptiveMultiLayerPerceptron(AdaptiveFeedForwardNetwork):
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

                 adaptable_learning_rate_policy=[1e-3, Xpolicy.constant],
                 # dropout_learning_rate_decay=None,
                 adaptable_update_interval=1,
                 # update_hidden_layer_dropout_only=False,

                 max_norm_constraint=0,
                 # learning_rate_decay_style=None,
                 # learning_rate_decay_parameter=0,
                 validation_interval=-1,
                 ):
        super(AdaptiveMultiLayerPerceptron, self).__init__(incoming=incoming,

                                                           objective_functions=objective_functions,
                                                           update_function=update_function,
                                                           learning_rate_policy=learning_rate_policy,
                                                           # learning_rate_decay,

                                                           adaptable_learning_rate_policy=adaptable_learning_rate_policy,
                                                           # dropout_learning_rate_decay,
                                                           adaptable_update_interval=adaptable_update_interval,

                                                           max_norm_constraint=max_norm_constraint,
                                                           validation_interval=validation_interval,
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


class DynamicMultiLayerPerceptron(DynamicFeedForwardNetwork):
    def __init__(self,
                 incoming,

                 dense_dimensions,
                 dense_nonlinearities,

                 layer_activation_types,
                 layer_activation_parameters=None,
                 layer_activation_styles=None,

                 objective_functions=objectives.categorical_crossentropy,
                 update_function=updates.nesterov_momentum,
                 learning_rate_policy=[1e-3, Xpolicy.constant],

                 adaptable_learning_rate_policy=[1e-3, Xpolicy.constant],
                 adaptable_update_interval=1,

                 prune_threshold_policies=None,
                 split_threshold_policies=None,
                 prune_split_interval=[0, 0],

                 max_norm_constraint=0,
                 validation_interval=-1,
                 ):
        super(DynamicMultiLayerPerceptron, self).__init__(incoming=incoming,

                                                          objective_functions=objective_functions,
                                                          update_function=update_function,
                                                          learning_rate_policy=learning_rate_policy,

                                                          adaptable_learning_rate_policy=adaptable_learning_rate_policy,
                                                          adaptable_update_interval=adaptable_update_interval,

                                                          prune_threshold_policies=prune_threshold_policies,
                                                          split_threshold_policies=split_threshold_policies,
                                                          prune_split_interval=prune_split_interval,

                                                          max_norm_constraint=max_norm_constraint,
                                                          validation_interval=validation_interval,
                                                          )
        assert len(dense_dimensions) == len(dense_nonlinearities)
        assert len(dense_dimensions) == len(layer_activation_parameters)
        assert len(dense_dimensions) == len(layer_activation_styles)

        number_of_dynamic_dropout_layers = sum(
            layer_activation_type is layers.DynamicDropoutLayer for layer_activation_type in layer_activation_types)
        assert (self.prune_threshold_policies is None) or len(
            self.prune_threshold_policies) == number_of_dynamic_dropout_layers
        assert (self.split_threshold_policies is None) or len(
            self.split_threshold_policies) == number_of_dynamic_dropout_layers

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

            neural_network = layers.DynamicDenseLayerGuo(neural_network, layer_dimension,
                                                         W=init.GlorotUniform(
                                                          gain=init.GlorotUniformGain[
                                                              layer_nonlinearity]),
                                                         nonlinearity=layer_nonlinearity)

        self._neural_network = neural_network

        self.build_functions()

    def adjust_network(self, train_dataset=None, validate_dataset=None, test_dataset=None):
        if self.epoch_index < self.prune_split_interval[0] \
                or self.prune_split_interval[1] <= 0 \
                or self.epoch_index % self.prune_split_interval[1] != 0:
            return

        if self.prune_threshold_policies is not None:
            prune_thresholds = [adjust_parameter_according_to_policy(prune_threshold_policy, self.epoch_index) for
                                prune_threshold_policy in self.prune_threshold_policies]
            self.prune_network(prune_thresholds=prune_thresholds, validate_dataset=validate_dataset,
                               test_dataset=test_dataset)
        if self.split_threshold_policies is not None:
            split_thresholds = [adjust_parameter_according_to_policy(split_threshold_policy, self.epoch_index) for
                                split_threshold_policy in self.split_threshold_policies]
            self.split_network(split_thresholds=split_thresholds, validate_dataset=validate_dataset,
                               test_dataset=test_dataset)

    def prune_network(self, prune_thresholds, validate_dataset=None, test_dataset=None, output_directory=None):
        architecture_changed = False

        dropout_layer_index = 0
        for pre_dropout_layer, dropout_layer, post_dropout_layer in zip(self.get_network_layers()[:-2],
                                                                        self.get_network_layers()[1:-1],
                                                                        self.get_network_layers()[2:]):

            if (not isinstance(pre_dropout_layer, layers.DynamicDenseLayerGuo)) or \
                    (not isinstance(dropout_layer, layers.DynamicDropoutLayer)) or \
                    (not isinstance(post_dropout_layer, layers.DynamicDenseLayerGuo)):
                continue

            # print("layer %s size %d" % (pre_dropout_layer, pre_dropout_layer.num_units))
            # print("layer %s size %s" % (dropout_layer, dropout_layer.input_shape))
            # print("layer %s size %d" % (post_dropout_layer, post_dropout_layer.num_units))

            prune_threshold = prune_thresholds[dropout_layer_index]
            dropout_layer_index += 1
            neuron_indices_to_prune, neuron_indices_to_keep = dropout_layer.find_neuron_indices_to_prune(
                prune_threshold)

            if len(neuron_indices_to_prune) == 0:
                continue

            architecture_changed = True
            old_size = len(neuron_indices_to_prune) + len(neuron_indices_to_keep)
            new_size = len(neuron_indices_to_keep)

            pre_dropout_layer.prune_output(neuron_indices_to_keep)
            dropout_layer.prune_activation_probability(neuron_indices_to_keep)
            post_dropout_layer.prune_input(neuron_indices_to_keep)

            print("Prune layer %s from %d to %d with threshold %g" % (pre_dropout_layer, old_size, new_size,
                                                                      prune_threshold))
            logger.info("Prune layer %s from %d to %d with threshold %g" % (pre_dropout_layer, old_size, new_size,
                                                                            prune_threshold))

        if not architecture_changed:
            return

        self.build_functions()

        # print("Performance on validate and test set after pruning...")
        # logger.info("Performance on validate and test set after pruning...")

        if validate_dataset is not None:
            output_file = None
            if output_directory is not None:
                output_file = os.path.join(output_directory, 'model.pkl')
            self.validate(validate_dataset, test_dataset, output_file)
        elif test_dataset is not None:
            self.test(test_dataset)

    def split_network(self, split_thresholds, validate_dataset=None, test_dataset=None, output_directory=None):
        architecture_changed = False

        dropout_layer_index = 0
        for pre_dropout_layer, dropout_layer, post_dropout_layer in zip(self.get_network_layers()[:-2],
                                                                        self.get_network_layers()[1:-1],
                                                                        self.get_network_layers()[2:]):

            if (not isinstance(pre_dropout_layer, layers.DynamicDenseLayerGuo)) or \
                    (not isinstance(dropout_layer, layers.DynamicDropoutLayer)) or \
                    (not isinstance(post_dropout_layer, layers.DynamicDenseLayerGuo)):
                continue

            # print("layer %s size %d" % (pre_dropout_layer, pre_dropout_layer.num_units))
            # print("layer %s size %s" % (dropout_layer, dropout_layer.input_shape))
            # print("layer %s size %d" % (post_dropout_layer, post_dropout_layer.num_units))

            split_threshold = split_thresholds[dropout_layer_index]
            dropout_layer_index += 1
            neuron_indices_to_split, neuron_indices_to_keep = dropout_layer.find_neuron_indices_to_split(
                split_threshold)

            if len(neuron_indices_to_split) == 0:
                continue

            architecture_changed = True
            old_size = len(neuron_indices_to_split) + len(neuron_indices_to_keep)
            new_size = 2 * len(neuron_indices_to_split) + len(neuron_indices_to_keep)

            pre_dropout_layer.split_output(neuron_indices_to_split)
            dropout_layer.split_activation_probability(neuron_indices_to_split)
            post_dropout_layer.split_input(neuron_indices_to_split)

            print("Split layer %s from %d to %d with threshold %g" % (pre_dropout_layer, old_size, new_size,
                                                                      split_threshold))
            logger.info("Split layer %s from %d to %d with threshold %g" % (pre_dropout_layer, old_size, new_size,
                                                                            split_threshold))

        if not architecture_changed:
            return

        self.build_functions()

        # print("Performance on validate and test set after pruning...")
        # logger.info("Performance on validate and test set after pruning...")

        if validate_dataset is not None:
            output_file = None
            if output_directory is not None:
                output_file = os.path.join(output_directory, 'model.pkl')
            self.validate(validate_dataset, test_dataset, output_file)
        elif test_dataset is not None:
            self.test(test_dataset)

        return


#
#
#
#
#

class DynamicMultiLayerPerceptronSurgery(FeedForwardNetwork):
    def __init__(self,
                 incoming,

                 dense_dimensions,
                 dense_nonlinearities,

                 # layer_activation_types,
                 layer_activation_parameters,
                 layer_activation_styles,

                 objective_functions=objectives.categorical_crossentropy,
                 update_function=updates.nesterov_momentum,
                 learning_rate_policy=[1e-3, Xpolicy.constant],

                 #
                 #
                 #
                 #
                 #

                 prune_threshold_policies=None,
                 splice_threshold_policies=None,
                 prune_split_interval=[0, 0],

                 max_norm_constraint=0,
                 validation_interval=-1,
                 ):
        super(DynamicMultiLayerPerceptronSurgery, self).__init__(incoming,
                                                                 objective_functions=objective_functions,
                                                                 update_function=update_function,
                                                                 learning_rate_policy=learning_rate_policy,
                                                                 # prune_policy=prune_policy,
                                                                 max_norm_constraint=max_norm_constraint,
                                                                 validation_interval=validation_interval,
                                                                 )

        self._prune_threshold_policies = prune_threshold_policies
        self._splice_threshold_policies = splice_threshold_policies
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

            neural_network = layers.WeightPruningDropoutLayer(neural_network,
                                                              activation_probability=activation_probability)

            layer_dimension = dense_dimensions[layer_index]
            layer_nonlinearity = dense_nonlinearities[layer_index]

            neural_network = layers.WeightPruningDenseLayer(neural_network, layer_dimension, W=init.GlorotUniform(
                gain=init.GlorotUniformGain[layer_nonlinearity]), nonlinearity=layer_nonlinearity)

        self._neural_network = neural_network

        self.build_functions()

    def train(self, train_dataset, minibatch_size, validate_dataset=None, test_dataset=None, output_directory=None):
        epoch_running_time = super(DynamicMultiLayerPerceptronSurgery, self).train(train_dataset, minibatch_size,
                                                                                   validate_dataset, test_dataset,
                                                                                   output_directory)

        # if self.epoch_index >= self._prune_policy[2] and self.epoch_index % self._prune_policy[1] == 0:
        epoch_running_time_temp = timeit.default_timer()
        self.adjust_network(train_dataset=train_dataset, validate_dataset=validate_dataset, test_dataset=test_dataset)
        epoch_running_time_temp = timeit.default_timer() - epoch_running_time_temp
        epoch_running_time += epoch_running_time_temp

        return epoch_running_time

    def adjust_network(self, train_dataset=None, validate_dataset=None, test_dataset=None):
        if self.epoch_index < self.prune_split_interval[0] \
                or self.prune_split_interval[1] <= 0 \
                or self.epoch_index % self.prune_split_interval[1] != 0:
            return

        if self._prune_threshold_policies is not None:
            prune_thresholds = [adjust_parameter_according_to_policy(prune_threshold_policy, self.epoch_index) for
                                prune_threshold_policy in self._prune_threshold_policies]
            self.prune_network(prune_thresholds=prune_thresholds, validate_dataset=validate_dataset,
                               test_dataset=test_dataset)
        if self._splice_threshold_policies is not None:
            splice_thresholds = [adjust_parameter_according_to_policy(splice_threshold_policy, self.epoch_index) for
                                 splice_threshold_policy in self._splice_threshold_policies]
            self.splice_network(splice_thresholds=splice_thresholds, validate_dataset=validate_dataset,
                                test_dataset=test_dataset)

        self.prune_synapses(train_dataset, validate_dataset, test_dataset)

    def prune_synapses(self, train_dataset=None, validate_dataset=None, test_dataset=None,
                       dropout_decay_style="elementwise"):
        connection_threshold = adjust_parameter_according_to_policy(self._prune_threshold_policies, self.epoch_index)

        layer_info_list = []
        for layer_0, layer_1 in zip(self.get_network_layers()[:-1], self.get_network_layers()[1:]):
            if (not isinstance(layer_0, layers.WeightPruningDropoutLayer)) or \
                    (not isinstance(layer_1, layers.WeightPruningDenseLayer)):
                continue

            # input_size = layers.get_output_shape(layer_0)[1:]
            # old_sizes.append(layer_1.num_units)
            '''
            if dropout_decay_style == "layerwise":
                C_old = numpy.sum(layer_1.mask)
            elif dropout_decay_style == "elementwise":
                C_old = numpy.sum(layer_1.mask, axis=1)
            '''
            C_old = numpy.sum(layer_1.mask, axis=1)
            neuron_indices_to_prune, neuron_indices_to_keep = layer_1.prune_weight(connection_threshold)
            # print "layer 0:", layer_0.input_shape, layer_0
            # print "layer 1:", layer_1.num_units, layer_1
            # print "neuron indices to prune:", len(neuron_indices_to_prune)  # , neuron_indices_to_prune
            # print "neuron indices to keep:", len(neuron_indices_to_keep)  # , neuron_indices_to_keep
            # print "=========="
            C_new = numpy.sum(layer_1.mask, axis=1)
            '''
            if dropout_decay_style == "layerwise":
                C_new = numpy.sum(layer_1.mask)
            elif dropout_decay_style == "elementwise":
                C_new = numpy.sum(layer_1.mask, axis=1)
            '''

            assert len(neuron_indices_to_keep) <= layer_0.input_shape, (
                len(neuron_indices_to_keep), neuron_indices_to_keep, layer_0.input_shape)
            layer_info_list.append(neuron_indices_to_keep)

            if numpy.all(C_old == C_new):
                continue
            print("Adjusting number of connections in layer %s from %d to %d" % (layer_1, numpy.sum(C_old),
                                                                                 numpy.sum(C_new)))
            logger.info("Adjusting number of connections in layer %s from %d to %d" % (layer_1, numpy.sum(C_old),
                                                                                       numpy.sum(C_new)))
            layer_0.decay_activation_probability(numpy.sqrt(C_new / C_old))

        layer_info_index = 0
        for layer_1, layer_2, layer_3 in zip(self.get_network_layers()[:-2], self.get_network_layers()[1:-1],
                                             self.get_network_layers()[2:]):
            if (not isinstance(layer_1, layers.WeightPruningDenseLayer)) or \
                    (not isinstance(layer_2, layers.WeightPruningDropoutLayer)) or \
                    (not isinstance(layer_3, layers.WeightPruningDenseLayer)):
                continue

            layer_info_index += 1
            neuron_indices_to_keep = layer_info_list[layer_info_index]

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

    network = DynamicMultiLayerPerceptronSurgery(
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
