import cPickle
import logging
import numpy
import os
import sys
import timeit

from .. import networks
from .. import layers
from .base import load_data, load_and_split_data

__all__ = [
    "train_dlenet",
]

def construct_dlenet_parser():
    from .lenet import construct_lenet_parser
    model_parser = construct_lenet_parser();

    model_parser.description = "convolutional dynamic le net argument";

    # model argument set
    model_parser.add_argument("--dropout_rate_update_interval", dest="dropout_rate_update_interval", type=int, action='store', default=0,
                              help="dropout rate update interval [0=no update]");
    model_parser.add_argument('--update_hidden_layer_dropout_only', dest="update_hidden_layer_dropout_only", action='store_true', default=False,
                              help="update hidden layer dropout only [False]")

    return model_parser;

def validate_dlenet_arguments(arguments):
    from .lenet import validate_lenet_arguments;
    arguments = validate_lenet_arguments(arguments);

    # model argument set
    assert (arguments.dropout_rate_update_interval >= 0);

    return arguments

def train_dlenet():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron
    This is demonstrated on MNIST.
    """

    arguments, additionals = construct_dlenet_parser().parse_known_args()
    settings = validate_dlenet_arguments(arguments);

    input_directory = settings.input_directory
    output_directory = settings.output_directory
    assert not os.path.exists(output_directory)
    os.mkdir(output_directory);

    logging.basicConfig(filename=os.path.join(output_directory, "model.log"), level=logging.DEBUG, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s');
    validation_data = settings.validation_data
    minibatch_size = settings.minibatch_size

    print "========== ==========", "parameters", "========== =========="
    for key, value in vars(settings).iteritems():
        print "%s=%s" % (key, value);
    print "========== ==========", "additional", "========== =========="
    for addition in additionals:
        print "%s" % (addition);

    logging.info("========== ==========" + "parameters" + "========== ==========")
    for key, value in vars(settings).iteritems():
        logging.info("%s=%s" % (key, value));
    logging.info("========== ==========" + "parameters" + "========== ==========")

    cPickle.dump(settings, open(os.path.join(output_directory, "settings.pkl"), 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

    #
    #
    #
    #
    #

    test_dataset = load_data(input_directory, dataset="test")

    if validation_data>=0:
        train_dataset_info, validate_dataset_info = load_and_split_data(input_directory, validation_data);
        train_dataset, train_indices = train_dataset_info;
        validate_dataset, validate_indices =validate_dataset_info;
        numpy.save(os.path.join(output_directory, "train.index.npy"), train_indices);
        numpy.save(os.path.join(output_directory, "validate.index.npy"), validate_indices);
    else:
        train_dataset = load_data(input_directory, dataset="train");
        validate_dataset = load_data(input_directory, dataset="validate");
    train_set_x, train_set_y = train_dataset;
    input_shape = list(train_set_x.shape[1:]);
    input_shape.insert(0, None)
    input_shape = tuple(input_shape)

    #
    #
    #
    #
    #

    network = networks.DynamicLeNet(
        incoming=input_shape,

        convolution_filters=settings.convolution_filters,
        convolution_nonlinearities=settings.convolution_nonlinearities,
        # convolution_filter_sizes=None,
        # maxpooling_sizes=None,
        pool_modes=settings.pool_modes,

        #local_convolution_filters=settings.local_convolution_filters,

        dense_dimensions=settings.dense_dimensions,
        dense_nonlinearities=settings.dense_nonlinearities,

        layer_activation_parameters=settings.layer_activation_parameters,
        layer_activation_styles=settings.layer_activation_styles,

        objective_functions=settings.objective,
        update_function=settings.update,

        learning_rate = settings.learning_rate,
        learning_rate_decay=settings.learning_rate_decay,
        #learning_rate_decay_style=settings.learning_rate_decay_style,
        #learning_rate_decay_parameter=settings.learning_rate_decay_parameter,

        dropout_rate_update_interval=settings.dropout_rate_update_interval,
        update_hidden_layer_dropout_only=settings.update_hidden_layer_dropout_only,

        validation_interval=settings.validation_interval,
    )

    '''
    convolution_filter_sizes = settings.convolution_filter_sizes,
    convolution_strides = settings.convolution_strides,
    convolution_pads = settings.convolution_pads,

    local_convolution_filter_sizes = settings.local_convolution_filter_sizes,
    local_convolution_strides = settings.local_convolution_strides,
    local_convolution_pads = settings.local_convolution_pads,

    pooling_sizes = settings.pooling_sizes,
    pooling_strides = settings.pooling_strides,
    '''

    network.set_regularizers(settings.regularizer);

    ########################
    # START MODEL TRAINING #
    ########################

    start_train = timeit.default_timer()
    # Finally, launch the training loop.
    # We iterate over epochs:

    if settings.debug:
        snapshot_retain_rates(network, output_directory)

    for epoch_index in range(settings.number_of_epochs):
        network.train(train_dataset, minibatch_size, validate_dataset, test_dataset, output_directory);

        if settings.snapshot_interval>0 and network.epoch_index % settings.snapshot_interval == 0:
            model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
            #cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

        print "PROGRESS: %f%%" % (100. * epoch_index / settings.number_of_epochs);

        if settings.debug:
            snapshot_retain_rates(network, output_directory);

    model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
    #cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

    end_train = timeit.default_timer()

    print "Optimization complete..."
    logging.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
        network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index));
    print >> sys.stderr, ('The code for file %s ran for %.2fm' % (os.path.split(__file__)[1], (end_train - start_train) / 60.))

def snapshot_retain_rates(network, output_directory):
    dropout_layer_index = 0;
    for network_layer in network.get_network_layers():
        if not isinstance(network_layer, layers.AdaptiveDropoutLayer):
            continue;

        layer_retain_probability = network_layer.activation_probability.eval();
        logging.info("retain rates stats: epoch %i, shape %s, average %f, minimum %f, maximum %f" % (
            network.epoch_index,
            layer_retain_probability.shape,
            numpy.mean(layer_retain_probability),
            numpy.min(layer_retain_probability),
            numpy.max(layer_retain_probability)));

        retain_rate_file = os.path.join(output_directory,
                                        "layer.%d.epoch.%d.npy" % (dropout_layer_index, network.epoch_index))
        numpy.save(retain_rate_file, layer_retain_probability);
        dropout_layer_index += 1;

if __name__ == '__main__':
    train_dlenet()