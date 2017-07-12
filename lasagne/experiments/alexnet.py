import cPickle
import logging
import numpy
import os
import sys
import timeit

from .. import networks
from .. import nonlinearities, objectives
from .base import load_data, load_and_split_data

__all__ = [
    "train_alexnet",
]

def construct_alexnet_parser():
    from .base import construct_discriminative_parser, add_convpool_options, add_dense_options, add_dropout_options

    model_parser = construct_discriminative_parser();
    model_parser = add_convpool_options(model_parser);
    model_parser = add_dense_options(model_parser);
    model_parser = add_dropout_options(model_parser);

    # model argument set 1
    model_parser.add_argument("--number_of_lrn_layers", dest="number_of_lrn_layers", type=int, action='store', default=2,
                              help="number of local response normalize layers [2]");
    model_parser.add_argument("--locally_connected_filters", dest="locally_connected_filters", action='store', default=None,
                              help="locally connected filter [None], example, '64,32' represents 64 and 32 filters");

    '''
    # model argument set 4
    model_parser.add_argument("--convolution_filter_sizes", dest="convolution_filter_sizes", action='store', default="5*5",
                              help="convolution filter sizes [5*5], example, '5*5,6*6' represents 5*5 and 6*6 filter size for convolution layers respectively");
    model_parser.add_argument("--convolution_strides", dest="convolution_strides", action='store', default="1*1",
                              help="convolution strides [1*1], example, '1*1,2*2' represents 1*1 and 2*2 stride size for convolution layers respectively");
    model_parser.add_argument("--convolution_pads", dest="convolution_pads", action='store', default="2",
                              help="convolution pads [2], example, '2,3' represents 2 and 3 pads for convolution layers respectively");

    # model argument set 5
    model_parser.add_argument("--locally_convolution_filter_sizes", dest="locally_convolution_filter_sizes", action='store', default="3*3",
                              help="locally convolution filter sizes [3*3], example, '5*5,6*6' represents 5*5 and 6*6 filter size for locally connected convolution layers respectively");
    model_parser.add_argument("--locally_convolution_strides", dest="locally_convolution_strides", action='store', default="1*1",
                              help="locally convolution strides [1*1], example, '1*1,2*2' represents 1*1 and 2*2 stride size for locally connected convolution layers respectively");
    model_parser.add_argument("--locally_convolution_pads", dest="locally_convolution_pads", action='store', default="1",
                              help="locally convolution pads [1], example, '2,3' represents 2 and 3 pads for locally connected convolution layers respectively");

    # model argument set 6
    model_parser.add_argument("--pooling_sizes", dest="pooling_sizes", action='store', default="3*3",
                              help="pooling sizes [3*3], example, '2*2,3*3' represents 2*2 and 3*3 pooling size respectively");
    model_parser.add_argument("--pooling_strides", dest="pooling_strides", action='store', default="2*2",
                              help="pooling strides [2*2], example, '2*2,3*3' represents 2*2 and 3*3 pooling stride respectively");
    '''

    #
    #
    #
    #
    #

    '''
    model_parser.add_argument("--pretrained_model_file", dest="pretrained_model_file",
                              help="pretrained model file [None]");
    model_parser.add_argument("--dae_regularizer_lambdas", dest="dae_regularizer_lambdas", nargs="+", type=float, action='store', default=0,
                              help="dae regularization lambda [0]")
    model_parser.add_argument("--layer_corruption_levels", dest="layer_corruption_levels", nargs="+", type=float, action='store', default=0,
                              help="layer corruption level for pre-training [0], either one number of a list of numbers, example, '0.2' represents 0.2 corruption level for all denoising auto encoders, or '0.2,0.5' represents 0.2 corruption level for first denoising auto encoder layer and 0.5 for second one respectively");
    '''

    return model_parser

def validate_alexnet_arguments(arguments):
    from .base import validate_discriminative_arguments, validate_convpool_arguments, validate_dense_arguments, \
        validate_dropout_arguments

    arguments = validate_discriminative_arguments(arguments);

    arguments = validate_convpool_arguments(arguments);
    number_of_convolution_layers = len(arguments.convolution_filters);

    arguments = validate_dense_arguments(arguments);
    number_of_dense_layers = len(arguments.dense_dimensions);

    arguments = validate_dropout_arguments(arguments, number_of_dense_layers);

    # model argument set 1
    assert arguments.number_of_lrn_layers<=number_of_convolution_layers

    if arguments.locally_connected_filters==None:
        arguments.locally_connected_filters = [];
    else:
        arguments.locally_connected_filters = [int(locally_connected_filter) for locally_connected_filter in arguments.locally_connected_filters.split(",")];

    '''
    # model argument set 4
    convolution_filter_sizes = arguments.convolution_filter_sizes;
    convolution_filter_sizes = [tuple(int(x) for x in token.split("*")) for token in convolution_filter_sizes.split(",")];
    if len(convolution_filter_sizes)==1:
        convolution_filter_sizes *= number_of_convolution_layers;
    assert len(convolution_filter_sizes) == number_of_convolution_layers;
    arguments.convolution_filter_sizes = convolution_filter_sizes

    convolution_strides = arguments.convolution_strides
    convolution_strides = [tuple(int(x) for x in token.split("*")) for token in convolution_strides.split(",")];
    if len(convolution_strides)==1:
        convolution_strides *= number_of_convolution_layers;
    assert len(convolution_strides)==number_of_convolution_layers;
    arguments.convolution_strides = convolution_strides

    convolution_pads = arguments.convolution_pads
    convolution_pads = [int(x) for x in convolution_pads.split(",")];
    if len(convolution_pads) == 1:
        convolution_pads *= number_of_convolution_layers;
    assert len(convolution_pads) == number_of_convolution_layers;
    arguments.convolution_pads = convolution_pads

    # model argument set 5
    local_convolution_filter_sizes = arguments.local_convolution_filter_sizes;
    local_convolution_filter_sizes = [tuple(int(x) for x in token.split("*")) for token in
                                      local_convolution_filter_sizes.split(",")];
    if len(local_convolution_filter_sizes) == 1:
        local_convolution_filter_sizes *= number_of_local_convolution_layers;
    assert len(convolution_filter_sizes) == number_of_local_convolution_layers;
    arguments.local_convolution_filter_sizes = local_convolution_filter_sizes

    local_convolution_strides = arguments.local_convolution_strides
    local_convolution_strides = [tuple(int(x) for x in token.split("*")) for token in local_convolution_strides.split(",")];
    if len(local_convolution_strides) == 1:
        local_convolution_strides *= number_of_local_convolution_layers;
    assert len(local_convolution_strides) == number_of_local_convolution_layers;
    arguments.local_convolution_strides = local_convolution_strides

    local_convolution_pads = arguments.local_convolution_pads
    local_convolution_pads = [int(x) for x in local_convolution_pads.split(",")];
    if len(local_convolution_pads) == 1:
        local_convolution_pads *= number_of_local_convolution_layers;
    assert len(local_convolution_pads) == number_of_local_convolution_layers;
    arguments.local_convolution_pads = local_convolution_pads

    # model argument set 6
    pooling_sizes = arguments.pooling_sizes;
    pooling_sizes = [tuple(int(x) for x in token.split("*")) for token in pooling_sizes.split(",")];
    if len(pooling_sizes) == 1:
        pooling_sizes *= number_of_convolution_layers;
    assert len(pooling_sizes) == number_of_convolution_layers;
    arguments.pooling_sizes=pooling_sizes

    pooling_strides = arguments.pooling_strides;
    pooling_strides = [tuple(int(x) for x in token.split("*")) for token in pooling_strides.split(",")];
    if len(pooling_strides) == 1:
        pooling_strides *= number_of_convolution_layers;
    assert len(pooling_strides) == number_of_convolution_layers;
    arguments.pooling_strides = pooling_strides;
    '''

    '''
    dae_regularizer_lambdas = arguments.dae_regularizer_lambdas
    if isinstance(dae_regularizer_lambdas, int):
        dae_regularizer_lambdas = [dae_regularizer_lambdas] * (number_of_layers - 1)
    assert len(dae_regularizer_lambdas) == number_of_layers - 1;
    assert (dae_regularizer_lambda >= 0 for dae_regularizer_lambda in dae_regularizer_lambdas)
    arguments.dae_regularizer_lambdas = dae_regularizer_lambdas;

    layer_corruption_levels = arguments.layer_corruption_levels;
    if isinstance(layer_corruption_levels, int):
        layer_corruption_levels = [layer_corruption_levels] * (number_of_layers - 1)
    assert len(layer_corruption_levels) == number_of_layers - 1;
    assert (layer_corruption_level >= 0 for layer_corruption_level in layer_corruption_levels)
    assert (layer_corruption_level <= 1 for layer_corruption_level in layer_corruption_levels)
    arguments.layer_corruption_levels = layer_corruption_levels;

    pretrained_model_file = arguments.pretrained_model_file;
    pretrained_model = None;
    if pretrained_model_file != None:
        assert os.path.exists(pretrained_model_file)
        pretrained_model = cPickle.load(open(pretrained_model_file, 'rb'));
    '''

    return arguments

def train_alexnet():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron
    This is demonstrated on MNIST.
    """

    arguments, additionals = construct_alexnet_parser().parse_known_args()
    settings = validate_alexnet_arguments(arguments);

    input_directory = settings.input_directory
    output_directory = settings.output_directory
    assert not os.path.exists(output_directory)
    os.mkdir(output_directory);

    logging.basicConfig(filename=os.path.join(output_directory, "model.log"), level=logging.DEBUG, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s');
    validation_data = settings.validation_data
    minibatch_size = settings.minibatch_size

    print "========== ========== ========== ========== =========="
    for key, value in vars(settings).iteritems():
        print "%s=%s" % (key, value);
    print "========== ========== ========== ========== =========="

    logging.info("========== ========== ========== ========== ==========")
    for key, value in vars(settings).iteritems():
        logging.info("%s=%s" % (key, value));
    logging.info("========== ========== ========== ========== ==========")

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

    network = networks.AlexNet(
        incoming=input_shape,

        convolution_filters=settings.convolution_filters,
        convolution_nonlinearities=settings.convolution_nonlinearities,
        # convolution_filter_sizes=None,
        # maxpooling_sizes=None,
        
        number_of_layers_to_LRN=settings.number_of_lrn_layers,
        pool_modes=settings.pool_modes,

        locally_connected_filters=settings.locally_connected_filters,

        dense_dimensions=settings.dense_dimensions,
        dense_nonlinearities=settings.dense_nonlinearities,

        layer_activation_parameters=settings.layer_activation_parameters,
        layer_activation_styles=settings.layer_activation_styles,

        objective_functions=settings.objective,
        update_function=settings.update,

        learning_rate = settings.learning_rate,
        learning_rate_decay=settings.learning_rate_decay,
        max_norm_constraint=settings.max_norm_constraint,
        #learning_rate_decay_style=settings.learning_rate_decay_style,
        #learning_rate_decay_parameter=settings.learning_rate_decay_parameter,
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
    for epoch_index in range(settings.number_of_epochs):
        network.train(train_dataset, minibatch_size, validate_dataset, test_dataset, output_directory);

        if settings.snapshot_interval>0 and network.epoch_index % settings.snapshot_interval == 0:
            model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
            #cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

        print "PROGRESS: %f%%" % (100. * epoch_index / settings.number_of_epochs);

    model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
    #cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

    end_train = timeit.default_timer()

    print "Optimization complete..."
    logging.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
        network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index));
    print >> sys.stderr, ('The code for file %s ran for %.2fm' % (os.path.split(__file__)[1], (end_train - start_train) / 60.))

if __name__ == '__main__':
    train_alexnet()