import argparse
import datetime
import logging
import numpy
import os
import random
import sys

from .. import layers
from .. import networks
from .. import nonlinearities, objectives, updates, utils, regularization

__all__ = [
    "construct_discriminative_parser",
    "validate_discriminative_arguments",
    "add_dense_options",
    "validate_dense_arguments",
    "add_dropout_options",
    "validate_dropout_arguments",
    #
    "load_and_split_data",
    "load_data",
    "load_mnist",
]

def construct_generic_parser():
    generic_parser = argparse.ArgumentParser(description="generic neural network arguments", add_help=True)

    # generic argument set 1
    generic_parser.add_argument("--input_directory", dest="input_directory", action='store', default=None,
                                help="input directory [None]");
    generic_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
                                help="output directory [None]");
    # generic_parser.add_argument("--logging_file", dest="logging_file", action='store', default=None,
    # help="logging file [None]");

    # generic argument set 2
    generic_parser.add_argument("--objective", dest="objective", action='store',
                                default="categorical_crossentropy",
                                help="objective function [categorical_crossentropy], example, 'squared_error' represents the neural network optimizes squared error");
    generic_parser.add_argument("--update", dest="update", action='store',
                                default="nesterov_momentum",
                                help="update function to minimize [nesterov_momentum], example, 'sgd' represents the stochastic gradient descent");
    generic_parser.add_argument("--regularizer", dest='regularizer', action='append', default=[],
                                help="regularizer function [None], example, " +
                                     "'l2:0.1'=l2-regularizer with lambda 0.1 applied over all layers, " +
                                     "'l1:0.1,0.2,0.3'=l1-regularizer with lambda 0.1, 0.2, 0.3 applied over three layers")

    # generic argument set 3
    generic_parser.add_argument("--minibatch_size", dest="minibatch_size", type=int, action='store', default=-1,
                                help="mini-batch size [-1]");
    generic_parser.add_argument("--number_of_epochs", dest="number_of_epochs", type=int, action='store', default=-1,
                                help="number of epochs [-1]");
    generic_parser.add_argument("--snapshot_interval", dest="snapshot_interval", type=int, action='store', default=0,
                                help="snapshot interval in number of epochs [0 - no snapshot]");

    # generic argument set 4
    generic_parser.add_argument("--learning_rate", dest="learning_rate", type=float, action='store', default=1e-2,
                                help="learning rate [1e-2]")
    generic_parser.add_argument("--learning_rate_decay", dest="learning_rate_decay", action='store', default=None,
                                help="learning rate decay [None], example, 'iteration,inverse_t,0.2,0.1', 'epoch,exponential,1.7,0.1', 'epoch,step,0.2,100'");
    '''
    generic_parser.add_argument("--learning_rate_decay_style", dest="learning_rate_decay_style", action='store',
                                default=None,
                                help="learning rate decay style [None], example, 'inverse_t', 'exponential'");
    generic_parser.add_argument("--learning_rate_decay_parameter", dest="learning_rate_decay_parameter", #type=float,
                                action='store', default=None,
                                help="learning rate decay [0 - no learning rate decay], example, half life iterations for inverse_t or exponential decay")
    '''
    generic_parser.add_argument('--debug', dest="debug", action='store_true', default=False, help="debug mode [False]")

    return generic_parser

def validate_generic_arguments(arguments):
    # generic argument set 4
    assert arguments.learning_rate > 0;
    if arguments.learning_rate_decay!=None:
        learning_rate_decay_tokens = arguments.learning_rate_decay.split(",");
        assert len(learning_rate_decay_tokens)==4;
        assert learning_rate_decay_tokens[0] in ["iteration", "epoch"];
        assert learning_rate_decay_tokens[1] in ["inverse_t", "exponential", "step"];
        learning_rate_decay_tokens[2] = float(learning_rate_decay_tokens[2]);
        learning_rate_decay_tokens[3] = float(learning_rate_decay_tokens[3]);
        arguments.learning_rate_decay = learning_rate_decay_tokens;

    # generic argument set 3
    assert arguments.minibatch_size > 0;
    assert arguments.number_of_epochs > 0;
    assert arguments.snapshot_interval >= 0;

    # generic argument set 2
    arguments.objective = getattr(objectives, arguments.objective)
    arguments.update = getattr(updates, arguments.update)

    regularizers = {};
    for regularizer_weight_mapping in arguments.regularizer:
        fields = regularizer_weight_mapping.split(":");
        regularizer_function = getattr(regularization, fields[0]);
        if len(fields)==1:
            regularizers[regularizer_function] = 1.0;
        elif len(fields)==2:
            tokens = fields[1].split(",");
            if len(tokens)==1:
                weight = float(tokens[0]);
            else:
                weight = [float(token) for token in tokens];
            regularizers[regularizer_function] = weight;
        else:
            logging.error("unrecognized regularizer function setting %s..." % (regularizer_weight_mapping));
    arguments.regularizer = regularizers;

    # generic argument set 1
    #self.input_directory = arguments.input_directory;
    assert os.path.exists(arguments.input_directory)

    output_directory = arguments.output_directory;
    assert (output_directory != None);
    if not os.path.exists(output_directory):
        os.mkdir(os.path.abspath(output_directory));
    # adjusting output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S-%f") + "";
    #suffix += "-%s" % ("mlp");
    output_directory = os.path.join(output_directory, suffix);
    assert not os.path.exists(output_directory)
    #os.mkdir(os.path.abspath(output_directory));
    arguments.output_directory = output_directory;

    return arguments;

def construct_discriminative_parser():
    model_parser = construct_generic_parser();

    # model argument set
    model_parser.add_argument("--validation_data", dest="validation_data", type=int, action='store', default=0,
                                help="validation data [0 - no validation data used], -1 - load validate.(feature|label).npy for validation]");
    model_parser.add_argument("--validation_interval", dest="validation_interval", type=int, action='store', default=1000,
                              help="validation interval in number of mini-batches [1000]");

    return model_parser;

def validate_discriminative_arguments(arguments):
    arguments = validate_generic_arguments(arguments);

    # model argument set
    assert arguments.validation_data >= -1;
    assert (arguments.validation_interval > 0);

    return arguments

def add_convpool_options(model_parser):
    # model argument set 1
    model_parser.add_argument("--convolution_filters", dest="convolution_filters", action='store', default=None,
                              help="number of convolution filters [None], example, '32,16' represents 32 and 16 filters for convolution layers respectively");
    model_parser.add_argument("--convolution_nonlinearities", dest="convolution_nonlinearities", action='store', default=None,
                              help="activation functions of convolution layers [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively");
    model_parser.add_argument("--pool_modes", dest="pool_modes", action='store', default="max",
                              help="pool modes after each convolution layers [max], set to none to omit, example, 'max,max,none,none,max'");

    return model_parser;

def validate_convpool_arguments(arguments):
    # model argument set 1
    assert arguments.convolution_filters != None
    conv_filters = arguments.convolution_filters.split(",")
    arguments.convolution_filters = [int(conv_filter) for conv_filter in conv_filters]

    assert arguments.convolution_nonlinearities != None
    conv_nonlinearities = arguments.convolution_nonlinearities.split(",")
    arguments.convolution_nonlinearities = [getattr(nonlinearities, conv_nonlinearity) for conv_nonlinearity in
                                            conv_nonlinearities]

    assert len(conv_filters) == len(conv_nonlinearities)

    assert arguments.pool_modes != None
    pool_modes = arguments.pool_modes.split(",")
    if len(pool_modes) == 1:
        pool_modes *= len(arguments.convolution_filters);
    for pool_mode_index in xrange(len(pool_modes)):
        if pool_modes[pool_mode_index].lower() == "none":
            pool_modes[pool_mode_index] = None;
    arguments.pool_modes = pool_modes

    assert len(conv_filters) == len(pool_modes)

    return arguments

def add_dense_options(model_parser):
    # model argument set 1
    model_parser.add_argument("--dense_dimensions", dest="dense_dimensions", action='store', default=None,
                              help="dimension of different layer [None], example, '100,500,10' represents 3 layers contains 100, 500, and 10 neurons respectively");
    model_parser.add_argument("--dense_nonlinearities", dest="dense_nonlinearities", action='store', default=None,
                              help="activation functions of different layer [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively");

    return model_parser;

def validate_dense_arguments(arguments):
    # model argument set 1
    assert arguments.dense_dimensions != None
    dense_dimensions = arguments.dense_dimensions.split(",")
    arguments.dense_dimensions = [int(dimensionality) for dimensionality in dense_dimensions]

    assert arguments.dense_nonlinearities != None
    dense_nonlinearities = arguments.dense_nonlinearities.split(",")
    arguments.dense_nonlinearities = [getattr(nonlinearities, dense_nonlinearity) for dense_nonlinearity in dense_nonlinearities]

    assert len(arguments.dense_nonlinearities) == len(arguments.dense_dimensions);

    return arguments

def add_dropout_options(model_parser):
    # model argument set 2
    model_parser.add_argument("--layer_activation_parameters", dest="layer_activation_parameters", action='store', default="1.0",
                              help="dropout probability of different layer [1], either one number of a list of numbers, example, '0.2' represents 0.2 dropout rate for all input+hidden layers, or '0.2,0.5' represents 0.2 dropout rate for input layer and 0.5 dropout rate for first hidden layer respectively");
    model_parser.add_argument("--layer_activation_styles", dest="layer_activation_styles", action='store', default="bernoulli",
                              help="dropout style different layer [bernoulli], example, 'bernoulli,beta-bernoulli' represents 2 layers with bernoulli and beta-bernoulli dropout respectively");

    return model_parser;

def validate_dropout_arguments(arguments, number_of_layers):
    # model argument set
    layer_activation_styles = arguments.layer_activation_styles;
    layer_activation_style_tokens = layer_activation_styles.split(",")
    if len(layer_activation_style_tokens) == 1:
        layer_activation_styles = [layer_activation_styles for layer_index in xrange(number_of_layers)]
    elif len(layer_activation_style_tokens) == number_of_layers:
        layer_activation_styles = layer_activation_style_tokens
        # [float(layer_activation_parameter) for layer_activation_parameter in layer_activation_parameter_tokens]
    assert len(layer_activation_styles) == number_of_layers;
    assert (layer_activation_style in set(
        ["bernoulli", "beta_bernoulli", "reciprocal_beta_bernoulli", "reverse_reciprocal_beta_bernoulli",
         "mixed_beta_bernoulli"]) for layer_activation_style in layer_activation_styles)
    arguments.layer_activation_styles = layer_activation_styles;

    layer_activation_parameters = arguments.layer_activation_parameters;
    layer_activation_parameter_tokens = layer_activation_parameters.split(",")
    if len(layer_activation_parameter_tokens) == 1:
        layer_activation_parameters = [layer_activation_parameters for layer_index in xrange(number_of_layers)]
    elif len(layer_activation_parameter_tokens) == number_of_layers:
        layer_activation_parameters = layer_activation_parameter_tokens
    assert len(layer_activation_parameters) == number_of_layers;

    for layer_index in xrange(number_of_layers):
        if layer_activation_styles[layer_index] == "bernoulli":
            layer_activation_parameters[layer_index] = float(layer_activation_parameters[layer_index])
            assert layer_activation_parameters[layer_index] <= 1;
            assert layer_activation_parameters[layer_index] > 0;
        elif layer_activation_styles[layer_index] == "beta_bernoulli" \
                or layer_activation_styles[layer_index] == "reciprocal_beta_bernoulli" \
                or layer_activation_styles[layer_index] == "reverse_reciprocal_beta_bernoulli" \
                or layer_activation_styles[layer_index] == "mixed_beta_bernoulli":
            layer_activation_parameter_tokens = layer_activation_parameters[layer_index].split("+");
            assert len(layer_activation_parameter_tokens) == 2;
            layer_activation_parameters[layer_index] = (float(layer_activation_parameter_tokens[0]),
                                                        float(layer_activation_parameter_tokens[1]))
            assert layer_activation_parameters[layer_index][0] > 0;
            assert layer_activation_parameters[layer_index][1] > 0;
            if layer_activation_styles[layer_index] == "mixed_beta_bernoulli":
                assert layer_activation_parameters[layer_index][0] < 1;
    arguments.layer_activation_parameters = layer_activation_parameters;

    return arguments

def load_data(input_directory, dataset="test"):
    data_set_x = numpy.load(os.path.join(input_directory, "%s.feature.npy" % dataset))
    data_set_y = numpy.load(os.path.join(input_directory, "%s.label.npy" % dataset))
    assert data_set_x.shape[0] == len(data_set_y);
    logging.info("successfully load data %s with %d to %s..." % (input_directory, data_set_x.shape[0], dataset))
    return (data_set_x, data_set_y)

def load_and_split_data(input_directory, number_of_validate_data=0):
    data_x = numpy.load(os.path.join(input_directory, "train.feature.npy"))
    data_y = numpy.load(os.path.join(input_directory, "train.label.npy"))
    assert data_x.shape[0] == len(data_y);

    assert number_of_validate_data>=0 and number_of_validate_data<len(data_y)
    indices = numpy.random.permutation(len(data_y));
    train_indices = indices[number_of_validate_data:]
    validate_indices = indices[:number_of_validate_data]

    train_set_x = data_x[train_indices, :]
    train_set_y = data_y[train_indices]
    train_dataset = (train_set_x, train_set_y)
    #numpy.save(os.path.join(output_directory, "train.index.npy"), train_indices);
    logging.info("successfully load data %s with %d to train..." % (input_directory, train_set_x.shape[0]))

    if len(validate_indices) > 0:
        validate_set_x = data_x[validate_indices, :]
        validate_set_y = data_y[validate_indices]
        validate_dataset = (validate_set_x, validate_set_y)
        #numpy.save(os.path.join(output_directory, "validate.index.npy"), validate_indices);
        logging.info("successfully load data %s with %d to validate..." % (input_directory, validate_set_x.shape[0]))
    else:
        validate_dataset = None;

    return (train_dataset, train_indices), (validate_dataset, validate_indices);

def split_train_data_to_cross_validate(input_directory, number_of_folds=5, output_directory=None):
    data_x = numpy.load(os.path.join(input_directory, "train.feature.npy"))
    data_y = numpy.load(os.path.join(input_directory, "train.label.npy"))
    assert data_x.shape[0] == len(data_y);
    number_of_data = len(data_y);

    assert number_of_folds >= 0 and number_of_folds < len(data_y)
    split_indices = range(0, number_of_data, number_of_data/number_of_folds);
    if len(split_indices)==number_of_folds+1:
        split_indices[-1] = number_of_data;
    elif len(split_indices)==number_of_folds:
        split_indices.append(number_of_data);
    else:
        logging.error("something went wrong...");
        sys.exit()
    assert len(split_indices)==number_of_folds+1;
    indices = range(len(data_y));
    random.shuffle(indices);

    if output_directory==None:
        output_directory = input_directory;

    fold_index = 0;
    for start_index, end_index in zip(split_indices[:-1], split_indices[1:]):
        fold_output_directory = os.path.join(output_directory, "folds.%d.index.%d" % (number_of_folds, fold_index));
        if not os.path.exists(fold_output_directory):
            os.mkdir(fold_output_directory);

        train_indices = indices[:start_index] + indices[end_index:];
        test_indices = indices[start_index:end_index];
        assert len(train_indices)>0;
        assert len(test_indices)>0;

        train_set_x = data_x[train_indices, :]
        train_set_y = data_y[train_indices]
        numpy.save(os.path.join(fold_output_directory, "train.feature.npy"), train_set_x);
        numpy.save(os.path.join(fold_output_directory, "train.label.npy"), train_set_y);
        numpy.save(os.path.join(fold_output_directory, "train.index.npy"), train_indices);

        test_set_x = data_x[test_indices, :]
        test_set_y = data_y[test_indices]
        numpy.save(os.path.join(fold_output_directory, "test.feature.npy"), test_set_x);
        numpy.save(os.path.join(fold_output_directory, "test.label.npy"), test_set_y);
        numpy.save(os.path.join(fold_output_directory, "test.index.npy"), test_indices);

        logging.info("successfully split data to %d for train and %d for test..." % (len(train_indices), len(test_indices)))
        logging.info("successfully generate fold %d to %s..." % (fold_index, fold_output_directory))

        print("successfully split data to %d for train and %d for test..." % (len(train_indices), len(test_indices)))
        print("successfully generate fold %d to %s..." % (fold_index, fold_output_directory))

        fold_index += 1;

    return;

#
#
#
#
#

def load_mnist():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = numpy.frombuffer(f.read(), numpy.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / numpy.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = numpy.frombuffer(f.read(), numpy.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    input_directory = sys.argv[1]
    number_of_folds = int(sys.argv[2])
    split_train_data_to_cross_validate(input_directory, number_of_folds);