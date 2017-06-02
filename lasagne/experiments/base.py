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
    "compile_generic_parser",
    "load_data_to_train_validate",
    "load_data",
    "Configuration",
    "load_mnist",
]

def compile_generic_parser():
    generic_parser = argparse.ArgumentParser(description="generic neural network arguments", add_help=True)

    # generic argument set 1
    generic_parser.add_argument("--input_directory", dest="input_directory", action='store', default=None,
                                help="input directory [None]");
    generic_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
                                help="output directory [None]");
    # generic_parser.add_argument("--logging_file", dest="logging_file", action='store', default=None,
    # help="logging file [None]");
    generic_parser.add_argument("--validation_data", dest="validation_data", type=int, action='store', default=-1,
                                help="validation data [-1 - load validate.(feature|label).npy for validation], 0 - no validation data used");

    # generic argument set 2
    generic_parser.add_argument("--objective_function", dest="objective_function", action='store',
                                default="categorical_crossentropy",
                                help="objective function [categorical_crossentropy], example, 'squared_error' represents the neural network optimizes squared error");
    generic_parser.add_argument("--update_function", dest="update_function", action='store',
                                default="nesterov_momentum",
                                help="update function to minimize [nesterov_momentum], example, 'sgd' represents the stochastic gradient descent");
    generic_parser.add_argument("--regularize_function", dest='regularize_functions', action='append',
                                default=[],
                                help='regularize function')

    # generic argument set 3
    generic_parser.add_argument("--minibatch_size", dest="minibatch_size", type=int, action='store', default=-1,
                                help="mini-batch size [-1]");
    generic_parser.add_argument("--number_of_epochs", dest="number_of_epochs", type=int, action='store', default=-1,
                                help="number of epochs [-1]");
    generic_parser.add_argument("--snapshot_interval", dest="snapshot_interval", type=int, action='store', default=-1,
                                help="snapshot interval in number of epochs [-1 - no snapshot]");

    # generic argument set 4
    generic_parser.add_argument("--learning_rate", dest="learning_rate", type=float, action='store', default=1e-2,
                                help="learning rate [1e-2]")
    generic_parser.add_argument("--learning_rate_decay_style", dest="learning_rate_decay_style", action='store',
                                default=None,
                                help="learning rate decay style [None], example, 'inverse_t', 'exponential'");
    generic_parser.add_argument("--learning_rate_decay_parameter", dest="learning_rate_decay_parameter", type=float,
                                action='store', default=0,
                                help="learning rate decay [0 - no learning rate decay], example, half life iterations for inverse_t or exponential decay")

    return generic_parser

'''
def load_data_to_test(input_directory):
    test_set_x = numpy.load(os.path.join(input_directory, "test.feature.npy"))
    test_set_y = numpy.load(os.path.join(input_directory, "test.label.npy"))
    assert test_set_x.shape[0] == len(test_set_y);
    test_dataset = (test_set_x, test_set_y);
    logging.info("successfully load data %s with %d to test..." % (input_directory, test_set_x.shape[0]))
    return test_dataset

def load_data_to_train_validate(input_directory, number_of_training_data=-1):
    data_x = numpy.load(os.path.join(input_directory, "train.feature.npy"))
    data_y = numpy.load(os.path.join(input_directory, "train.label.npy"))
    assert data_x.shape[0] == len(data_y);

    if number_of_training_data <= 0 or number_of_training_data >= len(data_y):
        number_of_training_data = len(data_y);
    indices = numpy.random.permutation(len(data_y));
    train_indices = indices[:number_of_training_data]
    validate_indices = indices[number_of_training_data:]

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
'''

class Configuration(object):
    def __init__(self, arguments):
        # generic argument set 4
        self.learning_rate = arguments.learning_rate;
        assert self.learning_rate > 0;
        self.learning_rate_decay_style = arguments.learning_rate_decay_style;
        assert self.learning_rate_decay_style == None or self.learning_rate_decay_style in ["inverse_t", "exponential"];
        self.learning_rate_decay_parameter = arguments.learning_rate_decay_parameter;
        assert self.learning_rate_decay_parameter >= 0;

        # generic argument set 3
        self.minibatch_size = arguments.minibatch_size;
        assert (self.minibatch_size > 0);
        self.number_of_epochs = arguments.number_of_epochs;
        assert (self.number_of_epochs > 0);
        self.snapshot_interval = arguments.snapshot_interval;
        # assert(options.snapshot_interval > 0);

        # generic argument set 2
        #objective_function = arguments.objective_function;
        self.objective_function = getattr(objectives, arguments.objective_function)
        #update_function = arguments.update_function;
        self.update_function = getattr(updates, arguments.update_function)

        self.regularizer_functions = {};
        for regularizer_weight_mapping in arguments.regularize_functions:
            fields = regularizer_weight_mapping.split(":");
            regularizer_function = getattr(regularization, fields[0]);
            if len(fields)==1:
                self.regularizer_functions[regularizer_function] = 1.0;
            elif len(fields)==2:
                tokens = fields[1].split(",");
                if len(tokens)==1:
                    weight = float(tokens[0]);
                else:
                    weight = [float(token) for token in tokens];
                self.regularizer_functions[regularizer_function] = weight;
            else:
                logging.error("unrecognized regularizer function setting %s..." % (regularizer_weight_mapping));

        # generic argument set 1
        self.input_directory = arguments.input_directory;
        assert os.path.exists(self.input_directory)

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
        self.output_directory = output_directory;

        self.validation_data = arguments.validation_data;

def load_data(input_directory, dataset="test"):
    data_set_x = numpy.load(os.path.join(input_directory, "%s.feature.npy" % dataset))
    data_set_y = numpy.load(os.path.join(input_directory, "%s.label.npy" % dataset))
    assert data_set_x.shape[0] == len(data_set_y);
    logging.info("successfully load data %s with %d to %s..." % (input_directory, data_set_x.shape[0], dataset))
    return (data_set_x, data_set_y)

def load_data_to_train_validate(input_directory, number_of_validate_data=0):
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