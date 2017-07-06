import argparse
import numpy

def main():
    #options = parse_args();
    #train_mlp(options)

    model_selector = argparse.ArgumentParser(description="model selector")
    model_selector.add_argument("--model", dest="model", action='store', default=None, help="model [None]");

    arguments, additionals = model_selector.parse_known_args()

    print "========== ========== ========== ========== =========="
    for key, value in vars(arguments).iteritems():
        print "%s=%s" % (key, value);
    print "========== ========== ========== ========== =========="
    print additionals
    print "========== ========== ========== ========== =========="


    if arguments.model == "mlp":
        from lasagne.experiments.mlp import train_mlp
        train_mlp()
    elif arguments.model == "dmlp":
        from lasagne.experiments.dmlp import train_dmlp
        train_dmlp()
    elif arguments.model == "cnn":
        from lasagne.experiments.cnn import train_cnn
        train_cnn()
    elif arguments.model == "dcnn":
        from lasagne.experiments.dcnn import train_dcnn
        train_dcnn()
    elif arguments.model == "snn":
        from lasagne.experiments.snn import train_snn
        train_snn()
    elif arguments.model == "fdn":
        from lasagne.experiments.fdn import train_fdn
        train_fdn()
    elif arguments.model == "vdn":
        from lasagne.experiments.vdn import train_vdn
        train_vdn()

if __name__ == '__main__':
    main();