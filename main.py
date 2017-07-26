import argparse


def main():
	model_selector = argparse.ArgumentParser(description="model selector")
	model_selector.add_argument("--model", dest="model", action='store', default=None, help="model [None]");

	arguments, additionals = model_selector.parse_known_args()

	print("========== ========== ========== ========== ==========")
	for key, value in vars(arguments).items():
		print("%s=%s" % (key, value));
	print("========== ========== ========== ========== ==========")
	print(additionals)
	print("========== ========== ========== ========== ==========")

	if arguments.model == "mlp":
		from lasagne.experiments.mlp import train_mlp
		train_mlp()
	elif arguments.model == "dmlp":
		from lasagne.experiments.mlpd import train_dmlp
		train_dmlp()
	elif arguments.model == "snn":
		from lasagne.experiments.snn import train_snn
		train_snn()
	elif arguments.model == "fdn":
		from lasagne.experiments.fdn import train_fdn
		train_fdn()
	elif arguments.model == "vdn":
		from lasagne.experiments.vdn import train_vdn
		train_vdn()
	elif arguments.model == "lenet":
		from lasagne.experiments.lenet import train_lenet
		train_lenet()
	elif arguments.model == "dlenet":
		from lasagne.experiments.lenetd import train_dlenet
		train_dlenet()
	elif arguments.model == "alexnet":
		from lasagne.experiments.alexnet import train_alexnet
		train_alexnet()
	elif arguments.model == "dalexnet":
		from lasagne.experiments.alexnetd import train_dalexnet
		train_dalexnet()
	elif arguments.model=="elman":
		from lasagne.experiments.elman import train_elman
		train_elman()


if __name__ == '__main__':
	main();
