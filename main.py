import argparse


def main():
	model_selector = argparse.ArgumentParser(description="model selector")
	model_selector.add_argument("--model", dest="model", action='store', default=None, help="model [None]")
	model_selector.add_argument("--mode", dest="mode", action='store', default="train", help="mode [None]")

	arguments, additionals = model_selector.parse_known_args()

	print("========== ========== ========== ========== ==========")
	for key, value in vars(arguments).items():
		print("%s=%s" % (key, value));
	print("========== ========== ========== ========== ==========")
	print(additionals)
	print("========== ========== ========== ========== ==========")

	if arguments.model == "mlp":
		from lasagne.experiments.mlp import start_mlp, resume_mlp
		if arguments.mode == "start":
			start_mlp()
		elif arguments.mode=="resume":
			resume_mlp()
	elif arguments.model == "mlpA":
		from lasagne.experiments.mlpA import start_mlpA
		start_mlpA()
	elif arguments.model == "mlpD":
		from lasagne.experiments.mlpD import start_mlpD
		start_mlpD()
	elif arguments.model == "mlpHan":
		from lasagne.experiments.mlpHan import train_mlpHan
		train_mlpHan()
	elif arguments.model == "snn":
		from lasagne.experiments.snn import train_snn
		train_snn()
	elif arguments.model == "lenet":
		from lasagne.experiments.lenet import start_lenet
		start_lenet()
	elif arguments.model == "lenetA":
		from lasagne.experiments.lenetA import train_lenetA
		train_lenetA()

	elif arguments.model == "alexnet":
		from lasagne.experiments.alexnet import train_alexnet
		train_alexnet()
	elif arguments.model == "elman":
		from lasagne.experiments.elman import train_elman
		train_elman()

	'''
	elif arguments.model == "dalexnet":
		from lasagne.experiments.alexnetd import train_dalexnet
		train_dalexnet()
	elif arguments.model == "delman":
		from lasagne.experiments.elmand import train_delman
		train_delman()
	elif arguments.model == "fdn":
		from lasagne.experiments.fdn import train_fdn
		train_fdn()
	elif arguments.model == "vdn":
		from lasagne.experiments.vdn import train_vdn
		train_vdn()
	'''


if __name__ == '__main__':
	main();
