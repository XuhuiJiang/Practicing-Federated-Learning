
import argparse
import json
import random

import datasets
from client import *
from server import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()

	with open(args.conf, 'r') as f:
		conf = json.load(f)	

	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	
	server = Server(conf, eval_datasets)

	# 列表
	clients = []
	# 10
	for c in range(conf["num_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, c))
		
	print("\n\n")
	# 20
	for e in range(conf["global_epochs"]):
		# 随机从客户端中选出k个候选客户端 k=5
		candidates = random.sample(clients, conf["k"])
		# 权重集合
		weight_accumulator = {}
		
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)
		# k=5,每个客户端进行一轮，进行5轮训练，每轮的epoch为3
		for c in candidates:
			diff = c.local_train(server.global_model)
			
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])

		server.model_aggregate(weight_accumulator)
		
		acc, loss = server.model_eval()

		print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
