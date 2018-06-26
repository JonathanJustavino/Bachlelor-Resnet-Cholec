import torch
import re


label_amount = 7
confusion = torch.zeros(label_amount, label_amount)
file = '2018-06-25_13-23dummy.txt'

matrix = [[7],[7]]

with open(file) as predictions:
	l = predictions.readline()
	l1 = predictions.readline()
	l2 = predictions.readline()
	l3 = predictions.readline()
	l4 = predictions.readline()
	l5 = predictions.readline()
	l6 = predictions.readline()
	l7 = predictions.readline()

	preds = re.sub("Predictions: ", "", l2) + l3
	print(preds)
	ground = re.sub("Labels:\s+", "", l4) + l5
	print(ground)