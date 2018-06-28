import re
import csv
from sklearn.metrics import confusion_matrix
import numpy
import os

directory = "/home/justaviju/PycharmProjects/resnet"
file_path = "2018-06-28_11-06_predictions.csv"
alt_file_path = "test.csv"
l1 ='Preparation'
l2 ='Calot Triangle Dissection'
l3 ='Cleaning Coagulation'
l4 ='Gallbladder Dissection'
l5 ='Gallbladder Retraction'
l6 ='Clipping Cutting'
l7 ='Gallbladder Packaging'
classes = [l1, l2, l3, l4, l5, l6, l7]
dimension = 7
confusion = [[0 for x in range(dimension)] for y in range(dimension)]
with open(os.path.join(directory, file_path), 'r') as file:
    predictions = []
    labels = []
    data = {
        "predictions": [],
        "labels": []
    }
    reader = csv.reader(file)
    for line in file:
        if re.match("Pred", line):
            newline = re.sub("Predictions:\s", "", line)
            newline.split()
            preds_to_int = 'preds = ' + newline
            exec(preds_to_int)
            data["predictions"] += preds
        if re.match("Label", line):
            newline2 = re.sub("Labels:\s*", "", line)
            newline2.split()
            labels_to_int = 'labls = ' + newline2
            exec(labels_to_int)
            data["labels"] += labls

total_labels = len(data["labels"])
total_predictions = len(data["predictions"])
net_classes = data["labels"]
net_preds = data["predictions"]

all_classes =len(net_classes)
all_preds =len(net_preds)
c = confusion_matrix(net_classes, net_preds)

print("L1, L2, L3, L4, L5, L6, L7 \n{}".format(c))
print("Labels:")


for i, label in enumerate(classes):
	print("L{}: {}".format(i + 1, label))