import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import re
import csv

print(__doc__)


plt.ion()

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
label_names = [l1, l2, l3, l4, l5, l6, l7]
label_numbers = [x + 1 for x in range(len(label_names))]
dimension = 7
confusion = [[0 for x in range(dimension)] for y in range(dimension)]
with open(file_path, 'r') as file:
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
net_labels = data["labels"]
net_preds = data["predictions"]

print(len(net_labels))
print(len(net_preds))
cnf_matrix = confusion_matrix(net_labels, net_preds)


print("Labels:")
# for i, label in enumerate(class_names):
#     print("L{}: {}".format(i, label), sep=", ", end="\t",)
print(label_names)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.legend(loc='center left')
    # legen = plt.subplot(111)
    # for i in range(7):
    #     ax.plot(i + 1, label_names[i])
 
    # box = ax.get_position()
    # ax.set_printoptions([box.x0, box.y0, box.width * 0.8, box.height])



    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(net_labels, net_preds)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=label_numbers,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=label_numbers, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
q_ = input("Input: ")