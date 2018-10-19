import math
from con_matrices import scheduler, no_scheduler


def precision(matrix):
    length = range(len(matrix))
    labels = list(length)
    for i in length:
        for j in length:
            labels[i] += matrix[i][j]
        labels[i] = math.ceil((matrix[i][i] / labels[i]) * 100.0) / 100.0
    return labels


def recall(matrix):
    length = range(len(matrix))
    labels = list(length)
    for j in length:
        for i in length:
            labels[j] += matrix[i][j]
        labels[j] = math.ceil((matrix[j][j] / labels[j]) * 100.0) / 100.0
    return labels


def accuracy(matrix):
    length = range(len(matrix))
    correct = 0
    predictions = 0
    for i in length:
        for j in length:
            predictions += matrix[i][j]
            if i == j:
                correct += matrix[i][j]
    return correct / predictions


def jaccard_score(matrix):
    nummerator = 0
    denominator = 0
    jaccard_idx = 0
    for i in range(len(matrix)):
        nummerator = matrix[i][i]
        for j in range(len(matrix[i])):
            if i == j:
                pass
            else:
                denominator += matrix[i][j]
                denominator += matrix[j][i]
        denominator += nummerator
        jaccard_idx += (nummerator / denominator)
        nummerator = 0
        denominator = 0
    return jaccard_idx / len(matrix)


k_fold_pre = [0 for i in range(7)]
avg_acc = 0

for set in scheduler['resnet18']:
    result = precision(scheduler['resnet18'][set])
    for i in range(len(result)):
        k_fold_pre[i] += result[i]


f = (lambda x: x / 4)
g = (lambda x: round(x, 2))
x = list(map(g, map(f, k_fold_pre)))

avg_pre = sum(x) / len(x)
print(round(avg_pre, 2))
