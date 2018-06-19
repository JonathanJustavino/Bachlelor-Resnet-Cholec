import os
import re
import matplotlib.pyplot as plt
import numpy as np


root_folder = '/media/data/ToolClassification/results'
resnet = 'resnet34'
resnet_type = 'train-last-layer'
document = '2018-06-18_16-46'

full_path = os.path.join(root_folder, resnet, resnet_type, document)


def visualize(data):
    data_sets = {
        '1': {'loss': [], 'accuracy': []},
        '2': {'loss': [], 'accuracy': []},
        '3': {'loss': [], 'accuracy': []},
        '4': {'loss': [], 'accuracy': []},
    }
    with open(data) as file:
        for line in file:
            set_match = re.match("Set: \d", line)
            loss = re.search("[0-9].[0-9]+", line)
            accuracy = re.search("Acc: [0-9].[0-9]+", line)
            if set_match:
                set_nr = re.sub("Set: ", "", set_match.group(0))
                loss = loss.group(0)
                accuracy = re.sub("\D+\s", "", accuracy.group(0))
                data_sets[set_nr]['loss'].append(float(loss))
                data_sets[set_nr]['accuracy'].append(float(accuracy))
        return data_sets


formatted_data = visualize(full_path)

acc = np.array(formatted_data['4']['accuracy'])


def retrieve_percentages(data):
    for set_nr in data:
        for i, value in enumerate(data[set_nr]['accuracy']):
            data[set_nr]['accuracy'][i] = round((value * 100), 2)


def display_accuracy(my_list):

    set_1 = my_list['1']['accuracy']
    set_2 = my_list['2']['accuracy']
    set_3 = my_list['3']['accuracy']
    set_4 = my_list['4']['accuracy']

    graph1, = plt.plot(set_1, 'r', label='Dataset 1')
    graph2, = plt.plot(set_2, 'g', label='Dataset 2')
    graph3, = plt.plot(set_3, 'b', label='Dataset 3')
    graph4, = plt.plot(set_4, 'y', label='Testset')
    plt.legend(handles=[graph1, graph2, graph3, graph4])
    plt.ylabel('Accuracy in %')
    plt.xlabel('Epoch')
    plt.title("{} {}".format(resnet, resnet_type))
    plt.axis([0, 40, 0, 100])
    plt.xticks([x * 4 for x in range(10)])
    plt.yticks([x * 10 for x in range(1, 11)])
    plt.grid(True)
    plt.show()

retrieve_percentages(formatted_data)

display_accuracy(formatted_data)
