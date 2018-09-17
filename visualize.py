import os
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread


root_folder = '/media/TCO/TCO-Studenten/justaviju/results/'
resnet_type = ''
full_path = sys.argv[1]
print(full_path)
try:
    validation_set = re.search("Valset[1-4]", full_path, re.IGNORECASE).group()
    validation_nr = re.sub("\D", "", validation_set)
    image_dir = re.sub("\d*.\d*.\d*.\d*.\d*$", "", full_path, re.IGNORECASE)
    print("Image DIR: ", image_dir)
except:
    validation_nr = -1


def format_data(data):
    data_sets = {
        '1': {'loss': [], 'accuracy': []},
        '2': {'loss': [], 'accuracy': []},
        '3': {'loss': [], 'accuracy': []},
        '4': {'loss': [], 'accuracy': []},
    }
    with open(data) as file:
        for line in file:
            type_match = re.match("resnet[0-9]+", line, re.IGNORECASE)
            if type_match:
                global resnet_type
                resnet_type = type_match.group()
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


formatted_data = format_data(full_path)

def retrieve_percentages(data):
    for set_nr in data:
        for i, value in enumerate(data[set_nr]['accuracy']):
            data[set_nr]['accuracy'][i] = round((value * 100), 2)


def display_accuracy(my_list):

    set_1 = my_list['1']['accuracy']
    set_2 = my_list['2']['accuracy']
    set_3 = my_list['3']['accuracy']
    set_4 = my_list['4']['accuracy']

    graph1, = plt.plot(set_1, 'r', label='Folder 1')
    graph2, = plt.plot(set_2, 'g', label='Folder 2')
    graph3, = plt.plot(set_3, 'b', label='Folder 3')
    graph4, = plt.plot(set_4, 'y', label='Folder 4')
    plt.legend(handles=[graph1, graph2, graph3, graph4])
    plt.ylabel('Accuracy in %')
    plt.xlabel('Epoch')
    plt.title("{} Test Set: {}".format(resnet_type, validation_nr))
    plt.axis([0, 40, 65, 100])   
    plt.grid(True)
    # plt.show()
    # plt.savefig('accuracy.png')
    plt.savefig('{}/accuracy.png'.format(image_dir))



def display_loss(my_list):
    set_1 = my_list['1']['loss']
    set_2 = my_list['2']['loss']
    set_3 = my_list['3']['loss']
    set_4 = my_list['4']['loss']

    graph1, = plt.plot(set_1, 'r', label='Folder 1')
    graph2, = plt.plot(set_2, 'g', label='Folder 2')
    graph3, = plt.plot(set_3, 'b', label='Folder 3')
    graph4, = plt.plot(set_4, 'y', label='Folder 4')
    plt.legend(handles=[graph1, graph2, graph3, graph4])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title("{} Test Set: {}".format(resnet_type, validation_nr))
    plt.axis([0, 50, 0, 3])
    plt.grid(True)
    # plt.show()
    # plt.savefig('loss.png')
    plt.savefig('{}/loss.png'.format(image_dir))


retrieve_percentages(formatted_data)

arguments = [formatted_data]

# t1 = Thread(target=display_loss, args=arguments)
# t2 = Thread(target=display_accuracy, args=arguments)

# t1.start()
# t2.start()

display_accuracy(formatted_data)
display_loss(formatted_data)
# plt.show()
