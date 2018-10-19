import numpy as np
import sys

###############################################################
# ########################scheduler########################## #
###############################################################
# Precision
###############################################################

precision_r18_val1_s = [0.55, 0.81, 0.54, 0.84, 0.63, 0.59, 0.72]
precision_r18_val2_s = [0.62, 0.84, 0.73, 0.88, 0.69, 0.61, 0.69]
precision_r18_val3_s = [0.58, 0.84, 0.66, 0.81, 0.61, 0.58, 0.66]
precision_r18_val4_s = [0.61, 0.87, 0.69, 0.81, 0.63, 0.56, 0.66]

precision_r34_val1_s = [0.53, 0.79, 0.61, 0.85, 0.60, 0.59, 0.73]
precision_r34_val2_s = [0.61, 0.80, 0.76, 0.88, 0.65, 0.67, 0.70]
precision_r34_val3_s = [0.55, 0.83, 0.66, 0.80, 0.62, 0.57, 0.68]
precision_r34_val4_s = [0.62, 0.90, 0.67, 0.81, 0.61, 0.54, 0.68]

precision_r50_val1_s = [0.55, 0.83, 0.58, 0.84, 0.63, 0.59, 0.75]
precision_r50_val2_s = [0.64, 0.85, 0.80, 0.87, 0.69, 0.72, 0.68]
precision_r50_val3_s = [0.56, 0.84, 0.67, 0.79, 0.67, 0.60, 0.69]
precision_r50_val4_s = [0.66, 0.89, 0.68, 0.81, 0.60, 0.63, 0.68]

precision_r152_val1_s = [0.51, 0.81, 0.52, 0.85, 0.56, 0.51, 0.73]
precision_r152_val2_s = [0.67, 0.84, 0.78, 0.87, 0.68, 0.70, 0.71]
precision_r152_val3_s = [0.59, 0.84, 0.68, 0.79, 0.67, 0.63, 0.70]
precision_r152_val4_s = [0.62, 0.88, 0.73, 0.83, 0.60, 0.60, 0.68]

precision_u152_val1_s = [0.53, 0.75, 0.62, 0.89, 0.60, 0.50, 0.75]
precision_u152_val2_s = [0.70, 0.82, 0.65, 0.90, 0.71, 0.73, 0.71]
precision_u152_val3_s = [0.58, 0.73, 0.64, 0.84, 0.71, 0.63, 0.62]
precision_u152_val4_s = [0.50, 0.86, 0.68, 0.83, 0.55, 0.64, 0.72]

precision_lstm18_val1_s = [0.17, 0.64, 0.27, 0.91, 0.35, 0.35, 0.65]
precision_lstm18_val2_s = [0.75, 0.80, 0.79, 0.95, 0.40, 0.52, 0.77]
precision_lstm18_val3_s = [0.75, 0.78, 0.65, 0.84, 0.57, 0.80, 0.79]
precision_lstm18_val4_s = [0.78, 0.90, 0.64, 0.89, 0.70, 0.79, 0.77]

###############################################################
# Recall
###############################################################

recall_r18_val1_s = [0.66, 0.83, 0.67, 0.78, 0.50, 0.61, 0.63]
recall_r18_val2_s = [0.60, 0.88, 0.70, 0.79, 0.69, 0.80, 0.61]
recall_r18_val3_s = [0.67, 0.78, 0.71, 0.79, 0.57, 0.74, 0.70]
recall_r18_val4_s = [0.63, 0.82, 0.64, 0.85, 0.55, 0.72, 0.67]

recall_r34_val1_s = [0.72, 0.83, 0.65, 0.76, 0.52, 0.63, 0.60]
recall_r34_val2_s = [0.63, 0.89, 0.68, 0.75, 0.71, 0.73, 0.66]
recall_r34_val3_s = [0.71, 0.77, 0.68, 0.78, 0.55, 0.75, 0.67]
recall_r34_val4_s = [0.64, 0.81, 0.67, 0.86, 0.54, 0.80, 0.64]

recall_r50_val1_s = [0.72, 0.82, 0.71, 0.78, 0.52, 0.69, 0.65]
recall_r50_val2_s = [0.68, 0.89, 0.70, 0.81, 0.71, 0.76, 0.71]
recall_r50_val3_s = [0.74, 0.78, 0.70, 0.80, 0.56, 0.73, 0.69]
recall_r50_val4_s = [0.67, 0.83, 0.68, 0.85, 0.53, 0.80, 0.66]

recall_r152_val1_s = [0.73, 0.81, 0.65, 0.72, 0.55, 0.80, 0.61]
recall_r152_val2_s = [0.69, 0.89, 0.70, 0.80, 0.73, 0.74, 0.70]
recall_r152_val3_s = [0.72, 0.78, 0.71, 0.81, 0.57, 0.72, 0.68]
recall_r152_val4_s = [0.67, 0.83, 0.67, 0.85, 0.55, 0.79, 0.66]

recall_u152_val1_s = [0.71, 0.88, 0.60, 0.71, 0.48, 0.78, 0.60]
recall_u152_val2_s = [0.69, 0.91, 0.77, 0.74, 0.69, 0.74, 0.68]
recall_u152_val3_s = [0.64, 0.84, 0.70, 0.72, 0.51, 0.65, 0.68]
recall_u152_val4_s = [0.70, 0.85, 0.65, 0.84, 0.47, 0.76, 0.58]

recall_lstm18_val1_s = [0.48, 0.87, 0.65, 0.52, 0.43, 0.79, 0.61]
recall_lstm18_val2_s = [0.50, 0.94, 0.82, 0.70, 0.77, 0.74, 0.84]
recall_lstm18_val3_s = [0.53, 0.88, 0.72, 0.82, 0.67, 0.69, 0.67]
recall_lstm18_val4_s = [0.68, 0.93, 0.73, 0.90, 0.63, 0.75, 0.65]

###############################################################
# #######################no-scheduler######################## #
###############################################################
# Precision
###############################################################

precision_r18_val1 = [0.55, 0.77, 0.56, 0.83, 0.54, 0.5, 0.67]
precision_r18_val2 = [0.58, 0.78, 0.73, 0.84, 0.64, 0.6, 0.66]
precision_r18_val3 = [0.56, 0.77, 0.65, 0.78, 0.6, 0.59, 0.65]
precision_r18_val4 = [0.53, 0.82, 0.65, 0.8, 0.52, 0.55, 0.62]

precision_r34_val1 = [0.57, 0.73, 0.53, 0.84, 0.54, 0.49, 0.69]
precision_r34_val2 = [0.59, 0.74, 0.75, 0.83, 0.64, 0.62, 0.65]
precision_r34_val3 = [0.54, 0.81, 0.63, 0.76, 0.57, 0.55, 0.63]
precision_r34_val4 = [0.54, 0.81, 0.63, 0.76, 0.57, 0.55, 0.63]

precision_r50_val1 = [0.61, 0.75, 0.53, 0.84, 0.58, 0.49, 0.70]
precision_r50_val2 = [0.62, 0.75, 0.70, 0.85, 0.60, 0.70, 0.65]
precision_r50_val3 = [0.52, 0.77, 0.64, 0.78, 0.59, 0.55, 0.66]
precision_r50_val4 = [0.60, 0.83, 0.66, 0.76, 0.54, 0.59, 0.61]

precision_r152_val1 = [0.55, 0.75, 0.58, 0.83, 0.59, 0.47, 0.69]
precision_r152_val2 = [0.61, 0.78, 0.70, 0.85, 0.64, 0.63, 0.67]
precision_r152_val3 = [0.55, 0.77, 0.67, 0.77, 0.62, 0.61, 0.65]
precision_r152_val4 = [0.51, 0.83, 0.64, 0.80, 0.55, 0.53, 0.58]

precision_u152_val1 = [0.61, 0.76, 0.49, 0.79, 0.56, 0.33, 0.62]
precision_u152_val2 = [0.63, 0.74, 0.64, 0.83, 0.64, 0.54, 0.59]
precision_u152_val3 = [0.59, 0.76, 0.63, 0.76, 0.59, 0.45, 0.58]
precision_u152_val4 = [0.49, 0.83, 0.56, 0.77, 0.50, 0.37, 0.58]

precision_lstm18_val1_cnn_with_s = [0.33, 0.72, 0.21, 0.88, 0.39, 0.38, 0.77]
precision_lstm18_val2_cnn_with_s = [0.78, 0.65, 0.84, 0.94, 0.52, 0.67, 0.83]
precision_lstm18_val3_cnn_with_s = [0.69, 0.71, 0.57, 0.88, 0.62, 0.52, 0.82]
precision_lstm18_val4_cnn_with_s = [0.56, 0.78, 0.62, 0.92, 0.54, 0.69, 0.83]

###############################################################
# Recall
###############################################################

recall_r18_val1 = [0.62, 0.81, 0.62, 0.74, 0.49, 0.55, 0.63]
recall_r18_val2 = [0.57, 0.86, 0.61, 0.73, 0.70, 0.68, 0.67]
recall_r18_val3 = [0.65, 0.77, 0.68, 0.77, 0.52, 0.59, 0.67]
recall_r18_val4 = [0.59, 0.81, 0.60, 0.80, 0.48, 0.66, 0.62]

recall_r34_val1 = [0.63, 0.82, 0.59, 0.71, 0.48, 0.54, 0.61]
recall_r34_val2 = [0.58, 0.86, 0.58, 0.70, 0.66, 0.63, 0.66]
recall_r34_val3 = [0.58, 0.77, 0.62, 0.78, 0.51, 0.62, 0.62]
recall_r34_val4 = [0.58, 0.77, 0.62, 0.78, 0.51, 0.62, 0.62]

recall_r50_val1 = [0.60, 0.84, 0.62, 0.72, 0.54, 0.54, 0.60]
recall_r50_val2 = [0.56, 0.88, 0.58, 0.72, 0.71, 0.61, 0.70]
recall_r50_val3 = [0.65, 0.78, 0.66, 0.74, 0.51, 0.61, 0.63]
recall_r50_val4 = [0.62, 0.81, 0.58, 0.81, 0.47, 0.67, 0.57]

recall_r152_val1 = [0.59, 0.82, 0.59, 0.73, 0.50, 0.59, 0.60]
recall_r152_val2 = [0.58, 0.88, 0.58, 0.74, 0.69, 0.66, 0.67]
recall_r152_val3 = [0.61, 0.78, 0.64, 0.77, 0.53, 0.62, 0.63]
recall_r152_val4 = [0.59, 0.80, 0.59, 0.80, 0.47, 0.70, 0.61]

recall_u152_val1 = [0.61, 0.78, 0.58, 0.72, 0.36, 0.51, 0.67]
recall_u152_val2 = [0.56, 0.85, 0.57, 0.69, 0.55, 0.58, 0.77]
recall_u152_val3 = [0.61, 0.75, 0.63, 0.75, 0.48, 0.53, 0.67]
recall_u152_val4 = [0.60, 0.74, 0.56, 0.79, 0.41, 0.60, 0.70]

recall_lstm18_val1_cnn_with_s = [0.42, 0.85, 0.64, 0.60, 0.42, 0.75, 0.59]
recall_lstm18_val2_cnn_with_s = [0.54, 0.96, 0.61, 0.66, 0.81, 0.63, 0.81]
recall_lstm18_val3_cnn_with_s = [0.65, 0.86, 0.74, 0.74, 0.52, 0.54, 0.71]
recall_lstm18_val4_cnn_with_s = [0.70, 0.95, 0.80, 0.78, 0.63, 0.60, 0.53]

###############################################################
# #########################Accuracy########################## #
# ########################w/scheduler######################## #
###############################################################

accuracy_s = {
    'resnet18': [0.7540051570359725, 0.7989087407724403, 0.7539409951563188, 0.7807262569832403],
    'resnet34': [0.7517648053430275, 0.7846795763346528, 0.7461690885072655, 0.7845393278354172],
    'resnet50': [0.7652280508940271, 0.8176313255590029, 0.7526860413914576, 0.7925645118382548],
    'resnet152': [0.7455298643107748, 0.8138226168824222, 0.7611845002201674, 0.7932074133191451],
    'resnetU152': [0.7453819165574671, 0.8037659141970686, 0.7311316600616469, 0.7861798350625167],
    'LSTM': [0.6365346409096674, 0.7948004707392746, 0.7783795684720387, 0.8473441518134256],
}
###############################################################
# #########################Accuracy########################## #
# #######################w/o scheduler####################### #
###############################################################

accuracy = {
    'resnet18': [0.7257894069408631, 0.7559002888627367, 0.7204817155060435, 0.7439478584729982],
    'resnet34': [0.7107410068901382, 0.7367069648015406, 0.722130169671262, 0.722130169671262],
    'resnet50': [0.7245424187344126, 0.7509361292393282, 0.7140246587406429, 0.7425290414117229],
    'resnet152': [0.7217314114215666, 0.7624906387076067, 0.7199031263760458, 0.7426842245277999],
    'resnetU152': [0.6966225641459187, 0.7250240718947256, 0.6928005284015852, 0.7116254322958233],
    'LSTM': [0.6736272562032379, 0.7480047073927464, 0.7362233328196209, 0.7872217788418906],
}
###############################################################

###############################################################
# ########################Best Accuracy###################### #
# ########################w/scheduler######################## #
###############################################################

best_accuracy_s = {
    'resnet18': [0.758634, 0.798909, 0.756121, 0.785337],
    'resnet34': [0.757450, 0.789408, 0.750903, 0.789971],
    'resnet50': [0.767870, 0.820648, 0.755394, 0.796067],
    'resnet152': [0.763601, 0.816519, 0.762021, 0.796666],
    'resnetU152': [0.771949, 0.817888, 0.757860, 0.793451],
    'LSTM': [0.793930, 0.826575, 0.782585, 0.850403],
}
###############################################################
# #######################Best Accuracy####################### #
# #######################w/o scheduler####################### #
###############################################################

best_accuracy = {
    'resnet18': [0.726867, 0.755900, 0.723542, 0.747938],
    'resnet34': [0.718561, 0.740558, 0.715302, 0.738007],
    'resnet50': [0.733990, 0.758083, 0.721775, 0.750732],
    'resnet152': [0.728030, 0.769595, 0.722127, 0.746586],
    'resnetU152': [0.696073, 0.733904, 0.693923, 0.718254],
    'LSTM': [0.793592, 0.783139, 0.792783, 0.866410],
}
###############################################################
# ###########################Jaccard######################### #
###############################################################

jaccard_r18_s = [7.397934919309667e-05, 7.489594812920622e-05, 7.70645029890018e-05, 7.759757895553658e-05]
jaccard_r34_s = [7.397934919309667e-05, 7.489594812920622e-05, 7.70645029890018e-05, 7.759757895553658e-05]
jaccard_r50_s = [7.397934919309667e-05, 7.489594812920622e-05, 7.70645029890018e-05, 7.759757895553658e-05]
jaccard_r152_s = [7.397934919309667e-05, 7.489594812920622e-05, 7.70645029890018e-05, 7.759757895553658e-05]
jaccard_u152_s = [7.397934919309667e-05, 7.489594812920622e-05, 7.70645029890018e-05, 7.759757895553658e-05]
jaccard_lstm_s = [7.397934919309667e-05, 7.489594812920622e-05, 7.70645029890018e-05, 7.759757895553658e-05]


jaccard_r18 = []
jaccard_r34 = []
jaccard_r50 = []
jaccard_r152 = []
jaccard_u152 = []
jaccard_lstm = []

###############################################################
###############################################################
###############################################################

prerec = {
    'scheduler': {
        'precision': {
            'resnet18': {
                1: precision_r18_val1_s,
                2: precision_r18_val2_s,
                3: precision_r18_val3_s,
                4: precision_r18_val4_s,
            },
            'resnet34': {
                1: precision_r34_val1_s,
                2: precision_r34_val2_s,
                3: precision_r34_val3_s,
                4: precision_r34_val4_s,
            },
            'resnet50': {
                1: precision_r50_val1_s,
                2: precision_r50_val2_s,
                3: precision_r50_val3_s,
                4: precision_r50_val4_s,
            },
            'resnet152': {
                1: precision_r152_val1_s,
                2: precision_r152_val2_s,
                3: precision_r152_val3_s,
                4: precision_r152_val4_s,
            },
            'unsuper152': {
                1: precision_u152_val1_s,
                2: precision_u152_val2_s,
                3: precision_u152_val3_s,
                4: precision_u152_val4_s,
            },
            'lstm18': {
                1: precision_lstm18_val1_cnn_with_s,
                2: precision_lstm18_val2_cnn_with_s,
                3: precision_lstm18_val3_cnn_with_s,
                4: precision_lstm18_val4_cnn_with_s,
            },
        },
        'recall': {
            'resnet18': {
                1: recall_r18_val1_s,
                2: recall_r18_val2_s,
                3: recall_r18_val3_s,
                4: recall_r18_val4_s,
            },
            'resnet34': {
                1: recall_r34_val1_s,
                2: recall_r34_val2_s,
                3: recall_r34_val3_s,
                4: recall_r34_val4_s,
            },
            'resnet50': {
                1: recall_r50_val1_s,
                2: recall_r50_val2_s,
                3: recall_r50_val3_s,
                4: recall_r50_val4_s,
            },
            'resnet152': {
                1: recall_r152_val1_s,
                2: recall_r152_val2_s,
                3: recall_r152_val3_s,
                4: recall_r152_val4_s,
            },
            'unsuper152': {
                1: recall_u152_val1_s,
                2: recall_u152_val2_s,
                3: recall_u152_val3_s,
                4: recall_u152_val4_s,
            },
            'lstm18': {
                1: recall_lstm18_val1_cnn_with_s,
                2: recall_lstm18_val2_cnn_with_s,
                3: recall_lstm18_val3_cnn_with_s,
                4: recall_lstm18_val4_cnn_with_s,
            },
        },
    },
    'no-scheduler': {
        'precision': {
            'resnet18': {
                1: precision_r18_val1,
                2: precision_r18_val2,
                3: precision_r18_val3,
                4: precision_r18_val4,
            },
            'resnet34': {
                1: precision_r34_val1,
                2: precision_r34_val2,
                3: precision_r34_val3,
                4: precision_r34_val4,
            },
            'resnet50': {
                1: precision_r50_val1,
                2: precision_r50_val2,
                3: precision_r50_val3,
                4: precision_r50_val4,
            },
            'resnet152': {
                1: precision_r152_val1,
                2: precision_r152_val2,
                3: precision_r152_val3,
                4: precision_r152_val4,
            },
            'unsuper152': {
                1: precision_u152_val1,
                2: precision_u152_val2,
                3: precision_u152_val3,
                4: precision_u152_val4,
            },
            'lstm18': {
                1: precision_lstm18_val1_cnn_with_s,
                2: precision_lstm18_val2_cnn_with_s,
                3: precision_lstm18_val3_cnn_with_s,
                4: precision_lstm18_val4_cnn_with_s,
            },
        },
        'recall': {
            'resnet18': {
                1: recall_r18_val1,
                2: recall_r18_val2,
                3: recall_r18_val3,
                4: recall_r18_val4,
            },
            'resnet34': {
                1: recall_r34_val1,
                2: recall_r34_val2,
                3: recall_r34_val3,
                4: recall_r34_val4,
            },
            'resnet50': {
                1: recall_r50_val1,
                2: recall_r50_val2,
                3: recall_r50_val3,
                4: recall_r50_val4,
            },
            'resnet152': {
                1: recall_r152_val1,
                2: recall_r152_val2,
                3: recall_r152_val3,
                4: recall_r152_val4,
            },
            'unsuper152': {
                1: recall_u152_val1,
                2: recall_u152_val2,
                3: recall_u152_val3,
                4: recall_u152_val4,
            },
            'lstm18': {
                1: recall_lstm18_val1_cnn_with_s,
                2: recall_lstm18_val2_cnn_with_s,
                3: recall_lstm18_val3_cnn_with_s,
                4: recall_lstm18_val4_cnn_with_s,
            },
        },
    },
}

###############################################################
###############################################################
###############################################################


def latex_it(metric, deviation):
    output = ''
    for value, devi in zip(metric, deviation):
        value = round(value, 2)
        devi = round(devi, 2)
        output += f'{value} $\pm$ {devi} & '
    return output + '\\\\\n'


def display(set1, set2, set3, set4):
    mean_pre = []
    rounded_pre = []
    std_pre = []
    std_pre_rounded = []
    metric = []
    for item_set_1, item_set_2, item_set_3, item_set_4 in zip(set1, set2, set3, set4):
        metric.append([item_set_1, item_set_2, item_set_3, item_set_4])

    for i in metric:
        mean = np.mean(i)
        std = np.std(i)
        mean_pre.append(mean)
        std_pre.append(std)
        rounded_pre.append(mean)
        std_pre_rounded.append(std)
    return rounded_pre, std_pre_rounded


def calculate_precision_scheduler():
    results = []
    display(precision_r18_val1_s, precision_r18_val2_s, precision_r18_val3_s, precision_r18_val4_s)
    display(precision_r34_val1_s, precision_r34_val2_s, precision_r34_val3_s, precision_r34_val4_s)
    display(precision_r50_val1_s, precision_r50_val2_s, precision_r50_val3_s, precision_r50_val4_s)
    display(precision_r152_val1_s, precision_r152_val2_s, precision_r152_val3_s, precision_r152_val4_s)
    display(precision_u152_val1_s, precision_u152_val2_s, precision_u152_val3_s, precision_u152_val4_s)
    results.append(display(precision_lstm18_val1_s, precision_lstm18_val2_s, precision_lstm18_val3_s, precision_lstm18_val4_s))
    return results


def calculate_recall_scheduler():
    results = []
    display(recall_r18_val1_s, recall_r18_val2_s, recall_r18_val3_s, recall_r18_val4_s)
    display(recall_r34_val1_s, recall_r34_val2_s, recall_r34_val3_s, recall_r34_val4_s)
    display(recall_r50_val1_s, recall_r50_val2_s, recall_r50_val3_s, recall_r50_val4_s)
    display(recall_r152_val1_s, recall_r152_val2_s, recall_r152_val3_s, recall_r152_val4_s)
    display(recall_u152_val1_s, recall_u152_val2_s, recall_u152_val3_s, recall_u152_val4_s)
    results.append(display(recall_lstm18_val1_s, recall_lstm18_val2_s, recall_lstm18_val3_s, recall_lstm18_val4_s))
    return results


def calculate_precision():
    results = []
    results.append(display(precision_r18_val1, precision_r18_val2, precision_r18_val3, precision_r18_val4))
    results.append(display(precision_r34_val1, precision_r34_val2, precision_r34_val3, precision_r34_val4))
    results.append(display(precision_r50_val1, precision_r50_val2, precision_r50_val3, precision_r50_val4))
    results.append(display(precision_r152_val1, precision_r152_val2, precision_r152_val3, precision_r152_val4))
    results.append(display(precision_u152_val1, precision_u152_val2, precision_u152_val3, precision_u152_val4))
    results.append(display(precision_lstm18_val1_cnn_with_s, precision_lstm18_val2_cnn_with_s, precision_lstm18_val3_cnn_with_s, precision_lstm18_val4_cnn_with_s))
    return results


def calculate_recall():
    results = []
    results.append(display(recall_r18_val1, recall_r18_val2, recall_r18_val3, recall_r18_val4))
    results.append(display(recall_r34_val1, recall_r34_val2, recall_r34_val3, recall_r34_val4))
    results.append(display(recall_r50_val1, recall_r50_val2, recall_r50_val3, recall_r50_val4))
    results.append(display(recall_r152_val1, recall_r152_val2, recall_r152_val3, recall_r152_val4))
    results.append(display(recall_u152_val1, recall_u152_val2, recall_u152_val3, recall_u152_val4))
    results.append(display(recall_lstm18_val1_cnn_with_s, recall_lstm18_val2_cnn_with_s, recall_lstm18_val3_cnn_with_s, recall_lstm18_val4_cnn_with_s))
    return results


def metrics(scheduler=True):
    pre = {}
    rec = {}
    if scheduler:
        print('Results with scheduler')
        pre = calculate_precision_scheduler()
        rec = calculate_recall_scheduler()
    else:
        print('Results without scheduler')
        pre = calculate_precision()
        rec = calculate_recall()
    precision = ''
    recall = ''
    for tupl1, tupl2 in zip(pre, rec):
        precision += latex_it(tupl1[0], tupl1[1])
        recall += latex_it(tupl2[0], tupl2[1])
    print(precision)
    print('\n\n')
    print(recall)


def calculate_accuracy(collection):
    for net in collection:
        print(net)
        average = 0
        for result in collection[net]:
            average += result
            print(round(result, 2))
        average /= 4
        rounded = average
        print('Average: ', round(rounded, 2))


def average_metric(collection, metric='precision'):
    result = 0

    sched_prec = []
    prec = []
    sched_rec = []
    rec = []
    for scheduler in collection:
        for metric in collection[scheduler]:
            for net in collection[scheduler][metric]:
                for subset in collection[scheduler][metric][net]:
                    precent = sum(collection[scheduler][metric][net][subset]) / len(collection[scheduler][metric][net][subset])
                    result += precent
                result /= len(collection[scheduler][metric][net])
                prerec[scheduler][metric][net] = round(result, 2)
                result = 0
    print(prerec)
    for scheduler in prerec:
        print(scheduler)
        for metric in prerec[scheduler]:
            print(metric)
            for net in prerec[scheduler][metric]:
                print(net + ' & ', end=' ')
            print('')
            for net in prerec[scheduler][metric]:
                print(str(int(prerec[scheduler][metric][net] * 100)) + ' & ', end='  ')
            print('')


def matrix_accuracy(matrix):
    nummerator = 0
    denominator = 0
    for i in range(len(matrix)):
        nummerator += matrix[i][i]
        for j in range(len(matrix[i])):
            denominator += matrix[i][j]
    result = nummerator / denominator
    # print(result)
    return result


m = [
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
]


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
    # jaccard_idx




# TODO: Confusion matrix der normierten auf addieren
# average_metric(prerec)
# metrics(scheduler=True)
# calculate_accuracy(accuracy)
print(jaccard_score(m))
