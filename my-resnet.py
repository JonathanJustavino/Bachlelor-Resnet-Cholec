import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

import os
import time
import copy
import sys
import datetime

from cholec80 import Cholec80


class MyResnet