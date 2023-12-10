import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
import scipy.stats as st
import numpy as np
import copy as cp
import random as rd
import torch
import torch.nn as nn
import gc
import time
import math
from collections import OrderedDict

seed = 123
np.random.seed(seed)
rd.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = "cuda"



