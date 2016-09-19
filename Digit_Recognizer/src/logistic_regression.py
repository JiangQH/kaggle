# this file use the logistic regression to do the classify job
# there can be preprocessing jobs, like feature-scaling, pca and so on

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

train = pd.read_csv('../dataSet/train.csv')
test = pd.read_csv('../dataSet/test.csv')