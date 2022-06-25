import pandas as pd
from collections import defaultdict
import math
import numpy as np

with open('./data/WSJ_02-21.pos', 'r') as fh:
    data = fh.readlines()

for line in data:
    print(line)
