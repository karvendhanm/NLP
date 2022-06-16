# Edit distance

# Here we will implement models that are 1 and 2 edit distance away.
# Data preprocessing

import numpy as np
import pandas as pd

from autocorrect.utils import process_data, get_count

word_l = process_data('./data/shakespeare.txt')
word_count_dict = get_count(word_l)

# String manipulations

