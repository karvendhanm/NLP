# Edit distance

# Here we will implement models that are 1 and 2 edit distance away.
# Data preprocessing

import numpy as np
import pandas as pd

from autocorrect.utils import *

word_l = process_data('./data/shakespeare.txt')
vocab = set(word_l)

word_count_dict = get_count(word_l)
probs = get_probs(word_count_dict)
delete_word_l = delete_letter(word="cans", verbose=True)
switch_word_l = switch_letter(word="eta", verbose=True)
replace_l = replace_letter(word='can', verbose=True)
insert_l = insert_letter('at', True)

# Combining the edits.

