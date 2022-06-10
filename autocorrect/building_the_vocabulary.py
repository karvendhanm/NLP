import re
from collections import Counter
import matplotlib.pyplot as plt

# tiny corpus of text
text = 'red pink pink blue blue yellow ORANGE BLUE BLUE PINK'
print(text)
print('string length: ', len(text))

# Preprocessing
# convert all the letters to lower case
text_lowercase = text.lower()
print(text_lowercase)
print('string length : ', len(text_lowercase))

regex_pattern = r'\w+'
words = re.findall(regex_pattern, text_lowercase)
print(words)
print('count : ', len(words))

# Create vocabulary
vocab = set(words)
print(vocab)
print("count : ", len(vocab))

# Add information with word counts
counts_a = dict()
for w in words:
    counts_a[w] = counts_a.get(w, 0) + 1
print(counts_a)
print('count : ', len(counts_a))

# create vocab including word count using collections.Counter
counts_b = dict()
counts_b = Counter(words)
print(counts_b)
print('count : ', len(counts_b))

# barchart of sorted word counts
d = {'blue': counts_b['blue'], 'pink': counts_b['pink'], 'red': counts_b['red'], 'yellow': counts_b['yellow'], 'orange': counts_b['orange']}
plt.bar(range(len(d)), list(d.values()), align='center', color=d.keys())
_ = plt.xticks(range(len(d)), list(d.keys()))
plt.show()



