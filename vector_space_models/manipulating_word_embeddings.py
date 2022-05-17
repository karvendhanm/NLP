import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

word_embeddings = pickle.load(open('./data/word_embeddings_subset.p', 'rb'))
len(word_embeddings)

def vec(w):
    return word_embeddings[w]

words = ['oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country',
        'continent', 'petroleum', 'joyful']

bag2d = np.array([vec(word) for word in words])

fig, ax = plt.subplots(figsize=(10, 10))

col1 = 3
col2 = 2

for word in bag2d:
    ax.arrow(0, 0, word[col1], word[col2], head_width=0.005, head_length=0.005, fc='r', ec='r', width = 1e-5)

ax.scatter(bag2d[:, col1], bag2d[:, col2])

for i in range(0, len(words)):
    ax.annotate(words[i], (bag2d[i, col1], bag2d[i, col2]))

plt.show()


second_words = ['sad', 'happy', 'town', 'village']
word_vectors = np.array([vec(word) for word in second_words])

fig, ax = plt.subplots(figsize=(10, 10))

col1 = 3
col2 = 2

for i in range(0, len(word_vectors)):
    ax.arrow(0, 0, word_vectors[i, col1], word_vectors[i, col2], head_width=0.005, head_length=0.005, fc='r', ec='r', width = 1e-5)

# arrow from village to town
village_vector = np.array(vec('village'))
town_vector = np.array(vec('town'))

bridge_vector = town_vector - village_vector
ax.arrow(village_vector[col1], village_vector[col2], bridge_vector[col1], bridge_vector[col2], head_width=0.005, head_length=0.005, fc='b', ec='b', width = 1e-5)

# arrow from sad to happy
sad_vector = np.array(vec('sad'))
happy_vector = np.array(vec('happy'))

bridge_vector = happy_vector - sad_vector
ax.arrow(sad_vector[col1], sad_vector[col2], bridge_vector[col1], bridge_vector[col2], head_width=0.005, head_length=0.005, fc='b', ec='b', width = 1e-5)


ax.scatter(word_vectors[:, col1], word_vectors[:, col2])

for i in range(0, len(word_vectors)):
    ax.annotate(second_words[i], (word_vectors[i, col1], word_vectors[i, col2]))

plt.show()

# Linear algebra on word embeddings
print(np.linalg.norm(vec('town')))
print(np.linalg.norm(vec('sad')))

# Predicting capitals
capital_vector = vec('France') - vec('Paris')
country_vector = vec('Madrid') + capital_vector

print(country_vector[:5])
vec('Spain')[:5]

diff = country_vector - vec('Spain')
diff[:5]

keys = word_embeddings.keys()
data = []
for key in keys:
    data.append(word_embeddings[key])

embedding = pd.DataFrame(data, index=keys)
def find_closest_word(v, k=1):

    diff = embedding.values - v
    delta = np.sum(diff*diff, axis=1)
    i = np.argmin(delta)
    return embedding.iloc[i].name

find_closest_word(country_vector)
find_closest_word(vec('Berlin') + capital_vector)
find_closest_word(vec('Beijing') + capital_vector)
print(find_closest_word(vec('Lisbon') + capital_vector))

find_closest_word((vec('Italy') - vec('Rome')) + vec('Madrid'))
doc = "Spain petroleum city king"

vdoc = [vec(x) for x in doc.split(" ")]
doc2vec = np.sum(vdoc, axis=0)
doc2vec

find_closest_word(doc2vec)





















