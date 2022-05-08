# confidence ellipse is a tool to represent Naive Bayes visually.

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv('./data/bayes_features.csv')

fig, ax = plt.subplots(figsize=(8,8))
colors = ['red', 'green']
sentiments = ['negative', 'positive']

index = data.index
for sentiment in data.sentiment.unique():
    ix = index[data.sentiment == sentiment]
    ax.scatter(data.iloc[ix].positive, data.iloc[ix].negative, c=colors[int(sentiment)], s=0.1, marker='*', label=sentiments[int(sentiment)])

plt.show()