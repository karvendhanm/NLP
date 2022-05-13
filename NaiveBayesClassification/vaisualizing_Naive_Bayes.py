import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from NaiveBayesClassification.utils import confidence_ellipse

data = pd.read_csv('./data/bayes_features.csv')
#### redundant #### STARTS HERE
# index = data.index
# for sentiment in data.sentiment.unique():
#     ix = index[data.sentiment == sentiment]
#     ax.scatter(data.iloc[ix].positive, data.iloc[ix].negative, s=0.1, c=colors[int(sentiment)],
#                marker="*", label=sentiments[int(sentiment)])
#
# ax.legend(loc='best')
# # Custom limits for this chart
# plt.xlim(-250,0)
# plt.ylim(-250,0)
#
# plt.xlabel("Positive") # x-axis label
# plt.ylabel("Negative") # y-axis label
# plt.show()
#### redundant #### ENDS HERE

# plotting confidence ellipse
fig, ax = plt.subplots(figsize=(8, 8))

colors = ['red', 'green']
sentiments = ['negative', 'positive']
index = data.index

for sentiment in data.sentiment.unique():
    ix = index[data.sentiment == sentiment]
    ax.scatter(data.iloc[ix].positive, data.iloc[ix].negative, c=colors[int(sentiment)],
               label =sentiments[int(sentiment)], s=0.1, marker="*")

plt.xlim(-200, 40)
plt.ylim(-200, 40)

plt.xlabel('positive')
plt.ylabel('negative')

data_pos = data.loc[data.sentiment == 1, :]
data_neg = data.loc[data.sentiment == 0, :]

#print confidence ellipse on 2d
confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=2, edgecolor='black', label=r'$2\sigma$')
confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=2, edgecolor='orange')

# Print confidence ellipses of 3 std
confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=3, edgecolor='black', linestyle=':', label=r'$3\sigma$')
confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=3, edgecolor='orange', linestyle=':')

ax.legend(loc='lower right')
plt.show()

data2 = data.copy(deep=True)
# The following 2 lines only modify the entries in the data frame where sentiment == 1
data2.negative[data.sentiment == 1] = data2.negative * 1.5 + 50 # Modify the negative attribute
data2.positive[data.sentiment == 1] = data2.positive / 1.5 - 50 # Modify the positive attribute


# Plot the samples using columns 1 and 2 of the matrix
fig, ax = plt.subplots(figsize = (8, 8))

colors = ['red', 'green'] # Define a color palete
sentiments = ['negative', 'positive']
index = data2.index

# Color base on sentiment
for sentiment in data2.sentiment.unique():
    ix = index[data2.sentiment == sentiment]
    ax.scatter(data2.iloc[ix].positive, data2.iloc[ix].negative, c=colors[int(sentiment)], s=0.1, marker='*', label=sentiments[int(sentiment)])

#ax.scatter(data2.positive, data2.negative, c=[colors[int(k)] for k in data2.sentiment], s = 0.1, marker='*')  # Plot a dot for tweet
# Custom limits for this chart
plt.xlim(-200,40)
plt.ylim(-200,40)

plt.xlabel("Positive") # x-axis label
plt.ylabel("Negative") # y-axis label

data_pos = data2[data2.sentiment == 1] # Filter only the positive samples
data_neg = data[data2.sentiment == 0] # Filter only the negative samples

# Print confidence ellipses of 2 std
confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=2, edgecolor='black', label=r'$2\sigma$' )
confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=2, edgecolor='orange')

# Print confidence ellipses of 3 std
confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=3, edgecolor='black', linestyle=':', label=r'$3\sigma$')
confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=3, edgecolor='orange', linestyle=':')
ax.legend(loc='lower right')

plt.show()















