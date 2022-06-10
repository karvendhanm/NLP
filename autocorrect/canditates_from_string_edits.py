word = 'dearz'

# Splits - find all the ways you can split a word into 2 parts.

splits_a = []
for i in range(len(word) + 1):
    splits_a.append([word[:i], word[i:]])

# for i in splits_a:
#     print(i)

splits_b = [[word[:i], word[i:]] for i in range(len(word)+1)]
for i in splits_b:
    print(i)

# Delete Edit
# delete a letter from each string in the splits list.
# what this does is effectively delete each possible letter from the original word being edited.

splits = splits_a
deletes = []

for L, R in splits:
    if R:
        print(L + R[1:], ' <--delete ', R[0])

deletes = [L + R[1:] for L, R in splits]

# We now have a list of candidate strings created after performing a delete edit in deletes list.
vocab = ['dean', 'deer', 'dear', 'fries', 'and', 'coke']
edits = list(deletes)

candidates=[]
vocab_set = set(vocab)
edits_set = set(edits)
candidates = vocab_set.intersection(edits_set)
print('candidate words : ', candidates)



