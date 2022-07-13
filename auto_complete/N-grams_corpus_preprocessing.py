import nltk
import re

nltk.download('punkt')

# text preprocessing steps

# lowercase
corpus = "Learning% makes 'me' happy. I am happy be-cause I am learning! :)"
corpus = corpus.lower()

# remove special characters
corpus = re.sub(r"[^a-zA-Z.?! ]+", r"", corpus)
print(corpus)

# text splitting
# split text by a delimiter to array
input_date="Sat May  9 07:33:35 CEST 2020"

# date parts
date_parts = input_date.split(" ")
print(f"date parts = {date_parts}")

time_parts = date_parts[4].split(":")
print(f'time parts = {time_parts}')

# sentence tokenizing:
# tokenize the sentence into an array of words.
sentence = 'i am happy because i am learning.'
tokenized_sentence = nltk.word_tokenize(sentence)
print(f'{sentence} -> {tokenized_sentence}')

word_lengths = [(word, len(word)) for word in tokenized_sentence]
print(f'the length of words is: \n{word_lengths}')


# N-grams
# sentence to n-gram
def sentence_to_trigram(tokenized_sentence, N = 3):
    '''
    prints all the trigrams in the given tokenized sentence.

    :param tokenized_sentence: the words list.
    :return: No output
    '''

    for i in range(len(tokenized_sentence) - N + 1):
        print(tokenized_sentence[i:i+N])


sentence_to_trigram(tokenized_sentence)

# prefix of an n-gram.
# For n-grams, we must prepend n-1 of characters at the begining of the sentence.
# when working with trigrams, you need to prepend 2 <s> and append one </s>

n = 3  # tri-gram
tokenized_sentence = ['i', 'am', 'happy', 'because', 'i', 'am', 'learning', '.']
tokenized_sentence = ['<s>'] * (n-1) + tokenized_sentence + ['<e>']
print(tokenized_sentence)




