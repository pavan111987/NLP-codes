from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys
import csv

top_words_considered=1000
max_sentence_length=1000

validation_split=0.2
file='q&a file.csv'
word_vector_file='glove.6B.300d.txt'


print ("Loading word vectors.....")
embeddings_index = {}
f = open(word_vector_file)
for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs



f.close()
print ("Word Vectors Loaded!!")
print('Found %s word vectors.' % len(embeddings_index))

word_vector_length=len(embeddings_index.get('dog'))


print ("Loading the data file....")
f=open(file)
f=csv.reader(f)
texts=[]
c=[]
for i in f:
  texts.append(i[0])
  c.append(i[1])


labels=[]
for i in c:
  labels.append((np.where(i==np.unique(c)))[0][0]) 
     

labels_index={}
for i in np.unique(c):
  for j in np.arange(len(c)):
    if (i==c[j]):
      labels_index[i]=labels[j]


print ("Datafile loaded!!")


c=[]
MAX_SEQUENCE_LENGTH = max_sentence_length
MAX_NB_WORDS = top_words_considered
EMBEDDING_DIM = word_vector_length
VALIDATION_SPLIT = validation_split



tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
  if i >= MAX_NB_WORDS:
    continue
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector




# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=3, batch_size=128,verbose=1)




sent='i have ear pain'

sequences = tokenizer.texts_to_sequences(sent)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


for lab,i in labels_index.items():
    if (i==np.argmax(model.predict(x_train[1001].reshape(1,1000)))):
       print (lab)




