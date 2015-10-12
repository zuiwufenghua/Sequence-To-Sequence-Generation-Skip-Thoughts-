# -*- coding: utf-8 -*-
# from __future__ import print_function
from keras.models import Sequential, slice_X
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Dropout, Dense
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
import numpy as np
import random, sys
from keras.callbacks import EarlyStopping
import re
import itertools
import shutil
from textblob import TextBlob
from textblob import WordList
import time
from nltk.corpus import wordnet as wn
import nltk
import random
from keras.preprocessing import sequence
from keras.datasets import imdb
import sys
import string


"""
The Idea is to first read all of the sentences of a book. Afterwards, the model should be able to write a sentence after it has been fed a previous sentence. 

In other words, if you feed the model "Stars are very bright in the night sky.", the model should write something like, "They shine all across the universe."


"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.


"""
'''Start of indexing your words here'''


number_of_iterations = 400 #max number of iterations/epochs you want to run for. 
start_validating_at_iteration = 0 #which iteration to start checking for overfitting
patience = 10 #number of times where validation loss has to rise before quitting algorithm
hidden_variables = 256 #number of hidden variables in all layers, keep this betwen 256 and 512
dropout = 0.5 #the dropout between each layer -- recommendation is 0.5
embedding_size = 256 #this should be 128 or 256 or 512
hidden_size = 256
batch_size = 1024
decoder_layers = 3 #number of lstm layers for decoding -- try 4, maybe eventually up to 8. 
time_steps = 50 #number to pad the vector by. Should be the max number of words in a sentence. 

vocabulary_size = 10000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"


'''*************************************PROCESSING ASTRONOMY BOOK TEXT**************************'''
# Read the data and append SENTENCE_START and SENTENCE_END tokens
print ("Reading text file...")
text = open('allarticles/AstronomyBookWikiAll_shortened.txt').read()

'''here you add code to filter out wikipedia references and more'''
text = re.sub('(Further reading|Bibliography|Books|External links|See also|Sources|References|Cited texts)\n\n(- .+\n|  .+\n)*', '',text)
text = re.sub('(Further reading|Bibliography|Books|External links|See also|Sources|References|Cited texts)\n(- .+\n|  .+\n)*', '',text)
text = re.sub('(Footnotes|References|Notes)\n\n\[\d*\].(.+\n|.+\n)*', '',text)
text = re.sub('\[\d*\].(.+\n|.+\n)*', '',text)
text = re.sub('Main article:.+\n', '',text)
text = re.sub('(²|³|¹|⁴|⁵|⁶|⁷|⁸|⁹|⁰)', '',text)
text = re.sub('(     |    |   |  )', '',text)
text = re.sub('(\n\n\n\n|\n\n\n\n\n)', '\n\n',text)
text = re.sub('\n\n\n', '\n\n',text)
text = re.sub('\n\n', ' ~ ',text)
text = re.sub('\n', ' ',text)

print ('-'*50)

sentences = (nltk.sent_tokenize(text.decode('utf-8').lower()))
# Append SENTENCE_START and SENTENCE_END
sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
#you can adjust the sentence start and end
print ("Parsed %d sentences." % (len(sentences)))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# print (tokenized_sentences) #this is not a list of lists

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print ("Found %d unique words tokens." % len(word_freq.items()))
# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1) #this is a list at this point of words in most frequently
index_to_word = ['padzero'] #indicates that you have a word for zero's



for x in vocab:
    index_to_word.append(x[0]) #this is based upon frequency of word used -- nice

index_to_word.append(unknown_token) #this is making the word for all unknown words here
print '-----------------------------------------------------------------------------------------------'
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
print
print

print ("Using vocabulary size %d." % vocabulary_size)
print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
 
# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
 
print ("\nExample sentence: '%s'" % sentences[0])
print ("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])


'''the x train should be the revers order of the first sentence

the y train should be the next sentence that comes after x sentence'''


X_train_array = ([[word_to_index[w] for w in sent[::-1]] for sent in tokenized_sentences[:-1]])
#the ::-1 is to reverse the list
X_train = sequence.pad_sequences(X_train_array, maxlen=time_steps)
y_train_array = ([[word_to_index[w] for w in sent] for sent in tokenized_sentences[1:]])
y_train = sequence.pad_sequences(y_train_array, maxlen=time_steps)

print '* '*50

'''------------------------------KERAS MODEL DESIGN HERE ----------------------------'''


print('Building model...')

increased_vocab = (int(len(vocab))+2+1)

print ('the increased vocab is')
print increased_vocab

model = Sequential()

#its important to realize that this embedding layer adds an extra dimension. 
model.add(Embedding(increased_vocab, embedding_size, input_length=time_steps, mask_zero=True))

#the masking will block out the zeros from padding -- really useful for recurrent layers

'''the input of the lstm is a 3d shape, the output is either a 2d or 3d shape. 

If you set the return_sequence to true, it will be a 3d shape


The below layer is your encoding layer'''

model.add(LSTM(output_dim=embedding_size))
# model.add(Dense(hidden_size)) #seen this in other models -- these two might be useful to test
# model.add(Activation('relu')

# '''this layer is the repeat vector, and somehow its useful, but i'm not sure why'''
model.add(RepeatVector(time_steps))

'''DECODER STARTS HERE'''

for z in xrange(decoder_layers-1):
    model.add(LSTM(output_dim=embedding_size, return_sequences=True))
    model.add(Dropout(dropout))


model.add(LSTM(output_dim=embedding_size, return_sequences=False))
model.add(Dropout(dropout))
    
'''their digits is the equivalent of how many words you want total in a sentence'''
''' something really important here is the decoder can be multiple lstms. This is where you would stack your lstms and apply dropouts!'''

model.add(Dense(output_dim=time_steps)) 

#stateful rnn's might be super useful here but i don't know 

model.add(Activation('time_distributed_softmax')) #this is good
print ('you are starting to compile')
model.compile(loss='categorical_crossentropy', optimizer='adam') #these settings are good
print ('you just finished compiling')


'''*************************Advanced Modifications For Much Later On in Development************'''

early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
def sample(a, temperature=1.0): #i want to use this in the prediction output but I'm not quite sure how to at this point. Not a priority right now. 
    a = np.log(a)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1,a,1))

'''**********************************Train the model*******************************'''

for iteration in range(1, number_of_iterations):
    print
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train, verbose=1, validation_split=0.2, callbacks=[early_stopping], nb_epoch=1, batch_size = batch_size, show_accuracy=True)

    random_sentence = random.choice(tokenized_sentences) #this is a list of words -- shoot
    print 'below is random_sentence'
    print random_sentence
    print
    reverse_random_sentence = random_sentence[::-1]
    copy_reverse_random_sentence = reverse_random_sentence
    print
    print('The starter sentence is: ')
    print
    for word in random_sentence:
        sys.stdout.write(word)
        sys.stdout.write(' ')

    # for diversity in [0.2, 0.4, 0.6, 0.75]:
    for diversity in [0.75]:
        
            print
            print('----- diversity:', diversity)

            #now convert the sentence you've chosen back into words 
            reverse_random_sentence = copy_reverse_random_sentence
            translated_reverse_random_sentence = []
            for eachword2 in reverse_random_sentence:
                translated_reverse_random_sentence.append(index_to_word.index(eachword2))


            for iteration in range(2):
                    
                #for the prediction model, you need to input a sentence in reverse order.
                #It should be a list of numbers that are represented by the words. 
                
                # translated_reverse_random_sentence_padded = sequence.pad_sequences(np.array(translated_reverse_random_sentence), maxlen=time_steps)
                sampletry = X_train[np.random.randint(10,size=1),:]

                #okay so the prediction function takes 2d input, because you gave it 2d input. Grrr....
                #but I notice it yields a one dimensional output -- i think this i because of the dense layer at the bottom? 

                preds = model.predict(sampletry,batch_size=batch_size,verbose=0)[0]
                print 
                print('preds is shown below')
                print (preds)   
            #     for eachword1 in preds:
            #         sys.stdout.write((index_to_word[eachword1]))
            #         sys.stdout.write(' ')
            #     translated_reverse_random_sentence = preds[::-1]
            # sys.stdout.flush()
            print
