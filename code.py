import random
import numpy as np
import csv
from gensim.models import Word2Vec
import theano
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

# this function returns one hot vector
def one_hot(x):
	y = np.zeros((3501))
	if x in output.keys():
		y[output[x]]=1
	else:
		y[3500]=1
	return y

# preprocessing
input_data=[]
output_data=[]
words=[]
with open('consolidated.csv') as csvfile:
	reader=csv.DictReader(csvfile)
	for row in reader:
		if int(row['valid'])==1:
			input_data.append(row['tweet'])
			output_data.append(row['corrected'])
			words.append([row['tweet'],row['corrected']])

input_data2=[]
for y in input_data:
	input_data2.append(text_to_word_sequence(y))

output_data2=[]
for i in output_data:
	output_data2.append(text_to_word_sequence(i))

total=[]
for x in output_data2:
	for y in x:
		total.append(y)
s = set(total)

a = []
for i in s:
	c = total.count(i)
	x = [c,i]
	a.append(x)

a.sort(reverse=True)

output = {}
for i in range(0,3500):
	output[a[i][1]]=i;

output['__unk__']=3500
data=random.sample(words,12000)
data_test=data[:int(0.0400*12000)]
data_train=data[int(0.0400*12000):]
data_train_corrected=[]
data_train_tweets=[]
data_test_corrected=[]
data_test_tweets=[]

for x in data_test:
	data_test_corrected.append(text_to_word_sequence(x[1]))
	data_test_tweets.append(text_to_word_sequence(x[0]))

for x in data_train:
	data_train_corrected.append(text_to_word_sequence(x[1]))
	data_train_tweets.append(text_to_word_sequence(x[0]))

one_hots=[]
lenk=42
for x in data_train_corrected:
	temp=[]	
	for y in x:
		temp.append(np.asarray(one_hot(y)))
	k=lenk-len(temp)
	for i in range(k):
		temp.append(np.zeros((3501)))
	one_hots.append(np.asarray(temp))

one_hots=np.asarray(one_hots)

one_hots_test=[]
for x in data_test_corrected:
	temp=[]	
	for y in x:
		temp.append(np.asarray(one_hot(y)))
	k=lenk-len(temp)
	for i in range(k):
		temp.append(np.zeros((3501)))
	one_hots_test.append(np.asarray(temp))

one_hots_test=np.asarray(one_hots_test)

model = Word2Vec(input_data2, min_count=1, workers=2,size=100)

inputs=[]
for x in data_train_tweets:
	temp=[]
	for y in x:
		temp.append(model.wv[y])
	k=lenk-len(temp)
	for i in range(k):
		temp.append(np.zeros((100)))
	inputs.append(np.asarray(temp))

inputs=np.asarray(inputs)

inputs_test=[]
for x in data_test_tweets:
	temp=[]
	for y in x:
		temp.append(model.wv[y])
	k=lenk-len(temp)
	for i in range(k):
		temp.append(np.zeros((100)))
	inputs_test.append(np.asarray(temp))

inputs_test=np.asarray(inputs_test)

# training
n_features = 100
n_timesteps_in = 42
n_timesteps_out = 2
model = Sequential()
model.add(LSTM(150, input_shape=(n_timesteps_in, n_features)))
model.add(RepeatVector(n_timesteps_in))
model.add(LSTM(150, return_sequences=True))
model.add(TimeDistributed(Dense(3501, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['acc'])
model.fit(inputs, one_hots, batch_size=80, validation_data=(inputs_test,one_hots_test),epochs=1)
