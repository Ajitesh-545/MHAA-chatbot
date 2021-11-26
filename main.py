import nltk 
from nltk.stem import WordNetLemmatizer 
import numpy as np
import random
import json
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.backend import dropout


lemmatizer=WordNetLemmatizer()

intents=json.loads(open('intents.json').read())

words=[]
documents=[]
classes=[]
ignore_letters=['!','?',',','.']

for intents in intents['intents']:
    for patterns in intents['patterns']:
        words_list=nltk.word_tokenize((patterns.lower()))
        words.extend(words_list)
        documents.append((words_list,intents['tag']))
        if intents['tag'] not in classes:
            classes.append(intents['tag'])

#stemming of words to root word
words=[lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words=sorted((set(words)))
classes=sorted(set(classes))


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#bag of words
training=[]
out_empty=[0]*len(classes)

for document in documents:
    bag=[]
    word_patterns=document[0]
    word_patterns=[lemmatizer.lemmatize(word) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    out_row=list(out_empty)
    out_row[classes.index(document[1])]=1
    training.append([bag,out_row])

#Splitting data 
random.shuffle(training)
training=np.array(training)

train_x=list(training[:,0])
train_y=list(training[:,1])

#Model building and training 
model=Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

hist=model.fit(np.array(train_x), np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('chatbot_model.h5',hist)
