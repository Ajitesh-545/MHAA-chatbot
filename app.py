import nltk
from nltk import probability 
from nltk.stem import WordNetLemmatizer 
import numpy as np
import random
import json
import pickle
from tensorflow.keras.models import load_model
from tensorflow.python.framework.op_def_registry import get

from flask import Flask, render_template, request

lemmatizer=WordNetLemmatizer()

#loading of data and model
intents=json.loads(open('intents.json').read())
words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=load_model('chatbot_model.h5')

#tokenize the input sentences 
def clean_up_sentence(sentence):
    sentence_words=nltk.word_tokenize(sentence.lower())
    sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#bag of word creation basically in array form which will be fed into the model
def bag_of_words(sentence):
    sentence_words=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow=bag_of_words(sentence)
    results=model.predict(np.array([bow]))[0]
    threshold_error=0.25
    final_result=[[i,res]for i, res in enumerate(results) if res>threshold_error]
    
    final_result.sort(key=lambda x:x[1], reverse=True)
    return_list=[]
    for r in final_result:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list,intents_json):
    tag=intents_list[0]['intent']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result=random.choice(i['responses']) 
            break
    return result
    
app = Flask(__name__)
@app.route("/")
def index():
    return render_template("/index.html")



@app.route("/get")
def get_bot_response():
    userText=request.args.get('msg')
    ints=predict_class(userText.lower())
    res=get_response(ints,intents)
    return res

    
if __name__ == "__main__":
    app.run(debug=True)
    