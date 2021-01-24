from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd
import transformers as ppb
import numpy as np
import torch 
import tweepy 
from tweepy.auth import OAuthHandler
import json
import pyrebase
from datetime import datetime

# Global variables
tweet_left_list = []
tweet_right_list = []

# Firebase config
config = {
    "apiKey": "AIzaSyB2c1Vd9lYzfbH4HMyM6wKAKqLRippKPes",
    "authDomain": "twitterbutunbiased.firebaseapp.com",
    "databaseURL": "https://twitterbutunbiased-default-rtdb.firebaseio.com/",
    "projectId": "twitterbutunbiased",
    "storageBucket": "twitterbutunbiased.appspot.com",
    "messagingSenderId": "509399748406",
    "appId": "1:509399748406:web:92b4b3e0ac11c2c32ced97",
    "measurementId": "G-RJ283WNDDM"
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()
FILE_NAME = 'last_saved_data_number.txt'

def retrieve_last_seen_id(file_name):
    f_read = open(file_name, 'r')
    last_seen_id = int(f_read.read().strip())
    f_read.close()
    return last_seen_id

def store_last_seen_id(last_seen_id, file_name):
    f_write = open(file_name, 'w')
    f_write.write(str(last_seen_id))
    f_write.close()
    return

### GET TWEETS USING HASHTAG ###
with open("twitter_credentials.json", "r") as file:
    creds = json.load(file)
 
consumer_key = creds["CONSUMER_KEY"]
consumer_secret = creds["CONSUMER_SECRET"]
access_key = creds["ACCESS_TOKEN"]
access_secret = creds["ACCESS_SECRET"]

# Authorization to consumer key and consumer secret 
auth = OAuthHandler(consumer_key, consumer_secret) 

# Access to user's access key and access secret 
auth.set_access_token(access_key, access_secret) 

# Calling api 
api = tweepy.API(auth) 

# get tweets using hashtag as input for tweepy
def get_tweets_hashtag(search_words, date_input, popularity_input, verification_input):
    tweet_list = []
    tweet_link_list = []

    if verification_input:
        tweets = tweepy.Cursor(api.search,
                q=search_words,
                lang="en",
                result_type="mixed").items(20)

        for tweet in tweets:
            if tweet.in_reply_to_status_id is None and tweet.__dict__['user'].__dict__['_json']['verified']:
                tweet_list.append(tweet.text)
                tweet_link_list.append("https://twitter.com/" + tweet.user.screen_name + "/status/" + str(tweet.id) + "?ref_src=twsrc%5Etfw")        
        
        return tweet_list, tweet_link_list       
    
    if date_input:
        since = date_input[:10]
        since_list = since.split("/")
        since = "{}-{}-{}".format(since_list[2], since_list[0], since_list[1])

        until = date_input[date_input.find("-") + 2:]
        until_list = until.split("/")
        until = "{}-{}-{}".format(until_list[2], until_list[0], until_list[1])

        tweets = tweepy.Cursor(api.search,
                q=search_words,
                lang="en",
                since=since,
                until=until,
                result_type="recent").items(20)
    
    elif popularity_input:
        tweets = tweepy.Cursor(api.search,
                q=search_words,
                lang="en",
                result_type="popular").items(20)
    
    else:
        tweets = tweepy.Cursor(api.search,
                q=search_words,
                lang="en",
                result_type="mixed").items(20)

    for tweet in tweets:
        if tweet.in_reply_to_status_id is None:
            tweet_list.append(tweet.text)
            tweet_link_list.append("https://twitter.com/" + tweet.user.screen_name + "/status/" + str(tweet.id) + "?ref_src=twsrc%5Etfw")
    
    return tweet_list, tweet_link_list


### INIT MODEL ###
Pkl_Filename = "Pickle_BERT_Model.pkl"
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'bert_files')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

with open(Pkl_Filename, 'rb') as file:  
    lr_clf = pickle.load(file)

### FEATURIZE LIST FROM TWITTER ###
def predict(tweet_list):
    predicted_list = []

    df = pd.DataFrame(tweet_list)
    df.columns = [1]

    df[1] = df[1].str.replace('RT', '')
    df[1] = df[1].str.replace(r'^https?:\/\/.*[\r\n]*', '')

    tokenized = df[1].apply(lambda x: tokenizer.encode(str(x), add_special_tokens=True))

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded[:])  
    attention_mask = torch.tensor(attention_mask[:])

    with torch.no_grad():
        last_hidden_states = model(input_ids.to(torch.int64), attention_mask=attention_mask)

    features = last_hidden_states[0][:,0,:].numpy()

    for i in range(len(tweet_list)):
        predicted_list.append(lr_clf.predict(features[i, np.newaxis])[0])

    return predicted_list

### FLASK ###
app = Flask(__name__)

@app.route('/')
def page():
    predicted_list = []
    return render_template('index.html')

@app.route('/tweets', methods=['GET','POST'])
def tweets():
    date_input = False
    popularity_input = False
    verification_input = False

    if request.args.get('searchinput') is not None:
        search_input = request.args.get('searchinput')
    else:
        search_input = request.form.get('searchinput')
        
        attribute_input = request.form.get('selectattribute')
        if attribute_input == "1":
            date_input = request.form.get('datefilter')
            print("The date input is {}".format(date_input), flush=True)
        elif attribute_input == "2":
            popularity_input = True
        elif attribute_input == "3":
            verification_input = True

    print("The search input is {}".format(search_input), flush=True)
    
    tweet_list, tweet_link_list = get_tweets_hashtag(str(search_input), date_input, popularity_input, verification_input)
    predicted_list = predict(tweet_list)

    global tweet_left_list
    global tweet_right_list
    
    tweet_left_list = []
    tweet_right_list = []
    tweet_left_link_list = []
    tweet_right_link_list = []

    for i in range(len(predicted_list)):
        if predicted_list[i] == 'HillaryClinton':
            tweet_left_list.append(tweet_list[i])
            tweet_left_link_list.append(tweet_link_list[i])
        else:
            tweet_right_list.append(tweet_list[i])
            tweet_right_link_list.append(tweet_link_list[i])

    return render_template('tweets.html', tweet_left_link_list=tweet_left_link_list, tweet_right_link_list=tweet_right_link_list)

@app.route('/send', methods=['POST'])
def send():
    global tweet_left_list
    global tweet_right_list
    
    last_seen_id = retrieve_last_seen_id(FILE_NAME)
    if "l" in request.form['form']:
        data = {
            "text": tweet_left_list[int(request.form['form'][2:]) - 1],
            "side": request.form['id']
        }
    else:
        data = {
            "text": tweet_right_list[int(request.form['form'][2:]) - 1],
            "side": request.form['id']
        }
    
    last_seen_id += 1
    db.child("data").child(last_seen_id).set(data)
    store_last_seen_id(last_seen_id, FILE_NAME)
    
    return '', 200

if __name__ == '__main__':
    app.run(use_reloader=True, debug=True)