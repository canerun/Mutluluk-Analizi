# -*- coding: utf-8 -*-
"""
Created on Wed May 26 02:40:52 2021

@author: Kutalmış
"""

import tweepy

from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re 
import nltk
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer

#Twitter API credentials
consumerKey="dSfolEQezmcaGCP8RmRypIXS6"
consumerSecret="OXB8k9tIeK6ONt8nPXrdXPhje6slTclbWZRlCj4ChmMsG989NE"
accessToken="545887288-aONnOFpmoGUhYVvDi74vz8xWdDPZqPOdkdrEkMiC"
accessTokenSecret="HlUUPcXKXnNbNb3oeAksIlz1NcyiorvtqL8vFZA80r2cz"
keys=[]

#Create teh authentication object
authenticate = tweepy.OAuthHandler(consumerKey,consumerSecret)

#Set the access token and acces token secret

authenticate.set_access_token(accessToken,accessTokenSecret)

#Create the API object while passing in the auth information
api = tweepy.API(authenticate,wait_on_rate_limit=True)

posts2 = api.user_timeline(screen_name="serakadigil",count=200,tweet_mode="extended",include_rts="false")
seraTweets=pd.DataFrame([tweet.full_text for tweet in posts2],columns=['Tweets'])
seraTweets['Kullanici']="serakadıgil"
seraTweets['İttifak']=1;

posts = api.user_timeline(screen_name="suleymansoylu",count=200,tweet_mode="extended",include_rts="false")
soyluTweets=pd.DataFrame([tweet.full_text for tweet in posts],columns=['Tweets'])
soyluTweets['Kullanici']="suleymansoylu"
soyluTweets['İttifak']=0;

posts1 = api.user_timeline(screen_name="dbdevletbahceli",count=200,tweet_mode="extended",include_rts="false")
bahceliTweets=pd.DataFrame([tweet.full_text for tweet in posts1],columns=['Tweets'])
bahceliTweets['Kullanici']="dbdevletbahceli"
bahceliTweets['İttifak']=0;

posts3 = api.user_timeline(screen_name="erdemmgul",count=200,tweet_mode="extended",include_rts="false")
erdemmgulTweets=pd.DataFrame([tweet.full_text for tweet in posts3],columns=['Tweets'])
erdemmgulTweets['Kullanici']="erdemmgul"
erdemmgulTweets['İttifak']=1;

posts4 = api.user_timeline(screen_name="fahrettinaltun",count=200,tweet_mode="extended",include_rts="false")
fahrettinaltunTweets=pd.DataFrame([tweet.full_text for tweet in posts4],columns=['Tweets'])
fahrettinaltunTweets['Kullanici']="fahrettinaltun"
fahrettinaltunTweets['İttifak']=0;

posts5 = api.user_timeline(screen_name="taylanyildiz",count=200,tweet_mode="extended",include_rts="false")
taylanyildizTweets=pd.DataFrame([tweet.full_text for tweet in posts5],columns=['Tweets'])
taylanyildizTweets['Kullanici']="taylanyildiz"
taylanyildizTweets['İttifak']=1;

posts6 = api.user_timeline(screen_name="meral_aksener",count=200,tweet_mode="extended",include_rts="false")
meralTweets=pd.DataFrame([tweet.full_text for tweet in posts6],columns=['Tweets'])
meralTweets['Kullanici']="meralTweets"
meralTweets['İttifak']=1;

posts7 = api.user_timeline(screen_name="MuratKoseMamak",count=200,tweet_mode="extended",include_rts="false")
koseTweets=pd.DataFrame([tweet.full_text for tweet in posts6],columns=['Tweets'])
koseTweets['Kullanici']="koseTweets"
koseTweets['İttifak']=1;

result = pd.concat([seraTweets,soyluTweets,erdemmgulTweets,bahceliTweets,fahrettinaltunTweets,taylanyildizTweets,koseTweets,meralTweets])
result.to_csv (r'C:/Users/P2/Desktop/pyton/twitter/result.csv', index = False, header=True)