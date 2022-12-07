import tweepy
import pandas as pd
import configparser
import re
import os
from textblob import TextBlob
from wordcloud import WordCloud
import streamlit as st
import datetime, pytz
import text2emotion as te
from geopy.geocoders import Nominatim
import nltk
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
######
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from PIL import Image, ImageFont
import snscrape.modules.twitter as sntwitter
from plotly import tools
from nrclex import NRCLex
import time
######
from collections import Counter	
from heapq import nlargest	
from rank_bm25 import *	
import json	
import string, unicodedata	
from nltk import word_tokenize, sent_tokenize, FreqDist	
from nltk.stem import LancasterStemmer	
from nltk.tokenize import TweetTokenizer	
from nltk.util import ngrams	
import preprocessor as p	
from functools import reduce	
import streamlit.components.v1 as components	
from pandas.api.types import (	
    is_categorical_dtype,	
    is_datetime64_any_dtype,	
    is_numeric_dtype,	
    is_object_dtype,	
)	
import streamlit as st	
from io import StringIO

try:
    nltk.data.path.append(os.getcwd() +"/nltk_models/")
    nltk.data.find('omw-1.4')
    nltk.data.find('wordnet')	
    nltk.data.find('stopwords')
except Exception:
    nltk.download('omw-1.4', download_dir=(os.getcwd() +"/nltk_models/"))
    nltk.download('wordnet', download_dir=(os.getcwd() +"/nltk_models/"))
    nltk.download('stopwords', download_dir=(os.getcwd() +"/nltk_models/"))

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"  # flags (iOS)
                           "]+", flags=re.UNICODE)

###### Function to Connect to Twitter API ###############
def twitter_connection():

    config = configparser.ConfigParser()
    config.read("config.ini")

    api_key = config["twitter"]["api_key"]
    api_key_secret = config["twitter"]["api_key_secret"]
    access_token = config["twitter"]["access_token"]

    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    api = tweepy.API(auth)

    return api

api = twitter_connection()


##### Clean Twitter Text ###########
def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
    text = re.sub('#', '', text) # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text) # Removing RT
    text = re.sub('https?:\/\/\S+', '', text)
    text = re.sub("\n","",text) # Removing hyperlink
    text = re.sub(":","",text) # Removing hyperlink
    text = re.sub("_","",text) # Removing hyperlink
    text = emoji_pattern.sub(r'', text)
    return text

####### Function to Extract Mentions from Tweets ##########
def extract_mentions(text):
    text = re.findall("(@[A-Za-z0–9\d\w]+)", text)
    return text

####### Function to Extract Hashtags from Tweets ##########
def extract_hastag(text):
    text = re.findall("(#[A-Za-z0–9\d\w]+)", text)
    return text
#When we calculate the sentiment of a text through TextBlob, it provides us numeric values for polarity and subjectivity. The numeric value for polarity describes how much a text is negative or positive. Similarly, subjectivity describes how much a text is objective or subjective.
# Each word in the lexicon has scores for:
# 1)     polarity: negative vs. positive    (-1.0 => +1.0)
# 2) subjectivity: objective vs. subjective (+0.0 => +1.0)
# 3)    intensity: modifies next word?      (x0.5 => x2.0)

def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

###### Create a function to get the polarity and then assign the Sentiment #########
def getPolarity(text):
   return  TextBlob(text).sentiment.polarity

def getAnalysis(score):
  if score < 0:
    return 'Negative'
  elif score == 0:
    return 'Neutral'
  else:
    return 'Positive'

####### Function to get the Satisfaction Score ##########
def getSatisfactionScore(value):
    return round((value + 1) / 2 * 10)

####### Function to Extract Emotions from Tweets ##########
def getEmotion(text):
    emotion = NRCLex(text)
    #print('\n', emotion.affect_frequencies)
    return emotion.affect_frequencies

####### Functions to Extract Promotors and Defractors from Tweets ##########
def getPromotors(score):
    if score > 8:
        return 1
    else:
        return 0

def getDefractors(score):
    if score <= 6:
        return 1
    else:
        return 0
		
####### Function to Find Coordinates of Tweets based on the Location Values ##########
def findCoordinates(location):
    geolocator = Nominatim(user_agent="my_user_agent")
    time.sleep(0.5)
    try:
        loc = geolocator.geocode(location)
        return pd.Series([loc.latitude, loc.longitude])
    except:
        return pd.Series(["No location" , "No location"])

@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)
def get_location_coordinates(data, vsentiment):
    #data = data.loc[(data['Analysis'] == 'Negative') | (data['Analysis'] == 'Neutral')].copy()
    data = data.loc[(data['Analysis'] == vsentiment)].copy()
    data.Location.fillna('', inplace=True)
    data = data.head(1000)
    data[['latitude', 'longitude']] = data['Location'].apply(findCoordinates)
    data = data.loc[data['latitude'] != 'No location']
    #data.to_csv('data_withloc.csv',index=False) 
    return data

####### Main Function to Pre-Process Tweets ##########
@st.cache(allow_output_mutation=True)
def preprocessing_data(word_query, number_of_tweets,start_dt,end_dt):

  # Creating list to append tweet data to
  tweets_list = []
  # Using TwitterSearchScraper to scrape data and append tweets to list
  q = word_query + ' since:' + str(start_dt) + ' until:' + str(end_dt)
  for tweet in sntwitter.TwitterSearchScraper(q).get_items():
    if len(tweets_list) == number_of_tweets:
     break
    else:
      tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.location, tweet.coordinates])   
  # Creating a dataframe from the tweets list above
  data = pd.DataFrame(tweets_list, columns=['Tweet Date','Tweet Id', 'Tweets','Location','Coordinates'])
  data["mentions"] = data["Tweets"].apply(extract_mentions)
  data["hastags"] = data["Tweets"].apply(extract_hastag)
  data['links'] = data['Tweets'].str.extract('(https?:\/\/\S+)', expand=False).str.strip()
  data['retweets'] = data['Tweets'].str.extract('(RT[\s@[A-Za-z0–9\d\w]+)', expand=False).str.strip()
  data['Original Tweet'] = data['Tweets']
  data.drop_duplicates(subset=['Tweets'], inplace=True)
  data = data.reset_index(drop=True)
  data['Tweets'] = data['Tweets'].apply(cleanTxt)
  discard = ["CNFTGiveaway", "GIVEAWAYPrizes", "Giveaway", "Airdrop", "GIVEAWAY", "makemoneyonline", "affiliatemarketing"]
  data = data[~data["Tweets"].str.contains('|'.join(discard))]
  data['Subjectivity'] = data['Tweets'].apply(getSubjectivity)
  data['Polarity'] = data['Tweets'].apply(getPolarity)
  data['Analysis'] = data['Polarity'].apply(getAnalysis)
  data['Satisfation score'] = data['Polarity'].apply(getSatisfactionScore)
  data.to_csv('data.csv',index=False) 
  return data

####### Function to Extract Mentions from Tweets ##########
@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)
def analyse_mention(data):

  mention = pd.DataFrame(data["mentions"].to_list()).add_prefix("mention_")

  try:
    mention = pd.concat([mention["mention_0"], mention["mention_1"], mention["mention_2"]], ignore_index=True)
  except:
    mention = pd.concat([mention["mention_0"]], ignore_index=True)
  
  #==mention = mention.value_counts().head(10)
  
  return mention

####### Function to Extract Hashtags from Tweets ##########
@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)	
def analyse_hastag(data):
  
  hastag = pd.DataFrame(data["hastags"].to_list()).add_prefix("hastag_")
  try:
    hastag = pd.concat([hastag["hastag_0"], hastag["hastag_1"], hastag["hastag_2"]], ignore_index=True)
  except:
    hastag = pd.concat([hastag["hastag_0"]], ignore_index=True)
  
  hastag = hastag.value_counts().head(10)

  return hastag

####### Function to Get and Plot Sentiments  ##########

@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)	
def get_sentiment_arr(data):
    analys = data["Analysis"].value_counts().reset_index().sort_values(by="index", ascending=False)
    analys.rename(columns = {'index':'Sentiment'}, inplace = True)
    return analys

@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)	
def sentiment_graph(data,number_of_tweets):
    top_labels = ['Positive', 'Neutral', 'Negative']

    colors = ['rgba(27,158,119,0.8)','rgba(246,207,113,0.8)','rgba(237,100,90,0.8)']
    #['rgba(27,158,119,0.8)','rgba(102,194,165,0.8)','rgba(179,226,205,0.8)']
    #['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)','rgba(122, 120, 168, 0.8)']

    analys = get_sentiment_arr(data)
    #print(analys)
    x_data = [list(analys['Analysis'])]
    y_data = ['Sentiments']
    fig = go.Figure()

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                orientation='h',
                marker=dict(
                    color=colors[i],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
                
            ))

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.15, 1]
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode='stack',
        paper_bgcolor='white', #'rgb(248, 248, 255)',
        plot_bgcolor='white', #rgb(248, 248, 255)',
            margin=dict(l=2, r=2, t=7, b=2), #dict(l=120, r=10, t=140, b=80), dict(l=10, r=20, t=20, b=330)
        showlegend=False,
        autosize=False,
        width=900,
        height=120,
    )

    annotations = []

    for yd, xd in zip(y_data, x_data):
        # labeling the y-axis
        annotations.append(dict(xref='paper', yref='y',
                                x=0.14, y=yd,
                                xanchor='right',
                                text=str(yd),
                                font=dict(family='Arial', size=14,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False, align='right'))
        # labeling the first percentage of each bar (x_axis)
        annotations.append(dict(xref='x', yref='y',
                                x=xd[0] / 2, y=yd,
                                text=str(round((xd[0] / number_of_tweets),2) * 100) + '%',
                                font=dict(family='Arial', size=14,
                                          color='rgb(248, 248, 255)'),
                                showarrow=False))
        # labeling the first Likert scale (on the top)
        if yd == y_data[-1]:
            annotations.append(dict(xref='x', yref='paper',
                                    x=xd[0] / 2, y=1.1,
                                    text=top_labels[0],
                                    font=dict(family='Arial', size=14,
                                              color='rgb(67, 67, 67)'),
                                    showarrow=False))
        space = xd[0]
        for i in range(1, len(xd)):
                # labeling the rest of percentages for each bar (x_axis)
                annotations.append(dict(xref='x', yref='y',
                                        x=space + (xd[i]/2), y=yd,
                                        text=str(round((xd[i] / number_of_tweets),2) *100)+ '%',
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(248, 248, 255)'),
                                        showarrow=False))
                # labeling the Likert scale
                if yd == y_data[-1]:
                    annotations.append(dict(xref='x', yref='paper',
                                            x=space + (xd[i]/2), y=1.1,
                                            text=top_labels[i],
                                            font=dict(family='Arial', size=14,
                                                      color='rgb(67, 67, 67)'),
                                            showarrow=False))
                space += xd[i]

    fig.update_layout(annotations=annotations)
    return fig

####### Function to Find Top 5 Mentions from Tweets ##########
@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)	
def get_top_mentions(data):
    mentions = data.explode('mentions')['mentions'].to_string(header=False,index=False).split('\n')
    mentions = [idx for idx in mentions] 
    mentions = ','.join([item.strip() for item in mentions if item.strip() != 'NaN'])   
    mentions = mentions.replace("[","")
    mentions = mentions.replace("]","")
    mentions_list = mentions.split(",")
    df = pd.DataFrame(mentions_list, columns=['Mentions'])
    df = df[df['Mentions'] != '']
    mentions_data = df['Mentions'].value_counts().reset_index().sort_values(by="Mentions", ascending=False).head(5)
    return mentions_data

####### Function to Find Top 5 Hashtags from Tweets ##########
@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)	
def get_top_hashtags(data):
    hashtags = data.explode('hastags')['hastags'].to_string(header=False,index=False).split('\n')
    hashtags = [idx for idx in hashtags] 
    hashtags = ','.join([item.strip() for item in hashtags if item.strip() != 'NaN'])   
    hashtags = hashtags.replace("[","")
    hashtags = hashtags.replace("]","")
    hashtags_list = hashtags.split(",")
    df = pd.DataFrame(hashtags_list, columns=['hashtags'])
    df = df[df['hashtags'] != '']
    hashtags_data = df['hashtags'].value_counts().reset_index().sort_values(by="hashtags", ascending=False).head(5)
    #print(hashtags_data)
    return hashtags_data

####### Function to Create a Horizontal Bar Chart for Top 5 Mentions and Tweets ##########
@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)	
def create_bar_chart(data):
    mentions_data = get_top_mentions(data)
    mentions_data = mentions_data.rename(columns={"Mentions":"Mentions Count","index":"Mentions"})
    hashtags_data = get_top_hashtags(data)
    hashtags_data = hashtags_data.rename(columns={"hashtags":"Hashtags Count","index":"Hashtags"})
    fig1 = px.bar(mentions_data,y='Mentions',x='Mentions Count',orientation='h', text_auto=True) #,color='rgb(248,156,116)')
    fig2 = px.bar(hashtags_data,y='Hashtags',x='Hashtags Count',orientation='h', text_auto=True)#, color='rgb(246,207,113)')
    fig = go.Figure(data = fig1.data + fig2.data)
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True,
        autosize=False,
        width=500,
        height=500,
        )
    fig.update_traces(marker_color=px.colors.qualitative.Pastel[0])
    return fig

####### Function to Create a Pie Chart for Emotion Analysis ##########
@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)	
def graph_emotion(data, word_q):
    colors = list(px.colors.qualitative.Pastel)
    combined_tweets = ' '.join(data['Tweets']) 
    emotion = getEmotion(combined_tweets)
    dataarr = pd.json_normalize(emotion)
    labels = dataarr.columns
    values = dataarr.values[0]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_traces(hoverinfo='label+percent', textinfo='label+percent', textfont_size=12,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    
    fig.update_layout(showlegend = False)
    return fig

####### Function to get the NPS Score for Brand ##########
@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)	
def get_nps_score(data):
    total_population = data.shape[0]
    promoters = (data.loc[data['Satisfation score'] > 8].shape[0] / total_population) * 100
    detractors = (data.loc[(data['Satisfation score'] >= 0) & (data['Satisfation score'] <=6)].shape[0] / total_population) * 100
    nps_score = promoters - detractors
    return nps_score

####### Function to Plot the WordCloud based on Hashtags ##########
def plot_hashtag_wc(data):
    orig_img_mask = np.array(Image.open('img/comment.png'))
    hashtag = data.explode('hastags')['hastags'].to_string(header=False,index=False).split('\n')
    hashtag = [idx for idx in hashtag if not re.findall("[^\u0000-\u05C0\u2100-\u214F]+", idx)]
    hastag_str = ','.join([item.strip() for item in hashtag if item.strip() != 'NaN'])
    wordcloud = WordCloud(background_color='white', width = 40, height=40, margin=0, colormap='Set1',mask=orig_img_mask,random_state=1,collocations=False).generate(hastag_str)
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return plt

#################################################################################################

def show_top_tweets(data,topic_select,sentiment):
    ### BM25 Function to add to return a new 'data' dataframe
    df = pd.DataFrame(data[data['Analysis'] == sentiment].sort_values(by='Satisfation score',ascending=False).head(10))
    df = pd.DataFrame(df, columns=("Tweets","Satisfation score"))
    return df

### Functions to return n serach terms for BM25 ###

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt

def get_n_search_terms(data, search_terms):
    d1 = data
    d1['Original Tweet'].fillna('',inplace=True)
    d1['clean_tweet'] = d1['Original Tweet'].apply(lambda x: remove_pattern(x, "https|RT|@[\w]*"))#np.vectorize(remove_pattern)(data['Original Tweet'], "https|RT|@[\w]*")
    #remove punctuations
    d1['clean_tweet'] = d1['clean_tweet'].str.replace("[^a-zA-Z#]", " ", regex=True)
    #lowering string
    d1['clean_tweet'] = d1['clean_tweet'].str.lower()
    #remove stop words
    stop_words = set(stopwords.words('english')) 

    d1['clean_tweet'] = [' '.join([w for w in x.lower().split() if w not in stop_words]) 
        for x in d1['clean_tweet'].tolist()]
    #remove words with len < 2
    d1['clean_tweet'] = d1['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    #tokenization
    tokenized_tweet = d1['clean_tweet'].apply(lambda x: list(ngrams(x.split(), 2)))
    
    l = reduce(lambda x, y: list(x)+list(y), zip(tokenized_tweet))
    flatten = [item for sublist in l for item in sublist]
    counts = Counter(flatten).most_common()
    
    terms = []
    for item in counts[:search_terms]:
        terms.append(' '.join([w for w in item[0]]))
    
    return terms  

##### BM25 scoring functions ######
def bm25_preprocess_data(data):
    #Removes Numbers
    data = data.astype(str).str.replace(r'\d+', '', regex=True)
    lower_text = data.str.lower()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    w_tokenizer =  TweetTokenizer()
 
    def lemmatize_text(text):
        return [(lemmatizer.lemmatize(w)) for w \
                       in w_tokenizer.tokenize((text))]
    def remove_punctuation(words):
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', (word))
            if new_word != '':
                new_words.append(new_word)
        return new_words
    words = lower_text.apply(lemmatize_text)
    words = words.apply(remove_punctuation)
    return pd.DataFrame(words)

def create_bm25_corpus(data):

    for i,v in enumerate(data['Original Tweet']):
        data.loc[i,'cleaned tweets'] = p.clean(v)
    pre_tweets = bm25_preprocess_data(data['cleaned tweets'])
    data['cleaned tweets'] = pre_tweets
    stop_words = set(stopwords.words('english'))
    data['cleaned tweets'] = data['cleaned tweets'].apply(lambda x: [item for item in \
                                        x if item not in stop_words])
    corpus = data['cleaned tweets'].to_list()
    
    return corpus

def get_bm25_score(data, corpus, query_ls, relevant_items, is_csv):
    results_ls = []
    bm_25 = BM25Okapi(corpus)
    
    if is_csv:
        for query in query_ls:
            tokenized_query = query.split(" ")
            doc_scores = bm_25.get_scores(tokenized_query)

            for item in doc_scores.argsort()[-relevant_items:][::-1]:
                results_ls.append((data.loc[item, 'Original Tweet'], str(doc_scores[item]), query))

        return pd.DataFrame(results_ls, columns=['Original Tweet', 'Relevance Score', 'Query']).to_csv('search_results.csv', index=False)
    else:
        for query in query_ls:
            tokenized_query = query.split(" ")
            doc_scores = bm_25.get_scores(tokenized_query)
            for item in doc_scores.argsort()[-relevant_items:][::-1]:
                results_ls.append((data.loc[item, 'Original Tweet'], data.loc[item, 'Satisfation score'], data.loc[item, 'Analysis']))
        df = pd.DataFrame(results_ls, columns=['Tweets', 'Satisfation score','Analysis'])
        return df


def bm25_scoring_func(data, num_topics, relevant_results, is_query_ls, is_csv, query_term=None):
    #Gets list of N most trending topics
    topic_ls = get_n_search_terms(data, num_topics)
    #print(topic_ls)
    corpus = create_bm25_corpus(data)
    
    if is_query_ls:
    
        if num_topics == len(topic_ls):
            return get_bm25_score(data, corpus, topic_ls, relevant_results, is_csv)
        else:
            raise ValueError("BM25 not computed correctly. Check the number of topics requested")
    
    else:
        return get_bm25_score(data, corpus, [query_term], relevant_results, is_csv)

##### BM25 scoring ends #######
def filter_dataframe(df: pd.DataFrame, topic_select) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """

    df = df.copy()

    df = df[df['Query'] == topic_select]

    return df


def show_top_tweets(data,topic_select,sentiment):
    ### BM25 Function to add to return a new 'data' dataframe
    d = bm25_scoring_func(data, 5, 200, False, False, topic_select)

    df = pd.DataFrame(d[d['Analysis'] == sentiment].sort_values(by='Satisfation score',ascending=False).head(10))
    df = pd.DataFrame(df, columns=("Tweets","Satisfation score"))
    return df

####### Function to Plot the NPS Trend ##########
@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)	
def calculate_nps_trend(data,linetype):
    # Calculate NPS Trends over the given range
    total_population = data.shape[0]
    data['Promotors'] = data['Satisfation score'].apply(getPromotors)
    data['Defractors'] = data['Satisfation score'].apply(getDefractors)
    data['Tweet Date'] = pd.to_datetime(data['Tweet Date'])
    if linetype == 'Hours':
        data['Range'] = data['Tweet Date'].dt.strftime('%H')
    else:
        data['Range'] = data['Tweet Date'].dt.strftime('%Y-%m-%d')
    Promotors = (data.groupby('Range')['Promotors'].sum() / total_population) * 100
    Defractors = (data.groupby('Range')['Defractors'].sum() / total_population) * 100
    nps_trend = pd.DataFrame(Promotors - Defractors)
    nps_trend.rename(columns={0 :'NPS_Score'}, inplace=True)
    fig = px.line(nps_trend, y='NPS_Score', x=nps_trend.index, markers=True)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)'
        )
    return fig