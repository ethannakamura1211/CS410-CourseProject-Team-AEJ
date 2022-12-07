from attr import has
import streamlit as st
from helper import preprocessing_data, analyse_mention, analyse_hastag, get_nps_score, get_location_coordinates,plot_hashtag_wc,graph_emotion,get_sentiment_arr,sentiment_graph,show_top_tweets,create_bar_chart, bm25_scoring_func, filter_dataframe, calculate_nps_trend
from datetime import datetime,timedelta
import numpy as np
import pandas as pd

bm25_relevant_result_cnt = 5
bm25_result_filename = "search_results.csv"
topic_select =""
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
     page_title="CS 410 - Tweet Analysis Project",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
)

with open('css/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html = True)

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>"""

######## Table styling  ###########################
th_props = [
  ('font-size', '14px'),
  ('text-align', 'center'),
  ('font-weight', 'bold'),
  ('color', '#ffffff'),
  ('background-color', '#006b54')
  ]
                               
td_props = [
  ('font-size', '12px'),
  ('height','400')
  ]
                                 
styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]
###################################################
col1, col2 = st.columns([1,2])
with col1:
    st.image("img/UIUC-Logo.png", width=250)
with col2:
    st.title("CS 410 - Tweets Sentiment Analysis")

st.sidebar.header('Choose your Brand')

function_option = st.sidebar.selectbox("Select Brand: ", ["Reebok", "Nike", "Apple", "UIUC", "Elon Musk","Puma"])
text_value = st.sidebar.text_input("or Type Brand Name:")
if (text_value != ''):
    word_query = text_value
else:
    word_query = function_option

st.sidebar.subheader("Tweets Date Range")
st.sidebar.text("(Default last 7 days)")

start_dt = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=7))
#with col3:
end_dt = st.sidebar.date_input("End Date", datetime.now())

st.sidebar.subheader("Number of Tweets")
number_of_tweets = st.sidebar.slider("How many tweets You want to collect from {}".format(word_query), min_value=1000, max_value=5000)

bm25_num_search_terms = st.sidebar.text_input("Number Trending Phrases (BM25):", "3")	
bm25_num_search_terms = int(bm25_num_search_terms)

col1, col2 = st.columns([1,2])
with col2:
    st.subheader(word_query + "'s Sentiment Analysis")

st.markdown("<hr/>", unsafe_allow_html = True)

load_button = st.sidebar.button("Analyze")

if "load_state" not in st.session_state:
    st.session_state.load_state = False

if load_button or st.session_state.load_state:
    st.session_state.load_state = True
    #data = pd.read_csv("data.csv", sep=",")
    data = preprocessing_data(word_query,number_of_tweets,start_dt,end_dt)
    mention = analyse_mention(data)
    hastag = analyse_hastag(data)
    hashtag_wc = plot_hashtag_wc(data)
    emotion = graph_emotion(data, word_query)
    bm25_scoring_func(data, bm25_num_search_terms, bm25_relevant_result_cnt, True, True)	
    bm25_res = pd.read_csv(bm25_result_filename)
    ##################################### Row 1 Widgets ##################################################
    sentiment_res = get_sentiment_arr(data).sort_values(by='Analysis',ascending=False)['Sentiment'][0]
    nps_score = round(get_nps_score(data),2)
    nps_delta = round(nps_score - 100)
    avg_satis_score = round(np.mean(data['Satisfation score']),2)
    var_satis_score = round(np.var(data['Satisfation score']),2)
    
    col3, col4, col5,col6 = st.columns(4)
    col3.metric('# of Tweets',number_of_tweets)
    col4.metric('Overall Sentiment',sentiment_res)
    col5.metric('NPS Score',nps_score,nps_delta)
    col6.metric('Avg. Satisfaction Score',avg_satis_score,var_satis_score)
    ######################################################################################################
    ##################################### Row 2 Widgets ##################################################
    st.markdown("<hr/>", unsafe_allow_html = True)
    col9, col10 = st.columns([1.3,3])
    with col9:
       st.markdown("##### Topics Identified using BM25 Method")
       topic_select = st.radio("Select a Topic to Filter Top 10 Tweets",set(bm25_res['Query'].to_list()), horizontal = True)	
    with col10:
        st.plotly_chart(sentiment_graph(data,number_of_tweets)) 

    col11, col12, col13 = st.columns(3)
    with col12:
        st.markdown("#### Top 10 Tweets for Selected Topic based on Sentiments")

    col14, col15, col16 = st.columns(3)
    with col14:
            df = show_top_tweets(data,topic_select,'Positive')
            nameDict={"Tweets":"Positive Tweets"}
            df = df.rename(columns=nameDict)
            df2=df.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            st.table(df2)
    with col15:
            df = show_top_tweets(data,topic_select,'Neutral')
            df = df.rename(columns={"Tweets":"Neutral Tweets"})
            df2=df.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            st.table(df2)
    with col16:
            df = show_top_tweets(data,topic_select,'Negative')
            df = df.rename(columns={"Tweets":"Negative Tweets"})
            df2=df.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
            st.markdown(hide_table_row_index, unsafe_allow_html=True)   
            st.table(df2)
    st.markdown("<hr/>", unsafe_allow_html = True)
 
    ######################################################################################################
    ##################################### Row 3 Widgets ##################################################
    col17, col18, col19 = st.columns(3)
    with col18:
        st.markdown("#### Tweets Analysis") #(Emotions, Topics, and EDA)
    with st.container():
        col20, col21, col22 = st.columns(3)
        with col20:
            st.markdown("##### Emotions based on Tweets")
            st.plotly_chart(emotion, use_container_width=True)
        with col21:
            st.markdown("##### Top 5 @Mentions and #Hashtags")
            st.plotly_chart(create_bar_chart(data))
        with col22:
            st.markdown("##### Topics discussed in Tweets")
            st.pyplot(hashtag_wc)
    ######################################################################################################
    ##################################### Row 4 Widgets ##################################################
    col23, col24 = st.columns(2)
    with col24:
         select_style = st.radio("Select the Trend Line Type",('Hours','Daily'), horizontal = True)	
         if select_style:
             st.markdown("##### NPS Trend for the selected Range")
             st.plotly_chart(calculate_nps_trend(data,select_style))
    with col23:
         sentiment_select = st.radio("Select the Sentiment to see the Location on the Map",('Positive','Negative'), horizontal = True)	
         if sentiment_select:
             map_location = get_location_coordinates(data,sentiment_select)
             st.markdown("##### Locations of tweets with " + sentiment_select + " Sentiments")
             st.map(map_location)
         
         
     