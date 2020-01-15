# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:53:06 2019

@author: Andrzej Różyc
"""

#%% Imports

import json
import pyodbc

import os, glob

import pandas as pd
import numpy as np

import pickle
from numba import jit

from sqlalchemy import create_engine

import datetime
from datetime import timedelta
from collections import Counter, OrderedDict

from sklearn.preprocessing import MinMaxScaler
import scipy.io.wavfile
import random

from inaSpeechSegmenter import Segmenter, seg2csv

#from GoogleS2T import Speech2Text
from CPS2T import *
from Vulgarisms import Vulgarisms_PL
from Emotions import Emotions
from Sentiments import TextSentiment
from LemmatizerPL import Lemmatizer
from CPStopwords import CPStopwords

from azure.storage.blob import BlockBlobService

#%% Export config

Export_to_CSV = False
Export_to_SQLDatabase = True

#%% Azure SQl and blobs and other global parameters

server = '...'
database = '...'
username = '...'
password = '...'
driver = 'ODBC Driver 17 for SQL Server'

blob_account_name = '...'
blob_account_key = '...'
blob_path = '...'
blob_container_name = '...'

data_directory = '/users/andrzej/VAdemo/'
stereo_audio_dir='/users/andrzej/VAdemo/Stereo/'
mono_audio_dir='/users/andrzej/VAdemo/Mono/'

emotions = ['', 'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
emotions_id = [1, 2, 3, 4, 5, 6, 7, 8]

#%% Create cloud db connections and storage connections

# Create db cursor for database operation
cnxn = pyodbc.connect('DRIVER={'+driver+'};PORT=1433;SERVER='+server+';PORT=1443;DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

# Create the BlockBlockService that is used to call the Blob service for the storage account
block_blob_service = BlockBlobService(account_name=blob_account_name, account_key=blob_account_key)

# Create a container.
block_blob_service.create_container(blob_container_name)

#Create sqlalchemy engine for pandas dataframe db operations
connectionstring = 'mssql+pyodbc://{uid}:{password}@{server}:1433/{database}?driver={driver}'.format(
    uid=username,
    password=password,
    server=server,
    database=database,
    driver=driver.replace(' ', '+'))

engn = create_engine(connectionstring, fast_executemany=True)

#%% Create all object needed for AI services

#gs2t = Speech2Text()
cps2t = PolMod()
s2tmodel = cps2t.speech_kac(False)
vlg = Vulgarisms_PL()
tsent = TextSentiment()
emop = Emotions()
lmm = Lemmatizer()
seg = Segmenter()
stw = CPStopwords()

stop_words = stw.stop_words
stop_words_lw = stw.stop_words_lw

#%% Read stuctures from SQL into dataframes

query = 'SELECT top 1 * FROM Calls'
dfsql = pd.read_sql(query, engn)
df_calls = pd.DataFrame(columns=dfsql.columns)

query = 'SELECT top 1 * FROM call_emotions'
dfsql = pd.read_sql(query, engn)
df_call_emotions = pd.DataFrame(columns=dfsql.columns)

query = 'SELECT top 1 * FROM call_data'
dfsql = pd.read_sql(query, engn)
df_call_data = pd.DataFrame(columns=dfsql.columns)

query = 'SELECT top 1 * FROM call_transcript'
dfsql = pd.read_sql(query, engn)
df_call_transcript = pd.DataFrame(columns=dfsql.columns)

query = 'SELECT top 1 * FROM consultant_topics'
dfsql = pd.read_sql(query, engn)
df_consultant_topics = pd.DataFrame(columns=dfsql.columns)

query = 'SELECT top 1 * FROM customer_topics'
dfsql = pd.read_sql(query, engn)
df_customer_topics = pd.DataFrame(columns=dfsql.columns)

#%% Helper functions

class OrderedCounter(Counter, OrderedDict):
    pass

def ClearDB():
    print('DB clear started...')

    cursor.execute('delete from call_data')
    print('20% done: call_data cleared.')

    cursor.execute('delete from call_emotions')
    print('40% done: call_emotions cleared.')

    cursor.execute('delete from  call_transcript') 
    print('50% done: call_transcript cleared.')

    cursor.execute("delete from call_time")
    print('60% done: consultant_topics cleared.')

    cursor.execute('delete from consultant_topics')
    print('70% done: consultant_topics cleared.')

    cursor.execute('delete  from customer_topics')
    print('80% done: customer_topics cleared.')

    cursor.execute('delete from calls')
    print('100% done: calls cleared.')

def PrepareTimeline():
    cursor.execute("insert into call_time(call_id, sec) select distinct call_id, sec from call_data")

def PrepareCallCRMId():
    cursor.execute("update calls set call_ID=Concat(trim(customer_number),'_', cast(id as varchar(20)))")


def ClearEmoInLowVolumeSeconds():
    cursor.execute("update call_emotions set [value_a]=0 FROM dbo.call_data INNER JOIN dbo.call_emotions ON dbo.call_data.call_id = dbo.call_emotions.call_id AND dbo.call_data.sec = dbo.call_emotions.sec Where [volume_a]<0.02")   
    cursor.execute("update call_emotions set [value_b]=0 FROM dbo.call_data INNER JOIN dbo.call_emotions ON dbo.call_data.call_id = dbo.call_emotions.call_id AND dbo.call_data.sec = dbo.call_emotions.sec Where [volume_b]<0.02")


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='same')
    #return np.convolve(data_set, weights, mode='valid')

def word_count(filename, window=200, cut_level=0.01):
    wc=0
    [Fs, x] = scipy.io.wavfile.read(data_directory+filename+'.wav')
    x=np.abs(x)
    scaler=MinMaxScaler()
    
    xp=x.reshape(-1,1)
    scaler.fit(xp)
    xpp=scaler.transform(xp)
    xppx=xpp.reshape(-1)
    
    xppxav=moving_average(xppx,int(window*Fs/1000))
    wp=np.zeros(xppxav.shape[0])
    wp[xppxav>cut_level]=1
    wpx=wp-np.roll(wp,1,axis=0)
    wpxx=np.zeros(xppxav.shape[0])
    wpxx[wpx>0]=1
    wc=sum(wpxx)
    return wc

def call_len(filename):
    [Fs, x] = scipy.io.wavfile.read(stereo_audio_dir+filename+'.wav')
    return x.shape[0]/Fs

def call_analysis(filename):
    [Fs, x] = scipy.io.wavfile.read(data_directory+filename+'.wav')
    x=np.abs(x)
    x=x.sum(axis=1) / 2
    x=moving_average(x,int(Fs))
    maxx=x.max()
    l=int(x.shape[0]/Fs)
    
    [Fs, cons_all] = scipy.io.wavfile.read(data_directory+'m'+filename+'1.wav')
    [Fs, cust_all] = scipy.io.wavfile.read(data_directory+'m'+filename+'0.wav')
    
    cons_all=np.abs(cons_all)/maxx
    
    cust_all=np.abs(cust_all)/maxx
    
    cons_all=moving_average(cons_all,int(Fs))
    cust_all=moving_average(cust_all,int(Fs))
    
    cons=np.zeros(int(cons_all.shape[0]/Fs))
    cons_time=0
    for i in range(0, int(cons_all.shape[0]/Fs)):
        cons[i]=cons_all[int(Fs/2+i*Fs)]
        if cons[i]>0.05: 
            cons_time+=1
            
    cust=np.zeros(int(cust_all.shape[0]/Fs))
    cust_time=0
    for i in range(0, int(cust_all.shape[0]/Fs)):
        cust[i]=cust_all[int(Fs/2+i*Fs)]
        if cust[i]>0.05: 
            cust_time+=1
            
        
    return l, cons, cust, cons_time, cust_time

def call_activity(filename):
    al=int(call_len(filename))
    
    print('')
    print('Processing: ', 'm'+filename+'1.wav')
    print('')

    if os.path.isfile(data_directory+filename+'1.csv'):
        df_co=pd.read_csv(data_directory+filename+'1.csv',sep='\t', header=None)
    else:
        segmentation = seg(data_directory+'m'+filename+'1.wav')
        seg2csv(segmentation, data_directory+filename+'1.csv')
        df_co=pd.DataFrame(segmentation)
        
    xgb=df_co.groupby(0).count()
    if xgb.loc[xgb.index=='Male'].shape[0]==0:
        gender_co=1
    elif  xgb.loc[xgb.index=='Female'].shape[0]==0:
        gender_co=0
    else:       
        if np.array(xgb.loc[xgb.index=='Male'][1])[0]>np.array(xgb.loc[xgb.index=='Female'][1])[0]:
            gender_co=0
        else:
            gender_co=1
        
    print('')
    print('Processing: ', 'm'+filename+'0.wav')
    print('')

    if os.path.isfile(data_directory+filename+'0.csv'):
        df_cu=pd.read_csv(data_directory+filename+'0.csv',sep='\t', header=None)
    else:
        segmentation = seg(data_directory+'m'+filename+'0.wav')
        seg2csv(segmentation, data_directory+filename+'0.csv')
        df_cu=pd.DataFrame(segmentation)

    xgb=df_cu.groupby(0).count()
    if xgb.loc[xgb.index=='Male'].shape[0]==0:
        gender_cu=1
    elif  xgb.loc[xgb.index=='Female'].shape[0]==0:
        gender_cu=0
    else:       
        if np.array(xgb.loc[xgb.index=='Male'][1])[0]>np.array(xgb.loc[xgb.index=='Female'][1])[0]:
            gender_cu=0
        else:
            gender_cu=1
    
    df_co.replace('NOACTIVITY',0, inplace=True)
    df_co.replace('Male',2, inplace=True)
    df_co.replace('Female',2, inplace=True)
    df_co.replace('Music',1, inplace=True)
    
    df_cu.replace('NOACTIVITY',0, inplace=True)
    df_cu.replace('Male',2, inplace=True)
    df_cu.replace('Female',2, inplace=True)
    df_cu.replace('Music',1, inplace=True)
    
    act_co=np.zeros(al)
    act_cu=np.zeros(al)
    
    ar_co=df_co.values
    ar_cu=df_cu.values
    
    for i in range(0,ar_co.shape[0]):
        for s in range(int(ar_co[i,1]),int(ar_co[i,2])):
            act_co[s]=int(ar_co[i,0])
            
    for i in range(0,ar_cu.shape[0]):
        for s in range(int(ar_cu[i,1]),int(ar_cu[i,2])):
            act_cu[s]=int(ar_cu[i,0])
    
    return df_co, df_cu, act_co, act_cu, gender_co, gender_cu

def load_call_emotions():
    df_emo=pd.read_csv(data_directory+'emo.csv')
    df_emo['file_name']=df_emo['file_name'].apply(lambda x: os.path.basename(x)[:-4])
    return df_emo
    
def main_emotion(emo, s):
    em = []
    for e in range(0,8):
        em.append(emo[e+1,s])        

    return em.index(max(em))+1

def call_transcript(filename):
    with open(data_directory+filename+'1.json', 'r') as f:
        data = json.load(f)
    arr_co=[]
    for i in range(0,len(data['results'])):
        arr_co.append(data['results'][i]['alternatives'][0]['transcript'])
    
    with open(data_directory+filename+'0.json', 'r') as f:
        data = json.load(f)
    arr_cu=[]
    for i in range(0,len(data['results'])):
        arr_cu.append(data['results'][i]['alternatives'][0]['transcript'])
        
    return arr_co, arr_cu

def sortSecond(val): 
    return val[1] 

def sortFirst(val): 
    return val[0] 

def main_topic(tr):
    c = OrderedCounter(lemma_arr(tr))
    del c['']
    if len(c)>0:
        return max(c, key=c.get)
    else:
        return ''

def topics(tr):
    c = OrderedCounter(lemma_arr(tr))
    del c['']
    return c.most_common(10)
    
def lemma_arr(text):
    words = text.split(' ')
    rwords = [w for w in words if not w in stop_words]
    rnwords = [w for w in rwords if not w == '']    
    lrnwords = lmm.lemmatize(rnwords)
    lrnwords = [w for w in lrnwords if not w == '']    
    lrnxwords = [w for w in lrnwords if not w in stop_words_lw]
    return lrnxwords

def lemma(text):   
    return ' '.join(lemma_arr(text))

def call_conversation(filename):
    transcript_text=''
    transcript_arr=[]

    if os.path.isfile(data_directory+'m'+filename+'1.trn'):
        tr1=pickle.load(open(data_directory+'m'+filename+'1.trn','rb'))
        r_co = tr1[0]
        tr_co = tr1[1]
    else:
        r_co,tr_co=cps2t.transcript_audio_file(s2tmodel, data_directory+'m'+filename+'1.wav')
        tr1=[r_co,tr_co]
        pickle.dump(tr1,open(data_directory+'m'+filename+'1.trn','wb'))
    
    if os.path.isfile(data_directory+'m'+filename+'0.trn'):
        tr2=pickle.load(open(data_directory+'m'+filename+'0.trn','rb'))
        r_cu = tr2[0]
        tr_cu = tr2[1]
    else:
        r_cu,tr_cu=cps2t.transcript_audio_file(s2tmodel, data_directory+'m'+filename+'0.wav')
        tr2=[r_cu,tr_cu]
        pickle.dump(tr2,open(data_directory+'m'+filename+'0.trn','wb'))
    
    wc_co=len(tr_co) 
    wc_cu=len(tr_cu)
    
    trw=[]
    for tra in tr_co:
        trw.append([tra[0],tra[1],''])
        
    for trb in tr_cu:
        trw.append([trb[0],'',trb[1]])
        
    trw.sort(key=sortFirst)
    
    transcript_arr=[]
    
    if trw[0][1]=='':
        tb=[0,'-','']
    else:
        tb=[0,'','-']
    ss=0
    ps=0
    buff=''
    for t in trw:
        if t[1]!='' and tb[1]=='':
            if ps>0:
                vco,re=vlg.CheckPhrase(buff)
                topic=main_topic(buff)
                transcript_arr.append([ss,'',buff,0,vco,'',topic])
            ss=t[0] #start second
            buff='' #staring buffering part
            ps=1 #conv side - consultant
            buff = buff + ' ' + t[1] #adding word to part
            tb=t 
        elif  t[1]!='' and tb[1]!='':
            buff = buff + ' ' + t[1] #adding word to part
            tb=t
        elif t[2]!='' and tb[2]=='':
            if ps>0:
                vco,re=vlg.CheckPhrase(buff)
                topic=main_topic(buff)
                transcript_arr.append([ss,buff,'',vco,0,topic,''])
            ss=t[0] #start second
            buff='' #staring buffering part
            ps=2 #conv side - consultant
            buff = buff + ' ' + t[2] #adding word to part
            tb=t 
        elif  t[2]!='' and tb[2]!='':
            buff = buff + ' ' + t[2] #adding word to part
            tb=t
            
    vco,re=vlg.CheckPhrase(buff)
    topic=main_topic(buff)
    lp=[ss,'','',0,0,'','']
    lp[ps]=buff
    lp[ps+2]=vco
    lp[ps+4]=topic
    transcript_arr.append(lp)
        
    text_co=''
    text_cu=''
    for tr in transcript_arr:
        if tr[1]!='':
            transcript_text=transcript_text+ ' | Consultant: '+ tr[1]
            text_co=text_co+' '+tr[1]
        else:
            transcript_text=transcript_text+ ' | Customer: '+ tr[2]
            text_cu=text_cu+' '+tr[2]
    
    return transcript_text, transcript_arr, wc_co, wc_cu, text_co, text_cu

#%% Listing audio files and uploading stereo audio files to the cloud

print("")
files_full=glob.glob(data_directory+'+*.wav')
files=[]
for file in files_full:
    files.append(os.path.basename(file)[:-4])
    block_blob_service.create_blob_from_path(blob_container_name, os.path.basename(file), file)
    print("Uploaded: " + file )

#%% Audio Activity Analysis

call_act_cos=[] 
call_act_cus=[] 
call_gender_cos=[] 
call_gender_cus=[]

for file in files:
    print('')    
    print('Audio Activity Analysis Processing: ',file)
    print('')

    df_co, df_cu, act_co, act_cu, gender_co, gender_cu=call_activity(file)
     
    call_act_cos.append(act_co)
    call_act_cus.append(act_cu)
    call_gender_cos.append(gender_co)
    call_gender_cus.append(gender_cu)
          
#%% Call Analysis
    
call_lengths=[]
call_cots=[]
call_cuts=[]
call_cons=[]
call_cust=[]

for file in files:
    print('')    
    print('Call Analysis Processing: ',file)
    print('')

    l, co,cu, cot, cut = call_analysis(file)
    call_lengths.append(l)
    call_cots.append(cot) #cons talking time
    call_cuts.append(cut) #cust talking time
    call_cons.append(co) #call volume - cons
    call_cust.append(cu) #call volume - cust

#%% Call Transcript
    
call_transcripts=[]

for file in files:
    print('')    
    print('Call Transcript Processing: ',file)
    print('')

    res=call_conversation(file)
    call_transcripts.append(res) #all transcription data
    
#%% Emotional Analysis

emo_e_co=[]
emo_g_co=[]
emo_s_co=[]
emo_i_co=[]
emo_e_cu=[]
emo_g_cu=[]
emo_s_cu=[]
emo_i_cu=[]

for file in files:
    print('')    
    print('Emotional Analysis Processing: ',file)
    print('')
    
    if os.path.isfile(data_directory+'m'+file+'1.emo'):
        emox_co = pickle.load(open(data_directory+'m'+file+'1.emo','rb'))
    else:
        emox_co = emop.recognize(data_directory+'m'+file+'1.wav')
        pickle.dump(emox_co,open(data_directory+'m'+file+'1.emo','wb'))
    
    emo_e_co.append(emox_co[0])
    emo_g_co.append(emox_co[1])
    emo_s_co.append(emox_co[2])
    emo_i_co.append(emox_co[3])
     
    if os.path.isfile(data_directory+'m'+file+'0.emo'):
        emox_cu = pickle.load(open(data_directory+'m'+file+'0.emo','rb'))
    else:
        emox_cu = emop.recognize(data_directory+'m'+file+'0.wav')
        pickle.dump(emox_cu,open(data_directory+'m'+file+'0.emo','wb'))
    
    emo_e_cu.append(emox_cu[0])
    emo_g_cu.append(emox_cu[1])
    emo_s_cu.append(emox_cu[2])
    emo_i_cu.append(emox_cu[3])

#%% Text sentiment analysis
     
call_s_co=[]
call_s_cu=[]

call_lw_co=[]
call_lw_cu=[]

tr_s_co=[]
tr_s_cu=[]

tr_lw_co=[]
tr_lw_cu=[]

call_topics_co=[]
call_topics_cu=[]

i=0
for tr in call_transcripts:
    call_lw_co.append(lemma(tr[4]))
    call_lw_cu.append(lemma(tr[5]))

    call_s_co.append(tsent.recognize(tr[4]))
    call_s_cu.append(tsent.recognize(tr[5]))

    call_topics_co.append(topics(tr[4]))
    call_topics_cu.append(topics(tr[5]))

    lw_co=[]
    s_co=[]
    lw_cu=[]
    s_cu=[]
    j=0
    for trp in tr[1]:
        if trp[1]=='':
            lw_co.append('')
            s_co.append(0)

            lw_cu.append(lemma(trp[2]))
            s_cu.append(tsent.recognize(trp[2]))
            
        elif trp[2]=='':
            lw_cu.append('')
            s_cu.append(0)

            lw_co.append(lemma(trp[1]))
            s_co.append(tsent.recognize(trp[1]))
        print('tr part done: ', i,j)
        j+=1
        
    tr_s_co.append(s_co)
    tr_s_cu.append(s_cu)
    
    tr_lw_co.append(lw_co)
    tr_lw_cu.append(lw_cu)
    
    print('Call done: ',i)
    i+=1
    
#%% Reading Excel data - call details

df_calls_info=pd.read_excel(data_directory+'calls.xlsx',index_col=0)

def sale_succ(call_id):
    if df_calls_info['[sale items]'][call_id]>0:
        return 1
    else:
        return 0

#%% Preparing result data - Main processing process

df_ncalls=pd.DataFrame(columns=df_calls.columns)
df_ncalls.drop(['id','call_audio_html', 'call_length_min','avg_speed_a', 'avg_speed_b', 'transcript_lw'], axis=1, inplace=True)

df_nemo = pd.DataFrame(columns=df_call_emotions.columns)
df_nemo.drop(['id'], axis=1, inplace=True)

df_ntran = pd.DataFrame(columns=df_call_transcript.columns)
df_ntran.drop(['id'], axis=1, inplace=True)

df_ndata = pd.DataFrame(columns=df_call_data.columns)
df_ndata.drop(['id'], axis=1, inplace=True)

df_ntopics_co=pd.DataFrame(columns=df_consultant_topics.columns)
df_ntopics_co.drop(['id'], axis=1, inplace=True)

df_ntopics_cu=pd.DataFrame(columns=df_customer_topics.columns)
df_ntopics_cu.drop(['id'], axis=1, inplace=True)

arr_ncalls=[]
arr_nemo=[]
arr_ntran=[]
arr_ndata=[]
arr_ntopics_co=[]
arr_ntopics_cu=[]

print('')
for i in range(0,len(files)):
    call_ID = i
    
    print('Starting call: ', i, 'Call ID: ', call_ID)
    
    arr_ncalls.append(
    {
    'id':i,
    'call_ID':call_ID,
    'call_file':blob_path+files[call_ID]+'.wav',
    'consultant_id':df_calls_info['[consultant id]'][call_ID],
    'customer_number':df_calls_info['[customer number]'][call_ID],
    'consultant_number':df_calls_info['[consultant number]'][call_ID],
    'call_date':df_calls_info['[call date]'][call_ID] ,
    'call_time':df_calls_info['[call time]'][call_ID],
    'call_length':call_lengths[call_ID],
    'talking_time_a':call_cots[call_ID],
    'talking_time_b':call_cuts[call_ID],
    'words_count_a':call_transcripts[call_ID][2],
    'words_count_b':call_transcripts[call_ID][3],
    'transcript':call_transcripts[call_ID][0],
    'transcript_a':call_transcripts[call_ID][4],
    'transcript_b':call_transcripts[call_ID][5],
    'transcript_a_lw':call_lw_co[call_ID],
    'transcript_b_lw':call_lw_cu[call_ID],
    'gender_a':call_gender_cos[call_ID],
    'gender_b':call_gender_cus[call_ID],
    'text_sentiment_a':call_s_co[call_ID],
    'text_sentiment_b':call_s_cu[call_ID],
    'product_id':df_calls_info['[product id]'][call_ID], 
    'campaign_id':df_calls_info['[campaign id]'][call_ID],
    'source_id':df_calls_info['[source id]'][call_ID], 
    'success_a':sale_succ(call_ID), 
    'success_b':sale_succ(call_ID), 
    'call_success':sale_succ(call_ID), 
    'sales_success':sale_succ(call_ID),
    'sales_items':df_calls_info['[sale items]'][call_ID], 
    'sales_value':df_calls_info['[sale value]'][call_ID], 
    'score_a':df_calls_info['[score consultant]'][call_ID], 
    'score_b':df_calls_info['[score customer]'][call_ID], 
    'call_score_a':0,
    'call_score_b':0, 
    'sales_score':0, 
    'sales_prob':0
    }
    )
    
    ix=0
    for tr in call_transcripts[call_ID][1]:
        arr_ntran.append({
                'call_id':i,
                'sec':tr[0],
                'text_a':tr[1],
                'text_b':tr[2],
                'text_a_lw':tr_lw_co[call_ID][ix],
                'text_b_lw':tr_lw_cu[call_ID][ix],
                'badwords_a':tr[3],
                'badwords_b':tr[4],
                'topic_a':tr[5],
                'topic_b':tr[6],
                'sentiment_a':tr_s_co[call_ID][ix],
                'sentiment_b':tr_s_cu[call_ID][ix]
                })
        ix+=1
        
    for ix in range(0, emo_e_co[call_ID].shape[1]):
        for e in range(1,9):
            arr_nemo.append({
                'call_id':i,
                'emotion_id':e,
                'sec':ix,
                'value_a':emo_e_co[call_ID][e,ix],
                'value_b':emo_e_cu[call_ID][e,ix]
                })
        
    l_emo_co=0
    l_emo_cu=0
    for s in range(0,len(call_cons[call_ID])):
        if s>=emo_e_co[call_ID].shape[1]:
            es=emo_e_co[call_ID].shape[1]-1
        else:
            es=s
                    
        me_co=main_emotion(emo_e_co[call_ID],es)
        me_cu=main_emotion(emo_e_cu[call_ID],es)
        
        ec_co=0
        if me_co!=l_emo_co:
            ec_co=1
        l_emo_co=me_co
        
        ec_cu=0
        if me_cu!=l_emo_cu:
            ec_cu=1
        l_emo_cu=me_cu
        
        arr_ndata.append({
                'call_id':i,
                'sec':s,
                'volume_a':call_cons[call_ID][s],
                'volume_b':call_cust[call_ID][s],
                'audio_activity_a':int(call_act_cos[call_ID][s]),
                'audio_activity_b':int(call_act_cus[call_ID][s]),
                'sentiment_a': emo_s_co[call_ID][es],
                'sentiment_b': emo_s_cu[call_ID][es],
                'intensity_a': emo_i_co[call_ID][es],
                'intensity_b': emo_i_cu[call_ID][es],
                'main_emotion_a':me_co,
                'main_emotion_b':me_cu,
                'emotion_change_a':ec_co,
                'emotion_change_b':ec_cu
               })
    
    tps=topics(call_transcripts[call_ID][4])
    for tp in tps:
        arr_ntopics_co.append({
                'call_id':i,
                'topic_text':tp[0],
                'topic_count':tp[1]
                })

    tps=topics(call_transcripts[call_ID][5])
    for tp in tps:
        arr_ntopics_cu.append({
                'call_id':i,
                'topic_text':tp[0],
                'topic_count':tp[1]
                })
        
    print( str(i)+', ', end='')

df_ncalls=df_ncalls.append(arr_ncalls,ignore_index=True)
df_nemo=df_nemo.append(arr_nemo,ignore_index=True)
df_ntran=df_ntran.append(arr_ntran,ignore_index=True)
df_ndata=df_ndata.append(arr_ndata,ignore_index=True)
df_ntopics_co=df_ntopics_co.append(arr_ntopics_co,ignore_index=True)
df_ntopics_cu=df_ntopics_cu.append(arr_ntopics_cu,ignore_index=True)

#%% Export Processed data to CSV files

if Export_to_CSV  
    df_ncalls.to_csv(data_directory+'calls.csv')
    df_nemo.to_csv(data_directory+'call_emotions.csv')
    df_ntran.to_csv(data_directory+'call_transcriptions.csv')
    df_ndata.to_csv(data_directory+'call_data.csv')
    df_ntopics_co.to_csv(data_directory+'call_consultant_topics.csv')
    df_ntopics_cu.to_csv(data_directory+'call_customer_topics.csv')

#%% Following sections will export data directly to database - warning: very slow
# Important update - using fast_executemany this is fast enough to run direct export to MS SQL database
# but still chunks like 10**4-10**5 should be considered

if Export_to_SQLDatabase
    #%% Calls export - fast
        
    df_ncalls.to_sql('calls', engn, if_exists='append', index=False)

    #%% Consultatn topics export - fast

    df_ntopics_co.to_sql('consultant_topics', engn, if_exists='append', index=False)

    #%% Customer topics export - fast

    df_ntopics_cu.to_sql('customer_topics', engn, if_exists='append', index=False)

    #%% Transcription parts export - slow, but can be executed

    df_ntran.to_sql('call_transcriptions', engn, if_exists='append', index=False)

    #%% Calls data export - large, very slow, should be done by bulk insert from csv

    df_ndata.to_sql('call_data', engn, if_exists='append', index=False)

    #%% Emotions export - large, very slow, should be done by bulk insert from csv
    
    df_nemo.to_sql('call_emotions', engn, if_exists='append', index=False)

#%% Following sections should be executed after export data to database

#%% Extracting seconds for calls time line filters

PrepareTimeline()

#%% Clear Emotions In Low Volume Seconds

ClearEmoInLowVolumeSeconds()