#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Dec 18 2019

Voice vibes extract.

@author: Andrzej Różyc

"""

import os

import numpy as np
import pandas as pd
import io

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import librosa
import h5py
import scipy
from sklearn.preprocessing import MinMaxScaler

Models = {
    'neutral': 'Emotion_1_L20_2_08K.h5',
    'calm': 'Emotion_2_L20_2_08K.h5',
    'happy': 'Emotion_3_L20_2_08K.h5',
    'engaged': 'Emotion_4_L20_2_08K.h5',
    'irritated': 'Emotion_5_L20_2_08K.h5',
    'fearful': 'Emotion_6_L20_2_08K.h5',
    'impatient': 'Emotion_7_L20_2_08K.h5',
    'surprised': 'Emotion_8_L20_2_08K.h5',
    'gender': 'Gender_L20_2_08K.h5',
    'sentiment':'Sentiment_L20_2_08K.h5'
}

vibes =['neutral','calm','happy','engaged','irritated','fearful','impatient','surprised']


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='same')

def audio_analyse(filename, window=200, cut_level=0.01):
    wc=0
    a_len=0
    [Fs, x] = scipy.io.wavfile.read(filename)

    a_len=x.shape[0]/Fs
    x=np.abs(x*0.9)
    
    xp=x.reshape(-1,1)
    scaler=MinMaxScaler()
    scaler.fit(xp)
    xpp=scaler.transform(xp)
    xppx=xpp.reshape(-1)

    df = pd.DataFrame(xppx, columns=['vol'])
    df['sec'] = df.index // Fs
    dfv = df.groupby('sec').mean()
    vol=np.array(dfv)

    vol=vol.reshape(-1,1)
    scaler=MinMaxScaler()
    scaler.fit(vol)
    vol=scaler.transform(vol)
    vol=vol.reshape(-1)

    xppxav=moving_average(xppx,int(window*Fs/1000))
    wp=np.zeros(xppxav.shape[0])
    wp[xppxav>cut_level]=1
    wpx=wp-np.roll(wp,1,axis=0)
    wpxx=np.zeros(xppxav.shape[0])
    wpxx[wpx>0]=1
    wc=sum(wpxx)

    return a_len, wc, vol

class GenderExtract:
    def __init__(self, block_blob_service, models_container):
        with io.BytesIO() as model_stream:
            block_blob_service.get_blob_to_stream(models_container, Models['gender'], model_stream)
            model_stream.seek(0)
            self.model = load_model(h5py.File(model_stream))

    def recognize(self, filename):
        a_len, wc, vol = audio_analyse(filename)

        empty_sec = vol < 0.1

        sample_length = 2
        samples = []
        FlMax = 0
        FwMax = 0
        
        length=round(a_len,0)
        for offset in range(0, (int(length)-sample_length)+1):
            x, Fs = librosa.load(path=filename, mono=True, duration=sample_length, offset=offset)
            mfcc = librosa.feature.mfcc(y=x, sr=Fs, hop_length=int(0.010*Fs), n_fft=int(0.025*Fs), n_mfcc=20)
            Fx = np.vstack((mfcc))
            F = np.transpose(Fx)
            samples.append(F)
            if FlMax < Fx.shape[1]:
                FlMax = Fx.shape[1]
            if FwMax < Fx.shape[0]:
                FwMax = Fx.shape[0]

        rsamples = []
        for sample in samples:
            rsamples.append(np.resize(sample, (FlMax, FwMax)))

        samples = np.array(rsamples)

        sa4d = samples.reshape(-1, samples.shape[1], samples.shape[2], 1)

        emo_pred = self.model.predict(sa4d)

        emo_pred_r = emo_pred.reshape(-1)
        emo_pred_r[empty_sec] = 0

        return emo_pred_r, a_len, wc, vol

class Sentiment:
    def __init__(self, block_blob_service, models_container):
        with io.BytesIO() as model_stream:
            block_blob_service.get_blob_to_stream(models_container, Models['sentiment'], model_stream)
            model_stream.seek(0)
            self.model = load_model(h5py.File(model_stream))

    def recognize(self, filename):
        a_len, wc, vol = audio_analyse(filename)

        empty_sec = vol < 0.1

        sample_length = 2
        samples = []
        FlMax = 0
        FwMax = 0
        
        length=round(a_len,0)
        for offset in range(0, (int(length)-sample_length)+1):
            x, Fs = librosa.load(path=filename, mono=True, duration=sample_length, offset=offset)
            mfcc = librosa.feature.mfcc(y=x, sr=Fs, hop_length=int(0.010*Fs), n_fft=int(0.025*Fs), n_mfcc=20)
            Fx = np.vstack((mfcc))
            F = np.transpose(Fx)
            samples.append(F)
            if FlMax < Fx.shape[1]:
                FlMax = Fx.shape[1]
            if FwMax < Fx.shape[0]:
                FwMax = Fx.shape[0]

        rsamples = []
        for sample in samples:
            rsamples.append(np.resize(sample, (FlMax, FwMax)))

        samples = np.array(rsamples)

        sa4d = samples.reshape(-1, samples.shape[1], samples.shape[2], 1)
        emo_pred = self.model.predict(sa4d)

        emo_pred_r = emo_pred.reshape(-1)
        emo_pred_r=emo_pred_r*2
        emo_pred_r=emo_pred_r-1
        emo_pred_r[empty_sec] = 0
        
        return emo_pred_r, a_len, wc, vol

class VibesExtract:
    def __init__(self, block_blob_service, models_container, emotion):
        with io.BytesIO() as model_stream:
            block_blob_service.get_blob_to_stream(models_container, Models[emotion], model_stream)
            model_stream.seek(0)
            self.model = load_model(h5py.File(model_stream))

    def recognize(self, filename):
        a_len, wc, vol = audio_analyse(filename)

        empty_sec = vol < 0.1

        sample_length = 2
        samples = []
        FlMax = 0
        FwMax = 0
        
        length=round(a_len,0)
        for offset in range(0, (int(length)-sample_length)+1):
            x, Fs = librosa.load(path=filename, mono=True, duration=sample_length, offset=offset)
            mfcc = librosa.feature.mfcc(y=x, sr=Fs, hop_length=int(0.010*Fs), n_fft=int(0.025*Fs), n_mfcc=20)
            Fx = np.vstack((mfcc))
            F = np.transpose(Fx)
            samples.append(F)
            if FlMax < Fx.shape[1]:
                FlMax = Fx.shape[1]
            if FwMax < Fx.shape[0]:
                FwMax = Fx.shape[0]

        rsamples = []
        for sample in samples:
            rsamples.append(np.resize(sample, (FlMax, FwMax)))

        samples = np.array(rsamples)

        sa4d = samples.reshape(-1, samples.shape[1], samples.shape[2], 1)

        emo_pred = self.model.predict(sa4d)

        emo_pred_r = emo_pred.reshape(-1)
        emo_pred_r[empty_sec] = 0

        return emo_pred_r, a_len, wc, vol

class VoiceStat:
    def recognize(self, filename):
        a_len, wc, vol = audio_analyse(filename)
        return a_len, wc, vol


class AllVibesExtract:
    def __init__(self, block_blob_service, models_container):
        self.models=[]
        for emotion in vibes:
            with io.BytesIO() as model_stream:
                block_blob_service.get_blob_to_stream(models_container, Models[emotion], model_stream)
                model_stream.seek(0)
                self.models.append(load_model(h5py.File(model_stream)))

    def recognize(self, filename):
        a_len, wc, vol = audio_analyse(filename)

        empty_sec = vol < 0.1

        sample_length = 2
        samples = []
        FlMax = 0
        FwMax = 0
        
        length=round(a_len,0)

        for offset in range(0, (int(length)-sample_length)+1):
            x, Fs = librosa.load(path=filename, mono=True, duration=sample_length, offset=offset)
            mfcc = librosa.feature.mfcc(y=x, sr=Fs, hop_length=int(0.010*Fs), n_fft=int(0.025*Fs), n_mfcc=20)
            Fx = np.vstack((mfcc))
            F = np.transpose(Fx)
            samples.append(F)
            if FlMax < Fx.shape[1]:
                FlMax = Fx.shape[1]
            if FwMax < Fx.shape[0]:
                FwMax = Fx.shape[0]

        rsamples = []
        for sample in samples:
            rsamples.append(np.resize(sample, (FlMax, FwMax)))

        samples = np.array(rsamples)
        sa4d = samples.reshape(-1, samples.shape[1], samples.shape[2], 1)
        
        emo_pred_list=[]
        for model in self.models:
            emo_pred = model.predict(sa4d)
            emo_pred_r = emo_pred.reshape(-1)
            emo_pred_r[empty_sec[:-1]] = 0
            emo_pred_list.append(emo_pred_r)

        return emo_pred_list, a_len, wc,vol