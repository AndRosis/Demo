
from flask import Flask, request, render_template, jsonify, Blueprint
from flask_restplus import Api, Resource, fields, reqparse
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from marshmallow import Schema, fields as ma_fields, post_load
from functools import wraps 

import os
import pickle
import secrets

from azure.storage.blob import BlockBlobService

import numpy as np

import json
import pyodbc

from subprocess import check_call, CalledProcessError

from werkzeug.security import generate_password_hash, check_password_hash

# additional install using apt-get - this library not installs properly using pip
try:
    check_call(['apt-get', 'install', '-y', 'libasound2'], stdout=open(os.devnull,'wb'))
except CalledProcessError as e:
    print(e.output)

# additional install using apt-get - this library not installs properly using pip
try:
    check_call(['apt-get', 'install', '-y', 'libssl1.0.2'], stdout=open(os.devnull,'wb'))
except CalledProcessError as e:
    print(e.output)


import azure.cognitiveservices.speech as speechsdk
import time
import wave

# additional install using apt-get - this library not installs properly using pip
try:
    check_call(['apt-get', 'install', '-y', 'libsndfile1'], stdout=open(os.devnull,'wb'))
except CalledProcessError as e:
    print(e.output)

from vibes import VibesExtract, AllVibesExtract, VoiceStat, GenderExtract, Sentiment

authorizations = {
    'apikey' : {
        'type' : 'apiKey',
        'in' : 'header',
        'name' : 'X-API-KEY'
    },
    'apiid' : {
        'type' : 'apiKey',
        'in' : 'header',
        'name' : 'X-API-ID'
    }
}
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

api = Api(app, version='1.0', title='DSC Voice Vibes API', description='Voice Vibes Extraction Services API', authorizations=authorizations, security=['apikey','apiid'])

app.config['APIKEY']='xxx'

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        token = None

        if 'X-API-KEY' in request.headers:
            token = request.headers['X-API-KEY']

        if not token:
            return {'message' : 'Token is missing.'}, 401

        if token != app.config['APIKEY']:
            return {'message' : 'Improper token.'}, 401

        print('TOKEN: {}'.format(token))
        return f(*args, **kwargs)

    return decorated

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def create_new_folder(local_dir):
    new_path = local_dir
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path

create_new_folder(app.config['UPLOAD_FOLDER'])

# transcription

def speech_recognize_continuous_from_file(filename, lang):
    """performs continuous speech recognition with input from an audio file"""
    speech_config = speechsdk.SpeechConfig(subscription=app.config['speech_key'], region=app.config['service_region'])
    speech_config.speech_recognition_language=lang
    #speech_config.request_word_level_timestamps()
    audio_config = speechsdk.audio.AudioConfig(filename=filename)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False
    all_res=[]
    def stop_cb(evt):
        """callback that stops continuous recognition upon receiving an event `evt`"""
        speech_recognizer.stop_continuous_recognition()
        nonlocal done
        done = True

    def handle_final_result(evt):
        """callback that handles continuous recognition results upon receiving an event `evt`"""
        all_res.append(evt.result.text)

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognized.connect(handle_final_result)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    return all_res

# database settings

app.config['server'] = '...'
app.config['database'] = '...'
app.config['username'] = '...'
app.config['password'] = '...'
app.config['driver'] = 'ODBC Driver 17 for SQL Server'

app.config['azure_account'] = '...'
app.config['azure_key'] = '...'

app.config['speech_key'] = "..."
app.config['service_region'] = "..."

app.config['sql_connection_string'] = 'DRIVER={' + app.config['driver'] + '};PORT=1433;SERVER=' + app.config['server']\
                                      + ';PORT=1443;DATABASE=' + app.config['database'] + ';UID=' \
                                      + app.config['username'] + ';PWD=' + app.config['password']

cnxn = pyodbc.connect(app.config['sql_connection_string'])
cursor = cnxn.cursor()

# containers settings

app.config['models_container_name'] = 'models'
app.config['uploads_container_name'] = 'uploads'

# blob services connection

block_blob_service = BlockBlobService(account_name=app.config['azure_account'], account_key=app.config['azure_key'])

# models objects creation

vibes =['neutral','calm','happy','engaged','irritated','fearful','impatient','surprised']

vex=[]
for vibe in vibes:
    ex = VibesExtract(block_blob_service, app.config['models_container_name'], vibe)
    vex.append(ex)

vs = VoiceStat()
all_vex = AllVibesExtract(block_blob_service, app.config['models_container_name'])
sent_vex = Sentiment(block_blob_service, app.config['models_container_name'])
gend_vex = GenderExtract(block_blob_service, app.config['models_container_name'])

# token services

def token_gen():
    return secrets.token_hex(32)

def create_client(client_name, client_surname, company, email, password, tel, nip, street, postal_code, town, country):
    password_hash = generate_password_hash(password)
    query = "insert into clients([name], surname, company, email, password_hash, tel, nip, street, postal_code, town, country) values('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')".format(client_name, client_surname, company, email, password_hash,tel, nip, street, postal_code, town, country)
    cursor.execute(query)
    client_id = cursor.execute('select @@IDENTITY').fetchone()[0]
    cnxn.commit()
    res={}
    res['client_id']=str(client_id)
    return res

def add_subscription(client_id, product_id, promotion_id, start_date, end_date):
    new_token = token_gen()

    query = "insert into subscriptions(token, client_id, product_id, promotion_id, start_date, end_date, active) values('{}','{}','{}','{}','{}','{}',{})".format(new_token, client_id, product_id, promotion_id, start_date, end_date, 1)
    cursor.execute(query)
    cnxn.commit()

    return new_token

def check_token(token):
    subscription_id = -1
    try:
        cursor.execute("SELECT token, id FROM subscriptions")
        row = cursor.fetchone()
        while row:
            if row[0] == token:
                subscription_id = row[1]
                break
            row = cursor.fetchone()
    
        return subscription_id
    except:
        return -2

# routing

@app.route('/')
def Home():
    return('Hello!')

# clients namespace
ns_clients = api.namespace('clients', description='Clients manipulation services')
api.add_namespace(ns_clients)

client_model=api.model('client_model',{
    'name': fields.String(required=True, description='Client name'),
    'surname': fields.String(required=True, description='Client surname'),
    'company': fields.String(required=False, description='Client company name'),
    'email': fields.String(required=True, description='E-mail'),
    'password': fields.String(required=True, description='Password'),
    'tel': fields.String(required=False, description='Telephone number'),
    'nip': fields.String(required=False, description='NIP'),
    'street': fields.String(required=True, description='Address - street posession and apartment'),
    'postal_code': fields.String(required=True, description='Address - postal code'),
    'town': fields.String(required=True, description='Address - town'),
    'country': fields.String(required=True, description='Address - country'),
})

@ns_clients.route('/client')
class Client(Resource):
    @ns_clients.expect(client_model)
    #@ns_clients.marshal_with(client_model)
    @token_required
    def post(self):
        res = create_client(request.json['name'],request.json['surname'],request.json['company'],request.json['email'],request.json['password'],request.json['tel'],request.json['nip'],request.json['street'],request.json['postal_code'],request.json['town'],request.json['country'])
        return res,201

# vibes namespace
ns_emo = api.namespace('vibes', description='Voice emotions extraction services')
api.add_namespace(ns_emo)

@ns_emo.route("/vibes_list")
class VibesList(Resource):
    @ns_emo.doc('List of vibes')
    @token_required
    def get(self):
        return vibes, 201

upload_parser = api.parser()
upload_parser.add_argument('audiofile', location='files', type=FileStorage, required=True, help='Audio file in the wmv format')

@ns_emo.route('/vibes')
class VibesExtraction(Resource):
    @ns_emo.doc('Voice vibes extraction')
    @ns_emo.expect(upload_parser)
    @token_required
    def post(self):
        token = None
        if 'X-API-ID' in request.headers:
            token = request.headers['X-API-ID']

        args = upload_parser.parse_args()
        uploaded_file = args['audiofile']
        
        ufn = app.config['UPLOAD_FOLDER']+uploaded_file.filename
        uploaded_file.save(ufn)

        blob_name = uploaded_file.filename
        block_blob_service.create_blob_from_stream(app.config['uploads_container_name'], blob_name, uploaded_file)
        
        #extract vibes
        
        vibes_pred_list,a_len, wc, vol = all_vex.recognize(ufn)            
        
        res={}

        res['seconds'] = str(a_len)
        res['words'] = str(wc)

        vol_res={}
        sec=0
        for v in vol:
            vol_res[sec]=str(v)
            sec += 1
        res['volume']=vol_res

        e=0
        for vibes_pred in vibes_pred_list:
            vibes_res={}
            sec=0
            for v in vibes_pred:
                vibes_res[sec]=str(v)
                sec += 1
            res[vibes[e]]=vibes_res
            e+=1

        os.remove(ufn)
        
        return res, 201

@ns_emo.route('/sentiment')
class SentimentExtraction(Resource):
    @ns_emo.doc('Voice sentiment recognition')
    @ns_emo.expect(upload_parser)
    @token_required
    def post(self):
        token = None

        client_id = 0
        subscription_id = 0

        if 'X-API-ID' in request.headers:
            token = request.headers['X-API-ID']
            subscription_id, client_id = check_token(token)

        args = upload_parser.parse_args()
        uploaded_file = args['audiofile']
        
        ufn = app.config['UPLOAD_FOLDER']+uploaded_file.filename
        uploaded_file.save(ufn)

        blob_name = str(client_id)+'__'+secure_filename(uploaded_file.filename)
        block_blob_service.create_blob_from_stream(app.config['uploads_container_name'], blob_name, uploaded_file)
        
        #extract vibes
        res = {}
        
        vibes_pred,a_len, wc, vol = sent_vex.recognize(ufn)            
        
        vibes_res={}
        sec=0
        for v in vibes_pred:
            vibes_res[sec]=str(v)
            sec += 1

        vol_res={}
        sec=0
        for v in vol:
            vol_res[sec]=str(v)
            sec += 1

        res['seconds'] = str(a_len)
        res['words'] = str(wc)
        res['volume']=vol_res
        res['sentiment']=vibes_res

        os.remove(ufn)
        
        return res, 201

@ns_emo.route('/gender')
class GenderExtraction(Resource):
    @ns_emo.doc('Voice gender recognition')
    @ns_emo.expect(upload_parser)
    @token_required
    def post(self):
        token = None
        if 'X-API-ID' in request.headers:
            token = request.headers['X-API-ID']

        args = upload_parser.parse_args()
        uploaded_file = args['audiofile']  
        
        ufn = app.config['UPLOAD_FOLDER']+uploaded_file.filename
        uploaded_file.save(ufn)

        blob_name = uploaded_file.filename
        block_blob_service.create_blob_from_stream(app.config['uploads_container_name'], blob_name, uploaded_file)
        
        #extract vibes
        res = {}
        
        vibes_pred,a_len, wc, vol = gend_vex.recognize(ufn)            
        
        vibes_res={}
        sec=0
        for v in vibes_pred:
            vibes_res[sec]=str(v)
            sec += 1

        vol_res={}
        sec=0
        for v in vol:
            vol_res[sec]=str(v)
            sec += 1

        res['seconds'] = str(a_len)
        res['words'] = str(wc)
        res['volume']=vol_res
        res['gender']=vibes_res
        
        os.remove(ufn)
        
        return res, 201

# text namespace
ns_text = api.namespace('text', description='Text extraction services')
api.add_namespace(ns_text)

@ns_text.route('/transcript_pl')
class TranscriptPL(Resource):
    @ns_text.doc('Voice transcription - Polish')
    @ns_text.expect(upload_parser)
    @token_required
    def post(self):
        lang='pl-PL'

        token = None
        if 'X-API-ID' in request.headers:
            token = request.headers['X-API-ID']

        args = upload_parser.parse_args()
        uploaded_file = args['audiofile']
        
        ufn = app.config['UPLOAD_FOLDER']+uploaded_file.filename
        uploaded_file.save(ufn)

        blob_name = uploaded_file.filename
        block_blob_service.create_blob_from_stream(app.config['uploads_container_name'], blob_name, uploaded_file)
        
        #transcript
        
        res_list = speech_recognize_continuous_from_file(ufn, lang)
                    
        trn = " ".join(res_list)
        res={}
        res['transcript']=trn
        #os.remove(ufn)
        
        return res, 201

# flask app run

if __name__ == '__main__':
    app.run()