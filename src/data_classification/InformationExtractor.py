#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statistics as stats
import matplotlib.pyplot as plt

import re
import string
import os
import sys
import json
import argparse

import spacy

import nltk
from nltk.tokenize import word_tokenize

import datetime
import sys
sys.path.append('.')
from src.io_handler import IOHandler


class InformationExtractor: 
    """Information Extractor
    
    This class extracts relevant information of tweets
    """

    def __init__(self):
        """Initialization of parameters
        """

        print("Generate Information Extractor")

    def extractInformation(self, df_tweets, ner_custom, language):
        """Extract relevant information from metadata and with NERs
        
        :param df_tweets: dataframe containing tweets
        :param ner_custom: customized NER for internal displacement tags

        :return: dataframe containing relevant information
        """

        #extract information from tweet metadata
        df_tweets = self.getTweetSummary(df_tweets, language)

        #extract information with NERs
        nlp = spacy.load("en_core_web_md")
        
        #initialize tag lists
        LOC_list = []
        DATE_list = []
        TERM_list = []
        FIGURE_list  = []
        UNIT_list  = []
        TRIGGER_list  = []

        #print(df_tweets.keys())
        if language == 'en':
            sentences = df_tweets['full_text'].to_list()
        else:
            sentences = df_tweets['full_text_translated'].to_list()

        for sen in sentences:
            loc, date, trigger, term, unit, fig = [], [], [], [], [], []
            try:
                #extract general information
                doc = nlp(sen)
                for ent in doc.ents:
                    if ent.label_ == 'DATE' or ent.label_ == 'TIME':
                        date.append(ent.text)
                    elif ent.label_ == 'GPE' or ent.label_ == 'LOC':
                        loc.append(ent.text)

                #extract internal displacement specific information
                doc_custom = ner_custom(sen)   
                for ent in doc_custom.ents:
                    if ent.label_ == 'B-trig':
                        trigger.append(ent.text)
                    elif ent.label_ == 'B-term':
                        term.append(ent.text)
                    elif ent.label_ == 'B-unit':
                        unit.append(ent.text)
                    elif ent.label_ == 'B-fig':
                        fig.append(ent.text)
                 
            except KeyboardInterrupt:
                print ('Interrupted')
                sys.exit()
            except:
                print("continue")

            LOC_list.append(loc)
            DATE_list.append(date)
            TERM_list.append(term)
            FIGURE_list.append(fig)
            UNIT_list.append(unit)
            TRIGGER_list.append(trigger)

        # add tag lists to tweet dataframe
        df_tweets['LOCATION'] = LOC_list
        df_tweets['DATE'] = DATE_list
        df_tweets['TRIGGER'] = TRIGGER_list
        df_tweets['TERM'] = TERM_list
        df_tweets['UNIT'] = UNIT_list
        df_tweets['FIGURE'] = FIGURE_list

        return df_tweets

    def getTweetSummary(self, df_tweets, language):
        """Extract relevant information from metadata
        
        :param df_tweets: dataframe containing tweets

        :return: dataframe containing relevant information from tweet metadata
        """

        df_results = pd.DataFrame()
        
        #extract relevant tweet information
        df_results["id_str"] = df_tweets["id_str"]
        try:
            df_results["created_at"] = df_tweets["created_at"]
        except:
            print("No tweet \"created at\" date available.")

        df_results["full_text"] = df_tweets["full_text"]
        df_results["full_text_preprocessed"] = df_tweets["full_text_preprocessed"]
        if not language == 'en':
            df_results['full_text_translated'] = df_tweets['full_text_translated']

        
        try:
            df_results["irrelevant"] = df_tweets["irrelevant"]
        except:
            print("Tweets not labelled.")    

        df_results["predicted_irrelevant"] = df_tweets["predicted_irrelevant"]
        df_results["retweeted"] = df_tweets["retweeted"]

        #extract relevant user information
        user_info = []
        for index, u in df_tweets.iterrows():
            tweet_user= eval(u["user"])
            user = {}
            user["id"] = tweet_user["id"]
            user["screen_name"] = tweet_user["screen_name"]
            user["location"] = tweet_user["location"]
            user["created_at"] = tweet_user["created_at"]
            user["verified"] = tweet_user["verified"]

            user_info.append(user)

        df_results["user_info"] = user_info

        return df_results
      

if __name__ == "__main__":

    #parse all arguments
    parser = argparse.ArgumentParser()

    #type of preprocessing
    parser.add_argument('-c', '--classifier', dest='model_type', action='store', type=str, default='svm', help='name of classification model')
    parser.add_argument('-m', '--mode', dest='mode', action='store', type=str, default='train', help='mode of classifier: train / test / predict')
    parser.add_argument('-l', '--language', dest='language', action='store', type=str, default='en', help='language of prediction tweets')
    parser.add_argument('--classifierLanguage', dest='class_lang', action='store', type=str, default='en', help='language of training tweets')

    parser.add_argument('--labelled', dest='labelled', action='store_true', default=False)
    parser.add_argument('--unlabelled', dest='labelled', action='store_false')

    parser.add_argument('--date', dest='date', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), default = None)

    args = parser.parse_args()

    class_lang = 'en'

    io = IOHandler()

    df_tweets = io.loadPredictionFile(args.model_type, args.language, args.class_lang, args.date, args.labelled)

    ie = InformationExtractor()
    df_results = ie.extractInformation(df_tweets, io.loadNER(), args.language)

    io.saveTweetSummary(df_results, args.model_type, args.language, args.class_lang, args.date, args.labelled)