#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import string
import sys
import re
import os
import pandas as pd
import argparse
import json
import datetime
from sklearn.model_selection import train_test_split

import nltk
nltk.download('punkt')
nltk.download('wordnet') 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 

import sys
sys.path.append('.')
from src.io_handler import IOHandler
from src.data_preprocessing.Translator import Translator
import demoji
demoji.download_codes()

STOPWORDS = ['i', 'im', 'ill', 'id', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "youre", "youve", "youll", "youd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'hes', 'him', 'his', 'himself', 'she', "shes", 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'theyre', 'theyll', 'them', 'their', 'theirs', 'themselves', 'what', 'whats', 'which', 'who', 'whom', 'this', 'that', 'thats', "thatll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', "shouldve", 'now', "could", 'would', 'u', "not", "isnt", "arent", "wasnt", "werent","havent","hasnt","hadnt","wont","wouldnt", "dont", "doesnt","didnt","cant","couldnt","shouldnt","mightnt","mustnt", "aint", "neednt" ]

class TweetPreprocessor():
    """Tweet Preprocessor
    
    This class preprocesses the given twitter data.
    """

    def __init__(self, language='en', labelled=False, verbose=True):
        """Initialization of parameters
        
        :param language: language of tweets that need to be preprocessed
        :param labelled: boolean whether tweets are labelled
        :param verbose: boolean for intermediate output
        """
        self.language = language
        self.labelled = labelled
        self.verbose = verbose

    def preprocess(self, df_tweets, triggers, terms, units):
        """Preprocess the given tweets
        
        :param df_tweets: dataframe containing tweets
        :param triggers: list of triggers
        :param terms: list of terms
        :param units: list of units

        :return: dataframe containing tweets, keywords and preprocessed text
        """

        #extract keywords
        if self.verbose:
            print("Extract keywords")
        df_tweets = self.extractKeyWords(df_tweets, triggers, terms, units)
        
        #preprocess tweet text
        if self.verbose:
            print("Process tweet text")
        df_tweets = self.preprocessText(df_tweets)

        return df_tweets
    

    def extractKeyWords(self, df_tweets, triggers, terms, units):
        """Extract keywords for given tweets
        
        :param df_tweets: dataframe containing tweets
        :param triggers: list of triggers
        :param terms: list of terms
        :param units: list of units

        :return: dataframe containing tweets and keywords
        """

        # stemming keywords
        ps = PorterStemmer()
        triggers = [(ps.stem(word), word) for word in triggers]
        terms = [(ps.stem(word), word) for word in terms]
        units = [(ps.stem(word), word) for word in units]
        

        # compare keywords and collect occurences
        dis_trigger = []
        dis_term = []
        dis_unit = []
        for tweet in df_tweets['full_text']:
            tokens = tweet.strip().split()
          
            trigger_list = []
            term_list = []
            unit_list = []
            for t in tokens:
                for pair in triggers:
                    if pair[0] in t:
                        trigger_list.append(pair[1])
                
                for pair in terms:
                    if pair[0] in t:
                        term_list.append(pair[1])
                 
                for pair in units:
                    if pair[0] in t:
                        unit_list.append(pair[1])

            dis_trigger.append(trigger_list)
            dis_term.append(term_list)
            dis_unit.append(unit_list)
        
        # add lists of keyword occurences to tweet dataframe
        df_tweets['trigger'] = dis_trigger
        df_tweets['term'] = dis_term
        df_tweets['unit'] = dis_unit

        return df_tweets
        

    def preprocessText(self, df_tweets):
        """Preprocess the tweets' text
        
        :param df_tweets: dataframe containing tweets
        
        :return: dataframe containing tweets, keywords and preprocessed text
        """

        sentences = df_tweets['full_text']

        #first preprocessing  before potential translation
        processed_tweets = []
        for tweet in sentences:
            tweet = demoji.replace(tweet, "")              
            tokens = tweet.strip().split()

            # Remove repeated letters and urls and user mentions
            pattern = re.compile(r"(.)\1{2,}")
            tokens = [pattern.sub(r"\1\1\1", i) for i in tokens]
            tokens = [re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", i) for i in tokens]
            
            # Join tokens after preprocessing to one tweet
            tweet = (" ").join(tokens)

            processed_tweets.append(tweet.strip())

        #if language is not english, translate tweets
        if self.language != "en":
            print(f"Translate {len(processed_tweets)} tweets from \'{self.language}\' to \'en\'")    
            trans = []
            for tweet in processed_tweets:
                translator = Translator(src_lang=self.language, dest_lang='en')
                try:
                    t = translator.translate(tweet)
                    print(t)
                    trans.append(t)
                except KeyboardInterrupt:
                    print ('Interrupted')
                    sys.exit()
                except Exception as e:
                    print(e)
                    trans.append("")

            df_tweets['full_text_translated'] = trans
            sentences = trans
        
        else:
            sentences = processed_tweets

        wn = WordNetLemmatizer()
        processed_tweets = []

        #full preprocessing
        for tweet in sentences:
            # Lowercase all words
            tweet = tweet.lower()
                
            tokens = tweet.strip().split()

            # Remove lemmatization 
            tokens = [wn.lemmatize(i) for i in tokens]

            # Remove stopwords
            tokens = [word for word in tokens if not word in STOPWORDS]
          
            # Join tokens after preprocessing to one tweet
            tweet = (" ").join(tokens)

            # Remove punctuation
            tweet = tweet.translate(str.maketrans('','',string.punctuation))

            processed_tweets.append(tweet.strip())

        df_tweets['full_text_preprocessed'] = processed_tweets

        return df_tweets


    def isNaN(self, string):
        return string != string


def run_preprocessing(io, language, labelled, date, verbose):
    """Run script for preprocessing tweets
    
    :param io: IOHandler for saving extracted tweets
    :param language: language of keywords for extraction
    :param labelled: boolean whether tweets are labelled
    :param verbose: boolean for intermediate output
    """

    df_tweets, filepath= io.combineTweetFiles(language, labelled, date, verbose)

    p = TweetPreprocessor(language, labelled, verbose)
    triggers, terms, units = io.getKeywords(language)
    df_tweets = p.preprocess(df_tweets, triggers, terms, units)
    io.saveTweets(df_tweets, f'{filepath}.csv')


if __name__ == "__main__":
    #parse all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', dest='language', action='store', type=str, default='en', help='language of tweets')
    parser.add_argument('--labelled', dest='labelled', action='store_true', default=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)

    parser.add_argument('--date', dest='date', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'))

    args = parser.parse_args()

    if args.verbose:
        print(f"Preprocessing Tweets of language {args.language} and labelled {args.labelled}")

    io = IOHandler()

    if args.date is None:
        date = io.getNewestExtractionDateRaw(args.language)
    else:
        date = args.date.strftime('%Y-%m-%d')

    #check whether input language is configured
    if not io.checkLanguage(args.language):
        print(f"language not known. Choose from the following languages: {io.listAllLanguages()}")
        sys.exit()

    run_preprocessing(io, args.language, args.labelled, date, args.verbose)
    
