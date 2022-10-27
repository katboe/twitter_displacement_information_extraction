#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import numpy as np
import pandas as pd
import datetime
import os
import json
import re
import gensim
from gensim.models import Word2Vec
from sklearn.utils import shuffle
import joblib
import spacy

CONFIG_PATH = "config/structure.json"

class IOHandler():
    """Input/Output Manager
    
    This class is responsible for all input/output related functions. It constructs correct
    filepaths and reads/writes the specified data
    """

    def __init__(self):
        """Initialization of parameters
        
        Load config files and initialize base directory
        """
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.BASE_DIR = os.path.abspath(os.path.join(curr_dir, '../'))   

        #load path structure
        with open(os.path.join(self.BASE_DIR, CONFIG_PATH)) as file:
            self.structure = json.load(file)

        #load keywords
        with open(os.path.join(self.BASE_DIR, self.structure['keywords'])) as file:
            self.keywords = json.load(file)


    def checkLanguage(self, language):
        #check whether language exists
        try:
            words = self.keywords[language]
            return True
        except:
            return False

    def checkModels(self, model_type):
        #check whether model type is configured
        with open(os.path.join(self.BASE_DIR, self.structure['models_config'])) as file:
            self.models_conf = json.load(file)
        
        try:
            model = self.models_conf[model_type]
            return True
        except:
            return False

    def checkModelExists(self, model_type, language):
        #check whether model is already trained
        if not os.path.exists(f"{os.path.join(self.BASE_DIR, self.structure['modelData'])}/{model_type}_{language}.joblib.pkl"):
            return False

        return True

    def listAllLanguages(self):
        return list(self.keywords)

    def listAllModels(self):
        if self.models_conf == None:
            with open(os.path.join(self.BASE_DIR, self.structure['models_config'])) as file:
                self.models_conf: dict[str, Union[str, list[str], int]] = json.load(file)

        return list(self.models_conf)

    def getDisplacementTriggers(self, language):
        return self.keywords[language]["displacement_triggers"]

    def getDisplacementTerms(self, language):
        return self.keywords[language]["displacement_terms"]

    def getDisplacementUnits(self, language):
        return self.keywords[language]["displacement_units"]

    def getKeywords(self, language):
        return self.getDisplacementTriggers(language), self.getDisplacementTerms(language), self.getDisplacementUnits(language)

    def combineTweetFiles(self, language, labelled, date, verbose):
        #combine all extracted tweet files
        dir_all = {}
        filepath = ""
        
        if not labelled:
            path = os.path.join(self.BASE_DIR, self.structure['rawTweets'])
            if date is None:
                date = getNewestExtractionDateRaw(language)
                dir_name = f'{date}-{language}'

            else:
                dir_name = f'{date}-{language}'

            subdir = os.path.join(path, dir_name)
            df_tweets = self.readTweetDirectory(os.path.join(self.BASE_DIR, subdir))
            
            #Write data to file
            print(f"Process tweets extracted on {date}")
            filepath = f'{os.path.join(self.BASE_DIR, self.structure["preprocessedPredict"])}/{date}-{language}'
            
        #labelled tweets that the model is trained on
        else:
            labelled_path = f"{os.path.join(self.BASE_DIR,  self.structure['labelledTweets'])}/labelled-{language}"
            df_all = self.readTweetDirectory(labelled_path)

            #Filter unlabeled data
            df_labelled = df_all.loc[(df_all['irrelevant'] == 0.0) | (df_all['irrelevant'] == 1.0)]

            if verbose:
                #Print some information about tweets
                print(f'\n------Tweet Statistics------')
                df_relevant = df_all.loc[(df_all['irrelevant'] == 0.0)]
                df_irrelevant = df_all.loc[(df_all['irrelevant'] == 1.0)]
                perc = len(df_relevant)/len(df_labelled)

                print(f'Number of labelled tweets: {len(df_labelled)}')
                print(f'Number of all tweets: {len(df_all)}')
                print(f'Number of relevant tweets: {len(df_relevant)}')
                print(f'Number of irrelevant tweets: {len(df_irrelevant)}')
                print("Percentage relevant/irrelevant: {0:.0%} / ".format(perc) + "{0:.0%}\n".format(1 - perc))

            df_tweets = df_labelled
            df_tweets = shuffle(df_tweets)
            df_tweets.reset_index(inplace=True, drop=True) 
            filepath=f'{os.path.join(self.BASE_DIR, self.structure["preprocessedLabel"])}/labelled_tweets_{language}'
        
        #save tweets
        self.saveTweets(df_tweets, f"{filepath}.csv")
        return df_tweets, filepath


    # ------------------- write/save functions -------------------

    def saveTweets(self, df_tweets, filepath):
        try:
            with open(f'{filepath}', 'w') as f:
                df_tweets.to_csv(f, index = False)
            print(f"Tweets saved to {filepath}")

        except:
            print(f"Error when saving tweets to filepath {filepath}")
            sys.exit()


    def makeTweetDirectory(self, language):
        dir_path = os.path.join(self.BASE_DIR, f"{self.structure['rawTweets']}/{datetime.date.today().strftime('%Y-%m-%d')}-{language}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        return dir_path

    def writeExtractedTweets(self, output_list, filepath, column_names, sep):
        with open( filepath, "w", encoding="utf-8-sig") as output_file:
            dict_writer: csv.DictWriter = csv.DictWriter(
                output_file,
                fieldnames=column_names,
                delimiter=sep,
            )
            dict_writer.writeheader()
            dict_writer.writerows(output_list)

        print(f"Tweets saved to filepath {filepath}")

    def saveModel(self, model, model_type, language):
        filepath = f"{os.path.join(self.BASE_DIR, self.structure['modelData'])}/{model_type}_{language}.joblib.pkl"
        joblib.dump(model, filepath, compress=9)
        print(f"Model saved to filepath {filepath}")

    # ------------------- load/read functions -------------------

    def loadTwitterConfig(self):
        with open(os.path.join(self.BASE_DIR, self.structure['query_config'])) as file:
            config: Dict[str, Union[str, List[str], int]] = json.load(file)
            return config

    def loadTwitterSecrets(self):
        with open(os.path.join(self.BASE_DIR, self.structure['secrets'])) as file:
            secrets: Dict[str, str] = json.load(file)
        return secrets

    def readTweetDirectory(self, path):
        
        dir_all = {}
        for filename in os.listdir(path):
            if "csv" in filename:
                df_single = self.readTweetFile(os.path.join(path, filename))
                if df_single is not None:
                    dir_all[filename] = df_single

        df_all = pd.concat(dir_all.values(), ignore_index=True, sort=False)
        return df_all

    def readTweetFile(self, filepath):
        try:
            df_single = pd.read_csv(filepath, sep=";")
            return df_single
        except:
            try:
                df_single = pd.read_csv(filepath, sep=",")
                return df_single
            except:
                print("Labelled tweet file not readable.")
                return None

    def loadModel(self, model_type, language):
        try:
            model = joblib.load(f"{os.path.join(self.BASE_DIR, self.structure['modelData'])}/{model_type}_{language}.joblib.pkl")
            return model

        except:
            print(f"Model not trained: {model_type}_{language}_full{full}.joblib.pkl")
            sys.exit()

    def loadEmbedding(self):
        print("Load Embedding.")
        emb = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(self.BASE_DIR, self.structure['word2vec']), binary=True, unicode_errors='ignore')
        emb_dim = emb.vector_size

        return emb, emb_dim
        
    def loadLabelledTweets(self, language):
        path = os.path.join(self.BASE_DIR, self.structure['preprocessedLabel'])
        filepath = f"{path}/labelled_tweets_{language}.csv"
        try:
            print(f"Load data from filepath: {filepath}")
            df_tweets = self.readTweetFile(filepath)
            return df_tweets
        except:
            print(f"No preprocessed labelled data for {language} available. File does not exist: {filepath}")
    
    def loadUnlabelledTweets(self, language,  date):
        path = os.path.join(self.BASE_DIR, self.structure['preprocessedPredict'])
        filename = f'{date.strftime("%Y-%m-%d")}-{language}.csv'
        
        filepath = os.path.join(path, filename)
        print(f"Load data from filepath: {filepath}")
        df_tweets = self.readTweetFile(filepath)
        return df_tweets

    def getPredictionFilename(self, model_type, language, class_lang, date, test):
        if date is None:
            if not test:
                print("Date needs to be defined in order to find prediction file.")
                sys.exit()
            date = ""#datetime.datetime.now().strftime("%Y-%m-%d")
        else:
            date = f'{date.strftime("%Y-%m-%d")}_'
        path = os.path.join(self.BASE_DIR, self.structure["resultsPredictions"])
        filename = f'fullTweets_{date}{model_type}_lang_{language}_classLang_{class_lang}'
        if test:
            filename = f'test_{filename}'

        return os.path.join(path, f'{filename}.csv')

    def loadPredictionFile(self, model_type, language, class_lang, date, labelled):
        filepath= self.getPredictionFilename(model_type, language, class_lang, date, labelled)
        df_tweets_full = self.readTweetFile(filepath)
        if df_tweets_full is None:
            print(f"No prediction file at filepath {filepath}")
            sys.exit()

        print(f"Prediction file loaded from filepath {filepath}")
        return df_tweets_full

    def saveTweetSummary(self, df_tweets, model_type, language, class_lang, date, test):
        if date is None:
            date = ""#datetime.datetime.now().strftime("%Y-%m-%d")
        else:
            date = f'{date.strftime("%Y-%m-%d")}_'

        path = os.path.join(self.BASE_DIR, self.structure["resultsPredictions"])
        filename = f'summary_{date}{model_type}_lang_{language}_classLang_{class_lang}'
        if test:
            filename = f'test_{filename}'

        self.saveTweets(df_tweets, os.path.join(path, f'{filename}.csv'))
        

    def getNewestExtractionDatePreprocessed(self, language):
        #find newest processed data available in given language
        print(language)
        path = os.path.join(self.BASE_DIR, self.structure['preprocessedPredict'])
        dates = [datetime.datetime.strptime(("-").join(file.split("-")[:3]), '%Y-%m-%d')  for file in os.listdir(path) if language in file]
        if dates == []:
            print(f"No preprocessed data in language \'{language}\' available")
            sys.exit()

        date = max(dates)
        print(f"Predict for date {date.strftime('%Y-%m-%d')}")
        return date

    def getNewestExtractionDateRaw(self, language):

        print("Find newest extracted tweets...")
        path = os.path.join(self.BASE_DIR, self.structure['rawTweets'])
        all_subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        all_subdirs = [subdir for subdir in all_subdirs if language in subdir]
        latest_subdir = max(all_subdirs, key=os.path.getmtime)
        date = ("-").join(os.path.basename(latest_subdir).split("-")[:3])

        return date

    def getParameter(self, model_type):
            
        try:
            try:
                model_conf = self.models_conf[model_type]
                return model_conf
            except:
                with open(os.path.join(self.BASE_DIR, self.structure['models_config'])) as file:
                    self.models_conf: Dict[str, Union[str, List[str], int]] = json.load(file)
            
        except:
            print(f'No parameter for model type {model_type} found')
            return None

     
    # ------------------- NER functions -------------------

    def readNERTrainingFile(self):
        # Read file with tweets, words and tags
        filename = f"{os.path.join(self.BASE_DIR, self.structure['nerData'])}/labelled_tweets_NER.csv"
        try:
            df = pd.read_csv(filename)
            return df
        except:
            print(f"Could not read NER training file at {filename}")
            sys.exit()

    def saveNER(self, nlp):
        nlp.to_disk(f"{os.path.join(self.BASE_DIR, self.structure['nerData'])}/custom_NER")

    def loadNER(self):
        try:
            nlp = spacy.load(f"{os.path.join(self.BASE_DIR, self.structure['nerData'])}/custom_NER")
            return nlp
        except:
            print("NER not trained yet.")
            sys.exit()

    def saveConfusionMatrix(self, figure, model_type, language, class_lang):
        print("Saving Confusion Matrix")
        path = os.path.join(self.BASE_DIR, self.structure["resultsModels"])
        filename = f'confusion_matrix_{model_type}_lang_{language}_classLang_{class_lang}.png'
        
        figure.savefig(os.path.join(path, filename), dpi=400)

