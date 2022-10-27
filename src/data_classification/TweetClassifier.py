#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
import statistics as stats

import argparse

import sklearn as skl
from sklearn import linear_model
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

from tpot import TPOTClassifier
from tpot import TPOTRegressor

import sys
sys.path.append('.')
from src.io_handler import IOHandler
from src.data_classification.InformationExtractor import InformationExtractor

import itertools

import datetime

SEED = 42

#----------------------------------------- Tweet Classifier Definition------------------------------------------------
class TweetClassifier():
    """Tweet Classifier
    
    This class trains and tests a binary tweet classifier and predicts labels for tweets.
    """

    def __init__(self, model_type='svm', param=None, emb=None, emb_dim=None, verbose=True):
        """Initialization of parameters
        
        :param model_type: type of classifier model
        :param param: dictionary of parameters for given model type
        :param emb: embedding for classification (if not bayes classifier)
        :param emb_dim: dimension  of embedding (if not bayes classifier)
        :param verbose: boolean for intermediate output
        :param cross_valid: boolean for cross validation
        """

        self.model_type = model_type
        self.verbose = verbose
        self.param = param

        # check whether embedding is needed
        if not self.model_type == 'bayes':
            if emb == None:
                print(f"Classifier with model_type {self.model_type} requires a word embedding.")
                sys.exit()
            
            else:
                self.emb, self.emb_dim = emb, emb_dim
            

    def train(self, df_train, cross_valid):
        """Train the binary classifier
        
        :param df_train: dataframe containing labelled training tweets
        :param cross_valid: boolean for cross validation

        :return: trained classifier model
        """

        # prepare training data
        X = self.transformDataToFeatures(df_train)
        y = df_train['irrelevant']


        if cross_valid:
            # if cross validation is set, test classifier with a Stratified KFold
            print("------Cross Validation------")

            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED)
            Bal_acc_scores = []
            counter = 0

            for train_index, test_index in cv.split(X, y):
                counter = counter + 1

                X_train, X_val = X.iloc[train_index], X.iloc[test_index]
                y_train, y_val = y.iloc[train_index], y.iloc[test_index]

                model = self.buildModel()
                
                if self.model_type == 'bayes':
                    test_ys_pred = []
                    model =  model.train(X_train['sentences'].tolist())
                    for feat in X_val['sentences']:
                        test_ys_pred.append((int)(model.classify(feat[0])))
                else:
                    model.fit(X_train, y_train.values.ravel())
                    test_ys_pred = model.predict(X_val)
               
                
                Bal_acc_scores.append(self.bal_score(y_val, test_ys_pred))
             
            print('Average Balanced Accuracy Score: {0:.2%}'.format(stats.mean(Bal_acc_scores)) + '; Standard Deviation: {0:.2%}'.format(stats.stdev(Bal_acc_scores)))

        # build model
        model = self.buildModel()

        # train model
        if self.model_type == 'bayes':
            print(X.shape)
            model =  model.train(X['sentences'].tolist())
        else:
            model.fit(X, y.values.ravel())

        return model

    def test(self, model, df_test):
        """Test the binary classifier
        
        :param model: binary classifier
        :param df_test: dataframe containing labelled testing tweets

        :return: tweets containing triggers, prediction labels
        """

        df_test, test_ys_pred  = self.predict(model, df_test)

        #get labels
        y_test = df_test['irrelevant']
        #compute test score
        test_ys_pred = np.asarray(test_ys_pred)
        print('Test Score (Balanced Accuracy): {0:.2%}\n'.format(self.bal_score(y_test, test_ys_pred.astype(int))))
        

        return df_test, test_ys_pred
        
    def predict(self, model, df_tweets):
        """Test the binary classifier
        
        :param model: binary classifier
        :param df_tweets: dataframe containing tweets

        :return: tweets containing triggers, prediction labels
        """

        #prepare tweets
        df_tweets = df_tweets[(df_tweets['trigger'].str.len() > 2)]
        if len(df_tweets) == 0:
            print("No tweets with trigger words found.")
            sys.exit()

        X_test = self.transformDataToFeatures(df_tweets)

        #predict labels
        if self.model_type == 'bayes':
            test_ys_pred = []
            for feat in X_test['sentences']:
                test_ys_pred.append((int)(model.classify(feat[0])))
        else:
            test_ys_pred = model.predict(X_test)
        
        return df_tweets, test_ys_pred

    def buildModel(self):
        """Build the model
        
        :return: binary classifier
        """

        #create Model with specified or default parameters
        if self.model_type == 'linear':
            try:
                loss_func = self.param["loss"]
            except:
                loss_func = "hinge"
                print("Using default parameters.")

            model = linear_model.SGDClassifier(loss=loss_func)

        elif self.model_type == 'randomforest':
            try:
                max_depth = self.param["max_depth"]
                random_state = self.param["random_state"]
                n_estimators = self.param["n_estimators"]
            except:
                max_depth = 10
                random_state = 0
                n_estimators = 10
                print("Using default parameters.")
            model = RandomForestClassifier(max_depth=max_depth, random_state=random_state, n_estimators=n_estimators)

        elif self.model_type == 'svm':
            try:
                kernel = self.param["kernel"]
            except:
                kernel = "rbf"
                print("Using default parameters.")
            model = svm.SVC(kernel='rbf')

        elif self.model_type == 'tpot':
            try: 
                verbosity = self.param["verbosity"]
                max_time_mins = self.param["max_time_mins"]
                max_eval_time_mins = self.param["max_eval_time_mins"]
                population_size = self.param["population_size"]
            except:
                verbosity=2 
                max_time_mins=2
                max_eval_time_mins=0.04
                population_size=15
                print("Using default parameters.")

            model = TPOTClassifier(verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=15)

        elif self.model_type == 'bayes':
            model =  NaiveBayesClassifier#.train(X_train['sentences'].tolist())

        else:
            print(f"Model {self.model_type} was not implemented.")
            sys.exit()

        return model

    def transformDataToFeatures(self, df_tweets):
        """Transform the tweet data to features
        
        :param df_tweets: dataframe containing tweets

        :return: dataframe containing features of tweets
        """

        #get tweet text
        sentences = df_tweets['full_text_preprocessed']
        sentences_tokens = [word_tokenize(tweet)  if not self.isNaN(tweet) else "" for tweet in sentences]
        
        if not self.model_type == 'bayes':

            #embed tweet text as vectors of emb_dim size
            X_sen = [self.embed2featureWords(row, self.emb, self.emb_dim) for row in sentences_tokens]

            #features consist of embedding vector and 'twitter user verified' status
            count = 0
            X_sen_full = []
            user_info = []
            for index, u in df_tweets.iterrows():
                user = eval(u["user"])
                sen = np.append(X_sen[count], user["verified"])
                count += 1
                X_sen_full.append(sen)

            X = pd.DataFrame(X_sen)

        else:
            #embed tweets as word occurence histograms if classifier is 'bayes'
            wordlist = nltk.FreqDist(list(itertools.chain.from_iterable(sentences_tokens)))
            word_features = wordlist.keys()

            sentences_tokens = []
            for index, sen in df_tweets.iterrows():
                tweet = word_tokenize(sen['full_text_preprocessed'])
                tweet_words = set(tweet)

                features = {}
                for word in word_features:
                    features['contains(%s)' % word] = (word in tweet_words)

                sentences_tokens.append((features, sen['irrelevant']))
            
            X = pd.DataFrame()
            X['sentences'] = sentences_tokens

        return X

    def embed2featureWords(self, sentence, emb, emb_dim):
        """Transform the tweet data to features
        
        :param sentence: list of word tokens
        :param emb: dictionary with word embedding
        :param emb_dim: dimension of word embedding

        :return: sentence embedding
        """

        #for each sentence compute mean of all word vectors
        featVec = np.zeros(emb_dim)
        count = 0
        for word in sentence:
            try:
                featVec += emb[word]
                count += 1
            except KeyError:
                count = count

        if count == 0:
            return featVec
        else:
            return featVec/(count)


    def isNaN(self, string):
        return string != string

    def bal_score(self, ys_true: np.ndarray, ys_pred: np.ndarray):
        return skl.metrics.balanced_accuracy_score(ys_true, ys_pred)

#------------------------------------ Tweet Classifier Definition Finished---------------------------------------------

#----------------------------------------------- Classifier Scripts-----------------------------------------------------

def trainClassifier(io, class_lang, model_type, verbose, cross_valid):
    """Train the classifier with the given parameters
        
    :param io: IOHandler
    :param class_lang: llanguage of training tweets
    :param model_type: type of classification model
    :param verbose: boolean for intermediate output
    :param cross_valid: boolean for cross validation

    """

    #load labelled training tweets of classifier language
    try: 
        df_tweets = io.loadLabelledTweets(class_lang)

        if verbose:
            # some statistics
            df_noTrigger = df_tweets[(df_tweets['trigger'].str.len() == 2)]
            df_withTrigger = df_tweets[(df_tweets['trigger'].str.len() > 2)]
            perc1 = len(df_withTrigger)/len(df_tweets)
            df_relevant_withTrigger = df_withTrigger.loc[(df_withTrigger['irrelevant'] == 0.0)]
            df_irrelevant_withTrigger = df_withTrigger.loc[(df_withTrigger['irrelevant'] == 1.0)]
            perc2 = len(df_relevant_withTrigger)/len(df_withTrigger)

            print('\n------Training Data Statistics------')
            print(f'Tweets with trigger word: {len(df_withTrigger)}')
            print(f'Tweets without trigger word: {len(df_noTrigger)}')
            print("Percentage with Trigger/without Trigger: {0:.0%} / ".format(perc1) + "{0:.0%}".format(1 - perc1))
            print(f'Tweets with trigger relevant: {len(df_relevant_withTrigger)}')
            print(f'Tweets with trigger irrelevant: {len(df_irrelevant_withTrigger)}')
            print("Percentage relevant/irrelevant: {0:.0%} / ".format(perc2) + "{0:.0%}\n".format(1 - perc2))

    except:
        print(f"Error while trying to read labelled preprocessed tweets in language \'{class_lang}\'")
        sys.exit()
    
    # HARD RULE: remove tweets without trigger words
    df_withTrigger = df_tweets[(df_tweets['trigger'].str.len() > 2)]

    #load word embedding
    emb, emb_dim = None, None
    if not model_type == 'bayes':
        emb, emb_dim = io.loadEmbedding()

    #load parameters for model type
    param = io.getParameter(model_type)
    clf = TweetClassifier(model_type, param, emb, emb_dim, verbose)
    
    #train/test classifier
    labels_trigger = df_withTrigger['irrelevant']
    df_train, df_test =  train_test_split(df_withTrigger, test_size=0.2, random_state=SEED, stratify=labels_trigger)
    print(f"Train/Test split: {len(df_train)}/{len(df_test)}")
    model = clf.train(df_train, False)
    df_result, labels = clf.test(model, df_test) 

    makeConfusionMatrix(io, df_result['irrelevant'], labels, model_type, class_lang, class_lang, False)

    # train classifier on full training set
    model = clf.train(df_withTrigger, cross_valid)
    print("Model trained\n")

    io.saveModel(model, model_type, class_lang)


def testClassifier(io, language, class_lang, model_type, verbose):
    """Test the classifier with the given parameters
    
    :param io: IOHandler
    :param language: language of testing tweets    
    :param class_lang: language of trained classifier
    :param model_type: type of classification model
    :param verbose: boolean for intermediate output

    """

    if language == class_lang:
        print("\nWARNING: Can't test the classifier on the same labelled tweets that it was trained on. Train and test classifier instead.\n")

    #load labelled test tweets of specified language
    try: 
        df_test = io.loadLabelledTweets(language)

        if df_test is None:
            sys.exit()

    except:
        print(f"No labelled data for language \'{language}\'")
        sys.exit()
    
    #load word embedding
    emb, emb_dim = None, None
    if not model_type == 'bayes':
        emb, emb_dim = io.loadEmbedding()

    #load parameters for model_type
    param = io.getParameter(model_type)
    #load the pretrained model
    model = io.loadModel(model_type, class_lang)

    #test the model on the given testing tweets
    clf = TweetClassifier(model_type, param, emb, emb_dim, verbose)
    df_result, labels = clf.test(model, df_test)

    makeConfusionMatrix(io, df_result['irrelevant'], labels, model_type, language, class_lang, True)
    
    #save prediction results
    df_result["predicted_irrelevant"] = np.asarray(labels).astype(int)
    filepath = io.getPredictionFilename(model_type, language, class_lang, None, True)
    io.saveTweets(df_result, filepath)

    #extract relevant information
    ie = InformationExtractor()
    df_results = ie.extractInformation(df_result, io.loadNER(), language)

    #save all results
    io.saveTweetSummary(df_results, model_type, language, class_lang, None, True)

def makeConfusionMatrix(io, y_test, test_ys_pred, model_type, language, class_lang, show):
    tn, fp, fn, tp = confusion_matrix(y_test, test_ys_pred).ravel()
    conf = np.asarray([[tp, fp], [fn, tn]])
    ax = sn.heatmap(conf, annot=True, annot_kws={"size": 16})

    ax.set(xlabel="Actual Irrelevant Label", ylabel = "Predicted Irrelevant Label")
    ax.set_title('Confusion matrix')
    figure = ax.get_figure() 
    figure.canvas.set_window_title('Confusion Matrix')  
    io.saveConfusionMatrix(figure, model_type, language, class_lang)
    if show:
        plt.show()


def classify(io, language, class_lang, model_type, verbose, date):
    """Predict labels for unlabelled tweets of specified language extracted on specified date
    
    :param io: IOHandler
    :param language: language of testing tweets    
    :param class_lang: language of trained classifier
    :param model_type: type of classification model
    :param verbose: boolean for intermediate output
    :param date: date of tweet extraction
    """

    #read unlabelled tweets of specified language
    try: 
        df_tweets = io.loadUnlabelledTweets(language, date)

    except:
        print(f"No data for language \'{language}\' extracted.")
        sys.exit()
    
    #load word embedding
    emb, emb_dim = None, None
    if not model_type == 'bayes':
        emb, emb_dim = io.loadEmbedding()

    #load parameters for model_type
    param = io.getParameter(model_type)
    #load the pretrained model
    model = io.loadModel(model_type, class_lang)
    
    #predict labels for tweets
    clf = TweetClassifier( model_type, param, emb, emb_dim, verbose)
    df_result, labels = clf.predict(model, df_tweets)

    #save prediction results
    df_result["predicted_irrelevant"] = np.asarray(labels).astype(int)
    filepath = io.getPredictionFilename(model_type, language, class_lang, date, False)
    io.saveTweets(df_result, filepath)

    #extract relevant information
    ie = InformationExtractor()
    df_results = ie.extractInformation(df_result, io.loadNER(), language)

    #save all results
    io.saveTweetSummary(df_results, model_type, language, class_lang, date, False)


if __name__ == "__main__":

    #parse all arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--classifier', dest='model_type', action='store', type=str, default='svm', help='name of classification model')
    parser.add_argument('-m', '--mode', dest='mode', action='store', type=str, default='train', help='mode of classifier: train / test / predict')
    parser.add_argument('-l', '--language', dest='language', action='store', type=str, default='en', help='language of prediction tweets')
    parser.add_argument('--classifierLanguage', dest='class_lang', action='store', type=str, default='en', help='language of training tweets')

    parser.add_argument('--cv', dest='cv', action='store_true', default=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)

    parser.add_argument('--date', dest='date', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'))

    args = parser.parse_args()

    io = IOHandler()

    #check whether input language is configured
    if not io.checkLanguage(args.language):
        print(f" Language not known. Choose from the following languages: {io.listAllLanguages()}")
        sys.exit()

    #check whether input language is configured
    if not io.checkLanguage(args.class_lang):
        print(f"Classifier language not known. Choose from the following languages: {io.listAllLanguages()}")
        sys.exit()

    #check whether model is configured
    if not io.checkModels(args.model_type):
        print(f"Classifier language not known. Choose from the following languages: {io.listAllModels()}")
        sys.exit()


    if args.mode == 'train':
        trainClassifier(io, args.class_lang, args.model_type, args.verbose, args.cv)

    elif args.mode == 'test':
        #if model doesn't exist already, train it first
        if not io.checkModelExists(args.model_type, args.class_lang):
            print("Classifier first needs to be trained.")
            trainClassifier(io, args.class_lang, args.model_type, args.verbose, args.cv)            
        

        testClassifier(io, args.language, args.class_lang, args.model_type, args.verbose)

    elif args.mode == 'predict':
        #if model doesn't exist already, train it first
        if not io.checkModelExists(args.model_type, args.class_lang):
            print("Classifier first needs to be trained.")
            trainClassifier(io, args.class_lang, args.model_type, args.verbose, args.cv)            
        
        #if no date specified, find newest extration date for specified language
        if args.date == None:
            args.date = io.getNewestExtractionDatePreprocessed(args.language)  

        classify(io, args.language, args.class_lang, args.model_type, args.verbose, args.date)

    else:
        print(f"Specified mode {args.mode} does not exist")