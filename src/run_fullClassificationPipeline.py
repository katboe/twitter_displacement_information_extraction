import sys
sys.path.append('.')

import argparse
from src.io_handler import IOHandler
from src.setup import setupEnvironment
from src.data_preprocessing.TweetPreprocessor import run_preprocessing
from src.data_acquisition.TweetExtractor import TweetExtractor
from src.data_classification.TweetClassifier import trainClassifier
from src.data_classification.TweetClassifier import classify


def run_fullClassificationPipeline(language, model_type, class_lang='en', verbose=False, cv=False):
    labelled = False

    io = IOHandler()

    #------------------------------------------
    # extract Tweets
    #------------------------------------------
    config = io.loadTwitterConfig()
    secrets = io.loadTwitterSecrets()
    e = TweetExtractor(config, secrets, verbose)
    
    triggers =  io.getDisplacementTriggers(language)
    terms = io.getDisplacementTerms(language)
    dir_path = io.makeTweetDirectory(language)
    e.extractTweets(io, dir_path, triggers, terms)

    #------------------------------------------
    # preprocess Tweets
    #------------------------------------------
   
    if not io.checkLanguage(language):
        print(f"language not known. Choose from the following languages: {io.listAllLanguages()}")
        sys.exit()

    if not io.checkLanguage(class_lang):
        print(f"language not known. Choose from the following languages: {io.listAllLanguages()}")
        sys.exit()

    date = io.getNewestExtractionDateRaw(language)
    run_preprocessing(io, language, labelled, date, verbose)
 
    #------------------------------------------
    # classify Tweets and extract Information
    #------------------------------------------
   
    #check whether model_type is configured
    if not io.checkModels(model_type):
        print(f"Classifier language not known. Choose from the following languages: {io.listAllModels()}")
        sys.exit()

    #check whether model is trained
    if not io.checkModelExists(model_type, class_lang):
            print("Classifier first needs to be trained.")
            trainClassifier(io, class_lang, model_type, verbose, cv)            
            
    date = io.getNewestExtractionDatePreprocessed(language)  

    classify(io, language, class_lang, model_type, verbose, date)

    #------------------------------------------
    # make Visualizations
    #------------------------------------------

if __name__ == "__main__":
    #parse all arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--classifier', dest='model_type', action='store', type=str, default='svm', help='name of classification model')
    parser.add_argument('-l', '--language', dest='language', action='store', type=str, default='en', help='language of prediction tweets')
    
    args = parser.parse_args()

    run_fullClassificationPipeline(args.language, args.model_type, class_lang='en', verbose=True, cv=True)

