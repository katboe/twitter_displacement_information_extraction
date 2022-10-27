import sys
sys.path.append('.')

import argparse
from src.io_handler import IOHandler
from src.setup import setupEnvironment
from src.data_preprocessing.TweetPreprocessor import run_preprocessing
from src.data_acquisition.TweetExtractor import TweetExtractor
from src.data_classification.TweetClassifier import trainClassifier



def run_trainClassifier(class_lang, model_type, verbose=False, cv=False):
    labelled = True

    io = IOHandler()

    #------------------------------------------
    # preprocess
    #------------------------------------------
   
    if not io.checkLanguage(class_lang):
        print(f"language not known. Choose from the following languages: {io.listAllLanguages()}")
        sys.exit()

    run_preprocessing(io, class_lang, labelled, None, verbose)

    #------------------------------------------
    # train
    #------------------------------------------
   
    #check whether model is configured
    if not io.checkModels(model_type):
        print(f"Classifier language not known. Choose from the following languages: {io.listAllModels()}")
        sys.exit()


    trainClassifier(io, class_lang, model_type, verbose, cv)


if __name__ == "__main__":
	run_trainClassifier('en', 'svm', True, True)

