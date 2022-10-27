import csv
import pandas as pd
import ast
import spacy
import random
from spacy.util import minibatch, compounding

import sys
sys.path.append('.')
from src.io_handler import IOHandler


class CustomNER:
    """Customized Named Entity Recognition model
    
    This class trains a customized NER for internal displacement tags
    """

    def __init__(self):
        print("Generate CustomNER")


    def format_train_data(self, df):
        """
        Format training data  containing full tweet, interesting words and associated tweets and rfind
        :param df: dataframe containing training tweets
        
        :return: properly formatted dataframe
        """

        # Filter dataframe for relevant columns
        df = df[['full_text', 'NER-Tags', 'NER-Words', 'rfind']]
        
        # Drop rows with NAN 
        df = df.dropna()
        tweets = []
        entities = []
        for idx, row in df.iterrows():
            # find words in tweet
            tweet = row['full_text']
            tags = ast.literal_eval(row['NER-Tags'])
            content = ast.literal_eval(row['NER-Words'])
            rfind = row['rfind']
            dic = self.find_terms(content, tags, tweet,rfind)
            tweets.append(tweet)
            entities.append(dic)
        
        #prepare resulting dataframe    
        df_result = pd.DataFrame()
        df_result["text"] = tweets
        df_result["entities"] = entities

        return df_result

    def find_terms(self, content, tags, tweet, rfind):
        """
        Find position of given words in tweet
        
        :param content: list of words contained in tweet
        :param tags: list of tags occuring in tweet
        :param tweet: raw tweet text 
        :param rfind: boolean whether word found
        """

        dic = {'entities': []}
        for word, tag in zip(content,tags):
            if rfind == False:
                start_position = tweet.find(word)
            elif rfind == True:
                start_position = tweet.rfind(word)
            end_position = start_position + len(word)
            lis = (start_position, end_position, tag)
            dic['entities'].append(lis)
        
        return(dic)

    def train(self, df_data):
        """
        Train NER on training data
        
        :param df_data: dataframe with training data
        
        :return: trained NER model
        """

        df_data = self.format_train_data(df_data)

        TRAIN_DATA = []
        for idx, row in df_data.iterrows():
            TRAIN_DATA.append((row['text'], row['entities']))
        print(TRAIN_DATA[0])

        # Setting up the pipeline and entity recognizer.
        model = None
        if model is not None:
            nlp = spacy.load(model)  # load existing spacy model
            print("Loaded model '%s'" % model)
        else:
            nlp = spacy.blank('en')  # create blank Language class
            print("Created blank 'en' model")
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner)
        else:
            ner = nlp.get_pipe('ner')

        # Add new entity labels to entity recognizer
        LABEL = ['B-trig', 'I-trig', 'B-term', 'I-term', 'B-fig', 'I-fig', 'B-time', 'I-time', 'B-unit', 'I-unit']
        for i in LABEL:
            ner.add_label(i)

        # Inititalizing optimizer
        if model is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.entity.create_optimizer()

        # Get names of other pipes to disable them during training to train # only NER and update the weights
        n_iter = 50
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            for itn in range(n_iter):
                random.shuffle(TRAIN_DATA)
                losses = {}
                batches = minibatch(TRAIN_DATA, 
                                    size=compounding(4., 32., 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    # Updating the weights
                    nlp.update(texts, annotations, sgd=optimizer, \
                               drop=0.35, losses=losses)
                print('Losses', losses)
        
        return nlp

def testNER():

    io = IOHandler()

    nlp = io.loadNER()
    
    # Test the trained model
    test_text = ["Watching how a second typhoon has wrecked Japan in just two weeks. It has already displaced millions of people. We should feel blessed that Uganda doesn't normally receive such catastrophic weather because only God knows how many lives we would have lost. #HAISHE", \
    "Evacuees displaced by Hurricane Laura can text LASHELTER to 211 for more information about the nearest shelter locations. There's more details in the link below! #HurricaneLauraRelief #HelpLouisiana https://t.co/j24ebrVCuL"]
    for i in test_text:
        doc = nlp(i)
        print(f"Entities in {i}")
        for ent in doc.ents:
            print(ent.label_, ent.text)
    
def trainNER():

    io = IOHandler()

    df_train = io.readNERTrainingFile()
    ner = CustomNER()
    nlp = ner.train(df_train)
    io.saveNER(nlp)


if __name__ == "__main__":
    trainNER()
