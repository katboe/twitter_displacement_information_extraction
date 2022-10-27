import numpy as np
import pandas as pd
import json
from typing import Dict, List, Union
import os

#define which languages are contained beside english
languages = {'en', 'fr', 'es'}


CURR_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURR_DIR, '../'))

with open(os.path.join(BASE_DIR,"config/structure.json")) as file:
    structure = json.load(file)

def setupEnvironment():
    #---------keyword lists to json file---------
    
    #read and process lists of keywords
    df_trigger = pd.read_csv(os.path.join(BASE_DIR, structure["triggerWords"]), index_col = 0)
    df_termUnit = pd.read_csv(os.path.join(BASE_DIR, structure["termUnitWords"]), index_col = 0)

    #split keywords into term and unit
    df_term = df_termUnit.loc[(df_termUnit['keyword_type'] == 'structure_term') | (df_termUnit['keyword_type'] == 'person_term')]
    df_unit = df_termUnit.loc[(df_termUnit['keyword_type'] == 'structure_unit') | (df_termUnit['keyword_type'] == 'person_unit')]

    keywords = {}

    for lang in languages:
        try:
            keywords_new = {}
            keywords_new['displacement_triggers'] = {}
            for  index, trigger in df_trigger.iterrows():
                keywords_new['displacement_triggers'][trigger[f'{lang.upper()}']] = trigger["displacement_type"]

            keywords_new['displacement_terms'] = {}
            for  index, term in df_term.iterrows():
                keywords_new['displacement_terms'][ term[f'{lang.upper()}']] = term['keyword_type']

            keywords_new['displacement_units'] = {} 
            for  index, unit in df_unit.iterrows():
                keywords_new['displacement_units'][ unit[f'{lang.upper()}']] = unit['keyword_type']

            keywords[lang] = keywords_new


        except: 
            print(f"keywords not available in language {lang}")

    with open(os.path.join(BASE_DIR, structure["keywords"]), "w") as f:
        json.dump(keywords, f)  
    
    print("Keywords copied to JSON file")

    #---------------------------------------------

if __name__ == "__main__":
    setupEnvironment()