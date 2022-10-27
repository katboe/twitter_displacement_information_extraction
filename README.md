# Social Media Monitoring Tool for Internal Displacement Data from Twitter

## Description
This repository contains the python implementation for training a model that tracks relevant tweets for internal displacement monitoring and extracts their important information. The project is a collaboration with the NGO IDMC and was performed as part of the Data Science Hackathon "Hack4Good" by the Analytics Club at ETH Zurich. General information on the project can be found in the general report (Report_General.pdf) and technical details are explained in the technical report (Report_Technical.pdf). A high-level blogpost on this project can be found on IDMC's [Expert Opinion](https://www.internal-displacement.org/expert-opinion/exploring-machine-learning-workflows-to-filter-out-internal-displacement-events-on) website.

## Installation and Prerequisits
1. Setup the environment by installing all requirements: `pip install -r requirements.txt`.

2. Download pretrained Word2Vec model (<https://drive.google.com/file/d/1lw5Hr6Xw0G0bMT1ZllrtMqEgCTrM7dzc/view>) and store it in `data/utils_data`.
 
3. Setup Twitter Developer Account and save credentials in `config/data_acquisition/secrets.json`
    1. Apply for Twitter Developer Account: <https://developer.twitter.com/en/apply-for-access>
    2. Follow Application Process
    3. After succesfull application, visit Developer Dashboard: <https://developer.twitter.com/en/dashboardCreate>
    4. Go to App Dashboard and create an app
    5. After succefull app creation, save API consumer and access keys and tokens to `config/data_acquisition/secrets.json`

4. Setup the internal displacement keywords (from csv in 'data/utils_data')
    `python src/setup.py`

5. Train the classifier on included labelled english tweets
    `python src/run_trainDefaultClassifier.py`

6. Train the custom Named Entity Recognition Model on the included training data
    `python src/run_trainNER.py`
    
## Usage

### Full Pipeline
Default Parameter:
```python src/run_fullClassificationPipeline.py```

Configure Parameter:
```python 
python src/run_fullClassificationPipeline.py  
        -l en | es | fr   
        -c svm | randomforest | linear | bayes
```

### Single Steps of the Pipeline
The steps of the pipeline can also be executed individually.

#### Extraction
This script extracts the tweets containing a combination of keywords and saves them in 'data/raw_data/raw_tweets'.
```python 
python src/data_acquisition/TweetExtractor.py
        --language en | es | fr   
        --verbose
```

#### Preprocessing
This script preprocesses the tweets saved in 'data/raw_data/raw_tweets' or 'data/raw_data/labelled_tweets' and saves the preprocessed tweets in 'data/preprocessed_data/predict_tweets' or 'data/preprocessed_data/labelled_tweets' respectively.
```python 
python src/data_preprocessing/TweetPreprocessor.py
        --language en | es | fr   
        --labelled
        --verbose
        --date YYYY-mm-dd (optional, default is newest extraction date)
```

#### Classification
These scripts train, test and use a binary classifier on the data in 'data/preprocessed_data' and saves/loads the classifier from 'data/models'.

Classify preprocessed unlabelled tweets. The predictions are saved in 'results/predictions', the 'summary' file contains the extracted relevant information.
```python 
python src/data_classification/TweetClassifier.py
        --mode predict
        --language en | es | fr   
        --classifier svm | randomforest | linear | bayes
        --date YYYY-mm-dd (optional, default is newest preprocessed date)
        --classifierLanguage en
        --verbose
```

Train new classifier on preprocessed labelled tweets:
```python 
python src/data_classification/TweetClassifier.py
        --mode train
        --classifierLanguage en
        --classifier svm | randomforest | linear | bayes
        --verbose
        --cv
```

Test classifier on preprocessed labelled tweets. The predictions are saved with a test-prefix in 'results/predictions', the 'summary' file contains the extracted relevant information.
```python 
python src/data_classification/TweetClassifier.py
        --mode test
        --language en | es | fr   
        --classifier svm | randomforest | linear | bayes
        --classifierLanguage en
        --verbose
```


#### Information Extraction

Extracting Information of already predicted tweets. However, this step is done automatically when predicting.
```python 
python src/data_classification/InformationExtractor.py
        --language en | es | fr   
        --classifier svm | randomforest | linear | bayes
        --classifierLanguage en
        --date YYYY-mm-dd (necessary)
        --labelled (if prediction of test tweets)
```


## Configuration

- More languages can be used by including their keywords to the keyword csv files in 'data/utils_data'.  

- Models parameters can be configured in the model config: 'config/data_classification/model_config.json'

- Translator implementation in 'src/data_preprocessing/Translator.py' can/should be modified e.g. with a google translator using a valid account.
