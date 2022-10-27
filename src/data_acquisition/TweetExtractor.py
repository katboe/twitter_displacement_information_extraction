#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import argparse
import datetime

import tweepy
from typing import Dict, List, Union

import sys
sys.path.append('.')
from src.io_handler import IOHandler


class TweetExtractor():
    """Tweet Extractor
    
    This class extracts tweets containing specified keywords.
    """

    def __init__(self, config, secrets, verbose=True):
        """Initialization of parameters
        
        :param config: config dictionary containing extraction parameters
        :param secrets: secret dictionary containing twitter credentials
        :param verbose: boolean for intermediate output
        """

        self.config = config
        self.secrets = secrets
        self.verbose = verbose

    def extractTweets(self, io, dir_path, triggers, terms):
        """Extract the tweets with given keywords

        :param io: IOHandler for saving extracted tweets
        :param dir_path: directory for saving extracted tweets
        :param triggers: list of triggers
        :param terms: list of terms
        """

        # authenticate tweepy API 
        try:
            auth: tweepy.OAuthHandler = tweepy.OAuthHandler(
                self.secrets["consumer_token"], self.secrets["consumer_secret"]
            )
            auth.set_access_token(self.secrets["access_token"], self.secrets["access_token_secret"])
        except:
            print("Twitter Credentials not valid. Check 'secrets.json' file.")

        api: tweepy.API = tweepy.API(
            auth,
            wait_on_rate_limit=True,
            wait_on_rate_limit_notify=True,
            timeout=self.config["timeout_duration"],
        )

        # extract tweets containing pairs of term and trigger words
        for trigger in triggers:

            try:
                keywords_pairs = []
                for term in  terms:
                    keywords_pairs.append(term + " " + trigger)
                
                output_list = []
                for keyword in keywords_pairs:
                    if self.verbose:
                        print(f"Starting keyword {keyword}.")
                    tweets: List[tweepy.models.Status] = [
                        tweet
                        for tweet in tweepy.Cursor(
                            api.search,
                            q=keyword + " -filter:retweets",
                            count=self.config["cursor_count"],
                            tweet_mode="extended",
                            until=datetime.date.today().strftime("%Y-%m-%d"),
                        ).items()
                    ]

                    output_list.extend([i._json for i in tweets])

            except tweepy.TweepError as ex:    
                print("Error with Twitter API. Maybe twitter credentials not valid. Check 'secrets.json' file.")
                print(ex)
                sys.exit()

            # for each trigger word, write extracted tweets to file
            filepath = f"{dir_path}/{datetime.date.today().strftime('%Y-%m-%d')}_{trigger}.csv"
            io.writeExtractedTweets(output_list, filepath, self.config["column_names"], self.config["separator"])

            if self.verbose:
                print(f"Finishing keyword {trigger}.")


def run_extraction(io, language):
    """Run script for extracting tweets
    
    :param io: IOHandler for saving extracted tweets
    :param language: language of keywords for extraction
    """

    config = io.loadTwitterConfig()
    secrets = io.loadTwitterSecrets()
    triggers =  io.getDisplacementTriggers(language)
    terms = io.getDisplacementTerms(language)
    dir_path = io.makeTweetDirectory(language)

    e = TweetExtractor(config, secrets, args.verbose)
    e.extractTweets(io, dir_path, triggers, terms)    



if __name__ == "__main__":
    # parse all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', dest='language', action='store', type=str, default='en', help='language of tweets')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)
   
    args = parser.parse_args()

    io = IOHandler()

    # check whether input language is configured
    if not io.checkLanguage(args.language):
        print(f"language not known. Choose from the following languages: {io.listAllLanguages()}")
        sys.exit()

    run_extraction(io, args.language)
    