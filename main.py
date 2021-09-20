import twitter
import pandas as pd
import tweepy as tw
import time
from configparser import ConfigParser
import re
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

config_parser = ConfigParser()
config_parser.read("config.ini")

consumer_key = config_parser.get('twitter_api', 'consumer_key')
consumer_secret = config_parser.get('twitter_api', 'consumer_secret')
access_token_key = config_parser.get('twitter_api', 'access_token_key')
access_token_secret = config_parser.get('twitter_api', 'access_token_secret')

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


def build_test_set(search_keyword):
    try:

        SLEEP_TIME = 900/180
        TOTAL_TWEETS = 3000
        MAX_TWEET_REQ = 1500

        tweets_fetched = []

        for i in range(2):
            tweets = tw.Cursor(api.search, q=search_keyword, lang="en").items(1500)
            for tweet in tweets:
                tweets_fetched.append({"text": tweet.text, "label": None})
            time.sleep(SLEEP_TIME)

        return tweets_fetched

    except:
        print("Unfortunately, something went wrong when acquiring tweets..")
        return None


class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])

    def process_tweets(self, list_of_tweets):
        processedTweets = []
        for tweet in list_of_tweets:
            processedTweets.append((self._process_tweet(tweet["text"]), tweet["label"]))
        return processedTweets

    def _process_tweet(self, tweet):
        tweet = tweet.lower()
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = word_tokenize(tweet)
        return [word for word in tweet if word not in self._stopwords]


def build_vocabulary(preprocessed_training_data):
    all_words = []

    for (words, sentiment) in preprocessed_training_data:
        all_words.extend(words)

    word_list = nltk.FreqDist(all_words)
    word_features = word_list.keys()

    return word_features


def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

# ------------------------------------------------------------------------


search_terms = ["Justin Trudeau", "Jagmeet Singh", "Erin O'Toole"]
result_tweet = ""

for search_term in search_terms:
    test_data_set = build_test_set(search_term)

    corpusFile = "/Users/amandatang/PycharmProjects/twitter_sentiment/corpus.csv"
    tweet_data_file = "/Users/amandatang/PycharmProjects/twitter_sentiment/tweetDataFile.csv"

    df = pd.read_csv(tweet_data_file, encoding='latin1', names=['tweet_id', 'text', 'label', 'topic'])
    training_data = df.to_dict(orient='records')

    tweet_processor = PreProcessTweets()
    preprocessed_training_set = tweet_processor.process_tweets(training_data)
    preprocessed_test_set = tweet_processor.process_tweets(test_data_set)

    word_features = build_vocabulary(preprocessed_training_set)
    training_features = nltk.classify.apply_features(extract_features, preprocessed_training_set)

    n_bayes_classifier = nltk.NaiveBayesClassifier.train(training_features)

    n_bayes_result_labels = [n_bayes_classifier.classify(extract_features(tweet[0])) for tweet in preprocessed_test_set]

    print(n_bayes_result_labels)
    print('Neutral: ' + str(n_bayes_result_labels.count('neutral')))
    print('Positive: ' + str(n_bayes_result_labels.count('positive')))
    print('Negative: ' + str(n_bayes_result_labels.count('negative')))

    if n_bayes_result_labels.count('positive') > n_bayes_result_labels.count('negative'):
        result_tweet += "Overall Positive Sentiment for " + search_term + "\n"
        result_tweet += "Positive Sentiment Percentage = " + \
                        str(100 * n_bayes_result_labels.count('positive') / len(n_bayes_result_labels)) + "%\n"
    else:
        result_tweet += "Overall Negative Sentiment for " + search_term + "\n"
        result_tweet += "Negative Sentiment Percentage = " + \
                        str(100 * n_bayes_result_labels.count('negative') / len(n_bayes_result_labels)) + "%\n"

# Send tweet with sentiment results
api.update_status(result_tweet)
