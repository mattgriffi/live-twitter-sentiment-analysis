
import tweepy

from data import DataSet
from voting_classifier import VotingClassifier


with open('keys.txt') as file:
    keys = [line.strip() for line in file.readlines()]

auth = tweepy.OAuthHandler(keys[0], keys[1])
auth.set_access_token(keys[2], keys[3])

api = tweepy.API(auth)

data = DataSet.get_data()
classifier = VotingClassifier()

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)
    classification = classifier.classify(DataSet.find_features(tweet.text, data.all_features))
    if classifier.get_most_recent_confidence() < 0.8:
        classification = 'unsure'
    print(f"Classification: {classification}\n")
