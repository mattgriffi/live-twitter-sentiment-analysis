
import tweepy

from data import DataSet
from voting_classifier import VotingClassifier


class KeywordStreamListener(tweepy.StreamListener):

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def on_status(self, status):
        text = status.text.strip()
        classification = self.classifier.classify(
            DataSet.find_features(text, DataSet.get_feature_list()))
        if self.classifier.get_most_recent_confidence() < 0.8:
            classification = 'unsure'
        print(status.text.strip())
        print(f"Classification: {classification}\n")

    def on_error(self, code):
        print(f"ERROR: {code}")


with open('keys.txt') as file:
    keys = [line.strip() for line in file.readlines()]

auth = tweepy.OAuthHandler(keys[0], keys[1])
auth.set_access_token(keys[2], keys[3])

api = tweepy.API(auth)

classifier = VotingClassifier()

stream_listener = KeywordStreamListener(classifier)
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
stream.filter(track=['Microsoft'])


# public_tweets = api.home_timeline()
# for tweet in public_tweets:
#     print(tweet.text)
#     classification = classifier.classify(DataSet.find_features(tweet.text, data.all_features))
#     if classifier.get_most_recent_confidence() < 1.0:
#         classification = 'unsure'
#     print(f"Classification: {classification}\n")
