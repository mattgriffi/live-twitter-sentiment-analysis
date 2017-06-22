import tweepy

from data import DataSet


class KeywordStreamListener(tweepy.StreamListener):

    def __init__(self, classifier, queue):
        super().__init__()
        self.classifier = classifier
        self.queue = queue

    def on_status(self, status):
        try:
            # Try to get the full text from extended tweets
            text = status.extended_tweet['full_text'].strip()
        except AttributeError:
            # Fall back to normal text field for non-extended tweets
            text = status.text.strip()
        if not text.startswith('RT @'):
            classification = self.classifier.classify(
                DataSet.find_features(text, DataSet.get_feature_list()))
            if self.classifier.get_most_recent_confidence() < 0.7:
                classification = 'unsure'
            if classification != 'unsure':
                self.queue.put((text.strip(), classification))

    def on_error(self, code):
        print(f"ERROR: {code}")


def start_tweepy(keyword, classifier, queue):

    with open('keys.txt') as file:
        keys = [line.strip() for line in file.readlines()]

    auth = tweepy.OAuthHandler(keys[0], keys[1])
    auth.set_access_token(keys[2], keys[3])

    api = tweepy.API(auth)

    stream_listener = KeywordStreamListener(classifier, queue)
    stream = tweepy.Stream(auth=api.auth, listener=stream_listener)

    stream.filter(track=[keyword], stall_warnings=True)
