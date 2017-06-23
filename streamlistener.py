import logging
import tweepy


class KeywordStreamListener(tweepy.StreamListener):

    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def on_status(self, status):
        try:
            # Try to get the full text from extended tweets
            text = status.extended_tweet['full_text'].strip()
        except AttributeError:
            # Fall back to normal text field for non-extended tweets
            text = status.text.strip()
        if not text.startswith('RT @'):
            self.queue.put(text.strip())

    def on_error(self, code):
        logging.error(code)
