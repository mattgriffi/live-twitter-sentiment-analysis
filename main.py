import logging
import multiprocessing

from graphing import graph
from votingclassifier import VotingClassifier
from streaming import start_tweepy

logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s - %(funcName)-30s - %(message)s')

KEYWORD = 'happy'


def main():

    classifier = VotingClassifier()
    queue = multiprocessing.Queue()

    streaming_process = multiprocessing.Process(target=start_tweepy,
                                                args=(KEYWORD, classifier, queue))
    graphing_process = multiprocessing.Process(target=graph, args=(queue, KEYWORD))

    streaming_process.start()
    graphing_process.start()
    graphing_process.join()
    streaming_process.terminate()


if __name__ == '__main__':
    main()
