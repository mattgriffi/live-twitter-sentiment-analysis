import multiprocessing


def main():

    keyword = 'hate'
    classifier = VotingClassifier()
    queue = multiprocessing.Queue()

    tweepy_process = multiprocessing.Process(target=start_tweepy,
                                             args=(keyword, classifier, queue))
    matplotlib_process = multiprocessing.Process(target=start_matplotlib,
                                                 args=(queue,))

    tweepy_process.start()
    matplotlib_process.start()
    tweepy_process.join()
    matplotlib_process.join()


if __name__ == '__main__':
    main()
