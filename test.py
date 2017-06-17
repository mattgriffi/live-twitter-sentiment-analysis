import nltk

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

from classifier_trainer import ClassifierTrainer
from data import DataSet


def test_algorithm_accuracy(algorithm_list, testing_set):
    """Tests and prints the accuracy of each algorithm in the list."""
    for algorithm in algorithm_list:
        print(f"\n{algorithm._clf.__class__.__name__:<20} "
              f"{nltk.classify.accuracy(algorithm, testing_set)}")

        for i in range(10):
            print(f"Test {i}: {algorithm._clf.__class__.__name__} -> "
                  f"{algorithm.classify(testing_set[i][0])} : {testing_set[i][1]}")


data = DataSet.get_data()

data2 = DataSet.get_data()

data3 = DataSet.get_data()

algorithm_list = [MultinomialNB, BernoulliNB, LogisticRegression, SGDClassifier,
                  LinearSVC, NuSVC]
trained_algorithm_list = ClassifierTrainer.get_trained_classifiers(algorithm_list)
trained_algorithm_list2 = ClassifierTrainer.get_trained_classifiers(algorithm_list)
test_algorithm_accuracy(trained_algorithm_list2, data.test_set)
