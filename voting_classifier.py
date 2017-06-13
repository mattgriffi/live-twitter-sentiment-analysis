"""This classifier takes a list of machine learning classification algorithms. It is used to
classify a text, then return a majority vote of the classifications determined by the multiple
classifiers."""


from nltk import ClassifierI


class VotingClassifier(ClassifierI):
    pass
