"""
Trains a linear SVM classifier using HOG features to seperate images
containing faces from images that do not contain faces (Use sklearn.svm.LinearSVC or a similar method for classification).
"""
from sklearn.svm import SVC


def train_classifier(x_train, y_train):
    svm = SVC(C=0.0001, gamma="auto", probability=True)
    svm.fit(x_train, y_train)
    return svm
