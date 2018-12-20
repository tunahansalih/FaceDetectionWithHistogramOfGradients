"""
Trains a linear SVM classifier using HOG features to seperate images
containing faces from images that do not contain faces (Use sklearn.svm.LinearSVC or a similar method for classification).
"""
from sklearn.svm import LinearSVC

def train_classifier(x_train, y_train):
    svm = LinearSVC()
    svm.fit(x_train, y_train)
    return svm