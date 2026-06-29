from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC

def get_knn(n_neighbors: int = 5):
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def get_naive_bayes():
    return GaussianNB()

def get_gaussian_process(random_state: int = 42):
    return GaussianProcessClassifier(random_state=random_state)

def get_svm(C: float = 1.0, kernel: str = 'rbf', random_state: int = 42):
    return SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
