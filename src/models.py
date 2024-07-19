from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


class BaseModel():
    name = "BaseModel"
        
    def fit(self, X, y) -> None:
        raise NotImplementedError()

    def predict(self, X) -> None:
        raise NotImplementedError()
    
    def eval(self, X, y) -> None:
        raise NotImplementedError()


class ModelKNN(BaseModel):
    name = "ModelKNN"

    def __init__(self) -> None:
        super().__init__()
        self.model = KNeighborsClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def eval(self, X, y):
        return self.model.score(X, y)


class ModelSVM(BaseModel):
    name = "ModelSVM"

    def __init__(self) -> None:
        super().__init__()
        self.model = svm.SVC(kernel='poly')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def eval(self, X, y):
        return self.model.score(X, y)
