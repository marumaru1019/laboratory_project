from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


class SVM_MODEL:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = LinearSVC(random_state=43)

    def fit(self):
        # モデルの学習。fit関数で行う。
        learn_model = self.model.fit(self.X_train, self.y_train)
        return learn_model

    def score(self, learn_model):
        # テストデータで試した正解率を返す
        accuracy = learn_model.score(self.X_test, self.y_test)
        print(f"正解率{accuracy}")

    def predict(self, learn_model):
        predicted = model.predict(self.X_test)
        print("classification report")
        print(classification_report(self.y_test, predicted))
