from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

class TextualDecisionTreeClassifier(DecisionTreeClassifier):

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        return super().fit(X, y_encoded)