import re

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class RuleBasedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, rules: dict):
        self.rules: dict = rules

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(list(self.rules.keys()) + list(y))
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        predictions = []
        
        for text in X:
            prediction = "null"
            
            for label, rules in self.rules.items():
                for rule in rules:
                    # The first rule that matches will be the prediction
                    # To check whether a keyword matches the text, we need to use regex
                    # otherwise the keyword "no" will match with "noise" for example
                    if re.search(r"\b" + re.escape(rule) + r"\b", text):
                        prediction = label
                        break
    
                # when a prediction is made, we do not look further
                if prediction != "null":
                    break
    
            predictions.append(prediction)
    
        return self.label_encoder_.transform(predictions)