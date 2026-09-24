from sklearn.base import BaseEstimator, TransformerMixin

from collections import Counter


# Since we are using the sklearn pipe pattern, we have to create our own _UNK_-handler
class UnknownHandler(BaseEstimator, TransformerMixin):
    def __init__(self, unknown_treshold=2):
        self.known_vocab = set()
        self.unknown_treshold = unknown_treshold

    def fit(self, X, y=None):
        all_text = " ".join(X)
        # Get the counts of each word
        word_counts = Counter(all_text.split())
        
        # If above treshold
        self.known_vocab = {
            word for word, count in word_counts.items() 
            if count >= self.unknown_treshold #Treshold value
        }
        return self

    def transform(self, X):
        # Replace rare or unseen words with the unk_token
        return [
            " ".join([word if word in self.known_vocab else "_UNK_" for word in doc.split()])
            for doc in X
        ]