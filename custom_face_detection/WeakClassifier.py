class WeakClassifier:
    def __init__(self, feature, threshold, polarity):
        self.feature = feature
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, integral):
        return 1 if self.polarity * self.feature.compute(integral) < self.polarity * self.threshold else 0
