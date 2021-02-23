import numpy as np
import math
import pickle
from Feature import Feature
from Integral import Integral
from Region import Region
from WeakClassifier import WeakClassifier


class ViolaJones:
    def __init__(self, weak_classifiers = 10):
        self.weak_classifiers = weak_classifiers
        self.alphas = []
        self.clfs = []
    
    def train(self, images, labels, pos_num, neg_num):
        weights = np.zeros(len(labels))
        training_data = []
        for fl_pair in range(len(labels)):
            training_data.append(Integral(images[fl_pair]))
            if labels[fl_pair] == 1:
                weights[fl_pair] = 1 / (2 * pos_num)
            else:
                weights[fl_pair] = 1 / (2 * neg_num) 
        
        feature_templates = self.build_features(training_data[0].image.shape)
        features = self.apply_features(feature_templates, training_data)

        for t in range(self.weak_classifiers):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.train_weak(features, labels, feature_templates, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, training_data, labels)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)  
            print("Chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), len(accuracy) - sum(accuracy), alpha))  
    
    def build_features(self, image_shape):
        height, width = image_shape
        features = []
        # TODO: play with minimum feature size
        for w in range(1, width+1):
            for h in range(1, height+1):
                x = 0
                while x + w < width:
                    y = 0
                    while y + h < height:
                        # 2 horizontally aligned blocks
                        root = Region(x,y,w, h)
                        right = Region(x+w, y, w,h)
                        # check if the VJ feature can be fit into the image
                        if x + 2 * w < width:
                            features.append(Feature([right], [root]))
                        bottom = Region(x, y+h, w, h)
                        # 2 vertically aligned blocks
                        if y + 2 * h < height:
                            features.append(Feature([root],[bottom]))
                        # 3 horizontally aligned blocks 
                        right2 = Region(x+2*w, y, w,h)
                        if x + 3 * w < width:
                            features.append(Feature([right], [right2, root]))
                        cross_bottom = Region(x+w, y+h, w, h)
                        if x + 2 * w < width and y + 2 * h < height:
                            features.append(Feature([right, bottom], [root, cross_bottom]))
                        y += 1
                    x += 1
        return features

    def apply_features(self, feature_templates, integrals):
        features = np.zeros((len(feature_templates), len(integrals)))
        count = 0
        for feature in feature_templates:
            features[count] = list(map(lambda integral: feature.compute(integral), integrals))
            count += 1
        return features
    
    def train_weak(self, features, labels, feature_templates, weights):
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, labels):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = []
        total_features = features.shape[0]
        for index, feature in enumerate(features):
            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print("Trained %d classifiers out of %d" % (len(classifiers), total_features))

            applied_feature = sorted(zip(weights, feature, labels), key=lambda x: x[1])

            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = feature_templates[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            
            clf = WeakClassifier(best_feature, best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers

    def select_best(self, classifiers, weights, integrals, labels):
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for integral, label, w in zip(integrals, labels, weights):
                correctness = abs(clf.classify(integral) - label)
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(labels)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy
    

    def classify(self, image):
        total = 0
        integral = Integral(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(integral)
        return 1 if total >= 0.5 * sum(self.alphas) else 0 

    def save(self, filename):
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)
    
    def test_model(self, images, labels):
        correct = 0
        for index, image in enumerate(images):
            correct += 1 if self.classify(image) == labels[index] else 0
        print(f" correct: {correct} out of {len(labels)} percentage: {correct/len(labels)}")
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
