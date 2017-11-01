from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import datasets 


class ModelSpace:
    """
    Should include:
    Logistic Regression
    KNN classification
    Support Vector Machines
    RandomForest
    Gradient Boosting
    """
    def __init__(self, model_list, random_subset=None):
        self.available_model = {
            'RFC' : RandomForestClassifier(),
            'GBC': GradientBoostingClassifier(),
            'LR': LogisticRegression(),
            'SVC': SVC(),
            'KNC': KNeighborsClassifier()
        }
        self.model_list = [self.available_model[model] for model in model_list]
        self.random_subset=random_subset

class CV:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __lshift__(self, other):
        score = []
        for model in other.model_list:
            score.append(cross_val_score(model, self.X, self.y).mean())
        score.sort()
        return score

class Validate:
    def __init__(self):
        pass
    
    def __lshift__(self, other):
        pass

class ChooseBest:
    def __init__(self, amount):
        self.amount=amount
    
    def __lshift__(self, other):
        return other[-self.amount:]
        
iris = datasets.load_iris()

X = iris.data
y = iris.target

print(ChooseBest(4) << (CV(X, y) << ModelSpace(['RFC', 'LR', 'SVC', 'KNC'])))