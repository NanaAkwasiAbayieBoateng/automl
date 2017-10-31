

class ModelSpace:
    def __init__(self, model_list, random_subset=None):
        self.model_list=model_list
        self.random_subset=random_subset

class CV:
    def __init__(self):
        pass

    def __lshift__(self, other):
        score = []
        for model in other.model_list:
            score.append(model)
        score.sort(key = len)
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
        return other[-3:]
        
        
        

print(ChooseBest(3) << (CV() << ModelSpace(['asdf', 'fgfahj','asaaadf1', 'fghj2','aaaaaaasdf2', 'fgaaaahj3','asdf4', 'fghj5'])))