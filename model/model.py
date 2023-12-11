from sklearn.linear_model import LogisticRegression
from config import conf
import numpy as np
import torch

class Model:
    """Base class of the classifier model"""
    def __init__(self) -> None:
        pass

    def train(self, x,y):
        pass

    def predict(self, input):
        pass
    
    def predict_prob(self, input):
        pass
    
    def test(self, x, y):
        pass

class ModelReal(Model):
    """Model used in real data experiments"""
    def __init__(self, m_name) -> None:
        super().__init__()
        # 'm_name' specifies the classifier name, i.e., DenseNet, PreResNet-110 or ResNet-110.
        with open(f"{conf.ROOT_DIR}/data/{m_name}.csv", "r") as f:
            csv = np.loadtxt(f, delimiter=',')
            self.model_logits = csv[:, 11:] 
            # Models keep stored the softmax outputs for each sample in test set,
            # so we only need the index of the correspondent sample to get the softmax output

        # Case 1: Uniform 
        # N, num_classes = self.model_logits.shape 
        # max_prob = 0.25
        # for i in range(len(self.model_logits)):
        #     argmax = np.argmax(self.model_logits[i])  
        #     deltas = np.array([(1-max_prob)/(num_classes-1) if ii != argmax else max_prob - self.model_logits[i, argmax] for ii in range(num_classes)], dtype=self.model_logits.dtype)
        #     self.model_logits[i, :] = self.model_logits[i, :] + deltas

        # Case 2: Top-k are near equivalent
        top_prob = 0.25
        adv = 0.05
        low_prob = (1 - (top_prob * 3) + adv) / 7
        for i in range(len(self.model_logits)):
            argmax = np.argmax(self.model_logits[i])
            if argmax < 3:
                self.model_logits[i, :] = np.array([top_prob] * 3 + [low_prob] * 7)
            elif argmax < 6:
                self.model_logits[i, :] = np.array([low_prob] * 3 + [top_prob] * 3 + [low_prob] * 4)
            else:
                self.model_logits[i, :] = np.array([low_prob] * 7 + [top_prob] * 3)
            self.model_logits[i, argmax] += adv
        
    def predict(self, input, return_tensor=False):

        self.model_logits_t = torch.tensor(self.model_logits[input], device=conf.device)
        y_hat = self.model_logits_t.multinomial(1, replacement=True, generator=conf.torch_rng)
        if not return_tensor:
            y_hat = y_hat.detach().cpu().numpy().flatten() 
        return y_hat
    
    def predict_prob(self, input):
        return self.model_logits[input]

    def test(self, x, y):
        y_hat = self.predict(input=x)
        return np.mean(y == y_hat)

class ModelSynthetic(Model):
    """Model used in synthetic data experiments"""
    def __init__(self) -> None:
        super().__init__()
        self.model = LogisticRegression(random_state=0,n_jobs=-1, max_iter=1000, multi_class='ovr')
        self.missing_classes = []
        
    def predict(self, input):
        return self.model.predict(input)

    def predict_prob(self, input):
        ret = self.model.predict_proba(input)
        # Fix model output to return 0 probabilty for unknown classes
        for missing_class in self.missing_classes:
            ret = np.insert(ret,missing_class,0.,axis=1)
        return ret

    def train(self, x,y):
        self.model = self.model.fit(x,y)
        # Find which classes the model did not learn at all 
        # (Needed to fix tensors later)
        sorted_classes = np.sort(self.model.classes_)
        all_classes = np.arange(conf.n_labels)
        if self.model.classes_.shape[0] < conf.n_labels:
            i = 0
            for j in all_classes:
                if j == sorted_classes[i]:
                    i+=1
                else:
                    self.missing_classes.append(j)

    def test(self, x, y):
        return self.model.score(x,y)