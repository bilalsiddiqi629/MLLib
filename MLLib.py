import numpy as np
import pandas as pd

class MLLib:
    def __init__ (self, X, Y, lam):
        self.Betas = np.zeros(len(X))
        self.X = X
        self.Y = Y
        self.lam = lam

    def return_betas(self):
        return self.Betas
    
    def return_pred(self, Xinput):
        try:
            result = np.dot(self.Betas, Xinput)
            return result
        except ValueError:
            print(f"Improper dimensions for X input. The number of betas is {len(self.Betas)}, and {len(Xinput)} values were inputted.")
            return
        except TypeError:
            print("Remember to encode your data!")
            return
    def train_linear(self):
        for x in range(len(self.Betas)):
            self.Betas[x] = (np.dot(self.X[x], np.transpose(self.X[x])) ** -1) * (np.dot(np.transpose(self.X[x]),self.Y))

    def mse(self):
        temp_total = 0
        for x in range(len(self.X)):
            temp_pred = self.return_pred(self.X[x])
            temp_total = temp_total + (self.Y[x] - temp_pred) ** 2
        
        mse = temp_total / len(self.X)
        return mse





        


