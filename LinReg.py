import numpy as np
import pandas as pd

class LinReg:
    def __init__ (self, X, Y, lam):
        self.Betas = np.zeros(len(X[:]))
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

    def mse(self):
        total = 0
        for x in range(len(self.X)):
            row = self.X.iloc[x]
            np_array = row.to_numpy()
            temp_pred = self.return_pred(np_array)
            total = total + ((self.Y[x] - temp_pred)**2)
        mse = total / len(self.X)
        return mse
    
        X.apply()

