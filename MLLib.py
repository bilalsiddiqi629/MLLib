import numpy as np
import pandas as pd

class MLLib:
    def __init__ (self, X, Y):
        self.X = X
        self.Y = Y

    def return_betas(self):
        return self.Betas

    def train_test_split(self, percent=75):
           # if isinstance(self.Y, pd.Series) or isinstance(self.X, pd.DataFrame):
            #    print("WARNING: Data isn't encoded. Auto encoding...")
             #   self.encode()
            if percent <= 0 or percent >= 100:
                raise ValueError("Percentage of training set must be between 0 and 100.")
                

            else:
                try:
                    rng = np.random.default_rng()
                    training_size = int(len(self.X) * (percent / 100))
                    indices = rng.choice(len(self.X), size=training_size, replace=False)
                    alt_indices = np.setdiff1d(np.arange(len(self.X)), indices)

                    testing_setX  = self.X.iloc[alt_indices].reset_index(drop=True)
                    testing_setY  = self.Y.iloc[alt_indices].reset_index(drop=True)
                    training_setX = self.X.iloc[indices].reset_index(drop=True)
                    training_setY = self.Y.iloc[indices].reset_index(drop=True)

                    return training_setX, training_setY, testing_setX, testing_setY

                except TypeError:
                    print("Remember to encode your data!")

    def linreg_train(self, x_train, y_train):
        arr = np.linalg.pinv(x_train.T @ x_train) @ (x_train.T @ y_train)
        self.Betas = arr
        return arr
    
    def linreg_predict(self, x_val):
        try:
            return self.Betas @ x_val
        except ValueError:
            return (f"ERROR: Improper dimensions. There are {len(self.Betas)} predictors but only {len(x_val)} inputs were provided.")
        except AttributeError:
            return ("ERROR: No betas were provided. Call the train method first to avoid this issue.")
    
    def linreg_ridge_train(self, x_train, y_train, lamb):
        identity_m = np.identity(x_train.shape[1] )
        arr = np.linalg.pinv(x_train.T @ x_train + lamb * (identity_m)) @ (x_train.T @ y_train)
        self.Betas = arr
        return arr

    
    def k_fold_cross_valid_lambda(self, k):
        lambdas = np.logspace(-4, 4, 10)
        error_matrix = np.zeros((k, len(lambdas)))
        for x in range(k):
            temptrainX, temptrainY, tempvalidX, tempvalidY = self.train_test_split()
            X = temptrainX.values
            Y = temptrainY.values
            Vx = tempvalidX.values
            Vy = tempvalidY.values

            XtX = X.T @ X        
            XtY = X.T @ Y
            n_features = XtX.shape[0]

            for y in range(len(lambdas)):
                beta_lamb = np.linalg.solve(XtX + y * np.eye(n_features), XtY)
                error_matrix[x, y] = np.sum((Vy - Vx @ beta_lamb) ** 2)
        
        min_idx = np.unravel_index(np.argmin(error_matrix), error_matrix.shape)
        return lambdas[min_idx[1]]

    def mse(self, x_test, y_test):
        try:
            temp_total = 0
            for x in range(len(x_test)):
                temp_pred = self.linreg_predict(x_test[x])
                temp_total = temp_total + (y_test[x] - temp_pred) ** 2
        
            mse = temp_total / len(x_test)
            return mse
        except AttributeError:
            return ("ERROR: No betas were provided. Call the train method first to avoid this issue.")
    
    def sigmoid(self, z):
         return 1 / (1 + np.exp(-z))

    def logreg_train(self, X_train, y_train, iterations, learning_rate):
        self.Betas = np.zeros(X_train.shape[1])  
        for i in range(iterations):
            prob_pred = self.sigmoid(np.dot(X_train, self.Betas))
            gradient = X_train.T @ (prob_pred - y_train) / X_train.shape[0]    
            self.Betas -= learning_rate * gradient  

        return self.Betas

    def logreg_predict(self, x_val):
        try:
            return self.sigmoid(np.dot(x_val, self.Betas))
        except ValueError:
            return (f"ERROR: Improper dimensions. There are {len(self.Betas)} predictors but only {len(x_val)} inputs were provided.")
        except AttributeError:
            return ("ERROR: No betas were provided. Call the train method first to avoid this issue.")
    
    def standard_scalar(self, Xinput):
        return (Xinput - np.mean(Xinput, axis=0)) / np.std(Xinput, axis=0)

    
    def standard_imputer(self, Xinput, method = 'mean'):
        if method == 'mean':
            replace = np.nanmean(Xinput)
        elif method == 'std':
            replace = np.nanstd(Xinput)
        elif method == 'median':
            replace = np.nanmedian(Xinput)
        
        Xinput[np.isnan(Xinput)] = replace
        return Xinput

    def encode(self):
        self.X = self.encoder(self.X)
        self.Y = self.encoder(self.Y)

    def encoder(self, df):
        if isinstance(df, pd.Series):
             if not pd.api.types.is_numeric_dtype(df):
                df = pd.Series(pd.factorize(df)[0])

        else:
            for col in df.select_dtypes(exclude="number").columns:
                df[col] = pd.factorize(df[col])[0]

        return df.to_numpy()
    


        


