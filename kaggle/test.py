import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# cluster_1 = cluster_1.drop('Churn', axis=1)

class LDA:
    def __init__(self) -> None:
        self.w = None

    def cov_calculate(self, X, Y=None):
        n = X.shape[0]
        X = X - np.mean(X, axis=0)
        Y = X if Y == None else Y - np.mean(Y, axis=0)
        res = (1 / (n - 1)) * np.matmul(X.T, Y)
        return res
    
    # 在给定数据上计算出 w
    def fit(self, X, y):
        X0, X1 = X[y.reshape(-1) == 0], X[y.reshape(-1) == 1]
        
        s0, s1 = self.cov_calculate(X0), self.cov_calculate(X1)
        print(s0 == np.cov(X0))
        sw = s0 + s1
        mu0, mu1 = X0.mean(0), X1.mean(0)
        mean_diff=np.atleast_1d(mu0 - mu1)

        U,S,V=np.linalg.svd(sw)
        #计算类内散度矩阵的逆
        s =np.dot(np.dot(V.T,np.linalg.pinv(np.diag(S))),U.T)
        self.w = s.dot(mean_diff)
        return self.w

    def fit_transform(self, X, y):
        self.fit(X, y)
        X_trans = X.dot(self.w)
        return X_trans
    
    def predict(self, X):
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred
    # print(w, w[0], w[1])


if __name__ == '__main__':
    data = pd.read_csv('kaggle\WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data['TotalCharges']=pd.to_numeric(data['TotalCharges'],errors='coerce')
    data.loc[data['TotalCharges'].isnull().values==True,'TotalCharges'] = data[data['TotalCharges'].isnull().values==True]['MonthlyCharges']
    data = data[['PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']]
    data['PaymentMethod'] = pd.factorize(data['PaymentMethod'])[0]
    data['Churn'] = pd.factorize(data['Churn'])[0]
    target = data['Churn'].values # 取得所有的 y
    data = data.drop('Churn', axis=1).to_numpy() # 取得 X
    # 在外面转成numpy送进去
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.30, stratify = target, random_state = 1)
    # lda = LDA()  
    # print(train_x[train_y.reshape(-1) == 0])
    # h = np.random.randn(1,4)
    # print(1 * (h < 0))
    lda = LDA()
    X_trans = lda.fit_transform(train_x, train_y)
    y_pred = lda.predict(test_x)
    # print(y_pred)
    print("测试集预测精度为acc=",np.sum(y_pred==test_y.reshape(-1))/len(y_pred))

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
    lda= skLDA(n_components=1, solver='svd')
    lda.fit(train_x,train_y)

    X_sklearn = lda.transform(train_x)
    y_pred_sklearn = lda.predict(test_x)
    # print(y_pred_sklearn)
    print(np.sum(np.array([y_pred == y_pred_sklearn])))
    print ('LDA的正确率:',lda.score(test_x,test_y))
