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
        
        # 类间散度矩阵什么时候算过吗？
        s0, s1 = self.cov_calculate(X0), self.cov_calculate(X1)
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

class Bayes:
    def __init__(self, n_classes = 2) -> None:
        self.dic = {}
        self.n_classes = n_classes
        for i in range(n_classes):
            self.dic[i] = {}

    def count(self, X):
        # 获得每个特征中每个数出现的概率
        dict_X = {}
        for i in range(X.shape[1]):
            unique, count_num=np.unique(X[:,i],return_counts=True) 
            data_count=dict(zip(unique,count_num / X.shape[0]))
            dict_X[i] = data_count
        return dict_X
    
    def fit(self, X, y):
        X0, X1 = X[y.reshape(-1) == 0], X[y.reshape(-1) == 1]
        self.dic[0] = self.count(X0)
        self.dic[1] = self.count(X1)
        self.P = np.array([X0.shape[0], X1.shape[0] ]) / X.shape[0]
    
    def predict(self, X):
        res = []
        for x in X:
            pos = np.ones(self.n_classes)
            for i in range(self.n_classes):
                for feature in range(X.shape[1]):
                    try:
                        pos[i] *= self.dic[i][feature][x[feature]]
                    except KeyError:
                        most_closed_index = np.argmin(np.abs(np.array(list(self.dic[i][feature])) - x[feature]))
                        most_closed = list(self.dic[i][feature])[most_closed_index]
                        pos[i] *= self.dic[i][feature][most_closed]
            pos = pos * self.P
            res.append(pos.argmax())
        return res

class LogisticRegression:
    def __init__(self) -> None:
        self.n = 5
        std = np.sqrt(2. / (5 + 1))
        self.w = np.random.normal(scale=std, size=5)
        self.b = np.random.normal(scale=std, size=1)
        pass
    
    def ObjectFunction(self, w, b, X, y):
        dot = np.dot(X, w) + b
        # 原本是求和，但因为 np.exp爆炸，这里改成mean好了一些
        return np.mean(y * dot - np.log(1 + np.exp(dot)), axis=0)
    
    def sigmoid(self, x):
        x = np.array(x)
        return 1. / (1. + np.exp(-x))

    def dL(self, w, b, X, y):
        dot = np.dot(X, w) + b
        distance = y - self.sigmoid(dot)
        distance = distance.reshape(-1, 1)
        return np.mean(distance * X, axis=0), np.mean(distance, axis=0)
    
    def fit(self, X, y, epoch=150, lr=0.0001):
        
        for i in range(epoch):
            dw, db = self.dL(self.w, self.b, X, y)
            self.w += lr * dw
            self.b += lr * db
        print('epoch:{}, L: {}'.format(i+1, self.ObjectFunction(self.w, self.b, X, y)))
        return
    
    def predict(self,X_test):
        return self.sigmoid(np.dot(X_test, self.w) + self.b) >= 0.5
    
def LDA_predict(train_x, test_x, train_y, test_y):
    lda = LDA()
    X_trans = lda.fit_transform(train_x, train_y)
    y_pred = lda.predict(test_x)
    # print(y_pred)
    print("(LDA)测试集预测精度为acc=",np.sum(y_pred==test_y.reshape(-1))/len(y_pred))

    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
    # lda= skLDA(n_components=1, solver='svd')
    # lda.fit(train_x,train_y)

    # X_sklearn = lda.transform(train_x)
    # y_pred_sklearn = lda.predict(test_x)
    # # print(y_pred_sklearn)
    # print(np.sum(np.array([y_pred == y_pred_sklearn])))
    # print ('LDA的正确率:',lda.score(test_x,test_y))

def bayes_predict(train_x, test_x, train_y, test_y):
    bayes = Bayes()
    bayes.fit(train_x, train_y)
    y_pred = bayes.predict(test_x)
    print("(bayes)测试集预测精度为acc=",np.sum(y_pred==test_y.reshape(-1))/len(y_pred))

def LR_predict(train_x, test_x, train_y, test_y):
    LR = LogisticRegression()
    LR.fit(train_x, train_y)
    y_pred = LR.predict(test_x)
    print("(LR)测试集预测精度为acc=",np.sum(y_pred==test_y.reshape(-1))/len(y_pred))


if __name__ == '__main__':
    data = pd.read_csv('kaggle\WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data['TotalCharges']=pd.to_numeric(data['TotalCharges'],errors='coerce')
    # data['tenure']=pd.to_numeric(data['tenure'],errors='coerce')
    data.loc[data['TotalCharges'].isnull().values==True,'TotalCharges'] = data[data['TotalCharges'].isnull().values==True]['MonthlyCharges']
    data = data[['MultipleLines','tenure','PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']]
    # data['PaymentMethod'] = pd.factorize(data['PaymentMethod'])[0]
    # data['MultipleLines'] = pd.factorize(data['MultipleLines'])[0]
    # data['Churn'] = pd.factorize(data['Churn'])[0]
    # dropFea = ['gender','PhoneService',
    #        'OnlineSecurity_No internet service', 'OnlineBackup_No internet service',
    #        'DeviceProtection_No internet service', 'TechSupport_No internet service',
    #        'StreamingTV_No internet service', 'StreamingMovies_No internet service',
    #        ]
    columns = data.columns.to_list()
    for key in columns:
        if key != 'TotalCharges':
            data[key] = pd.factorize(data[key])[0]
    # for column in dropFea:
    #     data[column] = pd.factorize(data[column])[0]
    # print(data.columns.to_list())
    # features = data.columns
    # key_dict = data.nunique().to_dict()
    # for key in features:
    #     if key_dict[key] <= 10:
    #         data[key] = pd.factorize(data[key])[0]
        # data[key]=pd.to_numeric(data[key],errors='coerce')
    # print(data.nunique().to_dict())
    
    target = data['Churn'].values # 取得所有的 y
    data = data.drop('Churn', axis=1).to_numpy() # 取得 X
    # print(data)
    # 在外面转成numpy送进去
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.30, stratify = target, random_state = 1)
    LDA_predict(train_x, test_x, train_y, test_y)
    bayes_predict(train_x, test_x, train_y, test_y)
    LR_predict(train_x, test_x, train_y, test_y)

    # 


