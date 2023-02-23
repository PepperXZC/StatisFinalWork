import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# cluster_1 = cluster_1.drop('Churn', axis=1)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA


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
        return np.array(y_pred)
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
        return np.array(res)

class LogReg:
    def __init__(self, sample_weight=None) -> None:
        self.n = 5
        std = np.sqrt(2. / (5 + 1))
        self.w = np.random.normal(scale=std, size=5) 
        self.b = np.random.normal(scale=std, size=1)
    
    def loss(self, w, b, X, y):
        d = np.dot(X, w) + b
        # a = self.sigmoid(d)
        # 原本是求和，但因为 np.exp爆炸，这里改成mean好了一些
        t = np.log(1. + np.exp(d))
        return np.mean(y * d -  t, axis=0)
    
    def sigmoid(self, x):
        x = np.exp(np.array(x))
        tell = np.where(np.isinf(x)==True) # 因为 exp 容易爆数量级，所以先记录一下exp过大的样本下标，其必判断为1
        res = x / (1. + x)
        res[tell] = 1
        return res

    def derivative(self, w, b, X, y):
        # 假设损失函数为 L2范数 
        dot = np.dot(X, w) + b
        distance = y - self.sigmoid(dot)
        distance = distance.reshape(-1, 1)
        return np.mean(distance * X, axis=0), np.mean(distance, axis=0)
    
    def fit(self, X, y, epoch=1000, lr=0.0001):
        for i in range(epoch):
            dw, db = self.derivative(self.w, self.b, X, y)
            # y1, y2, y3, loss = self.forward(X, y, self.w, self.b)
            # dw, db = self.gradients(y1, y2, y3, X, y)
            self.w += lr * dw
            self.b += lr * db
            if (i + 1) % 10 == 0:
            # 无法展示 loss 值，因为样本中的 0 代入这里会直接显示 nan，无法数据可视化
                print('epoch:{}, L: {}'.format(i+1, self.loss(self.w, self.b, X, y)))
        return
    
    def predict(self,X_test):
        tell = self.sigmoid(np.dot(X_test, self.w) + self.b) >= 0.5
        res = np.zeros(len(tell))
        res[tell] = 1
        return res

def evaluate(y_pred, y_true):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = (2 * P * R) / (P + R)
    print("查准率:{}, 查全率:{}，F1度量：{}".format(P, R, F1))
    return P, R, F1

def LDA_predict(train_x, test_x, train_y, test_y):
    lda = LDA()
    X_trans = lda.fit_transform(train_x, train_y)
    y_pred = lda.predict(test_x)
    # print(y_pred)
    print("(LDA)测试集预测精度为acc=",np.sum(y_pred==test_y.reshape(-1))/len(y_pred))
    evaluate(y_pred, test_y)
    lda= skLDA(n_components=1, solver='svd')
    lda.fit(train_x,train_y)

    X_sklearn = lda.transform(train_x)
    y_pred_sklearn = lda.predict(test_x)
    # print(y_pred_sklearn)
    # print(np.sum(np.array([y_pred == y_pred_sklearn])))
    print ('LDA的正确率:',lda.score(test_x,test_y))

def bayes_predict(train_x, test_x, train_y, test_y):
    bayes = Bayes()
    bayes.fit(train_x, train_y)
    y_pred = bayes.predict(test_x)
    print("(bayes)测试集预测精度为acc=",np.sum(y_pred==test_y.reshape(-1))/len(y_pred))
    evaluate(y_pred, test_y)
    return

def LR_predict(train_x, test_x, train_y, test_y):
    LR = LogReg()
    LR.fit(train_x, train_y)
    y_pred = LR.predict(test_x)

    print("(LR)测试集预测精度为acc=",np.sum(y_pred==test_y.reshape(-1))/len(y_pred))
    evaluate(y_pred, test_y)
    # LR = LogisticRegression().fit(train_x, train_y)
    # # y_pred = LR.predict(test_x)
    # print("(LR) sklearn版本正确率：", LR.score(test_x, test_y))

def get_data(path):
    data = pd.read_csv(path)
    data['TotalCharges']=pd.to_numeric(data['TotalCharges'],errors='coerce')
    # data['tenure']=pd.to_numeric(data['tenure'],errors='coerce')
    data.loc[data['TotalCharges'].isnull().values==True,'TotalCharges'] = data[data['TotalCharges'].isnull().values==True]['MonthlyCharges']
    data = data[['MultipleLines','tenure','PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']]
    columns = data.columns.to_list()
    for key in columns:
        if key != 'TotalCharges':
            data[key] = pd.factorize(data[key])[0]
    
    target = data['Churn'].values # 取得所有的 y
    data = data.drop('Churn', axis=1).to_numpy() # 取得 X
    return data, target

class AdaBoost:

    def __init__(self, m, T, clf):
        # 注意这里的分类器可以不是同一个模型：可以是m个Logistic，也可以是一些朴素贝叶斯+Logistic
        self.clf = [clf for _ in range(T)]
        # self.m = m
        self.T = 50 if T == None else T
        # 缓存基分类器和权重参数
        self.clf_arr = []
        self.alpha_arr = []
        self.y_y = None
 
    def fit(self, X, y):
        self.y_y = list(set(y))
        
        # train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.30, stratify = target, random_state = 1)
        num = X.shape[0]
        self.w = np.ones(num) / num
        self.beta = np.zeros(self.T)
        res = np.zeros(X.shape[1])

        for t in range(self.T):
            # 这里默认每个模型都有 sklearn 风格的 model.fit() 函数
            # 并且将 W 作为已知权重代入其中
            # D(i) 以 取1的概率大小 为准
            epsilon = 0
            self.clf[t].fit(X, y, sample_weight = self.w)
            # self.clf_arr.extend([self.clf])
            # 预测结果、预测概率
            y_pred = self.clf[t].predict(X)
            print("目前的精确度：{}",np.sum(y_pred==y.reshape(-1))/len(y_pred))
            not_equal = (y_pred != y).reshape(-1)
            epsilon = np.dot(self.w.reshape(-1), not_equal.T)
            if epsilon > 0.5:
                self.T = t - 1
                return
            beta_t = 1 / (1 - epsilon)
            self.beta[t] = beta_t
            self.w = self.w * (beta_t ** (1 - not_equal))
            self.w /= np.sum(self.w)
        self.beta = np.log(1 / self.beta)
        # for i in range(X.shape[1]):
        #     res[i] = 
        return

    def predict(self, X):
        res = np.zeros((len(self.y_y), X.shape[0]))
        self.beta = np.log(1 / self.beta) # (1, (len.self.T))
        for ans in range(len(self.y_y)):
            pred_array = np.zeros((self.T, X.shape[0]))
            for i in range(self.T):
                pred_array[i] = (self.clf[i].predict(X) == self.y_y[ans]).reshape(-1)
            res[ans] = np.dot(self.beta, pred_array)
        res = np.argmax(res, axis=0)
        return res
 
if __name__ == '__main__':
    
    data, target = get_data(path='WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # print(data)
    # 在外面转成numpy送进去
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.30, stratify = target, random_state = 1)
    # LDA_predict(train_x, test_x, train_y, test_y)
    # bayes_predict(train_x, test_x, train_y, test_y)
    LR_predict(train_x, test_x, train_y, test_y)
    # print()

    # 


