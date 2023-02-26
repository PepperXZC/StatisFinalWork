import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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
        self.w = None
        self.b = None
        
    def loss(self, w, b, X, y):
        d = np.dot(X, w) + b
        # a = self.sigmoid(d)
        # 原本是求和,但因为 np.exp爆炸,这里改成mean好了一些
        t = np.log(1. + np.exp(d))
        return np.mean(y * d -  t, axis=0)
    
    def sigmoid(self, x):
        x = np.exp(np.array(x))
        tell = np.where(np.isinf(x)==True) # 因为 exp 容易爆数量级,所以先记录一下exp过大的样本下标,其必判断为1
        res = x / (1. + x)
        res[tell] = 1
        return res

    def derivative(self, w, b, X, y):
        # 假设损失函数为 L2范数 
        dot = np.dot(X, w) + b
        distance = y - self.sigmoid(dot)
        distance = distance.reshape(-1, 1)
        return np.mean(distance * X, axis=0), np.mean(distance, axis=0)
    
    def fit(self, X, y, epoch=1000, lr=0.0005):
        # 将最后一列'TotalCharges'数据归一化
        fin = X.shape[1] - 1
        X[:, fin] = ( X[:, fin] - X[:, fin].mean() ) / X[:, fin].var()
        print(X)
        self.n = X.shape[1]
        std = np.sqrt(2. / (self.n + 1))
        self.w = np.random.normal(scale=std, size=self.n) 
        self.b = np.random.normal(scale=std, size=1)
        for i in range(epoch):
            dw, db = self.derivative(self.w, self.b, X, y)
            self.w += lr * dw
            self.b += lr * db
            if (i + 1) % 100 == 0:
            # 无法展示 loss 值,因为样本中的 0 代入这里会直接显示 nan,无法数据可视化
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
    print("查准率:{}, 查全率:{},F1度量：{}".format(P, R, F1))
    return P, R, F1

def LDA_predict(train_x, test_x, train_y, test_y):
    lda = LDA()
    X_trans = lda.fit_transform(train_x, train_y)
    y_pred = lda.predict(test_x)
    # print(y_pred)
    print("(LDA)测试集预测精度为acc=",np.sum(y_pred==test_y.reshape(-1))/len(y_pred))
    evaluate(y_pred, test_y)
    # 下面是 sklearn 标准的 LDA 函数，解锁注释以运行
    # lda= skLDA(n_components=1, solver='svd')
    # lda.fit(train_x,train_y)
    # X_sklearn = lda.transform(train_x)
    # y_pred_sklearn = lda.predict(test_x)
    # print(y_pred_sklearn)
    # print(np.sum(np.array([y_pred == y_pred_sklearn])))
    # print ('LDA的正确率:',lda.score(test_x,test_y))

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

def make_plot(path):
    # 分析是否为流失的顾客
    data = pd.read_csv(path)
    columns = data.columns.to_list()
    
    # 设置 plt.title 可以显示中文
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize']=5,5
    plt.pie(data['Churn'].value_counts(),labels=data['Churn'].value_counts().index,autopct='%.2f%%')
    plt.title("流失用户占比")
    plt.show()
    
    chinese_name = {
        'PhoneService':'电话服务',
        'MultipleLines':'多重号码',
        # 'InternetService':'连接宽带',
        'OnlineSecurity':'网络安全服务',
        'OnlineBackup':'线上备份',
        'DeviceProtection':'设备保护',
        'TechSupport':'技术支持',
        'StreamingTV':'网络电视',
        'StreamingMovies':'网络电影'
    }

    def del_no_phone(df1):
        temp_df = df1[df1['variable'] == '多重号码']
        del_list = temp_df[temp_df['是否拥有'] == 'No phone service'].index.to_list()
        return df1.drop(df1.index[del_list])

    cols = [j for (i, j) in chinese_name.items()]
    data.rename(columns=chinese_name, inplace=True)
    df_no = data[data['InternetService'] == 'No']
    plt.pie(df_no['Churn'].value_counts(),labels=df_no['Churn'].value_counts().index,autopct='%.2f%%')
    plt.title("未接入网的流失用户占比")
    plt.show()

    # data[data['连接宽带'] == 'DSL' and data['连接宽带'] == 'Fiber optic', ['连接宽带']] = 'Yes'
    # df1由data的列变成行（显示为列变量）并在新列值中列出所有关联值（每个样本提供至少一个）
    # 排除了所有 连接宽带 的数据。
    
    df1 = pd.melt(data[data['InternetService'] != 'No'][cols])
    df1.rename(columns={'value':'是否拥有'},inplace=True)
    df1 = del_no_phone(df1)
    # df1 = pd.melt(data[cols]) 
    # df1.drop(df1[df1['variable'] == '多重号码'] == 'No phone service', inplace=True)
    plt.figure(figsize=(10,5))
    ax = sns.countplot(data=df1, x='variable', hue='是否拥有')
    ax.set(xlabel='网络附加服务',ylabel='用户数量')
    plt.rcParams.update({'font.size':10})
    plt.title('所有用户与网络附加服务(第一组变量)相关关系')
    plt.show()

    # 非流失版本
    data = data[data['Churn'] == 'Yes']
    df1 = pd.melt(data[data['InternetService'] != 'No'][cols])
    df1.rename(columns={'value':'是否拥有'},inplace=True)
    df1 = del_no_phone(df1)
    plt.figure(figsize=(10,5))
    ax2 = sns.countplot(data=df1, x='variable', hue='是否拥有')
    ax2.set(xlabel='网络附加服务',ylabel='用户数量')
    plt.rcParams.update({'font.size':10})
    plt.title('已流失用户与网络附加服务(第一组变量)相关关系')
    plt.show()

    
def get_data(path):
    data = pd.read_csv(path)
    data['TotalCharges']=pd.to_numeric(data['TotalCharges'],errors='coerce')
    # data['tenure']=pd.to_numeric(data['tenure'],errors='coerce')
    data.loc[data['TotalCharges'].isnull().values==True,'TotalCharges'] = data[data['TotalCharges'].isnull().values==True]['MonthlyCharges']
    # 数据归一化
    # for key in ['TotalCharges']:
    #     data[key] = ( data[key] - data[key].min() ) / data[key].max() - data[key].min()
        # print(data[key].mean())
        # print(data[key].var())
        # print(data[key] - data[key].mean())
        # break
    data = data[['tenure','MonthlyCharges', 'TotalCharges', 'Churn']]
    columns = data.columns.to_list()
    for key in columns:
        if key != 'TotalCharges':
            data[key] = pd.factorize(data[key])[0]
    
    target = data['Churn'].values # 取得所有的 y
    data = data.drop('Churn', axis=1).to_numpy() # 取得 X
    return data, target
 
if __name__ == '__main__':
    
    data, target = get_data(path='WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # make_plot(path='WA_Fn-UseC_-Telco-Customer-Churn.csv')
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.30, stratify = target, random_state = 1)
    LDA_predict(train_x, test_x, train_y, test_y)
    bayes_predict(train_x, test_x, train_y, test_y)
    LR_predict(train_x, test_x, train_y, test_y)
    # print()

    # 


