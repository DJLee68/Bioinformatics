from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from tkinter import simpledialog
import numpy as np





class Classifier:
    alg_list = []
    
    def __init__(self):
        self.alg_list = ['Naive_bayes', 'k-Nearest Neighbor', 'Decision Tree', 'SVM', 'Random Forest', 'Logistic Regression']
        self.k = 0
    def predict(self, alg_idx, tr_data, tr_ans, ts_data):
        if alg_idx == 0:
            return self.nb_classifier(tr_data, tr_ans, ts_data)
        elif alg_idx == 1:
            return self.knn_classifier(tr_data, tr_ans, ts_data)
        elif alg_idx == 2:
            return self.dt_classifier(tr_data, tr_ans, ts_data)
        elif alg_idx == 3:
            return self.svm_classifier(tr_data, tr_ans, ts_data)
        elif alg_idx == 4:
            return self.rf_classifier(tr_data, tr_ans, ts_data)
        elif alg_idx == 5:
            return self.lgregression_classifier(tr_data, tr_ans, ts_data)
        
    def nb_classifier(self, tr_data, tr_ans, ts_data):
        gnb = GaussianNB()
        train_mdl = gnb.fit(tr_data, tr_ans)

        test_pred = train_mdl.predict(ts_data)
        self.k = 0
        return test_pred

    def svm_classifier(self, tr_data, tr_ans, ts_data):
        clf = SVC(kernel = 'rbf', C = 10.0, gamma=0.1)
        train_mdl = clf.fit(tr_data, tr_ans)
        
        test_pred = train_mdl.predict(ts_data)
        self.k = 0
        return test_pred
      
    def rf_classifier(self, tr_data, tr_ans, ts_data):
        rf = RandomForestClassifier()
        train_mdl = rf.fit(tr_data, tr_ans)
        
        test_pred = train_mdl.predict(ts_data)
        self.k = 0
        return test_pred
        
    def lgregression_classifier(self, tr_data, tr_ans, ts_data):
        
        lg = LogisticRegression()
        train_mdl = lg.fit(tr_data, tr_ans)
        test_pred = train_mdl.predict(ts_data)
        self.k = 0
        return test_pred
        
        
    def knn_classifier(self, tr_data, tr_ans, ts_data):
        self.k = 3
        """self.k = simpledialog.askinteger("파라미터 세팅", "k의 값을 결정하세요")"""
        nbrs = KNeighborsClassifier(n_neighbors=self.k)
        train_mdl = nbrs.fit(tr_data, tr_ans)

        test_pred = train_mdl.predict(ts_data)

        return test_pred

    def dt_classifier(self, tr_data, tr_ans, ts_data):
        dt = DecisionTreeClassifier()
        train_mdl = dt.fit(tr_data, tr_ans)

        test_pred = train_mdl.predict(ts_data)
        self.k = 0
        return test_pred

    def get_result(self, alg_idx, tr_data, tr_ans, ts_data, ts_ans):
        pred = self.predict(alg_idx, tr_data, tr_ans, ts_data)
        correct_count = (pred == ts_ans).sum()
        accuracy = correct_count / len(ts_ans)
        precision, recall, fbeta_score, support = precision_recall_fscore_support(ts_ans, pred)
        conf_mat = confusion_matrix(ts_ans, pred)

        return accuracy, precision, recall, fbeta_score, support, conf_mat

    def get_alg_list(self):
        return self.alg_list
