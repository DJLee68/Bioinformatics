import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

accuracy = 0
precision = 0 
recall = 0
fbeta_score = 0
support = 0
conf_mat = 0
k_num = 0
classifier_name = ""

def subset(data):
    m = data.shape[0]
    n = data.shape[1]
    n_b = 2**n-1
    quotient = int(data.shape[1]/15)+1
    reminder = int(data.shape[1]%15)
    static_num=2**15
    static_num1=2**15
    temp2=str(bin(n_b-1))
    temp2=temp2[2:]
    result_list=[]
    for k in range(quotient):
        if k==quotient-1:
            static_num=2**(reminder)
        result_array=np.zeros((static_num,m,n))
        for i in range(static_num):
            temp=str(bin(n_b-(k*static_num1)-i))
            temp=temp[2:]
            for j in range(len(temp2)-len(temp)):
                temp='0'+temp     
            result_array[i] = np.where(temp, data, 0)
          
        result_list.append(result_array)
        
    return result_list


def wrapper(tr_data, tr_ans, ts_data, ts_ans):
    global accuracy, precision, recall, fbeta_score, support, conf_mat, classifier_name, k_num
    classifiers=[GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(), SVC(), RandomForestClassifier(), LogisticRegression()]
    result_score = 0
    result_clf = type(classifiers.__contains__)
    result_tr_data = np.zeros((tr_data.shape))
    result_ts_data = np.zeros((ts_data.shape))
    for tr, ts in zip(subset(tr_data), subset(ts_data)):
        for s_tr, s_ts in zip(tr, ts):
            for clf in classifiers:
                clf.fit(s_tr.astype(float), tr_ans.astype(float))
                pred_y = clf.predict(s_ts.astype(float))
                temp_score = accuracy_score(ts_ans.astype(float), pred_y.astype(float))
                if result_score < temp_score:
                    result_score = temp_score
                    result_clf = clf
                    result_tr_data = s_tr.astype(float)
                    result_ts_data = s_ts.astype(float)
                                
    if type(result_clf) == type(KNeighborsClassifier()):
        k_num = 5                 
    
    accuracy = result_score
    classifier_name = result_clf
    precision, recall, fbeta_score, support = precision_recall_fscore_support(ts_ans.astype(float), pred_y.astype(float))           
    conf_mat = confusion_matrix(ts_ans.astype(float), pred_y.astype(float)) 
    return result_tr_data, result_ts_data