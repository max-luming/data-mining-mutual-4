import pandas as pd
import numpy as np
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
import os
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import random
path = 'data/data/wine/benchmarks/'
files = os.listdir(path)
random.shuffle(files)
files=files[:500]
train_csv = list(files)
print(files)

data_list = []
i=1
for fileitem in train_csv:
    tmp = pd.read_csv(path + fileitem)
    tmp_new=tmp[['ground.truth','fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol']]
    tmp_new.drop_duplicates()
    if i==1:
        tmp_new.to_csv("data/data/wine/prodata/gloable.csv",mode='a',index=False)
        i=0
    else:
        tmp_new.to_csv("data/data/wine/prodata/gloable.csv",mode='a',index=False, header=False)

 
df = pd.read_csv("data/data/wine/prodata/gloable.csv")
label=[]
X=df[['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol']]
count=0
for i in range(len(df)):
    if df['ground.truth'][i]=="nominal":
        label+=[0]
    if df['ground.truth'][i]=='anomaly':
        label+=[1]
        count+=1
Y=label
from pyod.utils.data import evaluate_print
random_state = np.random.RandomState(42)
outliers_fraction = count/len(df)
classifiers = {
    #基于接近度    
        'CBLOF':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
    #Proximity-Based
        'KNN': KNN(contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state)
}
for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X)
    y_pred=clf.labels_ 
    n_out=np.count_nonzero(y_pred)
    n_normal=len(y_pred)-n_out
    print("原数据的异常值数量为：%d;异常值数量：%d;正常值：%d" %(count,n_out,n_normal))
    sum_all=0
    #sum_out=0
    TP=0
    FP=0
    FN=0
    for i in range(len(df)):
        if(Y[i]==y_pred[i]):
            sum_all+=1
            if Y[i]==0:
                TP+=1
        if(Y[i]==1):
            if(y_pred[i]==0):
                FP+=1
        if(Y[i]==0):
            if(y_pred[i]==1):
                FN+=1
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1=2*precision*recall/(precision+recall)
    #print("整体识别准确率为: %4lf；离群点识别的准确率为：%4lf" % (sum_all/len(df),sum_out/n_out))
    print("算法：%s分析：整体识别准确率为: %4lf；precision: %4lf;recall：%4lf;F1:%4lf" %(clf_name,sum_all/len(df),precision,recall,F1))