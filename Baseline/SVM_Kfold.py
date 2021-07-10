#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:14:45 2018

@author: D. Escobar-Grisales
"""
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics


def SVM_kfold (matriz, labels, n_repet, n_kfold, shuffle=True):
    
    #Parametros y variables a usar 
    # vC=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 1000, 10000]
    # gm=[1e-4, 1e-3, 1e-2, 0.1, 1,10,50,100,1000]
    vC=[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    gm=[1e-4, 1e-3, 1e-2, 0.1, 1,10,50,100,1000]
    vC_total=[]
    gm_total=[]
    acc_total=[]
    sensi_total=[]
    speci_total=[]
    f1_total=[]
    auc_total=[]
    
    y_scores_final=[]
    z_final=[]
    y_final=[]

    for ite in range (0,n_repet):
        print("------------------- interaction = "+str(ite)+" --------------------")
      
        y_scores=[]
        z_total=[]
        y_total=[]
    
        kf = KFold(n_splits=n_kfold,shuffle=shuffle)
        kf.get_n_splits(matriz)
    
        for train, test in kf.split(matriz):
            X_train, X_test = matriz[train], matriz[test]
            y_train, y_test = labels[train], labels[test]
    
            #Restar la media y dividir la std
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
    
            #Calculo de hiperparametros
            parameters = {'kernel':['rbf'], 'C':vC, 'gamma': gm,}
            svr = svm.SVC()
            clf = GridSearchCV(svr, parameters,n_jobs=4,cv=9)
            clf.fit(X_train, y_train)
            C_ideal=clf.best_estimator_.C
            G_ideal=clf.best_estimator_.gamma
            vC_total.append(C_ideal)
            gm_total.append(G_ideal)
    
            #Entrenamiento y prediccion
            clf = SVC(C=C_ideal,gamma=G_ideal,kernel='rbf',class_weight='balanced')
            clf.fit(X_train, y_train)
            X_test=scaler.transform(X_test)
            Z=clf.predict(X_test)
            
            # Guardar score, predic y etiqueta
            y_scores.append(clf.decision_function(X_test))
            z_total.append(Z)
            y_total.append(y_test)
            
        y_total=np.hstack(y_total)
        z_total=np.hstack(z_total)
        y_scores=np.hstack(y_scores)
        
        #Calculo de metricas
        acc=accuracy_score(y_total,z_total)
        c_m=confusion_matrix(y_total,z_total)
        param=precision_recall_fscore_support(y_total, z_total, pos_label=1, average='weighted')
        sensi=(c_m[1][1])/(c_m[1][1]+c_m[1][0])
        speci=(c_m[0][0])/(c_m[0][0]+c_m[0][1])
        f1=param[2]
        fpr, tpr, thresholds = metrics.roc_curve(y_total, y_scores)
        auc=metrics.auc(fpr, tpr)
        
        print("Accuracy=",acc*100)
        print("Sensivity = ", sensi)
        print("Specificity = ", speci)
        print("F1score = ", f1)
        print("AUC= ", auc)
    
        y_final.append(y_total)
        z_final.append(z_total)
        y_scores_final.append(y_scores)

        acc_total.append(acc*100)
        sensi_total.append(sensi)
        speci_total.append(speci)
        f1_total.append(f1)
        auc_total.append(auc)
        
    if(np.size(np.where(np.isnan(sensi_total)))!=0):
        sensi_total=np.delete(sensi_total,np.where(np.isnan(sensi_total))[0])
    
    if(np.size(np.where(np.isnan(speci_total)))!=0):
        speci_total=np.delete(speci_total,np.where(np.isnan(speci_total))[0])
    
    print("------------------------ FINAL -------------------------")
    print("Total C = ", np.str(stats.mode(vC_total)[0][0]))
    print("Total Gamma = ", np.str(stats.mode(gm_total)[0][0]))
    print("Total Accuracy = " + np.str(np.mean(acc_total)) + " +/- " + np.str(np.std(acc_total)))
    print("Total Sensitivity = " + np.str(np.mean(sensi_total)) + " +/- " + np.str(np.std(sensi_total)))
    print("Total Recall = " + np.str(np.mean(speci_total)) + " +/- " + np.str(np.std(speci_total)))
    print("Total F1score = " + np.str(np.mean(f1_total)) + " +/- " + np.str(np.std(f1_total)))
    print("Total AUC = " + np.str(np.mean(auc_total)) + " +/- " + np.str(np.std(auc_total)))

    return [vC_total, gm_total, acc_total, sensi_total, speci_total, f1_total, auc_total, y_final, z_final, y_scores_final]
