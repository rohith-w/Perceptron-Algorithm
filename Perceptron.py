# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:45:43 2022

@author: Rohit Wardole
"""

import numpy as np
import pandas as pd

# single layer perceptron class used for binary classification
class Perceptron():
    # -- Inputs = features
    # -- MaxIter= number of iterations to be performed 
    # -- learning rate has been assumed to be =1, therefore it has been omitted
    def __init__(self):
        pass

    # train() runs the binary perceptron algorithm to calculate weights
    def train(self, X, y, l2 = False, coeff = 0, iterations=20):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for i in range(iterations):            
            for independent_var, dependent_var in zip(X,y):
                y_prediction = self.predict(independent_var)
                
                if dependent_var*y_prediction <= 0:
                    if l2:
                        self.weights = np.dot((1-2*coeff),self.weights) + dependent_var*independent_var
                    else:
                        self.weights += dependent_var*independent_var
                    self.bias += dependent_var   
        return self.weights,self.bias
          
    # predict() can be used to generate a best guess on a given feature set      
    # predict using label using the provided features
    def predict(self, X):
        return np.where(self.sigmoid(X) > 0.0, 1, -1) 
    
    # Calculate the activation score 
    def sigmoid(self, x):
        return np.dot(x, self.weights) + self.bias 

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true) * 100.0
    return accuracy       

def run_question_3(classification_1, classification_2):
    df_train = pd.read_csv("train.csv", header = None)
    df_test = pd.read_csv("test.csv", header = None)
    
    df_train_data = df_train.loc[(df_train[4] == classification_1) | (df_train[4] == classification_2)]
    #df_train_data = df_train_data.sample(frac=1, random_state=33).reset_index(drop = True)
    
    X_train = df_train_data.iloc[:, 0:4]
    X_train = X_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    X_train = X_train.values
    
    y_train = df_train_data.iloc[0:80, 4].values
    y_train = np.where(y_train == y_train[0], 1, -1)
          
    # Prepare Test Data
    df_test_data = df_test.loc[(df_test[4] == classification_1) | (df_test[4] == classification_2)]
    
    X_test = df_test_data.iloc[:, 0:4].values
    y_test = df_test_data.iloc[:, 4].values
    y_test = np.where(y_test == y_test[0], 1, -1)
    
    p = Perceptron()
    p.train(X_train, y_train)
    predictions = p.predict(X_test)
    
    print("<----TRAIN ACCURACY---->")
    train_acc = p.predict(X_train)
    print("The train accuracy for {} vs {} is: {}\n".format(classification_1, classification_2, accuracy(y_train,train_acc)))
    print("<----TEST ACCURACY---->")
    print("The test accuracy for {} vs {} is: {}".format(classification_1, classification_2, accuracy(y_test,predictions)))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
def question_3():
    
    run_question_3("class-1","class-2")
    run_question_3("class-2","class-3")
    run_question_3("class-1","class-3") 
    
def question_4():
    df_train = pd.read_csv("train.csv", header = None)
    df_test = pd.read_csv("test.csv", header = None)
    
    df_train = df_train.sample(frac=1, random_state=33).reset_index(drop = True)
    
    df_X_train = df_train.iloc[:, 0:4]
    df_X_train = df_X_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    df_X_train = df_X_train.values
    
    df_X_test = df_test.iloc[:, 0:4].values
    
    # Class 1 vs (Class 2 and Class 3)        
    df_y_train_1v23 = df_train.iloc[:, 4].values
    df_y_train_1v23 = np.where(df_y_train_1v23 == 'class-1', 1, -1)
    
    df_y_test_1v23 = df_test.iloc[:, 4].values
    df_y_test_1v23 = np.where(df_y_test_1v23 == 'class-1', 1, -1)
    
    p1 = Perceptron()
    p1.train(df_X_train, df_y_train_1v23,  iterations=20)
    predictions_1v23 = p1.sigmoid(df_X_test)
    predictions_1v23_train_acc = p1.sigmoid(df_X_train)
        
    # Class 2 vs (Class 1 and Class 3)    
    df_y_train_2v13 = df_train.iloc[:, 4].values
    df_y_train_2v13 = np.where(df_y_train_2v13 == 'class-2', 1, -1)
    
    df_y_test_2v13 = df_test.iloc[:, 4].values
    df_y_test_2v13 = np.where(df_y_test_2v13 == 'class-2', 1, -1)
    
    p1.train(df_X_train, df_y_train_2v13, iterations=20)
    predictions_2v13 = p1.sigmoid(df_X_test)
    predictions_2v13_train_acc = p1.sigmoid(df_X_train)
    
    # Class 3 vs (Class 1 and Class 2)    
    df_y_train_3v12 = df_train.iloc[:, 4].values
    df_y_train_3v12 = np.where(df_y_train_3v12 == 'class-3', 1, -1)
    
    df_y_test_3v12 = df_test.iloc[:, 4].values
    df_y_test_3v12 = np.where(df_y_test_3v12 == 'class-3', 1, -1)
    
    p1.train(df_X_train, df_y_train_3v12, iterations=20)
    predictions_3v12 = p1.sigmoid(df_X_test) 
    predictions_3v12_train_acc = p1.sigmoid(df_X_train)
    
    names = np.array(['class-%d'%(i+1) for i in range(3)])
    
    a = names[np.argmax((predictions_1v23_train_acc,predictions_2v13_train_acc,predictions_3v12_train_acc),0)]
    multi_class_accuracy_train_acc = np.sum(df_train[4] == a) /len(df_train[4]) * 100
    print("Multi Class Perceptron Train Accuracy: ",multi_class_accuracy_train_acc)  
    
    b = names[np.argmax((predictions_1v23,predictions_2v13,predictions_3v12),0)]
    multi_class_accuracy = np.sum(df_test[4] == b) /len(df_test[4]) * 100
    print("Multi Class Perceptron Test Accuracy: ",multi_class_accuracy)  

def question_5():
    
    coefficients = [0.01, 0.1, 1.0, 10.0, 100.0]  
    
    df_train = pd.read_csv("train.csv", header = None)
    df_test = pd.read_csv("test.csv", header = None)
    
    df_train = df_train.sample(frac=1, random_state=33).reset_index(drop = True)
    
    df_X_train = df_train.iloc[:, 0:4]
    #df_X_train = df_X_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    df_X_train = df_X_train.values
    
    df_X_test = df_test.iloc[:, 0:4].values
    
    # Class 1 vs (Class 2 and Class 2)    
    df_y_train_1v23 = df_train.iloc[:, 4].values
    df_y_train_1v23 = np.where(df_y_train_1v23 == 'class-1', 1, -1)
    
    df_y_test_1v23 = df_test.iloc[:, 4].values
    df_y_test_1v23 = np.where(df_y_test_1v23 == 'class-1', 1, -1)
        
    # Class 2 vs (Class 1 and Class 3)    
    df_y_train_2v13 = df_train.iloc[:, 4].values
    df_y_train_2v13 = np.where(df_y_train_2v13 == 'class-2', 1, -1)
    
    df_y_test_2v13 = df_test.iloc[:, 4].values
    df_y_test_2v13 = np.where(df_y_test_2v13 == 'class-2', 1, -1)
    
    # Class 3 vs (Class 1 and Class 2)    
    df_y_train_3v12 = df_train.iloc[:, 4].values
    df_y_train_3v12 = np.where(df_y_train_3v12 == 'class-3', 1, -1)
    
    df_y_test_3v12 = df_test.iloc[:, 4].values
    df_y_test_3v12 = np.where(df_y_test_3v12 == 'class-3', 1, -1)
    
    l2 = True
    
    for c in coefficients:
        
        p1 = Perceptron()
        p1.train(df_X_train,df_y_train_1v23, l2, c)
        predictions_1v23= p1.sigmoid(df_X_test)
        predictions_1v23_train= p1.sigmoid(df_X_train)
        
        p1.train(df_X_train,df_y_test_2v13,l2, c)
        predictions_2v13 = p1.sigmoid(df_X_test)
        predictions_2v13_train= p1.sigmoid(df_X_train)
        
        p1.train(df_X_train,df_y_test_3v12,l2, c)  
        predictions_3v12 = p1.sigmoid(df_X_test)
        predictions_3v12_train= p1.sigmoid(df_X_train)
        
        names = np.array(['class-%d'%(i+1) for i in range(3)])
        
        a = names[np.argmax((predictions_1v23_train,predictions_2v13_train,predictions_3v12_train),0)]
        multi_class_accuracy_train = np.sum(df_train[4] == a) /len(df_train[4]) * 100
        print("Train Accuracy for L2 Reg is {}:----->{}".format(c,multi_class_accuracy_train))
        
        b = names[np.argmax((predictions_1v23,predictions_2v13,predictions_3v12),0)]
        multi_class_accuracy = np.sum(df_test[4] == b) /len(df_test[4]) * 100
        print("Test Accuracy for L2 Reg is {}:----->{}".format(c,multi_class_accuracy))
        print("---------------------------------------------------------")
        
        
    print("====================================================\n")
    
if __name__ == "__main__":
    
    print("==================Answer 3==========================")
    question_3()
    print("==================Answer 4==========================")
    question_4()
    print("==================Answer 5==========================")
    question_5()