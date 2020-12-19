#!/usr/bin/env python
# coding: utf-8

# In[5]:


def plot_ROC(model, df):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt 
    from sklearn.preprocessing import LabelEncoder, LabelBinarizer
    from numpy import array     
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    
    df = df.sample(frac=1)
    
    X = df.Value
    le = LabelEncoder()
    lb = LabelBinarizer()
    Y = le.fit_transform(df.State)
    Y = lb.fit_transform(Y)

    model = OneVsRestClassifier(model)
    X_train,X_test, Y_train,Y_test = train_test_split(X, Y,train_size=0.7,random_state=1)
    #model.fit requires [[], [], []] format.
    X_train = array(X_train).reshape(-1, 1) 
    X_test = array(X_test).reshape(-1, 1) 
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):
        plt.figure(i+1)
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(3):
#         plt.figure(num=1, figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
        plt.figure(1)
#         plt.subplot(1,3,i+1)
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()


# In[6]:


##Pass model object, X as [...], Y as [[binary], ...].
##Y target class variables need to be Binarized.
##Pass ROC = True if want ROC and AUC of the model.
def test_model(df, model, ROC = False):
    from sklearn.preprocessing import LabelEncoder, LabelBinarizer
    from numpy import array     
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_validate
    from sklearn.multiclass import OneVsRestClassifier
    
    #Randomize
    df = df.sample(frac=1)
    
    X = df.Value
    le = LabelEncoder()
    lb = LabelBinarizer()
    Y = le.fit_transform(df.State)
    Y = lb.fit_transform(Y)

    model = OneVsRestClassifier(model)
    X_train,X_test, Y_train,Y_test = train_test_split(X, Y,train_size=0.7,random_state=1)
    #model.fit requires [[], [], []] format.
    X_train = array(X_train).reshape(-1, 1) 
    X_test = array(X_test).reshape(-1, 1) 
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    
    if ROC:
        plot_ROC(model, df)    
    
    scores = cross_validate(model, array(X).reshape(-1, 1), Y,
                            scoring=['recall_macro',
                                    'accuracy',
                                    'f1_macro',
                                    'precision_macro'], cv = 10)

    print("After 10 fold cross validation:")
    print(">>>Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean(),
                                           scores['test_accuracy'].std() * 2))
    print(">>>Misclassification Rate: %0.2f (+/- %0.2f)"% (1 - scores['test_accuracy'].mean(),
                                                        scores['test_accuracy'].std()*2))
    print(">>>F1_score: %0.2f (+/- %0.2f)" % (scores['test_f1_macro'].mean(),
                                           scores['test_f1_macro'].std()*2))
    print(">>>Precision: %0.2f (+/- %0.2f)" % (scores['test_precision_macro'].mean(),
                                           scores['test_precision_macro'].std()*2))
    print(">>>Recall: %0.2f (+/- %0.2f)" % (scores['test_recall_macro'].mean(),
                                           scores['test_recall_macro'].std()*2))
    CM = confusion_matrix(lb.inverse_transform(Y_test), lb.inverse_transform(Y_pred))
    print(">>>Confusion Matrix:")
    print(CM)
    print(">>>Sensitiviy: ")
    print("   Sensitivity of F = ", end = '')
    print(CM[0,0]/sum((CM.transpose())[0]))

    print("   Sensitivity of M = ", end = '')
    print(CM[1,1]/sum((CM.transpose())[1]))

    print("   Sensitivity of N = ", end = '')
    print(CM[2,2]/sum((CM.transpose())[2]))
    print("\n")
    


# In[6]:


def plot_multiple_ROC(model, df):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt 
    from sklearn.preprocessing import LabelEncoder, LabelBinarizer
    from numpy import array     
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    
    df = df.sample(frac=1)
    
    X = df.Value
    le = LabelEncoder()
    lb = LabelBinarizer()
    Y = le.fit_transform(df.State)
    Y = lb.fit_transform(Y)

    model = OneVsRestClassifier(model)
    X_train,X_test, Y_train,Y_test = train_test_split(X, Y,train_size=0.7,random_state=1)
    #model.fit requires [[], [], []] format.
    X_train = array(X_train).reshape(-1, 1) 
    X_test = array(X_test).reshape(-1, 1) 
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):
        plt.figure(i+1)
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return [fpr, tpr, roc_auc]

