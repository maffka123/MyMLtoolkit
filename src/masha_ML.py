import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rescalAllnumeric(df):
    '''min-max rescales numeric columns in dataframe
      also converts srtring columns to ordinal
      TODO: add more flexibility to which columns should be converted how (pd.to_dummy?)
    '''

    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OrdinalEncoder

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    newdf = df.select_dtypes(include=numerics)
    newdf_str = df.select_dtypes(include='object')

    newdf_scaled = StandardScaler().fit_transform(newdf)
    newdf_scaled = pd.DataFrame(newdf_scaled, index=newdf.index, columns=newdf.columns)
    newdf_str_dummy = OrdinalEncoder().fit_transform(newdf_str)
    newdf_str_dummy = pd.DataFrame(newdf_str_dummy, index=newdf_str.index, columns=newdf_str.columns)
    df.update(newdf_scaled)
    df.update(newdf_str_dummy)
    return df

def getXY(data,testvar,toremove=[],tested=False):
    '''Converts df into dependent and independent variables
       removes provided columns
       also removes vars after testing for importance
    ''''
    c = data.columns.tolist()
    c.remove(testvar)
    c = list(set(c).difference(set(toremove)))
    if tested:
        #To remove: got from previous iteration:
        #toremove=['age', 'job' ,'marital', 'education' ,'default' ,'housing' ,'loan' ,'month','day_of_week']
        toremove=['education', 'job' ,'marital' ,'day_of_week' ,'cons_conf_idx' ,'age','campaign']
        #toremove=['default', 'housing', 'loan', 'poutcome']
        c = list(set(c).difference(set(toremove)))
    X = data[c]
    Y = data[testvar]
    return X,Y

def check_and_fix_imbalance(X,y,plot=False):
    '''check if there is a strong imbalance in types of dytypoint to model
    TODO: return option without weight, but rather random selecting from variables
    '''

    from sklearn.utils import class_weight

    num1=sum(y)/len(y)
    num0=(len(y)-sum(y))/len(y)
    class_weights=[1,1]
    if abs(num1-num0)>0.6:
        class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
        #idx0=np.random.choice(np.where(y==0)[0],500,replace=False)
        #idx1 = np.random.choice(np.where(y == 1)[0], 500,replace=False)
        #X=X.iloc[idx0].append(X.iloc[idx1])
        #y=y.iloc[idx0].append(y.iloc[idx1])

    num0=[num0,num0*class_weights[0]]
    num1 = [num1, num1 * class_weights[1]]
    if plot:
        fig=plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
        p1=ax.bar([1,2], num1, color='#2DB973')
        p2=ax.bar([1,2], num0, color='#EED86C', bottom=num1)
        ax.legend((p1[0], p2[0]), ('number of 1', 'number of 0'))
        ax.set_xticks([1,2], ('Before correction', 'After correction'))
        ax.set_ylabel('Fraction')
        plt.show()
    return X,y,class_weights

def run_classifier(model,X_train, y_train,parameters):
    ''' runs classifier of choise using random search
        TODO: return grid search as an option
    '''
    from sklearn.model_selection import RandomizedSearchCV

    clf = RandomizedSearchCV(estimator = model, param_distributions =parameters, cv=5, scoring='f1')
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    model.set_params(**clf.best_params_)
    model.fit(X_train, y_train)
    return model

def plot_final(model,X_test,y_test, algo_name,ax):
    '''plots classification statistics
       ROC-curve, F1, accuracy for now
    '''
    
    from sklearn import metrics
    from sklearn.metrics import roc_curve

    prediction=model.predict(X_test)
    predictions = model.predict_proba(X_test)

    precision, recall, thresholds = roc_curve(y_test, predictions[:, 1])
    ax.plot(precision, recall)
    ax.plot([0, 1], [0, 1])
    ax.set_title('F1-score of %s: %0.2f\n Accuracy is: %0.f' % (algo_name,metrics.f1_score(y_test, prediction),metrics.accuracy_score(y_test, prediction)))
    return ax

def run_ML(X,Y,typeOFML,parameters):
    '''run machine learning based on sklearn
       runs only those tested by me
       TODO: default parameters
    '''

    from sklearn.model_selection import train_test_split
    
    if typeOFML=='RF': #random forest
    	from sklearn.ensemble import RandomForestClassifier
    	model = RandomForestClassifier(n_jobs=-1)
    	name='Random Forest'
    elif typeOFML=='KNN': #nearest neighbours
    	from sklearn.neighbors import KNeighborsClassifier
    	model = KNeighborsClassifier()
    	name='K-nearest Neighbours'
    elif typeOFML=='NB_Bernoulli': #Naive Bayes - Bernoulli
    	from sklearn.naive_bayes import BernoulliNB
    	model = BernoulliNB()
    	name='Naive Bayes - Bernoulli'
    elif typeOFML=='SVM': # SVM SVC
    	from sklearn import svm
    	model=svm.SVC(probability=True)
    	name='SVM SVC'
    else:
    	print('There is no such model!')
    	break()

    Y=Y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    model=run_classifier(model, X_train, y_train, parameters)
    plot_final(model, X_test, y_test, '%s'%name, ax)
    plt.show()


def SelectFeatures4RandomForest(X,Y,tresh):
    '''Select features from all in the Ran Forest data'''

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import SelectFromModel

    Y=Y.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

    clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    feat_labels=X.columns
    #for feature in zip(feat_labels, clf.feature_importances_):
    #    print(feature)
    sfm = SelectFromModel(clf, threshold=tresh)

    # Train the selector
    sfm.fit(X_train, y_train)
    for feature_list_index in sfm.get_support(indices=True):
        print(feat_labels[feature_list_index])
    return sfm.get_support(indices=True)
