'''
Created on 
@author: 51607
'''
# -*- coding: utf-8 -*-
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.metrics import recall_score
from scipy import signal
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as pl
import matplotlib
import math
from xdf import load_xdf
from CSP import CSP
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
#from enable import markers
from numpy import ndarray, shape

N = 500
fs = 5
n = [2*math.pi*fs*t/N for t in range(N)]

x = [math.sin(i) for i in n]
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    pl.figure()
    pl.title(title)
    if ylim is not None:
        pl.ylim(*ylim)
    pl.xlabel("Training examples")
    pl.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    pl.grid()

    pl.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    pl.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    pl.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    pl.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test score")

    pl.legend(loc="best")
    return pl




def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def readXDF(xdffile):
    tempnumber=1
    tempnumber2=0


    streams=load_xdf(xdffile)
#     for item in streams[0][0]:
#         print (item)
#     print ("---------------")

    SampleNumber=streams[0][tempnumber]['time_series'].__len__()
    MarkerNumber=streams[0][tempnumber2]['time_series'].__len__()
    trailNumber=int((MarkerNumber-1)/4)
    

    axis_x=np.array([streams[0][tempnumber]['time_stamps'][index] for index in range(SampleNumber)])
    #print (axis_x)
    signal1=np.array([streams[0][tempnumber]['time_series'][index][0] for index in range(SampleNumber)])
    signal2=np.array([streams[0][tempnumber]['time_series'][index][1] for index in range(SampleNumber)])
    signal3=np.array([streams[0][tempnumber]['time_series'][index][2] for index in range(SampleNumber)])
    signal4=np.array([streams[0][tempnumber]['time_series'][index][3] for index in range(SampleNumber)])
    signal5=np.array([streams[0][tempnumber]['time_series'][index][4] for index in range(SampleNumber)])
    signal6=np.array([streams[0][tempnumber]['time_series'][index][5] for index in range(SampleNumber)])
    signal7=np.array([streams[0][tempnumber]['time_series'][index][6] for index in range(SampleNumber)])
    signal8=np.array([streams[0][tempnumber]['time_series'][index][7] for index in range(SampleNumber)])
    markerStream=np.array([streams[0][tempnumber2]['time_series'][index] for index in range(MarkerNumber)])
    markerTime=np.array([streams[0][tempnumber2]['time_stamps'][index] for index in range(MarkerNumber)])
    return tempnumber,tempnumber2,streams,SampleNumber,MarkerNumber,trailNumber,axis_x,[signal1,signal2,signal3,signal4,signal5,signal6,signal7,signal8],markerStream,markerTime

def processing(tempnumber,tempnumber2,streams,SampleNumber,MarkerNumber,trailNumber,axis_x,mainsignal,markerStream,markerTime):

    markerResult=np.zeros(shape=[20])

    for index in range(2,82,4):

        if markerStream[index]==['Right']:

            markerResult[(index-2)//4]=1
            
    
    #print markerTime
    freqs = [[8,14],[18,24]]
    #freqs = [[2,4],[4,8],[8,13],[13,22],[22,30]] # delta, theta, alpha, low beta, high beta    
    #freqs = [[1,4],[4,8],[8,13],[13,22],[22,30],[1,30]]
    #freqs = [[1,4],[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32]]
    #freqs = [[1,3],[3,5],[5,7],[7,9],[9,11],[11,13],[13,15],[15,17],[17,19],[19,21],[21,23],[23,25],[25,27],[27,29],[29,31],[31,33],[33,35]]
    fs = 250.0  # Sample frequency (Hz)
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    Q = 30.0  # Quality factor
    w0 = f0/(fs/2)  # Normalized Frequency
    # Design notch filter
    b, a = signal.iirnotch(w0, Q)
#     print('12142124')
#     print(len(mainsignal))
    for index in range(8):
        mainsignal[index] = signal.filtfilt(b,a,mainsignal[index])


    filtersignal = [[butter_bandpass_filter(mainsignal[index], f[0], f[1],250) for index in range(8)] for f in freqs]
    #print(mainsignal[0].shape)
#     for f in freqs:
#         tempmainsignal=mainsignal
#         fs=250
#         nyq = 0.5 * fs
#         low = f[0] / nyq
#         high = f[1] / nyq
#         b, a = signal.butter(6, [low, high], btype='band')
#         for index in range(8):
#             tempmainsignal[index] = signal.lfilter(b, a, mainsignal[index])
#         filtersignal.append(tempmainsignal)


    # 
   

    countStart=0
    trailsTime=np.zeros(shape=[trailNumber,5])
    for index in range(MarkerNumber):
        if markerStream[index][0]=="start":
            trailsTime[countStart][0]=streams[0][tempnumber2]['time_stamps'][index]
            trailsTime[countStart][1]=streams[0][tempnumber2]['time_stamps'][index+1]
            trailsTime[countStart][2]=streams[0][tempnumber2]['time_stamps'][index+2]
            trailsTime[countStart][3]=streams[0][tempnumber2]['time_stamps'][index+3]
            if markerStream[index+1][0]=="right":
                trailsTime[countStart][4]=0
            else :
                trailsTime[countStart][4]=1
            countStart+=1
            
    for index in range(SampleNumber):
        if axis_x[index]>trailsTime[0][tempnumber2]:
            firstStartTime=index
            break
    trails=np.empty(shape=[20])
    temptrail=np.zeros(shape=[1000,8])
    trails=np.zeros(shape=[len(freqs),20,625,8])

    print(len(filtersignal))
    print(len(filtersignal[0]))
    print(trails.shape)
    for index4 in range(len(freqs)):
        for index in range(20):
            for index2 in range(625):
                for index3 in range(8):
                    trails[index4][index][index2][index3]=filtersignal[index4][index3][(index-1)*4750+index2+2750+firstStartTime]
    print(np.transpose(trails[0][0]).shape)
    for index in range(len(freqs)):
        for index2 in range(20):
            trails[index][index2] = preprocessing.normalize(trails[index][index2], norm='l2')
#             pl.subplot(421)
#             pl.plot(axis_x[:625],np.transpose(trails[index])[0])
#             pl.title(u'signal 1'+'     '+str(index))
#             pl.axis('tight')
#             pl.subplot(422)
#             pl.plot(axis_x[:625],np.transpose(trails[index])[1])
#             pl.title(u'signal 2')
#             pl.axis('tight')
#             pl.subplot(423)
#             pl.plot(axis_x[:625],np.transpose(trails[index])[2])
#             pl.title(u'signal 3')
#             pl.axis('tight')
#             pl.subplot(424)
#             pl.plot(axis_x[:625],np.transpose(trails[index])[3])
#             pl.title(u'signal 4')
#             pl.axis('tight')
#             pl.subplot(425)
#             pl.plot(axis_x[:625],np.transpose(trails[index])[4])
#             pl.title(u'signal 5')
#             pl.axis('tight')
#             pl.subplot(426)
#             pl.plot(axis_x[:625],np.transpose(trails[index])[5])
#             pl.title(u'signal 6')
#             pl.axis('tight')
#             pl.subplot(427)
#             pl.plot(axis_x[:625],np.transpose(trails[index])[6])
#             pl.title(u'signal 7')
#             pl.axis('tight')
#             pl.subplot(428)
#             pl.plot(axis_x[:625],np.transpose(trails[index])[7])
#             pl.title(u'signal 8')
#             pl.axis('tight')
#             pl.show()
    return markerResult,trails

def main():
    tempnumber1,tempnumber21,streams1,SampleNumber1,MarkerNumber1,trailNumber1,axis_x1,mainsignal1,markerStream1,markerTime1=readXDF("eric1.xdf")
    tempnumber2,tempnumber22,streams2,SampleNumber2,MarkerNumber2,trailNumber2,axis_x2,mainsignal2,markerStream2,markerTime2=readXDF("eric2.xdf")
    tempnumber3,tempnumber23,streams3,SampleNumber3,MarkerNumber3,trailNumber3,axis_x3,mainsignal3,markerStream3,markerTime3=readXDF("eric3.xdf")
    tempnumber4,tempnumber24,streams4,SampleNumber4,MarkerNumber4,trailNumber4,axis_x4,mainsignal4,markerStream4,markerTime4=readXDF("eric4.xdf")
    tempnumber5,tempnumber25,streams5,SampleNumber5,MarkerNumber5,trailNumber5,axis_x5,mainsignal5,markerStream5,markerTime5=readXDF("eric5.xdf")
    markerResult1,trails1=processing(tempnumber1,tempnumber21,streams1,SampleNumber1,MarkerNumber1,trailNumber1,axis_x1,mainsignal1,markerStream1,markerTime1)
    markerResult2,trails2=processing(tempnumber2,tempnumber22,streams2,SampleNumber2,MarkerNumber2,trailNumber2,axis_x2,mainsignal2,markerStream2,markerTime2)
    markerResult3,trails3=processing(tempnumber3,tempnumber23,streams3,SampleNumber3,MarkerNumber3,trailNumber3,axis_x3,mainsignal3,markerStream3,markerTime3)
    markerResult4,trails4=processing(tempnumber4,tempnumber24,streams4,SampleNumber4,MarkerNumber4,trailNumber4,axis_x4,mainsignal4,markerStream4,markerTime4)
    markerResult5,trails5=processing(tempnumber5,tempnumber25,streams5,SampleNumber5,MarkerNumber5,trailNumber5,axis_x5,mainsignal5,markerStream5,markerTime5)
    print(markerResult1)
    print(markerResult2)
    print(markerResult3)
    print(markerResult4)
    print(markerResult5)
    print(trails1.shape)
    freqs = [[8,14],[18,24]]
    #freqs = [[2,4],[4,8],[8,13],[13,22],[22,30]]
    print ("------------------------------------------------------------")
    left=[[],[]]
    right=[[],[]]
    #left=[[],[],[],[],[]]
    #right=[[],[],[],[],[]]
    for index2 in range(len(freqs)):
        for index in range(20):
            if markerResult1[index]==0:
                left[index2].append(np.transpose(trails1[index2][index]))
            elif markerResult1[index]==1:
                right[index2].append(np.transpose(trails1[index2][index]))
        for index in range(20):
            if markerResult2[index]==0:
                left[index2].append(np.transpose(trails2[index2][index]))
            elif markerResult2[index]==1:
                right[index2].append(np.transpose(trails2[index2][index]))
        for index in range(20):
            if markerResult3[index]==0:
                left[index2].append(np.transpose(trails3[index2][index]))
            elif markerResult3[index]==1:
                right[index2].append(np.transpose(trails3[index2][index]))
        for index in range(20):
            if markerResult4[index]==0:
                left[index2].append(np.transpose(trails4[index2][index]))
            elif markerResult4[index]==1:
                right[index2].append(np.transpose(trails4[index2][index]))
        for index in range(20):
            if markerResult5[index]==0:
                left[index2].append(np.transpose(trails5[index2][index]))
            elif markerResult5[index]==1:
                right[index2].append(np.transpose(trails5[index2][index]))
    result=[]
    print(len(left))
    print(len(left[0]))
    print(left[0][0].shape)
    for index in range(len(freqs)):
        result.append(CSP(left[index],right[index]))
    #result=CSP([np.transpose(trails1[0]),np.transpose(trails1[1]),np.transpose(trails1[2]),np.transpose(trails1[4]),np.transpose(trails1[6]),np.transpose(trails1[8]),np.transpose(trails1[10]),np.transpose(trails1[14]),np.transpose(trails1[15]),np.transpose(trails1[18])],[np.transpose(trails1[3]),np.transpose(trails1[5]),np.transpose(trails1[7]),np.transpose(trails1[9]),np.transpose(trails1[11]),np.transpose(trails1[12]),np.transpose(trails1[13]),np.transpose(trails1[16]),np.transpose(trails1[17]),np.transpose(trails1[19])])
#     result=CSP([np.transpose(trails2[1]),np.transpose(trails2[1]),np.transpose(trails2[2]),np.transpose(trails2[4]),np.transpose(trails2[6]),np.transpose(trails2[8]),np.transpose(trails2[10]),np.transpose(trails2[14]),np.transpose(trails2[15]),np.transpose(trails2[18])],[np.transpose(trails2[3]),np.transpose(trails2[5]),np.transpose(trails2[7]),np.transpose(trails2[9]),np.transpose(trails2[11]),np.transpose(trails2[12]),np.transpose(trails2[13]),np.transpose(trails2[16]),np.transpose(trails2[17]),np.transpose(trails2[19])])
#     result=CSP([np.transpose(trails3[0]),np.transpose(trails3[1]),np.transpose(trails2[2]),np.transpose(trails2[4]),np.transpose(trails2[6]),np.transpose(trails2[8]),np.transpose(trails2[10]),np.transpose(trails2[14]),np.transpose(trails2[15]),np.transpose(trails2[18])],[np.transpose(trails2[3]),np.transpose(trails2[5]),np.transpose(trails2[7]),np.transpose(trails2[9]),np.transpose(trails2[11]),np.transpose(trails2[12]),np.transpose(trails2[13]),np.transpose(trails2[16]),np.transpose(trails2[17]),np.transpose(trails2[19])])
#     result=CSP([np.transpose(trails4[0]),np.transpose(trails4[1]),np.transpose(trails2[2]),np.transpose(trails2[4]),np.transpose(trails2[6]),np.transpose(trails2[8]),np.transpose(trails2[10]),np.transpose(trails2[14]),np.transpose(trails2[15]),np.transpose(trails2[18])],[np.transpose(trails2[3]),np.transpose(trails2[5]),np.transpose(trails2[7]),np.transpose(trails2[9]),np.transpose(trails2[11]),np.transpose(trails2[12]),np.transpose(trails2[13]),np.transpose(trails2[16]),np.transpose(trails2[17]),np.transpose(trails2[19])])
#     result=CSP([np.transpose(trails5[0]),np.transpose(trails5[1]),np.transpose(trails2[2]),np.transpose(trails2[4]),np.transpose(trails2[6]),np.transpose(trails2[8]),np.transpose(trails2[10]),np.transpose(trails2[14]),np.transpose(trails2[15]),np.transpose(trails2[18])],[np.transpose(trails2[3]),np.transpose(trails2[5]),np.transpose(trails2[7]),np.transpose(trails2[9]),np.transpose(trails2[11]),np.transpose(trails2[12]),np.transpose(trails2[13]),np.transpose(trails2[16]),np.transpose(trails2[17]),np.transpose(trails2[19])])
    print ("csp")
    np.savetxt("fbCSP1.csv", result[0], delimiter=",")
    np.savetxt("fbCSP2.csv", result[1], delimiter=",")
    #np.savetxt("fbCSP3.csv", result[2], delimiter=",")
    #np.savetxt("fbCSP4.csv", result[3], delimiter=",")
    #np.savetxt("fbCSP5.csv", result[4], delimiter=",")
#     newresult= np.genfromtxt("fbCSP.csv", delimiter=',')

    x=np.zeros(shape=[len(freqs),100,8])
    x_final=np.zeros(shape=[100,8*len(freqs)])
    print(trails1[0][0].shape)
    for index3 in range(len(freqs)):
        for index in range(20):
            newtrail=np.dot(np.transpose(result[index3]),np.transpose(trails1[index3][index]))
            totalVar=0
            for index2 in range(8):
                totalVar+=np.var(newtrail[index2])
            for index2 in range(8):
                x[index3][index][index2]=math.log(np.var(newtrail[index2])/totalVar)
        
        for index in range(20,40):
            newtrail=np.dot(np.transpose(result[index3]),np.transpose(trails2[index3][index-20]))
            totalVar=0
            for index2 in range(8):
                totalVar+=np.var(newtrail[index2])
            for index2 in range(8):
                x[index3][index][index2]=math.log(np.var(newtrail[index2])/totalVar)
                
        for index in range(40,60):
            newtrail=np.dot(np.transpose(result[index3]),np.transpose(trails3[index3][index-40]))
            totalVar=0
            for index2 in range(8):
                totalVar+=np.var(newtrail[index2])
            for index2 in range(8):
                x[index3][index][index2]=math.log(np.var(newtrail[index2])/totalVar)
        for index in range(60,80):
            newtrail=np.dot(np.transpose(result[index3]),np.transpose(trails4[index3][index-60]))
            totalVar=0
            for index2 in range(8):
                totalVar+=np.var(newtrail[index2])
            for index2 in range(8):
                x[index3][index][index2]=math.log(np.var(newtrail[index2])/totalVar)
        for index in range(80,100):
            newtrail=np.dot(np.transpose(result[index3]),np.transpose(trails5[index3][index-80]))
            totalVar=0
            for index2 in range(8):
                totalVar+=np.var(newtrail[index2])
            for index2 in range(8):
                x[index3][index][index2]=math.log(np.var(newtrail[index2])/totalVar)




    x_final=np.concatenate(x,axis=1)
#     newtrail11=np.dot(np.transpose(result),np.transpose(trails1[10]))
#     totalVar=0
#     for index in range(8):
#         totalVar+=np.var(newtrail11[index])
#     for index in range(8):
#         x[9][index]=math.log(np.var(newtrail11[index])/totalVar)
#     
#     
     
#     example=np.zeros(shape=[10,8])
#     newtrail10=np.dot(np.transpose(result),np.transpose(trails1[9]))
#     totalVar=0
#     for index in range(8):
#         totalVar+=np.var(newtrail11[index])
#     for index in range(8):
#         example[0][index]=math.log(np.var(newtrail11[index])/totalVar)
#     
#     
#     for index in range(11,20):
#         newtrail=np.dot(np.transpose(result),np.transpose(trails1[index]))
#         totalVar=0
#         for index2 in range(8):
#             totalVar+=np.var(newtrail[index2])
#         for index2 in range(8):
#             example[index-10][index2]=math.log(np.var(newtrail[index2])/totalVar)
#     
    example=np.zeros(shape=[len(freqs),20,8])
    for index3 in range(len(freqs)):
        for index in range(20):
            newtrail2=np.dot(np.transpose(result[index3]),np.transpose(trails5[index3][index]))
            totalVar=0
            for index2 in range(8):
                totalVar+=np.var(newtrail2[index2])
            for index2 in range(8):
                example[index3][index][index2]=math.log(np.var(newtrail2[index2])/totalVar)
    example_final=np.concatenate(example,axis=1)
    
    #y=[0,0,0,1,0,1,0,1,0,1,0,1,1,1,0,0,1,1,0,1]
    #y=[1,0,1,0,1,1,1,0,0,1,0,0,0,1,1,0,1,1,0,1]
    y=list(markerResult1)
    list2=list(markerResult2)
    list3=list(markerResult3)
    list4=list(markerResult4)
    list5=list(markerResult5)
    y.extend(list2)
    y.extend(list3)
    y.extend(list4)
    y.extend(list5)

    #x = np.transpose(preprocessing.normalize(np.transpose(x), norm='l2'))
    
    scoring = ['precision_macro']
#     clf = svm.SVC(kernel='linear', C=1, random_state=0)
#     scores = cross_validate(clf, x_final, y, scoring=scoring,
#                         cv=5)
#     filename = 'svmmodel.sav'
#     pickle.dump(clf, open(filename, 'wb'))
#     print(scores)
    tre = tree.DecisionTreeClassifier()
    scores = cross_validate(tre, x_final, y, scoring=scoring, cv=5)
    tre.fit(x_final, y)
    #     filename = 'svmmodel.sav'
#     pickle.dump(clf, open(filename, 'wb'))
    print(scores)
    
    lda=LinearDiscriminantAnalysis(n_components=2)
    print(x_final.shape)
    lda.fit(x_final,y)
    scores = cross_validate(lda, x_final, y, scoring=scoring,
                                cv=5)
    print(scores)
    X_new = lda.transform(x_final)
    yy=np.zeros(shape=[len(X_new)])
    pl.scatter(X_new[:,0],X_new[:,0],marker='o',c=y)
    pl.show()
    print ("pre")
    filename = 'ldamodel.sav'
    pickle.dump(lda, open(filename, 'wb'))
    loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, Y_test)
    
    
#     title = "Learning Curves (Naive Bayes)"
#     # Cross validation with 100 iterations to get smoother mean test and train
#     # score curves, each time with 20% data randomly selected as a validation set.
#     cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
#     
#     estimator = GaussianNB()
#     plot_learning_curve(estimator, title, x_final, y, ylim=(0, 1.01), cv=cv, n_jobs=4)
#     
    title = r"Learning Curves Cross-validation score(LDA)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = LinearDiscriminantAnalysis(n_components=2)
    plot_learning_curve(estimator, title, x_final, y, (0, 1.01), cv=5, n_jobs=4)
    
    pl.show()
    #print (lda.predict(example))
    print(example.shape)
    print(example_final.shape)
    print (lda.predict(example_final))
    print (loaded_model.predict(example_final))
    print(markerResult5)
    print ("finish")
    
if __name__ == "__main__":
    main()
#FBCSP
#[[1,4],[4,8],[8,13],[13,22],[22,30]]
