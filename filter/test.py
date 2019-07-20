'''
Created on 
@author: 51607
'''
# -*- coding: utf-8 -*-
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as pl
import matplotlib
import math
from xdf import load_xdf
from CSP import CSP
import pickle
#from enable import markers
from numpy import ndarray, shape
from sklearn import tree

N = 500
fs = 5
n = [2*math.pi*fs*t/N for t in range(N)]

x = [math.sin(i) for i in n]

def readXDF(xdffile):
    tempnumber=1
    tempnumber2=0


    streams=load_xdf(xdffile)
    for item in streams[0][0]:
        print (item)
    print ("---------------")

    SampleNumber=streams[0][tempnumber]['time_series'].__len__()
    MarkerNumber=streams[0][tempnumber2]['time_series'].__len__()
    trailNumber=int((MarkerNumber-1)/4)
    

    axis_x=np.array([streams[0][tempnumber]['time_stamps'][index] for index in range(SampleNumber)])
    print (axis_x)
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
        print (markerStream[index])
        if markerStream[index]==['Right']:

            markerResult[(index-2)//4]=1
            
    
    #print markerTime


    # 
    fs = 250.0  # Sample frequency (Hz)
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    Q = 30.0  # Quality factor
    w0 = f0/(fs/2)  # Normalized Frequency
    # Design notch filter
    b, a = signal.iirnotch(w0, Q)
    for index in range(8):
        mainsignal[index] = signal.filtfilt(b,a,mainsignal[index])
    
    fs=250
    nyq = 0.5 * fs
    low = 8 / nyq
    high = 14 / nyq
    b, a = signal.butter(6, [low, high], btype='band')
    for index in range(8):
        mainsignal[index] = signal.lfilter(b, a, mainsignal[index])


#     pl.subplot(421)
#     pl.plot(axis_x[:120000],mainsignal[0])
#     pl.title(u'signal 1')
#     pl.axis('tight')
#     pl.subplot(422)
#     pl.plot(axis_x[:120000],mainsignal[1])
#     pl.title(u'signal 2')
#     pl.axis('tight')
#     pl.subplot(423)
#     pl.plot(axis_x[:120000],mainsignal[2])
#     pl.title(u'signal 3')
#     pl.axis('tight')
#     pl.subplot(424)
#     pl.plot(axis_x[:120000],mainsignal[3])
#     pl.title(u'signal 4')
#     pl.axis('tight')
#     pl.subplot(425)
#     pl.plot(axis_x[:120000],mainsignal[4])
#     pl.title(u'signal 5')
#     pl.axis('tight')
#     pl.subplot(426)
#     pl.plot(axis_x[:120000],mainsignal[5])
#     pl.title(u'signal 6')
#     pl.axis('tight')
#     pl.subplot(427)
#     pl.plot(axis_x[:120000],mainsignal[6])
#     pl.title(u'signal 7')
#     pl.axis('tight')
#     pl.subplot(428)
#     pl.plot(axis_x[:120000],mainsignal[7])
#     pl.title(u'signal 8')
#     pl.axis('tight')
#     pl.show()
#     nyq = 0.5 * fs
#     high = 24 / nyq
#     low= 18 / nyq
#     b, a = signal.butter(4, high, btype='low', analog = True)
#     for index in range(8):
#         mainsignal[index] = signal.lfilter(b,a,mainsignal[index])
#     b, a = signal.butter(4, low, btype='high', analog = True)
#     for index in range(8):
#         mainsignal[index] = signal.lfilter(b,a,mainsignal[index])
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
    trails=np.zeros(shape=[20,625,8])
    
    
    for index in range(20):
        for index2 in range(625):
            for index3 in range(8):
                trails[index][index2][index3]=mainsignal[index3][(index-1)*4750+index2+2750+firstStartTime]
    print(trails[0].shape)
    for index in range(20):
        print(trails[index].shape)
        trails[index] = preprocessing.normalize(trails[index], norm='l2')
#         pl.subplot(421)
#         pl.plot(axis_x[:625],np.transpose(trails[index])[0])
#         pl.title(u'signal 1'+'     '+str(index))
#         pl.axis('tight')
#         pl.subplot(422)
#         pl.plot(axis_x[:625],np.transpose(trails[index])[1])
#         pl.title(u'signal 2')
#         pl.axis('tight')
#         pl.subplot(423)
#         pl.plot(axis_x[:625],np.transpose(trails[index])[2])
#         pl.title(u'signal 3')
#         pl.axis('tight')
#         pl.subplot(424)
#         pl.plot(axis_x[:625],np.transpose(trails[index])[3])
#         pl.title(u'signal 4')
#         pl.axis('tight')
#         pl.subplot(425)
#         pl.plot(axis_x[:625],np.transpose(trails[index])[4])
#         pl.title(u'signal 5')
#         pl.axis('tight')
#         pl.subplot(426)
#         pl.plot(axis_x[:625],np.transpose(trails[index])[5])
#         pl.title(u'signal 6')
#         pl.axis('tight')
#         pl.subplot(427)
#         pl.plot(axis_x[:625],np.transpose(trails[index])[6])
#         pl.title(u'signal 7')
#         pl.axis('tight')
#         pl.subplot(428)
#         pl.plot(axis_x[:625],np.transpose(trails[index])[7])
#         pl.title(u'signal 8')
#         pl.axis('tight')
#         pl.show()
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
    
    

   

    print ("------------------------------------------------------------")
    left=[]
    right=[]
    for index in range(20):
        if markerResult1[index]==0:
            left.append(np.transpose(trails1[index]))
        elif markerResult1[index]==1:
            right.append(np.transpose(trails1[index]))
    for index in range(20):
        if markerResult2[index]==0:
            left.append(np.transpose(trails2[index]))
        elif markerResult2[index]==1:
            right.append(np.transpose(trails2[index]))
    for index in range(20):
        if markerResult3[index]==0:
            left.append(np.transpose(trails3[index]))
        elif markerResult3[index]==1:
            right.append(np.transpose(trails3[index]))
    for index in range(20):
        if markerResult4[index]==0:
            left.append(np.transpose(trails4[index]))
        elif markerResult4[index]==1:
            right.append(np.transpose(trails4[index]))
    for index in range(20):
        if markerResult5[index]==0:
            left.append(np.transpose(trails5[index]))
        elif markerResult5[index]==1:
            right.append(np.transpose(trails5[index]))
    print(len(left))
    print(left[0].shape)
    result=CSP(left,right)
    #result=CSP([np.transpose(trails1[0]),np.transpose(trails1[1]),np.transpose(trails1[2]),np.transpose(trails1[4]),np.transpose(trails1[6]),np.transpose(trails1[8]),np.transpose(trails1[10]),np.transpose(trails1[14]),np.transpose(trails1[15]),np.transpose(trails1[18])],[np.transpose(trails1[3]),np.transpose(trails1[5]),np.transpose(trails1[7]),np.transpose(trails1[9]),np.transpose(trails1[11]),np.transpose(trails1[12]),np.transpose(trails1[13]),np.transpose(trails1[16]),np.transpose(trails1[17]),np.transpose(trails1[19])])
#     result=CSP([np.transpose(trails2[1]),np.transpose(trails2[1]),np.transpose(trails2[2]),np.transpose(trails2[4]),np.transpose(trails2[6]),np.transpose(trails2[8]),np.transpose(trails2[10]),np.transpose(trails2[14]),np.transpose(trails2[15]),np.transpose(trails2[18])],[np.transpose(trails2[3]),np.transpose(trails2[5]),np.transpose(trails2[7]),np.transpose(trails2[9]),np.transpose(trails2[11]),np.transpose(trails2[12]),np.transpose(trails2[13]),np.transpose(trails2[16]),np.transpose(trails2[17]),np.transpose(trails2[19])])
#     result=CSP([np.transpose(trails3[0]),np.transpose(trails3[1]),np.transpose(trails2[2]),np.transpose(trails2[4]),np.transpose(trails2[6]),np.transpose(trails2[8]),np.transpose(trails2[10]),np.transpose(trails2[14]),np.transpose(trails2[15]),np.transpose(trails2[18])],[np.transpose(trails2[3]),np.transpose(trails2[5]),np.transpose(trails2[7]),np.transpose(trails2[9]),np.transpose(trails2[11]),np.transpose(trails2[12]),np.transpose(trails2[13]),np.transpose(trails2[16]),np.transpose(trails2[17]),np.transpose(trails2[19])])
#     result=CSP([np.transpose(trails4[0]),np.transpose(trails4[1]),np.transpose(trails2[2]),np.transpose(trails2[4]),np.transpose(trails2[6]),np.transpose(trails2[8]),np.transpose(trails2[10]),np.transpose(trails2[14]),np.transpose(trails2[15]),np.transpose(trails2[18])],[np.transpose(trails2[3]),np.transpose(trails2[5]),np.transpose(trails2[7]),np.transpose(trails2[9]),np.transpose(trails2[11]),np.transpose(trails2[12]),np.transpose(trails2[13]),np.transpose(trails2[16]),np.transpose(trails2[17]),np.transpose(trails2[19])])
#     result=CSP([np.transpose(trails5[0]),np.transpose(trails5[1]),np.transpose(trails2[2]),np.transpose(trails2[4]),np.transpose(trails2[6]),np.transpose(trails2[8]),np.transpose(trails2[10]),np.transpose(trails2[14]),np.transpose(trails2[15]),np.transpose(trails2[18])],[np.transpose(trails2[3]),np.transpose(trails2[5]),np.transpose(trails2[7]),np.transpose(trails2[9]),np.transpose(trails2[11]),np.transpose(trails2[12]),np.transpose(trails2[13]),np.transpose(trails2[16]),np.transpose(trails2[17]),np.transpose(trails2[19])])
    print ("csp")
    np.savetxt("CSP.csv", result, delimiter=",")
    newresult= np.genfromtxt("CSP.csv", delimiter=',')
    print (result.shape)
    print(newresult.shape)
    print(result[1])
    print(trails1.shape)
    x=np.zeros(shape=[100,8])
    for index in range(20):
        newtrail=np.dot(np.transpose(result),np.transpose(trails1[index]))
        totalVar=0
        for index2 in range(8):
            totalVar+=np.var(newtrail[index2])
        for index2 in range(8):
            x[index][index2]=math.log(np.var(newtrail[index2])/totalVar)
    
    for index in range(20,40):
        newtrail=np.dot(np.transpose(result),np.transpose(trails2[index-20]))
        totalVar=0
        for index2 in range(8):
            totalVar+=np.var(newtrail[index2])
        for index2 in range(8):
            x[index][index2]=math.log(np.var(newtrail[index2])/totalVar)
            
    for index in range(40,60):
        newtrail=np.dot(np.transpose(result),np.transpose(trails3[index-40]))
        totalVar=0
        for index2 in range(8):
            totalVar+=np.var(newtrail[index2])
        for index2 in range(8):
            x[index][index2]=math.log(np.var(newtrail[index2])/totalVar)
    for index in range(60,80):
        newtrail=np.dot(np.transpose(result),np.transpose(trails4[index-60]))
        totalVar=0
        for index2 in range(8):
            totalVar+=np.var(newtrail[index2])
        for index2 in range(8):
            x[index][index2]=math.log(np.var(newtrail[index2])/totalVar)
    for index in range(80,100):
        newtrail=np.dot(np.transpose(result),np.transpose(trails5[index-80]))
        totalVar=0
        for index2 in range(8):
            totalVar+=np.var(newtrail[index2])
        for index2 in range(8):
            x[index][index2]=math.log(np.var(newtrail[index2])/totalVar)





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
    example=np.zeros(shape=[20,8])
    for index in range(20):
        newtrail2=np.dot(np.transpose(result),np.transpose(trails5[index]))
        totalVar=0
        for index2 in range(8):
            totalVar+=np.var(newtrail2[index2])
        for index2 in range(8):
            example[index][index2]=math.log(np.var(newtrail2[index2])/totalVar)
    
    
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
    print(x.shape)
    x = np.transpose(preprocessing.normalize(np.transpose(x), norm='l2'))
    
    scoring = ['precision_macro']
#     clf = svm.SVC(kernel='linear', C=1, random_state=0)
#     scores = cross_validate(clf, x, y, scoring=scoring,
#                         cv=5)
#     filename = 'svmmodel.sav'
#     pickle.dump(clf, open(filename, 'wb'))
#     print(scores)
    tre = tree.DecisionTreeClassifier()
    scores = cross_validate(tre, x, y, scoring=scoring, cv=5)
    tre.fit(x, y)
    #     filename = 'svmmodel.sav'
#     pickle.dump(clf, open(filename, 'wb'))
    print(scores)

    lda=LinearDiscriminantAnalysis(n_components=2)
    lda.fit(x,y)
    scores = cross_validate(lda, x, y, scoring=scoring,
                        cv=5)
    print(scores)
    X_new = lda.transform(x)
    
    pl.scatter(X_new[:,0],X_new[:,0],marker='o',c=y)
    pl.show()
    print ("pre")
    filename = 'ldamodel.sav'
    pickle.dump(lda, open(filename, 'wb'))
    loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    
    #print (lda.predict(example))
    print(example.shape)
    print (lda.predict(example))
    #print (loaded_model.predict(example))
    print(markerResult5)
    print ("finish")
    
if __name__ == "__main__":
    main()
#FBCSP
#[[1,4],[4,8],[8,13],[13,22],[22,30]]
