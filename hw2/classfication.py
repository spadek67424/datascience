import numpy as np
import sys
import os
from sklearn import linear_model

def traindata(arg):
	data=np.genfromtxt(arg,delimiter=",")
	trainy=data[:,57]
	y=data.shape[1]
	trainx=np.delete(data,y-1,axis=1)
	return trainx,trainy
		
def testdata(arg):
	testx=np.genfromtxt(arg,delimiter=",")
	return testx
def regression (trainx,trainy,testx):
	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(trainx,trainy)
	testy=logreg.predict(testx)
	print(testy.shape)
	print(logreg.score(trainx,trainy))




if __name__ == '__main__':
	arg=sys.argv
	trainx,trainy=traindata(arg[2])
	testx=testdata(arg[3])
	if "R" in arg[1]:
		regression(trainx,trainy,testx)
	
	