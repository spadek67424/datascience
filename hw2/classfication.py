import numpy as np
import sys
import os
from sklearn import linear_model,tree
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
'''
remember to tune the parameter
'''

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
	print("R=",logreg.score(trainx,trainy))
	return testy
def des(trainx,trainy,testx):
	clf=tree.DecisionTreeClassifier()
	clf.fit(trainx,trainy)
	testy=clf.predict(testx)
	print("D=",clf.score(trainx,trainy))
	return testy
def svm(trainx,trainy,testx):
	sv=LinearSVC()
	sv.fit(trainx,trainy)
	testy=sv.predict(testx)
	print("S=",sv.score(trainx,trainy))
	return testy
def NN(trainx,trainy,testx):
	nn= MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=400,learning_rate_init=0.001)
	nn.fit(trainx,trainy)
	print("N=",nn.score(trainx,trainy))
	testy=nn.predict(testx)
	return testy

if __name__ == '__main__':
	arg=sys.argv
	trainx,trainy=traindata(arg[2])
	testx=testdata(arg[3])
	if "R" in arg[1]:
		regression(trainx,trainy,testx)
	if "D" in arg[1]:
		des(trainx,trainy,testx)
	if "S" in arg[1]:
		svm(trainx,trainy,testx)
	if "N" in arg[1]:
		NN(trainx,trainy,testx)
	