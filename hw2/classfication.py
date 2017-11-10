import numpy as np
import sys
import os
from sklearn import linear_model,tree
from sklearn.svm import LinearSVC,SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
'''
remember to tune the parameter
'''

def traindata(arg):
	data=np.genfromtxt(arg,delimiter=",")
	trainy=data[:,57]
	y=data.shape[1]
	trainx=np.delete(data,y-1,axis=1)
	return trainx,trainy

def crossdata(trainx,trainy):                              
	'''c means it's cross validation'''
	traincx,testcx,traincy,testcy=train_test_split(trainx,trainy,test_size=0.2)
	return traincx,testcx,traincy,testcy
def testdata(arg):
	testx=np.genfromtxt(arg,delimiter=",")
	return testx
def regression (trainx,trainy,testx):
	traincx,testcx,traincy,testcy=crossdata(trainx,trainy)
	print(traincy.shape)
	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(traincx,traincy)
	print("R=",logreg.score(testcx,testcy))
	testy=logreg.predict(testx)
	return testy
def des(trainx,trainy,testx):
	traincx,testcx,traincy,testcy=crossdata(trainx,trainy)
	clf=tree.DecisionTreeClassifier()
	clf.fit(traincx,traincy)
	testy=clf.predict(testx)
	print("D=",clf.score(testcx,testcy))
	return testy
def svm(trainx,trainy,testx):
	traincx,testcx,traincy,testcy=crossdata(trainx,trainy)
	sv=SVC(kernel="rbf")
	print("fdsaf")
	sv.fit(traincx,traincy)
	print("fdsaf")
	testy=sv.predict(testx)
	print("S=",sv.score(testcx,testcy))
	return testy
def NN(trainx,trainy,testx):
	traincx,testcx,traincy,testcy=crossdata(trainx,trainy)
	nn= MLPClassifier(hidden_layer_sizes=(1000,1000),max_iter=1000,learning_rate_init=0.001)
	nn.fit(traincx,traincy)
	print("N=",nn.score(testcx,testcy))
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
	