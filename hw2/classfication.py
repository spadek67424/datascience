import numpy as np
import sys
import os
from sklearn import linear_model,tree
from sklearn.svm import LinearSVC,SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize,StandardScaler
'''
remember to tune the parameter
'''

def traindata(arg):
	data=np.genfromtxt(arg,delimiter=",")
	data=shuffle(data)
	trainy=data[:,57]
	y=data.shape[1]
	trainx=np.delete(data,y-1,axis=1)
	#trainx=normalize(trainx)
	return trainx,trainy

def crossdata(trainx,trainy):                              
	'''c means it's cross validation'''
	traincx,testcx,traincy,testcy=train_test_split(trainx,trainy,test_size=0.2)
	'''
	nor=StandardScaler()
	traincx=nor.fit_transform(traincx)
	testcx=nor.fit_transform(testcx)
	'''
	return traincx,testcx,traincy,testcy
def forstandard(notstand):
	nor=StandardScaler()
	alstand=nor.fit_transform(notstand)
	return alstand
def testdata(arg):
	testx=np.genfromtxt(arg,delimiter=",")
	return testx
def regression (trainx,trainy,testx):
	trainx=forstandard(trainx)
	testx=forstandard(testx)
	traincx,testcx,traincy,testcy=crossdata(trainx,trainy)
	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(traincx,traincy)
	print("R=",logreg.score(testcx,testcy))
	testy=logreg.predict(testx)
	return testy
def des(trainx,trainy,testx):
	trainx=forstandard(trainx)
	testx=forstandard(testx)
	traincx,testcx,traincy,testcy=crossdata(trainx,trainy)
	clf=tree.DecisionTreeClassifier()
	clf.fit(traincx,traincy)
	testy=clf.predict(testx)
	print("D=",clf.score(testcx,testcy))
	return testy
def svm(trainx,trainy,testx):
	trainx=forstandard(trainx)
	testx=forstandard(testx)
	traincx,testcx,traincy,testcy=crossdata(trainx,trainy)
	sv=SVC()
	sv.fit(traincx,traincy)
	testy=sv.predict(testx)
	print("S=",sv.score(testcx,testcy))
	return testy
def NN(trainx,trainy,testx):
	trainx=forstandard(trainx)
	testx=forstandard(testx)
	traincx,testcx,traincy,testcy=crossdata(trainx,trainy)
	nn= MLPClassifier(hidden_layer_sizes=(100,100,200),max_iter=2000,solver="adam",learning_rate_init=0.001)
	''',learning_rate="adaptive")'''
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
	