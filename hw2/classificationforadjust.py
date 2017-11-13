import numpy as np
import sys
import os
from sklearn import linear_model,tree
from sklearn.svm import LinearSVC,SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.ensemble import RandomForestClassifier
import csv
'''
remember to tune the parameter
'''
R=list()
S=list()
D=list()
N=list()
rd=list()
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
	traincx,testcx,traincy,testcy=crossdata(trainx,trainy)
	'''
	trainx=forstandard(trainx)
	traincx=forstandard(traincx)
	testcx=forstandard(testcx)
	testx=forstandard(testx)
	'''
	#parameter_candidates = [{'C': [1,10, 100, 1000,10000]},]
	#clf2 = GridSearchCV(estimator=linear_model.LogisticRegression(),param_grid=parameter_candidates, n_jobs=-1,cv=5) 
	nor=StandardScaler()
	nor.fit(traincx)
	traincx=nor.transform(traincx)
	testcx=nor.transform(testcx)
	logreg = linear_model.LogisticRegression(C=1)
	logreg.fit(traincx,traincy)
	#clf2.fit(trainx,trainy)
	#print('Best `C`:',clf2.best_estimator_.C)
	print("R=",logreg.score(testcx,testcy))
	testy=logreg.predict(testx)
	R.append(logreg.score(testcx,testcy))
	'''
	with open('out.txt','a') as f:
		f.write(str(clf2.best_estimator_.C))
		f.write("\n")
	'''
	return testy
def ran(trainx,trainy,testx):
	traincx,testcx,traincy,testcy=crossdata(trainx,trainy)
	clf = RandomForestClassifier()
	clf.fit(traincx,traincy)
	testy=clf.predict(testx)
	print("rd",clf.score(testcx,testcy))
	rd.append(clf.score(testcx,testcy))
	return testy
def des(trainx,trainy,testx):
	traincx,testcx,traincy,testcy=crossdata(trainx,trainy)
	'''
	trainx=forstandard(trainx)
	traincx=forstandard(traincx)
	testcx=forstandard(testcx)
	testx=forstandard(testx)
	'''
	nor=StandardScaler()
	nor.fit(traincx)
	traincx=nor.transform(traincx)
	testcx=nor.transform(testcx)
	clf=tree.DecisionTreeClassifier()
	clf.fit(traincx,traincy)
	testy=clf.predict(testx)
	print("D=",clf.score(testcx,testcy))
	D.append(clf.score(testcx,testcy))
	return testy
def svm(trainx,trainy,testx):
	traincx,testcx,traincy,testcy=crossdata(trainx,trainy)
	'''
	trainx=forstandard(trainx)
	traincx=forstandard(traincx)
	testcx=forstandard(testcx)
	testx=forstandard(testx)
	'''
	nor=StandardScaler()
	nor.fit(traincx)
	traincx=nor.transform(traincx)
	testcx=nor.transform(testcx)
	sv=SVC()
	sv.fit(traincx,traincy)
	testy=sv.predict(testx)
	print("S=",sv.score(testcx,testcy))
	S.append(sv.score(testcx,testcy))
	return testy
def NN(trainx,trainy,testx):
	traincx,testcx,traincy,testcy=crossdata(trainx,trainy)
	'''
	trainx=forstandard(trainx)
	traincx=forstandard(traincx)
	testcx=forstandard(testcx)
	testx=forstandard(testx)
	'''
	nor=StandardScaler()
	nor.fit(traincx)
	traincx=nor.transform(traincx)
	testcx=nor.transform(testcx)
	nn= MLPClassifier(hidden_layer_sizes=(100,100,200),max_iter=2000,solver="adam",learning_rate_init=0.001)
	''',learning_rate="adaptive")'''
	nn.fit(traincx,traincy)
	print("N=",nn.score(testcx,testcy))
	testy=nn.predict(testx)
	N.append(nn.score(testcx,testcy))
	return testy

if __name__ == '__main__':
	arg=sys.argv
	trainx,trainy=traindata(arg[2])
	testx=testdata(arg[3])
	for i in range(1):
		if "R" in arg[1]:
			testy=regression(trainx,trainy,testx)
		if "D" in arg[1]:
			testy=des(trainx,trainy,testx)
			testy2=ran(trainx,trainy,testx)
		if "S" in arg[1]:
			testy=svm(trainx,trainy,testx)
		if "N" in arg[1]:
			testy=NN(trainx,trainy,testx)
	np.savetxt("predict.csv",testy)
	np.savetxt("R.csv",R)
	np.savetxt("S.csv",S)
	np.savetxt("D.csv",D)
	np.savetxt("N.csv",N)

	