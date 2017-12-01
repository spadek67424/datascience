from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import os
import sys
import pickle
def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict
if __name__=='__main__':
	data=list()
	label=list()
	'''
	b'data'
	b'labels'
	b'filenames'
	b'batch_label'

	for i in range()
	'''
	for i in range(1,len(sys.argv)):
		dicts=unpickle(sys.argv[i])
		#print(dicts.keys())
		data.append(dicts[b'data'])
		label.append(dicts[b'labels'])
		'''
		print(dicts[b'data'])
		input()
		print(dicts[b'labels'])
		input()
		'''
	trainx=np.array(data)
	trainy=np.array(label)	
	dicts=unpickle(sys.argv[6])
	testx=dicts[b'data']
	testy=dicts[b'labels']
	
	
	#print(dat.a)
