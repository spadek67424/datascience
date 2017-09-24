import urllib
import sys
import matplotlib.pyplot as plt
import urllib.request
import ssl
import os
arg=sys.argv
'''
for x in range(len(sys.argv)):
	print(x)
'''

ssl._create_default_https_context = ssl._create_unverified_context
url="https://ceiba.ntu.edu.tw/course/481ea4/hw1_data.csv"
webpage=urllib.request.urlopen(url)
html = webpage.read()
if os.path.exists('output.csv')!=True:
	with open('output.csv','wb') as f:
		f.write(html)
cal=list()
with open('output.csv','r') as f:
	for line in f:
		newline=line.replace("\n","")
		cal.append(newline.split(","))
count=0
cal2=list()
for row in cal:
	if row[1]!='' and count!=0:
		totalsmoke=float(row[1])*(float(row[2])/100)+float(row[3])*(float(row[4])/100)
		percent=float(row[1])+float(row[3])
		percent=totalsmoke/percent
		row.insert(5,totalsmoke)  # row[5] =totalsmoke
		row.insert(6,percent*100)
	count=1    # omit first row
print(cal)

'''
for a in cal:
	print(a)
'''