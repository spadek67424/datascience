import urllib
import sys
import matplotlib.pyplot as plt
import urllib.request
import ssl
arg=sys.argv
for x in range(len(sys.argv)):
	print(x)
ssl._create_default_https_context = ssl._create_unverified_context
url="https://ceiba.ntu.edu.tw/course/481ea4/hw1_data.csv"
webpage=urllib.request.urlopen(url)
html = webpage.read()

with open('output.csv','wb') as f:
	f.write(html)
cal=list()
with open('output.csv','r') as f:
	for line in f:
		newline=line.replace("\n","")
		cal.append(newline.split(","))
print(cal)
'''
for a in cal:
	print(a)
'''