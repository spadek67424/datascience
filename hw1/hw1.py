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
		row.insert(6,percent*100) # row[6] =totalper
	count=1    # omit first row

ite=len(sys.argv)
ite=int(ite)-1
xlabel=list()
for x in range(ite):
	cal2=list()
	st=arg[x+1]
	if 'E' in st:
		for x in range(5):
			cal2.append(cal[x+2])
		xlabel=cal[1][0]
	elif 'A' in st:
		for x in range(3):
			cal2.append(cal[x+8])
		xlabel=cal[7][0]
	elif 'W' in st:
		for x in range(3):
			cal2.append(cal[x+12])
		xlabel=cal[11][0]
	draw=list()
	draw2=list()
	draw3=list()
	xticks=list()
	countx=list()
	count=0
	maxy=0
	width=0.2
	

	
	if 'l' in st:
		for x in cal2:
			countx.append(count)
			count=count+1
			xticks.append(x[0])
			if int(round(float(x[2]),0))>maxy:
				maxy=int(round(float(x[2]),0))
			if int(round(float(x[4]),0))>maxy:
				maxy=int(round(x[4]),0)
			if int(round(float(x[6]),0))>maxy:
				maxy=int(round(float(x[6]),0))
			draw.append(x[2])
			plt.text(count-1,float(x[2])+1,round(float(x[2]),1))
			draw2.append(x[4])
			plt.text(count-1,float(x[4])+1,round(float(x[4]),1))
			draw3.append(x[6])
			plt.text(count-1,float(x[6])+1,round(float(x[6]),1))
		#plt.tick_params(labelsize=8)
		plt.xticks(countx,xticks,fontsize=7)
		plt.xlabel(xlabel)
		plt.xlim(-1,len(draw))
		plt.ylim(0,maxy+5)
		p1=plt.plot(draw)
		p2=plt.plot(draw2,'g')
		p3=plt.plot(draw3,'r')
		plt.legend((p1[0],p2[0],p3[0]),('men','female','total'))
		plt.show()

	elif 'b' in st:
		for x in cal2:
			countx.append(count)
			count=count+1
			xticks.append(x[0])
			if int(round(float(x[2]),0))>maxy:
				maxy=int(round(float(x[2]),0))
			if int(round(float(x[4]),0))>maxy:
				maxy=int(round(x[4]),0)
			if int(round(float(x[6]),0))>maxy:
				maxy=int(round(float(x[6]),0))
			draw.append(float(x[2]))
			draw2.append(float(x[4]))
			draw3.append(float(x[6]))
		plt.xticks(countx,xticks,fontsize=7)
		countx2=list()
		for c in countx:
			x=c-width
			countx2.append(x)
		countx=countx2
		dig=0
		for c in countx:
			plt.text(c,float(draw[dig])+1,round(float(draw[dig]),1))
			dig=dig+1
		p1=plt.bar(countx,draw,width)
		countx2=list()
		for c in countx:
			x=c+width
			countx2.append(x)
		countx=countx2
		dig=0
		for c in countx:
			plt.text(c,float(draw2[dig])+1,round(float(draw2[dig]),1))
			dig=dig+1
		p2=plt.bar(countx,draw2,width,color='red')
		countx2=list()
		for c in countx:
			x=c+width
			countx2.append(x)
		countx=countx2
		dig=0
		for c in countx:
			plt.text(c,float(draw3[dig])+1,round(float(draw3[dig]),1))
			dig=dig+1
		p3=plt.bar(countx,draw3,width,color='orange')
		plt.ylim(0,maxy+10)
		plt.legend((p1[0],p2[0],p3[0]),('men','female','total'))
		plt.show()
	elif 'p' in st:
		for x in cal2:
			countx.append(count)
			count=count+1
			xticks.append(x[0])
			draw.append(x[5])
		plt.pie(draw,labels=xticks,autopct='%1.1f%%')
		plt.show()
'''
plt.plot(draw)
plt.ylabel('some numbers')
plt.show()
'''
	#print(cal2)
'''
for a in cal:
	print(a)
'''