from cProfile import label
import matplotlib.pyplot as plt
import csv

X1=[]
Y1=[]
X2=[]
Y2=[]

with open('kalmann.txt', 'r') as datafile:
	plotting1 = csv.reader(datafile, delimiter=',')
	
	for ROWS in plotting1:
		X1.append(float(ROWS[0]))
		Y1.append(float(ROWS[1]))

with open('f_kalmann.txt', 'r') as datafile:
	plotting2 = csv.reader(datafile, delimiter=',')

	for ROWS in plotting2:
		X2.append(float(ROWS[0]))
		Y2.append(float(ROWS[1]))

plt.plot(X1,Y1 ,linewidth = '4',label = 'True Position')
plt.plot(X2,Y2,c = 'hotpink', label = 'Estimated position')
plt.title('Bot position in 2D')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()
