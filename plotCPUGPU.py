import matplotlib.pyplot as plt
import csv
  
x1 = []
y1 = []
x2 = []
y2 = []

  
with open('outputCPU.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        x1.append(int(row[0]))
        y1.append(float(row[1]))

with open('outputGPU.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        x2.append(int(row[0]))
        y2.append(float(row[1]))
  
plt.plot(x1, y1, '.', color = 'g', label = "CPU time")
plt.plot(x2, y2, '.', color = 'r', label = "GPU time")
plt.xticks(rotation = 25)
plt.xlabel('Frame')
plt.ylabel('Time')
plt.title('CPU vs GPU', fontsize = 20)
plt.grid()
plt.legend()
plt.show()
