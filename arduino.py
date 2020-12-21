import pyvisa
import serial
import numpy as np
import matplotlib.pyplot as plt
import time

ser = serial.Serial('/dev/cu.usbmodem141301', 9600, timeout=10)

t = np.zeros(100)
y = np.zeros(100)

plt.ion()
plt.figure()
li, = plt.plot(t, y)
plt.ylim(0, 5)
plt.xlabel("time[s]")
plt.ylabel("Voltage[V]")

ser.write(b' ')
data = ser.readline().strip().rsplit()
tInt = float(data[0])

while True:
    try:
        ser.write(b' ')
        data = ser.readline().strip().rsplit()
        # 配列をキューと見たてて要素を追加・削除
        t = np.append(t, (float(data[0])-tInt)/10**6)
        t = np.delete(t, 0)
        y = np.append(y, float(data[1]))
        y = np.delete(y, 0)

        li.set_xdata(t)
        li.set_ydata(y)
        plt.xlim(min(t), max(t))
        plt.ylim(0, 6)
        plt.draw()

    except KeyboardInterrupt:
        ser.close()
        break
