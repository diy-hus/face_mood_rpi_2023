import cv2 
import matplotlib.pyplot as plt 
import numpy as np
import random 
import time 
from collections import deque
from pyrf24 import RF24, RF24_PA_MAX, RF24_250KBPS

max_number = 36
size_fig = (2.5,1.5)

radio = RF24(22, 0)
radio.begin()
radio.setChannel(115)
radio.setPALevel(RF24_PA_MAX)
radio.setDataRate(RF24_250KBPS)
radio.openReadingPipe(0,0x0000000001)
radio.startListening()

class Timer:
    def __init__(self):
        self.start_time = self.start()
    def start(self):
        self.start_time = time.time()
    def estimate(self):
        return time.time() - self.start_time  
    
class DataReceive:
    def __init__(self):
        self.data_1 = deque([0]*max_number,maxlen=max_number)
        self.data_1_inter = 0
        self.data_2 = deque([0]*max_number,maxlen=max_number)
        self.data_2_inter = 0 
        self.data_3 = deque([0]*max_number,maxlen=max_number)
        self.data_3_inter = 0
        self.data_4 = deque([0]*max_number,maxlen=max_number)
        self.data_4_inter = 0

        self.data_1_pre = deque([0]*max_number,maxlen=max_number+2)
        self.data_1_inter_pre = 0
        self.data_2_pre = deque([0]*max_number,maxlen=max_number+2)
        self.data_2_inter_pre = 0 
        self.data_3_pre = deque([0]*max_number,maxlen=max_number+2)
        self.data_3_inter_pre = 0
        self.data_4_pre = deque([0]*max_number,maxlen=max_number+2)
        self.data_4_inter_pre = 0
    def add_data(self, node, id):
        if node == 1 and id == 3:
            self.data_1_inter += 1
        elif node == 2 and id == 3:
            self.data_2_inter += 1
        elif node == 3 and id == 3:
            self.data_3_inter += 1
        elif node == 4 and id == 3:
            self.data_4_inter += 1
    def agg_data(self):
        self.data_1.append(self.data_1_inter)
        self.data_2.append(self.data_2_inter)
        self.data_3.append(self.data_3_inter)
        self.data_4.append(self.data_4_inter)

        # self.data_1_inter = 0
        # self.data_2_inter = 0
        # self.data_3_inter = 0
        # self.data_4_inter = 0

        coeffs = np.polyfit(np.arange(len(self.data_1)), self.data_1, 3)
        p = np.poly1d(coeffs)
        new = p([len(self.data_1), len(self.data_1)+1])
        self.data_1_pre.extend(new)

        coeffs = np.polyfit(np.arange(len(self.data_2)), self.data_2, 3)
        p = np.poly1d(coeffs)
        new = p([len(self.data_2), len(self.data_2)+1])
        self.data_2_pre.extend(new)

        coeffs = np.polyfit(np.arange(len(self.data_3)), self.data_3, 3)
        p = np.poly1d(coeffs)
        new = p([len(self.data_3), len(self.data_3)+1])
        self.data_3_pre.extend(new)

        coeffs = np.polyfit(np.arange(len(self.data_4)), self.data_4, 3)
        p = np.poly1d(coeffs)
        new = p([len(self.data_4), len(self.data_4)+1])
        self.data_4_pre.extend(new)


        # self.data_2_pre.append(self.data_2_inter)
        # self.data_3_pre.append(self.data_3_inter)
        # self.data_4_pre.append(self.data_4_inter)

        # self.data_1_inter = 0
        # self.data_2_inter = 0
        # self.data_3_inter = 0
        # self.data_4_inter = 0

    def get_data(self):
        return self.data_1, self.data_2, self.data_3, self.data_4

    def get_data_pre(self):
        return self.data_1_pre, self.data_2_pre, self.data_3_pre, self.data_4_pre
    
timer1 = Timer()
timer1.start()
timer2 = Timer()
timer2.start()


windowName = "DIY-HUS"

def get_receive():
    if timer1.estimate()> 0.1:
        timer1.start()
        node = random.randint(1, 4)
        bt = random.randint(1, 4)
        return node, bt

def get_receive_real():
    data = radio.read(32)
    id_node = data[0]
    id_bt = data[1]
    return id_node, id_bt

data_receive = DataReceive()

def draw_fig(data, data_pre):
    fig = plt.figure(figsize=(size_fig))
    plt.plot(data, 'b:', color='blue', linewidth=2)
    plt.plot([int(i) for i in data_pre], color='red', )
    plt.xticks([])
    fig = plt.gcf()
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    w, h = fig.canvas.get_width_height()
    graph = np.fromstring(buf, dtype=np.uint8).reshape(h, w, 3)
    graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB)

    plt.clf()

    return graph

def draw(data_receive):
    data_receive.agg_data()
    data_1, data_2, data_3, data_4 = data_receive.get_data()
    data_1_pre, data_2_pre, data_3_pre, data_4_pre = data_receive.get_data_pre()

    graph1 = draw_fig(data_1, data_1_pre)
    graph2 = draw_fig(data_2, data_2_pre)
    graph3 = draw_fig(data_3, data_3_pre)
    graph4 = draw_fig(data_4, data_4_pre)

    return graph1, graph2, graph3, graph4




    

def main():
    is_fist = True
    while True:
        if is_fist:
            is_fist = False 
            #frame = np.zeros((592,1024,3), np.uint8)
            frame = cv2.imread("background.jpg")

        # receive = get_receive()
        receive = get_receive_real()
        if receive:
            data_receive.add_data(receive[0], receive[1])

        if timer2.estimate() > 1:
            timer2.start()
            
            graph1, graph2, graph3, graph4 = draw(data_receive) 
            # frame1 = np.concatenate([graph1, graph2],1)
            # frame2 = np.concatenate([graph4, graph3],1)
            # frame  = np.concatenate([frame1, frame2], 0)
            pos1 = (150,375)
            right_x = int(pos1[0]+size_fig[1]*100)
            right_y = int(pos1[1]+size_fig[0]*100)
            frame[pos1[0]:right_x, pos1[1]:right_y] = graph1
            
            pos2 = (150,710)
            right_x = int(pos2[0]+size_fig[1]*100)
            right_y = int(pos2[1]+size_fig[0]*100)
            frame[pos2[0]:right_x, pos2[1]:right_y] = graph2
            
            pos3 = (360,375)
            right_x = int(pos3[0]+size_fig[1]*100)
            right_y = int(pos3[1]+size_fig[0]*100)
            frame[pos3[0]:right_x, pos3[1]:right_y] = graph3
            
            pos4 = (360,710)
            right_x = int(pos4[0]+size_fig[1]*100)
            right_y = int(pos4[1]+size_fig[0]*100)
            frame[pos4[0]:right_x, pos4[1]:right_y] = graph4
            
        print(frame.shape)

        cv2.imshow(windowName, frame)

        if cv2.waitKey(1) == ord('q'):
            break

main()
