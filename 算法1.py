#-*- coding = utf-8 -*-
#@Time : 2020/11/15 01:42
#@Author : GoWitheFlow
#@File : 算法.py
#@Software : PyCharm


#把上面输出的字典结果弄到dict里就行

import numpy as np

L = []

emotion_labels = ['CALM','HAPPY','ANGRY','FEAR','DISGUSTED','SAD','SURPRISED']

for n in emotion_labels:
    for m in range(8):
        if dict['FaceDetails'][0]['Emotions'][m]['Type'] == n:
            L.append(dict['FaceDetails'][0]['Emotions'][m]['Confidence'])

coefficient_matrix = np.array(
    [[0.00,0.00,0.00],   # calm
    [2.77,1.21,1.42],    # happy
    [-1.98,1.10,0.60],   # angry
    [-0.93,1.30,-0.64],  # fear
    [-1.80,0.40,0.67],   # disgusted
    [-0.89,0.17,0.70],   # sad
    [1.72,1.71,0.22]])   # surprised

L_2 = []
for x in L:
    L_2.append([x])
emotion_matrix = np.array(L_2)

attention_score = (coefficient_matrix * emotion_matrix).sum()*1/3

print(attention_score)