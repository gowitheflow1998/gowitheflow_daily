#-*- coding = utf-8 -*-
#@Time : 2020/11/15 01:42
#@Author : GoWitheFlow
#@File : 算法.py
#@Software : PyCharm

# The below is the example of format when outputting facial information using Amazon Facial Rekonigtion API
'''
{ "FaceDetails": [ { "BoundingBox": { "Width": 0.49051496386528015, "Height": 0.6313714981079102, "Left": 0.23612090945243835, "Top": 0.24247881770133972 }, 
                    "AgeRange": { "Low": 12, "High": 22 }, "Smile": { "Value": false, "Confidence": 99.35020446777344 }, "Eyeglasses": { "Value": false, "Confidence": 
                    70.59939575195312 }, "Sunglasses": { "Value": false, "Confidence": 95.77267456054688 }, "Gender": { "Value": "Female", "Confidence": 93.57445526123047 }, 
                    "Beard": { "Value": false, "Confidence": 98.7693099975586 }, "Mustache": { "Value": false, "Confidence": 99.35382843017578 }, "EyesOpen": 
                    { "Value": true, "Confidence": 93.42809295654297 }, "MouthOpen": { "Value": false, "Confidence": 93.07030487060547 }, "Emotions": 
                    [ { "Type": "CALM", "Confidence": 92.20979309082031 }, { "Type": "SAD", "Confidence": 5.162755966186523 }, { "Type": "FEAR", "Confidence"
                    : 1.7833609580993652 }, { "Type": "SURPRISED", "Confidence": 0.3106754422187805 }, { "Type": "CONFUSED", "Confidence": 0.25853773951530457 }, 
                     { "Type": "DISGUSTED", "Confidence": 0.09852556884288788 }, { "Type": "ANGRY", "Confidence": 0.09079523384571075 }, { "Type": "HAPPY", "Confidence"
                    : 0.08555883169174194 } ], "Landmarks": [ { "Type": "eyeLeft", "X": 0.3920268416404724, "Y": 0.4967755079269409 },
                    { "Type": "eyeRight", "X": 0.6154553294181824, "Y": 0.5006048679351807 }, { "Type": "mouthLeft", "X": 0.4074469208717346, "Y": 0.748733401298523 }, 
                    { "Type": "mouthRight", "X": 0.5938295125961304, "Y": 0.7520964741706848 }, { "Type": "nose", "X": 0.49705785512924194, "Y": 0.6357306838035583 }, 
                    { "Type": "leftEyeBrowLeft", "X": 0.30983632802963257, "Y": 0.43735605478286743 }, { "Type": "leftEyeBrowRight", "X": 0.37308743596076965, 
                    "Y": 0.41009455919265747 }, { "Type": "leftEyeBrowUp", "X": 0.437092125415802, "Y": 0.4247143864631653 }, { "Type": "rightEyeBrowLeft", "X": 
                    0.5654298663139343, "Y": 0.4266926348209381 }, { "Type": "rightEyeBrowRight", "X": 0.6314858198165894, "Y": 0.41410425305366516 }, 
                    { "Type": "rightEyeBrowUp", "X": 0.6988897919654846, "Y": 0.44348254799842834 }, { "Type": "leftEyeLeft", "X": 0.35231930017471313, "Y": 0.4947666525840759 }
                    , { "Type": "leftEyeRight", "X": 0.43578603863716125, "Y": 0.5000283718109131 }, { "Type": "leftEyeUp", "X": 0.3909331262111664, "Y": 0.48436495661735535 },
                    { "Type": "leftEyeDown", "X": 0.39239880442619324, "Y": 0.5079828500747681 }, { "Type": "rightEyeLeft", "X": 0.5707411766052246, "Y": 0.5022863745689392 }, 
                    { "Type": "rightEyeRight", "X": 0.6554444432258606, "Y": 0.49984991550445557 }, { "Type": "rightEyeUp", "X": 0.6154654622077942, "Y": 0.4881286919116974 },
                    { "Type": "rightEyeDown", "X": 0.6139041781425476, "Y": 0.5117237567901611 }, { "Type": "noseLeft", "X": 0.45673295855522156, "Y": 0.6603248119354248 }, 
                    { "Type": "noseRight", "X": 0.5398502945899963, "Y": 0.6616655588150024 }, { "Type": "mouthUp", "X": 0.4972870945930481, "Y": 0.7206116318702698 }, 
                    { "Type": "mouthDown", "X": 0.4968832731246948, "Y": 0.7955513596534729 }, { "Type": "leftPupil", "X": 0.3920268416404724, "Y": 0.4967755079269409 }, 
                    { "Type": "rightPupil", "X": 0.6154553294181824, "Y": 0.5006048679351807 }, { "Type": "upperJawlineLeft", "X": 0.26449066400527954, "Y": 0.4957331418991089 }
                    , { "Type": "midJawlineLeft", "X": 0.3063112199306488, "Y": 0.7658522129058838 }, { "Type": "chinBottom", "X": 0.49744656682014465, "Y": 0.9228888750076294 }
                    , { "Type": "midJawlineRight", "X": 0.7037524580955505, "Y": 0.7718601226806641 }, { "Type": "upperJawlineRight", "X": 0.7532891035079956, 
                    "Y": 0.5031331777572632 } ], "Pose": {"Roll": -0.23215393722057343, "Yaw": -0.3328110873699188, "Pitch": 1.4646071195602417 }, "Quality":
                    { "Brightness": 92.05117797851562, "Sharpness": 92.22801208496094 }, "Confidence": 99.99906158447266 } ] }
'''

# dict = the dict above 

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
