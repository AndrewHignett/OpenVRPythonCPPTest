import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from mediapipe.framework.formats import landmark_pb2
from math import ceil
import tkinter
import time
import socket

port = 8888
firstCall = True

#for pings in range(10000):
    #client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #client_socket.settimeout(1.0)
    #message = b'test'
    #addr = ("127.0.0.1", 8888)
    #start = time.time()
    #client_socket.sendto(message, addr)
    #try:
    #    data, server = client_socket.recvfrom(1024)
    #    end = time.time()
    #    elapsed = end - start
    #    print(f'{data} {pings} {elapsed}')
    #except socket.timeout:
    #    print('REQUEST TIMED OUT')

def sendToServer(thisMessage):
     client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
     client_socket.settimeout(0.002)
     addr = ("127.0.0.1", port)
     client_socket.sendto((thisMessage), addr)

def connect():
    print("connect\n")
    #btnAddTrackers["state"] = "normal"
    btnConnect["state"] = "disabled"
    btnCalibrate["state"] = "normal"
    sendToServer(b'%')
    return 0;

def calibrate():
    pointsString = "+"
    first = True
    for point in calibrationTracking():
        for singularPoint in point:
            if (not first):
                pointsString += ","
                pointsString += str(singularPoint)
            else:
                pointsString += str(singularPoint)
                first = False

    #stick UI in python
    print(pointsString)
    print("big boi calibration\n")
    
    global firstCall
    if (firstCall):
        btnCalibrate2["state"] = "normal"    
        btnCalibrate["state"] = "disabled" 
        firstCall = False
    else:
        btnAddTrackers["state"] = "normal"
        btnCalibrate2["state"] = "disabled"
    #will need to send points to C++, from there calibration can be determined
    #sendToServer(b'+')
    sendToServer(str.encode(pointsString))
    #use left and right knew and ankles when we have one - realistically we only want to send the important points to the server
    return 0

def addTrackers():
    print("Add tracker\n")
    btnTracking["state"] = "normal"
    btnAddTrackers["state"] = "disabled"
    sendToServer(b'#')
    return 0;

#Need to stay still for this to work
def calibrationTracking():
    #worth repeating until we get goo points, but that can be set up later
    # For webcam input:
    capturedPoints = False
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 427)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    w, h = 3, 3#7
    calibrationPointsTracked = [[float(0.0) for x in range(w)] for y in range(h)]
    pointTrackingTracker = [0, 0, 0]
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if (results.pose_landmarks is not None):
                #print(results.pose_landmarks.landmark[17].visibility, results.pose_landmarks.landmark[18].visibility, results.pose_landmarks.landmark[19].visibility, results.pose_landmarks.landmark[20].visibility, results.pose_landmarks.landmark[0].visibility)
                #need to adjust this based on visibility and capture 5 points when appropraitely visible
                if ((results.pose_landmarks.landmark[17].visibility or results.pose_landmarks.landmark[19].visibility) > 0.6 and pointTrackingTracker[0] < 5):
                    calibrationPointsTracked[0][0] += results.pose_landmarks.landmark[17].x/10 + results.pose_landmarks.landmark[19].x/10#left controller
                    calibrationPointsTracked[0][1] += results.pose_landmarks.landmark[17].y/10 + results.pose_landmarks.landmark[19].y/10#left controller
                    calibrationPointsTracked[0][2] += results.pose_landmarks.landmark[17].z/10 + results.pose_landmarks.landmark[19].z/10#left controller
                    pointTrackingTracker[0] += 1
                if ((results.pose_landmarks.landmark[18].visibility or results.pose_landmarks.landmark[20].visibility) > 0.6 and pointTrackingTracker[1] < 5):
                    calibrationPointsTracked[1][0] += results.pose_landmarks.landmark[18].x/10 + results.pose_landmarks.landmark[20].x/10#right controller
                    calibrationPointsTracked[1][1] += results.pose_landmarks.landmark[18].y/10 + results.pose_landmarks.landmark[20].y/10#right controller
                    calibrationPointsTracked[1][2] += results.pose_landmarks.landmark[18].z/10 + results.pose_landmarks.landmark[20].z/10#right controller
                    pointTrackingTracker[1] += 1
                if (results.pose_landmarks.landmark[0].visibility > 0.6 and pointTrackingTracker[2] < 5):
                    calibrationPointsTracked[2][0] += results.pose_landmarks.landmark[0].x/5 #nose - hmd?
                    calibrationPointsTracked[2][1] += results.pose_landmarks.landmark[0].y/5 #nose - hmd?
                    calibrationPointsTracked[2][2] += results.pose_landmarks.landmark[0].z/5 #nose - hmd?    
                    pointTrackingTracker[2] += 1
                if (pointTrackingTracker[0] > 4 and pointTrackingTracker[1] > 4 and pointTrackingTracker[2] > 4):
                    capturedPoints = True
                print(pointTrackingTracker)
            if (cv2.waitKey(5) & 0xFF == 27)or(capturedPoints):
                #treat left controller as (0,0,0)
                #calibrationPointsTracked[1][0] -= calibrationPointsTracked[0][0]
                #calibrationPointsTracked[1][1] -= calibrationPointsTracked[0][1]
                #calibrationPointsTracked[1][2] -= calibrationPointsTracked[0][2]
                #calibrationPointsTracked[2][0] -= calibrationPointsTracked[0][0]
                #calibrationPointsTracked[2][1] -= calibrationPointsTracked[0][1]
                #calibrationPointsTracked[2][2] -= calibrationPointsTracked[0][2]
                #calibrationPointsTracked[0][0] = 0
                #calibrationPointsTracked[0][1] = 0
                #calibrationPointsTracked[0][2] = 0
                calibrationPointsTracked[0][0] -= calibrationPointsTracked[2][0]
                calibrationPointsTracked[0][1] -= calibrationPointsTracked[2][1]
                calibrationPointsTracked[0][2] -= calibrationPointsTracked[2][2]
                calibrationPointsTracked[1][0] -= calibrationPointsTracked[2][0]
                calibrationPointsTracked[1][1] -= calibrationPointsTracked[2][1]
                calibrationPointsTracked[1][2] -= calibrationPointsTracked[2][2]
                calibrationPointsTracked[2][0] = 0
                calibrationPointsTracked[2][1] = 0
                calibrationPointsTracked[2][2] = 0
                break
    cap.release()
    return calibrationPointsTracked

def tracking():
    #This is not threaded, so the UI will freeze on this
    btnTracking["text"] = "Stop tracking"
    # For webcam input:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 427)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():

            #define/clear allPoints list
            allPoints = []

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            '''
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            '''

            if (results.pose_landmarks is not None):
                #for data_point in results.pose_landmarks.landmark:
                    #print('x is', data_point.x, 'y is', data_point.y, 'z is', data_point.z, 'visibility is', data_point.visibility)

                #WHAT LEVEL OF VISIBILITY IS APPROPRIATE?
                from mediapipe.framework.formats import landmark_pb2
                landmark_subset = landmark_pb2.NormalizedLandmarkList(
                      landmark = [
                          results.pose_landmarks.landmark[27],#left ankle - tracker 1
                          results.pose_landmarks.landmark[28],#right ankle - tracker 2
                          results.pose_landmarks.landmark[0],#nose - hmd?
                          #results.pose_landmarks.landmark[25],#left knee
                          #results.pose_landmarks.landmark[26],#right knee
                          results.pose_landmarks.landmark[23],#left hip - for waist
                          results.pose_landmarks.landmark[24],#right hip - for waist
                          #results.pose_landmarks.landmark[1]#left eye inner, used for waist location
                          #hand position had be used for controllers 20 and 18 average gives rough midpoint of right controller and 17 and 19 left conntroller
                          results.pose_landmarks.landmark[17],#left pinky
                          results.pose_landmarks.landmark[19],#left index
                          results.pose_landmarks.landmark[18],#right pink
                          results.pose_landmarks.landmark[20]#right index
                          #IN THEORY THE WAIST POINT SHOULD BE 0,0,0 - we can treat it as this then modify accordingly (with rotations and x-y-z translationns based on hand positions
                      ]
                )
            
                #might be a good idea to smooth it out
                '''
                for i in range (8):
                    landmark_subset.landmark[i].x = ceil (landmark_subset.landmark[i].x * 100) / 100.0
                    landmark_subset.landmark[i].y = ceil (landmark_subset.landmark[i].y * 100) / 100.0
                    landmark_subset.landmark[i].z = ceil (landmark_subset.landmark[i].z * 100) / 100.0
                    landmark_subset.landmark[i].visibility = ceil (landmark_subset.landmark[i].visibility * 100) / 100.0
                '''
                #landmark_subset.landmark[7].x = results.pose_landmarks.landmark[23].x + (results.pose_landmarks.landmark[23].x - results.pose_landmarks.landmark[24].x)/2
                #landmark_subset.landmark[7].y = results.pose_landmarks.landmark[23].y + (results.pose_landmarks.landmark[23].y - results.pose_landmarks.landmark[24].y)/2
                #landmark_subset.landmark[7].z = 0.0 #midpoint where hips are, depth
                #landmark_subset.landmark[7].visibility = 1.0
            


                #currently adjusting based on nright hand
                pointsString = "";
                
                first = True;
                #for data_point in landmark_subset.landmark:
                #    allPoints.append(data_point.x)
                #    allPoints.append(data_point.y)
                #    allPoints.append(data_point.z)
                #    if (not first):
                #        #pointsString += "," + str(data_point.x) + "," + str(data_point.y) + "," + str(data_point.z)
                #        pointsString += "," + str(data_point.x - (results.pose_landmarks.landmark[18].x + results.pose_landmarks.landmark[20].x)/2) + "," + str(data_point.y - (results.pose_landmarks.landmark[18].y + results.pose_landmarks.landmark[20].y)/2) + "," + str(data_point.z - (results.pose_landmarks.landmark[18].z + results.pose_landmarks.landmark[20].z)/2)
                #    else:
                #        #pointsString += str(data_point.x) + "," + str(data_point.y) + "," + str(data_point.z)
                #        pointsString += str(data_point.x - (results.pose_landmarks.landmark[18].x + results.pose_landmarks.landmark[20].x)/2) + "," + str(data_point.y - (results.pose_landmarks.landmark[18].y + results.pose_landmarks.landmark[20].y)/2) + "," + str(data_point.z - (results.pose_landmarks.landmark[18].z + results.pose_landmarks.landmark[20].z)/2)
                #        first = False;
                
                #leftPointX = (results.pose_landmarks.landmark[17].x + results.pose_landmarks.landmark[19].x)/2
                #leftPointY = (results.pose_landmarks.landmark[17].y + results.pose_landmarks.landmark[19].y)/2
                #leftPointZ = (results.pose_landmarks.landmark[17].z + results.pose_landmarks.landmark[19].z)/2
                #hmdX = results.pose_landmarks.landmark[0].x - leftPointX
                #hmdY = results.pose_landmarks.landmark[0].y - leftPointY
                #hmdZ = results.pose_landmarks.landmark[0].z - leftPointZ
                #leftAnkleX = results.pose_landmarks.landmark[27].x - leftPointX
                #leftAnkleY = results.pose_landmarks.landmark[27].y - leftPointY
                #leftAnkleZ = results.pose_landmarks.landmark[27].z - leftPointZ
                #rightAnkleX = results.pose_landmarks.landmark[28].x - leftPointX
                #rightAnkleY = results.pose_landmarks.landmark[28].y - leftPointY
                #rightAnkleZ = results.pose_landmarks.landmark[28].z - leftPointZ
                hmdX = results.pose_landmarks.landmark[0].x
                hmdY = results.pose_landmarks.landmark[0].y
                hmdZ = results.pose_landmarks.landmark[0].z
                leftAnkleX = results.pose_landmarks.landmark[27].x - hmdX
                leftAnkleY = results.pose_landmarks.landmark[27].y - hmdY
                leftAnkleZ = results.pose_landmarks.landmark[27].z - hmdZ
                rightAnkleX = results.pose_landmarks.landmark[28].x - hmdX
                rightAnkleY = results.pose_landmarks.landmark[28].y - hmdY
                rightAnkleZ = results.pose_landmarks.landmark[28].z - hmdZ
                #pointsString += str((results.pose_landmarks.landmark[17].x + results.pose_landmarks.landmark[19].x)/2) + "," + str((results.pose_landmarks.landmark[17].y + results.pose_landmarks.landmark[19].y)/2) + "," + str((results.pose_landmarks.landmark[17].z + results.pose_landmarks.landmark[19].z)/2) + "," + str((results.pose_landmarks.landmark[18].x + results.pose_landmarks.landmark[20].x)/2) + "," + str((results.pose_landmarks.landmark[18].y + results.pose_landmarks.landmark[20].y)/2) + "," + str((results.pose_landmarks.landmark[18].z + results.pose_landmarks.landmark[20].z)/2) + "," + str(results.pose_landmarks.landmark[0].x) + "," + str(results.pose_landmarks.landmark[0].y) + "," + str(results.pose_landmarks.landmark[0].z)
                #pointsString += str(leftAnkleX) + "," + str(leftAnkleY) + "," + str(leftAnkleZ) + "," + str(rightAnkleX) + "," + str(rightAnkleY) + "," + str(rightAnkleZ) + "," + str(hmdX) + "," + str(hmdY) + "," + str(hmdZ)
                
                #live:
                #pointsString += str(leftAnkleX) + "," + str(leftAnkleY) + "," + str(leftAnkleZ) + "," + str(rightAnkleX) + "," + str(rightAnkleY) + "," + str(rightAnkleZ) + "," + str(hmdX) + "," + str(hmdY) + "," + str(hmdZ)
                #testing by intentionally mapping
                #'''
                leftAnkleX = results.pose_landmarks.landmark[17].x/2 + results.pose_landmarks.landmark[19].x/2 - hmdX
                leftAnkleY = results.pose_landmarks.landmark[17].y/2 + results.pose_landmarks.landmark[19].y/2 - hmdY
                leftAnkleZ = results.pose_landmarks.landmark[17].z/2 + results.pose_landmarks.landmark[19].z/2 - hmdZ
                rightAnkleX = results.pose_landmarks.landmark[18].z/2 + results.pose_landmarks.landmark[20].z/2 - hmdX
                rightAnkleY = results.pose_landmarks.landmark[18].z/2 + results.pose_landmarks.landmark[20].z/2 - hmdY
                rightAnkleZ = results.pose_landmarks.landmark[18].z/2 + results.pose_landmarks.landmark[20].z/2 - hmdZ
                pointsString += str(leftAnkleX) + "," + str(leftAnkleY) + "," + str(leftAnkleZ) + "," + str(rightAnkleX) + "," + str(rightAnkleY) + "," + str(rightAnkleZ) + "," + str(hmdX) + "," + str(hmdY) + "," + str(hmdZ)
                #'''


                image[0:480,0:640] = (0,0,0);
                mp_drawing.draw_landmarks(
                    image,
                    #landmark_subset)
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
 
                sendToServer(str.encode(pointsString))
                
                #client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                #client_socket.settimeout(0.002)
                #message = b'test'
                #addr = ("127.0.0.1", port)
                #start = time.time()
                #client_socket.sendto(str.encode(pointsString), addr)
                #end = time.time()
                #elapsed = end - start
                #print(f'{elapsed}')
                #probably necessary to wait for a reply before continuing, or could just go the relevant stuff then whatever we pick up next in the socket
                #try:
                #    data, server = client_socket.recvfrom(1024)
                #    end = time.time()
                #    elapsed = end - start
                #    print(f'{data} {pings} {elapsed}')
                #except socket.timeout:
                #    print('REQUEST TIMED OUT')

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
            
            #We then need to adjust based on head position, so it's all relative
            #^This may not be super smooth in terms of leg positions, but if we can get an idea of how much variation there is when
            #the head is "still", then we can choose not to move the points unless they're outside of this threshold
    cap.release()


#https://github.com/google/mediapipe/issues/2031#issuecomment-846123773

#Is it absolutely necessary to get an idea of room camera orientation? this can be calirbated
#yes
#if you stand up straight - hit calibrate, it should use ankle to knee (26-28, 25-27) as vertical
#evervything should get rotated around the midpoint - so, midpoint needs to be found

#waist midpoint, so worth centering on the poinnt and rotate around based on calibration



'''
PLAN

-Calibrate - gives angle of rotation
-!Work on head as midpoint
-!minus head x, y and z from all the others - head might need to be calculated differently from nose (especially if it disappears)
-!rotate around head based on calibration - calibration can be saved to a file and loaded in, if it exists (this would be good for a fixed camera)
-!x-y-z coordnates would be gathered at this point, this would just need to be scaled for steamvr
-!feed the above into tracker locations
-!add headset location to these coordinates - might be easier/better to use controller location (doesn't need quaternions, just x-y-z position)

-Try minusing instead of adding on the head position after mapping the point

EXTENDED FEATURES
-Tracker orientation
-calibrator file loading
-deal with head disappearing
-smooth out the shakes
-deal with trackers tempoarilly disappearing
-deal with trackers clearly in the wrong position - e.g. last few frame's positions, if too far - this may impact extreme speed motions
or even slow stuff - could average frames (introduces visible latency)
-button to rotate camera
'''

top = tkinter.Tk()

top.geometry("512x512")

btnConnect = tkinter.Button(top, text ="Connect", command = connect)
btnCalibrate = tkinter.Button(top, text ="Calibrate", command = calibrate)
btnCalibrate2 = tkinter.Button(top, text ="Calibrate2", command = calibrate)
btnAddTrackers = tkinter.Button(top, text ="Add Trackers", command = addTrackers)
btnTracking = tkinter.Button(top, text ="Start Tracking", command = tracking)

btnCalibrate["state"] = "disabled"
btnCalibrate2["state"] = "disabled"
btnAddTrackers["state"] = "disabled"
btnTracking["state"] = "disabled"

btnConnect.pack()
btnCalibrate.pack()
btnCalibrate2.pack()
btnAddTrackers.pack()
btnTracking.pack()
top.mainloop()

'''
1. I need to ensure the left and right controllers are the same as left and right hands, or track based on the head
2. The head may not be a great tracky point
3. The trackers jump all over the place, not totally sure what's causing that other than having some dodgy calibration points. I'll need to confirm visibility of points is good.
4. I need to make sure that x y and z directions are as expected - I'm pretty sure they're not, since moving my leg up generally makes it move in the y direction. Chances are that y and z are the other way around, since they can vary a little in 3d sometimes.
5. Does roration of trackers affect xyz directions at all

It may be good to ensure that the tracks are always within a radius of the person or up to a given distance from each other
Ignore points if too far - this should eliminate random crazy variations -somewhat set up

Could use 0 0 0 -> 0 0 0 as a calibration point
'''