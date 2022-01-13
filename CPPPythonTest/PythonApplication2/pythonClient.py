import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from mediapipe.framework.formats import landmark_pb2
from math import ceil
import tkinter



#Tried Boost, this is bad and does not work
#https://www.geeksforgeeks.org/how-to-call-c-c-from-python/ might be good

import time
import socket

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
     addr = ("127.0.0.1", 8888)
     client_socket.sendto((thisMessage), addr)

def calibrate():
    pointsString = "+"
    for point in calibrationTracking():
        pointsString += point
    #stick UI in python
    print("big boi calibration\n")
    btnConnect["state"] = "normal"
    btnCalibrate["state"] = "disabled"    
    #will need to send points to C++, from there calibration can be determined
    #sendToServer(b'+')
    sendToServer(str.encode(pointsString))
    #use left and right knew and ankles when we have one - realistically we only want to send the important points to the server
    return 0

def connect():
    print("connect\n")
    btnAddTrackers["state"] = "normal"
    btnConnect["state"] = "disabled"
    sendToServer(b'%')
    return 0;

def addTrackers():
    print("Add tracker\n")
    btnTracking["state"] = "normal"
    btnAddTrackers["state"] = "disabled"
    sendToServer(b'#')
    return 0;

def calibrationTracking():
    #worth repeating until we get goo points, but that can be set up later
    # For webcam input:
    capturedPoints = False
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 427)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
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
                calibrationPoints = [
                    results.pose_landmarks.landmark[27],#left ankle
                    results.pose_landmarks.landmark[28],#right ankle
                    results.pose_landmarks.landmark[25],#left knee
                    results.pose_landmarks.landmark[26],#right knee
                ]         
                capturedPoints = True
            if (cv2.waitKey(5) & 0xFF == 27)or(capturedPoints):
                break
    cap.release()
    return calibrationPoints

def tracking():
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
                from mediapipe.framework.formats import landmark_pb2
                landmark_subset = landmark_pb2.NormalizedLandmarkList(
                      landmark = [
                          results.pose_landmarks.landmark[27],#left ankle
                          results.pose_landmarks.landmark[28],#right ankle
                          results.pose_landmarks.landmark[0],#nose
                          #results.pose_landmarks.landmark[25],#left knee
                          #results.pose_landmarks.landmark[26],#right knee
                          results.pose_landmarks.landmark[23],#left hip
                          results.pose_landmarks.landmark[24],#right hip
                          #results.pose_landmarks.landmark[1]#left eye inner, used for waist location
                          #hand position had be used for controllers 20 and 18 average gives rough midpoint of right controller and 17 and 19 left conntroller
                          results.pose_landmarks.landmark[17],#left pinky
                          results.pose_landmarks.landmark[19],#left index
                          results.pose_landmarks.landmark[18],#right pink
                          results.pose_landmarks.landmark[20]#right index
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
            
                pointsString = "";
                i = 0
                for data_point in landmark_subset.landmark:
                    allPoints.append(data_point.x)
                    allPoints.append(data_point.y)
                    allPoints.append(data_point.z)
                    if (i < 1):
                        pointsString += "," + str(data_point.x) + "," + str(data_point.y) + "," + str(data_point.z)
                    else:
                        pointsString += str(data_point.x) + "," + str(data_point.y) + "," + str(data_point.z)
                    i += 1

                mp_drawing.draw_landmarks(
                    image,
                    #landmark_subset)
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


                client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                client_socket.settimeout(0.002)
                message = b'test'
                addr = ("127.0.0.1", 8888)
                start = time.time()
                client_socket.sendto(str.encode(pointsString), addr)
                end = time.time()
                elapsed = end - start
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

btnCalibrate = tkinter.Button(top, text ="Calibrate", command = calibrate)
btnConnect = tkinter.Button(top, text ="Connect", command = connect)
btnAddTrackers = tkinter.Button(top, text ="Add Trackers", command = addTrackers)
btnTracking = tkinter.Button(top, text ="Start Tracking", command = tracking)

btnConnect["state"] = "disabled"
btnAddTrackers["state"] = "disabled"
btnTracking["state"] = "disabled"

btnCalibrate.pack()
btnConnect.pack()
btnAddTrackers.pack()
btnTracking.pack()
top.mainloop()