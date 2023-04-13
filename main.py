import cv2
import mediapipe as mp

mpHand = mp.solutions.hands
hand = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand.process(imgRGB)
    if  result.multi_hand_landmarks:
        for handLM in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLM, mpHand.HAND_CONNECTIONS)

    cv2.imshow("Tracking", img)
    cv2.waitKey(1)