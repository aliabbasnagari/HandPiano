import cv2
import mediapipe as mp
import pygame.midi

mpHand = mp.solutions.hands
hand = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

pygame.midi.init()
output_id = pygame.midi.get_default_output_id()
midi_out = pygame.midi.Output(output_id)


A4 = 69
B4 = 71
C4 = 60
D4 = 62
E4 = 64
F4 = 65
G4 = 67
C5 = 72
D5 = 74
E5 = 76


velocity = 127


# A function takes x1, y1, x2 and y2 as parameters and returns the distance between them
def distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# A function that takes hand number and landmark number and returns midi note number


def get_midi_note_number(hand_number, landmark_number):
    if hand_number == 0:
        if landmark_number == 4:
            return C4
        elif landmark_number == 8:
            return D4
        elif landmark_number == 12:
            return E4
        elif landmark_number == 16:
            return F4
        elif landmark_number == 20:
            return G4
    elif hand_number == 1:
        if landmark_number == 4:
            return A4
        elif landmark_number == 8:
            return B4
        elif landmark_number == 12:
            return C5
        elif landmark_number == 16:
            return D5
        elif landmark_number == 20:
            return E5


finger_tips = [4, 8, 12, 16, 20]
key_press = [[False, False, False, False, False],
             [False, False, False, False, False]]

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand.process(imgRGB)
    h, w, c = img.shape
    if result.multi_hand_landmarks:
        handno = 0
        for handLM in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLM, mpHand.HAND_CONNECTIONS)

            for i in finger_tips:
                ind = int((i/4) - 1)
                if i == 4:
                    dist = distance(handLM.landmark[i].x * w, handLM.landmark[i].y * h,
                                    handLM.landmark[i + 1].x * w, handLM.landmark[i + 1].y * h)
                elif i != 4:
                    dist = distance(handLM.landmark[i].x * w, handLM.landmark[i].y * h,
                                    handLM.landmark[i - 3].x * w, handLM.landmark[i - 3].y * h)

                if dist < 70 and key_press[handno][ind] == False:
                    key_press[handno][ind] = True
                    midi_out.note_on(get_midi_note_number(handno, i), velocity)
                    print("Finger")
                elif dist >= 70:
                    key_press[handno][ind] = False

            for id, lm in enumerate(handLM.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

                if id == 5 or id == 9 or id == 13 or id == 17:
                    cv2.circle(img, (cx, cy), 10, (150, 0, 150), cv2.FILLED)
            handno += 1

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
        break

    cv2.imshow("Tracking", img)
    cv2.waitKey(1)
