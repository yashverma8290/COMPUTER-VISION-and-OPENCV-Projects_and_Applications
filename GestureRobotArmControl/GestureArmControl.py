import cv2
import mediapipe as mp
import serial
import time

# Initialize serial communication with Arduino
arduino = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Tip landmark IDs for each finger
fingerTips = [4, 8, 12, 16, 20]

def fingers_up(hand_landmarks):
    finger_states = []

    # Thumb: compare x of tip and base (depends on hand orientation, using simple logic here)
    # if hand_landmarks.landmark[fingerTips[0]].x < hand_landmarks.landmark[fingerTips[0] - 1].x:
    #     finger_states.append(1)
    # else:
    #     finger_states.append(0)

    # Other fingers: tip y should be less than pip y (raised if higher on image)
    for tip_id in fingerTips[0:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            finger_states.append(1)
        else:
            finger_states.append(0)

    return finger_states

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to grab frame")
        continue

    # Convert to RGB for MediaPipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            fingers = fingers_up(handLms)
            print(fingers)

            # Format data to send like $10101
            data = "$" + ''.join(map(str, fingers))
            #arduino.write(data.encode())

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
arduino.close()
