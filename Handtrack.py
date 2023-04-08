import cv2 as cv
import mediapipe as mp 


cap = cv.VideoCapture(0)
hands= mp.solutions.hands
handmp = hands.Hands()
mpDraw = mp.solutions.drawing_utils

while True: 
    _ret , image = cap.read()

    imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = handmp.process(imgRGB)

    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks: 
        for lm in results.multi_hand_landmarks:
            connections = hands.HAND_CONNECTIONS
            mpDraw.draw_landmarks(image, lm, connections, mpDraw.DrawingSpec(color=(0, 255, 0), thickness=6, circle_radius=2), mpDraw.DrawingSpec(color=(255, 0, 0), thickness=6, circle_radius=2))

    cv.imshow("handtrack",image)

    if cv.waitKey(1) == ord('q'): 
        break

cap.release()
cv.destroyAllWindows()
