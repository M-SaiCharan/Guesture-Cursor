import cv2
import mediapipe as mp
import pyautogui
import numpy as np

cap = cv2.VideoCapture(0)

hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

# Variables for smoothing
prev_x, prev_y = 0, 0
smoothening = 7

# Dragging state
dragging = False
right_click_done = False

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(frame_rgb)

    frame_height, frame_width, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                x, y = int(lm.x * frame_width), int(lm.y * frame_height)
                lm_list.append((id, x, y))

            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            if lm_list:
                index_finger = lm_list[8][1:]   # Index tip
                thumb = lm_list[4][1:]          # Thumb tip
                middle_finger = lm_list[12][1:] # Middle tip

                # Move Mouse
                x3 = np.interp(index_finger[0], (0, frame_width), (0, screen_width))
                y3 = np.interp(index_finger[1], (0, frame_height), (0, screen_height))

                curr_x = prev_x + (x3 - prev_x) / smoothening
                curr_y = prev_y + (y3 - prev_y) / smoothening

                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Distance calculations
                distance_index_thumb = np.hypot(thumb[0] - index_finger[0], thumb[1] - index_finger[1])
                distance_thumb_middle = np.hypot(thumb[0] - middle_finger[0], thumb[1] - middle_finger[1])
                distance_index_middle = np.hypot(index_finger[0] - middle_finger[0], index_finger[1] - middle_finger[1])

                # Left Click / Drag
                if distance_index_thumb < 30:
                    if not dragging:
                        dragging = True
                        pyautogui.mouseDown()
                else:
                    if dragging:
                        dragging = False
                        pyautogui.mouseUp()

                # Right Click
                if distance_thumb_middle < 30:
                    if not right_click_done:
                        right_click_done = True
                        pyautogui.rightClick()
                else:
                    right_click_done = False

                # Scroll
                if distance_index_middle < 40:  # Fingers close together means scrolling mode
                    fingers_center_y = (index_finger[1] + middle_finger[1]) // 2

                    if fingers_center_y < frame_height // 2 - 20:
                        pyautogui.scroll(20)  # Scroll up
                    elif fingers_center_y > frame_height // 2 + 20:
                        pyautogui.scroll(-20)  # Scroll down

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()