# pip install leap  # If available, or use Ultraleap’s LeapPython
# python your_script.py


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter, deque
import cv2 as cv
import numpy as np
import imageio
import Leap  # Leap Motion SDK

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", help='canvas width', type=int, default=960)
    parser.add_argument("--height", help='canvas height', type=int, default=540)
    return parser.parse_args()

def main():
    # Argument parsing
    args = get_args()
    cap_width = args.width
    cap_height = args.height
    use_brect = True

    # Leap Motion setup
    controller = Leap.Controller()
    controller.set_policy(Leap.Controller.POLICY_IMAGES)  # Enable image access

    # Model load
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    # FPS Measurement
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate and gesture history
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    # GIF recording variables
    recording = False
    frames = []
    max_recording_seconds = 5
    gif_fps = 10
    start_time = None

    mode = 0
    while True:
        fps = cvFpsCalc.get()
        frame = controller.frame()
        if not frame.is_valid:
            continue

        # Create a blank canvas for visualization
        image = np.zeros((cap_height, cap_width, 3), dtype=np.uint8)
        debug_image = copy.deepcopy(image)

        # Get Leap Motion landmarks
        landmarks = get_leap_landmarks(frame, cap_width, cap_height)
        if landmarks:
            pre_processed_landmark_list = pre_process_landmark(landmarks)
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            if hand_sign_id == 2:  # Point gesture
                point_history.append(landmarks[8])  # Index fingertip
            else:
                point_history.append([0, 0])

            pre_processed_point_history_list = pre_process_point_history(image, point_history)
            finger_gesture_id = 0
            if len(pre_processed_point_history_list) == (history_length * 2):
                finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
            finger_gesture_history.append(finger_gesture_id)
            most_common_fg_id = Counter(finger_gesture_history).most_common()

            # Visualization (simplified, no handedness from Leap yet)
            handedness = "Left" if hand.is_left() else "Right"
            
            brect = calc_bounding_rect(debug_image, landmarks)
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            debug_image = draw_landmarks(debug_image, landmarks)
            debug_image = draw_info_text(debug_image, brect, None,  # No handedness
                                        keypoint_classifier_labels[hand_sign_id],
                                        point_history_classifier_labels[most_common_fg_id[0][0]])
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, -1)  # No number logging yet

        # Process Key (ESC: exit, 'r': toggle recording)
        key = cv.waitKey(10)
        if key == 27:  # ESC to exit
            break
        elif key == ord('r'):
            if not recording:
                recording = True
                frames = []
                start_time = cv.getTickCount()
                print("Recording started...")
            else:
                recording = False
                save_gif(frames, gif_fps)
                print("Recording stopped and GIF saved.")
        number, mode = select_mode(key, mode)

        # Record frames if recording
        if recording:
            small_frame = cv.resize(debug_image, (480, 270), interpolation=cv.INTER_AREA)
            frames.append(cv.cvtColor(small_frame, cv.COLOR_BGR2RGB))
            elapsed = (cv.getTickCount() - start_time) / cv.getTickFrequency()
            if elapsed > max_recording_seconds:
                recording = False
                save_gif(frames, gif_fps)
                print("Recording auto-stopped and GIF saved.")

        # Display
        cv.imshow('Hand Gesture Recognition (Leap Motion)', debug_image)

    cv.destroyAllWindows()

def get_leap_landmarks(frame, width, height):
    hands = frame.hands
    landmark_list = []
    for hand in hands:
        # Map Leap's 3D coordinates to 2D image space (simplified)
        palm = hand.palm_position
        landmark_list.append([int(width / 2 + palm.x), int(height / 2 - palm.y)])  # Wrist (approx.)
        for finger in hand.fingers:
            tip = finger.bone(Leap.Bone.TYPE_DISTAL).next_joint
            landmark_list.append([int(width / 2 + tip.x), int(height / 2 - tip.y)])
    return landmark_list if len(landmark_list) >= 21 else None  # Ensure 21 landmarks

def calc_bounding_rect(image, landmarks):
    landmark_array = np.array(landmarks, dtype=int)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]  # Wrist as base
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list))
    return [n / max_value for n in temp_landmark_list] if max_value > 0 else temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = temp_point_history[0] if temp_point_history else (0, 0)
    for i in range(len(temp_point_history)):
        temp_point_history[i][0] = (temp_point_history[i][0] - base_x) / image_width
        temp_point_history[i][1] = (temp_point_history[i][1] - base_y) / image_height
    return list(itertools.chain.from_iterable(temp_point_history))

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Simplified drawing (adapt as needed)
        for i in range(len(landmark_point) - 1):
            cv.line(image, tuple(landmark_point[i]), tuple(landmark_point[i + 1]), (0, 255, 0), 2)
        for idx, point in enumerate(landmark_point):
            cv.circle(image, tuple(point), 5, (255, 255, 255), -1)
            cv.circle(image, tuple(point), 5, (0, 0, 0), 1)

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = "Hand:" + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    if finger_gesture_text:
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, tuple(point), 1 + int(index / 2), (152, 251, 152), 2)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, f"FPS:{fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, f"FPS:{fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, f"MODE:{mode_string[mode - 1]}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def save_gif(frames, fps):
    if not frames:
        print("No frames to save!")
        return
    output_path = "hand_gesture_demo1.gif"
    imageio.mimsave(output_path, frames, fps=fps, loop=0, palettesize=64)
    print(f"GIF saved as {output_path}. Check file size!")

if __name__ == '__main__':
    main()