#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter, deque
import cv2 as cv
import numpy as np
import mediapipe as mp
import imageio
import xr  # pyopenxr (pip install pyopenxr); adjust for your VR SDK

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)  # Webcam device
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()

def main():
    args = get_args()
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = True

    # Webcam setup
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # VR setup (OpenXR example)
    xr_context = xr.create_context()  # Initialize VR context
    xr_session = xr.create_session(xr_context)

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    recording = False
    frames = []
    max_recording_seconds = 5
    gif_fps = 10
    start_time = None
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Webcam capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # MediaPipe detection
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        # VR hand tracking
        vr_landmarks = get_vr_landmarks(xr_session, cap_width, cap_height)

        # Fuse landmarks (prefer VR if available, fallback to MediaPipe)
        if vr_landmarks:
            landmarks = vr_landmarks
            handedness = None  # VR-specific handedness logic needed
        elif results.multi_hand_landmarks:
            landmarks = calc_landmark_list(debug_image, results.multi_hand_landmarks[0])
            handedness = results.multi_handedness[0]
        else:
            landmarks = None
            handedness = None

        if landmarks:
            pre_processed_landmark_list = pre_process_landmark(landmarks)
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            if hand_sign_id == 2:
                point_history.append(landmarks[8])
            else:
                point_history.append([0, 0])

            pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
            finger_gesture_id = 0
            if len(pre_processed_point_history_list) == (history_length * 2):
                finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
            finger_gesture_history.append(finger_gesture_id)
            most_common_fg_id = Counter(finger_gesture_history).most_common()

            brect = calc_bounding_rect(debug_image, landmarks)
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            debug_image = draw_landmarks(debug_image, landmarks)
            debug_image = draw_info_text(debug_image, brect, handedness,
                                        keypoint_classifier_labels[hand_sign_id],
                                        point_history_classifier_labels[most_common_fg_id[0][0]])
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, -1)

        key = cv.waitKey(10)
        if key == 27:  # ESC
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

        if recording:
            small_frame = cv.resize(debug_image, (480, 270), interpolation=cv.INTER_AREA)
            frames.append(cv.cvtColor(small_frame, cv.COLOR_BGR2RGB))
            elapsed = (cv.getTickCount() - start_time) / cv.getTickFrequency()
            if elapsed > max_recording_seconds:
                recording = False
                save_gif(frames, gif_fps)
                print("Recording auto-stopped and GIF saved.")

        cv.imshow('Hand Gesture Recognition (VR + MediaPipe)', debug_image)

    cap.release()
    xr.destroy_session(xr_session)  # Cleanup VR
    xr.destroy_context(xr_context)
    cv.destroyAllWindows()

def get_vr_landmarks(session, width, height):
    frame_state = xr.predict_next_frame_state(session)
    hand_tracker = xr.create_hand_tracker(session)
    hands = xr.locate_hands(hand_tracker, frame_state)
    landmark_list = []
    if hands:
        for hand in hands:
            for joint in hand.joint_locations:  # Typically 26 joints
                # Map VR 3D coords to 2D image space (simplified)
                x = int(width * (joint.position.x + 0.5))  # Normalize [-0.5, 0.5] to [0, width]
                y = int(height * (0.5 - joint.position.y))  # Flip y-axis
                landmark_list.append([x, y])
    xr.destroy_hand_tracker(hand_tracker)
    return landmark_list[:21] if len(landmark_list) >= 21 else None  # Match MediaPipe’s 21 landmarks

def calc_bounding_rect(image, landmarks):
    landmark_array = np.array(landmarks, dtype=int)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]
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
    if 48 <= key <= 57:  # 0-9
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
        for i in range(len(landmark_point) - 1):
            cv.line(image, tuple(landmark_point[i]), tuple(landmark_point[i + 1]), (0, 255, 0), 2)
        for point in landmark_point:
            cv.circle(image, tuple(point), 5, (255, 255, 255), -1)
            cv.circle(image, tuple(point), 5, (0, 0, 0), 1)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = (handedness.classification[0].label[0:] if handedness else "VR") + ":" + hand_sign_text
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
    output_path = "hand_gesture_demo_vr.gif"
    imageio.mimsave(output_path, frames, fps=fps, loop=0, palettesize=64)
    print(f"GIF saved as {output_path}. Check file size!")

if __name__ == '__main__':
    main()