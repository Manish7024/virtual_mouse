import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# --- CONFIGURATION SETTINGS ---
# Screen dimensions
SCREEN_W, SCREEN_H = pyautogui.size()

# Hand tracking confidence levels
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8

# VERY IMPORTANT: This enables the safety feature to stop the script
# by moving your mouse to a screen corner.
pyautogui.FAILSAFE = True

# --- GESTURE & CONTROL SETTINGS ---
# Cursor smoothing factor (higher value = smoother but more lag)
SMOOTHING_FACTOR = 7

# Gesture distance thresholds (normalized distance)
# A smaller value means fingers must be closer to trigger an action.
CLICK_DISTANCE_THRESHOLD = 0.04  # For single click (index finger to thumb)
DOUBLE_CLICK_DISTANCE_THRESHOLD = 0.04  # For double click (middle finger to thumb)
SCROLL_MODE_DISTANCE_THRESHOLD = 0.05  # To enter scroll mode (index to middle finger)

# Scroll sensitivity
SCROLL_SENSITIVITY = 150

# Cooldown period (in seconds) to prevent multiple actions from a single gesture
ACTION_COOLDOWN = 0.5

# Visual feedback settings for clicks
CLICK_FEEDBACK_DURATION = 0.2

# --- INITIALIZATION ---
# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)
mp_draw = mp.solutions.drawing_utils

# --- RUNTIME VARIABLES ---
# For cursor smoothing
prev_x, prev_y = 0, 0

# To manage gesture cooldown
last_action_time = 0

# To manage scrolling state
prev_scroll_y = None

# To manage click visual feedback
click_pos, click_effect_time = None, 0


# Helper function to calculate distance between two points
def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two MediaPipe landmark points."""
    return np.linalg.norm(np.array([p1.x - p2.x, p1.y - p2.y]))


# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a natural, mirror-like view
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # Convert the BGR image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        lm = hand_landmarks.landmark

        # --- Get key finger landmarks ---
        thumb_tip = lm[4]
        index_tip = lm[8]
        middle_tip = lm[12]

        # Check if enough time has passed to perform a new action
        can_perform_action = (time.time() - last_action_time) > ACTION_COOLDOWN

        # --- GESTURE LOGIC ---

        # 1. SCROLL MODE: Activated when index and middle fingers are close
        if calculate_distance(index_tip, middle_tip) < SCROLL_MODE_DISTANCE_THRESHOLD:
            if prev_scroll_y is None:
                prev_scroll_y = index_tip.y

            scroll_diff = index_tip.y - prev_scroll_y

            # Scroll if vertical movement is significant
            if abs(scroll_diff) > 0.005:
                scroll_amount = int(-scroll_diff * SCROLL_SENSITIVITY)
                pyautogui.scroll(scroll_amount)

            prev_scroll_y = index_tip.y

        else:
            # If not scrolling, reset scroll tracker and handle cursor/clicks
            prev_scroll_y = None

            # 2. CURSOR MOVEMENT: Default action using the index finger
            screen_x = np.interp(index_tip.x, [0.1, 0.9], [0, SCREEN_W])
            screen_y = np.interp(index_tip.y, [0.1, 0.9], [0, SCREEN_H])

            curr_x = prev_x + (screen_x - prev_x) / SMOOTHING_FACTOR
            curr_y = prev_y + (screen_y - prev_y) / SMOOTHING_FACTOR
            pyautogui.moveTo(int(curr_x), int(curr_y))
            prev_x, prev_y = curr_x, curr_y

            # --- CLICK ACTIONS (only if not on cooldown) ---
            if can_perform_action:
                # 3. DOUBLE CLICK: Middle finger and thumb pinch
                if calculate_distance(middle_tip, thumb_tip) < DOUBLE_CLICK_DISTANCE_THRESHOLD:
                    pyautogui.doubleClick()
                    last_action_time = time.time()
                    click_pos = (int(middle_tip.x * frame_w), int(middle_tip.y * frame_h))
                    click_effect_time = time.time()

                # 4. SINGLE CLICK: Index finger and thumb pinch
                elif calculate_distance(index_tip, thumb_tip) < CLICK_DISTANCE_THRESHOLD:
                    pyautogui.click()
                    last_action_time = time.time()
                    click_pos = (int(index_tip.x * frame_w), int(index_tip.y * frame_h))
                    click_effect_time = time.time()

    # Draw visual feedback for clicks
    if click_pos and (time.time() - click_effect_time) < CLICK_FEEDBACK_DURATION:
        cv2.circle(frame, click_pos, 20, (0, 255, 0), 3)

    # Display the final frame
    cv2.imshow("Gesture Control (Press 'q' to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()