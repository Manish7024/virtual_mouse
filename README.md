# virtual_mouse

Hand Motion Controlled Cursor

A computer vision-based project that enables you to control your mouse cursor using simple hand gestures — no physical mouse required! This system uses a webcam feed to track hand movements in real-time and translates them into smooth, intuitive on-screen cursor control.

🚀 Features

🎯 Real-time Hand Tracking — Detects and tracks your hand using MediaPipe.

🖱️ Cursor Control — Move your index finger to control the cursor smoothly.

👆 Single Click — Pinch index finger and thumb.

✌️ Double Click — Pinch middle finger and thumb.

🧭 Scroll Mode — Bring index and middle finger together, then move vertically to scroll.

⚙️ Adjustable Sensitivity — Tweak thresholds and smoothing factors for personalized control.

🛑 Failsafe Protection — Instantly stop the script by moving your mouse to a screen corner.

🧠 Working Principle

Hand Detection: The webcam captures video frames and processes them with MediaPipe Hands to identify hand landmarks.

Gesture Recognition: Key fingertip positions (thumb, index, middle) are analyzed to determine gestures (click, double-click, scroll).

Cursor Mapping: The fingertip coordinates are mapped to the screen size using OpenCV and NumPy interpolation.

Action Execution: Based on gestures, PyAutoGUI triggers corresponding cursor or scroll actions on the system.

🧩 Technologies Used

Python 3

OpenCV — for image capture and visualization

MediaPipe — for hand tracking and landmark detection

PyAutoGUI — for controlling system-level cursor and clicks

NumPy — for efficient numerical operations
