import cv2
import mediapipe as mp
import urllib.request
import os

# landmark indices from MediaPipe hand model
# full map: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
INDEX_TIP  = 8   # tip of index finger
INDEX_PIP  = 6   # middle joint of index finger
MIDDLE_TIP = 12  # tip of middle finger
MIDDLE_PIP = 10  # middle joint of middle finger

class FingerTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 0 = first available camera, change to 1 or 2 if needed

        base_options = mp.tasks.BaseOptions(
            model_asset_path=self._get_model_path()
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1  # I only need one hand
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

        self.history = []  # stores last N positions for smoothing

    def _get_model_path(self):
        # MediaPipe needs a .task model file to detect hands
        # I download it automatically on first run so I don't have to ship it with the repo
        path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(path):
            print("Downloading hand landmark model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                path
            )
        return path

    def _index_finger_up(self, landmarks):
        # in image coordinates y=0 is at the top
        # so if the tip has a smaller y than the joint, the finger is pointing up
        return landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y

    def _middle_finger_up(self, landmarks):
        return landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y

    def get_finger_pos(self):
        # called every 30ms from update_camera() in app.py
        # returns the current frame, finger position, and whether I'm in drawing mode
        success, frame = self.cap.read()
        if not success:
            return None, None, False

        frame  = cv2.flip(frame, 1)  # mirror horizontally so movement feels natural
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.hands.detect(mp_img)

        if not result.hand_landmarks:
            return frame, None, False  # no hand detected, nothing to draw

        landmarks = result.hand_landmarks[0]
        tip       = landmarks[INDEX_TIP]

        # convert landmark coordinates (0-1) to pixel coordinates
        h, w, _ = frame.shape
        x = int(tip.x * w)
        y = int(tip.y * h)

        # keep the last 5 positions and use their average
        # this removes the shakiness that comes from raw MediaPipe output
        self.history.append((x, y))
        if len(self.history) > 5:
            self.history.pop(0)

        avg_x = int(sum(p[0] for p in self.history) / len(self.history))
        avg_y = int(sum(p[1] for p in self.history) / len(self.history))

        # drawing mode = index finger up AND middle finger down
        # middle finger acts as a pause button - raise it to stop drawing
        drawing = self._index_finger_up(landmarks) and not self._middle_finger_up(landmarks)

        # show visual feedback on the camera frame
        color = (0, 255, 100) if drawing else (200, 200, 200)
        cv2.circle(frame, (avg_x, avg_y), 12, color, -1)

        label = "DRAWING" if drawing else "move finger to draw"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame, (avg_x, avg_y), drawing

    def release(self):
        # called from stop_camera() in app.py when camera mode is turned off
        self.cap.release()
