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
        self.cap = cv2.VideoCapture(0)  # use my default webcam; if we plug in another camera I can switch this to 1 or 2

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
        # MediaPipe needs a .task model file to detect hands.
        # I keep it out of the repo and download it automatically the first time the app runs.
        path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(path):
            print("Downloading hand landmark model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                path
            )
        return path

    def _index_finger_up(self, landmarks):
        # image coordinates have y=0 at the top, so a lower y value means a point is higher in the frame.
        # I use this to check if my index finger tip is raised above the middle joint.
        return landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y

    def _middle_finger_up(self, landmarks):
        # same logic for the middle finger; I use this as a pause signal when drawing with the camera.
        return landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y

    def get_finger_pos(self):
        # called every 30ms from update_camera() in app.py.
        # I return the current camera frame, the smoothed finger position, and whether I'm currently drawing.
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

        # keep the last 5 positions and average them.
        # this smooths out the raw MediaPipe finger jitter so my cursor feels more stable.
        self.history.append((x, y))
        if len(self.history) > 5:
            self.history.pop(0)

        avg_x = int(sum(p[0] for p in self.history) / len(self.history))
        avg_y = int(sum(p[1] for p in self.history) / len(self.history))

        # I consider drawing active when my index finger is raised and my middle finger is lowered.
        # The middle finger becomes a simple pause control: raise it to stop drawing without removing your hand.
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
