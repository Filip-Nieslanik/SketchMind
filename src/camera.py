import cv2
import mediapipe as mp
import urllib.request
import os

INDEX_TIP  = 8   # tip of index finger
INDEX_PIP  = 6   # middle joint of index finger
MIDDLE_TIP = 12  # tip of middle finger
MIDDLE_PIP = 10  # middle joint of middle finger

class FingerTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 0 = first available camera

        base_options = mp.tasks.BaseOptions(
            model_asset_path=self._get_model_path()
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

        self.history = []  # I use this to smooth out the finger position

    def _get_model_path(self):
        path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(path):
            print("Downloading hand landmark model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                path
            )
        return path

    def _index_finger_up(self, landmarks):
        # tip has smaller y than the joint since y=0 is at the top of the image
        return landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y

    def _middle_finger_up(self, landmarks):
        return landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y

    def get_finger_pos(self):
        success, frame = self.cap.read()
        if not success:
            return None, None, False

        frame  = cv2.flip(frame, 1)  # mirror so it feels natural
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.hands.detect(mp_img)

        if not result.hand_landmarks:
            return frame, None, False

        landmarks = result.hand_landmarks[0]
        tip       = landmarks[INDEX_TIP]

        h, w, _ = frame.shape
        x = int(tip.x * w)
        y = int(tip.y * h)

        # average over last 5 frames to reduce shakiness
        self.history.append((x, y))
        if len(self.history) > 5:
            self.history.pop(0)

        avg_x = int(sum(p[0] for p in self.history) / len(self.history))
        avg_y = int(sum(p[1] for p in self.history) / len(self.history))

        # I draw when index is up and middle is down - middle acts as a pause button
        drawing = self._index_finger_up(landmarks) and not self._middle_finger_up(landmarks)

        color = (0, 255, 100) if drawing else (200, 200, 200)
        cv2.circle(frame, (avg_x, avg_y), 12, color, -1)

        label = "DRAWING" if drawing else "move finger to draw"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame, (avg_x, avg_y), drawing

    def release(self):
        self.cap.release()
