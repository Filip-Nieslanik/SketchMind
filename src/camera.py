import cv2
import mediapipe as mp
import urllib.request
import os

INDEX_TIP  = 8   # index finger tip
INDEX_PIP  = 6   # index finger middle joint
MIDDLE_TIP = 12  # middle finger tip
MIDDLE_PIP = 10  # middle finger middle joint

class FingerTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        base_options = mp.tasks.BaseOptions(
            model_asset_path=self._get_model_path()
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

        self.history = []

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
        # index finger is up when tip is higher than middle joint
        # in image coordinates y=0 is top, so smaller y = higher up
        return landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y

    def _middle_finger_up(self, landmarks):
        return landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y

    def get_finger_pos(self):
        success, frame = self.cap.read()
        if not success:
            return None, None, False

        frame  = cv2.flip(frame, 1)
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

        # smooth position
        self.history.append((x, y))
        if len(self.history) > 5:
            self.history.pop(0)

        avg_x = int(sum(p[0] for p in self.history) / len(self.history))
        avg_y = int(sum(p[1] for p in self.history) / len(self.history))

        # drawing mode = index finger up, middle finger down
        drawing = self._index_finger_up(landmarks) and not self._middle_finger_up(landmarks)

        # green dot when drawing, white when just tracking
        color = (0, 255, 100) if drawing else (200, 200, 200)
        cv2.circle(frame, (avg_x, avg_y), 12, color, -1)

        # show hint text on camera frame
        label = "DRAWING" if drawing else "move finger to draw"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame, (avg_x, avg_y), drawing

    def release(self):
        self.cap.release()