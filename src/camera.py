import cv2
import mediapipe as mp


INDEX_TIP = 8  # index finger tip landmark


class FingerTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        # new mediapipe API (0.10+)
        self.detector = mp.tasks.vision.HandLandmarker
        base_options  = mp.tasks.BaseOptions(
            model_asset_path=self._get_model_path()
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

        self.history = []

    def _get_model_path(self):
        import urllib.request, os
        path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(path):
            print("Downloading hand landmark model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                path
            )
        return path

    def get_finger_pos(self):
        success, frame = self.cap.read()
        if not success:
            return None, None

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.hands.detect(mp_img)

        if not result.hand_landmarks:
            return frame, None

        tip = result.hand_landmarks[0][INDEX_TIP]
        h, w, _ = frame.shape
        x = int(tip.x * w)
        y = int(tip.y * h)

        self.history.append((x, y))
        if len(self.history) > 5:
            self.history.pop(0)

        avg_x = int(sum(p[0] for p in self.history) / len(self.history))
        avg_y = int(sum(p[1] for p in self.history) / len(self.history))

        cv2.circle(frame, (avg_x, avg_y), 10, (0, 255, 100), -1)

        return frame, (avg_x, avg_y)

    def release(self):
        self.cap.release()
