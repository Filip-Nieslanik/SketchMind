import cv2
import mediapipe as mp


INDEX_TIP = 8  # mediapipe landmark index for index finger tip


class FingerTracker:
    def __init__(self):
        self.cap   = cv2.VideoCapture(0)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # smooth out finger position over last few frames
        self.history = []

    def get_finger_pos(self):
        success, frame = self.cap.read()
        if not success:
            return None, None

        frame = cv2.flip(frame, 1)  # mirror so it feels natural
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return frame, None

        hand = result.multi_hand_landmarks[0]
        tip  = hand.landmark[INDEX_TIP]

        h, w, _ = frame.shape
        x = int(tip.x * w)
        y = int(tip.y * h)

        # average last 5 positions so line doesnt shake
        self.history.append((x, y))
        if len(self.history) > 5:
            self.history.pop(0)

        avg_x = int(sum(p[0] for p in self.history) / len(self.history))
        avg_y = int(sum(p[1] for p in self.history) / len(self.history))

        cv2.circle(frame, (avg_x, avg_y), 10, (0, 255, 100), -1)

        return frame, (avg_x, avg_y)

    def release(self):
        self.cap.release()
