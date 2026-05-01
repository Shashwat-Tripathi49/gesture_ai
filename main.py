import cv2
import mediapipe as mp
import numpy as np
import math
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class ElasticString:
    def __init__(self, start_pos, color):
        self.color = color
        self.num_points = 25
        self.points = [np.array(start_pos, dtype=np.float64) for _ in range(self.num_points)]
        self.velocities = [np.array([0.0, 0.0], dtype=np.float64) for _ in range(self.num_points)]
        
        self.spring_constant = 0.35
        self.friction = 0.65
        self.gravity = 0.5

    def update(self, target_pos):
        self.points[0] = np.array(target_pos, dtype=np.float64)

        for i in range(1, self.num_points):
            prev = self.points[i - 1]
            curr = self.points[i]
            vel = self.velocities[i]

            dx = prev[0] - curr[0]
            dy = prev[1] - curr[1]

            ax = dx * self.spring_constant
            ay = dy * self.spring_constant

            vel[0] += ax
            vel[1] += ay
            
            vel[1] += self.gravity

            vel[0] *= self.friction
            vel[1] *= self.friction

            curr[0] += vel[0]
            curr[1] += vel[1]

    def draw(self, frame):
        for i in range(1, self.num_points):
            pt1 = (int(self.points[i-1][0]), int(self.points[i-1][1]))
            pt2 = (int(self.points[i][0]), int(self.points[i][1]))
            
            thickness = max(1, int(4 * (1 - (i / self.num_points) * 0.5)))
            
            cv2.line(frame, pt1, pt2, self.color, thickness, cv2.LINE_AA)

def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2, 
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.6
    )

    finger_tips = [4, 8, 12, 16, 20]
    
    left_colors = [
        (32, 176, 255),
        (246, 130, 59),
        (166, 184, 20),
        (94, 63, 244),
        (247, 85, 168)
    ]
    right_colors = [
        (11, 158, 245),
        (235, 99, 37),
        (136, 148, 13),
        (72, 29, 225),
        (234, 51, 147)
    ]

    active_strings = {}

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ret, frame = cap.read()
        if not ret:
            return

        pulse_time = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            timestamp_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            current_frame_keys = set()
            index_fingers = []

            if result.hand_landmarks:
                for idx, landmarks in enumerate(result.hand_landmarks):
                    handedness = result.handedness[idx][0].category_name
                    
                    palette = left_colors if handedness == 'Left' else right_colors
                    
                    hand_id = f"hand_{idx}_{handedness}"

                    for i, tip_idx in enumerate(finger_tips):
                        key = f"{hand_id}_{i}"
                        current_frame_keys.add(key)
                        
                        tip = landmarks[tip_idx]
                        tx, ty = int(tip.x * w), int(tip.y * h)
                        
                        if key not in active_strings:
                            active_strings[key] = ElasticString(start_pos=(tx, ty), color=palette[i])
                        
                        active_strings[key].update((tx, ty))

                        if i == 1:
                            index_fingers.append({
                                'hand_id': handedness,
                                'pos': (tx, ty)
                            })

            keys_to_delete = [k for k in active_strings.keys() if k not in current_frame_keys]
            for k in keys_to_delete:
                del active_strings[k]

            for key, string in active_strings.items():
                string.draw(frame)
                
            pulse_time += 0.15
            for i in range(len(index_fingers)):
                for j in range(i + 1, len(index_fingers)):
                    f1 = index_fingers[i]
                    f2 = index_fingers[j]
                    
                    if f1['hand_id'] != f2['hand_id']:
                        dx = f1['pos'][0] - f2['pos'][0]
                        dy = f1['pos'][1] - f2['pos'][1]
                        dist = math.hypot(dx, dy)
                        
                        if dist < 60:
                            pt1 = f1['pos']
                            pt2 = f2['pos']
                            
                            intensity = (math.sin(pulse_time) + 1) / 2
                            
                            thickness = int(3 + intensity * 4)
                            cv2.line(frame, pt1, pt2, (255, 255, 255), thickness, cv2.LINE_AA)
                       
            cv2.imshow("Gesture AI", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
