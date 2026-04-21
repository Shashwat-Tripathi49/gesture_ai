import cv2
import mediapipe as mp
import numpy as np
import math

# Use the Google MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class FlexibleString:
    """
    A custom physics simulation for a flexible string/ribbon 
    that realistically trails behind a moving coordinate.
    """
    def __init__(self, start_pos, num_segments=20, segment_length=10):
        self.num_segments = num_segments
        self.segment_length = segment_length
        # Initialize all particles at the starting finger position
        self.points = [list(start_pos) for _ in range(num_segments)]
        # Pick a random vibrant color for this string
        self.color = tuple(np.random.randint(100, 255, 3).tolist())

    def update(self, head_pos):
        # The very first point rigidly follows the finger tip
        self.points[0] = list(head_pos)

        # The subsequent points follow the previous point via inverse kinematics
        # It creates a satisfying, organic ribbon-waving effect.
        for i in range(1, self.num_segments):
            dx = self.points[i-1][0] - self.points[i][0]
            dy = self.points[i-1][1] - self.points[i][1]
            dist = math.hypot(dx, dy)
            
            # If the segment is stretched beyond its length, pull the next point up
            if dist > self.segment_length:
                ratio = self.segment_length / dist
                self.points[i][0] = self.points[i-1][0] - dx * ratio
                self.points[i][1] = self.points[i-1][1] - dy * ratio

    def draw(self, frame):
        # Render the flexible string by drawing thick, connected lines
        for i in range(1, self.num_segments):
            pt1 = (int(self.points[i-1][0]), int(self.points[i-1][1]))
            pt2 = (int(self.points[i][0]), int(self.points[i][1]))
            
            # Make the string taper off smoothly to look like a whip/ribbon
            thickness = max(1, int(15 * (1 - i / self.num_segments)))
            cv2.line(frame, pt1, pt2, self.color, thickness, cv2.LINE_AA)
            cv2.circle(frame, pt2, max(2, thickness-1), self.color, -1)


def main():
    # Setup hand tracking for BOTH hands (up to 10 strings at once)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2, 
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # MediaPipe landmark IDs for all 5 fingertips
    finger_tips = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
    
    # We maintain our non-perishable string objects here. 
    # Key = (hand_index, tip_id), Value = FlexibleString
    active_strings = {}

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            print("Failed to open camera.")
            return

        h, w, c = frame.shape
        print("Application is running. Press 'Q' inside the camera window to quit.")
        timestamp = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Mirror the camera so movements are natural
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            # Parse frame to MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp += 1
            result = landmarker.detect_for_video(mp_image, int(timestamp * 33.333))

            if result.hand_landmarks:
                for hand_idx, landmarks in enumerate(result.hand_landmarks):
                    # Attach a string to every single finger!
                    for tip_idx in finger_tips:
                        key = (hand_idx, tip_idx)
                        tip = landmarks[tip_idx]
                        tx, ty = int(tip.x * w), int(tip.y * h)
                        
                        # If a string isn't attached to this specific finger yet, spawn it!
                        if key not in active_strings:
                            active_strings[key] = FlexibleString(start_pos=(tx, ty))
                        
                        # Apply physics updates to make the string wave and follow the finger
                        active_strings[key].update((tx, ty))

            # Finally, draw all active, non-perishable strings on top of the live feed
            for key, string in active_strings.items():
                string.draw(frame)
                
            # Render instructions
            cv2.putText(frame, "Flexible Ribbons Trailing 10 Fingers | Quit: Q", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Gesture AI - Flexible Strings", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
