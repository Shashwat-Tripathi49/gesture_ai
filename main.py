import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Use the Google MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class FlexibleString:
    """
    A physics simulation for an organic, glowing ribbon trailing behind a finger.
    Features EMA temporal smoothing and Verlet Integration for momentum mapping.
    """
    def __init__(self, start_pos, num_segments=25, segment_length=15):
        self.num_segments = num_segments
        self.segment_length = segment_length
        
        # Physics nodes
        self.points = [np.array(start_pos, dtype=np.float64) for _ in range(num_segments)]
        self.prev_points = [np.array(start_pos, dtype=np.float64) for _ in range(num_segments)]
        
        # Physics setup
        self.gravity = np.array([0.0, 1.2]) # Y-axis gravity pulling strings down
        self.drag = 0.88 # Air friction map (lower = heavier stopping power)
        
        # Aesthetics (Pick a random hot color that blends down)
        self.base_color = tuple(np.random.randint(40, 255, 3).tolist())
        
        # EMA Filtering for webcam jitter removal
        self.smoothed_head = np.array(start_pos, dtype=np.float64)
        self.ema_alpha = 0.35 # Blend factor. Lower gives more lag but is buttery smooth

    def update(self, head_pos):
        # 1. EMA Filter the camera node
        raw_head = np.array(head_pos, dtype=np.float64)
        self.smoothed_head = self.ema_alpha * raw_head + (1.0 - self.ema_alpha) * self.smoothed_head
        
        # Manually set head to filtered node
        self.points[0] = np.copy(self.smoothed_head)
        self.prev_points[0] = np.copy(self.smoothed_head)
        
        # 2. Verlet Integration (Velocity Mapping)
        for i in range(1, self.num_segments):
            velocity = (self.points[i] - self.prev_points[i]) * self.drag
            self.prev_points[i] = np.copy(self.points[i])
            self.points[i] += velocity + self.gravity

        # 3. Geometric Relaxations
        # Multiple passes required so strings don't act like rubber bands
        iterations = 2
        for _ in range(iterations):
            for i in range(1, self.num_segments):
                dx = self.points[i-1][0] - self.points[i][0]
                dy = self.points[i-1][1] - self.points[i][1]
                dist = math.hypot(dx, dy)
                
                # Rigid constraint
                if dist > self.segment_length:
                    diff = dist - self.segment_length
                    nx = dx / dist
                    ny = dy / dist
                    self.points[i][0] += nx * diff
                    self.points[i][1] += ny * diff

    def draw(self, frame, glow_canvas):
        # Render the string onto two canvases simultaneously to prep for bloom
        for i in range(1, self.num_segments):
            pt1 = (int(self.points[i-1][0]), int(self.points[i-1][1]))
            pt2 = (int(self.points[i][0]), int(self.points[i][1]))
            
            # Smoothly taper off the tail
            thickness = max(1, int(25 * (1 - i / self.num_segments)))
            
            # Color gradient: Burn white hot at the finger tip, color at tail
            fraction = (i / self.num_segments)
            c = (
                int(255 * (1 - fraction) + self.base_color[0] * fraction),
                int(255 * (1 - fraction) + self.base_color[1] * fraction),
                int(255 * (1 - fraction) + self.base_color[2] * fraction)
            )
            
            # Draw highly blurred base onto the secondary glow canvas
            cv2.line(glow_canvas, pt1, pt2, c, thickness + 12, cv2.LINE_AA)
            cv2.circle(glow_canvas, pt2, max(2, thickness+2), c, -1)
            
            # Draw core dense element onto the video buffer
            cv2.line(frame, pt1, pt2, (255,255,255), max(1, int(thickness * 0.4)), cv2.LINE_AA)

def main():
    # Setup high confidence hand tracking
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2, 
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.6
    )

    finger_tips = [4, 8, 12, 16, 20]
    active_strings = {}

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            print("Failed to open camera.")
            return

        print("🔮 Magic Gesture AI is running. Press 'Q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Black frame to draw blooms onto
            glow_canvas = np.zeros_like(frame)

            # Accurate hardware clock for MediaPipe
            timestamp_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Route coordinates into our physics simulator
            if result.hand_landmarks:
                for hand_idx, landmarks in enumerate(result.hand_landmarks):
                    for tip_idx in finger_tips:
                        key = (hand_idx, tip_idx)
                        tip = landmarks[tip_idx]
                        tx, ty = int(tip.x * w), int(tip.y * h)
                        
                        if key not in active_strings:
                            active_strings[key] = FlexibleString(start_pos=(tx, ty))
                        
                        active_strings[key].update((tx, ty))

            # Trigger drawing
            for key, string in active_strings.items():
                string.draw(frame, glow_canvas)
                
            # --- The Neon Bloom Effect ---
            # Blur the background elements softly to emulate light scatter
            glow_canvas = cv2.GaussianBlur(glow_canvas, (55, 55), 0)
            
            # Composite blend the original video, the hot-center trails, and the heavy blur
            final_composed_frame = cv2.addWeighted(frame, 1.0, glow_canvas, 1.5, 0)
            
            # Draw UI
            cv2.putText(final_composed_frame, "Next Level Neon Physics | Quit: Q", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
            cv2.imshow("Gesture AI - Visual Overhaul", final_composed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

