import cv2
import mediapipe as mp
import numpy as np

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Initialize Camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to open camera.")
        return
        
    h, w, c = frame.shape
    
    # Create persistent canvas for the fading trails effect
    trail_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Interactive objects
    objects = [
        {"x": int(w*0.2), "y": int(h*0.5), "size": 60, "color": (0, 200, 255), "grabbed": False},
        {"x": int(w*0.8), "y": int(h*0.5), "size": 60, "color": (255, 100, 0), "grabbed": False}
    ]
    
    prev_index_pos = None

    print("App is running. Press 'Q' to quit in the video window.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for a natural mirror-like interaction
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        if trail_canvas.shape != (h,w,3):
            trail_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Fade out the previous trails slightly every frame. 
        # This gives the smooth "banishing" smoke/sparkler effect.
        trail_canvas = cv2.addWeighted(trail_canvas, 0.90, np.zeros_like(trail_canvas), 0.1, 0)
        
        # Convert BGR to RGB for mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = hands.process(rgb_frame)
        
        index_finger_pos = None
        thumb_pos = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index tip and thumb tip coordinates
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                
                index_finger_pos = (ix, iy)
                thumb_pos = (tx, ty)
        
        # Logic for Pinch and Draw
        if index_finger_pos and thumb_pos:
            # Calculate distance between thumb tip and index tip for pinch gesture
            distance = np.hypot(thumb_pos[0] - index_finger_pos[0], thumb_pos[1] - index_finger_pos[1])
            is_pinching = distance < 40  # Pinch trigger threshold
            
            # Draw a guiding circle on the index finger
            cv2.circle(frame, index_finger_pos, 8, (255, 255, 255), -1)
            
            grab_active = False

            # Check interaction with physical objects on screen
            for obj in objects:
                dist_to_obj = np.hypot(index_finger_pos[0] - obj["x"], index_finger_pos[1] - obj["y"])
                
                if is_pinching:
                    # If we pinch near an object (or already grabbing it), we drag it
                    if obj["grabbed"] or (dist_to_obj < obj["size"] and not any(o["grabbed"] for o in objects)):
                        obj["grabbed"] = True
                        obj["x"] = index_finger_pos[0]
                        obj["y"] = index_finger_pos[1]
                        grab_active = True
                else:
                    obj["grabbed"] = False
                    
            # If we are not grabbing things and not pinching, we just draw our fading trails!
            if not grab_active and not is_pinching:
                if prev_index_pos is not None:
                    # Draw smooth thick line
                    cv2.line(trail_canvas, prev_index_pos, index_finger_pos, (255, 180, 50), thickness=8)
                    # Add a glow point
                    cv2.circle(trail_canvas, index_finger_pos, 10, (150, 80, 20), -1)
            
            prev_index_pos = index_finger_pos
        else:
            prev_index_pos = None
            # Release all grabbed objects if hand disappears from frame
            for obj in objects:
                obj["grabbed"] = False

        # Overlay fading trails onto the main camera frame
        frame = cv2.addWeighted(frame, 1.0, trail_canvas, 1.0, 0)
        
        # Render the interactive objects
        for obj in objects:
            color = (0, 255, 150) if obj["grabbed"] else obj["color"]
            cv2.circle(frame, (obj["x"], obj["y"]), obj["size"], color, -1)
            cv2.circle(frame, (obj["x"], obj["y"]), obj["size"], (255, 255, 255), 3) # Outline

        # Display instructions
        cv2.putText(frame, "Draw: Point with Index | Grab: Pinch fingers on circles | Quit: Q", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Realistic Gesture AI", frame)
        
        # 'q' to break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
