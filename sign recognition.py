import cv2
import mediapipe as mp
import numpy as np
import autopy
from collections import deque
import time

class SignLanguageRecognition:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Gesture history for smoothing
        self.gesture_history = deque(maxlen=10)
        self.current_gesture = "None"
        self.confidence = 0.0
        
        # Sentence building
        self.sentence = []
        self.last_add_time = 0
        self.add_cooldown = 2.0  # seconds
        
        # Colors (BGR format)
        self.colors = {
            'bg': (245, 245, 245),
            'primary': (180, 100, 50),
            'success': (100, 200, 100),
            'warning': (50, 150, 255),
            'text': (50, 50, 50),
            'white': (255, 255, 255)
        }
        
        # Gesture database
        self.gesture_names = {
            'thumbs_up': 'Yes/Good',
            'thumbs_down': 'No/Bad',
            'peace': 'Peace/Two',
            'ok': 'OK',
            'pointing': 'You/That',
            'open_palm': 'Stop/Five',
            'fist': 'Zero/Fist',
            'one': 'One',
            'three': 'Three',
            'four': 'Four',
            'call_me': 'Call Me',
            'love': 'Love You'
        }
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_finger_extended(self, landmarks, finger_tip_id, finger_pip_id):
        """Check if a finger is extended"""
        return landmarks[finger_tip_id].y < landmarks[finger_pip_id].y
    
    def recognize_gesture(self, hand_landmarks):
        """Recognize hand gesture based on landmarks"""
        landmarks = hand_landmarks.landmark
        
        # Get finger states
        thumb_extended = landmarks[4].x < landmarks[3].x  # Thumb
        index_extended = self.is_finger_extended(landmarks, 8, 6)
        middle_extended = self.is_finger_extended(landmarks, 12, 10)
        ring_extended = self.is_finger_extended(landmarks, 16, 14)
        pinky_extended = self.is_finger_extended(landmarks, 20, 18)
        
        # Count extended fingers
        extended_fingers = sum([
            thumb_extended,
            index_extended,
            middle_extended,
            ring_extended,
            pinky_extended
        ])
        
        # Calculate distances for specific gestures
        thumb_index_dist = self.calculate_distance(landmarks[4], landmarks[8])
        thumb_pinky_dist = self.calculate_distance(landmarks[4], landmarks[20])
        
        # Gesture Recognition Logic
        
        # Fist (all fingers closed)
        if extended_fingers == 0:
            return 'fist', 0.95
        
        # One finger (index only)
        if extended_fingers == 1 and index_extended:
            return 'one', 0.92
        
        # Peace sign or Two (index and middle)
        if extended_fingers == 2 and index_extended and middle_extended:
            return 'peace', 0.90
        
        # Three fingers
        if extended_fingers == 3 and index_extended and middle_extended and ring_extended:
            return 'three', 0.88
        
        # Four fingers (no thumb)
        if extended_fingers == 4 and not thumb_extended:
            return 'four', 0.87
        
        # Open palm (all five fingers)
        if extended_fingers == 5:
            return 'open_palm', 0.93
        
        # OK sign (thumb and index form circle)
        if thumb_index_dist < 0.05 and middle_extended and ring_extended and pinky_extended:
            return 'ok', 0.85
        
        # Thumbs up
        if thumb_extended and landmarks[4].y < landmarks[3].y and extended_fingers == 1:
            return 'thumbs_up', 0.88
        
        # Thumbs down
        if thumb_extended and landmarks[4].y > landmarks[3].y and extended_fingers == 1:
            return 'thumbs_down', 0.88
        
        # Call me (thumb and pinky)
        if thumb_extended and pinky_extended and extended_fingers == 2:
            return 'call_me', 0.86
        
        # Love you (thumb, index, pinky)
        if thumb_extended and index_extended and pinky_extended and extended_fingers == 3:
            return 'love', 0.87
        
        return 'unknown', 0.0
    
    def draw_ui(self, frame):
        """Draw the user interface"""
        height, width = frame.shape[:2]
        
        # Create top bar
        cv2.rectangle(frame, (0, 0), (width, 80), self.colors['primary'], -1)
        
        # Title
        cv2.putText(frame, "Sign Language Recognition System", 
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['white'], 2)
        cv2.putText(frame, "AI-Powered Communication Assistant", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
        
        # Draw gesture detection box
        if self.current_gesture != "None" and self.current_gesture != "unknown":
            # Detection box
            box_width = 350
            box_height = 140
            box_x = width - box_width - 20
            box_y = 100
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), 
                         (box_x + box_width, box_y + box_height), 
                         self.colors['success'], -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Border
            cv2.rectangle(frame, (box_x, box_y), 
                         (box_x + box_width, box_y + box_height), 
                         self.colors['success'], 3)
            
            # Gesture name
            gesture_text = self.gesture_names.get(self.current_gesture, self.current_gesture)
            cv2.putText(frame, "DETECTED:", (box_x + 15, box_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
            cv2.putText(frame, gesture_text, (box_x + 15, box_y + 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['white'], 3)
            
            # Confidence bar
            cv2.putText(frame, f"Confidence: {self.confidence*100:.1f}%", 
                       (box_x + 15, box_y + 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
            
            bar_width = int((box_width - 30) * self.confidence)
            cv2.rectangle(frame, (box_x + 15, box_y + 105), 
                         (box_x + 15 + bar_width, box_y + 120),
                         self.colors['white'], -1)
            
        # Draw sentence panel
        panel_height = 120
        panel_y = height - panel_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (width, height), 
                     self.colors['bg'], -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        cv2.rectangle(frame, (0, panel_y), (width, height), 
                     self.colors['primary'], 3)
        
        # Sentence title
        cv2.putText(frame, "Constructed Sentence:", (20, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Display sentence
        sentence_text = " ".join(self.sentence) if self.sentence else "Gestures will appear here..."
        text_color = self.colors['text'] if self.sentence else (150, 150, 150)
        
        # Word wrap for long sentences
        max_width = width - 40
        words = sentence_text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
            (text_width, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
        
        # Display lines
        for i, line in enumerate(lines[:2]):  # Max 2 lines
            cv2.putText(frame, line, (20, panel_y + 60 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Instructions
        instructions = [
            "Press 'SPACE' to add gesture | 'C' to clear sentence | 'Q' to quit",
            "Show hand gestures to the camera for recognition"
        ]
        
        y_pos = 100
        for instruction in instructions:
            cv2.putText(frame, instruction, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y_pos += 25
        
        return frame
    
    def add_to_sentence(self):
        """Add current gesture to sentence"""
        current_time = time.time()
        if current_time - self.last_add_time >= self.add_cooldown:
            if self.current_gesture != "None" and self.current_gesture != "unknown":
                gesture_text = self.gesture_names.get(self.current_gesture, self.current_gesture)
                self.sentence.append(gesture_text)
                self.last_add_time = current_time
                print(f"Added to sentence: {gesture_text}")
    
    def clear_sentence(self):
        """Clear the sentence"""
        self.sentence = []
        print("Sentence cleared")
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks and recognize gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Recognize gesture
                gesture, confidence = self.recognize_gesture(hand_landmarks)
                self.gesture_history.append((gesture, confidence))
                
                # Smooth gesture detection
                if len(self.gesture_history) >= 5:
                    recent_gestures = [g[0] for g in list(self.gesture_history)[-5:]]
                    most_common = max(set(recent_gestures), key=recent_gestures.count)
                    if recent_gestures.count(most_common) >= 3:
                        self.current_gesture = most_common
                        avg_confidence = np.mean([g[1] for g in list(self.gesture_history)[-5:] 
                                                 if g[0] == most_common])
                        self.confidence = avg_confidence
        else:
            self.current_gesture = "None"
            self.confidence = 0.0
        
        # Draw UI
        frame = self.draw_ui(frame)
        
        return frame
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("=" * 60)
        print("Sign Language Recognition System Started")
        print("=" * 60)
        print("\nControls:")
        print("  SPACE  - Add current gesture to sentence")
        print("  C      - Clear sentence")
        print("  Q      - Quit application")
        print("\nSupported Gestures:")
        for gesture, name in self.gesture_names.items():
            print(f"  • {name}")
        print("\n" + "=" * 60)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                continue
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Display
            cv2.imshow('Sign Language Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.add_to_sentence()
            elif key == ord('c'):
                self.clear_sentence()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print("\nFinal Sentence:", " ".join(self.sentence))
        print("Application closed.")


if __name__ == "__main__":
    app = SignLanguageRecognition()
    app.run()
