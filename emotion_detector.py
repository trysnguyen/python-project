# emotion_detector.py
import cv2
import numpy as np
import os

class EmotionDetector:
    def __init__(self):
        self.emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Angry']
        
        # Load cascade classifiers
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_smile.xml')
        
        if self.face_cascade.empty():
            raise ValueError("Error: Could not load face cascade classifier")

        self.emotion_counts = {label: 0 for label in self.emotion_labels}

    def detect_emotion(self, frame):
        try:
            if frame is None:
                raise ValueError("No frame provided")

            output_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast for better feature detection
            gray = cv2.equalizeHist(gray)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                color_face_roi = frame[y:y+h, x:x+w]
                
                # Define regions
                face_height = face_roi.shape[0]
                face_width = face_roi.shape[1]
                
                # Upper face (for eyes)
                upper_face = face_roi[int(face_height*0.2):int(face_height*0.5), :]
                
                # Lower face (for smile detection)
                lower_face = face_roi[int(face_height*0.5):int(face_height*0.8), :]
                
                # Detect eyes
                eyes = self.eye_cascade.detectMultiScale(
                    upper_face,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(20, 20)
                )
                
                # Enhanced smile detection with multiple parameters
                smiles = self.smile_cascade.detectMultiScale(
                    lower_face,
                    scaleFactor=1.5,
                    minNeighbors=15,
                    minSize=(25, 25)
                )
                
                # Additional smile detection with different parameters
                smiles2 = self.smile_cascade.detectMultiScale(
                    lower_face,
                    scaleFactor=1.3,
                    minNeighbors=10,
                    minSize=(20, 20)
                )
                
                # Combine smile detections
                all_smiles = np.vstack((smiles, smiles2)) if len(smiles2) > 0 and len(smiles) > 0 else \
                           smiles if len(smiles) > 0 else \
                           smiles2 if len(smiles2) > 0 else \
                           np.array([])
                
                emotion, confidence = self._analyze_features(face_roi, upper_face, lower_face, eyes, all_smiles)
                
                # Update emotion counts
                self.emotion_counts[emotion] += 1

                # Draw rectangle around face
                color = self._get_emotion_color(emotion)
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)

                # Add emotion label with confidence
                label = f"{emotion} ({confidence:.0f}%)"
                label_position = (x, y - 10)
                cv2.putText(output_frame, label, label_position,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            return output_frame, faces

        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            return frame, []

    def _analyze_features(self, face_roi, upper_face, lower_face, eyes, smiles):
        """Enhanced emotion analysis with improved happy detection"""
        # Basic image statistics
        mean_intensity = np.mean(face_roi)
        std_intensity = np.std(face_roi)
        
        # Edge detection for expression lines
        edges = cv2.Canny(face_roi, 100, 200)
        edge_intensity = np.mean(edges)
        
        # Initialize confidence
        confidence = 0
        emotion = 'Neutral'
        
        # Happy detection (improved)
        if len(smiles) > 0:
            # Calculate smile characteristics
            smile_sizes = [w * h for (x, y, w, h) in smiles]
            max_smile_size = max(smile_sizes) if smile_sizes else 0
            smile_count = len(smiles)
            
            # Calculate cheek area intensity
            lower_half = lower_face[lower_face.shape[0]//2:, :]
            cheek_intensity = np.mean(lower_half)
            
            happy_confidence = 0
            # Multiple factors contribute to happy detection
            if smile_count > 0:
                happy_confidence += 30  # Base confidence for smile detection
                
                # Bonus for multiple smile detections
                happy_confidence += min(smile_count * 10, 30)
                
                # Bonus for smile size
                if max_smile_size > 1000:
                    happy_confidence += 20
                
                # Bonus for cheek intensity (smiling often increases brightness)
                if cheek_intensity > mean_intensity:
                    happy_confidence += 10
                
                # Bonus for appropriate edge intensity (natural smile lines)
                if 40 < edge_intensity < 100:
                    happy_confidence += 10
                
                if happy_confidence > confidence:
                    return 'Happy', min(happy_confidence, 100)
        
        # Surprise detection
        if len(eyes) >= 2:
            eye_height_ratio = self._calculate_eye_height_ratio(eyes)
            if eye_height_ratio > 0.15:
                surprise_confidence = min(eye_height_ratio * 300, 100)
                if surprise_confidence > confidence:
                    return 'Surprise', surprise_confidence
        
        # Sad detection
        if len(eyes) >= 2:
            mouth_curve = self._detect_mouth_curve(lower_face)
            if mouth_curve < -10:
                sad_confidence = min(70 - edge_intensity * 0.5, 100)
                if sad_confidence > confidence:
                    return 'Sad', sad_confidence
        
        # Angry detection
        if edge_intensity > 60:
            brow_intensity = np.mean(upper_face[:int(upper_face.shape[0]/2), :])
            if brow_intensity < mean_intensity:
                angry_confidence = min(edge_intensity * 0.8, 100)
                if angry_confidence > confidence:
                    return 'Angry', angry_confidence
        
        # Default to neutral
        return 'Neutral', max(60 - edge_intensity * 0.5, 30)

    def _calculate_eye_height_ratio(self, eyes):
        """Calculate the ratio of eye height to detect wide eyes"""
        if len(eyes) < 2:
            return 0
        total_height = sum(eye[3] for eye in eyes)
        return total_height / len(eyes) / 100

    def _detect_mouth_curve(self, lower_face):
        """Detect mouth curve direction"""
        edges = cv2.Canny(lower_face, 100, 200)
        height, width = edges.shape
        top_region = edges[:height//2, :]
        bottom_region = edges[height//2:, :]
        return np.mean(top_region) - np.mean(bottom_region)

    def _get_emotion_color(self, emotion):
        """Return color for emotion visualization"""
        colors = {
            'Happy': (0, 255, 0),     # Green
            'Sad': (255, 0, 0),       # Blue
            'Angry': (0, 0, 255),     # Red
            'Surprise': (255, 255, 0), # Cyan
            'Neutral': (128, 128, 128) # Gray
        }
        return colors.get(emotion, (0, 255, 0))

    def get_emotion_stats(self):
        """Return current emotion counts"""
        return self.emotion_counts

    def reset_stats(self):
        """Reset emotion counts"""
        self.emotion_counts = {label: 0 for label in self.emotion_labels}