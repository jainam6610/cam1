import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
import numpy as np
import pandas as pd
import datetime
import os
import json
import threading
import queue
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class SecuritySystem:
    def __init__(self, config_path="config.json"):
        self.setup_directories()
        self.load_config(config_path)
        self.setup_models()
        self.setup_detection_buffers()
        self.setup_logging()
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ["logs", "dataset/fire", "dataset/fight", "dataset/normal", 
                "models", "alerts", "reports"]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_config(self, config_path):
        """Load configuration with defaults"""
        default_config = {
            "fire_threshold": 0.7,
            "fight_threshold": 0.75,
            "face_detection_scale": 1.1,
            "face_detection_neighbors": 4,
            "buffer_size": 30,
            "alert_cooldown": 5,
            "video_quality": 0.8,
            "detection_interval": 2
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.config = {**default_config, **config}
        else:
            self.config = default_config
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
    
    def setup_models(self):
        """Initialize detection models with better architectures"""
        # Face detection with multiple cascades for better accuracy
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Enhanced models for fire and fight detection
        self.fire_model = self.create_enhanced_model(num_classes=2)
        self.fight_model = self.create_enhanced_model(num_classes=3)  # normal, fight, violence
        
        # Motion detection background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )
        
        # Enhanced image transforms with augmentation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Multi-scale transform for better detection
        self.multi_scale_transforms = [
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ]) for size in [224, 256, 288]
        ]
    
    def create_enhanced_model(self, num_classes):
        """Create enhanced model with better architecture"""
        model = resnet50(pretrained=True)
        
        # Add dropout and batch normalization for better generalization
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )
        
        model.eval()
        return model
    
    def setup_detection_buffers(self):
        """Setup buffers for temporal consistency"""
        buffer_size = self.config["buffer_size"]
        self.fire_buffer = deque(maxlen=buffer_size)
        self.fight_buffer = deque(maxlen=buffer_size)
        self.motion_buffer = deque(maxlen=buffer_size)
        self.face_buffer = deque(maxlen=buffer_size)
        
        # Alert cooldown timers
        self.last_fire_alert = 0
        self.last_fight_alert = 0
        self.alert_cooldown = self.config["alert_cooldown"]
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        self.detection_log = []
        self.daily_stats = {
            'fire_detections': 0,
            'fight_detections': 0,
            'face_detections': 0,
            'total_frames': 0,
            'alerts_sent': 0
        }
    
    def enhanced_face_detection(self, frame):
        """Enhanced face detection with multiple methods"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        
        # Frontal face detection
        frontal_faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=self.config["face_detection_scale"],
            minNeighbors=self.config["face_detection_neighbors"],
            minSize=(30, 30)
        )
        
        # Profile face detection
        profile_faces = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config["face_detection_scale"],
            minNeighbors=self.config["face_detection_neighbors"],
            minSize=(30, 30)
        )
        
        # Combine detections and remove duplicates
        all_faces = np.vstack((frontal_faces, profile_faces)) if len(profile_faces) > 0 and len(frontal_faces) > 0 else frontal_faces
        
        # Non-maximum suppression to remove overlapping detections
        if len(all_faces) > 0:
            faces = self.non_max_suppression(all_faces, 0.3)
        
        return faces
    
    def non_max_suppression(self, boxes, overlap_threshold):
        """Remove overlapping bounding boxes"""
        if len(boxes) == 0:
            return []
        
        boxes = boxes.astype(np.float32)
        pick = []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], 
                           np.where(overlap > overlap_threshold)[0])))
        
        return boxes[pick].astype(np.int32)
    
    def multi_scale_prediction(self, model, frame, threshold):
        """Enhanced prediction with multiple scales and temporal consistency"""
        predictions = []
        confidences = []
        
        with torch.no_grad():
            # Multi-scale prediction
            for transform in self.multi_scale_transforms:
                img_tensor = transform(frame).unsqueeze(0)
                output = model(img_tensor)
                prob = torch.softmax(output, dim=1)
                confidence, pred = torch.max(prob, 1)
                
                predictions.append(pred.item())
                confidences.append(confidence.item())
            
            # Ensemble prediction
            final_pred = max(set(predictions), key=predictions.count)
            avg_confidence = np.mean(confidences)
            
            return final_pred, avg_confidence
    
    def detect_motion(self, frame):
        """Enhanced motion detection"""
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate motion intensity
        motion_pixels = cv2.countNonZero(fg_mask)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_ratio = motion_pixels / total_pixels
        
        return motion_ratio, fg_mask
    
    def analyze_color_histograms(self, frame):
        """Analyze color histograms for fire detection enhancement"""
        # Convert to HSV for better fire detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Fire color ranges (red, orange, yellow)
        fire_lower1 = np.array([0, 50, 50])
        fire_upper1 = np.array([10, 255, 255])
        fire_lower2 = np.array([170, 50, 50])
        fire_upper2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, fire_lower1, fire_upper1)
        mask2 = cv2.inRange(hsv, fire_lower2, fire_upper2)
        fire_mask = cv2.bitwise_or(mask1, mask2)
        
        fire_ratio = cv2.countNonZero(fire_mask) / (frame.shape[0] * frame.shape[1])
        
        return fire_ratio, fire_mask
    
    def temporal_consistency_check(self, buffer, prediction, threshold=0.6):
        """Check temporal consistency to reduce false positives"""
        buffer.append(prediction)
        
        if len(buffer) >= 5:  # Need at least 5 frames
            recent_predictions = list(buffer)[-5:]
            positive_ratio = sum(recent_predictions) / len(recent_predictions)
            return positive_ratio >= threshold
        
        return False
    
    def log_detection(self, event_type, frame, confidence, metadata=None):
        """Enhanced logging with metadata"""
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Save image with timestamp overlay
        frame_copy = frame.copy()
        cv2.putText(frame_copy, f"{event_type.upper()} - {timestamp_str}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save to appropriate directory
        path = f"logs/{event_type}_{timestamp_str}.jpg"
        cv2.imwrite(path, frame_copy)
        
        # Log to data structure
        log_entry = {
            'timestamp': timestamp_str,
            'event_type': event_type,
            'confidence': confidence,
            'path': path,
            'metadata': metadata or {}
        }
        
        self.detection_log.append(log_entry)
        
        # Update daily stats
        if event_type == 'fire':
            self.daily_stats['fire_detections'] += 1
        elif event_type == 'fight':
            self.daily_stats['fight_detections'] += 1
        
        print(f"[DETECTED] {event_type.upper()} - Confidence: {confidence:.2f} at {timestamp_str}")
        
        return log_entry
    
    def should_send_alert(self, event_type):
        """Check if alert should be sent based on cooldown"""
        current_time = datetime.datetime.now().timestamp()
        
        if event_type == 'fire':
            if current_time - self.last_fire_alert > self.alert_cooldown:
                self.last_fire_alert = current_time
                return True
        elif event_type == 'fight':
            if current_time - self.last_fight_alert > self.alert_cooldown:
                self.last_fight_alert = current_time
                return True
        
        return False
    
    def send_alert(self, event_type, confidence, metadata):
        """Send alert (placeholder for actual alert system)"""
        if self.should_send_alert(event_type):
            alert_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'event_type': event_type,
                'confidence': confidence,
                'metadata': metadata
            }
            
            # Save alert to file
            alert_file = f"alerts/alert_{event_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(alert_file, 'w') as f:
                json.dump(alert_data, f, indent=2)
            
            self.daily_stats['alerts_sent'] += 1
            print(f"[ALERT] {event_type.upper()} alert sent!")
    
    def generate_daily_report(self):
        """Generate daily statistics report"""
        report = {
            'date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'statistics': self.daily_stats,
            'detections': self.detection_log[-50:]  # Last 50 detections
        }
        
        report_file = f"reports/daily_report_{datetime.datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    def run_detection(self, video_source=0):
        """Main detection loop with enhanced features"""
        cap = cv2.VideoCapture(video_source)
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        detection_interval = self.config["detection_interval"]
        
        print("AI Security System Started. Press 'q' to quit, 's' to save report.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame_count += 1
            self.daily_stats['total_frames'] += 1
            
            # Process every nth frame for efficiency
            if frame_count % detection_interval == 0:
                # Enhanced face detection
                faces = self.enhanced_face_detection(frame)
                self.face_buffer.append(len(faces))
                
                # Draw face rectangles with eye detection
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Detect eyes within face region
                    face_roi = frame[y:y+h, x:x+w]
                    eyes = self.eye_cascade.detectMultiScale(face_roi)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 1)
                
                # Motion detection
                motion_ratio, motion_mask = self.detect_motion(frame)
                self.motion_buffer.append(motion_ratio > 0.02)
                
                # Fire detection with color analysis
                fire_pred, fire_conf = self.multi_scale_prediction(
                    self.fire_model, frame, self.config["fire_threshold"]
                )
                fire_color_ratio, fire_mask = self.analyze_color_histograms(frame)
                
                # Enhanced fire detection combining ML and color analysis
                fire_detected = (fire_pred == 1 and fire_conf > self.config["fire_threshold"]) or \
                               (fire_color_ratio > 0.1 and motion_ratio > 0.05)
                
                if fire_detected and self.temporal_consistency_check(self.fire_buffer, True):
                    metadata = {
                        'ml_confidence': fire_conf,
                        'color_ratio': fire_color_ratio,
                        'motion_ratio': motion_ratio,
                        'faces_detected': len(faces)
                    }
                    self.log_detection("fire", frame, max(fire_conf, fire_color_ratio), metadata)
                    self.send_alert("fire", max(fire_conf, fire_color_ratio), metadata)
                    cv2.putText(frame, f"FIRE DETECTED! Conf: {fire_conf:.2f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.fire_buffer.append(False)
                
                # Fight detection
                fight_pred, fight_conf = self.multi_scale_prediction(
                    self.fight_model, frame, self.config["fight_threshold"]
                )
                
                # Enhanced fight detection considering motion and people
                fight_detected = (fight_pred > 0 and fight_conf > self.config["fight_threshold"] and 
                                len(faces) >= 2 and motion_ratio > 0.1)
                
                if fight_detected and self.temporal_consistency_check(self.fight_buffer, True):
                    metadata = {
                        'ml_confidence': fight_conf,
                        'motion_ratio': motion_ratio,
                        'faces_detected': len(faces),
                        'fight_type': 'violence' if fight_pred == 2 else 'fight'
                    }
                    self.log_detection("fight", frame, fight_conf, metadata)
                    self.send_alert("fight", fight_conf, metadata)
                    cv2.putText(frame, f"FIGHT DETECTED! Conf: {fight_conf:.2f}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.fight_buffer.append(False)
                
                # Display statistics
                stats_text = f"Faces: {len(faces)} | Motion: {motion_ratio:.3f} | Frame: {frame_count}"
                cv2.putText(frame, stats_text, (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Enhanced AI Security System", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                break
            elif key == ord('s'):  # 's' to save report
                self.generate_daily_report()
                print("Daily report saved!")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Generate final report
        self.generate_daily_report()
        print("Security system stopped. Final report generated.")

# Usage
if __name__ == "__main__":
    # Initialize the enhanced security system
    security_system = SecuritySystem()
    
    # Run the detection system
    security_system.run_detection(video_source=0)  # Use 0 for webcam, or path for video file
