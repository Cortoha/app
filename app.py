import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
import argparse
import cv2
import sys
import os
from pathlib import Path
from collections import defaultdict

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None


# LOGGING SETUP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# WHALE IDENTIFICATION MODEL - DinoV2 backbone with ArcFace loss


class ArcFaceLoss(nn.Module):
    """ArcFace Loss for whale identification"""
    def __init__(self, embedding_dim, num_classes, s=64.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        W = F.normalize(self.W, dim=1)
        logits = torch.mm(embeddings, W.t())
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        theta = torch.arange(len(labels), device=labels.device).unsqueeze(1) != labels.unsqueeze(0)
        theta = theta.float() * theta + (~theta).float() * (theta - self.m)
        loss = F.cross_entropy(torch.cos(theta), labels)
        return loss

class WhaleIdentificationModel(nn.Module):
    """DinoV2 backbone + custom head for whale identification"""
    def __init__(self, num_classes, embedding_dim=256, pretrained=True):
        super(WhaleIdentificationModel, self).__init__()
        
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        except:
            logger.warning("Could not load DinoV2 from hub, using random weights")
            self.backbone = None
        
        backbone_feat_dim = 768
        
        if self.backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.head = nn.Sequential(
            nn.Linear(backbone_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        self.arcface_loss = ArcFaceLoss(embedding_dim, num_classes, s=64.0, m=0.5)
        self.ce_loss = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
    
    def forward(self, x, labels=None, return_embedding=False):
        if self.backbone is not None:
            features = self.backbone(x)
        else:
            features = torch.randn(x.shape[0], 768, device=x.device)
        
        embeddings = self.head(features)
        
        if return_embedding:
            return embeddings
        
        if labels is not None:
            arcface_loss = self.arcface_loss(embeddings, labels)
            logits = self.classifier(embeddings)
            ce_loss = self.ce_loss(logits, labels)
            total_loss = 0.7 * arcface_loss + 0.3 * ce_loss
            return type('obj', (object,), {
                'loss': total_loss,
                'arcface_loss': arcface_loss,
                'ce_loss': ce_loss,
                'logits': logits,
                'embeddings': embeddings
            })()
        
        logits = self.classifier(embeddings)
        return logits

# BASE WHALE DETECTOR - Common methods for video and photo processing


class BaseWhaleDetector:
    """Base class with common detection methods"""
    
    def __init__(self, yolo_path='./yolo/best.pt', embeddings_path='./embeddings/all_embeddings.npy',
                 labels_path='./embeddings/all_labels.npy', checkpoint_path='./models/best_model_whale.pth',
                 device='cuda'):
        
        if YOLO is None:
            raise RuntimeError("ultralytics not installed")
        
        self.device = device
        self.top_k = 5
        self.top_neighbors = 100
        
        logger.info(f"Loading YOLO model from {yolo_path}")
        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.to(device)
        logger.info("YOLO loaded")
        
        logger.info(f"Loading embeddings from {embeddings_path}")
        self.db_embeddings = np.load(embeddings_path)
        self.db_labels = np.load(labels_path)
        self.db_embeddings_normalized = self._normalize(self.db_embeddings)
        logger.info(f"Loaded {len(self.db_embeddings)} embeddings")
        
        self.label_to_id = self._load_id_map()
        self.db_labels_original = np.array([
            self.label_to_id.get(int(label), str(label)) for label in self.db_labels
        ])
        logger.info(f"Loaded {len(np.unique(self.db_labels))} unique whale IDs")
        
        logger.info(f"Loading embedding model from {checkpoint_path}")
        num_classes = len(np.unique(self.db_labels))
        self.model = WhaleIdentificationModel(
            num_classes=num_classes,
            embedding_dim=self.db_embeddings.shape[1]
        ).to(device)
        
        logger.info(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Weights loaded")
        
        self.model.eval()
        logger.info("Ready for inference")
    
    def _load_id_map(self):
        """Load whale_ids.json"""
        # Определяем базовый путь (работает для dev и PyInstaller)
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = '.'
        
        search_paths = [
            os.path.join(base_path, 'whale_id.json'),
            './whale_id.json', 
            '../whale_id.json', 
            '../../whale_id.json'
        ]
        
        for path in search_paths:
            if Path(path).exists():
                try:
                    with open(path) as f:
                        mapping = json.load(f)
                        logger.info(f"Loaded whale_ids.json from {path}")
                        return {int(k): v for k, v in mapping.items()}
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
                    continue
        
        logger.warning("whale_ids.json not found! Using numeric IDs")
        return {i: str(i) for i in range(len(np.unique(self.db_labels)))}
    
    @staticmethod
    def _normalize(embeddings):
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
    
    @staticmethod
    def _cosine_similarity(vec1, vec2_normalized):
        """Calculate cosine similarity"""
        vec1_normalized = vec1 / (np.linalg.norm(vec1) + 1e-8)
        return np.dot(vec2_normalized, vec1_normalized)
    
    def get_embedding_from_region(self, region_rgb):
        """Get embedding vector from whale region"""
        region_resized = cv2.resize(region_rgb, (392, 392))
        image_normalized = region_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(image_tensor, return_embedding=True).cpu().numpy()[0]
        
        return embedding
    
    def identify_embedding(self, embedding):
        """Identify whale from embedding vector
        
        ALGORITHM - for EACH whale detected:
        1. Get its unique embedding from the region (bbox from YOLO)
        2. Find 100 nearest neighbors in database
        3. Count occurrences of each whale in top-100
        4. Return top-5 whales by occurrence count
        
        This means EACH whale gets its OWN top-5!
        """
        similarities = self._cosine_similarity(embedding, self.db_embeddings_normalized)
        top_100_indices = np.argsort(similarities)[::-1][:self.top_neighbors]
        
        whale_counts = {}
        whale_similarities = {}
        
        for idx in top_100_indices:
            whale_id = self.db_labels_original[idx]
            similarity = similarities[idx]
            
            if whale_id not in whale_counts:
                whale_counts[whale_id] = 0
                whale_similarities[whale_id] = []
            
            whale_counts[whale_id] += 1
            whale_similarities[whale_id].append(similarity)
        
        sorted_whales = sorted(
            whale_counts.items(),
            key=lambda x: (x[1], np.mean(whale_similarities[x[0]])),
            reverse=True
        )
        
        top_5_whales = sorted_whales[:5]
        top_5_ids = [w[0] for w in top_5_whales]
        top_5_counts = [w[1] for w in top_5_whales]
        top_5_avg_sim = [np.mean(whale_similarities[w[0]]) for w in top_5_whales]
        
        best_label = top_5_ids[0] if top_5_ids else "Unknown"
        best_count = top_5_counts[0] if top_5_counts else 0
        
        return {
            'top_whale_ids': top_5_ids,
            'top_whale_counts': top_5_counts,
            'top_whale_avg_similarities': top_5_avg_sim,
            'best_label': best_label,
            'best_count': best_count,
            'best_avg_similarity': top_5_avg_sim[0] if top_5_avg_sim else 0.0
        }


# WHALE VIDEO DETECTOR


class WhaleVideoDetector(BaseWhaleDetector):
    """Whale detector in video with ready embeddings"""
    
    def extract_frames(self, video_path, start_time=None, end_time=None, fps_extraction=5):
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video duration: {duration:.1f}s, FPS: {fps:.1f}, Total frames: {total_frames}")
        
        start_frame = int((start_time or 0) * fps)
        end_frame = int((end_time or duration) * fps)
        frame_interval = int(fps / fps_extraction)
        
        logger.info(f"Extracting {fps_extraction} frames/sec")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame, frame_idx / fps, frame_idx
            frame_idx += frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        cap.release()
    
    def process_video(self, video_path, output_dir='./whale_detections', start_time=None, end_time=None, fps=5):
        """Process video and detect whales
        
        FOR EACH FRAME:
        - Detect whales with YOLO (get bbox)
        - FOR EACH whale in frame:
          - Extract region by bbox
          - Get embedding (unique vector for this whale)
          - Identify: find top-5 matches for THIS whale's embedding
          - Store results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing video: {Path(video_path).name}")
        
        whale_detections = defaultdict(lambda: {
            'frames': [],
            'top_matches': [],
            'best_frame': None,
            'best_count': 0
        })
        
        first_frame_data = None
        frame_count = 0
        
        for frame, timestamp, frame_idx in self.extract_frames(video_path, start_time, end_time, fps):
            frame_count += 1
            
            results = self.yolo_model(frame, conf=0.5, verbose=False)
            detections = []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append({'bbox': (x1, y1, x2, y2)})
            
            if not detections:
                continue
            
            if first_frame_data is None:
                first_frame_data = (frame, timestamp, detections)
            
            logger.info(f"Frame {frame_count} at {timestamp:.1f}s - Found {len(detections)} whale(s)")
            
            for det_idx, detection in enumerate(detections):
                x1, y1, x2, y2 = detection['bbox']
                
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                mask[max(0, y1-5):min(frame.shape[0], y2+5), max(0, x1-5):min(frame.shape[1], x2+5)] = 255
                
                region = frame[y1:y2, x1:x2].copy()
                region_masked = cv2.bitwise_and(region, region, mask=mask[y1:y2, x1:x2])
                region_rgb = cv2.cvtColor(region_masked, cv2.COLOR_BGR2RGB)
                
                embedding = self.get_embedding_from_region(region_rgb)
                
                identification = self.identify_embedding(embedding)
                whale_id = identification['best_label']
                best_count = identification['best_count']
                
                logger.info(f"  Whale {det_idx+1}: {whale_id} (top-100 count: {best_count})")
                logger.info(f"    Top-5 for this whale: {identification['top_whale_ids']}")
                
                whale_detections[whale_id]['frames'].append({
                    'frame': frame.copy(),
                    'bbox': detection['bbox'],
                    'timestamp': timestamp
                })
                whale_detections[whale_id]['top_matches'].append(identification)
                
                if best_count > whale_detections[whale_id]['best_count']:
                    whale_detections[whale_id]['best_count'] = best_count
                    whale_detections[whale_id]['best_frame'] = {
                        'frame': frame.copy(),
                        'bbox': detection['bbox'],
                        'timestamp': timestamp,
                        'whale_id': whale_id,
                        'top_matches': identification
                    }
        
        logger.info(f"Processed {frame_count} frames")
        
        self._save_video_results(whale_detections, output_dir)
        
        if first_frame_data:
            self._visualize_video(whale_detections, first_frame_data, output_dir)
        
        logger.info("Video processing completed")
        
        return whale_detections
    
    def _visualize_video(self, whale_detections, first_frame_data, output_dir):
        """Create visualization of video results
        
        KEY FIX: Show TOP-5 SPECIFIC to each whale, not global top-5
        """
        frame, timestamp, detections = first_frame_data
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 0, 128), (0, 128, 128)]
        
        for det_idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            
            for whale_id in whale_detections:
                found = False
                whale_specific_matches = None
                
                for frame_info in whale_detections[whale_id]['frames']:
                    if frame_info['bbox'] == detection['bbox']:
                        if whale_detections[whale_id]['top_matches']:
                            whale_specific_matches = whale_detections[whale_id]['top_matches'][0]
                        
                        color = colors[det_idx % len(colors)]
                        
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 4)
                        
                        cv2.putText(frame_rgb, str(whale_id), (x1 + 5, y1 + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)
                        
                        # Немного сдвигаем блок текста, чтобы он меньше перекрывал bbox и надписи
                        text_x = x2 + 25 if x2 + 320 < w else max(10, x1 - 320)
                        text_y = max(y1 + 40, 40)
                        
                        cv2.putText(frame_rgb, "Top matches:", (text_x, text_y + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                        
                        if whale_specific_matches:
                            top_ids = whale_specific_matches['top_whale_ids'][:5]
                            top_counts = whale_specific_matches['top_whale_counts'][:5]
                            
                            for rank, (w_id, count) in enumerate(zip(top_ids, top_counts), 1):
                                text = f"{rank}. {w_id} ({count})"
                                cv2.putText(frame_rgb, text, (text_x, text_y + 25 * (rank + 1)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                        
                        found = True
                        break
                if found:
                    break
        
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        output_path = output_dir / 'whale_detection_result.jpg'
        cv2.imwrite(str(output_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"Saved visualization: {output_path}")
    
    def _save_video_results(self, whale_detections, output_dir):
        """Save video results to JSON and best frames as images"""
        results = {}
        for whale_id, info in whale_detections.items():
            if info['best_frame']:
                # Save best frame as image
                best_frame_data = info['best_frame']
                frame = best_frame_data['frame']
                bbox = best_frame_data['bbox']
                x1, y1, x2, y2 = bbox
                
                # Draw visualization on best frame
                frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
                color = (0, 255, 0)
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 4)
                cv2.putText(frame_rgb, str(whale_id), (x1 + 5, y1 + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)
                
                # Add top matches info
                top_matches = best_frame_data.get('top_matches', {})
                top_ids = top_matches.get('top_whale_ids', [])[:5]
                top_counts = top_matches.get('top_whale_counts', [])[:5]
                
                h, w = frame_rgb.shape[:2]
                text_x = x2 + 15 if x2 + 300 < w else max(10, x1 - 300)
                text_y = max(y1, 30)
                
                cv2.putText(frame_rgb, "Top matches:", (text_x, text_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                
                for rank, (w_id, count) in enumerate(zip(top_ids, top_counts), 1):
                    text = f"{rank}. {w_id} ({count})"
                    cv2.putText(frame_rgb, text, (text_x, text_y + 25 * (rank + 1)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame_path = output_dir / f'whale_{whale_id}_best_frame.jpg'
                cv2.imwrite(str(frame_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logger.info(f"Saved best frame for {whale_id}: {frame_path}")
                
                results[str(whale_id)] = {
                    'detections': len(info['frames']),
                    'best_count': int(info['best_count']),
                    'timestamps': [float(f['timestamp']) for f in info['frames']]
                }
        
        output_path = output_dir / 'detection_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results: {output_path}")


# WHALE PHOTO IDENTIFIER


class WhalePhotoIdentifier(BaseWhaleDetector):
    """Whale identifier from photos with binary mask"""
    
    def identify_photo(self, image_path, output_dir='./whale_photo_results'):
        """Identify whale in photo"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Identifying whale in photo: {image_path}")
        
        image = cv2.imread(image_path)
        logger.info(f"Image shape: {image.shape}")
        
        logger.info("Running YOLO detection")
        results = self.yolo_model(image, conf=0.5, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                detections.append({'bbox': (x1, y1, x2, y2), 'confidence': conf})
        
        logger.info(f"Found {len(detections)} whale(s)")
        
        results_data = []
        
        for det_idx, detection in enumerate(detections):
            logger.info(f"Processing whale {det_idx+1}")
            
            x1, y1, x2, y2 = detection['bbox']
            logger.info(f"BBox: ({x1}, {y1}) -> ({x2}, {y2})")
            
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            mask[max(0, y1-5):min(image.shape[0], y2+5), max(0, x1-5):min(image.shape[1], x2+5)] = 255
            
            region = image[y1:y2, x1:x2].copy()
            region_masked = cv2.bitwise_and(region, region, mask=mask[y1:y2, x1:x2])
            region_rgb = cv2.cvtColor(region_masked, cv2.COLOR_BGR2RGB)
            
            logger.info("Creating embedding vector")
            embedding = self.get_embedding_from_region(region_rgb)
            logger.info(f"Embedding shape: {embedding.shape}")
            
            logger.info("Comparing with database")
            identification = self.identify_embedding(embedding)
            
            best_whale_id = identification['best_label']
            best_count = identification['best_count']
            top_whale_ids = identification['top_whale_ids']
            top_whale_counts = identification['top_whale_counts']
            top_whale_avg_sim = identification['top_whale_avg_similarities']
            
            logger.info(f"Best match: {best_whale_id} (top-100 count: {best_count}, avg_sim: {identification['best_avg_similarity']:.4f})")
            
            logger.info("Top 5 whales:")
            for rank, (whale, count, avg_sim) in enumerate(zip(top_whale_ids, top_whale_counts, top_whale_avg_sim), 1):
                logger.info(f"  {rank}. {whale} - count: {count}, avg_sim: {avg_sim:.4f}")
            
            results_data.append({
                'whale_index': det_idx + 1,
                'bbox': [int(x) for x in detection['bbox']],
                'confidence': float(detection['confidence']),
                'best_match': str(best_whale_id),
                'best_count': int(best_count),
                'top_5_whales': [
                    {'rank': rank, 'whale_id': str(w), 'count': int(c)}
                    for rank, (w, c) in enumerate(zip(top_whale_ids, top_whale_counts), 1)
                ]
            })
            
            self._save_photo_visualization(image, detection, best_whale_id, best_count,
                                          top_whale_ids, top_whale_counts, top_whale_avg_sim,
                                          output_dir, det_idx)
        
        self._save_photo_results(results_data, output_dir)
        
        logger.info("Photo identification completed")
        
        return results_data
    
    def _save_photo_visualization(self, image, detection, best_whale_id, best_count,
                                 top_whale_ids, top_whale_counts, top_whale_avg_sim,
                                 output_dir, idx):
        """Save photo visualization results"""
        x1, y1, x2, y2 = detection['bbox']
        h, w = image.shape[:2]
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        color = (0, 255, 0)
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 4)
        cv2.putText(frame_rgb, str(best_whale_id), (x1 + 5, y1 + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)
        
        # Сдвигаем блок с TOP списком немного ниже и правее,
        # чтобы он меньше перекрывал bbox и надписи на изображении
        text_x = x2 + 25 if x2 + 380 < w else max(10, x1 - 380)
        text_y = max(y1 + 40, 40)
        
        cv2.putText(frame_rgb, "Top whales:", (text_x, text_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        for rank, (whale, count, avg_sim) in enumerate(zip(top_whale_ids[:5], top_whale_counts[:5], top_whale_avg_sim[:5]), 1):
            text = f"{rank}. {whale}"
            cv2.putText(frame_rgb, text, (text_x, text_y + 25 * (rank + 1)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)
        
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        output_path = output_dir / f'whale_{idx+1}_identification.jpg'
        cv2.imwrite(str(output_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"Saved visualization: {output_path}")
    
    def _save_photo_results(self, results_data, output_dir):
        """Save photo results to JSON"""
        output_path = output_dir / 'photo_identification_results.json'
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Saved results: {output_path}")
        

# MAIN


def main():
    parser = argparse.ArgumentParser(
        description='Whale Detection & Identification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python whale_system_clean_FIXED.py video.mp4
  python whale_system_clean_FIXED.py video.mp4 --start 10 --end 120
  python whale_system_clean_FIXED.py --video video.mp4
  python whale_system_clean_FIXED.py whale.jpg
  python whale_system_clean_FIXED.py --photo whale.jpg
        """
    )
    
    parser.add_argument('input_file', nargs='?', default=None, help='Video or photo file')
    parser.add_argument('--video', default=None, help='Process video file')
    parser.add_argument('--photo', default=None, help='Identify whale in photo')
    parser.add_argument('--yolo', default='./yolo/best.pt', help='Path to YOLO model')
    parser.add_argument('--embeddings', default='./embeddings/all_embeddings.npy', help='Path to embeddings')
    parser.add_argument('--labels', default='./embeddings/all_labels.npy', help='Path to labels')
    parser.add_argument('--checkpoint', default='./models/best_model_whale.pth', help='Path to model checkpoint')
    parser.add_argument('--start', type=float, default=None, help='Start time in seconds')
    parser.add_argument('--end', type=float, default=None, help='End time in seconds')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second to extract')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    video_file = None
    photo_file = None
    
    if args.video:
        video_file = args.video
    elif args.photo:
        photo_file = args.photo
    elif args.input_file:
        input_path = Path(args.input_file)
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']:
            video_file = args.input_file
        elif input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
            photo_file = args.input_file
        else:
            logger.error(f"Unknown file type: {input_path.suffix}")
            return
    else:
        parser.print_help()
        logger.error("Please provide video or photo file")
        return
    
    logger.info("Whale Detection & Identification System")
    logger.info(f"Configuration:")
    logger.info(f"  YOLO: {args.yolo}")
    logger.info(f"  Embeddings: {args.embeddings}")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Device: {args.device}")
    
    try:
        device = torch.device(args.device)
        
        if video_file:
            output_dir = args.output or './whale_detections'
            detector = WhaleVideoDetector(
                yolo_path=args.yolo,
                embeddings_path=args.embeddings,
                labels_path=args.labels,
                checkpoint_path=args.checkpoint,
                device=device
            )
            
            detector.process_video(
                video_path=video_file,
                output_dir=output_dir,
                start_time=args.start,
                end_time=args.end,
                fps=args.fps
            )
        
        elif photo_file:
            output_dir = args.output or './whale_photo_results'
            identifier = WhalePhotoIdentifier(
                yolo_path=args.yolo,
                embeddings_path=args.embeddings,
                labels_path=args.labels,
                checkpoint_path=args.checkpoint,
                device=device
            )
            
            identifier.identify_photo(photo_file, output_dir)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == '__main__':
    main()
