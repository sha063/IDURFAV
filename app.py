import streamlit as st
import numpy as np
import faiss
import logging
import os
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation
import insightface
import cv2
from PIL import Image
import gdown
import io

# Initialize logging (Streamlit-friendly: console + optional file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Cache InsightFace model loading
@st.cache_resource
def load_face_model():
    try:
        face_analysis = insightface.app.FaceAnalysis(det_name='retinaface_r50_v1', rec_name='arcface_r100_v1')
        face_analysis.prepare(ctx_id=-1, det_size=(640, 640))  # CPU mode
        logging.info("InsightFace initialized successfully")
        return face_analysis
    except Exception as e:
        logging.error(f"InsightFace initialization failed: {str(e)}")
        raise

# Download and load .npy from Google Drive (cached)
@st.cache_resource
def load_existing_features():
    try:
        file_id = "YOUR_FILE_ID_HERE"  # Replace with your actual Google Drive file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "features3_fusion_merged.npy"
        if not os.path.exists(output):
            with st.spinner("Downloading large feature database (241 MB)... This may take a minute."):
                gdown.download(url, output, quiet=False)
        data = np.load(output, allow_pickle=True).item()
        feature_vectors = data.get('feature_vectors', [])
        image_paths = data.get('image_paths', [])
        if not feature_vectors:
            raise ValueError("No feature vectors found in the database")
        dimension = len(feature_vectors[0])
        logging.info(f"Loaded {len(feature_vectors)} feature vectors with dimension {dimension}")
        return feature_vectors, image_paths
    except Exception as e:
        logging.error(f"Failed to load features: {str(e)}")
        st.error(f"Error loading database: {str(e)}")
        raise

# Build FAISS index (cached)
@st.cache_resource
def build_feature_index(_feature_vectors):  # Use hashable param for cache
    try:
        if not _feature_vectors:
            raise ValueError("Feature vector list is empty")
        dimension = len(_feature_vectors[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(_feature_vectors).astype(np.float32))
        logging.info(f"Built FAISS index with {len(_feature_vectors)} vectors of dimension {dimension}")
        return index
    except Exception as e:
        logging.error(f"Failed to build FAISS index: {str(e)}")
        st.error(f"Error building index: {str(e)}")
        raise

def preprocess_image(image: np.ndarray, contrast_factor: float = 1.0, resolution_factor: float = 1.0) -> np.ndarray:
    """Preprocess image with optional contrast and resolution adjustment."""
    try:
        if image is None or len(image.shape) != 3:
            raise ValueError("Invalid image: Image is None or not in correct format")
        max_width = int(640 * resolution_factor)
        height, width = image.shape[:2]
        if width > max_width:
            scale = max_width / width
            image = cv2.resize(image, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)

        if contrast_factor != 1.0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=contrast_factor, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return image
    except Exception as e:
        logging.warning(f"Image preprocessing failed: {str(e)}")
        raise

def estimate_pose(points: np.ndarray) -> Tuple[float, float, float]:
    """Estimate yaw, pitch, and roll angles from landmarks."""
    try:
        if not isinstance(points, np.ndarray) or points.shape != (68, 2):
            raise ValueError(f"Invalid points shape: {points.shape}")
        left_eye_center = np.mean(points[36:42], axis=0)
        right_eye_center = np.mean(points[42:48], axis=0)
        nose_tip = points[30]

        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0] + 1e-6
        roll = np.degrees(np.arctan2(dY, dX))

        eye_midpoint = (left_eye_center + right_eye_center) / 2
        pitch = np.degrees(np.arctan2(nose_tip[1] - eye_midpoint[1], 100.0))

        face_center = np.mean(points[0:17], axis=0)
        yaw = np.degrees(np.arctan2(nose_tip[0] - face_center[0], 100.0))

        return roll, pitch, yaw
    except Exception as e:
        logging.warning(f"Pose estimation failed: {str(e)}")
        return 0.0, 0.0, 0.0

def compute_curvature(points: np.ndarray, indices: range) -> float:
    """Compute curvature using quadratic fit."""
    try:
        if not isinstance(points, np.ndarray) or points.shape != (68, 2):
            raise ValueError(f"Invalid points shape: {points.shape}")
        selected_points = points[list(indices)]
        x, y = selected_points[:, 0], selected_points[:, 1]
        coeffs = np.polyfit(x, y, 2)
        return abs(coeffs[0])
    except Exception as e:
        logging.warning(f"Curvature computation failed: {str(e)}")
        return 0.0

def extract_facial_features(image: np.ndarray, model) -> Dict:
    """Extract features using InsightFace with pose and curvature for 519-dimensional vectors."""
    try:
        processed_image = preprocess_image(image)
        faces = model.get(processed_image)
        if not faces:
            return {'features': None, 'status': 'failed', 'reason': 'No faces detected', 'detection_method': 'insightface', 'confidence': 0.0}

        areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
        max_idx = np.argmax(areas)
        selected_face = faces[max_idx]
        embedding = selected_face.embedding / np.linalg.norm(selected_face.embedding)
        confidence = selected_face.det_score

        additional_features = np.zeros(7, dtype=np.float32)
        landmarks = selected_face.landmark
        if landmarks is not None and isinstance(landmarks, np.ndarray) and landmarks.shape == (68, 2):
            roll, pitch, yaw = estimate_pose(landmarks)
            jaw_curvature = compute_curvature(landmarks, range(0, 17))
            left_eyebrow_curvature = compute_curvature(landmarks, range(17, 22))
            right_eyebrow_curvature = compute_curvature(landmarks, range(22, 27))
            lip_curvature = compute_curvature(landmarks, range(48, 55))

            additional_features = np.array([roll / 180.0, pitch / 180.0, yaw / 180.0,
                                            jaw_curvature, left_eyebrow_curvature,
                                            right_eyebrow_curvature, lip_curvature], dtype=np.float32)
        else:
            confidence *= 0.8

        features = np.concatenate([embedding, additional_features])
        return {'features': features, 'status': 'success', 'reason': None, 'detection_method': 'insightface', 'confidence': confidence}
    except Exception as e:
        logging.error(f"Feature extraction failed: {str(e)}")
        return {'features': None, 'status': 'failed', 'reason': f'Exception: {str(e)}', 'detection_method': 'insightface', 'confidence': 0.0}

def reverse_image_search(query_image_np: np.ndarray, index: faiss.Index, image_paths: List[str], model, top_k: int = 20) -> List[Dict]:
    try:
        query_result = extract_facial_features(query_image_np, model)
        if query_result['status'] == 'failed':
            raise ValueError(f"Failed to extract features from query image: {query_result['reason']}")

        query_vector = query_result['features'].reshape(1, -1).astype(np.float32)
        if query_vector.shape[1] != index.d:
            raise ValueError(f"Query vector dimension {query_vector.shape[1]} does not match index dimension {index.d}")
        distances, indices = index.search(query_vector, top_k)

        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx < len(image_paths):
                results.append({
                    'image_path': image_paths[idx],
                    'distance': float(distances[0][i]),
                    'index': int(idx)
                })
        return results
    except Exception as e:
        logging.error(f"Reverse image search failed: {str(e)}")
        raise

# Streamlit UI
st.title("🔍 Face-Based Reverse Image Search")

uploaded_file = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Load resources
    with st.spinner("Loading model, database, and index..."):
        model = load_face_model()
        feature_vectors, image_paths = load_existing_features()
        index = build_feature_index(tuple(map(tuple, feature_vectors)))  # Tuple for hashing in cache

    # Perform search
    with st.spinner("Performing search..."):
        try:
            results = reverse_image_search(image_np, index, image_paths, model, top_k=20)
            st.subheader("Top Matches:")
            for result in results:
                st.write(f"Match: {os.path.basename(result['image_path'])}, Distance: {result['distance']:.4f}")
                # If image_paths are URLs, add: st.image(result['image_path'], width=150)
            st.success("Search complete!")
        except Exception as e:
            st.error(f"Error during search: {str(e)}")