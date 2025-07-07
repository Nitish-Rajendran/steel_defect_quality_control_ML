import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Parameters ---
IMG_SIZE = 128
THRESHOLD = 0.0005  # Lowered threshold for better sensitivity
MODEL_PATH = 'models/improved_autoencoder.keras'

def load_autoencoder_model():
    """Load the trained autoencoder model"""
    try:
        model = load_model(MODEL_PATH)
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"✗ Could not load image: {image_path}")
        return None
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def calculate_anomaly_score(original, reconstructed):
    """Calculate multiple anomaly metrics"""
    # Mean Squared Error
    mse = mean_squared_error(original.flatten(), reconstructed.flatten())
    
    # Mean Absolute Error
    mae = mean_absolute_error(original.flatten(), reconstructed.flatten())
    
    # Structural Similarity Index (SSIM-like)
    # Calculate correlation between original and reconstructed
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    # Normalize
    orig_norm = (orig_flat - np.mean(orig_flat)) / np.std(orig_flat)
    recon_norm = (recon_flat - np.mean(recon_flat)) / np.std(recon_flat)
    
    # Correlation coefficient
    correlation = np.corrcoef(orig_norm, recon_norm)[0, 1]
    if np.isnan(correlation):
        correlation = 0
    
    # Combined anomaly score (higher = more anomalous)
    anomaly_score = mse * (1 - correlation)
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'anomaly_score': anomaly_score
    }

def calculate_confidence(anomaly_score, threshold):
    """Calculate confidence score based on distance from threshold"""
    # Normalize confidence: 0-100%
    distance_from_threshold = abs(anomaly_score - threshold)
    max_expected_distance = threshold * 2  # Reasonable range
    
    confidence = max(0, min(100, (1 - distance_from_threshold / max_expected_distance) * 100))
    
    # Boost confidence if clearly good or clearly defective
    if anomaly_score < threshold * 0.5:  # Clearly good
        confidence = min(100, confidence + 20)
    elif anomaly_score > threshold * 1.5:  # Clearly defective
        confidence = min(100, confidence + 20)
    
    return confidence

def test_image(model, image_path):
    """Test a single image for defects with improved metrics"""
    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    # Preprocess image
    img = preprocess_image(image_path)
    if img is None:
        return None
    
    # Prepare input for model
    img_input = np.expand_dims(img, axis=0)
    
    # Get reconstruction
    reconstructed = model.predict(img_input, verbose=0)
    
    # Calculate anomaly metrics
    metrics = calculate_anomaly_score(img, reconstructed[0])
    
    # Determine if defective using anomaly score
    is_defective = metrics['anomaly_score'] > THRESHOLD
    result = "DEFECTIVE" if is_defective else "GOOD"
    
    # Calculate confidence
    confidence = calculate_confidence(metrics['anomaly_score'], THRESHOLD)
    
    # Print results
    print(f"Anomaly Score: {metrics['anomaly_score']:.6f}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Result: {result}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Correlation: {metrics['correlation']:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title(f"Original Image\n{os.path.basename(image_path)}")
    axes[0].axis('off')
    
    # Reconstructed image
    axes[1].imshow(reconstructed[0])
    axes[1].set_title(f"Reconstructed Image\nAnomaly: {metrics['anomaly_score']:.4f}")
    axes[1].axis('off')
    
    # Difference image (highlight defects)
    diff = np.abs(img - reconstructed[0])
    diff_normalized = diff / np.max(diff)  # Normalize for better visibility
    axes[2].imshow(diff_normalized)
    axes[2].set_title(f"Difference (Defects)\nCorrelation: {metrics['correlation']:.3f}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'image_path': image_path,
        'is_defective': is_defective,
        'result': result,
        'confidence': confidence,
        'anomaly_score': metrics['anomaly_score'],
        'correlation': metrics['correlation'],
        'original': img,
        'reconstructed': reconstructed[0],
        'difference': diff_normalized
    }

def test_multiple_images(model, folder_path, max_images=5):
    """Test multiple images from a folder"""
    if not os.path.exists(folder_path):
        print(f"✗ Folder not found: {folder_path}")
        return
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"✗ No images found in {folder_path}")
        return
    
    # Limit number of images
    image_files = image_files[:max_images]
    
    print(f"\n{'='*70}")
    print(f"Testing {len(image_files)} images from: {folder_path}")
    print(f"{'='*70}")
    
    results = []
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, filename)
        print(f"\n[{i}/{len(image_files)}] Testing: {filename}")
        
        result = test_image(model, image_path)
        if result:
            results.append(result)
    
    return results

def analyze_results(results):
    """Analyze and display summary of results"""
    if not results:
        return
    
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    total = len(results)
    defective_count = sum(1 for r in results if r['is_defective'])
    good_count = total - defective_count
    
    print(f"Total images tested: {total}")
    print(f"Defective detected: {defective_count}")
    print(f"Good detected: {good_count}")
    
    # Average confidence
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"Average confidence: {avg_confidence:.1f}%")
    
    # Anomaly score statistics
    anomaly_scores = [r['anomaly_score'] for r in results]
    print(f"Anomaly score range: {min(anomaly_scores):.6f} - {max(anomaly_scores):.6f}")
    print(f"Average anomaly score: {np.mean(anomaly_scores):.6f}")
    
    # Correlation statistics
    correlations = [r['correlation'] for r in results]
    print(f"Average correlation: {np.mean(correlations):.3f}")

def main():
    """Main function"""
    global THRESHOLD
    
    print("Improved Steel Brick Defect Detection - Static Image Testing")
    print("=" * 70)
    
    # Load model
    model = load_autoencoder_model()
    if model is None:
        return
    
    while True:
        print("\n" + "="*70)
        print("Choose an option:")
        print("1. Test a specific image")
        print("2. Test good images (from dataset/test/good/)")
        print("3. Test defective images (from dataset/test/defective/)")
        print("4. Adjust threshold (current: {:.6f})".format(THRESHOLD))
        print("5. Exit")
        print("="*70)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Test specific image
            image_path = input("Enter the full path to the image: ").strip()
            if os.path.exists(image_path):
                test_image(model, image_path)
            else:
                print("✗ File not found!")
        
        elif choice == '2':
            # Test good images
            good_folder = 'dataset/test/good/'
            num_images = input("How many images to test? (default 3): ").strip()
            num_images = int(num_images) if num_images.isdigit() else 3
            results = test_multiple_images(model, good_folder, max_images=num_images)
            analyze_results(results)
        
        elif choice == '3':
            # Test defective images
            defective_folder = 'dataset/test/defective/'
            num_images = input("How many images to test? (default 3): ").strip()
            num_images = int(num_images) if num_images.isdigit() else 3
            results = test_multiple_images(model, defective_folder, max_images=num_images)
            analyze_results(results)
        
        elif choice == '4':
            # Adjust threshold
            new_threshold = input(f"Enter new threshold (current: {THRESHOLD:.6f}): ").strip()
            try:
                THRESHOLD = float(new_threshold)
                print(f"✓ Threshold updated to: {THRESHOLD:.6f}")
            except ValueError:
                print("✗ Invalid threshold value!")
        
        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 