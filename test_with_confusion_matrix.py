import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Parameters ---
IMG_SIZE = 128
THRESHOLD = 0.008
MODEL_PATH = 'models/improved_autoencoder.keras'
GOOD_DIR = 'dataset/test/good/'
DEFECTIVE_DIR = 'dataset/test/defective/'
CONFUSION_PNG = 'confusion_matrix.png'

# --- Load model ---
def load_autoencoder_model():
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

# --- Preprocess image ---
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

# --- Anomaly score ---
def anomaly_score(original, reconstructed):
    return mean_squared_error(original.flatten(), reconstructed.flatten())

def confidence_score(anomaly, threshold):
    conf = max(0, min(100, (1 - abs(anomaly-threshold)/(2*threshold)) * 100))
    return round(conf, 1)

# --- Test a single image ---
def test_image(model, image_path, threshold):
    img = preprocess_image(image_path)
    if img is None:
        return None
    img_input = np.expand_dims(img, axis=0)
    reconstructed = model.predict(img_input, verbose=0)[0]
    anomaly = anomaly_score(img, reconstructed)
    result = 'DEFECTIVE' if anomaly > threshold else 'GOOD'
    confidence = confidence_score(anomaly, threshold)
    return {
        'filename': os.path.basename(image_path),
        'anomaly_score': anomaly,
        'confidence': confidence,
        'result': result
    }

# --- Test all images in a folder ---
def test_folder(model, folder, label, threshold):
    results = []
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fname in files:
        path = os.path.join(folder, fname)
        res = test_image(model, path, threshold)
        if res:
            res['expected'] = label
            results.append(res)
            print(f"{fname:25} | Expected: {label:10} | Predicted: {res['result']:10} | Anomaly: {res['anomaly_score']:.5f} | Confidence: {res['confidence']}%")
    return results

# --- Plot and save confusion matrix ---
def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Confusion matrix saved to {save_path}")
    return cm

# --- Main ---
def main():
    print("Testing All Images and Creating Confusion Matrix...")
    model = load_autoencoder_model()
    all_results = []
    print("\nTesting GOOD images:")
    good_results = test_folder(model, GOOD_DIR, 'GOOD', THRESHOLD)
    print("\nTesting DEFECTIVE images:")
    defective_results = test_folder(model, DEFECTIVE_DIR, 'DEFECTIVE', THRESHOLD)
    all_results.extend(good_results)
    all_results.extend(defective_results)

    # --- Summary ---
    y_true = [r['expected'] for r in all_results]
    y_pred = [r['result'] for r in all_results]
    acc = accuracy_score(y_true, y_pred)
    print("\n--- SUMMARY ---")
    print(f"Total images tested: {len(all_results)}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"GOOD images: {len(good_results)}")
    print(f"DEFECTIVE images: {len(defective_results)}")

    # --- Print classification report ---
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['GOOD', 'DEFECTIVE']))

    # --- Print confusion matrix as text ---
    cm = confusion_matrix(y_true, y_pred, labels=['GOOD', 'DEFECTIVE'])
    print("Confusion Matrix (rows: Actual, cols: Predicted):")
    print("         GOOD  DEFECTIVE")
    print(f"GOOD     {cm[0,0]:4d}  {cm[0,1]:9d}")
    print(f"DEFECT   {cm[1,0]:4d}  {cm[1,1]:9d}")

    # --- Confusion Matrix Plot ---
    plot_confusion_matrix(y_true, y_pred, labels=['GOOD', 'DEFECTIVE'], save_path=CONFUSION_PNG)

if __name__ == "__main__":
    main() 