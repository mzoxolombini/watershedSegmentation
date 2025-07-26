import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import urllib.request
import os
from sklearn.decomposition import PCA
from skimage import exposure, filters, morphology, segmentation, color, measure
from sklearn.metrics import adjusted_rand_score
import ssl
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


# Disable SSL verification and add retry mechanism
ssl._create_default_https_context = ssl._create_unverified_context


def robust_download(url, filename, max_retries=3):
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, filename)
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait before retrying
    return False


def load_dataset(name='IndianPines'):
    os.makedirs('data', exist_ok=True)
    datasets = {
        'IndianPines': {
            'url': "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
            'gt_url': "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
            'img_key': 'indian_pines_corrected',
            'gt_key': 'indian_pines_gt'
        },
        'Salinas': {
            'url': "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
            'gt_url': "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
            'img_key': 'salinas_corrected',
            'gt_key': 'salinas_gt'
        },
        'PaviaU': {
            'url': "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
            'gt_url': "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
            'img_key': 'paviaU',
            'gt_key': 'paviaU_gt'
        }
    }

    dataset = datasets[name]
    img_path = f'data/{name}_corrected.mat'
    gt_path = f'data/{name}_gt.mat'

    # Download with retry mechanism
    if not os.path.exists(img_path):
        print(f"Downloading {name} image data...")
        if not robust_download(dataset['url'], img_path):
            print(f"Failed to download image data after multiple attempts")
            return None, None

    if not os.path.exists(gt_path):
        print(f"Downloading {name} ground truth data...")
        if not robust_download(dataset['gt_url'], gt_path):
            print(f"Failed to download ground truth data after multiple attempts")
            return None, None

    # Load .mat files with error handling
    try:
        img_data = loadmat(img_path, verify_compressed_data_integrity=False)
        gt_data = loadmat(gt_path, verify_compressed_data_integrity=False)

        # Find correct keys automatically if default keys don't exist
        img_key = dataset['img_key']
        gt_key = dataset['gt_key']

        if img_key not in img_data:
            img_key = [k for k in img_data.keys() if not k.startswith('__')][0]
        if gt_key not in gt_data:
            gt_key = [k for k in gt_data.keys() if not k.startswith('__')][0]

        return img_data[img_key], gt_data[gt_key]

    except Exception as e:
        print(f"Error loading {name} data: {str(e)}")
        return None, None

def compute_rcmg(pca_img):
    """
    Compute the RCMG image by summing gradient magnitudes of PCA components.
    """
    gradients = []
    for i in range(3):  # PCA has 3 channels
        gx = filters.sobel_h(pca_img[:, :, i])
        gy = filters.sobel_v(pca_img[:, :, i])
        grad_mag = np.sqrt(gx**2 + gy**2)
        gradients.append(grad_mag)
    rcimg = np.sum(gradients, axis=0)
    rcimg = (rcimg - rcimg.min()) / (rcimg.max() - rcimg.min())  # Normalize
    return rcimg


from scipy import ndimage as ndi
from skimage.feature import peak_local_max

def apply_whed(regions):
    """
    Assign unlabeled pixels (label 0) to the nearest labeled region using WHED.
    """
    labeled = regions > 0
    distance, (inds_x, inds_y) = ndi.distance_transform_edt(~labeled, return_indices=True)
    regions_whed = np.copy(regions)
    unlabeled_mask = ~labeled
    regions_whed[unlabeled_mask] = regions[inds_x[unlabeled_mask], inds_y[unlabeled_mask]]
    return regions_whed


def majority_vote(class_map, regions):
    out = np.zeros_like(class_map)
    for label in np.unique(regions):
        if label == 0:
            continue
        mask = regions == label
        votes = class_map[mask]
        votes = votes[votes >= 0]  # Remove invalid (-1) predictions
        if len(votes) == 0:
            continue
        majority = np.bincount(votes).argmax()
        out[mask] = majority
    return out


def watershed_segmentation(img_rgb, min_distance=10):
    # Convert RGB to grayscale
    gray = color.rgb2gray(img_rgb)

    # Enhance contrast
    p2, p98 = np.percentile(gray, (2, 98))
    gray = exposure.rescale_intensity(gray, in_range=(p2, p98))

    # Threshold to binary foreground
    thresh = filters.threshold_otsu(gray)
    binary = gray > thresh

    # Compute distance transform
    distance = ndi.distance_transform_edt(binary)

    # Identify local maxima as markers
    coords = peak_local_max(distance, labels=binary, min_distance=min_distance, footprint=np.ones((3, 3)))
    markers = np.zeros_like(gray, dtype=int)
    for idx, (r, c) in enumerate(coords, 1):
        markers[r, c] = idx

    markers = ndi.label(markers)[0]

    # Apply watershed on inverted distance
    labels = segmentation.watershed(-distance, markers, mask=binary)

    return labels



def process_dataset(name):
    print(f"\nProcessing {name} dataset...")
    img, gt = load_dataset(name)
    if img is None or gt is None:
        return None

    # Normalize and reshape
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    h, w, b = img.shape
    img_2d = img.reshape(-1, b)

    # Prepare ground truth
    gt_flat = gt.flatten()
    mask = gt_flat > 0
    labels_gt = gt_flat[mask] - 1  # Ensure labels are 0-indexed

    # Standardize features
    scaler = StandardScaler()
    img_2d_scaled = scaler.fit_transform(img_2d)

    X = img_2d_scaled[mask]
    y = labels_gt

    # Train pixel-wise SVM
    clf = SVC(kernel='rbf', C=100, gamma=0.01)
    clf.fit(X, y)

    # Predict entire map
    preds = np.full(gt_flat.shape, -1)
    preds[mask] = clf.predict(X)
    class_map = preds.reshape(h, w)

    # Create RGB image from PCA and segment
    pca = PCA(n_components=3)
    img_pca = pca.fit_transform(img_2d).reshape(h, w, 3)
    img_pca = (img_pca - np.min(img_pca)) / (np.max(img_pca) - np.min(img_pca))
    regions = watershed_segmentation(img_pca)
    regions = apply_whed(regions)

    # Majority vote inside watershed regions
    class_map_mv = majority_vote(class_map, regions)

    # Evaluation
    y_pred = class_map_mv.flatten()[mask]
    oa = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    aa = np.mean(np.diag(cm) / cm.sum(axis=1))

    print(f"OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}")

    # Visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(img_pca)
    plt.title('RGB Composite')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(regions, cmap='nipy_spectral')
    plt.title('Watershed + WHED')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(class_map_mv, cmap='nipy_spectral')
    plt.title('Final Classification')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return {
        'OA': oa,
        'AA': aa,
        'Kappa': kappa
    }


if __name__ == "__main__":
    # Clear data directory for fresh downloads
    if os.path.exists('data'):
        for f in os.listdir('data'):
            os.remove(f'data/{f}')

    results = {}
    for name in ['IndianPines', 'Salinas', 'PaviaU']:
        try:
            metrics = process_dataset(name)
            if metrics:
                results[name] = metrics
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")

    print("\nFinal Results:")
    for name, metrics in results.items():
        print(f"{name}: OA = {metrics['OA']:.4f}, AA = {metrics['AA']:.4f}, Kappa = {metrics['Kappa']:.4f}")
