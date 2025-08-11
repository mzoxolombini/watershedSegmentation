# psoOptimisedWatershed.py (updated with confidence-based filtering)
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import urllib.request
import os
import shutil
import errno
from sklearn.decomposition import PCA
from skimage import exposure, filters, morphology, segmentation, color, measure, feature
from sklearn.metrics import adjusted_rand_score, jaccard_score
import ssl
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from pyswarm import pso
from sklearn.model_selection import cross_val_score
import socket
from skimage.color import rgb2gray
import signal
from contextlib import contextmanager
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from skimage.transform import resize
from joblib import Memory
import sys
import threading
from skimage import exposure
from skimage.morphology import disk, binary_dilation

memory = Memory(location='cache_dir', verbose=0)

# ================== CONFIGURATION PARAMETERS ==================
CONFIG = {
    'datasets': ['IndianPines', 'Salinas', 'PaviaU'],
    'pso_params': {
        'n_particles': 5,
        'max_iter': 5,
        'timeout': 14400  # 4 hours
    },
    'downsample_factor': 2,
    'watershed_params': {
        'conf_thresh': 0.60,
        'dilation_radius': 3
    },
    'classifiers': {
        'RF': {'n_estimators': 50, 'random_state': 42},
        'SVM': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf', 'probability': True},
        'KNN': {'n_neighbors': 5}
    }
}


# ================== CACHE DIRECTORY SETUP ==================
CACHE_DIR = 'watershed_cache'


def setup_cache():
    if os.path.exists(CACHE_DIR):
        try:
            shutil.rmtree(CACHE_DIR)
        except OSError as e:
            if e.errno != errno.ENOENT:
                print(f"Warning: Could not clear cache directory {CACHE_DIR}: {e}")
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create cache directory {CACHE_DIR}: {e}")


setup_cache()
memory = Memory(CACHE_DIR, verbose=0)

socket.setdefaulttimeout(30)
ssl._create_default_https_context = ssl._create_unverified_context


# ================== TIMEOUT HANDLING ==================
class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    if sys.platform == "win32":
        timer = None

        def timeout_handler():
            raise TimeoutException(f"Timed out after {seconds} seconds")

        timer = threading.Timer(seconds, timeout_handler)
        timer.start()
        try:
            yield
        finally:
            if timer is not None:
                timer.cancel()
    else:
        def signal_handler(signum, frame):
            raise TimeoutException(f"Timed out after {seconds} seconds")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)


# ================== DATASET LOADING ==================
@memory.cache
def load_dataset(name='IndianPines', downsample_factor=2):
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

    os.makedirs('data', exist_ok=True)
    dataset = datasets[name]
    img_path = f'data/{name}_corrected.mat'
    gt_path = f'data/{name}_gt.mat'

    if not os.path.exists(img_path):
        print(f"Downloading {name} image data...")
        urllib.request.urlretrieve(dataset['url'], img_path)
    if not os.path.exists(gt_path):
        print(f"Downloading {name} ground truth data...")
        urllib.request.urlretrieve(dataset['gt_url'], gt_path)

    img_data = loadmat(img_path)[dataset['img_key']]
    gt_data = loadmat(gt_path)[dataset['gt_key']]

    if downsample_factor > 1:
        img_data = resize(img_data,
                          (img_data.shape[0] // downsample_factor,
                           img_data.shape[1] // downsample_factor,
                           img_data.shape[2]),
                          anti_aliasing=True)
        gt_data = resize(gt_data,
                         (gt_data.shape[0] // downsample_factor,
                          gt_data.shape[1] // downsample_factor),
                         anti_aliasing=False,
                         preserve_range=True,
                         order=0).astype(int)

    return img_data, gt_data


# ================== EVALUATION ==================
def evaluate(true_mask, pred_mask):
    valid = true_mask > 0
    if not np.any(valid):
        return {'OA': 0.0, 'AA': 0.0, 'Kappa': 0.0, 'Dice': 0.0, 'IoU': 0.0, 'Confusion': None}
    true = true_mask[valid]
    pred = pred_mask[valid]
    if len(np.unique(pred)) < 2:
        return {'OA': 0.0, 'AA': 0.0, 'Kappa': 0.0, 'Dice': 0.0, 'IoU': 0.0, 'Confusion': None}
    oa = accuracy_score(true, pred)
    kappa = cohen_kappa_score(true, pred)
    dice = f1_score(true, pred, average='macro')
    iou = jaccard_score(true, pred, average='macro')
    cm = confusion_matrix(true, pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        aa_per_class = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        aa_per_class[np.isnan(aa_per_class)] = 0
    aa = np.mean(aa_per_class)
    return {'OA': oa, 'AA': aa, 'Kappa': kappa, 'Dice': dice, 'IoU': iou, 'Confusion': cm}


# ================== CONFIDENCE-BASED REFINEMENT ==================
def compute_max_probability_map(clf, img_2d_scaled, h, w):
    """
    Returns a 2D array of max prediction probabilities for each pixel.
    clf: fitted classifier with predict_proba
    img_2d_scaled: (N, B) scaled pixels
    """
    if not hasattr(clf, "predict_proba"):
        return np.zeros((h, w), dtype=float), None

    probs = clf.predict_proba(img_2d_scaled)  # shape (N, n_classes)
    maxp = probs.max(axis=1).reshape(h, w)
    return maxp, probs


def refine_predictions_by_confidence(preds_2d, proba_map, conf_thresh=0.6, dilation_radius=3):
    """
    Refine low-confidence regions by assigning them the majority label found among nearby high-confidence pixels.
    preds_2d : (H,W) int predicted labels (in GT label space: 1..C, 0 background)
    proba_map : (H,W) float max probs per pixel
    """
    h, w = preds_2d.shape
    high_conf_mask = (proba_map >= conf_thresh) & (preds_2d > 0)
    low_conf_mask = (proba_map < conf_thresh) & (preds_2d > 0)

    if not np.any(low_conf_mask):
        return preds_2d.copy()

    low_labels = measure.label(low_conf_mask, connectivity=2)
    refined = preds_2d.copy()
    selem = disk(dilation_radius)

    for rl in np.unique(low_labels):
        if rl == 0:
            continue
        region_mask = (low_labels == rl)
        dilated = binary_dilation(region_mask, selem)
        neighbor_mask = dilated & high_conf_mask
        neighbor_labels = preds_2d[neighbor_mask]

        if neighbor_labels.size > 0:
            maj = int(np.bincount(neighbor_labels.astype(int)).argmax())
            refined[region_mask] = maj
        else:
            neighbor_mask2 = dilated & (preds_2d > 0)
            neighbor_labels2 = preds_2d[neighbor_mask2]
            if neighbor_labels2.size > 0:
                maj2 = int(np.bincount(neighbor_labels2.astype(int)).argmax())
                refined[region_mask] = maj2

    return refined


# ================== SEGMENTATION + POSTPROCESS ==================
def segment_and_postprocess_hsi(image, gt_mask,
                                pca_components=3, sigma=1.0, thresh=0.5,
                                marker_min_distance=5, min_size=64, connectivity=2):
    """
    Robust watershed segmentation for hyperspectral image.
    Returns cleaned labeled regions (0 background).
    """
    h, w, b = image.shape
    reshaped = image.reshape(-1, b)

    # PCA intensity
    n_comp = min(pca_components, b)
    pca = PCA(n_components=n_comp)
    pca_flat = pca.fit_transform(reshaped)
    pca_img = pca_flat.reshape(h, w, n_comp)

    # intensity (first PC)
    intensity = pca_img[..., 0]

    # smoothing + gradient
    smoothed = filters.gaussian(intensity, sigma=sigma, preserve_range=True)
    gradient = filters.sobel(smoothed)

    # marker detection: try peak_local_max on inverted gradient
    try:
        local_max = peak_local_max(-gradient, indices=False, min_distance=marker_min_distance,
                                   labels=gt_mask, footprint=np.ones((3, 3)))
        markers, _ = ndi.label(local_max)
        if markers.max() == 0:
            raise ValueError("no markers from peak_local_max")
    except Exception:
        # fallback: low-gradient areas as markers (percentile)
        marker_mask = np.zeros_like(gradient, dtype=bool)
        valid_vals = gradient[gt_mask]
        if valid_vals.size > 0:
            cutoff = np.percentile(valid_vals, thresh * 100)
            marker_mask = (gradient < cutoff) & gt_mask
        markers = measure.label(marker_mask, connectivity=connectivity)

    # watershed (use gradient as elevation)
    segmented = segmentation.watershed(gradient, markers, mask=gt_mask)

    # Post-process: convert to binary mask, remove small objects, holes, relabel
    binary = segmented > 0
    if binary.any():
        cleaned = morphology.remove_small_objects(binary, min_size=min_size, connectivity=connectivity)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_size)
        cleaned_labels = measure.label(cleaned, connectivity=connectivity)
    else:
        cleaned_labels = np.zeros((h, w), dtype=int)

    return cleaned_labels


def assign_region_labels_by_majority(cleaned_labels, pixel_preds, background_mask=None):
    """
    Assign majority-voted class label to each region.
    pixel_preds: 2D int array with predicted class labels (in GT label space, i.e., 1..C)
    """
    h, w = cleaned_labels.shape
    region_assigned = np.zeros((h, w), dtype=pixel_preds.dtype)
    for region_label in np.unique(cleaned_labels):
        if region_label == 0:
            continue
        mask = cleaned_labels == region_label
        if not np.any(mask):
            continue
        vals = pixel_preds[mask]
        if vals.size == 0:
            continue
        counts = np.bincount(vals.astype(np.int64))
        majority = int(np.argmax(counts))
        region_assigned[mask] = majority
    if background_mask is not None:
        region_assigned[~background_mask] = 0
    return region_assigned


# ================== CACHED DECODER ==================
@memory.cache
def cached_decode_particle(particle_tuple, image, gt, classifiers, scaler, gt_mask, h, w,
                           conf_thresh=0.60, dilation_radius=3):
    try:
        particle = np.array(particle_tuple)
        pca_components = max(1, min(10, int(round(particle[0]))))
        sigma = max(0.1, min(5, float(particle[1])))
        thresh = max(0.01, min(0.99, float(particle[2])))
        min_size = max(1, min(500, int(round(particle[3]))))
        classifier_idx = min(2, max(0, int(round(particle[4]))))

        # segmentation + postprocess
        cleaned_labels = segment_and_postprocess_hsi(
            image, gt_mask,
            pca_components=pca_components, sigma=sigma, thresh=thresh,
            marker_min_distance=3, min_size=min_size, connectivity=2
        )

        # pixel-wise predictions (use scaler)
        reshaped = image.reshape(-1, image.shape[-1])
        img_2d_scaled = scaler.transform(reshaped)
        clf_key = ['RF', 'SVM', 'KNN'][classifier_idx]
        clf = classifiers[clf_key]

        # Get per-pixel probabilities & predictions
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(img_2d_scaled)  # (N, C)
            preds_flat = np.argmax(probs, axis=1)  # 0..C-1
            maxp_flat = probs.max(axis=1)
        else:
            preds_flat = clf.predict(img_2d_scaled)
            maxp_flat = np.zeros_like(preds_flat, dtype=float)

        preds_flat = preds_flat.astype(np.int64) + 1  # shift to GT label space
        preds_2d = preds_flat.reshape(h, w)
        proba_map = maxp_flat.reshape(h, w)

        preds_2d[~gt_mask] = 0
        proba_map[~gt_mask] = 0.0

        # refine low-confidence pixels by neighbor majority
        refined_preds = refine_predictions_by_confidence(preds_2d, proba_map,
                                                         conf_thresh=conf_thresh,
                                                         dilation_radius=dilation_radius)

        # assign region labels by majority vote from refined pixels
        final_map = assign_region_labels_by_majority(cleaned_labels, refined_preds, background_mask=gt_mask)
        return final_map

    except Exception as e:
        print(f"Error in cached_decode_particle: {e}")
        return np.zeros((h, w), dtype=int)


# ================== OPTIMIZED WATERSHED PSO ==================
class WatershedPSOOptimizer:
    def __init__(self, image, ground_truth):
        print("[Init] Initializing with optimized settings...")
        self.image = image
        self.gt = ground_truth
        self.gt_mask = self.gt > 0
        self.h, self.w = self.gt.shape

        # Precompute features & scaler
        self.b = self.image.shape[2]
        self.scaler = StandardScaler()
        img_2d = self.image.reshape(-1, self.b)
        self.img_2d_scaled = self.scaler.fit_transform(img_2d)

        # Prepare training data (labels shifted to 0..C-1)
        gt_flat = self.gt.flatten()
        self.mask = gt_flat > 0
        self.labels_gt = gt_flat[self.mask] - 1

        X = self.img_2d_scaled[self.mask]
        y = self.labels_gt

        # Initialize classifiers from config
        self.classifiers = {
            'RF': RandomForestClassifier(n_jobs=-1, **CONFIG['classifiers']['RF']),
            'SVM': SVC(**CONFIG['classifiers']['SVM']),
            'KNN': KNeighborsClassifier(n_jobs=-1, **CONFIG['classifiers']['KNN'])
        }

        for name, clf in self.classifiers.items():
            print(f"Training {name}...")
            clf.fit(X, y)

    def decode_particle(self, particle):
        try:
            pca_components = max(1, min(10, int(round(particle[0]))))
            sigma = max(0.1, min(5, float(particle[1])))
            thresh = max(0.01, min(0.99, float(particle[2])))
            min_size = max(1, min(500, int(round(particle[3]))))
            classifier_idx = min(2, max(0, int(round(particle[4]))))

            cleaned_labels = segment_and_postprocess_hsi(
                self.image, self.gt_mask,
                pca_components=pca_components,
                sigma=sigma,
                thresh=thresh,
                marker_min_distance=3,
                min_size=min_size,
                connectivity=2
            )

            clf_key = ['RF', 'SVM', 'KNN'][classifier_idx]
            clf = self.classifiers[clf_key]

            # pixel-wise probabilities
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(self.img_2d_scaled)  # (N, C)
                preds_flat = np.argmax(probs, axis=1)  # 0..C-1
                maxp_flat = probs.max(axis=1)
            else:
                preds_flat = clf.predict(self.img_2d_scaled)
                maxp_flat = np.zeros_like(preds_flat, dtype=float)

            preds_flat = preds_flat.astype(np.int64) + 1
            preds_2d = preds_flat.reshape(self.h, self.w)
            proba_map = maxp_flat.reshape(self.h, self.w)

            preds_2d[~self.gt_mask] = 0
            proba_map[~self.gt_mask] = 0.0

            # refine using config parameters
            refined_preds = refine_predictions_by_confidence(
                preds_2d,
                proba_map,
                conf_thresh=CONFIG['watershed_params']['conf_thresh'],
                dilation_radius=CONFIG['watershed_params']['dilation_radius']
            )

            final_map = assign_region_labels_by_majority(
                cleaned_labels,
                refined_preds,
                background_mask=self.gt_mask
            )
            return final_map
        except Exception as e:
            print(f"Error in decode_particle (refined): {e}")
            return np.zeros_like(self.gt)

    def evaluate_segmentation(self, class_map):
        return evaluate(self.gt, class_map)

    def optimize(self):
        print(f"[Optimize] Starting PSO with {CONFIG['pso_params']['n_particles']} particles and {CONFIG['pso_params']['max_iter']} iterations")
        lb = [1, 0.1, 0.01, 1, 0]
        ub = [10, 5, 0.99, 500, 2]

        def fitness(x):
            class_map = cached_decode_particle(
                tuple(x),
                self.image,
                self.gt,
                self.classifiers,
                self.scaler,
                self.gt_mask,
                self.h,
                self.w,
                conf_thresh=CONFIG['watershed_params']['conf_thresh'],
                dilation_radius=CONFIG['watershed_params']['dilation_radius']
            )
            score = self.evaluate_segmentation(class_map)['OA']
            return -score

        try:
            with time_limit(CONFIG['pso_params']['timeout']):
                start_time = time.time()
                best_params, best_fitness = pso(
                    fitness,
                    lb, ub,
                    swarmsize=CONFIG['pso_params']['n_particles'],
                    maxiter=CONFIG['pso_params']['max_iter'],
                    debug=True
                )
                elapsed = time.time() - start_time
                print(f"Optimization completed in {elapsed:.2f} seconds")
                final_segmentation = self.decode_particle(best_params)
                final_metrics = self.evaluate_segmentation(final_segmentation)
                return best_params, final_segmentation, final_metrics

        except TimeoutException as e:
            print(str(e))
            return None, None, None
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return None, None, None


# ================== VISUALIZATION ==================
def visualize_results(img, gt, segmentation, metrics):
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    if img.shape[2] > 3:
        pca = PCA(n_components=3)
        img_rgb = pca.fit_transform(img.reshape(-1, img.shape[2])).reshape(img.shape[0], img.shape[1], 3)
        img_rgb = exposure.rescale_intensity(img_rgb, in_range='image', out_range=(0, 1))
    else:
        img_rgb = img
    plt.imshow(img_rgb)
    plt.title('RGB Composite')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(gt, cmap='nipy_spectral')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(segmentation, cmap='nipy_spectral')
    title = (f"Segmentation Results\nOA: {metrics['OA']:.3f}, Kappa: {metrics['Kappa']:.3f}\n"
             f"Dice: {metrics['Dice']:.3f}, IoU: {metrics['IoU']:.3f}")
    plt.title(title)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# ================== MAIN PIPELINE ==================
def process_dataset(name, config):
    print(f"\nProcessing {name} dataset (downsampled by {config['downsample_factor']}x)")
    img, gt = load_dataset(name, downsample_factor=config['downsample_factor'])
    img = (img - img.min()) / (img.max() - img.min())

    optimizer = WatershedPSOOptimizer(img, gt)
    params, seg, metrics = optimizer.optimize()  # Removed the parameters here

    if metrics:
        print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            if metric != 'Confusion':
                print(f"{metric:>10}: {value:.4f}")
        visualize_results(img, gt, seg, metrics)
    return metrics


if __name__ == "__main__":
    results = {}
    for name in CONFIG['datasets']:
        try:
            print(f"\n{'=' * 40}\nProcessing {name} dataset\n{'=' * 40}")
            results[name] = process_dataset(name, CONFIG)
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")

    print("\nFinal Results Across Datasets:")
    print(f"{'Dataset':<12} {'OA':>8} {'AA':>8} {'Kappa':>8} {'Dice':>8} {'IoU':>8}")
    for name, metrics in results.items():
        if metrics:
            print(
                f"{name:<12} {metrics['OA']:8.4f} {metrics['AA']:8.4f} {metrics['Kappa']:8.4f} {metrics['Dice']:8.4f} {metrics['IoU']:8.4f}")

    # Clean up cache optionally
    try:
        shutil.rmtree(CACHE_DIR)
    except OSError as e:
        print(f"Note: Cache directory {CACHE_DIR} could not be removed - you may need to delete it manually: {e}")