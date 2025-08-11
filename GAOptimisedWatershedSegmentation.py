import os
import sys
import time
import errno
import ssl
import urllib.request
import shutil
import socket
import threading
from contextlib import contextmanager
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import seaborn as sns
from scipy.io import loadmat
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from skimage import filters, morphology, segmentation, measure, exposure
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.morphology import disk, binary_dilation
from joblib import Memory
import multiprocessing

import pygad

try:
    import PyQt5
    matplotlib.use('Qt5Agg')  # Use Qt backend
except ImportError:
    matplotlib.use('Agg')  # Fall back to non-interactive backend
    print("PyQt5 not found - plots will not be interactive")
import matplotlib.pyplot as plt

memory = Memory(location='cache_dir', verbose=0)

# ================== CONFIGURATION PARAMETERS ==================
CONFIG = {
    'datasets': ['IndianPines', 'Salinas', 'PaviaU'],
    'ga_params': {
        'population_size': 15,
        'generations': 50,  # Max generations
        'parents_mating': 5,
        'mutation_percent_genes': 20,
        'saturate_generations': 10, # Early stopping patience
        'timeout': 14400,
    },
    'downsample_factor': 2,
    'watershed_params': {
        'conf_thresh': 0.60,
        'dilation_radius': 3
    },
    'classifiers': {
        'RF': {'n_estimators': 50, 'random_state': 42, 'n_jobs': -1},
        'SVM': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf', 'probability': True},
        'KNN': {'n_neighbors': 5, 'n_jobs': -1}
    }
}

# ================== DATASET-SPECIFIC FITNESS CONFIG ==================
DATASET_FITNESS_CONFIG = {
    'IndianPines': {
        'w_smoothness': 0.20, 'target_regions': 1500, 'min_coverage_pct': 0.10
    },
    'Salinas': {
        'w_smoothness': 0.15, 'target_regions': 2000, 'min_coverage_pct': 0.10
    },
    'PaviaU': {
        'w_smoothness': 0.05, 'target_regions': 6000, 'min_coverage_pct': 0.02
    },
    'default': {
        'w_smoothness': 0.15, 'target_regions': 2000, 'min_coverage_pct': 0.10
    }
}

# ================== CACHE DIRECTORY SETUP ==================
CACHE_DIR = 'watershed_cache'
memory = Memory(location=CACHE_DIR, verbose=0)

socket.setdefaulttimeout(30)
ssl._create_default_https_context = ssl._create_unverified_context


def setup_cache():
    CACHE_DIR = 'watershed_cache'
    if os.path.exists(CACHE_DIR):
        try: shutil.rmtree(CACHE_DIR)
        except OSError as e:
            if e.errno != errno.ENOENT: print(f"Warning: Could not clear cache directory {CACHE_DIR}: {e}")
    try: os.makedirs(CACHE_DIR, exist_ok=True)
    except OSError as e: print(f"Warning: Could not create cache directory {CACHE_DIR}: {e}")
setup_cache()

@memory.cache
def cached_decode_particle_with_pca(particle_tuple, pca_img, image, gt, classifiers, scaler, gt_mask, h, w,
                                    conf_thresh=0.60, dilation_radius=3):
    try:
        particle = np.array(particle_tuple)
        pca_components = max(1, min(pca_img.shape[2], int(round(particle[0]))))
        sigma = max(0.1, min(5, float(particle[1])))
        thresh = max(0.01, min(0.99, float(particle[2])))
        min_size = max(1, min(500, int(round(particle[3]))))
        classifier_idx = min(2, max(0, int(round(particle[4]))))
        use_pca_img = pca_img[..., :pca_components]
        cleaned_labels = segment_and_postprocess_hsi_using_pca_img(
            use_pca_img, gt_mask, n_comp=pca_components, sigma=sigma, thresh=thresh,
            marker_min_distance=3, min_size=min_size, connectivity=2
        )
        reshaped = image.reshape(-1, image.shape[-1])
        img_2d_scaled = scaler.transform(reshaped)
        clf_key = ['RF', 'SVM', 'KNN'][classifier_idx]
        clf = classifiers[clf_key]
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(img_2d_scaled)
            preds_flat = np.argmax(probs, axis=1)
            maxp_flat = probs.max(axis=1)
        else:
            preds_flat = clf.predict(img_2d_scaled)
            maxp_flat = np.zeros_like(preds_flat, dtype=float)
        preds_flat = preds_flat.astype(np.int64) + 1
        preds_2d = preds_flat.reshape(h, w)
        proba_map = maxp_flat.reshape(h, w)
        preds_2d[~gt_mask] = 0
        proba_map[~gt_mask] = 0.0
        refined_preds = refine_predictions_by_confidence(preds_2d, proba_map,
                                                         conf_thresh=conf_thresh,
                                                         dilation_radius=dilation_radius)
        final_map = assign_region_labels_by_majority(cleaned_labels, refined_preds, background_mask=gt_mask)
        return final_map
    except Exception as e:
        print(f"Error in cached_decode_particle_with_pca: {e}")
        return np.zeros((h, w), dtype=int)


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

        import signal
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
    from sklearn.metrics import jaccard_score
    iou = jaccard_score(true, pred, average='macro')
    cm = confusion_matrix(true, pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        aa_per_class = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        aa_per_class[np.isnan(aa_per_class)] = 0
        aa = np.mean(aa_per_class)
    return {'OA': oa, 'AA': aa, 'Kappa': kappa, 'Dice': dice, 'IoU': iou, 'Confusion': cm}


# ================== CONFIDENCE-BASED REFINEMENT ==================
def refine_predictions_by_confidence(preds_2d, proba_map, conf_thresh=0.6, dilation_radius=3):
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
def segment_and_postprocess_hsi_using_pca_img(pca_img, gt_mask,
                                              n_comp=3, sigma=1.0, thresh=0.5,
                                              marker_min_distance=5, min_size=64, connectivity=2):
    h, w, _ = pca_img.shape
    intensity = pca_img[..., 0]
    smoothed = filters.gaussian(intensity, sigma=sigma, preserve_range=True)
    gradient = filters.sobel(smoothed)
    try:
        local_max = peak_local_max(-gradient, indices=False, min_distance=marker_min_distance,
                                   labels=gt_mask, footprint=np.ones((3, 3)))
        markers, _ = ndi.label(local_max)
        if markers.max() == 0:
            raise ValueError("no markers from peak_local_max")
    except Exception:
        marker_mask = np.zeros_like(gradient, dtype=bool)
        valid_vals = gradient[gt_mask]
        if valid_vals.size > 0:
            cutoff = np.percentile(valid_vals, thresh * 100)
            marker_mask = (gradient < cutoff) & gt_mask
        markers = measure.label(marker_mask, connectivity=connectivity)
    segmented = segmentation.watershed(gradient, markers, mask=gt_mask)
    binary = segmented > 0
    if binary.any():
        cleaned = morphology.remove_small_objects(binary, min_size=min_size, connectivity=connectivity)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_size)
        cleaned_labels = measure.label(cleaned, connectivity=connectivity)
    else:
        cleaned_labels = np.zeros((h, w), dtype=int)
    return cleaned_labels


# ============== ASSIGN REGION LABELS ================
def assign_region_labels_by_majority(cleaned_labels, pixel_preds, background_mask=None):
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
def cached_decode_particle_with_pca(particle_tuple, pca_img, image, gt, classifiers, scaler, gt_mask, h, w,
                                    conf_thresh=0.60, dilation_radius=3):
    try:
        particle = np.array(particle_tuple)
        pca_components = max(1, min(pca_img.shape[2], int(round(particle[0]))))
        sigma = max(0.1, min(5, float(particle[1])))
        thresh = max(0.01, min(0.99, float(particle[2])))
        min_size = max(1, min(500, int(round(particle[3]))))
        classifier_idx = min(2, max(0, int(round(particle[4]))))
        use_pca_img = pca_img[..., :pca_components]
        cleaned_labels = segment_and_postprocess_hsi_using_pca_img(
            use_pca_img, gt_mask, n_comp=pca_components, sigma=sigma, thresh=thresh,
            marker_min_distance=3, min_size=min_size, connectivity=2
        )
        reshaped = image.reshape(-1, image.shape[-1])
        img_2d_scaled = scaler.transform(reshaped)
        clf_key = ['RF', 'SVM', 'KNN'][classifier_idx]
        clf = classifiers[clf_key]
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(img_2d_scaled)
            preds_flat = np.argmax(probs, axis=1)
            maxp_flat = probs.max(axis=1)
        else:
            preds_flat = clf.predict(img_2d_scaled)
            maxp_flat = np.zeros_like(preds_flat, dtype=float)
        preds_flat = preds_flat.astype(np.int64) + 1
        preds_2d = preds_flat.reshape(h, w)
        proba_map = maxp_flat.reshape(h, w)
        preds_2d[~gt_mask] = 0
        proba_map[~gt_mask] = 0.0
        refined_preds = refine_predictions_by_confidence(preds_2d, proba_map,
                                                         conf_thresh=conf_thresh,
                                                         dilation_radius=dilation_radius)
        final_map = assign_region_labels_by_majority(cleaned_labels, refined_preds, background_mask=gt_mask)
        return final_map
    except Exception as e:
        print(f"Error in cached_decode_particle_with_pca: {e}")
        return np.zeros((h, w), dtype=int)

# ========== GLOBAL EVAL ARGS for GA (picklable) ============
GLOBAL_EVAL_ARGS = {}

# ====== GA FITNESS WRAPPER (ADAPTIVE) ========
def ga_fitness_wrapper(ga_instance, solution, solution_idx):
    args = GLOBAL_EVAL_ARGS
    if not args: return 0.0

    dataset_name = getattr(ga_instance, 'dataset_name', 'default')
    cfg = DATASET_FITNESS_CONFIG.get(dataset_name, DATASET_FITNESS_CONFIG['default'])
    w_smoothness = cfg['w_smoothness']
    target_regions = cfg['target_regions']
    min_coverage_pct = cfg['min_coverage_pct']

    try:
        class_map = cached_decode_particle_with_pca(tuple(solution), **args)
        metrics = evaluate(args['gt'], class_map)
        oa = metrics.get('OA', 0.0)
        aa = metrics.get('AA', 0.0)

        base_fitness = 0.55 * oa + 0.45 * aa

        labeled_pixel_count = np.count_nonzero(class_map)
        total_pixels_in_gt = np.count_nonzero(args['gt_mask'])
        coverage_ratio = labeled_pixel_count / total_pixels_in_gt if total_pixels_in_gt > 0 else 0.0
        coverage_penalty = 0.0
        if coverage_ratio < min_coverage_pct:
            coverage_penalty = 0.5 * (1.0 - (coverage_ratio / (min_coverage_pct + 1e-12)))

        num_regions = int(len(np.unique(class_map)) - (1 if 0 in np.unique(class_map) else 0))
        smoothness_penalty = 0.0
        if num_regions > target_regions:
            excess_ratio = (num_regions - target_regions) / (target_regions + 1e-12)
            smoothness_penalty = min(0.6, w_smoothness * excess_ratio)

        final_fitness = base_fitness - coverage_penalty - smoothness_penalty
        return float(max(0.0, min(1.0, final_fitness)))
    except Exception as e:
        print(f"GA fitness eval error: {e}")
        return 0.0

# ================== OPTIMIZED WATERSHED GA ==================
class WatershedGAOptimizer:
    def __init__(self, image, ground_truth, dataset_name):
        print(f"[Init] Initializing GA optimizer for {dataset_name}...")
        self.image = image
        self.gt = ground_truth
        self.dataset_name = dataset_name
        self.gt_mask = self.gt > 0
        self.h, self.w = self.gt.shape
        self.b = self.image.shape[2]
        max_comp = min(10, self.b)
        reshaped = self.image.reshape(-1, self.b)
        self.pca_model = PCA(n_components=max_comp)
        pca_flat = self.pca_model.fit_transform(reshaped)
        self.pca_img = pca_flat.reshape(self.h, self.w, max_comp)
        self.scaler = StandardScaler()
        img_2d = self.image.reshape(-1, self.b)
        self.img_2d_scaled = self.scaler.fit_transform(img_2d)
        gt_flat = self.gt.flatten()
        mask = gt_flat > 0
        y = gt_flat[mask].astype(int) - 1
        X = self.img_2d_scaled[mask]
        self.classifiers = {
            'RF': RandomForestClassifier(**CONFIG['classifiers']['RF']),
            'SVM': SVC(**CONFIG['classifiers']['SVM']),
            'KNN': KNeighborsClassifier(**CONFIG['classifiers']['KNN'])
        }
        for name, clf in self.classifiers.items():
            print(f"Training {name}...")
            clf.fit(X, y)

    def optimize(self):
        print(f"[Optimize] Starting GA for {self.dataset_name} dataset...")
        global GLOBAL_EVAL_ARGS
        GLOBAL_EVAL_ARGS = {
            'pca_img': self.pca_img, 'image': self.image, 'gt': self.gt,
            'classifiers': self.classifiers, 'scaler': self.scaler, 'gt_mask': self.gt_mask,
            'h': self.h, 'w': self.w
        }
        gene_space = [
            {'low': 1, 'high': self.pca_img.shape[2], 'step': 1},
            {'low': 0.1, 'high': 5},
            {'low': 0.01, 'high': 0.99},
            {'low': 10, 'high': 500, 'step': 1},
            {'low': 0, 'high': 2, 'step': 1}
        ]
        ga_instance = pygad.GA(
            num_generations=CONFIG['ga_params']['generations'],
            num_parents_mating=CONFIG['ga_params']['parents_mating'],
            fitness_func=ga_fitness_wrapper,
            sol_per_pop=CONFIG['ga_params']['population_size'],
            num_genes=len(gene_space),
            gene_space=gene_space,
            mutation_percent_genes=CONFIG['ga_params']['mutation_percent_genes'],
            parent_selection_type="sss",
            crossover_type="single_point",
            mutation_type="random",
            stop_criteria=[f"saturate_{CONFIG['ga_params']['saturate_generations']}"]
        )
        ga_instance.dataset_name = self.dataset_name

        try:
            ga_instance.run()
            best_solution, best_fitness, _ = ga_instance.best_solution()
            print(f"[Result] Best solution from GA: {best_solution} with fitness: {best_fitness:.4f}")

            refined_solution, refined_fitness = self.local_search_refinement(best_solution, gene_space)

            final_seg = cached_decode_particle_with_pca(tuple(refined_solution), **GLOBAL_EVAL_ARGS)
            final_metrics = evaluate(self.gt, final_seg)
            print(f"[Done] Optimization completed for {self.dataset_name}.")

            return refined_solution, final_seg, final_metrics, ga_instance
        except Exception as e:
            print(f"Error during GA optimization for {self.dataset_name}: {e}")
            return None, None, None, None

    def local_search_refinement(self, initial_solution, gene_space):
        print("--- Starting Local Search Refinement ---")
        current_solution = np.array(initial_solution)
        current_fitness = ga_fitness_wrapper(None, current_solution, 0)
        steps = 20
        for step_num in range(steps):
            best_neighbor, best_neighbor_fitness = current_solution, current_fitness
            for i in range(len(current_solution)):
                gene_info = gene_space[i]
                low, high = gene_info['low'], gene_info['high']
                step_size = 1 if 'step' in gene_info and gene_info['step'] == 1 else (high - low) * 0.02
                for delta in [-step_size, step_size]:
                    candidate = current_solution.copy()
                    candidate[i] = np.clip(candidate[i] + delta, low, high)
                    if 'step' in gene_info and gene_info['step'] == 1:
                        candidate[i] = round(candidate[i])
                    candidate_fitness = ga_fitness_wrapper(None, candidate, 0)
                    if candidate_fitness > best_neighbor_fitness:
                        best_neighbor, best_neighbor_fitness = candidate, candidate_fitness
            if best_neighbor_fitness > current_fitness:
                print(f"Local Search Step {step_num + 1}: Fitness improved from {current_fitness:.4f} to {best_neighbor_fitness:.4f}")
                current_solution, current_fitness = best_neighbor, best_neighbor_fitness
            else:
                print("Local search converged. No further improvement found.")
                break
        print(f"--- Local Search Finished. Final Fitness: {current_fitness:.4f} ---")
        return current_solution, current_fitness


# ================== VISUALIZATION ==================
def visualize_results(img, gt, segmentation, metrics):
    # Create a new figure
    fig = plt.figure(figsize=(15, 5), num='Hyperspectral Segmentation Results')

    # RGB Composite subplot
    ax1 = plt.subplot(131)
    if img.shape[2] > 3:
        pca = PCA(n_components=3)
        img_rgb = pca.fit_transform(img.reshape(-1, img.shape[2])).reshape(img.shape[0], img.shape[1], 3)
        img_rgb = exposure.rescale_intensity(img_rgb, in_range='image', out_range=(0, 1))
    else:
        img_rgb = img
    ax1.imshow(img_rgb)
    ax1.set_title('RGB Composite')
    ax1.axis('off')

    # Ground Truth subplot
    ax2 = plt.subplot(132)
    ax2.imshow(gt, cmap='nipy_spectral')
    ax2.set_title('Ground Truth')
    ax2.axis('off')

    # Segmentation Results subplot
    ax3 = plt.subplot(133)
    im = ax3.imshow(segmentation, cmap='nipy_spectral')
    title = (f"Segmentation Results\nOA: {metrics['OA']:.3f}, Kappa: {metrics['Kappa']:.3f}\n"
             f"Dice: {metrics['Dice']:.3f}, IoU: {metrics['IoU']:.3f}")
    ax3.set_title(title)
    ax3.axis('off')

    plt.tight_layout()

    # Check if we're using an interactive backend
    if matplotlib.get_backend().lower() != 'agg':
        try:
            plt.show(block=False)
            plt.pause(0.1)  # Allow time for the window to appear
            print("Close the plot window to continue...")
            while plt.fignum_exists(fig.number):
                plt.pause(0.5)  # Keep the window open
        except:
            plt.show(block=True)  # Fallback if non-interactive
    else:
        # Save to file if not interactive
        output_path = f"{name}_results.png"
        plt.savefig(output_path)
        print(f"Results saved to {output_path}")

    plt.close(fig)


# ================== MAIN PIPELINE ==================
def process_dataset(name, config):
    try:
        print(f"\nProcessing {name} dataset (downsampled by {config['downsample_factor']}x)")
        img, gt = load_dataset(name, downsample_factor=config['downsample_factor'])
        img = (img - img.min()) / (img.max() - img.min())
        optimizer = WatershedGAOptimizer(img, gt, name)

        params, seg, metrics, ga_instance = optimizer.optimize()

        if metrics:
            print("\nPerformance Metrics:")
            for metric, value in metrics.items():
                if metric != 'Confusion':
                    print(f"{metric:>10}: {value:.4f}")
            visualize_results(img, gt, seg, metrics)
        return metrics
    except Exception as e:
        print(f"Error processing {name}: {str(e)}")
        return None


if __name__ == "__main__":
    multiprocessing.freeze_support()
    results = {}
    for name in CONFIG['datasets']:
        try:
            print(f"\n{'=' * 40}\nProcessing {name} dataset\n{'=' * 40}")
            results[name] = process_dataset(name, CONFIG)
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
    print("\nFinal Results Across Datasets:")
    print(f"{'Dataset':<12} {'OA':>8} {'AA':>8} {'Kappa':>8} {'Dice':>8} {'IoU':>8}")
    if results:
        for name, metrics in results.items():
            if metrics:
                print(
                    f"{name:<12} {metrics['OA']:8.4f} {metrics['AA']:8.4f} {metrics['Kappa']:8.4f} {metrics['Dice']:8.4f} {metrics['IoU']:8.4f}")
    try:
        shutil.rmtree(CACHE_DIR)
    except OSError as e:
        print(f"Note: Cache directory {CACHE_DIR} could not be removed - you may need to delete it manually: {e}")