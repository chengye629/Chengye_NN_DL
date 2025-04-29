
import numpy as np
from scipy.ndimage import shift, rotate, zoom

def augment_image(img):
    """Apply random transformations to a single image."""
    img_aug = img.copy()

    C, H, W = img_aug.shape

    # Random shift
    if np.random.rand() < 0.5:
        shift_vals = (0, np.random.uniform(-2, 2), np.random.uniform(-2, 2))  # no shift on channel axis
        img_aug = shift(img_aug, shift=shift_vals, mode='nearest', order=1)

    # Random rotation
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-15, 15)
        for c in range(C):
            img_aug[c] = rotate(img_aug[c], angle=angle, reshape=False, mode='nearest', order=1)

    # Random zoom
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.9, 1.1)
        for c in range(C):
            zoomed = zoom(img_aug[c], zoom=factor, mode='nearest', order=1)
            # Center crop or pad back to (H, W)
            if factor < 1.0:
                pad_h = (H - zoomed.shape[0]) // 2
                pad_w = (W - zoomed.shape[1]) // 2
                img_aug[c] = np.pad(zoomed, ((pad_h, H - zoomed.shape[0] - pad_h),
                                             (pad_w, W - zoomed.shape[1] - pad_w)), mode='constant')
            else:
                crop_h = (zoomed.shape[0] - H) // 2
                crop_w = (zoomed.shape[1] - W) // 2
                img_aug[c] = zoomed[crop_h:crop_h+H, crop_w:crop_w+W]

    return img_aug

def augment_batch(X, y, expand_times=2):
    """
    Augment the dataset by creating more samples.
    expand_times = 2 means for each original image, generate 2 new versions
    """
    original_shape = X.shape

    if len(X.shape) == 2 and X.shape[1] == 28*28:
        X = X.reshape(-1, 1, 28, 28)

    X_aug = []
    y_aug = []

    for img, label in zip(X, y):
        X_aug.append(img)  # Keep original
        y_aug.append(label)
        for _ in range(expand_times):
            img_aug = augment_image(img)
            X_aug.append(img_aug)
            y_aug.append(label)

    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)

    if original_shape[1] == 28*28:
        X_aug = X_aug.reshape(-1, 28*28)

    return X_aug, y_aug

