# ============================================================
# EXPERIMENT 3: Contrast Stretching + Histogram Equalization
# ============================================================

from pathlib import Path              # Path = safe file paths on Windows/Linux/Mac
import cv2                            # cv2 = OpenCV for image read/convert/save
import numpy as np                    # np = NumPy for array math
import matplotlib.pyplot as plt       # plt = Matplotlib for display
import argparse                       # argparse = read command-line arguments (--input)


def contrast_stretch(gray: np.ndarray) -> np.ndarray:
    """
    gray: grayscale image (uint8 array)

    This function performs contrast stretching:
    - find min intensity (gmin)
    - find max intensity (gmax)
    - map [gmin, gmax] -> [0, 255]
    """

    gmin = np.min(gray)               # minimum pixel value in the whole image
    gmax = np.max(gray)               # maximum pixel value in the whole image

    if gmax == gmin:                  # if all pixels are same, no contrast to stretch
        return gray.copy()            # return a copy to keep original unchanged

    gray_f = gray.astype(np.float32)  # convert to float to avoid integer rounding errors
    stretched = (gray_f - gmin) * 255.0 / (gmax - gmin)  # linear stretching formula
    return stretched.astype(np.uint8) # convert back to uint8 (0..255)


def main():
    # --------------------------------------------------------
    # Read command-line argument --input "C:\path\to\image.jpg"
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Experiment 3: Contrast Stretching & Histogram Equalization"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Absolute path of input image (example: C:\\Users\\...\\h.jpg)"
    )
    args = parser.parse_args()

    # Path(args.input) converts the input string into a Path object
    input_path = Path(args.input)

    # outputs folder in your ipcv-lab project
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # cv2.imread reads image from disk (OpenCV loads in BGR format)
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        raise FileNotFoundError(f"OpenCV could not read: {input_path}")

    # Convert image to grayscale for histogram operations (single channel)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # A) Contrast Stretching
    stretched = contrast_stretch(gray)

    # B) Histogram Equalization (OpenCV built-in for uint8 grayscale)
    hist_eq = cv2.equalizeHist(gray)

    # Save outputs
    cv2.imwrite(str(out_dir / "exp03_gray.png"), gray)
    cv2.imwrite(str(out_dir / "exp03_contrast_stretched.png"), stretched)
    cv2.imwrite(str(out_dir / "exp03_hist_equalized.png"), hist_eq)

    # Display results
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Original Gray")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(stretched, cmap="gray")
    plt.title("Contrast Stretched")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(hist_eq, cmap="gray")
    plt.title("Histogram Equalized")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.hist(gray.ravel(), bins=256)
    plt.title("Histogram (Original Gray)")

    plt.tight_layout()
    plt.show()

    print("=== Experiment 3 Done ===")
    print("Input:", input_path)
    print("Saved in:", out_dir.resolve())


if __name__ == "__main__":
    main()