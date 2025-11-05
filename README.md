#  Roof Segmentation from Satellite Imagery

##  Overview
This project focuses on **automatic roof segmentation** from aerial or satellite images using **deep learning**.  
The model identifies roof regions in urban scenes — a task useful for **urban planning, disaster response, solar panel mapping**, and more.

It uses a **U-Net–based architecture** with selective fine-tuning and **boundary-aware loss** to improve precision along building edges.

---

##  Key Features
-  **U-Net architecture** (lightweight version for small datasets)  
-  **Selective fine-tuning** of decoder and encoder layers  
-  **Custom loss function** combining:
  - Binary Cross-Entropy (BCE)
  - Dice Loss
  - Boundary-aware term (based on Sobel edges)
-  **Consistent image–mask augmentation**
-  **Automatic post-processing filter** to clean up small or noisy predictions
-  Metrics: Accuracy, Dice coefficient, and IoU (Intersection-over-Union)

---

##  Model Architecture
The model is based on a **Small U-Net** encoder–decoder:

**Input (256×256×3)**

↓

**Encoder:**

32 → 64 → 128 → 256 feature blocks with MaxPooling

↓

**Bottleneck (512 filters)**

↓

**Decoder:**

**Transposed convolutions + skip connections**

↓

**Output (256×256×1) with sigmoid activation**


Optionally, you can switch to **EfficientNetB0 backbone** for improved feature extraction.

---

##  Training Strategy
The training is performed in **two or three stages** for better generalization:

- **Stage 1 – Train Decoder Only**
- **Stage 2 – Fine-Tune Entire Model**
- **(Optional) Stage 3 – Dynamic Loss Balancing**

Gradually reduce the boundary weight as the model learns better structure.

##  Evaluation Metrics

- Dice Coefficient — measures overlap quality
- IoU (Intersection over Union) — measures segmentation accuracy
- Accuracy — measures per-pixel correctness

##  Data

-Input: Aerial or satellite RGB images (256×256)
-Target: Binary roof masks (white = roof, black = background)
-Dataset Size: ~25 manually labeled images

You can optionally use:


##  Data Augmentation

Augmentation is applied identically to both images and masks:

- Horizontal flips
- Random rotations
- Brightness variation

This ensures spatial consistency between image and mask.

##  Postprocessing

After prediction, each mask is cleaned using:
- Morphological closing
- Contour area filtering
- (Optional) Shape filter to exclude non-rectangular blobs

post_pred = postprocess_mask(raw_pred, threshold=0.5)

---

##  Results
- Metric	Score
- Validation Dice	~0.84
- Validation IoU	~0.72
- Accuracy	~0.96

Detects most roofs correctly
Occasionally misclassifies wide roads as flat rooftops (can be reduced with shape filters or dataset refinement)

##  Requirements
- tensorflow >= 2.10
- numpy
- matplotlib
- opencv-python
- scikit-learn

---

##  Author

Developed by: Daria Kolbasova\

Environment: Kaggle Notebook

Framework: TensorFlow / Keras

##  Future Work

Integrate transformer-based segmentation (e.g., SegFormer)

Expand dataset to more roof materials

Add color-invariant augmentation

Improve road/roof differentiation using spectral cues
