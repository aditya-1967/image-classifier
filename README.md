# Scene Image Classifier: CNN from Scratch vs Transfer Learning

A deep-dive comparison of two approaches to image classification on the Intel Image Classification dataset — a custom CNN trained from scratch versus a fine-tuned ResNet18. Built to understand not just *what* works, but *why*.

---

## Results

| Model | Val Accuracy | Parameters | Training Strategy |
|---|---|---|---|
| SimpleCNN (scratch) | 85.6% | 102K | 15 epochs, Adam, CosineAnnealingLR |
| ResNet18 (fine-tuned) | 93.9% | 11.2M | 5 epochs head-only + 10 epochs full fine-tune |

**8.3% accuracy improvement purely from reusing ImageNet features — without a single architecture change.**

---

## Training Curves

![Training Curves](src\training_curves.png)

ResNet18 opens at 86% accuracy on epoch 1 — before meaningfully training on this dataset. That's the weight of 1.2 million ImageNet images already baked into the backbone. SimpleCNN starts at 60% and has to learn every edge, texture, and shape from scratch. The gap never closes.

The loss curves tell the same story: ResNet starts confident (loss ~0.4) and converges smoothly. SimpleCNN starts uncertain (loss ~0.92) and takes the full 15 epochs to stabilise.

---

## Grad-CAM Visualisations

![Grad-CAM](src\gradcam.png)

Grad-CAM (Gradient-weighted Class Activation Mapping) visualises which regions of the image the model attended to when making its prediction. Red = high attention, blue = low attention.

- **Buildings** → model focuses on glass facades, angular edges, and architectural grid patterns
- The model is not cheating — it attends to semantically meaningful regions, not background noise or image artefacts

---

## Dataset

**Intel Image Classification** — 6 scene categories, ~14K training images, 3K validation images.

| Class | Train Images |
|---|---|
| buildings | 2,191 |
| forest | 2,271 |
| glacier | 2,404 |
| mountain | 2,512 |
| sea | 2,274 |
| street | 2,382 |

Download: [kaggle.com/puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

---

## Architecture

### SimpleCNN (from scratch)

Three convolutional blocks feeding into a fully connected classifier. Built to understand the fundamentals before reaching for pretrained models.

```
Input:          (B,   3, 150, 150)
Block 1:        (B,  32,  75,  75)   Conv2d → BatchNorm → ReLU → MaxPool
Block 2:        (B,  64,  37,  37)   Conv2d → BatchNorm → ReLU → MaxPool
Block 3:        (B, 128,  18,  18)   Conv2d → BatchNorm → ReLU → MaxPool
AvgPool:        (B, 128,   1,   1)   AdaptiveAvgPool2d(1)
Flatten:        (B, 128)
Linear + ReLU:  (B,  64)
Dropout(0.4):   (B,  64)
Output:         (B,   6)             one logit per class
```

**102,342 trainable parameters.**

### ResNet18 (transfer learning)

ImageNet-pretrained backbone with the final FC layer replaced. Trained in two stages for stability.

**Stage 1 — Head only (5 epochs, lr=1e-3):**
Freeze the entire backbone. Only train the new `Linear(512 → 6)` layer — 3,078 parameters out of 11 million. Forces the head to learn sensible weights before touching the backbone.

**Stage 2 — Full fine-tune (10 epochs, lr=1e-4):**
Unfreeze everything. Fine-tune the whole network at a much lower learning rate to preserve the pretrained features while specialising toward scene classification.

---

## Concepts Explained

### What does Conv2d do?
Slides a small filter (e.g. 3×3) over the image and computes dot products at each spatial location — detecting local patterns like edges, corners, and textures. Each Conv2d layer learns its own set of filters automatically during training.

### Why MaxPool?
Downsamples spatially by taking the maximum value in each 2×2 window. This halves the height and width (150→75→37→18), reducing computation and adding **translation invariance** — a feature detected slightly off-centre still survives pooling.

### What is BatchNorm?
Normalises activations within a mini-batch to zero mean and unit variance, then learns a scale and shift on top. Stabilises training by preventing activations from growing very large or collapsing to near-zero. Practical effect: faster training, less sensitivity to learning rate choice, mild regularisation.

### Why Dropout?
Randomly zeros 40% of neurons during training. Forces the network to not rely on any single neuron — every feature has to be learnable by multiple paths. Reduces overfitting. Disabled automatically during `model.eval()`.

### Why ReLU?
`f(x) = max(0, x)`. Without non-linearity, stacking layers is mathematically equivalent to a single layer — you can always collapse linear transformations into one. ReLU breaks linearity so deep networks can learn complex, non-linear decision boundaries.

### What does AdaptiveAvgPool2d(1) do?
Collapses any spatial dimension to 1×1 by averaging. Removes the dependency on input image size — the classifier always receives a fixed-length vector regardless of how large the feature maps are.

### Why freeze then unfreeze in transfer learning?
If you unfreeze the entire network at once with a randomly initialised head, the large random gradients from the head corrupt the pretrained backbone on the first few updates. Freezing first forces the head to stabilise before letting those gradients propagate deeper.

### What does ResNet solve?
**Vanishing gradients.** In very deep plain CNNs, gradients shrink as they backpropagate through many layers (multiply by numbers <1 repeatedly). Early layers stop learning. ResNet adds **skip connections** (`output = F(x) + x`) that provide a direct gradient highway bypassing layer stacks. This makes networks of 18, 50, even 152 layers trainable.

### What is transfer learning?
ImageNet-pretrained weights already encode general visual knowledge — edges in early layers, textures in middle layers, object parts in later layers. Reusing them for a new task requires far less data and training time than random initialisation. The pretrained model starts close to a good solution; we just specialise it.

### What is Grad-CAM?
Backpropagates the model's prediction score through the network, stops at the last conv layer, and asks: "which feature maps had the largest gradient — i.e. mattered most to this prediction?" Weights each feature map by its average gradient, sums them into a single spatial map, and applies ReLU. The result is a heatmap showing which image regions drove the decision.

We use the **last** conv layer specifically because that's where features are most semantic — it detects whole concepts (towers, facades, canopies) rather than low-level edges.

---

## Interview Q&A

**Q: Your CNN from scratch hit 85.6%. ResNet18 hit 93.9%. Why the gap?**

ResNet18 starts with weights trained on 1.2 million ImageNet images. Its early layers already detect universal visual features — edges, colour gradients, textures — that transfer directly to any image classification task. SimpleCNN initialises randomly and must learn all of these from 14K images in 15 epochs. The pretrained model also has 11M parameters giving it far greater representational capacity, though that alone doesn't explain the gap — a randomly initialised ResNet18 would perform much worse without pretraining.

**Q: Why do you normalise with ImageNet mean and std even though this isn't ImageNet data?**

The pretrained ResNet18 was trained with inputs normalised to ImageNet statistics. Its weights — learned activations, batch norm parameters — are calibrated to that input distribution. Feeding unnormalised inputs would shift the distribution the network expects, degrading performance. For SimpleCNN it matters less, but we use the same transforms for consistency.

**Q: What would you try next to improve beyond 93.9%?**

- **Larger ResNet variant** — ResNet34 or ResNet50 for more capacity
- **More aggressive augmentation** — CutMix, MixUp, or RandomErasing
- **Label smoothing** — replace hard 0/1 targets with 0.1/0.9 to prevent overconfidence
- **Test-time augmentation (TTA)** — average predictions over multiple augmented versions of each test image
- **Per-class metrics** — a confusion matrix would reveal which classes are hardest and guide targeted data collection

**Q: Why do we call `opt.zero_grad()` at every batch?**

PyTorch accumulates gradients by default — `loss.backward()` adds to existing gradients rather than replacing them. Without zeroing, batch N's gradients bleed into batch N+1's update. We need a clean slate each batch.

**Q: Why does validation use `@torch.no_grad()`?**

During training, PyTorch builds a computation graph of every operation to enable backpropagation. This costs memory and time. During validation we never call `loss.backward()` — we're only measuring accuracy. `@torch.no_grad()` tells PyTorch not to build the graph, saving both VRAM and compute.

**Q: Why save the best checkpoint rather than the last epoch?**

The last epoch is not necessarily the best one. Validation accuracy can peak at epoch 12 then slightly regress by epoch 15 due to overfitting. Saving the best means deployment uses the weights that actually performed best on unseen data.

---

## Limitations

- **Fixed validation set** — repeated experiments risk overfitting to it. A truly held-out test set would give a more honest estimate of generalisation.
- **Coarse Grad-CAM** — heatmaps are 5×5 spatial resolution (after all MaxPool downsampling). Grad-CAM++ or Score-CAM would give finer maps.
- **No confusion matrix** — overall accuracy hides per-class performance. Glacier vs mountain is likely the hardest pair; a confusion matrix would confirm this.
- **Balanced dataset** — this dataset is roughly class-balanced (~2,300 images per class). Real-world deployment would need per-class metrics and potentially weighted sampling.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/aditya-1967/image-classifier.git
cd image-classifier

# Create virtual environment (Python 3.12 required — PyTorch has no 3.14 wheels yet)
python -m venv cv-env
cv-env\Scripts\activate        # Windows
source cv-env/bin/activate     # Linux/Mac

# Install dependencies (CUDA 12.1 build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib seaborn pillow tqdm kaggle

# Download dataset
kaggle datasets download -d puneet6060/intel-image-classification
tar -xf intel-image-classification.zip -d data/

# Train
python src/train.py
```

**Requirements:** Python 3.12, CUDA-capable GPU recommended (tested on RTX 4050 6GB)

---

## Stack

PyTorch · torchvision · matplotlib · Intel Image Classification (Kaggle)

---

## Project Series

This is Project 2 of a 12-week ML portfolio build covering three tracks:

| Track | Project |
|---|---|
| LLMs | Semantic Search Engine |
| **Computer Vision** | **Scene Image Classifier ← you are here** |
| Classical ML | Credit Risk Classifier |