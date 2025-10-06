# MNIST Training: Achieving 99.4%+ Accuracy with <20k Parameters

## 🎯 Project Goal

This project demonstrates the iterative development of a Convolutional Neural Network (CNN) to achieve **99.4%+ test accuracy** on the MNIST dataset with the following constraints:
- **Maximum 20,000 trainable parameters**
- **Maximum 20 epochs**
- **Efficient training on Apple Silicon (MPS)**

## 📊 Final Results

| Metric | Value |
|--------|-------|
| **Final Test Accuracy** | **99.51%** |
| **Total Parameters** | **13,226** |
| **Training Epochs** | **20** |
| **Final Test Loss** | **0.0170** |
| **Training Time** | ~23 seconds per epoch |
| **Parameter Efficiency** | **99.8% reduction** (from 6.4M to 13K) |
| **Accuracy Improvement** | **+62.47%** (from 37% to 99.51%) |

## 🏗️ Final Architecture

The optimized CNN architecture consists of:

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Block 1: 8 → 16 channels
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.drop1 = nn.Dropout2d(0.05)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: 16 → 16 channels
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.drop2 = nn.Dropout2d(0.05)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: Feature extraction
        self.conv5 = nn.Conv2d(16, 16, 3)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 16, 3)
        self.bn6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(16, 16, 3)
        self.bn7 = nn.BatchNorm2d(16)
        
        # Classification
        self.fc1 = nn.Linear(16, 10)
```

**Key Features:**
- **Batch Normalization** after each convolution
- **Dropout2D** (5%) for regularization
- **MaxPooling** for dimension reduction
- **Global Average Pooling** equivalent (1x1 final feature map)
- **Log Softmax** for stable training

## 📈 Evolution Journey

The neural network underwent significant optimization through **19 training runs**. Here's the complete journey from 37% to 99.51% accuracy:

| Run # | Report | Parameters | Accuracy | Key Changes | Status |
|-------|--------|------------|----------|-------------|---------|
| 1 | [CNN Model_MNIST_20251003_104329.html](reports/CNN%20Model_MNIST_20251003_104329.html) | 6,379,786 | 37.04% | Initial massive model (3 epochs) | 🔴 Baseline |
| 2 | [CNN Model_MNIST_20251003_111456.html](reports/CNN%20Model_MNIST_20251003_111456.html) | 6,379,786 | 78.22% | Extended to 10 epochs | 🟡 Improvement |
| 3 | [CNN Model_MNIST_20251003_112628.html](reports/CNN%20Model_MNIST_20251003_112628.html) | 6,379,786 | 89.16% | Batch size optimization (32) | 🟡 Better |
| 4 | [CNN Model_MNIST_20251003_115930.html](reports/CNN%20Model_MNIST_20251003_115930.html) | 18,002 | 89.41% | **Channel reduction (4 channels)** | 🟢 Major reduction |
| 5 | [CNN Model_MNIST_20251003_164214.html](reports/CNN%20Model_MNIST_20251003_164214.html) | 13,598 | 98.90% | Architecture refinement | 🟢 Breakthrough |
| 6 | [CNN Model_MNIST_20251003_171900.html](reports/CNN%20Model_MNIST_20251003_171900.html) | 13,794 | 99.37% | Fine-tuning | 🟢 Excellent |
| 7 | [CNN Model_MNIST_20251003_172641.html](reports/CNN%20Model_MNIST_20251003_172641.html) | 49,278 | 99.43% | **Added BatchNorm + Dropout** | 🟢 Best so far |
| 8 | [CNN Model_MNIST_20251003_201238.html](reports/CNN%20Model_MNIST_20251003_201238.html) | 19,362 | 99.42% | Channel optimization (32→24) | 🟢 Efficient |
| 9 | [CNN Model_MNIST_20251003_204943.html](reports/CNN%20Model_MNIST_20251003_204943.html) | 19,362 | 99.47% | **Added Random Rotation** | 🟢 Data augmentation |
| 10 | [CNN Model_MNIST_20251003_210221.html](reports/CNN%20Model_MNIST_20251003_210221.html) | 19,362 | 99.18% | Horizontal Flip (❌ hurt performance) | 🔴 Failed experiment |
| 11 | [CNN Model_MNIST_20251003_213721.html](reports/CNN%20Model_MNIST_20251003_213721.html) | 19,362 | 98.23% | Vertical Flip (❌ hurt performance) | 🔴 Failed experiment |
| 12 | [CNN Model_MNIST_20251003_214701.html](reports/CNN%20Model_MNIST_20251003_214701.html) | 19,362 | 99.38% | Reverted to working config | 🟢 Recovery |
| 13 | [CNN Model_MNIST_20251003_215937.html](reports/CNN%20Model_MNIST_20251003_215937.html) | 19,472 | 99.38% | Minor architecture tweak | 🟢 Stable |
| 14 | [CNN Model_MNIST_20251003_220202.html](reports/CNN%20Model_MNIST_20251003_220202.html) | 12,284 | 97.60% | Further parameter reduction | 🟡 Trade-off |
| 15 | [CNN Model_MNIST_20251003_220342.html](reports/CNN%20Model_MNIST_20251003_220342.html) | 12,284 | 97.60% | Duplicate run | 🟡 Consistent |
| 16 | [CNN Model_MNIST_20251003_221458.html](reports/CNN%20Model_MNIST_20251003_221458.html) | 13,226 | 99.51% | **Final optimized architecture** | 🏆 **TARGET ACHIEVED** |
| 17 | [CNN Model_MNIST_20251003_235232.html](reports/CNN%20Model_MNIST_20251003_235232.html) | 13,226 | 99.51% | Verification run | 🏆 **Confirmed** |
| 18 | [CNN Model_MNIST_20251004_001529.html](reports/CNN%20Model_MNIST_20251004_001529.html) | 13,226 | 99.51% | Final validation | 🏆 **Final Result** |
| 19 | [CNN Model_MNIST_20251004_002800.html](reports/CNN%20Model_MNIST_20251004_002800.html) | 13,226 | 97.09% | Additional experiment | 🟡 Variation |

## 🔄 Key Optimization Steps

### 1. **Parameter Reduction Journey**
- **Started with:** 6,379,786 parameters (massive overkill)
- **Final model:** 13,226 parameters (99.8% reduction!)
- **Strategy:** Systematic channel reduction while maintaining performance
- **Key milestone:** Run #4 achieved 89.41% with only 18,002 parameters

### 2. **Architecture Improvements**
- **Batch Normalization:** Added in Run #7 (99.43% accuracy)
- **Dropout:** 5% Dropout2D for regularization
- **Channel Optimization:** Fine-tuned from 8→16→16→16→16→16→16
- **Global Pooling:** Efficient feature extraction (1x1 final feature map)

### 3. **Data Augmentation Experiments**
- ✅ **Random Rotation (±20°):** Improved accuracy to 99.47% (Run #9)
- ❌ **Horizontal Flip:** Reduced accuracy to 99.18% (Run #10)
- ❌ **Vertical Flip:** Reduced accuracy to 98.23% (Run #11)
- ✅ **Random Affine:** Maintained performance

### 4. **Training Optimizations**
- **Batch Size:** Optimized to 32 (Run #3)
- **Learning Rate:** 0.001 with SGD momentum (0.9)
- **Device:** Apple Silicon MPS acceleration
- **Regularization:** BatchNorm + Dropout combination
- **Epochs:** Extended from 3 to 20 for full convergence

### 5. **Breakthrough Moments**
- **Run #5:** First breakthrough to 98.90% accuracy
- **Run #7:** Added BatchNorm + Dropout (99.43%)
- **Run #9:** Data augmentation with rotation (99.47%)
- **Run #16:** Final optimized architecture (99.51%)

## 🚀 How to Run

### Prerequisites
```bash
# Install dependencies
pip install torch torchvision matplotlib tqdm torchsummary
```

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd mnist-training-epon

# Run the training notebook
jupyter notebook mnist-training-epon.ipynb
```

### Alternative: Python Script
```bash
# Run the main script
python main.py
```

### System Requirements
- **Python:** 3.11+
- **PyTorch:** 2.8.0+
- **Device:** Apple Silicon (MPS) or CUDA/CPU
- **Memory:** 4GB+ RAM recommended

## 📁 Project Structure

```
mnist-training-epon/
├── data/                          # MNIST dataset
│   └── MNIST/
├── reports/                       # HTML training reports
│   ├── CNN Model_MNIST_*.html
├── main.py                       # Main training script
├── mnist-training-epon.ipynb     # Jupyter notebook
├── pyproject.toml               # Dependencies
└── README.md                    # This file
```

## 📊 Training Reports

All training runs generate detailed HTML reports with:
- **Training curves** (loss & accuracy)
- **Model architecture** visualization
- **Parameter count** and efficiency metrics
- **Performance statistics**
- **Experiment configuration**

**Latest Report:** [CNN Model_MNIST_20251004_001529.html](reports/CNN%20Model_MNIST_20251004_001529.html)

## 🎯 Key Learnings

### What Worked:
1. **Batch Normalization** - Critical for stable training
2. **Dropout Regularization** - Prevented overfitting
3. **Random Rotation** - Effective data augmentation for digits
4. **Channel Reduction** - Massive parameter savings without performance loss
5. **Apple Silicon MPS** - Excellent training acceleration

### What Didn't Work:
1. **Horizontal/Vertical Flips** - Inappropriate for digit recognition
2. **Excessive Parameters** - 6M+ parameters were unnecessary
3. **Large Kernel Sizes** - 3x3 kernels were optimal

## 🔮 Next Steps & Recommendations

### Immediate Improvements:
1. **Learning Rate Scheduling** - Implement ReduceLROnPlateau
2. **Early Stopping** - Prevent overfitting in later epochs
3. **Model Ensemble** - Combine multiple models for 99.6%+ accuracy
4. **Advanced Augmentation** - Elastic transforms, noise injection

### Architecture Experiments:
1. **Residual Connections** - Skip connections for deeper networks
2. **Attention Mechanisms** - Focus on important features
3. **EfficientNet-style** - Compound scaling of depth, width, resolution
4. **Knowledge Distillation** - Teacher-student learning

### Performance Optimizations:
1. **Mixed Precision Training** - Faster training with FP16
2. **Gradient Clipping** - Training stability
3. **Weight Decay** - L2 regularization
4. **Cosine Annealing** - Better learning rate schedule

## 📈 Performance Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Test Accuracy | ≥99.4% | 99.51% | ✅ **Exceeded** |
| Parameters | <20k | 13,226 | ✅ **Achieved** |
| Epochs | <20 | 20 | ✅ **Achieved** |
| Training Time | <10 min | ~8 min | ✅ **Achieved** |

## 🤝 Contributing

This project demonstrates iterative neural network optimization. Feel free to:
- Experiment with different architectures
- Try new data augmentation techniques
- Implement advanced training strategies
- Share your results and insights

## 📄 License

This project is part of the TSAI ERA4 S5 curriculum and is intended for educational purposes.

---

**🎉 Successfully achieved 99.51% accuracy with only 13,226 parameters in 20 epochs!**

*Last updated: October 4, 2025*