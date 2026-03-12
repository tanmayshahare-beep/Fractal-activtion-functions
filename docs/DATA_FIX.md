# PyTorch Data Preprocessing Fix

## The Problem: Double Scaling

The most common cause of poor PyTorch performance compared to TensorFlow is **incorrect data preprocessing**.

### What Was Wrong

```python
# WRONG - Double scaling!
x_train = x_train.astype('float32') / 255.0  # First scaling: 0-255 -> 0-1
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),  # Second scaling: treats 0-1 as 0-255, scales to 0-0.0039!
])
```

**Result:** Input values become ~0.004 instead of ~1.0, causing:
- Tiny gradients
- Slow learning
- Poor accuracy (~47% instead of >98%)

### The Fix

```python
# CORRECT - Let ToTensor() handle scaling
x_train = x_train.astype('float32')  # Keep 0-255 range
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy to PIL (keeps 0-255)
    transforms.ToTensor(),    # Scale 0-255 -> 0-1
])
```

**Result:** Input values are correctly in [0, 1] range.

## How ToTensor() Works

`transforms.ToTensor()` expects:
- **Input:** PIL Image with values 0-255 (uint8)
- **Output:** Tensor with values 0-1 (float32)

If you pass a PIL Image with values already in 0-1 range, ToTensor() will still convert it to tensor but won't scale (since it expects uint8). However, the conversion path matters.

## Files Fixed

- ✅ `experiments/mnist_efta.py` - Removed `/255.0` before transform
- ✅ `experiments/fashion_mnist_efta.py` - Removed `/255.0` before transform

## Verification

Add this check before training:

```python
sample_images, sample_labels = next(iter(train_loader))
print(f"Batch shape: {sample_images.shape}")   # Should be [128, 1, 28, 28]
print(f"Image min: {sample_images.min():.3f}, max: {sample_images.max():.3f}")  # Should be 0.0 and 1.0
```

Expected output:
```
Batch shape: torch.Size([128, 1, 28, 28])
Image min: 0.000, max: 1.000
```

## Expected Results After Fix

| Dataset | Previous (Broken) | After Fix |
|---------|------------------|-----------|
| MNIST | ~47% after 7 epochs | >95% after 5 epochs |
| Fashion-MNIST | ~40% after 7 epochs | >85% after 5 epochs |

## Other Potential Issues (If Still Poor Performance)

1. **Model has softmax at end** - Remove it! CrossEntropyLoss expects raw logits
2. **Wrong learning rate** - Should be 0.001 for Adam
3. **Custom activation bugs** - Test FTA/EFTA layers independently
4. **Missing model.train()** - Required for BatchNorm and Dropout
5. **Not loading best model** - Ensure best weights are restored after training
