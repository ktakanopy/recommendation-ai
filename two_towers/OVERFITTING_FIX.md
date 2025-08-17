# Overfitting Fix: Recommended Settings

## 🚨 **Current Problem**
Your model is overfitting due to weak negative sampling and settings optimized for tiny datasets.

## 🎯 **Recommended Settings**

### **Option 1: Strong Negative Sampling (Recommended)**
```bash
# Use sampled softmax with many negatives
ML_loss_type=sampled_softmax
ML_num_sampled_negatives=5000
ML_sample_users=500          # More data
ML_num_epochs=5              # Fewer epochs
ML_layer_sizes="[64, 32]"    # Simpler model
ML_learning_rate=0.001
ML_batch_size=128
```

### **Option 2: Hard Negative Mining**
```bash
# Use hard negatives for better learning
ML_loss_type=explicit_negatives
ML_num_negatives=20          # Much more negatives
ML_use_hard_negatives=true
ML_hard_negative_ratio=0.7
ML_sample_users=500
ML_num_epochs=5
ML_layer_sizes="[64, 32]"
```

### **Option 3: Conservative In-Batch**
```bash
# Go back to in-batch but with better settings
ML_loss_type=in_batch
ML_temperature=0.05          # Lower temperature for harder learning
ML_sample_users=1000         # More data
ML_num_epochs=3              # Very few epochs
ML_layer_sizes="[32]"        # Much simpler model
ML_batch_size=256            # Larger batches = more negatives
```

## 🔧 **Quick Fixes You Can Try Now**

### **Immediate Fix (Environment Variables)**
```bash
# Test with sampled softmax (should work best)
cd two_towers/
ML_loss_type=sampled_softmax ML_num_sampled_negatives=3000 ML_sample_users=500 ML_num_epochs=3 poetry run python scripts/model_training.py
```

### **Edit Settings File**
Update `two_towers/infrastructure/config/settings.py`:

```python
# Replace the problematic settings
sample_users: int | None = 500          # Was: 50
num_epochs: int = 5                     # Was: 10  
layer_sizes: List[int] = [64, 32]       # Was: [32, 32, 32]
loss_type: str = "sampled_softmax"      # Was: "explicit_negatives"
num_negatives: int = 20                 # Was: 4
num_sampled_negatives: int = 3000       # Was: 1000
batch_size: int = 128                   # Was: 64
```

## 📊 **Expected Results**

With these fixes, you should see:
- **Higher precision**: 6-12% (vs current 3.7%)
- **Better generalization**: Val loss closer to train loss
- **Faster convergence**: Clear improvements in 2-3 epochs
- **No overfitting**: Stable validation metrics

## 🎯 **Why These Work**

1. **More negatives** = harder learning task = better discrimination
2. **More data** = less prone to overfitting
3. **Simpler model** = fewer parameters to overfit
4. **Fewer epochs** = early stopping before overfitting
5. **Sampled softmax** = most robust negative sampling strategy

Try Option 1 first - it should give you the best results!