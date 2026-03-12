"""
GPU Diagnostic Script
Check if TensorFlow and PyTorch can detect and use your GPU
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}\n")

# ============== TensorFlow GPU Check ==============
print("="*60)
print("TENSORFLOW GPU CHECK")
print("="*60)

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"\n✓ GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"  GPU {i}: {details.get('device_name', 'Unknown')}")
            except:
                print(f"  GPU {i}: {gpu.name}")
        
        # Test GPU computation
        print("\nTesting GPU computation...")
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [2.0, 2.0]])
                c = tf.matmul(a, b)
                print(f"  Matrix multiplication result: {c.numpy()}")
                print("  ✓ GPU computation successful!")
        except Exception as e:
            print(f"  ✗ GPU computation failed: {e}")
    else:
        print("\n✗ No GPUs detected by TensorFlow")
        print("\nPossible solutions:")
        print("  1. Check if NVIDIA drivers are installed")
        print("  2. Verify CUDA installation: nvcc --version")
        print("  3. Check cuDNN is in PATH")
        print("  4. Try: pip install tensorflow-cpu (then reinstall tensorflow)")
        
        # List all physical devices
        print("\nAll physical devices:")
        for device in tf.config.list_physical_devices():
            print(f"  - {device.device_type}: {device.name}")
    
    # Show TensorFlow build info
    print(f"\nTensorFlow build info:")
    print(f"  Built with CUDA: {tf.test.is_built_with_cuda()}")
    
except ImportError as e:
    print(f"✗ TensorFlow not installed: {e}")
except Exception as e:
    print(f"✗ TensorFlow error: {e}")


# ============== PyTorch GPU Check ==============
print("\n" + "="*60)
print("PYTORCH GPU CHECK")
print("="*60)

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"\n✓ CUDA is available!")
        print(f"  CUDA version (PyTorch): {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        print(f"  Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test GPU computation
        print("\nTesting GPU computation...")
        try:
            device = torch.device('cuda:0')
            x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(device)
            y = torch.tensor([[1.0, 1.0], [2.0, 2.0]]).to(device)
            z = torch.matmul(x, y)
            print(f"  Matrix multiplication result: {z.cpu().numpy()}")
            print("  ✓ GPU computation successful!")
        except Exception as e:
            print(f"  ✗ GPU computation failed: {e}")
    else:
        print("\n✗ CUDA not available in PyTorch")
        print("\nPossible solutions:")
        print("  1. Reinstall PyTorch with CUDA support:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("  2. Check NVIDIA drivers")
        print("  3. Verify CUDA installation")
        
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")
except Exception as e:
    print(f"✗ PyTorch error: {e}")


# ============== System CUDA Check ==============
print("\n" + "="*60)
print("SYSTEM CUDA CHECK")
print("="*60)

import subprocess
import os

# Check nvcc (CUDA compiler)
try:
    result = subprocess.run(['nvcc', '--version'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ CUDA compiler (nvcc) found:")
        print(f"  {result.stdout.split('\\n')[3] if len(result.stdout.split('\\n')) > 3 else result.stdout[:200]}")
    else:
        print("✗ nvcc not found in PATH")
except FileNotFoundError:
    print("✗ nvcc not found - CUDA Toolkit may not be installed")
except Exception as e:
    print(f"✗ Error checking nvcc: {e}")

# Check CUDA environment variables
print("\nCUDA environment variables:")
cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'PATH']
for var in cuda_vars:
    value = os.environ.get(var, 'Not set')
    if value != 'Not set':
        if 'cuda' in value.lower() or var == 'PATH':
            cuda_in_path = 'cuda' in value.lower() if var == 'PATH' else True
            if cuda_in_path:
                print(f"  {var}: {value[:100]}...")


# ============== Summary ==============
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

tf_gpu = False
torch_gpu = False

try:
    import tensorflow as tf
    tf_gpu = len(tf.config.list_physical_devices('GPU')) > 0
except:
    pass

try:
    import torch
    torch_gpu = torch.cuda.is_available()
except:
    pass

if tf_gpu or torch_gpu:
    print("✓ At least one framework can use GPU!")
    if tf_gpu:
        print("  - TensorFlow: GPU available")
    if torch_gpu:
        print("  - PyTorch: GPU available")
else:
    print("✗ Neither TensorFlow nor PyTorch can use GPU")
    print("\nRecommended steps:")
    print("  1. Install/Update NVIDIA GPU drivers")
    print("  2. Install CUDA Toolkit 11.8")
    print("  3. Install cuDNN 8.x")
    print("  4. Add CUDA to PATH:")
    print("     C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
    print("     C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\libnvvp")
    print("  5. Restart your computer")
    print("  6. Reinstall TensorFlow:")
    print("     pip uninstall tensorflow")
    print("     pip install tensorflow")
