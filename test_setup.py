"""
Quick Setup Validation Script
Run this after installation to verify everything works correctly.

Usage:
    python test_setup.py
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("=" * 60)
    print("🐍 Python Version Check")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Current Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 9 <= version.minor <= 11:
        print("✅ Python version is compatible (3.9-3.11)")
        return True
    else:
        print("⚠️  Warning: Python 3.9-3.11 recommended")
        print("   Python 3.12+ may have compatibility issues")
        return True  # Still continue


def check_dependencies():
    """Check if all required packages are installed"""
    print("\n" + "=" * 60)
    print("📦 Dependencies Check")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'Torchvision',
        'ultralytics': 'YOLOv8 (Ultralytics)',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'albumentations': 'Albumentations',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed")
        return True


def check_pytorch_device():
    """Check PyTorch device availability"""
    print("\n" + "=" * 60)
    print("🔥 PyTorch Device Check")
    print("=" * 60)
    
    import torch
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print("\n✅ GPU mode ready for training")
    else:
        print("\nℹ️  CPU mode only")
        print("   Training will be slower but functional")
        print("   For GPU training, use AWS SageMaker")
    
    return True


def check_model_imports():
    """Test importing project modules"""
    print("\n" + "=" * 60)
    print("🏗️  Project Modules Check")
    print("=" * 60)
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        
        from models.detection_model import YOLOv8Detector
        print("✅ YOLOv8Detector")
        
        from models.classification_model import ResNet50Classifier
        print("✅ ResNet50Classifier")
        
        from training.classification_trainer import ClassificationTrainer
        print("✅ ClassificationTrainer")
        
        print("\n✅ All project modules accessible")
        return True
        
    except Exception as e:
        print(f"\n❌ Module import failed: {e}")
        return False


def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("🧪 Visual Dog Emotion Recognition - Setup Validation")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run checks
    results.append(check_python_version())
    results.append(check_dependencies())
    results.append(check_pytorch_device())
    results.append(check_model_imports())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    if all(results):
        print("\n🎉 SUCCESS! Your environment is ready.")
        print("\nNext steps:")
        print("1. Configure Kaggle API credentials")
        print("2. Run your first experiment:")
        print("   python experiments/exp04_classification_baseline.py")
        print("\nSee QUICKSTART.md for detailed instructions.\n")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("See QUICKSTART.md troubleshooting section.\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
