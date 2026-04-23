(base) sagemaker-user@default:~/CNN_A3$ python experiments/exp01_detection_baseline.py --resume
2026-04-23 04:26:48 - exp01_detection_baseline - INFO - ================================================================================
2026-04-23 04:26:48 - exp01_detection_baseline - INFO - RESUMING EXPERIMENT: exp01_detection_baseline
2026-04-23 04:26:48 - exp01_detection_baseline - INFO - Resuming from directory: outputs/exp01_detection_baseline/run_20260423_041040
2026-04-23 04:26:48 - exp01_detection_baseline - INFO - ================================================================================
2026-04-23 04:26:48 - exp01_detection_baseline - INFO - 
[Step 1/5] Loading dataset configuration...
2026-04-23 04:26:48 - exp01_detection_baseline - INFO - Dataset config loaded from: data/processed/detection/dataset.yaml
2026-04-23 04:26:48 - exp01_detection_baseline - INFO - Dataset root: data/processed/detection
2026-04-23 04:26:48 - exp01_detection_baseline - INFO - Classes: 1 (['dog'])
2026-04-23 04:26:48 - exp01_detection_baseline - INFO - [Step 2/5] Initializing model and trainer...
2026-04-23 04:26:48 - exp01_detection_baseline - INFO - Model config: {'backbone': 'm', 'input_size': 640, 'confidence_threshold': 0.5, 'nms_iou_threshold': 0.45, 'pretrained': True}
2026-04-23 04:26:48 - exp01_detection_baseline - INFO - Training config: {'learning_rate': 0.001, 'batch_size': 16, 'epochs': 150, 'optimizer': 'adam', 'weight_decay': 0.0001, 'early_stopping_patience': 0, 'use_amp': True, 'gradient_accumulation_steps': 1, 'warmup_epochs': 10, 'scheduler': 'cosine', 'resume': True}
Using device: cuda
2026-04-23 04:26:49 - exp01_detection_baseline - INFO - 
[Step 3/5] Training model...
================================================================================
DETECTION MODEL TRAINING
================================================================================
Model config: {'backbone': 'm', 'input_size': 640, 'confidence_threshold': 0.5, 'nms_iou_threshold': 0.45, 'pretrained': True}
Training config: {'learning_rate': 0.001, 'batch_size': 16, 'epochs': 150, 'optimizer': 'adam', 'weight_decay': 0.0001, 'early_stopping_patience': 0, 'use_amp': True, 'gradient_accumulation_steps': 1, 'warmup_epochs': 10, 'scheduler': 'cosine', 'resume': True}
Output directory: outputs/exp01_detection_baseline/run_20260423_041040

  Starting fresh training...
Ultralytics 8.4.41 🚀 Python-3.12.9 torch-2.2.2+cu118 CUDA:0 (Tesla T4, 14912MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=0.5, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=data/processed/detection/dataset.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=150, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.45, keras=False, kobj=1.0, line_width=None, lr0=0.001, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8m.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=run_20260423_041040, nbs=64, nms=False, opset=None, optimize=False, optimizer=ADAM, overlap_mask=True, patience=0, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sagemaker-user/CNN_A3/outputs/exp01_detection_baseline/run_20260423_041040, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=10, warmup_momentum=0.8, weight_decay=0.0001, workers=8, workspace=None
 Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1      1392  ultralytics.nn.modules.conv.Conv             [3, 48, 3, 2]                 
  1                  -1  1     41664  ultralytics.nn.modules.conv.Conv             [48, 96, 3, 2]                
  2                  -1  2    111360  ultralytics.nn.modules.block.C2f             [96, 96, 2, True]             
  3                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  4                  -1  4    813312  ultralytics.nn.modules.block.C2f             [192, 192, 4, True]           
  5                  -1  1    664320  ultralytics.nn.modules.conv.Conv             [192, 384, 3, 2]              
  6                  -1  4   3248640  ultralytics.nn.modules.block.C2f             [384, 384, 4, True]           
  7                  -1  1   1991808  ultralytics.nn.modules.conv.Conv             [384, 576, 3, 2]              
  8                  -1  2   3985920  ultralytics.nn.modules.block.C2f             [576, 576, 2, True]           
  9                  -1  1    831168  ultralytics.nn.modules.block.SPPF            [576, 576, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  2   1993728  ultralytics.nn.modules.block.C2f             [960, 384, 2]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  2    517632  ultralytics.nn.modules.block.C2f             [576, 192, 2]                 
 16                  -1  1    332160  ultralytics.nn.modules.conv.Conv             [192, 192, 3, 2]              
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  2   1846272  ultralytics.nn.modules.block.C2f             [576, 384, 2]                 
 19                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  2   4207104  ultralytics.nn.modules.block.C2f             [960, 576, 2]                 
 22        [15, 18, 21]  1   3776275  ultralytics.nn.modules.head.Detect           [1, 16, None, [192, 384, 576]]
Model summary: 170 layers, 25,856,899 parameters, 25,856,883 gradients, 79.1 GFLOPs

Transferred 469/475 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2381.0±944.9 MB/s, size: 93.4 KB)
train: Scanning /home/sagemaker-user/CNN_A3/data/processed/detection/labels/train.cache... 2461 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 2461/2461 491.5Mit/s 0.0s
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.2±0.3 ms, read: 1622.5±1218.1 MB/s, size: 104.4 KB)
val: Scanning /home/sagemaker-user/CNN_A3/data/processed/detection/labels/val.cache... 2462 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 2462/2462 134.1Mit/s 0.0s
optimizer: Adam(lr=0.001, momentum=0.937) with parameter groups 77 weight(decay=0.0), 84 weight(decay=0.0001), 83 bias(decay=0.0)
Plotting labels to /home/sagemaker-user/CNN_A3/outputs/exp01_detection_baseline/run_20260423_041040/labels.jpg... 
2026/04/23 04:26:57 WARNING mlflow.utils.autologging_utils: MLflow sklearn autologging is known to be compatible with 0.24.1 <= scikit-learn <= 1.6.1, but the installed version is 1.7.2. If you encounter errors during autologging, try upgrading / downgrading scikit-learn to a compatible version, or try upgrading MLflow.
2026/04/23 04:26:57 INFO mlflow.tracking.fluent: Autologging successfully enabled for sklearn.
2026/04/23 04:26:57 WARNING mlflow.utils.autologging_utils: MLflow transformers autologging is known to be compatible with 4.35.2 <= transformers <= 4.51.2, but the installed version is 4.57.6. If you encounter errors during autologging, try upgrading / downgrading transformers to a compatible version, or try upgrading MLflow.
2026/04/23 04:26:58 INFO mlflow.bedrock: Enabled auto-tracing for Bedrock. Note that MLflow can only trace boto3 service clients that are created after this call. If you have already created one, please recreate the client by calling `boto3.client`.
2026/04/23 04:26:58 INFO mlflow.tracking.fluent: Autologging successfully enabled for boto3.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1776918418.612577    6093 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1776918418.618283    6093 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2026/04/23 04:27:01 WARNING mlflow.utils.autologging_utils: MLflow keras autologging is known to be compatible with 3.0.2 <= keras <= 3.9.2, but the installed version is 3.13.2. If you encounter errors during autologging, try upgrading / downgrading keras to a compatible version, or try upgrading MLflow.
2026/04/23 04:27:01 INFO mlflow.tracking.fluent: Autologging successfully enabled for keras.
2026/04/23 04:27:01 INFO mlflow.tracking.fluent: Autologging successfully enabled for tensorflow.
2026/04/23 04:27:03 INFO mlflow.tracking.fluent: Autologging successfully enabled for transformers.
MLflow: logging run_id(e304451f65d948139d57e24c94143d78) to runs/mlflow
MLflow: view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri runs/mlflow'
MLflow: disable with 'yolo settings mlflow=False'
Image sizes 640 train, 640 val
Using 4 dataloader workers
Logging results to /home/sagemaker-user/CNN_A3/outputs/exp01_detection_baseline/run_20260423_041040
Starting training for 150 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/150      6.37G      1.313      1.365      1.567         35        640: 39% ━━━━╸─────── 60/154 2.1it/s 29.3s<44.8sCorrupt JPEG data: premature end of data segment
      1/150      6.37G      1.269      1.165       1.52         39        640: 69% ━━━━━━━━──── 106/154 2.1it/s 51.4s<23.0sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
      1/150      6.37G      1.264       1.11       1.52         43        640: 86% ━━━━━━━━━━── 133/154 2.1it/s 1:04<10.1s
      1/150      6.37G      1.264      1.109      1.519         42        640: 87% ━━━━━━━━━━── 134/154 2.1it/s 1:05<9.6s
      1/150      6.37G      1.262      1.106      1.519         40        640: 88% ━━━━━━━━━━╸─ 135/154 2.1it/s 1:05<9.1s
      1/150      6.37G      1.248       1.08      1.509         30        640: 100% ━━━━━━━━━━━━ 154/154 2.1it/s 1:14
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.1it/s 24.5s
                   all       2462       2541      0.904      0.402      0.392      0.226

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/150      6.56G       1.19     0.8888      1.511         31        640: 38% ━━━━╸─────── 58/154 1.9it/s 28.6s<50.1s


      2/150      6.57G      1.216     0.8956      1.503         45        640: 86% ━━━━━━━━━━── 133/154 2.0it/s 1:06<10.8sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment
      2/150      6.57G       1.22     0.9006       1.51         26        640: 100% ━━━━━━━━━━━━ 154/154 2.0it/s 1:16
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.5it/s 22.3s
                   all       2462       2541      0.952      0.689      0.678      0.437