((.venv) ) sagemaker-user@default:~/CNN_A3$ python experiments/exp06_classification_GoogLeNet.py 
2026-04-26 02:29:01 - exp06_classification_GoogLeNet - INFO - ================================================================================
2026-04-26 02:29:01 - exp06_classification_GoogLeNet - INFO - STARTING EXPERIMENT: exp06_classification_GoogLeNet
2026-04-26 02:29:01 - exp06_classification_GoogLeNet - INFO - Model: GoogLeNet/Inception v1 (with auxiliary classifiers)
2026-04-26 02:29:01 - exp06_classification_GoogLeNet - INFO - ================================================================================
2026-04-26 02:29:01 - exp06_classification_GoogLeNet - INFO - 
[Step 1/4] Loading preprocessed data...
2026-04-26 02:29:45 - exp06_classification_GoogLeNet - INFO - Data loaded successfully:
2026-04-26 02:29:45 - exp06_classification_GoogLeNet - INFO -   Train: 6527 samples, shape: (224, 224, 3)
2026-04-26 02:29:45 - exp06_classification_GoogLeNet - INFO -   Valid: 1865 samples, shape: (224, 224, 3)
2026-04-26 02:29:45 - exp06_classification_GoogLeNet - INFO -   Test: 933 samples, shape: (224, 224, 3)
2026-04-26 02:29:45 - exp06_classification_GoogLeNet - INFO - 
[Step 2/4] Initializing model and trainer...
2026-04-26 02:29:45 - exp06_classification_GoogLeNet - INFO - Model config: {'num_classes': 5, 'dropout_rate': 0.5, 'pretrained': True, 'freeze_backbone': True, 'use_auxiliary': True}
2026-04-26 02:29:45 - exp06_classification_GoogLeNet - INFO - Training config: {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 120, 'optimizer': 'adam', 'weight_decay': 0.0001, 'early_stopping_patience': 15, 'use_amp': True, 'gradient_accumulation_steps': 1, 'label_smoothing': 0.1, 'class_weighting': True}
Experiment directory created: outputs/exp06_classification_GoogLeNet/run_20260426_022945
2026-04-26 02:29:45 - exp06_classification_GoogLeNet - INFO - Total parameters: 10,440,599
2026-04-26 02:29:45 - exp06_classification_GoogLeNet - INFO - Trainable parameters: 3,815,695
Using device: cuda
2026-04-26 02:29:45 - exp06_classification_GoogLeNet - INFO - 
[Step 3/4] Training model...
================================================================================
CLASSIFICATION MODEL TRAINING
================================================================================
Model config: {'num_classes': 5, 'dropout_rate': 0.5, 'pretrained': True, 'freeze_backbone': True, 'use_auxiliary': True}
Training config: {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 120, 'optimizer': 'adam', 'weight_decay': 0.0001, 'early_stopping_patience': 15, 'use_amp': True, 'gradient_accumulation_steps': 1, 'label_smoothing': 0.1, 'class_weighting': True}
Output directory: outputs/exp06_classification_GoogLeNet/run_20260426_022945
Class weights: [    0.99954      1.0003     0.99954      1.0003      1.0003]

================================================================================
PHASE 1: Training with frozen backbone
================================================================================
Training Epoch 1:   0%|                                                                                                                                | 0/203 [00:00<?, ?it/s]/home/sagemaker-user/CNN_A3/.venv/lib/python3.12/site-packages/torch/nn/modules/conv.py:456: UserWarning: Applied workaround for CuDNN issue, install nvrtc.so (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:80.)
  return F.conv2d(input, weight, bias, self.stride,
Training Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.17it/s, loss=2.4734, acc=31.63%]
[Phase 1 (Frozen)] Epoch 1/10 | Train Loss: 2.4734 | Train Acc: 0.3163 | Val Loss: 1.4583 | Val Acc: 0.3925
  ✓ New best model saved (Val Acc: 0.3925)
Training Epoch 2: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.38it/s, loss=2.3562, acc=35.10%]
[Phase 1 (Frozen)] Epoch 2/10 | Train Loss: 2.3562 | Train Acc: 0.3510 | Val Loss: 1.4283 | Val Acc: 0.4236
  ✓ New best model saved (Val Acc: 0.4236)
Training Epoch 3: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:22<00:00,  8.91it/s, loss=2.3144, acc=36.01%]
[Phase 1 (Frozen)] Epoch 3/10 | Train Loss: 2.3144 | Train Acc: 0.3601 | Val Loss: 1.4160 | Val Acc: 0.4273
  ✓ New best model saved (Val Acc: 0.4273)
Training Epoch 4: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:22<00:00,  8.85it/s, loss=2.2921, acc=37.39%]
[Phase 1 (Frozen)] Epoch 4/10 | Train Loss: 2.2921 | Train Acc: 0.3739 | Val Loss: 1.4202 | Val Acc: 0.4075
Training Epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:22<00:00,  8.97it/s, loss=2.2789, acc=36.28%]
[Phase 1 (Frozen)] Epoch 5/10 | Train Loss: 2.2789 | Train Acc: 0.3628 | Val Loss: 1.4284 | Val Acc: 0.4306
  ✓ New best model saved (Val Acc: 0.4306)
Training Epoch 6: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:22<00:00,  8.91it/s, loss=2.2625, acc=36.73%]
[Phase 1 (Frozen)] Epoch 6/10 | Train Loss: 2.2625 | Train Acc: 0.3673 | Val Loss: 1.4366 | Val Acc: 0.4268
Training Epoch 7: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.58it/s, loss=2.2400, acc=37.79%]
[Phase 1 (Frozen)] Epoch 7/10 | Train Loss: 2.2400 | Train Acc: 0.3779 | Val Loss: 1.4032 | Val Acc: 0.4370
  ✓ New best model saved (Val Acc: 0.4370)
Training Epoch 8: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.68it/s, loss=2.2502, acc=37.42%]
[Phase 1 (Frozen)] Epoch 8/10 | Train Loss: 2.2502 | Train Acc: 0.3742 | Val Loss: 1.4175 | Val Acc: 0.4322
Training Epoch 9: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:22<00:00,  9.04it/s, loss=2.2322, acc=37.48%]
[Phase 1 (Frozen)] Epoch 9/10 | Train Loss: 2.2322 | Train Acc: 0.3748 | Val Loss: 1.4036 | Val Acc: 0.4456
  ✓ New best model saved (Val Acc: 0.4456)
Training Epoch 10: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.63it/s, loss=2.2221, acc=37.85%]
[Phase 1 (Frozen)] Epoch 10/10 | Train Loss: 2.2221 | Train Acc: 0.3785 | Val Loss: 1.4327 | Val Acc: 0.4097

================================================================================
PHASE 2: Fine-tuning with unfrozen backbone
================================================================================
Training Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.24it/s, loss=2.1823, acc=40.41%]
[Phase 2 (Fine-tune)] Epoch 1/110 | Train Loss: 2.1823 | Train Acc: 0.4041 | Val Loss: 1.3481 | Val Acc: 0.4853
  ✓ New best model saved (Val Acc: 0.4853)
Training Epoch 2: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.35it/s, loss=2.1320, acc=43.21%]
[Phase 2 (Fine-tune)] Epoch 2/110 | Train Loss: 2.1320 | Train Acc: 0.4321 | Val Loss: 1.3154 | Val Acc: 0.4965
  ✓ New best model saved (Val Acc: 0.4965)
Training Epoch 3: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.39it/s, loss=2.0884, acc=46.51%]
[Phase 2 (Fine-tune)] Epoch 3/110 | Train Loss: 2.0884 | Train Acc: 0.4651 | Val Loss: 1.2891 | Val Acc: 0.5083
  ✓ New best model saved (Val Acc: 0.5083)
Training Epoch 4: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.21it/s, loss=2.0686, acc=47.48%]
[Phase 2 (Fine-tune)] Epoch 4/110 | Train Loss: 2.0686 | Train Acc: 0.4748 | Val Loss: 1.2644 | Val Acc: 0.5169
  ✓ New best model saved (Val Acc: 0.5169)
Training Epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.27it/s, loss=2.0302, acc=49.85%]
[Phase 2 (Fine-tune)] Epoch 5/110 | Train Loss: 2.0302 | Train Acc: 0.4985 | Val Loss: 1.2452 | Val Acc: 0.5303
  ✓ New best model saved (Val Acc: 0.5303)
Training Epoch 6: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.16it/s, loss=2.0122, acc=49.83%]
[Phase 2 (Fine-tune)] Epoch 6/110 | Train Loss: 2.0122 | Train Acc: 0.4983 | Val Loss: 1.2265 | Val Acc: 0.5458
  ✓ New best model saved (Val Acc: 0.5458)
Training Epoch 7: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.64it/s, loss=1.9866, acc=51.72%]
[Phase 2 (Fine-tune)] Epoch 7/110 | Train Loss: 1.9866 | Train Acc: 0.5172 | Val Loss: 1.2063 | Val Acc: 0.5630
  ✓ New best model saved (Val Acc: 0.5630)
Training Epoch 8: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.36it/s, loss=1.9662, acc=53.34%]
[Phase 2 (Fine-tune)] Epoch 8/110 | Train Loss: 1.9662 | Train Acc: 0.5334 | Val Loss: 1.1907 | Val Acc: 0.5668
  ✓ New best model saved (Val Acc: 0.5668)
Training Epoch 9: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.69it/s, loss=1.9447, acc=53.53%]
[Phase 2 (Fine-tune)] Epoch 9/110 | Train Loss: 1.9447 | Train Acc: 0.5353 | Val Loss: 1.1836 | Val Acc: 0.5710
  ✓ New best model saved (Val Acc: 0.5710)
Training Epoch 10: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.63it/s, loss=1.9290, acc=54.88%]
[Phase 2 (Fine-tune)] Epoch 10/110 | Train Loss: 1.9290 | Train Acc: 0.5488 | Val Loss: 1.1734 | Val Acc: 0.5668
Training Epoch 11: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.35it/s, loss=1.9054, acc=55.85%]
[Phase 2 (Fine-tune)] Epoch 11/110 | Train Loss: 1.9054 | Train Acc: 0.5585 | Val Loss: 1.1645 | Val Acc: 0.5727
  ✓ New best model saved (Val Acc: 0.5727)
Training Epoch 12: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.30it/s, loss=1.9006, acc=55.82%]
[Phase 2 (Fine-tune)] Epoch 12/110 | Train Loss: 1.9006 | Train Acc: 0.5582 | Val Loss: 1.1536 | Val Acc: 0.5861
  ✓ New best model saved (Val Acc: 0.5861)
Training Epoch 13: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.36it/s, loss=1.8770, acc=57.07%]
[Phase 2 (Fine-tune)] Epoch 13/110 | Train Loss: 1.8770 | Train Acc: 0.5707 | Val Loss: 1.1448 | Val Acc: 0.5893
  ✓ New best model saved (Val Acc: 0.5893)
Training Epoch 14: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.42it/s, loss=1.8607, acc=58.25%]
[Phase 2 (Fine-tune)] Epoch 14/110 | Train Loss: 1.8607 | Train Acc: 0.5825 | Val Loss: 1.1338 | Val Acc: 0.5893
Training Epoch 15: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.39it/s, loss=1.8453, acc=57.62%]
[Phase 2 (Fine-tune)] Epoch 15/110 | Train Loss: 1.8453 | Train Acc: 0.5762 | Val Loss: 1.1241 | Val Acc: 0.5989
  ✓ New best model saved (Val Acc: 0.5989)
Training Epoch 16: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.56it/s, loss=1.8322, acc=59.22%]
[Phase 2 (Fine-tune)] Epoch 16/110 | Train Loss: 1.8322 | Train Acc: 0.5922 | Val Loss: 1.1160 | Val Acc: 0.5973
Training Epoch 17: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.57it/s, loss=1.8122, acc=60.73%]
[Phase 2 (Fine-tune)] Epoch 17/110 | Train Loss: 1.8122 | Train Acc: 0.6073 | Val Loss: 1.1097 | Val Acc: 0.6005
  ✓ New best model saved (Val Acc: 0.6005)
Training Epoch 18: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.55it/s, loss=1.8113, acc=60.45%]
[Phase 2 (Fine-tune)] Epoch 18/110 | Train Loss: 1.8113 | Train Acc: 0.6045 | Val Loss: 1.1032 | Val Acc: 0.6059
  ✓ New best model saved (Val Acc: 0.6059)
Training Epoch 19: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.40it/s, loss=1.8044, acc=60.71%]
[Phase 2 (Fine-tune)] Epoch 19/110 | Train Loss: 1.8044 | Train Acc: 0.6071 | Val Loss: 1.0989 | Val Acc: 0.6113
  ✓ New best model saved (Val Acc: 0.6113)
Training Epoch 20: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.51it/s, loss=1.7728, acc=62.15%]
[Phase 2 (Fine-tune)] Epoch 20/110 | Train Loss: 1.7728 | Train Acc: 0.6215 | Val Loss: 1.0916 | Val Acc: 0.6172
  ✓ New best model saved (Val Acc: 0.6172)
Training Epoch 21: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.42it/s, loss=1.7760, acc=61.64%]
  Learning rate reduced to 0.000070
[Phase 2 (Fine-tune)] Epoch 21/110 | Train Loss: 1.7760 | Train Acc: 0.6164 | Val Loss: 1.0899 | Val Acc: 0.6155
Training Epoch 22: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.16it/s, loss=1.7560, acc=62.55%]
[Phase 2 (Fine-tune)] Epoch 22/110 | Train Loss: 1.7560 | Train Acc: 0.6255 | Val Loss: 1.0830 | Val Acc: 0.6225
  ✓ New best model saved (Val Acc: 0.6225)
Training Epoch 23: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.24it/s, loss=1.7625, acc=62.45%]
[Phase 2 (Fine-tune)] Epoch 23/110 | Train Loss: 1.7625 | Train Acc: 0.6245 | Val Loss: 1.0829 | Val Acc: 0.6172
Training Epoch 24: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:25<00:00,  8.04it/s, loss=1.7475, acc=62.90%]
[Phase 2 (Fine-tune)] Epoch 24/110 | Train Loss: 1.7475 | Train Acc: 0.6290 | Val Loss: 1.0820 | Val Acc: 0.6198
Training Epoch 25: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.24it/s, loss=1.7470, acc=62.39%]
[Phase 2 (Fine-tune)] Epoch 25/110 | Train Loss: 1.7470 | Train Acc: 0.6239 | Val Loss: 1.0792 | Val Acc: 0.6182
Training Epoch 26: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.52it/s, loss=1.7162, acc=64.84%]
[Phase 2 (Fine-tune)] Epoch 26/110 | Train Loss: 1.7162 | Train Acc: 0.6484 | Val Loss: 1.0745 | Val Acc: 0.6279
  ✓ New best model saved (Val Acc: 0.6279)
Training Epoch 27: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.31it/s, loss=1.7160, acc=65.09%]
[Phase 2 (Fine-tune)] Epoch 27/110 | Train Loss: 1.7160 | Train Acc: 0.6509 | Val Loss: 1.0699 | Val Acc: 0.6327
  ✓ New best model saved (Val Acc: 0.6327)
Training Epoch 28: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.40it/s, loss=1.7220, acc=64.27%]
[Phase 2 (Fine-tune)] Epoch 28/110 | Train Loss: 1.7220 | Train Acc: 0.6427 | Val Loss: 1.0675 | Val Acc: 0.6284
Training Epoch 29: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.34it/s, loss=1.7074, acc=65.06%]
[Phase 2 (Fine-tune)] Epoch 29/110 | Train Loss: 1.7074 | Train Acc: 0.6506 | Val Loss: 1.0647 | Val Acc: 0.6338
  ✓ New best model saved (Val Acc: 0.6338)
Training Epoch 30: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.47it/s, loss=1.6986, acc=65.13%]
[Phase 2 (Fine-tune)] Epoch 30/110 | Train Loss: 1.6986 | Train Acc: 0.6513 | Val Loss: 1.0637 | Val Acc: 0.6343
  ✓ New best model saved (Val Acc: 0.6343)
Training Epoch 31: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.51it/s, loss=1.6860, acc=66.33%]
[Phase 2 (Fine-tune)] Epoch 31/110 | Train Loss: 1.6860 | Train Acc: 0.6633 | Val Loss: 1.0627 | Val Acc: 0.6322
Training Epoch 32: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.26it/s, loss=1.6776, acc=66.58%]
[Phase 2 (Fine-tune)] Epoch 32/110 | Train Loss: 1.6776 | Train Acc: 0.6658 | Val Loss: 1.0612 | Val Acc: 0.6354
  ✓ New best model saved (Val Acc: 0.6354)
Training Epoch 33: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.59it/s, loss=1.6704, acc=67.20%]
[Phase 2 (Fine-tune)] Epoch 33/110 | Train Loss: 1.6704 | Train Acc: 0.6720 | Val Loss: 1.0568 | Val Acc: 0.6354
Training Epoch 34: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.35it/s, loss=1.6786, acc=66.27%]
[Phase 2 (Fine-tune)] Epoch 34/110 | Train Loss: 1.6786 | Train Acc: 0.6627 | Val Loss: 1.0533 | Val Acc: 0.6386
  ✓ New best model saved (Val Acc: 0.6386)
Training Epoch 35: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.57it/s, loss=1.6493, acc=67.36%]
[Phase 2 (Fine-tune)] Epoch 35/110 | Train Loss: 1.6493 | Train Acc: 0.6736 | Val Loss: 1.0579 | Val Acc: 0.6354
Training Epoch 36: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.43it/s, loss=1.6345, acc=69.01%]
[Phase 2 (Fine-tune)] Epoch 36/110 | Train Loss: 1.6345 | Train Acc: 0.6901 | Val Loss: 1.0615 | Val Acc: 0.6402
  ✓ New best model saved (Val Acc: 0.6402)
Training Epoch 37: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.51it/s, loss=1.6411, acc=67.64%]
[Phase 2 (Fine-tune)] Epoch 37/110 | Train Loss: 1.6411 | Train Acc: 0.6764 | Val Loss: 1.0528 | Val Acc: 0.6413
  ✓ New best model saved (Val Acc: 0.6413)
Training Epoch 38: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.50it/s, loss=1.6500, acc=66.92%]
[Phase 2 (Fine-tune)] Epoch 38/110 | Train Loss: 1.6500 | Train Acc: 0.6692 | Val Loss: 1.0586 | Val Acc: 0.6375
Training Epoch 39: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.60it/s, loss=1.6252, acc=68.63%]
[Phase 2 (Fine-tune)] Epoch 39/110 | Train Loss: 1.6252 | Train Acc: 0.6863 | Val Loss: 1.0494 | Val Acc: 0.6434
  ✓ New best model saved (Val Acc: 0.6434)
Training Epoch 40: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.47it/s, loss=1.6263, acc=68.84%]
[Phase 2 (Fine-tune)] Epoch 40/110 | Train Loss: 1.6263 | Train Acc: 0.6884 | Val Loss: 1.0526 | Val Acc: 0.6434
Training Epoch 41: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.23it/s, loss=1.6249, acc=68.92%]
  Learning rate reduced to 0.000049
[Phase 2 (Fine-tune)] Epoch 41/110 | Train Loss: 1.6249 | Train Acc: 0.6892 | Val Loss: 1.0514 | Val Acc: 0.6408
Training Epoch 42: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.44it/s, loss=1.6116, acc=70.09%]
[Phase 2 (Fine-tune)] Epoch 42/110 | Train Loss: 1.6116 | Train Acc: 0.7009 | Val Loss: 1.0509 | Val Acc: 0.6375
Training Epoch 43: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.46it/s, loss=1.6127, acc=69.69%]
[Phase 2 (Fine-tune)] Epoch 43/110 | Train Loss: 1.6127 | Train Acc: 0.6969 | Val Loss: 1.0519 | Val Acc: 0.6418
Training Epoch 44: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.37it/s, loss=1.5991, acc=70.50%]
[Phase 2 (Fine-tune)] Epoch 44/110 | Train Loss: 1.5991 | Train Acc: 0.7050 | Val Loss: 1.0508 | Val Acc: 0.6418
Training Epoch 45: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.20it/s, loss=1.6025, acc=69.78%]
[Phase 2 (Fine-tune)] Epoch 45/110 | Train Loss: 1.6025 | Train Acc: 0.6978 | Val Loss: 1.0518 | Val Acc: 0.6477
  ✓ New best model saved (Val Acc: 0.6477)
Training Epoch 46: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.18it/s, loss=1.5971, acc=70.14%]
[Phase 2 (Fine-tune)] Epoch 46/110 | Train Loss: 1.5971 | Train Acc: 0.7014 | Val Loss: 1.0510 | Val Acc: 0.6472
Training Epoch 47: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.47it/s, loss=1.5916, acc=70.44%]
[Phase 2 (Fine-tune)] Epoch 47/110 | Train Loss: 1.5916 | Train Acc: 0.7044 | Val Loss: 1.0488 | Val Acc: 0.6450
Training Epoch 48: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.17it/s, loss=1.5935, acc=70.50%]
[Phase 2 (Fine-tune)] Epoch 48/110 | Train Loss: 1.5935 | Train Acc: 0.7050 | Val Loss: 1.0503 | Val Acc: 0.6440
Training Epoch 49: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.49it/s, loss=1.5903, acc=70.21%]
[Phase 2 (Fine-tune)] Epoch 49/110 | Train Loss: 1.5903 | Train Acc: 0.7021 | Val Loss: 1.0502 | Val Acc: 0.6456
Training Epoch 50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.60it/s, loss=1.5777, acc=71.37%]
[Phase 2 (Fine-tune)] Epoch 50/110 | Train Loss: 1.5777 | Train Acc: 0.7137 | Val Loss: 1.0558 | Val Acc: 0.6461
Training Epoch 51: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.26it/s, loss=1.5647, acc=71.98%]
[Phase 2 (Fine-tune)] Epoch 51/110 | Train Loss: 1.5647 | Train Acc: 0.7198 | Val Loss: 1.0526 | Val Acc: 0.6477
Training Epoch 52: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.59it/s, loss=1.5738, acc=71.18%]
[Phase 2 (Fine-tune)] Epoch 52/110 | Train Loss: 1.5738 | Train Acc: 0.7118 | Val Loss: 1.0514 | Val Acc: 0.6408
Training Epoch 53: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.50it/s, loss=1.5568, acc=72.04%]
[Phase 2 (Fine-tune)] Epoch 53/110 | Train Loss: 1.5568 | Train Acc: 0.7204 | Val Loss: 1.0550 | Val Acc: 0.6483
  ✓ New best model saved (Val Acc: 0.6483)
Training Epoch 54: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.47it/s, loss=1.5690, acc=71.04%]
[Phase 2 (Fine-tune)] Epoch 54/110 | Train Loss: 1.5690 | Train Acc: 0.7104 | Val Loss: 1.0533 | Val Acc: 0.6456
Training Epoch 55: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:25<00:00,  7.94it/s, loss=1.5738, acc=70.83%]
[Phase 2 (Fine-tune)] Epoch 55/110 | Train Loss: 1.5738 | Train Acc: 0.7083 | Val Loss: 1.0499 | Val Acc: 0.6466
Training Epoch 56: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.14it/s, loss=1.5568, acc=72.28%]
[Phase 2 (Fine-tune)] Epoch 56/110 | Train Loss: 1.5568 | Train Acc: 0.7228 | Val Loss: 1.0498 | Val Acc: 0.6450
Training Epoch 57: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.51it/s, loss=1.5407, acc=72.86%]
[Phase 2 (Fine-tune)] Epoch 57/110 | Train Loss: 1.5407 | Train Acc: 0.7286 | Val Loss: 1.0529 | Val Acc: 0.6472
Training Epoch 58: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.37it/s, loss=1.5501, acc=72.44%]
[Phase 2 (Fine-tune)] Epoch 58/110 | Train Loss: 1.5501 | Train Acc: 0.7244 | Val Loss: 1.0523 | Val Acc: 0.6450
Training Epoch 59: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.68it/s, loss=1.5430, acc=73.17%]
[Phase 2 (Fine-tune)] Epoch 59/110 | Train Loss: 1.5430 | Train Acc: 0.7317 | Val Loss: 1.0501 | Val Acc: 0.6461
Training Epoch 60: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.30it/s, loss=1.5459, acc=72.55%]
[Phase 2 (Fine-tune)] Epoch 60/110 | Train Loss: 1.5459 | Train Acc: 0.7255 | Val Loss: 1.0506 | Val Acc: 0.6477
Training Epoch 61: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.52it/s, loss=1.5391, acc=72.72%]
  Learning rate reduced to 0.000034
[Phase 2 (Fine-tune)] Epoch 61/110 | Train Loss: 1.5391 | Train Acc: 0.7272 | Val Loss: 1.0509 | Val Acc: 0.6477
Training Epoch 62: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.50it/s, loss=1.5338, acc=73.17%]
[Phase 2 (Fine-tune)] Epoch 62/110 | Train Loss: 1.5338 | Train Acc: 0.7317 | Val Loss: 1.0531 | Val Acc: 0.6509
  ✓ New best model saved (Val Acc: 0.6509)
Training Epoch 63: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.29it/s, loss=1.5171, acc=74.25%]
[Phase 2 (Fine-tune)] Epoch 63/110 | Train Loss: 1.5171 | Train Acc: 0.7425 | Val Loss: 1.0568 | Val Acc: 0.6515
  ✓ New best model saved (Val Acc: 0.6515)
Training Epoch 64: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.48it/s, loss=1.5283, acc=73.78%]
[Phase 2 (Fine-tune)] Epoch 64/110 | Train Loss: 1.5283 | Train Acc: 0.7378 | Val Loss: 1.0537 | Val Acc: 0.6515
Training Epoch 65: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.31it/s, loss=1.5204, acc=73.83%]
[Phase 2 (Fine-tune)] Epoch 65/110 | Train Loss: 1.5204 | Train Acc: 0.7383 | Val Loss: 1.0550 | Val Acc: 0.6472
Training Epoch 66: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.44it/s, loss=1.5130, acc=74.42%]
[Phase 2 (Fine-tune)] Epoch 66/110 | Train Loss: 1.5130 | Train Acc: 0.7442 | Val Loss: 1.0576 | Val Acc: 0.6472
Training Epoch 67: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.58it/s, loss=1.5039, acc=74.62%]
[Phase 2 (Fine-tune)] Epoch 67/110 | Train Loss: 1.5039 | Train Acc: 0.7462 | Val Loss: 1.0596 | Val Acc: 0.6477
Training Epoch 68: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.56it/s, loss=1.5157, acc=74.15%]
[Phase 2 (Fine-tune)] Epoch 68/110 | Train Loss: 1.5157 | Train Acc: 0.7415 | Val Loss: 1.0558 | Val Acc: 0.6509
Training Epoch 69: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.52it/s, loss=1.5222, acc=73.52%]
[Phase 2 (Fine-tune)] Epoch 69/110 | Train Loss: 1.5222 | Train Acc: 0.7352 | Val Loss: 1.0595 | Val Acc: 0.6499
Training Epoch 70: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.58it/s, loss=1.5218, acc=73.86%]
[Phase 2 (Fine-tune)] Epoch 70/110 | Train Loss: 1.5218 | Train Acc: 0.7386 | Val Loss: 1.0565 | Val Acc: 0.6509
Training Epoch 71: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.60it/s, loss=1.5007, acc=74.69%]
[Phase 2 (Fine-tune)] Epoch 71/110 | Train Loss: 1.5007 | Train Acc: 0.7469 | Val Loss: 1.0589 | Val Acc: 0.6504
Training Epoch 72: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.33it/s, loss=1.5036, acc=74.17%]
[Phase 2 (Fine-tune)] Epoch 72/110 | Train Loss: 1.5036 | Train Acc: 0.7417 | Val Loss: 1.0595 | Val Acc: 0.6552
  ✓ New best model saved (Val Acc: 0.6552)
Training Epoch 73: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.76it/s, loss=1.4819, acc=75.26%]
[Phase 2 (Fine-tune)] Epoch 73/110 | Train Loss: 1.4819 | Train Acc: 0.7526 | Val Loss: 1.0621 | Val Acc: 0.6579
  ✓ New best model saved (Val Acc: 0.6579)
Training Epoch 74: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.35it/s, loss=1.4932, acc=74.37%]
[Phase 2 (Fine-tune)] Epoch 74/110 | Train Loss: 1.4932 | Train Acc: 0.7437 | Val Loss: 1.0619 | Val Acc: 0.6584
  ✓ New best model saved (Val Acc: 0.6584)
Training Epoch 75: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.62it/s, loss=1.4946, acc=74.62%]
[Phase 2 (Fine-tune)] Epoch 75/110 | Train Loss: 1.4946 | Train Acc: 0.7462 | Val Loss: 1.0607 | Val Acc: 0.6525
Training Epoch 76: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.59it/s, loss=1.4909, acc=75.51%]
[Phase 2 (Fine-tune)] Epoch 76/110 | Train Loss: 1.4909 | Train Acc: 0.7551 | Val Loss: 1.0635 | Val Acc: 0.6515
Training Epoch 77: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.22it/s, loss=1.4842, acc=75.45%]
[Phase 2 (Fine-tune)] Epoch 77/110 | Train Loss: 1.4842 | Train Acc: 0.7545 | Val Loss: 1.0673 | Val Acc: 0.6574
Training Epoch 78: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.19it/s, loss=1.4798, acc=75.66%]
[Phase 2 (Fine-tune)] Epoch 78/110 | Train Loss: 1.4798 | Train Acc: 0.7566 | Val Loss: 1.0687 | Val Acc: 0.6509
Training Epoch 79: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.51it/s, loss=1.4768, acc=75.66%]
[Phase 2 (Fine-tune)] Epoch 79/110 | Train Loss: 1.4768 | Train Acc: 0.7566 | Val Loss: 1.0706 | Val Acc: 0.6493
Training Epoch 80: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.44it/s, loss=1.4788, acc=75.65%]
[Phase 2 (Fine-tune)] Epoch 80/110 | Train Loss: 1.4788 | Train Acc: 0.7565 | Val Loss: 1.0675 | Val Acc: 0.6568
Training Epoch 81: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.22it/s, loss=1.4853, acc=75.65%]
  Learning rate reduced to 0.000024
[Phase 2 (Fine-tune)] Epoch 81/110 | Train Loss: 1.4853 | Train Acc: 0.7565 | Val Loss: 1.0687 | Val Acc: 0.6552
Training Epoch 82: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.33it/s, loss=1.4707, acc=76.26%]
[Phase 2 (Fine-tune)] Epoch 82/110 | Train Loss: 1.4707 | Train Acc: 0.7626 | Val Loss: 1.0706 | Val Acc: 0.6547
Training Epoch 83: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.35it/s, loss=1.4705, acc=76.72%]
[Phase 2 (Fine-tune)] Epoch 83/110 | Train Loss: 1.4705 | Train Acc: 0.7672 | Val Loss: 1.0726 | Val Acc: 0.6477
Training Epoch 84: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.35it/s, loss=1.4769, acc=75.55%]
[Phase 2 (Fine-tune)] Epoch 84/110 | Train Loss: 1.4769 | Train Acc: 0.7555 | Val Loss: 1.0661 | Val Acc: 0.6542
Training Epoch 85: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.31it/s, loss=1.4663, acc=76.17%]
[Phase 2 (Fine-tune)] Epoch 85/110 | Train Loss: 1.4663 | Train Acc: 0.7617 | Val Loss: 1.0688 | Val Acc: 0.6509
Training Epoch 86: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.40it/s, loss=1.4528, acc=76.92%]
[Phase 2 (Fine-tune)] Epoch 86/110 | Train Loss: 1.4528 | Train Acc: 0.7692 | Val Loss: 1.0685 | Val Acc: 0.6509
Training Epoch 87: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.51it/s, loss=1.4641, acc=76.15%]
[Phase 2 (Fine-tune)] Epoch 87/110 | Train Loss: 1.4641 | Train Acc: 0.7615 | Val Loss: 1.0716 | Val Acc: 0.6520
Training Epoch 88:   2%|██▎                                                                                           | 5/203 [00:00<00:29,  6.78it/s, loss=1.4126, acc=75.00%]
Training Epoch 88:   2%|██▎                                                                                           | 5/203 [00:00<00:29,  6.78it/s, loss=1.4078, acc=75.52%]
Training Epoch 88:   3%|███▏                                                                                          | 7/203 [00:01<00:26,  7.46it/s, loss=1.4342, acc=74.61%]
Training Epoch 88:   4%|████▏                                                                                         | 9/203 [00:01<00:24,  8.06it/s, loss=1.4141, acc=75.69%]
Training Epoch 88:   4%|████▏                                                                                         | 9/203 [00:01<00:24,  8.06it/s, loss=1.4270, acc=75.94%]
Training Epoch 88:   5%|█████                                                                                        | 11/203 [00:01<00:22,  8.54it/s, loss=1.4143, acc=77.86%]
Training Epoch 88:   6%|█████▉                                                                                       | 13/203 [00:01<00:21,  8.68it/s, loss=1.4183, acc=78.35%]
Training Epoch 88: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.53it/s, loss=1.4477, acc=77.19%]
[Phase 2 (Fine-tune)] Epoch 88/110 | Train Loss: 1.4477 | Train Acc: 0.7719 | Val Loss: 1.0753 | Val Acc: 0.6477
Training Epoch 89: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.53it/s, loss=1.4575, acc=76.03%]
[Phase 2 (Fine-tune)] Epoch 89/110 | Train Loss: 1.4575 | Train Acc: 0.7603 | Val Loss: 1.0706 | Val Acc: 0.6558

Early stopping triggered after 89 epochs

================================================================================
TRAINING COMPLETE
================================================================================
Best validation accuracy: 0.6584
Best model saved to: outputs/exp06_classification_GoogLeNet/run_20260426_022945/model/best_model.pth
2026-04-26 03:12:54 - exp06_classification_GoogLeNet - INFO - Training completed successfully!
2026-04-26 03:12:54 - exp06_classification_GoogLeNet - INFO - 
[Step 4/4] Evaluating model on test set...
2026-04-26 03:12:54 - exp06_classification_GoogLeNet - INFO - Using class names: ['angry', 'happy', 'relax', 'frown', 'alert']
================================================================================
CLASSIFICATION MODEL EVALUATION
================================================================================

Overall Metrics:
  Accuracy: 0.6206
  Precision: 0.6118
  Recall: 0.6206
  F1-Score: 0.6139

Per-Class Metrics:
  angry:
    Precision: 0.5159
    Recall: 0.4355
    F1-Score: 0.4723
  happy:
    Precision: 0.7600
    Recall: 0.8128
    F1-Score: 0.7855
  relax:
    Precision: 0.6955
    Recall: 0.8226
    F1-Score: 0.7537
  frown:
    Precision: 0.5663
    Recall: 0.5027
    F1-Score: 0.5326
  alert:
    Precision: 0.5211
    Recall: 0.5294
    F1-Score: 0.5252

Confusion Matrix:
[[ 81  33  23  17  32]
 [ 20 152   3   3   9]
 [  2   2 153  19  10]
 [ 19   2  32  94  40]
 [ 35  11   9  33  99]]

Metrics saved to: outputs/exp06_classification_GoogLeNet/run_20260426_022945/logs/evaluation_metrics.json
Detailed report saved to: outputs/exp06_classification_GoogLeNet/run_20260426_022945/logs/classification_report.txt
Report saved to: outputs/exp06_classification_GoogLeNet/run_20260426_022945/logs/experiment_report.md
2026-04-26 03:12:55 - exp06_classification_GoogLeNet - INFO - 
================================================================================
2026-04-26 03:12:55 - exp06_classification_GoogLeNet - INFO - EXPERIMENT COMPLETED SUCCESSFULLY
2026-04-26 03:12:55 - exp06_classification_GoogLeNet - INFO - ================================================================================
2026-04-26 03:12:55 - exp06_classification_GoogLeNet - INFO - Results saved to: outputs/exp06_classification_GoogLeNet/run_20260426_022945
2026-04-26 03:12:55 - exp06_classification_GoogLeNet - INFO - Best Validation Accuracy: 0.6584
2026-04-26 03:12:55 - exp06_classification_GoogLeNet - INFO - Test Accuracy: 0.6206
2026-04-26 03:12:55 - exp06_classification_GoogLeNet - INFO - Test Precision: 0.6118
2026-04-26 03:12:55 - exp06_classification_GoogLeNet - INFO - Test Recall: 0.6206
2026-04-26 03:12:55 - exp06_classification_GoogLeNet - INFO - Test F1-Score: 0.6139