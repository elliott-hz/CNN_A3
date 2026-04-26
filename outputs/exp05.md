(.venv) sagemaker-user@default:~/CNN_A3$ python experiments/exp05_classification_AlexNet.py 
2026-04-26 02:40:23 - exp05_classification_AlexNet - INFO - ================================================================================
2026-04-26 02:40:23 - exp05_classification_AlexNet - INFO - STARTING EXPERIMENT: exp05_classification_AlexNet
2026-04-26 02:40:23 - exp05_classification_AlexNet - INFO - Model: AlexNet (5 conv layers + 3 FC layers)
2026-04-26 02:40:23 - exp05_classification_AlexNet - INFO - ================================================================================
2026-04-26 02:40:23 - exp05_classification_AlexNet - INFO - 
[Step 1/4] Loading preprocessed data...
2026-04-26 02:41:09 - exp05_classification_AlexNet - INFO - Data loaded successfully:
2026-04-26 02:41:09 - exp05_classification_AlexNet - INFO -   Train: 6527 samples, shape: (224, 224, 3)
2026-04-26 02:41:09 - exp05_classification_AlexNet - INFO -   Valid: 1865 samples, shape: (224, 224, 3)
2026-04-26 02:41:09 - exp05_classification_AlexNet - INFO -   Test: 933 samples, shape: (224, 224, 3)
2026-04-26 02:41:09 - exp05_classification_AlexNet - INFO - 
[Step 2/4] Initializing model and trainer...
2026-04-26 02:41:09 - exp05_classification_AlexNet - INFO - Model config: {'num_classes': 5, 'dropout_rate': 0.5, 'pretrained': True, 'freeze_backbone': True}
2026-04-26 02:41:09 - exp05_classification_AlexNet - INFO - Training config: {'learning_rate': 0.01, 'batch_size': 64, 'epochs': 200, 'optimizer': 'sgd', 'momentum': 0.9, 'weight_decay': 0.0005, 'early_stopping_patience': 30, 'use_amp': True, 'gradient_accumulation_steps': 1, 'label_smoothing': 0.1, 'class_weighting': True, 'lr_decay_factor': 0.5, 'lr_decay_interval': 40}
Experiment directory created: outputs/exp05_classification_AlexNet/run_20260426_024109
2026-04-26 02:41:10 - exp05_classification_AlexNet - INFO - Total parameters: 65,822,509
2026-04-26 02:41:10 - exp05_classification_AlexNet - INFO - Trainable parameters: 63,352,813
Using device: cuda
2026-04-26 02:41:10 - exp05_classification_AlexNet - INFO - 
[Step 3/4] Training model...
================================================================================
CLASSIFICATION MODEL TRAINING
================================================================================
Model config: {'num_classes': 5, 'dropout_rate': 0.5, 'pretrained': True, 'freeze_backbone': True}
Training config: {'learning_rate': 0.01, 'batch_size': 64, 'epochs': 200, 'optimizer': 'sgd', 'momentum': 0.9, 'weight_decay': 0.0005, 'early_stopping_patience': 30, 'use_amp': True, 'gradient_accumulation_steps': 1, 'label_smoothing': 0.1, 'class_weighting': True, 'lr_decay_factor': 0.5, 'lr_decay_interval': 40}
Output directory: outputs/exp05_classification_AlexNet/run_20260426_024109
Class weights: [    0.99954      1.0003     0.99954      1.0003      1.0003]

================================================================================
PHASE 1: Training with frozen backbone
================================================================================
Training Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.53it/s, loss=1.5356, acc=33.66%]
[Phase 1 (Frozen)] Epoch 1/10 | Train Loss: 1.5356 | Train Acc: 0.3366 | Val Loss: 1.4032 | Val Acc: 0.4263
  ✓ New best model saved (Val Acc: 0.4263)
Training Epoch 2: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.63it/s, loss=1.4936, acc=35.72%]
[Phase 1 (Frozen)] Epoch 2/10 | Train Loss: 1.4936 | Train Acc: 0.3572 | Val Loss: 1.3574 | Val Acc: 0.4477
  ✓ New best model saved (Val Acc: 0.4477)
Training Epoch 3: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.47it/s, loss=1.4828, acc=36.56%]
[Phase 1 (Frozen)] Epoch 3/10 | Train Loss: 1.4828 | Train Acc: 0.3656 | Val Loss: 1.3640 | Val Acc: 0.4499
  ✓ New best model saved (Val Acc: 0.4499)
Training Epoch 4: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.70it/s, loss=1.4794, acc=36.34%]
[Phase 1 (Frozen)] Epoch 4/10 | Train Loss: 1.4794 | Train Acc: 0.3634 | Val Loss: 1.4384 | Val Acc: 0.4257
Training Epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.61it/s, loss=1.4786, acc=36.62%]
[Phase 1 (Frozen)] Epoch 5/10 | Train Loss: 1.4786 | Train Acc: 0.3662 | Val Loss: 1.4068 | Val Acc: 0.4134
Training Epoch 6: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.73it/s, loss=1.4818, acc=35.43%]
[Phase 1 (Frozen)] Epoch 6/10 | Train Loss: 1.4818 | Train Acc: 0.3543 | Val Loss: 1.3699 | Val Acc: 0.4461
Training Epoch 7: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.61it/s, loss=1.4793, acc=37.00%]
[Phase 1 (Frozen)] Epoch 7/10 | Train Loss: 1.4793 | Train Acc: 0.3700 | Val Loss: 1.3727 | Val Acc: 0.4241
Training Epoch 8: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.45it/s, loss=1.4614, acc=38.29%]
[Phase 1 (Frozen)] Epoch 8/10 | Train Loss: 1.4614 | Train Acc: 0.3829 | Val Loss: 1.3704 | Val Acc: 0.4381
Training Epoch 9: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.48it/s, loss=1.4639, acc=37.55%]
[Phase 1 (Frozen)] Epoch 9/10 | Train Loss: 1.4639 | Train Acc: 0.3755 | Val Loss: 1.3622 | Val Acc: 0.4515
  ✓ New best model saved (Val Acc: 0.4515)
Training Epoch 10: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.62it/s, loss=1.4671, acc=38.21%]
[Phase 1 (Frozen)] Epoch 10/10 | Train Loss: 1.4671 | Train Acc: 0.3821 | Val Loss: 1.3715 | Val Acc: 0.4638
  ✓ New best model saved (Val Acc: 0.4638)

================================================================================
PHASE 2: Fine-tuning with unfrozen backbone
================================================================================
Training Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.55it/s, loss=1.4326, acc=40.19%]
[Phase 2 (Fine-tune)] Epoch 1/190 | Train Loss: 1.4326 | Train Acc: 0.4019 | Val Loss: 1.3098 | Val Acc: 0.4987
  ✓ New best model saved (Val Acc: 0.4987)
Training Epoch 2: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.59it/s, loss=1.3995, acc=42.61%]
[Phase 2 (Fine-tune)] Epoch 2/190 | Train Loss: 1.3995 | Train Acc: 0.4261 | Val Loss: 1.2911 | Val Acc: 0.5029
  ✓ New best model saved (Val Acc: 0.5029)
Training Epoch 3: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.59it/s, loss=1.3848, acc=43.90%]
[Phase 2 (Fine-tune)] Epoch 3/190 | Train Loss: 1.3848 | Train Acc: 0.4390 | Val Loss: 1.2854 | Val Acc: 0.5121
  ✓ New best model saved (Val Acc: 0.5121)
Training Epoch 4: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.54it/s, loss=1.3895, acc=42.88%]
[Phase 2 (Fine-tune)] Epoch 4/190 | Train Loss: 1.3895 | Train Acc: 0.4288 | Val Loss: 1.2826 | Val Acc: 0.5062
Training Epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.57it/s, loss=1.3709, acc=44.91%]
[Phase 2 (Fine-tune)] Epoch 5/190 | Train Loss: 1.3709 | Train Acc: 0.4491 | Val Loss: 1.2669 | Val Acc: 0.5190
  ✓ New best model saved (Val Acc: 0.5190)
Training Epoch 6: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.60it/s, loss=1.3731, acc=44.69%]
[Phase 2 (Fine-tune)] Epoch 6/190 | Train Loss: 1.3731 | Train Acc: 0.4469 | Val Loss: 1.2583 | Val Acc: 0.5244
  ✓ New best model saved (Val Acc: 0.5244)
Training Epoch 7: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.45it/s, loss=1.3555, acc=45.82%]
[Phase 2 (Fine-tune)] Epoch 7/190 | Train Loss: 1.3555 | Train Acc: 0.4582 | Val Loss: 1.2530 | Val Acc: 0.5303
  ✓ New best model saved (Val Acc: 0.5303)
Training Epoch 8: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.48it/s, loss=1.3512, acc=45.31%]
[Phase 2 (Fine-tune)] Epoch 8/190 | Train Loss: 1.3512 | Train Acc: 0.4531 | Val Loss: 1.2439 | Val Acc: 0.5217
Training Epoch 9: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.57it/s, loss=1.3380, acc=47.37%]
[Phase 2 (Fine-tune)] Epoch 9/190 | Train Loss: 1.3380 | Train Acc: 0.4737 | Val Loss: 1.2379 | Val Acc: 0.5340
  ✓ New best model saved (Val Acc: 0.5340)
Training Epoch 10: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.57it/s, loss=1.3415, acc=47.06%]
[Phase 2 (Fine-tune)] Epoch 10/190 | Train Loss: 1.3415 | Train Acc: 0.4706 | Val Loss: 1.2391 | Val Acc: 0.5244
Training Epoch 11: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.64it/s, loss=1.3320, acc=47.08%]
[Phase 2 (Fine-tune)] Epoch 11/190 | Train Loss: 1.3320 | Train Acc: 0.4708 | Val Loss: 1.2437 | Val Acc: 0.5249
Training Epoch 12: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.56it/s, loss=1.3223, acc=48.14%]
[Phase 2 (Fine-tune)] Epoch 12/190 | Train Loss: 1.3223 | Train Acc: 0.4814 | Val Loss: 1.2256 | Val Acc: 0.5282
Training Epoch 13: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.60it/s, loss=1.3130, acc=48.81%]
[Phase 2 (Fine-tune)] Epoch 13/190 | Train Loss: 1.3130 | Train Acc: 0.4881 | Val Loss: 1.2216 | Val Acc: 0.5373
  ✓ New best model saved (Val Acc: 0.5373)
Training Epoch 14: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.55it/s, loss=1.3181, acc=48.16%]
[Phase 2 (Fine-tune)] Epoch 14/190 | Train Loss: 1.3181 | Train Acc: 0.4816 | Val Loss: 1.2164 | Val Acc: 0.5346
Training Epoch 15: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.52it/s, loss=1.3057, acc=48.59%]
[Phase 2 (Fine-tune)] Epoch 15/190 | Train Loss: 1.3057 | Train Acc: 0.4859 | Val Loss: 1.2119 | Val Acc: 0.5399
  ✓ New best model saved (Val Acc: 0.5399)
Training Epoch 16: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.61it/s, loss=1.3058, acc=49.10%]
[Phase 2 (Fine-tune)] Epoch 16/190 | Train Loss: 1.3058 | Train Acc: 0.4910 | Val Loss: 1.2072 | Val Acc: 0.5432
  ✓ New best model saved (Val Acc: 0.5432)
Training Epoch 17: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.45it/s, loss=1.2975, acc=50.14%]
[Phase 2 (Fine-tune)] Epoch 17/190 | Train Loss: 1.2975 | Train Acc: 0.5014 | Val Loss: 1.2059 | Val Acc: 0.5501
  ✓ New best model saved (Val Acc: 0.5501)
Training Epoch 18: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:23<00:00,  4.38it/s, loss=1.3012, acc=50.05%]
[Phase 2 (Fine-tune)] Epoch 18/190 | Train Loss: 1.3012 | Train Acc: 0.5005 | Val Loss: 1.1977 | Val Acc: 0.5566
  ✓ New best model saved (Val Acc: 0.5566)
Training Epoch 19: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.55it/s, loss=1.2905, acc=50.56%]
[Phase 2 (Fine-tune)] Epoch 19/190 | Train Loss: 1.2905 | Train Acc: 0.5056 | Val Loss: 1.2049 | Val Acc: 0.5432
Training Epoch 20: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.48it/s, loss=1.2852, acc=50.26%]
[Phase 2 (Fine-tune)] Epoch 20/190 | Train Loss: 1.2852 | Train Acc: 0.5026 | Val Loss: 1.1942 | Val Acc: 0.5539
Training Epoch 21: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.50it/s, loss=1.2816, acc=51.28%]
[Phase 2 (Fine-tune)] Epoch 21/190 | Train Loss: 1.2816 | Train Acc: 0.5128 | Val Loss: 1.1975 | Val Acc: 0.5448
Training Epoch 22: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.51it/s, loss=1.2809, acc=50.80%]
[Phase 2 (Fine-tune)] Epoch 22/190 | Train Loss: 1.2809 | Train Acc: 0.5080 | Val Loss: 1.1911 | Val Acc: 0.5480
Training Epoch 23: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.42it/s, loss=1.2717, acc=51.62%]
[Phase 2 (Fine-tune)] Epoch 23/190 | Train Loss: 1.2717 | Train Acc: 0.5162 | Val Loss: 1.1899 | Val Acc: 0.5485
Training Epoch 24: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.57it/s, loss=1.2658, acc=52.31%]
[Phase 2 (Fine-tune)] Epoch 24/190 | Train Loss: 1.2658 | Train Acc: 0.5231 | Val Loss: 1.1852 | Val Acc: 0.5491
Training Epoch 25: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.57it/s, loss=1.2651, acc=52.00%]
[Phase 2 (Fine-tune)] Epoch 25/190 | Train Loss: 1.2651 | Train Acc: 0.5200 | Val Loss: 1.1790 | Val Acc: 0.5501
Training Epoch 26: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.52it/s, loss=1.2635, acc=52.31%]
[Phase 2 (Fine-tune)] Epoch 26/190 | Train Loss: 1.2635 | Train Acc: 0.5231 | Val Loss: 1.1863 | Val Acc: 0.5464
Training Epoch 27: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.52it/s, loss=1.2587, acc=52.69%]
[Phase 2 (Fine-tune)] Epoch 27/190 | Train Loss: 1.2587 | Train Acc: 0.5269 | Val Loss: 1.1711 | Val Acc: 0.5571
  ✓ New best model saved (Val Acc: 0.5571)
Training Epoch 28: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.56it/s, loss=1.2577, acc=52.55%]
[Phase 2 (Fine-tune)] Epoch 28/190 | Train Loss: 1.2577 | Train Acc: 0.5255 | Val Loss: 1.1743 | Val Acc: 0.5587
  ✓ New best model saved (Val Acc: 0.5587)
Training Epoch 29: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.60it/s, loss=1.2487, acc=53.51%]
[Phase 2 (Fine-tune)] Epoch 29/190 | Train Loss: 1.2487 | Train Acc: 0.5351 | Val Loss: 1.1714 | Val Acc: 0.5576
Training Epoch 30: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.53it/s, loss=1.2453, acc=53.37%]
[Phase 2 (Fine-tune)] Epoch 30/190 | Train Loss: 1.2453 | Train Acc: 0.5337 | Val Loss: 1.1623 | Val Acc: 0.5641
  ✓ New best model saved (Val Acc: 0.5641)
Training Epoch 31: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.47it/s, loss=1.2404, acc=54.21%]
[Phase 2 (Fine-tune)] Epoch 31/190 | Train Loss: 1.2404 | Train Acc: 0.5421 | Val Loss: 1.1627 | Val Acc: 0.5678
  ✓ New best model saved (Val Acc: 0.5678)
Training Epoch 32: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:23<00:00,  4.34it/s, loss=1.2384, acc=54.07%]
[Phase 2 (Fine-tune)] Epoch 32/190 | Train Loss: 1.2384 | Train Acc: 0.5407 | Val Loss: 1.1639 | Val Acc: 0.5689
  ✓ New best model saved (Val Acc: 0.5689)
Training Epoch 33: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.55it/s, loss=1.2343, acc=54.66%]
[Phase 2 (Fine-tune)] Epoch 33/190 | Train Loss: 1.2343 | Train Acc: 0.5466 | Val Loss: 1.1636 | Val Acc: 0.5748
  ✓ New best model saved (Val Acc: 0.5748)
Training Epoch 34: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.50it/s, loss=1.2313, acc=54.33%]
[Phase 2 (Fine-tune)] Epoch 34/190 | Train Loss: 1.2313 | Train Acc: 0.5433 | Val Loss: 1.1559 | Val Acc: 0.5710
Training Epoch 35: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.58it/s, loss=1.2244, acc=55.23%]
[Phase 2 (Fine-tune)] Epoch 35/190 | Train Loss: 1.2244 | Train Acc: 0.5523 | Val Loss: 1.1583 | Val Acc: 0.5748
Training Epoch 36: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.47it/s, loss=1.2170, acc=55.60%]
[Phase 2 (Fine-tune)] Epoch 36/190 | Train Loss: 1.2170 | Train Acc: 0.5560 | Val Loss: 1.1544 | Val Acc: 0.5716
Training Epoch 37: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.53it/s, loss=1.2245, acc=55.82%]
[Phase 2 (Fine-tune)] Epoch 37/190 | Train Loss: 1.2245 | Train Acc: 0.5582 | Val Loss: 1.1551 | Val Acc: 0.5635
Training Epoch 38: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.51it/s, loss=1.2133, acc=55.34%]
[Phase 2 (Fine-tune)] Epoch 38/190 | Train Loss: 1.2133 | Train Acc: 0.5534 | Val Loss: 1.1560 | Val Acc: 0.5678
Training Epoch 39: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:24<00:00,  4.04it/s, loss=1.2118, acc=55.62%]
[Phase 2 (Fine-tune)] Epoch 39/190 | Train Loss: 1.2118 | Train Acc: 0.5562 | Val Loss: 1.1480 | Val Acc: 0.5807
  ✓ New best model saved (Val Acc: 0.5807)
Training Epoch 40: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:27<00:00,  3.70it/s, loss=1.2167, acc=55.32%]
[Phase 2 (Fine-tune)] Epoch 40/190 | Train Loss: 1.2167 | Train Acc: 0.5532 | Val Loss: 1.1479 | Val Acc: 0.5791
Training Epoch 41: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:27<00:00,  3.61it/s, loss=1.2064, acc=55.82%]
  Learning rate reduced from 0.000100 to 0.000050
[Phase 2 (Fine-tune)] Epoch 41/190 | Train Loss: 1.2064 | Train Acc: 0.5582 | Val Loss: 1.1448 | Val Acc: 0.5764
Training Epoch 42: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:27<00:00,  3.69it/s, loss=1.2063, acc=56.14%]
[Phase 2 (Fine-tune)] Epoch 42/190 | Train Loss: 1.2063 | Train Acc: 0.5614 | Val Loss: 1.1446 | Val Acc: 0.5802
Training Epoch 43: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:27<00:00,  3.70it/s, loss=1.2006, acc=56.33%]
[Phase 2 (Fine-tune)] Epoch 43/190 | Train Loss: 1.2006 | Train Acc: 0.5633 | Val Loss: 1.1412 | Val Acc: 0.5791
Training Epoch 44: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:24<00:00,  4.13it/s, loss=1.2037, acc=56.51%]
[Phase 2 (Fine-tune)] Epoch 44/190 | Train Loss: 1.2037 | Train Acc: 0.5651 | Val Loss: 1.1413 | Val Acc: 0.5834
  ✓ New best model saved (Val Acc: 0.5834)
Training Epoch 45: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.49it/s, loss=1.1923, acc=56.56%]
[Phase 2 (Fine-tune)] Epoch 45/190 | Train Loss: 1.1923 | Train Acc: 0.5656 | Val Loss: 1.1405 | Val Acc: 0.5807
Training Epoch 46: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.57it/s, loss=1.1948, acc=57.21%]
[Phase 2 (Fine-tune)] Epoch 46/190 | Train Loss: 1.1948 | Train Acc: 0.5721 | Val Loss: 1.1397 | Val Acc: 0.5823
Training Epoch 47: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.42it/s, loss=1.1927, acc=56.67%]
[Phase 2 (Fine-tune)] Epoch 47/190 | Train Loss: 1.1927 | Train Acc: 0.5667 | Val Loss: 1.1409 | Val Acc: 0.5780
Training Epoch 48: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.56it/s, loss=1.1849, acc=57.27%]
[Phase 2 (Fine-tune)] Epoch 48/190 | Train Loss: 1.1849 | Train Acc: 0.5727 | Val Loss: 1.1368 | Val Acc: 0.5834
Training Epoch 49: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.60it/s, loss=1.1866, acc=57.60%]
[Phase 2 (Fine-tune)] Epoch 49/190 | Train Loss: 1.1866 | Train Acc: 0.5760 | Val Loss: 1.1354 | Val Acc: 0.5818
Training Epoch 50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.49it/s, loss=1.1954, acc=56.28%]
[Phase 2 (Fine-tune)] Epoch 50/190 | Train Loss: 1.1954 | Train Acc: 0.5628 | Val Loss: 1.1354 | Val Acc: 0.5850
  ✓ New best model saved (Val Acc: 0.5850)
Training Epoch 51: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.50it/s, loss=1.1923, acc=56.68%]
[Phase 2 (Fine-tune)] Epoch 51/190 | Train Loss: 1.1923 | Train Acc: 0.5668 | Val Loss: 1.1342 | Val Acc: 0.5861
  ✓ New best model saved (Val Acc: 0.5861)
Training Epoch 52: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.68it/s, loss=1.1756, acc=57.52%]
[Phase 2 (Fine-tune)] Epoch 52/190 | Train Loss: 1.1756 | Train Acc: 0.5752 | Val Loss: 1.1342 | Val Acc: 0.5823
Training Epoch 53: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.62it/s, loss=1.1841, acc=56.99%]
[Phase 2 (Fine-tune)] Epoch 53/190 | Train Loss: 1.1841 | Train Acc: 0.5699 | Val Loss: 1.1337 | Val Acc: 0.5839
Training Epoch 54: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.54it/s, loss=1.1801, acc=57.44%]
[Phase 2 (Fine-tune)] Epoch 54/190 | Train Loss: 1.1801 | Train Acc: 0.5744 | Val Loss: 1.1322 | Val Acc: 0.5834
Training Epoch 55: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.57it/s, loss=1.1810, acc=57.04%]
[Phase 2 (Fine-tune)] Epoch 55/190 | Train Loss: 1.1810 | Train Acc: 0.5704 | Val Loss: 1.1348 | Val Acc: 0.5802
Training Epoch 56: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.56it/s, loss=1.1800, acc=57.07%]
[Phase 2 (Fine-tune)] Epoch 56/190 | Train Loss: 1.1800 | Train Acc: 0.5707 | Val Loss: 1.1305 | Val Acc: 0.5855
Training Epoch 57: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.59it/s, loss=1.1731, acc=57.95%]
[Phase 2 (Fine-tune)] Epoch 57/190 | Train Loss: 1.1731 | Train Acc: 0.5795 | Val Loss: 1.1273 | Val Acc: 0.5850
Training Epoch 58: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.51it/s, loss=1.1761, acc=57.04%]
[Phase 2 (Fine-tune)] Epoch 58/190 | Train Loss: 1.1761 | Train Acc: 0.5704 | Val Loss: 1.1324 | Val Acc: 0.5791
Training Epoch 59: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.53it/s, loss=1.1735, acc=57.70%]
[Phase 2 (Fine-tune)] Epoch 59/190 | Train Loss: 1.1735 | Train Acc: 0.5770 | Val Loss: 1.1288 | Val Acc: 0.5882
  ✓ New best model saved (Val Acc: 0.5882)
Training Epoch 60: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.51it/s, loss=1.1698, acc=58.14%]
[Phase 2 (Fine-tune)] Epoch 60/190 | Train Loss: 1.1698 | Train Acc: 0.5814 | Val Loss: 1.1282 | Val Acc: 0.5855
Training Epoch 61: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.62it/s, loss=1.1731, acc=57.89%]
[Phase 2 (Fine-tune)] Epoch 61/190 | Train Loss: 1.1731 | Train Acc: 0.5789 | Val Loss: 1.1316 | Val Acc: 0.5834
Training Epoch 62: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.64it/s, loss=1.1617, acc=58.77%]
[Phase 2 (Fine-tune)] Epoch 62/190 | Train Loss: 1.1617 | Train Acc: 0.5877 | Val Loss: 1.1250 | Val Acc: 0.5882
Training Epoch 63: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.58it/s, loss=1.1655, acc=58.45%]
[Phase 2 (Fine-tune)] Epoch 63/190 | Train Loss: 1.1655 | Train Acc: 0.5845 | Val Loss: 1.1252 | Val Acc: 0.5930
  ✓ New best model saved (Val Acc: 0.5930)
Training Epoch 64: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.62it/s, loss=1.1666, acc=58.01%]
[Phase 2 (Fine-tune)] Epoch 64/190 | Train Loss: 1.1666 | Train Acc: 0.5801 | Val Loss: 1.1327 | Val Acc: 0.5802
Training Epoch 65:  27%|████████████████████████▊                                                                    | 27/101 [00:06<00:16,  4.41it/s, loss=1.1800, acc=58.76%]

Training Epoch 65:  29%|██████████████████████████▋                                                                  | 29/101 [00:06<00:15,  4.54it/s, loss=1.1767, acc=59.17%]


Training Epoch 65:  31%|████████████████████████████▌                                                                | 31/101 [00:07<00:15,  4.59it/s, loss=1.1730, acc=59.38%]
Training Epoch 65: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.46it/s, loss=1.1617, acc=59.22%]
[Phase 2 (Fine-tune)] Epoch 65/190 | Train Loss: 1.1617 | Train Acc: 0.5922 | Val Loss: 1.1306 | Val Acc: 0.5823
Training Epoch 66: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.54it/s, loss=1.1575, acc=58.88%]
[Phase 2 (Fine-tune)] Epoch 66/190 | Train Loss: 1.1575 | Train Acc: 0.5888 | Val Loss: 1.1248 | Val Acc: 0.5786
Training Epoch 67: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.54it/s, loss=1.1541, acc=59.17%]
[Phase 2 (Fine-tune)] Epoch 67/190 | Train Loss: 1.1541 | Train Acc: 0.5917 | Val Loss: 1.1235 | Val Acc: 0.5914
Training Epoch 68: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.60it/s, loss=1.1558, acc=58.93%]
[Phase 2 (Fine-tune)] Epoch 68/190 | Train Loss: 1.1558 | Train Acc: 0.5893 | Val Loss: 1.1225 | Val Acc: 0.5893
Training Epoch 69: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.48it/s, loss=1.1656, acc=58.91%]
[Phase 2 (Fine-tune)] Epoch 69/190 | Train Loss: 1.1656 | Train Acc: 0.5891 | Val Loss: 1.1244 | Val Acc: 0.5855
Training Epoch 70: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.58it/s, loss=1.1522, acc=59.48%]
[Phase 2 (Fine-tune)] Epoch 70/190 | Train Loss: 1.1522 | Train Acc: 0.5948 | Val Loss: 1.1225 | Val Acc: 0.5845
Training Epoch 71: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.57it/s, loss=1.1559, acc=59.53%]
[Phase 2 (Fine-tune)] Epoch 71/190 | Train Loss: 1.1559 | Train Acc: 0.5953 | Val Loss: 1.1273 | Val Acc: 0.5828
Training Epoch 72: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.57it/s, loss=1.1604, acc=59.25%]
[Phase 2 (Fine-tune)] Epoch 72/190 | Train Loss: 1.1604 | Train Acc: 0.5925 | Val Loss: 1.1211 | Val Acc: 0.5914
Training Epoch 73: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.66it/s, loss=1.1518, acc=59.61%]
[Phase 2 (Fine-tune)] Epoch 73/190 | Train Loss: 1.1518 | Train Acc: 0.5961 | Val Loss: 1.1194 | Val Acc: 0.5920
Training Epoch 74: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.60it/s, loss=1.1583, acc=59.11%]
[Phase 2 (Fine-tune)] Epoch 74/190 | Train Loss: 1.1583 | Train Acc: 0.5911 | Val Loss: 1.1214 | Val Acc: 0.5914
Training Epoch 75: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.60it/s, loss=1.1501, acc=59.42%]
[Phase 2 (Fine-tune)] Epoch 75/190 | Train Loss: 1.1501 | Train Acc: 0.5942 | Val Loss: 1.1206 | Val Acc: 0.5952
  ✓ New best model saved (Val Acc: 0.5952)
Training Epoch 76: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.48it/s, loss=1.1456, acc=60.16%]
[Phase 2 (Fine-tune)] Epoch 76/190 | Train Loss: 1.1456 | Train Acc: 0.6016 | Val Loss: 1.1226 | Val Acc: 0.5979
  ✓ New best model saved (Val Acc: 0.5979)
Training Epoch 77: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.60it/s, loss=1.1397, acc=60.84%]
[Phase 2 (Fine-tune)] Epoch 77/190 | Train Loss: 1.1397 | Train Acc: 0.6084 | Val Loss: 1.1183 | Val Acc: 0.5903
Training Epoch 78: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.59it/s, loss=1.1413, acc=59.72%]
[Phase 2 (Fine-tune)] Epoch 78/190 | Train Loss: 1.1413 | Train Acc: 0.5972 | Val Loss: 1.1185 | Val Acc: 0.5925
Training Epoch 79: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.49it/s, loss=1.1414, acc=60.10%]
[Phase 2 (Fine-tune)] Epoch 79/190 | Train Loss: 1.1414 | Train Acc: 0.6010 | Val Loss: 1.1231 | Val Acc: 0.5828
Training Epoch 80: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.57it/s, loss=1.1424, acc=59.61%]
[Phase 2 (Fine-tune)] Epoch 80/190 | Train Loss: 1.1424 | Train Acc: 0.5961 | Val Loss: 1.1188 | Val Acc: 0.5877
Training Epoch 81: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.63it/s, loss=1.1342, acc=60.38%]
  Learning rate reduced from 0.000050 to 0.000025
[Phase 2 (Fine-tune)] Epoch 81/190 | Train Loss: 1.1342 | Train Acc: 0.6038 | Val Loss: 1.1242 | Val Acc: 0.5871
Training Epoch 82: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.54it/s, loss=1.1418, acc=59.28%]
[Phase 2 (Fine-tune)] Epoch 82/190 | Train Loss: 1.1418 | Train Acc: 0.5928 | Val Loss: 1.1210 | Val Acc: 0.5887
Training Epoch 83: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.53it/s, loss=1.1341, acc=60.15%]
[Phase 2 (Fine-tune)] Epoch 83/190 | Train Loss: 1.1341 | Train Acc: 0.6015 | Val Loss: 1.1182 | Val Acc: 0.5845
Training Epoch 84: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.54it/s, loss=1.1334, acc=61.23%]
[Phase 2 (Fine-tune)] Epoch 84/190 | Train Loss: 1.1334 | Train Acc: 0.6123 | Val Loss: 1.1200 | Val Acc: 0.5898
Training Epoch 85: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.46it/s, loss=1.1389, acc=59.65%]
[Phase 2 (Fine-tune)] Epoch 85/190 | Train Loss: 1.1389 | Train Acc: 0.5965 | Val Loss: 1.1190 | Val Acc: 0.5936
Training Epoch 86: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.55it/s, loss=1.1344, acc=61.29%]
[Phase 2 (Fine-tune)] Epoch 86/190 | Train Loss: 1.1344 | Train Acc: 0.6129 | Val Loss: 1.1188 | Val Acc: 0.5850
Training Epoch 87: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.64it/s, loss=1.1339, acc=59.79%]
[Phase 2 (Fine-tune)] Epoch 87/190 | Train Loss: 1.1339 | Train Acc: 0.5979 | Val Loss: 1.1187 | Val Acc: 0.5941
Training Epoch 88: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.42it/s, loss=1.1344, acc=60.43%]
[Phase 2 (Fine-tune)] Epoch 88/190 | Train Loss: 1.1344 | Train Acc: 0.6043 | Val Loss: 1.1168 | Val Acc: 0.5925
Training Epoch 89:  98%|███████████████████████████████████████████████████████████████████████████████████████████▏ | 99/101 [00:21<00:00,  4.95it/s, loss=1.1309, acc=61.02%]
Training Epoch 89:  99%|███████████████████████████████████████████████████████████████████████████████████████████ | 100/101 [00:21<00:00,  4.47it/s, loss=1.1311, acc=61.02%]
Training Epoch 89: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  5.17it/s, loss=1.1309, acc=61.03%]
Training Epoch 89: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.57it/s, loss=1.1309, acc=61.03%]
[Phase 2 (Fine-tune)] Epoch 89/190 | Train Loss: 1.1309 | Train Acc: 0.6103 | Val Loss: 1.1173 | Val Acc: 0.5861
Training Epoch 90: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.55it/s, loss=1.1329, acc=60.27%]
[Phase 2 (Fine-tune)] Epoch 90/190 | Train Loss: 1.1329 | Train Acc: 0.6027 | Val Loss: 1.1183 | Val Acc: 0.5887
Training Epoch 91: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.52it/s, loss=1.1317, acc=60.77%]
[Phase 2 (Fine-tune)] Epoch 91/190 | Train Loss: 1.1317 | Train Acc: 0.6077 | Val Loss: 1.1159 | Val Acc: 0.5866
Training Epoch 92: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.54it/s, loss=1.1267, acc=61.63%]
[Phase 2 (Fine-tune)] Epoch 92/190 | Train Loss: 1.1267 | Train Acc: 0.6163 | Val Loss: 1.1159 | Val Acc: 0.5903
Training Epoch 93: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.55it/s, loss=1.1286, acc=60.19%]
[Phase 2 (Fine-tune)] Epoch 93/190 | Train Loss: 1.1286 | Train Acc: 0.6019 | Val Loss: 1.1165 | Val Acc: 0.5930
Training Epoch 94: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.52it/s, loss=1.1304, acc=60.23%]
[Phase 2 (Fine-tune)] Epoch 94/190 | Train Loss: 1.1304 | Train Acc: 0.6023 | Val Loss: 1.1149 | Val Acc: 0.5909
Training Epoch 95: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.58it/s, loss=1.1264, acc=61.22%]
[Phase 2 (Fine-tune)] Epoch 95/190 | Train Loss: 1.1264 | Train Acc: 0.6122 | Val Loss: 1.1152 | Val Acc: 0.5871
Training Epoch 96: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:23<00:00,  4.38it/s, loss=1.1265, acc=60.74%]
[Phase 2 (Fine-tune)] Epoch 96/190 | Train Loss: 1.1265 | Train Acc: 0.6074 | Val Loss: 1.1143 | Val Acc: 0.5930
Training Epoch 97: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.55it/s, loss=1.1313, acc=60.16%]
[Phase 2 (Fine-tune)] Epoch 97/190 | Train Loss: 1.1313 | Train Acc: 0.6016 | Val Loss: 1.1134 | Val Acc: 0.5936
Training Epoch 98: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.51it/s, loss=1.1245, acc=60.44%]
[Phase 2 (Fine-tune)] Epoch 98/190 | Train Loss: 1.1245 | Train Acc: 0.6044 | Val Loss: 1.1141 | Val Acc: 0.5946
Training Epoch 99: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.52it/s, loss=1.1251, acc=60.58%]
[Phase 2 (Fine-tune)] Epoch 99/190 | Train Loss: 1.1251 | Train Acc: 0.6058 | Val Loss: 1.1139 | Val Acc: 0.5898
Training Epoch 100: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.61it/s, loss=1.1288, acc=60.84%]
[Phase 2 (Fine-tune)] Epoch 100/190 | Train Loss: 1.1288 | Train Acc: 0.6084 | Val Loss: 1.1136 | Val Acc: 0.5903
Training Epoch 101: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.58it/s, loss=1.1290, acc=60.43%]
[Phase 2 (Fine-tune)] Epoch 101/190 | Train Loss: 1.1290 | Train Acc: 0.6043 | Val Loss: 1.1133 | Val Acc: 0.5882
Training Epoch 102: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.58it/s, loss=1.1211, acc=61.29%]
[Phase 2 (Fine-tune)] Epoch 102/190 | Train Loss: 1.1211 | Train Acc: 0.6129 | Val Loss: 1.1181 | Val Acc: 0.5909
Training Epoch 103: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.56it/s, loss=1.1265, acc=59.95%]
[Phase 2 (Fine-tune)] Epoch 103/190 | Train Loss: 1.1265 | Train Acc: 0.5995 | Val Loss: 1.1141 | Val Acc: 0.5903
Training Epoch 104: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.48it/s, loss=1.1179, acc=61.51%]
[Phase 2 (Fine-tune)] Epoch 104/190 | Train Loss: 1.1179 | Train Acc: 0.6151 | Val Loss: 1.1132 | Val Acc: 0.5882
Training Epoch 105: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:21<00:00,  4.63it/s, loss=1.1270, acc=60.77%]
[Phase 2 (Fine-tune)] Epoch 105/190 | Train Loss: 1.1270 | Train Acc: 0.6077 | Val Loss: 1.1137 | Val Acc: 0.5957
Training Epoch 106: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:22<00:00,  4.48it/s, loss=1.1158, acc=61.99%]
[Phase 2 (Fine-tune)] Epoch 106/190 | Train Loss: 1.1158 | Train Acc: 0.6199 | Val Loss: 1.1137 | Val Acc: 0.5925

Early stopping triggered after 106 epochs

================================================================================
TRAINING COMPLETE
================================================================================
Best validation accuracy: 0.5979
Best model saved to: outputs/exp05_classification_AlexNet/run_20260426_024109/model/best_model.pth
2026-04-26 03:28:58 - exp05_classification_AlexNet - INFO - Training completed successfully!
2026-04-26 03:28:58 - exp05_classification_AlexNet - INFO - 
[Step 4/4] Evaluating model on test set...
2026-04-26 03:28:58 - exp05_classification_AlexNet - INFO - Using class names: ['angry', 'happy', 'relax', 'frown', 'alert']
================================================================================
CLASSIFICATION MODEL EVALUATION
================================================================================

Overall Metrics:
  Accuracy: 0.6195
  Precision: 0.6151
  Recall: 0.6195
  F1-Score: 0.6111

Per-Class Metrics:
  angry:
    Precision: 0.5882
    Recall: 0.3763
    F1-Score: 0.4590
  happy:
    Precision: 0.7364
    Recall: 0.8663
    F1-Score: 0.7961
  relax:
    Precision: 0.6837
    Recall: 0.7204
    F1-Score: 0.7016
  frown:
    Precision: 0.5143
    Recall: 0.5775
    F1-Score: 0.5441
  alert:
    Precision: 0.5532
    Recall: 0.5561
    F1-Score: 0.5547

Confusion Matrix:
[[ 70  32  19  28  37]
 [ 15 162   5   2   3]
 [  5   4 134  32  11]
 [ 12   4  30 108  33]
 [ 17  18   8  40 104]]

Metrics saved to: outputs/exp05_classification_AlexNet/run_20260426_024109/logs/evaluation_metrics.json
Detailed report saved to: outputs/exp05_classification_AlexNet/run_20260426_024109/logs/classification_report.txt
Report saved to: outputs/exp05_classification_AlexNet/run_20260426_024109/logs/experiment_report.md
2026-04-26 03:28:58 - exp05_classification_AlexNet - INFO - 
================================================================================
2026-04-26 03:28:58 - exp05_classification_AlexNet - INFO - EXPERIMENT COMPLETED SUCCESSFULLY
2026-04-26 03:28:58 - exp05_classification_AlexNet - INFO - ================================================================================
2026-04-26 03:28:58 - exp05_classification_AlexNet - INFO - Results saved to: outputs/exp05_classification_AlexNet/run_20260426_024109
2026-04-26 03:28:58 - exp05_classification_AlexNet - INFO - Best Validation Accuracy: 0.5979
2026-04-26 03:28:58 - exp05_classification_AlexNet - INFO - Test Accuracy: 0.6195
2026-04-26 03:28:58 - exp05_classification_AlexNet - INFO - Test Precision: 0.6151
2026-04-26 03:28:58 - exp05_classification_AlexNet - INFO - Test Recall: 0.6195
2026-04-26 03:28:58 - exp05_classification_AlexNet - INFO - Test F1-Score: 0.6111