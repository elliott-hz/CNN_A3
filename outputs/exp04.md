(.venv) (base) sagemaker-user@default:~/CNN_A3$ python experiments/exp04_classification_baseline.py 
2026-04-26 01:56:56 - exp04_classification_baseline - INFO - ================================================================================
2026-04-26 01:56:56 - exp04_classification_baseline - INFO - STARTING EXPERIMENT: exp04_classification_baseline
2026-04-26 01:56:56 - exp04_classification_baseline - INFO - ================================================================================
2026-04-26 01:56:56 - exp04_classification_baseline - INFO - 
[Step 1/4] Loading preprocessed data...
2026-04-26 01:57:48 - exp04_classification_baseline - INFO - Data loaded successfully:
2026-04-26 01:57:48 - exp04_classification_baseline - INFO -   Train: 6527 samples, shape: (224, 224, 3)
2026-04-26 01:57:48 - exp04_classification_baseline - INFO -   Valid: 1865 samples, shape: (224, 224, 3)
2026-04-26 01:57:48 - exp04_classification_baseline - INFO -   Test: 933 samples, shape: (224, 224, 3)
2026-04-26 01:57:48 - exp04_classification_baseline - INFO - 
[Step 2/4] Initializing model and trainer...
2026-04-26 01:57:48 - exp04_classification_baseline - INFO - Model config: {'num_classes': 5, 'dropout_rate': 0.5, 'pretrained': True, 'freeze_backbone': True, 'additional_fc_layers': False, 'use_batch_norm': True}
2026-04-26 01:57:48 - exp04_classification_baseline - INFO - Training config: {'learning_rate': 0.0005, 'batch_size': 32, 'epochs': 120, 'optimizer': 'adam', 'weight_decay': 0.0005, 'early_stopping_patience': 0, 'use_amp': True, 'gradient_accumulation_steps': 1, 'label_smoothing': 0.1, 'class_weighting': True}
Experiment directory created: outputs/exp04_classification_baseline/run_20260426_015748
Using device: cuda
2026-04-26 01:57:50 - exp04_classification_baseline - INFO - 
[Step 3/4] Training model...
================================================================================
CLASSIFICATION MODEL TRAINING
================================================================================
Model config: {'num_classes': 5, 'dropout_rate': 0.5, 'pretrained': True, 'freeze_backbone': True, 'additional_fc_layers': False, 'use_batch_norm': True}
Training config: {'learning_rate': 0.0005, 'batch_size': 32, 'epochs': 120, 'optimizer': 'adam', 'weight_decay': 0.0005, 'early_stopping_patience': 0, 'use_amp': True, 'gradient_accumulation_steps': 1, 'label_smoothing': 0.1, 'class_weighting': True}
Output directory: outputs/exp04_classification_baseline/run_20260426_015748
Class weights: [    0.99954      1.0003     0.99954      1.0003      1.0003]

================================================================================
PHASE 1: Training with frozen backbone
================================================================================
Training Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.68it/s, loss=1.4631, acc=40.15%]
[Phase 1 (Frozen)] Epoch 1/10 | Train Loss: 1.4631 | Train Acc: 0.4015 | Val Loss: 1.3924 | Val Acc: 0.4563
  ✓ New best model saved (Val Acc: 0.4563)
Training Epoch 2: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.45it/s, loss=1.3352, acc=49.09%]
[Phase 1 (Frozen)] Epoch 2/10 | Train Loss: 1.3352 | Train Acc: 0.4909 | Val Loss: 1.3232 | Val Acc: 0.5046
  ✓ New best model saved (Val Acc: 0.5046)
Training Epoch 3: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.34it/s, loss=1.2853, acc=51.37%]
[Phase 1 (Frozen)] Epoch 3/10 | Train Loss: 1.2853 | Train Acc: 0.5137 | Val Loss: 1.2837 | Val Acc: 0.5217
  ✓ New best model saved (Val Acc: 0.5217)
Training Epoch 4: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.58it/s, loss=1.2549, acc=53.46%]
[Phase 1 (Frozen)] Epoch 4/10 | Train Loss: 1.2549 | Train Acc: 0.5346 | Val Loss: 1.2602 | Val Acc: 0.5282
  ✓ New best model saved (Val Acc: 0.5282)
Training Epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:25<00:00,  8.05it/s, loss=1.2594, acc=53.85%]
[Phase 1 (Frozen)] Epoch 5/10 | Train Loss: 1.2594 | Train Acc: 0.5385 | Val Loss: 1.2543 | Val Acc: 0.5362
  ✓ New best model saved (Val Acc: 0.5362)
Training Epoch 6: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.46it/s, loss=1.2428, acc=53.69%]
[Phase 1 (Frozen)] Epoch 6/10 | Train Loss: 1.2428 | Train Acc: 0.5369 | Val Loss: 1.2481 | Val Acc: 0.5394
  ✓ New best model saved (Val Acc: 0.5394)
Training Epoch 7: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:23<00:00,  8.56it/s, loss=1.2324, acc=55.57%]
[Phase 1 (Frozen)] Epoch 7/10 | Train Loss: 1.2324 | Train Acc: 0.5557 | Val Loss: 1.2363 | Val Acc: 0.5464
  ✓ New best model saved (Val Acc: 0.5464)
Training Epoch 8: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.43it/s, loss=1.2309, acc=54.45%]
[Phase 1 (Frozen)] Epoch 8/10 | Train Loss: 1.2309 | Train Acc: 0.5445 | Val Loss: 1.2352 | Val Acc: 0.5442
Training Epoch 9: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.35it/s, loss=1.2162, acc=55.62%]
[Phase 1 (Frozen)] Epoch 9/10 | Train Loss: 1.2162 | Train Acc: 0.5562 | Val Loss: 1.2232 | Val Acc: 0.5512
  ✓ New best model saved (Val Acc: 0.5512)
Training Epoch 10: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:24<00:00,  8.16it/s, loss=1.2168, acc=55.48%]
[Phase 1 (Frozen)] Epoch 10/10 | Train Loss: 1.2168 | Train Acc: 0.5548 | Val Loss: 1.2209 | Val Acc: 0.5517
  ✓ New best model saved (Val Acc: 0.5517)

================================================================================
PHASE 2: Fine-tuning with unfrozen backbone
================================================================================
Training Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:29<00:00,  6.99it/s, loss=1.1917, acc=57.64%]
[Phase 2 (Fine-tune)] Epoch 1/110 | Train Loss: 1.1917 | Train Acc: 0.5764 | Val Loss: 1.1707 | Val Acc: 0.5737
  ✓ New best model saved (Val Acc: 0.5737)
Training Epoch 2: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:31<00:00,  6.46it/s, loss=1.1574, acc=59.58%]
[Phase 2 (Fine-tune)] Epoch 2/110 | Train Loss: 1.1574 | Train Acc: 0.5958 | Val Loss: 1.1323 | Val Acc: 0.5930
  ✓ New best model saved (Val Acc: 0.5930)
Training Epoch 3: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:29<00:00,  6.79it/s, loss=1.1131, acc=61.67%]
[Phase 2 (Fine-tune)] Epoch 3/110 | Train Loss: 1.1131 | Train Acc: 0.6167 | Val Loss: 1.0972 | Val Acc: 0.6129
  ✓ New best model saved (Val Acc: 0.6129)
Training Epoch 4: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:30<00:00,  6.68it/s, loss=1.0922, acc=63.13%]
[Phase 2 (Fine-tune)] Epoch 4/110 | Train Loss: 1.0922 | Train Acc: 0.6313 | Val Loss: 1.0676 | Val Acc: 0.6252
  ✓ New best model saved (Val Acc: 0.6252)
Training Epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:29<00:00,  6.78it/s, loss=1.0608, acc=64.35%]
[Phase 2 (Fine-tune)] Epoch 5/110 | Train Loss: 1.0608 | Train Acc: 0.6435 | Val Loss: 1.0426 | Val Acc: 0.6408
  ✓ New best model saved (Val Acc: 0.6408)
Training Epoch 6: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:30<00:00,  6.62it/s, loss=1.0451, acc=65.56%]
[Phase 2 (Fine-tune)] Epoch 6/110 | Train Loss: 1.0451 | Train Acc: 0.6556 | Val Loss: 1.0259 | Val Acc: 0.6499
  ✓ New best model saved (Val Acc: 0.6499)
Training Epoch 7: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.25it/s, loss=1.0210, acc=66.87%]
[Phase 2 (Fine-tune)] Epoch 7/110 | Train Loss: 1.0210 | Train Acc: 0.6687 | Val Loss: 1.0080 | Val Acc: 0.6590
  ✓ New best model saved (Val Acc: 0.6590)
Training Epoch 8: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.46it/s, loss=1.0100, acc=68.04%]
[Phase 2 (Fine-tune)] Epoch 8/110 | Train Loss: 1.0100 | Train Acc: 0.6804 | Val Loss: 1.0000 | Val Acc: 0.6686
  ✓ New best model saved (Val Acc: 0.6686)
Training Epoch 9: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.43it/s, loss=0.9984, acc=68.04%]
[Phase 2 (Fine-tune)] Epoch 9/110 | Train Loss: 0.9984 | Train Acc: 0.6804 | Val Loss: 0.9870 | Val Acc: 0.6735
  ✓ New best model saved (Val Acc: 0.6735)
Training Epoch 10: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.66it/s, loss=0.9847, acc=69.49%]
[Phase 2 (Fine-tune)] Epoch 10/110 | Train Loss: 0.9847 | Train Acc: 0.6949 | Val Loss: 0.9775 | Val Acc: 0.6799
  ✓ New best model saved (Val Acc: 0.6799)
Training Epoch 11: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.32it/s, loss=0.9694, acc=70.37%]
[Phase 2 (Fine-tune)] Epoch 11/110 | Train Loss: 0.9694 | Train Acc: 0.7037 | Val Loss: 0.9679 | Val Acc: 0.6912
  ✓ New best model saved (Val Acc: 0.6912)
Training Epoch 12: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.12it/s, loss=0.9527, acc=70.49%]
[Phase 2 (Fine-tune)] Epoch 12/110 | Train Loss: 0.9527 | Train Acc: 0.7049 | Val Loss: 0.9626 | Val Acc: 0.6863
Training Epoch 13: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.12it/s, loss=0.9499, acc=71.18%]
[Phase 2 (Fine-tune)] Epoch 13/110 | Train Loss: 0.9499 | Train Acc: 0.7118 | Val Loss: 0.9584 | Val Acc: 0.6981
  ✓ New best model saved (Val Acc: 0.6981)
Training Epoch 14: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.33it/s, loss=0.9301, acc=72.23%]
[Phase 2 (Fine-tune)] Epoch 14/110 | Train Loss: 0.9301 | Train Acc: 0.7223 | Val Loss: 0.9555 | Val Acc: 0.6954
Training Epoch 15: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.18it/s, loss=0.9222, acc=72.78%]
[Phase 2 (Fine-tune)] Epoch 15/110 | Train Loss: 0.9222 | Train Acc: 0.7278 | Val Loss: 0.9505 | Val Acc: 0.7013
  ✓ New best model saved (Val Acc: 0.7013)
Training Epoch 16: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.52it/s, loss=0.9074, acc=73.45%]
[Phase 2 (Fine-tune)] Epoch 16/110 | Train Loss: 0.9074 | Train Acc: 0.7345 | Val Loss: 0.9484 | Val Acc: 0.7067
  ✓ New best model saved (Val Acc: 0.7067)
Training Epoch 17: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.42it/s, loss=0.9031, acc=74.01%]
[Phase 2 (Fine-tune)] Epoch 17/110 | Train Loss: 0.9031 | Train Acc: 0.7401 | Val Loss: 0.9409 | Val Acc: 0.7072
  ✓ New best model saved (Val Acc: 0.7072)
Training Epoch 18: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.35it/s, loss=0.8866, acc=74.85%]
[Phase 2 (Fine-tune)] Epoch 18/110 | Train Loss: 0.8866 | Train Acc: 0.7485 | Val Loss: 0.9407 | Val Acc: 0.7078
  ✓ New best model saved (Val Acc: 0.7078)
Training Epoch 19: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.68it/s, loss=0.8838, acc=75.17%]
[Phase 2 (Fine-tune)] Epoch 19/110 | Train Loss: 0.8838 | Train Acc: 0.7517 | Val Loss: 0.9407 | Val Acc: 0.7051
Training Epoch 20: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.33it/s, loss=0.8747, acc=75.37%]
[Phase 2 (Fine-tune)] Epoch 20/110 | Train Loss: 0.8747 | Train Acc: 0.7537 | Val Loss: 0.9371 | Val Acc: 0.7158
  ✓ New best model saved (Val Acc: 0.7158)
Training Epoch 21: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.47it/s, loss=0.8661, acc=76.29%]
  Learning rate reduced to 0.000035
[Phase 2 (Fine-tune)] Epoch 21/110 | Train Loss: 0.8661 | Train Acc: 0.7629 | Val Loss: 0.9378 | Val Acc: 0.7185
  ✓ New best model saved (Val Acc: 0.7185)
Training Epoch 22: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.24it/s, loss=0.8548, acc=76.12%]
[Phase 2 (Fine-tune)] Epoch 22/110 | Train Loss: 0.8548 | Train Acc: 0.7612 | Val Loss: 0.9386 | Val Acc: 0.7169
Training Epoch 23: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.37it/s, loss=0.8495, acc=76.82%]
[Phase 2 (Fine-tune)] Epoch 23/110 | Train Loss: 0.8495 | Train Acc: 0.7682 | Val Loss: 0.9343 | Val Acc: 0.7174
Training Epoch 24: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.40it/s, loss=0.8409, acc=77.57%]
[Phase 2 (Fine-tune)] Epoch 24/110 | Train Loss: 0.8409 | Train Acc: 0.7757 | Val Loss: 0.9330 | Val Acc: 0.7137
Training Epoch 25: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.48it/s, loss=0.8359, acc=77.37%]
[Phase 2 (Fine-tune)] Epoch 25/110 | Train Loss: 0.8359 | Train Acc: 0.7737 | Val Loss: 0.9348 | Val Acc: 0.7137
Training Epoch 26: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.44it/s, loss=0.8269, acc=78.00%]
[Phase 2 (Fine-tune)] Epoch 26/110 | Train Loss: 0.8269 | Train Acc: 0.7800 | Val Loss: 0.9373 | Val Acc: 0.7126
Training Epoch 27: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.47it/s, loss=0.8179, acc=78.51%]
[Phase 2 (Fine-tune)] Epoch 27/110 | Train Loss: 0.8179 | Train Acc: 0.7851 | Val Loss: 0.9345 | Val Acc: 0.7131
Training Epoch 28: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.20it/s, loss=0.8243, acc=78.49%]
[Phase 2 (Fine-tune)] Epoch 28/110 | Train Loss: 0.8243 | Train Acc: 0.7849 | Val Loss: 0.9379 | Val Acc: 0.7185
Training Epoch 29: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.43it/s, loss=0.8094, acc=78.96%]
[Phase 2 (Fine-tune)] Epoch 29/110 | Train Loss: 0.8094 | Train Acc: 0.7896 | Val Loss: 0.9357 | Val Acc: 0.7185
Training Epoch 30: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.64it/s, loss=0.8055, acc=79.33%]
[Phase 2 (Fine-tune)] Epoch 30/110 | Train Loss: 0.8055 | Train Acc: 0.7933 | Val Loss: 0.9371 | Val Acc: 0.7137
Training Epoch 31: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.49it/s, loss=0.8041, acc=79.54%]
[Phase 2 (Fine-tune)] Epoch 31/110 | Train Loss: 0.8041 | Train Acc: 0.7954 | Val Loss: 0.9369 | Val Acc: 0.7169
Training Epoch 32: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.45it/s, loss=0.7922, acc=80.06%]
[Phase 2 (Fine-tune)] Epoch 32/110 | Train Loss: 0.7922 | Train Acc: 0.8006 | Val Loss: 0.9357 | Val Acc: 0.7180
Training Epoch 33: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.46it/s, loss=0.7983, acc=79.53%]
[Phase 2 (Fine-tune)] Epoch 33/110 | Train Loss: 0.7983 | Train Acc: 0.7953 | Val Loss: 0.9371 | Val Acc: 0.7153
Training Epoch 34: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.55it/s, loss=0.7842, acc=80.82%]
[Phase 2 (Fine-tune)] Epoch 34/110 | Train Loss: 0.7842 | Train Acc: 0.8082 | Val Loss: 0.9358 | Val Acc: 0.7223
  ✓ New best model saved (Val Acc: 0.7223)
Training Epoch 35: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.65it/s, loss=0.7787, acc=80.39%]
[Phase 2 (Fine-tune)] Epoch 35/110 | Train Loss: 0.7787 | Train Acc: 0.8039 | Val Loss: 0.9361 | Val Acc: 0.7201
Training Epoch 36: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.24it/s, loss=0.7720, acc=81.73%]
[Phase 2 (Fine-tune)] Epoch 36/110 | Train Loss: 0.7720 | Train Acc: 0.8173 | Val Loss: 0.9385 | Val Acc: 0.7158
Training Epoch 37: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.37it/s, loss=0.7674, acc=81.76%]
[Phase 2 (Fine-tune)] Epoch 37/110 | Train Loss: 0.7674 | Train Acc: 0.8176 | Val Loss: 0.9353 | Val Acc: 0.7217
Training Epoch 38: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.60it/s, loss=0.7564, acc=82.48%]
[Phase 2 (Fine-tune)] Epoch 38/110 | Train Loss: 0.7564 | Train Acc: 0.8248 | Val Loss: 0.9379 | Val Acc: 0.7174
Training Epoch 39: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.23it/s, loss=0.7587, acc=82.00%]
[Phase 2 (Fine-tune)] Epoch 39/110 | Train Loss: 0.7587 | Train Acc: 0.8200 | Val Loss: 0.9407 | Val Acc: 0.7153
Training Epoch 40: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.41it/s, loss=0.7504, acc=82.57%]
[Phase 2 (Fine-tune)] Epoch 40/110 | Train Loss: 0.7504 | Train Acc: 0.8257 | Val Loss: 0.9383 | Val Acc: 0.7180
Training Epoch 41: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.39it/s, loss=0.7445, acc=82.50%]
  Learning rate reduced to 0.000024
[Phase 2 (Fine-tune)] Epoch 41/110 | Train Loss: 0.7445 | Train Acc: 0.8250 | Val Loss: 0.9420 | Val Acc: 0.7169
Training Epoch 42: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.21it/s, loss=0.7443, acc=82.48%]
[Phase 2 (Fine-tune)] Epoch 42/110 | Train Loss: 0.7443 | Train Acc: 0.8248 | Val Loss: 0.9426 | Val Acc: 0.7142
Training Epoch 43: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.10it/s, loss=0.7370, acc=83.74%]
[Phase 2 (Fine-tune)] Epoch 43/110 | Train Loss: 0.7370 | Train Acc: 0.8374 | Val Loss: 0.9427 | Val Acc: 0.7174
Training Epoch 44: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.25it/s, loss=0.7382, acc=83.64%]
[Phase 2 (Fine-tune)] Epoch 44/110 | Train Loss: 0.7382 | Train Acc: 0.8364 | Val Loss: 0.9457 | Val Acc: 0.7164
Training Epoch 45: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.29it/s, loss=0.7292, acc=83.84%]
[Phase 2 (Fine-tune)] Epoch 45/110 | Train Loss: 0.7292 | Train Acc: 0.8384 | Val Loss: 0.9478 | Val Acc: 0.7169
Training Epoch 46: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.63it/s, loss=0.7278, acc=83.90%]
[Phase 2 (Fine-tune)] Epoch 46/110 | Train Loss: 0.7278 | Train Acc: 0.8390 | Val Loss: 0.9459 | Val Acc: 0.7180
Training Epoch 47: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.29it/s, loss=0.7240, acc=84.22%]
[Phase 2 (Fine-tune)] Epoch 47/110 | Train Loss: 0.7240 | Train Acc: 0.8422 | Val Loss: 0.9471 | Val Acc: 0.7131
Training Epoch 48: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.26it/s, loss=0.7234, acc=84.08%]
[Phase 2 (Fine-tune)] Epoch 48/110 | Train Loss: 0.7234 | Train Acc: 0.8408 | Val Loss: 0.9499 | Val Acc: 0.7147
Training Epoch 49: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.53it/s, loss=0.7186, acc=84.53%]
[Phase 2 (Fine-tune)] Epoch 49/110 | Train Loss: 0.7186 | Train Acc: 0.8453 | Val Loss: 0.9495 | Val Acc: 0.7115
Training Epoch 50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.46it/s, loss=0.7059, acc=85.19%]
[Phase 2 (Fine-tune)] Epoch 50/110 | Train Loss: 0.7059 | Train Acc: 0.8519 | Val Loss: 0.9537 | Val Acc: 0.7126
Training Epoch 51: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.56it/s, loss=0.7024, acc=85.50%]
[Phase 2 (Fine-tune)] Epoch 51/110 | Train Loss: 0.7024 | Train Acc: 0.8550 | Val Loss: 0.9526 | Val Acc: 0.7131
Training Epoch 52: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.49it/s, loss=0.7076, acc=84.84%]
[Phase 2 (Fine-tune)] Epoch 52/110 | Train Loss: 0.7076 | Train Acc: 0.8484 | Val Loss: 0.9570 | Val Acc: 0.7051
Training Epoch 53: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.13it/s, loss=0.6985, acc=86.01%]
[Phase 2 (Fine-tune)] Epoch 53/110 | Train Loss: 0.6985 | Train Acc: 0.8601 | Val Loss: 0.9546 | Val Acc: 0.7158
Training Epoch 54: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.27it/s, loss=0.6899, acc=86.62%]
[Phase 2 (Fine-tune)] Epoch 54/110 | Train Loss: 0.6899 | Train Acc: 0.8662 | Val Loss: 0.9534 | Val Acc: 0.7105
Training Epoch 55: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.20it/s, loss=0.6970, acc=85.79%]
[Phase 2 (Fine-tune)] Epoch 55/110 | Train Loss: 0.6970 | Train Acc: 0.8579 | Val Loss: 0.9607 | Val Acc: 0.7094
Training Epoch 56: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.43it/s, loss=0.6961, acc=85.93%]
[Phase 2 (Fine-tune)] Epoch 56/110 | Train Loss: 0.6961 | Train Acc: 0.8593 | Val Loss: 0.9545 | Val Acc: 0.7153
Training Epoch 57: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.25it/s, loss=0.6847, acc=86.51%]
[Phase 2 (Fine-tune)] Epoch 57/110 | Train Loss: 0.6847 | Train Acc: 0.8651 | Val Loss: 0.9618 | Val Acc: 0.7067
Training Epoch 58: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.32it/s, loss=0.6843, acc=86.59%]
[Phase 2 (Fine-tune)] Epoch 58/110 | Train Loss: 0.6843 | Train Acc: 0.8659 | Val Loss: 0.9591 | Val Acc: 0.7137
Training Epoch 59: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.66it/s, loss=0.6805, acc=86.50%]
[Phase 2 (Fine-tune)] Epoch 59/110 | Train Loss: 0.6805 | Train Acc: 0.8650 | Val Loss: 0.9633 | Val Acc: 0.7131
Training Epoch 60: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:30<00:00,  6.69it/s, loss=0.6829, acc=86.87%]
[Phase 2 (Fine-tune)] Epoch 60/110 | Train Loss: 0.6829 | Train Acc: 0.8687 | Val Loss: 0.9667 | Val Acc: 0.7088
Training Epoch 61: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:30<00:00,  6.63it/s, loss=0.6782, acc=87.27%]
  Learning rate reduced to 0.000017
[Phase 2 (Fine-tune)] Epoch 61/110 | Train Loss: 0.6782 | Train Acc: 0.8727 | Val Loss: 0.9632 | Val Acc: 0.7056
Training Epoch 62: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:31<00:00,  6.49it/s, loss=0.6719, acc=87.72%]
[Phase 2 (Fine-tune)] Epoch 62/110 | Train Loss: 0.6719 | Train Acc: 0.8772 | Val Loss: 0.9695 | Val Acc: 0.7062
Training Epoch 63: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:29<00:00,  6.78it/s, loss=0.6651, acc=87.68%]
[Phase 2 (Fine-tune)] Epoch 63/110 | Train Loss: 0.6651 | Train Acc: 0.8768 | Val Loss: 0.9681 | Val Acc: 0.7088
Training Epoch 64: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:30<00:00,  6.59it/s, loss=0.6633, acc=87.96%]
[Phase 2 (Fine-tune)] Epoch 64/110 | Train Loss: 0.6633 | Train Acc: 0.8796 | Val Loss: 0.9767 | Val Acc: 0.7035
Training Epoch 65: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:29<00:00,  6.81it/s, loss=0.6698, acc=87.75%]
[Phase 2 (Fine-tune)] Epoch 65/110 | Train Loss: 0.6698 | Train Acc: 0.8775 | Val Loss: 0.9719 | Val Acc: 0.7056
Training Epoch 66: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.47it/s, loss=0.6628, acc=87.64%]
[Phase 2 (Fine-tune)] Epoch 66/110 | Train Loss: 0.6628 | Train Acc: 0.8764 | Val Loss: 0.9743 | Val Acc: 0.7062
Training Epoch 67: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.38it/s, loss=0.6595, acc=87.87%]
[Phase 2 (Fine-tune)] Epoch 67/110 | Train Loss: 0.6595 | Train Acc: 0.8787 | Val Loss: 0.9781 | Val Acc: 0.7072
Training Epoch 68: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.35it/s, loss=0.6680, acc=87.47%]
[Phase 2 (Fine-tune)] Epoch 68/110 | Train Loss: 0.6680 | Train Acc: 0.8747 | Val Loss: 0.9778 | Val Acc: 0.7067
Training Epoch 69: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.69it/s, loss=0.6556, acc=88.30%]
[Phase 2 (Fine-tune)] Epoch 69/110 | Train Loss: 0.6556 | Train Acc: 0.8830 | Val Loss: 0.9778 | Val Acc: 0.7083
Training Epoch 70: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.16it/s, loss=0.6525, acc=88.36%]
[Phase 2 (Fine-tune)] Epoch 70/110 | Train Loss: 0.6525 | Train Acc: 0.8836 | Val Loss: 0.9783 | Val Acc: 0.7094
Training Epoch 71: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.03it/s, loss=0.6494, acc=88.65%]
[Phase 2 (Fine-tune)] Epoch 71/110 | Train Loss: 0.6494 | Train Acc: 0.8865 | Val Loss: 0.9789 | Val Acc: 0.7099
Training Epoch 72: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.49it/s, loss=0.6484, acc=89.01%]
[Phase 2 (Fine-tune)] Epoch 72/110 | Train Loss: 0.6484 | Train Acc: 0.8901 | Val Loss: 0.9823 | Val Acc: 0.7062
Training Epoch 73: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.28it/s, loss=0.6495, acc=88.99%]
[Phase 2 (Fine-tune)] Epoch 73/110 | Train Loss: 0.6495 | Train Acc: 0.8899 | Val Loss: 0.9819 | Val Acc: 0.7013
Training Epoch 74: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.39it/s, loss=0.6428, acc=89.30%]
[Phase 2 (Fine-tune)] Epoch 74/110 | Train Loss: 0.6428 | Train Acc: 0.8930 | Val Loss: 0.9848 | Val Acc: 0.7051
Training Epoch 75: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.51it/s, loss=0.6461, acc=89.09%]
[Phase 2 (Fine-tune)] Epoch 75/110 | Train Loss: 0.6461 | Train Acc: 0.8909 | Val Loss: 0.9842 | Val Acc: 0.7056
Training Epoch 76: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.33it/s, loss=0.6425, acc=88.99%]
[Phase 2 (Fine-tune)] Epoch 76/110 | Train Loss: 0.6425 | Train Acc: 0.8899 | Val Loss: 0.9808 | Val Acc: 0.7110
Training Epoch 77: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.61it/s, loss=0.6392, acc=89.30%]
[Phase 2 (Fine-tune)] Epoch 77/110 | Train Loss: 0.6392 | Train Acc: 0.8930 | Val Loss: 0.9882 | Val Acc: 0.7126
Training Epoch 78: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.43it/s, loss=0.6337, acc=89.70%]
[Phase 2 (Fine-tune)] Epoch 78/110 | Train Loss: 0.6337 | Train Acc: 0.8970 | Val Loss: 0.9828 | Val Acc: 0.7051
Training Epoch 79: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.28it/s, loss=0.6389, acc=89.22%]
[Phase 2 (Fine-tune)] Epoch 79/110 | Train Loss: 0.6389 | Train Acc: 0.8922 | Val Loss: 0.9860 | Val Acc: 0.7078
Training Epoch 80: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.43it/s, loss=0.6254, acc=90.09%]
[Phase 2 (Fine-tune)] Epoch 80/110 | Train Loss: 0.6254 | Train Acc: 0.9009 | Val Loss: 0.9833 | Val Acc: 0.7056
Training Epoch 81: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.58it/s, loss=0.6296, acc=89.61%]
  Learning rate reduced to 0.000012
[Phase 2 (Fine-tune)] Epoch 81/110 | Train Loss: 0.6296 | Train Acc: 0.8961 | Val Loss: 0.9898 | Val Acc: 0.7062
Training Epoch 82: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.01it/s, loss=0.6258, acc=90.15%]
[Phase 2 (Fine-tune)] Epoch 82/110 | Train Loss: 0.6258 | Train Acc: 0.9015 | Val Loss: 0.9901 | Val Acc: 0.7024
Training Epoch 83: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.69it/s, loss=0.6254, acc=89.98%]
[Phase 2 (Fine-tune)] Epoch 83/110 | Train Loss: 0.6254 | Train Acc: 0.8998 | Val Loss: 0.9968 | Val Acc: 0.7024
Training Epoch 84: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.43it/s, loss=0.6241, acc=90.07%]
[Phase 2 (Fine-tune)] Epoch 84/110 | Train Loss: 0.6241 | Train Acc: 0.9007 | Val Loss: 0.9925 | Val Acc: 0.7088
Training Epoch 85: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.49it/s, loss=0.6303, acc=89.72%]
[Phase 2 (Fine-tune)] Epoch 85/110 | Train Loss: 0.6303 | Train Acc: 0.8972 | Val Loss: 0.9914 | Val Acc: 0.6997
Training Epoch 86: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.58it/s, loss=0.6258, acc=89.93%]
[Phase 2 (Fine-tune)] Epoch 86/110 | Train Loss: 0.6258 | Train Acc: 0.8993 | Val Loss: 0.9882 | Val Acc: 0.7040
Training Epoch 87: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.64it/s, loss=0.6209, acc=90.46%]
[Phase 2 (Fine-tune)] Epoch 87/110 | Train Loss: 0.6209 | Train Acc: 0.9046 | Val Loss: 0.9920 | Val Acc: 0.7046
Training Epoch 88: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.33it/s, loss=0.6140, acc=90.96%]
[Phase 2 (Fine-tune)] Epoch 88/110 | Train Loss: 0.6140 | Train Acc: 0.9096 | Val Loss: 0.9937 | Val Acc: 0.6971
Training Epoch 89: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.47it/s, loss=0.6156, acc=90.76%]
[Phase 2 (Fine-tune)] Epoch 89/110 | Train Loss: 0.6156 | Train Acc: 0.9076 | Val Loss: 0.9983 | Val Acc: 0.7024
Training Epoch 90: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.41it/s, loss=0.6187, acc=90.29%]
[Phase 2 (Fine-tune)] Epoch 90/110 | Train Loss: 0.6187 | Train Acc: 0.9029 | Val Loss: 0.9996 | Val Acc: 0.7008
Training Epoch 91: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.24it/s, loss=0.6139, acc=90.86%]
[Phase 2 (Fine-tune)] Epoch 91/110 | Train Loss: 0.6139 | Train Acc: 0.9086 | Val Loss: 0.9963 | Val Acc: 0.7121
Training Epoch 92: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.25it/s, loss=0.6090, acc=91.39%]
[Phase 2 (Fine-tune)] Epoch 92/110 | Train Loss: 0.6090 | Train Acc: 0.9139 | Val Loss: 0.9976 | Val Acc: 0.7067
Training Epoch 93: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.42it/s, loss=0.6101, acc=90.83%]
[Phase 2 (Fine-tune)] Epoch 93/110 | Train Loss: 0.6101 | Train Acc: 0.9083 | Val Loss: 0.9973 | Val Acc: 0.7035
Training Epoch 94: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.43it/s, loss=0.6095, acc=91.12%]
[Phase 2 (Fine-tune)] Epoch 94/110 | Train Loss: 0.6095 | Train Acc: 0.9112 | Val Loss: 0.9961 | Val Acc: 0.7024
Training Epoch 95: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.49it/s, loss=0.6085, acc=91.36%]
[Phase 2 (Fine-tune)] Epoch 95/110 | Train Loss: 0.6085 | Train Acc: 0.9136 | Val Loss: 1.0002 | Val Acc: 0.7013
Training Epoch 96: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.25it/s, loss=0.6085, acc=91.09%]
[Phase 2 (Fine-tune)] Epoch 96/110 | Train Loss: 0.6085 | Train Acc: 0.9109 | Val Loss: 0.9975 | Val Acc: 0.7056
Training Epoch 97: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.46it/s, loss=0.6030, acc=91.79%]
[Phase 2 (Fine-tune)] Epoch 97/110 | Train Loss: 0.6030 | Train Acc: 0.9179 | Val Loss: 1.0005 | Val Acc: 0.7046
Training Epoch 98: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.27it/s, loss=0.6140, acc=90.67%]
[Phase 2 (Fine-tune)] Epoch 98/110 | Train Loss: 0.6140 | Train Acc: 0.9067 | Val Loss: 0.9983 | Val Acc: 0.7024
Training Epoch 99: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.64it/s, loss=0.6075, acc=90.90%]
[Phase 2 (Fine-tune)] Epoch 99/110 | Train Loss: 0.6075 | Train Acc: 0.9090 | Val Loss: 1.0025 | Val Acc: 0.7067
Training Epoch 100: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.51it/s, loss=0.6010, acc=91.61%]
[Phase 2 (Fine-tune)] Epoch 100/110 | Train Loss: 0.6010 | Train Acc: 0.9161 | Val Loss: 1.0014 | Val Acc: 0.7035
Training Epoch 101: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.43it/s, loss=0.6062, acc=91.24%]
  Learning rate reduced to 0.000008
[Phase 2 (Fine-tune)] Epoch 101/110 | Train Loss: 0.6062 | Train Acc: 0.9124 | Val Loss: 0.9999 | Val Acc: 0.7067
Training Epoch 102: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.45it/s, loss=0.6003, acc=91.32%]
[Phase 2 (Fine-tune)] Epoch 102/110 | Train Loss: 0.6003 | Train Acc: 0.9132 | Val Loss: 1.0097 | Val Acc: 0.7003
Training Epoch 103: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.36it/s, loss=0.6038, acc=91.41%]
[Phase 2 (Fine-tune)] Epoch 103/110 | Train Loss: 0.6038 | Train Acc: 0.9141 | Val Loss: 1.0060 | Val Acc: 0.7029
Training Epoch 104: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.60it/s, loss=0.6008, acc=91.67%]
[Phase 2 (Fine-tune)] Epoch 104/110 | Train Loss: 0.6008 | Train Acc: 0.9167 | Val Loss: 1.0044 | Val Acc: 0.7024
Training Epoch 105: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.40it/s, loss=0.6014, acc=91.72%]
[Phase 2 (Fine-tune)] Epoch 105/110 | Train Loss: 0.6014 | Train Acc: 0.9172 | Val Loss: 1.0039 | Val Acc: 0.7024
Training Epoch 106: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.52it/s, loss=0.6078, acc=91.19%]
[Phase 2 (Fine-tune)] Epoch 106/110 | Train Loss: 0.6078 | Train Acc: 0.9119 | Val Loss: 1.0111 | Val Acc: 0.6971
Training Epoch 107: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.33it/s, loss=0.5997, acc=91.75%]
[Phase 2 (Fine-tune)] Epoch 107/110 | Train Loss: 0.5997 | Train Acc: 0.9175 | Val Loss: 1.0051 | Val Acc: 0.7029
Training Epoch 108: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:28<00:00,  7.11it/s, loss=0.5950, acc=92.07%]
[Phase 2 (Fine-tune)] Epoch 108/110 | Train Loss: 0.5950 | Train Acc: 0.9207 | Val Loss: 1.0063 | Val Acc: 0.7008
Training Epoch 109: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:26<00:00,  7.57it/s, loss=0.6035, acc=91.56%]
[Phase 2 (Fine-tune)] Epoch 109/110 | Train Loss: 0.6035 | Train Acc: 0.9156 | Val Loss: 1.0156 | Val Acc: 0.7003
Training Epoch 110: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 203/203 [00:27<00:00,  7.38it/s, loss=0.5979, acc=91.73%]
[Phase 2 (Fine-tune)] Epoch 110/110 | Train Loss: 0.5979 | Train Acc: 0.9173 | Val Loss: 1.0071 | Val Acc: 0.6965

================================================================================
TRAINING COMPLETE
================================================================================
Best validation accuracy: 0.7223
Best model saved to: outputs/exp04_classification_baseline/run_20260426_015748/model/best_model.pth
2026-04-26 02:58:32 - exp04_classification_baseline - INFO - Training completed successfully!
2026-04-26 02:58:32 - exp04_classification_baseline - INFO - 
[Step 4/4] Evaluating model on test set...
2026-04-26 02:58:32 - exp04_classification_baseline - INFO - Using class names: ['angry', 'happy', 'relax', 'frown', 'alert']
================================================================================
CLASSIFICATION MODEL EVALUATION
================================================================================

Overall Metrics:
  Accuracy: 0.7031
  Precision: 0.7052
  Recall: 0.7031
  F1-Score: 0.7034

Per-Class Metrics:
  angry:
    Precision: 0.6333
    Recall: 0.6129
    F1-Score: 0.6230
  happy:
    Precision: 0.8939
    Recall: 0.8556
    F1-Score: 0.8743
  relax:
    Precision: 0.7817
    Recall: 0.8280
    F1-Score: 0.8042
  frown:
    Precision: 0.6491
    Recall: 0.5936
    F1-Score: 0.6201
  alert:
    Precision: 0.5680
    Recall: 0.6257
    F1-Score: 0.5954

Confusion Matrix:
[[114  14  11  11  36]
 [ 21 160   0   2   4]
 [  2   1 154  18  11]
 [ 10   0  28 111  38]
 [ 33   4   4  29 117]]

Metrics saved to: outputs/exp04_classification_baseline/run_20260426_015748/logs/evaluation_metrics.json
Detailed report saved to: outputs/exp04_classification_baseline/run_20260426_015748/logs/classification_report.txt
Report saved to: outputs/exp04_classification_baseline/run_20260426_015748/logs/experiment_report.md
2026-04-26 02:58:34 - exp04_classification_baseline - INFO - 
================================================================================
2026-04-26 02:58:34 - exp04_classification_baseline - INFO - EXPERIMENT COMPLETED SUCCESSFULLY
2026-04-26 02:58:34 - exp04_classification_baseline - INFO - ================================================================================
2026-04-26 02:58:34 - exp04_classification_baseline - INFO - Results saved to: outputs/exp04_classification_baseline/run_20260426_015748
2026-04-26 02:58:34 - exp04_classification_baseline - INFO - Best Validation Accuracy: 0.7223
2026-04-26 02:58:34 - exp04_classification_baseline - INFO - Test Accuracy: 0.7031