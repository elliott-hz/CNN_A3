# Fix Plan for exp07_detection_rcnn

## Issues Found
1. **Only runs 1 epoch**: `src/models/rcnn_detection_model.py` has debug `break` statements that force early exit after epoch 0.
2. **detection_evaluator.py can't run**: It uses `model.model.val()` which is YOLOv8-only, incompatible with RCNNDetector.

## Steps
- [x] Step 1: Fix rcnn_detection_model.py — remove debug breaks and fix loss averaging
- [x] Step 2: Fix detection_evaluator.py — add RCNN-compatible evaluation logic
- [ ] Step 3: Test run to verify fixes work

