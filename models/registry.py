"""Single source of truth for model IDs and version pins.

Every model call in the codebase goes through modules in models/ so that a
preview-API change (rfdetr keypoint schema, inference SDK) touches one place.
"""

# Detector v0: RF-DETR Small bootstrap, trained on the forked UArizona
# "Basketball Shooting Robot" dataset in the owner's workspace. Trained on
# Roboflow ONCE, outside this repo — never retrain or relaunch it; a weak M1
# result feeds the M6 fine-tune instead.
DETECTOR_V0_MODEL_ID = "aarnavs-space/basketball-shooting-robot-kbsro-1-rfdetr-small-t1"

# The source dataset also carries `made`, `shoot`, `person`, etc. The engine
# consumes only these two classes; everything else is dropped at the adapter.
DETECTOR_CLASSES = ("ball", "rim")

# Detector v0 runs locally through the `inference` runtime (ONNX), loaded by
# model ID. Installed and pinned at M1:
INFERENCE_VERSION_PIN = "1.3.10"

# Pose: RF-DETR Keypoint (RFDETRKeypointPreview), pretrained COCO-17
# checkpoint, zero-shot. rfdetr is a preview API — keep it pinned. Keypoint
# ordering VERIFIED EMPIRICALLY on 1.9.2 (indexed overlay on real footage,
# 2026-08-11): standard COCO-17 order, indices below. Re-verify the overlay
# if this pin ever moves (the schema default changed around 1.8.x).
RFDETR_VERSION_PIN = "1.9.2"

COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
