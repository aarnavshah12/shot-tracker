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

# Pose: RF-DETR Keypoint, pretrained COCO-17 checkpoint, zero-shot (M3).
# rfdetr is a preview API — the exact pin is chosen when the package is
# installed at M3 and recorded here. NOTE: the keypoint schema default
# changed to active-first in 1.8.x; verify which side of that line the pin
# lands on before parsing keypoints.
RFDETR_VERSION_PIN: str | None = None  # set at install time, e.g. "1.8.1"
