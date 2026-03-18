```python
import os
import sys
import platform

if platform.system() == "Linux":
    os.environ["BACKBONE_DIR"] = "/home/namu/myspace/NAMU/backbones"
    os.environ["DATASET_DIR"] = "/home/namu/myspace/NAMU/datasets"
    PROJECT_DIR = "/home/namu/myspace/NAMU/defectvad_rev"
    SOURCE_DIR = "/home/namu/myspace/NAMU/defectvad_rev/src"
else:
    os.environ["BACKBONE_DIR"] = "d:\\Non_Documents\\backbones"
    os.environ["DATASET_DIR"] = "e:\\datasets"
    PROJECT_DIR = "d:\\Non_Documents\\2026\\defectvad_rev"
    SOURCE_DIR = "d:\\Non_Documents\\2026\\defectvad_rev\\src"

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)
```

```python
from defectvad.common.utils import load_weights
from defectvad.common.mvtec import get_dataloader
from defectvad.common.evaluator import Evaluator
from defectvad.common.predictor import Predictor
from defectvad.common.visualizer import Visualizer

BATCH_SIZE = 16
IMG_SIZE = 256
CROP_SIZE = None
NORMALIZE = True
MAX_EPOCHS = 10
VALIDATE = True

from defectvad.models.stfpm.torch_model import STFPMModel

# resnet18 / resnet50
model = STFPMModel(
backbone="resnet18",
layers=["layer1", "layer2", "layer3"],
)
EXPERIMENT_NAME = "mvtec_01_stfpm_resnet18"

OUTPUT_DIR = "D:\\Non_Documents\\2026\\defectvad_rev\\tests\mvtec1\\outputs"
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, EXPERIMENT_NAME, f"{EXPERIMENT_NAME}.pth")
NUM_SAMPLES = 5

load_weights(model, weights_path=WEIGHTS_PATH)
```

```python
def show_anomaly(category, num_samples=5):
    LOADER_KWARGS = dict(
        data_dir=os.path.join(os.environ["DATASET_DIR"], "mvtec"),
        img_size=IMG_SIZE,
        crop_size=CROP_SIZE,
        normalize=NORMALIZE,
    )
    test_loader = get_dataloader("test", category=category, batch_size=1, **LOADER_KWARGS)
    image_results = Evaluator(model).evaluate_image_level(test_loader)
    print(f">> {category}: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))

    preds = Predictor(model).predict(test_loader)
    visualizer = Visualizer(preds)
    visualizer.show_anomaly(max_samples=num_samples, denormalize=NORMALIZE)
```

```python
show_anomaly("carpet", num_samples=NUM_SAMPLES)
show_anomaly("grid", num_samples=NUM_SAMPLES)
show_anomaly("leather", num_samples=NUM_SAMPLES)
show_anomaly("tile", num_samples=NUM_SAMPLES)
show_anomaly("wood", num_samples=NUM_SAMPLES)
```
