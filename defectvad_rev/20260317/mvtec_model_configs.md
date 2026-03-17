## DefectVAD Model Configs

#### 1. STFPM

```python
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.stfpm.torch_model import STFPMModel
    from defectvad.models.stfpm.torch_trainer import STFPMTrainer

    # resnet18 / resnet50
    model = STFPMModel(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )
    trainer = STFPMTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 2. Reverse Distillation

```python
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.reversedistill.torch_model import ReverseDistillationModel
    from defectvad.models.reversedistill.torch_trainer import ReverseDistillTrainer

    # resnet18 / wide_resnet50_2
    model = ReverseDistillationModel(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            input_size=(256, 256),
            anomaly_map_mode="add",
            pre_trained=True,
    )
    trainer = ReverseDistillTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 3. EfficientAD

```python
    BATCH_SIZE = 1      # EfficientAd!!!
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = False   # EfficientAd!!!
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.efficientad.torch_model import EfficientAdModel
    from defectvad.models.efficientad.torch_trainer import EfficientAdTrainer

    # small / medium
    model = EfficientAdModel(
            teacher_out_channels=384,
            model_size="small",
            padding=False,
            pad_maps=True,
    )
    trainer = EfficientAdTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 4. CFlow

```python
    BATCH_SIZE = 8
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True
    MAX_EPOCHS = 1      # test for single category
    VALIDATE = True

    from defectvad.models.cflow.torch_model import CflowModel
    from defectvad.models.cflow.torch_trainer import CflowTrainer

    # resnet18 / wide_resnet50_2
    model = CflowModel(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"], 
            fiber_batch_size=64, 
            decoder="freia-cflow", 
            condition_vector=128, 
            coupling_blocks=8,
            clamp_alpha=1.9, 
            permute_soft=False,
    )
    trainer = CflowTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 5. FastFlow

```python
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.fastflow.torch_model import FastflowModel
    from defectvad.models.fastflow.torch_trainer import FastflowTrainer

    # resnet18 / wide_resnet50_2 / cait / deit
    model = FastflowModel(
            input_size=(256, 256),
            backbone="resnet18",
            pre_trained=True,
            flow_steps=8,
            conv3x3_only=False,
            hidden_ratio=1.0,
    )
    trainer = FastflowTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 6. CSFlow

```python
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.csflow.torch_model import CsFlowModel
    from defectvad.models.csflow.torch_trainer import CsflowTrainer

    # efficientnet_b5
    model = CsFlowModel(
            input_size=(256, 256),
            cross_conv_hidden_channels=1024,
            n_coupling_blocks=4,
            clamp=3,
            num_channels=3,
    )
    trainer = CsflowTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 7. UFlow

```python
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.uflow.torch_model import UflowModel
    from defectvad.models.uflow.torch_trainer import UflowTrainer

    # mcait: (448, 448)
    # resnet18 / wide_resnet50_2: (256, 256)
    model = UflowModel(
            input_size=(256, 256),
            backbone="wide_resnet50_2",
            flow_steps=4,
            affine_clamp=2.0,
            affine_subnet_channels_ratio=1.0,
            permute_soft=False,
    )
    trainer = UflowTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 8. Patchcore

```python
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = 224
    NORMALIZE = True
    MAX_EPOCHS = 1      # memory-based model
    VALIDATE = False    # memory-based model

    from defectvad.models.patchcore.torch_model import PatchcoreModel
    from defectvad.models.patchcore.torch_trainer import PatchcoreTrainer

    # resnet18 / wide_resnet50_2
    model = PatchcoreModel(
            backbone="resnet18",
            pre_trained=True,
            layers=["layer2", "layer3"],
            num_neighbors=9,
    )
    trainer = PatchcoreTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 9. PaDIM

```python
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = 224
    NORMALIZE = True
    MAX_EPOCHS = 1      # memory-based model
    VALIDATE = False    # memory-based model

    from defectvad.models.padim.torch_model import PadimModel
    from defectvad.models.padim.torch_trainer import PadimTrainer

    # resnet18 / wide_resnet50_2
    model = PadimModel(
            backbone="resnet18", 
            layers=["layer1", "layer2", "layer3"], 
            n_features=None,
    )
    trainer = PadimTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 10. CFA

```python
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = 224
    NORMALIZE = True
    MAX_EPOCHS = 1      # memory-based model
    VALIDATE = False    # memory-based model

    from defectvad.models.cfa.torch_model import CfaModel
    from defectvad.models.cfa.torch_trainer import CfaTrainer

    # vgg19_bn, resnet18, wide_resnet50_2, efficientnet_b5
    model = CfaModel(
            backbone="wide_resnet50_2",
            gamma_c=1,
            gamma_d=2,
            num_nearest_neighbors=3,
            num_hard_negative_features=3,
            radius=0.5,     # 1e-5
    )
    trainer = CfaTrainer(model, evaluator=Evaluator(model) if VALIDATE else None, radius=0.5)
```

#### 11. DFM

```python
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = 224
    NORMALIZE = True
    MAX_EPOCHS = 1      # memory-based model
    VALIDATE = False    # memory-based model

    from defectvad.models.dfm.torch_model import DFMModel
    from defectvad.models.dfm.torch_trainer import DFMTrainer

    # resnet18 / resnet50
    model = DFMModel(
            backbone="resnet18",
            layer="layer3", 
            pooling_kernel_size=4, 
            n_comps=0.97,
            score_type="fre",
    )
    trainer = DFMTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 12. DKKDE

```python
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = 224
    NORMALIZE = True
    MAX_EPOCHS = 1      # memory-based model
    VALIDATE = False    # memory-based model

    from defectvad.models.dfkde.torch_model import DfkdeModel
    from defectvad.models.dfkde.torch_trainer import DfkdeTrainer

    # resnet18 / resnet50
    model = DfkdeModel(
            backbone="resnet18", 
            layers=["layer4"], 
            n_pca_components=16, 
            feature_scaling_method="scale", 
            max_training_points=40000,
    )
    trainer = DfkdeTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 13. GANomaly

```python
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.ganomaly.torch_model import GanomalyModel
    from defectvad.models.ganomaly.torch_trainer import GanomalyTrainer

    model = GanomalyModel(
            input_size=(IMG_SIZE, IMG_SIZE), 
            num_input_channels=3, 
            n_features=64, 
            latent_vec_size=100,
            extra_layers=0, 
            add_final_conv_layer=True,
    )
    trainer = GanomalyTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 14. FRE

```python
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.fre.torch_model import FREModel
    from defectvad.models.fre.torch_trainer import FRETrainer

    # resnet18(16384), resnet50 (65536)
    model = FREModel(
        backbone="resnet18", 
        layer="layer3", 
        pooling_kernel_size=2,
        input_dim=16384,
        latent_dim=220,
    )
    trainer = FRETrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 15. DRAEM

```python
    BATCH_SIZE = 4      # DRAEM!!!
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = False   # DRAEM!!!
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.draem.torch_model import DraemModel
    from defectvad.models.draem.torch_trainer import DraemTrainer

    model = DraemModel(sspcab=False)
    trainer = DraemTrainer(model, evaluator=Evaluator(model) if VALIDATE else None, enable_sspcab=False)
```

#### 16. DSR

```python
BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = False   # DSR!!!
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.dsr.torch_model import DsrModel
    from defectvad.models.dsr.torch_trainer import DsrTrainer

    model = DsrModel(latent_anomaly_strength=0.2)
    trainer = DsrTrainer(model, evaluator=Evaluator(model) if VALIDATE else None, upsampling_train_ratio=0.7)
```

#### 17. SupersimpleNet

```python
    BATCH_SIZE = 8
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.supersimplenet.torch_model import SupersimplenetModel
    from defectvad.models.supersimplenet.torch_trainer import SupersimplenetTrainer

    # wide_resnet50_2
    supervised = False
    model = SupersimplenetModel(
            perlin_threshold=0.2,
            backbone="wide_resnet50_2.tv_in1k",
            layers=["layer2", "layer3"],
            # stop_grad=False if supervised else True,
            stop_grad=False,
            adapt_cls_features=False,
    )
    trainer = SupersimplenetTrainer(model, evaluator=Evaluator(model) if VALIDATE else None, supervised=supervised)
```

#### 18. UniNet

```python
    BATCH_SIZE = 4
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.uninet.torch_model import UniNetModel
    from defectvad.models.uninet.torch_trainer import UniNetTrainer
    from defectvad.models.uninet.loss import UniNetLoss

    # wide_resnet50_2
    model = UniNetModel(
            student_backbone="wide_resnet50_2",
            teacher_backbone="wide_resnet50_2",
            loss=UniNetLoss(temperature=0.1)
    )
    trainer = UniNetTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 19. Dinomaly

```python
    BATCH_SIZE = 8
    IMG_SIZE = 448      # Dinomaly!!!
    CROP_SIZE = 392     # Dinomaly!!!
    NORMALIZE = True
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.dinomaly.torch_model import DinomalyModel
    from defectvad.models.dinomaly.torch_trainer import DinomalyTrainer

    # dinov2reg_vit_small_14, dinov2_vit_small_14
    # dinov2reg_vit_base_14, dinov2_vit_base_14
    # dinov2reg_vit_large_14, dinov2_vit_large_14
    model = DinomalyModel(
            encoder_name="dinov2reg_vit_small_14",
            bottleneck_dropout=0.2,
            decoder_depth=8,
            target_layers=None,
            fuse_layer_encoder=None,
            fuse_layer_decoder=None,
            remove_class_token=False,
    )
    trainer = DinomalyTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```

#### 20. AnomalyDINO

```python
    BATCH_SIZE = 8
    IMG_SIZE = 256
    CROP_SIZE = 224
    NORMALIZE = False   # AnomalyDINO!!!
    MAX_EPOCHS = 1      # memory-based model
    VALIDATE = False    # memory-based model

    from defectvad.models.anomalydino.torch_model import AnomalyDINOModel
    from defectvad.models.anomalydino.torch_trainer import AnomalyDINOTrainer

    # dinov2_vit_small_14 / dinov2_vit_base_14 / # dinov2_vit_large_14 
    model = AnomalyDINOModel(
        encoder_name="dinov2_vit_small_14",
        num_neighbours=1,
        masking=False,       # default: False
        coreset_subsampling=False,
        sampling_ratio=0.1
    )
    trainer = AnomalyDINOTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)
```
