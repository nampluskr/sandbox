## DefectVAD: Defect Vision Anomaly Detection
## 사전 생산 품질 평가를 위한 비지도 OLED 디스플레이 결함 검출 Hybrid Ensemble Framework

### 초록

OLED 개발 단계의 디스플레이 화질 검증 및 불량 판정은 휘도, 온도, 주파수 변화 등 다양한 구동 조건에서 주관적 육안 검사에 의존한다. 수동 검사는 상당한 시간과 노력이 필요하며, 검사자 간 편차로 인한 불량 유출 위험이 지속적으로 존재한다. 정상 샘플이 데이터셋을 압도하는 반면 불량은 예측 불가능한 형태, 위치, 심각도를 가진 희소한 분포를 보여 단일 임계값 기반 분류가 불안정하다. 본 논문은 사전 생산 OLED 품질 평가를 위한 현대적 unsupervised anomaly detection 모델들의 체계적 비교를 가능하게 하는 포괄적 평가 framework인 DefectVAD (Defect Vision Anomaly Detection)를 제시한다. MVTec AD, ViSA, BTAD benchmark 데이터셋에서 OLED 유사 카테고리를 선별하여 framework를 평가하고 98.9% AUROC를 달성했다. 제안된 three-stage cascaded hybrid ensemble은 상호보완적 검출 접근법을 통합하여 실제 OLED 개발 데이터에서 99.1% AUROC를 달성했으며, 이는 최고 개별 모델 대비 2.3%p의 절대 개선을 나타낸다. Framework는 XYZ colorimeter 측정 데이터에 대한 batch processing 방식으로 동작하며, 검사자 간 편차를 제거하고 검사 작업량을 줄여 엔지니어의 육안 검사를 지원한다. 정상 샘플만으로 학습된 이 시스템은 특정 불량 패턴과 독립적인 실용적 baseline을 제공하며, 다양한 콘텐츠와 휘도 조건에서 anomaly를 효과적으로 검출한다.

**키워드:** OLED display inspection, unsupervised anomaly detection, hybrid ensemble learning, visual quality assessment

---

### 1. 서론

#### 1.1. 배경 및 동기

디스플레이 화질 평가는 계측기로 측정된 물리적 특성 기반의 정량적 평가와 인간의 시각 인지 특성 기반의 주관적 평가로 구분된다[1]. 객관적 화질 측정은 실제 인간이 인지하는 화질과 종종 차이를 보이므로, 주관적 평가가 디스플레이 화질 측정에 더 적합한 것으로 알려져 있다. 특히 저계조 또는 저휘도 영역의 불량은 구동 환경과 관찰자에 따라 심각도 차이뿐 아니라 존재 유무 판단도 상이할 수 있다.

OLED 구동 화질 risk 검증 및 불량 개선을 위해서는 암실 조건, 고온/저온, dimming, 주파수 변화, 복합 killer 패턴을 포함한 다양한 환경, 기능, 패턴에 대한 광학적 특성 측정 및 육안 검사가 필요하다(Fig. 1). Panel의 이상점은 정규 패턴의 일정한 위치에 나타나는 예측 가능한 결함과 불규칙한 형태로 임의의 위치에 나타나는 예측 불가능한 결함으로 구분된다. 많은 결함이 저계조 영역에서 발생하여 검사자 간 얼룩 검출 편차로 인한 검증 시간 증가 및 불량 유출을 초래할 수 있다.

현재 수동 검사 관행은 여러 과제에 직면해 있다: (1) **검사자 간 편차**: 특히 저휘도 영역에서 미묘한 Mura 결함에 대해 검사자마다 검출 임계값이 개인별로 크게 차이를 보임; (2) **시간 집약적 프로세스**: 다양한 구동 조건에서의 종합적 화질 검증은 설정, 측정, 육안 검사를 포함하여 panel variant당 상당한 시간이 소요됨; (3) **희소한 불량 분포**: 정상 샘플이 개발 평가 데이터셋의 대다수를 차지하는 반면 불량은 예측 불가능한 형태, 위치, 심각도를 나타냄; (4) **주관적 판단 경계**: 검출 임계값 근처의 저계조 Mura 패턴은 전문 검사자 간에도 합의가 부족함.

Deep learning 접근법은 MVTec AD[2]를 포함한 benchmark 데이터셋에서 인상적인 성능을 보이지만, 산업 적용 시 architecture별 한계가 드러난다. PatchCore[3]와 같은 memory-based 방법은 textural pattern에서 우수하지만 structural defect에서는 어려움을 겪는다. CFlow[4] 및 FastFlow[5]를 포함한 normalizing flow 접근법은 우수한 기하학적 인식을 달성하지만 광범위한 학습이 필요하다. STFPM[6]과 같은 knowledge distillation 기법은 효율성을 제공하지만 민감도를 희생한다. Dinomaly[7]와 같은 foundation model은 더 높은 resource 요구사항과 함께 강건한 일반화를 제공한다.

본 논문은 개발 단계 품질 평가에서 동작하는 AI 지원 사전 검사 시스템을 통해 수동 검사의 한계를 해결한다. 처리량을 목표로 하는 대량 생산 자동화와 달리, 우리 시스템은 다음을 통해 엔지니어링 평가를 지원한다: (1) batch 사전 검사를 통한 2D 촬상 및 화질 검증 시간 단축, (2) 향상된 검출 능력을 통한 불량 유출 최소화, (3) 일관된 자동화 분석을 통한 검사자 간 편차 제거, (4) 전문가 판단이 필요한 중요 사례에 집중 가능.

#### 1.2. 기여

본 논문은 개발 단계 품질 평가에 맞춤화된 체계적 평가 및 전략적 ensemble 통합을 통해 단일 모델의 한계를 해결한다:

1. **DefectVAD Framework**: MVTec AD[2], ViSA, BTAD benchmark와 XYZ colorspace 전처리를 지원하는 custom OLED 개발 데이터를 위한 통합 인터페이스에서 현대적 unsupervised 모델의 비교 분석을 가능하게 하는 modular 평가 시스템
2. **포괄적 Benchmark 검증**: OLED 유사 카테고리를 선별한 공개 benchmark 데이터셋에서 체계적 평가를 수행하여 98.9% AUROC 달성, OLED 적용 전 framework 효과성 입증
3. **Three-Stage Hybrid Ensemble**: Batch processing workflow에 최적화된 전략적 통합을 통해 실제 OLED 데이터에서 99.1% AUROC 달성하는 cascaded architecture
4. **사전 생산 준비 시스템**: 중요 사례에 대한 human-in-the-loop 검토를 지원하는 엔지니어링 평가용 batch processing pipeline

---

### 2. DefectVAD Framework

#### 2.1. 시스템 Architecture

DefectVAD는 Anomalib framework[9]에서 적용된 계층적 구성을 통해 체계적 평가를 가능하게 하는 4개의 modular layer로 구성된다:

**Configuration Management Layer**: Hierarchical YAML 기반 configuration 시스템은 pretrained backbone weight 경로, 데이터셋 디렉터리, 데이터 로딩을 위한 운영체제별 설정, 이미지 크기 및 전처리 변환을 정의하는 데이터셋 사양, 동적 경로 해상도를 갖춘 모델 hyperparameter를 포함한 환경 설정을 관리한다.

**Data Abstraction Layer**: 통합 데이터셋 인터페이스는 여러 format에 걸쳐 로딩, 전처리, augmentation을 표준화한다. Framework는 5,354개 이미지로 구성된 15개 object 및 texture 카테고리를 가진 MVTec AD[2]; 10,821개 고해상도 이미지를 포함한 12개 산업 카테고리의 ViSA 데이터셋; 3개 제조 카테고리의 BTAD 데이터셋; batch 전처리 pipeline을 통한 XYZ colorspace에서 RGB로의 변환이 필요한 custom OLED 개발 데이터를 지원한다. Factory pattern 구현은 format별 세부사항을 추상화하여 동적 데이터셋 인스턴스화를 가능하게 한다.

**Model Layer**: 각 모델은 다양한 architecture에서 일관된 동작을 보장하는 공통 인터페이스 패턴을 구현한다[9]. 모든 모델은 forward 계산, loss 계산, 선택적 pixel-level localization과 함께 anomaly score를 생성하는 inference 메서드를 정의하는 추상 base에서 상속된다. Directory 구조는 STFPM[6], EfficientAD[8], PatchCore[3], normalizing flow[4,5], reconstruction 기반 방법[11], foundation model[7]을 포함하여 모델 family별로 구현을 구성한다.

**Training Engine**: 통합 training framework는 확장 가능한 hook 시스템을 통해 모델별 customization을 가능하게 하면서 공통 패턴을 추상화한다. Base trainer는 training loop 조정, checkpoint 관리, early stopping, learning rate scheduling, metric logging을 구현한다.

#### 2.2. 데이터셋 및 전처리

**OLED 유사 평가를 위한 Benchmark 데이터셋 선별**:
실제 OLED 데이터에 적용하기 전에 framework를 검증하기 위해, 공개 benchmark에서 OLED 결함 특성과 유사한 카테고리를 선별한다. MVTec AD[2]에서는 Carpet, Leather, Tile (휘도 불균일을 나타내는 Mura 결함과 유사한 texture 카테고리); Grid, Transistor (line 결함과 유사한 structural 카테고리); Metal Nut, Screw (pixel anomaly와 유사한 point defect를 가진 카테고리)를 선별한다. ViSA 데이터셋에서는 PCB 카테고리 (color 불균일과 유사한 texture variation)를 선별한다. BTAD 데이터셋에서는 산업 결함 패턴을 나타내는 3개 제조 카테고리 전부를 활용한다. 이 선별은 4가지 OLED 결함 유형을 대표하는 카테고리에서 총 2,847개 training 이미지와 1,329개 test 이미지를 포함한다.

**OLED 개발 평가 데이터**:
다양한 구동 조건에서 1920×1080 pixel 해상도로 XYZ tristimulus colorimeter 측정을 통해 획득한 실제 OLED 개발 단계 데이터셋은 세 가지 subset으로 구성된다. Training set은 전문가 합의 검사를 통해 검증된 정상 panel을 포함한다. Validation set은 hyperparameter tuning을 위해 예약된 정상 panel을 포함한다. Test set은 네 가지 결함 카테고리로 분포된 panel을 포함한다:

- **Mura 결함**: 특히 색차가 인간 검출 임계값에 근접하는 저계조 영역에서 어려운 구름 또는 얼룩 패턴으로 나타나는 휘도 불균일
- **Color 불균일**: RGB OLED 발광층의 차등 노화에서 발생하는 국부적 색차를 가진 chromatic aberration 패턴을 나타내는 RGB 색 편차
- **Line 결함**: 종종 저휘도에서 미묘하며 좁은 폭에 걸친 공정 불규칙성에서 발생하는 선형 artifact
- **Pixel anomaly**: Dead pixel 및 고해상도 보존이 필요한 색 편차를 포함한 개별 pixel 불규칙성

**XYZ to RGB 변환 Pipeline**:
XYZ colorspace 측정은 natural image에서 사전학습된 deep learning 모델에 적합한 RGB format으로 체계적 변환을 거친다. Pipeline 단계는 다음을 포함한다: (1) XYZ tristimulus 값 로딩, (2) CIE 표준 illuminant D65 color transformation matrix 적용, (3) sRGB 적합성을 위한 지수 2.2의 gamma 보정 수행, (4) [0,1] 범위로 intensity 값 정규화, (5) 8-bit PNG format으로 인코딩. Batch processing은 병렬 CPU 실행을 통해 분당 약 1,000개 이미지의 처리량을 달성한다.

#### 2.3. 실험 설정

**Training Protocol**: 모델별 configuration은 일관된 평가 framework를 유지한다. Optimization은 개선된 일반화를 위해 AdamW algorithm (β₁=0.9, β₂=0.999, ε=1e-8, weight decay=0.01)을 사용한다. Learning rate는 1e-4에서 1e-3 범위의 모델별 초기화와 함께 cosine annealing schedule을 따른다. Data augmentation은 random horizontal flipping (p=0.5), random rotation (±15°), color jittering (brightness/contrast ±10%)을 적용한다. Early stopping은 patience 10 epoch로 validation AUROC를 모니터링한다. 재현성은 모든 실험에서 고정 random seed 42를 통해 보장된다.

**Evaluation Metric**: Image-level metric은 ranking 품질을 측정하는 Area Under ROC Curve (AUROC)와 Area Under Precision-Recall Curve (AUPR)를 포함한다. Localization 가능 모델에 대해 계산된 pixel-level metric은 pixel AUROC 및 pixel AUPR를 포함한다. Threshold 의존 metric은 여러 threshold 선택 전략에서 계산된 F1-score, precision, recall, accuracy를 포함한다. 통계적 검증은 5-fold stratified cross-validation을 사용한다. 성능 metric은 95% 신뢰구간과 함께 평균으로 보고된다. 통계적 유의성은 α = 0.05로 paired t-test를 통해 결정된다.

---

### 3. Benchmark 평가 결과

#### 3.1. Benchmark 데이터셋에서의 전체 성능

실제 OLED 데이터에 적용하기 전에 먼저 공개 benchmark 데이터셋에서 DefectVAD framework를 검증한다. Table I은 MVTec AD, ViSA, BTAD 데이터셋의 선별된 OLED 유사 카테고리에 대한 대표 모델 성능을 요약한다.

**TABLE I: BENCHMARK 데이터셋 (OLED 유사 카테고리)에서의 모델 성능**

| Model             | Architecture Category  | AUROC (%) | F1-Score | Params (M) |
|-------------------|------------------------|-----------|----------|------------|
| Dinomaly-Base-322 | Foundation Model [7]   | 98.1      | 0.941    | 86.2       |
| FastFlow-CaiT     | Normalizing Flow [5]   | 97.8      | 0.935    | 55.8       |
| PatchCore         | Memory-Based [3]       | 97.2      | 0.928    | 23.5       |
| CFlow             | Normalizing Flow [4]   | 97.5      | 0.931    | 45.3       |
| STFPM             | Knowledge Distill. [6] | 95.8      | 0.912    | 38.7       |
| EfficientAD-Small | Knowledge Distill. [8] | 95.6      | 0.908    | 12.3       |

Foundation model[7]은 benchmark 카테고리 전반에 걸쳐 우수한 일관성을 보이며 최고 성능 (98.1% AUROC)을 달성한다. Normalizing flow model[4,5]은 강력한 성능 (97.5-97.8%)을 보인다. Memory-based 방법[3]은 compact parameter로 견고한 성능 (97.2%)을 보인다. Knowledge distillation 접근법[6,8]은 효율성 이점과 함께 적당한 성능 (95.6-95.8%)을 보인다.

#### 3.2. Benchmark에서의 결함 유형별 분석

Table II는 benchmark 데이터셋에서 architecture-defect affinity pattern을 보여주며, 후속 OLED 적용을 위한 specialization 전략을 검증한다.

**TABLE II: BENCHMARK 데이터에서의 결함 유형별 성능**

| Defect Type (Benchmark 카테고리)              | Best Model      | AUROC (%) | F1-Score |
|-------------------------------------------------|-----------------|-----------|----------|
| Mura-like (Carpet, Leather, Tile)              | PatchCore [3]   | 97.8      | 0.936    |
| Color-like (PCB texture variants)               | Dinomaly [7]    | 98.6      | 0.948    |
| Line-like (Grid, Transistor)                    | FastFlow [5]    | 98.2      | 0.942    |
| Pixel-like (Metal Nut, Screw point defects)    | Dinomaly-Large  | 98.4      | 0.945    |

Benchmark 검증은 architecture-defect affinity 가설을 확인한다: Memory-based PatchCore[3]는 texture-like Mura 패턴에서 우수함 (97.8% AUROC). Foundation model Dinomaly[7]는 color variation (98.6%)과 pixel-level defect (98.4%)에서 우수한 성능을 달성한다. Normalizing flow FastFlow[5]는 structural line pattern에서 강점을 보인다 (98.2%). 이러한 발견은 OLED 적용을 위한 ensemble 구성 및 routing 전략을 안내한다.

#### 3.3. Benchmark에서의 상호보완성 분석

**Score 상관관계 분석**: Benchmark 데이터에서 모델 anomaly score 간의 pairwise Pearson 상관계수는 architectural diversity를 정량화한다:

|             | PatchCore | Dinomaly | FastFlow | EfficientAD |
|-------------|-----------|----------|----------|-------------|
| PatchCore   | 1.00      | 0.68     | 0.64     | 0.51        |
| Dinomaly    | 0.68      | 1.00     | 0.61     | 0.47        |
| FastFlow    | 0.64      | 0.61     | 1.00     | 0.53        |
| EfficientAD | 0.51      | 0.47     | 0.53     | 1.00        |

적당한 양의 상관관계 (0.47-0.68)는 모델들이 다른 anomaly 측면을 포착함을 확인한다. 이러한 상관관계는 ensemble 통합에 최적이다: 모델들은 명백한 사례에서는 일치하지만 어려운 샘플에서는 차이를 보인다.

**오류 패턴 분석**: False negative 검사는 상호보완성을 드러낸다. PatchCore는 texture 경계에서 38개의 고유 false negative를 생성한다. Dinomaly는 조명 변화에서 31개의 고유 오류를 생성한다. FastFlow는 기하학적 구조가 부족한 미묘한 패턴에서 42개의 고유 사례를 놓친다. 결합된 coverage는 89%에 도달하여 모든 benchmark anomaly의 89%에서 최소 하나의 모델이 올바르게 검출한다. 모델 쌍 간 22% 미만의 오류 중복은 진정한 상호보완성을 나타낸다.

#### 3.4. Benchmark 데이터에서의 Ensemble 검증

상호보완성 분석을 기반으로 three-stage cascaded ensemble을 구성하고 OLED 적용 전에 benchmark 데이터셋에서 검증한다.

**TABLE III: BENCHMARK ENSEMBLE 성능**

| Configuration          | AUROC (%) | F1-Score | Best Single 대비 개선 |
|------------------------|-----------|----------|-----------------------|
| Best Single (Dinomaly) | 98.1      | 0.941    | Baseline              |
| Two-Model Ensemble     | 98.5      | 0.951    | +0.4 pp               |
| Three-Model Ensemble   | 98.9      | 0.957    | +0.8 pp               |
| Four-Model Ensemble    | 99.0      | 0.959    | +0.9 pp               |

Three-model ensemble은 benchmark 데이터에서 98.9% AUROC를 달성하여 최고 개별 모델 (Dinomaly 98.1%) 대비 +0.8%p 개선을 보인다. Four-model configuration은 복잡도 증가와 함께 최소한의 추가 이득 (+0.1 pp)을 제공하여 three-model 최적성을 확인한다. 이러한 benchmark 결과는 실제 OLED 개발 데이터에 적용하기 전에 ensemble 접근법을 검증한다.

---

### 4. Hybrid Ensemble Architecture

#### 4.1. Three-Stage Cascaded 설계

Benchmark 검증을 기반으로 batch 평가 workflow에 최적화된 three-stage cascaded architecture를 설계한다 (Fig. 2).

**Stage One: Fast Screening**
초기 filtering은 효율적인 batch processing을 달성하는 EfficientAD-Small model[8] (12.3M parameter, benchmark에서 95.6% AUROC)을 사용한다. Dual threshold 전략은 샘플을 세 가지 카테고리로 분리한다: 0.28 미만의 score를 가진 확실한 정상 (약 70%); 0.78을 초과하는 score를 가진 확실한 이상 (약 5%); Stage Two 분석이 필요한 중간 score를 가진 불확실 (약 25%). Conservative threshold는 정상 분류에서 더 높은 false positive rate를 허용하면서 recall을 우선시하여 불량을 놓치지 않도록 보장한다.

**Stage Two: Defect-Type Routing with Selective Ensemble**
불확실한 샘플은 spatial variance, brightness distribution, edge density를 검사하는 경량 pattern 분류를 거친다. Classification rule은 benchmark 데이터에서 검증된 전문화된 조합으로 샘플을 routing한다: 희소한 peak를 가진 낮은 spatial variance (<0.06)는 PatchCore[3] (w=0.65) + Dinomaly[7] (w=0.35)를 trigger하는 Mura pattern을 나타낸다. 0.12를 초과하는 RGB channel variance는 Dinomaly (w=0.75) + EfficientAD[8] (w=0.25)를 trigger하는 color 불균일을 시사한다. 높은 edge density (>0.28)는 FastFlow[5] (w=0.60) + Dinomaly (w=0.40)를 trigger하는 line defect를 나타낸다. 모호한 패턴은 weight (0.42/0.38/0.20)로 전체 three-model ensemble을 trigger한다.

**Stage Three: Confidence-Based Review Queue**
최종 분류는 confidence 추정을 통합한다: confidence = |score - threshold|. 낮은 confidence 샘플 (confidence < 0.18)은 전문가 검토 (전체의 약 3%)를 위해 flag된다. 높은 confidence 샘플은 자동화된 분류를 받는다. Human-in-the-loop는 간단한 사례를 자동화하면서 중요한 결정이 검증을 받도록 보장한다.

#### 4.2. Score Fusion 전략

**Weighted Voting 메커니즘**: 최종 anomaly score는 다음과 같이 계산된다: S_final = 0.42·S_PatchCore + 0.38·S_Dinomaly + 0.20·S_EfficientAD, 여기서 weight는 unity로 합산된다.

**Weight 최적화 프로세스**: Validation set에서의 grid search는 11×11 = 121개 weight 조합을 평가한다. 결과: Uniform weighting (각 0.333)은 benchmark에서 98.6% AUROC 달성; 개별 AUROC에 비례한 performance-based weighting은 98.7% 달성; adaptive validation-최적화 weighting은 최종 configuration으로 선택된 98.9% 달성.

**Category-Specific Threshold 최적화**: 결함 유형별 최적 threshold는 F1-score 최대화를 통해 결정된다:

| Defect Category | Optimal Threshold | F1-Score (Benchmark) | Strategic Rationale                |
|-----------------|-------------------|----------------------|------------------------------------|
| Mura-like       | 0.241             | 0.963                | 미묘한 패턴을 위한 낮은 threshold  |
| Color-like      | 0.196             | 0.978                | Recall 우선 최저 threshold         |
| Line-like       | 0.273             | 0.968                | Structural confidence를 위한 높은 threshold |
| Pixel-like      | 0.205             | 0.973                | 균형잡힌 precision-recall trade-off |

Category-specific thresholding은 global threshold (0.948) 대비 benchmark F1-score를 0.9%p 개선한다 (0.957 vs. 0.948).

---

### 5. OLED 개발 데이터에 대한 적용

#### 5.1. 실제 OLED 데이터에서의 성능

Benchmark 데이터셋에서의 검증 (Section 3) 후, 최적화된 ensemble을 실제 OLED 개발 평가 데이터에 적용한다. Table IV는 포괄적 성능 비교를 제시한다.

**TABLE IV: 실제 OLED 개발 데이터에서의 성능**

| Model/Ensemble        | AUROC (%) | AUPR (%) | F1-Score | Precision (%) | Recall (%) |
|----------------------|-----------|----------|----------|---------------|------------|
| PatchCore [3]        | 96.2      | 95.3     | 0.924    | 96.8          | 95.1       |
| Dinomaly-Base [7]    | 96.8      | 96.1     | 0.931    | 97.3          | 95.8       |
| FastFlow-CaiT [5]    | 96.5      | 95.7     | 0.927    | 97.0          | 95.4       |
| EfficientAD-Small [8]| 94.8      | 93.6     | 0.908    | 95.2          | 93.9       |
| **Hybrid Ensemble**  | **99.1**  | **98.3** | **0.964**| **98.7**      | **98.2**   |
| 절대 개선            | +2.3 pp   | +2.2 pp  | +0.033   | +1.4 pp       | +2.4 pp    |
| 상대 개선            | +2.4%     | +2.3%    | +3.5%    | +1.4%         | +2.5%      |

Hybrid ensemble은 실제 OLED 데이터에서 99.1% AUROC를 달성하며, 최고 개별 모델 (Dinomaly 96.8%) 대비 2.3%p 절대 개선 (2.4% 상대)을 나타낸다. 통계적 유의성은 paired t-test (p = 0.004 < α = 0.05)를 통해 확인되었다. 주목할 만하게, OLED 데이터 성능 (99.1%)은 benchmark 성능 (98.9%)을 초과하여 학습된 ensemble 전략이 목표 domain으로 성공적으로 전이되었음을 시사한다.

**OLED 데이터에서의 카테고리별 성능**:

| Defect Category      | Hybrid Ensemble | Best Individual | Best Model    | 절대 개선 |
|----------------------|-----------------|-----------------|---------------|----------|
| Mura                 | 98.9%           | 96.7%           | PatchCore [3] | +2.2 pp  |
| Color Non-uniformity | 99.4%           | 97.2%           | Dinomaly [7]  | +2.2 pp  |
| Line                 | 99.1%           | 96.9%           | FastFlow [5]  | +2.2 pp  |
| Pixel                | 99.2%           | 97.1%           | Dinomaly      | +2.1 pp  |

모든 OLED 결함 카테고리에 걸친 일관된 이득 (2.1-2.2 pp)은 ensemble 강건성을 검증한다. Benchmark에서 관찰된 architecture-defect affinity (Table II)가 실제 OLED 데이터로 성공적으로 전이된다: PatchCore는 Mura에서 우수 (96.7% 개별), Dinomaly는 color 불균일 (97.2%)과 pixel defect (97.1%)에서 우수, FastFlow는 line defect (96.9%)에서 우수.

#### 5.2. OLED 데이터에서의 Ablation Study

**Ensemble 크기 최적화**:

| Configuration | AUROC (%) | F1-Score | Params (M) | Efficiency Score |
|--------------|-----------|----------|------------|------------------|
| Best Single  | 96.8      | 0.931    | 86.2       | 1.123            |
| Two Models   | 98.2      | 0.947    | 109.7      | 0.895            |
| Three Models | 99.1      | 0.964    | 122.0      | 0.812            |
| Four Models  | 99.2      | 0.966    | 167.3      | 0.593            |

*Efficiency Score = AUROC / (Params/100)

Three-model configuration이 최적: 허용 가능한 122M parameter로 99.1% AUROC. Four-model은 37% parameter 증가와 함께 최소한의 이득 (+0.1 pp)을 제공하여 수확체감을 확인.

**Component 기여도 분석**:

| 제거된 Component    | 결과 AUROC | 성능 저하 | 영향 분석                      |
|--------------------|-----------|----------|-------------------------------|
| None (전체 시스템)  | 99.1%     | Baseline | 완전한 시스템 성능              |
| Remove PatchCore   | 98.5%     | -0.6 pp  | Mura 검출이 크게 저하됨         |
| Remove Dinomaly    | 98.2%     | -0.9 pp  | 전체 성능이 가장 크게 저하됨     |
| Remove EfficientAD | 98.7%     | -0.4 pp  | Edge case에서 상호보완성 손실   |
| Remove Stage One   | 99.1%     | 0.0 pp   | 정확도 영향 없음, 효율성 감소    |
| Remove Type Routing| 98.8%     | -0.3 pp  | Specialization 손실            |

Dinomaly 제거가 가장 큰 저하 (-0.9 pp)를 일으켜 중심 ensemble 역할을 확인. 모든 component가 의미있게 기여하여 architecture를 검증.

#### 5.3. 개발 단계 통합 분석

**운영 통합**: DefectVAD는 modular architecture를 통해 유연한 개발 평가 통합을 가능하게 한다. Batch processing은 편리한 시간대에 실행된다. 품질 보증은 anomaly heatmap, 모델 score, confidence metric, 정상 reference와 함께 검토 인터페이스를 통해 낮은 confidence 샘플 (~3%)을 제시한다. 전문가 annotation은 지속적 개선 pipeline에 feed된다.

**엔지니어링 의사결정 지원**: 개발 단계 평가는 다음을 우선시한다: (1) 자동화된 사전 검사를 통한 검사 시간 단축 (70% 확실한 정상은 자동으로 분류), (2) 일관된 분석을 통한 검사자 간 편차 제거, (3) borderline 사례에 집중 (3% 전문가 검토를 위해 flag), (4) unsupervised learning을 통한 신규 결함 패턴 검출.

**Color Non-uniformity 검출 영향**: Color 불균일 통합은 전통적인 휘도 중심 검사의 중요한 gap을 해결한다. XYZ colorspace 측정은 미묘한 RGB channel 편차 검출을 가능하게 한다. Dinomaly의 fine-grained feature 추출은 OLED color defect에서 효과적임을 증명한다 (개별 97.2%, ensemble 99.4%), 주관적 색 인지 판단에 대한 의존도를 줄인다.

#### 5.4. Benchmark-to-OLED 전이 분석

Benchmark 검증 (Section 3)과 OLED 적용 (Section 5.1) 간의 비교는 성공적인 지식 전이를 드러낸다:

| Metric              | Benchmark 성능 | OLED 성능 | 전이 성공     |
|---------------------|---------------|----------|--------------|
| Ensemble AUROC      | 98.9%         | 99.1%    | +0.2 pp gain |
| Architecture Affinity| 검증됨        | 확인됨    | Pattern 유지 |
| 최적 Ensemble 크기   | 3 models      | 3 models | 일관됨        |
| Weight Configuration| 0.42/0.38/0.20| 0.42/0.38/0.20 | 변경 없음 |

OLED 성능이 benchmark 결과를 초과 (+0.2 pp)하여 ensemble 전략이 효과적으로 일반화됨을 나타낸다. Benchmark에서 검증된 architecture-defect affinity가 OLED domain으로 성공적으로 전이되어 방법론 강건성을 확인한다.

#### 5.5. 한계 및 향후 방향

**현재 한계**: Single-panel 분석은 spatial 또는 temporal 일관성을 활용하지 않고 display를 독립적으로 처리한다. Static ensemble weight는 잠재적 결함 패턴 변화에도 불구하고 고정된 채로 유지된다. 결함 분류체계는 신규 결함 유형에 대한 확장이 필요한 4개의 미리 정의된 카테고리로 제한된다.

**향후 방향**: Model distillation은 99%+ 정확도를 유지하면서 ensemble 지식을 단일 경량 network로 전이할 수 있다. Active learning은 전문가 feedback을 기반으로 weight를 동적으로 조정할 수 있다. Thermal imaging 및 depth sensing을 통합한 multi-modal 통합은 characterization을 향상시킬 수 있다. LCD 및 microLED로 확장하는 transfer learning은 domain adaptation 기법이 필요하다.

---

### 6. 결론

본 논문은 개발 단계의 OLED 디스플레이 화질 평가를 위한 포괄적 평가 framework인 DefectVAD (Defect Vision Anomaly Detection)를 제시했다. OLED 유사 카테고리를 선별한 공개 benchmark (MVTec AD, ViSA, BTAD)에서의 체계적 검증을 통해 98.9% AUROC를 달성하여 framework 효과성을 확인했다. Benchmark에서 검증된 architecture-defect affinity pattern—texture-like Mura를 위한 PatchCore (97.8%), color variation (98.6%)과 pixel defect (98.4%)를 위한 Dinomaly, line pattern을 위한 FastFlow (98.2%)—이 ensemble 설계를 안내했다.

PatchCore[3], Dinomaly[7], EfficientAD[8]를 adaptive weighted fusion을 통해 통합하는 three-stage cascaded hybrid ensemble은 실제 OLED 개발 데이터에서 99.1% AUROC를 달성하여 통계적 유의성 (p = 0.004)과 함께 최고 개별 모델 대비 2.3%p 절대 개선 (2.4% 상대)을 나타냈다. 모든 결함 카테고리에 걸친 일관된 이득 (2.1-2.2 pp)은 ensemble 강건성을 검증한다: Mura 98.9%, color 불균일 99.4%, line defect 99.1%, pixel anomaly 99.2%.

Benchmark 검증 (98.9% AUROC)에서 OLED 적용 (99.1% AUROC)으로의 성공적인 전이는 방법론 강건성을 입증한다. Architecture-defect affinity와 최적 ensemble configuration (3 model, weight 0.42/0.38/0.20)이 benchmark에서 목표 domain으로 효과적으로 전이되어 일반화 능력을 확인한다.

Framework는 개발 단계 과제를 해결한다: 자동화된 사전 검사를 통한 검사 시간 단축 (70% 자동 분류), 일관된 분석을 통한 검사자 간 편차 제거, confidence 기반 검토를 통한 엔지니어링 의사결정 지원 (3% 전문가 검증), unsupervised learning을 통한 신규 패턴 검출. 정상 샘플만으로 학습된 이 접근법은 특정 결함 패턴과 독립적인 실용적 baseline을 제공한다.

체계적 평가는 다양한 결함 유형에서 단일 architecture가 지배하지 않음을 확인했다. Memory-based 방법[3,10]은 textural pattern에서 우수, normalizing flow[4,5]는 structural anomaly에서 우수, foundation model[7]은 일관된 일반화에서 우수—전략적 통합이 개별 한계를 극복하는 상호보완적 강점을 활용한다.

향후 방향에는 배포 효율성을 위한 model distillation, 동적 최적화를 위한 active learning, 향상된 characterization을 위한 multi-modal sensor fusion, 추가 display 기술로의 transfer learning이 포함된다. 포괄적 평가 방법론과 modular framework architecture는 개발 단계 품질 관리에서 실용적 배포를 향한 산업 anomaly detection 연구 발전의 기반을 제공하며, OLED 개발의 신뢰할 수 있는 품질 관리 시스템에 기여한다.

---

### References

[1] Lee, S. et al. Mura defect detection using selective noise filtering based on just-noticeable-difference model. IEEE Trans. Semicond. Manuf. **31**, 381–390 (2018)

[2] Bergmann, P. et al. MVTec AD—A comprehensive real-world dataset for unsupervised anomaly detection. Proc. IEEE Conf. Computer Vision and Pattern Recognition, 9592–9600 (2019)

[3] Roth, K. et al. Towards total recall in industrial anomaly detection. Proc. IEEE Conf. Computer Vision and Pattern Recognition, 14318–14328 (2022)

[4] Gudovskiy, D. et al. CFLOW-AD: Real-time unsupervised anomaly detection with localization via conditional normalizing flows. Proc. IEEE Winter Conf. Applications of Computer Vision, 1819–1828 (2022)

[5] Yu, J. et al. FastFlow: Unsupervised anomaly detection and localization via 2D normalizing flows. arXiv:2111.07677 (2021)

[6] Wang, G. et al. Student-teacher feature pyramid matching for anomaly detection. Proc. British Machine Vision Conf. (2021)

[7] Jiang, Y. et al. Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection. arXiv:2405.14325 (2024)

[8] Batzner, K. et al. EfficientAD: Accurate visual anomaly detection at millisecond-level latencies. Proc. IEEE Winter Conf. Applications of Computer Vision, 5183–5193 (2024)

[9] Akcay, S. et al. Anomalib: A deep learning library for anomaly detection. Proc. IEEE Int. Conf. Image Processing, 1706–1710 (2022)

[10] Defard, T. et al. PaDiM: A patch distribution modeling framework for anomaly detection and localization. Proc. Int. Conf. Pattern Recognition, 475–489 (2021)

[11] Zavrtanik, V. et al. DRAEM: A discriminatively trained reconstruction embedding for surface anomaly detection. Proc. IEEE Int. Conf. Computer Vision, 8330–8339 (2021)

[12] Cheon, J. et al. Convolutional neural network for Mura defect classification in TFT-LCD manufacturing. J. Soc. Inf. Display **27**, 597–605 (2019)

---

### Figures

**Fig. 1. 구동 화질 검사용 OLED Display Killer Pattern 예시**
[다양한 조건에서의 test pattern을 보여주는 구동 화질 검사 pattern 이미지를 위한 placeholder: 암실, 고온/저온, dimming level, 주파수 변화, 저계조 영역의 복합 killer pattern]

**Fig. 2. DefectVAD Three-Stage Cascaded Architecture**

```
┌──────────────────────────────────────────────────────────┐
│    Benchmark 검증 단계 (MVTec/ViSA/BTAD)                 │
│    OLED 유사 카테고리: 2,847 train / 1,329 test          │
│    Ensemble 성능: 98.9% AUROC                            │
└───────────────────────┬──────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│      OLED 개발 평가 데이터 적용                           │
│    (XYZ Colorspace → RGB Batch 변환)                     │
└───────────────────────┬──────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────────┐
        │      Stage 1: Fast Screening          │
        │   Model: EfficientAD-Small [8]        │
        │   Params: 12.3M | Bench AUROC: 95.6%  │
        │   전략: Dual threshold (0.28/0.78)    │
        └───────────────┬───────────────────────┘
                        ↓
        ┌───────────────┴────────────────┐
        ↓               ↓                ↓
   Score < 0.28    0.28 ≤ S ≤ 0.78   Score > 0.78
   확실한 정상      불확실한 사례      확실한 이상
      (~70%)           (~25%)            (~5%)
        ↓               ↓                ↓
     분류됨         다음 Stage          분류됨
                        ↓
        ┌───────────────────────────────────────┐
        │   Stage 2: Defect-Type Routing        │
        │   Pattern 분류 & Selective Ensemble   │
        │   (benchmark에서 검증)                 │
        └───────────────┬───────────────────────┘
                        ↓
     ┌────────┬─────────┼─────────┬─────────┐
     ↓        ↓         ↓         ↓         ↓
   Mura     Color     Line    Unknown    Mixed
  Pattern    Non-   Pattern  Pattern   Pattern
           uniform.
     ↓        ↓         ↓         ↓         ↓
 PatchCore Dinomaly FastFlow  Full    Adaptive
  +Dinomaly +EfficAD +Dinomaly Ensemble Strategy
 (0.65/0.35)(0.75/0.25)(0.60/0.40)(0.42/0.38/0.20)
     ↓        ↓         ↓         ↓         ↓
     └────────┴─────────┴─────────┴─────────┘
                        ↓
      ┌───────────────────────────────────────┐
      │   Stage 3: Confidence Assessment      │
      │   Metric: |Score - Threshold|         │
      │   Threshold: 0.18                     │
      └─────────────────┬─────────────────────┘
                        ↓
            ┌───────────┴────────────┐
            ↓                        ↓
     High Confidence          Low Confidence
     (Confidence ≥ 0.18)      (Confidence < 0.18)
        (~97%)                    (~3%)
            ↓                        ↓
     자동화된 결정            전문가 Review Queue
     최종 분류               엔지니어링 검증
            ↓                        ↓
            └────────┬───────────────┘
                     ↓
        ┌────────────────────────────────────────┐
        │    OLED 데이터에서의 최종 성능           │
        │   Image-Level AUROC: 99.1%             │
        │   F1-Score: 0.964                      │
        │   개선: Best single 대비 +2.3 pp        │
        │   통계적 유의성: p = 0.004              │
        │   Benchmark → OLED 전이: +0.2 pp       │
        └────────────────────────────────────────┘
```

**Figure Caption**: Benchmark 데이터셋에서 검증 (98.9% AUROC) 후 OLED 개발 데이터에 적용 (99.1% AUROC)된 three-stage cascaded architecture. Stage one은 효율적인 screening (12.3M params)을 사용하여 70% 명백한 정상 사례를 제거한다. Stage two는 benchmark에서 검증된 전문화된 조합으로 defect-type routing을 적용한다: Mura-like → PatchCore+Dinomaly, Color-like → Dinomaly+EfficientAD, Line-like → FastFlow+Dinomaly. Stage three는 3% 전문가 검증을 위한 confidence 기반 review를 구현한다. Benchmark에서 OLED로의 성공적인 전이는 최고 개별 모델 대비 2.3 pp 절대 개선과 함께 방법론 강건성을 입증한다.
