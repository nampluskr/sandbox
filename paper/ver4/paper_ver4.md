## DefectVAD: Defect Vision Anomaly Detection
## Hybrid Ensemble Framework for Unsupervised OLED Display Defect Detection in Pre-Production Quality Assessment

### Abstract

Display quality verification and defect judgment in OLED development rely on subjective visual inspection under various driving conditions including luminance, temperature, and frequency variations. Manual inspection requires significant time and effort, with persistent risk of defect escape due to inter-inspector variability. Normal samples dominate the dataset while defects are sparse with unpredictable shapes, positions, and severity levels, making single-threshold classification unstable. This paper presents DefectVAD (Defect Vision Anomaly Detection), a comprehensive evaluation framework enabling systematic comparison of contemporary unsupervised anomaly detection models for pre-production OLED quality assessment. We evaluate the framework on benchmark datasets (MVTec AD, ViSA, BTAD) selecting categories analogous to OLED defect types, achieving 98.9% AUROC on benchmark data. The proposed three-stage cascaded hybrid ensemble integrating complementary detection approaches demonstrates 99.1% AUROC on actual OLED development data, representing 2.3 percentage point absolute improvement over best individual model. The framework operates in batch processing mode on XYZ colorimeter measurements, supporting engineers' visual inspection by reducing inter-inspector variance and inspection workload. Trained exclusively on normal samples, the system provides a practical baseline independent of specific defect patterns, effectively detecting anomalies across diverse content and luminance conditions.

**Keywords:** OLED display inspection, unsupervised anomaly detection, hybrid ensemble learning, visual quality assessment

---

### 1. Introduction

#### 1.1. Background and Motivation

Display quality evaluation consists of quantitative assessment based on physical characteristics measured by instruments and subjective assessment based on human visual perception [1]. Objective quality measurements often differ from actual human-perceived quality, making subjective evaluation more appropriate for display quality assessment. Particularly, defects at low grayscale or low luminance levels exhibit not only severity differences but also presence/absence ambiguity depending on driving environment and observers.

OLED driving quality risk verification and defect improvement require optical characteristic measurements and visual inspection across diverse environments, functions, and patterns including dark room conditions, high/low temperature, dimming, frequency variations, and complex killer patterns (Fig. 1). Panel anomalies include predictable defects appearing at fixed locations in regular patterns and unpredictable defects appearing irregularly at arbitrary positions. Many defects manifest at low grayscale levels, causing increased verification time and potential defect escape due to inter-inspector detection variance.

Current manual inspection practices face several challenges: (1) **Inter-inspector variability**: Different inspectors exhibit varying sensitivity to subtle Mura defects, especially in low-luminance regions where detection thresholds vary significantly among individuals; (2) **Time-intensive process**: Comprehensive quality verification under multiple driving conditions requires substantial time per panel variant including setup, measurement, and visual inspection; (3) **Sparse defect distribution**: Normal samples constitute majority of development evaluation datasets while defects exhibit unpredictable morphology, location, and severity; (4) **Subjective decision boundaries**: Low-grayscale Mura patterns near detection threshold lack consensus among expert inspectors.

Deep learning approaches demonstrate impressive performance on benchmark datasets including MVTec AD [2], yet industrial deployment reveals architecture-specific limitations. Memory-based methods such as PatchCore [3] excel at textural patterns but struggle with structural defects. Normalizing flow approaches including CFlow [4] and FastFlow [5] achieve superior geometric recognition yet require extensive training. Knowledge distillation techniques such as STFPM [6] offer efficiency but sacrifice sensitivity. Foundation models like Dinomaly [7] provide robust generalization with higher resource demands.

This paper addresses manual inspection limitations through AI-assisted pre-screening system operating in development phase quality assessment. Unlike mass production automation targeting throughput, our system supports engineering evaluation by: (1) reducing 2D imaging and quality verification time through batch pre-screening, (2) minimizing defect escape through improved detection capability, (3) eliminating inter-inspector variance through consistent automated analysis, and (4) enabling focus on critical cases requiring expert judgment.

#### 1.2. Contributions

This paper addresses single-model limitations through systematic evaluation and strategic ensemble integration tailored for development phase quality assessment:

1. **DefectVAD Framework**: Modular evaluation system enabling comparative analysis of contemporary unsupervised models across unified interfaces supporting MVTec AD [2], ViSA, BTAD benchmarks and custom OLED development data with XYZ colorspace preprocessing
2. **Benchmark Validation**: Systematic evaluation on public benchmark datasets selecting OLED-analogous categories achieving 98.9% AUROC, demonstrating framework effectiveness before OLED application
3. **Three-Stage Hybrid Ensemble**: Cascaded architecture achieving 99.1% AUROC on actual OLED data through strategic integration optimized for batch processing workflow
4. **Pre-Production Ready System**: Batch processing pipeline supporting engineering evaluation with human-in-the-loop review for critical cases

---

### 2. DefectVAD Framework

#### 2.1. System Architecture

DefectVAD comprises four modular layers enabling systematic evaluation through hierarchical organization adapted from Anomalib framework [9]:

**Configuration Management Layer**: Hierarchical YAML-based configuration system manages environment setup including paths for pretrained backbone weights, dataset directories, operating system-specific settings for data loading, dataset specifications defining image dimensions and preprocessing transformations, and model hyperparameters with dynamic path resolution.

**Data Abstraction Layer**: Unified dataset interface standardizes loading, preprocessing, and augmentation across multiple formats. The framework supports MVTec AD [2] with 15 object and texture categories comprising 5,354 images; ViSA dataset containing 12 industrial categories with 10,821 high-resolution images; BTAD dataset with 3 manufacturing categories; and custom OLED development data requiring XYZ colorspace to RGB conversion through batch preprocessing pipeline. Factory pattern implementation enables dynamic dataset instantiation abstracting format-specific details.

**Model Layer**: Each model implements common interface patterns ensuring consistent behavior across diverse architectures [9]. All models inherit from abstract base defining forward computation, loss computation, and inference methods producing anomaly scores with optional pixel-level localization. Directory structure organizes implementations by model family including STFPM [6], EfficientAD [8], PatchCore [3], normalizing flows [4,5], reconstruction-based methods [11], and foundation models [7].

**Training Engine**: Unified training framework abstracts common patterns while enabling model-specific customizations through extensible hook system. Base trainer implements training loop coordination, checkpoint management, early stopping, learning rate scheduling, and metric logging.

#### 2.2. Dataset and Preprocessing

**Benchmark Dataset Selection for OLED Analogous Evaluation**:
To validate the framework before applying to actual OLED data, we select categories from public benchmarks analogous to OLED defect characteristics. From MVTec AD [2], we select: Carpet, Leather, Tile (texture categories analogous to Mura defects exhibiting luminance non-uniformity); Grid, Transistor (structural categories analogous to line defects); Metal Nut, Screw (categories with point defects analogous to pixel anomalies). From ViSA dataset, we select: PCB categories (texture variations analogous to color non-uniformity). From BTAD dataset, we utilize all three manufacturing categories exhibiting industrial defect patterns. This selection totals 2,847 training images and 1,329 test images across categories representing four OLED defect types.

**OLED Development Evaluation Data**:
Actual OLED development phase dataset acquired through XYZ tristimulus colorimeter measurements at 1920×1080 pixel resolution under various driving conditions comprises three subsets. Training set contains normal panels verified through expert consensus. Validation set includes normal panels for hyperparameter tuning. Test set encompasses panels distributed across four defect categories:

- **Mura defects**: Luminance non-uniformity appearing as cloudy or spotty patterns, particularly challenging at low grayscale levels where color differences approach human detection threshold
- **Color non-uniformity**: RGB color deviation exhibiting chromatic aberration patterns with localized color differences, arising from differential aging of RGB OLED emissive layers
- **Line defects**: Linear artifacts from process irregularities spanning narrow widths, often subtle at low luminance
- **Pixel anomalies**: Individual pixel irregularities including dead pixels and color deviations requiring high-resolution preservation

**XYZ to RGB Conversion Pipeline**:
XYZ colorspace measurements undergo systematic conversion to RGB format suitable for deep learning models pretrained on natural images. Pipeline stages include: (1) loading raw XYZ tristimulus values, (2) applying CIE standard illuminant D65 color transformation matrix, (3) performing gamma correction with exponent 2.2 for sRGB conformance, (4) normalizing intensity values to [0,1] range, and (5) encoding as 8-bit PNG format. Batch processing achieves throughput of approximately 1,000 images per minute through parallel CPU execution.

#### 2.3. Experimental Setup

**Training Protocol**: Model-specific configurations maintain consistent evaluation framework. Optimization employs AdamW algorithm (β₁=0.9, β₂=0.999, ε=1e-8, weight decay=0.01) for improved generalization. Learning rate follows cosine annealing schedule with model-appropriate initialization ranging 1e-4 to 1e-3. Data augmentation applies random horizontal flipping (p=0.5), random rotation (±15°), and color jittering (brightness/contrast ±10%). Early stopping monitors validation AUROC with patience of 10 epochs. Reproducibility ensured through fixed random seed of 42.

**Evaluation Metrics**: Image-level metrics include Area Under ROC Curve (AUROC) measuring ranking quality and Area Under Precision-Recall Curve (AUPR). Pixel-level metrics computed for localization-capable models include pixel AUROC and pixel AUPR. Threshold-dependent metrics comprise F1-score, precision, recall, and accuracy computed at multiple threshold selection strategies. Statistical validation employs 5-fold stratified cross-validation. Performance metrics reported as mean with 95% confidence intervals. Statistical significance determined through paired t-tests with α = 0.05.

---

### 3. Benchmark Evaluation Results

#### 3.1. Overall Performance on Benchmark Datasets

We first validate DefectVAD framework on public benchmark datasets before applying to OLED data. Table I summarizes representative model performance on selected OLED-analogous categories from MVTec AD, ViSA, and BTAD datasets.

**TABLE I: MODEL PERFORMANCE ON BENCHMARK DATASETS (OLED-ANALOGOUS CATEGORIES)**

| Model             | Architecture Category  | AUROC (%) | F1-Score | Params (M) |
|-------------------|------------------------|-----------|----------|------------|
| Dinomaly-Base-322 | Foundation Model [7]   | 98.1      | 0.941    | 86.2       |
| FastFlow-CaiT     | Normalizing Flow [5]   | 97.8      | 0.935    | 55.8       |
| PatchCore         | Memory-Based [3]       | 97.2      | 0.928    | 23.5       |
| CFlow             | Normalizing Flow [4]   | 97.5      | 0.931    | 45.3       |
| STFPM             | Knowledge Distill. [6] | 95.8      | 0.912    | 38.7       |
| EfficientAD-Small | Knowledge Distill. [8] | 95.6      | 0.908    | 12.3       |

Foundation models [7] achieve highest performance (98.1% AUROC) demonstrating superior consistency across benchmark categories. Normalizing flow models [4,5] show strong performance (97.5-97.8%). Memory-based methods [3] demonstrate solid performance (97.2%) with compact parameters. Knowledge distillation approaches [6,8] show moderate performance (95.6-95.8%) with efficiency advantages.

#### 3.2. Defect-Type Specific Analysis on Benchmarks

Table II demonstrates architecture-defect affinity patterns on benchmark datasets, validating specialization strategies for subsequent OLED application.

**TABLE II: DEFECT-TYPE SPECIFIC PERFORMANCE ON BENCHMARK DATA**

| Defect Type (Benchmark Categories)              | Best Model      | AUROC (%) | F1-Score |
|-------------------------------------------------|-----------------|-----------|----------|
| Mura-like (Carpet, Leather, Tile)              | PatchCore [3]   | 97.8      | 0.936    |
| Color-like (PCB texture variants)               | Dinomaly [7]    | 98.6      | 0.948    |
| Line-like (Grid, Transistor)                    | FastFlow [5]    | 98.2      | 0.942    |
| Pixel-like (Metal Nut, Screw point defects)    | Dinomaly-Large  | 98.4      | 0.945    |

Benchmark validation confirms architecture-defect affinity hypothesis: Memory-based PatchCore [3] excels at texture-like Mura patterns (97.8% AUROC). Foundation model Dinomaly [7] achieves superior performance on color variations (98.6%) and pixel-level defects (98.4%). Normalizing flow FastFlow [5] demonstrates strength on structural line patterns (98.2%). These findings guide ensemble composition and routing strategies for OLED application.

#### 3.3. Complementarity Analysis on Benchmarks

**Score Correlation Analysis**: Pairwise Pearson correlation coefficients between model anomaly scores on benchmark data quantify architectural diversity:

|             | PatchCore | Dinomaly | FastFlow | EfficientAD |
|-------------|-----------|----------|----------|-------------|
| PatchCore   | 1.00      | 0.68     | 0.64     | 0.51        |
| Dinomaly    | 0.68      | 1.00     | 0.61     | 0.47        |
| FastFlow    | 0.64      | 0.61     | 1.00     | 0.53        |
| EfficientAD | 0.51      | 0.47     | 0.53     | 1.00        |

Moderate positive correlations (0.47-0.68) confirm models capture different anomaly aspects. These correlations are optimal for ensemble integration: models agree on obvious cases while differing on challenging samples.

**Error Pattern Analysis**: False negative examination reveals complementarity. PatchCore generates 38 unique false negatives on texture boundaries. Dinomaly produces 31 unique errors on lighting variations. FastFlow misses 42 unique cases on subtle patterns lacking geometric structure. Combined coverage reaches 89% with at least one model correctly detecting 89% of all benchmark anomalies. Error overlap below 22% between model pairs indicates true complementarity.

#### 3.4. Ensemble Validation on Benchmark Data

Based on complementarity analysis, we construct three-stage cascaded ensemble and validate on benchmark datasets before OLED application.

**TABLE III: BENCHMARK ENSEMBLE PERFORMANCE**

| Configuration          | AUROC (%) | F1-Score | Improvement vs. Best Single |
|------------------------|-----------|----------|----------------------------|
| Best Single (Dinomaly) | 98.1      | 0.941    | Baseline                   |
| Two-Model Ensemble     | 98.5      | 0.951    | +0.4 pp                    |
| Three-Model Ensemble   | 98.9      | 0.957    | +0.8 pp                    |
| Four-Model Ensemble    | 99.0      | 0.959    | +0.9 pp                    |

Three-model ensemble achieves 98.9% AUROC on benchmark data, demonstrating +0.8 percentage point improvement over best individual model (Dinomaly 98.1%). Four-model configuration provides minimal additional gain (+0.1 pp) with increased complexity, confirming three-model optimality. These benchmark results validate the ensemble approach before applying to actual OLED development data.

---

### 4. Hybrid Ensemble Architecture

#### 4.1. Three-Stage Cascaded Design

Based on benchmark validation, we design three-stage cascaded architecture optimized for batch evaluation workflow (Fig. 2).

**Stage One: Fast Screening**
Initial filtering employs EfficientAD-Small model [8] (12.3M parameters, 95.6% AUROC on benchmarks) achieving efficient batch processing. Dual threshold strategy separates samples into three categories: certain normal with scores below 0.28 (approximately 70%); certain anomaly with scores exceeding 0.78 (approximately 5%); uncertain with intermediate scores (approximately 25%) requiring Stage Two analysis. Conservative thresholds prioritize recall over precision, ensuring no defects missed at cost of higher false positive rate in normal classification.

**Stage Two: Defect-Type Routing with Selective Ensemble**
Uncertain samples undergo lightweight pattern classification examining spatial variance, brightness distribution, and edge density. Classification rules route samples to specialized combinations validated on benchmark data: Low spatial variance (<0.06) with sparse peaks indicates Mura pattern triggering PatchCore [3] (w=0.65) + Dinomaly [7] (w=0.35). RGB channel variance exceeding 0.12 suggests color non-uniformity triggering Dinomaly (w=0.75) + EfficientAD [8] (w=0.25). High edge density (>0.28) indicates line defect triggering FastFlow [5] (w=0.60) + Dinomaly (w=0.40). Ambiguous patterns trigger full three-model ensemble with weights (0.42/0.38/0.20).

**Stage Three: Confidence-Based Review Queue**
Final classification incorporates confidence estimation: confidence = |score - threshold|. Low confidence samples (confidence < 0.18) flagged for expert review (approximately 3% of total). High confidence samples receive automated classification. Human-in-the-loop ensures critical decisions receive validation while automating straightforward cases.

#### 4.2. Score Fusion Strategy

**Weighted Voting Mechanism**: Final anomaly score computed as: S_final = 0.42·S_PatchCore + 0.38·S_Dinomaly + 0.20·S_EfficientAD, where weights sum to unity.

**Weight Optimization Process**: Grid search on validation set evaluates 11×11 = 121 weight combinations. Results: Uniform weighting (0.333 each) achieves 98.6% AUROC on benchmarks; performance-based weighting proportional to individual AUROCs achieves 98.7%; adaptive validation-optimized weighting achieves 98.9% selected as final configuration.

**Category-Specific Threshold Optimization**: Per-defect-type optimal thresholds determined through F1-score maximization:

| Defect Category      | Optimal Threshold | F1-Score (Benchmark) | Strategic Rationale                     |
|----------------------|-------------------|----------------------|-----------------------------------------|
| Mura-like            | 0.241             | 0.963                | Lower threshold for subtle patterns     |
| Color-like           | 0.196             | 0.978                | Lowest threshold prioritizes recall     |
| Line-like            | 0.273             | 0.968                | Higher threshold for structural confidence |
| Pixel-like           | 0.205             | 0.973                | Balanced precision-recall trade-off     |

Category-specific thresholding improves benchmark F1-score by 0.9 percentage points over global threshold (0.957 vs. 0.948).

---

### 5. Application to OLED Development Data

#### 5.1. Performance on Actual OLED Data

After validation on benchmark datasets (Section 3), we apply the optimized ensemble to actual OLED development evaluation data. Table IV presents comprehensive performance comparison.

**TABLE IV: PERFORMANCE ON ACTUAL OLED DEVELOPMENT DATA**

| Model/Ensemble        | AUROC (%) | AUPR (%) | F1-Score | Precision (%) | Recall (%) |
|----------------------|-----------|----------|----------|---------------|------------|
| PatchCore [3]        | 96.2      | 95.3     | 0.924    | 96.8          | 95.1       |
| Dinomaly-Base [7]    | 96.8      | 96.1     | 0.931    | 97.3          | 95.8       |
| FastFlow-CaiT [5]    | 96.5      | 95.7     | 0.927    | 97.0          | 95.4       |
| EfficientAD-Small [8]| 94.8      | 93.6     | 0.908    | 95.2          | 93.9       |
| **Hybrid Ensemble**  | **99.1**  | **98.3** | **0.964**| **98.7**      | **98.2**   |
| Absolute Improvement | +2.3 pp   | +2.2 pp  | +0.033   | +1.4 pp       | +2.4 pp    |
| Relative Improvement | +2.4%     | +2.3%    | +3.5%    | +1.4%         | +2.5%      |

The hybrid ensemble achieves 99.1% AUROC on actual OLED data, representing 2.3 percentage point absolute improvement (2.4% relative) over best individual model (Dinomaly 96.8%). Statistical significance confirmed through paired t-test (p = 0.004 < α = 0.05). Notably, OLED data performance (99.1%) exceeds benchmark performance (98.9%), suggesting successful transfer of learned ensemble strategies to target domain.

**Per-Category Performance on OLED Data**:

| Defect Category      | Hybrid Ensemble | Best Individual | Best Model    | Absolute Gain |
|----------------------|-----------------|-----------------|---------------|---------------|
| Mura                 | 98.9%           | 96.7%           | PatchCore [3] | +2.2 pp       |
| Color Non-uniformity | 99.4%           | 97.2%           | Dinomaly [7]  | +2.2 pp       |
| Line                 | 99.1%           | 96.9%           | FastFlow [5]  | +2.2 pp       |
| Pixel                | 99.2%           | 97.1%           | Dinomaly      | +2.1 pp       |

Consistent gains (2.1-2.2 pp) across all OLED defect categories validate ensemble robustness. Architecture-defect affinities observed on benchmarks (Table II) transfer successfully to actual OLED data: PatchCore excels at Mura (96.7% individual), Dinomaly at color non-uniformity (97.2%) and pixel defects (97.1%), FastFlow at line defects (96.9%).

#### 5.2. Ablation Studies on OLED Data

**Ensemble Size Optimization**:

| Configuration        | AUROC (%) | F1-Score | Params (M) | Efficiency Score |
|---------------------|-----------|----------|------------|------------------|
| Best Single         | 96.8      | 0.931    | 86.2       | 1.123            |
| Two Models          | 98.2      | 0.947    | 109.7      | 0.895            |
| Three Models        | 99.1      | 0.964    | 122.0      | 0.812            |
| Four Models         | 99.2      | 0.966    | 167.3      | 0.593            |

*Efficiency Score = AUROC / (Params/100)

Three-model configuration optimal: 99.1% AUROC with acceptable 122M parameters. Four-model provides minimal gain (+0.1 pp) with 37% parameter increase, confirming diminishing returns.

**Component Contribution Analysis**:

| Removed Component   | Resulting AUROC | Performance Drop | Impact Analysis                        |
|---------------------|-----------------|------------------|----------------------------------------|
| None (Full System)  | 99.1%           | Baseline         | Complete system performance            |
| Remove PatchCore    | 98.5%           | -0.6 pp          | Mura detection degrades significantly  |
| Remove Dinomaly     | 98.2%           | -0.9 pp          | Overall performance suffers most       |
| Remove EfficientAD  | 98.7%           | -0.4 pp          | Complementarity loss affects edges     |
| Remove Stage One    | 99.1%           | 0.0 pp           | No accuracy impact, efficiency reduced |
| Remove Type Routing | 98.8%           | -0.3 pp          | Specialization loss                    |

Dinomaly removal causes largest drop (-0.9 pp) confirming central ensemble role. All components contribute meaningfully validating architecture.

#### 5.3. Development Phase Integration Analysis

**Operational Integration**: DefectVAD enables flexible development evaluation integration through modular architecture. Batch processing executes during convenient periods. Quality assurance presents low-confidence samples (~3%) through review interface with anomaly heatmaps, model scores, confidence metrics, and normal references. Expert annotations feed continuous improvement pipeline.

**Supporting Engineering Decisions**: Development phase evaluation prioritizes: (1) reducing inspection time through automated pre-screening (70% certain normal automatically classified), (2) eliminating inter-inspector variance through consistent analysis, (3) enabling focus on borderline cases (3% flagged for expert review), and (4) detecting novel defect patterns through unsupervised learning.

**Color Non-uniformity Detection Impact**: Color non-uniformity integration addresses critical gap in traditional luminance-focused inspection. XYZ colorspace measurements enable detection of subtle RGB channel deviations. Dinomaly's fine-grained feature extraction proves effective (97.2% individual, 99.4% ensemble on OLED color defects), reducing reliance on subjective color perception judgment.

#### 5.4. Benchmark-to-OLED Transfer Analysis

Comparison between benchmark validation (Section 3) and OLED application (Section 5.1) reveals successful knowledge transfer:

| Metric               | Benchmark Performance | OLED Performance | Transfer Success |
|----------------------|----------------------|------------------|------------------|
| Ensemble AUROC       | 98.9%                | 99.1%            | +0.2 pp gain     |
| Architecture Affinity| Validated            | Confirmed        | Patterns hold    |
| Optimal Ensemble Size| 3 models             | 3 models         | Consistent       |
| Weight Configuration | 0.42/0.38/0.20       | 0.42/0.38/0.20   | Unchanged        |

OLED performance exceeds benchmark results (+0.2 pp), indicating ensemble strategies generalize effectively. Architecture-defect affinities validated on benchmarks transfer successfully to OLED domain, confirming methodology robustness.

#### 5.5. Limitations and Future Directions

**Current Limitations**: Single-panel analysis treats displays independently without exploiting spatial or temporal consistency. Static ensemble weights remain fixed despite potential defect pattern variations. Defect taxonomy limited to four predefined categories requires extension for emerging defect types.

**Future Directions**: Model distillation could transfer ensemble knowledge to single lightweight network maintaining 99%+ accuracy. Active learning could dynamically adjust weights based on expert feedback. Multi-modal integration incorporating thermal imaging and depth sensing could enhance characterization. Transfer learning extending to LCD and microLED requires domain adaptation techniques.

---

### 6. Conclusion

This paper presented DefectVAD (Defect Vision Anomaly Detection), a comprehensive evaluation framework for OLED display quality assessment in development phase. Through systematic validation on public benchmarks (MVTec AD, ViSA, BTAD) selecting OLED-analogous categories, we demonstrated 98.9% AUROC confirming framework effectiveness. Architecture-defect affinity patterns validated on benchmarks—PatchCore for texture-like Mura (97.8%), Dinomaly for color variations (98.6%) and pixel defects (98.4%), FastFlow for line patterns (98.2%)—guided ensemble design.

The three-stage cascaded hybrid ensemble integrating PatchCore [3], Dinomaly [7], and EfficientAD [8] through adaptive weighted fusion achieved 99.1% AUROC on actual OLED development data, representing 2.3 percentage point absolute improvement (2.4% relative) over best individual model with statistical significance (p = 0.004). Consistent gains across all defect categories (2.1-2.2 pp) validate ensemble robustness: Mura 98.9%, color non-uniformity 99.4%, line defects 99.1%, pixel anomalies 99.2%.

Successful transfer from benchmark validation (98.9% AUROC) to OLED application (99.1% AUROC) demonstrates methodology robustness. Architecture-defect affinities and optimal ensemble configuration (3 models, weights 0.42/0.38/0.20) transfer effectively from benchmarks to target domain, confirming generalization capability.

The framework addresses development phase challenges: reducing inspection time through automated pre-screening (70% automatic classification), eliminating inter-inspector variance through consistent analysis, supporting engineering decisions through confidence-based review (3% expert validation), and detecting novel patterns through unsupervised learning. Trained exclusively on normal samples, the approach provides practical baseline independent of specific defect patterns.

Systematic evaluation confirmed no single architecture dominates across diverse defect types. Memory-based methods [3,10] excel at textural patterns, normalizing flows [4,5] at structural anomalies, foundation models [7] at consistent generalization—strategic integration leverages complementary strengths overcoming individual limitations.

Future directions include model distillation for deployment efficiency, active learning for dynamic optimization, multi-modal sensor fusion for enhanced characterization, and transfer learning to additional display technologies. The comprehensive evaluation methodology and modular framework provide foundation for advancing industrial anomaly detection toward practical deployment in development phase quality control, contributing to reliable quality management systems in OLED development.

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

**Fig. 1. OLED Display Killer Pattern Examples for Driving Quality Inspection**
[Placeholder for driving quality inspection pattern images showing various test patterns under different conditions: dark room, high/low temperature, dimming levels, frequency variations, and complex killer patterns at low grayscale levels]

**Fig. 2. DefectVAD Three-Stage Cascaded Architecture**

```
┌──────────────────────────────────────────────────────────┐
│    Benchmark Validation Phase (MVTec/ViSA/BTAD)         │
│    OLED-analogous categories: 2,847 train / 1,329 test  │
│    Ensemble Performance: 98.9% AUROC                     │
└───────────────────────┬──────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│      OLED Development Evaluation Data Application        │
│    (XYZ Colorspace → RGB Batch Conversion)               │
└───────────────────────┬──────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────────┐
        │      Stage 1: Fast Screening          │
        │   Model: EfficientAD-Small [8]        │
        │   Params: 12.3M | Bench AUROC: 95.6%  │
        │   Strategy: Dual threshold (0.28/0.78)│
        └───────────────┬───────────────────────┘
                        ↓
        ┌───────────────┴────────────────┐
        ↓               ↓                ↓
   Score < 0.28    0.28 ≤ S ≤ 0.78   Score > 0.78
   Certain Normal  Uncertain Cases   Certain Anomaly
      (~70%)           (~25%)            (~5%)
        ↓               ↓                ↓
   CLASSIFIED      Next Stage       CLASSIFIED
                        ↓
        ┌───────────────────────────────────────┐
        │   Stage 2: Defect-Type Routing        │
        │   Pattern Classification & Selective  │
        │   Ensemble (validated on benchmarks)  │
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
     Automated Decision       Expert Review Queue
     Final Classification     Engineering Validation
            ↓                        ↓
            └────────┬───────────────┘
                     ↓
        ┌────────────────────────────────────────┐
        │    Final Performance on OLED Data      │
        │   Image-Level AUROC: 99.1%             │
        │   F1-Score: 0.964                      │
        │   Improvement: +2.3 pp over best single│
        │   Statistical Significance: p = 0.004  │
        │   Benchmark → OLED Transfer: +0.2 pp   │
        └────────────────────────────────────────┘
```

**Figure Caption**: Three-stage cascaded architecture validated on benchmark datasets (98.9% AUROC) then applied to OLED development data (99.1% AUROC). Stage one employs efficient screening (12.3M params) eliminating 70% obvious normal cases. Stage two applies defect-type routing with specialized combinations validated on benchmarks: Mura-like → PatchCore+Dinomaly, Color-like → Dinomaly+EfficientAD, Line-like → FastFlow+Dinomaly. Stage three implements confidence-based review flagging 3% for expert validation. Successful transfer from benchmarks to OLED demonstrates methodology robustness with 2.3 pp absolute improvement over best individual model.
