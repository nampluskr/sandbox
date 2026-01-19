## DefectVAD: Defect Vision Anomaly Detection
## Hybrid Ensemble Framework for Unsupervised OLED Display Defect Detection in Pre-Production Quality Assessment

### Abstract

Display quality verification and defect judgment in OLED development rely on subjective visual inspection under various driving conditions including luminance, temperature, and frequency variations. Manual inspection requires significant time and effort, with persistent risk of defect escape due to inter-inspector variability. Normal samples dominate the dataset while defects are sparse with unpredictable shapes, positions, and severity levels, making single-threshold classification unstable. This paper presents DefectVAD (Defect Vision Anomaly Detection), a comprehensive evaluation framework enabling systematic comparison of 20 unsupervised anomaly detection models for pre-production OLED quality assessment. We propose a three-stage cascaded hybrid ensemble integrating PatchCore, Dinomaly, and EfficientAD through adaptive weighted fusion, achieving 99.2% image-level AUROC—outperforming individual models by 3.8%. The framework operates in batch processing mode on XYZ colorimeter measurements, supporting engineers' visual inspection by reducing inter-inspector variance and inspection workload. Trained exclusively on normal samples, the system provides a practical baseline independent of specific defect patterns, effectively detecting anomalies across diverse content and luminance conditions. The proposed method is immediately applicable to actual evaluation workflows and contributes to building reliable quality control systems in OLED development.

**Keywords:** OLED display inspection, unsupervised anomaly detection, hybrid ensemble learning, visual quality assessment

---

### 1. Introduction

#### 1.1. Background and Motivation

Display quality evaluation consists of quantitative assessment based on physical characteristics measured by instruments and subjective assessment based on human visual perception [1]. Objective quality measurements often differ from actual human-perceived quality, making subjective evaluation more appropriate for display quality assessment. Particularly, defects at low grayscale or low luminance levels exhibit not only severity differences but also presence/absence ambiguity depending on driving environment and observers.

OLED driving quality risk verification and defect improvement require optical characteristic measurements and visual inspection across diverse environments, functions, and patterns including dark room conditions, high/low temperature, dimming, frequency variations, and complex killer patterns (Fig. 1). Panel anomalies include predictable defects appearing at fixed locations in regular patterns and unpredictable defects appearing irregularly at arbitrary positions. Many defects manifest at low grayscale levels, causing increased verification time and potential defect escape due to inter-inspector detection variance.

Current manual inspection practices face several challenges: (1) **Inter-inspector variability**: Different inspectors exhibit varying sensitivity to subtle Mura defects, especially in low-luminance regions below 1.5 cd/m² where detection thresholds vary by ±30% among individuals; (2) **Time-intensive process**: Comprehensive quality verification under multiple driving conditions requires 2-4 hours per panel variant including setup, measurement, and visual inspection; (3) **Sparse defect distribution**: Normal samples constitute 85-90% of development evaluation datasets while defects exhibit unpredictable morphology, location, and severity; (4) **Subjective decision boundaries**: Low-grayscale Mura patterns near detection threshold lack consensus among expert inspectors, with inter-rater agreement κ = 0.68 indicating substantial but imperfect concordance.

Deep learning approaches demonstrate impressive performance on benchmark datasets including MVTec AD [2], yet industrial deployment reveals architecture-specific limitations. Memory-based methods such as PatchCore [3] excel at textural patterns but struggle with structural defects. Normalizing flow approaches including CFlow [4] and FastFlow [5] achieve superior geometric recognition yet require extensive training. Knowledge distillation techniques such as STFPM [6] offer efficiency but sacrifice sensitivity. Foundation models like Dinomaly [7] provide robust generalization with higher resource demands.

This paper addresses manual inspection limitations through AI-assisted pre-screening system operating in development phase quality assessment. Unlike mass production automation targeting throughput, our system supports engineering evaluation by: (1) reducing 2D imaging and quality verification time through batch pre-screening, (2) minimizing defect escape through improved detection capability, (3) eliminating inter-inspector variance through consistent automated analysis, and (4) enabling focus on critical cases requiring expert judgment.

#### 1.2. Contributions

This paper addresses single-model limitations through systematic evaluation and strategic ensemble integration tailored for development phase quality assessment:

1. **DefectVAD Framework**: Modular evaluation system enabling comparative analysis of 20 state-of-the-art unsupervised models across unified interfaces supporting MVTec AD [2], VisA, and custom OLED development data with XYZ colorspace preprocessing
2. **Comprehensive Comparative Analysis**: Systematic evaluation on OLED development data revealing quantitative complementarity with score correlations ranging 0.49 to 0.71 and error overlap below 20%
3. **Three-Stage Hybrid Ensemble**: Cascaded architecture achieving 99.2% AUROC through strategic integration optimized for batch processing workflow
4. **Pre-Production Ready System**: Batch processing pipeline supporting engineering evaluation with human-in-the-loop review for critical cases

---

### 2. DefectVAD Framework

#### 2.1. System Architecture

DefectVAD comprises four modular layers enabling systematic evaluation through hierarchical organization adapted from Anomalib framework [9]:

**Configuration Management Layer**: Hierarchical YAML-based configuration system manages environment setup including paths for pretrained backbone weights, dataset directories, operating system-specific settings for data loading, dataset specifications defining image dimensions and preprocessing transformations, and model hyperparameters with dynamic path resolution.

**Data Abstraction Layer**: Unified dataset interface standardizes loading, preprocessing, and augmentation across multiple formats. The framework supports MVTec AD [2] with 15 object and texture categories comprising 5,354 images; VisA dataset containing 12 industrial categories with 10,821 high-resolution images; and custom OLED development data requiring XYZ colorspace to RGB conversion through batch preprocessing pipeline. Factory pattern implementation enables dynamic dataset instantiation abstracting format-specific details.

**Model Layer**: Each model implements common interface patterns ensuring consistent behavior across diverse architectures [9]. All models inherit from abstract base defining forward computation, loss computation, and inference methods producing anomaly scores with optional pixel-level localization. Directory structure organizes implementations by model family including STFPM [6], EfficientAD [8], PatchCore [3], normalizing flows [4,5], reconstruction-based methods [11], and foundation models [7].

**Training Engine**: Unified training framework abstracts common patterns while enabling model-specific customizations through extensible hook system. Base trainer implements training loop coordination, checkpoint management, early stopping, learning rate scheduling, and metric logging.

#### 2.2. Dataset and Preprocessing

**OLED Development Evaluation Data**:
Development phase dataset acquired through XYZ tristimulus colorimeter measurements at 1920×1080 pixel resolution under various driving conditions comprises three subsets representing typical engineering evaluation scenarios. Training set contains 3,078 normal panels verified through expert consensus inspection showing no visible defects under standard viewing conditions. Validation set includes 769 normal panels reserved for hyperparameter tuning and threshold optimization. Test set encompasses 1,523 panels distributed across four defect categories encountered in development evaluation:

- **Mura defects** (673 images, 44.2%): Luminance non-uniformity appearing as cloudy or spotty patterns, particularly challenging at low grayscale levels (L < 10) where ΔE < 1.5 approaches human detection threshold
- **Color non-uniformity** (389 images, 25.5%): RGB color deviation exhibiting chromatic aberration patterns with color difference ΔE > 1.5 in localized regions, arising from differential aging of RGB OLED emissive layers or non-uniform color filter characteristics
- **Line defects** (281 images, 18.4%): Linear artifacts from process irregularities spanning 1-3 pixels width, often subtle at low luminance
- **Pixel anomalies** (180 images, 11.9%): Individual pixel irregularities including dead pixels and color deviations requiring high-resolution preservation

**XYZ to RGB Conversion Pipeline**:
XYZ colorspace measurements undergo systematic conversion to RGB format suitable for deep learning models pretrained on natural images. Pipeline stages include: (1) loading raw XYZ tristimulus values from measurement files, (2) applying CIE standard illuminant D65 color transformation matrix, (3) performing gamma correction with exponent 2.2 for sRGB conformance, (4) normalizing intensity values to [0,1] range, and (5) encoding as 8-bit PNG format. Batch processing achieves throughput of approximately 1,000 images per minute through parallel CPU execution, reducing storage from 12MB per image in raw XYZ to 2-3MB in compressed RGB with negligible perceptual quality loss.

#### 2.3. Experimental Setup

**Training Protocol**: Model-specific configurations maintain consistent evaluation framework. Optimization employs AdamW algorithm (β₁=0.9, β₂=0.999, ε=1e-8, weight decay=0.01) for improved generalization. Learning rate follows cosine annealing schedule with initialization ranging 1e-4 to 1e-3. Data augmentation applies random horizontal flipping (p=0.5), random rotation (±15°), and color jittering (brightness/contrast ±10%). Early stopping monitors validation AUROC with patience of 10 epochs. Reproducibility ensured through fixed random seed of 42.

**Evaluation Metrics**: Image-level metrics include Area Under ROC Curve (AUROC) measuring ranking quality and Area Under Precision-Recall Curve (AUPR) emphasizing imbalanced dataset performance. Pixel-level metrics computed for localization-capable models include pixel AUROC and pixel AUPR. Threshold-dependent metrics comprise F1-score, precision, recall, and accuracy computed at four threshold selection strategies. Statistical validation employs 5-fold stratified cross-validation maintaining defect type distributions. Performance metrics reported as mean with 95% confidence intervals. Statistical significance determined through paired t-tests with α = 0.05.

---

### 3. Comparative Evaluation Results

#### 3.1. Overall Performance

Table I summarizes top-five models demonstrating significant architectural variation across the accuracy-efficiency spectrum, essential for batch processing optimization in development evaluation workflows.

**TABLE I: TOP-FIVE MODEL PERFORMANCE**

| Model             | Architecture Category  | AUROC (%) | F1-Score | Params (M) |
|-------------------|------------------------|-----------|----------|------------|
| Dinomaly-Base-322 | Foundation Model [7]   | 98.5      | 0.943    | 86.2       |
| FastFlow-CaiT     | Normalizing Flow [5]   | 98.3      | 0.939    | 55.8       |
| PatchCore         | Memory-Based [3]       | 97.8      | 0.931    | 23.5       |
| CFlow             | Normalizing Flow [4]   | 97.9      | 0.933    | 45.3       |
| STFPM             | Knowledge Distill. [6] | 96.8      | 0.921    | 38.7       |

Category-wise performance aggregation reveals systematic patterns. Foundation models [7] achieve highest average 98.1% AUROC with lowest standard deviation 0.8% demonstrating superior consistency across defect categories. Normalizing flow models [4,5] average 97.6% (σ=1.1%). Memory-based methods [3,10] average 96.9% (σ=1.3%). Knowledge distillation approaches [6] average 95.8% (σ=1.7%). Reconstruction-based methods [11] average 91.2% (σ=2.4%).

#### 3.2. Defect-Specific Analysis

Table II demonstrates architecture-defect affinity patterns validating complementarity hypothesis underlying ensemble design.

**TABLE II: DEFECT-TYPE SPECIFIC PERFORMANCE**

| Defect Type         | Best Model      | AUROC (%) | Key Detection Mechanism                     |
|---------------------|-----------------|-----------|---------------------------------------------|
| Mura                | PatchCore [3]   | 98.4      | Local distribution modeling                 |
| Color Non-uniformity| Dinomaly [7]    | 99.1      | Fine-grained RGB channel features           |
| Line                | FastFlow [5]    | 98.7      | Geometric pattern recognition               |
| Pixel               | Dinomaly-Large  | 98.9      | High-resolution preservation                |

**Performance Analysis by Defect Characteristics**:
Mura defects characterized by textural non-uniformity favor memory-based methods [3] through patch-level distribution modeling. PatchCore coreset selection retains representative texture patterns while maintaining computational efficiency. XYZ colorspace acquisition provides advantage through direct luminance channel access enabling sensitive detection of color differences below 1.5 ΔE units imperceptible in standard RGB imaging.

Color non-uniformity presenting chromatic aberration patterns benefits from foundation models [7] processing at higher resolutions with enhanced RGB channel feature extraction. Dinomaly operating at 322×322 pixel resolution preserves fine-grained color information essential for detecting subtle color deviations (ΔE > 1.5) arising from differential OLED emissive layer aging—challenging for standard 224×224 resolution models. XYZ to RGB conversion through CIE D65 transformation maintains color fidelity critical for chromatic defect detection.

Line defects exhibiting structural patterns favor normalizing flow models [5] through precise geometric modeling. FastFlow 2D flow architecture effectively captures linear structures spanning 1-3 pixels width and 50-800 pixels length across multiple spatial scales.

Pixel anomalies demanding single-pixel precision benefit from foundation models processing maximum resolution, maintaining pixel-level information often lost through aggressive downsampling in standard pipelines.

#### 3.3. Complementarity Analysis

**Score Correlation Analysis**: Pairwise Pearson correlation coefficients between model anomaly scores quantify architectural diversity essential for effective ensemble integration:

|             | PatchCore | Dinomaly | FastFlow | EfficientAD |
|-------------|-----------|----------|----------|-------------|
| PatchCore   | 1.00      | 0.71     | 0.67     | 0.52        |
| Dinomaly    | 0.71      | 1.00     | 0.64     | 0.49        |
| FastFlow    | 0.67      | 0.64     | 1.00     | 0.55        |
| EfficientAD | 0.52      | 0.49     | 0.55     | 1.00        |

Moderate positive correlations ranging 0.49-0.71 confirm models capture different aspects of anomaly patterns rather than redundant information. Observed moderate correlations optimal for ensemble integration: models agree on obvious cases while differing on challenging samples.

**Error Pattern Analysis**: Detailed examination of false negative cases reveals true complementarity. PatchCore generates 32 unique false negatives primarily on textural boundary ambiguities. Dinomaly produces 28 unique errors predominantly on challenging lighting conditions. FastFlow misses 35 unique cases mainly subtle Mura patterns lacking strong geometric structure. Combined coverage reaches 91% with at least one model correctly detecting anomaly in 91% of all positive test cases. Error overlap below 20% between any model pair indicates true complementarity—models systematically fail on different samples rather than exhibiting common systematic failures.

---

### 4. Hybrid Ensemble Architecture

#### 4.1. Three-Stage Cascaded Design

Proposed architecture leverages computational efficiency while maximizing detection accuracy through hierarchical processing strategy optimized for batch evaluation workflow (Fig. 2).

**Stage One: Fast Screening**
Initial filtering employs EfficientAD-Small model [8] with compact 12.3M parameters achieving efficient batch processing. Dual threshold strategy separates samples into three categories: (1) certain normal with anomaly scores below 0.25 comprising approximately 70% classified as normal requiring no further processing; (2) certain anomaly with scores exceeding 0.75 comprising approximately 5% immediately classified as defective; (3) uncertain with scores between 0.25-0.75 comprising approximately 25% requiring deeper analysis in subsequent stages. Conservative thresholds deliberately tolerate higher false positive rate in normal classification while maintaining near-perfect recall on anomalies, prioritizing not missing defects over computational efficiency.

**Stage Two: Defect-Type Routing with Selective Ensemble**
Uncertain samples undergo lightweight pattern classification examining spatial variance, local brightness maxima count, and edge density. Classification rules route samples to specialized model combinations: Low spatial variance (<0.05) with sparse bright spots (<3) indicates Mura pattern triggering PatchCore [3] (w=0.70) + Dinomaly [7] (w=0.30). RGB channel variance exceeding threshold suggests color non-uniformity triggering Dinomaly (w=0.80) + EfficientAD [8] (w=0.20), leveraging Dinomaly's fine-grained color feature sensitivity. High edge density (>0.3) indicates line defect triggering FastFlow [5] (w=0.60) + Dinomaly (w=0.40). Ambiguous patterns trigger full three-model ensemble (0.42/0.38/0.20).

**Stage Three: Confidence-Based Review Queue**
Final classification incorporates confidence estimation guiding human review prioritization. Confidence metric computed as |score - threshold| quantifies decision certainty. Low confidence samples (confidence < 0.15) flagged for expert review comprising approximately 2.5% of total volume. High confidence samples receive automated classification without manual intervention. Human-in-the-loop mechanism ensures critical decisions receive expert validation while automating straightforward cases, supporting engineering evaluation workflow rather than replacing expert judgment.

#### 4.2. Score Fusion Strategy

**Weighted Voting Mechanism**: Final anomaly score computed as weighted linear combination: S_final = 0.42·S_PatchCore + 0.38·S_Dinomaly + 0.20·S_EfficientAD. Weights sum to unity ensuring score interpretability.

**Weight Optimization Process**: Grid search on validation set explores weight space with exhaustive search over 11×11 grid evaluating 121 weight combinations maximizing validation AUROC. Adaptive optimization outperforms alternatives: Uniform weighting (0.333 each) achieves 98.7% AUROC; performance-based weighting proportional to individual AUROCs achieves 99.0%; adaptive validation-optimized weighting achieves 99.2% selected as final configuration.

Weight interpretation: PatchCore receives highest weight 0.42 reflecting importance of Mura detection given 44.2% prevalence; Dinomaly receives moderate weight 0.38 for balanced cross-category performance including color non-uniformity detection; EfficientAD receives lowest weight 0.20 contributing complementary perspective with minimal parameter overhead.

**Category-Specific Threshold Optimization**: Per-defect-type optimal thresholds determined through F1-score maximization on validation set:

| Defect Category      | Optimal Threshold | F1-Score | Strategic Rationale                          |
|----------------------|-------------------|----------|----------------------------------------------|
| Mura                 | 0.234             | 0.967    | Lower threshold increases sensitivity        |
| Color Non-uniformity | 0.189             | 0.982    | Lowest threshold prioritizes high recall     |
| Line                 | 0.267             | 0.971    | Higher threshold requires structural confidence |
| Pixel                | 0.198             | 0.976    | Balanced precision-recall trade-off          |

Category-specific thresholding improves overall F1-score by 1.0 percentage point over global threshold (0.961 vs. 0.951) by accommodating distinct score distributions across defect types.

#### 4.3. Batch Processing Workflow

**Batch Processing Architecture**: DefectVAD operates in offline batch mode suitable for development phase evaluation workflow. Daily evaluation batches undergo systematic processing: (1) XYZ colorspace measurement data collection from 2D colorimeter, (2) RGB batch conversion through parallel CPU processing achieving ~1,000 images per minute throughput, (3) Three-stage ensemble inference distributing computational load across specialized model combinations, (4) Confidence-based classification generating automated decisions for high-confidence cases (~97.5%) and review queue for low-confidence cases (~2.5%).

**Efficiency Optimization**: Cascaded architecture achieves computational efficiency through strategic sample routing. Stage one fast screening eliminates ~70% obvious normal cases using lightweight EfficientAD-Small (12.3M parameters), avoiding unnecessary inference on straightforward samples. Stage two selective ensemble processes ~25% uncertain cases through defect-type routing, applying specialized model combinations rather than full ensemble for all samples. This selective processing reduces overall computational burden while maintaining detection accuracy, demonstrating 42.6% efficiency improvement compared to full ensemble baseline.

---

### 5. Results and Discussion

#### 5.1. Performance Comparison

Table III demonstrates consistent improvement across all evaluation metrics with statistical significance confirmed through paired t-test (p = 0.003 < α = 0.05).

**TABLE III: HYBRID ENSEMBLE VS. INDIVIDUAL MODEL PERFORMANCE**

| Metric            | Hybrid Ensemble | Best Individual | Absolute Gain | Relative Gain |
|-------------------|-----------------|-----------------|---------------|---------------|
| Image-Level AUROC | 99.2%           | 98.5% (Dinomaly)| +0.7 pp       | +3.8%         |
| Image-Level AUPR  | 98.4%           | 97.6% (Dinomaly)| +0.8 pp       | +0.8%         |
| F1-Score          | 0.961           | 0.943 (Dinomaly)| +0.018        | +1.9%         |
| Precision         | 98.9%           | 97.1%           | +1.8 pp       | +1.9%         |
| Recall            | 99.1%           | 98.3%           | +0.8 pp       | +0.8%         |

**Per-Category Performance Analysis**:

| Defect Category      | Hybrid Ensemble | Best Individual | Best Model      | Absolute Gain |
|----------------------|-----------------|-----------------|-----------------|---------------|
| Mura                 | 99.1%           | 98.4%           | PatchCore [3]   | +0.7 pp       |
| Color Non-uniformity | 99.6%           | 99.1%           | Dinomaly [7]    | +0.5 pp       |
| Line                 | 99.3%           | 98.7%           | FastFlow [5]    | +0.6 pp       |
| Pixel                | 99.4%           | 98.9%           | Dinomaly-Large  | +0.5 pp       |

Universal performance gains across all defect categories demonstrate ensemble robustness rather than overfitting to specific defect types. Consistent improvements ranging 0.5-0.7 percentage points validate complementarity hypothesis.

#### 5.2. Ablation Studies

**Ensemble Size Optimization**:

| Model Combination | Image AUROC | F1-Score | Params (M) | Efficiency Score |
|-------------------|-------------|----------|------------|------------------|
| Best Single       | 98.5%       | 0.943    | 86.2       | 1.142            |
| Two Models        | 98.9%       | 0.954    | 109.7      | 0.901            |
| Three Models      | 99.2%       | 0.961    | 122.0      | 0.813            |
| Four Models       | 99.3%       | 0.963    | 167.3      | 0.593            |
| Five Models       | 99.3%       | 0.964    | 212.6      | 0.467            |

*Efficiency Score = AUROC / (Params/100)

Three-model configuration achieves optimal accuracy-efficiency balance. Additional models beyond three provide less than 0.1% AUROC gain while increasing parameter count disproportionately. Efficiency score peaks at two-model configuration but three-model achieves best absolute AUROC with acceptable parameter overhead.

**Component Contribution Analysis**:

| Removed Component   | Resulting AUROC | Performance Drop | Impact Analysis                        |
|---------------------|-----------------|------------------|----------------------------------------|
| None (Full System)  | 99.2%           | Baseline         | Complete system performance            |
| Remove PatchCore    | 98.7%           | -0.5 pp          | Mura detection degrades significantly  |
| Remove Dinomaly     | 98.4%           | -0.8 pp          | Overall performance suffers most       |
| Remove EfficientAD  | 98.9%           | -0.3 pp          | Complementarity loss affects marginal cases |
| Remove Stage One    | 99.2%           | 0.0 pp           | No accuracy impact, efficiency reduced |
| Remove Type Routing | 99.0%           | -0.2 pp          | Specialization loss                    |

All components contribute meaningfully to final system performance validating architectural design decisions. Dinomaly removal causes largest drop confirming its central role across defect categories including color non-uniformity detection. Stage one screening provides pure efficiency gain without accuracy penalty.

**Model Efficiency Analysis**:

| Model               | Params (M) | AUROC (%) | Efficiency Score | Deployment Suitability |
|---------------------|------------|-----------|------------------|------------------------|
| EfficientAD-Small   | 12.3       | 96.1      | 7.81             | Excellent for Stage 1  |
| PatchCore           | 23.5       | 97.8      | 4.16             | Good balance           |
| STFPM               | 38.7       | 96.8      | 2.50             | Moderate efficiency    |
| CFlow               | 45.3       | 97.9      | 2.16             | Lower efficiency       |
| FastFlow-CaiT       | 55.8       | 98.3      | 1.76             | Acceptable for Stage 2 |
| Dinomaly-Base       | 86.2       | 98.5      | 1.14             | Best accuracy          |

Efficiency score (AUROC / Params×100) reveals EfficientAD-Small as optimal Stage 1 screener balancing accuracy and model size. PatchCore offers excellent efficiency for Mura-specific detection. Dinomaly achieves highest absolute accuracy justifying larger parameter count for critical Stage 2 ensemble role.

#### 5.3. Development Phase Integration Analysis

**Operational Integration**: DefectVAD framework enables flexible development evaluation integration through modular architecture. Batch processing executes during convenient periods while model inference supports engineering evaluation workflow. Quality assurance integration presents low-confidence samples (~2.5%) through human review interface displaying original image alongside anomaly heatmap, individual model scores, confidence metrics, side-by-side normal reference comparison, and annotation tools for ground truth feedback.

Expert review decisions feed continuous improvement pipeline augmenting training dataset with challenging samples from actual development evaluation. Periodic retraining conducted as needed incorporates accumulated feedback improving model adaptation to evolving defect patterns encountered in development phase. Model weights support hot-swapping enabling updates without system rebuild. Modular architecture permits individual ensemble component upgrades independently through standardized interfaces.

**Supporting Engineering Decisions**: Unlike automated production inspection targeting defect escape prevention through 100% screening, development phase evaluation prioritizes: (1) reducing visual inspection time by pre-screening obvious normal/defective cases, (2) eliminating inter-inspector variance through consistent automated analysis providing objective reference, (3) enabling engineers to focus on borderline cases requiring expert judgment, and (4) detecting novel defect patterns not previously characterized through unsupervised learning approach.

Threshold adjustment enables precision-recall trade-off optimization per evaluation priorities without retraining. Conservative thresholds increase recall minimizing missed defects at cost of higher review workload. Aggressive thresholds increase precision reducing unnecessary inspections at risk of missing subtle defects. Category-specific thresholds tunable through configuration files enabling rapid adaptation to changing evaluation standards.

**Color Non-uniformity Detection Impact**: Integration of color non-uniformity as primary defect category (25.5% of test set) addresses critical gap in traditional luminance-focused inspection. XYZ colorspace measurements provide direct access to chromatic information enabling detection of subtle RGB channel deviations (ΔE > 1.5) arising from differential OLED aging [12]. Dinomaly's fine-grained feature extraction at 322×322 resolution proves particularly effective for chromatic defect detection, achieving 99.1% individual AUROC and contributing to ensemble's 99.6% category-specific performance. This capability reduces reliance on subjective color perception judgment among inspectors, standardizing chromatic defect classification.

#### 5.4. Limitations and Future Directions

**Current System Limitations**: Single-panel analysis treats each display independently without exploiting spatial consistency across panel array or temporal consistency in development iterations. Static ensemble weights remain fixed despite potential variations in defect patterns across development phases. Defect taxonomy limited to four predefined categories requires framework extension for emerging defect types encountered in new product development.

**Future Research Directions**: Model distillation research could transfer knowledge from three-model ensemble to single lightweight network maintaining 99%+ accuracy while reducing parameter count for resource-constrained deployment. Active learning strategies could dynamically adjust ensemble weights based on expert review feedback optimizing performance for current development phase defect distribution. Multi-modal integration incorporating additional sensor modalities including thermal imaging detecting temperature anomalies and depth sensing identifying surface irregularities could enhance defect characterization beyond single-modality limitations. Transfer learning research extending framework to additional display technologies including LCD and microLED requires domain adaptation techniques accounting for different defect characteristics while leveraging shared knowledge from OLED evaluation experience.

---

### 6. Conclusion

This paper presented DefectVAD (Defect Vision Anomaly Detection), a comprehensive evaluation framework enabling systematic comparison of 20 state-of-the-art unsupervised anomaly detection models for OLED display quality assessment in development phase evaluation. Through modular architecture supporting unified dataset abstraction, model training coordination, and performance evaluation across diverse benchmarks [2] and custom industrial data, we revealed complementary architectural strengths across defect types with quantitative validation demonstrating score correlations ranging 0.49-0.71 and error overlap below 20%.

The proposed three-stage cascaded hybrid ensemble strategically integrates PatchCore [3] providing texture expertise for Mura defects, Dinomaly [7] offering balanced cross-category performance including color non-uniformity detection, and EfficientAD [8] enabling efficient screening through adaptive weighted fusion with category-specific threshold optimization. Operating within batch processing environment, the ensemble achieves 99.2% image-level AUROC and 0.961 F1-score—representing 3.8% relative improvement over best individual models with statistical significance (p = 0.003).

Performance validation demonstrates consistent gains across all defect categories: Mura defects improve from 98.4% to 99.1%, color non-uniformity from 99.1% to 99.6%, line defects from 98.7% to 99.3%, and pixel anomalies from 98.9% to 99.4%. Model efficiency analysis reveals optimal three-model configuration balancing accuracy (99.2% AUROC) with parameter efficiency (122M total parameters, efficiency score 0.813).

Systematic evaluation confirmed no single architecture dominates across diverse OLED defect types, validating hybrid approaches exploiting model complementarity as superior solution for specialized industrial vision applications. Memory-based methods [3,10] excel at textural patterns, normalizing flows [4,5] at structural anomalies, and foundation models [7] at consistent generalization including chromatic defect detection—strategic integration leverages these complementary strengths overcoming individual model limitations.

The framework addresses critical challenges in development phase quality assessment: reducing visual inspection time through automated pre-screening, eliminating inter-inspector variance through consistent analysis, and supporting engineering decisions through confidence-based review prioritization. Trained exclusively on normal samples, the unsupervised approach provides practical baseline independent of specific defect patterns, effectively detecting anomalies across diverse content and luminance conditions without requiring defect annotations.

Future research directions include model distillation for deployment efficiency, active learning for dynamic optimization responding to development phase feedback, multi-modal sensor fusion for enhanced characterization, and transfer learning to additional display technologies. The comprehensive evaluation methodology and modular framework architecture established through this work provide foundation for advancing industrial anomaly detection research toward practical deployment in development phase quality control applications, contributing to building reliable quality management systems in OLED development.

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
│        Input: OLED Development Evaluation Batch          │
│    (XYZ Colorspace → RGB Batch Conversion)               │
└───────────────────────┬──────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────────┐
        │      Stage 1: Fast Screening          │
        │   Model: EfficientAD-Small [8]        │
        │   Params: 12.3M                       │
        │   Strategy: Dual threshold filtering  │
        └───────────────┬───────────────────────┘
                        ↓
        ┌───────────────┴────────────────┐
        ↓               ↓                ↓
   Score < 0.25    0.25 ≤ S ≤ 0.75   Score > 0.75
   Certain Normal  Uncertain Cases   Certain Anomaly
      (~70%)           (~25%)            (~5%)
        ↓               ↓                ↓
   CLASSIFIED      Next Stage       CLASSIFIED
                        ↓
        ┌───────────────────────────────────────┐
        │   Stage 2: Defect-Type Routing        │
        │   Pattern Classification & Selective  │
        │   Model Ensemble                      │
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
 (0.7/0.3) (0.8/0.2) (0.6/0.4) (0.42/0.38/0.20)
     ↓        ↓         ↓         ↓         ↓
     └────────┴─────────┴─────────┴─────────┘
                        ↓
      ┌───────────────────────────────────────┐
      │   Stage 3: Confidence Assessment      │
      │   Metric: |Score - Threshold|         │
      │   Human-in-the-Loop Review Queue      │
      └─────────────────┬─────────────────────┘
                        ↓
            ┌───────────┴────────────┐
            ↓                        ↓
     High Confidence          Low Confidence
     (Confidence ≥ 0.15)      (Confidence < 0.15)
        (~97.5%)                  (~2.5%)
            ↓                        ↓
     Automated Decision       Expert Review Queue
     Final Classification     Engineering Validation
            ↓                        ↓
            └────────┬───────────────┘
                     ↓
        ┌────────────────────────────────────────┐
        │       Final Output Statistics           │
        │   Image-Level AUROC: 99.2%             │
        │   F1-Score: 0.961                      │
        │   Total Params: 122M (3 models)        │
        │   Supports Development Phase           │
        │   Quality Assessment Workflow          │
        └────────────────────────────────────────┘
```

**Figure Caption**: Three-stage cascaded architecture optimized for development phase quality assessment. Stage one employs fast screening with lightweight EfficientAD-Small (12.3M params) eliminating ~70% obvious normal cases. Stage two applies defect-type routing directing uncertain samples to specialized model combinations optimized for Mura, color non-uniformity, line, and pixel defects. Stage three implements confidence-based review queue flagging ambiguous cases (~2.5%) for expert engineering validation, supporting human decision-making rather than replacing it.
