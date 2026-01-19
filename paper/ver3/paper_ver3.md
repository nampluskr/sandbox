## DefectVAD: Defect Vision Anomaly Detection
## Hybrid Ensemble Framework for Unsupervised OLED Display Defect Detection in Pre-Production Quality Assessment

### Abstract

Display quality verification and defect judgment in OLED development rely on subjective visual inspection under various driving conditions including luminance, temperature, and frequency variations. Manual inspection requires significant time and effort, with persistent risk of defect escape due to inter-inspector variability. Normal samples dominate the dataset while defects are sparse with unpredictable shapes, positions, and severity levels, making single-threshold classification unstable. This paper presents DefectVAD (Defect Vision Anomaly Detection), a comprehensive evaluation framework enabling systematic comparison of contemporary unsupervised anomaly detection models for pre-production OLED quality assessment. We propose a three-stage cascaded hybrid ensemble integrating complementary detection approaches through adaptive weighted fusion, demonstrating substantial improvement over individual models in preliminary evaluation. The framework operates in batch processing mode on XYZ colorimeter measurements, supporting engineers' visual inspection by reducing inter-inspector variance and inspection workload. Trained exclusively on normal samples, the system provides a practical baseline independent of specific defect patterns, effectively detecting anomalies across diverse content and luminance conditions. The proposed method is immediately applicable to actual evaluation workflows and contributes to building reliable quality control systems in OLED development.

**Keywords:** OLED display inspection, unsupervised anomaly detection, hybrid ensemble learning, visual quality assessment

---

### 1. Introduction

#### 1.1. Background and Motivation

Display quality evaluation consists of quantitative assessment based on physical characteristics measured by instruments and subjective assessment based on human visual perception [1]. Objective quality measurements often differ from actual human-perceived quality, making subjective evaluation more appropriate for display quality assessment. Particularly, defects at low grayscale or low luminance levels exhibit not only severity differences but also presence/absence ambiguity depending on driving environment and observers.

OLED driving quality risk verification and defect improvement require optical characteristic measurements and visual inspection across diverse environments, functions, and patterns including dark room conditions, high/low temperature, dimming, frequency variations, and complex killer patterns (Fig. 1). Panel anomalies include predictable defects appearing at fixed locations in regular patterns and unpredictable defects appearing irregularly at arbitrary positions. Many defects manifest at low grayscale levels, causing increased verification time and potential defect escape due to inter-inspector detection variance.

Current manual inspection practices face several challenges: (1) **Inter-inspector variability**: Different inspectors exhibit varying sensitivity to subtle Mura defects, especially in low-luminance regions where detection thresholds vary significantly among individuals; (2) **Time-intensive process**: Comprehensive quality verification under multiple driving conditions requires substantial time per panel variant including setup, measurement, and visual inspection; (3) **Sparse defect distribution**: Normal samples constitute majority of development evaluation datasets while defects exhibit unpredictable morphology, location, and severity; (4) **Subjective decision boundaries**: Low-grayscale Mura patterns near detection threshold lack consensus among expert inspectors, with inter-rater agreement indicating substantial but imperfect concordance.

Deep learning approaches demonstrate impressive performance on benchmark datasets including MVTec AD [2], yet industrial deployment reveals architecture-specific limitations. Memory-based methods such as PatchCore [3] excel at textural patterns but struggle with structural defects. Normalizing flow approaches including CFlow [4] and FastFlow [5] achieve superior geometric recognition yet require extensive training. Knowledge distillation techniques such as STFPM [6] offer efficiency but sacrifice sensitivity. Foundation models like Dinomaly [7] provide robust generalization with higher resource demands.

This paper addresses manual inspection limitations through AI-assisted pre-screening system operating in development phase quality assessment. Unlike mass production automation targeting throughput, our system supports engineering evaluation by: (1) reducing 2D imaging and quality verification time through batch pre-screening, (2) minimizing defect escape through improved detection capability, (3) eliminating inter-inspector variance through consistent automated analysis, and (4) enabling focus on critical cases requiring expert judgment.

#### 1.2. Contributions

This paper addresses single-model limitations through systematic evaluation and strategic ensemble integration tailored for development phase quality assessment:

1. **DefectVAD Framework**: Modular evaluation system enabling comparative analysis of contemporary unsupervised models across unified interfaces supporting MVTec AD [2], VisA, and custom OLED development data with XYZ colorspace preprocessing
2. **Comprehensive Comparative Analysis**: Systematic evaluation on OLED development data revealing quantitative complementarity across diverse architectural approaches
3. **Three-Stage Hybrid Ensemble**: Cascaded architecture achieving substantial improvement through strategic integration optimized for batch processing workflow
4. **Pre-Production Ready System**: Batch processing pipeline supporting engineering evaluation with human-in-the-loop review for critical cases

---

### 2. DefectVAD Framework

#### 2.1. System Architecture

DefectVAD comprises four modular layers enabling systematic evaluation through hierarchical organization adapted from Anomalib framework [9]:

**Configuration Management Layer**: Hierarchical YAML-based configuration system manages environment setup including paths for pretrained backbone weights, dataset directories, operating system-specific settings for data loading, dataset specifications defining image dimensions and preprocessing transformations, and model hyperparameters with dynamic path resolution.

**Data Abstraction Layer**: Unified dataset interface standardizes loading, preprocessing, and augmentation across multiple formats. The framework supports MVTec AD [2] with multiple object and texture categories; VisA dataset containing industrial categories with high-resolution images; and custom OLED development data requiring XYZ colorspace to RGB conversion through batch preprocessing pipeline. Factory pattern implementation enables dynamic dataset instantiation abstracting format-specific details.

**Model Layer**: Each model implements common interface patterns ensuring consistent behavior across diverse architectures [9]. All models inherit from abstract base defining forward computation, loss computation, and inference methods producing anomaly scores with optional pixel-level localization. Directory structure organizes implementations by model family including STFPM [6], EfficientAD [8], PatchCore [3], normalizing flows [4,5], reconstruction-based methods [11], and foundation models [7].

**Training Engine**: Unified training framework abstracts common patterns while enabling model-specific customizations through extensible hook system. Base trainer implements training loop coordination, checkpoint management, early stopping, learning rate scheduling, and metric logging.

#### 2.2. Dataset and Preprocessing

**OLED Development Evaluation Data**:
Development phase dataset acquired through XYZ tristimulus colorimeter measurements at high resolution under various driving conditions comprises three subsets representing typical engineering evaluation scenarios. Training set contains normal panels verified through expert consensus inspection showing no visible defects under standard viewing conditions. Validation set includes normal panels reserved for hyperparameter tuning and threshold optimization. Test set encompasses panels distributed across four defect categories encountered in development evaluation:

- **Mura defects**: Luminance non-uniformity appearing as cloudy or spotty patterns, particularly challenging at low grayscale levels where color differences approach human detection threshold
- **Color non-uniformity**: RGB color deviation exhibiting chromatic aberration patterns with localized color differences, arising from differential aging of RGB OLED emissive layers or non-uniform color filter characteristics
- **Line defects**: Linear artifacts from process irregularities spanning narrow widths, often subtle at low luminance
- **Pixel anomalies**: Individual pixel irregularities including dead pixels and color deviations requiring high-resolution preservation

**XYZ to RGB Conversion Pipeline**:
XYZ colorspace measurements undergo systematic conversion to RGB format suitable for deep learning models pretrained on natural images. Pipeline stages include: (1) loading raw XYZ tristimulus values from measurement files, (2) applying CIE standard illuminant D65 color transformation matrix, (3) performing gamma correction for sRGB conformance, (4) normalizing intensity values to standard range, and (5) encoding as compressed format. Batch processing achieves efficient throughput through parallel CPU execution, reducing storage requirements substantially with negligible perceptual quality loss.

#### 2.3. Experimental Setup

**Training Protocol**: Model-specific configurations maintain consistent evaluation framework. Optimization employs AdamW algorithm with standard hyperparameters for improved generalization. Learning rate follows cosine annealing schedule with model-appropriate initialization. Data augmentation applies random horizontal flipping, random rotation, and color jittering. Early stopping monitors validation metrics with appropriate patience. Reproducibility ensured through fixed random seed across all experiments.

**Evaluation Metrics**: Image-level metrics include Area Under ROC Curve (AUROC) measuring ranking quality and Area Under Precision-Recall Curve (AUPR) emphasizing imbalanced dataset performance. Pixel-level metrics computed for localization-capable models include pixel AUROC and pixel AUPR. Threshold-dependent metrics comprise F1-score, precision, recall, and accuracy computed at multiple threshold selection strategies. Statistical validation employs stratified cross-validation maintaining defect type distributions. Performance metrics reported with confidence intervals. Statistical significance determined through paired statistical tests.

---

### 3. Comparative Evaluation Results

#### 3.1. Overall Performance

Table I summarizes representative models demonstrating architectural variation across the accuracy-efficiency spectrum, essential for batch processing optimization in development evaluation workflows.

**TABLE I: REPRESENTATIVE MODEL PERFORMANCE**

| Model             | Architecture Category  | AUROC      | F1-Score   | Params     |
|-------------------|------------------------|------------|------------|------------|
| Foundation Model  | Transformer-based [7]  | High       | High       | Large      |
| Normalizing Flow  | 2D Flow [5]           | High       | High       | Medium     |
| Memory-based      | Coreset [3]           | Moderate   | Moderate   | Moderate   |
| Normalizing Flow  | Conditional [4]        | Moderate   | Moderate   | Medium     |
| Knowledge Distill.| Teacher-Student [6]    | Moderate   | Moderate   | Medium     |

Category-wise performance aggregation reveals systematic patterns. Foundation models achieve highest average performance with lowest variance demonstrating superior consistency across defect categories. Normalizing flow models show competitive performance with moderate variance. Memory-based methods demonstrate moderate performance with acceptable variance. Knowledge distillation approaches show lower performance with higher variance. Reconstruction-based methods show lowest performance with highest variance.

#### 3.2. Defect-Specific Analysis

Table II demonstrates architecture-defect affinity patterns validating complementarity hypothesis underlying ensemble design.

**TABLE II: DEFECT-TYPE AFFINITY PATTERNS**

| Defect Type         | Best Architecture | Performance | Key Detection Mechanism                     |
|---------------------|-------------------|-------------|---------------------------------------------|
| Mura                | Memory-based [3]  | Superior    | Local distribution modeling                 |
| Color Non-uniformity| Foundation [7]    | Superior    | Fine-grained RGB channel features           |
| Line                | Normalizing Flow [5]| Superior  | Geometric pattern recognition               |
| Pixel               | Foundation        | Superior    | High-resolution preservation                |

**Performance Analysis by Defect Characteristics**:
Mura defects characterized by textural non-uniformity favor memory-based methods [3] through patch-level distribution modeling. PatchCore coreset selection retains representative texture patterns while maintaining computational efficiency. XYZ colorspace acquisition provides advantage through direct luminance channel access enabling sensitive detection of subtle color differences imperceptible in standard RGB imaging.

Color non-uniformity presenting chromatic aberration patterns benefits from foundation models [7] processing at higher resolutions with enhanced RGB channel feature extraction. Higher resolution processing preserves fine-grained color information essential for detecting subtle color deviations arising from differential OLED emissive layer aging. XYZ to RGB conversion through CIE D65 transformation maintains color fidelity critical for chromatic defect detection.

Line defects exhibiting structural patterns favor normalizing flow models [5] through precise geometric modeling. 2D flow architecture effectively captures linear structures across multiple spatial scales through hierarchical feature extraction.

Pixel anomalies demanding single-pixel precision benefit from foundation models processing maximum resolution, maintaining pixel-level information often lost through aggressive downsampling in standard pipelines.

#### 3.3. Complementarity Analysis

**Score Correlation Analysis**: Pairwise correlation coefficients between model anomaly scores quantify architectural diversity essential for effective ensemble integration:

|             | Memory | Foundation | Flow | Distillation |
|-------------|--------|-----------|------|--------------|
| Memory      | 1.00   | Moderate  | Moderate | Low      |
| Foundation  | Moderate | 1.00    | Moderate | Low      |
| Flow        | Moderate | Moderate | 1.00 | Low          |
| Distillation| Low    | Low       | Low  | 1.00         |

Moderate positive correlations confirm models capture different aspects of anomaly patterns rather than redundant information. Observed correlations optimal for ensemble integration: models agree on obvious cases while differing on challenging samples.

**Error Pattern Analysis**: Detailed examination of false negative cases reveals true complementarity. Memory-based methods generate unique false negatives primarily on textural boundary ambiguities. Foundation models produce unique errors predominantly on challenging lighting conditions. Flow models miss unique cases mainly subtle patterns lacking strong geometric structure. Combined coverage demonstrates substantial improvement with multiple models correctly detecting anomalies in majority of positive test cases. Low error overlap between model pairs indicates true complementarity—models systematically fail on different samples rather than exhibiting common systematic failures.

---

### 4. Hybrid Ensemble Architecture

#### 4.1. Three-Stage Cascaded Design

Proposed architecture leverages computational efficiency while maximizing detection accuracy through hierarchical processing strategy optimized for batch evaluation workflow (Fig. 2).

**Stage One: Fast Screening**
Initial filtering employs lightweight model [8] with compact parameters achieving efficient batch processing. Dual threshold strategy separates samples into three categories: (1) certain normal with low anomaly scores classified as normal requiring no further processing (majority); (2) certain anomaly with high scores immediately classified as defective (small fraction); (3) uncertain with intermediate scores requiring deeper analysis in subsequent stages (moderate fraction). Conservative thresholds deliberately tolerate higher false positive rate in normal classification while maintaining near-perfect recall on anomalies, prioritizing not missing defects over computational efficiency.

**Stage Two: Defect-Type Routing with Selective Ensemble**
Uncertain samples undergo lightweight pattern classification examining spatial variance, local brightness maxima count, and edge density. Classification rules route samples to specialized model combinations: Low spatial variance with sparse bright spots indicates Mura pattern triggering memory-based plus foundation model combination. RGB channel variance exceeding threshold suggests color non-uniformity triggering foundation plus efficient model combination, leveraging fine-grained color feature sensitivity. High edge density indicates line defect triggering flow plus foundation combination. Ambiguous patterns trigger full ensemble.

**Stage Three: Confidence-Based Review Queue**
Final classification incorporates confidence estimation guiding human review prioritization. Confidence metric computed as score distance from threshold quantifies decision certainty. Low confidence samples flagged for expert review comprising small fraction of total volume. High confidence samples receive automated classification without manual intervention. Human-in-the-loop mechanism ensures critical decisions receive expert validation while automating straightforward cases, supporting engineering evaluation workflow rather than replacing expert judgment.

#### 4.2. Score Fusion Strategy

**Weighted Voting Mechanism**: Final anomaly score computed as weighted linear combination of ensemble member scores. Weights sum to unity ensuring score interpretability.

**Weight Optimization Process**: Grid search on validation set explores weight space evaluating multiple combinations maximizing validation performance. Adaptive optimization outperforms alternatives: Uniform weighting achieves baseline performance; performance-based weighting proportional to individual model performance achieves improved performance; adaptive validation-optimized weighting achieves best performance selected as final configuration.

Weight interpretation: Memory-based method receives highest weight reflecting importance of dominant defect category; foundation model receives moderate weight for balanced cross-category performance including color non-uniformity detection; efficient model receives lowest weight contributing complementary perspective with minimal parameter overhead.

**Category-Specific Threshold Optimization**: Per-defect-type optimal thresholds determined through F1-score maximization on validation set:

| Defect Category      | Threshold Strategy | Performance | Strategic Rationale                          |
|----------------------|-------------------|-------------|----------------------------------------------|
| Mura                 | Lower             | High        | Increases sensitivity to subtle patterns     |
| Color Non-uniformity | Lowest            | Highest     | Prioritizes high recall                      |
| Line                 | Higher            | High        | Requires structural confidence               |
| Pixel                | Moderate          | High        | Balanced precision-recall trade-off          |

Category-specific thresholding improves overall performance over global threshold by accommodating distinct score distributions across defect types.

#### 4.3. Batch Processing Workflow

**Batch Processing Architecture**: DefectVAD operates in offline batch mode suitable for development phase evaluation workflow. Evaluation batches undergo systematic processing: (1) XYZ colorspace measurement data collection from 2D colorimeter, (2) RGB batch conversion through parallel CPU processing achieving efficient throughput, (3) Three-stage ensemble inference distributing computational load across specialized model combinations, (4) Confidence-based classification generating automated decisions for high-confidence cases and review queue for low-confidence cases.

**Efficiency Optimization**: Cascaded architecture achieves computational efficiency through strategic sample routing. Stage one fast screening eliminates majority obvious normal cases using lightweight model with compact parameters, avoiding unnecessary inference on straightforward samples. Stage two selective ensemble processes moderate fraction uncertain cases through defect-type routing, applying specialized model combinations rather than full ensemble for all samples. This selective processing reduces overall computational burden while maintaining detection accuracy, demonstrating substantial efficiency improvement compared to full ensemble baseline.

---

### 5. Results and Discussion

#### 5.1. Performance Comparison

Table III demonstrates consistent improvement across evaluation metrics with statistical significance confirmed through appropriate statistical testing.

**TABLE III: HYBRID ENSEMBLE VS. INDIVIDUAL MODEL PERFORMANCE**

| Metric            | Hybrid Ensemble | Best Individual | Improvement   |
|-------------------|-----------------|-----------------|---------------|
| Image-Level AUROC | Superior        | High (Foundation)| Substantial   |
| Image-Level AUPR  | Superior        | High (Foundation)| Moderate      |
| F1-Score          | Superior        | High (Foundation)| Moderate      |
| Precision         | Superior        | High            | Moderate      |
| Recall            | Superior        | High            | Moderate      |

**Per-Category Performance Analysis**:

| Defect Category      | Hybrid Ensemble | Best Individual | Best Architecture | Improvement |
|----------------------|-----------------|-----------------|-------------------|-------------|
| Mura                 | Superior        | High            | Memory [3]        | Moderate    |
| Color Non-uniformity | Superior        | High            | Foundation [7]    | Moderate    |
| Line                 | Superior        | High            | Flow [5]          | Moderate    |
| Pixel                | Superior        | High            | Foundation        | Moderate    |

Universal performance gains across all defect categories demonstrate ensemble robustness rather than overfitting to specific defect types. Consistent improvements validate complementarity hypothesis.

#### 5.2. Ablation Studies

**Ensemble Size Optimization**:

| Model Combination | Image AUROC | F1-Score | Params | Efficiency |
|-------------------|-------------|----------|--------|------------|
| Best Single       | High        | High     | Large  | Moderate   |
| Two Models        | Higher      | Higher   | Larger | Lower      |
| Three Models      | Highest     | Highest  | Larger | Optimal    |
| Four Models       | Marginal    | Marginal | Much Larger | Lower |
| Five Models       | Minimal     | Minimal  | Much Larger | Lowest |

Three-model configuration achieves optimal accuracy-efficiency balance. Additional models beyond three provide minimal gains while increasing parameter count disproportionately. Efficiency score peaks at optimal configuration balancing accuracy with parameter overhead.

**Component Contribution Analysis**:

| Removed Component   | Resulting Performance | Drop    | Impact Analysis                        |
|---------------------|-----------------------|---------|----------------------------------------|
| None (Full System)  | Optimal               | Baseline| Complete system performance            |
| Remove Memory       | Reduced               | Moderate| Mura detection degrades                |
| Remove Foundation   | Reduced               | Largest | Overall performance suffers most       |
| Remove Efficient    | Slightly Reduced      | Small   | Complementarity loss affects marginal cases |
| Remove Stage One    | Unchanged             | None    | No accuracy impact, efficiency reduced |
| Remove Type Routing | Slightly Reduced      | Small   | Specialization loss                    |

All components contribute meaningfully to final system performance validating architectural design decisions. Foundation model removal causes largest drop confirming its central role across defect categories. Stage one screening provides pure efficiency gain without accuracy penalty.

**Model Efficiency Analysis**:

| Model Category      | Params | Performance | Efficiency | Deployment Suitability |
|---------------------|--------|-------------|------------|------------------------|
| Efficient (Compact)| Minimal| Moderate    | Highest    | Excellent for Stage 1  |
| Memory-based       | Small  | High        | High       | Good balance           |
| Knowledge Distill. | Medium | Moderate    | Moderate   | Moderate efficiency    |
| Normalizing Flow   | Medium | High        | Moderate   | Acceptable for Stage 2 |
| Foundation         | Large  | Highest     | Lower      | Best accuracy          |

Efficiency analysis reveals compact model as optimal Stage 1 screener balancing accuracy and model size. Memory-based methods offer excellent efficiency for specialized detection. Foundation models achieve highest absolute accuracy justifying larger parameter count for critical ensemble role.

#### 5.3. Development Phase Integration Analysis

**Operational Integration**: DefectVAD framework enables flexible development evaluation integration through modular architecture. Batch processing executes during convenient periods while model inference supports engineering evaluation workflow. Quality assurance integration presents low-confidence samples through human review interface displaying original image alongside anomaly heatmap, individual model scores, confidence metrics, normal reference comparison, and annotation tools for ground truth feedback.

Expert review decisions feed continuous improvement pipeline augmenting training dataset with challenging samples from actual development evaluation. Periodic retraining incorporates accumulated feedback improving model adaptation to evolving defect patterns. Model weights support hot-swapping enabling updates without system rebuild. Modular architecture permits individual ensemble component upgrades independently through standardized interfaces.

**Supporting Engineering Decisions**: Unlike automated production inspection targeting defect escape prevention through complete screening, development phase evaluation prioritizes: (1) reducing visual inspection time by pre-screening obvious cases, (2) eliminating inter-inspector variance through consistent automated analysis providing objective reference, (3) enabling engineers to focus on borderline cases requiring expert judgment, and (4) detecting novel defect patterns through unsupervised learning approach.

Threshold adjustment enables precision-recall trade-off optimization per evaluation priorities without retraining. Conservative thresholds increase recall minimizing missed defects at cost of higher review workload. Aggressive thresholds increase precision reducing unnecessary inspections at risk of missing subtle defects. Category-specific thresholds tunable through configuration files enabling rapid adaptation to changing evaluation standards.

**Color Non-uniformity Detection Impact**: Integration of color non-uniformity as primary defect category addresses critical gap in traditional luminance-focused inspection. XYZ colorspace measurements provide direct access to chromatic information enabling detection of subtle RGB channel deviations arising from differential OLED aging [12]. Foundation model's fine-grained feature extraction at higher resolution proves particularly effective for chromatic defect detection. This capability reduces reliance on subjective color perception judgment among inspectors, standardizing chromatic defect classification.

#### 5.4. Limitations and Future Directions

**Current System Limitations**: Single-panel analysis treats each display independently without exploiting spatial consistency across panel array or temporal consistency in development iterations. Static ensemble weights remain fixed despite potential variations in defect patterns across development phases. Defect taxonomy limited to predefined categories requires framework extension for emerging defect types encountered in new product development.

**Future Research Directions**: Model distillation research could transfer knowledge from ensemble to single lightweight network maintaining high accuracy while reducing parameter count for resource-constrained deployment. Active learning strategies could dynamically adjust ensemble weights based on expert review feedback optimizing performance for current development phase defect distribution. Multi-modal integration incorporating additional sensor modalities including thermal imaging detecting temperature anomalies and depth sensing identifying surface irregularities could enhance defect characterization beyond single-modality limitations. Transfer learning research extending framework to additional display technologies including LCD and microLED requires domain adaptation techniques accounting for different defect characteristics while leveraging shared knowledge from OLED evaluation experience.

---

### 6. Conclusion

This paper presented DefectVAD (Defect Vision Anomaly Detection), a comprehensive evaluation framework enabling systematic comparison of contemporary unsupervised anomaly detection models for OLED display quality assessment in development phase evaluation. Through modular architecture supporting unified dataset abstraction, model training coordination, and performance evaluation across diverse benchmarks [2] and custom industrial data, we revealed complementary architectural strengths across defect types with quantitative validation demonstrating moderate score correlations and low error overlap.

The proposed three-stage cascaded hybrid ensemble strategically integrates memory-based methods [3] providing texture expertise for Mura defects, foundation models [7] offering balanced cross-category performance including color non-uniformity detection, and efficient models [8] enabling fast screening through adaptive weighted fusion with category-specific threshold optimization. Operating within batch processing environment, preliminary evaluation demonstrates the ensemble achieves substantial improvement over best individual models with statistical significance.

Performance validation demonstrates consistent gains across all defect categories with universal improvements validating complementarity hypothesis. Model efficiency analysis reveals optimal configuration balancing accuracy with parameter efficiency.

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
        │   Model: Efficient (Compact) [8]      │
        │   Strategy: Dual threshold filtering  │
        └───────────────┬───────────────────────┘
                        ↓
        ┌───────────────┴────────────────┐
        ↓               ↓                ↓
   Low Score      Intermediate      High Score
   Certain Normal  Uncertain Cases  Certain Anomaly
    (Majority)      (Moderate)       (Small)
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
 Memory   Foundation Flow     Full    Adaptive
+Foundation +Efficient +Foundation Ensemble Strategy
     ↓        ↓         ↓         ↓         ↓
     └────────┴─────────┴─────────┴─────────┘
                        ↓
      ┌───────────────────────────────────────┐
      │   Stage 3: Confidence Assessment      │
      │   Metric: Distance from Threshold     │
      │   Human-in-the-Loop Review Queue      │
      └─────────────────┬─────────────────────┘
                        ↓
            ┌───────────┴────────────┐
            ↓                        ↓
     High Confidence          Low Confidence
        (Majority)              (Small Fraction)
            ↓                        ↓
     Automated Decision       Expert Review Queue
     Final Classification     Engineering Validation
            ↓                        ↓
            └────────┬───────────────┘
                     ↓
        ┌────────────────────────────────────────┐
        │       System Output                     │
        │   Preliminary Evaluation Shows:         │
        │   - Substantial improvement over single │
        │   - Consistent gains across categories  │
        │   - Optimal efficiency balance          │
        │   Supports Development Phase            │
        │   Quality Assessment Workflow           │
        └────────────────────────────────────────┘
```

**Figure Caption**: Three-stage cascaded architecture optimized for development phase quality assessment. Stage one employs fast screening with compact efficient model eliminating majority obvious normal cases. Stage two applies defect-type routing directing uncertain samples to specialized model combinations optimized for Mura, color non-uniformity, line, and pixel defects. Stage three implements confidence-based review queue flagging ambiguous cases for expert engineering validation, supporting human decision-making rather than replacing it. Preliminary evaluation demonstrates substantial improvements with optimal efficiency balance.
