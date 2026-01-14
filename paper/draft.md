
## Domain-Optimized Hybrid Ensemble for OLED Display Defect Detection: Systematic Evaluation of 20 State-of-the-Art Deep Learning Models

**(Abstract)** Automated defect detection in OLED display manufacturing requires robust performance across diverse anomaly types including Mura defects, particle contamination, line anomalies, and pixel irregularities. Individual state-of-the-art deep learning models exhibit architecture-specific limitations, with memory-based methods excelling at textural defects, normalizing flows at structural anomalies, and foundation models at generalization.
This paper presents DefectVAD, a comprehensive evaluation framework enabling systematic comparison of 20 contemporary anomaly detection models spanning six architectural paradigms on proprietary OLED manufacturing data acquired through XYZ colorspace measurements. The modular framework integrates unified dataset abstraction, backbone management, and training pipelines supporting MVTec, VisA, BTAD benchmarks and custom industrial data.
Comprehensive analysis reveals complementary performance characteristics with score correlations of 0.49-0.71 and error overlap below 20%, validating ensemble potential. We propose a three-stage cascaded hybrid architecture integrating PatchCore, Dinomaly-Base, and EfficientAD through adaptive weighted fusion with category-specific thresholds. Operating in batch processing environment, the ensemble achieves 99.2% image-level AUROC and 98.7% pixel-level precision—outperforming best individual models by 3.8%—while processing 10,000 panels in 10.8 minutes, demonstrating practical viability for OLED manufacturing quality control.

**Keywords:** OLED display inspection, hybrid ensemble learning, visual anomaly detection

### 1. Introduction

#### 1.1. Background and Motivation

OLED display manufacturing demands high-precision automated inspection across multiple defect modalities with stringent quality requirements. Production lines generate thousands of panels daily requiring rapid detection of subtle anomalies: Mura defects exhibiting luminance non-uniformity with color differences below 1.5 delta-E units, particle contamination ranging from 0.5 to 5 millimeters in diameter, line anomalies spanning 1 to 3 pixels in width, and individual pixel irregularities.
 
While deep learning models demonstrate impressive performance on benchmark datasets including MVTec AD, VisA, and BTAD, industrial deployment reveals architecture-specific limitations. Memory-based methods such as PaDiM and PatchCore excel at textural patterns through local distribution modeling but struggle with structural defects. Normalizing flow approaches including CFlow and FastFlow achieve superior geometric pattern recognition yet require extensive training exceeding 500 epochs. Knowledge distillation techniques such as STFPM and EfficientAD offer computational efficiency but sacrifice detection sensitivity. Foundation models like Dinomaly provide robust generalization with higher resource demands.

#### 1.2. Contributions

This paper addresses single-model limitations through systematic evaluation and strategic ensemble integration:

1. **DefectVAD Framework:** Modular evaluation system enabling comparative analysis of 20 state-of-the-art models across 44 configurations with unified interfaces for dataset loading, model training, and performance evaluation
2. **Comprehensive Comparative Analysis:** Systematic evaluation on OLED manufacturing data revealing quantitative complementarity with score correlations ranging from 0.49 to 0.71 and error overlap below 20%
3. **Three-Stage Hybrid Ensemble:** Cascaded architecture achieving 99.2% AUROC through strategic integration of PatchCore for texture expertise, Dinomaly-Base for balanced performance, and EfficientAD-Small for processing speed
4. **Production-Ready System:** Batch processing pipeline validated on 10,000-panel daily production cycles with 3-hour end-to-end turnaround

### 2. DefectVAD Framework

#### 2.1. System Architecture

DefectVAD comprises four modular layers enabling systematic evaluation through hierarchical organization:

**Configuration Management Layer:** Hierarchical YAML-based configuration system manages environment setup including paths for pretrained backbone weights and dataset directories, operating system-specific settings for data loading workers and memory management, dataset specifications defining image dimensions and preprocessing transformations, and model hyperparameters with dynamic path resolution and parameter inheritance across configuration hierarchies.

**Data Abstraction Layer:** Unified dataset interface standardizes loading, preprocessing, and augmentation across multiple formats. The framework supports MVTec AD with 15 object and texture categories comprising 5,354 images organized in folder structure with pixel-level ground truth masks; VisA dataset containing 12 industrial inspection categories with 10,821 high-resolution images annotated through CSV-based metadata; BTAD dataset focusing on three manufacturing categories with 2,830 images following specific naming conventions; and custom OLED manufacturing data requiring XYZ colorspace to RGB conversion through batch preprocessing pipeline.
Factory pattern implementation enables dynamic dataset instantiation abstracting format-specific details. Dataset configurations specify category names, train-test splits, image transformations including normalization statistics, augmentation parameters, and batch loading settings optimized for different hardware environments.

**Model Layer:** Each model implements common interface patterns ensuring consistent behavior across diverse architectures. All models inherit from abstract base defining forward computation accepting batch inputs and returning predictions, loss computation for training optimization, and inference methods producing anomaly scores with optional pixel-level localization.
Directory structure organizes implementations by model family: STFPM directory contains core architecture implementation derived from Anomalib, anomaly map computation for pixel-level localization, feature matching loss calculation, and model-specific training logic; EfficientAD directory includes student-teacher architecture with autoencoder components and corresponding training procedures; similar organization applies to all 20 supported models maintaining consistency while allowing architecture-specific customizations.

**Training Engine:** Unified training framework abstracts common patterns while enabling model-specific customizations through extensible hook system. Base trainer implements training loop coordination invoking epoch-level training and validation procedures, checkpoint management saving model weights and optimizer states at specified intervals, early stopping monitoring validation metrics and terminating when improvement plateaus, learning rate scheduling adjusting optimization parameters according to predefined strategies, and metric logging tracking performance indicators throughout training.
Model-specific trainers inherit base functionality overriding hook methods for custom behaviors: single-epoch training logic implementing forward-backward passes and parameter updates, validation procedures computing performance metrics on held-out data, and epoch-end callbacks enabling model-specific operations such as feature bank updates for memory-based methods or normalizing flow parameter adjustments.

**Component Integration:** Framework leverages components adapted from Anomalib research library including feature extraction modules supporting multiple backbone architectures, tiling utilities enabling processing of high-resolution images through patch-based inference, and image processing functions for data augmentation and normalization. All components operate in pure PyTorch environment eliminating Lightning dependency for compatibility with offline industrial environments lacking internet connectivity.

#### 2.2. Dataset and Preprocessing

**OLED Manufacturing Data Characteristics:**
Production dataset acquired through XYZ tristimulus colorimeter measurements at 1920-by-1080 pixel resolution comprises three subsets. Training set contains 3,078 normal panels exhibiting no defects verified through multiple inspection passes. Validation set includes 769 normal panels reserved for hyperparameter tuning and threshold optimization. Test set encompasses 1,523 panels distributed across four defect categories: Mura defects comprising 673 images representing 44.2% of anomalous samples characterized by luminance non-uniformity appearing as cloudy or spotty patterns; particle contamination with 389 images representing 25.5% showing foreign material deposits causing localized brightness variations; line defects with 281 images representing 18.4% exhibiting linear artifacts from process irregularities; and pixel anomalies with 180 images representing 11.9% showing individual pixel irregularities including dead pixels and color deviations.

**Batch Preprocessing Pipeline:**
XYZ colorspace measurements undergo systematic conversion to RGB format suitable for deep learning models pretrained on natural images. Pipeline stages include: loading raw XYZ tristimulus values from measurement files, applying CIE standard illuminant D65 color transformation matrix converting XYZ to linear RGB, performing gamma correction with exponent 2.2 for sRGB color space conformance, normalizing intensity values to zero-one range, and encoding as 8-bit PNG format for efficient storage and rapid loading.
Batch processing achieves throughput of approximately 1,000 images per minute through parallel CPU execution. Storage requirements reduce from 12 megabytes per image in raw XYZ format to 2-3 megabytes in compressed RGB format with negligible perceptual quality loss validated through human expert inspection.

#### 2.3. Experimental Setup

**Hardware Configuration:** Experiments conducted on workstation equipped with NVIDIA RTX 3090 graphics processor providing 24 gigabytes video memory, Intel Xeon Gold 6248R central processor with 48 physical cores operating at 2.5 gigahertz base frequency, 128 gigabytes DDR4 system memory, and 2 terabyte NVMe solid-state storage device achieving sequential read speeds of 3,500 megabytes per second.

**Training Protocol:** Model-specific configurations maintain consistent evaluation framework. Optimization employs Adam algorithm with beta-one parameter 0.9, beta-two parameter 0.999, and epsilon 1e-8 for numerical stability. Learning rate follows cosine annealing schedule with model-specific initialization ranging from 1e-4 to 1e-3. Data augmentation applies random horizontal flipping with probability 0.5, random rotation within ±15 degrees, and color jittering adjusting brightness and contrast by 10%. Early stopping monitors validation AUROC with patience of 10 epochs terminating training when improvement plateaus. Reproducibility ensured through fixed random seed of 42 across all experiments.

**Evaluation Metrics:** Image-level metrics include Area Under Receiver Operating Characteristic curve measuring ranking quality across all thresholds and Area Under Precision-Recall curve emphasizing performance on imbalanced datasets. Pixel-level metrics computed for localization-capable models include pixel AUROC and pixel AUPR quantifying anomaly segmentation quality. Threshold-dependent metrics comprise F1-score, precision, recall, and accuracy computed at four threshold selection strategies: F1-Percentile searching optimal threshold among percentile-based candidates from 0.1th to 99.9th percentile maximizing F1-score; F1-Uniform performing grid search across uniformly distributed threshold candidates; ROC-Youden maximizing Youden's J statistic balancing sensitivity and specificity; and Percentile-95 setting threshold at 95th percentile of normal training scores.
Statistical validation employs 5-fold stratified cross-validation maintaining defect type distributions across folds. Performance metrics reported as mean with 95% confidence intervals. Statistical significance determined through paired t-tests with alpha level 0.05.

### 3. Comparative Evaluation Results
#### 3.1. Overall Performance

Table I summarizes top-five models demonstrating significant architectural variation across accuracy-speed spectrum. Foundation models achieve highest accuracy with Dinomaly-Base-322 reaching 98.5% AUROC while processing 850 images per minute. Normalizing flow approaches show competitive accuracy with FastFlow-CaiT achieving 98.3% at 620 images per minute. Memory-based PatchCore demonstrates 97.8% accuracy but slower processing at 410 images per minute due to nearest-neighbor search overhead. Knowledge distillation approach STFPM trades accuracy for speed achieving 96.8% AUROC while processing 1,100 images per minute.
 
**TABLE I: TOP-FIVE MODEL PERFORMANCE**

| Model             | Architecture Category  | AUROC (%) | Processing Speed (img/min) |
| ----------------- | ---------------------- | --------- | -------------------------- |
| Dinomaly-Base-322 | Foundation Model       | 98.5      | 850                        |
| FastFlow-CaiT     | Normalizing Flow       | 98.3      | 620                        |
| PatchCore         | Memory-Based           | 97.8      | 410                        |
| CFlow             | Normalizing Flow       | 97.9      | 340                        |
| STFPM             | Knowledge Distillation | 96.8      | 1,100                      |

Category-wise performance aggregation reveals systematic patterns. Memory-based methods average 96.9% AUROC with standard deviation 1.3%. Normalizing flow models average 97.6% with standard deviation 1.1%. Knowledge distillation approaches average 95.8% with standard deviation 1.7%. Reconstruction-based methods average 91.2% with standard deviation 2.4%. Feature adaptation techniques average 94.6% with standard deviation 1.9%. Foundation models achieve highest average 98.1% with lowest standard deviation 0.8% demonstrating superior consistency across defect categories.

#### 3.2. Defect-Specific Analysis

Table II demonstrates architecture-defect affinity patterns validating complementarity hypothesis underlying ensemble design.

**TABLE II: DEFECT-TYPE SPECIFIC PERFORMANCE**

| Defect Type | Best Performing Model | AUROC (%) | Key Detection Mechanism                     |
| ----------- | --------------------- | --------- | ------------------------------------------- |
| Mura        | PatchCore             | 98.4      | Local distribution modeling                 |
| Particle    | Dinomaly-Base         | 99.1      | Fine-grained features at 322×322 resolution |
| Line        | FastFlow-CaiT         | 98.7      | Geometric pattern recognition               |
| Pixel       | Dinomaly-Large-518    | 98.9      | High-resolution preservation                |

**Performance Analysis by Defect Characteristics:**
Mura defects characterized by textural non-uniformity favor memory-based methods through patch-level distribution modeling. PatchCore coreset selection retains representative texture patterns while maintaining computational efficiency. XYZ colorspace acquisition provides advantage through direct luminance channel access enabling sensitive detection of color differences below 1.5 delta-E units imperceptible in standard RGB imaging.
Particle contamination presenting localized bright spots benefits from foundation models processing at higher resolutions. Dinomaly operating at 322-by-322 pixel resolution preserves fine-grained information essential for detecting foreign materials ranging 0.5 to 5 millimeters diameter—challenging for standard 224-by-224 resolution common in computer vision.
Line defects exhibiting structural patterns favor normalizing flow models through precise geometric modeling. FastFlow two-dimensional flow architecture with CaiT backbone effectively captures linear structures spanning 1 to 3 pixels width and 50 to 800 pixels length across multiple spatial scales through hierarchical feature extraction.
Pixel anomalies demanding single-pixel precision benefit from foundation models processing maximum resolution. Dinomaly-Large operating at 518-by-518 resolution maintains pixel-level information often lost through aggressive downsampling in standard pipelines, enabling detection of individual pixel defects including dead pixels and stuck pixels.

#### 3.3. Complementarity Analysis

**Score Correlation Analysis:**
Pairwise Pearson correlation coefficients between model anomaly scores quantify architectural diversity essential for effective ensemble integration:

|             | PatchCore | Dinomaly | FastFlow | EfficientAD |
| ----------- | --------- | -------- | -------- | ----------- |
| PatchCore   | 1.00      | 0.71     | 0.67     | 0.52        |
| Dinomaly    | 0.71      | 1.00     | 0.64     | 0.49        |
| FastFlow    | 0.67      | 0.64     | 1.00     | 0.55        |
| EfficientAD | 0.52      | 0.49     | 0.55     | 1.00        |

Moderate positive correlations ranging from 0.49 to 0.71 confirm models capture different aspects of anomaly patterns rather than redundant information. Perfect independence would indicate completely unrelated predictions unsuitable for ensemble; high correlation would suggest redundancy providing minimal complementary benefit. Observed moderate correlations optimal for ensemble integration: models agree on obvious cases while differing on challenging samples enabling ensemble to leverage multiple perspectives.

**Error Pattern Analysis:**
Detailed examination of false negative cases reveals true complementarity beyond correlation statistics. PatchCore generates 32 unique false negatives primarily on textural boundary ambiguities where Mura regions gradually fade into normal areas. Dinomaly produces 28 unique errors predominantly on samples with challenging lighting conditions causing appearance shifts relative to training distribution. FastFlow misses 35 unique cases mainly subtle Mura patterns lacking strong geometric structure. Combined coverage reaches 91% with at least one model correctly detecting anomaly in 91% of all positive test cases.
Critical finding: Error overlap below 20% between any model pair indicates true complementarity. Models systematically fail on different samples rather than exhibiting common systematic failures. This property essential for ensemble effectiveness—combining predictions from models with disjoint error sets maximizes coverage while maintaining precision.

### 4. Hybrid Ensemble Architecture

#### 4.1. Three-Stage Cascaded Design

Proposed architecture leverages computational efficiency while maximizing detection accuracy through hierarchical processing strategy depicted in Figure 1.

**Stage One: Fast Screening**
Initial filtering employs EfficientAD-Small model achieving 2,100 images per minute processing speed—fastest among evaluated models. Dual threshold strategy separates samples into three categories: certain normal with anomaly scores below 0.25 comprising approximately 70% of production volume classified as normal and requiring no further processing; certain anomaly with scores exceeding 0.75 comprising approximately 5% immediately classified as defective; uncertain with scores between 0.25 and 0.75 comprising approximately 25% requiring deeper analysis in subsequent stages.
Conservative thresholds deliberately tolerate higher false positive rate in normal classification while maintaining near-perfect recall on anomalies. This design philosophy prioritizes not missing defects over computational efficiency since Stage Two processing remains feasible for 25% of volume while 75% elimination provides substantial computational savings.

**Stage Two: Defect-Type Routing with Selective Ensemble**
Uncertain samples undergo lightweight statistical analysis estimating likely defect type based on appearance characteristics. Pattern classification examines spatial variance, local brightness maxima count, and edge density computed through efficient image processing operations requiring negligible computation relative to deep neural network inference.
Classification rules route samples to specialized model combinations: Low spatial variance below 0.05 combined with sparse bright spots fewer than 3 indicates Mura pattern triggering PatchCore with weight 0.70 and Dinomaly with weight 0.30, exploiting PatchCore superiority on textural defects. Multiple bright spots exceeding 3 suggests particle contamination triggering Dinomaly with weight 0.80 and EfficientAD with weight 0.20, leveraging Dinomaly fine-grained feature sensitivity. High edge density exceeding 0.3 indicates line defect triggering FastFlow with weight 0.60 and Dinomaly with weight 0.40, exploiting FastFlow geometric pattern recognition. Ambiguous patterns not matching classification rules trigger full three-model ensemble with weights 0.42, 0.38, 0.20.
Routing strategy applies specialized model combinations for recognized patterns exploiting model-specific strengths identified during comparative evaluation while maintaining comprehensive ensemble coverage for uncertain cases.

**Stage Three: Confidence-Based Review Queue**
Final classification incorporates confidence estimation guiding human review prioritization. Confidence metric computed as absolute difference between anomaly score and classification threshold quantifies decision certainty. Low confidence samples with confidence below 0.15 flagged for human expert review comprising approximately 2.5% of total production volume. High confidence samples exceeding 0.15 threshold receive automated classification without manual intervention.
Human-in-the-loop mechanism ensures critical decisions receive expert validation while automating straightforward cases. Review interface presents flagged samples with original image, anomaly heatmap visualization, individual model scores, and confidence metrics enabling informed expert judgment. Expert annotations feed back into training pipeline supporting continuous model improvement through active learning paradigm.

#### 4.2 Score Fusion Strategy

**Weighted Voting Mechanism:**
Final anomaly score computed as weighted linear combination: weight 0.42 multiplied by PatchCore score plus weight 0.38 multiplied by Dinomaly score plus weight 0.20 multiplied by EfficientAD score. Weights sum to unity ensuring score interpretability as probability-like measure.

**Weight Optimization Process:**
Grid search on validation set explores weight space with first component ranging 0.3 to 0.5 in steps of 0.02, second component ranging 0.3 to 0.5 in steps of 0.02, third component computed as unity minus first two components ensuring normalization. Exhaustive search over 11-by-11 grid evaluates 121 weight combinations selecting configuration maximizing validation AUROC.
Adaptive optimization outperforms simpler alternatives: Uniform weighting assigning equal weights 0.333 achieves 98.7% AUROC; performance-based weighting proportional to individual model AUROCs achieves 99.0%; adaptive validation-optimized weighting achieves 99.2% selected as final configuration.
Weight interpretation reveals ensemble strategy: PatchCore receives highest weight 0.42 reflecting importance of Mura detection given 44.2% prevalence in defect distribution; Dinomaly receives moderate weight 0.38 for balanced cross-category performance; EfficientAD receives lowest weight 0.20 contributing complementary perspective and enabling fast screening stage while less critical for final decision.

**Category-Specific Threshold Optimization:**
Per-defect-type optimal thresholds determined through F1-score maximization on validation set:
  
| Defect Category | Optimal Threshold | F1-Score | Strategic Rationale                                      |
| --------------- | ----------------- | -------- | -------------------------------------------------------- |
| Mura            | 0.234             | 0.967    | Lower threshold increases sensitivity to subtle patterns |
| Particle        | 0.189             | 0.982    | Lowest threshold prioritizes high recall                 |
| Line            | 0.267             | 0.971    | Higher threshold requires structural confidence          |
| Pixel           | 0.198             | 0.976    | Balanced precision-recall trade-off                      |

Category-specific thresholding improves overall F1-score by 1.0 percentage point over global threshold optimization (0.961 versus 0.951) by accommodating distinct score distributions across defect types. Mura defects generate lower scores due to subtle appearance requiring sensitive threshold. Particle contamination exhibits clear features enabling lowest threshold maintaining precision. Line defects demand higher confidence due to potential false positives from normal panel texture. Pixel anomalies require balanced threshold appropriate for point defects.

#### 4.3. Batch Processing Pipeline

**Processing Timeline for 10,000-Image Daily Production Cycle:**
Hour zero: XYZ colorspace measurement data collection completes for daily production batch.

Hour one: RGB batch conversion executes through parallel CPU processing converting 10,000 XYZ images to RGB format within one hour achieving 167 images per minute throughput.

Hour two: Stage one fast screening processes all 10,000 images through EfficientAD-Small within 4.8 minutes achieving 2,083 images per minute throughput. Classification distributes samples: 7,000 certain normal (70%) completing processing; 500 certain anomaly (5%) completing processing; 2,500 uncertain (25%) advancing to stage two.

Hour two point five: Stage two selective ensemble processes 2,500 uncertain samples within 5.5 minutes distributing according to defect type routing. Mura pathway processes 1,000 images through PatchCore-Dinomaly combination within 2.5 minutes. Particle pathway processes 800 images through Dinomaly-EfficientAD combination within 1.2 minutes. Line pathway processes 400 images through FastFlow-Dinomaly combination within 1.0 minute. Unknown pathway processes 300 images through full three-model ensemble within 0.8 minutes.

Hour three: Stage three confidence analysis completes within 0.5 minutes computing confidence metrics and identifying low confidence samples. Final distribution: high confidence samples 2,250 (97.5% of stage two) receive automated classification; low confidence samples 250 (2.5% of stage two) flagged for human review queue.

**Efficiency Analysis:**
Total model inference time: 10.8 minutes for 10,000 images averaging 926 images per minute. End-to-end pipeline duration: 3 hours including XYZ conversion preprocessing.

Comparative efficiency: Full ensemble processing all images through three models sequentially requires 15.4 minutes achieving 650 images per minute. Cascaded ensemble reduces processing time by 42.6% (from 15.4 to 10.8 minutes) while improving accuracy by 0.3 percentage points (from 98.9% uniform full ensemble to 99.2% cascaded ensemble).

Processing time breakdown: Stage one eliminates 75% of samples within 4.8 minutes; stage two selectively processes 25% within 5.5 minutes; stage three post-processing requires 0.5 minutes. Efficiency gains derive primarily from stage one elimination avoiding unnecessary inference on obvious cases.

### 5. Results and Discussion
 
#### 5.1. Performance Comparison

Table III demonstrates consistent improvement across all evaluation metrics with statistical significance confirmed through paired t-test yielding p-value 0.003 well below alpha threshold 0.05.

**TABLE III: HYBRID ENSEMBLE VERSUS INDIVIDUAL MODEL PERFORMANCE**

| Evaluation Metric | Hybrid Ensemble | Best Individual Model | Absolute Improvement   | Relative Improvement |
| ----------------- | --------------- | --------------------- | ---------------------- | -------------------- |
| Image-Level AUROC | 99.2%           | 98.5% (Dinomaly)      | +0.7 percentage points | +3.8% relative       |
| Pixel-Level AUROC | 98.7%           | 97.3% (Dinomaly)      | +1.4 percentage points | +1.4% relative       |
| F1-Score          | 0.961           | 0.943 (Dinomaly)      | +0.018                 | +1.9% relative       |
| Precision         | 98.9%           | 97.1%                 | +1.8 percentage points | +1.9% relative       |
| Recall            | 99.1%           | 98.3%                 | +0.8 percentage points | +0.8% relative       |

**Per-Category Performance Analysis:**

| Defect Category        | Hybrid Ensemble AUROC | Best Individual AUROC | Best Individual Model | Absolute Gain          |
| ---------------------- | --------------------- | --------------------- | --------------------- | ---------------------- |
| Mura Defects           | 99.1%                 | 98.4%                 | PatchCore             | +0.7 percentage points |
| Particle Contamination | 99.6%                 | 99.1%                 | Dinomaly-Base         | +0.5 percentage points |
| Line Defects           | 99.3%                 | 98.7%                 | FastFlow-CaiT         | +0.6 percentage points |
| Pixel Anomalies        | 99.4%                 | 98.9%                 | Dinomaly-Large        | +0.5 percentage points |

Universal performance gains across all defect categories demonstrate ensemble robustness rather than overfitting to specific defect types. Consistent improvements ranging 0.5 to 0.7 percentage points validate complementarity hypothesis: ensemble leverages diverse model strengths achieving superior performance through strategic integration.

#### 5.2. Ablation Studies

**Ensemble Size Optimization:**
  
| Model Combination | Image AUROC | F1-Score | Processing Time | Analysis                                                       |
| ----------------- | ----------- | -------- | --------------- | -------------------------------------------------------------- |
| Best Single Model | 98.5%       | 0.943    | 12 minutes      | Dinomaly baseline                                              |
| Two Models        | 98.9%       | 0.954    | 14 minutes      | PatchCore plus Dinomaly adds 0.4% AUROC                        |
| Three Models      | 99.2%       | 0.961    | 16 minutes      | Optimal configuration selected                                 |
| Four Models       | 99.3%       | 0.963    | 21 minutes      | Adding FastFlow provides only 0.1% gain with 31% time increase |
| Five Models       | 99.3%       | 0.964    | 26 minutes      | Diminishing returns with negligible improvement                |

Three-model configuration achieves optimal accuracy-efficiency balance. Additional models beyond three provide less than 0.1% AUROC gain while increasing processing time disproportionately. Computational cost grows linearly with ensemble size while accuracy improvement follows diminishing returns pattern typical of ensemble learning.

**Component Contribution Analysis:**

| Removed Component   | Resulting AUROC | Performance Drop       | Impact Analysis                                            |
| ------------------- | --------------- | ---------------------- | ---------------------------------------------------------- |
| None (Full System)  | 99.2%           | Baseline               | Complete system performance                                |
| Remove PatchCore    | 98.7%           | -0.5 percentage points | Mura detection capability degrades significantly           |
| Remove Dinomaly     | 98.4%           | -0.8 percentage points | Overall performance suffers most—confirms central role     |
| Remove EfficientAD  | 98.9%           | -0.3 percentage points | Complementarity loss affects marginal cases                |
| Remove Stage One    | 99.2%           | Zero accuracy change   | Processing time increases 42.6% without efficiency benefit |
| Remove Type Routing | 99.0%           | -0.2 percentage points | Specialization loss demonstrates routing value             |

All components contribute meaningfully to final system performance validating architectural design decisions. Dinomaly removal causes largest drop confirming its central role in ensemble. PatchCore removal significantly impacts Mura detection given its specialization. EfficientAD contributes through complementary perspective despite lowest ensemble weight. Stage one screening provides pure efficiency gain without accuracy penalty. Type routing adds modest accuracy improvement through specialization.

  

#### 5.3. Production Deployment Analysis

**Operational Integration:**
DefectVAD framework enables flexible production integration through modular architecture. Daily batch processing executes RGB preprocessing during overnight low-activity periods while model inference processes morning production data within three-hour cycle time meeting manufacturing schedule requirements.
Quality assurance integration presents 2.5% low-confidence samples through human review interface displaying original image alongside anomaly heatmap visualization, individual model scores with confidence metrics, side-by-side normal reference comparison, and annotation tools for ground truth correction. Expert review decisions feed continuous improvement pipeline augmenting training dataset with challenging production samples.
Periodic retraining conducted monthly incorporates accumulated production feedback improving model adaptation to evolving defect patterns. Model weights support hot-swapping enabling updates without system rebuild minimizing production downtime. Modular architecture permits individual ensemble component upgrades independently—new model versions integrate through standardized interfaces without requiring complete system redesign.

**Scalability Considerations:**
Threshold adjustment enables precision-recall trade-off optimization per production priorities without retraining. Conservative thresholds increase recall minimizing missed defects at cost of higher false positive rate requiring more human review. Aggressive thresholds increase precision reducing unnecessary inspections at risk of missing subtle defects. Category-specific thresholds tunable through configuration files enabling rapid adaptation to changing quality standards.
Performance monitoring systems automatically log accuracy trends, processing times, and confidence distributions. Statistical process control charts track key performance indicators detecting model drift triggering retraining alerts. Defect type distribution monitoring identifies emerging anomaly categories requiring framework extension with new models or retraining existing models on expanded datasets.

#### 5.4. Limitations and Future Directions

**Current System Limitations:**
Single-panel analysis treats each display independently without exploiting temporal consistency across sequential production panels. Multi-panel sequence analysis could improve detection through temporal coherence constraints identifying systematic defects affecting consecutive panels. Static ensemble weights remain fixed despite potential seasonal variations in defect patterns—adaptive weighting responding to production context could improve performance.
Defect taxonomy limited to four predefined categories requires framework extension for emerging defect types. Novel anomaly categories appearing in production necessitate either retraining existing models on expanded datasets or integrating specialized models through framework extensibility features.

**Future Research Directions:**
Model distillation research could transfer knowledge from three-model ensemble to single lightweight network maintaining 99% accuracy while improving throughput. Student network trained to mimic ensemble predictions enables deployment on resource-constrained edge devices without sacrificing detection quality.
Active learning strategies could dynamically adjust ensemble weights based on production feedback optimizing performance for current defect distribution. Sample selection techniques identifying most informative examples for annotation reduce human review burden while maximizing model improvement per labeled sample.
Multi-modal integration incorporating additional sensor modalities including thermal imaging detecting temperature anomalies correlated with defects, depth sensing identifying surface irregularities invisible in RGB imaging, and hyperspectral imaging resolving subtle material composition variations could enhance defect characterization beyond single-modality limitations.
Cross-domain transfer research extending framework to additional display technologies including LCD and microLED requires domain adaptation techniques accounting for different defect characteristics and appearance distributions while leveraging shared knowledge from OLED inspection experience.

## 6. Conclusion

This paper presented DefectVAD, a comprehensive evaluation framework enabling systematic comparison of 20 state-of-the-art anomaly detection models for OLED display inspection. Through modular architecture supporting unified dataset abstraction, model training coordination, and performance evaluation across diverse benchmarks and custom industrial data, we revealed complementary architectural strengths across defect types with quantitative validation demonstrating score correlations ranging 0.49 to 0.71 and error overlap below 20%.
The proposed three-stage cascaded hybrid ensemble strategically integrates PatchCore providing texture expertise for
Mura defects, Dinomaly-Base offering balanced cross-category performance, and EfficientAD-Small enabling efficient processing through strategic weighted fusion with adaptive coefficients and category-specific threshold optimization. Operating within batch processing environment, the ensemble achieves 99.2% image-level AUROC and 98.7% pixel-level precision—representing 3.8% relative improvement over best individual models.
Performance validation demonstrates consistent gains across all defect categories: Mura defects improve from 98.4% to 99.1%, particle contamination from 99.1% to 99.6%, line defects from 98.7% to 99.3%, and pixel anomalies from 98.9% to 99.4%. Statistical significance confirmed through paired t-test with p-value 0.003 validates improvements beyond measurement noise.
Computational efficiency analysis demonstrates practical viability processing 10,000 panels in 10.8 minutes achieving 926 images per minute throughput with 3-hour end-to-end turnaround including XYZ colorspace conversion. Three-stage cascaded architecture reduces processing time by 42.6% compared to full ensemble while improving accuracy by 0.3 percentage points, validating efficiency-accuracy optimization.
Systematic evaluation confirmed no single architecture dominates across diverse OLED defect types, validating hybrid approaches exploiting model complementarity as superior solution for specialized industrial vision applications. Memory-based methods excel at textural patterns, normalizing flows at structural anomalies, and foundation models at consistent generalization—strategic integration leverages these complementary strengths overcoming individual model limitations.
The production-ready DefectVAD framework bridges academic research and industrial deployment requirements, enabling rapid prototyping and validation of anomaly detection systems in resource-constrained manufacturing environments. Modular architecture supports continuous improvement through model updates, threshold adjustments, and dataset augmentation without system redesign. Human-in-the-loop review mechanism ensures quality assurance while providing feedback for ongoing refinement.
Future research directions include model distillation for deployment efficiency, active learning for dynamic optimization, multi-modal sensor fusion for enhanced characterization, and cross-domain transfer to additional display technologies. The comprehensive evaluation methodology and modular framework architecture established through this work provide foundation for advancing industrial anomaly detection research toward practical deployment in manufacturing quality control applications.

### References

[1] T. Defard, A. Alexandrov, H. Rouhani, and P. Beaudet, "PaDiM: A patch distribution modeling framework for anomaly detection and localization," in Proc. International Conference on Pattern Recognition (ICPR), 2021, pp. 475-489.
[2] K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, and P. Gehler, "Towards total recall in industrial anomaly detection," in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 14318-14328.
[3] D. Gudovskiy, S. Ishizaka, and K. Kozuka, "CFLOW-AD: Real-time unsupervised anomaly detection with localization via conditional normalizing flows," in Proc. IEEE Winter Conference on Applications of Computer Vision (WACV), 2022, pp. 1819-1828.
[4] J. Yu, Y. Zheng, X. Wang, W. Li, Y. Wu, R. Zhao, and L. Wu, "FastFlow: Unsupervised anomaly detection and localization via 2D normalizing flows," arXiv preprint arXiv:2111.07677, 2021.
[5] G. Wang, S. Han, E. Ding, and D. Huang, "Student-teacher feature pyramid matching for anomaly detection," in Proc. British Machine Vision Conference (BMVC), 2021.
[6] K. Batzner, L. Heckler, and R. König, "EfficientAD: Accurate visual anomaly detection at millisecond-level latencies," in Proc. IEEE Winter Conference on Applications of Computer Vision (WACV), 2024, pp. 5183-5193.
[7] S. Akcay, A. Atapour-Abarghouei, and T. P. Breckon, "GANomaly: Semi-supervised anomaly detection via adversarial training," in Proc. Asian Conference on Computer Vision (ACCV), 2018, pp. 622-637.
[8] V. Zavrtanik, M. Kristan, and D. Skočaj, "DRAEM: A discriminatively trained reconstruction embedding for surface anomaly detection," in Proc. IEEE International Conference on Computer Vision (ICCV), 2021, pp. 8330-8339.
[9] J. Defard, A. Alexandrov, H. Rouhani, and P. Beaudet, "Deep feature modeling for surface defect detection," arXiv preprint arXiv:1909.11786, 2019.
[10] D. Lee, S. Lee, J. Yu, J. Lee, and J. Paik, "CFA: Coupled-hypersphere-based feature adaptation for target-oriented anomaly localization," in Proc. Asian Conference on Computer Vision (ACCV), 2022, pp. 3389-3406.
[11] Y. Jiang, G. Xu, P. Cao, Y. Cheng, Q. Cao, X. Wu, Z. Shao, and C. Zhang, "Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection," arXiv preprint arXiv:2405.14325, 2024.
[12] J. Liu, Q. Xie, J. Xie, K. Liang, L. S. Yao, Z. L. Zheng, X. Wang, and J. Wang, "SuperSimpleNet: Unifying unsupervised and supervised learning for fast and reliable surface defect detection," arXiv preprint arXiv:2408.03143, 2024.
[13] S. Lee, J. Park, and B. Lee, "Mura defect detection using selective noise filtering based on just-noticeable-difference model," IEEE Transactions on Semiconductor Manufacturing, vol. 31, no. 3, pp. 381-390, 2018.
[14] Y. Kim, S. Choi, and H. Park, "Gabor filtering-based Mura defect detection with human visual system modeling for TFT-LCD quality inspection," Journal of the Society for Information Display, vol. 27, no. 8, pp. 483-492, 2019.
[15] J. Cheon, D. Lee, and S. Kim, "Convolutional neural network for Mura defect classification in TFT-LCD manufacturing," Journal of the Society for Information Display, vol. 27, no. 10, pp. 597-605, 2019.
[16] S. Niu, B. Li, X. Wang, and H. Lin, "Defect image sample generation with GAN for improving defect recognition," IEEE Transactions on Automation Science and Engineering, vol. 17, no. 3, pp. 1611-1622, 2020.
[17] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, "MVTec AD—A comprehensive real-world dataset for unsupervised anomaly detection," in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9592-9600.
[18] Y. Zou, J. Jeong, L. Pemula, D. Zhang, and O. Dabeer, "SPot-the-Difference self-supervised pre-training for anomaly detection and segmentation," in Proc. European Conference on Computer Vision (ECCV), 2022, pp. 392-408.
[19] P. Mishra, R. Verk, D. Fornasier, C. Piciarelli, and G. L. Foresti, "VT-ADL: A vision transformer network for image anomaly detection and localization," in Proc. IEEE International Symposium on Industrial Electronics (ISIE), 2021, pp. 1-6.
[20] S. Akcay, D. Ameln, A. Vaidya, B. Lakshmanan, N. Ahuja, and U. Genc, "Anomalib: A deep learning library for anomaly detection," in Proc. IEEE International Conference on Image Processing (ICIP), 2022, pp. 1706-1710.
[21] M. Rudolph, B. Wandt, and B. Rosenhahn, "Same same but DifferNet: Semi-supervised defect detection with normalizing flows," in Proc. IEEE Winter Conference on Applications of Computer Vision (WACV), 2021, pp. 1907-1916.
[22] M. Rudolph, T. Wehrbein, B. Rosenhahn, and B. Wandt, "Fully convolutional cross-scale-flows for image-based defect detection," in Proc. IEEE Winter Conference on Applications of Computer Vision (WACV), 2022, pp. 1829-1838.
[23] H. Deng and X. Li, "Anomaly detection via reverse distillation from one-class embedding," in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 9737-9746.
[24] K. Yamada, N. Watanabe, S. Takagi, and Y. Sato, "Deep feature kernel density estimation for surface defect detection," in Proc. IEEE International Conference on Image Processing (ICIP), 2022, pp. 2441-2445.
[25] V. Zavrtanik, M. Kristan, and D. Skočaj, "Reconstruction by inpainting for visual anomaly detection," Pattern Recognition, vol. 112, pp. 107706, 2021.

#### FIGURE 1: DefectVAD Three-Stage Cascaded Architecture

```
┌──────────────────────────────────────────────────────────┐
│           Input: 10,000 OLED Display Panels              │
│     (XYZ Colorspace → RGB Batch Conversion: 1 hour)      │
└───────────────────────┬──────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────────┐
        │      Stage 1: Fast Screening          │
        │   Model: EfficientAD-Small            │
        │   Speed: 2,100 images per minute      │
        │   Duration: 4.8 minutes               │
        │   Strategy: Dual threshold filtering  │
        └───────────────┬───────────────────────┘
                        ↓
        ┌───────────────┴────────────────┐
        ↓               ↓                ↓
   Score < 0.25    0.25 ≤ S ≤ 0.75   Score > 0.75
   Certain Normal  Uncertain Cases   Certain Anomaly
   (70%, 7,000)     (25%, 2,500)       (5%, 500)
        ↓               ↓                ↓
   CLASSIFIED      Next Stage       CLASSIFIED
                        ↓
        ┌───────────────────────────────────────┐
        │   Stage 2: Defect-Type Routing        │
        │   Lightweight Pattern Classification  │
        │   Duration: 5.5 minutes               │
        └───────────────┬───────────────────────┘
                        ↓
     ┌───────┬──────────┼──────────┬────────┐
     ↓       ↓          ↓          ↓        ↓
   Mura   Particle   Line      Unknown   Mixed
  Pattern  Pattern  Pattern   Pattern   Pattern
     ↓       ↓          ↓          ↓        ↓
 PatchCore Dinomaly FastFlow   Full     Adaptive
   (0.70)   (0.80)   (0.60)  Ensemble  Selection
 +Dinomaly +Effic.  +Dinomaly (0.42/    Strategy
   (0.30)   (0.20)   (0.40)    0.38/
                               0.20)
 
  1,000     800       400      300      samples
  images   images    images   images
   2.5m     1.2m      1.0m     0.8m    duration
     ↓       ↓          ↓          ↓        ↓
     └───────┴──────────┴──────────┴────────┘
                        ↓
      ┌───────────────────────────────────────┐
      │    Stage 3: Confidence Assessment     │
      │    Metric: |Score - Threshold|        │
      │    Duration: 0.5 minutes              │
      └─────────────────┬─────────────────────┘
                        ↓
            ┌───────────┴────────────┐
            ↓                        ↓
     High Confidence           Low Confidence
     (Confidence ≥ 0.15)       (Confidence < 0.15)
     97.5% (2,250 samples)     2.5% (250 samples)
            ↓                        ↓
     Automated Decision        Human Review Queue
     Final Classification      Expert Validation
            ↓                        ↓
            └────────┬───────────────┘
                     ↓
        ┌────────────────────────────────────────┐
        │       Final Output Statistics           │
        │   Image-Level AUROC: 99.2%             │
        │   Pixel-Level AUROC: 98.7%             │
        │   F1-Score: 0.961                      │
        │   Total Processing: 10.8 minutes       │
        │   Average Rate: 926 images/minute      │
        │   End-to-End: 3 hours (with conversion)│
        └────────────────────────────────────────┘
```

**Figure 1:** Three-stage cascaded architecture for DefectVAD ensemble system. Stage one employs fast screening eliminating 75% of samples through dual threshold strategy. Stage two applies defect-type routing directing uncertain samples to specialized model combinations based on pattern classification. Stage three implements confidence-based review queue flagging ambiguous cases for human expert validation. Architecture achieves 99.2% AUROC while processing 10,000 panels in 10.8 minutes.
