#show figure.where(kind: table): set block(breakable: true)
#import "@preview/tablem:0.3.0": tablem, three-line-table

#import "@preview/lilaq:0.5.0" as lq
#set text(font: "New Computer Modern")
#show raw.where(block: true): it => [
  #set text(font: "Iosevka NF")
  #v(-4mm)
  #h(1pt) #block(inset: 8pt, fill: gray.lighten(90%), width: 100%)[#it] #h(1pt)
]

#show raw.where(block: false): it => [
  #set text(font: "Iosevka NF")
  #h(1pt) #box(outset: 3pt, fill: gray.lighten(80%))[#it] #h(1pt)
]

#show heading.where(level: 2): it => [
  #align(left)[#text(size: 12pt)[#it]]
]

#show heading.where(level: 1): it => [
  #align(center)[#text(size: 18pt)[#it]]
]
#set par(justify: true)
#set page(paper: "a4")

#align(center)[
  #v(20mm)
  #smallcaps()[#text(size: 24pt, weight: "bold")[UCS615]] \

  #smallcaps()[#text(
    size: 24pt,
    weight: "bold",
  )[Deep Learning]]
]

#place(center + horizon, dy: -20pt)[
  #image("assets/thapar.png", width: 50%)
]

#place(center + bottom, dy: -30pt)[
  #v(-20mm)
  #smallcaps()[#text(weight: "bold", size: 20pt)[Project Report]]
  #v(2mm)
  #text(size: 16pt)[Submitted By: \
    #table(columns: 2, stroke: 0pt, align: left)[Nitish][*102303239*][Himanshu Sardana][*102303244*][Tavish Sood][*102303246*][Aishani Shreya][*102303250*]
  ]

  #text(size: 16pt)[Submitted To: *Dr. Ritu Rana*] \

  #text(size: 16pt)[Session: *January to May 2026*]
]

#pagebreak()

#outline(title: "Table of Contents", depth: 1)

#pagebreak()

= Introduction
#v(5mm)
This project addresses the *Recursion Cellular Image Classification* (RxRx1) challenge, a Kaggle competition focused on predicting which siRNA (small interfering RNA) was applied to a given well of cells based on high-throughput microscopy images. The dataset comprises 6-channel fluorescence microscopy images across multiple experimental batches, cell types, and imaging sites. The core difficulty lies in the high intra-class variability caused by experimental noise and batch effects, making robust feature extraction essential.

We experiment with several deep convolutional neural network architectures --- ResNet-50, DenseNet-121, and others --- adapting them to handle 6-channel inputs (instead of the standard 3-channel RGB) and training them under controlled splits to evaluate their ability to generalize across biological replicates.

#pagebreak()
= Literature Review

== Deep Learning for Biomedical Image Classification

The application of deep convolutional neural networks (CNNs) to biomedical imaging has transformed the field over the past decade. Unlike natural images, microscopy data presents unique challenges: high dynamic range, multi-channel fluorescence, experimental batch effects, and substantial biological variability between replicates. @lecun1998gradient established the foundational principles of CNNs with gradient-based learning, while later architectures such as AlexNet @krizhevsky2012imagenet and VGGNet @simonyan2015very demonstrated that depth and hierarchical feature extraction are critical for complex visual recognition tasks.

In the context of high-content screening (HCS) and cellular imaging, deep learning models have been employed for phenotype classification, drug mechanism prediction, and genetic perturbation identification. The key requirement is robustness to nuisance variation --- changes in illumination, staining intensity, or cellular confluence that should not affect the class prediction.

== Residual Networks

*Residual Networks (ResNet)*, introduced by @he2016deep, addressed the degradation problem in very deep networks by introducing skip (shortcut) connections. Instead of learning a direct mapping $cal(H)(x)$, each residual block learns a residual function $cal(F)(x) = cal(H)(x) - x$, allowing the network to bypass layers when they are not beneficial. This reformulation preserves gradient flow during backpropagation and enables architectures with hundreds of layers to be trained effectively.

ResNet-50, specifically, employs a *bottleneck* design: each block contains three convolutions (1$times$1, 3$times$3, 1$times$1) that first reduce, then process, then restore the channel dimension. This design reduces computational cost while maintaining representational power. ResNet-50 has become a standard backbone in transfer learning because its pre-trained ImageNet weights provide strong initial feature extractors that generalize well across domains, including microscopy.

== Dense Convolutional Networks

*Dense Convolutional Networks (DenseNet)*, proposed by @huang2017densely, extend the idea of skip connections to an extreme: every layer within a dense block is directly connected to every other layer in a feed-forward manner. Formally, the $l$-th layer receives feature maps from all preceding layers $[x_0, x_1, ..., x_(l-1)]$ as input. This dense connectivity encourages feature reuse, reduces the number of parameters needed, and strengthens gradient flow throughout the network.

DenseNet-121 consists of four dense blocks separated by transition layers that apply convolution and pooling to compress the feature space. With a growth rate of 32, each layer adds only 32 new feature maps, keeping the model compact. Empirical studies have shown that DenseNets often achieve better generalization than ResNets on limited training data, a property particularly relevant for biological imaging datasets where annotated samples can be scarce.

== EfficientNet

*EfficientNet*, proposed by @tan2019efficientnet, introduced a novel compound scaling method that uniformly scales network depth, width, and resolution using a principled
$ φ $ ratio. The core building block is the *Mobile Inverted Bottleneck (MBConv)*, which applies depthwise separable convolutions followed by squeeze-and-excitation optimization. EfficientNet-B0, the base variant, achieves competitive accuracy with 5.3M parameters --- significantly smaller than ResNet-50's 25.6M --- while maintaining strong transfer learning performance. Its compound scaling strategy ensures optimal trade-offs between computational cost and accuracy, making it particularly suitable for high-throughput screening scenarios where both precision and inference speed matter.

== ResNeXt

*ResNeXt*, introduced by @xie2017aggregated, extends ResNet's residual blocks with the concept of *aggregated residual transformations*. Instead of learning complex spatial mappings directly, ResNeXt decomposes each bottleneck block's transformation into multiple parallel paths (cardinality) operating at reduced width. This design increases representational diversity without significantly increasing computational complexity. ResNeXt-50 (32×4d) --- used in this project --- has a cardinality of 32 and a bottleneck width of 4, striking an effective balance between model capacity and training efficiency. The aggregated structure has been shown to outperform vanilla ResNets on ImageNet while maintaining similar parameter counts.

== Vision Transformer (ViT)

*Vision Transformer (ViT)*, introduced by @dosovitskiy2021vit, adapts the transformer architecture originally designed for natural language processing to computer vision tasks. Images are split into fixed-size patches (16×16 tokens), linearly embedded, and processed by a standard transformer encoder with multi-head self-attention. Without any convolution layers, ViT-Base/16 achieves competitive results on ImageNet when pre-trained on large datasets (JFT-300M), though it underperforms CNNs on smaller datasets due to lack of inductive biases.

== ConvNeXt and ConvNeXt V2

*ConvNeXt*, proposed by @liu2022convnext, modernizes the classic convolutional architecture by incorporating design principles from vision transformers: inverted bottlenecks, large kernel sizes (7×7 depthwise convolutions), and layer normalization. ConvNeXt V2 @liu2023convnextv2 further extends this with a *Global Response Normalization (GRN)* layer that improves channel-wise competition, enabling co-design with masked autoencoders for self-supervised pre-training. ConvNeXt V2 nano (5.2M params) offers a lightweight option suitable for resource-constrained settings.

== Chaotic CNN

*Chaotic CNN* architectures leverage chaos theory to enhance feature extraction through nonlinear dynamical systems. The *skew-tent map*, *logistic map*, and *sine map* are one-dimensional discrete dynamical systems that exhibit deterministic chaos. By incorporating these maps into CNN architectures, the model can capture complex nonlinear patterns in cellular images that linear convolutions may miss. The skew-tent map $f(x) = {(x/p "for" x < p), ( (1-x)/(1-p) "for" x ≥ p):}$ and logistic map $x_(n+1) = μ x_n (1 - x_n)$ provide distinct bifurcation behaviors that enrich the feature space with chaotic dynamics.

== Multi-Channel Input Adaptation
#v(3mm)
Standard CNN backbones are designed for 3-channel RGB images, whereas fluorescence microscopy frequently captures 4, 5, or 6 channels representing different cellular components (e.g., nuclei, endoplasmic reticulum, actin). A common transfer-learning strategy, as described in the @recursion2019rxrx1 competition discussions, is to modify the first convolutional layer to accept $C$ channels while keeping all deeper layers intact. Pre-trained weights from the original 3-channel filter are replicated or averaged across the new channels. This approach preserves the low-level edge and texture detectors learned from ImageNet while adapting the input layer to the new modality.

== Test-Time Augmentation and Metric Learning
#v(3mm)
Beyond architecture choice, two advanced techniques have proven effective in cellular image competitions. *Test-Time Augmentation (TTA)* involves averaging model predictions across multiple augmented views of the same image --- such as horizontal flips, vertical flips, and rotations --- at inference time. This reduces prediction variance and often yields a 1--3% accuracy improvement without retraining.

*Metric learning* approaches, particularly the ArcFace loss @deng2019arcface, reformulate classification as an embedding learning problem. Instead of a standard linear classifier, ArcFace applies an angular margin penalty to the cosine similarity between image embeddings and class centroids. This enforces a more discriminative feature space, which is especially valuable when the number of classes is large (e.g., 1108 siRNAs) and inter-class differences are subtle.

#pagebreak()

= Methodology

== ResNet-50
#v(5mm)
*ResNet-50* is a 50-layer convolutional neural network that uses residual (skip) connections to mitigate vanishing gradients. For this project, the first convolutional layer was modified from 3 input channels to 6 channels to accommodate the 6 fluorescence channels of the microscopy images. Pre-trained ImageNet weights were used for initialization, with the RGB weights replicated across the additional 3 channels.

=== Architecture Details
- *Stem*: 7$times$7 convolution with 64 filters, stride 2, followed by max pooling.
- *Residual Stages*: Four stages with bottleneck blocks (filter sizes 64, 128, 256, 512) and identity shortcuts.
- *Global Average Pooling*: Reduces spatial dimensions before the classifier.
- *Classifier*: A single fully-connected layer mapping to 1108 classes (one per siRNA).

=== Training Configuration
- *Loss*: Cross-Entropy Loss
- *Optimizer*: AdamW with learning rate $3 times 10^(-4)$ and weight decay $10^(-4)$
- *Scheduler*: Cosine Annealing over 10 epochs
- *Image Size*: 320$times$320
- *Batch Size*: 32

== DenseNet-121
#v(5mm)
*DenseNet-121* is a 121-layer CNN where each layer receives feature maps from all preceding layers within a dense block. This dense connectivity promotes feature reuse and strengthens gradient flow. Like ResNet-50, the first convolution (`conv0`) was adapted to accept 6 channels by replicating pretrained weights across all input channels.

=== Architecture Details
- *Stem*: 7$times$7 convolution with 64 filters, stride 2.
- *Dense Blocks*: Four dense blocks with transition layers in between; growth rate of 32.
- *Compression*: Transition layers use a compression factor of 0.5 to control model size.
- *Classifier*: Global average pooling followed by a linear layer to 1108 classes.

=== Training Configuration
- *Loss*: Cross-Entropy Loss
- *Optimizer*: AdamW with learning rate $3 times 10^(-4)$ and weight decay $10^(-4)$
- *Scheduler*: Cosine Annealing over 20 epochs
- *Image Size*: 320$times$320
- *Batch Size*: 32
- *Augmentation*: Random horizontal/vertical flips, rotations
- *TTA*: 6-pass test-time augmentation at inference
- *Cell Types*: All four (HEPG2, HUVEC, RPE, U2OS)

== EfficientNet-B0
#v(5mm)
*EfficientNet-B0* is a lightweight yet powerful CNN that uses compound scaling to uniformly scale depth, width, and resolution. Unlike ResNet and DenseNet which were designed for ImageNet's 3-channel inputs, EfficientNet's Mobile Inverted Bottleneck (MBConv) blocks use depthwise separable convolutions, making it highly parameter-efficient while maintaining strong representational power.

This model was trained on *all four cell types* (HEPG2, HUVEC, RPE, U2OS) with the full dataset of 36,517 training samples and 19,897 test samples, providing a more comprehensive evaluation across biological conditions.

=== Architecture Details
- *Stem*: 3$times$3 convolution with 32 filters.
- *MBConv Blocks*: Seven stages of Mobile Inverted Bottleneck blocks with squeeze-and-excitation (SE) optimization.
- *Head*: Global average pooling followed by a fully-connected layer to 1108 classes.
- *Efficient Scaling*: Uniformly scales width, depth, and resolution for optimal accuracy--efficiency trade-off.

=== Training Configuration
- *Loss*: Cross-Entropy Loss
- *Optimizer*: AdamW with learning rate $3 times 10^(-4)$ and weight decay $10^(-4)$
- *Scheduler*: Cosine Annealing over 10 epochs
- *Image Size*: 320$times$320
- *Batch Size*: 32

=== Test-Time Augmentation
For final submission, TTA was applied using dual-site + flip augmentation (horizontal and vertical) across both imaging sites, generating 6 augmented views per sample for ensemble prediction.

== EfficientNet-B4
#v(5mm)
*EfficientNet-B4* is a larger variant in the EfficientNet family, scaled with a higher compound coefficient than B0. It uses the same Mobile Inverted Bottleneck (MBConv) blocks with squeeze-and-excitation optimization but with increased depth, width, and input resolution. This model was trained on all four cell types without data augmentation or TTA.

=== Architecture Details
- *Stem*: 3$times$3 convolution with 48 filters.
- *MBConv Blocks*: Seven stages with expanded channel dimensions compared to B0.
- *Head*: Global average pooling followed by a fully-connected layer to 1108 classes.

=== Training Configuration
- *Loss*: Cross-Entropy Loss
- *Optimizer*: AdamW with learning rate $3 times 10^(-4)$ and weight decay $10^(-4)$
- *Scheduler*: Cosine Annealing over 14 epochs
- *Two-Phase Training*: Frozen backbone (1 epoch) → Full fine-tuning (13 epochs)
- *Image Size*: 320$times$320
- *Batch Size*: 32
- *Cell Types*: All four (HEPG2, HUVEC, RPE, U2OS)

== ConvNeXt V2 (Tiny)
#v(5mm)
*ConvNeXt V2* is a modernized convolutional architecture that incorporates design principles from vision transformers while retaining the efficiency of standard convolutions. The tiny variant uses *Global Response Normalization (GRN)* to improve channel competition and features a fully-convolutional design with inverted bottlenecks and large-kernel depthwise convolutions (7×7). This model was trained on all four cell types.

=== Architecture Details
- *Stem*: 4×4 convolution with 96 filters.
- *Stages*: Four stages with varying depths; inverted bottleneck blocks with large-kernel convolutions.
- *GRN*: Global Response Normalization in FFN layers for improved channel diversity.
- *Head*: Global average pooling followed by a linear layer to 1108 classes.

=== Training Configuration
- *Loss*: Cross-Entropy Loss
- *Optimizer*: AdamW with learning rate $3 × 10^(-4)$ and weight decay $10^(-4)$
- *Scheduler*: Cosine Annealing over 20 epochs
- *Image Size*: 320×320
- *Batch Size*: 32 (effective 64 with gradient accumulation)
- *TTA*: 6-pass test-time augmentation at inference
- *Cell Types*: All four (HEPG2, HUVEC, RPE, U2OS)

== Vision Transformer (ViT-Base/16)
#v(5mm)
*Vision Transformer (ViT)* applies the transformer encoder architecture to image classification by splitting images into 16×16 patches. Unlike CNNs, ViT relies entirely on self-attention mechanisms to capture global dependencies. The base variant uses 12 transformer layers with 12 attention heads per layer, totaling approximately 86M parameters. Training used mixed precision (FP16) for memory efficiency.

=== Architecture Details
- *Patch Embedding*: 16×16 patches projected to 768-dimensional space.
- *Transformer Encoder*: 12 layers, 12 heads, 768-dim attention.
- *Classification Head*: [CLS] token → linear classifier.
- *Input*: 224×224 images (upsampled from 320×320).

=== Training Configuration
- *Loss*: Cross-Entropy Loss
- *Optimizer*: AdamW with learning rate $3 × 10^(-4)$
- *Scheduler*: Cosine Annealing over 20 epochs
- *Mixed Precision*: FP16 for memory optimization

== Chaotic CNN (DenseNet-121 + Chaotic Maps)
#v(5mm)
*Chaotic CNN* augments the DenseNet-121 backbone with *chaotic dynamical systems* integrated into the feature extraction pipeline. These maps generate deterministic chaotic sequences that modulate channel features, introducing nonlinear dynamics inspired by chaos theory. Three map variants were experimented with:

=== Architecture Details
- *Backbone*: Standard DenseNet-121 with pretrained ImageNet weights.
- *Chaotic Module*: Chaotic sequence generator applied to channel-wise feature modulation.
- *Variants Tested*: Skew-tent map, Logistic map ($μ=4$), Sine map.

=== Training Configuration
- *Loss*: Cross-Entropy Loss
- *Optimizer*: AdamW with learning rate $3 × 10^(-4)$
- *Scheduler*: Cosine Annealing over 20 epochs
- *Image Size*: 320×320
- *Batch Size*: 32

== ResNeXt-50 (32×4d)
#v(5mm)
*ResNeXt-50 (32×4d)* extends ResNet-50 with aggregated residual transformations. Instead of a single wide convolution path, it uses 32 parallel paths (cardinality=32) each with width 4, providing greater representational diversity. This model was trained on *all four cell types* (HEPG2, HUVEC, RPE, U2OS) using an experiment-level train-validation split.

=== Architecture Details
- *Stem*: 7×7 convolution with 64 filters, stride 2.
- *Aggregated Bottleneck Blocks*: 32 parallel paths per block.
- *Stages*: Four stages with (3, 4, 6, 3) blocks.
- *Classifier*: Global average pooling → 1108 classes.

=== Training Configuration
- *Loss*: Cross-Entropy Loss
- *Optimizer*: AdamW with learning rate $3 × 10^(-4)$, weight decay $10^(-4)$
- *Scheduler*: Cosine Annealing over 30 epochs
- *Two-Phase Training*: Frozen backbone (1 epoch) → Full fine-tuning (29 epochs)
- *Image Size*: 224×224

== Inception V3
#v(5mm)
*Inception V3*, introduced by Szegedy et al., uses factorized convolutions and auxiliary classifiers to improve computational efficiency. It applies asymmetric convolutions (e.g., 1$times$7 followed by 7$times$1) and multi-scale feature extraction through parallel inception modules. This model was trained with test-time augmentation (TTA).

=== Architecture Details
- *Stem*: Multiple convolution and pooling layers with asymmetric filters.
- *Inception Modules*: Parallel convolutions at multiple scales (1$times$1, 3$times$3, 5$times$5) with pooling.
- *Auxiliary Classifier*: Additional classification head for regularization (disabled during transfer learning).
- *Head*: Global average pooling → 1108 classes.

=== Training Configuration
- *Loss*: Cross-Entropy Loss
- *Optimizer*: AdamW with learning rate $3 times 10^(-4)$
- *Scheduler*: Cosine Annealing over 10 epochs
- *Two-Phase Training*: Frozen backbone (1 epoch) → Full fine-tuning (9 epochs)
- *Image Size*: 320$times$320
- *Batch Size*: 32
- *TTA*: Multi-pass test-time augmentation at inference

#pagebreak()

= Implementation Details
#v(5mm)
The models were implemented in PyTorch and trained on an NVIDIA Tesla T4 GPU via Kaggle Notebooks. The dataset was filtered to the *HUVEC* cell type for the completed runs, yielding 17,689 training samples and 8,846 test samples. An 85/15 stratified train-validation split was used.

Because the original siRNA labels were sparse (ranging up to values larger than 1107), a label encoder was applied to remap them to a contiguous 0--1107 range before training. This prevents indexing errors in the final classification layer.

== ResNet-50 Training Log
The model was trained for 10 epochs. The loss and accuracy values recorded after each epoch are summarized below:

#figure(
  caption: [ResNet-50 Training and Validation Metrics],
)[
  #three-line-table(columns: (1fr, 2fr, 2fr, 2fr, 2fr))[
    | *Epoch* | *Train Loss* | *Train Acc* | *Val Loss* | *Val Acc* |
    | 1 | 6.9417 | 0.15% | 6.7151 | 0.41% |
    | 2 | 6.1782 | 1.40% | 5.3732 | 5.01% |
    | 3 | 4.6563 | 11.50% | 4.3221 | 14.73% |
    | 4 | 3.3380 | 29.94% | 3.5857 | 24.27% |
    | 5 | 2.4297 | 48.22% | 3.2956 | 30.33% |
    | 6 | 1.8331 | 61.93% | 3.1245 | 34.06% |
    | 7 | 1.4440 | 71.74% | 3.0479 | 35.27% |
    | 8 | 1.2069 | 78.32% | 2.9688 | 36.44% |
    | 9 | 1.0592 | 81.96% | 2.9528 | 37.26% |
    | 10 | 0.9937 | 83.66% | 2.9565 | 37.49% |
  ]
]

*Best Validation Accuracy: 37.49%* (Epoch 10)

== DenseNet-121 (Augmentation + TTA) Training Log
The model was trained for 20 epochs on all four cell types (HEPG2, HUVEC, RPE, U2OS) with augmentation and test-time augmentation (TTA). Its training progression is summarized below:

#figure(
  caption: [DenseNet-121 (Aug + TTA) Training and Validation Metrics],
)[
  #three-line-table(columns: (1fr, 2fr, 2fr, 2fr, 2fr))[
    | *Epoch* | *Train Loss* | *Train Acc* | *Val Loss* | *Val Acc* |
    | 1 | 7.0167 | 0.16% | 6.7316 | 0.86% |
    | 2 | 6.6803 | 0.76% | 6.3504 | 1.95% |
    | 3 | 6.2441 | 2.65% | 5.8501 | 5.37% |
    | 4 | 5.8060 | 6.13% | 5.4522 | 9.38% |
    | 5 | 5.4203 | 10.43% | 4.9985 | 15.79% |
    | 6 | 5.1210 | 14.87% | 4.7622 | 20.41% |
    | 7 | 4.8672 | 19.00% | 4.5965 | 22.32% |
    | 8 | 4.6387 | 22.76% | 4.3184 | 27.12% |
    | 9 | 4.4313 | 26.66% | 4.1164 | 31.90% |
    | 10 | 4.2441 | 30.27% | 3.9788 | 34.84% |
    | 11 | 4.0835 | 33.72% | 3.8590 | 37.56% |
    | 12 | 3.9349 | 36.55% | 3.7567 | 39.78% |
    | 13 | 3.8072 | 39.36% | 3.6743 | 41.12% |
    | 14 | 3.7031 | 41.77% | 3.6075 | 42.66% |
    | 15 | 3.6009 | 43.91% | 3.5349 | 44.72% |
    | 16 | 3.5162 | 45.96% | 3.4949 | 46.17% |
    | 17 | 3.4648 | 47.26% | 3.4411 | 47.42% |
    | 18 | 3.4218 | 48.38% | 3.4562 | 46.81% |
    | 19 | 3.4080 | 48.62% | 3.4728 | 46.62% |
    | 20 | 3.3847 | 49.39% | 3.4377 | 47.26% |
  ]
]

*Best Validation Accuracy: 47.42%* (Epoch 17)

== EfficientNet-B0 Training Log
The model was trained for 10 epochs on all four cell types (HEPG2, HUVEC, RPE, U2OS). Its training progression is summarized below:

#figure(
  caption: [EfficientNet-B0 Training and Validation Metrics],
)[
  #three-line-table(columns: (1fr, 2fr, 2fr, 2fr, 2fr))[
    | *Epoch* | *Train Loss* | *Train Acc* | *Val Loss* | *Val Acc* |
    | 1 | 6.6601 | 1.18% | 6.0470 | 3.98% |
    | 2 | 5.6555 | 7.46% | 5.2280 | 13.25% |
    | 3 | 4.9103 | 17.14% | 4.6951 | 21.23% |
    | 4 | 4.3488 | 26.18% | 4.3486 | 28.20% |
    | 5 | 3.9275 | 34.60% | 4.0928 | 32.18% |
    | 6 | 3.5940 | 41.91% | 3.8343 | 37.73% |
    | 7 | 3.3287 | 48.39% | 3.6166 | 42.42% |
    | 8 | 3.1454 | 52.75% | 3.5372 | 44.36% |
    | 9 | 3.0048 | 56.55% | 3.5236 | 44.23% |
    | 10 | 2.9406 | 57.88% | 3.4870 | 44.85% |
  ]
]

*Best Validation Accuracy: 44.85%* (Epoch 10)

== EfficientNet-B4 Training Log
The model was trained for 14 epochs on all four cell types (HEPG2, HUVEC, RPE, U2OS) without augmentation or TTA. Its training progression is summarized below:

#figure(
  caption: [EfficientNet-B4 Training and Validation Metrics],
)[
  #three-line-table(columns: (1fr, 2fr, 2fr, 2fr, 2fr))[
    | *Epoch* | *Train Loss* | *Train Acc* | *Val Loss* | *Val Acc* |
    | 1 (Frozen) | 7.0375 | 0.21% | 6.7248 | 0.82% |
    | 2 | 6.5534 | 1.47% | 6.1183 | 3.85% |
    | 3 | 5.8667 | 6.34% | nan | 6.93% |
    | 4 | 5.0988 | 15.56% | 5.5418 | 10.04% |
    | 5 | 4.2990 | 29.13% | 5.5827 | 10.83% |
    | 6 | 3.5100 | 45.89% | 5.7409 | 10.63% |
    | 7 | 2.8412 | 62.30% | 5.9402 | 10.08% |
  ]
]

*Best Validation Accuracy: 10.83%* (Epoch 5) — Validation loss became unstable (nan at epoch 3); severe overfitting observed.

== ConvNeXt V2 Tiny Training Log
The model was trained for 20 epochs on all four cell types with TTA, but early stopping triggered at epoch 6 due to lack of improvement:

#figure(
  caption: [ConvNeXt V2 Tiny Training and Validation Metrics],
)[
  #three-line-table(columns: (1fr, 2fr, 2fr, 2fr, 2fr))[
    | *Epoch* | *Train Loss* | *Train Acc* | *Val Loss* | *Val Acc* |
    | 1 | 6.9913 | 0.21% | 6.6528 | 1.04% |
    | 2 | 6.8814 | 0.17% | 25.8076 | 0.09% |
    | 3 | 6.7558 | 0.31% | 16.1042 | 0.11% |
    | 4 | 6.6470 | 0.41% | 11.8655 | 0.20% |
    | 5 | 6.5632 | 0.55% | 21.9518 | 0.11% |
    | 6 | 6.4848 | 0.77% | 10.9303 | 0.39% |
  ]
]

*Best Validation Accuracy: 1.04%* (Epoch 1) — Training plateaued; model failed to learn meaningful features.

== ViT-Base/16 Training Log
The model was trained for 4 epochs (early stopping triggered due to no improvement):

#figure(
  caption: [Vision Transformer (ViT-Base/16) Training and Validation Metrics],
)[
  #three-line-table(columns: (1fr, 2fr, 2fr, 2fr, 2fr))[
    | *Epoch* | *Train Loss* | *Train Acc* | *Val Loss* | *Val Acc* |
    | 1 | 7.1742 | 0.09% | 7.0832 | 0.11% |
    | 2 | 7.0888 | 0.05% | 7.0719 | 0.04% |
    | 3 | 7.0717 | 0.05% | 7.0687 | 0.04% |
    | 4 | 7.0590 | 0.06% | 7.0645 | 0.00% |
  ]
]

*Best Validation Accuracy: 0.11%* (Epoch 1) — ViT failed to learn on this dataset; attributed to insufficient inductive biases for microscopy data and small training set.

== Chaotic CNN (Skew-Tent Map) Training Log
The model was trained for 11 epochs:

#figure(
  caption: [Chaotic CNN (Skew-Tent Map) Training and Validation Metrics],
)[
  #three-line-table(columns: (1fr, 2fr, 2fr, 2fr, 2fr))[
    | *Epoch* | *Train Loss* | *Train Acc* | *Val Loss* | *Val Acc* |
    | 1 | 6.9772 | 0.16% | 6.7598 | 0.26% |
    | 2 | 6.5127 | 0.42% | 6.2786 | 1.66% |
    | 3 | 5.8726 | 2.91% | 5.5577 | 6.48% |
    | 4 | 5.1724 | 8.37% | 5.0375 | 10.59% |
    | 5 | 4.5370 | 16.08% | 4.7442 | 13.45% |
    | 6 | 3.9398 | 25.56% | 4.4028 | 17.26% |
    | 7 | 3.3544 | 37.65% | 4.0126 | 21.59% |
    | 8 | 2.7914 | 50.75% | 3.8145 | 24.30% |
    | 9 | 2.2361 | 64.46% | 3.6379 | 28.71% |
    | 10 | 1.7423 | 77.23% | 3.5004 | 30.56% |
    | 11 | 1.3089 | 87.26% | 3.5608 | 29.43% |
  ]
]

*Best Validation Accuracy: 30.56%* (Epoch 10)

== Chaotic CNN (Logistic Map) Training Log
The model was trained for 16 epochs with early stopping:

#figure(
  caption: [Chaotic CNN (Logistic Map) Training and Validation Metrics],
)[
  #three-line-table(columns: (1fr, 2fr, 2fr, 2fr, 2fr))[
    | *Epoch* | *Train Loss* | *Train Acc* | *Val Loss* | *Val Acc* |
    | 1 | 7.0205 | 0.11% | 6.8873 | 0.30% |
    | 2 | 6.7422 | 0.24% | 6.5280 | 0.49% |
    | 3 | 6.2747 | 1.08% | 5.9932 | 2.94% |
    | 4 | 5.6682 | 4.13% | 5.4359 | 5.43% |
    | 5 | 4.9676 | 10.29% | 4.8561 | 12.89% |
    | 6 | 4.3126 | 18.90% | 4.5669 | 14.39% |
    | 7 | 3.6854 | 30.28% | 4.0990 | 20.08% |
    | 8 | 3.1021 | 41.98% | 4.0316 | 21.25% |
    | 9 | 2.5016 | 56.46% | 3.7037 | 27.69% |
    | 10 | 1.9386 | 69.80% | 3.5602 | 30.03% |
    | 11 | 1.4201 | 82.08% | 3.5174 | 30.93% |
    | 12 | 1.0176 | 90.58% | 3.5377 | 30.90% |
    | 13 | 0.7226 | 95.58% | 3.4864 | 32.22% |
    | 14 | 0.5295 | 98.09% | 3.5189 | 31.95% |
    | 15 | 0.3954 | 99.10% | 3.4949 | 31.39% |
    | 16 | 0.3135 | 99.55% | 3.5238 | 31.09% |
  ]
]

*Best Validation Accuracy: 32.22%* (Epoch 13) — Early stopping at epoch 16

== Chaotic CNN (Sine Map) Training Log
The model was trained for 20 epochs:

#figure(
  caption: [Chaotic CNN (Sine Map) Training and Validation Metrics],
)[
  #three-line-table(columns: (1fr, 2fr, 2fr, 2fr, 2fr))[
    | *Epoch* | *Train Loss* | *Train Acc* | *Val Loss* | *Val Acc* |
    | 1 | 6.9896 | 0.15% | 6.7119 | 0.34% |
    | 2 | 6.4316 | 0.73% | 6.0738 | 2.71% |
    | 3 | 5.6728 | 4.16% | 5.3186 | 7.57% |
    | 4 | 4.8648 | 11.07% | 4.8792 | 10.36% |
    | 5 | 4.1446 | 21.39% | 4.2816 | 17.97% |
    | 6 | 3.5054 | 32.43% | 4.0361 | 21.89% |
    | 7 | 2.9111 | 45.67% | 3.7019 | 25.47% |
    | 8 | 2.3345 | 60.31% | 3.5687 | 28.22% |
    | 9 | 1.7819 | 73.89% | 3.4274 | 31.09% |
    | 10 | 1.3347 | 84.28% | 3.3930 | 32.74% |
    | 11 | 0.9477 | 91.85% | 3.3316 | 32.18% |
    | 12 | 0.6780 | 96.28% | 3.3743 | 33.01% |
    | 13 | 0.4804 | 98.55% | 3.3372 | 33.12% |
    | 14 | 0.3609 | 99.60% | 3.3181 | 32.74% |
    | 15 | 0.2782 | 99.71% | 3.3120 | 33.53% |
    | 16 | 0.2266 | 99.95% | 3.3146 | 33.72% |
    | 17 | 0.1930 | 99.93% | 3.3173 | 33.27% |
    | 18 | 0.1740 | 99.96% | 3.3082 | 33.35% |
    | 19 | 0.1619 | 99.95% | 3.3094 | 33.42% |
    | 20 | 0.1558 | 99.97% | 3.3082 | 33.53% |
  ]
]

*Best Validation Accuracy: 33.72%* (Epoch 17)

== ResNeXt-50 Training Log
The model was trained for 30 epochs with two-phase training (frozen backbone → full fine-tuning):

#figure(
  caption: [ResNeXt-50 (32×4d) Training and Validation Metrics],
)[
  #three-line-table(columns: (1fr, 2fr, 2fr, 2fr, 2fr))[
    | *Epoch* | *Train Loss* | *Train Acc* | *Val Loss* | *Val Acc* |
    | 1 (Frozen) | 7.0385 | 0.16% | 6.7345 | 0.66% |
    | 2 | 6.7942 | 0.58% | 6.5552 | 1.06% |
    | 3 | 6.5252 | 1.24% | 6.2617 | 2.70% |
    | 4 | 6.2142 | 2.94% | 5.9632 | 4.96% |
    | 5 | 5.9072 | 5.41% | 5.5324 | 9.42% |
    | 6 | 5.5918 | 8.79% | 5.3963 | 12.10% |
    | 7 | 5.3152 | 12.66% | 5.0698 | 15.56% |
    | 8 | 5.0909 | 15.85% | 4.8948 | 19.85% |
    | 9 | 4.8703 | 19.61% | 4.6125 | 22.77% |
    | 10 | 4.6861 | 22.63% | 4.4282 | 26.57% |
    | 11 | 4.5131 | 25.80% | 11.9006 | 28.05% |
    | 12 | 4.3458 | 28.33% | 4.2065 | 32.58% |
    | 13 | 4.2000 | 31.50% | 3.9630 | 35.64% |
    | 14 | 4.0717 | 34.00% | 3.7505 | 40.60% |
    | 15 | 3.9505 | 36.61% | 3.7764 | 41.30% |
    | 16 | 3.8484 | 38.64% | 3.9199 | 40.12% |
    | 17 | 3.7410 | 40.68% | 3.6179 | 44.77% |
    | 18 | 3.6357 | 42.83% | 4.0202 | 45.92% |
    | 19 | 3.5496 | 44.85% | 3.4070 | 47.10% |
    | 20 | 3.4740 | 46.63% | 3.3077 | 49.98% |
    | 21 | 3.4210 | 47.91% | 3.2326 | 51.59% |
    | 22 | 3.3496 | 49.35% | 3.2198 | 51.50% |
    | 23 | 3.2863 | 50.93% | 3.1764 | 53.78% |
    | 24 | 3.2474 | 52.09% | 3.1779 | 53.01% |
    | 25 | 3.2142 | 52.61% | 3.1636 | 53.10% |
    | 26 | 3.1766 | 53.74% | 3.1395 | 53.72% |
    | 27 | 3.1581 | 54.21% | 3.0644 | 55.07% |
    | 28 | 3.1435 | 54.46% | 3.0838 | 54.69% |
    | 29 | 3.1353 | 54.90% | 3.1006 | 54.35% |
    | 30 | 3.1280 | 54.76% | 3.1018 | 54.69% |
  ]
]

*Best Validation Accuracy: 55.07%* (Epoch 27) — Achieved on all four cell types

== Inception V3 Training Log
The model was trained for 10 epochs with test-time augmentation (TTA). Its training progression is summarized below:

#figure(
  caption: [Inception V3 Training and Validation Metrics],
)[
  #three-line-table(columns: (1fr, 2fr, 2fr, 2fr, 2fr))[
    | *Epoch* | *Train Loss* | *Train Acc* | *Val Loss* | *Val Acc* |
    | 1 (Frozen) | 7.0471 | 0.16% | 6.8068 | 0.72% |
    | 2 | 6.8515 | 0.31% | 7.1554 | 1.16% |
    | 3 | 6.5686 | 0.83% | 6.5015 | 1.40% |
    | 4 | 6.3339 | 1.52% | 7.2565 | 2.20% |
    | 5 | 6.1303 | 2.69% | 6.3189 | 3.53% |
    | 6 | 5.9130 | 4.11% | 5.6685 | 6.80% |
    | 7 | 5.7124 | 6.03% | 5.6866 | 8.52% |
    | 8 | 5.5515 | 7.90% | 5.5133 | 9.79% |
    | 9 | 5.4378 | 9.08% | 5.5648 | 9.81% |
    | 10 | 5.3892 | 10.07% | 6.0558 | 8.27% |
  ]
]

*Best Validation Accuracy: 9.81%* (Epoch 9)

== Observations
- DenseNet-121 (Aug + TTA) consistently outperformed ResNet-50 on validation accuracy across all epochs, peaking at *47.42%* versus *37.49%* when trained on all four cell types.
- ResNet-50 showed stronger signs of overfitting after Epoch 8: training accuracy climbed to 83.66% while validation accuracy plateaued near 37%.
- DenseNet-121 maintained a healthier train--val gap, suggesting that dense feature reuse provides better generalization on this cellular imaging task.
- Both models started from near-random accuracy ($approx$ 0.1--0.4%), confirming the difficulty of the 1108-way classification problem.
- *ConvNeXt V2 Tiny* failed to learn meaningful features, achieving only 1.04% validation accuracy. This is attributed to the need for pre-trained weights on ImageNet, which may not transfer well to 6-channel microscopy data without proper adaptation.
- *Vision Transformer (ViT-Base/16)* similarly failed to learn, reaching only 0.11% accuracy. ViT requires large-scale pre-training to develop inductive biases; without this, it underperforms even simple CNN baselines on microscopy data.
- *Chaotic CNN* variants showed moderate performance: Skew-Tent (30.56%), Logistic (32.22%), and Sine (33.72%). The chaotic dynamics introduced additional regularization, but the performance gains over standard DenseNet-121 were not observed.
- *Inception V3* with TTA achieved 9.81% validation accuracy, showing slow but steady improvement before overfitting at epoch 10.
- *EfficientNet-B4* without augmentation or TTA suffered from severe overfitting and unstable validation loss (nan at epoch 3), peaking at only 10.83%.
- *ResNeXt-50* achieved the best performance across all models at *55.07%* validation accuracy, trained on all four cell types with 30 epochs and a two-phase training strategy.

#pagebreak()
= Results and Discussion
#v(5mm)

#figure(
  caption: [Model Comparison: Validation Accuracies and Best Performance],
)[
  #three-line-table(columns: (2fr, 2fr, 2fr, 2fr))[
    | *Model* | *Best Val Acc* | *Epoch* | *Cell Type* |
    | ResNet-50 (Baseline) | 38.09% | 11 | HUVEC |
    | ResNet-50 | 37.49% | 10 | HUVEC |
    | DenseNet-121 (Aug + TTA) | 47.42% | 17 | All |
    | DenseNet-121 (No Aug) | 46.65% | 10 | HUVEC |
    | EfficientNet-B0 | 44.85% | 10 | All |
    | EfficientNet-B4 (No Aug, No TTA) | 10.83% | 5 | All |
    | ConvNeXt V2 Tiny | 1.04% | 1 | All |
    | ViT-Base/16 | 0.11% | 1 | HUVEC |
    | Chaotic CNN (Skew-Tent) | 30.56% | 10 | HUVEC |
    | Chaotic CNN (Logistic) | 32.22% | 13 | HUVEC |
    | Chaotic CNN (Sine) | 33.72% | 17 | HUVEC |
    | Inception V3 (TTA) | 9.81% | 9 | All |
    | ResNeXt-50 (32×4d) | 55.07% | 27 | All |
  ]
]
#v(2mm)
#box(inset: 5pt)[
  #text(size: 9pt)[*Key Findings:*
    • ResNeXt-50 achieved best at 55.07%
    • DenseNet-121 (Aug + TTA) outperformed ResNet-50 at 47.42%
    • ConvNeXt/ViT failed (ImageNet transfer)
    • Chaotic CNN variants: 30--34%
    • EfficientNet-B4 and Inception V3 underperformed]
]

#v(3mm)
=== Overfitting Onset Analysis

The table below records the first epoch at which training accuracy exceeded validation accuracy for each model, indicating the onset of overfitting:

#figure(
  caption: [Overfitting Onset by Model],
)[
  #three-line-table(columns: (2fr, 2fr, 2fr, 2fr))[
    | *Model* | *Overfitting Epoch* | *Best Val Acc* | *Total Epochs* |
    | ResNet-50 | 4 | 37.49% | 10 |
    | DenseNet-121 (Aug + TTA) | 18 | 47.42% | 20 |
    | EfficientNet-B0 | 5 | 44.85% | 10 |
    | EfficientNet-B4 (No Aug, No TTA) | 4 | 10.83% | 14 |
    | ConvNeXt V2 Tiny | 2 | 1.04% | 6 |
    | ViT-Base/16 | 2 | 0.11% | 4 |
    | Chaotic CNN (Skew-Tent) | 5 | 30.56% | 11 |
    | Chaotic CNN (Logistic) | 6 | 32.22% | 16 |
    | Chaotic CNN (Sine) | 4 | 33.72% | 20 |
    | Inception V3 (TTA) | 10 | 9.81% | 10 |
    | ResNeXt-50 (32×4d) | 26 | 55.07% | 30 |
  ]
]

DenseNet-121 with augmentation and TTA demonstrated the strongest resistance to overfitting, with the train--val gap only inverting at epoch 18. In contrast, ConvNeXt V2 Tiny and ViT began overfitting almost immediately (epoch 2), reflecting their inability to learn discriminative features for this microscopy task. ResNeXt-50 maintained generalization until epoch 26, underscoring the benefit of aggregated residual transformations and training on all four cell types.

#v(3mm)
=== Validation Loss Curves (Epochs 1--10)

#figure(
  caption: [Validation Loss Comparison Across All CNN Models (Epochs 1--10)],
)[
  #align(center)[
    #lq.diagram(
      width: 12cm,
      height: 10cm,
      xlabel: [Epoch],
      ylabel: [Validation Loss],
      xlim: (1, 10),
      ylim: (2.0, 7.5),
      legend: (position: top + right),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          6.7151,
          5.3732,
          4.3221,
          3.5857,
          3.2956,
          3.1245,
          3.0479,
          2.9688,
          2.9528,
          2.9565,
        ),
        label: [ResNet-50],
        mark: "o",
      ),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          6.7316,
          6.3504,
          5.8501,
          5.4522,
          4.9985,
          4.7622,
          4.5965,
          4.3184,
          4.1164,
          3.9788,
        ),
        label: [DenseNet-121 (Aug + TTA)],
        mark: "x",
      ),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          6.0470,
          5.2280,
          4.6951,
          4.3486,
          4.0928,
          3.8343,
          3.6166,
          3.5372,
          3.5236,
          3.4870,
        ),
        label: [EfficientNet-B0],
        mark: "s",
      ),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          6.7598,
          6.2786,
          5.5577,
          5.0375,
          4.7442,
          4.4028,
          4.0126,
          3.8145,
          3.6379,
          3.5004,
        ),
        label: [Chaotic (Skew-Tent)],
        mark: "^",
      ),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          6.8873,
          6.5280,
          5.9932,
          5.4359,
          4.8561,
          4.5669,
          4.0990,
          4.0316,
          3.7037,
          3.5602,
        ),
        label: [Chaotic (Logistic)],
        mark: "v",
      ),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          6.7119,
          6.0738,
          5.3186,
          4.8792,
          4.2816,
          4.0361,
          3.7019,
          3.5687,
          3.4274,
          3.3930,
        ),
        label: [Chaotic (Sine)],
        mark: "d",
      ),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          6.7345,
          6.5552,
          6.2617,
          5.9632,
          5.5324,
          5.3963,
          5.0698,
          4.8948,
          4.6125,
          4.4282,
        ),
        label: [ResNeXt-50],
        mark: "+",
      ),
    )
  ]
]
#text(
  size: 8pt,
)[*ConvNeXt V2 and ViT plateaued near random-loss (~7.0); omitted for clarity*]

#v(3mm)
=== Training Loss Curves (Epochs 1--10)

#figure(
  caption: [Training Loss Comparison Across All CNN Models (Epochs 1--10)],
)[
  #align(center)[
    #lq.diagram(
      width: 12cm,
      height: 8cm,
      xlabel: [Epoch],
      ylabel: [Training Loss],
      xlim: (1, 10),
      ylim: (0, 8),
      legend: (position: top + right),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          6.9417,
          6.1782,
          4.6563,
          3.3380,
          2.4297,
          1.8331,
          1.4440,
          1.2069,
          1.0592,
          0.9937,
        ),
        label: [ResNet-50],
        mark: "o",
      ),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          7.0167,
          6.6803,
          6.2441,
          5.8060,
          5.4203,
          5.1210,
          4.8672,
          4.6387,
          4.4313,
          4.2441,
        ),
        label: [DenseNet-121 (Aug + TTA)],
        mark: "x",
      ),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          6.6601,
          5.6555,
          4.9103,
          4.3488,
          3.9275,
          3.5940,
          3.3287,
          3.1454,
          3.0048,
          2.9406,
        ),
        label: [EfficientNet-B0],
        mark: "s",
      ),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          6.9772,
          6.5127,
          5.8726,
          5.1724,
          4.5370,
          3.9398,
          3.3544,
          2.7914,
          2.2361,
          1.7423,
        ),
        label: [Chaotic (Skew-Tent)],
        mark: "^",
      ),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          7.0205,
          6.7422,
          6.2747,
          5.6682,
          4.9676,
          4.3126,
          3.6854,
          3.1021,
          2.5016,
          1.9386,
        ),
        label: [Chaotic (Logistic)],
        mark: "v",
      ),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          6.9896,
          6.4316,
          5.6728,
          4.8648,
          4.1446,
          3.5054,
          2.9111,
          2.3345,
          1.7819,
          1.3347,
        ),
        label: [Chaotic (Sine)],
        mark: "d",
      ),
      lq.plot(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (
          7.0385,
          6.7942,
          6.5252,
          6.2142,
          5.9072,
          5.5918,
          5.3152,
          5.0909,
          4.8703,
          4.6861,
        ),
        label: [ResNeXt-50],
        mark: "+",
      ),
    )
  ]
]
#text(
  size: 8pt,
)[*ResNeXt-50 shows higher training loss due to training on all 4 cell types*]

#v(4mm)DenseNet-121 (Aug + TTA)'s superior performance can be attributed to its dense connectivity, which encourages feature reuse and mitigates overfitting. When combined with augmentation and TTA on all four cell types, it achieved 47.42% validation accuracy. ResNet-50 appears to memorize training patterns more aggressively, as evidenced by the large divergence between training and validation accuracy.

EfficientNet-B0, trained on all four cell types, achieved 44.85%---competitive with DenseNet-121 despite the increased task complexity of multi-cell-type classification. Its efficient architecture and TTA-based submission strategy demonstrate that lightweight models can rival heavier architectures when combined with inference-time optimizations.


Future work will explore *dual-site* training (combining images from both imaging sites), *test-time augmentation* (TTA), and *metric-learning* heads such as ArcFace to further improve generalization across experimental batches.

#pagebreak()

= Conclusion
#v(5mm)
This study evaluated eleven deep learning architectures for siRNA classification on the RxRx1 dataset: ResNet-50, DenseNet-121 (with and without augmentation), EfficientNet-B0 and EfficientNet-B4, ConvNeXt V2 Tiny, Vision Transformer (ViT-Base/16), three Chaotic CNN variants (Skew-Tent, Logistic, Sine maps), Inception V3, and ResNeXt-50. DenseNet-121 (Aug + TTA) achieved a best validation accuracy of 47.42% across all cell types, significantly outperforming ResNet-50's 37.49%. EfficientNet-B0, trained on all four cell types, achieved 44.85% with an additional improvement via TTA-based submission. EfficientNet-B4 without augmentation severely overfit, reaching only 10.83%. ConvNeXt V2 Tiny and ViT failed to learn meaningful features, highlighting the importance of proper transfer learning adaptation for microscopy data. Inception V3 achieved 9.81% with TTA. The Chaotic CNN variants achieved moderate accuracies (30.56%--33.72%), while ResNeXt-50 achieved the best overall performance at 55.07% validation accuracy on all four cell types, demonstrating the effectiveness of aggregated residual transformations for large-scale biomedical image classification.

#pagebreak()
#bibliography("refs.bib", style: "ieee", title: [References])
