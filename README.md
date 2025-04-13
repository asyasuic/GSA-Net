# Balanced Feature Representation in GSA-Net: A Lightweight Deep Learning Model for Accurate Medical Image Segmentation
The Visual Computer
Medical image segmentation is crucial for modern medical diagnosis, yet traditional manual methods are time-consuming and inefficient. This paper proposes GSA-Net, an innovative lightweight deep learning model tailored for this task. GSA-Net introduces a novel symmetric encoder-decoder structure with balanced stage computation ratios (1:1:3:1) to enhance multi-scale feature representation while maintaining structural balance. The model incorporates the Dilated Wide-field Ghost Attention (DWGA) module, which combines dilated convolution's expanded receptive field with Ghost module's parameter efficiency and coordinate attention's positional sensitivity. Furthermore, GSA-Net employs complementary lightweight components, including an Expansion-Compression (EC) bridge for feature refinement and a Lightweight Multi-scale Feature Aggregation (LMFA) module for context integration. Comprehensive evaluations on Kvasir-SEG, ISIC2018, and GLaS benchmarks demonstrate state-of-the-art performance, achieving Dice scores of 90.0%, 90.4%, and 91.6%, respectively, while reducing parameters by 37% and FLOPs by 48.6% compared to ConvUNeXt. The model's exceptional accuracy-efficiency trade-off and robust generalization ability make it highly suitable for clinical deployment in resource-constrained environm![Figure](https://github.com/user-attachments/assets/e0c46cf2-cd0c-4c19-9d0e-9c2ceb63ad13)

