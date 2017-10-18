# Deep-Learning-On-Medical-Image
Paper LIst about Deep Learning on Medical Image

## Brain Tumor Segmentation 

### Review

- [x] [Multimodal Brain MRI Tumor Segmentation via Convolutional Neural Networks]()  (2017,   **\*\*\***)

> Glioma are the most common family of brain tumors, with a subset of glioma known as glioblastoma forming the most common and some of the highest-mortality and economically costly forms of brain cancer. Patients are diagnosed based on manual segmentation and analysis of multimodal MRI scans, but due to the labor-intensive nature of the manual segmentation process and mistakes or disagreement between manual segmentations, there exists a need for a fast and robust automated segmentation algorithm. Convolutional neural networks (CNNs) have been shown to be extremely effective for a variety of visual recognition and semantic segmentation tasks. Here, we present three
> novel CNN-based architectures for glioma segmentation for images from the MICCAI BraTS Challenge dataset. We also explore transfer learning between the BraTS dataset and other neuroimaging datasets by applying models pretrained on the BraTS dataset to segmenting images from the Rembrandt dataset. Our results show that patch-wise approaches trained on a balanced training set of tumor and non-tumor patches delivers strong segmentation results with mean dice score of 0.86. The results from transfer learning show that applying models pre-trained on the BraTS dataset to other neuroimaging datasets is promising but requires further work.

- [ ] [The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)](http://ieeexplore.ieee.org/abstract/document/6975210/)  (2015, **\*\*\*\*\***)

> In this paper we report the set-up and results of the Multimodal Brain Tumor Image Segmentation Benchmark (BRATS) organized in conjunction with the MICCAI 2012 and 2013 conferences. Twenty state-of-the-art tumor segmentation algorithms were applied to a set of 65 multi-contrast MR scans of low- and high-grade glioma patients - manually annotated by up to four raters - and to 65 comparable scans generated using tumor image simulation software. Quantitative evaluations revealed considerable disagreement between the human raters in segmenting various tumor sub-regions (Dice scores in the range 74%-85%), illustrating the difficulty of this task. We found that different algorithms worked best for different sub-regions (reaching performance comparable to human inter-rater variability), but that no single algorithm ranked in the top for all sub-regions simultaneously. Fusing several good algorithms using a hierarchical majority vote yielded segmentations that consistently ranked above all individual algorithms, indicating remaining opportunities for further methodological improvements. The BRATS image data and manual annotations continue to be publicly available through an online evaluation system as an ongoing benchmarking resource.

- [ ] [Glioma Dynamics and Computational Models: A Review of Segmentation, Registration, and In Silico Growth Algorithms and their Clinical Applications](http://www.ingentaconnect.com/content/ben/cmir/2007/00000003/00000004/art00007)  (2007, **\*\***)

> Tracking gliomas dynamics on MRI has became more and more important for therapeutic management. Powerful computational tools have been recently developed in this context enabling in silico growth on a virtual brain that can be matched with real 3D segmented evolution through registration between atlases and patient brain MRI data. In this paper, we provide an extensive review of existing algorithms for the three computational tasks involved in patient-specific tumor modeling: image segmentation, image registration, and in silico growth modelling (with special emphasis on the proliferation-diffusion model). Accuracy and limits of the reviewed algorithms are systematically discussed. Finally applications of these methods for both clinical practice and fundamental research are also discussed. 

- [ ] [A survey of MRI-based medical image analysis for brain tumor studies](https://www.researchgate.net/publication/237070108_A_survey_of_MRI-based_medical_image_analysis_for_brain_tumor_studies) (2013, **\*\*\***) 

>MRI-based medical image analysis for brain tumor studies is gaining attention in recent times due to an increased need for efficient and objective evaluation of large amounts of data. While the pioneering approaches applying automated methods for the analysis of brain tumor images date back almost two decades, the current methods are becoming more mature and coming closer to routine clinical application. This review aims to provide a comprehensive overview by giving a brief introduction to brain tumors and imaging of brain tumors first. Then, we review the state of the art in segmentation, registration and modeling related to tumor-bearing brain images with a focus on gliomas. The objective in the segmentation is outlining the tumor including its sub-compartments and surrounding tissues, while the main challenge in registration and modeling is the handling of morphological changes caused by the tumor. The qualities of different approaches are discussed with a focus on methods that can be applied on standard clinical imaging protocols. Finally, a critical assessment of the current state is performed and future developments and trends are addressed, giving special attention to recent developments in radiological tumor assessment guidelines. 

### Generative Models

- [ ] K. Van Leemput, F. Maes, D. Vandermeulen, P. Suetens, "Automated model-based bias field correction of MR images of the brain", *IEEE Trans. Med. Imag.*, vol. 18, no. 10, pp. 885-896, Oct. 1999.  
- [ ] M. R. Kaus, S. K. Warfield, A. Nabavi, P. M. Black, F. A. Jolesz, R. Kikinis, "Automated segmentation of MR images of brain tumors", *Radiology*, vol. 218, no. 2, pp. 586-591, Feb. 2001.
- [ ] M. Prastawa, E. Bullitt, S. Ho, G. Gerig, "A brain tumor segmentation framework based on outlier detection", *Med. Image Anal.*, vol. 8, pp. 275-283, 2004.
- [ ] K. M. Pohl, J. Fisher, J. J. Levitt, M. E. Shenton, R. Kikinis, W. E. L. Grimson, W. M. Wells, "A unifying approach to registration segmentation and intensity correction", *Proc. MICCAI*, pp. 310-318, 2005
- [ ] F. O. Kaster, B. H. Menze, M.-A. Weber, F. A. Hamprecht, "Comparative validation of graphical models for learning tumor segmentations from noisy manual annotations", *Proc. MICCAI-MCV*, 2010.
- [ ] B. Fischl, "Whole brain segmentation: Automated labeling of neuroanatomical structures in the human brain", *Neuron.*, vol. 33, no. 3, pp. 341-355, 2002
- [ ] J. Ashburner, K. J. Friston, "Unified segmentation", *Neuroimage*, vol. 26, no. 3, pp. 839-851, 2005.
- [ ] A. Gooya, "GLISTR: Glioma image segmentation and registration", *IEEE Trans. Med. Imag.*, vol. 31, no. 10, pp. 1941-1954, Oct. 2012.

### Discriminative Models

- [ ] [D. Cobzas, N. Birkbeck, M. Schmidt, M. Jagersand, A. Murtha, "3D variational brain tumor segmentation using a high dimensional feature set", *Proc. ICCV*, pp. 1-8, 2007.](http://ieeexplore.ieee.org/document/4409130/)  (2007, **\*\***)
- [ ] [S. Ho, E. Bullitt, G. Gerig, "Level-set evolution with region competition: Automatic 3D segmentation of brain tumors", Proc. ICPR, pp. 532-535, 2002.](http://ieeexplore.ieee.org/document/1044788) (2002, **\*\***)
- [ ] [C. Lee, S. Wang, A. Murtha, R. Greiner, "Segmenting brain tumors using pseudo conditional random fields", Proc. MICCAI, pp. 359-366, 2008.](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwiMt6KjuvTWAhVFYo8KHQefAVAQFgguMAE&url=https%3A%2F%2Fwebdocs.cs.ualberta.ca%2F~btap%2FPapers%2FChihoon_MICCAI_2008.pdf&usg=AOvVaw0p2lAIDa7wjKiGVZqIRGAo)  (2008, **\*\***)
- [ ] [S. Bauer, L.-P. Nolte, M. Reyes, "Fully automatic segmentation of brain tumor images using support vector machine classification in combination with hierarchical conditional random field regularization", Proc. MICCAI, pp. 354-361, 2011.](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0ahUKEwjwjr6nu_TWAhUFtI8KHf8VAbQQFggxMAI&url=http%3A%2F%2Fftp.mauricioreyes.me%2FPublications%2FBauerMiccai2011.pdf&usg=AOvVaw3GVzM3HAks9pcF6nZGMVI2) (2011, **\*\***)
- [ ] [W. Wu, A. Y. Chen, L. Zhao, J. J. Corso, "Brain tumor detection and segmentation in a conditional random fields framework with pixel-pairwise affinity and superpixel-level features", Int. J. Comput. Assist. Radiol. Surg., pp. 1-13, 2013.]()  (2013, **\*\***)

### CNN-Based Models

- [ ] [Fully Convolutional Networks for Semantic Segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html) (2015, **\*\*\*\***)

> Convolutional networks are powerful visual models that yield hierarchies of features. We show that convolutional networks by themselves, trained end-to-end, pixels-to-pixels, exceed the state-of-the-art in semantic segmentation. Our key insight is to build "fully convolutional" networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning. We define and detail the space of fully convolutional networks, explain their application to spatially dense prediction tasks, and draw connections to prior models. We adapt contemporary classification networks (AlexNet, the VGG net, and GoogLeNet) into fully convolutional networks and transfer their learned representations by fine-tuning to the segmentation task. We then define a skip architecture that combines semantic information from a deep, coarse layer with appearance information from a shallow, fine layer to produce accurate and detailed segmentations. Our fully convolutional network achieves state-of-the-art segmentation of PASCAL VOC (20% relative improvement to 62.2% mean IU on 2012), NYUDv2, and SIFT Flow, while inference takes less than one fifth of a second for a typical image.

## MICCAI 2017

- [ ] [Transfer Learning for Domain Adaptation in MRI: Application in Brain Lesion Segmentation](http://arxiv.org/abs/1702.07841v1) (2017, **\*\*\*\***)

> Magnetic Resonance Imaging (MRI) is widely used in routine clinical diagnosis and treatment. However, variations in MRI acquisition protocols result in different appearances of normal and diseased tissue in the images. Convolutional neural networks (CNNs), which have shown to be successful in many medical image analysis tasks, are typically sensitive to the variations in imaging protocols. Therefore, in many cases, networks trained on data acquired with one MRI protocol, do not perform satisfactorily on data acquired with different protocols. This limits the use of models trained with large annotated legacy datasets on a new dataset with a different domain which is often a recurring situation in clinical settings. In this study, we aim to answer the following central questions regarding domain adaptation in medical image analysis: Given a fitted legacy model, 1) How much data from the new domain is required for a decent adaptation of the original network?; and, 2) What portion of the pre-trained model parameters should be retrained given a certain number of the new domain training samples? To address these questions, we conducted extensive experiments in white matter hyperintensity segmentation task. We trained a CNN on legacy MR images of brain and evaluated the performance of the domain-adapted network on the same task with images from a different domain. We then compared the performance of the model to the surrogate scenarios where either the same trained network is used or a new network is trained from scratch on the new dataset.The domain-adapted network tuned only by two training examples achieved a Dice score of 0.63 substantially outperforming a similar network trained on the same set of examples from scratch.

## Others



- [x] [Low-dose CT denoising with convolutional neural network](http://arxiv.org/abs/1610.00321v1)  

> To reduce the potential radiation risk, low-dose CT has attracted much attention. However, simply lowering the radiation dose will lead to significant deterioration of the image quality. In this paper, we propose a noise reduction method for low-dose CT via deep neural network without accessing original projection data. A deep convolutional neural network is trained to transform low-dose CT images towards normal-dose CT images, patch by patch. Visual and quantitative evaluation demonstrates a competing performance of the proposed method.

