# Contextual Range-View Projection for 3D LiDAR Point Clouds

This repository contains the code, data processing scripts, and supplementary material for our paper:

**Contextual Range-View Projection for 3D LiDAR Point Clouds**  
*Seyedali Mousavi, Seyedhamidreza Mousavi, Masoud Daneshtalab*  
Mälardalen University

---

## 📖 Overview
Range-view projection converts 3D LiDAR point clouds into 2D image-like representations, enabling efficient processing with 2D CNNs.  
However, a key challenge is the **many-to-one conflict**, where multiple 3D points are mapped to the same pixel. Standard approaches simply keep the closest point (smallest depth), which ignores object structure and semantic relevance.

In this work, we propose two improvements to the projection stage:

- **Centerness-Aware Projection (CAP)**  
  Prioritizes points closer to the geometric center of object instances (“things” such as cars, bicycles, pedestrians) over boundary or noisy points.  

- **Class-Aware Projection**  
  Assigns weights to semantic classes, allowing task-relevant categories to be emphasized while background (“stuff”) can be down-weighted.  

These strategies directly refine range-view projections while remaining compatible with existing LiDAR pipelines. On SemanticKITTI, our approach improves semantic segmentation performance by up to **+1.8% mIoU** on instance-level classes.

---

## 📂 Repository Structure
