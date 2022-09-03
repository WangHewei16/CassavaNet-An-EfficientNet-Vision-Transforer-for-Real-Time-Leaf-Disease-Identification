## EfficientNet-Vit-Model-for-Cassava-Leaf-Disease-Classification

### 1. Background of the problem to be solved
Cassava, Africa's second-largest provider of carbohydrates, is a critical food security crop grown by smallholder farmers due to its ability to withstand harsh conditions. This starchy root is grown on at least 80% of Sub-Saharan African household farms, but viral diseases are a major source of low yields. It may be possible to identify common diseases and treat them using data science.


Existing disease detection methods necessitate farmers enlisting the assistance of government-funded agricultural experts to visually inspect and diagnose the plants. This suffers from being labor-intensive, scarce, and expensive. As an added challenge, effective solutions for farmers must perform well under important constraints, as African farmers might only have access to low-bandwidth mobile-quality cameras.


The dataset for this competition consists of 21,367 labeled images collected during a regular survey in Uganda. The majority of the images were crowdsourced from farmers who took photos of their gardens and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with Makerere University's AI lab in Kampala. This is in a format that most closely resembles what farmers would need to diagnose in the field. Our task is to categorize each cassava image into one of four disease categories or one healthy leaf category. To assist farmers in quickly identifying diseased plants, potentially saving their crops before irreversible damage occurs.

### 2. Pipeline
This problem is a single-label image classification problem with large differences in the amount of data from various categories and high data noise. Designed pipeline is shown in the figure below, Use resize, crop, flip, normalize and other preprocessing methods, and then input into two backbones: `Vit` and `EfficientNet`. Adapt `nn.CrossEntropyLoss()` as loss function, using the `LabelSmoothing` anti-noise technique, and choosing a different learning rate strategy for these two backbones. Lastly, do a simple ensemble such as `tst_preds = 0.452*tst_preds_vit + 0.548*test_preds_eff`.
<div align=center><img src="https://github.com/WangHewei16/EfficientNet-Vit-Model-for-Cassava-Leaf-Disease-Classification/blob/main/images/pipeline.png" width="950"/></div>


### 3. BackBone
#### 3.1 EfficientNet
The figure below shows the architecture of EfficientNet. [[Paper Link](https://arxiv.org/pdf/1905.11946.pdf)]
<div align=center><img src="https://github.com/WangHewei16/EfficientNet-Vit-Model-for-Cassava-Leaf-Disease-Classification/blob/main/images/EfficientNet%20diagram.png" width="700"/></div>

#### 3.2 Vision Transformer (Vit)
The figure below shows the architecture of EfficientNet. Converting images to sequence into Transformer. [[Paper Link](https://arxiv.org/pdf/2010.11929.pdf)]
<div align=center><img src="https://github.com/WangHewei16/EfficientNet-Vit-Model-for-Cassava-Leaf-Disease-Classification/blob/main/images/Vit%20diagram.png" width="590"/></div>

### 4. Learning rate strategy
Use `Cosine Annealing` strategy for EfficientNet backbone and adapt `ReduceLROnPlateau` strategy for Vit backbone.
<div align=center><img src="https://github.com/WangHewei16/EfficientNet-Vit-Model-for-Cassava-Leaf-Disease-Classification/blob/main/images/learning%20rate%20strategy.png" width="600"/></div>


### 5. K-Fold cross validation skill
Implement K-Fold Cross Validation for each model to improve respective and ensemble effect.

<div align=center><img src="https://github.com/WangHewei16/EfficientNet-Vit-Model-for-Cassava-Leaf-Disease-Classification/blob/main/images/k-fold%20cross%20validation.png" width="450"/></div>
