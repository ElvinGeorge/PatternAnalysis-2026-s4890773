# GFNet for Identifying Alzheimer's Disease in Brain MRIs

## The Algorithm
Alzheimer’s disease (AD) is a neurodegenerative condition characterised by difficulties with memory, fatigue, confusion and physical decline. When clinicians suspect AD, it is important that any brain MRIs produced are correctly interpreted. False positives lead to unnecessary testing and use of hospital resources, while false negatives lead to delayed treatment and care, and potentially a misdiagnosis.

The goal of this algorithm is to assist clinicians by being able to analyse brain scans and identify if someone has AD or is cognitively normal (CN). It will achieve this by being able to analyse a brain MRI and determine whether the scan indicates AD or CN, and a probability of it being AD (P(AD)). The algorithm is a GFNet and works in four stages. The first step is patch embedding. The 224x224 image is split into a 16x16 grid of non-overlapping patches. Each patch is projected onto a 256-dimensional vector to produce a 14x14 grid of tokens. The next step is global filtering. The 12 GFNet blocks normalise the token and apply a 2D fast Fourier transform (FFT) on the 14x14 grid. The result is multiplied by a learnable filter and then transformed back with the inverse FFT. The third step is channel mixing. Within each block, a two-layer MLP is applied to each token to mix information across feature channels. Finally, classification occurs. After the 12 blocks, the tokens are normalised and averaged into a single 256-dimensional vector. This is mapped to two logits, and SoftMax is used to create a probability.

![GFNet Pipeline](README_images\gfnet_pipeline.png)


## Feasibility Review
### User Need, Scope & Acceptance Criteria
The intended users are clinicians who can use this tool for automated screening to improve productivity and reduce diagnostic errors or misses. The scope will be limited to Alzheimer’s Disease and a positive/negative classification with confidence, using just MRI scans.  

-	Ambiguous cases: Confidence scores (A probability of AD) should be reliably produced for all MRIs. This confidence score should correlate with diagnostic correctness.

-	Handling: For all high confidence negatives, a negative case should be reported. For medium or high confidence positives, a positive case should be reported, and further testing commence. For any medium confidence negatives, or any low confidence result, there should be a human review process.

-	False negatives: Maximising recall is the primary. focus. This is because a false negative leads to undetected AD, whilst a false positive leads to an unneeded test. The model will have be considered successful if it can achieve a recall greater than 80%.

-	Overall positives: Although maximising recall is the primary focus, false positives lead to unnecessary health scares and biopsies, causing strain on the healthcare system. Thus, it is important the specificity is greater than 80%.

-	Computational limits: To process a significant number of images and automate this, it is important the inference time is minimised. The model is considered successful if it can achieve a median inference time of 10ms or less, per image, on a standard A100 GPU. 

### Model Choice & Course Concepts
The data will be split by patients. This is important to prevent data leakage and inflated accuracy scores. It will then be split into approximately 70% training , 15% testing and 15% validation. These percentages were chosen since ADNI is a large dataset (>10 000 entries) and this is a recommended split, and CV is not feasible due to training times.
GFNet will be chosen for the model, with a CNN as the baseline. GFNets  have been used before AD classification, most notably by Xu et.al. in 2025. With MRI AD vs CN, they achieved an accuracy of 92.53% and were able to improve this to 98.91% by using a modified GFNet that also incorporated PET scans. This study serves as proof that GFNets, even with just MRI data, can detect AD at a reasonable accuracy rate. 
A typical CNN will be chosen for the baseline. This is since this task is fundamentally computer vision, where a model is looking for signs of AD such as progressive hippocampal and cortical atrophy with ventricular enlargement.

### Preliminary Feasibility Evidence
This is feasible and has been done before as stated in the previous section. The data itself is not split by patient yet, and this will have to be done to prevent data leakage. With an A100 and using the ADNI dataset, a GFNet is estimated to take less than 3hrs to train, while a CNN would be at most 30m (estimates from previous runs I have done). I have access to an A100 so there are no issues with resources.
Risks, Budget & Fallback
I have 80 computing credits with Google Collab, and also free access to UQ’s Rangpur Cluster. This will give me access to A100s and other GPUs to train on. The primary risk of either model is data leakage due to patients not being kept together on datasets, leading to overconfidence and inflated accuracy scores. If GFNets do not work, I can try a ConvNeXt instead.

## Dependencies and Reproducibility
### Dependencies
| Package      | Version |
|--------------|---------|
| Python       | 3.13.16 |
| torch        | 2.11.0 (CUDA 13.0) |
| torchvision  | 0.26.0  |
| scikit-learn | 1.6.1   |
| matplotlib   | 3.10.0  |
| Pillow       | 11.3.0  |
| pandas       | 2.2.3   |

### Reproducibility
The hardware used to run the models was a NVIDIA A100-SXM4-40GB. The hardware is important when it comes to reproducing model training and inference times.\
The test train val split is reproducible since it was done with randomstate=42. Training however was not seeded, meaning that model to model variation will exist. In my testing, I have run the GFNet model 3 times on 100 epochs, and the accuracy between them was fairly consistent, ranging between 0.867 and 0.899.\
As a whole, if the scripts were run again on any GPU, the results achieved are highly reproducible (aside for inference time unless it were run on a similarly powerful GPU).
## Example Inputs and Outputs and Plots
### True Positive Case:
The input was a AD brain MRI\
**IMAGE ID:** 1005735_104\
![True Positive](README_images\example-input-true-pos-1005735_104.jpeg)\
**Prediction made:** AD\
**P(AD):** 0.958\
There were 1994 images that were true postives (43.35%)

### True Negative Case:
The input was a CN brain MRI\
**IMAGE ID:** 998432_107\
![True Negative](README_images\example-input-true-neg-998432_107.jpeg)\
**Prediction made:** CN\
**P(AD):** 0.006\
There were 2048 images that were true negatives (44.52%)

### False Positive Case:
The input was a CN brain MRI\
**IMAGE ID:** 1084935_110\
![False Positive](README_images\example-input-false-pos-1084935_110.jpeg)\
**Prediction made:** AD\
**P(AD):** 0.533\
There were 312 images that were false positives (6.78%)

### False Negative Case:
The input was a CN brain MRI\
**IMAGE ID:** 373618_86\
![False Negative](README_images\example-input-false-negative-373618_86.jpeg)\
**Prediction made:** CN\
**P(AD):** 0.008\
There were 246 images that were false negatives (5.35%)

### Confusion Matrix 
![Confusion Matrix](..\GFNet_48907736\GFNet\outputs\GFNet_confusion_matrix.png)\
As stated above, there were 1994 images that were true postives (43.35%), 2048 images that were true negatives (44.52%), 312 images that were false positives (6.78%) and 246 images that were false negatives (5.35%).


### Accuracy Curve
![Confusion Matrix](..\GFNet_48907736\GFNet\outputs\GFNet_accuracy_curve.png)\
By the end of training, the model reached a validation accuracy of 87.2%, and a training accuracy of 98.9%.

### ROC Curve
![ROC Curve](..\GFNet_48907736\GFNet\outputs\GFNet_roc_curve.png)\
By the end of training, the model reached a AUROC of accuracy of 94.5%.


## Preprocessing and Data Split
### Dataset
The dataset was provided by the course on the /home/groups/comp3710/ADNI. Within the folder, there was a test and train folder, and within each folder two classes, AD and CN. Files are named patientID_sliceNumber.jpeg.\
There were 14860 images with AD, and 15660 that were CN. In total, there were 30520 images. There was no major class imbalance, with AD making up 48.69% of images. There were 1526 patients, with 20 images per patient.

### Pre-Processing
The AD images were pooled and so were the CN images. After that, all the images in either set were grouped by patient and then randomly split between test train and validation sets. This produced six groups of images.\
The images were then converted to grayscale (Grayscale(1)), scaled to 244x244 (Resize((224, 224))) and finally converted to tensors and normalised (ToTensor() + Normalize([0.5], [0.5])). This was done to all images to ensure consistency amongst colour, size and scale.\
For training, RandomHorizontalFlip() and RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)) was also applied. The reason random horizontal flips and affine transformations were applied was to reduce the chance the model was just memorising the inputs and subsequently improve generalisation. The effect of these transformations were tested when I ran a model without them and only reached 0.802 accuracy and 0.882 AUROC, compared to 0.879 accuracy and 0.945 AUROC with them. This further justified the use of these transformations.

### Data Split
The data was split as 70% training, 15% validation and 15% testing. Splits were done by patient since brainscans from the same patient are typically similar. This meant if a patient's brainscan was in the testing and training set, then the final test accuracy will be skewed. The split 70 / 15 / 15 was chosen under advice from a tutor and by this [article](https://medium.com/@kiranvutukuri/19-model-evaluation-basics-train-validation-and-test-splits-ed5bb26b088d), which stated for large dataset (>10000 items), it is better to have a larger test and validation set since the training set has enough items already. 

## Investigation
### Benchmarking Against CNN
Both models were evaluated on the same test set of 4600 images (AD: 2240, CN: 2360). Both models were also trained on the same set of 21360 images (AD: 10400, CN: 10960).

| Metric                     | CNN baseline | GFNet        
|----------------------------|--------------|-------------
| Accuracy                   | 0.838        | **0.879**    
| Macro-F1                   | 0.838        | **0.879**    
| Recall (AD)                | 0.838        | **0.890**    
| Specificity                | 0.839        | **0.868**    
| AUROC                      | 0.910        | **0.945**    

The GFNet outperforms the CNN on every metric. It improves accuracy by 4.1 percentage points, from 83.8% to 87.9% and AUROC by 0.035, from 0.910 to 0.945. Recall and specificity both improve from 83.8% and 83.9% to 87.9% and 89.0%, respectively. Overall, it is evident the GFNet is superior in classifying AD vs CN brain MRIs. This difference in performance is very valuable since the increased specificity and recall mean less missed diagnosis from false negatives, and less wastage of hospital resources from false positives.

### Computational Efficiency
Models were trained and evaluated on a Nvidia A100 40GB GPU.

| Metric                       | CNN baseline              | GFNet                 
|------------------------------|---------------------------|----------
| Parameter count              | **389,314**               | 7,124,226
| Peak GPU VRAM (batch of 64)  | 846.1 MB                  | **194.3 MB**
| Median inference time        | **0.83 ms** per image     | 5.73 ms per image
| Input size                   | 1 × 224 × 224 (greyscale) | 1 × 224 × 224 (greyscale)

The GFNet has 7 124 226 parameters compared to only 389 314 on the baseline CNN. On a Nvidia A100 GPU, the GFNet takes 5.73ms per slice, meaning its max throughput would be around 175 images per second. This is more than fast enough, as at 20 slices per patient, this is 524 patients a minute. Although the CNN is significantly faster at 0.83ms per image, this extra speed is not necessary. The peak VRAM usage during inference is 194.3MB for the GFNet and 846.1MB for the CNN (with a batch of 64). Although this means that both models can run on consumer grade GPUs easily and not hit memory limits, the GFNet is significantly more efficient. This low memory usage is likely since the model converts the image into a 14x14 token grid, reducing the space needed. To summarise, both models are computationally efficient, with the CNN holding the advantage with less parameters, while the GFNet requires less VRAM during inference.

### GFNet Error Autopsies 
#### High Confidence False Positive
**IMAGE ID**: 1177833_94\
![High Confidence False Positive](README_images\GFNet_heatmap_1177833_94.png)\
**Truth:** CN\
**Prediction:** AD\
**P(AD):** 0.998

NEED TO WRITE HERE 


#### Medium Confidence False Positive
**IMAGE ID**: 921879_91\
![Medium Confidence False Positive](README_images\GFNet_heatmap_921879_91.png)\
**Truth:** CN\
**Prediction:** AD\
**P(AD):** 0.712

NEED TO WRITE HERE 


#### Medium Confidence False Negative
**IMAGE ID**: 1343923_105\
![Medium Confidence False Negative](README_images\GFNet_heatmap_1343923_105.png)\
**Truth:** AD\
**Prediction:** CN\
**P(AD):** 0.243

NEED TO WRITE HERE


#### High Confidence False Negative
**IMAGE ID**: 373618_88\
![High Confidence False Negative](README_images\GFNet_heatmap_373618_88.png)\
**Truth:** AD\
**Prediction:** CN\
**P(AD):** 0.003

NEED TO WRITE HERE 


### Recomendations and Trade-Offs
**Accuracy vs Cost:** Although the GFNet has an accuracy 4.1 percentage points higher (from 83.8% to 87.9%), this comes at the cost of ~18x more parameters and ~7x the inference time. Despite this, the GFNet still remains the better option. This is because with an inference time 5.73ms per slice, it could process 524 patients a minute which is more than enough. Additionally, the increase in specificity from 83.9% to 89.0% is significant since it directly means there are less false postives and a reduction in time and resources spent on unnecessary testing. The increase in recall from 83.8% to 87.9% also means that there is a reduction in false negatives, meaning less patients with AD are missed.

**Decision Threshold:** The decision threshold is currently set as P(AD) > 0.5. This is a tradeoff since by decreasing it, although there will be less false negatives, there will be more false positives. False negatives, however, are far less impactful then false postives since they just lead to extra testing (as compared to a missed diagnosis). Thus, the decision threshold should be lowered. The high AUROC of 0.945 supports this trade-off, so the threshold should be reduced based on the resources available to handle false positives.

**Slice Level Prediction:** The model classifies per slice, however diagnosis happen at the per patient level. P(AD) should be averaged across all 20 slices per patient to determine the diagnosis. This would produce more stable and reliable predictions.

**Incorporating PET Scans:** The study by [Xu et.al. in 2025](https://dl.acm.org/doi/full/10.1145/3715931.3715943) improved accuracy with CN vs AD from 92.53% with just MRI scans, to 98.91% with a modified GFNet that uses PET scans and MRI scans. By modifying my model to also use PET scan data, the accuracy, recall and specificity can all likely be improved. 

## AI Disclosure 
Claude Opus 5.5 was used to generate gfnet_pipeline.png\
I did this by giving it modules.py for my GFNet and asking it to generate a flow diagram that accurately outlines the model architecture.\
I verified that the output image was correct by checking the diagram against the model structure I have made.

Claude Opus 5.5 was used to generate heatmap.py\
I did this by giving into my GFNet structure and asking it to generate a script that can take an image and model file, and output a two heatmaps (Grad-CAM and a gradient x input).\
I have verified this heatmap is correct by going over the code, and then also checking heatmaps to see if there are any obvious issues. I also modified the code to remove it from printing P(AD) and labels on the image since they will be written underneath in the report.