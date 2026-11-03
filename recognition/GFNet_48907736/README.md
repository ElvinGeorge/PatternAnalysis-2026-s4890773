# GFNet for Identifying Alzheimer's Disease in Brain MRIs

## The Algorithm
Alzheimer’s disease (AD) is a neurodegenerative condition characterised by difficulties with memory, fatigue, confusion and physical decline. When clinicians suspect AD, it is important that any brain MRIs produced are correctly interpreted. False positives lead to unnecessary testing and use of hospital resources, while false negatives lead to delayed treatment and care, and potentially a misdiagnosis.

The goal of this algorithm is to assist clinicians by being able to analyse brain scans and identify if someone has AD or is cognitively normal (CN). It will achieve this by being able to analyse a brain MRI and determine whether the scan indicates AD or CN, and a probability of it being AD (P(AD)). The algorithm is a GFNet and works in four stages. The first step is patch embedding. The 224x224 image is split into a 16x16 grid of non-overlapping patches. Each patch is projected onto a 256-dimensional vector to produce a 14x14 grid of tokens. The next step is global filtering. The 12 GFNet blocks normalise the token and apply a 2D fast Fourier transform (FFT) on the 14x14 grid. The result is multiplied by a learnable filter and then transformed back with the inverse FFT. The third step is channel mixing. Within each block, a two-layer MLP is applied to each token to mix information across feature channels. Finally, classification occurs. After the 12 blocks, the tokens are normalised and averaged into a single 256-dimensional vector. This is mapped to two logits, and SoftMax is used to create a probability.

![GFNet Pipeline](gfnet_pipeline.png)


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


## Dependencies 



## Example Inputs and Outputs and Plots


## Preprocessing and Data Split

## Investigation

### Benchmarking Against CNN

### Computational Efficiency

### Error Autopsies 

### Recomendations and Trade-Offs

## AI Disclosure 
