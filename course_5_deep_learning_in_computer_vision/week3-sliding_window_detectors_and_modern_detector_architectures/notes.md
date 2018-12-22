# Week 3 notes


## Sliding Windows

### Object detection problem
- Things - Certain size and shape
- Stuff - Uniform texture, no certain shape or size
- Bounding box (x,y,w,h) and class label
- IoU = Area of overlap / Area of union
   - Hit if: Correct label and IoU over certain threshold
   - IoU > p: True positive
   - IoU < p: False positive
   - Missed ground truth: False negative
- AP = \frac{1}{11}\sum_{r\in\{0, 0.1, \ldots 1\}} p(r)
    - mAP = mean AP (over all classes)
- Miss rate vs false positives per image (also found by varying the threshold)
- Important with annotation protocol

### Sliding window
   - Problem: Various size objects and various aspect ration
   - Scan image various size and aspect ratio
   - Vary image size, keep scanning box fixed
   - Images scanned = Number of scales X Number of aspect ratios
   - Select local maxima in case of multiple responses

### HoG - Histogram of Gradient
- Keep information about gradient (as apposed to Canny edges, where only keep magnitude)
- Information about weak gradients are kept
- Frequency vs orientation histogram for cells in image
   - Cells are concatenated into blocks and normalised
   - Linear SVM for person/non-person classifier

### Detector training
- Objects << non-objects -> Skew dataset
- Add random negative examples for non-objects
    - Search for high-confidence false detections, add this as a hard negative example
    - Retrain and repeat
    - Should be mined, but should not rely on them entirely

### Viola-Jones face
- Pre deep-learning
- 1 MP image -> classify up to ~1M windows, less than 1 fp in the image -> fp rate lower than 10^-6
- Need to filter windows without faces
- Should be real-time
- Viola-Jones
    - Haar features (apply special filters)
    - Integral images: At one pixel: Sum of pixel values above and to the left
    - Sum within rectangle of original image uses only 3 additions from integral image
    - Feature selection: Can get a lot of rectangle features from 24 x 24 image, instead find good features with linear combination with boosting
        - Boosting: Weak learners for features O(Rounds X Examples X Features)
        - 2 features can give 50 % detection rate
        - 200 can give 98 %
    - Attentional cascade:
        - Simple classifiers rejects sub-windows, positive triggers evaluation of next set of classifiers and so on
        - A false at any level is automatically rejected
        - Expensive classifiers used in the end
 - 15 Hz detection in 2001

### Attention cascade
- CNN are very slow, but can be used in last classifier
- Can replace all classifiers with NN, but have to be smart for real time detection
    - Bounding box can be used between stages, and will be input to next classifier (will be a prediction of b-box transformation)
    - Non-max suppression


## Modern Architectures

### Region based CNN (R-CNN)
- Select proposals, then apply strong CNN to proposals, classification with for example SVMs
- Selective search: High recall, but low precision (NOT NN)
- Training
    - Pre-train CNN on large dataset for image classification
    - Fine-tune CNN for object detection
    - Train linear prediction: CNN-features to SVM
- Disadvantages
    - Redundant computations
    - Need for rescale to fixed resolution
    - External algorithm for hypothesis
    - Complicated training with high file system load

### Fast R-CNN to Faster R-CNN
- Region of interest needs to fit into input of classifier (e.g. 244x244), so need to crop and rescale -> degrading image quality
- Spatial Pyramid Pooling (SPP)
    - Changing last pooling layer
    - Bag of visual words
        - Region of interest
        - Region of interest divided into 4 cells
        - Region of interest divided into 16 cells
        - Concatenated (all cells have the same number of maps)
- Features are extracted after SPP, leads to 160x faster than R-CNN
- Softmax instead of SVM
- Multi-task loss: Classification, b-box regression
- Faster to train, and around 3 Hz detection with 68,1% mAP on VOC

### Faster R-CNN
- Fast R-CNN + Regional Proposal Netwrok
   - No external hypothesis generation methon
   - Single NN
- RPN
    - Slide window on feature map
    - Window position provides localisation info
    - B-box provides finer localisation
    - At each sliding widow position
        - Set of k-object proposals with different size and aspect ratio (anchors)
        - B-box with reference to anchors
        - Positive if anchor with IoU>0.7, negative if IoU<0.3
    - Hard for very different scales (can make branches for each scale)

### Region based Fully Convolution Network
- Faster R-CNN
    - Many overlapping hypotheses
    - Features in box classifier computed independently for each box
    - Region component for each proposal
- R-FCN
    - Box classifiers precomputed for whole image
    - Crops are taken from last layer prior to prediction
    - FC layers are replaced with CNN layers prior to last CNN layer
    - Weakly dependent on number of proposals

### Single shot detectors
- One stage detectors, no proposal generating network
- YOLO:
   - Split image into grid
   - Each cell predict 2 boxes with confidence at cell center
   - All predictions from all cells are combined
   - Probability for each cell is calculated
   - Non-maximum suppression
   - Fixed output size
- R-CNN: 20 s/im, Fast R-CNN: 2s/img, Faster R-CNN: 140 ms/im, YOLO: 22ms/im, but 1 % lower mAP
- SSD extension of YOLO, and outperforms YOLO

### Speed vs accuracy
- Most accurate: Faster R-CNN (with certain hyper parameters

