# Week 2 notes

## Image classification
- Image classification - assign input a label
- Humans able to recognise objects in low res great accuracy quickly
- One often calls the CNN part the feature extractor and the fully connected part the classifier
- Formula for new height and widths of new layers (search for Summary): http://cs231n.github.io/convolutional-networks/
- NOTE: If we apply a 5x5 filter, the real size of the filter is 5x5xprevious_depth [serach for "Local Connectivity", "Numpy examples" and "Convolution Demo" in the link above]. This will give 5x5x(previous_depth + 1) trainable parameters (the +1 is the bias, one for each filter). This means that one applied filter corresponds one 2 dimensional output matrix
- LeNet5 (not shown on slide, but max pool has a stride of 2) 400 dimensional feature vector prior to last convolution layer which have depth of 1
- AlexNet 60 million parameters
- VGG 138 million Parameters - two stacked 3x3 conv layers without pooling => higher receptive field, more linearities, but less parameters
- Inception v3 25 million parameters(?) - computational efficiency in mind
- Inception block: Reduce computational complexity and efficient use of local image structure. 4 different feature maps are concatenated on depth in the end. Block stride of 1 and enough padding to get same WxH
- Can replace 5x5 conv with 2 stacked 3x3 filters (same receptive field). This is computationally more efficient
- Further 3x3 filters can be replaced with 1x3 followed by 3x1 (decomposing of filters)
- Cannot stack more layers forever - need blocks, batchNorm layers, residual connections or similar tricks

## ResNet and beyond
- Deeper networks suffer from degradation (not caused by overfitting - no regularisation)
- Plain network: H(x) by several weight layers
- Residual networks F(x) = H(x) - x (input is added after layers)
- ResNets helps backprop in deep networks
- Maybe possible to replace pooling with convolution using higher stride
- Stochastic models: Shallow net during training, deep during inference 25% speed-up, 25 % relative improvement for networks with over 1000 layers
- Densely connected: Connect layers directly with all following layers - parameter efficient (no relearning redundant features, better gradient flow)

## Fine-grained image recognition
- Classify visually very similar objects (distinguish species of birds, model of aircraft etc., not just: Is this a dog or a cat)
- High intra-class variance, low inter-class variance
- Part localisation: Find parts of image (beak, wing etc.) -> align -> recognise
- Divide and conquer: One model for family, one for genus, one for species

## Detection and classification of facial attributes
- Attribute: Aspect of visual appearance (age, gender, race etc.)
- Facial attributes as basis for image retrieval
- Global approach: No need to annotate parts or landmarks
- Local: Detect parts, concatenate local features
- Overfitting is a common problem, use aggressive augmentation and strong regularisation together with appropriate loss

## Content-based image retrieval (CBIR)
- Query by image content
    - Caption
    - Example image
    - Form of content features
    - Sketch
- Similarity
    - Content: Color is similar
    - Near duplicates (colour changes, compressed etc)
    - Same object or scene
    - Scene geometry
    - Same scenes, but with high visual distinctiveness (semantic)
- Pipeline: Extract descriptors, perform efficient NN search

## Semantic image embeddings using CNN
- Extraction of feature vector
- First layers: Low level texture features
- Last layers: High level concepts

## Indexing structures for efficient retrieval of semantic neighbours
- Indexing required for CBIR to scale to billions of images
- Vector quantisation via k-means -> adaptive indexing, but slow for large image collections
- Semantic hashing - efficient for binary codes, but better suited for uniform vector distribution
- Trade-off k-means and semantic hashing

## Face verification
- Face verification - are these the same person
- Open-set - match may be absent
- Closed-set - match always present
- Align face -> extract features -> same person?
- Triplet loss: 2 matching and 1 non-matching thumbnail images, difference between non-matching face should be larger than matching faces

## Re-identification problem
- Identification unique individual across multiple cameras
- Hard due to highly varying images (overlaps, lightning, low resolution etc.)
- Standard approach: Similarity between two images compared against treshold
- May be effective with part-based models

## Facial key points regression
- Points like eyes, nose etc regressed to an image
- Local: Locate parts, must include facial shape constraint
    - Sliding window: Extract features (SIFT/Hog/CNN), classify via SVM, ANN or similar
- Global: Regression from entire image
- Statistical methods
    - Methods like PCA for training
    - Inference: Iteratively minimise

## CNN for key point regression
- Ensemble methods
    - Option 1 - Simple averaging
    - Option 2 - Coarse-to-fine prediction: Initial model refined by subsequent methods

