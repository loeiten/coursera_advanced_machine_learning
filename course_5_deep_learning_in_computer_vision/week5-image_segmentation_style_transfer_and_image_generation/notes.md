# Week 5 notes

## Segmentation
- Image segmentation - Splitting image into fragments, group pixels by criterion
- Semantic segmentation - Every group in image belongs to a certain class
- Instance segmentation - Each object of a group is marked with own id
- Object extraction - Select a given object
- Co-segmentation - Extract the same image from different images
- Unsupervised segmentation - Statistical characteristic of group are homogeneous

## Oversegmentation
- Segmentation methods that meets following requirements are called over-segmentation/super pixel methods
    - Boundaries of segment approx boundaries of object
    - Segment contained within object
    - Small objects described by their own segment
    - Uniform distribution over the image
- In over segmentation one can iteratively segment objects into subcomponents
- Methods
    - Heuristic methods (region growth, split and merge regions)
    - Graph-based
    - Energy-based methods (Snakes, TurobPixels)
    - Clustering-based (Mean shift, QuickShift, SLIC)
- Mean-shift
    - Each pixel is assigned a feature vector (color, HOG, etc.)
    - Space features are grouped togehter
    - Maxima of data element density should be cluster centres
    - For each point, find nearest mode of distribution density (moving along density gradient)
    - A cluster is a group of points for which the search leads to the same mode of distribution
- SLIC: Simple Linear Iterative Clustering
    - Initialize clusters over the grid at distance s
    - Search over a region
    - Use k-means to find cluster centres
    - Stop when change is less than threshold

Deep learning for image segmentation
- Fully Convolutional Networks:
    - Take pertained CNN, convolutionalize fc layers
    - Use multi-class entropy loss function (each pixel in heat map has a class)
    - Fine tune fully convolutional network
    - Problem: Decrease of resolution
- Encoder-decoder
    - Encoder downsamples
    - Decoder upsamples with unspooling layers
    - Transposed convolution: Kernel and strides are the same, but use zero padding with appropriate size
        - Bilinear interpolation
        - Bed of nails
        - Max location switches
- U-net
    - Skip connections with concatenations

## Human pose estimation as image segmentation
- Top-down
    - Person detector, pose estimation for single person
    - Disadvantage: If people estimator fails -> no retrieval
    - Disadvantage: Runtime proportional to number of people
- Bottom-up
    - Synthesised Gaussian with a fixed variance entered at GT joint position
- DeepCut
    - Initial detection and pairwise graphs
    - Clusters that belong to one person, each part is labelled
    - Final prediction
- Realtime Multi-Person 2D
    - Detection using Confidence Maps
    - Part Affinity Fields (vector field that connects body parts)

## Style transfer
- Content c and style t are presented
- Goal: Produce output x matching content and style
- Content and style similarity: Feature activations in a convolutional network
- From a pertained model (e.g. AlexNet trained on ImageNet)
    - Extract content targets deep in the network
    - Similarity between x and c: L_content = |F^l(x)-F^l(c)|_2^2, where F^l is the output at the layer
    - Extract style from early in the network
    - Similarity between t and c: L_style = |G^l(x)-G^l(t)|_2^2, where G^l is the Gram matrix (sum_k F^l_{i,k} F^l_{j,k}) the layer
    - Total loss = a L_content + b L_style
    - Produce output x^* such that L = argmin_x L(x)

## GANs
- Paper of the year: https://arxiv.org/abs/1708.05509
- Generator (G) creates fake images
- The fake and real images are fed into discriminant (D), which outputs whether or not the current image is real
- D and G trains jointly, alternate between training different parts of the network
- The input x to D can come from the real distribution, or a fake image G(z), where z is some noise
- D tries to make D(G(z)) to 0
- G tries to make G(D(z)) to 1
- D and G have different loss functions
- Tries to solve a saddle problem
- DCGAN (Deep convolutional GAN)
    - Batch normalisation in both D and G
    - Last layer of G and first of D are not Batch normalised, so that model can learn correct mean and scale of data distribution
    - No pooling or unpooling layers
    - Adam instead of SGD

## Image to image translations
- Day to night, sketch to real, are to map, BW to color
- Training data not available
- Generator G_{AB} maps x_A to x_AB
- D_B classifies between real x_B and fake x_{AB} by minimising GAN loss
- Reconstructional loss: Add a generator G_{BA} which takes x_{AB} as input and outputs x_{ABA}, which should be similar to x_A
- Can even try to reconstruct real and generated image

