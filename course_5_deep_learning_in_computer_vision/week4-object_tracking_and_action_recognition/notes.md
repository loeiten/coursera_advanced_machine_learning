# Week 4 notes

## Working with videos
- Videos can be large
- Not very generalisable from different angles
- https://www.biomotionlab.ca/demos/
- Optical flow: Vector field of apparent motion of pixels between frames
    - Visualized by point vectors for small set of frames, or color coding (hue is orientation and length by saturation)
    - Angle error
    - Endpoint error
    - Can generate GT from fluorescent image or 3D data
- Flow estimation from FlowNet: Either concatenate two images into 6 channels, or to input channels which are later concatenated

## Object tracking
- Single object localised in first frame
- Model-free: Only know object at first frame
    -  Not learning the model of object to be tracked
    - Can't train detector
    - Usual to build some descriptor of object appearance, search area in next frame etc.
- Only previous frames can be used
- Challenging (changes of objects and environment, similarity between objects)
- Ground truth hard to get: 1 example of object tracking = 1 video
- votchallenge.net
- Can collect large number of sequences, cluster similar sequences
- Accuracy - average overlap during successful tracking
- Robustness - number of times a tracker drifts of target
- Expected Average Overlap: Re-initialization at overlap 0
- Speed metric: Equivalent Filter Operations
    - Tracking time divided by time to perform filter operation
    - MAX filter in 30x30 window for all pixels in 600x600 image
- Iterative object tracking
    - Model init from frame -> feature vector -> search in local neighbourhood -> maximising visual similarity
- Object representation
    - Object template (can be tracked with cross-correlation)
    - Set of object parts or
    - Appearance features
- Color-based tracking (object-background + object-distractor)
    - Computes Region of Interest of object and background
    - Maximize likelihood Ratio
    - For similar objects can add distractors to other similar objects
- Sum of Template and Pixel-wise Learners (staple)

## CNN to VOT
- Object-background classifier
    - Region of highest score
    - Online training and regularly update to handle changes
    - Multi-Domain Network
        - One offline component (like VGG)
        - Multiple branches for online video-specific (domain specific) component (fully connected)
        - Hard negative mining (mining of false positives and adding them to negative bounding boxes)
- Regression of position
    - Previous and current frame as input
    - Offline learning
- Generic Object Tracking Using Regression Networks (GOTURN)
    - Train NN on network on collection of images (original and shifted) and videos without bboxes
    - Regress position of objects from previous frame
    - 100 Hz for GPU
- Accurate methods are usually slower than 1 FPS

## Multiple Object Tracking (MOT)
- Long-term tracking
- Two variants
    - Detection based tracking
    - Detection free tracking
- Tracking errors
    - ID switch
    - Fragmentation (frag) - missed detection
- MOT Accuracy (robustness) = 1 - sum(FN+FP + IDSW)/sum(GT)
- MOT Precision = avg(sum(dissimilarities from GT))
- Mostly tracked (tracked over 80 %)
- Mostly lost (tracked less than 20 %)
- Partially tracked
- Challenging, important with good detector
- One frame and multi-frame methods
- IoU tracking
- Simple Online and Realtime Tracking (SOR)
- Can use RNN
- Affinity modeling, appearance modeling through re-identification methods

## Action recognition
- Action (can have different meaning in different contexts)
- Event (collection of smaller actions possibly performed by several performers)
- Assign label to action or event
- Find location of event
- Easier to collect and annotate datasets compared to VOT
- Non CNN
    - Harris 3D detector: Distinctive neighbours in space and time, look at distribution of gradients
    - Histogram of gradients (HOG), histogram of optical flow (HOF)
    - Visual words (cluster descriptors)
    - Dense sampling and dense trajectories helps
    - Improved trajectories include filtered trajectories (for example optical flow minus camera movement
- CNN
    - Can use whole spacetime volume
    - Two scale model: Fovea stream (central part 89X89) and Context stream (whole frame downscaled to 89X89)
    - Single frame
    - First and last frame (late fusion)
    - Couple of center frames (early fusion)
    - Slow fusion, different frames to different inputs which are concatenated later in network
    - Two stream: Spatial CNN, temporal CNN

## Localization
- Drinking or smoking vs random is "easy"
- Drinking vs smoking hard
- Human pose
- Best result are obtained by classifying space-time volumes in temporal window along with object trajectory

