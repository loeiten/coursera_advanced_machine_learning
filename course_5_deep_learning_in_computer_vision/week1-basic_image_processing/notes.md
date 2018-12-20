# Week 1 notes

## Computer vision
- Photogrammetry (distance estimation from images)
- The Marr Prize
- Viola-Jones face detector (2001)
- Eye movements and vision (Yarbus, A.L.)
- Around one angular minute of higher resolution from the eye
- The eyes are seccading (rapid change of fixation)
- Brain makes the image by combining several fixation points
-  The succession of fixation point depends on the mental task

## Color
- Our perception of color is physiological
- Metamers: Two spectra which gives us the sense of the same colors
- Color matching experiments: Test light vs a combination of RGB -> sometimes not possible to match test colors wiht available primaries
- Grassman's Laws: Color matching appears to be linear
- Color diagram: Gamut
- Cannot cover the entire gamut with three points
- Light perception is non-uniform

## Image preprocessing

### Brightness correction

- Brigthness histogram: x-axis = brightness values from white to black, y-axis: number of pixels
- Problems: Brightness range is not fully used or values are concentrated around certain values
- Point operators (points treated independently when correcting)
- Linear correction: (y - y_min)*(255 - 0)/(y_max - y_min)
- Gamma correction for contrast enhancement (non-linear)
- Free form brightness transformation
- Histogram equalization

### Image convolution
- Image averaging for reducing for noise
- Filters with convolution to smooth, sharpen, blur etc
     - Box filter
     - Gaussian filter
     - Negative Laplacian of Gaussian

### Edge detection
- Edge strength is given by the gradient magnitude
- Derivatives through finite differences
- Smooth before finding derivatives
- Missing precision and connectivity between edge point if only use gradient strength
- Canny edge detection takes this into account
- Non maximum suppression: Take away non maximum points
- Edge linking: Construct tangent to edge curve and use to predict the next point
- Hysteresis thresholding

