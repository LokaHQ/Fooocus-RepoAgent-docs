## FunctionDef centered_canny(x, canny_low_threshold, canny_high_threshold)
**centered_canny**: The function of centered_canny is to apply the Canny edge detection algorithm to a grayscale image and normalize the output.

**parameters**: The parameters of this Function.
· parameter1: x - A 2D numpy array representing a grayscale image, which must be of type uint8.
· parameter2: canny_low_threshold - The lower threshold for the Canny edge detection algorithm.
· parameter3: canny_high_threshold - The upper threshold for the Canny edge detection algorithm.

**Code Description**: The centered_canny function takes a 2D numpy array as input, which represents a grayscale image, and two threshold values for the Canny edge detection algorithm. The function first asserts that the input is indeed a numpy array, checks that it is two-dimensional, and confirms that its data type is uint8, which is standard for grayscale images. 

Once the input validation is complete, the function applies the Canny edge detection using the OpenCV library's cv2.Canny method, which detects edges in the image based on the provided low and high threshold values. The resulting output is a binary image where edges are marked. The output is then converted to a float32 data type and normalized by dividing by 255.0, which scales the pixel values to a range between 0 and 1. This normalization is often useful for further image processing tasks.

The centered_canny function is called by the centered_canny_color function, which extends its functionality to handle color images. In centered_canny_color, the input image is expected to be a 3D numpy array (representing an RGB image). The function iterates over each color channel (red, green, and blue), applying centered_canny to each channel separately. The results are then stacked back together to form a 3D array that represents the edges detected in the color image.

**Note**: It is important to ensure that the input image for centered_canny is a valid grayscale image. The function will raise an assertion error if the input does not meet the specified criteria. Additionally, the thresholds provided should be chosen carefully to achieve the desired edge detection results.

**Output Example**: A possible appearance of the code's return value could be a 2D numpy array where pixel values are in the range of 0.0 to 1.0, representing the normalized edges detected in the input grayscale image. For example, an output might look like:
```
array([[0., 0., 1., ..., 0., 0.],
       [0., 1., 1., ..., 0., 0.],
       [0., 0., 0., ..., 0., 0.],
       ...,
       [0., 0., 0., ..., 1., 0.],
       [0., 0., 0., ..., 0., 0.]])
```
## FunctionDef centered_canny_color(x, canny_low_threshold, canny_high_threshold)
**centered_canny_color**: The function of centered_canny_color is to apply the centered Canny edge detection algorithm to each color channel of a color image and return the combined result.

**parameters**: The parameters of this Function.
· parameter1: x - A 3D numpy array representing a color image in RGB format, which must have three channels.
· parameter2: canny_low_threshold - The lower threshold for the Canny edge detection algorithm.
· parameter3: canny_high_threshold - The upper threshold for the Canny edge detection algorithm.

**Code Description**: The centered_canny_color function begins by asserting that the input x is a numpy array and that it is a 3D array with three channels, confirming it represents a color image. The function then processes each color channel (red, green, and blue) individually by calling the centered_canny function. This function applies the Canny edge detection algorithm to each channel, using the specified low and high threshold values for edge detection.

The results from the centered_canny function for each channel are collected into a list, which is subsequently stacked along the third dimension to create a 3D numpy array. This output represents the edges detected in the original color image across all three channels.

The centered_canny_color function is called by the pyramid_canny_color function, which utilizes it to perform edge detection on resized versions of the input image at various scales. This hierarchical relationship allows pyramid_canny_color to accumulate edge information from multiple resolutions, enhancing the overall edge detection process.

**Note**: It is crucial to ensure that the input image is a valid RGB image with three channels. The function will raise an assertion error if the input does not meet the specified criteria. Additionally, the thresholds provided should be chosen carefully to achieve the desired edge detection results.

**Output Example**: A possible appearance of the code's return value could be a 3D numpy array where each pixel value represents the edges detected in the respective color channel. For example, an output might look like:
```
array([[[0., 0., 0.],
        [0., 1., 0.],
        [1., 1., 1.],
        ...,
        [0., 0., 0.],
        [0., 0., 0.]],

       [[0., 0., 0.],
        [1., 1., 1.],
        [1., 1., 1.],
        ...,
        [0., 0., 0.],
        [0., 0., 0.]],

       ...,
       
       [[0., 0., 0.],
        [0., 0., 0.],
        [1., 0., 0.],
        ...,
        [1., 0., 0.],
        [0., 0., 0.]]])
```
## FunctionDef pyramid_canny_color(x, canny_low_threshold, canny_high_threshold)
**pyramid_canny_color**: The function of pyramid_canny_color is to perform multi-scale Canny edge detection on a color image, accumulating edge information from various resolutions.

**parameters**: The parameters of this Function.
· parameter1: x - A 3D numpy array representing a color image in RGB format, which must have three channels.
· parameter2: canny_low_threshold - The lower threshold for the Canny edge detection algorithm.
· parameter3: canny_high_threshold - The upper threshold for the Canny edge detection algorithm.

**Code Description**: The pyramid_canny_color function begins by asserting that the input x is a numpy array and that it is a 3D array with three channels, confirming it represents a color image. The function retrieves the height (H), width (W), and number of channels (C) of the input image. It initializes an accumulator variable, acc_edge, to store the accumulated edge information.

The function then iterates over a predefined set of scaling factors (k) ranging from 0.2 to 1.0. For each scaling factor, it calculates the new dimensions (Hs, Ws) for the resized image. The input image is resized to these dimensions using the cv2.resize function with INTER_AREA interpolation, which is suitable for reducing image size.

Next, the centered_canny_color function is called with the resized image and the specified Canny thresholds. This function applies the Canny edge detection algorithm to each color channel of the resized image, returning the edges detected in a 3D numpy array.

If acc_edge is None (the first iteration), it is initialized with the edges detected in the current resized image. For subsequent iterations, acc_edge is resized to match the dimensions of the newly detected edges using INTER_LINEAR interpolation. The accumulated edges are updated by blending the previous accumulated edges with the newly detected edges, applying a weight of 0.75 to the previous edges and 0.25 to the current edges.

Finally, the function returns the accumulated edge information, which represents the edges detected across multiple resolutions of the input image.

The pyramid_canny_color function is called by the canny_pyramid function, which utilizes it to perform edge detection on the input image at multiple scales. The result from pyramid_canny_color is then summed across the color channels to produce a single edge map, which is further processed to normalize the pixel values.

**Note**: It is essential to ensure that the input image is a valid RGB image with three channels. The function will raise an assertion error if the input does not meet the specified criteria. Additionally, the thresholds provided should be chosen carefully to achieve the desired edge detection results.

**Output Example**: A possible appearance of the code's return value could be a 3D numpy array where each pixel value represents the edges detected in the respective color channel. For example, an output might look like:
```
array([[[0., 0., 0.],
        [0., 1., 0.],
        [1., 1., 1.],
        ...,
        [0., 0., 0.],
        [0., 0., 0.]],

       [[0., 0., 0.],
        [1., 1., 1.],
        [1., 1., 1.],
        ...,
        [0., 0., 0.],
        [0., 0., 0.]],

       ...,
       
       [[0., 0., 0.],
        [0., 0., 0.],
        [1., 0., 0.],
        ...,
        [1., 0., 0.],
        [0., 0., 0.]]])
```
## FunctionDef norm255(x, low, high)
**norm255**: The function of norm255 is to normalize a 2D numpy array to a range of 0 to 255 based on specified percentile values.

**parameters**: The parameters of this Function.
· x: A 2D numpy array of type float32 that needs to be normalized.
· low: An integer representing the lower percentile threshold for normalization (default is 4).
· high: An integer representing the upper percentile threshold for normalization (default is 96).

**Code Description**: The norm255 function takes a 2D numpy array as input and normalizes its values to a scale of 0 to 255. It first checks that the input is a valid 2D numpy array of type float32. The function then calculates the minimum and maximum values of the array based on the specified percentiles (low and high). The normalization process involves subtracting the minimum value from each element of the array and then dividing by the range (maximum - minimum). Finally, the normalized values are scaled by multiplying by 255. The resulting array is returned, effectively transforming the original data into a format suitable for image processing or visualization.

This function is called by other functions within the same module, specifically canny_pyramid and cpds. In canny_pyramid, norm255 is used to normalize the result of a color Canny edge detection operation, ensuring that the output image is properly scaled for display. In cpds, norm255 normalizes the result of a decolorization process, which enhances the contrast of the image. Both functions rely on norm255 to ensure that their outputs are within the appropriate range for further processing or visualization.

**Note**: It is important to ensure that the input array is of the correct type and dimensionality before calling this function. The output will be clipped to the range of 0 to 255 and converted to an unsigned 8-bit integer format.

**Output Example**: A possible appearance of the code's return value could be a numpy array such as:
```
array([[  0,  64, 128, 192, 255],
       [  0,  64, 128, 192, 255],
       [  0,  64, 128, 192, 255]], dtype=uint8)
```
## FunctionDef canny_pyramid(x, canny_low_threshold, canny_high_threshold)
**canny_pyramid**: The function of canny_pyramid is to perform multi-scale Canny edge detection on a color image and normalize the resulting edge map.

**parameters**: The parameters of this Function.
· parameter1: x - A 3D numpy array representing a color image in RGB format, which must have three channels.
· parameter2: canny_low_threshold - The lower threshold for the Canny edge detection algorithm.
· parameter3: canny_high_threshold - The upper threshold for the Canny edge detection algorithm.

**Code Description**: The canny_pyramid function is designed to enhance edge detection in images by utilizing a multi-scale approach. It begins by invoking the pyramid_canny_color function, which applies Canny edge detection across various resolutions of the input image. This function is crucial as it accumulates edge information from different scales, ensuring that structures present in the image are not missed due to resolution discrepancies.

The result from pyramid_canny_color is a 3D numpy array where edge information is stored across the color channels. The canny_pyramid function then sums the edge information across these channels, resulting in a single 2D array that represents the overall edge map of the image.

To prepare the edge map for further processing or visualization, the function normalizes the pixel values using the norm255 function. This normalization scales the values to a range of 0 to 255 based on specified percentile thresholds, ensuring that the output is suitable for display and analysis.

The canny_pyramid function is called within the apply_control_nets function in the async_worker module. In this context, it processes images that are part of asynchronous tasks related to control networks. Specifically, it applies the Canny edge detection only if the preprocessor is not being skipped, ensuring that the edge detection step is integrated into the overall image processing workflow.

**Note**: It is essential to ensure that the input image is a valid RGB image with three channels. The function will raise an assertion error if the input does not meet the specified criteria. Additionally, the thresholds provided for Canny edge detection should be chosen carefully to achieve the desired results.

**Output Example**: A possible appearance of the code's return value could be a 2D numpy array where each pixel value represents the edges detected in the image. For example, an output might look like:
```
array([[  0,  64, 128, 192, 255],
       [  0,  64, 128, 192, 255],
       [  0,  64, 128, 192, 255]], dtype=uint8)
```
## FunctionDef cpds(x)
**cpds**: The function of cpds is to perform a decolorization process on an input image, enhancing its contrast and normalizing the result for further processing.

**parameters**: The parameters of this Function.
· x: A 3D numpy array representing the input image in the format of height x width x channels (typically RGB).

**Code Description**: The cpds function begins by applying a Gaussian blur to the input image `x` using OpenCV's GaussianBlur method, which helps in reducing noise and detail in the image. The blurred image is then processed using OpenCV's decolor method, which separates the image into two components: density and boost. 

The function converts the raw image, density, and boost components to float32 type for accurate calculations. It calculates an offset based on the difference between the raw image and the boost component, which is then used to enhance the contrast of the density image. The final result is obtained by summing the density and the calculated offset.

To ensure the output is suitable for image display, the result is normalized using the norm255 function, which scales the pixel values to a range of 0 to 255 based on specified percentile thresholds. The output is clipped to ensure all values remain within this range and is converted to an unsigned 8-bit integer format.

The cpds function is called within the apply_control_nets function in the async_worker module. In this context, it processes images as part of a series of tasks related to control networks. Specifically, it is invoked when the preprocessing step is not skipped, ensuring that the images are appropriately decolorized before further processing. This integration highlights the function's role in enhancing image quality for subsequent tasks in the pipeline.

**Note**: It is crucial to ensure that the input image is in the correct format (3D numpy array) before calling this function. The output will be clipped to the range of 0 to 255 and converted to an unsigned 8-bit integer format, making it suitable for display or further image processing.

**Output Example**: A possible appearance of the code's return value could be a numpy array such as:
```
array([[  0,  64, 128, 192, 255],
       [  0,  64, 128, 192, 255],
       [  0,  64, 128, 192, 255]], dtype=uint8)
```
