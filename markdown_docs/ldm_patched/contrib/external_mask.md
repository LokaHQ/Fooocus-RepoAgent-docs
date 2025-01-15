## FunctionDef composite(destination, source, x, y, mask, multiplier, resize_source)
**composite**: The function of composite is to blend a source tensor into a destination tensor at specified coordinates, optionally using a mask to control the blending process.

**parameters**: The parameters of this Function.
· parameter1: destination - A PyTorch tensor representing the target image or feature map where the source will be blended.
· parameter2: source - A PyTorch tensor that contains the image or feature map to be blended into the destination.
· parameter3: x - An integer representing the x-coordinate in the destination where the blending will start.
· parameter4: y - An integer representing the y-coordinate in the destination where the blending will start.
· parameter5: mask (optional) - A PyTorch tensor that defines the blending mask; if None, a default mask of ones is used.
· parameter6: multiplier (default=8) - An integer that scales the coordinates for blending.
· parameter7: resize_source (default=False) - A boolean indicating whether the source tensor should be resized to match the dimensions of the destination tensor.

**Code Description**: The composite function is designed to blend a source tensor into a destination tensor at specified (x, y) coordinates. It first ensures that the source tensor is on the same device as the destination tensor. If the resize_source parameter is set to True, the source tensor is resized to match the height and width of the destination tensor using bilinear interpolation. 

The function then adjusts the source tensor to match the batch size of the destination tensor by calling the repeat_to_batch_size function, which ensures that the source tensor has the correct number of elements to avoid dimension mismatches during blending.

Next, the function calculates the valid blending coordinates by clamping the x and y values based on the dimensions of the source and destination tensors, ensuring that the blending does not attempt to overwrite pixels that are out of bounds. The left and top coordinates are derived from the adjusted x and y values, while the right and bottom coordinates are calculated based on the dimensions of the source tensor.

If a mask is provided, it is resized to match the dimensions of the source tensor; otherwise, a default mask of ones is created. The mask is then used to determine which parts of the source tensor will be blended into the destination tensor. The blending is performed by multiplying the source tensor portion by the mask and combining it with the corresponding portion of the destination tensor, which is masked by the inverse of the blending mask.

Finally, the function updates the destination tensor with the blended result and returns it. This function is essential in scenarios where images or feature maps need to be combined, such as in image processing or deep learning applications.

**Note**: It is important to ensure that the input tensors are PyTorch tensors and that the coordinates (x, y) are within the valid range to avoid runtime errors. Additionally, the mask, if provided, should be compatible with the dimensions of the source tensor.

**Output Example**: For a destination tensor of shape (1, 3, 256, 256) and a source tensor of shape (1, 3, 128, 128) blended at coordinates (50, 50), the output would be a tensor of shape (1, 3, 256, 256) where the specified portion of the source tensor is blended into the destination tensor at the defined coordinates.
## ClassDef LatentCompositeMasked
**LatentCompositeMasked**: The function of LatentCompositeMasked is to perform a composite operation on latent images using specified parameters.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required and optional input types for the composite operation.
· RETURN_TYPES: Specifies the return type of the composite function.
· FUNCTION: Indicates the name of the function that will be executed.
· CATEGORY: Categorizes the class within the latent processing framework.

**Code Description**: The LatentCompositeMasked class is designed to facilitate the compositing of latent images. It contains a class method INPUT_TYPES that outlines the necessary and optional parameters for the composite operation. The required parameters include 'destination' and 'source', both of which are expected to be of type "LATENT". Additionally, it requires two integer parameters, 'x' and 'y', which define the position for the compositing operation, constrained by a minimum of 0 and a maximum defined by MAX_RESOLUTION, with a step of 8. The 'resize_source' parameter is a boolean that determines whether the source image should be resized during the operation. An optional parameter 'mask' of type "MASK" can also be provided to apply a mask to the composite operation.

The RETURN_TYPES attribute indicates that the output of the composite function will be of type "LATENT". The FUNCTION attribute specifies that the method to be executed is named "composite". The CATEGORY attribute classifies this operation under "latent", indicating its relevance to latent image processing.

The core functionality is encapsulated in the composite method, which takes the defined parameters and performs the compositing operation. It begins by creating a copy of the destination latent image and cloning its samples. The source samples are then retrieved, and the composite function is called with the destination, source, x, y, mask, a fixed value of 8, and the resize_source flag. The result of this operation is assigned to the 'samples' key of the output dictionary, which is then returned as a tuple containing the modified output.

**Note**: When utilizing this class, ensure that the input parameters adhere to the specified types and constraints to avoid runtime errors. The mask parameter is optional but can significantly affect the output if provided.

**Output Example**: A possible return value from the composite method could look like this:
{
    "samples": <modified latent image data>
}
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required and optional input types for a specific operation involving latent variables.

**parameters**: The parameters of this Function.
· s: This parameter is typically a placeholder for the state or context in which the function is called, although it is not utilized within the function body.

**Code Description**: The INPUT_TYPES function constructs a dictionary that categorizes input parameters into "required" and "optional" sections. The "required" section includes the following parameters:
- "destination": This parameter expects a tuple containing the string "LATENT", indicating that the destination for the operation is a latent variable.
- "source": Similar to "destination", this parameter also expects a tuple with the string "LATENT", specifying that the source of the operation is a latent variable.
- "x": This parameter is an integer that represents the x-coordinate. It has a default value of 0, with constraints that it must be a non-negative integer (minimum 0) and cannot exceed a maximum value defined by MAX_RESOLUTION. The step size for this parameter is set to 8.
- "y": This parameter is analogous to "x" but represents the y-coordinate, with the same constraints and default value.
- "resize_source": This boolean parameter indicates whether the source should be resized. It defaults to False.

The "optional" section includes:
- "mask": This parameter expects a value of type "MASK", which is used for additional processing or masking in the operation.

The function ultimately returns this structured dictionary, which can be utilized by other components of the system to validate and process inputs accordingly.

**Note**: It is important to ensure that the values provided for "x" and "y" adhere to the defined constraints to avoid runtime errors. Additionally, the presence of the "mask" parameter is optional, meaning that the function can operate without it if not specified.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "destination": ("LATENT",),
        "source": ("LATENT",),
        "x": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
        "y": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
        "resize_source": ("BOOLEAN", {"default": False}),
    },
    "optional": {
        "mask": ("MASK",),
    }
}
***
### FunctionDef composite(self, destination, source, x, y, resize_source, mask)
**composite**: The function of composite is to blend a source image into a destination image at specified coordinates, optionally using a mask.

**parameters**: The parameters of this Function.
· destination: The destination image where the source will be blended. It is expected to be a dictionary containing a key "samples" that holds the image data.
· source: The source image that will be blended into the destination. Similar to destination, it is a dictionary with a key "samples".
· x: The x-coordinate in the destination image where the blending will start.
· y: The y-coordinate in the destination image where the blending will start.
· resize_source: A boolean indicating whether the source image should be resized to fit the destination.
· mask: An optional parameter that can be used to specify a mask for blending. If not provided, the function will blend without a mask.

**Code Description**: The composite function begins by creating a copy of the destination image to preserve the original data. It then clones the "samples" from the destination image into a new variable, ensuring that any modifications do not affect the original destination. The source image's "samples" are directly referenced. The function then calls another composite function (presumably defined elsewhere) to perform the actual blending operation, passing in the cloned destination, the source samples, the x and y coordinates, the mask (if provided), a constant value of 8, and the resize_source flag. Finally, the function returns a tuple containing the modified output image.

**Note**: It is important to ensure that the destination and source images are properly formatted as dictionaries with the "samples" key. The mask parameter is optional but can significantly affect the blending outcome if provided. The function assumes that the composite function it calls is capable of handling the blending logic.

**Output Example**: A possible return value of the function could look like this:
{
    "samples": <blended image data>
}
***
## ClassDef ImageCompositeMasked
**ImageCompositeMasked**: The function of ImageCompositeMasked is to composite one image onto another at specified coordinates, optionally using a mask and resizing the source image.

**attributes**: The attributes of this Class.
· destination: The target image onto which the source image will be composited. It is required and must be of type IMAGE.
· source: The image that will be composited onto the destination image. It is required and must be of type IMAGE.
· x: The x-coordinate where the source image will be placed on the destination image. It is required, of type INT, with a default value of 0, and must be within the range of 0 to MAX_RESOLUTION.
· y: The y-coordinate where the source image will be placed on the destination image. It is required, of type INT, with a default value of 0, and must be within the range of 0 to MAX_RESOLUTION.
· resize_source: A boolean flag indicating whether the source image should be resized to fit the destination image. It is required, with a default value of False.
· mask: An optional parameter of type MASK that can be used to define areas of the source image that should be composited onto the destination image.

**Code Description**: The ImageCompositeMasked class provides a method for compositing images, allowing developers to overlay one image onto another at specified coordinates. The INPUT_TYPES class method defines the required and optional parameters for the composite operation. The composite method takes the destination image, the source image, the x and y coordinates for placement, the resize flag, and an optional mask. The destination image is cloned and its dimensions are adjusted to accommodate the compositing operation. The source image is also adjusted accordingly. The composite function is then called, which performs the actual image compositing based on the provided parameters. The output is returned as a tuple containing the resulting image.

**Note**: When using this class, ensure that the destination and source images are compatible in terms of dimensions and formats. The mask, if provided, should also align with the dimensions of the source image to achieve the desired compositing effect.

**Output Example**: A possible return value of the composite method could be an image that visually represents the source image composited onto the destination image at the specified (x, y) coordinates, with any masked areas applied, resulting in a new image that combines elements from both the source and destination.
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define the required and optional input types for the ImageCompositeMasked operation.

**parameters**: The parameters of this Function.
· destination: Specifies the type of the destination image, which must be of type "IMAGE".
· source: Specifies the type of the source image, which must also be of type "IMAGE".
· x: An integer representing the x-coordinate for positioning, with a default value of 0, a minimum value of 0, a maximum value defined by MAX_RESOLUTION, and a step increment of 1.
· y: An integer representing the y-coordinate for positioning, with a default value of 0, a minimum value of 0, a maximum value defined by MAX_RESOLUTION, and a step increment of 1.
· resize_source: A boolean indicating whether to resize the source image, with a default value of False.
· mask: An optional parameter that specifies a mask type, which is of type "MASK".

**Code Description**: The INPUT_TYPES function returns a dictionary that categorizes input parameters into two sections: "required" and "optional". The "required" section includes parameters that must be provided for the function to operate correctly, such as the destination and source images, as well as the x and y coordinates for positioning the source image. The x and y parameters are constrained by minimum and maximum values to ensure they fall within acceptable limits, which are defined by the constant MAX_RESOLUTION. Additionally, the resize_source parameter allows the user to specify whether the source image should be resized, defaulting to False. The "optional" section includes the mask parameter, which allows for additional functionality if a mask is provided.

**Note**: It is important to ensure that the values provided for x and y are within the defined limits to avoid errors during execution. The mask parameter is optional and can be omitted if not needed.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "destination": ("IMAGE",),
        "source": ("IMAGE",),
        "x": ("INT", {"default": 0, "min": 0, "max": 1920, "step": 1}),
        "y": ("INT", {"default": 0, "min": 0, "max": 1080, "step": 1}),
        "resize_source": ("BOOLEAN", {"default": False}),
    },
    "optional": {
        "mask": ("MASK",),
    }
}
***
### FunctionDef composite(self, destination, source, x, y, resize_source, mask)
**composite**: The function of composite is to blend a source image onto a destination image at specified coordinates, optionally using a mask and resizing the source image.

**parameters**: The parameters of this Function.
· destination: The destination image onto which the source image will be composited.  
· source: The source image that will be blended onto the destination image.  
· x: The x-coordinate where the source image will be placed on the destination image.  
· y: The y-coordinate where the source image will be placed on the destination image.  
· resize_source: A boolean indicating whether the source image should be resized during the compositing process.  
· mask: An optional parameter that specifies a mask to control the blending of the source image with the destination image.  

**Code Description**: The composite function begins by creating a clone of the destination image and changing its dimensions using the `movedim` method to adjust the channel order. This is necessary for proper image processing. The function then calls another composite function (presumably defined elsewhere) with the modified destination and source images, along with the specified x and y coordinates, the mask (if provided), a constant value of 1, and the resize_source flag. The result of this operation is then adjusted back to the original channel order using `movedim` again before being returned as a single-element tuple. This function is essential for image manipulation tasks where layering images is required, particularly in graphics applications.

**Note**: It is important to ensure that the dimensions of the destination and source images are compatible for compositing. If a mask is used, it should also match the dimensions of the source image to avoid errors during the blending process.

**Output Example**: The return value of the composite function would be a tuple containing the composited image, which may look like an array of pixel values representing the blended result of the destination and source images. For instance, if the destination was a blue square and the source was a red circle, the output might show a blend of blue and red pixels where the circle overlaps the square.
***
## ClassDef MaskToImage
**MaskToImage**: The function of MaskToImage is to convert a mask into an image format.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Specifies the required input types for the class method.
· CATEGORY: Defines the category under which this class is categorized.
· RETURN_TYPES: Indicates the type of output that the class method will return.
· FUNCTION: The name of the function that will be executed to perform the conversion.

**Code Description**: The MaskToImage class is designed to transform a mask into an image representation. It contains a class method called INPUT_TYPES, which defines the expected input for the class. The input is required to be of type "MASK". The class is categorized under "mask", indicating its functionality relates to mask processing. The RETURN_TYPES attribute specifies that the output will be of type "IMAGE", and the FUNCTION attribute indicates that the core functionality is encapsulated in the method named "mask_to_image".

The main method, mask_to_image, takes a single parameter, mask, which is expected to be a multidimensional array representing the mask. Inside this method, the mask is reshaped to have a specific dimensionality suitable for image processing. The reshape operation modifies the mask's shape to include an additional dimension, which is then moved to the end of the array using the movedim function. This is followed by an expansion of the dimensions to create a three-channel image, effectively converting the mask into an RGB image format. The result is returned as a tuple containing the transformed image.

**Note**: When using this class, ensure that the input mask is correctly formatted as a multidimensional array. The output will be a tuple, so it is important to access the first element to retrieve the image.

**Output Example**: A possible appearance of the code's return value could be a 3D NumPy array with dimensions corresponding to the height, width, and color channels of the image, such as (height, width, 3), where each pixel value represents the RGB color channels.
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define the required input types for a specific operation involving masks.

**parameters**: The parameters of this Function.
· s: This parameter is not utilized within the function but is included to maintain a consistent function signature.

**Code Description**: The INPUT_TYPES function returns a dictionary that specifies the required input types for a particular operation. The dictionary contains a single key, "required", which maps to another dictionary. This inner dictionary defines the expected input parameters for the operation. In this case, it specifies that a parameter named "mask" is required, and its type is denoted as "MASK". The structure indicates that the function is designed to enforce the presence of a mask input, which is likely essential for the subsequent processing or functionality that relies on this input.

**Note**: It is important to ensure that the input provided to the function adheres to the specified type "MASK" to avoid errors during execution. The function does not perform any validation or processing of the input; it solely defines the expected input structure.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "mask": ("MASK",)
    }
}
***
### FunctionDef mask_to_image(self, mask)
**mask_to_image**: The function of mask_to_image is to convert a mask tensor into an image tensor format suitable for further processing.

**parameters**: The parameters of this Function.
· mask: A tensor representing the mask to be converted, typically with dimensions that include height and width.

**Code Description**: The mask_to_image function takes a mask tensor as input and reshapes it to prepare it for image representation. The input mask is expected to have a specific shape, where the last two dimensions correspond to the height and width of the mask. The function first reshapes the mask using the reshape method, changing its dimensions to (-1, 1, mask.shape[-2], mask.shape[-1]). This transformation effectively adds a new dimension to the mask, allowing it to be treated as a batch of single-channel images. 

Next, the movedim method is utilized to change the order of the dimensions, moving the newly added dimension to the last position. This step is crucial for ensuring that the data is structured correctly for subsequent operations. Finally, the expand method is called to replicate the single-channel data across three channels, resulting in a three-channel image representation. The function returns a tuple containing the transformed result.

**Note**: It is important to ensure that the input mask tensor has the correct shape before calling this function. The function assumes that the last two dimensions of the mask correspond to the spatial dimensions (height and width) of the mask.

**Output Example**: If the input mask tensor has a shape of (2, 1, 64, 64), the output will be a tuple containing a tensor of shape (2, 64, 64, 3), where each pixel in the original mask is replicated across three channels.
***
## ClassDef ImageToMask
**ImageToMask**: The function of ImageToMask is to convert a specified color channel of an image into a mask.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types for the class method.
· CATEGORY: Specifies the category of the class, which is "mask".
· RETURN_TYPES: Indicates the type of output returned by the class method.
· FUNCTION: The name of the function that processes the input.

**Code Description**: The ImageToMask class is designed to create a mask from a given image based on a specified color channel. It contains a class method called INPUT_TYPES that outlines the expected inputs for the class. The required inputs include an image of type "IMAGE" and a channel that can be one of the following: "red", "green", "blue", or "alpha". The CATEGORY attribute categorizes this class under "mask", indicating its purpose in image processing.

The RETURN_TYPES attribute specifies that the output of the class method will be of type "MASK". The FUNCTION attribute defines the name of the method responsible for the processing, which is "image_to_mask". 

The core functionality is implemented in the image_to_mask method, which takes two parameters: image and channel. The method first defines a list of channels corresponding to the color components of the image. It then extracts the specified channel from the image using NumPy indexing, where the channel index is determined by finding the index of the provided channel in the channels list. The resulting mask, which is a 2D array representing the selected channel, is returned as a single-element tuple.

**Note**: When using this class, ensure that the input image is in a compatible format that includes the specified color channels. The channel parameter must be one of the predefined options; otherwise, an error may occur.

**Output Example**: If the input image is a 4D NumPy array representing an RGBA image and the channel specified is "red", the output will be a 2D NumPy array containing only the red channel values, effectively creating a mask that highlights the red component of the original image.
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define the required input types for an image processing function.

**parameters**: The parameters of this Function.
· parameter1: s - This parameter is not utilized within the function body and serves as a placeholder for potential future use or for compatibility with a specific function signature.

**Code Description**: The INPUT_TYPES function returns a dictionary that specifies the required input types for an image processing operation. The dictionary contains a single key, "required", which maps to another dictionary. This inner dictionary defines two required inputs: "image" and "channel". The "image" key is associated with a tuple containing the string "IMAGE", indicating that the input must be of the type IMAGE. The "channel" key is associated with a tuple containing a list of strings: "red", "green", "blue", and "alpha". This indicates that the channel input can be one of these four color channels, allowing for flexibility in specifying which channel of the image is to be processed.

**Note**: It is important to ensure that the inputs provided to any function utilizing INPUT_TYPES conform to the specified types. The function does not perform any validation on the inputs; it merely defines the expected structure.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "image": ("IMAGE",),
        "channel": (["red", "green", "blue", "alpha"],)
    }
}
***
### FunctionDef image_to_mask(self, image, channel)
**image_to_mask**: The function of image_to_mask is to extract a specific channel from an image and return it as a mask.

**parameters**: The parameters of this Function.
· image: A 4-dimensional array representing the image data, where the dimensions correspond to height, width, channels, and color depth.
· channel: A string indicating which channel to extract from the image. It should be one of the following: "red", "green", "blue", or "alpha".

**Code Description**: The image_to_mask function is designed to take an image and a specified color channel as inputs. The function first defines a list of channel names: "red", "green", "blue", and "alpha". It then uses the index of the specified channel to extract the corresponding data from the image array. The extraction is performed using NumPy-style slicing, which allows for efficient access to the desired channel data. The resulting mask, which is a 3-dimensional array containing only the data for the specified channel, is returned as a single-element tuple.

This function is particularly useful in image processing tasks where it is necessary to isolate a specific color channel for analysis or manipulation. By returning the mask as a tuple, the function maintains consistency with other functions that may return multiple outputs.

**Note**: It is important to ensure that the input image is in the correct format and contains the specified channel. If the channel provided does not exist in the image, an IndexError may occur. Users should validate the input before calling this function to avoid runtime errors.

**Output Example**: If the input image is a 4D NumPy array with shape (height, width, channels, depth) and the specified channel is "red", the output might look like this:
```
(array([[0.1, 0.2, 0.3],
         [0.4, 0.5, 0.6],
         [0.7, 0.8, 0.9]]),)
```
This output represents the extracted red channel data from the original image, encapsulated in a tuple.
***
## ClassDef ImageColorToMask
**ImageColorToMask**: The function of ImageColorToMask is to convert an image into a mask based on a specified color.

**attributes**: The attributes of this Class.
· INPUT_TYPES: A class method that defines the required input types for the image and color parameters.
· CATEGORY: A string that categorizes the functionality of the class, set to "mask".
· RETURN_TYPES: A tuple indicating the type of output returned by the class, which is "MASK".
· FUNCTION: A string that specifies the function name to be used, which is "image_to_mask".

**Code Description**: The ImageColorToMask class is designed to create a binary mask from an input image based on a specified color. The class contains a class method INPUT_TYPES that specifies the required inputs: an image of type "IMAGE" and a color of type "INT". The color parameter has constraints including a default value of 0, a minimum value of 0, a maximum value of 0xFFFFFF, a step of 1, and a display type of "color". The class is categorized under "mask" and specifies that it returns a "MASK" type output.

The core functionality is implemented in the method image_to_mask, which takes an image tensor and a color integer as inputs. Inside this method, the image tensor is first clamped to ensure that its values are within the range of 0 to 1.0, then scaled to a range of 0 to 255, and rounded to the nearest integer. The method then constructs a temporary representation of the image by combining the red, green, and blue channels into a single integer value using bitwise operations. A mask is generated by comparing this temporary representation to the specified color, where pixels matching the color are set to 255 (white) and all other pixels are set to 0 (black). The resulting mask is returned as a float tensor.

**Note**: When using this class, ensure that the input image is properly formatted as a tensor and that the color value is within the specified range. The output mask will be a binary representation where the specified color is highlighted.

**Output Example**: If the input image contains a pixel with the RGB value (255, 0, 0) and the specified color is set to 16711680 (which corresponds to red), the output mask will have a value of 255 for that pixel and 0 for all other pixels. The output mask tensor might look like this for a simple case:

```
[[[255, 0, 0],
  [0, 0, 0]],
 
 [[0, 0, 0],
  [255, 0, 0]]]
```
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types for an image processing function that involves color manipulation.

**parameters**: The parameters of this Function.
· parameter1: s - This parameter is not utilized within the function body but is typically used to represent the state or context in which the function is called.

**Code Description**: The INPUT_TYPES function constructs a dictionary that specifies the required input types for processing an image with a color parameter. The returned dictionary contains a single key, "required", which maps to another dictionary detailing the expected inputs. 

- The first entry under "required" is "image", which is expected to be of type "IMAGE". This indicates that the function requires an image input for processing.
- The second entry is "color", which is expected to be of type "INT". This entry includes additional constraints:
  - "default": 0, which sets the default value of the color parameter to 0.
  - "min": 0, indicating that the minimum allowable value for the color is 0.
  - "max": 0xFFFFFF, which sets the maximum allowable value for the color to 16777215 in decimal (the maximum value for a 24-bit RGB color).
  - "step": 1, which specifies that the color value can be incremented in steps of 1.
  - "display": "color", which suggests that the input for this parameter should be presented in a color picker format in user interfaces.

This structured approach ensures that the function receives the correct types of inputs, facilitating proper image processing based on the specified color.

**Note**: It is important to ensure that the inputs conform to the specified types and constraints to avoid errors during the image processing operation. The color input should be provided in a format that adheres to the defined range and type.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "image": ("IMAGE",),
        "color": ("INT", {"default": 0, "min": 0, "max": 16777215, "step": 1, "display": "color"})
    }
}
***
### FunctionDef image_to_mask(self, image, color)
**image_to_mask**: The function of image_to_mask is to convert an image into a binary mask based on a specified color.

**parameters**: The parameters of this Function.
· image: A tensor representing the input image, typically in the range [0, 1] with shape (N, C, H, W), where N is the batch size, C is the number of channels (usually 3 for RGB), H is the height, and W is the width of the image.
· color: An integer representing the target color in the format of a 24-bit integer, where the red, green, and blue channels are combined into a single integer.

**Code Description**: The image_to_mask function processes the input image to create a binary mask that highlights pixels matching a specified color. The function first clamps the pixel values of the input image to ensure they are within the range [0, 1.0] and then scales these values to the range [0, 255] by multiplying by 255. The resulting values are rounded and converted to integers. 

Next, the function constructs a single integer representation of the RGB color by performing bitwise operations: it shifts the red channel (first channel) left by 16 bits, the green channel (second channel) left by 8 bits, and adds the blue channel (third channel). This creates a 24-bit integer that represents the color in the format expected for comparison.

The function then generates a mask by comparing the constructed integer representation of the image's pixels to the specified color. If a pixel matches the color, it is assigned a value of 255 in the mask; otherwise, it is assigned a value of 0. The resulting mask is converted to a float tensor and returned as a single-element tuple.

**Note**: It is important to ensure that the input image is in the correct format and range before calling this function. The color parameter must be a valid 24-bit integer corresponding to the desired RGB color.

**Output Example**: If the input image contains pixels that match the specified color (e.g., a bright red represented by the integer 0xFF0000), the output mask will be a tensor where pixels matching this color are set to 255 and all other pixels are set to 0. For instance, if the input image has dimensions (1, 3, 2, 2) and the specified color is red, the output might look like:
tensor([[255, 0],
        [0, 255]])
***
## ClassDef SolidMask
**SolidMask**: The function of SolidMask is to generate a solid mask tensor based on specified dimensions and a value.

**attributes**: The attributes of this Class.
· INPUT_TYPES: A class method that defines the required input types for the SolidMask functionality.  
· CATEGORY: A string that categorizes the SolidMask under "mask".  
· RETURN_TYPES: A tuple indicating the return type of the solid method, which is a "MASK".  
· FUNCTION: A string that specifies the name of the function to be executed, which is "solid".  

**Code Description**: The SolidMask class is designed to create a solid mask tensor filled with a specified float value. The class includes a class method INPUT_TYPES that outlines the required inputs for the solid method. These inputs include a float value (defaulting to 1.0, with a range between 0.0 and 1.0), and two integers representing the width and height of the mask (both defaulting to 512, with a minimum of 1 and a maximum defined by MAX_RESOLUTION). The CATEGORY attribute categorizes this class under "mask", while RETURN_TYPES specifies that the output will be a "MASK". The FUNCTION attribute indicates that the core functionality is encapsulated in the "solid" method. 

The solid method itself takes three parameters: value, width, and height. It utilizes the PyTorch library to create a tensor filled with the specified value, having the shape of (1, height, width). The tensor is created on the CPU and is of type float32. The method returns a tuple containing the generated tensor.

**Note**: When using this class, ensure that the input parameters adhere to the specified ranges to avoid errors. The MAX_RESOLUTION constant should be defined elsewhere in the code to set the upper limit for width and height.

**Output Example**: An example of the output from the solid method when called with parameters (value=0.5, width=512, height=512) would be a tensor of shape (1, 512, 512) filled with the value 0.5. The output tensor would look like this:

```
tensor([[[0.5000, 0.5000, 0.5000, ..., 0.5000, 0.5000, 0.5000],
         [0.5000, 0.5000, 0.5000, ..., 0.5000, 0.5000, 0.5000],
         ...
         [0.5000, 0.5000, 0.5000, ..., 0.5000, 0.5000, 0.5000]]])
```
### FunctionDef INPUT_TYPES(cls)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return a dictionary that specifies the required input types and their constraints for a particular class.

**parameters**: The parameters of this Function.
· cls: This parameter refers to the class itself and is used to define class-level attributes.

**Code Description**: The INPUT_TYPES function constructs and returns a dictionary containing the required input types for the class. The dictionary has a single key, "required", which maps to another dictionary that defines three specific input parameters: "value", "width", and "height". Each of these parameters has a tuple as its value, where the first element indicates the data type (e.g., "FLOAT" or "INT") and the second element is another dictionary that specifies the default value, minimum value, maximum value, and step size for that parameter.

- The "value" parameter is of type "FLOAT" with a default value of 1.0, a minimum of 0.0, a maximum of 1.0, and a step size of 0.01. This indicates that the value must be a floating-point number within the specified range.
- The "width" parameter is of type "INT" with a default value of 512, a minimum of 1, a maximum defined by the constant MAX_RESOLUTION, and a step size of 1. This means that the width must be an integer within the specified range.
- The "height" parameter is also of type "INT" with the same constraints as the "width" parameter.

This structured approach allows for clear validation of input parameters when instances of the class are created or when methods are invoked that require these inputs.

**Note**: It is important to ensure that the values provided for "width" and "height" do not exceed the defined MAX_RESOLUTION constant, as this could lead to errors or unexpected behavior in the application.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
        "height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
    }
}
***
### FunctionDef solid(self, value, width, height)
**solid**: The function of solid is to create a tensor filled with a specified value.

**parameters**: The parameters of this Function.
· parameter1: value - A float value that will fill the tensor.
· parameter2: width - An integer representing the width of the tensor.
· parameter3: height - An integer representing the height of the tensor.

**Code Description**: The solid function generates a tensor of shape (1, height, width) filled with the specified float value. It utilizes the PyTorch library to create this tensor, ensuring that it is of type float32 and located on the CPU. The function begins by calling `torch.full`, which constructs a new tensor initialized to the provided value. The resulting tensor is then returned as a single-element tuple.

**Note**: It is important to ensure that the parameters width and height are positive integers, as negative or zero values would lead to an invalid tensor shape. Additionally, the function currently defaults to creating the tensor on the CPU; if GPU support is needed, modifications would be necessary.

**Output Example**: If the function is called as `solid(5.0, 4, 3)`, the return value would be a tensor that looks like this:
```
tensor([[[5.0, 5.0, 5.0, 5.0],
         [5.0, 5.0, 5.0, 5.0],
         [5.0, 5.0, 5.0, 5.0]]])
```
***
## ClassDef InvertMask
**InvertMask**: The function of InvertMask is to invert a given mask.

**attributes**: The attributes of this Class.
· INPUT_TYPES: A class method that defines the required input types for the class. It specifies that the input must include a "mask" of type "MASK".  
· CATEGORY: A constant that categorizes the functionality of the class, set to "mask".  
· RETURN_TYPES: A constant that defines the type of output returned by the class, which is a "MASK".  
· FUNCTION: A constant that specifies the name of the function that will be executed, which is "invert".  

**Code Description**: The InvertMask class is designed to perform a specific operation on a mask, which is a common data structure used in image processing and computer vision tasks. The primary functionality of this class is encapsulated in the `invert` method. This method takes a single parameter, `mask`, which is expected to be a numerical array representing the mask to be inverted. The inversion is achieved by subtracting the mask values from 1.0, effectively flipping the binary values of the mask (where 0 becomes 1 and 1 becomes 0). The result of this operation is returned as a tuple containing the inverted mask.

The class also defines its input and output types clearly. The `INPUT_TYPES` class method specifies that the class requires a "mask" input of type "MASK". The `RETURN_TYPES` attribute indicates that the output will also be of type "MASK". This clear definition of input and output types helps ensure that the class can be integrated smoothly into larger systems that utilize masks.

**Note**: It is important to ensure that the input mask is properly formatted and contains numerical values, as the inversion operation relies on this. Users should also be aware that the output will be a tuple, even though it contains only one element, which is the inverted mask.

**Output Example**: Given an input mask of `[[0, 1], [1, 0]]`, the output of the `invert` method would be `[[1.0, 0.0], [0.0, 1.0]]`. The output is returned as a tuple: `([[1.0, 0.0], [0.0, 1.0]],)`.
### FunctionDef INPUT_TYPES(cls)
**INPUT_TYPES**: The function of INPUT_TYPES is to define the required input types for the mask processing functionality.

**parameters**: The parameters of this Function.
· None

**Code Description**: The INPUT_TYPES function is a class method that returns a dictionary specifying the required input types for the operation of the class it belongs to. The returned dictionary contains a single key, "required", which maps to another dictionary. This inner dictionary defines the expected input parameters for the function. In this case, it specifies that a parameter named "mask" is required, and its type is designated as "MASK". This structure is typically used in frameworks that require explicit definitions of input types for validation or processing purposes, ensuring that the necessary data is provided before executing further operations.

**Note**: It is important to ensure that the input provided matches the expected type "MASK" to avoid errors during processing. This function does not take any parameters and is intended to be called on the class itself.

**Output Example**: The function would return the following structure:
{
    "required": {
        "mask": ("MASK",)
    }
}
***
### FunctionDef invert(self, mask)
**invert**: The function of invert is to compute the inverse of a given mask.

**parameters**: The parameters of this Function.
· mask: A numerical array or matrix representing the mask to be inverted.

**Code Description**: The invert function takes a single parameter, `mask`, which is expected to be a numerical array or matrix. The function performs an element-wise operation where it subtracts each element of the `mask` from 1.0. This operation effectively inverts the values in the mask, meaning that if the original mask contains values close to 1, the output will have values close to 0, and vice versa. The result of this operation is stored in the variable `out`. Finally, the function returns a tuple containing the inverted mask as its sole element. This design allows for easy integration into other parts of the code that may expect a tuple as output.

**Note**: It is important to ensure that the input `mask` contains values in the range [0, 1] to avoid unexpected results. Values outside this range may lead to outputs that do not conform to the expected behavior of a mask inversion.

**Output Example**: If the input mask is an array like [0.2, 0.5, 0.8], the output of the invert function would be (array([0.8, 0.5, 0.2]),).
***
## ClassDef CropMask
**CropMask**: The function of CropMask is to crop a specified region from a given mask.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types for the cropping operation, including the mask and the dimensions for cropping.
· CATEGORY: Specifies the category of the operation, which is "mask".
· RETURN_TYPES: Indicates the type of output returned by the crop function, which is a mask.
· FUNCTION: The name of the function that performs the cropping operation, which is "crop".

**Code Description**: The CropMask class is designed to facilitate the cropping of a mask image based on specified coordinates and dimensions. The class contains a class method INPUT_TYPES that outlines the required inputs for the cropping operation. These inputs include:
- `mask`: The mask to be cropped, which is expected to be of type "MASK".
- `x`: The x-coordinate of the top-left corner of the cropping rectangle, defined as an integer with a default value of 0 and constrained between 0 and MAX_RESOLUTION.
- `y`: The y-coordinate of the top-left corner of the cropping rectangle, also defined as an integer with a default value of 0 and constrained similarly.
- `width`: The width of the cropping rectangle, defined as an integer with a default value of 512 and constrained to be at least 1 and at most MAX_RESOLUTION.
- `height`: The height of the cropping rectangle, defined as an integer with a default value of 512 and similarly constrained.

The class also defines a method named `crop`, which takes the mask and the cropping parameters (x, y, width, height) as inputs. Inside this method, the mask is reshaped to ensure it has the correct dimensions for processing. The cropping operation is performed by slicing the mask array to extract the specified region, which is then returned as the output.

**Note**: When using the CropMask class, ensure that the input mask is in the correct format and that the specified coordinates and dimensions do not exceed the boundaries of the mask. The values for x and y should be within the dimensions of the mask, and width and height should be chosen to avoid out-of-bounds errors.

**Output Example**: If the input mask has a shape of (1, 1024, 1024) and the parameters are x=100, y=100, width=512, height=512, the output will be a mask of shape (1, 512, 512) containing the cropped region from the original mask.
### FunctionDef INPUT_TYPES(cls)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types for a specific operation involving a mask and its dimensions.

**parameters**: The parameters of this Function.
· mask: This parameter is of type "MASK" and is required for the operation.
· x: This parameter is of type "INT" with a default value of 0, a minimum value of 0, a maximum value defined by MAX_RESOLUTION, and a step increment of 1.
· y: This parameter is of type "INT" with a default value of 0, a minimum value of 0, a maximum value defined by MAX_RESOLUTION, and a step increment of 1.
· width: This parameter is of type "INT" with a default value of 512, a minimum value of 1, a maximum value defined by MAX_RESOLUTION, and a step increment of 1.
· height: This parameter is of type "INT" with a default value of 512, a minimum value of 1, a maximum value defined by MAX_RESOLUTION, and a step increment of 1.

**Code Description**: The INPUT_TYPES function is a class method that returns a dictionary specifying the required input types for a particular operation. The dictionary contains a single key "required", which maps to another dictionary detailing the parameters needed. Each parameter is associated with its type and constraints. The "mask" parameter is essential and must be provided as a "MASK" type. The parameters "x" and "y" represent the coordinates and are both integers with specified default values, minimum and maximum limits, and step increments. The "width" and "height" parameters define the dimensions of the mask and also follow similar constraints. The use of MAX_RESOLUTION ensures that the values do not exceed a predefined limit, which is crucial for maintaining the integrity of the operation.

**Note**: It is important to ensure that the values for "x", "y", "width", and "height" are within the specified ranges to avoid errors during execution. The default values provide a starting point, but users should adjust them according to their specific requirements.

**Output Example**: An example of the return value from the INPUT_TYPES function could look like this:
{
    "required": {
        "mask": ("MASK",),
        "x": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
        "y": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
        "width": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
        "height": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
    }
}
***
### FunctionDef crop(self, mask, x, y, width, height)
**crop**: The function of crop is to extract a specific rectangular region from a given mask.

**parameters**: The parameters of this Function.
· parameter1: mask - A multi-dimensional array representing the mask from which a region will be extracted. The last two dimensions of this array represent the height and width of the mask.
· parameter2: x - An integer representing the starting x-coordinate (horizontal position) from which the crop will begin.
· parameter3: y - An integer representing the starting y-coordinate (vertical position) from which the crop will begin.
· parameter4: width - An integer representing the width of the rectangular region to be extracted from the mask.
· parameter5: height - An integer representing the height of the rectangular region to be extracted from the mask.

**Code Description**: The crop function begins by reshaping the input mask to ensure that it has a consistent structure, specifically transforming it into a 3D array where the first dimension is flattened. This reshaping is done using the `reshape` method, which modifies the shape of the mask to have the dimensions (-1, mask.shape[-2], mask.shape[-1]). The `-1` allows NumPy to automatically determine the size of the first dimension based on the total number of elements and the specified dimensions for the last two. 

Following the reshaping, the function extracts a sub-region from the mask using array slicing. The slice is defined by the coordinates (x, y) and the specified width and height. The resulting output is a new array containing only the specified rectangular area of the original mask. Finally, the function returns this cropped area as a single-element tuple.

**Note**: It is important to ensure that the specified coordinates (x, y) and the dimensions (width, height) do not exceed the boundaries of the original mask. If they do, it may result in an error or unexpected behavior.

**Output Example**: If the input mask is a 4x4 array and the parameters are x=1, y=1, width=2, height=2, the function will return a tuple containing a 2x2 array that represents the cropped region:
```
(array([[mask[1][1], mask[1][2]],
         [mask[2][1], mask[2][2]]]),)
```
***
## ClassDef MaskComposite
**MaskComposite**: The function of MaskComposite is to combine two mask images using specified operations at given coordinates.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types for the class method.
· CATEGORY: Specifies the category of the operation, which is "mask".
· RETURN_TYPES: Indicates the type of output returned by the class method.
· FUNCTION: The name of the function that performs the operation, which is "combine".

**Code Description**: The MaskComposite class is designed to facilitate the combination of two mask images through various operations such as multiplication, addition, subtraction, and bitwise operations (AND, OR, XOR). The class provides a class method INPUT_TYPES that outlines the necessary parameters for the combination process, including the destination mask, source mask, x and y coordinates for positioning, and the operation type. 

The combine method reshapes the destination and source masks to ensure they are compatible for the operation. It calculates the visible area based on the provided x and y coordinates, ensuring that the operation does not exceed the bounds of the destination mask. Depending on the specified operation, the method performs the corresponding mathematical or bitwise operation on the overlapping regions of the destination and source masks. The output is then clamped to ensure that the values remain within the range of 0.0 to 1.0, which is typical for mask images.

**Note**: When using the MaskComposite class, ensure that the dimensions of the destination and source masks are compatible. The x and y parameters should be within the bounds of the destination mask to avoid indexing errors. The operations performed will affect the final output, so choose the operation based on the desired effect on the mask images.

**Output Example**: A possible return value of the combine method could be a tensor representing the combined mask, where the overlapping regions have been modified according to the specified operation. For instance, if the destination mask is a tensor of shape (1, 256, 256) and the source mask is also of shape (1, 256, 256), the output will be a tensor of the same shape, containing the result of the operation applied to the specified regions.
### FunctionDef INPUT_TYPES(cls)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types for a specific operation involving masks.

**parameters**: The parameters of this Function.
· destination: Specifies the type of destination input, which is expected to be of type "MASK".
· source: Specifies the type of source input, which is also expected to be of type "MASK".
· x: An integer input representing the x-coordinate, with constraints on its default value, minimum, maximum, and step.
· y: An integer input representing the y-coordinate, with similar constraints as the x parameter.
· operation: A list of string options representing the mathematical operations that can be performed, including "multiply", "add", "subtract", "and", "or", and "xor".

**Code Description**: The INPUT_TYPES function is a class method that constructs and returns a dictionary containing the required input types for a mask operation. The dictionary is structured with a key "required", which maps to another dictionary that specifies the expected inputs. The "destination" and "source" keys indicate that both inputs must be of the type "MASK". The "x" and "y" parameters are defined as integers, each with a default value of 0, a minimum value of 0, a maximum value defined by the constant MAX_RESOLUTION, and a step increment of 1. The "operation" parameter is a list that allows the user to select from a predefined set of operations that can be applied to the masks.

**Note**: It is important to ensure that the inputs provided conform to the specified types and constraints to avoid errors during execution. The MAX_RESOLUTION constant must be defined elsewhere in the code for the function to operate correctly.

**Output Example**: An example of the return value from the INPUT_TYPES function could look like this:
{
    "required": {
        "destination": ("MASK",),
        "source": ("MASK",),
        "x": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
        "y": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
        "operation": (["multiply", "add", "subtract", "and", "or", "xor"],),
    }
}
***
### FunctionDef combine(self, destination, source, x, y, operation)
**combine**: The function of combine is to blend a source tensor into a destination tensor at specified coordinates using a defined operation.

**parameters**: The parameters of this Function.
· destination: A tensor that serves as the base onto which the source tensor will be combined.  
· source: A tensor that contains the data to be combined with the destination tensor.  
· x: An integer representing the x-coordinate (horizontal position) in the destination tensor where the source tensor will be placed.  
· y: An integer representing the y-coordinate (vertical position) in the destination tensor where the source tensor will be placed.  
· operation: A string that specifies the mathematical operation to be performed during the combination. Valid operations include "multiply", "add", "subtract", "and", "or", and "xor".

**Code Description**: The combine function begins by reshaping the destination tensor to ensure it has the appropriate dimensions for processing. It then reshapes the source tensor similarly. The function calculates the boundaries for the portion of the source tensor that will be combined with the destination tensor based on the provided x and y coordinates. It determines the visible width and height of the source tensor that can be placed within the destination tensor without exceeding its dimensions.

The function extracts the relevant portions of both the source and destination tensors. Depending on the specified operation, it performs the corresponding mathematical operation on the extracted portions. The operations supported include:
- "multiply": Element-wise multiplication of the source and destination portions.
- "add": Element-wise addition of the source and destination portions.
- "subtract": Element-wise subtraction of the source portion from the destination portion.
- "and": Bitwise AND operation on the boolean representations of the source and destination portions.
- "or": Bitwise OR operation on the boolean representations of the source and destination portions.
- "xor": Bitwise XOR operation on the boolean representations of the source and destination portions.

After performing the operation, the output tensor is clamped to ensure that all values remain within the range of 0.0 to 1.0. Finally, the function returns the modified output tensor as a single-element tuple.

**Note**: It is important to ensure that the dimensions of the source tensor do not exceed the boundaries of the destination tensor when specifying the x and y coordinates. The operation parameter must be one of the predefined strings; otherwise, the function may not behave as expected.

**Output Example**: If the destination tensor is a 3x3 matrix with values [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]] and the source tensor is a 2x2 matrix with values [[0.1, 0.2], [0.3, 0.4]], and the operation is "add" with coordinates (1, 1), the output might look like [[0.1, 0.2, 0.3], [0.4, 0.6, 0.8], [0.7, 0.8, 0.9]].
***
## ClassDef FeatherMask
**FeatherMask**: The function of FeatherMask is to apply a feathering effect to a given mask based on specified boundary parameters.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types for the feathering operation, including the mask and boundary parameters (left, top, right, bottom).
· CATEGORY: Specifies the category of the operation, which is "mask".
· RETURN_TYPES: Indicates the type of output returned by the feathering operation, which is a "MASK".
· FUNCTION: The name of the function that performs the feathering operation, which is "feather".

**Code Description**: The FeatherMask class is designed to apply a feathering effect to a mask, which is a common operation in image processing. The class contains a class method INPUT_TYPES that specifies the required inputs for the feathering process. The inputs include a mask of type "MASK" and four integer parameters (left, top, right, bottom) that define the extent of the feathering effect on each side of the mask. Each of these parameters has a default value of 0 and must be within the range of 0 to MAX_RESOLUTION.

The class also defines a method called feather, which takes the mask and the four boundary parameters as arguments. Inside this method, the mask is reshaped and cloned to create an output tensor. The method then calculates the feathering effect for each boundary by iterating over the specified ranges and applying a feathering rate that gradually decreases the opacity of the mask towards the edges. The feathering rate is determined by the current position relative to the boundary size, ensuring a smooth transition from fully opaque to fully transparent.

Finally, the method returns the modified mask as a tuple containing the output tensor.

**Note**: When using the FeatherMask class, ensure that the input mask is properly formatted and that the boundary parameters are set within the allowed range to avoid unexpected behavior. The feathering effect will only be applied if the boundary parameters are greater than zero.

**Output Example**: A possible appearance of the code's return value could be a tensor representing a mask where the edges have been softened, resulting in a gradient effect from the center of the mask to the edges, with values closer to zero at the boundaries and values closer to one towards the center.
### FunctionDef INPUT_TYPES(cls)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types for a specific operation involving a mask.

**parameters**: The parameters of this Function.
· mask: This parameter is of type "MASK" and is required for the operation.
· left: This parameter is of type "INT" with a default value of 0, a minimum value of 0, a maximum value defined by MAX_RESOLUTION, and a step increment of 1.
· top: This parameter is of type "INT" with a default value of 0, a minimum value of 0, a maximum value defined by MAX_RESOLUTION, and a step increment of 1.
· right: This parameter is of type "INT" with a default value of 0, a minimum value of 0, a maximum value defined by MAX_RESOLUTION, and a step increment of 1.
· bottom: This parameter is of type "INT" with a default value of 0, a minimum value of 0, a maximum value defined by MAX_RESOLUTION, and a step increment of 1.

**Code Description**: The INPUT_TYPES function is a class method that returns a dictionary specifying the required input types for a certain functionality. The dictionary contains a single key "required" which maps to another dictionary that defines the parameters needed for the operation. The "mask" parameter is essential and must be provided as a "MASK" type. The parameters "left", "top", "right", and "bottom" are all of type "INT" and are used to define the boundaries of a rectangular area. Each of these integer parameters has specific constraints: they must be non-negative (minimum value of 0), cannot exceed a maximum value defined by the constant MAX_RESOLUTION, and can only be incremented in steps of 1. The default value for each of these integer parameters is set to 0.

**Note**: It is important to ensure that the values provided for the integer parameters do not exceed the defined MAX_RESOLUTION to avoid errors during execution. The "mask" parameter is mandatory and must be supplied for the function to operate correctly.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "mask": ("MASK",),
        "left": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
        "top": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
        "right": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
        "bottom": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
    }
}
***
### FunctionDef feather(self, mask, left, top, right, bottom)
**feather**: The function of feather is to apply a feathering effect to the edges of a given mask by modifying pixel values based on specified boundary parameters.

**parameters**: The parameters of this Function.
· mask: A tensor representing the mask to which the feathering effect will be applied. It is expected to have at least three dimensions, where the last two dimensions correspond to the spatial dimensions (height and width) of the mask.
· left: An integer specifying the number of pixels from the left edge of the mask to be feathered.
· top: An integer specifying the number of pixels from the top edge of the mask to be feathered.
· right: An integer specifying the number of pixels from the right edge of the mask to be feathered.
· bottom: An integer specifying the number of pixels from the bottom edge of the mask to be feathered.

**Code Description**: The feather function begins by reshaping the input mask tensor to ensure that it has the correct dimensions for processing. It then initializes an output tensor that is a clone of the reshaped mask. The function calculates the effective boundaries for feathering by taking the minimum of the specified parameters and the dimensions of the mask. 

The feathering effect is applied in four loops corresponding to each edge of the mask: left, right, top, and bottom. For each pixel along the specified edges, a feathering rate is calculated based on its position relative to the edge. This rate is used to scale the pixel values, gradually reducing their intensity as they approach the edge of the mask. The left and top edges are processed in ascending order, while the right and bottom edges are processed in descending order. Finally, the function returns the modified output tensor as a single-element tuple.

**Note**: It is important to ensure that the mask tensor has sufficient dimensions and that the feathering parameters do not exceed the dimensions of the mask. The feathering effect is most effective when the left, right, top, and bottom parameters are set to values that create a smooth transition at the edges.

**Output Example**: A possible appearance of the code's return value could be a tensor of the same shape as the input mask, where the pixel values at the edges have been modified to create a gradient effect, resulting in softer transitions at the boundaries. For instance, if the input mask had pixel values of 1 at the center and 0 at the edges, the output might show values gradually decreasing from 1 to 0 towards the edges, depending on the specified feathering parameters.
***
## ClassDef GrowMask
**GrowMask**: The function of GrowMask is to expand or contract a given mask based on specified parameters.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types for the class method.
· CATEGORY: Specifies the category of the class, which is "mask".
· RETURN_TYPES: Indicates the types of output returned by the class method.
· FUNCTION: Specifies the name of the function that will be executed, which is "expand_mask".

**Code Description**: The GrowMask class is designed to manipulate image masks by expanding or contracting them using morphological operations. The class provides a class method INPUT_TYPES that outlines the required inputs for the mask manipulation process. The inputs include a mask of type "MASK", an integer for the expansion size, and a boolean indicating whether to use tapered corners in the expansion process.

The CATEGORY attribute categorizes this class under "mask", while RETURN_TYPES specifies that the output will also be of type "MASK". The FUNCTION attribute indicates that the core functionality of the class is encapsulated in the method named "expand_mask".

The expand_mask method takes three parameters: mask, expand, and tapered_corners. The method first determines the value of 'c', which is set to 0 if tapered_corners is True, otherwise it is set to 1. This value is used to create a kernel, which is a 3x3 numpy array that defines the structure for the morphological operation. 

The input mask is reshaped to ensure it has the correct dimensions for processing. The method then iterates over each mask in the batch, converting it to a numpy array for manipulation. Depending on the value of the expand parameter, the method applies either grey erosion or grey dilation using the defined kernel. If expand is negative, erosion is applied, which reduces the mask size; if positive, dilation is applied, which increases the mask size. After processing, the output is converted back to a PyTorch tensor and collected into a list. Finally, the method returns a stacked tensor of the processed masks.

**Note**: Users should ensure that the input mask is in the correct format and that the expand value is within the defined limits. The tapered_corners parameter affects the shape of the expansion and should be set according to the desired output.

**Output Example**: An example output of the expand_mask method could be a tensor representing a modified mask where the original mask has been expanded or contracted based on the specified parameters. For instance, if the input mask was a binary image of a shape, the output could be a larger or smaller binary image reflecting the morphological changes applied.
### FunctionDef INPUT_TYPES(cls)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types for the GrowMask class.

**parameters**: The parameters of this Function.
· mask: A required input of type MASK, which is expected to be provided by the user.
· expand: A required input of type INT, which has a default value of 0 and must be within the range defined by -MAX_RESOLUTION to MAX_RESOLUTION, with a step increment of 1.
· tapered_corners: A required input of type BOOLEAN, which has a default value of True.

**Code Description**: The INPUT_TYPES function is a class method that returns a dictionary specifying the required input types for the GrowMask functionality. The returned dictionary contains a single key "required", which maps to another dictionary that defines three specific inputs: "mask", "expand", and "tapered_corners". Each of these inputs has an associated type and, where applicable, additional constraints or default values. The "mask" input is of type MASK, indicating that it is a necessary component for the operation of the GrowMask. The "expand" input is of type INT, which allows for a numerical value that can be adjusted within a specified range, ensuring that the value adheres to the limits set by MAX_RESOLUTION. The "tapered_corners" input is of type BOOLEAN, providing a simple true/false option that defaults to True, allowing users to specify whether they want tapered corners in the mask.

**Note**: It is important for users to ensure that the inputs provided conform to the specified types and constraints to avoid errors during execution. The use of default values for "expand" and "tapered_corners" allows for flexibility in user input, but users should be aware of the implications of these defaults on the functionality of the GrowMask.

**Output Example**: An example of the return value from the INPUT_TYPES function could look like this:
{
    "required": {
        "mask": ("MASK",),
        "expand": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
        "tapered_corners": ("BOOLEAN", {"default": True}),
    },
}
***
### FunctionDef expand_mask(self, mask, expand, tapered_corners)
**expand_mask**: The function of expand_mask is to modify a given mask by expanding or contracting its features based on the specified parameters.

**parameters**: The parameters of this Function.
· mask: A tensor representing the input mask that needs to be expanded or contracted. It is expected to have at least two dimensions, with the last two dimensions representing the spatial dimensions of the mask.
· expand: An integer indicating the amount of expansion (positive value) or contraction (negative value) to be applied to the mask.
· tapered_corners: A boolean flag that determines whether the corners of the expansion kernel should be tapered (if True) or not (if False).

**Code Description**: The expand_mask function processes the input mask by reshaping it and applying morphological operations to expand or contract its features. The function begins by determining the value of `c`, which is set to 0 if tapered_corners is True, and 1 otherwise. This value is used to define the structure of the kernel, which is a 3x3 array that will be utilized for the morphological operations. The mask is reshaped to ensure that it has the correct dimensions for processing.

The function then iterates over each mask in the reshaped mask tensor. For each mask, it converts the tensor to a NumPy array to facilitate the use of the SciPy library's morphological functions. Depending on the value of the `expand` parameter, the function applies either grey erosion (if expand is negative) or grey dilation (if expand is positive) using the defined kernel. This process effectively modifies the mask by either shrinking or enlarging the features within it.

After processing, the modified output is converted back to a PyTorch tensor and collected into a list. Finally, the function returns a tuple containing a single tensor that is the stacked result of all processed masks.

**Note**: It is important to ensure that the input mask is in the correct format and shape before calling this function. The function relies on the SciPy library for morphological operations, so it is necessary to have this library installed and imported in the environment where this function is used.

**Output Example**: If the input mask is a tensor of shape (2, 5, 5) and the expand parameter is set to 1 with tapered_corners set to False, the output might be a tensor of shape (2, 5, 5) where the features of the mask have been expanded according to the morphological dilation operation. The exact values will depend on the initial content of the mask.
***
