## FunctionDef resize_mask(mask, shape)
**resize_mask**: The function of resize_mask is to resize a given mask tensor to a specified shape using bilinear interpolation.

**parameters**: The parameters of this Function.
· parameter1: mask - A tensor representing the mask to be resized. It is expected to have at least two dimensions, with the last two dimensions representing height and width.
· parameter2: shape - A tuple containing two integers that specify the target height and width to which the mask should be resized.

**Code Description**: The resize_mask function takes a mask tensor and a target shape as input. It first reshapes the mask tensor to ensure it has the correct dimensions for processing, specifically transforming it into a 4D tensor with a shape of (-1, 1, height, width). The function then utilizes the PyTorch function `torch.nn.functional.interpolate` to resize the mask to the specified dimensions using bilinear interpolation. After resizing, the function removes the singleton dimension added during reshaping by calling `squeeze(1)`, resulting in a 3D tensor that retains the original batch size and the new height and width.

This function is called within the `join_image_with_alpha` method of the JoinImageWithAlpha class. In this context, the resize_mask function is used to adjust the alpha mask to match the dimensions of the input image. The alpha mask is inverted (1.0 - alpha) after resizing, and then it is concatenated with the RGB channels of the image to create an output image that includes the alpha channel. This integration is crucial for compositing images with transparency, ensuring that the final output image has the correct dimensions and alpha values.

**Note**: It is important to ensure that the mask tensor has the appropriate dimensions before calling this function. The shape parameter must be a tuple of two integers representing the desired output dimensions. The function assumes that the mask is a single-channel tensor.

**Output Example**: If the input mask has a shape of (1, 256, 256) and the target shape is (128, 128), the output of the resize_mask function would be a tensor with a shape of (1, 128, 128), representing the resized mask.
## ClassDef PorterDuffMode
**PorterDuffMode**: The function of PorterDuffMode is to define various blending modes for compositing images.

**attributes**: The attributes of this Class.
· ADD: Represents the addition blending mode.
· CLEAR: Represents the clear blending mode, which results in a transparent output.
· DARKEN: Represents the darken blending mode, which selects the darker of the two colors.
· DST: Represents the destination mode, which outputs the destination image.
· DST_ATOP: Represents the destination atop mode, which combines the source and destination images based on their alpha values.
· DST_IN: Represents the destination in mode, which outputs the source where it overlaps with the destination.
· DST_OUT: Represents the destination out mode, which outputs the destination where it does not overlap with the source.
· DST_OVER: Represents the destination over mode, which outputs the destination image over the source.
· LIGHTEN: Represents the lighten blending mode, which selects the lighter of the two colors.
· MULTIPLY: Represents the multiply blending mode, which multiplies the source and destination colors.
· OVERLAY: Represents the overlay blending mode, which combines multiply and screen modes based on the destination color.
· SCREEN: Represents the screen blending mode, which results in a lighter output.
· SRC: Represents the source mode, which outputs the source image.
· SRC_ATOP: Represents the source atop mode, which combines the source and destination images based on the destination's alpha.
· SRC_IN: Represents the source in mode, which outputs the source where it overlaps with the destination.
· SRC_OUT: Represents the source out mode, which outputs the source where it does not overlap with the destination.
· SRC_OVER: Represents the source over mode, which outputs the source image over the destination.
· XOR: Represents the exclusive or mode, which combines the source and destination images where they do not overlap.

**Code Description**: The PorterDuffMode class is an enumeration that defines a set of constants representing different Porter-Duff compositing modes used in image processing. Each mode specifies how two images (source and destination) should be blended together based on their alpha (transparency) values. The modes can be utilized in various image compositing functions, such as the porter_duff_composite function, which takes two images and their respective alpha channels along with a specified blending mode to produce a resulting image and alpha channel.

The porter_duff_composite function utilizes the PorterDuffMode enumeration to determine how to blend the source and destination images based on the selected mode. For instance, if the ADD mode is selected, the function will add the pixel values of the source and destination images, clamping the result to ensure it remains within valid bounds. Similarly, other modes like CLEAR, DARKEN, and MULTIPLY define specific blending behaviors that are executed within the function based on the mode provided.

The INPUT_TYPES function in the PorterDuffImageComposite class also references the PorterDuffMode enumeration to define the expected input types for the compositing operation, allowing users to select from the available blending modes when performing image compositing tasks.

**Note**: When using the PorterDuffMode enumeration, it is important to ensure that the images being processed are compatible in terms of dimensions and channels to avoid runtime errors during compositing operations.
## FunctionDef porter_duff_composite(src_image, src_alpha, dst_image, dst_alpha, mode)
**porter_duff_composite**: The function of porter_duff_composite is to blend two images based on specified Porter-Duff compositing modes.

**parameters**: The parameters of this Function.
· src_image: A torch.Tensor representing the source image to be blended.
· src_alpha: A torch.Tensor representing the alpha (transparency) channel of the source image.
· dst_image: A torch.Tensor representing the destination image to be blended with the source image.
· dst_alpha: A torch.Tensor representing the alpha (transparency) channel of the destination image.
· mode: An instance of PorterDuffMode that specifies the blending mode to be used for compositing the images.

**Code Description**: The porter_duff_composite function performs image compositing by blending a source image with a destination image according to the specified Porter-Duff mode. The function begins by checking the blending mode provided as an argument. It supports various modes such as ADD, CLEAR, DARKEN, and MULTIPLY, among others, each defining a specific way to combine the pixel values of the source and destination images based on their alpha channels.

For instance, in the ADD mode, the function computes the output alpha as the clamped sum of the source and destination alpha values, and similarly, it computes the output image by clamping the sum of the source and destination pixel values. In contrast, the CLEAR mode results in a fully transparent output, while the DARKEN mode selects the darker pixel value from the source and destination images.

The function returns two tensors: out_image, which contains the blended image, and out_alpha, which contains the resulting alpha channel. This function is called within the composite method of the PorterDuffImageComposite class, which processes batches of images. The composite method prepares the input images and their alpha channels, ensuring they are compatible in dimensions before invoking porter_duff_composite for each image pair in the batch. The results from porter_duff_composite are collected and returned as a stacked tensor of output images and their corresponding alpha channels.

**Note**: When using the porter_duff_composite function, it is essential to ensure that the input images and alpha channels are of compatible dimensions to avoid runtime errors. Additionally, the selected PorterDuffMode must be valid to ensure the correct blending behavior is applied.

**Output Example**: An example output of the function could be two tensors where out_image contains the blended pixel values of the source and destination images, and out_alpha contains the resulting alpha values, both shaped according to the input images. For instance, if the input images are of size (batch_size, height, width, channels), the output tensors will also have the same shape.
## ClassDef PorterDuffImageComposite
**PorterDuffImageComposite**: The function of PorterDuffImageComposite is to perform image compositing using the Porter-Duff algorithm, combining source images and their respective alpha masks with destination images and masks based on a specified mode.

**attributes**: The attributes of this Class.
· INPUT_TYPES: A class method that defines the required input types for the compositing operation, including source images, source alpha masks, destination images, destination alpha masks, and the compositing mode.
· RETURN_TYPES: Specifies the output types of the composite function, which are an image and a mask.
· FUNCTION: The name of the function that performs the compositing operation, which is "composite".
· CATEGORY: The category under which this class is classified, which is "mask/compositing".

**Code Description**: The PorterDuffImageComposite class is designed to facilitate the compositing of images using the Porter-Duff blending modes. It includes a class method INPUT_TYPES that outlines the necessary inputs for the compositing process. The required inputs are:
- source: A tensor representing the source image(s).
- source_alpha: A tensor representing the alpha mask(s) for the source image(s).
- destination: A tensor representing the destination image(s).
- destination_alpha: A tensor representing the alpha mask(s) for the destination image(s).
- mode: A list of available Porter-Duff modes, with a default set to "DST".

The class defines a method named composite, which takes the aforementioned parameters and processes them. Inside the composite method, the batch size is determined by the minimum length of the input tensors. The method iterates through each image in the batch, ensuring that the source and destination images have the same number of channels. If the dimensions of the alpha masks do not match the corresponding images, the method upscales them to ensure compatibility using a bicubic interpolation method.

The core of the compositing operation is handled by the porter_duff_composite function, which applies the specified Porter-Duff mode to blend the source and destination images along with their alpha masks. The results are collected into lists and finally returned as stacked tensors representing the output images and their respective alpha masks.

**Note**: It is important to ensure that the input images and masks are properly formatted and that their dimensions are compatible before invoking the composite method. The class relies on the presence of the porter_duff_composite function and the PorterDuffMode enumeration for its operations.

**Output Example**: The output of the composite method will be a tuple containing two tensors: the first tensor represents the resulting composite image, and the second tensor represents the alpha mask of the composite image. For instance, if the input images are of size (batch_size, height, width, channels), the output will also be of the same shape for the image tensor, while the alpha mask will have a shape of (batch_size, height, width).
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define the required input types for the image compositing operation.

**parameters**: The parameters of this Function.
· s: This parameter is a placeholder for the function's input, which is not utilized within the function body.

**Code Description**: The INPUT_TYPES function returns a dictionary that specifies the required input types for an image compositing operation using the Porter-Duff blending modes. The returned dictionary contains a single key, "required", which maps to another dictionary that outlines the expected inputs.

The "required" dictionary includes the following keys:
- "source": This key expects an input of type "IMAGE", which represents the source image to be composited.
- "source_alpha": This key expects an input of type "MASK", which represents the alpha mask for the source image, determining its transparency.
- "destination": This key expects an input of type "IMAGE", which represents the destination image onto which the source image will be composited.
- "destination_alpha": This key expects an input of type "MASK", which represents the alpha mask for the destination image.
- "mode": This key expects a list of blending modes derived from the PorterDuffMode enumeration. It provides a default value of PorterDuffMode.DST, which indicates that the destination image will be used as the output when no blending is applied.

The INPUT_TYPES function is integral to the PorterDuffImageComposite class, as it defines the structure of inputs necessary for the compositing process. By specifying the required types, it ensures that users provide the correct data formats when invoking the compositing functionality. This validation helps prevent runtime errors and facilitates a smoother integration of the compositing operation within larger image processing workflows.

**Note**: When utilizing the INPUT_TYPES function, it is essential to ensure that the provided images and masks are compatible in terms of dimensions and channels to avoid any issues during the compositing process.

**Output Example**: An example of the return value from the INPUT_TYPES function could look like this:
{
    "required": {
        "source": ("IMAGE",),
        "source_alpha": ("MASK",),
        "destination": ("IMAGE",),
        "destination_alpha": ("MASK",),
        "mode": (["ADD", "CLEAR", "DARKEN", "DST", "DST_ATOP", "DST_IN", "DST_OUT", "DST_OVER", "LIGHTEN", "MULTIPLY", "OVERLAY", "SCREEN", "SRC", "SRC_ATOP", "SRC_IN", "SRC_OUT", "SRC_OVER", "XOR"], {"default": "DST"}),
    },
}
***
### FunctionDef composite(self, source, source_alpha, destination, destination_alpha, mode)
**composite**: The function of composite is to blend batches of source and destination images using specified Porter-Duff compositing modes.

**parameters**: The parameters of this Function.
· source: A torch.Tensor representing the batch of source images to be blended.  
· source_alpha: A torch.Tensor representing the alpha (transparency) channels of the source images.  
· destination: A torch.Tensor representing the batch of destination images to be blended with the source images.  
· destination_alpha: A torch.Tensor representing the alpha (transparency) channels of the destination images.  
· mode: An instance of PorterDuffMode that specifies the blending mode to be used for compositing the images.

**Code Description**: The composite function processes batches of images by blending each pair of source and destination images according to their respective alpha channels and the specified Porter-Duff compositing mode. The function begins by determining the minimum batch size from the input tensors to ensure that all inputs are processed correctly.

For each image in the batch, the function extracts the corresponding source and destination images along with their alpha channels. It asserts that both images have the same number of channels to maintain compatibility during blending. If the dimensions of the alpha channels do not match the dimensions of the images, the function uses the common_upscale utility to resize the alpha channels to match the dimensions of the destination images. Similarly, if the source image dimensions do not match the destination image dimensions, the function upscales the source image accordingly.

Once the images and alpha channels are appropriately sized, the function calls the porter_duff_composite function, passing the source image, source alpha, destination image, destination alpha, and the specified blending mode. The porter_duff_composite function performs the actual blending operation based on the selected Porter-Duff mode, which determines how the pixel values of the source and destination images are combined.

The results from the porter_duff_composite function, which include the blended image and the resulting alpha channel, are collected into lists. After processing all images in the batch, the function stacks the output images and alpha channels into tensors and returns them as a tuple.

This function is integral to the image compositing workflow, allowing for efficient blending of multiple images in a single operation while ensuring that the dimensions and alpha channels are correctly handled.

**Note**: It is crucial to ensure that the input tensors for source, source_alpha, destination, and destination_alpha are all of compatible dimensions to avoid runtime errors. Additionally, the specified mode must be a valid PorterDuffMode to ensure the correct blending behavior is applied.

**Output Example**: An example output of the function could be a tuple containing two tensors: the first tensor representing the blended images with shape (batch_size, height, width, channels), and the second tensor representing the resulting alpha values with shape (batch_size, height, width). For instance, if the input batch size is 4 and the images are of size 256x256 with 3 color channels, the output would be two tensors of shape (4, 256, 256, 3) and (4, 256, 256), respectively.
***
## ClassDef SplitImageWithAlpha
**SplitImageWithAlpha**: The function of SplitImageWithAlpha is to separate an image into its RGB components and its alpha channel.

**attributes**: The attributes of this Class.
· INPUT_TYPES: A class method that defines the required input types for the function.  
· CATEGORY: A string that categorizes the functionality of the class, specifically under "mask/compositing".  
· RETURN_TYPES: A tuple indicating the types of outputs returned by the function, which are "IMAGE" and "MASK".  
· FUNCTION: A string that specifies the name of the function to be executed, which is "split_image_with_alpha".  

**Code Description**: The SplitImageWithAlpha class is designed to handle image processing tasks, specifically for images that include an alpha channel. The class contains a class method, INPUT_TYPES, which specifies that the required input is an image of type "IMAGE". The class is categorized under "mask/compositing", indicating its purpose in image manipulation and compositing tasks. 

The core functionality is provided by the method split_image_with_alpha, which takes a tensor representation of an image as input. This method processes the input image tensor, which is expected to have a shape that includes an alpha channel (i.e., four channels: red, green, blue, and alpha). The method performs the following operations:
1. It extracts the RGB components from the input image tensor, creating a list of tensors that contain only the first three channels (i.e., the RGB channels).
2. It checks for the presence of an alpha channel. If the input image has more than three channels, it extracts the alpha channel; otherwise, it generates a tensor of ones with the same shape as the first channel, effectively treating it as fully opaque.
3. The method then stacks the RGB components and computes the mask by subtracting the alpha values from 1.0, resulting in a mask that represents the transparency of the image.

The final output of the method is a tuple containing two elements: the stacked RGB images and the computed mask.

**Note**: It is important to ensure that the input image tensor has the correct shape and number of channels (at least 3 for RGB and optionally 4 for RGBA) to avoid runtime errors. The method assumes that the input is a batch of images.

**Output Example**: A possible appearance of the code's return value could be:
- RGB Images: A tensor of shape (N, H, W, 3), where N is the number of images, H is the height, and W is the width.
- Mask: A tensor of shape (N, H, W), representing the transparency of each pixel in the images.
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define the required input types for the image processing function.

**parameters**: The parameters of this Function.
· parameter1: s - This parameter is typically used as a placeholder for the function's context or state but is not utilized within the function body.

**Code Description**: The INPUT_TYPES function returns a dictionary that specifies the required input types for the image processing functionality. The dictionary contains a single key, "required", which maps to another dictionary. This inner dictionary defines the expected input parameters for the function. In this case, it specifies that an input named "image" is required, and its type is designated as "IMAGE". The return structure indicates that the function expects an image input to operate correctly, ensuring that any calling code adheres to this requirement.

**Note**: It is important to ensure that the input provided to the function matches the specified type "IMAGE". Failure to do so may result in errors or unexpected behavior during the image processing operations.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "image": ("IMAGE",)
    }
}
***
### FunctionDef split_image_with_alpha(self, image)
**split_image_with_alpha**: The function of split_image_with_alpha is to separate an image tensor into its RGB components and alpha channel, returning both as stacked tensors.

**parameters**: The parameters of this Function.
· image: A torch.Tensor representing a batch of images, where each image may contain an alpha channel.

**Code Description**: The split_image_with_alpha function processes a batch of images represented as a tensor. It assumes that each image in the batch can have four channels (RGBA) or three channels (RGB). The function performs the following operations:

1. It extracts the RGB components from each image in the batch. This is done using a list comprehension that slices the first three channels of each image tensor. The result is a list of tensors, each containing only the RGB channels.

2. It then extracts the alpha channel from each image. If an image has more than three channels, it takes the fourth channel as the alpha channel. If an image has only three channels, it creates a tensor of ones with the same shape as the first channel of the image, effectively treating it as fully opaque. This is also done using a list comprehension.

3. Finally, the function stacks the RGB components and the alpha channels into two separate tensors. The alpha channels are inverted (1.0 - alpha) to represent the transparency instead of opacity. The result is a tuple containing the stacked RGB images and the stacked inverted alpha channels.

**Note**: It is important to ensure that the input tensor has the correct shape and number of channels. The function expects a tensor of shape (N, H, W, C), where N is the number of images, H is the height, W is the width, and C is the number of channels. If the input tensor does not meet these requirements, the function may not behave as expected.

**Output Example**: If the input tensor is a batch of two images, each with four channels (RGBA), the output might look like this:
- RGB Tensor: A tensor of shape (2, H, W, 3) containing the RGB components of the two images.
- Inverted Alpha Tensor: A tensor of shape (2, H, W) containing the inverted alpha values for the two images.
***
## ClassDef JoinImageWithAlpha
**JoinImageWithAlpha**: The function of JoinImageWithAlpha is to combine an image with an alpha mask to produce a composite image.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Specifies the required input types for the class, which include an image and an alpha mask.
· CATEGORY: Defines the category under which this class is classified, specifically "mask/compositing".
· RETURN_TYPES: Indicates the type of output returned by the class, which is an image.
· FUNCTION: The name of the function that performs the main operation, which is "join_image_with_alpha".

**Code Description**: The JoinImageWithAlpha class is designed to facilitate the compositing of images using an alpha mask. It contains a class method INPUT_TYPES that outlines the necessary inputs for the operation, specifically requiring an image of type "IMAGE" and an alpha mask of type "MASK". The class is categorized under "mask/compositing", indicating its purpose in image processing tasks related to masking.

The core functionality is encapsulated in the method join_image_with_alpha, which takes two parameters: image and alpha, both of which are expected to be PyTorch tensors. The method first determines the batch size by taking the minimum length of the image and alpha tensors. It then initializes an empty list to store the output images.

The alpha mask is processed by resizing it to match the dimensions of the input image, and it is inverted (1.0 - alpha) to prepare it for compositing. The method then iterates over each image in the batch, concatenating the RGB channels of the image with the corresponding alpha channel. This is achieved by appending the alpha channel as a new dimension to the image tensor. The resulting composite images are stacked into a single tensor, which is returned as a tuple.

**Note**: It is important to ensure that the dimensions of the image and alpha mask are compatible for the compositing operation. The alpha mask should be a single channel that corresponds to the dimensions of the input image.

**Output Example**: A possible return value of the join_image_with_alpha method could be a tensor of shape (batch_size, height, width, 4), where the last dimension represents the RGBA channels of the composite images. For instance, if the input batch size is 2 and the image dimensions are 256x256, the output could look like this:
```
tensor([[[[R, G, B, A], ...],
          ...],
         [[...],
          ...]])
```
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define the required input types for processing images with an alpha mask.

**parameters**: The parameters of this Function.
· s: This parameter is not utilized within the function and serves as a placeholder for potential future use or for maintaining a consistent function signature.

**Code Description**: The INPUT_TYPES function returns a dictionary that specifies the required input types for an operation involving images and alpha masks. The returned dictionary contains a single key, "required", which maps to another dictionary. This inner dictionary defines two keys: "image" and "alpha". The "image" key is associated with a tuple containing the string "IMAGE", indicating that the input must be of type IMAGE. Similarly, the "alpha" key is associated with a tuple containing the string "MASK", indicating that the input must be of type MASK. This structured return value ensures that the function clearly communicates the expected types of inputs necessary for further processing.

**Note**: It is important to ensure that the inputs provided to the function conform to the specified types, as deviations may lead to errors or unexpected behavior during processing.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "image": ("IMAGE",),
        "alpha": ("MASK",)
    }
}
***
### FunctionDef join_image_with_alpha(self, image, alpha)
**join_image_with_alpha**: The function of join_image_with_alpha is to combine an image tensor with an alpha mask tensor to produce an output tensor that includes an alpha channel.

**parameters**: The parameters of this Function.
· parameter1: image - A tensor of shape (N, C, H, W) representing a batch of images, where N is the batch size, C is the number of channels (expected to be at least 3 for RGB), H is the height, and W is the width of the images.
· parameter2: alpha - A tensor of shape (N, H, W) representing the alpha mask for each image in the batch, where N is the batch size, H is the height, and W is the width.

**Code Description**: The join_image_with_alpha function processes a batch of images and their corresponding alpha masks to create output images that incorporate the alpha channel. The function begins by determining the minimum batch size between the image and alpha tensors to ensure that both inputs are compatible. It initializes an empty list, out_images, to store the resulting images.

The alpha mask is resized to match the dimensions of the input images using the resize_mask function, which is called within this method. This function adjusts the alpha mask to the correct size using bilinear interpolation, ensuring that the alpha values correspond accurately to the pixel data of the images. After resizing, the alpha mask is inverted (1.0 - alpha) to prepare it for compositing.

For each image in the batch, the function concatenates the RGB channels of the image (the first three channels) with the resized alpha mask (expanded to a third dimension) along the channel dimension (dim=2). This results in an output tensor that contains the original RGB data along with the alpha channel, effectively creating an image with transparency.

Finally, the function stacks all the processed images into a single tensor and returns it as a tuple. This output is crucial for applications that require image compositing with transparency, allowing for the integration of images with varying levels of opacity.

**Note**: It is essential to ensure that the image and alpha tensors are appropriately shaped before calling this function. The alpha tensor must have a single channel, and its dimensions should be compatible with the corresponding image tensor. The function assumes that the input image tensor has at least three channels.

**Output Example**: If the input image tensor has a shape of (2, 3, 256, 256) and the alpha tensor has a shape of (2, 128, 128), the output of the join_image_with_alpha function would be a tensor with a shape of (2, 4, 256, 256), where the last dimension includes the RGB channels and the alpha channel for each image in the batch.
***
