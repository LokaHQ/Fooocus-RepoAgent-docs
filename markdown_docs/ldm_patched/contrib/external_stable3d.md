## FunctionDef camera_embeddings(elevation, azimuth)
**camera_embeddings**: The function of camera_embeddings is to generate camera embeddings based on specified elevation and azimuth angles.

**parameters**: The parameters of this Function.
· elevation: A float value representing the elevation angle in degrees.
· azimuth: A float value representing the azimuth angle in degrees.

**Code Description**: The camera_embeddings function takes two parameters, elevation and azimuth, which are both expected to be in degrees. It converts these parameters into PyTorch tensors to facilitate tensor operations. The function computes a set of embeddings that represent the camera's orientation in a 3D space.

The embeddings are calculated as follows:
1. The elevation is transformed to a polar coordinate by subtracting it from 90 degrees, which aligns with the Zero123 polar coordinate system.
2. The sine and cosine of the azimuth angle are computed to represent the horizontal orientation.
3. A constant value representing the zenith angle (90 degrees) is included in the embeddings.

The resulting embeddings are stacked into a tensor with an additional dimension added, making it suitable for further processing in neural networks.

This function is called within the encode methods of two classes: StableZero123_Conditioning and StableZero123_Conditioning_Batched. In both cases, camera_embeddings is utilized to generate camera embeddings that are concatenated with image embeddings obtained from a vision model. 

In StableZero123_Conditioning, the camera embeddings are generated for a single set of elevation and azimuth values, while in StableZero123_Conditioning_Batched, the function is called in a loop to create embeddings for a batch of images, incrementing the elevation and azimuth for each iteration. This allows for the generation of diverse camera perspectives in a batch processing context.

**Note**: It is important to ensure that the elevation and azimuth values are provided in degrees, as the function performs conversions based on this assumption.

**Output Example**: A possible appearance of the code's return value could be a tensor of shape (1, 1, 4) containing the calculated embeddings, such as:
```
tensor([[[ 1.5708,  0.8660,  0.5000,  1.5708]]])  # Example output for specific elevation and azimuth
```
## ClassDef StableZero123_Conditioning
**StableZero123_Conditioning**: The function of StableZero123_Conditioning is to encode images and camera parameters into a conditioning format suitable for 3D models.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types for the encoding process, including various parameters such as clip_vision, init_image, vae, width, height, batch_size, elevation, and azimuth.
· RETURN_TYPES: Specifies the types of outputs returned by the encode function, which are CONDITIONING, CONDITIONING, and LATENT.
· RETURN_NAMES: Names for the returned outputs, which are positive, negative, and latent.
· FUNCTION: The name of the function that performs the encoding, which is "encode".
· CATEGORY: The category under which this class is organized, specifically "conditioning/3d_models".

**Code Description**: The StableZero123_Conditioning class is designed to facilitate the encoding of images along with camera parameters into a format that can be utilized for conditioning in 3D models. The class defines a class method `INPUT_TYPES` that outlines the necessary inputs for the encoding process. These inputs include:
- `clip_vision`: An object responsible for encoding images.
- `init_image`: The initial image to be processed.
- `vae`: A Variational Autoencoder used for encoding pixel data.
- `width` and `height`: Dimensions for the output, with constraints on their values.
- `batch_size`: The number of images to process in a single batch.
- `elevation` and `azimuth`: Camera parameters that define the viewpoint.

The `encode` method takes these inputs and performs the following operations:
1. It utilizes the `clip_vision` object to encode the `init_image`, producing an output that contains image embeddings.
2. The image embeddings are pooled and prepared for further processing.
3. The initial image is upscaled to the specified width and height using bilinear interpolation.
4. The pixel data is then encoded using the provided `vae`.
5. Camera embeddings are generated based on the elevation and azimuth parameters.
6. The pooled embeddings and camera embeddings are concatenated to form the conditioning data.

The method returns three outputs:
- `positive`: A list containing the conditioning data and the encoded latent image.
- `negative`: A list containing zeroed embeddings and a zeroed latent image for contrastive purposes.
- `latent`: A tensor initialized to zeros, representing the latent space for the batch.

**Note**: It is important to ensure that the input parameters adhere to the specified constraints, particularly for width, height, and batch size, to avoid runtime errors during the encoding process.

**Output Example**: A possible appearance of the code's return value could be:
```
positive: [[<tensor of shape (1, C, H, W)>, {"concat_latent_image": <tensor of shape (1, C, H, W)>}]]
negative: [[<tensor of shape (1, C, H, W)>, {"concat_latent_image": <tensor of shape (1, C, H, W)>}]]
latent: {"samples": <tensor of shape (batch_size, 4, height // 8, width // 8)>}
```
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types for a specific process in the application.

**parameters**: The parameters of this Function.
· parameter1: s - This parameter is a placeholder for any input that may be passed to the function, although it is not utilized within the function body.

**Code Description**: The INPUT_TYPES function constructs and returns a dictionary that specifies the required input types for various parameters needed in the application. The dictionary contains a single key, "required", which maps to another dictionary detailing the specific input parameters and their types. Each parameter is associated with a tuple that includes the type of the input and, in some cases, additional constraints or default values.

The parameters defined in the returned dictionary include:
- "clip_vision": This parameter expects a value of type "CLIP_VISION".
- "init_image": This parameter expects a value of type "IMAGE".
- "vae": This parameter expects a value of type "VAE".
- "width": This parameter expects an integer ("INT") with a default value of 256. It has constraints for minimum (16), maximum (defined by ldm_patched.contrib.external.MAX_RESOLUTION), and a step increment of 8.
- "height": Similar to "width", this parameter expects an integer ("INT") with a default value of 256, and the same constraints for minimum, maximum, and step.
- "batch_size": This parameter expects an integer ("INT") with a default value of 1, a minimum of 1, and a maximum of 4096.
- "elevation": This parameter expects a floating-point number ("FLOAT") with a default value of 0.0, a minimum of -180.0, and a maximum of 180.0.
- "azimuth": This parameter also expects a floating-point number ("FLOAT") with the same constraints as "elevation".

This structured approach ensures that the inputs are validated against defined types and constraints, facilitating error handling and improving the robustness of the application.

**Note**: It is important to ensure that the values provided for each parameter adhere to the specified types and constraints to avoid runtime errors. The maximum resolution for "width" and "height" should be checked against the defined constant in the external module.

**Output Example**: An example of the return value from the INPUT_TYPES function would be:
{
    "required": {
        "clip_vision": ("CLIP_VISION",),
        "init_image": ("IMAGE",),
        "vae": ("VAE",),
        "width": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 8}),
        "height": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 8}),
        "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        "elevation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
        "azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
    }
}
***
### FunctionDef encode(self, clip_vision, init_image, vae, width, height, batch_size, elevation, azimuth)
**encode**: The function of encode is to process an initial image and generate conditioning data for a model by encoding the image and camera parameters.

**parameters**: The parameters of this Function.
· clip_vision: An object responsible for encoding images using a vision model.  
· init_image: The input image that needs to be encoded.  
· vae: A Variational Autoencoder used for encoding pixel data.  
· width: An integer specifying the desired width of the encoded image.  
· height: An integer specifying the desired height of the encoded image.  
· batch_size: An integer representing the number of samples to process in a batch.  
· elevation: A float value representing the elevation angle in degrees for camera embedding.  
· azimuth: A float value representing the azimuth angle in degrees for camera embedding.  

**Code Description**: The encode function begins by utilizing the clip_vision object to encode the input image (init_image) into a feature representation. This is achieved through the method `encode_image`, which outputs image embeddings. The resulting embeddings are then unsqueezed to add a new dimension, preparing them for concatenation with other embeddings.

Next, the function processes the input image to upscale it to the specified width and height using the `common_upscale` function. This function takes care of resizing the image while maintaining the aspect ratio and applying the specified interpolation method. The upscaled image is then sliced to retain only the RGB channels, which are necessary for further processing.

The function then encodes the pixel data using the provided Variational Autoencoder (vae) by calling its `encode` method with the upscaled pixel data. This step generates latent representations of the image.

Following this, the function generates camera embeddings by calling the `camera_embeddings` function with the specified elevation and azimuth parameters. These embeddings represent the camera's orientation in a 3D space and are crucial for conditioning the model on the perspective from which the image was captured.

The final conditioning data is created by concatenating the pooled image embeddings with the camera embeddings, ensuring that the camera embeddings are repeated to match the batch size. This concatenation forms a tensor that combines both image and camera information.

The function prepares two sets of outputs: positive and negative conditioning data. The positive set includes the concatenated embeddings and the latent representation from the VAE, while the negative set consists of zero tensors of the same shape as the pooled embeddings and latent representation. Additionally, a tensor of zeros is created to represent latent samples, which is returned alongside the positive and negative conditioning data.

This function is integral to the workflow of models that require both image and camera information for generating outputs, ensuring that the model can leverage the spatial context provided by the camera parameters along with the visual features of the image.

**Note**: It is important to ensure that the input image (init_image) is in the correct format and that the elevation and azimuth values are provided in degrees, as the function relies on these parameters for generating camera embeddings.

**Output Example**: A possible appearance of the code's return value could be a tuple containing the positive conditioning data, negative conditioning data, and a dictionary with latent samples, such as:
```
(positive, negative, {"samples": tensor of shape (batch_size, 4, height // 8, width // 8)})
```
***
## ClassDef StableZero123_Conditioning_Batched
**StableZero123_Conditioning_Batched**: The function of StableZero123_Conditioning_Batched is to encode images and generate conditioning data for 3D models based on specified parameters.

**attributes**: The attributes of this Class.
· clip_vision: An instance of CLIP_VISION used for encoding images.
· init_image: The initial image to be processed.
· vae: An instance of VAE (Variational Autoencoder) used for encoding pixel data.
· width: The width of the output image, with a default of 256 and constraints on its range.
· height: The height of the output image, with a default of 256 and constraints on its range.
· batch_size: The number of images to process in a single batch, with a default of 1.
· elevation: The elevation angle for camera embeddings, with a default of 0.0.
· azimuth: The azimuth angle for camera embeddings, with a default of 0.0.
· elevation_batch_increment: The increment for elevation in each batch, with a default of 0.0.
· azimuth_batch_increment: The increment for azimuth in each batch, with a default of 0.0.

**Code Description**: The StableZero123_Conditioning_Batched class is designed to facilitate the encoding of images into a format suitable for conditioning 3D models. The primary method, `encode`, takes multiple parameters including a vision model, an initial image, a VAE model, and various configuration settings such as width, height, batch size, and camera angles (elevation and azimuth). 

The `encode` method begins by using the `clip_vision` instance to encode the `init_image`, producing an output that contains image embeddings. These embeddings are then pooled and prepared for further processing. The initial image is upscaled to the specified width and height, and the first three channels of the pixel data are extracted for encoding.

Next, the method generates camera embeddings for each image in the batch. It iteratively adjusts the elevation and azimuth angles based on the specified increments for each batch item. The camera embeddings are concatenated with the pooled image embeddings to create a conditioning tensor.

The method returns three components: a positive conditioning tensor, a negative conditioning tensor (initialized to zeros), and a latent tensor filled with zeros. The positive tensor contains the concatenated conditioning data, while the negative tensor serves as a placeholder for negative conditioning. The latent tensor is structured to match the expected output dimensions for the batch.

**Note**: It is important to ensure that the input parameters adhere to the specified constraints, particularly for width, height, and batch size, to avoid runtime errors. The class is intended for use in scenarios where conditioning data for 3D models is required, and proper initialization of the CLIP_VISION and VAE instances is crucial for successful encoding.

**Output Example**: A possible return value from the `encode` method could look like this:
```python
(
    [[condition_tensor, {"concat_latent_image": latent_tensor}]],
    [[torch.zeros_like(condition_tensor), {"concat_latent_image": torch.zeros_like(latent_tensor)}]],
    {"samples": latent_tensor, "batch_index": [0, 0, 0]}
)
```
Where `condition_tensor` is the concatenated conditioning data, `latent_tensor` is a tensor of zeros with the shape [batch_size, 4, height // 8, width // 8], and `batch_index` indicates the index of each sample in the batch.
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types and their constraints for a specific processing function.

**parameters**: The parameters of this Function.
· s: This parameter is a placeholder and is not utilized within the function body.

**Code Description**: The INPUT_TYPES function constructs a dictionary that specifies the required input types for a particular operation. The dictionary contains a single key, "required", which maps to another dictionary detailing various input parameters. Each input parameter is associated with a tuple that defines its type and additional constraints. 

The following input parameters are defined:

- **clip_vision**: This parameter is expected to be of type "CLIP_VISION".
- **init_image**: This parameter is expected to be of type "IMAGE".
- **vae**: This parameter is expected to be of type "VAE".
- **width**: This parameter is an integer ("INT") with a default value of 256. It has constraints for minimum (16), maximum (defined by ldm_patched.contrib.external.MAX_RESOLUTION), and a step increment of 8.
- **height**: Similar to width, this is also an integer ("INT") with the same constraints and default value.
- **batch_size**: This parameter is an integer ("INT") with a default value of 1, a minimum of 1, and a maximum of 4096.
- **elevation**: This parameter is a floating-point number ("FLOAT") with a default value of 0.0, a minimum of -180.0, and a maximum of 180.0.
- **azimuth**: This parameter is also a floating-point number ("FLOAT") with the same constraints as elevation.
- **elevation_batch_increment**: This is a floating-point number ("FLOAT") with a default value of 0.0 and the same range as elevation.
- **azimuth_batch_increment**: This is a floating-point number ("FLOAT") with the same constraints as azimuth.

The function returns this structured dictionary, which can be utilized by other components of the system to validate and process input data effectively.

**Note**: It is important to ensure that the input values adhere to the specified types and constraints to avoid errors during processing. The maximum resolution for width and height should be defined in the external module referenced.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "clip_vision": ("CLIP_VISION",),
        "init_image": ("IMAGE",),
        "vae": ("VAE",),
        "width": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 8}),
        "height": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 8}),
        "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        "elevation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
        "azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
        "elevation_batch_increment": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
        "azimuth_batch_increment": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
    }
}
***
### FunctionDef encode(self, clip_vision, init_image, vae, width, height, batch_size, elevation, azimuth, elevation_batch_increment, azimuth_batch_increment)
**encode**: The function of encode is to process an initial image and generate a set of embeddings that combine image and camera parameters for further use in a neural network.

**parameters**: The parameters of this Function.
· clip_vision: An object responsible for encoding the initial image into a feature representation.
· init_image: A tensor representing the initial image to be processed.
· vae: A variational autoencoder used to encode the image pixels.
· width: An integer specifying the target width for the encoded image.
· height: An integer specifying the target height for the encoded image.
· batch_size: An integer indicating the number of images to process in a batch.
· elevation: A float value representing the starting elevation angle in degrees for camera embeddings.
· azimuth: A float value representing the starting azimuth angle in degrees for camera embeddings.
· elevation_batch_increment: A float value indicating the increment for the elevation angle for each image in the batch.
· azimuth_batch_increment: A float value indicating the increment for the azimuth angle for each image in the batch.

**Code Description**: The encode function begins by utilizing the clip_vision object to encode the initial image (init_image) into a feature representation, specifically extracting the image embeddings. The output embeddings are then unsqueezed to add an additional dimension, preparing them for concatenation with camera embeddings.

Next, the function processes the initial image to upscale it to the specified width and height using the common_upscale utility. This step ensures that the image dimensions are suitable for further processing. The resulting pixel data is then sliced to retain only the RGB channels, which are subsequently encoded using the variational autoencoder (vae).

The function then initializes an empty list to store camera embeddings. A loop iterates over the specified batch size, generating camera embeddings for each image by calling the camera_embeddings function with the current elevation and azimuth values. The elevation and azimuth angles are incremented for each iteration to create a diverse set of camera perspectives.

After generating the camera embeddings, the function concatenates the pooled image embeddings with the camera embeddings, ensuring that both sets of embeddings have the same batch size by utilizing the repeat_to_batch_size utility. This concatenated tensor serves as the conditioning input for the model.

The function prepares two sets of outputs: a positive set containing the concatenated embeddings and the encoded image tensor, and a negative set initialized with zeros. Finally, it returns these outputs along with a latent tensor initialized to zeros, which represents the latent space for the batch of images.

This encode function is integral to the processing pipeline within the StableZero123_Conditioning_Batched class, where it facilitates the generation of embeddings that incorporate both visual and spatial information, enabling the model to learn from diverse perspectives.

**Note**: It is crucial to ensure that the input image is properly formatted and that the elevation and azimuth parameters are provided in degrees. The function assumes that the input dimensions and types are consistent with the expected shapes for processing.

**Output Example**: A possible appearance of the code's return value could be a tuple containing two lists and a dictionary, such as:
```
(positive, negative, {"samples": latent, "batch_index": [0] * batch_size})
```
Where `positive` contains the concatenated embeddings and encoded image, `negative` contains zeroed embeddings, and `latent` is a tensor of shape (batch_size, 4, height // 8, width // 8) initialized to zeros.
***
