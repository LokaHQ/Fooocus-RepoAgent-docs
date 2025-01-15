## ClassDef SD_4XUpscale_Conditioning
**SD_4XUpscale_Conditioning**: The function of SD_4XUpscale_Conditioning is to upscale images while conditioning them with positive and negative inputs.

**attributes**: The attributes of this Class.
· INPUT_TYPES: A class method that defines the required input types for the encoding function, including images, positive and negative conditioning, scale ratio, and noise augmentation.
· RETURN_TYPES: A tuple indicating the types of outputs returned by the encode method, which are CONDITIONING, CONDITIONING, and LATENT.
· RETURN_NAMES: A tuple containing the names of the returned outputs, which are positive, negative, and latent.
· FUNCTION: A string that specifies the name of the function to be executed, which is "encode".
· CATEGORY: A string that categorizes the functionality of the class, specifically under "conditioning/upscale_diffusion".

**Code Description**: The SD_4XUpscale_Conditioning class is designed to perform image upscaling while applying conditioning based on provided positive and negative inputs. The class defines a method called `encode`, which takes several parameters: images, positive conditioning, negative conditioning, scale ratio, and noise augmentation. 

The `INPUT_TYPES` class method specifies the required inputs, where:
- "images" must be of type IMAGE,
- "positive" and "negative" must be of type CONDITIONING,
- "scale_ratio" is a FLOAT with a default value of 4.0, and it must be between 0.0 and 10.0,
- "noise_augmentation" is also a FLOAT with a default value of 0.0, constrained between 0.0 and 1.0.

The `encode` method begins by calculating the new width and height of the images based on the provided scale ratio. It ensures that the dimensions are at least 1 by rounding the original dimensions multiplied by the scale ratio. The method then uses a utility function to upscale the image pixels using bilinear interpolation.

Next, the method prepares the output for positive and negative conditioning. For each element in the positive conditioning list, it creates a new entry that includes the original conditioning data along with the upscaled image pixels and the specified noise augmentation. The same process is repeated for the negative conditioning list.

Finally, the method initializes a latent tensor filled with zeros, which has dimensions based on the number of images and the upscaled dimensions. The method returns a tuple containing the processed positive conditioning, negative conditioning, and the latent representation.

**Note**: It is important to ensure that the input images and conditioning data are correctly formatted and that the scale ratio and noise augmentation parameters are set within their defined limits to avoid errors during processing.

**Output Example**: An example of the output returned by the `encode` method could look like this:
(
    [   # List of positive conditioning outputs
        (positive_conditioning_id_1, {'concat_image': upscaled_pixels, 'noise_augmentation': 0.1}),
        (positive_conditioning_id_2, {'concat_image': upscaled_pixels, 'noise_augmentation': 0.1}),
    ],
    [   # List of negative conditioning outputs
        (negative_conditioning_id_1, {'concat_image': upscaled_pixels, 'noise_augmentation': 0.1}),
        (negative_conditioning_id_2, {'concat_image': upscaled_pixels, 'noise_augmentation': 0.1}),
    ],
    {'samples': latent_tensor}  # Latent representation
)
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types for a specific processing function in the context of image conditioning.

**parameters**: The parameters of this Function.
· s: This parameter is typically a placeholder for the state or context in which the function is executed, though it is not utilized within the function body.

**Code Description**: The INPUT_TYPES function constructs a dictionary that specifies the required input types for an image processing operation. The dictionary returned by the function contains a single key, "required", which maps to another dictionary detailing the specific inputs needed. Each input is associated with its type and, where applicable, additional constraints or default values. 

The inputs defined are as follows:
- "images": This input is expected to be of type "IMAGE", indicating that the function requires image data for processing.
- "positive": This input is also of type "CONDITIONING", which suggests that it is used for conditioning the positive aspects of the image.
- "negative": Similar to "positive", this input is of type "CONDITIONING" and is used for conditioning the negative aspects of the image.
- "scale_ratio": This input is of type "FLOAT" and has constraints defined within a dictionary. The default value is set to 4.0, with a minimum value of 0.0, a maximum value of 10.0, and a step increment of 0.01. This parameter likely controls the scaling factor applied during processing.
- "noise_augmentation": This input is also of type "FLOAT" and includes constraints similar to "scale_ratio". The default value is 0.0, with a minimum of 0.0, a maximum of 1.0, and a step increment of 0.001. This parameter likely adjusts the level of noise added to the images during processing.

**Note**: It is important to ensure that all required inputs are provided in the correct format and within the specified constraints to avoid errors during processing.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "images": ("IMAGE",),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "scale_ratio": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.01}),
        "noise_augmentation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
    }
}
***
### FunctionDef encode(self, images, positive, negative, scale_ratio, noise_augmentation)
**encode**: The function of encode is to process a batch of images by upscaling them and preparing them for further analysis by associating them with positive and negative samples.

**parameters**: The parameters of this Function.
· images: A tensor of shape (N, C, H, W) representing a batch of images, where N is the number of samples, C is the number of channels, H is the height, and W is the width.  
· positive: A list of positive sample tuples, where each tuple contains an identifier and a dictionary of attributes related to the sample.  
· negative: A list of negative sample tuples, similar to positive but representing negative samples.  
· scale_ratio: A float that determines the factor by which the images will be upscaled.  
· noise_augmentation: A boolean or a parameter that specifies whether to apply noise augmentation to the images.

**Code Description**: The encode function begins by calculating the new dimensions (width and height) for the upscaled images based on the provided scale_ratio. It ensures that the new dimensions are at least 1 pixel in size. The function then calls the common_upscale function from the ldm_patched.modules.utils module to upscale the input images. The images are first transformed by moving the last dimension to the second position and scaling the pixel values to the range [-1, 1].

The upscaled pixel data is stored in the variable `pixels`, which is then used to update the attributes of the positive and negative sample tuples. For each positive sample, a copy of the sample's attributes is created, and the upscaled image data along with the noise augmentation parameter is added to the dictionary. This updated tuple is appended to the out_cp list. A similar process is followed for the negative samples, resulting in the out_cn list.

Finally, the function initializes a latent tensor filled with zeros, which has a shape corresponding to the number of images and their upscaled dimensions. The function returns a tuple containing the lists of processed positive and negative samples, along with the latent tensor.

The encode function is integral to the image processing workflow, as it prepares the images for subsequent operations by ensuring they are correctly upscaled and associated with their respective sample identifiers and attributes. It relies on the common_upscale function to handle the actual resizing of the images, thus maintaining a modular approach to image processing.

**Note**: It is important to ensure that the input images are in the correct shape and data type before calling the encode function. Additionally, the scale_ratio should be a positive value to avoid invalid dimensions during the upscaling process.

**Output Example**: Given an input tensor of shape (2, 3, 64, 64) representing two images with 3 color channels and a size of 64x64, calling encode with positive samples and negative samples would return a tuple containing two lists of processed samples and a latent tensor of shape (2, 4, 16, 16), where the images have been upscaled and prepared for further analysis.
***
