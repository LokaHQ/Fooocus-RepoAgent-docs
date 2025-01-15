## FunctionDef erode_or_dilate(x, k)
**erode_or_dilate**: The function of erode_or_dilate is to perform morphological operations on an image, specifically erosion or dilation, based on the input parameters.

**parameters**: The parameters of this Function.
· x: The input image on which the morphological operation will be performed. It is expected to be a NumPy array representing the image data.
· k: An integer that determines the type of operation. A positive value indicates dilation, a negative value indicates erosion, and zero indicates no operation.

**Code Description**: The erode_or_dilate function is designed to apply morphological transformations to an image using the OpenCV library. The function begins by converting the parameter k to an integer. It then checks the value of k to determine which operation to perform. If k is greater than zero, the function applies dilation to the input image x using a 3x3 kernel of ones, repeating the operation k times. Dilation is a process that enlarges the boundaries of objects in an image, which can be useful for closing small holes or gaps. Conversely, if k is less than zero, the function applies erosion to the image, using the same kernel but repeating the operation -k times. Erosion reduces the boundaries of objects, which can help in removing small-scale noise or separating objects that are close together. If k is zero, the function simply returns the original image without any modifications.

The erode_or_dilate function is called within the apply_image_input function of the async_worker module. Specifically, it is invoked when the inpaint_erode_or_dilate parameter of the async_task is not zero. This indicates that the user has requested a morphological operation to be applied to the inpainting mask before further processing. The resulting mask is then used in the inpainting process, which is part of the image enhancement workflow.

**Note**: It is important to ensure that the input image x is in the correct format (a NumPy array) and that the k parameter is an integer. The function assumes that the input image is a binary mask when performing erosion or dilation.

**Output Example**: If the input image is a binary mask with small holes, applying erode_or_dilate with k = 1 might result in a mask where those holes are closed, while applying it with k = -1 could separate connected components in the mask.
## FunctionDef resample_image(im, width, height)
**resample_image**: The function of resample_image is to resize an image to specified dimensions using a high-quality resampling filter.

**parameters**: The parameters of this Function.
· im: A NumPy array representing the image to be resized.
· width: The desired width of the output image.
· height: The desired height of the output image.

**Code Description**: The resample_image function takes an image represented as a NumPy array and resizes it to the specified width and height. It first converts the NumPy array into a PIL Image object using `Image.fromarray(im)`. This conversion is necessary because the resizing operation is performed using the PIL library, which provides various resampling filters for high-quality image resizing.

The function then uses the `resize` method of the PIL Image object to change the dimensions of the image. The resampling filter used is `LANCZOS`, which is known for producing high-quality results, especially when downsampling images. After resizing, the function converts the PIL Image back into a NumPy array using `np.array(im)` and returns this array as the output.

This function is called within the `apply_upscale` function in the modules/async_worker.py file. Specifically, it is used to resize images during the upscaling process. The `apply_upscale` function checks the desired upscale factor and uses resample_image to ensure that the image is resized appropriately before further processing. Additionally, it is also called in the `apply_image_input` function to resize masks associated with inpainting tasks, ensuring that the mask dimensions match the input image dimensions.

**Note**: When using this function, it is important to ensure that the input image is in the correct format (a NumPy array) and that the specified width and height are valid integers. The quality of the resized image will depend on the choice of the resampling filter, with LANCZOS being a suitable option for most cases.

**Output Example**: A possible appearance of the code's return value could be a NumPy array representing a resized image, for example, an array of shape (256, 256, 3) for an image resized to 256 pixels in height and 256 pixels in width with three color channels (RGB).
## FunctionDef resize_image(im, width, height, resize_mode)
**resize_image**: The function of resize_image is to resize an image to specified dimensions while allowing for different resizing modes.

**parameters**: The parameters of this Function.
· im: The image to resize, provided as a NumPy array.
· width: The target width to resize the image to.
· height: The target height to resize the image to.
· resize_mode: The mode to use when resizing the image, which can be:
  0: Resize the image to the specified width and height without maintaining the aspect ratio.
  1: Resize the image to fill the specified width and height while maintaining the aspect ratio, centering the image within the dimensions and cropping any excess.
  2: Resize the image to fit within the specified width and height while maintaining the aspect ratio, centering the image within the dimensions and filling any empty space with data from the image.

**Code Description**: The resize_image function takes an image and resizes it according to the specified width, height, and resize_mode. The function begins by converting the input image from a NumPy array to a PIL Image object. It defines a nested function, resize, which performs the actual resizing using the LANCZOS resampling filter for high-quality results.

The function then evaluates the resize_mode parameter:
- If resize_mode is 0, it directly resizes the image to the specified dimensions without preserving the aspect ratio.
- If resize_mode is 1, it calculates the appropriate dimensions to maintain the aspect ratio, resizes the image accordingly, and then centers it within a new blank image of the specified dimensions, cropping any excess.
- If resize_mode is 2, it similarly calculates the dimensions to maintain the aspect ratio, resizes the image, and centers it within a new blank image, but fills any empty space with data from the resized image.

Finally, the function returns the resized image as a NumPy array.

This function is called within the apply_control_nets function in the async_worker module. Specifically, it is used to preprocess images for various control net tasks, ensuring that the images are resized to the appropriate dimensions before further processing. This is crucial for maintaining consistency in image dimensions across different tasks, which is essential for the subsequent operations that rely on these images.

**Note**: It is important to ensure that the input image is in the correct format (NumPy array) and that the specified width and height are appropriate for the intended use case. The resize_mode should be chosen based on the desired outcome regarding aspect ratio preservation and image centering.

**Output Example**: A possible appearance of the code's return value could be a NumPy array representing a resized image, for instance, an array of shape (height, width, 3) for a color image, where each pixel is represented by three values corresponding to the RGB color channels.
### FunctionDef resize(im, w, h)
**resize**: The function of resize is to change the dimensions of an image to specified width and height.

**parameters**: The parameters of this Function.
· parameter1: im - This is the image object that you want to resize. It should be an instance of an image class that supports the resize method, typically from the PIL (Pillow) library.
· parameter2: w - This is an integer representing the desired width of the resized image.
· parameter3: h - This is an integer representing the desired height of the resized image.

**Code Description**: The resize function takes an image object and two integers as input parameters. It utilizes the resize method of the image object to change its dimensions to the specified width (w) and height (h). The resizing process employs the LANCZOS resampling filter, which is known for producing high-quality results, especially when reducing the size of an image. The function returns a new image object that has been resized to the specified dimensions.

**Note**: When using this function, ensure that the image object passed as the first parameter is properly loaded and compatible with the resize method. The LANCZOS filter is particularly effective for downscaling images, but it may not be the best choice for all types of images or resizing scenarios. It is advisable to test the output to ensure it meets the desired quality standards.

**Output Example**: If the function is called with an image object `img`, a width of 200, and a height of 100, the return value would be a new image object that has dimensions of 200 pixels in width and 100 pixels in height.
***
## FunctionDef get_shape_ceil(h, w)
**get_shape_ceil**: The function of get_shape_ceil is to calculate the smallest multiple of 64 that is greater than or equal to the square root of the product of height and width.

**parameters**: The parameters of this Function.
· h: The height of the image or shape.
· w: The width of the image or shape.

**Code Description**: The get_shape_ceil function takes two parameters, h and w, which represent the height and width of an image or shape, respectively. It computes the square root of the product of these two dimensions, divides the result by 64.0, and then applies the math.ceil function to round up to the nearest whole number. Finally, it multiplies this result by 64.0 to ensure that the output is a multiple of 64. This function is particularly useful in scenarios where dimensions need to conform to specific constraints, such as when preparing images for processing in machine learning models or graphical applications that require dimensions to be aligned to certain grid sizes.

The get_shape_ceil function is called within other functions in the project, such as apply_upscale and get_image_shape_ceil. In apply_upscale, it is used to determine the appropriate dimensions for an image after it has been upscaled, ensuring that the new dimensions meet the required constraints. Similarly, get_image_shape_ceil utilizes get_shape_ceil to derive the shape ceiling based on the dimensions of an image passed to it. This interconnectedness highlights the importance of get_shape_ceil in maintaining consistent and valid dimensions throughout the image processing workflow.

**Note**: It is important to ensure that the input parameters h and w are positive integers, as negative or zero values would lead to invalid calculations and potentially raise errors.

**Output Example**: For example, if h = 150 and w = 200, the function would calculate:
1. sqrt(150 * 200) = sqrt(30000) ≈ 173.205
2. 173.205 / 64.0 ≈ 2.707
3. math.ceil(2.707) = 3
4. 3 * 64.0 = 192.0

Thus, the output of get_shape_ceil(150, 200) would be 192.0.
## FunctionDef get_image_shape_ceil(im)
**get_image_shape_ceil**: The function of get_image_shape_ceil is to calculate the shape ceiling for an image based on its dimensions.

**parameters**: The parameters of this Function.
· im: A numpy array representing the image whose shape ceiling is to be calculated.

**Code Description**: The get_image_shape_ceil function takes a single parameter, im, which is expected to be a numpy array representing an image. The function extracts the height (H) and width (W) of the image from its shape attribute. It then calls the get_shape_ceil function, passing the height and width as arguments. The purpose of get_shape_ceil is to compute the smallest multiple of 64 that is greater than or equal to the square root of the product of the height and width of the image. This is particularly useful in image processing tasks where dimensions need to conform to specific constraints, such as when preparing images for machine learning models or graphical applications that require dimensions to be aligned to certain grid sizes.

The get_image_shape_ceil function is called within the apply_vary function in the async_worker module. In this context, it is used to determine the appropriate shape ceiling for an input image before further processing, ensuring that the image dimensions meet the necessary requirements for subsequent operations.

**Note**: It is important to ensure that the input parameter im is a valid numpy array representing an image. The function assumes that the image has at least two dimensions (height and width). If the input does not meet these criteria, it may lead to errors during execution.

**Output Example**: For example, if the input image has a shape of (150, 200, 3), the function would calculate:
1. H = 150
2. W = 200
3. The output of get_shape_ceil(150, 200) would be computed based on the logic defined in the get_shape_ceil function, resulting in a shape ceiling value that is a multiple of 64.
## FunctionDef set_image_shape_ceil(im, shape_ceil)
**set_image_shape_ceil**: The function of set_image_shape_ceil is to adjust the dimensions of an image to the nearest multiple of 64 that is less than or equal to a specified ceiling value.

**parameters**: The parameters of this Function.
· im: A NumPy array representing the image whose dimensions are to be adjusted.
· shape_ceil: A numeric value representing the desired maximum dimension for the image.

**Code Description**: The set_image_shape_ceil function begins by converting the shape_ceil parameter to a float. It then retrieves the original dimensions (height and width) of the input image, im. A loop is initiated to iteratively adjust the height (H) and width (W) of the image until the calculated shape ceiling, obtained from the get_shape_ceil function, is sufficiently close to the specified shape_ceil. The loop runs a maximum of 256 iterations to prevent infinite loops in case of unexpected behavior.

Within the loop, the current shape ceiling is calculated using the get_shape_ceil function, which computes the smallest multiple of 64 that is greater than or equal to the square root of the product of the current height and width. If the absolute difference between the current shape ceiling and the specified shape_ceil is less than 0.1, the loop breaks, indicating that the dimensions are close enough to the desired ceiling.

If the dimensions of the image have changed (i.e., if H is not equal to H_origin or W is not equal to W_origin), the function calls the resample_image function to resize the image to the new dimensions. The resample_image function is responsible for resizing the image using a high-quality resampling filter, ensuring that the output image maintains good visual quality.

This function is called within other functions in the project, such as apply_vary and apply_upscale. In apply_vary, it is used to ensure that the input image dimensions conform to a specified ceiling before further processing. Similarly, in apply_upscale, it adjusts the dimensions of the image after an upscale operation to meet the required constraints for subsequent processing steps.

**Note**: It is important to ensure that the input image is in the correct format (a NumPy array) and that the specified shape_ceil is a valid numeric value. The function is designed to handle images of varying sizes, but the final dimensions will always be multiples of 64, which is a common requirement in image processing tasks.

**Output Example**: A possible appearance of the code's return value could be a NumPy array representing an image resized to dimensions that are multiples of 64, for example, an array of shape (1024, 1024, 3) for an image resized to 1024 pixels in height and 1024 pixels in width with three color channels (RGB).
## FunctionDef HWC3(x)
**HWC3**: The function of HWC3 is to convert an image to a height-width-channel format while ensuring the correct number of channels.

**parameters**: The parameters of this Function.
· x: A NumPy array representing the image, which must be of data type uint8.

**Code Description**: The HWC3 function is designed to process images represented as NumPy arrays. It begins by asserting that the input image `x` is of the correct data type, specifically `np.uint8`, which is commonly used for image data. The function checks the number of dimensions of the input image. If the image is two-dimensional (i.e., a grayscale image), it adds a third dimension to convert it into a three-dimensional format by repeating the single channel. The function then asserts that the image is now three-dimensional and extracts its height (H), width (W), and number of channels (C).

The function ensures that the number of channels is either 1, 3, or 4. If the image has 3 channels (indicating it is already in RGB format), it returns the image as is. If the image has only 1 channel (grayscale), it duplicates this channel three times to create an RGB image. If the image has 4 channels (RGBA), it processes the image to blend the color channels with the alpha channel, effectively creating a new image that combines the color information with transparency. The resulting image is clipped to ensure pixel values remain within the valid range (0-255) and is returned as a uint8 array.

The HWC3 function is called in various parts of the project, notably within the `apply_control_nets` and `apply_image_input` functions in the `modules/async_worker.py/worker` module. In `apply_control_nets`, HWC3 is used to preprocess images before applying control net tasks, ensuring that the images are in the correct format for further processing. In `apply_image_input`, it is utilized to prepare images for inpainting and other tasks, ensuring that the images conform to the expected height-width-channel format.

**Note**: It is crucial to ensure that the input image is of the correct data type and dimensions before calling this function. Any deviation from the expected format may result in assertion errors or unexpected behavior.

**Output Example**: A possible appearance of the code's return value could be a NumPy array representing an RGB image with dimensions (height, width, 3) or an RGBA image with dimensions (height, width, 4), depending on the input provided. For example, an input grayscale image of shape (256, 256) would be transformed to an output shape of (256, 256, 3) after processing.
## FunctionDef remove_empty_str(items, default)
**remove_empty_str**: The function of remove_empty_str is to filter out empty strings from a list and optionally return a default value if the resulting list is empty.

**parameters**: The parameters of this Function.
· items: A list of strings that may contain empty strings.
· default: An optional value that will be returned as a list if the filtered list is empty.

**Code Description**: The remove_empty_str function takes a list of strings as input and removes any empty strings from that list. It uses a list comprehension to iterate through the provided list, retaining only those elements that are not empty. If the resulting list is empty and a default value is provided, the function returns a list containing that default value. If the resulting list is not empty, it simply returns the filtered list.

This function is particularly useful in scenarios where empty strings may cause issues or where a default value is needed to ensure that a list always contains at least one element. In the context of the project, remove_empty_str is called within the process_prompt function of the worker module. It is used to process the prompts and negative prompts by filtering out any empty lines before further processing. This ensures that the prompts passed to the model are valid and meaningful, preventing potential errors or unexpected behavior during model inference.

**Note**: When using remove_empty_str, it is important to consider the implications of the default parameter. If the default value is not provided and the input list contains only empty strings, the function will return an empty list, which may not be the desired outcome in some cases.

**Output Example**: 
If the input to remove_empty_str is `["Hello", "", "World", ""]` and the default is `"Default"`, the output will be `["Hello", "World"]`. If the input is `["", ""]` and the default is `"Default"`, the output will be `["Default"]`.
## FunctionDef join_prompts
**join_prompts**: The function of join_prompts is to concatenate a variable number of string inputs into a single string, separated by commas.

**parameters**: The parameters of this Function.
· *args: A variable number of arguments that can be of any type, which will be converted to strings.
· **kwargs: A variable number of keyword arguments that are not used in the function but can be accepted.

**Code Description**: The join_prompts function takes a variable number of arguments and processes them to create a single string output. It begins by converting each argument in *args to a string and filtering out any empty strings. The resulting list, named prompts, contains only non-empty string representations of the input arguments. 

If the prompts list is empty (i.e., no valid input was provided), the function returns an empty string. If there is only one valid prompt, it returns that single prompt. If there are multiple valid prompts, it joins them into a single string, with each prompt separated by a comma and a space. This function is useful for creating a formatted string from a collection of inputs, ensuring that only meaningful data is included in the output.

**Note**: It is important to ensure that the inputs provided are convertible to strings. The function does not handle any exceptions related to type conversion, so passing non-convertible types may lead to unexpected behavior.

**Output Example**: 
- If the function is called as join_prompts("Hello", "", "World", "Python"), the output will be "Hello, World, Python".
- If called with no arguments, join_prompts() will return an empty string "".
- If called with a single argument, join_prompts("Single") will return "Single".
## FunctionDef generate_temp_filename(folder, extension)
**generate_temp_filename**: The function of generate_temp_filename is to create a temporary filename for storing output files, including a timestamp and a random number for uniqueness.

**parameters**: The parameters of this Function.
· folder: A string representing the directory where the temporary file will be saved. The default value is './outputs/'.
· extension: A string representing the file extension for the temporary file. The default value is 'png'.

**Code Description**: The generate_temp_filename function is designed to generate a unique temporary filename based on the current date and time, along with a random number. It begins by retrieving the current date and time using the datetime module. The date is formatted as "YYYY-MM-DD" and the time as "YYYY-MM-DD_HH-MM-SS". A random integer between 1000 and 9999 is generated to ensure that the filename remains unique even if multiple files are created in quick succession.

The filename is constructed by concatenating the time string, the random number, and the specified file extension. The function then combines the folder path, date string, and filename to create the full path for the temporary file. Finally, it returns a tuple containing the date string, the absolute path of the generated filename, and the filename itself.

This function is called by other functions within the project, specifically in modules/private_logger.py. For instance, in the get_current_html_path function, generate_temp_filename is used to create a temporary filename for storing HTML logs. The generated filename is then utilized to construct the path for the HTML file that will log metadata and images.

In the log function, generate_temp_filename is also called to create a temporary filename for saving images. The function ensures that the output directory exists before saving the image, and it handles various image formats based on the specified output format. The generated filename is crucial for organizing and storing logs and images systematically.

**Note**: It is important to ensure that the specified folder exists before calling this function, as it does not create the folder structure. The function assumes that the provided folder path is valid and accessible.

**Output Example**: An example output of the function could be a tuple like the following:
('2023-10-05', '/absolute/path/to/outputs/2023-10-05/2023-10-05_14-30-15_1234.png', '2023-10-05_14-30-15_1234.png')
## FunctionDef sha256(filename, use_addnet_hash, length)
**sha256**: The function of sha256 is to compute the SHA-256 hash of a specified file, with an option to use an alternative hashing method for safetensors formatted files.

**parameters**: The parameters of this Function.
· filename: A string representing the path to the file for which the SHA-256 hash needs to be calculated.
· use_addnet_hash: A boolean indicating whether to use the addnet_hash_safetensors function for computing the hash instead of the standard method.
· length: An optional integer specifying the length of the hash to return; if None, the full hash is returned.

**Code Description**: The sha256 function is designed to compute the SHA-256 hash of a file specified by the filename parameter. It provides flexibility by allowing the user to choose between two methods of hash calculation: the standard method using the calculate_sha256 function or an alternative method using the addnet_hash_safetensors function when the use_addnet_hash parameter is set to True.

When use_addnet_hash is True, the function opens the specified file in binary read mode and passes the file object to the addnet_hash_safetensors function. This function is specifically tailored to handle binary streams formatted as safetensors, ensuring that the hash is computed correctly based on the structure of the data.

If use_addnet_hash is False, the sha256 function calls the calculate_sha256 function, which reads the file in chunks and computes the hash using the standard SHA-256 algorithm. This method is suitable for general file types and is efficient for large files due to its chunked reading approach.

The sha256 function also includes an optional length parameter, which allows the user to specify a desired length for the resulting hash. If length is provided, the function returns only the first 'length' characters of the computed hash. If length is None, the full hash is returned.

This function is called by the sha256_from_cache function located in the modules/hash_cache.py file. The sha256_from_cache function checks if the hash for a given filepath is already cached. If not, it invokes the sha256 function to calculate the hash, stores it in the cache, and saves the cache to a file for future reference. This caching mechanism enhances performance by avoiding redundant hash calculations for files that have already been processed.

**Note**: It is important to ensure that the specified file exists and is accessible; otherwise, a FileNotFoundError will be raised. Additionally, when using the addnet_hash_safetensors method, the binary stream must be correctly formatted as safetensors to ensure accurate hash calculations.

**Output Example**: An example of the output from the sha256 function could be a string like "6dcd4ce23d88e2ee9568ba546c007c63a5c3e1c2f2c9c5e8c3a8e2c7c5e7e1c0", which represents the SHA-256 hash of the file's contents. If the length parameter is set to 8, the output might be "6dcd4ce2".
## FunctionDef addnet_hash_safetensors(b)
**addnet_hash_safetensors**: The function of addnet_hash_safetensors is to compute the SHA-256 hash of a binary stream formatted as safetensors.

**parameters**: The parameters of this Function.
· b: A binary stream (file-like object) from which the hash will be calculated.

**Code Description**: The addnet_hash_safetensors function is designed to compute the SHA-256 hash of a binary file that follows the safetensors format. The function begins by initializing a SHA-256 hash object using the hashlib library. It sets a block size of 1 MB for reading the binary data in chunks. 

The function first seeks to the beginning of the binary stream and reads the first 8 bytes, which are expected to represent the size of the data in little-endian format. This size is then used to calculate the offset for the actual data, which is read in chunks until the end of the stream is reached. Each chunk read is updated into the SHA-256 hash object. Finally, the function returns the hexadecimal digest of the hash, which represents the computed SHA-256 hash of the entire binary content.

This function is called by the sha256 function located in the same module. When the parameter `use_addnet_hash` is set to True, the sha256 function opens the specified file in binary read mode and passes the file object to addnet_hash_safetensors to compute the hash. If `use_addnet_hash` is False, the sha256 function calls a different method, calculate_sha256, to compute the hash.

**Note**: It is important to ensure that the binary stream passed to addnet_hash_safetensors is correctly formatted as safetensors, as the function relies on the first 8 bytes to determine the size of the data. Improper formatting may lead to incorrect hash calculations.

**Output Example**: An example of the output from addnet_hash_safetensors could be a string like "3a1f9d4e0e7c8b1f4c8d5e4f2b3a1e1f6c4b5a1e2d3c4b5a6e7f8c9d0e1f2a3", which represents the SHA-256 hash of the binary data processed.
## FunctionDef calculate_sha256(filename)
**calculate_sha256**: The function of calculate_sha256 is to compute the SHA-256 hash of a file specified by its filename.

**parameters**: The parameters of this Function.
· filename: A string representing the path to the file for which the SHA-256 hash needs to be calculated.

**Code Description**: The calculate_sha256 function takes a single parameter, filename, which should be the path to a file on the filesystem. It initializes a SHA-256 hash object using the hashlib library. The function reads the file in chunks of 1 MB (1024 * 1024 bytes) to efficiently handle large files. For each chunk read from the file, it updates the hash object with the chunk's data. Once the entire file has been processed, the function returns the hexadecimal digest of the hash, which is a string representation of the computed SHA-256 hash.

This function is called by the sha256 function located in the same module (modules/util.py). In the sha256 function, if the parameter use_addnet_hash is set to True, it computes the hash using an alternative method (addnet_hash_safetensors). If use_addnet_hash is False, it directly calls calculate_sha256 to obtain the SHA-256 hash of the specified file. The sha256 function also allows for an optional length parameter, which can truncate the resulting hash to a specified length.

**Note**: It is important to ensure that the file specified by the filename parameter exists and is accessible; otherwise, a FileNotFoundError will be raised. Additionally, the function assumes that the file is in binary format, as it opens the file with the "rb" mode.

**Output Example**: An example of the output returned by calculate_sha256 when called with a valid filename could be a string like "6dcd4ce23d88e2ee9568ba546c007c63a5c3e1c2f2c9c5e8c3a8e2c7c5e7e1c0", which represents the SHA-256 hash of the file's contents.
## FunctionDef quote(text)
**quote**: The function of quote is to format a given text into a JSON-compatible string if it contains specific characters.

**parameters**: The parameters of this Function.
· text: The input text that needs to be formatted.

**Code Description**: The quote function checks if the provided text contains any of the following characters: a comma (','), a newline ('\n'), or a colon (':'). If none of these characters are present, the function simply returns the original text as is. However, if any of these characters are found, the function utilizes the json.dumps method to convert the text into a JSON string. The ensure_ascii=False parameter is specified to allow for the inclusion of non-ASCII characters in the output, ensuring that the text is preserved in its original form without escaping.

This function is utilized within the to_string method of the A1111MetadataParser class found in the modules/meta_parser.py file. In this context, the quote function is called to format values from a dictionary before they are concatenated into a string representation of metadata. Specifically, it is used to ensure that any values that may contain special characters are properly formatted, preventing potential issues with string representation in the final output.

**Note**: It is important to ensure that the input to the quote function is a string or can be converted to a string, as the function relies on string operations to check for the presence of specific characters.

**Output Example**: 
- If the input is "Hello, World", the output will be: `"Hello, World"`
- If the input is "Hello World", the output will be: `Hello World`
## FunctionDef unquote(text)
**unquote**: The function of unquote is to decode a JSON string that is enclosed in double quotes.

**parameters**: The parameters of this Function.
· text: A string that may be a JSON-encoded value enclosed in double quotes.

**Code Description**: The unquote function checks if the input string, referred to as 'text', is empty or does not start and end with double quotes. If either condition is true, it returns the original string unchanged. If the string is properly formatted with double quotes, the function attempts to parse it as a JSON object using the json.loads method. If the parsing is successful, it returns the resulting object. If an exception occurs during parsing, it catches the exception and returns the original string instead. 

This function is utilized within the to_json method of the A1111MetadataParser class found in the modules/meta_parser.py file. In this context, unquote is called to process values that are expected to be JSON strings. Specifically, it checks if the value starts and ends with double quotes before attempting to decode it. This ensures that any valid JSON string is correctly parsed into a Python object, while any invalid or non-JSON strings are returned as-is, preventing errors in the overall data processing workflow.

**Note**: It is important to ensure that the input string is formatted correctly as a JSON string for successful parsing. If the input does not meet the expected format, the function will return the original string, which may lead to further processing issues if not handled appropriately.

**Output Example**: If the input to unquote is '"{"key": "value"}"', the output will be {'key': 'value'}. If the input is 'not a json string', the output will be 'not a json string'.
## FunctionDef unwrap_style_text_from_prompt(style_text, prompt)
**unwrap_style_text_from_prompt**: The function of unwrap_style_text_from_prompt is to check if the provided style text wraps around the prompt and return a modified prompt without the style text if it does.

**parameters**: The parameters of this Function.
· parameter1: style_text - A string representing the style text that may wrap around the prompt.
· parameter2: prompt - A string representing the prompt that is being checked against the style text.

**Code Description**: The unwrap_style_text_from_prompt function is designed to analyze the relationship between a given style text and a prompt. It checks if the prompt is encapsulated within the style text, specifically looking for the placeholder "{prompt}" within the style text. If the placeholder is found, the function attempts to split the style text into two parts: the text before and after the placeholder. It then searches for these parts in the prompt to determine if the prompt is indeed wrapped by the style text.

If the prompt is successfully identified as being wrapped, the function extracts the inner prompt text (the part of the prompt that is not part of the style text) and returns a tuple containing:
1. A boolean value indicating that the style text wraps the prompt (True).
2. The modified prompt with the style text removed.
3. The real prompt text that was extracted.

If the style text does not contain the placeholder or if the prompt does not match the expected structure, the function checks if the prompt ends with the style text. If it does, it returns a similar tuple indicating that the style text is at the end of the prompt.

The function is called by extract_original_prompts, which uses it to determine if the provided style and negative prompt match the respective style texts. If both checks are successful, extract_original_prompts returns the modified prompts without the style text. If either check fails, it returns the original prompts, indicating that the style did not match.

**Note**: It is important to ensure that the style text contains the placeholder "{prompt}" only once. If it appears multiple times, the function will raise an error and print a message indicating the issue, but it will not modify the original style text.

**Output Example**: 
For example, if the style_text is "Elegant: {prompt}" and the prompt is "Elegant: A beautiful sunset", the function would return:
(True, "A beautiful sunset", "A beautiful sunset")
## FunctionDef extract_original_prompts(style, prompt, negative_prompt)
**extract_original_prompts**: The function of extract_original_prompts is to evaluate a given style against a prompt and a negative prompt, returning modified prompts if a match is found.

**parameters**: The parameters of this Function.
· parameter1: style - An object representing the style, which contains attributes for prompt and negative prompt.
· parameter2: prompt - A string representing the main prompt that is being evaluated.
· parameter3: negative_prompt - A string representing the negative prompt that is being evaluated.

**Code Description**: The extract_original_prompts function is designed to determine if a specified style matches the provided prompt and negative prompt. It first checks if both the style's prompt and negative prompt are empty. If they are, the function immediately returns False along with the original prompt and negative prompt, indicating that no match can be made.

Next, the function utilizes the unwrap_style_text_from_prompt function to assess whether the style's prompt wraps around the main prompt. If a match is found, it extracts the inner prompt text and returns True, along with the modified prompt that excludes the style text. The same process is repeated for the negative prompt.

If either the positive or negative prompt does not match the respective style, the function returns False, along with the original prompt and negative prompt, indicating that the style did not match.

This function is called by extract_styles_from_prompt, which iterates through a list of applicable styles to find matches. If a match is found, it updates the prompt and negative prompt accordingly and continues to search for additional styles until no more matches can be found. The results from extract_original_prompts are crucial for determining how styles are applied to prompts in the context of the overall functionality of the project.

**Note**: It is important to ensure that the style text used in conjunction with this function is structured correctly to avoid unexpected results. The function relies on the unwrap_style_text_from_prompt to accurately identify and extract prompts, so any issues with the style text format may lead to incorrect behavior.

**Output Example**: For instance, if the style object has a prompt of "Elegant: {prompt}" and the provided prompt is "Elegant: A beautiful sunset", the function would return:
(True, "A beautiful sunset", "A beautiful sunset")
## FunctionDef extract_styles_from_prompt(prompt, negative_prompt)
**extract_styles_from_prompt**: The function of extract_styles_from_prompt is to extract applicable styles from a given prompt and its negative counterpart, returning a list of extracted style names along with the modified prompts.

**parameters**: The parameters of this Function.
· parameter1: prompt - A string representing the main prompt that is being evaluated for style extraction.  
· parameter2: negative_prompt - A string representing the negative prompt that is being evaluated alongside the main prompt.

**Code Description**: The extract_styles_from_prompt function is designed to analyze a main prompt and a negative prompt to identify and extract applicable styles defined in the project. It begins by initializing two lists: one for storing the names of extracted styles and another for holding instances of PromptStyle, which encapsulate the styles available for extraction.

The function iterates over the styles defined in modules.sdxl_styles.styles, creating PromptStyle instances for each style. It then enters a loop where it attempts to match the provided prompts against the styles in the applicable_styles list. For each style, it calls the extract_original_prompts function, which evaluates whether the style matches the current prompt and negative prompt. If a match is found, the function updates the prompts accordingly, records the style name, and continues searching for additional matches until no more applicable styles can be found.

If there are remaining unresolved prompts after the matching process, the function checks if any prompt expansion is necessary. It does this by examining the first word of the prompt and determining if it appears multiple times, which may indicate that a real prompt can be extracted. The function then appends the appropriate expansion style if applicable.

Finally, the function returns a list of extracted style names in reverse order, along with the modified real prompt and negative prompt. This output is crucial for subsequent processing in the project, particularly in the context of the A1111MetadataParser class, where the extracted styles and prompts are utilized to construct a structured JSON representation of metadata.

The extract_styles_from_prompt function is called within the to_json method of the A1111MetadataParser class. This method processes metadata strings, separating them into main and negative prompts, and then invokes extract_styles_from_prompt to identify styles and modify the prompts accordingly. The results from extract_styles_from_prompt are integrated into the final data structure returned by the to_json method, highlighting the importance of this function in the overall workflow of metadata parsing and style extraction.

**Note**: It is essential to ensure that the prompts provided to this function are structured correctly to facilitate accurate style extraction. Any discrepancies in the prompt format may lead to unexpected results during the extraction process.

**Output Example**: For instance, if the input prompts are "Elegant: A beautiful sunset" and "Not Elegant", and the styles defined include "Elegant" with a corresponding prompt of "Elegant: {prompt}", the function might return:
(["Elegant"], "A beautiful sunset", "Not Elegant").
## ClassDef PromptStyle
**PromptStyle**: The function of PromptStyle is to represent a style with its associated prompts.

**attributes**: The attributes of this Class.
· name: A string representing the name of the style.  
· prompt: A string containing the prompt associated with the style.  
· negative_prompt: A string containing the negative prompt associated with the style.  

**Code Description**: The PromptStyle class is a simple data structure that inherits from NamedTuple, allowing it to store three attributes: name, prompt, and negative_prompt. Each instance of PromptStyle encapsulates the information related to a specific style, including its name and the prompts that define it. This structure is particularly useful in scenarios where styles need to be extracted and processed, as it provides a clear and organized way to manage the associated data.

The PromptStyle class is utilized within the extract_styles_from_prompt function, which is responsible for extracting applicable styles from a given prompt and its negative counterpart. In this function, a list of PromptStyle instances is created by iterating over predefined styles stored in modules.sdxl_styles.styles. Each style is instantiated as a PromptStyle object, allowing the function to easily access and manipulate the style's attributes during the extraction process.

As the extract_styles_from_prompt function processes the input prompts, it checks for matches against the styles represented by the PromptStyle instances. When a match is found, the corresponding style is extracted, and its name is added to the list of extracted styles. This relationship highlights the importance of the PromptStyle class in facilitating the extraction and organization of style-related data, making it an integral part of the overall functionality of the prompt extraction process.

**Note**: When using the PromptStyle class, it is essential to ensure that the attributes are correctly populated with valid string values to maintain the integrity of the style representation. Additionally, understanding the context in which PromptStyle is used, particularly within the extract_styles_from_prompt function, will enhance the effectiveness of its application in style extraction tasks.
## FunctionDef is_json(data)
**is_json**: The function of is_json is to determine whether a given string is a valid JSON object.

**parameters**: The parameters of this Function.
· data: A string that is expected to be in JSON format.

**Code Description**: The is_json function attempts to parse the input string `data` using the `json.loads()` method from the json module. If the parsing is successful, it asserts that the result is a dictionary, which is the expected structure for a JSON object. If either the parsing fails (raising a ValueError) or the assertion fails (raising an AssertionError), the function returns False, indicating that the input string is not valid JSON. If both checks pass, the function returns True, confirming that the input string is a valid JSON object.

This function is utilized in multiple locations within the project. For instance, in the `read_info_from_image` function found in `modules/meta_parser.py`, is_json is called to validate the 'parameters' extracted from a file's info. If the parameters are in JSON format, they are subsequently parsed into a Python dictionary. This ensures that the function can handle the data correctly and avoid errors during processing.

Additionally, in the `parse_meta` function located in `webui.py`, is_json is used to check if the `raw_prompt_txt` input is a valid JSON string. If it is valid, the string is parsed into a JSON object. This validation step is crucial for the function's logic, as it dictates the flow of data handling based on whether the input is valid JSON or not.

**Note**: It is important to ensure that the input string is properly formatted as JSON. The function will return False for any malformed JSON strings, which could lead to unexpected behavior in the calling functions if not handled appropriately.

**Output Example**: 
- Input: '{"name": "John", "age": 30}'
- Output: True

- Input: 'Invalid JSON string'
- Output: False
## FunctionDef get_filname_by_stem(lora_name, filenames)
**get_filname_by_stem**: The function of get_filname_by_stem is to retrieve the filename that matches a specified stem from a list of filenames.

**parameters**: The parameters of this Function.
· lora_name: A string representing the stem name to be matched against the filenames.
· filenames: A list of strings containing the filenames to be searched.

**Code Description**: The get_filname_by_stem function iterates through a list of filenames and checks if the stem (the filename without its extension) matches the provided lora_name. It utilizes the Path class from the pathlib module to extract the stem of each filename. If a match is found, the function returns the corresponding filename. If no matches are found after checking all filenames, the function returns None.

This function is called within the parse_lora_references_from_prompt function, which processes a prompt string to identify and extract references to "lora" files. When a match is found in the prompt, parse_lora_references_from_prompt attempts to verify the existence of the lora file by calling get_filname_by_stem with the extracted lora name and a list of filenames. This relationship is crucial as it ensures that only valid lora files are considered when processing the prompt, thereby preventing unintended side effects from invalid references.

**Note**: It is important to ensure that the filenames provided to the get_filname_by_stem function are accurate and formatted correctly, as any discrepancies may lead to a failure in finding the intended match.

**Output Example**: If the lora_name is "example" and the filenames list contains ["example.safetensors", "test.safetensors"], the function would return "example.safetensors". If there is no match, it would return None.
## FunctionDef get_file_from_folder_list(name, folders)
**get_file_from_folder_list**: The function of get_file_from_folder_list is to locate a specified file within a list of directories and return its absolute path if found.

**parameters**: The parameters of this Function.
· name: A string representing the name of the file to be located.
· folders: A list or a single string representing the directories to search for the specified file.

**Code Description**: The get_file_from_folder_list function begins by checking if the provided folders parameter is a list. If it is not, the function converts it into a list containing the single folder. The function then iterates over each folder in the folders list, constructing the absolute path of the file by joining the folder path with the file name. It uses the os.path.abspath and os.path.realpath functions to ensure that the path is both absolute and resolves any symbolic links. If the constructed path points to an existing file (checked using os.path.isfile), the function returns the absolute path of that file. If the file is not found in any of the specified folders, the function defaults to returning the absolute path constructed from the first folder in the list, regardless of whether the file exists there or not.

This function is utilized in various parts of the project to ensure that required files are accessible. For instance, in the download_models function within launch.py, get_file_from_folder_list is called to verify the existence of a default model file in the specified checkpoint paths. If the file is not found, it checks for alternative model names, thereby allowing the application to fallback on previously downloaded models without requiring a new download. Similarly, in the refresh_base_model and refresh_refiner_model functions within modules/default_pipeline.py, get_file_from_folder_list is employed to locate the base and refiner model files, ensuring that the correct models are loaded into the application. Additionally, in the set_data method of the MetadataParser class, it is used to retrieve the paths for base models and LoRAs, further emphasizing its role in file management across the project.

**Note**: It is important to ensure that the folders parameter is either a list or a single string; otherwise, the function will convert it to a list, which may lead to unexpected behavior if not handled properly. 

**Output Example**: An example return value of the function could be: "/absolute/path/to/folder/model_file.bin" if the file "model_file.bin" is found in the specified folder. If not found, it may return "/absolute/path/to/folder/model_file.bin" based on the first folder in the list, regardless of the file's existence.
## FunctionDef get_enabled_loras(loras, remove_none)
**get_enabled_loras**: The function of get_enabled_loras is to filter and return a list of enabled Loras based on specific criteria.

**parameters**: The parameters of this Function.
· loras: A list of tuples, where each tuple contains information about a Lora, including its enabled status, name, and weight.
· remove_none: A boolean flag indicating whether to exclude Loras with a name of 'None' from the result.

**Code Description**: The get_enabled_loras function processes a list of Loras, which are represented as tuples. Each tuple is expected to contain three elements: a boolean indicating if the Lora is enabled, a string representing the Lora's name, and a float representing its weight. The function constructs a new list by iterating through the input list and applying the following filters:

1. It checks if the first element of the tuple (the enabled status) is True.
2. If the remove_none parameter is set to True, it further checks that the second element (the Lora's name) is not equal to the string 'None'. If remove_none is False, this check is bypassed.

The result is a list of tuples, each containing the name and weight of the enabled Loras. This function is particularly useful in contexts where Loras need to be selectively applied based on their enabled status and name validity.

In the project, get_enabled_loras is called within the __init__ method of the AsyncTask class located in the modules/async_worker.py file. Here, it is used to initialize the performance_loras attribute by processing a list of Lora data generated from the input arguments. This integration indicates that the function plays a crucial role in setting up the task's parameters, ensuring that only valid and enabled Loras are considered for further processing.

**Note**: It is important to ensure that the input list of Loras is structured correctly, as the function relies on the expected tuple format to operate effectively.

**Output Example**: An example of the possible return value of the function could be:
```python
[("Lora1", 0.75), ("Lora2", 1.0)]
```
This output indicates that two Loras are enabled, with their respective weights.
## FunctionDef parse_lora_references_from_prompt(prompt, loras, loras_limit, skip_file_check, prompt_cleanup, deduplicate_loras, lora_filenames)
**parse_lora_references_from_prompt**: The function of parse_lora_references_from_prompt is to process a prompt string to identify and extract references to "lora" files, while managing their limits and deduplication.

**parameters**: The parameters of this Function.
· prompt: A string containing the prompt from which LORA references are to be extracted.
· loras: A list of tuples, where each tuple contains a string representing the LORA filename and a float representing its weight.
· loras_limit: An integer specifying the maximum number of LORA references to return.
· skip_file_check: A boolean indicating whether to skip the check for the existence of LORA files.
· prompt_cleanup: A boolean indicating whether to clean up the prompt by removing unnecessary spaces and formatting.
· deduplicate_loras: A boolean indicating whether to deduplicate LORA references in the output.
· lora_filenames: An optional list of strings representing the filenames of LORA files to check against.

**Code Description**: The parse_lora_references_from_prompt function begins by creating a copy of the loras list to prevent unintended side effects. It initializes variables to store found LORA references and the cleaned prompt. The function then splits the input prompt by commas and iterates through each token. For each token, it uses a regular expression to find matches for LORA references in the format `<lora:name:weight>`. If no matches are found, the token is added to the prompt_without_loras string. When matches are found, the function constructs the LORA filename and checks its existence using the get_filname_by_stem function. Valid LORA references are added to the found_loras list, and the token is updated to remove the matched LORA reference.

After processing all tokens, the function cleans up the prompt if the prompt_cleanup parameter is set to True. It then deduplicates the found LORA references based on the deduplicate_loras parameter and combines them with the original loras list. Finally, the function returns a tuple containing the updated list of LORA references (limited by loras_limit) and the cleaned prompt.

This function is called within the process_prompt function, which is responsible for preparing prompts for image generation tasks. The parse_lora_references_from_prompt function ensures that only valid LORA files are considered when processing the prompt, thereby preventing unintended side effects from invalid references. Additionally, it interacts with the get_filname_by_stem function to verify the existence of LORA files and the cleanup_prompt function to format the prompt correctly.

**Note**: It is important to ensure that the input prompt is well-formed and that the lora_filenames provided are accurate. The function assumes that the prompt will contain comma-separated values and that LORA references will follow the specified format.

**Output Example**: Given an input prompt such as "some prompt, very cool, <lora:hey-lora:0.4>, cool", with an empty loras list and a loras_limit of 5, the function would return:
(
    [('hey-lora.safetensors', 0.4)], 
    'some prompt, very cool, cool'
)
## FunctionDef remove_performance_lora(filenames, performance)
**remove_performance_lora**: The function of remove_performance_lora is to filter out filenames from a list that correspond to a specific performance mode's LoRA filename.

**parameters**: The parameters of this Function.
· filenames: A list of strings representing the filenames to be filtered.
· performance: An instance of the Performance class or None, representing the performance mode whose associated LoRA filename should be removed from the list.

**Code Description**: The remove_performance_lora function begins by creating a copy of the input list of filenames, named loras_without_performance. This copy will be modified to exclude any filenames that match the LoRA filename associated with the provided performance mode. 

If the performance parameter is None, the function immediately returns the copied list, as there is no specific performance mode to filter against. 

When a valid performance instance is provided, the function calls the lora_filename method on the performance instance to retrieve the corresponding LoRA filename. It then iterates through each filename in the original list. For each filename, it checks if it matches the retrieved performance_lora filename. If a match is found, that filename is removed from the loras_without_performance list.

Finally, the function returns the filtered list of filenames, which no longer includes any that correspond to the specified performance mode's LoRA filename.

This function is called within the process_prompt function in the async_worker module. In this context, it is used to ensure that the list of LoRA filenames passed to the image processing pipeline does not include any filenames that are associated with the performance mode selected by the user. This is crucial for maintaining the integrity of the processing parameters, as including the performance LoRA could lead to unintended behavior or conflicts during image generation tasks.

**Note**: It is important to ensure that the performance parameter is an instance of the Performance class to retrieve the correct LoRA filename. If the performance is not set or is invalid, the function will return the original list of filenames without any modifications.

**Output Example**: An example output for the function might look like this:
```
['hey-lora.safetensors']
``` 
This output indicates that the filename associated with the specified performance mode has been successfully removed from the original list of filenames.
## FunctionDef cleanup_prompt(prompt)
**cleanup_prompt**: The function of cleanup_prompt is to clean up a given prompt string by removing extra spaces and ensuring proper formatting of comma-separated values.

**parameters**: The parameters of this Function.
· prompt: A string that contains the prompt to be cleaned.

**Code Description**: The cleanup_prompt function processes a string input, referred to as 'prompt', to enhance its readability and formatting. The function first uses a regular expression to replace multiple consecutive spaces with a single space. It then applies another regular expression to reduce multiple consecutive commas to a single comma. 

After these initial replacements, the function initializes an empty string called 'cleaned_prompt'. It splits the modified prompt into tokens based on commas and iterates through each token. During this iteration, it trims whitespace from each token and checks if the token is empty. If a token is not empty, it appends the token followed by a comma and a space to the 'cleaned_prompt' string. Finally, the function returns the 'cleaned_prompt' string, excluding the last two characters (which are an extra comma and space).

This function is called by the parse_lora_references_from_prompt function, which is responsible for parsing LORA references from a given prompt. Within parse_lora_references_from_prompt, the prompt is processed to separate LORA references from the main prompt content. After constructing a string that excludes LORA references, the function calls cleanup_prompt to ensure that the resulting prompt is well-formatted and free of unnecessary whitespace or punctuation. This relationship highlights the importance of cleanup_prompt in maintaining the integrity and clarity of the prompt data being processed.

**Note**: It is important to ensure that the input prompt is a well-formed string. The function assumes that the input will be a string containing comma-separated values.

**Output Example**: Given an input prompt such as "  apple,  banana, , orange,   ,  grape, , ", the function would return "apple, banana, orange, grape".
## FunctionDef apply_wildcards(wildcard_text, rng, i, read_wildcards_in_order)
**apply_wildcards**: The function of apply_wildcards is to process a string containing placeholders and replace them with corresponding values from external files based on a specified order or randomness.

**parameters**: The parameters of this Function.
· wildcard_text: A string that may contain placeholders formatted as `__placeholder__` which need to be replaced.
· rng: A random number generator instance used for selecting random words from the list of possible replacements.
· i: An index used to determine the order of replacement when `read_wildcards_in_order` is set to True.
· read_wildcards_in_order: A boolean that indicates whether the wildcards should be replaced in a sequential order or randomly.

**Code Description**: The apply_wildcards function is designed to replace placeholders in a given text with corresponding words from external files. It first identifies all placeholders in the input string using a regular expression. If no placeholders are found, it returns the original string. If placeholders are present, the function iterates through a maximum depth defined by `modules.config.wildcards_max_bfs_depth`. For each placeholder, it attempts to find a matching file in the `modules.config.wildcard_filenames` list. If a matching file is found, it reads the contents of that file, which should contain a list of words. The function then replaces the placeholder in the original text with either a word from the list based on the index `i` (if `read_wildcards_in_order` is True) or a randomly selected word from the list (if False). If the file is missing or empty, the placeholder is replaced with its own name. The function also prints debug information during processing, including warnings for missing files and the current state of the text after each replacement. If the maximum depth is reached without resolving all placeholders, a warning is printed, and the function returns the current state of the text.

This function is called within the `process_prompt` function in the `modules/async_worker.py/worker` module. It is used to process the main prompt and negative prompt for image generation tasks. The prompts may contain wildcards that need to be resolved to create specific variations of the prompts for different images. The `apply_wildcards` function ensures that the prompts are dynamically generated based on the available wildcard files, allowing for a more flexible and creative image generation process.

**Note**: It is important to ensure that the wildcard files are correctly formatted and accessible, as missing or empty files will lead to the placeholders being replaced with their own names, which may not be the intended behavior.

**Output Example**: Given the input `wildcard_text` as "This is a __color__ car.", and assuming the corresponding file for `color` contains the words "red", "blue", "green", if `read_wildcards_in_order` is True and `i` is 1, the output would be "This is a blue car." If `read_wildcards_in_order` is False, the output could be any random color from the list, such as "This is a red car."
## FunctionDef get_image_size_info(image, aspect_ratios)
**get_image_size_info**: The function of get_image_size_info is to calculate and return the size and aspect ratio information of a given image, along with a recommended size based on predefined aspect ratios.

**parameters**: The parameters of this Function.
· parameter1: image (np.ndarray) - A NumPy array representing the image whose size information is to be retrieved.
· parameter2: aspect_ratios (list) - A list of aspect ratios in the format 'width*height' that will be used to recommend a suitable size for the image.

**Code Description**: The get_image_size_info function begins by attempting to convert the input NumPy array into a PIL Image object. It retrieves the width and height of the image and calculates the aspect ratio by dividing the width by the height, rounding the result to two decimal places. The function then computes the greatest common divisor (GCD) of the width and height to derive the least common multiple (LCM) ratio, formatted as 'width:height'. 

Next, the function identifies the closest aspect ratio from the provided list by calculating the absolute difference between the calculated ratio and each aspect ratio in the list. It selects the aspect ratio that minimizes this difference, extracts the recommended width and height, and computes their ratio and LCM in the same manner as before.

The function constructs a string containing the original image size, its ratio, and the recommended size and ratio, returning this information as a formatted string. If any error occurs during the process, such as issues with the image format or conversion, the function catches the exception and returns an error message indicating the problem.

This function is called by the trigger_show_image_properties function in the webui.py module. The trigger_show_image_properties function takes an image as input, calls get_image_size_info with the image and a list of aspect ratios defined in modules.flags.sdxl_aspect_ratios, and updates the user interface with the resulting size information.

**Note**: It is important to ensure that the input image is a valid NumPy array and that the aspect ratios provided are in the correct format ('width*height') to avoid errors during execution.

**Output Example**: A possible return value of the function could be:
"Image Size: 1920 x 1080, Ratio: 1.78, 16:9
Recommended Size: 1280 x 720, Recommended Ratio: 1.78, 16:9"
