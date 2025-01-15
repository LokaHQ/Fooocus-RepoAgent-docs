## FunctionDef _compute_zero_padding(kernel_size)
**_compute_zero_padding**: The function of _compute_zero_padding is to calculate the amount of zero padding required for a given kernel size in a 2D convolution operation.

**parameters**: The parameters of this Function.
· kernel_size: A tuple of two integers or a single integer representing the size of the kernel. If a single integer is provided, it is used for both dimensions (height and width).

**Code Description**: The _compute_zero_padding function takes a kernel_size parameter, which can either be a single integer or a tuple containing two integers. It first calls the _unpack_2d_ks function to convert the kernel_size into a standardized tuple format representing the height (ky) and width (kx) of the kernel. The function then calculates the required zero padding for both dimensions by applying the formula (ky - 1) // 2 for height and (kx - 1) // 2 for width. This calculation is essential in convolution operations, where zero padding is used to maintain the spatial dimensions of the output feature maps.

The _compute_zero_padding function is called by the _bilateral_blur function within the same module. In _bilateral_blur, it is used to determine the amount of padding needed before processing the input tensor. This ensures that the convolution operation can be performed correctly without altering the dimensions of the input data. The padding values obtained from _compute_zero_padding are then utilized to pad the input tensor appropriately before further processing, such as unfolding the tensor for bilateral filtering.

**Note**: It is important to ensure that the kernel_size parameter is either a single integer or a tuple of two integers. Providing an invalid input type will result in an assertion error from the _unpack_2d_ks function.

**Output Example**: If the input kernel_size is (5, 3), the function will return (2, 1). If the input is 4, the function will return (1, 1).
## FunctionDef _unpack_2d_ks(kernel_size)
**_unpack_2d_ks**: The function of _unpack_2d_ks is to convert a kernel size input into a standardized tuple format representing the height and width of a 2D kernel.

**parameters**: The parameters of this Function.
· kernel_size: A tuple of two integers or a single integer representing the size of the kernel. If a single integer is provided, it is used for both dimensions (height and width).

**Code Description**: The _unpack_2d_ks function takes a kernel_size parameter, which can either be a single integer or a tuple containing two integers. If a single integer is provided, it assigns this value to both the height (ky) and width (kx) of the kernel. If a tuple is provided, the function asserts that the length of the tuple is exactly 2, ensuring that both dimensions are specified. The function then converts both dimensions to integers and returns them as a tuple. 

This function is called by other functions within the module, such as _compute_zero_padding, get_gaussian_kernel2d, and _bilateral_blur. In _compute_zero_padding, the unpacked kernel size is used to calculate the amount of zero padding needed for the kernel. In get_gaussian_kernel2d, it is used to determine the dimensions of the Gaussian kernel being generated. Similarly, in _bilateral_blur, the unpacked dimensions are utilized to prepare the input tensor for processing, ensuring that the operations performed on the tensor are consistent with the specified kernel size.

**Note**: It is important to ensure that the kernel_size parameter is either a single integer or a tuple of two integers. Providing an invalid input type will result in an assertion error.

**Output Example**: If the input kernel_size is (5, 3), the function will return (5, 3). If the input is 4, the function will return (4, 4).
## FunctionDef gaussian(window_size, sigma)
**gaussian**: The function of gaussian is to generate a normalized Gaussian kernel based on the specified window size and standard deviation.

**parameters**: The parameters of this Function.
· window_size: An integer that specifies the size of the Gaussian kernel to be generated. It determines the number of elements in the output tensor.
· sigma: A Tensor or float that represents the standard deviation of the Gaussian distribution. It controls the spread of the kernel.
· device: An optional parameter that specifies the device (CPU or GPU) on which the tensor will be allocated. It can be None.
· dtype: An optional parameter that specifies the data type of the tensor. It can be None.

**Code Description**: The gaussian function creates a one-dimensional Gaussian kernel. It first calculates the batch size from the shape of the sigma parameter, which is expected to be a tensor. The function then generates a range of values centered around zero, which are adjusted based on the window size. If the window size is even, the function adds 0.5 to the range to ensure proper centering. The Gaussian values are computed using the formula for the Gaussian function, which involves exponentiation of the negative squared values divided by twice the square of sigma. Finally, the function normalizes the Gaussian values by dividing by their sum, ensuring that the output kernel sums to one.

This function is called by the get_gaussian_kernel1d function, which serves as a wrapper to simplify the process of obtaining a Gaussian kernel. The get_gaussian_kernel1d function takes in parameters such as kernel_size and sigma, and it directly passes these parameters to the gaussian function along with optional device and dtype parameters. This relationship allows for a modular design where the gaussian function can be reused in different contexts, enhancing code maintainability and clarity.

**Note**: When using this function, ensure that the sigma parameter is appropriately sized if it is a tensor, as the function expects it to have a shape that includes the batch size. Additionally, the window_size should be a positive integer to avoid errors during tensor operations.

**Output Example**: A possible appearance of the code's return value could be a tensor like the following, assuming a window_size of 5 and a sigma of 1.0:
```
tensor([[0.05399, 0.24197, 0.39894, 0.24197, 0.05399]])
```
## FunctionDef get_gaussian_kernel1d(kernel_size, sigma, force_even)
**get_gaussian_kernel1d**: The function of get_gaussian_kernel1d is to generate a one-dimensional Gaussian kernel based on the specified kernel size and standard deviation.

**parameters**: The parameters of this Function.
· kernel_size: An integer that specifies the size of the Gaussian kernel to be generated. It determines the number of elements in the output tensor.  
· sigma: A Tensor or float that represents the standard deviation of the Gaussian distribution. It controls the spread of the kernel.  
· force_even: A boolean that, when set to True, forces the kernel size to be even. This can be useful for certain applications where even-sized kernels are required.  
· device: An optional parameter that specifies the device (CPU or GPU) on which the tensor will be allocated. It can be None.  
· dtype: An optional parameter that specifies the data type of the tensor. It can be None.  

**Code Description**: The get_gaussian_kernel1d function serves as a wrapper around the gaussian function, simplifying the process of obtaining a one-dimensional Gaussian kernel. It accepts parameters such as kernel_size and sigma, which are essential for generating the kernel. The function directly passes these parameters to the gaussian function along with optional device and dtype parameters. This design promotes modularity, allowing the gaussian function to be reused in different contexts, thus enhancing code maintainability and clarity.

The gaussian function, which is called within get_gaussian_kernel1d, generates a normalized Gaussian kernel based on the specified window size and standard deviation. It calculates the batch size from the shape of the sigma parameter and generates a range of values centered around zero, adjusting for even-sized kernels if necessary. The Gaussian values are computed using the Gaussian function formula and normalized to ensure the output kernel sums to one.

The get_gaussian_kernel1d function is also utilized by the get_gaussian_kernel2d function, which generates a two-dimensional Gaussian kernel. In this context, get_gaussian_kernel1d is called twice to create one-dimensional kernels for both the y and x dimensions, which are then combined to form the final two-dimensional kernel. This demonstrates the utility of get_gaussian_kernel1d in generating Gaussian kernels of varying dimensions while maintaining a consistent interface.

**Note**: When using this function, ensure that the sigma parameter is appropriately sized if it is a tensor, as the function expects it to have a shape that includes the batch size. Additionally, the kernel_size should be a positive integer to avoid errors during tensor operations.

**Output Example**: A possible appearance of the code's return value could be a tensor like the following, assuming a kernel_size of 5 and a sigma of 1.0:
```
tensor([[0.05399, 0.24197, 0.39894, 0.24197, 0.05399]])
```
## FunctionDef get_gaussian_kernel2d(kernel_size, sigma, force_even)
**get_gaussian_kernel2d**: The function of get_gaussian_kernel2d is to generate a two-dimensional Gaussian kernel based on specified kernel size and standard deviation values.

**parameters**: The parameters of this Function.
· kernel_size: A tuple of two integers or a single integer representing the size of the kernel. If a single integer is provided, it is used for both dimensions (height and width).  
· sigma: A tuple of two floats or a Tensor representing the standard deviation for the Gaussian distribution in both dimensions. This controls the spread of the kernel.  
· force_even: A boolean that, when set to True, forces the kernel size to be even. This can be useful for certain applications where even-sized kernels are required.  
· device: An optional parameter that specifies the device (CPU or GPU) on which the tensor will be allocated. It can be None.  
· dtype: An optional parameter that specifies the data type of the tensor. It can be None.  

**Code Description**: The get_gaussian_kernel2d function is designed to create a two-dimensional Gaussian kernel by first unpacking the kernel size and standard deviation parameters. It utilizes the _unpack_2d_ks function to convert the kernel_size input into a standardized tuple format representing the height and width of the kernel. The sigma parameter is transformed into a tensor, ensuring it is compatible with the specified device and data type.

The function then calls get_gaussian_kernel1d twice to generate one-dimensional Gaussian kernels for both the y and x dimensions. These one-dimensional kernels are then combined to produce the final two-dimensional Gaussian kernel by performing an outer product. This approach allows for the creation of a Gaussian kernel that accurately reflects the specified standard deviations in both dimensions.

The get_gaussian_kernel2d function is called by the _bilateral_blur function, which applies bilateral filtering to an input tensor. In this context, get_gaussian_kernel2d is used to generate the spatial kernel based on the provided kernel size and sigma_space parameters. The resulting kernel is then combined with a color kernel to perform the bilateral blur operation, ensuring that the blurring effect is spatially aware and preserves edges.

**Note**: When using this function, ensure that the sigma parameter is appropriately sized if it is a tensor, as the function expects it to have a shape that includes the batch size. Additionally, the kernel_size should be a positive integer or a valid tuple to avoid errors during tensor operations.

**Output Example**: A possible appearance of the code's return value could be a tensor like the following, assuming a kernel_size of (5, 5) and a sigma of (1.0, 1.0):
```
tensor([[0.00376, 0.01576, 0.02391, 0.01576, 0.00376],
        [0.01576, 0.06624, 0.09973, 0.06624, 0.01576],
        [0.02391, 0.09973, 0.14957, 0.09973, 0.02391],
        [0.01576, 0.06624, 0.09973, 0.06624, 0.01576],
        [0.00376, 0.01576, 0.02391, 0.01576, 0.00376]])
```
## FunctionDef _bilateral_blur(input, guidance, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)
**_bilateral_blur**: The function of _bilateral_blur is to perform bilateral filtering on an input tensor, optionally guided by another tensor, to achieve edge-preserving smoothing.

**parameters**: The parameters of this Function.
· input: A Tensor representing the input image or data to be processed.  
· guidance: An optional Tensor that serves as a guidance image for the bilateral filtering. If not provided, the input tensor is used as the guidance.  
· kernel_size: A tuple of two integers or a single integer specifying the size of the kernel used for filtering.  
· sigma_color: A float or Tensor that defines the standard deviation for color space, influencing how colors are blended.  
· sigma_space: A tuple of floats or a Tensor representing the standard deviation for spatial distance, affecting the spatial extent of the filter.  
· border_type: A string that specifies the type of border handling during padding, with 'reflect' being the default.  
· color_distance_type: A string that determines the method for calculating color distance, accepting either 'l1' or 'l2'.

**Code Description**: The _bilateral_blur function implements bilateral filtering, which is a technique that smooths images while preserving edges. It begins by checking if the sigma_color parameter is a Tensor, in which case it adjusts its shape and device compatibility. The function then unpacks the kernel size into its height (ky) and width (kx) components and computes the necessary zero padding for the input tensor using the _compute_zero_padding function.

The input tensor is padded according to the specified border type, and the padded input is unfolded into a format suitable for applying the bilateral filter. If a guidance tensor is provided, it is also padded and unfolded; otherwise, the input tensor is used as the guidance.

The function calculates the color distance between the guidance and the input tensor based on the specified color_distance_type (either 'l1' or 'l2'). This distance is used to create a color kernel that influences how much neighboring pixels contribute to the final output based on their color similarity. Additionally, a spatial kernel is generated using the get_gaussian_kernel2d function, which creates a Gaussian kernel based on the specified kernel size and sigma_space.

The final kernel is a product of the spatial and color kernels, and the output is computed by applying this kernel to the unfolded input tensor, normalizing by the sum of the kernel weights. The resulting tensor is returned as the output of the function.

The _bilateral_blur function is called by several other functions within the module, including bilateral_blur, adaptive_anisotropic_filter, and joint_bilateral_blur. In bilateral_blur, it is invoked without a guidance tensor, applying the filter directly to the input. In adaptive_anisotropic_filter, it uses a computed guidance tensor based on the input to enhance the filtering effect. In joint_bilateral_blur, it applies the filter using both the input and a specified guidance tensor, allowing for more controlled smoothing based on external information.

**Note**: It is essential to ensure that the kernel_size parameter is valid and that the sigma parameters are appropriately sized if they are tensors. Providing invalid inputs may lead to runtime errors.

**Output Example**: A possible appearance of the code's return value could be a tensor representing the filtered image, maintaining the original dimensions of the input tensor while exhibiting reduced noise and preserved edges.
## FunctionDef bilateral_blur(input, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)
**bilateral_blur**: The function of bilateral_blur is to apply bilateral filtering to an input tensor, enhancing edge preservation while reducing noise.

**parameters**: The parameters of this Function.
· input: A Tensor representing the input image or data to be processed.  
· kernel_size: A tuple of two integers or a single integer specifying the size of the kernel used for filtering, defaulting to (13, 13).  
· sigma_color: A float or Tensor that defines the standard deviation for color space, influencing how colors are blended, defaulting to 3.0.  
· sigma_space: A tuple of floats or a Tensor representing the standard deviation for spatial distance, affecting the spatial extent of the filter, defaulting to 3.0.  
· border_type: A string that specifies the type of border handling during padding, with 'reflect' being the default.  
· color_distance_type: A string that determines the method for calculating color distance, accepting either 'l1' or 'l2'.

**Code Description**: The bilateral_blur function serves as a high-level interface for performing bilateral filtering on an input tensor. It internally calls the _bilateral_blur function, passing the necessary parameters while omitting the guidance tensor, which means the filtering is applied directly to the input tensor itself.

The function begins by accepting the input tensor and various parameters that control the filtering process, including kernel size, sigma values for color and space, border handling type, and the method for calculating color distance. The default values for kernel size, sigma_color, and sigma_space are set to ensure reasonable defaults for common use cases.

When bilateral_blur is invoked, it calls the _bilateral_blur function, which implements the core logic of bilateral filtering. This lower-level function performs several key operations: it checks the type of sigma_color, unpacks the kernel size, computes necessary padding, and prepares the input tensor for filtering. The filtering process involves calculating color and spatial kernels based on the provided parameters and applying these kernels to the input tensor to produce the final output.

The bilateral_blur function is called by the forward method of the BilateralBlur class, which serves as a wrapper around the bilateral_blur function. This design allows for a more object-oriented approach to applying bilateral filtering, where the parameters can be set as attributes of the BilateralBlur instance and reused across multiple calls to the forward method.

**Note**: It is essential to ensure that the kernel_size parameter is valid and that the sigma parameters are appropriately sized if they are tensors. Providing invalid inputs may lead to runtime errors.

**Output Example**: A possible appearance of the code's return value could be a tensor representing the filtered image, maintaining the original dimensions of the input tensor while exhibiting reduced noise and preserved edges.
## FunctionDef adaptive_anisotropic_filter(x, g)
**adaptive_anisotropic_filter**: The function of adaptive_anisotropic_filter is to apply an adaptive anisotropic filtering technique to an input tensor, optionally guided by another tensor, to achieve edge-preserving smoothing.

**parameters**: The parameters of this Function.
· x: A Tensor representing the input image or data to be processed.  
· g: An optional Tensor that serves as a guidance image for the filtering process. If not provided, the input tensor is used as the guidance.

**Code Description**: The adaptive_anisotropic_filter function begins by checking if the guidance tensor `g` is provided. If `g` is None, it defaults to using the input tensor `x` as the guidance. The function then computes the standard deviation and mean of the guidance tensor across the specified dimensions (1, 2, 3), which correspond to the color channels and spatial dimensions of the input tensor. A small constant (1e-5) is added to the standard deviation to prevent division by zero.

Next, the function normalizes the guidance tensor by subtracting the mean and dividing by the standard deviation, resulting in a guidance tensor that is centered and scaled. This normalized guidance is then used as input to the _bilateral_blur function, which performs bilateral filtering on the input tensor `x`. The bilateral blur is configured with a kernel size of (13, 13), and both sigma_color and sigma_space parameters are set to 3.0. The border handling type is set to 'reflect', and the color distance type is specified as 'l1'.

The output of the _bilateral_blur function is returned as the result of the adaptive_anisotropic_filter function. This output represents the filtered image, which retains important edge information while reducing noise.

The adaptive_anisotropic_filter function is called by the patched_sampling_function, where it is utilized to process the positive epsilon values derived from the input tensor. The filtered result is then combined with the original positive epsilon to create a final epsilon value, which is used to adjust the input tensor. This integration highlights the role of adaptive_anisotropic_filter in enhancing the quality of the sampling process by providing a smoother representation of the input data while preserving essential features.

**Note**: It is important to ensure that the input tensor `x` and the optional guidance tensor `g` are properly formatted and compatible in terms of dimensions. Providing invalid inputs may lead to runtime errors.

**Output Example**: A possible appearance of the code's return value could be a tensor representing the filtered image, maintaining the original dimensions of the input tensor while exhibiting reduced noise and preserved edges.
## FunctionDef joint_bilateral_blur(input, guidance, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)
**joint_bilateral_blur**: The function of joint_bilateral_blur is to apply joint bilateral filtering on an input tensor using a guidance tensor to achieve edge-preserving smoothing.

**parameters**: The parameters of this Function.
· input: A Tensor representing the input image or data to be processed.  
· guidance: A Tensor that serves as a guidance image for the bilateral filtering.  
· kernel_size: A tuple of two integers or a single integer specifying the size of the kernel used for filtering.  
· sigma_color: A float or Tensor that defines the standard deviation for color space, influencing how colors are blended.  
· sigma_space: A tuple of floats or a Tensor representing the standard deviation for spatial distance, affecting the spatial extent of the filter.  
· border_type: A string that specifies the type of border handling during padding, with 'reflect' being the default.  
· color_distance_type: A string that determines the method for calculating color distance, accepting either 'l1' or 'l2'.

**Code Description**: The joint_bilateral_blur function is designed to perform joint bilateral filtering, which is a technique that smooths images while preserving edges by utilizing a guidance tensor. This function calls the _bilateral_blur function to execute the filtering process. 

The function accepts an input tensor and a guidance tensor, along with parameters that define the filtering behavior, such as kernel size, sigma values for color and space, border handling type, and color distance calculation method. The kernel size can be specified as either a single integer or a tuple, allowing for flexibility in defining the filter's dimensions. The sigma_color parameter influences how colors are blended during the filtering process, while sigma_space determines the spatial extent of the filter.

The joint_bilateral_blur function directly passes its parameters to the _bilateral_blur function, which performs the actual filtering operation. This relationship allows joint_bilateral_blur to leverage the capabilities of _bilateral_blur while providing additional context through the guidance tensor. The guidance tensor is crucial as it directs the filtering process, enabling more controlled smoothing based on external information.

The joint_bilateral_blur function is called within the forward method of the JointBilateralBlur class. This method serves as an interface for applying the joint bilateral blur operation on input data, effectively encapsulating the functionality of joint_bilateral_blur and making it accessible for use in neural network architectures or other applications requiring image processing.

**Note**: It is essential to ensure that the kernel_size parameter is valid and that the sigma parameters are appropriately sized if they are tensors. Providing invalid inputs may lead to runtime errors.

**Output Example**: A possible appearance of the code's return value could be a tensor representing the filtered image, maintaining the original dimensions of the input tensor while exhibiting reduced noise and preserved edges.
## ClassDef _BilateralBlur
**_BilateralBlur**: The function of _BilateralBlur is to provide a base class for implementing bilateral blur operations in neural networks.

**attributes**: The attributes of this Class.
· kernel_size: Defines the size of the kernel used for the bilateral blur operation, which can be a tuple of two integers or a single integer.
· sigma_color: Represents the standard deviation for the color space, which can be a float or a Tensor.
· sigma_space: Specifies the standard deviation for the spatial space, which can be a tuple of two floats or a Tensor.
· border_type: Indicates the type of border handling to be used, defaulting to 'reflect'.
· color_distance_type: Determines the method for calculating color distance, defaulting to "l1".

**Code Description**: The _BilateralBlur class inherits from torch.nn.Module, making it a part of the PyTorch neural network module system. It initializes several parameters essential for performing bilateral blur operations, including kernel size, sigma values for color and space, border type, and color distance type. The constructor method (__init__) sets these parameters and calls the superclass constructor to ensure proper initialization within the PyTorch framework.

This class serves as a foundational component for two derived classes: BilateralBlur and JointBilateralBlur. The BilateralBlur class implements a forward method that applies the bilateral blur operation to a given input tensor using the parameters defined in _BilateralBlur. Similarly, the JointBilateralBlur class extends this functionality by allowing an additional guidance tensor, which can be used to influence the blurring effect based on another image.

Both derived classes utilize the attributes defined in _BilateralBlur to perform their respective operations, ensuring consistency and reusability of the core parameters across different types of bilateral blur implementations.

**Note**: When using this class, it is essential to ensure that the parameters provided are compatible with the intended input data. The kernel size should be appropriate for the dimensions of the input tensor, and the sigma values should be chosen based on the desired blurring effect.

**Output Example**: An instance of _BilateralBlur could be represented as follows:
_BilateralBlur(kernel_size=(5, 5), sigma_color=25.0, sigma_space=(10.0, 15.0), border_type='reflect', color_distance_type='l1')
### FunctionDef __init__(self, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)
**__init__**: The function of __init__ is to initialize an instance of the BilateralBlur class with specified parameters for image processing.

**parameters**: The parameters of this Function.
· kernel_size: A tuple of two integers or a single integer that defines the size of the kernel used for the bilateral blur operation. If a single integer is provided, it is interpreted as the size for both dimensions of the kernel.
· sigma_color: A float or Tensor that represents the standard deviation in the color space. It controls how much the colors can vary when applying the blur effect.
· sigma_space: A tuple of two floats or a Tensor that specifies the standard deviation in the coordinate space. It determines how far the influence of a pixel extends in the spatial domain.
· border_type: A string that defines the border mode to be used when the kernel overlaps the image borders. The default value is 'reflect'.
· color_distance_type: A string that indicates the method used to compute the distance between colors. The default value is "l1".

**Code Description**: The __init__ function serves as the constructor for the BilateralBlur class. It initializes the instance with parameters that dictate how the bilateral blur effect will be applied to images. The kernel_size parameter allows for flexible kernel dimensions, accommodating both square and rectangular kernels. The sigma_color parameter influences the blurring effect based on color differences, while sigma_space controls the spatial extent of the blur. The border_type parameter ensures that the behavior at the edges of the image is defined, with 'reflect' being the default behavior. Lastly, the color_distance_type parameter allows the user to specify how color differences are calculated, which can affect the overall appearance of the blur.

**Note**: It is important to ensure that the parameters provided are compatible with the intended image processing tasks. The kernel size should be a positive integer, and the sigma values should be chosen based on the desired level of blurring. Additionally, users should be aware of the implications of different border types and color distance calculations on the final output.
***
### FunctionDef __repr__(self)
**__repr__**: The function of __repr__ is to provide a string representation of the BilateralBlur object, summarizing its key attributes.

**parameters**: The parameters of this Function.
· None

**Code Description**: The __repr__ method is designed to return a formatted string that represents the current state of the BilateralBlur object. It utilizes an f-string to concatenate the class name with its attributes, which include kernel_size, sigma_color, sigma_space, border_type, and color_distance_type. This method is particularly useful for debugging and logging purposes, as it allows developers to quickly understand the configuration of an instance of the BilateralBlur class. The output string is structured to clearly indicate the class type and its relevant parameters, making it easier to identify the specific settings being used.

**Note**: It is important to ensure that the attributes referenced in the __repr__ method are properly initialized in the class constructor. This will guarantee that the string representation accurately reflects the object's state.

**Output Example**: An example of the return value from this method could look like:
"BilateralBlur(kernel_size=5, sigma_color=75.0, sigma_space=75.0, border_type=cv2.BORDER_DEFAULT, color_distance_type='L2')"
***
## ClassDef BilateralBlur
**BilateralBlur**: The function of BilateralBlur is to apply bilateral blur operations to input tensors in neural networks.

**attributes**: The attributes of this Class.
· kernel_size: Defines the size of the kernel used for the bilateral blur operation, which can be a tuple of two integers or a single integer.  
· sigma_color: Represents the standard deviation for the color space, which can be a float or a Tensor.  
· sigma_space: Specifies the standard deviation for the spatial space, which can be a tuple of two floats or a Tensor.  
· border_type: Indicates the type of border handling to be used, defaulting to 'reflect'.  
· color_distance_type: Determines the method for calculating color distance, defaulting to "l1".  

**Code Description**: The BilateralBlur class inherits from the _BilateralBlur base class, which is designed to facilitate bilateral blur operations within the PyTorch framework. The primary functionality of the BilateralBlur class is encapsulated in its forward method, which takes an input tensor and applies the bilateral blur effect using the parameters defined in the _BilateralBlur class. 

The forward method calls the bilateral_blur function, passing in the input tensor along with the kernel size, sigma values for color and space, border type, and color distance type. This method effectively performs the blurring operation by leveraging the attributes inherited from _BilateralBlur, ensuring that the blurring parameters are consistently applied.

The relationship between BilateralBlur and its base class _BilateralBlur is crucial, as the latter provides the foundational attributes and initialization necessary for the bilateral blur operation. By extending _BilateralBlur, BilateralBlur can utilize the established parameters while implementing its specific behavior in the forward method.

**Note**: When utilizing the BilateralBlur class, it is essential to ensure that the input tensor is compatible with the parameters defined, particularly the kernel size and sigma values, to achieve the desired blurring effect.

**Output Example**: An instance of BilateralBlur could be represented as follows:
BilateralBlur(kernel_size=(5, 5), sigma_color=25.0, sigma_space=(10.0, 15.0), border_type='reflect', color_distance_type='l1')
### FunctionDef forward(self, input)
**forward**: The function of forward is to apply bilateral filtering to an input tensor using predefined parameters.

**parameters**: The parameters of this Function.
· input: A Tensor representing the input image or data to be processed.

**Code Description**: The forward method is a member of the BilateralBlur class and is responsible for executing the bilateral filtering process on the provided input tensor. It takes a single parameter, `input`, which is expected to be a Tensor containing the image or data that needs to be filtered. 

Internally, the forward method calls the `bilateral_blur` function, passing the input tensor along with several attributes of the BilateralBlur instance, including `kernel_size`, `sigma_color`, `sigma_space`, `border_type`, and `color_distance_type`. These attributes are set when the BilateralBlur instance is created and dictate how the filtering is performed.

The `bilateral_blur` function is designed to enhance edge preservation while reducing noise in the input tensor. It achieves this by applying a bilateral filter, which considers both the spatial distance and the color difference between pixels when determining how to blend them. The function handles various parameters that control the filtering process, ensuring that it can be tailored to different use cases.

The forward method serves as a high-level interface for users of the BilateralBlur class, allowing them to apply bilateral filtering without needing to understand the underlying implementation details. By encapsulating the filtering logic within the bilateral_blur function, the forward method promotes code reuse and maintainability.

**Note**: It is important to ensure that the input tensor is appropriately formatted and that the parameters set in the BilateralBlur instance are valid. Invalid inputs may lead to runtime errors during the filtering process.

**Output Example**: A possible appearance of the code's return value could be a tensor representing the filtered image, maintaining the original dimensions of the input tensor while exhibiting reduced noise and preserved edges.
***
## ClassDef JointBilateralBlur
**JointBilateralBlur**: The function of JointBilateralBlur is to apply a joint bilateral blur operation to an input tensor using a guidance tensor.

**attributes**: The attributes of this Class.
· kernel_size: Defines the size of the kernel used for the bilateral blur operation, which can be a tuple of two integers or a single integer.  
· sigma_color: Represents the standard deviation for the color space, which can be a float or a Tensor.  
· sigma_space: Specifies the standard deviation for the spatial space, which can be a tuple of two floats or a Tensor.  
· border_type: Indicates the type of border handling to be used, defaulting to 'reflect'.  
· color_distance_type: Determines the method for calculating color distance, defaulting to "l1".  

**Code Description**: The JointBilateralBlur class inherits from the _BilateralBlur base class, which provides foundational functionality for bilateral blur operations in neural networks. The primary purpose of the JointBilateralBlur class is to extend the bilateral blur operation by incorporating an additional guidance tensor. This allows the blurring effect to be influenced by another image, which can enhance the quality of the output by preserving edges and important features based on the guidance image.

The forward method of the JointBilateralBlur class takes two parameters: `input`, which is the tensor to be blurred, and `guidance`, which is the tensor that guides the blurring process. The method calls the `joint_bilateral_blur` function, passing the input tensor, guidance tensor, and the various parameters defined in the _BilateralBlur class, such as kernel size, sigma values for color and space, border type, and color distance type. This encapsulation allows for a clean and efficient implementation of the joint bilateral blur operation while leveraging the established parameters from the base class.

The relationship with its callees is significant, as the JointBilateralBlur class relies on the functionality provided by the _BilateralBlur class to manage the core attributes and methods necessary for performing bilateral blur operations. This design promotes code reuse and maintains consistency across different types of bilateral blur implementations.

**Note**: When using this class, it is crucial to ensure that the input and guidance tensors are compatible in terms of dimensions and data types. The kernel size should be appropriate for the dimensions of the input tensor, and the sigma values should be selected based on the desired blurring effect. Proper configuration of these parameters will yield optimal results in the joint bilateral blur operation.

**Output Example**: An instance of JointBilateralBlur could be represented as follows:
JointBilateralBlur(kernel_size=(5, 5), sigma_color=25.0, sigma_space=(10.0, 15.0), border_type='reflect', color_distance_type='l1')
### FunctionDef forward(self, input, guidance)
**forward**: The function of forward is to apply joint bilateral filtering on an input tensor using a guidance tensor.

**parameters**: The parameters of this Function.
· input: A Tensor representing the input image or data to be processed.  
· guidance: A Tensor that serves as a guidance image for the bilateral filtering.  

**Code Description**: The forward method is a member of the JointBilateralBlur class and is responsible for executing the joint bilateral blur operation. It takes two parameters: `input`, which is the tensor containing the image or data to be filtered, and `guidance`, which is a tensor that provides additional information to guide the filtering process. 

This method calls the `joint_bilateral_blur` function, passing along the input tensor, guidance tensor, and several parameters that define the filtering behavior, including `kernel_size`, `sigma_color`, `sigma_space`, `border_type`, and `color_distance_type`. These parameters are essential for controlling the characteristics of the filtering operation. 

The `joint_bilateral_blur` function itself is designed to perform joint bilateral filtering, a technique that smooths images while preserving edges by utilizing the guidance tensor. The filtering process is executed by the `_bilateral_blur` function, which is invoked within `joint_bilateral_blur`. This hierarchical structure allows the forward method to encapsulate the functionality of joint bilateral filtering, making it accessible for use in various applications, such as neural network architectures or image processing tasks.

The forward method ultimately returns a Tensor that represents the filtered image, which retains the original dimensions of the input tensor while exhibiting reduced noise and preserved edges. 

**Note**: It is important to ensure that the parameters passed to the joint_bilateral_blur function are valid. Invalid inputs, such as incorrect kernel sizes or improperly sized sigma parameters, may lead to runtime errors.

**Output Example**: A possible appearance of the code's return value could be a tensor representing the filtered image, maintaining the original dimensions of the input tensor while exhibiting reduced noise and preserved edges.
***
