## FunctionDef loglinear_interp(t_steps, num_steps)
**loglinear_interp**: The function of loglinear_interp is to perform log-linear interpolation on a given array of decreasing numbers.

**parameters**: The parameters of this Function.
· parameter1: t_steps - An array of decreasing numbers that represent the original data points to be interpolated.
· parameter2: num_steps - An integer that specifies the number of steps for the interpolation output.

**Code Description**: The loglinear_interp function takes an array of decreasing numbers (t_steps) and performs log-linear interpolation to generate a new array of values. The function begins by creating a normalized array (xs) that spans from 0 to 1, with the same length as the input array t_steps. It then computes the natural logarithm of the reversed t_steps array (ys), which prepares the data for interpolation in the logarithmic space.

Next, the function generates another normalized array (new_xs) that defines the new interpolation points based on the specified num_steps. Using NumPy's interp function, it interpolates the logarithmic values (ys) at the new x-coordinates (new_xs). After obtaining the interpolated logarithmic values (new_ys), the function exponentiates these values to revert back to the original scale and reverses the order of the resulting array to maintain the original decreasing order of t_steps.

The loglinear_interp function is called within the get_sigmas method of the AlignYourStepsScheduler class. In this context, it is used to adjust the noise levels (sigmas) based on the number of steps specified. If the length of the sigmas array does not match the expected number of steps, the loglinear_interp function is invoked to interpolate the noise levels accordingly. This ensures that the output sigmas array is appropriately sized for further processing, particularly in scenarios where denoising is applied.

**Note**: It is important to ensure that the input array t_steps is strictly decreasing, as the function relies on this property for accurate interpolation. Additionally, the num_steps parameter should be a positive integer to avoid errors during the interpolation process.

**Output Example**: An example output of the loglinear_interp function could be an array such as [0.1, 0.05, 0.02, 0.01], representing the interpolated values corresponding to the specified num_steps based on the input t_steps.
## ClassDef AlignYourStepsScheduler
**AlignYourStepsScheduler**: The function of AlignYourStepsScheduler is to compute a series of sigma values based on the model type, number of steps, and denoising factor.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types for the method, including model type, steps, and denoise parameters.  
· RETURN_TYPES: Specifies the return type of the method, which is a tuple containing "SIGMAS".  
· CATEGORY: Indicates the category under which this scheduler falls, specifically "sampling/custom_sampling/schedulers".  
· FUNCTION: The name of the function that will be executed, which is "get_sigmas".

**Code Description**: The AlignYourStepsScheduler class is designed to facilitate the generation of sigma values used in various sampling processes. The primary method, get_sigmas, takes three parameters: model_type, steps, and denoise. 

The method begins by determining the total number of steps to be used. If the denoise value is less than 1.0, it adjusts the total_steps based on the denoise factor, ensuring that if denoise is 0.0, an empty tensor is returned. 

Next, the method retrieves the noise levels corresponding to the specified model type from the NOISE_LEVELS dictionary. If the length of the noise levels does not match the expected number of steps, it performs a logarithmic linear interpolation to adjust the sigma values accordingly. 

Finally, the method slices the sigma values to return only the last total_steps + 1 values, setting the last value to 0 before converting the list to a FloatTensor and returning it as a tuple. 

This class is invoked within the calculate_sigmas_scheduler_hacked function found in the modules/sample_hijack.py file. When the scheduler_name is "align_your_steps", it determines the model type based on the latent format of the model and calls the get_sigmas method of AlignYourStepsScheduler to obtain the required sigma values. This integration allows for a flexible and customizable approach to sigma generation, accommodating different model types and sampling strategies.

**Note**: It is essential to ensure that the denoise parameter is within the specified range (0.0 to 1.0) to avoid unexpected behavior. The steps parameter must also be an integer within the defined limits (10 to 10000).

**Output Example**: A possible return value from the get_sigmas method could be a tensor containing the last few sigma values, such as:
```
tensor([0.1234, 0.5678, 0.9101, 0.0000])
```
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types for the configuration of the Align Your Steps Scheduler.

**parameters**: The parameters of this Function.
· s: This parameter is not utilized within the function body and serves as a placeholder for potential future use or to maintain a consistent function signature.

**Code Description**: The INPUT_TYPES function constructs a dictionary that specifies the required input types for the Align Your Steps Scheduler. The returned dictionary contains a single key, "required", which maps to another dictionary detailing the specific input parameters needed for the scheduler's operation. 

The inner dictionary includes the following keys and their corresponding value types:
- "model_type": This key expects a list of strings, specifically one of the following values: "SD1", "SDXL", or "SVD". This indicates the type of model that will be used in the scheduler.
- "steps": This key expects an integer value, which is constrained by a set of rules: it must be at least 10 (minimum) and can go up to 10,000 (maximum). The default value for this parameter is set to 10.
- "denoise": This key expects a floating-point number that represents the denoising level. The value must be between 0.0 (minimum) and 1.0 (maximum), with a default value of 1.0. Additionally, the value can be incremented in steps of 0.01.

This structured approach ensures that the inputs are validated against defined types and constraints, promoting robust configuration of the scheduler.

**Note**: It is important to ensure that the input values adhere to the specified types and constraints to avoid runtime errors. The function does not handle input validation; it merely defines the expected structure.

**Output Example**: A possible return value of the INPUT_TYPES function would look like this:
{
    "required": {
        "model_type": (["SD1", "SDXL", "SVD"], ),
        "steps": ("INT", {"default": 10, "min": 10, "max": 10000}),
        "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
    }
}
***
### FunctionDef get_sigmas(self, model_type, steps, denoise)
**get_sigmas**: The function of get_sigmas is to compute a tensor of noise levels (sigmas) based on the specified model type, number of steps, and denoising factor.

**parameters**: The parameters of this Function.
· parameter1: model_type - A string that indicates the type of model being used, which determines the noise levels to be applied.
· parameter2: steps - An integer representing the total number of steps for which the noise levels are to be calculated.
· parameter3: denoise - A float value that specifies the degree of denoising to be applied, where values less than 1.0 indicate a reduction in the number of effective steps.

**Code Description**: The get_sigmas function begins by initializing the total_steps variable to the value of the steps parameter. If the denoise parameter is less than 1.0, the function checks if denoise is less than or equal to 0.0. In such a case, it returns an empty tensor, indicating that no noise levels can be computed. If denoise is greater than 0.0, the function calculates the total_steps as the rounded product of steps and denoise, effectively reducing the number of steps based on the denoising factor.

Next, the function retrieves the noise levels (sigmas) corresponding to the specified model_type from a predefined NOISE_LEVELS dictionary. If the length of the sigmas array does not match the expected number of steps (steps + 1), the function calls loglinear_interp to perform log-linear interpolation on the sigmas array, adjusting it to the correct size.

After ensuring the sigmas array is appropriately sized, the function slices the last (total_steps + 1) elements from the sigmas array and sets the last element to 0, which is a common practice in noise scheduling to indicate no noise at the final step. Finally, the function returns a tuple containing a FloatTensor of the computed sigmas.

The get_sigmas function is called within the calculate_sigmas_scheduler_hacked function, which is responsible for selecting the appropriate sigma calculation method based on the specified scheduler name. In the case of the "align_your_steps" scheduler, the get_sigmas function is invoked with the model type determined by the latent format of the model. This integration allows for the dynamic adjustment of noise levels based on the model's characteristics and the specified denoising factor.

**Note**: It is essential to ensure that the denoise parameter is within the range [0.0, 1.0] to avoid unexpected behavior. Additionally, the model_type must correspond to a valid entry in the NOISE_LEVELS dictionary to ensure accurate noise level retrieval.

**Output Example**: An example output of the get_sigmas function could be a tensor such as [0.1, 0.05, 0.02, 0.0], representing the computed noise levels for the specified model type and steps, with the last value indicating no noise at the final step.
***
