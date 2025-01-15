## ClassDef BasicScheduler
**BasicScheduler**: The function of BasicScheduler is to calculate and return a series of sigma values based on the specified model, scheduler, number of steps, and denoise factor.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types for the scheduler, including the model, scheduler name, number of steps, and denoise factor.
· RETURN_TYPES: Specifies the type of output returned by the function, which is a tuple containing "SIGMAS".
· CATEGORY: Indicates the category under which this scheduler falls, specifically "sampling/custom_sampling/schedulers".
· FUNCTION: The name of the function that will be executed, which is "get_sigmas".

**Code Description**: The BasicScheduler class is designed to facilitate the calculation of sigma values used in sampling processes. It includes a class method, INPUT_TYPES, which outlines the necessary inputs for the scheduler. The required inputs are:
- model: A reference to the model being used, which is expected to be of type "MODEL".
- scheduler: A string representing the name of the scheduler, selected from predefined options in ldm_patched.modules.samplers.SCHEDULER_NAMES.
- steps: An integer indicating the number of steps for the sampling process, with a default value of 20 and constraints that it must be between 1 and 10,000.
- denoise: A floating-point number that represents the denoising factor, with a default of 1.0 and constraints that it must be between 0.0 and 1.0, with increments of 0.01.

The class also defines a method, get_sigmas, which takes the aforementioned parameters and computes the sigma values. Inside this method, the total number of steps is adjusted based on the denoise factor; if denoise is less than 1.0, total_steps is recalculated as steps divided by denoise. The model is then loaded onto the GPU using ldm_patched.modules.model_management.load_models_gpu. Subsequently, the method calls ldm_patched.modules.samplers.calculate_sigmas_scheduler to compute the sigma values based on the model, scheduler, and total_steps. The resulting sigma values are sliced to return only the last (steps + 1) values, which are then returned as a tuple.

**Note**: It is important to ensure that the model and scheduler provided are compatible and correctly configured. The denoise factor should be carefully chosen, as it directly affects the total number of steps and the resulting sigma values.

**Output Example**: An example of the output returned by the get_sigmas method could look like this:
(sigmas_tensor,) where sigmas_tensor is a tensor containing the calculated sigma values, for instance, a tensor of shape (21,) with values such as [0.1, 0.2, 0.3, ..., 2.0].
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types for a scheduling model configuration.

**parameters**: The parameters of this Function.
· parameter1: s - This parameter is typically used to represent the state or context in which the function is called, although it is not utilized within the function body.

**Code Description**: The INPUT_TYPES function constructs and returns a dictionary that specifies the required input types for a scheduling model. The dictionary contains a single key, "required", which maps to another dictionary detailing the specific inputs needed. The inputs include:
- "model": This input expects a value from a predefined set of model names, indicated by the tuple ("MODEL",).
- "scheduler": This input expects a value from the available scheduler names, which are sourced from the ldm_patched.modules.samplers.SCHEDULER_NAMES.
- "steps": This input is an integer that specifies the number of steps. It has a default value of 20 and must be within the range of 1 to 10,000.
- "denoise": This input is a floating-point number that indicates the denoising factor. It has a default value of 1.0 and must be between 0.0 and 1.0, with increments of 0.01.

The structure of the returned dictionary ensures that all necessary parameters for the scheduling model are clearly defined, along with their expected types and constraints.

**Note**: When using this function, ensure that the inputs provided conform to the specified types and constraints to avoid errors during model configuration.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "model": ("MODEL",),
        "scheduler": (ldm_patched.modules.samplers.SCHEDULER_NAMES,),
        "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
        "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
    }
}
***
### FunctionDef get_sigmas(self, model, scheduler, steps, denoise)
**get_sigmas**: The function of get_sigmas is to compute a sequence of sigma values for a given model and scheduler over a specified number of steps, adjusting for the level of denoising.

**parameters**: The parameters of this Function.
· model: An object representing the model from which sigma values are derived, which must have the necessary sampling attributes initialized.
· scheduler: A string indicating the scheduling method to be used for generating sigma values.
· steps: An integer representing the total number of steps for which sigma values are to be calculated.
· denoise: A float value indicating the level of denoising, which influences the total number of steps.

**Code Description**: The get_sigmas function begins by determining the total number of steps to be used for sigma calculation. If the denoise parameter is less than 1.0, it adjusts the total_steps by dividing the original steps by the denoise value, effectively increasing the number of steps to account for the denoising process. 

Next, the function calls the load_models_gpu function from the model_management module to load the specified model onto the GPU. This ensures that the model is ready for inference and that the necessary resources are allocated for computation.

Following the model loading, the function invokes calculate_sigmas_scheduler from the samplers module. This function is responsible for generating the sigma values based on the provided model, scheduler, and the adjusted total_steps. The resulting sigma values are then moved to the CPU for further processing and are sliced to retain only the last (steps + 1) values, which are relevant for the current operation.

The get_sigmas function returns a tuple containing the computed sigma values. This function plays a crucial role in the broader context of noise scheduling and sampling processes, as it provides the necessary sigma values that are used in various sampling strategies throughout the project.

**Note**: It is essential to ensure that the model passed to the function has the appropriate sampling attributes initialized, particularly the sigma_min and sigma_max values, to avoid runtime errors. Additionally, the scheduler parameter must be valid to ensure correct functionality.

**Output Example**: A possible return value from the get_sigmas function when called with a model, scheduler, steps=5, and denoise=0.8 might look like this:
```
(tensor([0.0000, 0.1000, 0.3000, 0.6000, 1.0000, 0.0000]),)
```
This output represents a tuple containing a tensor of sigma values generated for the specified steps, with the last value being 0.0.
***
## ClassDef KarrasScheduler
**KarrasScheduler**: The function of KarrasScheduler is to generate a sequence of sigma values used in the Karras diffusion sampling process.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input parameters for the class methods, including their types and constraints.  
· RETURN_TYPES: Specifies the type of output returned by the class methods.  
· CATEGORY: Indicates the category under which this class is organized within the project.  
· FUNCTION: The name of the method that will be executed to perform the main functionality of the class.

**Code Description**: The KarrasScheduler class is designed to facilitate the generation of sigma values for the Karras diffusion sampling method. It contains a class method called INPUT_TYPES that specifies the required parameters for the sigma generation process. These parameters include:
- steps: An integer that defines the number of steps in the sampling process, with a default value of 20 and a range between 1 and 10,000.
- sigma_max: A floating-point number representing the maximum sigma value, with a default of 14.614642 and a range from 0.0 to 1000.0.
- sigma_min: A floating-point number representing the minimum sigma value, with a default of 0.0291675 and a range from 0.0 to 1000.0.
- rho: A floating-point number that influences the distribution of sigma values, with a default of 7.0 and a range from 0.0 to 100.0.

The RETURN_TYPES attribute indicates that the output of the class methods will be a tuple containing the generated sigma values. The CATEGORY attribute categorizes this class under "sampling/custom_sampling/schedulers," which helps in organizing the codebase. The FUNCTION attribute specifies that the primary method for generating sigma values is named "get_sigmas."

The get_sigmas method takes the defined parameters and utilizes the k_diffusion_sampling.get_sigmas_karras function to compute the sigma values based on the provided inputs. The method returns a tuple containing the computed sigmas, which can then be used in various sampling applications.

**Note**: When using the KarrasScheduler class, ensure that the input parameters adhere to the specified constraints to avoid errors during the sigma generation process. The parameters should be carefully chosen based on the requirements of the specific sampling task.

**Output Example**: An example of the output from the get_sigmas method could be a tuple containing an array of sigma values, such as:
([14.614642, 14.0, 13.5, ..., 0.0291675],)
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types and their constraints for a specific configuration.

**parameters**: The parameters of this Function.
· s: This parameter is a placeholder for the function input, which is not utilized within the function body.

**Code Description**: The INPUT_TYPES function constructs and returns a dictionary that specifies the required input parameters for a certain process. The dictionary contains a single key, "required", which maps to another dictionary detailing the specific parameters needed. Each parameter is defined by a tuple that includes its data type and a dictionary of constraints. 

The parameters defined in this function are:
- "steps": This parameter is of type "INT" and has constraints that set its default value to 20, with a minimum value of 1 and a maximum value of 10,000.
- "sigma_max": This parameter is of type "FLOAT" with a default value of approximately 14.614642. It has constraints that allow values from 0.0 to 1000.0, with a step increment of 0.01 and no rounding applied.
- "sigma_min": This parameter is also of type "FLOAT", with a default value of approximately 0.0291675. Its constraints mirror those of "sigma_max", allowing values from 0.0 to 1000.0, with a step of 0.01 and no rounding.
- "rho": This parameter is of type "FLOAT" with a default value of 7.0. It has constraints that permit values from 0.0 to 100.0, with a step of 0.01 and no rounding.

This structured approach ensures that the inputs are validated against specified types and constraints, promoting robust configuration management.

**Note**: It is important to adhere to the defined constraints when providing input values to ensure proper functionality and avoid errors during execution.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
        "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 1000.0, "step": 0.01, "round": False}),
        "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 1000.0, "step": 0.01, "round": False}),
        "rho": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
    }
}
***
### FunctionDef get_sigmas(self, steps, sigma_max, sigma_min, rho)
**get_sigmas**: The function of get_sigmas is to generate a noise schedule based on the Karras et al. (2022) methodology.

**parameters**: The parameters of this Function.
· steps: An integer representing the number of noise levels to generate in the schedule.
· sigma_max: A float indicating the maximum sigma value in the noise schedule.
· sigma_min: A float indicating the minimum sigma value in the noise schedule.
· rho: A float that controls the shape of the noise schedule.

**Code Description**: The get_sigmas function is designed to create a noise schedule by invoking the get_sigmas_karras function from the k_diffusion_sampling module. It takes four parameters: steps, sigma_max, sigma_min, and rho. The steps parameter specifies how many noise levels are to be generated, while sigma_max and sigma_min define the range of sigma values in the schedule. The rho parameter influences the shape of the noise schedule, allowing for flexibility in how the noise levels transition from maximum to minimum.

Upon execution, get_sigmas calls the get_sigmas_karras function with the provided parameters, which constructs the noise schedule according to the principles established by Karras et al. (2022). The output of get_sigmas_karras is then returned as a tuple containing the generated sigmas. This function serves as a critical component in various sampling techniques within the project, as it provides the necessary noise levels for processes that require a structured noise schedule.

The get_sigmas function is utilized by other functions in the KarrasScheduler class, which rely on the generated noise schedule for their respective operations. This includes functions that handle sampling and noise management, ensuring that the noise levels are appropriately defined for the tasks at hand.

**Note**: When using get_sigmas, it is important to ensure that the input parameters are accurately specified, particularly the steps, sigma_max, sigma_min, and rho, as they directly affect the generated noise schedule.

**Output Example**: If the function is called with parameters steps=5, sigma_min=0.1, sigma_max=1.0, and rho=7.0, a possible return value could be a tuple containing a tensor like `(tensor([0.0000, 0.1000, 0.3000, 0.6000, 1.0000]),)`, where the tensor represents the generated noise levels.
***
## ClassDef ExponentialScheduler
**ExponentialScheduler**: The function of ExponentialScheduler is to generate a series of sigma values based on an exponential decay function.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input parameters for the class method.
· RETURN_TYPES: Specifies the type of output returned by the class method.
· CATEGORY: Indicates the category under which this class is organized.
· FUNCTION: Names the function that will be executed to generate the output.

**Code Description**: The ExponentialScheduler class is designed to facilitate the generation of sigma values using an exponential decay approach. It contains a class method called INPUT_TYPES that outlines the necessary parameters for its operation. The required parameters include:
- `steps`: An integer that determines the number of sigma values to generate, with a default value of 20 and a permissible range from 1 to 10,000.
- `sigma_max`: A floating-point number representing the maximum sigma value, defaulting to approximately 14.614642, with a range from 0.0 to 1000.0.
- `sigma_min`: A floating-point number indicating the minimum sigma value, defaulting to approximately 0.0291675, also within the range of 0.0 to 1000.0.

The class defines a return type of "SIGMAS", indicating that the output will consist of the generated sigma values. The CATEGORY attribute categorizes this class under "sampling/custom_sampling/schedulers", which helps in organizing the functionality within the broader context of sampling techniques.

The core functionality is encapsulated in the `get_sigmas` method, which takes the defined parameters as input. This method utilizes the `k_diffusion_sampling.get_sigmas_exponential` function to compute the sigma values based on the provided `steps`, `sigma_min`, and `sigma_max`. The result is returned as a tuple containing the generated sigma values.

**Note**: When using the ExponentialScheduler, ensure that the input parameters are within the specified ranges to avoid errors. The method is designed to handle a variety of scenarios, but adhering to the defined constraints will ensure optimal performance.

**Output Example**: An example of the output returned by the `get_sigmas` method might look like this: 
```
([0.0291675, 0.058335, 0.11667, 0.23334, 0.46668, 0.93336, 1.86672, 3.73344, 7.46688, 14.614642],)
``` 
This output represents a list of sigma values generated based on the specified parameters.
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types and their constraints for a specific configuration.

**parameters**: The parameters of this Function.
· steps: An integer parameter that specifies the number of steps, with a default value of 20, and must be between 1 and 10000.
· sigma_max: A floating-point parameter that represents the maximum sigma value, with a default of 14.614642, and must be between 0.0 and 1000.0, allowing increments of 0.01.
· sigma_min: A floating-point parameter that represents the minimum sigma value, with a default of 0.0291675, and must be between 0.0 and 1000.0, allowing increments of 0.01.

**Code Description**: The INPUT_TYPES function is designed to return a dictionary that specifies the required input parameters for a particular operation. The dictionary contains a key "required" which maps to another dictionary detailing three specific parameters: "steps", "sigma_max", and "sigma_min". Each parameter is associated with its data type (either "INT" for integer or "FLOAT" for floating-point) and a set of constraints. The "steps" parameter is constrained to be an integer within the range of 1 to 10000, while both "sigma_max" and "sigma_min" are constrained floating-point numbers that must be within the range of 0.0 to 1000.0. Additionally, these floating-point parameters can be adjusted in increments of 0.01, and they do not require rounding.

**Note**: It is important to ensure that the values provided for each parameter adhere to the specified constraints to avoid errors during execution. The default values can be overridden by the user, but they must still comply with the defined limits.

**Output Example**: A possible return value of the INPUT_TYPES function could look like this:
{
    "required": {
        "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
        "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 1000.0, "step": 0.01, "round": False}),
        "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 1000.0, "step": 0.01, "round": False})
    }
}
***
### FunctionDef get_sigmas(self, steps, sigma_max, sigma_min)
**get_sigmas**: The function of get_sigmas is to generate a sequence of sigma values based on an exponential noise schedule.

**parameters**: The parameters of this Function.
· steps: An integer representing the number of sigma values to generate in the schedule.  
· sigma_max: A float that specifies the maximum value of sigma in the schedule.  
· sigma_min: A float that specifies the minimum value of sigma in the schedule.  

**Code Description**: The get_sigmas function utilizes the get_sigmas_exponential function from the k_diffusion_sampling module to create a series of sigma values that follow an exponential distribution. It takes three parameters: steps, sigma_max, and sigma_min. The steps parameter determines how many sigma values will be generated, while sigma_max and sigma_min define the range of these values. The function calls get_sigmas_exponential with these parameters, which constructs the exponential noise schedule and returns the generated sigma values. The output of get_sigmas is a tuple containing the generated sigma values, ensuring compatibility with other components that may expect a tuple format.

The get_sigmas function is integral to the ExponentialScheduler class, which is part of the external_custom_sampler module. This relationship indicates that get_sigmas is used to facilitate the scheduling of noise values in various sampling processes, contributing to the overall functionality of the noise scheduling mechanism within the project.

**Note**: It is essential to provide valid input values for steps, sigma_max, and sigma_min to ensure the correct generation of sigma values. The function assumes that sigma_max is greater than sigma_min to produce a meaningful schedule.

**Output Example**: If the input parameters are steps=5, sigma_min=0.1, and sigma_max=1.0, the return value of get_sigmas might be a tuple containing a tensor like `(torch.tensor([0.1, 0.215, 0.462, 1.0, 0.0]),)`, where the last element is the appended zero.
***
## ClassDef PolyexponentialScheduler
**PolyexponentialScheduler**: The function of PolyexponentialScheduler is to generate a sequence of sigma values using a polyexponential decay function.

**attributes**: The attributes of this Class.
· steps: An integer representing the number of steps for the sigma generation, with a default value of 20, and must be between 1 and 10000.
· sigma_max: A float indicating the maximum value of sigma, defaulting to 14.614642, with a permissible range from 0.0 to 1000.0.
· sigma_min: A float representing the minimum value of sigma, with a default of 0.0291675, and must be between 0.0 and 1000.0.
· rho: A float that controls the decay rate, defaulting to 1.0, and must be within the range of 0.0 to 100.0.

**Code Description**: The PolyexponentialScheduler class is designed to facilitate the generation of sigma values for sampling processes. It includes a class method, INPUT_TYPES, which defines the expected input parameters for the sigma generation function. This method specifies that the required inputs are steps, sigma_max, sigma_min, and rho, along with their respective data types and constraints. The RETURN_TYPES attribute indicates that the output of the class will be a tuple containing the generated sigmas. The CATEGORY attribute categorizes this class under custom sampling schedulers. The core functionality is encapsulated in the get_sigmas method, which takes the specified parameters and utilizes the k_diffusion_sampling.get_sigmas_polyexponential function to compute the sigmas based on the provided inputs. The method returns the computed sigmas as a single-element tuple.

**Note**: It is important to ensure that the input parameters adhere to the specified constraints to avoid errors during execution. The values of sigma_max and sigma_min should be set appropriately to achieve the desired range of sigma values.

**Output Example**: An example output of the get_sigmas method when called with steps=20, sigma_max=14.614642, sigma_min=0.0291675, and rho=1.0 might look like this: 
([14.614642, 14.123456, 13.987654, ..., 0.0291675],) 
This output represents a tuple containing a list of sigma values generated according to the polyexponential decay function.
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types and their constraints for a specific configuration.

**parameters**: The parameters of this Function.
· steps: An integer representing the number of steps, with a default value of 20, and must be between 1 and 10000.
· sigma_max: A floating-point number representing the maximum sigma value, with a default of 14.614642, and must be between 0.0 and 1000.0, with a step of 0.01.
· sigma_min: A floating-point number representing the minimum sigma value, with a default of 0.0291675, and must be between 0.0 and 1000.0, with a step of 0.01.
· rho: A floating-point number representing the rho value, with a default of 1.0, and must be between 0.0 and 100.0, with a step of 0.01.

**Code Description**: The INPUT_TYPES function is designed to provide a structured output that specifies the required input parameters for a particular operation or configuration. It returns a dictionary containing a single key, "required", which maps to another dictionary. This inner dictionary defines four parameters: "steps", "sigma_max", "sigma_min", and "rho". Each parameter is associated with a tuple that includes its data type and a dictionary of constraints. The constraints specify default values, minimum and maximum allowable values, and additional properties such as step increments and rounding behavior. This structured approach ensures that users of the function have clear guidelines on what inputs are necessary and the valid ranges for those inputs.

**Note**: It is important to adhere to the specified constraints for each parameter to ensure proper functionality. Users should ensure that the values provided fall within the defined ranges to avoid errors during execution.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
        "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 1000.0, "step": 0.01, "round": False}),
        "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 1000.0, "step": 0.01, "round": False}),
        "rho": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
    }
}
***
### FunctionDef get_sigmas(self, steps, sigma_max, sigma_min, rho)
**get_sigmas**: The function of get_sigmas is to generate a sequence of sigma values based on a polynomial noise schedule.

**parameters**: The parameters of this Function.
· steps: An integer representing the number of sigma values to generate.  
· sigma_max: A float that specifies the maximum sigma value.  
· sigma_min: A float that specifies the minimum sigma value.  
· rho: A float that controls the shape of the polynomial ramp.

**Code Description**: The get_sigmas function is a method of the PolyexponentialScheduler class, which is designed to facilitate the generation of sigma values used in diffusion sampling processes. This function calls the get_sigmas_polyexponential function from the k_diffusion_sampling module, passing four parameters: steps, sigma_max, sigma_min, and rho. The purpose of this call is to obtain a tensor of sigma values that adhere to a polynomial schedule in logarithmic space.

The get_sigmas_polyexponential function constructs a polynomial in the log sigma noise schedule by creating a ramp tensor that decreases linearly from 1 to 0 over the specified number of steps. This ramp is raised to the power of rho, allowing for control over the shape of the polynomial ramp. The logarithmic interpolation between sigma_max and sigma_min is performed, and the resulting tensor is exponentiated to convert it back to the original scale. The output from get_sigmas_polyexponential is then returned as a tuple containing the tensor of sigma values.

This method is crucial for the PolyexponentialScheduler, as it provides the necessary sigma values that dictate the noise schedule for various sampling techniques. The proper definition of input parameters, particularly steps, sigma_min, sigma_max, and rho, is essential to ensure the function operates without errors.

**Note**: It is important to ensure that the input parameters are defined correctly to avoid runtime errors. The values for sigma_min and sigma_max should be chosen such that sigma_min is less than sigma_max to maintain a valid range for the generated sigma values.

**Output Example**: If the input parameters are steps=5, sigma_min=0.1, sigma_max=1.0, and rho=1.0, the return value of get_sigmas might look like a tuple containing a tensor such as `(torch.tensor([0.1, 0.275, 0.615, 0.866, 1.0]),)`.
***
## ClassDef SDTurboScheduler
**SDTurboScheduler**: The function of SDTurboScheduler is to compute sigma values for a given model based on specified steps and a denoising factor.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types for the class method.
· RETURN_TYPES: Specifies the return type of the class method.
· CATEGORY: Indicates the category under which this scheduler falls.
· FUNCTION: The name of the function that will be executed to obtain the sigmas.

**Code Description**: The SDTurboScheduler class is designed to facilitate the generation of sigma values used in sampling processes within machine learning models, particularly in diffusion models. The class contains a class method called INPUT_TYPES, which outlines the required inputs: a model, the number of steps (an integer), and a denoising factor (a float). The method specifies default values and constraints for these inputs, ensuring that they fall within acceptable ranges.

The RETURN_TYPES attribute indicates that the output of the class method will be a tuple containing the computed sigmas. The CATEGORY attribute categorizes this scheduler under "sampling/custom_sampling/schedulers," which helps in organizing different sampling strategies within the project. The FUNCTION attribute specifies that the core functionality of this class is encapsulated in the method named "get_sigmas."

The get_sigmas method takes three parameters: model, steps, and denoise. It calculates the starting step based on the denoise factor, which influences how much noise is applied during the sampling process. The method then generates a range of timesteps, reverses them, and selects a subset based on the starting step and the number of steps requested. Subsequently, it computes the sigma values using the model's sampling method and appends a zero value to the end of the sigmas tensor before returning it.

In the project, the SDTurboScheduler is invoked within the calculate_sigmas_scheduler_hacked function found in the modules/sample_hijack.py file. When the scheduler name is "turbo," the get_sigmas method of the SDTurboScheduler class is called to obtain the sigma values. This integration highlights the role of the SDTurboScheduler in providing a specific sampling strategy that can be selected based on user-defined parameters.

**Note**: Users should ensure that the model provided to the get_sigmas method is compatible with the expected input types, and that the steps and denoise parameters are within the defined limits to avoid runtime errors.

**Output Example**: A possible appearance of the code's return value could be a tensor containing sigma values, such as: 
```
tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0])
```
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return a structured dictionary that specifies the required input types for a model configuration.

**parameters**: The parameters of this Function.
· parameter1: s - This parameter is not utilized within the function body and serves as a placeholder for potential future use or for maintaining a consistent function signature.

**Code Description**: The INPUT_TYPES function constructs and returns a dictionary that outlines the necessary input types for a specific model configuration. The returned dictionary contains a single key, "required", which maps to another dictionary detailing the specific inputs needed. 

Within this inner dictionary, there are three keys:
- "model": This key is associated with a tuple containing the string "MODEL". This indicates that the input for the model must be of type "MODEL".
- "steps": This key is linked to a tuple that includes the string "INT" and a dictionary defining constraints on the integer input. The constraints specify that the default value is 1, with a minimum value of 1 and a maximum value of 10.
- "denoise": This key is associated with a tuple that includes the string "FLOAT" and a dictionary that outlines the parameters for the floating-point input. The default value is set to 1.0, with a minimum of 0, a maximum of 1.0, and a step increment of 0.01.

This structured approach ensures that the inputs are clearly defined, allowing for validation and proper handling of the model configuration.

**Note**: It is important to ensure that the inputs provided conform to the specified types and constraints to avoid errors during model execution. The function does not perform any validation itself; it merely defines the expected structure.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "model": ("MODEL",),
        "steps": ("INT", {"default": 1, "min": 1, "max": 10}),
        "denoise": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01})
    }
}
***
### FunctionDef get_sigmas(self, model, steps, denoise)
**get_sigmas**: The function of get_sigmas is to compute the sigma values used in the sampling process of a model based on the specified number of steps and the denoising factor.

**parameters**: The parameters of this Function.
· model: An instance of a model that contains the sampling method and sigma parameters.
· steps: An integer representing the number of steps for which sigma values are to be calculated.
· denoise: A float value indicating the level of denoising to be applied, which influences the starting step for sigma calculation.

**Code Description**: The get_sigmas function begins by determining the starting step for sigma calculation based on the denoise parameter. It calculates the starting step as 10 minus the product of 10 and the denoise value, effectively adjusting the starting point based on how much denoising is desired. The function then generates a tensor of timesteps by creating a range from 1 to 10, multiplying each by 100, and subtracting 1. This tensor is then flipped and sliced to obtain the relevant timesteps starting from the calculated start_step for the specified number of steps.

Next, the function calls the model's sampling method to compute the sigma values corresponding to the generated timesteps. After obtaining the sigma values, it appends a zero tensor of size one to the end of the sigma tensor, ensuring that the output has a consistent shape. Finally, the function returns the modified sigma tensor as a single-element tuple.

This function is called within the calculate_sigmas_scheduler_hacked function, which is responsible for selecting the appropriate sigma calculation method based on the provided scheduler name. When the scheduler name is "turbo", the get_sigmas function is invoked with a denoise value of 1.0, indicating that no denoising is to be applied. This integration highlights the role of get_sigmas in the broader context of sigma calculation for different sampling strategies, specifically within the turbo scheduler.

**Note**: It is important to ensure that the model passed to this function has the necessary sampling methods and sigma parameters defined, as the function relies on these to compute the sigma values correctly.

**Output Example**: A possible return value of the function could be a tensor containing sigma values such as (tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0]),).
***
## ClassDef VPScheduler
**VPScheduler**: The function of VPScheduler is to generate a sequence of sigma values used in variational posterior sampling.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types and their constraints for the class method.
· RETURN_TYPES: Specifies the type of output returned by the class method.
· CATEGORY: Indicates the category under which this class is organized.
· FUNCTION: Specifies the name of the function that will be executed.

**Code Description**: VPScheduler is a class designed to facilitate the generation of sigma values for variational posterior sampling. It contains a class method called INPUT_TYPES that outlines the required parameters for its operation, including 'steps', 'beta_d', 'beta_min', and 'eps_s'. Each parameter is associated with specific data types and constraints, such as default values, minimum and maximum limits, and step sizes. The RETURN_TYPES attribute indicates that the output of the class will be a tuple containing sigma values. The CATEGORY attribute categorizes this class under "sampling/custom_sampling/schedulers", which helps in organizing the functionality within the broader context of the project. The FUNCTION attribute specifies that the main operation of this class is executed through the 'get_sigmas' method. 

The 'get_sigmas' method takes four parameters: steps (an integer representing the number of steps), beta_d (a float representing the diffusion coefficient), beta_min (a float representing the minimum beta value), and eps_s (a float representing a small epsilon value for numerical stability). This method calls an external function, 'k_diffusion_sampling.get_sigmas_vp', to compute the sigma values based on the provided parameters and returns these values as a tuple.

**Note**: When using this class, ensure that the input parameters adhere to the specified constraints to avoid runtime errors. The default values provided can be adjusted as needed, but they should remain within the defined limits.

**Output Example**: An example of the output from the 'get_sigmas' method could be a tuple containing an array of sigma values, such as: ([0.1, 0.2, 0.3, ..., 19.9],).
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return a dictionary of required input parameters for a specific configuration.

**parameters**: The parameters of this Function.
· s: This parameter is a placeholder for the function's input, which is not utilized within the function body.

**Code Description**: The INPUT_TYPES function constructs a dictionary that specifies the required input parameters for a certain process or algorithm. The dictionary contains a single key, "required", which maps to another dictionary that defines various parameters. Each parameter is associated with a tuple that specifies its data type and a dictionary of constraints. The parameters included are:

- "steps": This parameter is of type "INT" and has a default value of 20. It accepts integer values ranging from 1 to 10,000.
- "beta_d": This parameter is of type "FLOAT" with a default value of 19.9. It allows floating-point values between 0.0 and 1000.0, with a step size of 0.01. The rounding option is set to False, indicating that values do not need to be rounded.
- "beta_min": Similar to "beta_d", this parameter is also of type "FLOAT" with a default value of 0.1, allowing values from 0.0 to 1000.0, with the same step size and rounding option.
- "eps_s": This parameter is another "FLOAT" type, with a default value of 0.001. It accepts values from 0.0 to 1.0, with a step size of 0.0001 and no rounding.

The function effectively standardizes the input requirements for the associated process, ensuring that users provide valid and expected values.

**Note**: It is important to ensure that the input values provided adhere to the specified constraints to avoid errors during execution. The default values serve as initial settings but can be adjusted as needed within the defined ranges.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
        "beta_d": ("FLOAT", {"default": 19.9, "min": 0.0, "max": 1000.0, "step": 0.01, "round": False}),
        "beta_min": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1000.0, "step": 0.01, "round": False}),
        "eps_s": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001, "round": False}),
    }
}
***
### FunctionDef get_sigmas(self, steps, beta_d, beta_min, eps_s)
**get_sigmas**: The function of get_sigmas is to generate a tensor of sigma values based on a specified number of steps and diffusion parameters.

**parameters**: The parameters of this Function.
· steps: An integer representing the number of time steps for which the sigma values are to be generated.  
· beta_d: A float that controls the diffusion rate.  
· beta_min: A float that sets the minimum value for the beta parameter.  
· eps_s: A small float value used to define the lower bound of the time steps.  

**Code Description**: The get_sigmas function serves as a wrapper around the k_diffusion_sampling.get_sigmas_vp function, which constructs a continuous VP noise schedule. It takes four parameters: steps, beta_d, beta_min, and eps_s. The function calls get_sigmas_vp with these parameters, where 'steps' determines the number of time steps for which the noise schedule is generated, while beta_d, beta_min, and eps_s control the characteristics of the noise schedule. The output of get_sigmas is a tuple containing the generated tensor of sigma values. This integration allows the VPScheduler class to effectively utilize the noise schedule generation within its broader sampling framework, ensuring that the generated sigmas are consistent with the diffusion process defined by the input parameters.

**Note**: It is essential to provide valid input parameters to ensure the correct generation of sigma values. The function is designed to work within the context of the VPScheduler class, which is part of the external_custom_sampler module.

**Output Example**: If the input parameter 'steps' is set to 5, the function might return a tuple containing a tensor similar to (torch.tensor([sigma_1, sigma_2, sigma_3, sigma_4, sigma_5]),), where each sigma_i represents the computed noise value for the corresponding time step.
***
## ClassDef SplitSigmas
**SplitSigmas**: The function of SplitSigmas is to split a list of sigma values into two parts based on a specified step index.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types for the class method.
· RETURN_TYPES: Specifies the types of values returned by the class method.
· CATEGORY: Indicates the category under which this class is organized.
· FUNCTION: The name of the function that performs the main operation of the class.

**Code Description**: The SplitSigmas class is designed to facilitate the splitting of a list of sigma values into two separate lists based on a given index, referred to as "step." The class contains a class method INPUT_TYPES that outlines the expected input parameters: "sigmas," which is a list of sigma values, and "step," which is an integer that determines the index at which the split occurs. The method also defines constraints for the "step" parameter, setting a default value of 0 and allowing values from 0 to 10,000. The RETURN_TYPES attribute indicates that the output will consist of two lists of sigma values. The CATEGORY attribute categorizes this class under "sampling/custom_sampling/sigmas." The main functionality is encapsulated in the get_sigmas method, which takes the input parameters and returns two lists: the first list contains sigma values from the start up to the specified step index (inclusive), while the second list contains the remaining sigma values from the step index onward.

**Note**: When using this class, ensure that the "step" parameter does not exceed the length of the "sigmas" list to avoid index errors. The "sigmas" input should be a valid list of sigma values for the function to operate correctly.

**Output Example**: For an input of sigmas = [0.1, 0.2, 0.3, 0.4, 0.5] and step = 2, the return value would be ([0.1, 0.2, 0.3], [0.4, 0.5]).
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define the required input types for a specific configuration involving sigmas and a step parameter.

**parameters**: The parameters of this Function.
· parameter1: s - This parameter is a placeholder for any input that may be passed to the function, though it is not utilized within the function body.

**Code Description**: The INPUT_TYPES function returns a dictionary that specifies the required input types for a certain operation. The dictionary contains a single key, "required", which maps to another dictionary detailing the specific inputs needed. Within this inner dictionary, there are two keys: "sigmas" and "step". 

- The "sigmas" key is associated with a tuple containing the string "SIGMAS", indicating that this input type is expected to be of a specific format or category labeled as SIGMAS.
- The "step" key is linked to a tuple that includes the string "INT" and a second dictionary that defines constraints for this integer input. The constraints specify a default value of 0, a minimum value of 0, and a maximum value of 10000. This means that the step parameter must be an integer within this defined range, with 0 being the default if no value is provided.

Overall, the function is structured to ensure that the inputs adhere to the specified types and constraints, facilitating proper validation and usage in subsequent operations.

**Note**: It is important to ensure that the inputs provided to the function conform to the defined types and constraints to avoid errors during execution. The "sigmas" input should be appropriately formatted as expected, and the "step" input must be an integer within the specified range.

**Output Example**: A possible return value of the INPUT_TYPES function could look like this:
{
    "required": {
        "sigmas": ("SIGMAS", ),
        "step": ("INT", {"default": 0, "min": 0, "max": 10000})
    }
}
***
### FunctionDef get_sigmas(self, sigmas, step)
**get_sigmas**: The function of get_sigmas is to split a list of sigma values into two separate lists based on a specified step index.

**parameters**: The parameters of this Function.
· parameter1: sigmas - A list of sigma values that need to be split.
· parameter2: step - An integer index that determines the point at which the list will be divided.

**Code Description**: The get_sigmas function takes two parameters: a list of sigma values and an integer step. It creates two new lists from the original list of sigmas. The first list, sigmas1, contains all elements from the beginning of the sigmas list up to and including the index specified by step. The second list, sigmas2, contains all elements from the index specified by step to the end of the sigmas list. The function then returns a tuple containing both lists. This allows for easy access to two segments of the original list, which can be useful in various applications, such as sampling or data processing tasks.

**Note**: It is important to ensure that the step parameter is within the bounds of the sigmas list to avoid index errors. If step is greater than the length of sigmas, the function will still return a valid tuple, but sigmas1 will contain all elements of sigmas, and sigmas2 will be empty.

**Output Example**: If the input sigmas is [0.1, 0.2, 0.3, 0.4, 0.5] and the step is 2, the function will return the following:
([0.1, 0.2, 0.3], [0.3, 0.4, 0.5])
***
## ClassDef FlipSigmas
**FlipSigmas**: The function of FlipSigmas is to manipulate a tensor of sigmas by flipping its values and ensuring the first element is not zero.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Specifies the required input types for the class method.  
· RETURN_TYPES: Defines the output types returned by the class method.  
· CATEGORY: Indicates the category under which this class is organized.  
· FUNCTION: Represents the name of the method that performs the main functionality.

**Code Description**: The FlipSigmas class is designed to handle a tensor of sigmas, which are typically used in sampling processes. The class contains a class method `INPUT_TYPES` that specifies the expected input format, which requires a single parameter named `sigmas`. The `RETURN_TYPES` attribute indicates that the output will also be a tensor of sigmas. The `CATEGORY` attribute categorizes this class under "sampling/custom_sampling/sigmas", which helps in organizing the functionality within a larger framework. The `FUNCTION` attribute specifies that the main operation is performed by the `get_sigmas` method.

The `get_sigmas` method takes the input tensor `sigmas`, flips it along the first dimension (dimension 0), and checks if the first element of the flipped tensor is zero. If it is zero, the method sets it to a small value of 0.0001 to avoid issues that may arise from having a zero value. Finally, the method returns the modified tensor as a single-element tuple.

**Note**: It is important to ensure that the input tensor is of the correct shape and type as specified in the `INPUT_TYPES`. The method is designed to handle tensors efficiently, but users should be aware of the implications of modifying the first element to a small value if it was originally zero.

**Output Example**: If the input tensor `sigmas` is given as `tensor([0, 1, 2])`, the output after processing through `get_sigmas` would be `tensor([0.0001, 1, 2])`.
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define the required input types for a specific operation involving sigmas.

**parameters**: The parameters of this Function.
· s: This parameter is not utilized within the function but is typically included to maintain a consistent function signature.

**Code Description**: The INPUT_TYPES function returns a dictionary that specifies the required input types for a process. The dictionary contains a key "required", which maps to another dictionary. This inner dictionary has a key "sigmas" that is associated with a tuple containing a single string "SIGMAS". This structure indicates that the function expects an input labeled "sigmas" of type "SIGMAS". The design of this function allows for easy extension and modification of input requirements in the future.

**Note**: It is important to ensure that any input provided to the function adheres to the specified structure, as deviations may lead to errors in processing. The function is primarily used to enforce input validation for subsequent operations that depend on the "sigmas" input.

**Output Example**: An example of the return value from the INPUT_TYPES function would be:
{
    "required": {
        "sigmas": ("SIGMAS", )
    }
}
***
### FunctionDef get_sigmas(self, sigmas)
**get_sigmas**: The function of get_sigmas is to manipulate the input tensor of sigmas by flipping it and ensuring the first element is non-zero.

**parameters**: The parameters of this Function.
· sigmas: A tensor that contains sigma values, which are expected to be manipulated by the function.

**Code Description**: The get_sigmas function takes a tensor input called sigmas. The first operation performed is flipping the tensor along the first dimension (dimension 0). This means that the order of the elements in the tensor is reversed. After flipping, the function checks if the first element of the flipped tensor is equal to zero. If it is, the function assigns a very small value (0.0001) to this first element to prevent it from being zero, which could lead to issues in subsequent computations that rely on this value being non-zero. Finally, the function returns the modified sigmas tensor as a single-element tuple.

**Note**: It is important to ensure that the input sigmas tensor has at least one element before calling this function to avoid index errors. Additionally, the function assumes that the input is a PyTorch tensor, as it utilizes the flip method specific to PyTorch.

**Output Example**: If the input sigmas tensor is [0, 1, 2], the output after processing would be (tensor([0.0001, 2, 1]),). If the input is [3, 4, 5], the output would be (tensor([5, 4, 3]),).
***
## ClassDef KSamplerSelect
**KSamplerSelect**: The function of KSamplerSelect is to retrieve a sampler object based on a specified sampler name.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types for the class method.
· RETURN_TYPES: Specifies the types of values returned by the class method.
· CATEGORY: Indicates the category under which this class is organized.
· FUNCTION: The name of the function that this class implements.

**Code Description**: The KSamplerSelect class is designed to facilitate the selection of a sampler object from a predefined list of sampler names. It contains a class method called INPUT_TYPES, which specifies that the input to the class must include a parameter named "sampler_name." This parameter is expected to be one of the values from the ldm_patched.modules.samplers.SAMPLER_NAMES collection. The class also defines RETURN_TYPES, which indicates that the output of the class will be a tuple containing a single element labeled "SAMPLER." The CATEGORY attribute categorizes this class under "sampling/custom_sampling/samplers," which helps in organizing the functionality within the broader context of the project.

The core functionality of the class is encapsulated in the get_sampler method. This instance method takes a single argument, sampler_name, and uses it to retrieve the corresponding sampler object by calling ldm_patched.modules.samplers.sampler_object(sampler_name). The method then returns a tuple containing the retrieved sampler object. This design allows for a straightforward way to access different sampler implementations based on user input.

**Note**: When using this class, ensure that the sampler_name provided is valid and exists within the SAMPLER_NAMES collection to avoid errors during the retrieval process.

**Output Example**: If the input sampler_name is "default_sampler," the return value of the get_sampler method might look like this: (SamplerObjectInstance,). Here, SamplerObjectInstance represents the actual sampler object that corresponds to the specified name.
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define the required input types for a specific sampler configuration.

**parameters**: The parameters of this Function.
· s: This parameter is typically used as a context or state object, though its specific usage is not detailed in the provided code.

**Code Description**: The INPUT_TYPES function returns a dictionary that specifies the required input types for a sampler. The dictionary contains a single key, "required", which maps to another dictionary. This inner dictionary defines the expected input parameters for the sampler configuration. In this case, it specifies that the "sampler_name" parameter is required, and its value is a tuple containing the reference to `ldm_patched.modules.samplers.SAMPLER_NAMES`. This indicates that the "sampler_name" must be one of the predefined sampler names available in the `SAMPLER_NAMES` collection.

The structure of the returned dictionary is crucial for ensuring that the correct parameters are provided when configuring the sampler. By enforcing the requirement for "sampler_name", the function helps maintain the integrity of the sampler's configuration process.

**Note**: It is important to ensure that the input provided for "sampler_name" matches one of the valid entries defined in `ldm_patched.modules.samplers.SAMPLER_NAMES` to avoid errors during the sampler's execution.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "sampler_name": (["sampler1", "sampler2", "sampler3"],)
    }
} 
This output indicates that the sampler configuration requires a "sampler_name" that must be one of the specified sampler options.
***
### FunctionDef get_sampler(self, sampler_name)
**get_sampler**: The function of get_sampler is to retrieve a sampling object based on the specified sampler name.

**parameters**: The parameters of this Function.
· sampler_name: A string that specifies the name of the sampling method to be instantiated. This name is passed to the sampler_object function to create the appropriate sampler instance.

**Code Description**: The get_sampler function is a method within the KSamplerSelect class that serves to obtain a specific sampling object based on the provided sampler_name. It utilizes the sampler_object function from the ldm_patched.modules.samplers module to create and return an instance of the desired sampler.

When get_sampler is called, it takes the sampler_name as an argument and passes it to the sampler_object function. The sampler_object function acts as a factory method, determining which specific sampler class to instantiate based on the input name. This design allows for flexibility in selecting different sampling strategies, such as "uni_pc", "uni_pc_bh2", or "ddim", among others.

The output of the get_sampler function is a tuple containing the instantiated sampler object. This output can then be used by other components in the project that require a specific sampling method for their operations. The relationship between get_sampler and sampler_object is crucial, as get_sampler relies on the latter to ensure that the correct sampler is instantiated based on user input.

**Note**: When using the get_sampler function, it is essential to provide a valid sampler_name that corresponds to one of the recognized sampling methods. Users should also be aware of the specific parameters and configurations required by the instantiated sampler classes to achieve the desired sampling results.

**Output Example**: The output of the get_sampler function could be a tuple containing an instance of a sampler class, such as (UNIPC(model_wrap=<model>, sigmas=[0.1, 0.2, 0.3], ...),) or (UNIPCBH2(model_wrap=<model>, sigmas=[0.1, 0.2, 0.3], ...),) depending on the sampler_name provided.
***
## ClassDef SamplerDPMPP_2M_SDE
**SamplerDPMPP_2M_SDE**: The function of SamplerDPMPP_2M_SDE is to create a sampler for the DPM++ 2M SDE algorithm based on specified parameters.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types for the sampler configuration.
· RETURN_TYPES: Specifies the type of output returned by the sampler.
· CATEGORY: Indicates the category under which this sampler falls.
· FUNCTION: The name of the function that will be called to obtain the sampler.

**Code Description**: The SamplerDPMPP_2M_SDE class is designed to facilitate the creation of a sampling object for the DPM++ 2M SDE algorithm. It includes a class method, INPUT_TYPES, which outlines the necessary parameters that must be provided when initializing the sampler. These parameters include:
- solver_type: A choice between 'midpoint' and 'heun', which determines the numerical method used for solving differential equations.
- eta: A floating-point value that represents a scaling factor, with a default of 1.0 and a range from 0.0 to 100.0.
- s_noise: Another floating-point value that specifies the noise level, also defaulting to 1.0 and within the same range as eta.
- noise_device: A selection between 'gpu' and 'cpu' that indicates the device on which the sampling will be performed.

The class also defines RETURN_TYPES, which indicates that the output will be of type "SAMPLER". The CATEGORY attribute categorizes this sampler under "sampling/custom_sampling/samplers". The FUNCTION attribute specifies that the method to be called for obtaining the sampler is "get_sampler".

The core functionality is encapsulated in the get_sampler method, which takes the specified parameters and determines the appropriate sampler name based on the noise_device. If the noise_device is set to 'cpu', the sampler name is "dpmpp_2m_sde"; if it is set to 'gpu', the sampler name becomes "dpmpp_2m_sde_gpu". The method then calls the ksampler function from the ldm_patched.modules.samplers module, passing the sampler name along with a dictionary containing the eta, s_noise, and solver_type parameters. Finally, the method returns a tuple containing the created sampler.

**Note**: When using this class, ensure that the input parameters adhere to the specified types and ranges to avoid errors during sampler creation. The choice of noise_device will affect the performance and capabilities of the sampler.

**Output Example**: An example of the return value when calling get_sampler with appropriate parameters might look like this:
(sampler_object,) where sampler_object is an instance of the sampler configured with the provided parameters.
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types for a specific sampling process.

**parameters**: The parameters of this Function.
· s: This parameter is a placeholder that is typically used to represent the state or context in which the function is called. It is not utilized within the function body.

**Code Description**: The INPUT_TYPES function constructs a dictionary that specifies the required input parameters for a sampling method. The dictionary contains a single key, "required", which maps to another dictionary detailing the specific input parameters needed. 

The input parameters defined are as follows:
- "solver_type": This parameter accepts a list of string values, specifically 'midpoint' and 'heun', indicating the types of solvers that can be used in the sampling process.
- "eta": This parameter is of type FLOAT and has a default value of 1.0. It is constrained to a minimum of 0.0 and a maximum of 100.0, with a step increment of 0.01. The "round" option is set to False, allowing for decimal values.
- "s_noise": Similar to "eta", this parameter is also of type FLOAT, with the same default, minimum, maximum, and step constraints.
- "noise_device": This parameter accepts a list of string values, specifically 'gpu' and 'cpu', indicating the devices that can be used for noise generation.

The function returns this structured dictionary, which is essential for ensuring that the correct types and constraints are enforced when the sampling process is executed.

**Note**: It is important to ensure that the input values provided to the sampling process adhere to the specified types and constraints to avoid runtime errors or unexpected behavior.

**Output Example**: An example of the return value from the INPUT_TYPES function would be:
{
    "required": {
        "solver_type": (['midpoint', 'heun'], ),
        "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
        "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
        "noise_device": (['gpu', 'cpu'], )
    }
}
***
### FunctionDef get_sampler(self, solver_type, eta, s_noise, noise_device)
**get_sampler**: The function of get_sampler is to create and return a sampler instance based on the specified solver type and noise parameters.

**parameters**: The parameters of this Function.
· solver_type: A string that specifies the type of solver to be used in the sampling process.
· eta: A float value representing a parameter that influences the sampling process.
· s_noise: A float value indicating the amount of noise to be applied during sampling.
· noise_device: A string that determines the device to be used for noise generation, either 'cpu' or 'gpu'.

**Code Description**: The get_sampler function is designed to instantiate a sampling object based on the provided parameters. It first checks the value of the noise_device parameter to determine whether to use a CPU or GPU for the sampling process. If the noise_device is set to 'cpu', the function assigns the string "dpmpp_2m_sde" to the sampler_name variable. Conversely, if the noise_device is set to any other value (implicitly assumed to be 'gpu'), it assigns "dpmpp_2m_sde_gpu" to sampler_name.

Following this, the function calls the ksampler function from the ldm_patched.modules.samplers module, passing the determined sampler_name along with a dictionary containing the eta, s_noise, and solver_type parameters as extra options. The ksampler function is responsible for creating an instance of the KSAMPLER class, which encapsulates the logic for the specified sampling method.

The get_sampler function ultimately returns a tuple containing the created sampler instance. This design allows for flexibility in selecting different sampling strategies based on the device and parameters provided, making it suitable for various sampling scenarios within the project.

The get_sampler function is typically called by other components within the project that require a configured sampler for generating samples, ensuring that the appropriate settings are applied based on the context of the sampling operation.

**Note**: When using the get_sampler function, ensure that the parameters provided are valid and correspond to the expected types. The noise_device should be either 'cpu' or 'gpu' to avoid unexpected behavior. Additionally, the eta and s_noise values should be chosen based on the specific requirements of the sampling process.

**Output Example**: The output of the get_sampler function could be a tuple containing an instance of the KSAMPLER class, such as `(KSAMPLER(sampler_function=<function dpmpp_2m_sde_function>, extra_options={'eta': 0.5, 's_noise': 0.1, 'solver_type': 'some_solver_type'}),)`, indicating the configured sampler ready for use in the sampling process.
***
## ClassDef SamplerDPMPP_SDE
**SamplerDPMPP_SDE**: The function of SamplerDPMPP_SDE is to create a sampling object based on specified parameters for noise and sampling configuration.

**attributes**: The attributes of this Class.
· eta: A floating-point value that controls the noise level in the sampling process, with a default of 1.0 and a range from 0.0 to 100.0.
· s_noise: A floating-point value representing the standard deviation of the noise, with a default of 1.0 and a range from 0.0 to 100.0.
· r: A floating-point value that influences the sampling rate, with a default of 0.5 and a range from 0.0 to 100.0.
· noise_device: A string that specifies the device to be used for noise generation, which can either be 'gpu' or 'cpu'.

**Code Description**: The SamplerDPMPP_SDE class is designed to facilitate the creation of a sampler object tailored for different noise configurations in a sampling process. It contains a class method, INPUT_TYPES, which defines the required input parameters for the sampler. These parameters include eta, s_noise, r, and noise_device, each with specific data types and constraints. The method returns a dictionary that outlines the expected input types, ensuring that users provide valid values when creating a sampler.

The RETURN_TYPES attribute indicates that the output of the class's functionality will be a tuple containing a single element of type "SAMPLER". The CATEGORY attribute categorizes this class under "sampling/custom_sampling/samplers", making it easier for users to locate it within a larger framework.

The core functionality is encapsulated in the get_sampler method, which takes the parameters eta, s_noise, r, and noise_device as inputs. Based on the value of noise_device, the method determines the appropriate sampler name, either "dpmpp_sde" for CPU or "dpmpp_sde_gpu" for GPU. It then calls the ksampler function from the ldm_patched.modules.samplers module, passing the sampler name and a dictionary of parameters. The method concludes by returning a tuple containing the created sampler object.

**Note**: Users should ensure that the values provided for eta, s_noise, and r fall within the specified ranges to avoid errors. Additionally, the choice of noise_device should be made based on the available hardware to optimize performance.

**Output Example**: A possible return value from the get_sampler method could look like this: 
(sampler_object,) 
where sampler_object is an instance of the sampler created based on the provided parameters.
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return a dictionary of required input types for a sampling process.

**parameters**: The parameters of this Function.
· parameter1: s - This parameter is an input that is typically used to represent the state or context in which the function operates. It is not utilized within the function body.

**Code Description**: The INPUT_TYPES function constructs and returns a dictionary that specifies the required input parameters for a sampling process. The dictionary is structured with a single key "required", which maps to another dictionary containing the following keys and their respective value types:

- "eta": This key corresponds to a floating-point number. It has a default value of 1.0, with a minimum value of 0.0 and a maximum value of 100.0. The step size for increments is set to 0.01, and rounding is disabled.
  
- "s_noise": Similar to "eta", this key also represents a floating-point number with the same default, minimum, maximum, and step specifications.
  
- "r": This key represents another floating-point number, with a default of 0.5, a minimum of 0.0, a maximum of 100.0, a step size of 0.01, and rounding disabled.
  
- "noise_device": This key is defined as a list containing two string options: 'gpu' and 'cpu'. This indicates that the user can select either of these devices for noise processing.

The structure of the returned dictionary ensures that users of the sampling function are aware of the necessary parameters and their constraints, facilitating proper input handling and validation.

**Note**: It is important to ensure that the values provided for "eta", "s_noise", and "r" fall within the specified ranges to avoid errors during the sampling process. Additionally, the "noise_device" must be selected from the provided options.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
        "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
        "r": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
        "noise_device": (['gpu', 'cpu'], )
    }
}
***
### FunctionDef get_sampler(self, eta, s_noise, r, noise_device)
**get_sampler**: The function of get_sampler is to create and return a sampler instance based on the specified parameters and the device type.

**parameters**: The parameters of this Function.
· eta: A float value representing the trade-off between exploration and exploitation during the sampling process.
· s_noise: A float value indicating the standard deviation of the noise to be added during sampling.
· r: An integer that specifies the number of steps or iterations for the sampling process.
· noise_device: A string that indicates the device type for noise processing, which can either be 'cpu' or 'gpu'.

**Code Description**: The get_sampler function is designed to generate a sampler instance tailored to the specified noise device and sampling parameters. It begins by checking the value of the noise_device parameter. If the noise_device is set to 'cpu', it assigns the string "dpmpp_sde" to the sampler_name variable, indicating that the CPU version of the DPM++ sampling method will be used. Conversely, if the noise_device is set to any other value (typically 'gpu'), it assigns "dpmpp_sde_gpu" to sampler_name, indicating the GPU version.

Following this, the function calls the ksampler function from the ldm_patched.modules.samplers module, passing the sampler_name along with a dictionary containing the parameters eta, s_noise, and r. The ksampler function acts as a factory for creating an instance of the KSAMPLER class, which is responsible for executing the sampling process based on the specified method and options.

The get_sampler function ultimately returns a tuple containing the created sampler instance. This design allows for flexibility in sampling strategies, enabling users to easily switch between CPU and GPU implementations based on their computational resources.

The get_sampler function is typically invoked by other components within the project that require a configured sampler for generating samples, ensuring that the appropriate sampling method is utilized based on the device and parameters provided.

**Note**: When using the get_sampler function, ensure that the eta, s_noise, and r parameters are set correctly to achieve the desired sampling behavior. Additionally, verify that the noise_device parameter is accurately specified to avoid any issues with sampler instantiation.

**Output Example**: The output of the get_sampler function could be a tuple containing an instance of the KSAMPLER class, such as `(KSAMPLER(sampler_function=<function dpmpp_sde_function>, extra_options={'eta': 0.5, 's_noise': 0.1, 'r': 10}),)`, indicating that the configured sampler is ready for use in the sampling process.
***
## ClassDef SamplerTCD
**SamplerTCD**: The function of SamplerTCD is to create a custom sampler with a specified parameter.

**attributes**: The attributes of this Class.
· INPUT_TYPES: A class method that defines the required input types for the sampler, specifically the 'eta' parameter.
· RETURN_TYPES: A tuple indicating the type of return value, which is "SAMPLER".
· CATEGORY: A string that categorizes the sampler within the sampling framework.
· FUNCTION: A string that specifies the function name to be called, which is "get_sampler".

**Code Description**: The SamplerTCD class is designed to facilitate the creation of a custom sampling mechanism. It includes a class method, INPUT_TYPES, which specifies the required input for the sampler. In this case, the input is a floating-point number 'eta' that has a default value of 0.3 and is constrained to a range between 0.0 and 1.0, with increments of 0.01. The RETURN_TYPES attribute indicates that the output of the class will be of type "SAMPLER". The CATEGORY attribute categorizes this sampler under "sampling/custom_sampling/samplers", which helps in organizing different sampling methods. The FUNCTION attribute defines the name of the method that will be executed to obtain the sampler, which is "get_sampler". 

The core functionality is encapsulated in the get_sampler method, which takes an optional parameter 'eta' (defaulting to 0.3). This method utilizes the ldm_patched.modules.samplers.ksampler function to create a sampler of type "tcd", passing the 'eta' parameter as part of its configuration. The method returns a tuple containing the created sampler.

**Note**: When using this class, ensure that the 'eta' parameter is within the specified range to avoid errors. The default value can be adjusted as needed, but it must adhere to the defined constraints.

**Output Example**: An example of the return value when calling the get_sampler method with eta set to 0.5 might look like this:
(sampler_instance,) where sampler_instance is an object representing the custom sampler configured with the specified 'eta' value.
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types for a specific configuration parameter.

**parameters**: The parameters of this Function.
· s: This parameter is not utilized within the function and serves as a placeholder for potential future use or for maintaining a consistent function signature.

**Code Description**: The INPUT_TYPES function returns a dictionary that specifies the required input types for a configuration setting. In this case, the dictionary contains a single key "required", which maps to another dictionary that defines the parameter "eta". The "eta" parameter is expected to be of type "FLOAT". Additionally, it includes a set of constraints: a default value of 0.3, a minimum value of 0.0, a maximum value of 1.0, and a step increment of 0.01. This structure allows for validation and ensures that the input for "eta" adheres to the specified type and constraints, facilitating proper configuration in the context where this function is utilized.

**Note**: It is important to ensure that any input provided for the "eta" parameter falls within the defined range and adheres to the specified type to avoid errors during execution. The function is designed to be extensible, allowing for additional parameters to be added in the future if needed.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "eta": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
    }
}
***
### FunctionDef get_sampler(self, eta)
**get_sampler**: The function of get_sampler is to create and return a configured sampler instance for the TCD sampling method.

**parameters**: The parameters of this Function.
· eta: A float value that specifies the parameter for the sampling process, with a default value of 0.3.

**Code Description**: The get_sampler function is designed to instantiate a sampler specifically for the TCD (Temporal Consistency Diffusion) method by utilizing the ksampler function from the ldm_patched.modules.samplers module. When called, it takes an optional parameter `eta`, which is passed as part of the configuration options to the ksampler function. 

The function calls `ksampler` with the sampler name "tcd" and a dictionary containing the `eta` parameter. The ksampler function then processes this request by determining the appropriate sampling function based on the provided sampler name. It returns an instance of the KSAMPLER class, which is tailored for the TCD sampling method with the specified options.

The get_sampler function is typically invoked by other components within the project that require a sampling strategy for TCD. By encapsulating the creation of the sampler, it simplifies the process of obtaining a configured sampler instance, ensuring that the necessary parameters are correctly set up for subsequent sampling operations.

**Note**: When using the get_sampler function, ensure that the `eta` parameter is set according to the requirements of the sampling process, as it influences the behavior of the TCD sampler.

**Output Example**: The output of the get_sampler function could be a tuple containing an instance of the KSAMPLER class, such as `(KSAMPLER(sampler_function=<function corresponding to TCD>, extra_options={'eta': 0.3}, inpaint_options={}),)`, indicating that the sampler is ready for use in the sampling process.
***
## ClassDef SamplerCustom
**SamplerCustom**: The function of SamplerCustom is to perform custom sampling on latent images with optional noise addition.

**attributes**: The attributes of this Class.
· INPUT_TYPES: Defines the required input types for the sampling function, including model, noise parameters, configuration, and conditioning inputs.  
· RETURN_TYPES: Specifies the types of outputs returned by the sample function, which are two latent outputs.  
· RETURN_NAMES: Names the outputs returned by the sample function as "output" and "denoised_output".  
· FUNCTION: Indicates the name of the function that will be executed, which is "sample".  
· CATEGORY: Categorizes the functionality of the class under "sampling/custom_sampling".  

**Code Description**: The SamplerCustom class is designed to facilitate custom sampling from a model using latent images. It includes a class method, INPUT_TYPES, which outlines the necessary input parameters for the sampling process. These inputs include a model, a boolean to determine if noise should be added, a seed for the noise generation, a configuration float, and conditioning inputs for both positive and negative samples. The sample method is the core function of this class, which takes in the specified parameters and processes them to generate samples.

Within the sample method, the latent image is extracted from the input, and based on the add_noise parameter, either a tensor of zeros or generated noise is prepared. If noise is to be added, the method utilizes a helper function to prepare the noise based on the latent image and the provided noise seed. The method also checks for an optional noise mask in the latent input.

The sampling process is executed using another utility function that handles the custom sampling logic, taking into account the model, noise, configuration, sampler type, and conditioning inputs. The results are stored in an output dictionary, which is then returned as two outputs: the sampled latent and a denoised version of the latent, if applicable. The denoised output is processed through the model if available.

**Note**: It is important to ensure that the input parameters adhere to the specified types and constraints, particularly for the noise seed and configuration values. The class is intended for use in scenarios where custom sampling techniques are required, and users should be familiar with the underlying model and its expected inputs.

**Output Example**: A possible appearance of the code's return value could be:
{
  "samples": tensor([[...], [...], ...]),  // Latent samples generated
  "batch_index": [0, 1, 2],  // Indices of the samples in the batch
  "noise_mask": tensor([[...], [...], ...])  // Optional noise mask if provided
}
### FunctionDef INPUT_TYPES(s)
**INPUT_TYPES**: The function of INPUT_TYPES is to define and return the required input types for a specific sampling model configuration.

**parameters**: The parameters of this Function.
· s: This parameter is typically used to represent the state or context in which the function is called, although it is not utilized within the function body.

**Code Description**: The INPUT_TYPES function constructs a dictionary that specifies the required input types for a sampling model. The dictionary contains a single key "required", which maps to another dictionary detailing various input parameters necessary for the model's operation. Each input parameter is associated with a tuple that defines its type and additional constraints.

The parameters included in the returned dictionary are as follows:
- "model": This parameter is of type "MODEL" and is required for the function to operate.
- "add_noise": This is a boolean parameter with a default value of True, indicating whether noise should be added to the model's output.
- "noise_seed": This integer parameter has a default value of 0 and is constrained to a range from 0 to 0xffffffffffffffff, allowing for a wide range of seed values for noise generation.
- "cfg": This float parameter has a default value of 8.0 and is constrained to a minimum of 0.0 and a maximum of 100.0, with a step of 0.1 and rounding to two decimal places. It likely represents a configuration value for the model.
- "positive": This parameter is of type "CONDITIONING" and is required for the model's operation.
- "negative": This parameter is also of type "CONDITIONING" and is required for the model's operation.
- "sampler": This parameter is of type "SAMPLER" and is required for the model's operation.
- "sigmas": This parameter is of type "SIGMAS" and is required for the model's operation.
- "latent_image": This parameter is of type "LATENT" and is required for the model's operation.

The structure of the returned dictionary is designed to ensure that all necessary inputs are clearly defined, along with their types and any relevant constraints.

**Note**: It is important to ensure that all required parameters are provided when utilizing this function, as missing parameters may lead to errors during model execution.

**Output Example**: A possible appearance of the code's return value could be:
{
    "required": {
        "model": ("MODEL",),
        "add_noise": ("BOOLEAN", {"default": True}),
        "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "sampler": ("SAMPLER",),
        "sigmas": ("SIGMAS",),
        "latent_image": ("LATENT",)
    }
}
***
### FunctionDef sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image)
**sample**: The function of sample is to generate samples from a latent image using a specified model, noise, and various conditioning parameters.

**parameters**: The parameters of this Function.
· model: An object representing the machine learning model used for generating samples.
· add_noise: A boolean flag indicating whether to add noise to the latent image.
· noise_seed: An integer seed for initializing the random number generator for noise generation.
· cfg: A configuration parameter that influences the sampling behavior.
· positive: A list of conditioning tuples representing positive conditions for the model.
· negative: A list of conditioning tuples representing negative conditions for the model.
· sampler: An object responsible for generating samples from the model.
· sigmas: A tensor representing the sigma values used in the sampling process.
· latent_image: A tensor representing the latent image to be processed.

**Code Description**: The sample function is designed to facilitate the sampling process by generating samples based on a latent image and various input parameters. It begins by extracting the latent representation from the provided latent_image tensor. If the add_noise flag is set to False, the function initializes a noise tensor filled with zeros, matching the size and data type of the latent image. If add_noise is True, it calls the prepare_noise function from the ldm_patched.modules.sample module to generate random noise based on the latent image and the specified noise_seed. The prepare_noise function ensures that the generated noise is consistent across different runs by using the provided seed.

The function then checks for the presence of a noise mask in the latent dictionary. If a noise mask exists, it is extracted for later use during sampling. The function prepares a callback using the prepare_callback function from the ldm_patched.utils.latent_visualization module. This callback is intended to provide real-time feedback and visualization of the latent representations during the sampling process.

Next, the sample function invokes the sample_custom function from the ldm_patched.modules.sample module. This function is responsible for generating the actual samples from the model using the prepared noise, configuration, sampler, and other parameters. The samples generated are then stored in the output dictionary along with the original latent representation.

If the x0_output dictionary contains a key "x0", indicating that the initial latent representation was processed, the function further processes this representation using the model's process_latent_out method to obtain a denoised output. If no such key exists, the denoised output is simply set to the original output.

The sample function ultimately returns a tuple containing the original latent representation with the generated samples and the denoised output, providing a comprehensive result of the sampling process.

**Note**: When using the sample function, ensure that the model, noise_seed, and other parameters are correctly specified to achieve the desired sampling results. The add_noise parameter controls whether noise is incorporated into the sampling process, which can significantly affect the diversity of the generated samples.

**Output Example**: A possible return value from the sample function could be a tuple structured as follows:
```python
({
    "samples": tensor([[...], [...], ...]),  # Generated samples
    "batch_index": [...],                     # Indices of the batches
    "noise_mask": tensor([[...], [...], ...]) # Optional noise mask if provided
}, {
    "samples": tensor([[...], [...], ...])   # Denoised output samples
})
```
***
