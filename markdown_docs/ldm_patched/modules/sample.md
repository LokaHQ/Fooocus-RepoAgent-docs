## FunctionDef prepare_noise(latent_image, seed, noise_inds)
**prepare_noise**: The function of prepare_noise is to create random noise based on a given latent image and a specified seed.

**parameters**: The parameters of this Function.
· latent_image: A tensor representing the latent image from which noise will be generated. It defines the shape and data type of the noise to be created.
· seed: An integer value used to initialize the random number generator, ensuring reproducibility of the generated noise.
· noise_inds: An optional parameter that, if provided, specifies indices for generating unique noise tensors. If set to None, noise will be generated without any specific indexing.

**Code Description**: The prepare_noise function generates random noise tensors that can be used in various sampling processes. It utilizes the PyTorch library to create noise, ensuring that the generated noise matches the shape and data type of the provided latent_image tensor. 

When the function is called, it first sets the random seed using `torch.manual_seed(seed)`, which allows for consistent noise generation across different runs. If noise_inds is not provided, the function generates a tensor of random values with the same size as latent_image. This tensor is created using `torch.randn`, which produces values from a standard normal distribution.

If noise_inds is provided, the function first identifies unique indices from the noise_inds array. It then generates noise for each unique index, ensuring that the same noise tensor is reused for repeated indices. This is achieved by creating a list of noise tensors and then indexing into this list using the inverse of the unique indices to construct the final noise tensor. The resulting noise tensor is concatenated along the first dimension and returned.

The prepare_noise function is called by other components in the project, such as common_ksampler and SamplerCustom's sample method. In these contexts, it is used to generate noise that is either added to the latent image or set to zero based on the disable_noise flag. This integration highlights the function's role in controlling the noise component of sampling processes, which is crucial for generating diverse outputs in models that rely on latent representations.

**Note**: It is important to ensure that the latent_image tensor is properly initialized and that the seed is set appropriately to achieve the desired reproducibility in noise generation.

**Output Example**: A possible return value of the function could be a tensor of shape (batch_size, channels, height, width) filled with random values, such as:
```
tensor([[[-0.1234, 0.5678, ...],
         [0.9101, -0.2345, ...],
         ...],
        [[0.6789, -0.1234, ...],
         [-0.5678, 0.9101, ...],
         ...],
        ...])
```
## FunctionDef prepare_mask(noise_mask, shape, device)
**prepare_mask**: The function of prepare_mask is to ensure that the noise mask is of proper dimensions for further processing.

**parameters**: The parameters of this Function.
· parameter1: noise_mask - A PyTorch tensor representing the noise mask that needs to be resized and adjusted.
· parameter2: shape - A tuple indicating the desired output shape, typically containing batch size and spatial dimensions.
· parameter3: device - A string or object specifying the device (e.g., CPU or GPU) where the tensor should be allocated.

**Code Description**: The prepare_mask function takes a noise mask tensor and adjusts its dimensions to match the specified shape. Initially, the noise mask is reshaped to ensure it has the appropriate number of channels for interpolation. The function uses PyTorch's `torch.nn.functional.interpolate` to resize the noise mask to the target spatial dimensions defined in the shape parameter, utilizing bilinear interpolation for smooth resizing. 

After resizing, the function duplicates the noise mask across the channel dimension to match the required number of channels specified in the shape parameter. This is achieved using `torch.cat`, which concatenates the resized noise mask multiple times along the channel dimension. 

Next, the function calls `repeat_to_batch_size`, which is defined in the ldm_patched.modules.utils module. This function adjusts the size of the noise mask tensor to match the specified batch size, ensuring that the tensor has the correct number of elements for processing. Finally, the adjusted noise mask is moved to the specified device (CPU or GPU) using the `.to(device)` method, making it ready for subsequent operations.

The prepare_mask function is called within the prepare_sampling function, where it is used to process the noise mask before it is utilized in the model's inference pipeline. This ensures that the noise mask is correctly sized and formatted, which is crucial for the model to function properly without dimension mismatches.

**Note**: It is important to ensure that the input noise_mask is a valid PyTorch tensor and that the shape parameter accurately reflects the desired output dimensions to avoid runtime errors.

**Output Example**: For an input noise_mask tensor of shape (1, 1, 64, 64) and a shape parameter of (4, 3, 128, 128), the output would be a tensor of shape (4, 3, 128, 128) where the original noise mask is resized and repeated accordingly.
## FunctionDef get_models_from_cond(cond, model_type)
**get_models_from_cond**: The function of get_models_from_cond is to extract specific model types from a given list of conditions.

**parameters**: The parameters of this Function.
· parameter1: cond - A list of conditions, where each condition is expected to be a dictionary containing model types.
· parameter2: model_type - A string that specifies the type of model to be extracted from the conditions.

**Code Description**: The get_models_from_cond function iterates through a list of conditions (cond) and checks each condition to see if it contains the specified model type (model_type). If the model type is found within a condition, the corresponding model is added to a list called models. This function ultimately returns a list of models that match the specified type.

This function is utilized in several other functions within the project, including get_additional_models and sample_custom. In get_additional_models, it is called twice to gather control models from both positive and negative conditioning inputs. The results are then combined to form a set of control networks, which are further processed to retrieve additional model details and inference memory requirements. In the sample and sample_custom functions, get_models_from_cond is again called to clean up additional models based on the positive and negative conditioning inputs before returning the final samples. This demonstrates that get_models_from_cond plays a crucial role in filtering and managing models based on specific conditions, ensuring that only relevant models are processed in subsequent operations.

**Note**: It is important to ensure that the input list (cond) contains dictionaries with the expected structure, as the function relies on the presence of the model_type key within these dictionaries to function correctly.

**Output Example**: An example return value of get_models_from_cond could be a list like ['model1', 'model2'], where 'model1' and 'model2' are the models extracted from the conditions that matched the specified model_type.
## FunctionDef convert_cond(cond)
**convert_cond**: The function of convert_cond is to transform a list of conditioning tuples into a standardized format suitable for further processing in machine learning models.

**parameters**: The parameters of this Function.
· cond: A list of tuples, where each tuple contains a conditioning identifier and a dictionary of conditioning data.

**Code Description**: The convert_cond function iterates over a list of conditioning tuples, where each tuple consists of a conditioning identifier and a dictionary containing conditioning data. For each tuple, it creates a copy of the conditioning data dictionary and checks if the first element of the tuple (the conditioning identifier) is not None. If it is not None, the function initializes a CONDCrossAttn object with this identifier and adds it to the dictionary under the key "c_crossattn". It also assigns the conditioning identifier to the key "cross_attn" in the dictionary. The modified dictionary, which now includes the updated model conditions, is appended to an output list.

This function is called by other components in the project, such as the prepare_sampling function and the patch method in the PerpNeg class. In prepare_sampling, convert_cond is used to process both positive and negative conditioning inputs, ensuring they are in the correct format before further operations. In the patch method of the PerpNeg class, convert_cond is utilized to handle empty conditioning data, allowing for the subsequent calculation of noise predictions based on the processed conditioning data.

The output of convert_cond is a list of dictionaries, each representing a conditioning configuration that can be utilized in various model operations, particularly those involving cross-attention mechanisms.

**Note**: It is important to ensure that the input list of conditioning tuples is structured correctly, as the function expects each tuple to contain a valid identifier and a dictionary. Improperly formatted input may lead to unexpected behavior or errors during processing.

**Output Example**: An example of the output from the convert_cond function might look like this:
```python
[
    {"model_conds": {"c_crossattn": <CONDCrossAttn object>, ...}, "cross_attn": <identifier>, ...},
    ...
]
```
## FunctionDef get_additional_models(positive, negative, dtype)
**get_additional_models**: The function of get_additional_models is to load additional models based on positive and negative conditioning inputs.

**parameters**: The parameters of this Function.
· parameter1: positive - A list of conditions that represent positive model types to be extracted.
· parameter2: negative - A list of conditions that represent negative model types to be extracted.
· parameter3: dtype - A data type that specifies the type of models being processed, which is used to calculate inference memory requirements.

**Code Description**: The get_additional_models function is designed to gather and return additional models based on specified positive and negative conditions. It begins by calling the get_models_from_cond function twice to extract control models from both the positive and negative conditions. This results in a set of unique control networks, which are then processed to compile a list of control models and calculate their total inference memory requirements.

For each model in the control networks, the function retrieves the models associated with that network and accumulates the inference memory requirements based on the provided dtype. Subsequently, it calls get_models_from_cond again to extract additional models of type "gligen" from both the positive and negative conditions. The gligen models are filtered to retain only the relevant model instances.

Finally, the function combines the lists of control models and gligen models, returning both the complete list of models and the total inference memory required for their operation. 

This function is utilized by other functions within the project, such as prepare_sampling and refiner_switch. In prepare_sampling, get_additional_models is called to gather models that will be loaded onto the GPU alongside the main model, ensuring that all necessary models are available for the sampling process. In refiner_switch, it is used to load models specific to a refiner, allowing for dynamic model management based on the current conditions.

**Note**: It is essential to ensure that the input lists (positive and negative) are structured correctly, as the function relies on the presence of specific model types within these conditions to function effectively.

**Output Example**: A possible return value of get_additional_models could be a tuple like (['control_model1', 'control_model2', 'gligen_model1'], 256), where the first element is a list of models extracted from the conditions and the second element is the total inference memory required for those models.
## FunctionDef cleanup_additional_models(models)
**cleanup_additional_models**: The function of cleanup_additional_models is to clean up additional models that were loaded during the sampling process.

**parameters**: The parameters of this Function.
· models: A collection of model instances that may have a cleanup method.

**Code Description**: The cleanup_additional_models function iterates through a list of model instances provided as the 'models' parameter. For each model in this collection, it checks if the model has a 'cleanup' method. If the method exists, it calls this method on the model. This function is essential for managing resources and ensuring that any additional models that were loaded during the sampling process are properly cleaned up, thus preventing memory leaks or other resource-related issues.

This function is called within the sample and sample_custom functions, which are responsible for generating samples based on the provided model and input parameters. After the sampling process is completed, cleanup_additional_models is invoked with the 'models' variable, which contains the models prepared for sampling. This ensures that any additional models that were utilized during the sampling are appropriately cleaned up.

Furthermore, cleanup_additional_models is also called with a set of models derived from the positive and negative conditions. This highlights its role in maintaining the integrity of the model management system by ensuring that all models, regardless of their source, are cleaned up after use.

**Note**: It is important to ensure that the models passed to this function are instances that have a 'cleanup' method defined. Failure to do so may result in an AttributeError if the method is called on an instance that does not support it.
## FunctionDef prepare_sampling(model, noise_shape, positive, negative, noise_mask)
**prepare_sampling**: The function of prepare_sampling is to prepare the necessary components for sampling by processing model conditions and loading additional models onto the GPU.

**parameters**: The parameters of this Function.
· parameter1: model - The primary machine learning model that will be used for sampling.
· parameter2: noise_shape - A tuple representing the shape of the noise tensor, typically including batch size and spatial dimensions.
· parameter3: positive - A list of conditioning tuples that represent positive conditions for the model.
· parameter4: negative - A list of conditioning tuples that represent negative conditions for the model.
· parameter5: noise_mask - An optional PyTorch tensor representing the noise mask that may be used during the sampling process.

**Code Description**: The prepare_sampling function begins by determining the device on which the model is loaded, which is crucial for ensuring that all tensors are processed on the correct hardware (CPU or GPU). It then converts the positive and negative conditioning inputs into a standardized format using the convert_cond function. This function processes each conditioning tuple, ensuring that they are structured correctly for further operations.

If a noise mask is provided, the function prepares it for use by calling the prepare_mask function. This function ensures that the noise mask tensor has the appropriate dimensions and is moved to the correct device, which is essential for compatibility with the model's input requirements.

Next, the prepare_sampling function retrieves additional models that may be needed during the sampling process by calling the get_additional_models function. This function takes the positive and negative conditions and the model's data type as inputs, returning a list of additional models and the total inference memory required for these models.

The function then loads the main model and any additional models onto the GPU using the load_models_gpu function. This step is critical for ensuring that all necessary models are available for inference without running into memory issues.

Finally, prepare_sampling returns the real model along with the processed positive and negative conditions, the prepared noise mask, and the list of additional models. This output is essential for the subsequent sampling operations, which rely on these components to generate samples.

The prepare_sampling function is called by other functions such as sample and sample_custom. In these functions, prepare_sampling is utilized to ensure that all necessary components are ready before the actual sampling process begins. This highlights the function's role in setting up the environment for model inference.

**Note**: It is important to ensure that the input parameters, especially the conditioning tuples and noise mask, are structured correctly to avoid runtime errors during processing.

**Output Example**: A possible return value from the prepare_sampling function could be a tuple like (real_model, positive_conditions, negative_conditions, prepared_noise_mask, additional_models), where each element is appropriately processed and ready for use in the sampling pipeline.
## FunctionDef sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, disable_noise, start_step, last_step, force_full_denoise, noise_mask, sigmas, callback, disable_pbar, seed)
**sample**: The function of sample is to generate samples from a generative model using specified noise and conditioning inputs.

**parameters**: The parameters of this Function.
· parameter1: model - The primary machine learning model used for generating samples.
· parameter2: noise - A tensor representing the noise input for the sampling process.
· parameter3: steps - An integer indicating the number of steps to be taken during the sampling.
· parameter4: cfg - A configuration parameter that influences the sampling process.
· parameter5: sampler_name - A string specifying the name of the sampler to be used.
· parameter6: scheduler - A string indicating the scheduling strategy for the sampling process.
· parameter7: positive - A list of conditioning tuples representing positive conditions for the model.
· parameter8: negative - A list of conditioning tuples representing negative conditions for the model.
· parameter9: latent_image - A tensor representing the latent image input for the sampling.
· parameter10: denoise - A float value controlling the denoising process (default is 1.0).
· parameter11: disable_noise - A boolean flag to disable noise during sampling (default is False).
· parameter12: start_step - An optional integer indicating the starting step for sampling (default is None).
· parameter13: last_step - An optional integer indicating the last step for sampling (default is None).
· parameter14: force_full_denoise - A boolean flag to force full denoising during sampling (default is False).
· parameter15: noise_mask - An optional tensor representing the noise mask to be used during sampling (default is None).
· parameter16: sigmas - An optional tensor representing the noise levels at each step (default is None).
· parameter17: callback - An optional function to be called during the sampling process (default is None).
· parameter18: disable_pbar - A boolean flag to disable the progress bar during sampling (default is False).
· parameter19: seed - An optional integer for random seed initialization (default is None).

**Code Description**: The sample function is designed to facilitate the generation of samples from a generative model by preparing the necessary inputs and invoking the appropriate sampling mechanism. Initially, it calls the prepare_sampling function to set up the model and conditioning inputs, which includes loading the model onto the appropriate device and processing the positive and negative conditions. The noise tensor and latent image are also moved to the device specified by the model.

Next, an instance of the KSampler class is created, which is responsible for managing the sampling process. The KSampler is initialized with the real model, the number of steps, the device, and various parameters such as the sampler name, scheduler, and denoising options. The sample method of the KSampler instance is then called, passing in the noise, conditioning inputs, and other parameters required for sampling.

After the sampling process is completed, the generated samples are moved to the intermediate device using the intermediate_device function, ensuring that they are compatible with further processing. The cleanup_additional_models function is invoked to manage and release any additional models that were loaded during the sampling process, preventing memory leaks.

The sample function is called by other functions within the project, such as common_ksampler and ksampler, which serve as higher-level interfaces for generating samples. These functions prepare the necessary inputs and invoke the sample function, demonstrating its role as a core component in the sampling workflow.

**Note**: It is crucial to ensure that all input parameters, especially the conditioning inputs and noise tensor, are structured correctly to avoid runtime errors during the sampling process. Additionally, the use of the callback function and progress bar options should be configured based on user preferences.

**Output Example**: A possible return value of the sample function could be a tensor representing the generated samples, which may appear as a multi-dimensional array of floating-point values corresponding to the generated images or data points.
## FunctionDef sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask, callback, disable_pbar, seed)
**sample_custom**: The function of sample_custom is to generate samples from a model using specified noise and conditioning parameters.

**parameters**: The parameters of this Function.
· model: The primary machine learning model used for sampling.
· noise: A tensor representing the noise data to be utilized in the sampling process.
· cfg: A configuration parameter that influences the sampling behavior.
· sampler: An object responsible for generating samples from the model.
· sigmas: A tensor representing the sigma values used in the sampling process.
· positive: A list of conditioning tuples that represent positive conditions for the model.
· negative: A list of conditioning tuples that represent negative conditions for the model.
· latent_image: A tensor representing the latent image to be processed.
· noise_mask: An optional tensor representing the noise mask (default is None).
· callback: A callable function for progress tracking during sampling (default is None).
· disable_pbar: A boolean flag to disable the progress bar during sampling (default is False).
· seed: An optional integer value used for random seed initialization (default is None).

**Code Description**: The sample_custom function is designed to facilitate the sampling process by preparing the necessary components and invoking the sampling mechanism. It begins by calling the prepare_sampling function, which processes the model and conditions, ensuring that all required models are loaded onto the appropriate device. This function returns the real model, copies of the positive and negative conditions, the prepared noise mask, and any additional models needed for sampling.

Once the components are prepared, the noise tensor and latent image are transferred to the device specified by the model. The sigmas tensor is also moved to the correct device to ensure compatibility during the sampling process.

The core sampling operation is performed by invoking the sample function from the ldm_patched.modules.samplers module. This function generates samples based on the provided model, noise, conditions, and other parameters. The resulting samples are then transferred to an intermediate device for further processing.

After the sampling is completed, the cleanup_additional_models function is called to manage resources by cleaning up any additional models that were loaded during the sampling process. This is crucial for preventing memory leaks and ensuring efficient resource management.

The sample_custom function is called by the sample method in the SamplerCustom class, which is responsible for orchestrating the overall sampling workflow. The SamplerCustom class prepares the necessary inputs, including noise and latent images, and then invokes sample_custom to generate the final output samples.

**Note**: It is important to ensure that all input parameters, especially the conditioning tuples and noise mask, are structured correctly to avoid runtime errors during processing. Additionally, the models and tensors should be compatible with the specified device to ensure smooth execution.

**Output Example**: A possible return value from the sample_custom function could be a tensor representing the generated samples, structured as follows:
```python
torch.Size([batch_size, channels, height, width])
```
This tensor contains the sampled outputs generated by the model based on the provided conditions and noise inputs.
