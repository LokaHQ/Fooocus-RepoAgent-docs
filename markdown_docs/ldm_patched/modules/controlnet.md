## FunctionDef broadcast_image_to(tensor, target_batch_size, batched_number)
**broadcast_image_to**: The function of broadcast_image_to is to adjust the size of a tensor to match a specified target batch size by either trimming or repeating its elements.

**parameters**: The parameters of this Function.
· tensor: A PyTorch tensor that contains the data to be adjusted.
· target_batch_size: An integer representing the desired size of the output tensor.
· batched_number: An integer indicating how many times the tensor should be batched.

**Code Description**: The broadcast_image_to function begins by determining the current batch size of the input tensor using its shape. If the current batch size is 1, it simply returns the tensor as is, since no adjustment is necessary. 

Next, the function calculates how many elements should be included per batch by dividing the target batch size by the batched number. It then slices the tensor to keep only the first 'per_batch' elements. If the number of elements per batch exceeds the size of the tensor, the function concatenates copies of the tensor to meet the required size, ensuring that the total number of elements matches 'per_batch'. 

After adjusting the tensor, the function checks if the current batch size matches the target batch size. If it does, the adjusted tensor is returned. If not, the tensor is repeated 'batched_number' times to create a new tensor that meets the target batch size.

This function is called within the get_control methods of two classes: ControlNet and T2IAdapter. In both cases, it is used to ensure that the conditional hint tensor (cond_hint) matches the batch size of the noisy input tensor (x_noisy). This is crucial for maintaining consistency in the dimensions of the tensors being processed, as mismatched sizes could lead to errors during model inference. By using broadcast_image_to, both classes can effectively manage the input tensor sizes, allowing for seamless integration into the control mechanisms of the models.

**Note**: It is important to ensure that the input tensor is appropriately shaped before calling this function, as unexpected shapes may lead to unintended behavior. 

**Output Example**: If the input tensor has a shape of (2, 3, 64, 64) and the target_batch_size is 6 with a batched_number of 2, the output tensor would have a shape of (6, 3, 64, 64), created by repeating the original tensor elements.
## ClassDef ControlBase
**ControlBase**: The function of ControlBase is to serve as a foundational class for managing control hints and their processing in a control network.

**attributes**: The attributes of this Class.
· cond_hint_original: Stores the original conditional hint provided for processing.
· cond_hint: Holds the processed conditional hint.
· strength: A float value representing the strength of the control hint, defaulting to 1.0.
· timestep_percent_range: A tuple indicating the range of timesteps as percentages, defaulting to (0.0, 1.0).
· global_average_pooling: A boolean flag indicating whether to apply global average pooling, defaulting to False.
· timestep_range: A tuple representing the actual range of timesteps after processing.
· device: Specifies the device (CPU or GPU) on which the computations will be performed.
· previous_controlnet: A reference to a previous ControlBase instance, allowing for chaining of control networks.

**Code Description**: The ControlBase class is designed to manage the control hints used in various control networks. It initializes with an optional device parameter, which determines where the computations will be executed. If no device is provided, it defaults to the device obtained from the model management module. The class maintains the original and processed conditional hints, the strength of the hint, and the range of timesteps for processing.

The method `set_cond_hint` allows users to set the conditional hint along with its strength and timestep percentage range. The `pre_run` method prepares the timestep range based on a provided function that maps percentages to actual timesteps. It also allows for the chaining of control networks by invoking the `pre_run` method of the previous control network if it exists.

The `set_previous_controlnet` method establishes a link to a previous ControlBase instance, enabling a sequence of control networks to be processed. The `cleanup` method is responsible for releasing resources and resetting attributes, ensuring that the instance is ready for reuse or disposal.

The `get_models` method retrieves models from the current and previous control networks, while the `copy_to` method facilitates the copying of relevant attributes to another ControlBase instance. The `inference_memory_requirements` method calculates the memory requirements for inference, considering the previous control network if applicable.

The `control_merge` method is a critical function that combines inputs, outputs, and previous control states into a structured output. It applies the strength factor to the control inputs and outputs, manages data types, and handles the merging of previous control states.

ControlBase is utilized by several derived classes, such as ControlNet, ControlLora, and T2IAdapter. Each of these classes extends the functionality of ControlBase to implement specific control mechanisms tailored to their respective models. For instance, ControlNet uses ControlBase to manage control hints while interfacing with a control model, whereas T2IAdapter focuses on adapting text-to-image models with control hints.

**Note**: It is essential to ensure that the conditional hints are appropriately sized and compatible with the input data dimensions. The strength parameter should be adjusted based on the desired influence of the control hints on the output.

**Output Example**: An example output from the `control_merge` method could look like this:
{
  'input': [tensor1, tensor2],
  'middle': [tensor3],
  'output': [tensor4, tensor5]
}
Where `tensor1`, `tensor2`, etc., are processed tensors representing the control inputs and outputs.
### FunctionDef __init__(self, device)
**__init__**: The function of __init__ is to initialize an instance of the ControlBase class with specified parameters and default values.

**parameters**: The parameters of this Function.
· device: Optional; specifies the device (CPU, GPU, or XPU) on which the operations will be performed. If not provided, the function will determine the device automatically.

**Code Description**: The __init__ function serves as the constructor for the ControlBase class. It initializes several instance variables that are essential for the operation of the class. 

1. The function begins by setting default values for several attributes:
   - `self.cond_hint_original` and `self.cond_hint` are initialized to None, indicating that no conditional hints have been provided at the time of instantiation.
   - `self.strength` is set to 1.0, which likely represents a default strength parameter for some control mechanism within the class.
   - `self.timestep_percent_range` is initialized to a tuple (0.0, 1.0), defining the range for timestep percentages.
   - `self.global_average_pooling` is set to False, indicating that global average pooling is not enabled by default.
   - `self.timestep_range` is initialized to None, suggesting that no specific timestep range has been defined at this point.

2. The function then checks if the `device` parameter has been provided. If it is None, the function calls `ldm_patched.modules.model_management.get_torch_device()` to determine the appropriate device for PyTorch operations based on the current system configuration. This function is crucial as it ensures that the class operates on the optimal hardware available, whether it be a CPU, GPU, or XPU.

3. The determined or provided device is then assigned to `self.device`, which will be used throughout the class for executing operations.

4. Finally, `self.previous_controlnet` is initialized to None, indicating that there is no previous control network associated with this instance at the time of creation.

The __init__ function is fundamental for setting up the ControlBase class, ensuring that all necessary parameters are initialized and that the class is ready for subsequent operations. The relationship with the `get_torch_device` function is particularly important, as it allows the ControlBase class to adapt to the hardware environment dynamically, enhancing its flexibility and performance.

**Note**: It is important to ensure that the device parameter is correctly specified or that the system is properly configured to allow `get_torch_device` to function as intended. This will prevent issues related to device compatibility during the execution of operations within the ControlBase class.
***
### FunctionDef set_cond_hint(self, cond_hint, strength, timestep_percent_range)
**set_cond_hint**: The function of set_cond_hint is to set the conditional hint, strength, and timestep percent range for a control net.

**parameters**: The parameters of this Function.
· cond_hint: This parameter represents the conditional hint that will be applied to the control net. It is expected to be in a format compatible with the control net's requirements.
· strength: This parameter defines the strength of the conditional hint, with a default value of 1.0. It determines how strongly the hint influences the control net's behavior.
· timestep_percent_range: This parameter is a tuple that specifies the range of percent values for the timestep, with a default value of (0.0, 1.0). It indicates the portion of the timestep over which the conditional hint will be applied.

**Code Description**: The set_cond_hint function is responsible for initializing and storing the conditional hint, its strength, and the timestep percent range within the instance of the class it belongs to. When invoked, it assigns the provided cond_hint to the instance variable cond_hint_original, sets the strength of the hint, and defines the range of timesteps during which this hint will be effective. The function then returns the instance itself, allowing for method chaining.

This function is called within the apply_controlnet method located in the ControlNetApplyAdvanced class. In this context, when applying control nets to images, the apply_controlnet method first checks if the strength is zero; if so, it returns the positive and negative inputs unchanged. If strength is greater than zero, it processes the control hint by moving the image dimensions and prepares to apply the control net. For each conditioning (positive and negative), it checks if a control net has already been created for the current conditioning. If not, it creates a new control net by copying the provided control_net and invoking set_cond_hint with the control_hint, strength, and the specified timestep percent range. This establishes the necessary parameters for the control net to function correctly in the image processing pipeline.

**Note**: It is important to ensure that the cond_hint provided is compatible with the control net's requirements to avoid runtime errors. Additionally, the strength parameter should be set appropriately to achieve the desired influence of the conditional hint on the control net's output.

**Output Example**: A possible appearance of the code's return value could be an instance of the control net class with the following attributes set:
- cond_hint_original: <value of cond_hint>
- strength: <value of strength>
- timestep_percent_range: (start_percent, end_percent)
***
### FunctionDef pre_run(self, model, percent_to_timestep_function)
**pre_run**: The function of pre_run is to prepare the control net for execution by setting the timestep range and invoking the pre_run method of any previous control net if it exists.

**parameters**: The parameters of this Function.
· model: This parameter represents the model that the control net will operate on. It is expected to be an object that contains the necessary data and methods for the control net's functionality.
· percent_to_timestep_function: This is a function that converts a percentage value into a corresponding timestep value. It is used to map the percentage range defined in the control net to actual timesteps.

**Code Description**: The pre_run function begins by calculating the timestep range based on the provided percent_to_timestep_function. It takes the first and second elements of the timestep_percent_range attribute, applies the percent_to_timestep_function to each, and assigns the resulting values to the timestep_range attribute. This effectively sets the operational range for the control net in terms of timesteps, which is crucial for its execution. 

Following this, the function checks if there is a previous control net (indicated by the previous_controlnet attribute). If such a control net exists, the pre_run method is called recursively on it, passing along the model and the percent_to_timestep_function. This ensures that any preceding control nets are also prepared for execution, maintaining a chain of preparation across potentially multiple control nets.

**Note**: It is important to ensure that the percent_to_timestep_function is correctly defined and capable of handling the input values from timestep_percent_range. Additionally, the previous_controlnet should be properly initialized before invoking this method to avoid potential errors during execution.
***
### FunctionDef set_previous_controlnet(self, controlnet)
**set_previous_controlnet**: The function of set_previous_controlnet is to set the previous control network for the current instance.

**parameters**: The parameters of this Function.
· controlnet: This parameter represents the control network that is to be set as the previous control network for the current instance.

**Code Description**: The set_previous_controlnet function is a method defined within a class that manages control networks. Its primary purpose is to assign a given control network to the instance variable previous_controlnet. This allows the instance to keep track of the control network that was used prior to the current one, facilitating operations that may require reference to the previous state or configuration.

The function takes a single argument, controlnet, which is expected to be an object representing a control network. When invoked, the function assigns this controlnet to the instance variable self.previous_controlnet and then returns the instance itself (self). This pattern is commonly used in object-oriented programming to allow for method chaining, enabling subsequent method calls on the same instance.

In the context of its caller, the apply_controlnet function from the ControlNetApplyAdvanced class utilizes set_previous_controlnet to maintain a reference to the control network that was previously applied. During the application of control networks to images, if a previous control network exists, it is passed to the set_previous_controlnet method. This ensures that the current control network can be aware of its predecessor, which may be crucial for operations that depend on the history of applied control networks.

**Note**: It is important to ensure that the controlnet parameter passed to this function is valid and properly initialized to avoid potential errors when accessing the previous control network.

**Output Example**: An example of the return value after invoking set_previous_controlnet with a control network object could look like this:
```
<ControlNetworkInstance(previous_controlnet=<SomeControlNetworkObject>)>
```
***
### FunctionDef cleanup(self)
**cleanup**: The function of cleanup is to release resources and reset certain attributes of the ControlNet object.

**parameters**: The parameters of this Function.
· There are no parameters for this function.

**Code Description**: The cleanup function is responsible for managing the cleanup process of the ControlNet object. It performs the following actions:
1. It first checks if the `previous_controlnet` attribute is not None. If it exists, it calls the `cleanup` method on the `previous_controlnet` object. This ensures that any resources or references held by the previous ControlNet instance are properly released, preventing memory leaks or dangling references.
2. Next, it checks if the `cond_hint` attribute is not None. If it exists, it deletes the `cond_hint` attribute and sets it to None. This effectively removes the reference to `cond_hint`, allowing for garbage collection of the object it pointed to, if there are no other references to it.
3. Finally, it sets the `timestep_range` attribute to None, indicating that the current instance no longer has a valid timestep range associated with it.

This method is crucial for maintaining the integrity of the ControlNet object and ensuring that resources are managed efficiently.

**Note**: It is important to call the cleanup function when the ControlNet object is no longer needed to ensure that all associated resources are released properly. Failure to do so may lead to memory leaks or unintended behavior in the application.
***
### FunctionDef get_models(self)
**get_models**: The function of get_models is to retrieve a list of models from the current and previous controlnet instances.

**parameters**: The parameters of this Function.
· There are no parameters for this function.

**Code Description**: The get_models function is designed to return a list of models associated with the current instance of the ControlBase class. It initializes an empty list named 'out' to store the models. The function then checks if the attribute 'previous_controlnet' is not None, indicating that there is a previous controlnet instance available. If this condition is met, it calls the get_models method on the previous_controlnet instance and appends the returned models to the 'out' list. Finally, the function returns the 'out' list, which may contain models from the previous controlnet instance or remain empty if there is no previous instance.

**Note**: It is important to ensure that the previous_controlnet attribute is properly initialized before calling this function to avoid unexpected behavior. If there is no previous controlnet, the function will simply return an empty list.

**Output Example**: A possible appearance of the code's return value could be:
```python
[]
```
or, if there are models in the previous controlnet:
```python
['model1', 'model2', 'model3']
```
***
### FunctionDef copy_to(self, c)
**copy_to**: The function of copy_to is to transfer specific attributes from the current instance to another instance of a similar class.

**parameters**: The parameters of this Function.
· parameter1: c - An instance of a class that is intended to receive the copied attributes.

**Code Description**: The copy_to function is designed to facilitate the copying of specific attributes from the current object to another object, referred to as 'c'. This function assigns the values of four attributes from the current instance to the corresponding attributes of the target instance 'c'. The attributes being copied are:
- cond_hint_original: This likely represents some original condition hints used in the context of the control model.
- strength: This attribute may denote the intensity or weight of the control applied.
- timestep_percent_range: This attribute could indicate the range of time steps, expressed as a percentage, relevant to the operation of the model.
- global_average_pooling: This boolean or numerical attribute may determine whether global average pooling is applied in the model's processing.

The copy_to function is invoked by the copy methods of three different classes: ControlNet, ControlLora, and T2IAdapter. Each of these classes creates a new instance of itself (or its respective class) and then calls the copy_to function to transfer relevant attributes from the current instance to the newly created instance. This ensures that the new instance retains the necessary configuration and state from the original instance, allowing for consistent behavior across instances.

**Note**: It is important to ensure that the target instance 'c' is of a compatible type that contains the attributes being copied. Failure to do so may result in attribute errors or unexpected behavior.
***
### FunctionDef inference_memory_requirements(self, dtype)
**inference_memory_requirements**: The function of inference_memory_requirements is to calculate the memory requirements for inference based on the previous control net.

**parameters**: The parameters of this Function.
· dtype: This parameter specifies the data type for which the memory requirements are being calculated.

**Code Description**: The inference_memory_requirements function checks if there is a previous control net available. If the previous_controlnet attribute is not None, it recursively calls the inference_memory_requirements method of the previous control net, passing the dtype parameter to it. This allows the function to accumulate memory requirements from the previous control net. If there is no previous control net (i.e., previous_controlnet is None), the function returns 0, indicating that there are no memory requirements to account for in the current context.

This function is useful in scenarios where multiple control nets may be chained together, allowing for a dynamic assessment of memory needs based on the configuration of these nets. By leveraging the previous control net's memory requirements, it ensures that the current net can effectively manage its memory usage in relation to its predecessor.

**Note**: It is important to ensure that the previous_controlnet is properly initialized before calling this function to avoid unexpected behavior. Additionally, the dtype parameter should be compatible with the expected data types used in the previous control net.

**Output Example**: If the previous control net requires 512 MB of memory for the specified dtype, the function will return 512. If there is no previous control net, the function will return 0.
***
### FunctionDef control_merge(self, control_input, control_output, control_prev, output_dtype)
**control_merge**: The function of control_merge is to merge control inputs, outputs, and previous control states while applying specified transformations.

**parameters**: The parameters of this Function.
· control_input: A list of control input tensors that may be modified based on the strength attribute.
· control_output: A list of control output tensors that may undergo transformations and adjustments.
· control_prev: A dictionary containing previous control states for 'input', 'middle', and 'output'.
· output_dtype: The desired data type for the output tensors.

**Code Description**: The control_merge function is designed to process and merge multiple control tensors, which are essential in controlling the behavior of a model during inference or training. The function begins by initializing an output dictionary with keys 'input', 'middle', and 'output', each associated with an empty list.

If control_input is provided, the function iterates through each tensor in control_input. For each tensor, it applies a scaling factor defined by the strength attribute of the class. If the tensor's data type does not match the specified output_dtype, it converts the tensor to the appropriate type. The processed tensors are then inserted into the 'input' list of the output dictionary.

Next, if control_output is provided, the function processes each tensor similarly. It distinguishes between the last tensor in the control_output list, which is categorized as 'middle', and all others, which are categorized as 'output'. The function applies global average pooling if the global_average_pooling attribute is set to True, scales the tensor, and converts it to the output_dtype if necessary. The resulting tensors are appended to their respective lists in the output dictionary.

Finally, if control_prev is provided, the function merges the previous control states with the current outputs. It iterates over the keys 'input', 'middle', and 'output', checking if there are corresponding previous values. If a previous value exists and the current output at that index is None, it replaces the current output with the previous value. If both values exist, it combines them based on their shapes, ensuring that the larger tensor is preserved.

The control_merge function is called by the get_control methods of both the ControlNet and T2IAdapter classes. In these contexts, it is used to integrate control signals generated from noisy inputs and conditions, facilitating the model's ability to produce coherent outputs based on the provided control mechanisms.

**Note**: It is important to ensure that the control_input, control_output, and control_prev tensors are compatible in terms of dimensions and data types to avoid runtime errors during merging.

**Output Example**: A possible appearance of the code's return value could be:
{
    'input': [tensor1, tensor2, ...],
    'middle': [tensor3],
    'output': [tensor4, tensor5, ...]
}
Where tensor1, tensor2, etc., are the processed control tensors after applying the necessary transformations.
***
## ClassDef ControlNet
**ControlNet**: The function of ControlNet is to manage and process control hints in a control network, leveraging a specified control model for enhanced functionality.

**attributes**: The attributes of this Class.
· control_model: The control model used for processing inputs and generating outputs.
· load_device: Specifies the device on which the control model is loaded.
· control_model_wrapped: An instance of ModelPatcher that wraps the control model for additional functionality.
· global_average_pooling: A boolean flag indicating whether to apply global average pooling.
· model_sampling_current: Holds the current model sampling instance for processing.
· manual_cast_dtype: Specifies the data type for manual casting during processing.
· cond_hint: Holds the processed conditional hint for the control model.
· cond_hint_original: Stores the original conditional hint provided for processing.
· strength: A float value representing the strength of the control hint, defaulting to 1.0.
· timestep_percent_range: A tuple indicating the range of timesteps as percentages, defaulting to (0.0, 1.0).
· timestep_range: A tuple representing the actual range of timesteps after processing.
· previous_controlnet: A reference to a previous ControlBase instance, allowing for chaining of control networks.

**Code Description**: The ControlNet class extends the ControlBase class, inheriting its attributes and methods while adding specific functionality for managing control hints in a control network. The constructor initializes the control model and wraps it using the ModelPatcher for enhanced capabilities. It also sets up the device for computation and initializes other relevant attributes.

The `get_control` method is a core function that processes noisy input data (`x_noisy`) along with the current timestep (`t`), conditional data (`cond`), and the batch size (`batched_number`). It first checks if there is a previous control network and retrieves its control if applicable. It then validates the timestep against a defined range and processes the conditional hint to ensure it matches the dimensions of the input data. The method calculates the input for the control model, invokes the model with the processed data, and merges the results with any previous control outputs.

The `copy` method creates a duplicate of the current ControlNet instance, preserving its configuration. The `get_models` method retrieves the models associated with the current instance, including the wrapped control model. The `pre_run` method prepares the instance for execution by setting up the current model sampling. The `cleanup` method resets the model sampling instance and releases resources.

ControlNet is utilized by the ControlLora class, which extends its functionality by integrating control weights and additional configurations. The `load_controlnet` function is responsible for loading a ControlNet instance from a checkpoint file, handling various configurations and ensuring the control model is properly set up.

**Note**: It is essential to ensure that the conditional hints are appropriately sized and compatible with the input data dimensions. The strength parameter should be adjusted based on the desired influence of the control hints on the output.

**Output Example**: An example output from the `get_control` method could look like this:
{
  'input': [tensor1, tensor2],
  'middle': [tensor3],
  'output': [tensor4, tensor5]
}
Where `tensor1`, `tensor2`, etc., are processed tensors representing the control inputs and outputs.
### FunctionDef __init__(self, control_model, global_average_pooling, device, load_device, manual_cast_dtype)
**__init__**: The function of __init__ is to initialize an instance of the ControlNet class, setting up the control model and its associated parameters.

**parameters**: The parameters of this Function.
· control_model: The model that will be controlled and modified by the ControlNet instance.
· global_average_pooling: A boolean flag indicating whether to apply global average pooling (default is False).
· device: The device on which the model will be loaded for computation (default is None).
· load_device: The device from which the model will be loaded (default is None).
· manual_cast_dtype: The data type to which the model's parameters should be cast manually (default is None).

**Code Description**: The __init__ method is responsible for initializing the ControlNet class. It begins by calling the parent class's __init__ method with the specified device, ensuring that the base functionality is set up correctly. The control_model parameter is assigned to the instance variable self.control_model, which holds the model that this instance will manage.

Next, the load_device parameter is stored in self.load_device, which specifies the device from which the model will be loaded. The method then creates an instance of the ModelPatcher class, passing the control_model and load_device as arguments. This instance is assigned to self.control_model_wrapped, allowing the ControlNet class to manage and apply patches to the control model's weights and structure effectively. The ModelPatcher is crucial for modifying the model's behavior without altering its architecture directly.

The global_average_pooling parameter is stored in self.global_average_pooling, determining whether global average pooling will be applied during processing. The model_sampling_current variable is initialized to None, which will later be used to track the current sampling state of the model. Finally, the manual_cast_dtype parameter is assigned to self.manual_cast_dtype, allowing for manual control over the data type of the model's parameters.

This initialization process is vital for setting up the ControlNet instance, enabling it to interact with the control model and manage its behavior through the ModelPatcher. The integration of these components allows for dynamic adjustments to the model's functionality, enhancing its capabilities in various applications.

**Note**: When initializing the ControlNet class, it is important to ensure that the control_model is compatible with the ModelPatcher and that the specified devices are correctly configured to avoid runtime errors. Additionally, the global_average_pooling and manual_cast_dtype parameters should be set according to the specific requirements of the application to ensure optimal performance.
***
### FunctionDef get_control(self, x_noisy, t, cond, batched_number)
**get_control**: The function of get_control is to compute the control output based on noisy input, timestep, conditional data, and batch size.

**parameters**: The parameters of this Function.
· x_noisy: A tensor representing the noisy input data that will be processed to generate control outputs.  
· t: A tensor indicating the current timesteps for the model's operation.  
· cond: A dictionary containing conditional data necessary for the control model, including context for cross-attention.  
· batched_number: An integer that specifies how many times the tensor should be batched, influencing the output size.

**Code Description**: The get_control function is designed to generate control outputs by processing the noisy input tensor (x_noisy) in conjunction with the provided timestep (t), conditional data (cond), and the specified batch size (batched_number). 

Initially, the function checks if there is a previous control state available through the attribute `previous_controlnet`. If it exists, it recursively calls the get_control method of the previous control network, passing along the same parameters. This allows for a hierarchical control mechanism where the current control can depend on the previous state.

Next, the function evaluates the timestep range defined by the attribute `timestep_range`. If the current timestep (t[0]) falls outside the specified range, it returns the previous control output if available; otherwise, it returns None. This conditional check ensures that control outputs are only generated within a valid operational timeframe.

The function then determines the appropriate data type for processing. It defaults to the data type of the control model but can be overridden by a manually specified type through the `manual_cast_dtype` attribute. This flexibility allows for compatibility with various tensor types.

Following this, the function checks the compatibility of the conditional hint tensor (`cond_hint`) with the noisy input tensor. If the dimensions do not match, it utilizes the `common_upscale` function to resize the conditional hint to match the dimensions of the noisy input, ensuring that both tensors can be processed together without dimension mismatches.

The context for the control model is extracted from the conditional data, and any additional conditional data (y) is also prepared for processing. The current timestep is calculated using the `model_sampling_current.timestep` method, and the noisy input is adjusted using `model_sampling_current.calculate_input`.

Finally, the control model is invoked with the processed noisy input, conditional hint, timesteps, and context. The resulting control output is then merged with any previous control outputs using the `control_merge` function. This merging process integrates the current control output with any historical data, ensuring a coherent output that reflects both the current and previous states.

The get_control function is integral to the operation of control mechanisms within the ControlNet and T2IAdapter classes, facilitating the generation of control outputs that guide the model's behavior during inference.

**Note**: It is crucial to ensure that the input tensors (x_noisy, cond) are correctly shaped and compatible with the control model to avoid runtime errors. Additionally, the function's behavior is sensitive to the specified timestep range and the presence of previous control states.

**Output Example**: A possible appearance of the code's return value could be a tensor representing the control output, which may look like:
{
    'control_output': tensor([[...], [...], ...]),
    'previous_control': tensor([[...], [...], ...])
} 
Where the tensors contain the processed control data ready for further use in the model.
***
### FunctionDef copy(self)
**copy**: The function of copy is to create a new instance of the ControlNet class and transfer specific attributes from the current instance to the new instance.

**parameters**: The parameters of this Function.
· parameter1: None

**Code Description**: The copy function is responsible for instantiating a new ControlNet object using the current instance's attributes. It initializes the new object with the same control model, global average pooling setting, load device, and manual cast data type as the current instance. After creating the new instance, the function calls the copy_to method to transfer specific attributes from the current instance to the newly created instance. The attributes transferred include cond_hint_original, strength, timestep_percent_range, and global_average_pooling, which are essential for maintaining the configuration and state of the ControlNet instance.

The copy function is called within the apply_controlnet function, which is part of the ControlNetApplyAdvanced class. In this context, apply_controlnet utilizes the copy function to create a new ControlNet instance that is configured based on the current state of the control model. This allows for the application of control mechanisms to images while preserving the necessary parameters and settings from the original instance. The copy function ensures that each new instance of ControlNet can operate independently while retaining the relevant attributes from its predecessor.

**Note**: It is crucial to ensure that the new instance created by the copy function is compatible with the attributes being copied. Any discrepancies in attribute types or names may lead to errors during execution.

**Output Example**: A possible appearance of the code's return value could be a new ControlNet object with attributes set as follows:
- control_model: <current_instance.control_model>
- global_average_pooling: <current_instance.global_average_pooling>
- load_device: <current_instance.load_device>
- manual_cast_dtype: <current_instance.manual_cast_dtype>
- cond_hint_original: <current_instance.cond_hint_original>
- strength: <current_instance.strength>
- timestep_percent_range: <current_instance.timestep_percent_range>
***
### FunctionDef get_models(self)
**get_models**: The function of get_models is to retrieve a list of models, including a specific control model.

**parameters**: The parameters of this Function.
· There are no parameters for this function.

**Code Description**: The get_models function is designed to extend the functionality of a parent class method also named get_models. It first calls the parent class's get_models method using `super().get_models()`, which retrieves a list of models defined in the parent class. This list is stored in the variable `out`. Subsequently, the function appends an additional model, referred to as `self.control_model_wrapped`, to the list `out`. Finally, the modified list, which now includes both the parent class models and the control model, is returned. This function is particularly useful in scenarios where additional models need to be included in the existing model list without altering the original implementation of the parent class.

**Note**: It is important to ensure that `self.control_model_wrapped` is properly defined and initialized within the class before calling this function. Failure to do so may result in an AttributeError.

**Output Example**: A possible appearance of the code's return value could be:
```python
['model1', 'model2', 'model3', 'control_model']
```
In this example, 'model1', 'model2', and 'model3' are models retrieved from the parent class, while 'control_model' represents the additional model wrapped by `self.control_model_wrapped`.
***
### FunctionDef pre_run(self, model, percent_to_timestep_function)
**pre_run**: The function of pre_run is to prepare the model for execution by invoking the superclass's pre_run method and setting the current sampling model.

**parameters**: The parameters of this Function.
· model: An object representing the model that is to be prepared for execution.
· percent_to_timestep_function: A function that determines the relationship between percentage completion and time steps.

**Code Description**: The pre_run function begins by calling the pre_run method of its superclass, ensuring that any necessary initialization or setup defined in the parent class is executed. This is crucial for maintaining the integrity of the class hierarchy and ensuring that all inherited behaviors are properly initialized. Following this, the function assigns the current sampling model from the provided model object to the instance variable model_sampling_current. This assignment allows the instance to keep track of the model's sampling state, which may be used later in the execution process.

**Note**: It is important to ensure that the model parameter passed to pre_run is correctly instantiated and contains a valid model_sampling attribute. Failure to do so may result in errors during execution or unexpected behavior. Additionally, any modifications to the superclass's pre_run method should be carefully considered, as they may impact the functionality of this method.
***
### FunctionDef cleanup(self)
**cleanup**: The function of cleanup is to reset the current model sampling state and invoke the parent class's cleanup method.

**parameters**: The parameters of this Function.
· There are no parameters for this function.

**Code Description**: The cleanup function is designed to reset the state of the current model sampling by setting the attribute `model_sampling_current` to `None`. This indicates that there is no active model sampling at the time of cleanup. Following this, the function calls the `cleanup` method from the parent class using `super().cleanup()`. This ensures that any additional cleanup processes defined in the parent class are executed, maintaining the integrity of the class hierarchy and ensuring that all necessary cleanup operations are performed.

**Note**: It is important to call this cleanup method when the object is no longer needed or before reinitializing the object to prevent memory leaks or unintended behavior due to lingering state. This function does not take any parameters and is intended to be called without arguments.
***
## ClassDef ControlLoraOps
**ControlLoraOps**: The function of ControlLoraOps is to provide custom linear and convolutional operations with additional capabilities for handling weights and biases in a neural network context.

**attributes**: The attributes of this Class.
· Linear: A nested class that implements a custom linear layer.
· Conv2d: A nested class that implements a custom 2D convolutional layer.

**Code Description**: The ControlLoraOps class serves as a container for two specialized neural network layer implementations: Linear and Conv2d. Both of these nested classes inherit from `torch.nn.Module`, which is the base class for all neural network modules in PyTorch.

The Linear class is designed to perform linear transformations on input data. It initializes with parameters such as `in_features`, `out_features`, and optional parameters for bias, device, and dtype. The forward method computes the output by applying a linear transformation to the input. It utilizes a helper function `ldm_patched.modules.ops.cast_bias_weight` to retrieve the appropriate weights and biases. If the `up` attribute is not None, it adds a computed matrix product of `up` and `down` to the weight before applying the linear transformation.

Similarly, the Conv2d class implements a 2D convolutional layer. It is initialized with parameters such as `in_channels`, `out_channels`, `kernel_size`, and other convolution-specific parameters like stride, padding, and dilation. The forward method also calls `ldm_patched.modules.ops.cast_bias_weight` to obtain the weights and biases. If the `up` attribute is set, it modifies the weight in a manner analogous to the Linear class before performing the convolution operation.

The ControlLoraOps class is utilized by other classes in the project, specifically `control_lora_ops`, which inherits from ControlLoraOps and combines it with additional functionalities from `ldm_patched.modules.ops.disable_weight_init` and `ldm_patched.modules.ops.manual_cast`. This indicates that the operations defined in ControlLoraOps are integral to the behavior of these derived classes, allowing them to leverage the custom linear and convolutional operations while also integrating weight initialization and casting functionalities.

**Note**: When using the ControlLoraOps class, it is essential to ensure that the `up` and `down` attributes are properly initialized if they are to be utilized in the forward computations. Additionally, the use of the `cast_bias_weight` function is crucial for the correct functioning of the linear and convolutional operations.

**Output Example**: A possible output from the Linear class's forward method could be a tensor representing the transformed input data after applying the linear operation, while the Conv2d class's forward method would return a tensor representing the feature map resulting from the convolution operation.
### ClassDef Linear
**Linear**: The function of Linear is to implement a linear transformation layer in a neural network.

**attributes**: The attributes of this Class.
· in_features: An integer representing the number of input features to the linear layer.  
· out_features: An integer representing the number of output features from the linear layer.  
· weight: A tensor that holds the weights of the linear transformation.  
· up: An optional tensor used for additional transformations, if applicable.  
· down: An optional tensor used for additional transformations, if applicable.  
· bias: An optional tensor that holds the bias values for the linear transformation, if enabled.  

**Code Description**: The Linear class inherits from `torch.nn.Module`, making it a part of the PyTorch neural network module system. The constructor (`__init__`) initializes the linear layer with specified input and output feature sizes, as well as optional parameters for bias, device, and data type. The `weight`, `up`, `down`, and `bias` attributes are initialized to `None` at the start.

The `forward` method defines the computation performed at every call of the layer. It first retrieves the weight and bias tensors by calling the `cast_bias_weight` function from the `ldm_patched.modules.ops` module, which is expected to handle the casting of these tensors based on the input. If the `up` tensor is not `None`, it performs an additional transformation by computing the matrix multiplication of the flattened `up` and `down` tensors, reshaping the result to match the weight's shape and type. This transformed weight is then added to the original weight before applying the linear transformation using `torch.nn.functional.linear`. If `up` is `None`, it simply applies the linear transformation using the original weight and bias.

**Note**: When using this class, ensure that the input tensor's shape matches the expected dimensions based on `in_features`. Additionally, if `up` and `down` tensors are utilized, they should be appropriately initialized to avoid runtime errors.

**Output Example**: Given an input tensor of shape (batch_size, in_features), the output of the forward method will be a tensor of shape (batch_size, out_features) after applying the linear transformation. For example, if `in_features` is 3 and `out_features` is 2, an input tensor of shape (4, 3) would yield an output tensor of shape (4, 2).
#### FunctionDef __init__(self, in_features, out_features, bias, device, dtype)
**__init__**: The function of __init__ is to initialize an instance of the class with specified input and output features, along with optional parameters for bias, device, and data type.

**parameters**: The parameters of this Function.
· in_features: An integer representing the number of input features for the layer.  
· out_features: An integer representing the number of output features for the layer.  
· bias: A boolean indicating whether to include a bias term in the layer. Defaults to True.  
· device: An optional parameter specifying the device on which the tensor will be allocated (e.g., CPU or GPU).  
· dtype: An optional parameter specifying the data type of the tensor (e.g., float32, float64).  

**Code Description**: The __init__ function is a constructor that initializes an instance of the class. It takes in several parameters that define the characteristics of the layer. The parameters in_features and out_features are essential as they determine the dimensions of the input and output tensors, respectively. The bias parameter allows the user to specify whether a bias term should be included in the layer's computations, with a default value of True indicating that a bias will be included.

The function also initializes several attributes: 
- self.weight is set to None, indicating that the weight tensor has not yet been defined or initialized.
- self.up and self.down are also set to None, which may be placeholders for future operations or transformations that will be defined later in the class.
- self.bias is initialized to None, which suggests that if bias is included, it will be defined later in the class.

The factory_kwargs dictionary is created to hold the device and dtype parameters, which can be used later for tensor creation, ensuring that the tensors are allocated on the correct device and with the specified data type.

**Note**: It is important to ensure that the in_features and out_features parameters are set correctly, as they directly affect the layer's functionality. Additionally, users should be aware that the actual weight and bias tensors will need to be initialized separately after the instance is created.
***
#### FunctionDef forward(self, input)
**forward**: The function of forward is to perform a linear transformation on the input tensor using the layer's weight and bias parameters.

**parameters**: The parameters of this Function.
· input: A tensor that serves as the input for the linear transformation.

**Code Description**: The forward function is responsible for executing the linear transformation operation in the context of a neural network layer. It begins by invoking the cast_bias_weight function, which transfers the model's weight and bias to the appropriate device and data type based on the input tensor. This ensures that the operations performed within the forward method are compatible with the device on which the input tensor resides.

Once the weight and bias are obtained, the function checks if the 'up' attribute of the layer is not None. If 'up' is defined, it indicates that an additional transformation is to be applied. In this case, the function computes a matrix multiplication between the flattened 'up' and 'down' attributes, reshapes the result to match the weight's shape, and adds it to the weight tensor. This combined weight is then used in the linear transformation along with the bias.

If the 'up' attribute is None, the function simply performs the linear transformation using the weight and bias obtained from the cast_bias_weight function. The linear transformation is executed using PyTorch's functional API, specifically the torch.nn.functional.linear method, which efficiently computes the output tensor.

The forward method is a critical component of the Linear layer within the ControlLoraOps class, and it is designed to ensure that the linear transformation is performed correctly and efficiently, taking into account the device and data type of the input tensor. The use of the cast_bias_weight function within this method is essential for maintaining consistency and compatibility across different hardware configurations.

**Note**: It is important to ensure that the input tensor is properly defined with a specific device and data type to prevent any unexpected behavior during the execution of the forward method.

**Output Example**: If the input tensor is a 2D tensor of shape (batch_size, input_features) and the weight tensor is of shape (output_features, input_features), the function will return a tensor of shape (batch_size, output_features) after applying the linear transformation.
***
***
### ClassDef Conv2d
**Conv2d**: The function of Conv2d is to implement a 2D convolutional layer in a neural network.

**attributes**: The attributes of this Class.
· in_channels: The number of input channels for the convolutional layer.  
· out_channels: The number of output channels produced by the convolution.  
· kernel_size: The size of the convolutional kernel. This can be a single integer or a tuple of two integers.  
· stride: The stride of the convolution. This controls how much the kernel moves at each step. Default is 1.  
· padding: The amount of padding added to both sides of the input. Default is 0.  
· dilation: The spacing between kernel elements. Default is 1.  
· groups: The number of groups for grouped convolution. Default is 1, which means standard convolution.  
· bias: A boolean indicating whether to include a bias term. Default is True.  
· padding_mode: The mode for padding, which can be 'zeros' or other types. Default is 'zeros'.  
· device: The device on which the tensor is allocated (e.g., CPU or GPU).  
· dtype: The desired data type of the tensor.

**Code Description**: The Conv2d class inherits from `torch.nn.Module`, making it a part of the PyTorch neural network module system. The constructor initializes several parameters that define the behavior of the convolutional layer, including the number of input and output channels, the size of the kernel, stride, padding, dilation, and whether to use bias. The class also initializes attributes for weights and biases, which are set to None initially, and additional attributes for upsampling and downsampling operations.

The `forward` method defines the computation performed at every call. It first retrieves the weight and bias using a helper function `ldm_patched.modules.ops.cast_bias_weight`. If the `up` attribute is not None, it performs a convolution operation that includes an additional term computed from the `up` and `down` attributes. This term is reshaped to match the weight's shape and is added to the convolution operation. If `up` is None, it simply performs the standard convolution using the weight and bias.

**Note**: When using this class, ensure that the input tensor matches the expected shape based on the number of input channels. The `up` and `down` attributes should be properly initialized if they are to be used in the convolution operation. The class is designed to work within the PyTorch framework, so familiarity with PyTorch's tensor operations is beneficial.

**Output Example**: A possible output of the `forward` method could be a tensor representing the result of the convolution operation, which would have a shape determined by the input dimensions, kernel size, stride, and padding. For instance, if the input tensor has a shape of (batch_size, in_channels, height, width) and the convolution is performed with appropriate parameters, the output tensor might have a shape of (batch_size, out_channels, new_height, new_width), where new_height and new_width are calculated based on the convolution parameters.
#### FunctionDef __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
**__init__**: The function of __init__ is to initialize an instance of the Conv2d class with specified parameters for a 2D convolutional layer.

**parameters**: The parameters of this Function.
· in_channels: The number of input channels for the convolution operation.  
· out_channels: The number of output channels produced by the convolution.  
· kernel_size: The size of the convolutional kernel (filter). This can be a single integer or a tuple of two integers.  
· stride: The stride of the convolution. This determines how much the filter moves at each step. Default is 1.  
· padding: The amount of padding added to both sides of the input. Default is 0.  
· dilation: The spacing between kernel elements. Default is 1.  
· groups: The number of groups for grouped convolution. Default is 1, which means standard convolution.  
· bias: A boolean indicating whether to include a bias term in the convolution. Default is True.  
· padding_mode: The mode of padding, which can be 'zeros' or other types. Default is 'zeros'.  
· device: The device on which the tensor is allocated (e.g., CPU or GPU). Default is None.  
· dtype: The desired data type of the tensor. Default is None.  

**Code Description**: The __init__ function is a constructor for the Conv2d class, which is designed to create a 2D convolutional layer in a neural network. It takes several parameters that define the characteristics of the convolutional layer. The in_channels and out_channels parameters specify the number of input and output channels, respectively. The kernel_size parameter defines the dimensions of the convolutional filter, which can be specified as a single integer (for square filters) or as a tuple (for rectangular filters). 

The stride parameter controls how the filter moves across the input image, while the padding parameter determines how much padding is added to the input to control the spatial dimensions of the output. The dilation parameter allows for dilated convolutions, which can expand the receptive field of the kernel without increasing the number of parameters. The groups parameter enables grouped convolutions, which can be useful for certain architectures. 

The bias parameter indicates whether a bias term should be included in the convolution operation. The padding_mode parameter specifies the method of padding used, with 'zeros' being the default. The device and dtype parameters allow for flexibility in specifying where the tensors are stored and their data types. 

The constructor also initializes several attributes, including weight and bias, which are set to None initially, as well as up, down, and other parameters related to the convolution operation.

**Note**: It is important to ensure that the in_channels parameter matches the number of channels in the input data when using this class. Additionally, the kernel_size, stride, and padding should be chosen carefully to achieve the desired output dimensions.
***
#### FunctionDef forward(self, input)
**forward**: The function of forward is to perform a convolution operation on the input tensor using the layer's weight and bias parameters, potentially modified by an additional transformation if applicable.

**parameters**: The parameters of this Function.
· input: A tensor that serves as the input data for the convolution operation.

**Code Description**: The forward function is responsible for executing the convolution operation in the Conv2d layer of the ControlLoraOps class. It begins by calling the cast_bias_weight function, which retrieves the weight and bias tensors from the layer, ensuring they are transferred to the appropriate device and data type based on the input tensor. This is crucial for maintaining compatibility and performance across different hardware configurations.

If the layer has an additional transformation defined by the attribute 'up', the function computes a modified weight by adding the result of a matrix multiplication between the flattened 'up' and 'down' tensors. This modified weight is then reshaped to match the dimensions of the original weight tensor and is cast to the same data type as the input tensor. The convolution operation is performed using the PyTorch function torch.nn.functional.conv2d, which takes the input tensor, the computed weight (either modified or original), the bias, and other parameters such as stride, padding, dilation, and groups.

If the 'up' attribute is not defined (i.e., it is None), the function simply performs the convolution using the original weight and bias without any modification. This design allows for flexibility in how the convolution is applied, accommodating different configurations of the layer.

The forward function is a critical component of the Conv2d layer's operation, as it directly influences how input data is transformed through the neural network. By leveraging the cast_bias_weight function, it ensures that the parameters are correctly set up for the convolution, thereby optimizing performance and ensuring consistency in tensor operations.

**Note**: It is essential that the input tensor is properly defined with respect to its device and data type to avoid any unexpected behavior during the convolution operation.

**Output Example**: If the input tensor has a shape of (batch_size, channels, height, width) and is on a CUDA device with dtype float32, the function will return the result of the convolution operation, which will also be a tensor on the CUDA device with dtype float32, shaped according to the convolution parameters.
***
***
## ClassDef ControlLora
**ControlLora**: The function of ControlLora is to manage and process control weights within a control network, enhancing the functionality of the underlying control model.

**attributes**: The attributes of this Class.
· control_weights: A dictionary containing the weights used for controlling the model's behavior.
· global_average_pooling: A boolean flag indicating whether to apply global average pooling during processing.
· manual_cast_dtype: Specifies the data type for manual casting during processing.
· control_model: An instance of the ControlNet class that processes inputs and generates outputs.

**Code Description**: The ControlLora class extends the ControlNet class, inheriting its attributes and methods while adding specific functionality for integrating control weights into the control network. The constructor initializes the control weights and sets up the global average pooling option and device for computation.

The `pre_run` method prepares the ControlLora instance for execution by configuring the control model based on the provided model and its configuration. It copies the UNet configuration, adjusts the hint channels according to the control weights, and determines the appropriate operations class based on whether manual casting is required. The control model is then instantiated and moved to the appropriate device.

The `copy` method creates a duplicate of the current ControlLora instance, preserving its configuration and control weights. The `cleanup` method releases resources associated with the control model, ensuring that it is properly disposed of when no longer needed.

The `get_models` method retrieves the models associated with the current instance, including the control model. The `inference_memory_requirements` method calculates the memory requirements for inference based on the control weights and the specified data type.

ControlLora is utilized by the `load_controlnet` function, which is responsible for loading a ControlLora instance from a checkpoint file. This function checks for the presence of control weights in the checkpoint data and initializes the ControlLora instance accordingly. If control weights are not found, it falls back to loading a ControlNet instance.

**Note**: It is essential to ensure that the control weights are correctly formatted and compatible with the expected input dimensions of the control model. The global average pooling option should be set based on the specific requirements of the model being used.

**Output Example**: An example output from the `get_models` method could look like this:
{
  'control_model': <ControlNet instance>,
  'control_weights': { 'weight1': tensor1, 'weight2': tensor2 }
}
Where `tensor1`, `tensor2`, etc., are the processed tensors representing the control weights associated with the control model.
### FunctionDef __init__(self, control_weights, global_average_pooling, device)
**__init__**: The function of __init__ is to initialize an instance of the ControlLora class, setting up control weights and pooling options.

**parameters**: The parameters of this Function.
· control_weights: This parameter holds the control weights that will be used for processing within the control network.  
· global_average_pooling: A boolean flag indicating whether to apply global average pooling during processing. The default value is False.  
· device: This parameter specifies the device (CPU or GPU) on which the computations will be performed. If not provided, it defaults to None.

**Code Description**: The __init__ method is the constructor for the ControlLora class, which inherits from the ControlBase class. This method initializes the instance by calling the constructor of the parent class ControlBase with the device parameter. This ensures that the foundational attributes and functionalities defined in ControlBase are properly set up for the ControlLora instance.

The control_weights parameter is assigned to the instance variable self.control_weights, which is essential for the control mechanisms that the ControlLora class will implement. The global_average_pooling parameter is also stored in the instance variable self.global_average_pooling, allowing the class to determine whether to apply global average pooling during its operations.

By invoking ControlBase.__init__(self, device), the constructor ensures that all attributes defined in ControlBase, such as cond_hint_original, cond_hint, strength, timestep_percent_range, and others, are initialized. This establishes a solid foundation for the ControlLora class to build upon, enabling it to manage control hints and their processing effectively.

The relationship with its callees is significant, as ControlLora extends the functionalities of ControlBase, which is designed to manage control hints in various control networks. This hierarchical structure allows ControlLora to leverage the methods and attributes of ControlBase, ensuring that it can handle control weights and pooling options while maintaining the integrity of the control hint management system.

**Note**: It is important to ensure that the control_weights provided are compatible with the expected input dimensions of the control network. Additionally, the device parameter should be specified according to the available computational resources to optimize performance.
***
### FunctionDef pre_run(self, model, percent_to_timestep_function)
**pre_run**: The function of pre_run is to prepare the ControlLora model for execution by configuring its parameters and initializing the control model with the appropriate weights.

**parameters**: The parameters of this Function.
· parameter1: model - An instance of the model that contains the configuration and weights to be used for initializing the ControlLora model.  
· parameter2: percent_to_timestep_function - A function that maps a percentage to a timestep, used for controlling the diffusion process.

**Code Description**: The pre_run method is a crucial step in the setup of the ControlLora model, which is a part of the larger ControlNet architecture. This method begins by invoking the pre_run method of its superclass to ensure any necessary initialization is performed. It then copies the UNet configuration from the provided model, specifically removing the "out_channels" attribute, which is not needed for the ControlLora model.

Next, the method sets the "hint_channels" in the controlnet_config based on the shape of the input weights for the hint block. This is essential for ensuring that the model can correctly process the hint inputs during inference. The manual_cast_dtype is also set based on the model's configuration, which determines how data types are handled within the model.

The method then defines a nested class, control_lora_ops, which inherits from ControlLoraOps and either disables weight initialization or applies manual casting based on whether manual_cast_dtype is specified. This class is assigned to the "operations" key in the controlnet_config, along with the appropriate data type.

Following this, the ControlNet model is instantiated using the modified controlnet_config, and it is moved to the appropriate device (CPU or GPU) using the get_torch_device function. This ensures that the model is ready for computation on the available hardware.

The method then retrieves the state dictionary of the diffusion model and the newly created control model. It iterates through the state dictionary of the diffusion model, attempting to set the corresponding weights in the control model using the set_attr utility function. This is critical for initializing the control model with the correct parameters.

Finally, the method updates the control model's weights based on the control_weights attribute, ensuring that all necessary parameters are correctly set before the model is used for inference or training.

The pre_run method is integral to the functioning of the ControlLora class, as it prepares the model with the necessary configurations and weights, facilitating its operation within the broader context of image processing tasks managed by the ControlNet architecture.

**Note**: It is important to ensure that the model passed to pre_run is correctly configured and that the control_weights are properly initialized. Additionally, users should be aware that any changes to the model's configuration after calling pre_run may require re-initialization of the ControlLora model to reflect those changes.
#### ClassDef control_lora_ops
**control_lora_ops**: The function of control_lora_ops is to extend the functionalities of ControlLoraOps while incorporating features from disable_weight_init.

**attributes**: The attributes of this Class.
· Inherits attributes from ControlLoraOps: This includes the nested classes Linear and Conv2d, which implement custom linear and convolutional operations respectively.
· Inherits attributes from disable_weight_init: This includes modified versions of layers that disable weight initialization.

**Code Description**: The control_lora_ops class is a specialized implementation that inherits from both ControlLoraOps and disable_weight_init. By doing so, it combines the capabilities of custom linear and convolutional operations with the ability to disable weight initialization for these layers. 

The ControlLoraOps class provides a framework for defining custom neural network layers, specifically designed to handle weights and biases in a flexible manner. It includes nested classes for Linear and Conv2d operations, which are essential for performing linear transformations and 2D convolutions, respectively. These operations are enhanced by the ability to manage additional parameters like `up` and `down`, which can modify the weights during the forward pass.

On the other hand, the disable_weight_init class offers modified versions of standard PyTorch layers that do not initialize weights. This is particularly useful in scenarios where weight initialization may interfere with specific training strategies or model behaviors. By inheriting from disable_weight_init, the control_lora_ops class ensures that any instances of Linear or Conv2d created through it will not undergo weight initialization unless explicitly managed.

The integration of these two classes allows for a versatile approach to building neural network architectures that require both custom operations and specific weight management strategies. This makes control_lora_ops a crucial component in the broader context of the project, enabling developers to create models that leverage advanced operational capabilities while maintaining control over weight initialization.

**Note**: When utilizing the control_lora_ops class, it is important to be aware of the implications of disabling weight initialization. Users should ensure that the behavior of the model aligns with their training objectives, particularly in relation to the management of the `up` and `down` attributes in the Linear and Conv2d operations. Proper initialization and handling of these parameters are essential for achieving the desired performance in neural network training and inference.
***
#### ClassDef control_lora_ops
**control_lora_ops**: The function of control_lora_ops is to extend the functionalities of ControlLoraOps and manual_cast, providing specialized operations for neural network layers with custom weight handling.

**attributes**: The attributes of this Class.
· Inherits from ControlLoraOps: This allows access to custom linear and convolutional operations defined in ControlLoraOps.
· Inherits from manual_cast: This enables specific weight casting behavior during forward passes.

**Code Description**: The control_lora_ops class is a specialized implementation that combines features from both the ControlLoraOps and manual_cast classes. By inheriting from ControlLoraOps, it gains access to the custom Linear and Conv2d layers that are designed to handle weights and biases in a neural network context. This allows for enhanced flexibility in defining how inputs are transformed through linear and convolutional operations.

Additionally, by inheriting from manual_cast, control_lora_ops ensures that the forward methods of these layers can utilize custom weight casting behavior. This is particularly important in scenarios where weight initialization needs to be controlled or modified dynamically. The ldm_patched_cast_weights attribute, which is set to True in the manual_cast class, indicates that the layers will apply specific transformations to weights and biases during their forward computations.

The integration of these two classes allows control_lora_ops to be utilized in various components of the project that require both custom layer operations and specific weight handling. This makes it a critical component for ensuring that neural network models can operate effectively under the defined constraints and behaviors.

**Note**: When using the control_lora_ops class, it is essential to ensure that any necessary attributes, such as weights and biases, are properly initialized. Additionally, understanding the implications of the ldm_patched_cast_weights attribute is crucial for controlling the behavior of the model during training and inference.
***
***
### FunctionDef copy(self)
**copy**: The function of copy is to create a new instance of the ControlLora class and transfer relevant attributes from the current instance to this new instance.

**parameters**: The parameters of this Function.
· parameter1: None

**Code Description**: The copy function is responsible for instantiating a new ControlLora object using the current instance's control weights and global average pooling attribute. It then invokes the copy_to method to transfer specific attributes from the current instance to the newly created ControlLora instance. The attributes transferred include cond_hint_original, strength, timestep_percent_range, and global_average_pooling, which are essential for maintaining the state and configuration of the control model.

This function is called within the context of the apply_controlnet function found in the ControlNetApplyAdvanced class. The apply_controlnet function utilizes the copy method to ensure that each control net being applied retains the necessary attributes from its source instance. This is crucial for the correct functioning of the control mechanism in the model, allowing for consistent behavior across different instances of control nets.

**Note**: It is important to ensure that the target instance created by the copy method is compatible with the attributes being copied. Any mismatch may lead to attribute errors or unexpected behavior during execution.

**Output Example**: A possible appearance of the code's return value could be a new ControlLora instance with the same attributes as the original instance, ready to be used in further processing within the control net application.
***
### FunctionDef cleanup(self)
**cleanup**: The function of cleanup is to release resources associated with the control model and perform necessary cleanup operations.

**parameters**: The parameters of this Function.
· There are no parameters for this function.

**Code Description**: The cleanup function is responsible for properly deallocating resources associated with the control model within the class. It begins by deleting the attribute `control_model`, which is expected to hold a reference to an object or resource that needs to be cleaned up. By using the `del` statement, it ensures that the reference to the control model is removed from the instance, allowing Python's garbage collector to reclaim the memory if there are no other references to that object. Following this, the function sets `self.control_model` to `None`, explicitly indicating that the control model is no longer in use. This step is a good practice as it helps prevent accidental access to a deleted or invalid object. Finally, the function calls `super().cleanup()`, which invokes the cleanup method of the parent class. This ensures that any additional cleanup logic defined in the parent class is executed, maintaining the integrity of the class hierarchy and ensuring that all resources are properly managed.

**Note**: It is important to call the cleanup function when the control model is no longer needed to avoid memory leaks and ensure that all resources are released appropriately. This function should be used in contexts where the lifecycle of the control model is managed, such as when an object is being destroyed or when the control model is being replaced with a new instance.
***
### FunctionDef get_models(self)
**get_models**: The function of get_models is to retrieve models from the current and previous control networks.

**parameters**: The parameters of this Function.
· None

**Code Description**: The `get_models` method is a straightforward function that serves to gather models from the current instance of the class and any linked previous control networks. It begins by invoking the `get_models` method from its parent class, `ControlBase`, which is responsible for managing control hints and their processing. This call ensures that any models associated with the current control network are included in the output. If there is a reference to a previous control network (i.e., `previous_controlnet` is not None), the method will also retrieve models from that instance by appending the results to the output list. The final output is a consolidated list of models from both the current and any previous control networks, thus facilitating the management of multiple control networks in a structured manner.

The relationship with its callees is significant as it leverages the functionality provided by the `ControlBase` class, ensuring that the model retrieval process is consistent and integrated with the overall control network management system. This method is particularly useful in scenarios where multiple control networks are used in sequence or in conjunction, allowing for a comprehensive view of all models involved.

**Note**: It is essential to ensure that the previous control network is properly set up before calling this method to avoid unexpected results or errors in model retrieval.

**Output Example**: A possible appearance of the code's return value could look like this:
```python
[
  model1,
  model2,
  model3
]
```
Where `model1`, `model2`, and `model3` represent the models retrieved from the current and previous control networks.
***
### FunctionDef inference_memory_requirements(self, dtype)
**inference_memory_requirements**: The function of inference_memory_requirements is to calculate the memory requirements for inference based on control weights and the specified data type.

**parameters**: The parameters of this Function.
· dtype: The data type for which the memory size is to be calculated.

**Code Description**: The inference_memory_requirements function computes the total memory required for inference by leveraging the control weights associated with the current instance of the ControlLora class. It first calls the calculate_parameters function from the utils module to determine the number of parameters related to the control weights. This count is then multiplied by the size in bytes of the specified data type, which is obtained by calling the dtype_size function from the model_management module. The result of this multiplication gives the total memory requirement for the control weights.

Additionally, the function incorporates the memory requirements of any previous control network by invoking the inference_memory_requirements method of the ControlBase class, from which ControlLora inherits. This ensures that if there is a previous control network, its memory requirements are included in the total calculation. The final return value is the sum of the memory requirements for the current control weights and any previous control networks, providing a comprehensive assessment of the memory needed for inference.

**Note**: It is essential to ensure that the dtype parameter passed to this function is a valid PyTorch data type to avoid errors during the memory size calculation. The function is designed to work seamlessly with the control weights and their associated data types.

**Output Example**: An example output from the inference_memory_requirements function could be a numerical value representing the total memory requirement in bytes, such as 2048000, indicating that 2,048,000 bytes of memory are needed for inference with the specified control weights and data type.
***
## FunctionDef load_controlnet(ckpt_path, model)
**load_controlnet**: The function of load_controlnet is to load a ControlNet model from a specified checkpoint file, handling various configurations and ensuring the model is properly set up for inference or training.

**parameters**: The parameters of this Function.
· ckpt_path: A string representing the file path to the checkpoint from which the ControlNet model will be loaded.
· model: An optional parameter that can be passed to integrate with an existing model, enhancing the loading process if provided.

**Code Description**: The load_controlnet function is responsible for loading a ControlNet model from a specified checkpoint file. It begins by utilizing the load_torch_file function to read the checkpoint data, which is expected to contain the model's weights and configuration. The function checks if the loaded data includes "lora_controlnet"; if so, it returns an instance of the ControlLora class, which is a specialized version of ControlNet designed to manage control weights.

If the checkpoint data contains keys indicative of a diffusers format, the function proceeds to extract and configure the ControlNet model accordingly. It retrieves the UNet data type using the unet_dtype function and constructs the controlnet configuration using the unet_config_from_diffusers_unet function. The function then maps the keys from the controlnet data to the expected format for the ControlNet model, ensuring that all necessary parameters are correctly aligned.

The function also handles the loading of weights into the ControlNet model. If the loaded data indicates that the model is a "diff controlnet" and a model is provided, it merges the loaded weights with those of the existing model. This is done to enhance the model's capabilities by integrating the control weights effectively.

In cases where the controlnet configuration is not found, the function defaults to creating a new instance of the ControlNet class using the model_config_from_unet function to derive the necessary configuration from the UNet data. The function also manages the device on which the model will be loaded, utilizing the get_torch_device function to ensure compatibility with the available hardware.

Finally, the function checks for any leftover keys in the controlnet data that were not mapped to the new configuration and prints them for debugging purposes. The function concludes by returning the instantiated ControlNet or ControlLora model, ready for use in subsequent operations.

The load_controlnet function is integral to the initialization and configuration of ControlNet models within the project, serving as a bridge between checkpoint data and model instantiation. It is called by higher-level functions that require a ControlNet model for processing tasks, ensuring that the model is correctly set up based on the provided checkpoint.

**Note**: Users should ensure that the checkpoint file exists at the specified path and that the data contained within it is structured correctly to avoid runtime errors during the loading process. Additionally, the model parameter should be provided if integration with an existing model is desired.

**Output Example**: A possible return value from load_controlnet could be an instance of ControlNet or ControlLora, initialized with the loaded weights and configuration, ready for inference or further training. For example:
```
ControlNet(model_config, loaded_weights)
```
### ClassDef WeightsLoader
**WeightsLoader**: The function of WeightsLoader is to serve as a neural network module for loading and managing weights in a PyTorch-based model.

**attributes**: The attributes of this Class.
· No attributes are defined within the WeightsLoader class as it currently stands.

**Code Description**: The WeightsLoader class inherits from `torch.nn.Module`, which is a base class for all neural network modules in PyTorch. This inheritance indicates that WeightsLoader is intended to be used as part of a neural network architecture. However, as it is currently implemented, the class does not contain any methods or attributes, which means it does not perform any specific functionality or hold any data. 

In a typical implementation, one would expect the WeightsLoader class to include methods for loading weights from a file or a specific source, initializing weights for a model, or managing the transfer of weights between different layers of a neural network. The absence of such methods suggests that the class is either a placeholder for future development or is intended to be extended by other classes that will provide the necessary functionality.

**Note**: It is important to recognize that the WeightsLoader class, as it stands, does not provide any operational capabilities. Developers intending to use this class should be aware that they will need to implement additional methods and attributes to fulfill the intended purpose of loading and managing weights in a neural network context.
***
## ClassDef T2IAdapter
**T2IAdapter**: The function of T2IAdapter is to adapt a text-to-image model with control hints for image generation processes.

**attributes**: The attributes of this Class.
· t2i_model: The text-to-image model that the adapter is designed to work with.
· channels_in: The number of input channels for the model.
· control_input: A variable that holds the processed control input for the model.

**Code Description**: The T2IAdapter class extends the ControlBase class, inheriting its functionality for managing control hints and their processing. It is specifically tailored for adapting text-to-image models, allowing for the integration of control hints during the image generation process.

The constructor `__init__` initializes the T2IAdapter with a specified text-to-image model (`t2i_model`), the number of input channels (`channels_in`), and an optional device parameter that determines where computations will be performed (CPU or GPU). The constructor also initializes the `control_input` attribute to None, which will later hold the processed control input.

The method `scale_image_to` adjusts the dimensions of an image to ensure they are compatible with the model's requirements. It calculates the width and height based on the model's `unshuffle_amount`, ensuring that the dimensions are multiples of this value.

The `get_control` method is a critical function that processes the noisy input image (`x_noisy`), the current timestep (`t`), and any conditional hints (`cond`). It first checks if there is a previous control network and retrieves its control if applicable. It then verifies if the current timestep is within the defined range. If the conditional hints are not set or do not match the dimensions of the noisy input, it recalculates the conditional hints, ensuring they are appropriately sized for processing.

If the `control_input` is None, the method prepares the text-to-image model for processing by moving it to the appropriate data type and device, then generates the control input using the model. The method finally returns a merged output of the control inputs and any previous control states, ensuring that the output is structured correctly for further processing.

The `copy` method creates a duplicate of the T2IAdapter instance, allowing for the reuse of the model and its parameters in different contexts.

The T2IAdapter is called by the `load_t2i_adapter` function, which initializes an instance of T2IAdapter based on the provided model data. This function checks for the presence of an adapter in the input data, processes the model weights, and constructs the appropriate model before returning an instance of T2IAdapter. This establishes a direct relationship between the loading mechanism and the T2IAdapter, ensuring that the adapter is correctly configured with the necessary model parameters.

**Note**: When using the T2IAdapter, it is essential to ensure that the conditional hints are correctly sized and compatible with the input data dimensions. Proper handling of the device and data types is crucial for optimal performance during image generation.

**Output Example**: A possible output from the `get_control` method could look like this:
{
  'input': [tensor1, tensor2],
  'middle': [tensor3],
  'output': [tensor4, tensor5]
}
Where `tensor1`, `tensor2`, etc., are processed tensors representing the control inputs and outputs, structured for further use in the image generation pipeline.
### FunctionDef __init__(self, t2i_model, channels_in, device)
**__init__**: The function of __init__ is to initialize an instance of the class with specified parameters.

**parameters**: The parameters of this Function.
· t2i_model: This parameter represents the model used for text-to-image conversion. It is expected to be an object that handles the specific operations related to this functionality.  
· channels_in: This parameter indicates the number of input channels that the model will process. It is typically an integer value that defines the dimensionality of the input data.  
· device: This optional parameter specifies the device on which the model will run, such as a CPU or GPU. If not provided, it defaults to None.

**Code Description**: The __init__ function serves as the constructor for the class, ensuring that when an instance is created, it is properly set up with the necessary attributes. The function first calls the constructor of the parent class using `super().__init__(device)`, which initializes the base class with the specified device. This is crucial for ensuring that the instance can leverage any inherited functionality related to device management. Following this, the function assigns the provided `t2i_model` to the instance variable `self.t2i_model`, allowing the instance to utilize this model for its operations. The `channels_in` parameter is similarly stored in `self.channels_in`, which will be used later in the processing of input data. Additionally, the `control_input` attribute is initialized to None, indicating that it will be set later in the workflow, but is currently not holding any value.

**Note**: It is important to ensure that the `t2i_model` provided is compatible with the expected input channels defined by `channels_in`. Additionally, if the `device` parameter is not specified, the instance will default to a None value, which may affect performance if the model is intended to run on a specific hardware accelerator.
***
### FunctionDef scale_image_to(self, width, height)
**scale_image_to**: The function of scale_image_to is to adjust the dimensions of an image to the nearest multiple of a specified unshuffle amount.

**parameters**: The parameters of this Function.
· parameter1: width - The desired width of the image to be scaled.
· parameter2: height - The desired height of the image to be scaled.

**Code Description**: The scale_image_to function takes two parameters, width and height, which represent the dimensions of an image. It retrieves the unshuffle amount from the t2i_model associated with the instance of the class. The function then calculates the new width and height by rounding each dimension up to the nearest multiple of the unshuffle amount using the math.ceil function. This ensures that the resulting dimensions are compatible with the model's requirements for processing images. The function returns the adjusted width and height as a tuple.

This function is called within the get_control method of the T2IAdapter class. In get_control, the scale_image_to function is utilized to determine the appropriate dimensions for the cond_hint based on the shape of the x_noisy input tensor. The dimensions are scaled to ensure that they align with the model's expectations, particularly when the cond_hint is either None or does not match the expected dimensions derived from x_noisy. This scaling is critical for maintaining the integrity of the image processing pipeline, as it prevents dimension mismatches that could lead to errors during model inference.

**Note**: It is important to ensure that the width and height passed to the scale_image_to function are in a format that is compatible with the model's requirements. The function assumes that the unshuffle amount is a positive integer.

**Output Example**: For example, if the input width is 150 and the height is 200, and the unshuffle amount is 32, the function would return (160, 224) after rounding both dimensions to the nearest multiples of 32.
***
### FunctionDef get_control(self, x_noisy, t, cond, batched_number)
**get_control**: The function of get_control is to generate control signals based on noisy input, time step, conditional hints, and batch size.

**parameters**: The parameters of this Function.
· x_noisy: A tensor representing the noisy input data, typically shaped as (batch_size, channels, height, width).
· t: A tensor indicating the current time step(s) for processing.
· cond: A tensor that provides conditional hints to guide the control generation.
· batched_number: An integer that specifies how many times the tensor should be batched.

**Code Description**: The get_control function is a critical component of the T2IAdapter class, designed to produce control signals that influence the model's output based on various inputs. The function begins by checking if there is a previous control state available through the previous_controlnet attribute. If it exists, it recursively calls the get_control method of the previous control network to obtain the prior control signals.

Next, the function evaluates the timestep range defined in the class. If the current time step (t) falls outside the specified range, it returns the previous control signals if available; otherwise, it returns None. This mechanism ensures that control generation is only performed within valid time steps, maintaining the integrity of the processing pipeline.

The function then checks the compatibility of the conditional hints (cond_hint) with the noisy input (x_noisy). If the dimensions of cond_hint do not match the expected dimensions derived from x_noisy, it deletes the existing cond_hint and resets the control input. It subsequently calculates the appropriate dimensions for cond_hint by calling the scale_image_to function, which adjusts the dimensions to the nearest multiples of a specified unshuffle amount.

Once the dimensions are set, the function utilizes the common_upscale function to upscale the original conditional hints to the newly calculated dimensions. This ensures that the conditional hints are appropriately sized for processing. If the batch size of the noisy input does not match that of cond_hint, the broadcast_image_to function is called to adjust the size of cond_hint to match the batch size of x_noisy.

If the control input has not been initialized, the function prepares the t2i_model for processing by moving it to the appropriate data type and device. It then generates the control input by passing the conditional hints through the t2i_model. After processing, the model is returned to the CPU to free up resources.

Finally, the function prepares the control input for merging. It checks if the model is in a specific state (xl) and adjusts the control input accordingly. The control_merge function is then called to combine the generated control inputs, any middle control states, and the previous control signals, ensuring that the output is coherent and aligned with the model's requirements.

This function is integral to the control mechanisms of the T2IAdapter, allowing for dynamic adjustments based on noisy inputs and conditions, thereby enhancing the model's ability to produce desired outputs.

**Note**: It is essential to ensure that the input tensors (x_noisy, cond) are correctly shaped and compatible before invoking this function. Mismatched dimensions may lead to runtime errors or unintended behavior during processing.

**Output Example**: A possible appearance of the code's return value could be:
{
    'input': [tensor1, tensor2, ...],
    'middle': [tensor3],
    'output': [tensor4, tensor5, ...]
}
Where tensor1, tensor2, etc., are the processed control tensors after applying the necessary transformations.
***
### FunctionDef copy(self)
**copy**: The function of copy is to create a new instance of the T2IAdapter class and transfer relevant attributes from the current instance to the new instance.

**parameters**: The parameters of this Function.
· parameter1: None

**Code Description**: The copy function is responsible for instantiating a new T2IAdapter object using the current instance's t2i_model and channels_in attributes. After creating this new instance, the function invokes the copy_to method, which is defined in the ControlBase class, to transfer specific attributes from the current instance to the newly created T2IAdapter instance. The attributes that are copied include cond_hint_original, strength, timestep_percent_range, and global_average_pooling, ensuring that the new instance retains the necessary configuration and state from the original instance.

This function is called within the apply_controlnet function found in the ControlNetApplyAdvanced class. In this context, apply_controlnet utilizes the copy function to create a new control net instance that is configured based on the current state of the control net being applied. The apply_controlnet function manages the application of control nets to images, and it requires the ability to create copies of control nets with specific configurations. By calling the copy function, it ensures that each control net applied retains the relevant attributes from the original instance, allowing for consistent behavior across different instances of control nets.

**Note**: It is essential to ensure that the attributes being copied are compatible with the target instance. Any discrepancies in attribute names or types may lead to errors during the copying process.

**Output Example**: A possible appearance of the code's return value could be an instance of T2IAdapter with attributes set as follows:
```
T2IAdapter(
    t2i_model=<model_instance>,
    channels_in=<number_of_channels>,
    cond_hint_original=<original_condition_hint>,
    strength=<strength_value>,
    timestep_percent_range=<time_step_range>,
    global_average_pooling=<pooling_value>
)
```
***
## FunctionDef load_t2i_adapter(t2i_data)
**load_t2i_adapter**: The function of load_t2i_adapter is to load and configure a text-to-image adapter model based on the provided state dictionary.

**parameters**: The parameters of this Function.
· t2i_data: A dictionary containing the model weights and configuration data for the text-to-image adapter.

**Code Description**: The load_t2i_adapter function is responsible for initializing and configuring a text-to-image adapter model based on the input state dictionary (t2i_data). The function first checks if the 'adapter' key exists in the provided t2i_data. If it does, it updates t2i_data to reference the contents of this 'adapter' key. 

Next, the function checks for specific keys in the t2i_data to determine the format of the model weights. If the keys indicate a diffusers format, it creates a mapping (prefix_replace) to adjust the keys in the state dictionary to match the expected format for the model. This is done using the state_dict_prefix_replace utility function, which replaces prefixes in the keys of the state dictionary.

The function then examines the keys in the t2i_data to determine the number of input channels (cin) based on the presence of certain weight keys. If the key "body.0.in_conv.weight" is found, it initializes an instance of the Adapter_light class with the appropriate input channels and configuration. If the key "conv_in.weight" is found instead, it initializes an instance of the Adapter class, taking into account additional parameters such as kernel size (ksize) and whether to use convolutional layers.

After initializing the model, the function attempts to load the state dictionary into the model instance. It captures any missing or unexpected keys during this process and prints warnings if any discrepancies are found. Finally, the function returns an instance of the T2IAdapter class, which is constructed using the initialized model and the number of input channels.

The load_t2i_adapter function is called by the load_controlnet function, which is responsible for loading control network data from a checkpoint file. If the control network data does not contain the necessary keys for direct loading, it invokes load_t2i_adapter to handle the loading of the text-to-image adapter model. This establishes a clear relationship between the loading mechanism for control networks and the text-to-image adapter configuration.

**Note**: When using the load_t2i_adapter function, ensure that the input data (t2i_data) is structured correctly and contains the necessary keys for successful model initialization. Proper handling of the state dictionary is crucial to avoid runtime errors and ensure the model is configured as intended.

**Output Example**: A possible output from the load_t2i_adapter function could be an instance of T2IAdapter initialized with the loaded model and input channels, ready for use in text-to-image generation tasks. For example:
```
T2IAdapter(model_ad, model_ad.input_channels)
```
