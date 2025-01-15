## ClassDef MetadataScheme
**MetadataScheme**: The function of MetadataScheme is to define a set of constants representing different metadata schemes used in the application.

**attributes**: The attributes of this Class.
· FOOOCUS: Represents the 'fooocus' metadata scheme.
· A1111: Represents the 'a1111' metadata scheme.

**Code Description**: The MetadataScheme class is an enumeration that provides a structured way to define and manage different metadata schemes within the application. It inherits from the Enum class, which allows for the creation of enumerated constants. In this case, two constants are defined: FOOOCUS and A1111, each associated with a string value that represents the respective metadata scheme.

The MetadataScheme class is utilized in various parts of the project, particularly in the modules that handle metadata parsing and processing. For instance, in the AsyncTask class within the async_worker module, an instance of MetadataScheme is created based on the arguments passed to the constructor. This instance determines how metadata will be handled during the execution of tasks.

Additionally, the MetadataScheme is referenced in the get_scheme methods of different metadata parser classes, such as A1111MetadataParser and FooocusMetadataParser. These methods return the corresponding MetadataScheme constant, indicating which scheme is being used for parsing metadata. The get_metadata_parser function also utilizes the MetadataScheme to match the appropriate parser based on the provided scheme.

Furthermore, the read_info_from_image function retrieves metadata from image files and attempts to determine the appropriate MetadataScheme based on the extracted information. If the metadata scheme cannot be identified, it falls back to default values, ensuring that the application can handle various scenarios gracefully.

**Note**: When using the MetadataScheme class, it is important to ensure that the correct scheme is selected based on the context of the application. This will facilitate accurate metadata handling and processing throughout the system.
## ClassDef OutputFormat
**OutputFormat**: The function of OutputFormat is to define and manage the supported output formats for images.

**attributes**: The attributes of this Class.
· PNG: Represents the PNG image format.
· JPEG: Represents the JPEG image format.
· WEBP: Represents the WEBP image format.

**Code Description**: The OutputFormat class is an enumeration that defines three specific image output formats: PNG, JPEG, and WEBP. Each format is represented by a string value corresponding to its file extension. The class also includes a class method named `list`, which returns a list of the values of the defined formats. This method utilizes a lambda function to map over the enumeration members and extract their values, providing a convenient way to retrieve all supported formats in a list format.

The OutputFormat class is utilized in the `log` function found in the `modules/private_logger.py` file. Within this function, the output format is determined based on the provided `output_format` parameter. Depending on the value of this parameter, the function saves the image in the specified format using the appropriate settings. For instance, if the output format is set to `OutputFormat.PNG.value`, the image is saved as a PNG file, potentially with additional metadata embedded. Similarly, if the format is JPEG or WEBP, the image is saved accordingly with specific quality settings.

This relationship highlights the importance of the OutputFormat class in ensuring that images are saved in the correct format as specified by the user or the default configuration. It serves as a centralized definition of the available formats, promoting consistency and clarity throughout the image logging process.

**Note**: When using the OutputFormat class, ensure that the specified output format is one of the defined attributes (PNG, JPEG, WEBP) to avoid unexpected behavior in the image saving process.

**Output Example**: An example of the output from the `list` method would be:
```python
['png', 'jpeg', 'webp']
```
### FunctionDef list(cls)
**list**: The function of list is to return a list of values from the enumeration class.

**parameters**: The parameters of this Function.
· parameter1: cls - This parameter represents the enumeration class from which the values will be extracted.

**Code Description**: The list function is a class method that generates and returns a list containing the values of the enumeration members defined in the class. It utilizes the built-in `map` function in conjunction with a lambda function to iterate over each member of the enumeration class (`cls`). For each member, it retrieves the `value` attribute, which is assumed to be defined for each enumeration member. The result of the `map` function is then converted into a list before being returned.

This function is particularly useful in scenarios where a developer needs to obtain a simple list of values from an enumeration, which can be used for display purposes, validation, or other logic that requires the enumeration values in a list format.

In the context of the project, the list function may be called from other modules, such as `webui.py`. Although there is no specific documentation or raw code provided for the call in `webui.py`, it can be inferred that the function would be utilized to retrieve the values of an enumeration for rendering in a user interface or for processing input related to the enumeration.

**Note**: It is important to ensure that the enumeration class has a defined `value` attribute for each member; otherwise, the function may raise an AttributeError.

**Output Example**: If the enumeration class has members with values like `RED`, `GREEN`, and `BLUE`, the output of the list function would be: `['RED', 'GREEN', 'BLUE']`.
***
## ClassDef PerformanceLoRA
**PerformanceLoRA**: The function of PerformanceLoRA is to define various performance-related configurations for loading models in a machine learning context.

**attributes**: The attributes of this Class.
· QUALITY: Represents a quality performance setting, currently set to None.  
· SPEED: Represents a speed performance setting, currently set to None.  
· EXTREME_SPEED: Represents an extreme speed performance setting, associated with the filename 'sdxl_lcm_lora.safetensors'.  
· LIGHTNING: Represents a lightning performance setting, associated with the filename 'sdxl_lightning_4step_lora.safetensors'.  
· HYPER_SD: Represents a hyper speed performance setting, associated with the filename 'sdxl_hyper_sd_4step_lora.safetensors'.  

**Code Description**: The PerformanceLoRA class is an enumeration that categorizes different performance configurations for loading models, specifically in the context of deep learning or machine learning frameworks. Each attribute corresponds to a specific performance setting, with some attributes linked to specific model files that can be downloaded and utilized in the application.

The attributes QUALITY and SPEED are placeholders and currently do not hold any value, while EXTREME_SPEED, LIGHTNING, and HYPER_SD are associated with specific filenames that indicate the model files to be used for those performance settings. These filenames are critical for the downloading functions defined in the project, as they specify which model file to retrieve from a given URL.

The PerformanceLoRA class is utilized in several functions within the modules/config.py file, specifically in the downloading_sdxl_lcm_lora, downloading_sdxl_lightning_lora, and downloading_sdxl_hyper_sd_lora functions. Each of these functions calls the load_file_from_url function, passing the corresponding PerformanceLoRA attribute's value as the file_name parameter. This indicates that the model files associated with EXTREME_SPEED, LIGHTNING, and HYPER_SD are intended to be downloaded and used based on the performance configuration selected.

Additionally, the PerformanceLoRA class is referenced in the lora_filename method, which checks if the name of the current instance matches any of the members of the PerformanceLoRA enumeration and returns the associated value if it does. This method is likely used to dynamically retrieve the appropriate model filename based on the performance setting in use.

In the test suite, the PerformanceLoRA class is also referenced to validate the functionality of parsing and stripping performance-related tokens from prompts. This ensures that the correct model filenames are being processed according to the specified performance settings.

**Note**: It is important to ensure that the model files corresponding to the PerformanceLoRA attributes are available at the specified URLs for successful downloads. Additionally, developers should be aware of the current None values for QUALITY and SPEED, which may need to be defined or updated based on future requirements.
## ClassDef Steps
**Steps**: The function of Steps is to define a set of constants representing different performance levels.

**attributes**: The attributes of this Class.
· QUALITY: Represents a performance level with a value of 60.  
· SPEED: Represents a performance level with a value of 30.  
· EXTREME_SPEED: Represents a performance level with a value of 8.  
· LIGHTNING: Represents a performance level with a value of 4.  
· HYPER_SD: Represents a performance level with a value of 4.  

**Code Description**: The Steps class is an enumeration that inherits from IntEnum, allowing for the definition of symbolic names bound to unique, constant values. Each attribute in the Steps class corresponds to a specific performance level, facilitating the categorization of performance metrics in a clear and manageable way. 

The class also includes a class method `keys`, which returns a list of the names of the members of the Steps enumeration. This method utilizes the `__members__` attribute of the Steps class to extract the names of the defined performance levels, making it easier to access and utilize these names programmatically.

The Steps class is utilized in various parts of the project. For instance, in the `by_steps` method of the Performance module, the Steps class is referenced to convert a given step value (either an integer or string) into its corresponding enumeration member. This method ensures that the performance level can be accurately retrieved based on the provided input.

Additionally, in the `steps` method, the Steps class is used to determine the integer value associated with the current instance's name, checking if it exists within the defined members of the Steps enumeration. This allows for a straightforward way to retrieve performance values based on the enumeration.

In the `get_steps` function, the Steps class is leveraged to validate and match performance levels against a provided source dictionary. It checks if the performance name corresponds to any of the keys in the Steps enumeration and appends the appropriate results based on the matching criteria.

Lastly, in the `MetadataParser` class, the Steps class is employed to initialize the `steps` attribute with a default value of `SPEED`, establishing a baseline performance level for instances of the MetadataParser.

**Note**: It is important to ensure that any input provided to methods interacting with the Steps class is valid and corresponds to the defined performance levels to avoid errors in retrieval and processing.

**Output Example**: An example of the output when calling `Steps.keys()` would be: `['QUALITY', 'SPEED', 'EXTREME_SPEED', 'LIGHTNING', 'HYPER_SD']`.
### FunctionDef keys(cls)
**keys**: The function of keys is to return a list of all member names from the Steps enumeration.

**parameters**: The parameters of this Function.
· There are no parameters for this function.

**Code Description**: The keys function is a class method that retrieves all the member names of the Steps enumeration. It utilizes the `__members__` attribute of the Steps class, which contains a mapping of member names to their corresponding enumeration values. The function employs the `map` function along with a lambda function to iterate over the members, effectively creating a list of the member names. The `list` function then converts the map object into a list, which is returned as the output of the function. This method is particularly useful for obtaining a list of all available keys in the Steps enumeration, which can be utilized in various contexts, such as validation, display, or further processing.

**Note**: It is important to ensure that the Steps enumeration is defined and contains members before calling this function, as it relies on the existence of these members to generate the list.

**Output Example**: An example of the possible return value of the keys function could be:
```python
['STEP_ONE', 'STEP_TWO', 'STEP_THREE']
``` 
This output represents a list of member names from the Steps enumeration, assuming those members are defined within it.
***
## ClassDef StepsUOV
**StepsUOV**: The function of StepsUOV is to define a set of constants representing different performance levels.

**attributes**: The attributes of this Class.
· QUALITY: Represents a performance level with a value of 36.  
· SPEED: Represents a performance level with a value of 18.  
· EXTREME_SPEED: Represents a performance level with a value of 8.  
· LIGHTNING: Represents a performance level with a value of 4.  
· HYPER_SD: Represents a performance level with a value of 4.  

**Code Description**: The StepsUOV class is an enumeration that inherits from the IntEnum class, which allows for the creation of enumerated constants that can also be treated as integers. Each constant in the StepsUOV class corresponds to a specific performance level, with associated integer values that may be used in calculations or comparisons. The constants defined in this class are intended to provide a clear and organized way to reference different performance metrics within the application.

The StepsUOV class is utilized in the method `steps_uov` found in the Performance module. This method checks if the name of the current object (presumably an instance of a class that includes this method) corresponds to any of the defined members in the StepsUOV enumeration. If a match is found, it returns the integer value associated with that member; otherwise, it returns None. This functionality allows for dynamic retrieval of performance levels based on the object's name, facilitating the integration of performance metrics into the broader application logic.

**Note**: When using the StepsUOV class, it is important to ensure that the names used to access its members are valid and correspond to the defined constants. This will prevent unexpected None returns and ensure that the performance levels are accurately represented in the application.
## ClassDef Performance
**Performance**: The function of Performance is to define various performance modes and provide utility methods related to these modes.

**attributes**: The attributes of this Class.
· QUALITY: Represents the quality performance mode.
· SPEED: Represents the speed performance mode.
· EXTREME_SPEED: Represents the extreme speed performance mode.
· LIGHTNING: Represents the lightning performance mode.
· HYPER_SD: Represents the hyper SD performance mode.

**Code Description**: The Performance class is an enumeration that defines different performance modes for an application. Each mode is represented as a unique member of the enum, allowing for easy reference and comparison throughout the codebase. The class includes several class methods that provide utility functions related to the performance modes.

1. **list(cls)**: This class method returns a list of tuples, where each tuple contains the name and value of each performance mode. This is useful for displaying available performance options in a user interface or for logging purposes.

2. **values(cls)**: This class method returns a list of the values associated with each performance mode. This can be used when only the string representation of the performance modes is needed.

3. **by_steps(cls, steps)**: This class method takes an integer or string representing the number of steps and returns the corresponding performance mode based on predefined mappings. This allows for dynamic selection of performance modes based on user input or configuration settings.

4. **has_restricted_features(cls, x)**: This class method checks if a given performance mode (or its value) has restricted features. It returns a boolean indicating whether the mode is one of EXTREME_SPEED, LIGHTNING, or HYPER_SD, which are considered to have limitations in functionality.

5. **steps(self)**: This instance method returns the number of steps associated with the current performance mode. If the mode does not have a corresponding entry in the Steps enumeration, it returns None.

6. **steps_uov(self)**: This instance method returns the number of steps associated with the current performance mode in the context of upscaling or varying operations. Similar to the steps method, it returns None if there is no corresponding entry.

7. **lora_filename(self)**: This instance method returns the filename associated with the current performance mode's LoRA (Low-Rank Adaptation) model. If the mode does not have a corresponding entry in the PerformanceLoRA enumeration, it returns None.

The Performance class is utilized in various parts of the project, particularly in the AsyncTask class within the async_worker module. When initializing an AsyncTask, the performance mode is selected based on user input and is used to determine the processing parameters for image generation tasks. For example, the performance_selection attribute of AsyncTask is set using the Performance enum, and methods like steps() and steps_uov() are called to retrieve the appropriate step counts for the selected performance mode. This integration ensures that the application can dynamically adjust its processing behavior based on the chosen performance mode, optimizing for either quality or speed as required by the user.

**Note**: It is important to ensure that the performance mode selected is compatible with the intended processing tasks, as certain modes may impose restrictions on available features or processing capabilities.

**Output Example**: An example output for the list method might look like this:
[
    ('QUALITY', 'Quality'),
    ('SPEED', 'Speed'),
    ('EXTREME_SPEED', 'Extreme Speed'),
    ('LIGHTNING', 'Lightning'),
    ('HYPER_SD', 'Hyper-SD')
]
### FunctionDef list(cls)
**list**: The function of list is to return a list of tuples containing the names and values of the class members.

**parameters**: The parameters of this Function.
· cls: The class from which the members are being extracted.

**Code Description**: The list function is a class method that generates a list of tuples, where each tuple consists of the name and value of each member of the class it belongs to. It utilizes the built-in `map` function to iterate over the class members, applying a lambda function that extracts the `name` and `value` attributes of each member. The result of the `map` function is then converted into a list and returned.

This function is closely related to the values function defined in the same module. The values function also operates on the class members but focuses solely on returning a list of their values. While list provides a more comprehensive view by including both names and values, values simplifies the output to just the values. This relationship indicates that both functions serve to expose different aspects of the class members, allowing for flexible usage depending on the needs of the caller.

**Note**: When using this function, it is important to ensure that the class members have both `name` and `value` attributes defined, as the function relies on these attributes to construct the output.

**Output Example**: An example of the possible return value of the list function could be:
```
[('Member1', 1), ('Member2', 2), ('Member3', 3)]
``` 
This output represents a list of tuples where each tuple corresponds to a member of the class, with the first element being the member's name and the second element being its value.
***
### FunctionDef values(cls)
**values**: The function of values is to return a list of the values of the class members.

**parameters**: The parameters of this Function.
· cls: The class from which the member values are being extracted.

**Code Description**: The values function is a class method that generates a list containing only the values of the class members. It utilizes the built-in `map` function to iterate over the class members, applying a lambda function that extracts the `value` attribute of each member. The result of the `map` function is then converted into a list and returned.

This function is closely related to the list function defined in the same module. While the list function provides a comprehensive view by returning both the names and values of the class members, the values function simplifies the output to include only the values. This relationship indicates that both functions serve to expose different aspects of the class members, allowing for flexible usage depending on the needs of the caller.

The values function is called within the load_parameter_button_click function in the modules/meta_parser.py file. In this context, it is used to check if a specified performance value exists within the defined class members of Performance. This ensures that the performance value being processed is valid and corresponds to an existing member of the Performance class.

**Note**: When using this function, it is important to ensure that the class members have a `value` attribute defined, as the function relies on this attribute to construct the output.

**Output Example**: An example of the possible return value of the values function could be:
```
[1, 2, 3]
```
This output represents a list of values corresponding to the members of the class, where each element is the value of a member.
***
### FunctionDef by_steps(cls, steps)
**by_steps**: The function of by_steps is to retrieve a performance level based on a specified step value.

**parameters**: The parameters of this Function.
· steps: An integer or string representing the step value for which the corresponding performance level is to be retrieved.

**Code Description**: The by_steps function is a class method that takes a parameter named steps, which can be either an integer or a string. This method converts the steps parameter into an integer and then retrieves the corresponding performance level from the Steps enumeration class. The conversion is done using the Steps class, which defines a set of constants representing different performance levels, such as QUALITY, SPEED, EXTREME_SPEED, LIGHTNING, and HYPER_SD. Each of these constants is associated with a unique integer value.

The by_steps function is utilized within the to_json method of the A1111MetadataParser class. Specifically, it is called when the to_json method processes the input metadata to extract performance information. If the 'steps' key is present in the data dictionary and the 'performance' key is not already set, the by_steps function is invoked to determine the performance level based on the provided steps value. If the value is valid and corresponds to one of the defined performance levels, the method assigns the corresponding performance value to the 'performance' key in the data dictionary.

This integration ensures that the performance level is accurately retrieved and included in the final output of the to_json method, which constructs a structured representation of the metadata.

**Note**: It is essential to provide valid input for the steps parameter to ensure that the corresponding performance level can be retrieved without errors. Invalid inputs may lead to exceptions being raised during the retrieval process.

**Output Example**: An example of the output when calling by_steps with a valid input could be: `30`, which corresponds to the SPEED performance level when steps is set to 'SPEED'.
***
### FunctionDef has_restricted_features(cls, x)
**has_restricted_features**: The function of has_restricted_features is to determine if a given Performance object or its value is classified as having restricted features.

**parameters**: The parameters of this Function.
· cls: The class reference, typically used to access class-level attributes.
· x: An instance of Performance or a value that needs to be checked.

**Code Description**: The has_restricted_features function checks if the input parameter x is an instance of the Performance class. If it is, the function retrieves the value associated with that instance. The function then checks if this value is one of the predefined restricted features, which are EXTREME_SPEED, LIGHTNING, or HYPER_SD. These restricted features are represented by their respective values, which are accessed through the class reference cls. The function returns a boolean value: True if x corresponds to one of the restricted features, and False otherwise.

In the context of the project, this function may be called from other modules, such as webui.py. Although there is no direct documentation or code snippet provided for the call in webui.py, it can be inferred that this function is likely used to enforce certain restrictions or conditions based on the performance characteristics of an object. This could be important for ensuring that only specific performance levels are allowed in certain operations or configurations within the web interface.

**Note**: It is important to ensure that the input parameter x is either a Performance object or a valid value that can be checked against the restricted features. Passing an invalid type may lead to unexpected behavior or errors.

**Output Example**: If the input x is an instance of Performance with a value of EXTREME_SPEED, the function would return True. If x is a value like NORMAL_SPEED, the function would return False.
***
### FunctionDef steps(self)
**steps**: The function of steps is to retrieve the integer value associated with the performance level defined by the current instance's name.

**parameters**: The parameters of this Function.
· None

**Code Description**: The steps function is a method that returns an integer value corresponding to the performance level of the current instance, as defined in the Steps enumeration. It checks if the name of the current instance exists within the members of the Steps enumeration. If the name is found, it retrieves the associated value from the Steps enumeration; otherwise, it returns None. This functionality is essential for determining the performance metrics of an instance based on its name, allowing for a clear and manageable way to access performance levels.

The steps method is called within the AsyncTask class during its initialization process. Specifically, when an AsyncTask instance is created, the performance_selection attribute is set based on the provided arguments, and the steps method is invoked to obtain the corresponding performance value. This value is then stored in the steps attribute of the AsyncTask instance, which is used later in the processing workflow to determine how many steps are required for various tasks, such as image processing and enhancement.

The relationship between the steps method and its callers is crucial for the overall functionality of the AsyncTask class. By leveraging the Steps enumeration, the steps method ensures that the performance levels are consistently applied throughout the task execution, contributing to the efficiency and effectiveness of the processing operations.

**Note**: It is important to ensure that the name assigned to the instance corresponds to one of the defined performance levels in the Steps enumeration to avoid returning None, which could lead to unexpected behavior in the processing logic.

**Output Example**: An example of the output when calling steps on an instance with the name "SPEED" would be: `30`. If the instance name is not found in the Steps enumeration, the output would be: `None`.
***
### FunctionDef steps_uov(self)
**steps_uov**: The function of steps_uov is to retrieve the integer value associated with the performance level defined by the name of the current object, if it exists within the StepsUOV enumeration.

**parameters**: The parameters of this Function.
· self: The instance of the class that contains the steps_uov method.

**Code Description**: The steps_uov method is designed to return an integer value that corresponds to a performance level defined in the StepsUOV enumeration. It first checks if the name attribute of the current object (self.name) exists as a member within the StepsUOV enumeration. If a match is found, it retrieves the value associated with that member using `StepsUOV[self.name].value`. If the name does not correspond to any member in the enumeration, the method returns None. This functionality allows for dynamic access to performance metrics based on the object's name, facilitating integration with other components of the application.

The steps_uov method is called within the prepare_upscale function located in the async_worker module. In this context, it is used to determine the number of steps for an upscale operation based on the performance level selected by the user. Specifically, if the uov_method includes 'upscale' and does not include 'fast', the method retrieves the performance steps using performance.steps_uov(). This value is then utilized to control the number of processing steps in the upscale operation, ensuring that the operation aligns with the desired performance level.

**Note**: When using the steps_uov method, it is essential to ensure that the name attribute of the object corresponds to a valid member of the StepsUOV enumeration. Failure to do so will result in a return value of None, which may affect the execution of related processes that depend on this value.

**Output Example**: If the name of the current object is "QUALITY", the method would return 36, as defined in the StepsUOV enumeration. If the name is "UNKNOWN", the method would return None.
***
### FunctionDef lora_filename(self)
**lora_filename**: The function of lora_filename is to retrieve the associated filename for a specific performance setting if it exists within the PerformanceLoRA enumeration.

**parameters**: The parameters of this Function.
· self: The instance of the class that contains the lora_filename method.

**Code Description**: The lora_filename method checks if the name attribute of the current instance is a member of the PerformanceLoRA enumeration. If the name exists within the enumeration, it returns the corresponding value, which is the filename associated with that performance setting. If the name does not match any member of the PerformanceLoRA, the method returns None. This functionality is crucial for dynamically determining the appropriate model filename based on the performance configuration selected by the user.

The lora_filename method is called within the load_parameter_button_click function, which processes user input and metadata. Specifically, it checks if the performance setting provided in the input is valid and retrieves the corresponding filename using lora_filename. This filename is then used to prevent duplicate entries of performance LoRAs when loading multiple LoRA configurations. 

Additionally, the lora_filename method is referenced in the remove_performance_lora function, which removes any filenames from a list that match the performance LoRA filename. This ensures that the filenames associated with the specified performance setting are not included in the final list of filenames, maintaining the integrity of the performance configurations.

**Note**: It is essential to ensure that the name attribute of the instance accurately reflects one of the members of the PerformanceLoRA enumeration to retrieve the correct filename. If the name does not match any member, the method will return None, which may affect the functionality of the calling methods that rely on this filename.

**Output Example**: If the name attribute of the instance is set to "EXTREME_SPEED", the lora_filename method would return 'sdxl_lcm_lora.safetensors'. If the name is set to "UNKNOWN", it would return None.
***
