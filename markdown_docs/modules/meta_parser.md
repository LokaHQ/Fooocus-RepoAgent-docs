## FunctionDef load_parameter_button_click(raw_metadata, is_generating, inpaint_mode)
**load_parameter_button_click**: The function of load_parameter_button_click is to process metadata, either in the form of a dictionary or a JSON string, and extract various parameters to update the results list accordingly.

**parameters**: The parameters of this Function.
· raw_metadata: dict | str - The raw metadata input, which can be either a dictionary or a JSON string representing the parameters to be loaded.
· is_generating: bool - A flag indicating whether the process is currently generating outputs, affecting how the results are updated.
· inpaint_mode: str - A string representing the mode of inpainting, which influences the retrieval of specific parameters.

**Code Description**: The load_parameter_button_click function begins by checking the type of raw_metadata. If it is a string, it attempts to parse it into a dictionary using json.loads. The function then asserts that the resulting loaded_parameter_dict is indeed a dictionary. 

Next, it initializes a results list that will track the success of various parameter retrievals. The function proceeds to call several helper functions to extract specific parameters from the loaded metadata. These helper functions include get_image_number, get_str, get_list, get_number, get_resolution, get_seed, get_inpaint_engine_version, get_inpaint_method, get_adm_guidance, and get_freeu. Each of these functions retrieves a specific type of data (e.g., strings, numbers, lists) and appends the results to the results list.

The function also handles the performance parameter by checking if it exists in the Performance enumeration. If valid, it retrieves the associated filename to prevent duplicate entries when processing LoRA configurations. The function iterates through a predefined maximum number of LoRAs, calling get_lora to extract LoRA-related information.

Finally, based on the is_generating flag, the function updates the results list to indicate whether the UI should reflect the current state of the parameters. The results list is then returned, containing all the processed parameters and their corresponding states.

This function is called in various contexts within the project, such as in the preset_selection_change function, where it processes preset metadata, and in the trigger_metadata_import function, which handles metadata extracted from images. In both cases, load_parameter_button_click serves as a central point for loading and processing parameters, ensuring that the application can dynamically adjust its behavior based on user input or metadata.

**Note**: It is essential to ensure that the raw_metadata input is correctly formatted as either a dictionary or a JSON string to avoid parsing errors. Additionally, the helper functions called within load_parameter_button_click should be properly defined to handle the expected keys and data types in the metadata.

**Output Example**: A possible return value of the load_parameter_button_click function could be a list such as:
```
[True, 'Sample Prompt', 'Sample Negative Prompt', ['Style1', 'Style2'], 30, 10, 1.5, 'ModelName', 'SamplerName', 'SchedulerName', 12345, 'v1.2.3', 'method_a', True, 0.5, 0.5, 0.5, 0.5, True, 1.0, 'LoRA1', 0.8, 'LoRA2', 0.6]
```
This output represents the successful extraction and processing of various parameters from the input metadata.
## FunctionDef get_str(key, fallback, source_dict, results, default)
**get_str**: The function of get_str is to retrieve a string value from a specified dictionary using a primary key and a fallback key, appending the result to a list of results.

**parameters**: The parameters of this Function.
· parameter1: key (str) - The primary key to look for in the source dictionary.
· parameter2: fallback (str | None) - The fallback key to use if the primary key is not found.
· parameter3: source_dict (dict) - The dictionary from which to retrieve the value.
· parameter4: results (list) - A list to which the retrieved value or an update will be appended.
· parameter5: default (optional) - A default value to return if neither the key nor the fallback is found.

**Code Description**: The get_str function attempts to fetch a string value from the source_dict using the provided key. If the key does not exist, it tries to retrieve the value using the fallback key. If neither key is found, it returns a default value if provided. The function ensures that the retrieved value is of type string, asserting this condition. If the value is successfully retrieved, it appends this value to the results list and returns it. In case of any exceptions during this process, the function appends a generic update to the results list and returns None. 

This function is called within the load_parameter_button_click function, which processes metadata that can either be a dictionary or a JSON string. The load_parameter_button_click function prepares a results list and populates it with various parameters extracted from the loaded metadata. The get_str function is specifically used to extract string parameters such as 'prompt', 'negative_prompt', 'performance', and others. The results list is then used to manage the state of the user interface, indicating whether the parameters were successfully loaded or if updates are needed.

**Note**: It is important to ensure that the source_dict contains the expected keys to avoid unnecessary exceptions. The function is designed to handle missing keys gracefully by utilizing a fallback mechanism.

**Output Example**: A possible return value of the get_str function could be "This is a prompt." if the key 'prompt' exists in the source_dict, or None if both the key and fallback are not found.
## FunctionDef get_list(key, fallback, source_dict, results, default)
**get_list**: The function of get_list is to retrieve a list from a source dictionary based on a specified key or a fallback key, and append it to a results list.

**parameters**: The parameters of this Function.
· key: A string representing the primary key to look for in the source dictionary.
· fallback: A string or None that serves as an alternative key if the primary key is not found.
· source_dict: A dictionary from which the value is retrieved.
· results: A list where the retrieved value will be appended.
· default: An optional parameter that defines a default value to use if neither the key nor the fallback is found.

**Code Description**: The get_list function attempts to retrieve a value from the source_dict using the provided key. If the key does not exist, it attempts to retrieve the value using the fallback key. If neither key is found, it uses the default value if provided. The retrieved value is then evaluated using the eval function, which expects the value to be a string representation of a list. An assertion checks that the evaluated result is indeed a list. If the assertion passes, the list is appended to the results list. In case of any exceptions during this process, such as a missing key or an evaluation error, the function appends a default update to the results list.

This function is called within the load_parameter_button_click function, which processes a metadata dictionary or string. The get_list function specifically retrieves a list of styles from the loaded parameters. It plays a crucial role in ensuring that the styles are correctly fetched and included in the results, which are ultimately used for further processing or display. The integration of get_list within load_parameter_button_click highlights its importance in handling user-defined parameters effectively.

**Note**: It is important to ensure that the value associated with the key or fallback is a valid string representation of a list to avoid evaluation errors. Additionally, the use of eval should be approached with caution, as it can execute arbitrary code if the input is not controlled.
## FunctionDef get_number(key, fallback, source_dict, results, default, cast_type)
**get_number**: The function of get_number is to retrieve a numeric value from a given dictionary based on a specified key or a fallback key, apply a type cast to it, and append the result to a list.

**parameters**: The parameters of this Function.
· key: A string representing the primary key to look for in the source dictionary.
· fallback: A string or None that serves as a secondary key to look for if the primary key is not found.
· source_dict: A dictionary from which the value is retrieved.
· results: A list where the retrieved and casted value will be appended.
· default: An optional value that will be used if neither the key nor the fallback is found in the source dictionary.
· cast_type: A type that defines how the retrieved value should be cast, defaulting to float.

**Code Description**: The get_number function attempts to retrieve a value from the source_dict using the provided key. If the key does not exist, it tries to retrieve the value using the fallback key. If neither key is found, it uses the default value if provided. The retrieved value is then asserted to be non-null and is cast to the specified cast_type (which defaults to float). The casted value is appended to the results list. In case of any exceptions during this process, an update is appended to the results list instead.

This function is called within the load_parameter_button_click function, which processes a dictionary of parameters (raw_metadata). The load_parameter_button_click function first checks if the input is a string and converts it to a dictionary if necessary. It then initializes a results list and populates it with various parameters by calling several helper functions, including get_number. The get_number function is specifically used to retrieve numeric parameters such as 'overwrite_switch', 'guidance_scale', 'sharpness', 'adaptive_cfg', 'clip_skip', and 'refiner_switch'. Each of these calls contributes to building a comprehensive list of results based on the provided metadata.

**Note**: It is important to ensure that the source_dict contains the expected keys and that the values are convertible to the specified cast_type to avoid exceptions. Additionally, the function's behavior when encountering exceptions is to append a generic update to the results, which may need to be handled appropriately in the context of the application.
## FunctionDef get_image_number(key, fallback, source_dict, results, default)
**get_image_number**: The function of get_image_number is to retrieve and validate the number of images specified in a source dictionary, ensuring it adheres to a defined maximum limit, and appending the result to a provided list.

**parameters**: The parameters of this Function.
· key: A string representing the primary key to look for in the source dictionary.
· fallback: A string or None that serves as an alternative key if the primary key is not found.
· source_dict: A dictionary from which the image number is retrieved.
· results: A list where the retrieved image number will be appended.
· default: An optional value that serves as a fallback if neither the key nor the fallback is found.

**Code Description**: The get_image_number function attempts to extract an image number from the source_dict using the specified key. If the key is not present, it looks for the fallback key. If neither key is found, it uses the default value provided. The function asserts that the retrieved value is not None, ensuring that a valid number is obtained. It then converts this value to an integer and applies a minimum constraint to ensure it does not exceed a predefined maximum limit, defined as modules.config.default_max_image_number. The final validated image number is appended to the results list. In the event of any exceptions during this process, such as a failure to convert the value to an integer, the function appends a default value of 1 to the results list.

This function is called within the load_parameter_button_click function, which processes a dictionary of parameters (raw_metadata) that may be in string format or already as a dictionary. The load_parameter_button_click function first ensures that the raw_metadata is a dictionary and initializes a results list to track the outcomes of various parameter retrievals. The get_image_number function is specifically called to retrieve the 'image_number' from the loaded parameters, contributing to the overall results list that is returned at the end of the load_parameter_button_click function. This indicates that get_image_number plays a crucial role in validating and ensuring the integrity of the image number parameter within the broader context of loading and processing parameters for further operations.

**Note**: It is important to ensure that the source_dict contains valid keys as expected by the get_image_number function. Additionally, users should be aware that if both the key and fallback are missing, the function will default to appending the value 1 to the results list, which may not reflect the intended behavior if not properly handled.
## FunctionDef get_steps(key, fallback, source_dict, results, default)
**get_steps**: The function of get_steps is to retrieve and validate performance step values from a source dictionary based on specified keys.

**parameters**: The parameters of this Function.
· key: A string representing the primary key to look up in the source dictionary.  
· fallback: A string or None, representing an alternative key to use if the primary key is not found.  
· source_dict: A dictionary containing the data from which to retrieve the performance step value.  
· results: A list that will be populated with the result of the function's execution.  
· default: An optional parameter that specifies a default value to return if both the key and fallback are not found.

**Code Description**: The get_steps function attempts to extract a performance step value from the provided source_dict using the specified key and fallback. It first retrieves the value associated with the key; if that is not found, it attempts to retrieve the value associated with the fallback key. If neither key yields a value, the function uses the provided default value.

Once a value is retrieved, the function asserts that it is not None and converts it to an integer. It then checks if the performance name, derived from the source_dict, matches any keys in the Steps enumeration and whether the corresponding value matches the retrieved integer. If no matching performance candidates are found, the integer value is appended to the results list. If a match is found or an error occurs during the process, -1 is appended to the results list instead.

The get_steps function is called within the load_parameter_button_click function, which processes a dictionary of parameters (raw_metadata). This function is responsible for loading various parameters, including performance steps, and populating the results list with the outcomes of each parameter retrieval. The results list is then used to update the user interface based on the success or failure of these operations.

**Note**: It is important to ensure that the source_dict contains valid keys and values that correspond to the expected performance levels defined in the Steps enumeration to avoid errors in retrieval and processing.

**Output Example**: A possible output when calling get_steps with a valid key and no matching performance candidates might be: `[30]`, assuming the integer value associated with 'steps' is 30 and there are no performance matches. If an error occurs or a match is found, the output would be `[-1]`.
## FunctionDef get_resolution(key, fallback, source_dict, results, default)
**get_resolution**: The function of get_resolution is to retrieve and format the resolution specified by a key from a source dictionary, handling potential fallbacks and updating a results list accordingly.

**parameters**: The parameters of this Function.
· key: A string representing the key used to look up the resolution in the source dictionary.  
· fallback: A string or None that serves as an alternative key if the primary key is not found.  
· source_dict: A dictionary containing resolution values associated with various keys.  
· results: A list that will be updated with the formatted resolution and other relevant values.  
· default: An optional parameter that specifies a default value to return if neither the key nor the fallback is found.

**Code Description**: The get_resolution function attempts to retrieve a resolution value from the source_dict using the provided key. If the key does not exist, it attempts to use the fallback key. If neither key is found, it defaults to a specified value. The retrieved resolution is expected to be in a format that can be evaluated to yield width and height values. 

The function uses the eval function to parse the resolution string into width and height integers. It then calls the add_ratio function from the modules.config module to format the width and height into a string representation of the aspect ratio. This formatted ratio is checked against a list of available aspect ratios. If the formatted ratio exists in the available aspect ratios, it appends the formatted ratio along with two placeholder values (-1) to the results list. If the ratio does not exist, it appends updates to the results list based on the width and height values.

In the context of the project, get_resolution is called by the load_parameter_button_click function, which is responsible for processing various parameters from a metadata dictionary. Within this function, get_resolution is specifically invoked to handle the 'resolution' parameter, ensuring that the resolution is correctly retrieved and formatted before being added to the results list. This indicates that get_resolution plays a critical role in managing resolution data as part of a broader parameter loading process.

**Note**: It is important to ensure that the resolution string retrieved from the source dictionary is in the correct format that can be evaluated. Any deviation from the expected format may lead to errors during execution. Additionally, the use of eval should be approached with caution, as it can execute arbitrary code if the input is not properly controlled.
## FunctionDef get_seed(key, fallback, source_dict, results, default)
**get_seed**: The function of get_seed is to retrieve a seed value from a source dictionary based on a specified key and a fallback option, and append the result to a provided results list.

**parameters**: The parameters of this Function.
· key: A string representing the primary key to look up in the source dictionary.
· fallback: A string or None that serves as a secondary key to use if the primary key is not found.
· source_dict: A dictionary from which the seed value is to be retrieved.
· results: A list that will store the outcome of the function's operation.
· default: An optional value that will be used if neither the key nor the fallback is found in the source dictionary.

**Code Description**: The get_seed function attempts to retrieve a value associated with the specified key from the source_dict. If the key is not present, it will try to find the value using the fallback key. If neither key is found, the function will use the default value provided. The retrieved value is then converted to an integer and appended to the results list. If the retrieval process encounters any exceptions, the function appends two update calls to the results list, indicating an error or a need for further action.

This function is called within the load_parameter_button_click function, which processes a metadata dictionary or string to extract various parameters. The get_seed function specifically retrieves the 'seed' value from the loaded parameter dictionary. This value is essential for ensuring reproducibility in processes that rely on random number generation, as it serves as the initial input for generating random sequences. The results list, which is modified by get_seed, is used to collect the outcomes of all parameter retrievals, facilitating further processing or display in the application.

**Note**: It is important to ensure that the source_dict contains valid keys and that the results list is properly initialized before calling this function. Additionally, be aware that the function does not handle specific exceptions; it is advisable to implement error handling in the calling function to manage potential issues effectively.
## FunctionDef get_inpaint_engine_version(key, fallback, source_dict, results, inpaint_mode, default)
**get_inpaint_engine_version**: The function of get_inpaint_engine_version is to retrieve the version of the inpainting engine based on specified keys and conditions.

**parameters**: The parameters of this Function.
· key: A string that represents the primary key used to look up the inpaint engine version in the source dictionary.  
· fallback: A string or None that serves as a secondary key to use if the primary key is not found.  
· source_dict: A dictionary containing the metadata from which the inpaint engine version is extracted.  
· results: A list that accumulates results and updates during the function execution.  
· inpaint_mode: A string that indicates the mode of inpainting, which affects how the results are processed.  
· default: An optional parameter that specifies a default value to return if neither the key nor the fallback is found.

**Code Description**: The get_inpaint_engine_version function attempts to retrieve the inpaint engine version from the source_dict using the provided key and fallback. It first checks if the value associated with the key exists; if not, it looks for the fallback key. If neither is found, it defaults to the specified default value. The function asserts that the retrieved value is a string and is part of the predefined inpaint engine versions stored in modules.flags.inpaint_engine_versions. Depending on the inpaint_mode, it either appends the retrieved version to the results list or updates the results with a call to gr.update(). In the event of an error (such as a missing key or an invalid value), it appends 'empty' to the results and returns None. 

This function is called within load_parameter_button_click, which is responsible for processing raw metadata and loading various parameters, including the inpaint engine version. The results list is used to collect all relevant information, which is then utilized for further processing or display. The integration of get_inpaint_engine_version within load_parameter_button_click highlights its role in ensuring that the correct inpainting engine version is fetched and made available for subsequent operations.

**Note**: It is important to ensure that the keys used for retrieval are valid and that the source_dict contains the expected structure to avoid runtime errors. The function is designed to handle exceptions gracefully, providing a fallback mechanism to maintain stability.

**Output Example**: A possible return value of the function could be a string such as "v1.2.3" if the inpaint engine version is successfully retrieved, or None if an error occurs during the process.
## FunctionDef get_inpaint_method(key, fallback, source_dict, results, default)
**get_inpaint_method**: The function of get_inpaint_method is to retrieve the inpainting method specified by a key from a source dictionary, with a fallback option and to update the results list accordingly.

**parameters**: The parameters of this Function.
· parameter1: key (str) - The key used to look up the inpainting method in the source dictionary.
· parameter2: fallback (str | None) - An alternative key to use if the primary key is not found; can be None.
· parameter3: source_dict (dict) - The dictionary from which to retrieve the inpainting method.
· parameter4: results (list) - A list that will be updated with the retrieved inpainting method and additional entries.
· parameter5: default (str | None) - A default value to return if neither the key nor the fallback is found in the source dictionary.

**Code Description**: The get_inpaint_method function attempts to retrieve a string value representing an inpainting method from the provided source_dict using the specified key. If the key is not found, it checks for a fallback key. If neither is found, it returns a default value if provided. The function asserts that the retrieved value is a string and is part of the predefined inpaint options available in modules.flags.inpaint_options. If successful, the function appends the retrieved method to the results list and duplicates this entry based on the value of modules.config.default_enhance_tabs. In the event of an exception, it appends an update to the results list and performs the same duplication as in the successful case. 

This function is called within load_parameter_button_click, where it is used to retrieve the inpainting method from the loaded parameter dictionary (raw_metadata). The results list is populated with various parameters, including the inpainting method, which is crucial for configuring the inpainting process in the application. The results list is then used to update the user interface, indicating whether the operation is generating or not.

**Note**: It is important to ensure that the source_dict contains valid keys and that the inpainting method is part of the allowed options to avoid assertion errors. Proper handling of the fallback and default parameters is essential for robust functionality.

**Output Example**: A possible return value of the function could be a string such as "method_a" if the key "inpaint_method" exists in source_dict and is valid, or None if no valid method is found and no default is provided. The results list would then contain entries like ["method_a", "method_a", ..., "method_a"] based on the value of modules.config.default_enhance_tabs.
## FunctionDef get_adm_guidance(key, fallback, source_dict, results, default)
**get_adm_guidance**: The function of get_adm_guidance is to retrieve and process guidance values from a source dictionary based on specified keys and append the results to a provided list.

**parameters**: The parameters of this Function.
· key: A string representing the primary key to look up in the source dictionary.
· fallback: A string or None, representing a secondary key to use if the primary key is not found.
· source_dict: A dictionary from which values are retrieved based on the specified keys.
· results: A list to which the processed values will be appended.
· default: An optional parameter that specifies a default value to use if neither the key nor the fallback is found.

**Code Description**: The get_adm_guidance function attempts to retrieve a value from the source_dict using the provided key. If the key is not found, it will attempt to retrieve a value using the fallback key. If neither key is found, it will use the default value if provided. The retrieved value is expected to be a string that can be evaluated into three components (p, n, e). These components are then converted to floats and appended to the results list. If an error occurs during this process (for example, if the value cannot be evaluated), the function appends three updates to the results list, which are likely placeholders or default values.

This function is called within the load_parameter_button_click function, which is responsible for processing a dictionary of parameters (raw_metadata). The load_parameter_button_click function first ensures that the raw_metadata is in the correct format (a dictionary) and initializes a results list. It then calls various helper functions to extract and process different parameters from the loaded dictionary. The get_adm_guidance function specifically retrieves the 'adm_guidance' value from the loaded parameters, processes it, and appends the results to the results list. This integration indicates that get_adm_guidance plays a crucial role in ensuring that the ADM guidance value is correctly handled as part of the overall parameter loading and processing workflow.

**Note**: It is important to ensure that the values in the source_dict are formatted correctly for evaluation, as improper formatting may lead to runtime errors. Additionally, the function does not specify the type of updates appended in case of exceptions, which may require further clarification in the context of the overall application.
## FunctionDef get_freeu(key, fallback, source_dict, results, default)
**get_freeu**: The function of get_freeu is to retrieve and process specific values from a source dictionary based on provided keys, appending the results to a list.

**parameters**: The parameters of this Function.
· key: A string representing the primary key to look up in the source dictionary.
· fallback: A string or None, representing a secondary key to use if the primary key is not found.
· source_dict: A dictionary from which values are retrieved based on the provided keys.
· results: A list that will be populated with the results of the function's processing.
· default: An optional parameter that serves as a fallback value if neither the key nor the fallback is found in the source dictionary.

**Code Description**: The get_freeu function attempts to retrieve a value from the source_dict using the provided key. If the key is not found, it tries to retrieve the value using the fallback key. If neither key is present, it defaults to the provided default value. The retrieved value is expected to be a string that can be evaluated into four components (b1, b2, s1, s2). These components are then converted to floats and appended to the results list, along with a boolean value indicating success.

In the event of an exception during this process, the function appends a boolean value of False to the results list, followed by four calls to gr.update(), which likely updates some graphical interface elements or state. This function is called within load_parameter_button_click, which processes a metadata dictionary or string, extracts various parameters, and ultimately calls get_freeu to handle specific values related to 'freeu'. The results from get_freeu are integrated into the overall results list, which is returned by load_parameter_button_click.

**Note**: It is important to ensure that the values being evaluated in get_freeu are in the correct format to avoid exceptions. Additionally, the use of eval should be approached with caution, as it can execute arbitrary code if the input is not properly sanitized.
## FunctionDef get_lora(key, fallback, source_dict, results, performance_filename)
**get_lora**: The function of get_lora is to retrieve and process LoRA (Low-Rank Adaptation) parameters from a source dictionary and append the results to a provided list.

**parameters**: The parameters of this Function.
· key: A string representing the key to look up in the source dictionary for LoRA data.
· fallback: A string or None, which serves as an alternative key if the primary key is not found in the source dictionary.
· source_dict: A dictionary containing the LoRA data from which values will be extracted.
· results: A list that will be populated with the processed results of the LoRA data.
· performance_filename: A string or None that represents the filename associated with performance, used to prevent duplication of performance LoRAs.

**Code Description**: The get_lora function is designed to extract LoRA-related information from a given source dictionary. It attempts to retrieve the value associated with the specified key. If the key is not found, it uses the fallback key. The retrieved value is expected to be a string formatted as "enabled : name : weight". The function splits this string into components, determining whether the LoRA is enabled, its name, and its weight.

If the retrieved name matches the performance_filename, an exception is raised to avoid processing the same LoRA twice. The weight is converted to a float, and the results list is updated with the enabled status, name, and weight. In the event of any errors during this process, the function appends default values (True, 'None', and 1) to the results list.

The get_lora function is called within the load_parameter_button_click function, which is responsible for loading various parameters from a metadata dictionary. Specifically, get_lora is invoked in a loop that iterates up to a predefined maximum number of LoRAs, allowing for the retrieval of multiple LoRA entries. This integration ensures that the LoRA data is processed and included in the results, which can be used for further operations or user interface updates.

**Note**: It is important to ensure that the source dictionary contains the expected format for LoRA data. Any discrepancies in the data format may lead to exceptions being raised, which are handled by appending default values to the results. Additionally, care should be taken to manage the performance_filename to prevent duplication of LoRA entries.
## FunctionDef parse_meta_from_preset(preset_content)
**parse_meta_from_preset**: The function of parse_meta_from_preset is to transform a preset configuration dictionary into a structured format suitable for further processing.

**parameters**: The parameters of this Function.
· preset_content: A dictionary containing preset configuration data.

**Code Description**: The parse_meta_from_preset function begins by asserting that the input parameter, preset_content, is of type dictionary. It initializes an empty dictionary named preset_prepared to hold the processed output. The function then iterates over a predefined mapping of possible preset keys, which are sourced from modules.config.possible_preset_keys.

During each iteration, the function checks for specific keys such as "default_loras" and "default_aspect_ratio". For "default_loras", it retrieves the corresponding values from the configuration or the input dictionary, and constructs a combined string representation for each Lora, which is stored in the preset_prepared dictionary. For "default_aspect_ratio", it processes the aspect ratio string to extract width and height values, ensuring to handle both the input dictionary and default configuration appropriately.

For other keys, the function assigns values from the input dictionary if they exist; otherwise, it falls back to the default values from the configuration. Additionally, for keys related to styles and aspect ratios, the function ensures that the values are converted to strings.

The function ultimately returns the preset_prepared dictionary, which contains the processed metadata ready for further use.

This function is called within the preset_selection_change function in the webui.py module. Here, it is utilized to prepare the preset content for model launching and downloading operations. The output of parse_meta_from_preset is critical as it feeds into subsequent processes that handle model downloads and parameter settings based on the user's selected preset.

**Note**: It is essential to ensure that the input to this function is a well-formed dictionary, as the function relies on specific keys being present to function correctly.

**Output Example**: An example of the return value from parse_meta_from_preset could look like this:
{
    'base_model': 'model_v1',
    'checkpoint_downloads': {'model_v1': 'url_to_model'},
    'lora_combined_1': 'Lora1 : 0.5',
    'default_aspect_ratio': ('16', '9'),
    'default_styles': 'style1, style2'
}
## ClassDef MetadataParser
**MetadataParser**: The function of MetadataParser is to serve as an abstract base class for parsing metadata related to different models and formats.

**attributes**: The attributes of this Class.
· raw_prompt: A string that holds the raw prompt input.
· full_prompt: A string that contains the full prompt after processing.
· raw_negative_prompt: A string that holds the raw negative prompt input.
· full_negative_prompt: A string that contains the full negative prompt after processing.
· steps: An integer representing the number of steps for processing, defaulting to a speed value.
· base_model_name: A string that stores the name of the base model.
· base_model_hash: A string that stores the hash of the base model for verification.
· refiner_model_name: A string that stores the name of the refiner model, if applicable.
· refiner_model_hash: A string that stores the hash of the refiner model for verification.
· loras: A list that holds tuples of LoRA names and their corresponding weights.
· vae_name: A string that stores the name of the VAE model.

**Code Description**: The MetadataParser class is an abstract base class designed to define a common interface for metadata parsing across different implementations. It includes several attributes that store various pieces of metadata related to model prompts, steps, and model names. The class defines three abstract methods: `get_scheme`, `to_json`, and `to_string`, which must be implemented by any subclass. These methods are intended to provide specific functionality for retrieving the metadata scheme, converting metadata to JSON format, and converting metadata to a string format, respectively.

The `set_data` method is a concrete implementation that allows subclasses to set the metadata attributes based on the provided parameters. This method processes the input parameters, computes the hashes for the base and refiner models, and populates the list of LoRAs with their names, weights, and hashes. The method utilizes helper functions such as `get_file_from_folder_list` and `sha256_from_cache` to retrieve file paths and compute hashes, ensuring that the metadata is accurately represented.

The MetadataParser class is utilized by its subclasses, A1111MetadataParser and FooocusMetadataParser, which implement the abstract methods to provide specific parsing logic for different metadata schemes. The `get_metadata_parser` function is responsible for instantiating the appropriate parser based on the specified metadata scheme, returning an instance of either A1111MetadataParser or FooocusMetadataParser. This design allows for extensibility and modularity in handling various metadata formats.

**Note**: It is important to ensure that any subclass of MetadataParser implements all abstract methods to avoid runtime errors. Additionally, the `set_data` method should be called to initialize the metadata attributes before attempting to use the parsing methods.
### FunctionDef __init__(self)
**__init__**: The function of __init__ is to initialize an instance of the MetadataParser class with default attribute values.

**parameters**: The parameters of this Function.
· There are no parameters for this __init__ method.

**Code Description**: The __init__ method serves as the constructor for the MetadataParser class. When an instance of MetadataParser is created, this method is automatically invoked to set up the initial state of the object. The method initializes several attributes to their default values, which are essential for the functioning of the MetadataParser.

The attributes initialized in this method include:
- `raw_prompt`: A string that holds the raw input prompt, initialized to an empty string.
- `full_prompt`: A string that contains the complete prompt, also initialized to an empty string.
- `raw_negative_prompt`: A string for the raw negative prompt, initialized to an empty string.
- `full_negative_prompt`: A string for the complete negative prompt, initialized to an empty string.
- `steps`: An integer that represents the performance level, initialized to the value associated with the `SPEED` attribute from the Steps enumeration. This establishes a default performance level for the MetadataParser instance.
- `base_model_name`: A string that holds the name of the base model, initialized to an empty string.
- `base_model_hash`: A string that holds the hash of the base model, initialized to an empty string.
- `refiner_model_name`: A string for the name of the refiner model, initialized to an empty string.
- `refiner_model_hash`: A string for the hash of the refiner model, initialized to an empty string.
- `loras`: A list that is initialized as an empty list, intended to hold any additional parameters or configurations.
- `vae_name`: A string that holds the name of the Variational Autoencoder (VAE), initialized to an empty string.

The initialization of the `steps` attribute with a default value from the Steps enumeration is particularly significant, as it ensures that every instance of MetadataParser starts with a predefined performance level, which can be crucial for subsequent processing and functionality within the class.

**Note**: It is important to ensure that any modifications to the attributes after initialization are done with valid data types and values to maintain the integrity of the MetadataParser instance.
***
### FunctionDef get_scheme(self)
**get_scheme**: The function of get_scheme is to retrieve the metadata scheme associated with the parser implementation.

**parameters**: The parameters of this Function.
· None

**Code Description**: The get_scheme function is defined within a class that presumably implements a metadata parser. It is intended to return an instance of the MetadataScheme enumeration, which defines a set of constants representing different metadata schemes used in the application. However, the function raises a NotImplementedError, indicating that any subclass inheriting from this class must provide its own implementation of the get_scheme method.

The MetadataScheme class, which is referenced in this function, contains constants such as FOOOCUS and A1111, each representing a specific metadata scheme. This structured approach allows for consistent handling of various metadata formats throughout the application.

The get_scheme function is called within the log function of the private_logger module. In this context, the log function utilizes the metadata_parser parameter, which is expected to be an instance of a class that inherits from the MetadataParser class. When the log function is executed, it attempts to call metadata_parser.get_scheme(). The value returned from this call is then used to embed the corresponding metadata scheme into the saved image file's PNG info or EXIF data, depending on the output format specified.

This design ensures that the correct metadata scheme is applied during the logging process, facilitating accurate metadata handling and processing. The get_scheme method serves as a critical point of integration between the metadata parser implementations and the logging functionality, reinforcing the importance of implementing this method in any derived classes.

**Note**: It is essential for developers to implement the get_scheme method in any subclass of the MetadataParser to ensure that the logging and metadata handling functions correctly reflect the intended metadata scheme. Failure to implement this method will result in a NotImplementedError being raised, which will disrupt the functionality of any components relying on it.
***
### FunctionDef to_json(self, metadata)
**to_json**: The function of to_json is to convert metadata into a JSON-compatible dictionary format.

**parameters**: The parameters of this Function.
· metadata: This parameter can be either a dictionary or a string that represents the metadata to be converted.

**Code Description**: The to_json function is defined within the MetadataParser class and is intended to transform the provided metadata into a JSON format. However, it is important to note that this function raises a NotImplementedError, indicating that the actual implementation of this function has not been provided. This suggests that the function is meant to be overridden in a subclass of MetadataParser where the specific logic for converting metadata to JSON will be defined.

The to_json function is called within the trigger_metadata_import function in the webui.py module. In this context, the trigger_metadata_import function first reads metadata from an image file using the read_info_from_image function. If the parameters are successfully retrieved, it then obtains an appropriate metadata parser based on the metadata scheme. The to_json function is subsequently invoked on this metadata parser with the parameters obtained from the image. The output of the to_json function is expected to be a dictionary that can be further processed or utilized in the application.

Since the to_json function is not implemented, any attempt to call it will result in an error unless it is properly defined in a subclass. This highlights the importance of ensuring that subclasses provide the necessary functionality to handle the conversion of metadata to JSON format.

**Note**: Users of this function should be aware that it is currently not implemented and will raise an error if called directly. It is essential to implement this function in a subclass to ensure proper functionality.
***
### FunctionDef to_string(self, metadata)
**to_string**: The function of to_string is to convert metadata into a string representation.

**parameters**: The parameters of this Function.
· metadata: A dictionary containing metadata that needs to be converted to a string.

**Code Description**: The to_string function is defined within the MetadataParser class and is intended to be overridden by subclasses. The function takes a single parameter, metadata, which is expected to be a dictionary. The purpose of this function is to provide a string representation of the metadata passed to it. However, the implementation raises a NotImplementedError, indicating that this function must be implemented in a subclass for it to be functional.

This function is called within the log function located in the modules/private_logger.py file. In the log function, if a metadata_parser is provided (an instance of MetadataParser or its subclass), the to_string method is invoked with a copy of the metadata dictionary. The result of this call is stored in the parsed_parameters variable. This string representation of the metadata is then utilized when saving images in different formats (PNG, JPEG, WEBP) and when generating an HTML log file that includes metadata information about the processed images.

The relationship between to_string and its caller, log, is crucial as it allows the log function to embed relevant metadata into the saved images and the generated HTML log. This integration ensures that users can access and review the metadata associated with each image, enhancing the overall functionality of the logging process.

**Note**: It is important to implement the to_string method in any subclass of MetadataParser to ensure that the log function can successfully convert metadata into a string format. Without this implementation, the log function will not be able to log any metadata, potentially leading to a loss of important information during image processing.
***
### FunctionDef set_data(self, raw_prompt, full_prompt, raw_negative_prompt, full_negative_prompt, steps, base_model_name, refiner_model_name, loras, vae_name)
**set_data**: The function of set_data is to initialize and set various metadata attributes related to prompts, models, and configurations for the MetadataParser class.

**parameters**: The parameters of this Function.
· raw_prompt: A string representing the raw positive prompt input.
· full_prompt: A string representing the full positive prompt input.
· raw_negative_prompt: A string representing the raw negative prompt input.
· full_negative_prompt: A string representing the full negative prompt input.
· steps: An integer indicating the number of steps for processing.
· base_model_name: A string representing the name of the base model to be used.
· refiner_model_name: A string representing the name of the refiner model, if applicable.
· loras: A list of tuples containing LoRA names and their corresponding weights.
· vae_name: A string representing the name of the Variational Autoencoder (VAE) model.

**Code Description**: The set_data function is responsible for setting up the internal state of the MetadataParser instance by assigning values to various attributes based on the provided parameters. It begins by assigning the raw and full prompts, as well as the raw and full negative prompts, to their respective instance variables. The steps parameter is also stored, along with the base model name, which is processed to extract the stem (the base name without the file extension) using the Path class from the pathlib module.

The function then retrieves the file path of the base model using the get_file_from_folder_list function, which searches for the specified model name within a predefined list of checkpoint directories. Once the file path is obtained, the SHA-256 hash of the base model is calculated and stored using the sha256_from_cache function, which either retrieves the hash from a cache or computes it if not already cached.

If a refiner model name is provided and is not empty or 'None', the function similarly processes this name, retrieves its file path, and calculates its hash. The function then initializes the loras attribute as an empty list and iterates over the provided loras parameter, which is expected to be a list of tuples. For each LoRA, it checks if the name is not 'None', retrieves the corresponding file path, calculates its hash, and appends a tuple containing the LoRA name (stem), weight, and hash to the loras list.

Finally, the function processes the vae_name parameter to store its stem in the instance variable. This comprehensive setup ensures that the MetadataParser instance is properly initialized with all necessary metadata, which can be utilized later in the processing workflow.

The set_data function is called within the save_and_log function in the async_worker module. In this context, it is used to populate the metadata for images being processed, ensuring that the relevant prompts, model names, and configurations are correctly set before logging the results. This integration highlights the importance of the set_data function in maintaining accurate metadata throughout the image processing pipeline.

**Note**: It is crucial to ensure that the provided model names and paths are valid and accessible; otherwise, the functions get_file_from_folder_list and sha256_from_cache may raise exceptions if the files cannot be found or accessed. Additionally, the loras parameter should be structured as a list of tuples to avoid runtime errors during iteration.
***
## ClassDef A1111MetadataParser
**A1111MetadataParser**: The function of A1111MetadataParser is to parse and convert metadata specific to the A1111 model format into JSON and string representations.

**attributes**: The attributes of this Class.
· fooocus_to_a1111: A dictionary mapping various metadata keys to their corresponding human-readable labels used in the A1111 format.

**Code Description**: The A1111MetadataParser class extends the MetadataParser abstract base class, implementing specific functionality for handling metadata associated with the A1111 model format. It provides methods to retrieve the metadata scheme, convert metadata to JSON format, and convert metadata to a string format.

The `get_scheme` method returns the specific metadata scheme for A1111, which is defined as `MetadataScheme.A1111`. This method is essential for identifying the type of metadata being processed.

The `to_json` method takes a string containing metadata and processes it to extract relevant information. It splits the metadata into lines, identifies prompts and negative prompts, and uses regular expressions to extract additional parameters such as resolution, steps, and model names. The method constructs a dictionary containing the parsed data, which includes the prompt, negative prompt, styles, and various model-related attributes. It also handles specific cases, such as loading performance based on the number of steps and resolving LoRA weights and hashes.

The `to_string` method converts the parsed metadata dictionary back into a formatted string representation. It constructs a string that includes the positive and negative prompts along with the generation parameters, ensuring that the output is human-readable and structured according to the A1111 format.

The class also includes a static method, `add_extension_to_filename`, which appends the appropriate file extension to model names based on the provided filenames. This method ensures that the correct file paths are used when referencing models.

The A1111MetadataParser class is instantiated through the `get_metadata_parser` function, which matches the specified metadata scheme to return an appropriate parser instance. This design allows for modularity and extensibility in handling different metadata formats, as the function can return either A1111MetadataParser or other parsers like FooocusMetadataParser based on the input.

**Note**: It is important to ensure that the metadata string passed to the `to_json` method is correctly formatted to avoid parsing errors. Additionally, when using the `to_string` method, the metadata dictionary should contain all necessary keys to generate a complete output string.

**Output Example**: 
A possible return value from the `to_json` method might look like this:
```json
{
    "prompt": "A beautiful sunset over the mountains",
    "negative_prompt": "No people, no buildings",
    "resolution": "(1920, 1080)",
    "steps": "50",
    "sampler": "Euler",
    "guidance_scale": "7.5",
    "styles": "[\"landscape\", \"vibrant\"]",
    "base_model": "model_name",
    "version": "1.0"
}
```
### FunctionDef get_scheme(self)
**get_scheme**: The function of get_scheme is to return the specific metadata scheme associated with the A1111 metadata parser.

**parameters**: The parameters of this Function.
· None

**Code Description**: The get_scheme function is a method defined within the A1111MetadataParser class. Its primary purpose is to provide a way to retrieve the metadata scheme that the parser is designed to handle. When invoked, this method returns the constant MetadataScheme.A1111, which is part of the MetadataScheme enumeration defined in the MetadataScheme class.

The relationship between get_scheme and the MetadataScheme class is integral to the functionality of the metadata parsing system within the application. By returning MetadataScheme.A1111, the get_scheme method indicates that the A1111 metadata parser is specifically tailored to process metadata conforming to the A1111 scheme. This allows other components of the application, such as metadata processing modules or task handlers, to understand which scheme is being utilized and to act accordingly.

The get_scheme method is typically called in contexts where the metadata parser needs to communicate its capabilities or when the application requires the identification of the metadata scheme being used for parsing operations. This ensures that the correct parsing logic is applied based on the metadata scheme returned by the method.

**Note**: It is important to use the get_scheme method in conjunction with other components that rely on the metadata scheme for accurate metadata handling. This will help maintain consistency and correctness in the processing of metadata throughout the application.

**Output Example**: The output of the get_scheme method would be the value associated with the A1111 metadata scheme, which can be represented as follows: `MetadataScheme.A1111`.
***
### FunctionDef to_json(self, metadata)
**to_json**: The function of to_json is to convert a metadata string into a structured JSON-like dictionary format.

**parameters**: The parameters of this Function.
· metadata: A string representing the metadata that needs to be parsed and converted into a structured dictionary.

**Code Description**: The to_json function processes a given metadata string to extract relevant information and organize it into a dictionary format suitable for JSON serialization. The function begins by initializing two empty strings, metadata_prompt and metadata_negative_prompt, which will hold the main prompt and the negative prompt extracted from the metadata.

The function splits the input metadata string into lines and checks the last line for specific parameters using regular expressions. If the last line contains fewer than three parameters, it is appended to the list of lines for further processing.

As the function iterates through each line, it identifies whether the line pertains to the negative prompt by checking for a specific prefix defined in the class attribute fooocus_to_a1111. If the prefix is found, the line is processed as part of the negative prompt; otherwise, it is added to the main prompt.

After separating the prompts, the function calls extract_styles_from_prompt, which analyzes the prompts to extract applicable styles and modify them accordingly. This function returns a list of found styles, the cleaned main prompt, and the negative prompt.

The resulting data dictionary is then populated with the extracted prompt and negative prompt. The function also processes any additional parameters found in the last line of the metadata string, such as resolution and various model identifiers, updating the data dictionary accordingly. The unquote function is utilized to decode any JSON strings that are enclosed in double quotes, ensuring that valid JSON values are correctly parsed.

The function further handles specific keys like 'steps' to determine performance levels using the Performance class's by_steps method. It also updates model-related keys with their corresponding filenames by invoking the add_extension_to_filename function.

Finally, the function checks for any LoRA (Low-Rank Adaptation) weights or hashes and processes them to include relevant information in the data dictionary. The completed data dictionary is then returned, providing a structured representation of the parsed metadata.

The to_json function is integral to the A1111MetadataParser class, facilitating the transformation of unstructured metadata strings into a structured format that can be easily utilized in subsequent processing steps.

**Note**: It is crucial to ensure that the input metadata string is formatted correctly to avoid parsing errors. Any discrepancies in the expected format may lead to incomplete or inaccurate data being returned.

**Output Example**: A possible return value of the to_json function could look like this:
{
    'prompt': 'A beautiful sunset',
    'negative_prompt': 'Not Elegant',
    'resolution': '(1920, 1080)',
    'styles': "['Elegant']",
    'performance': 'Quality',
    'sampler': 'Karras',
    'base_model': 'model_name',
    'lora_combined_1': 'lora_filename : 0.5'
}
***
### FunctionDef to_string(self, metadata)
**to_string**: The function of to_string is to convert a metadata dictionary into a formatted string representation suitable for output.

**parameters**: The parameters of this Function.
· metadata: A dictionary containing various metadata attributes related to the generation process.

**Code Description**: The to_string method processes a given metadata dictionary to create a structured string that encapsulates the relevant information for a generation task. It begins by extracting key-value pairs from the metadata dictionary, specifically focusing on attributes such as resolution, sampler, scheduler, and various generation parameters.

The method evaluates the resolution to obtain width and height values, which are then used to format the output. It checks if the specified sampler exists in a predefined SAMPLERS dictionary and adjusts its representation accordingly, particularly if the scheduler is set to 'karras'. 

A dictionary named generation_params is constructed to hold the formatted parameters, mapping them from internal representations to user-friendly strings. This includes parameters like steps, seed, resolution, guidance scale, and sharpness, among others. The method also conditionally includes additional parameters if certain conditions are met, such as the presence of a refiner model or specific keys in the metadata.

The method further processes any LoRA (Low-Rank Adaptation) models by collecting their names and weights, ensuring they are included in the final output if applicable. It also appends the version of the metadata and the creator's information if available.

Finally, the method constructs the output string by concatenating the positive and negative prompts along with the formatted generation parameters. The quote function is utilized to ensure that any values containing special characters are properly formatted for JSON compatibility. This is crucial for maintaining the integrity of the output string.

The to_string method is integral to the A1111MetadataParser class, providing a clear and structured representation of metadata that can be easily interpreted by users or other components of the system.

**Note**: It is important to ensure that the metadata dictionary passed to this function contains all the required keys to avoid KeyErrors during processing. The output string is designed to be human-readable, making it suitable for logging or display purposes.

**Output Example**: An example of the output string generated by this method could be:
"Prompt: A beautiful sunset, Negative prompt: None, steps: 50, sampler: Euler, seed: 12345, resolution: 1920x1080, guidance_scale: 7.5, sharpness: 1.0"
***
### FunctionDef add_extension_to_filename(data, filenames, key)
**add_extension_to_filename**: The function of add_extension_to_filename is to update a specified key in a data dictionary with the corresponding filename if the filename matches the key's value.

**parameters**: The parameters of this Function.
· parameter1: data - A dictionary containing metadata where the specified key's value may be updated.
· parameter2: filenames - A list of filenames to be checked against the value of the specified key in the data dictionary.
· parameter3: key - A string representing the key in the data dictionary whose value is to be compared with the filenames.

**Code Description**: The add_extension_to_filename function iterates through a list of filenames and checks if the stem (the filename without its extension) of each filename matches the value associated with the specified key in the data dictionary. If a match is found, the function updates the value of that key in the data dictionary to the full filename (including its extension) and exits the loop. This function is particularly useful in scenarios where filenames need to be resolved to their full paths based on metadata, ensuring that the correct filename is associated with the corresponding metadata entry.

This function is called within the to_json method of the A1111MetadataParser class. In to_json, after parsing the metadata string into a structured dictionary, the function is invoked to ensure that the 'vae', 'base_model', and 'refiner_model' keys in the data dictionary are updated with the correct filenames from predefined lists (vae_filenames and model_filenames). This integration ensures that the metadata is not only structured but also accurately reflects the filenames that correspond to the models and configurations being used.

**Note**: It is essential to ensure that the filenames provided in the list are correctly formatted and that the key exists in the data dictionary to avoid any potential errors during execution.
***
## ClassDef FooocusMetadataParser
**FooocusMetadataParser**: The function of FooocusMetadataParser is to parse and manage metadata specific to the Fooocus model format.

**attributes**: The attributes of this Class.
· raw_prompt: A string that holds the raw prompt input.  
· full_prompt: A string that contains the full prompt after processing.  
· raw_negative_prompt: A string that holds the raw negative prompt input.  
· full_negative_prompt: A string that contains the full negative prompt after processing.  
· steps: An integer representing the number of steps for processing, defaulting to a speed value.  
· base_model_name: A string that stores the name of the base model.  
· base_model_hash: A string that stores the hash of the base model for verification.  
· refiner_model_name: A string that stores the name of the refiner model, if applicable.  
· refiner_model_hash: A string that stores the hash of the refiner model for verification.  
· loras: A list that holds tuples of LoRA names and their corresponding weights.  
· vae_name: A string that stores the name of the VAE model.  

**Code Description**: The FooocusMetadataParser class extends the MetadataParser abstract base class, implementing the methods required for parsing metadata specific to the Fooocus model. The class provides functionality to retrieve the metadata scheme, convert metadata to JSON format, and convert metadata to a string format.

The `get_scheme` method returns the specific metadata scheme associated with Fooocus, which is defined as `MetadataScheme.FOOOCUS`. This method is essential for identifying the type of metadata being processed.

The `to_json` method takes a dictionary of metadata as input and processes it to replace certain values with corresponding filenames from predefined lists. It checks for specific keys such as 'base_model', 'refiner_model', and 'vae', and utilizes the `replace_value_with_filename` static method to perform the replacements. This method ensures that the metadata is formatted correctly for JSON output, omitting any entries with empty or 'None' values.

The `to_string` method converts the metadata into a string format, organizing it into a dictionary and including additional attributes such as `full_prompt`, `steps`, and model names. It also formats the 'lora_combined_' entries by stripping folder paths and returning a clean representation. The final output is a JSON string that is sorted for consistency.

The `replace_value_with_filename` static method is responsible for matching values against a list of filenames. It checks if the value corresponds to a filename stem and returns the appropriate filename if a match is found. This method is crucial for ensuring that the metadata accurately reflects the file structure used in the project.

The FooocusMetadataParser class is instantiated through the `get_metadata_parser` function, which selects the appropriate parser based on the provided metadata scheme. This function uses a match-case structure to return an instance of FooocusMetadataParser when the scheme is `MetadataScheme.FOOOCUS`. This design allows for modular handling of different metadata formats, ensuring that the correct parser is utilized for each scheme.

**Note**: It is important to ensure that the metadata attributes are set correctly before invoking the parsing methods. The `set_data` method from the parent class, MetadataParser, should be called to initialize the attributes with the relevant metadata before using the `to_json` or `to_string` methods.

**Output Example**: A possible output of the `to_json` method when provided with a metadata dictionary might look like this:
```json
{
    "base_model": "path/to/base_model_file",
    "refiner_model": "path/to/refiner_model_file",
    "vae": "path/to/vae_file",
    "loras": [
        "lora1 : 0.5",
        "lora2 : 0.8"
    ],
    "full_prompt": "This is the full prompt.",
    "full_negative_prompt": "This is the full negative prompt.",
    "steps": 50,
    "created_by": "Author Name"
}
```
### FunctionDef get_scheme(self)
**get_scheme**: The function of get_scheme is to return the specific metadata scheme used by the FooocusMetadataParser.

**parameters**: The parameters of this Function.
· None

**Code Description**: The get_scheme function is a method defined within the FooocusMetadataParser class. Its primary purpose is to return a constant value from the MetadataScheme enumeration, specifically the FOOOCUS constant. This indicates that the FooocusMetadataParser is designed to handle metadata according to the 'fooocus' scheme.

When invoked, the get_scheme method does not take any parameters and directly returns the value associated with MetadataScheme.FOOOCUS. This design allows for a clear and consistent way to identify the metadata scheme being utilized by the parser. The relationship between get_scheme and the MetadataScheme class is crucial, as it ensures that the parser adheres to the defined constants for metadata handling.

The FOOOCUS constant is part of a broader set of metadata schemes defined in the MetadataScheme class, which includes other schemes such as A1111. By using this enumeration, the application can maintain a structured approach to metadata parsing, allowing different parsers to specify which scheme they are implementing.

The get_scheme method is typically called in contexts where the metadata scheme needs to be identified or verified, such as during the initialization of metadata processing tasks or when interfacing with other components that require knowledge of the current metadata scheme.

**Note**: It is important to ensure that the FooocusMetadataParser is used in scenarios where the 'fooocus' metadata scheme is applicable, as this will facilitate accurate metadata parsing and processing.

**Output Example**: The output of the get_scheme function would be the value 'fooocus', representing the FOOOCUS metadata scheme.
***
### FunctionDef to_json(self, metadata)
**to_json**: The function of to_json is to process a metadata dictionary and replace specific values with corresponding filenames based on predefined conditions.

**parameters**: The parameters of this Function.
· parameter1: metadata - A dictionary containing metadata key-value pairs that may include model names and configurations.

**Code Description**: The to_json function iterates through the provided metadata dictionary, examining each key-value pair. If the value is empty or the string 'None', it is skipped. For keys that match 'base_model' or 'refiner_model', the function calls replace_value_with_filename, passing the key, value, and a list of model filenames from the configuration. Similarly, if the key starts with 'lora_combined_', it uses the lora_filenames list, and for the key 'vae', it utilizes the vae_filenames list. The replace_value_with_filename function is essential here, as it determines if the value can be replaced with a corresponding filename based on specific matching criteria. After processing all relevant keys, the modified metadata dictionary is returned.

This function is integral to ensuring that the metadata is accurately represented with the correct filenames, enhancing the clarity and usability of the metadata structure. The relationship with the replace_value_with_filename function is crucial, as it handles the logic for matching and replacing values with filenames, ensuring that the metadata is not only valid but also meaningful.

**Note**: It is important to ensure that the values in the metadata dictionary are formatted correctly, especially when they are expected to contain a name and weight separated by ' : '. Additionally, the function assumes that the filenames provided in the configuration are valid and accessible.

**Output Example**: A possible return value could be a modified metadata dictionary such as {'base_model': 'example_filename.model', 'refiner_model': 'refiner_filename.model', 'vae': 'vae_filename.vae'} after processing the input metadata.
***
### FunctionDef to_string(self, metadata)
**to_string**: The function of to_string is to convert metadata into a JSON string format.

**parameters**: The parameters of this Function.
· metadata: A list of tuples, where each tuple contains a label, a key, and a value representing metadata information.

**Code Description**: The to_string function processes a list of metadata tuples and constructs a dictionary that includes both the provided metadata and additional attributes from the class instance. The function begins by iterating over the metadata list, checking for keys that start with 'lora_combined_'. If such a key is found, it splits the associated value into a name and weight, modifies the name to remove any folder paths, and updates the metadata list with the new formatted value.

Next, the function creates a dictionary called `res` that maps keys to their corresponding values from the metadata list. It then adds several additional key-value pairs to this dictionary, including 'full_prompt', 'full_negative_prompt', 'steps', 'base_model', and 'base_model_hash', which are attributes of the class instance. If the 'refiner_model_name' is not an empty string or 'None', it also adds 'refiner_model' and 'refiner_model_hash' to the dictionary. The function continues by adding 'vae' and 'loras' attributes to the dictionary.

If the configuration specifies a creator for the metadata, it includes a 'created_by' entry in the dictionary. Finally, the function returns a JSON string representation of the sorted dictionary.

**Note**: It is important to ensure that the metadata list is structured correctly, as the function relies on the expected format of tuples. Additionally, the function assumes that the values associated with keys starting with 'lora_combined_' can be split into two parts.

**Output Example**: A possible return value of the function could look like this:
```json
{
    "base_model": "ModelName",
    "base_model_hash": "abc123",
    "created_by": "UserName",
    "full_negative_prompt": "No negative prompts",
    "full_prompt": "This is a full prompt",
    "loras": ["Lora1", "Lora2"],
    "refiner_model": "RefinerModelName",
    "refiner_model_hash": "def456",
    "steps": 50,
    "vae": "VaeName"
}
```
***
### FunctionDef replace_value_with_filename(key, value, filenames)
**replace_value_with_filename**: The function of replace_value_with_filename is to replace a given value with a corresponding filename based on specific conditions.

**parameters**: The parameters of this Function.
· parameter1: key - A string representing the key in the metadata dictionary that is being processed.  
· parameter2: value - A string representing the value associated with the key, which may contain a name and weight or just a name.  
· parameter3: filenames - A list of filenames that will be checked against the value to find a match.

**Code Description**: The replace_value_with_filename function iterates through a list of filenames to determine if the provided value corresponds to any of the filenames based on specific criteria. If the key starts with 'lora_combined_', the function splits the value into a name and weight. It then checks if the name matches the stem (the filename without its extension) of the current filename. If a match is found, it returns a string that combines the filename and the weight. If the key does not start with 'lora_combined_', the function checks if the value matches the stem of the filename directly. If a match is found, it returns the filename. If no matches are found after checking all filenames, the function returns None.

This function is called within the to_json method of the FooocusMetadataParser class. The to_json method processes a metadata dictionary, replacing specific values based on the keys 'base_model', 'refiner_model', and 'vae' with the corresponding filenames from predefined lists (model_filenames, lora_filenames, and vae_filenames). The replace_value_with_filename function is crucial for ensuring that the metadata is accurately represented with the correct filenames, enhancing the usability and clarity of the metadata structure.

**Note**: It is important to ensure that the value passed to the function is formatted correctly, especially when it is expected to contain a name and weight separated by ' : '. Additionally, the function assumes that the filenames provided in the list are valid and accessible.

**Output Example**: A possible return value could be "example_filename.lora : 0.75" if the key is 'lora_combined_example', the value is "example : 0.75", and "example_filename.lora" is present in the filenames list. If no matches are found, the function would return None.
***
## FunctionDef get_metadata_parser(metadata_scheme)
**get_metadata_parser**: The function of get_metadata_parser is to instantiate and return the appropriate metadata parser based on the specified metadata scheme.

**parameters**: The parameters of this Function.
· metadata_scheme: An instance of the MetadataScheme enumeration that specifies which metadata parser to use.

**Code Description**: The get_metadata_parser function utilizes a match-case structure to determine which metadata parser to instantiate based on the provided metadata_scheme parameter. It checks the value of metadata_scheme against predefined constants in the MetadataScheme enumeration, specifically MetadataScheme.FOOOCUS and MetadataScheme.A1111. 

If the metadata_scheme is MetadataScheme.FOOOCUS, the function returns an instance of the FooocusMetadataParser class. If the metadata_scheme is MetadataScheme.A1111, it returns an instance of the A1111MetadataParser class. If the metadata_scheme does not match any of the defined cases, the function raises a NotImplementedError, indicating that the requested metadata scheme is not supported.

This function is called by various components in the project, including the worker function in the async_worker module and the save_and_log function. In the worker function, get_metadata_parser is invoked to obtain the correct parser based on the async_task's metadata_scheme attribute. This allows the worker to process metadata appropriately for different tasks, ensuring that the correct parsing logic is applied based on the metadata format being used.

In the save_and_log function, get_metadata_parser is used to create a metadata parser instance when the async_task.save_metadata_to_images attribute is set to true. The parser is then utilized to set the relevant metadata attributes before logging the image and its associated metadata.

**Note**: It is essential to ensure that the metadata_scheme provided to the get_metadata_parser function corresponds to one of the defined schemes in the MetadataScheme enumeration to avoid runtime errors. Additionally, any subclass of MetadataParser that is instantiated must implement the required abstract methods to function correctly.

**Output Example**: A possible return value from the get_metadata_parser function when called with MetadataScheme.FOOOCUS might look like this:
```python
FooocusMetadataParser()
```
## FunctionDef read_info_from_image(file)
**read_info_from_image**: The function of read_info_from_image is to extract parameters and metadata scheme information from an image file.

**parameters**: The parameters of this Function.
· file: An object representing the image file from which metadata and parameters are to be extracted.

**Code Description**: The read_info_from_image function begins by creating a copy of the 'info' dictionary from the provided file object. It then attempts to extract specific keys: 'parameters', 'fooocus_scheme', and 'exif'. If 'parameters' is found and is a valid JSON string, it is parsed into a Python dictionary. If 'exif' data is available, the function retrieves the UserComment and MakerNote from the EXIF metadata, which are then processed similarly to 'parameters'.

The function checks if 'parameters' is a valid JSON string using the is_json function. If it is valid, it is parsed; otherwise, if 'exif' is present, it attempts to extract 'parameters' and 'metadata_scheme' from the EXIF data. If the parsing of 'parameters' fails, the function defaults to creating a MetadataScheme instance based on the type of 'parameters' extracted. If 'parameters' is a dictionary, it defaults to the FOOOCUS scheme; if it is a string, it defaults to the A1111 scheme.

Finally, the function returns a tuple containing the parsed 'parameters' and the determined 'metadata_scheme'. This function is called by other functions such as trigger_metadata_preview and trigger_metadata_import in the webui.py module. In trigger_metadata_preview, it retrieves the parameters and metadata scheme to prepare results for a preview. In trigger_metadata_import, it uses the extracted information to either print an error message if parameters are not found or to parse the parameters using the appropriate metadata parser.

**Note**: It is essential to ensure that the input file contains the necessary metadata for the function to operate correctly. If the metadata is missing or malformed, the function may return None for parameters or metadata_scheme, which should be handled appropriately by the calling functions.

**Output Example**: 
- Input: An image file with valid parameters and metadata.
- Output: ({"name": "John", "age": 30}, MetadataScheme.FOOOCUS)
## FunctionDef get_exif(metadata, metadata_scheme)
**get_exif**: The function of get_exif is to create and return an EXIF metadata object containing specific metadata information for an image.

**parameters**: The parameters of this Function.
· metadata: A string or None that represents user comments to be embedded in the EXIF data.
· metadata_scheme: A string that represents the metadata scheme to be embedded in the EXIF data.

**Code Description**: The get_exif function initializes an EXIF object using the Image.Exif() class from the Pillow library. It sets specific EXIF tags with the provided metadata and a software version string. The function assigns the user comments to the tag identified by 0x9286, the software name to the tag 0x0131, and the metadata scheme to the tag 0x927C. After populating these fields, the function returns the constructed EXIF object.

This function is called within the log function of the modules/private_logger.py file. In the log function, when saving an image in JPEG or WEBP format, the get_exif function is invoked to retrieve the EXIF data that includes the parsed metadata and the metadata scheme. This integration allows the log function to embed relevant metadata into the saved image, enhancing the image's informational context.

**Note**: It is important to ensure that the metadata parameter is properly formatted as a string or None, as this will directly affect the EXIF data embedded in the image. Additionally, the metadata_scheme should be a valid string to avoid any inconsistencies in the EXIF data.

**Output Example**: A possible appearance of the code's return value could be an EXIF object containing the following tags:
- UserComment (0x9286): "This is a sample user comment."
- Software (0x0131): "Fooocus v1.0.0"
- MakerNote (0x927C): "Sample Metadata Scheme"
