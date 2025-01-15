## FunctionDef set_output_directory(output_dir)
**set_output_directory**: The function of set_output_directory is to set the global output directory for the application.

**parameters**: The parameters of this Function.
· output_dir: A string representing the path to the desired output directory.

**Code Description**: The set_output_directory function is designed to assign a specified directory path to a global variable named output_directory. When this function is called with a string argument (output_dir), it updates the value of output_directory to the provided path. This allows other parts of the application to reference the output_directory variable to determine where output files or data should be saved. The use of the global keyword indicates that the function modifies a variable that exists outside its local scope, ensuring that the updated value persists throughout the application.

**Note**: It is important to ensure that the output_dir provided is a valid directory path. If an invalid path is set, subsequent operations that rely on output_directory may fail or produce errors. Additionally, since this function modifies a global variable, care should be taken to avoid unintended side effects in other parts of the code that may also interact with output_directory.
## FunctionDef set_temp_directory(temp_dir)
**set_temp_directory**: The function of set_temp_directory is to set a global temporary directory variable.

**parameters**: The parameters of this Function.
· temp_dir: A string representing the path to the temporary directory that will be set.

**Code Description**: The set_temp_directory function is designed to assign a value to a global variable named temp_directory. When this function is called with a string argument (temp_dir), it updates the global temp_directory variable to hold the specified path. This allows other parts of the program to access the temporary directory path consistently throughout the application. The use of the global keyword indicates that the function modifies a variable that is defined outside its local scope, ensuring that the updated value persists beyond the function's execution.

**Note**: It is important to ensure that the temp_dir provided is a valid directory path. Additionally, since this function modifies a global variable, care should be taken to avoid unintended side effects in other parts of the code that may rely on the value of temp_directory.
## FunctionDef set_input_directory(input_dir)
**set_input_directory**: The function of set_input_directory is to set the global variable input_directory to the specified input directory path.

**parameters**: The parameters of this Function.
· input_dir: A string representing the path to the input directory that needs to be set.

**Code Description**: The set_input_directory function is designed to update a global variable named input_directory with the value provided as the input_dir parameter. When this function is called, it first declares the variable input_directory as global, which allows it to modify the variable that exists outside of the function's local scope. The function then assigns the value of input_dir to input_directory, effectively changing the global state of input_directory to the new path specified by the user. This function is useful in scenarios where the input directory needs to be dynamically set based on user input or configuration settings.

**Note**: It is important to ensure that the input_dir parameter is a valid directory path to avoid potential issues when accessing the input_directory later in the code. Additionally, since input_directory is a global variable, any changes made by this function will affect all parts of the code that reference input_directory.
## FunctionDef get_output_directory
**get_output_directory**: The function of get_output_directory is to return the global output directory.

**parameters**: The parameters of this Function.
· There are no parameters for this function.

**Code Description**: The get_output_directory function is a simple utility that accesses and returns the value of a global variable named output_directory. This function does not take any parameters and is designed to provide a centralized way to retrieve the output directory used throughout the application. 

In the context of the project, this function is called by several classes within the ldm_patched/contrib/external.py and ldm_patched/contrib/external_model_merging.py modules. For instance, in the SaveLatent, SaveImage, SaveAnimatedWEBP, SaveAnimatedPNG, CheckpointSave, CLIPSave, and VAESave classes, the output_dir attribute is initialized by calling get_output_directory. This indicates that these classes rely on the output directory for their operations, likely to save files or data generated during their execution.

Additionally, the get_output_directory function is also utilized in other functions within the path_utils.py module, such as get_directory_by_type and annotated_filepath. In get_directory_by_type, it is called when the type_name is "output", ensuring that the correct output directory is returned based on the specified type. In annotated_filepath, it is used to determine the base directory when the name ends with "[output]". This further emphasizes the importance of the output_directory in the overall functionality of the project.

**Note**: It is important to ensure that the global variable output_directory is properly initialized before calling this function to avoid returning an undefined value.

**Output Example**: An example of the possible return value of this function could be a string representing a file path, such as "/home/user/project/output".
## FunctionDef get_temp_directory
**get_temp_directory**: The function of get_temp_directory is to return the global temporary directory path.

**parameters**: The parameters of this Function.
· There are no parameters for this function.

**Code Description**: The get_temp_directory function is a simple utility that retrieves the value of a global variable named temp_directory. This function does not take any parameters and directly accesses the global scope to return the current value of temp_directory. 

In the context of its usage within the project, get_temp_directory is called in multiple places. For instance, in the __init__ method of the PreviewImage class located in ldm_patched/contrib/external.py, it is used to initialize the output_dir attribute. This indicates that the temporary directory is likely intended for storing intermediate or temporary files related to image processing tasks.

Additionally, the function is also called in get_directory_by_type, which is defined in the same path_utils.py module. This function checks the type of directory requested and returns the appropriate directory path based on the type name provided. When the type name is "temp", it calls get_temp_directory to obtain the path for temporary files.

Moreover, get_temp_directory is utilized in the annotated_filepath function, which processes file names to determine their associated base directories based on specific suffixes. If the name ends with "[temp]", it retrieves the temporary directory path using get_temp_directory.

**Note**: It is important to ensure that the global variable temp_directory is properly initialized before calling this function to avoid returning an undefined value.

**Output Example**: An example of the possible return value of get_temp_directory could be a string representing a file path, such as "/tmp/my_temp_directory".
## FunctionDef get_input_directory
**get_input_directory**: The function of get_input_directory is to return the current input directory path used in the application.

**parameters**: The parameters of this Function.
· There are no parameters for this function.

**Code Description**: The get_input_directory function is a simple utility that accesses a global variable named input_directory and returns its value. This function is crucial for retrieving the path where input files are expected to be located within the application. The global variable input_directory must be defined elsewhere in the code for this function to operate correctly.

This function is called by several other functions within the project, specifically in the INPUT_TYPES methods of the LoadLatent, LoadImage, and LoadImageMask classes located in the ldm_patched/contrib/external.py file. Each of these methods utilizes get_input_directory to obtain the path to the input directory, which is then used to list files that match specific criteria (e.g., files with a ".latent" extension for LoadLatent, or all files for LoadImage and LoadImageMask). This demonstrates the function's role in providing a centralized way to access the input directory, ensuring consistency across different parts of the code that require access to input files.

Additionally, get_input_directory is also referenced in other utility functions within the path_utils module, such as get_directory_by_type, annotated_filepath, get_annotated_filepath, and exists_annotated_filepath. These functions rely on get_input_directory to determine the appropriate base directory for file operations, further emphasizing its importance in the overall file management system of the application.

**Note**: It is essential to ensure that the global variable input_directory is properly initialized before calling this function to avoid unexpected behavior or errors.

**Output Example**: A possible return value of the get_input_directory function could be a string representing a file path, such as "/path/to/input/directory".
## FunctionDef get_directory_by_type(type_name)
**get_directory_by_type**: The function of get_directory_by_type is to return the appropriate directory path based on the specified type name.

**parameters**: The parameters of this Function.
· type_name: A string that specifies the type of directory to retrieve. It can be "output", "temp", or "input".

**Code Description**: The get_directory_by_type function serves as a utility to retrieve directory paths based on the provided type_name parameter. It checks the value of type_name against three predefined options: "output", "temp", and "input". Depending on the value of type_name, the function calls one of three other functions: get_output_directory, get_temp_directory, or get_input_directory.

- If type_name is "output", the function invokes get_output_directory, which returns the global output directory path. This is essential for operations that require saving output files.
- If type_name is "temp", it calls get_temp_directory, which provides the path to a temporary directory used for intermediate file storage.
- If type_name is "input", the function calls get_input_directory to obtain the path for the input directory, where files to be processed are located.

If the type_name does not match any of the specified options, the function returns None, indicating that an invalid type was provided. This design ensures that the function only returns valid directory paths based on recognized types, thereby maintaining the integrity of directory management within the application.

The get_directory_by_type function is integral to the overall file management system in the project, as it centralizes the logic for retrieving directory paths. It is called in various parts of the code where different types of directories are needed, ensuring consistency and reducing redundancy in directory access.

**Note**: It is important to ensure that the global variables corresponding to the output_directory, temp_directory, and input_directory are properly initialized before calling this function to avoid returning undefined values.

**Output Example**: A possible return value of the get_directory_by_type function could be a string representing a file path, such as "/home/user/project/output" if type_name is "output", "/tmp/my_temp_directory" if type_name is "temp", or "/path/to/input/directory" if type_name is "input". If an invalid type_name is provided, the function would return None.
## FunctionDef annotated_filepath(name)
**annotated_filepath**: The function of annotated_filepath is to process a given file name and determine its associated base directory based on specific suffixes.

**parameters**: The parameters of this Function.
· name: A string representing the file name that may include an annotation indicating its type (e.g., "[output]", "[input]", or "[temp]").

**Code Description**: The annotated_filepath function analyzes the provided name to identify if it ends with specific suffixes that denote the type of directory to be used. If the name ends with "[output]", it calls the get_output_directory function to retrieve the global output directory and removes the suffix from the name. Similarly, if the name ends with "[input]", it invokes get_input_directory to obtain the input directory and strips the suffix. If the name ends with "[temp]", it uses get_temp_directory to get the temporary directory and also removes the suffix. If the name does not match any of these suffixes, the function returns the original name along with a None value for the base directory.

The function is utilized by other functions within the path_utils module, specifically get_annotated_filepath and exists_annotated_filepath. In get_annotated_filepath, annotated_filepath is called to determine the base directory for constructing a full file path. If no base directory is found, it falls back to a default directory or retrieves the input directory. In exists_annotated_filepath, annotated_filepath is similarly called to check if a file exists at the constructed path based on the processed name and determined base directory.

This function plays a crucial role in managing file paths within the application, ensuring that files are correctly located based on their intended use (output, input, or temporary).

**Note**: It is important to ensure that the global directory variables (output_directory, input_directory, temp_directory) are properly initialized before calling this function to avoid returning undefined values.

**Output Example**: A possible return value of the annotated_filepath function could be a tuple such as ("my_file", "/home/user/project/output") if the input name was "my_file[output]".
## FunctionDef get_annotated_filepath(name, default_dir)
**get_annotated_filepath**: The function of get_annotated_filepath is to construct a full file path based on a given name and an optional default directory.

**parameters**: The parameters of this Function.
· name: A string representing the file name that may include an annotation indicating its type (e.g., "[output]", "[input]", or "[temp]").
· default_dir: An optional string representing a default directory path to use if no base directory is determined from the name.

**Code Description**: The get_annotated_filepath function begins by calling the annotated_filepath function with the provided name. This function processes the name to determine if it contains any specific suffixes that indicate the type of directory to be used (e.g., output, input, or temporary). The annotated_filepath function returns a tuple consisting of the processed name and its associated base directory.

If the base directory returned by annotated_filepath is None, get_annotated_filepath checks if a default directory has been provided. If a default directory is specified, it uses that as the base directory. If no default directory is provided, it calls the get_input_directory function to retrieve the current input directory path used in the application as a fallback.

Finally, the function constructs and returns the full file path by joining the determined base directory with the processed name using the os.path.join method.

This function is utilized in several places within the project, specifically in the LoadLatent, LoadImage, and LoadImageMask classes located in the ldm_patched/contrib/external.py file. In the load method of LoadLatent, get_annotated_filepath is called to obtain the path of the latent file, which is then used to load the file into memory. Similarly, in the load_image method of LoadImage and load_image method of LoadImageMask, get_annotated_filepath is called to retrieve the path of the image file, ensuring that the correct file is accessed based on the provided name.

The function plays a crucial role in managing file paths within the application, ensuring that files are correctly located based on their intended use and providing a consistent way to access files across different components of the project.

**Note**: It is essential to ensure that the global directory variables (output_directory, input_directory, temp_directory) are properly initialized before calling this function to avoid returning undefined values.

**Output Example**: A possible return value of the get_annotated_filepath function could be a string representing a file path, such as "/path/to/input/directory/my_file".
## FunctionDef exists_annotated_filepath(name)
**exists_annotated_filepath**: The function of exists_annotated_filepath is to check if a file with a specified name exists in an appropriate directory, which may be determined based on annotations in the file name.

**parameters**: The parameters of this Function.
· name: A string representing the file name that may include an annotation indicating its type (e.g., "[output]", "[input]", or "[temp]").

**Code Description**: The exists_annotated_filepath function begins by calling the annotated_filepath function with the provided name parameter. This function processes the name to determine if it includes any annotations that specify a base directory. If the name has an annotation, annotated_filepath will return the processed name and the corresponding base directory. If no base directory is found (i.e., it returns None), the function defaults to using the directory returned by the get_input_directory function, which retrieves the current input directory path used in the application.

Once the base directory is established, the function constructs the full file path by joining the base directory with the processed name using the os.path.join method. Finally, it checks for the existence of the constructed file path using os.path.exists and returns a boolean value indicating whether the file exists.

This function is utilized in several validation methods within the LoadLatent, LoadImage, and LoadImageMask classes located in the ldm_patched/contrib/external.py file. Specifically, these classes call exists_annotated_filepath to validate the presence of necessary files (latent files for LoadLatent and image files for LoadImage and LoadImageMask) before proceeding with their operations. If the file does not exist, a message indicating the invalid file is returned.

The relationship between exists_annotated_filepath and its callees emphasizes its role in ensuring that the application has access to the required files, thereby preventing errors that could arise from missing resources.

**Note**: It is important to ensure that the global directory variables (output_directory, input_directory, temp_directory) are properly initialized before calling the annotated_filepath function to avoid returning undefined values.

**Output Example**: A possible return value of the exists_annotated_filepath function could be a boolean value such as True if the file exists at the constructed path, or False if it does not.
## FunctionDef add_model_folder_path(folder_name, full_folder_path)
**add_model_folder_path**: The function of add_model_folder_path is to manage the association between folder names and their corresponding full folder paths in a global dictionary.

**parameters**: The parameters of this Function.
· parameter1: folder_name - A string representing the name of the folder to be added or updated in the global dictionary.
· parameter2: full_folder_path - A string representing the complete path of the folder that corresponds to the folder_name.

**Code Description**: The add_model_folder_path function is designed to update a global dictionary named folder_names_and_paths, which maps folder names to their respective paths. When the function is called, it first checks if the provided folder_name already exists in the global dictionary. If it does, the function appends the full_folder_path to the list of paths associated with that folder_name. This allows for multiple paths to be associated with a single folder name. If the folder_name does not exist in the dictionary, the function initializes a new entry with the folder_name as the key. The value for this key is a tuple containing a list with the full_folder_path and an empty set. The empty set is presumably reserved for future use, possibly to store additional metadata or properties related to the folder.

**Note**: It is important to ensure that the global variable folder_names_and_paths is properly initialized before calling this function. Additionally, users should be aware that the function does not perform any validation on the folder_name or full_folder_path inputs, so it is the caller's responsibility to provide valid strings.
## FunctionDef get_folder_paths(folder_name)
**get_folder_paths**: The function of get_folder_paths is to retrieve a list of folder paths associated with a specified folder name.

**parameters**: The parameters of this Function.
· folder_name: A string representing the name of the folder for which paths are to be retrieved.

**Code Description**: The get_folder_paths function accesses a predefined data structure, referred to as `folder_names_and_paths`, to obtain the paths associated with the provided folder_name. It specifically returns a copy of the first element of the list corresponding to the folder_name key in the `folder_names_and_paths` dictionary. This function is crucial for various components within the project that require access to specific folder paths, particularly for loading resources or configurations.

The function is called by several other components in the project, such as load_checkpoint methods in different loaders (e.g., CheckpointLoader, DiffusersLoader, unCLIPCheckpointLoader). In these contexts, get_folder_paths is used to obtain the paths for "embeddings" or "diffusers," which are essential for loading models and checkpoints correctly. By providing these paths, the function facilitates the seamless integration of resources needed for model operations, ensuring that the loaders can locate the necessary files during execution.

**Note**: It is important to ensure that the folder_name passed to the function exists in the `folder_names_and_paths` dictionary; otherwise, it may lead to a KeyError.

**Output Example**: An example of the output from get_folder_paths("embeddings") could be a list of strings such as:
["/path/to/embeddings/folder1", "/path/to/embeddings/folder2"]
## FunctionDef recursive_search(directory, excluded_dir_names)
**recursive_search**: The function of recursive_search is to perform a recursive search through a specified directory, returning a list of relative file paths and a dictionary of directories with their last modified times.

**parameters**: The parameters of this Function.
· parameter1: directory - A string representing the path of the directory to search within. This must be a valid directory path.
· parameter2: excluded_dir_names - An optional list of directory names that should be excluded from the search. If not provided, defaults to an empty list.

**Code Description**: The recursive_search function begins by checking if the provided directory is indeed a valid directory using os.path.isdir. If the directory is invalid, it returns an empty list and an empty dictionary. If the excluded_dir_names parameter is not provided, it initializes it as an empty list.

The function initializes two variables: result, which will store the relative paths of the files found, and dirs, which will store the directories along with their last modified times. It attempts to add the initial directory to the dirs dictionary while handling potential errors, such as a FileNotFoundError, which may occur if the directory cannot be accessed.

The function then utilizes os.walk to traverse the directory structure. It filters out any subdirectories that are listed in excluded_dir_names. For each file found, it calculates the relative path from the base directory and appends it to the result list. For each subdirectory, it attempts to retrieve and store its last modified time in the dirs dictionary, again handling any access errors gracefully.

The function ultimately returns a tuple containing the list of relative file paths and the dictionary of directories with their last modified times.

This function is called by the get_filename_list_ function, which is responsible for gathering a list of filenames from specified folders. Within get_filename_list_, the recursive_search function is invoked for each folder in the folder_names_and_paths global variable, excluding directories named ".git". The results from recursive_search are then filtered and combined to produce a sorted list of filenames and a comprehensive dictionary of folder paths and their modification times.

**Note**: When using this function, ensure that the directory path provided is valid and accessible. Be aware that any directories specified in excluded_dir_names will not be included in the search results.

**Output Example**: An example of the return value from recursive_search could be:
(
    ['subfolder/file1.txt', 'subfolder/file2.txt', 'anotherfolder/file3.txt'],
    {
        '/path/to/directory': 1633072800.0,
        '/path/to/directory/subfolder': 1633072900.0,
        '/path/to/directory/anotherfolder': 1633073000.0
    }
)
## FunctionDef filter_files_extensions(files, extensions)
**filter_files_extensions**: The function of filter_files_extensions is to filter a list of files based on specified file extensions.

**parameters**: The parameters of this Function.
· parameter1: files - A list of file names (strings) that need to be filtered.
· parameter2: extensions - A list of file extensions (strings) that will be used to filter the files. If this list is empty, all files will be included.

**Code Description**: The filter_files_extensions function takes two arguments: a list of file names and a list of desired file extensions. It uses the built-in filter function along with a lambda expression to check each file's extension against the provided list of extensions. The os.path.splitext function is utilized to extract the file extension from each file name, and the lower method ensures that the comparison is case-insensitive. The function returns a sorted list of the filtered file names. 

This function is called within the get_filename_list_ function, which is responsible for retrieving a list of files from specified folders. In get_filename_list_, the filter_files_extensions function is used to refine the list of files obtained from a recursive search, ensuring that only files with the specified extensions are included in the final output. This relationship highlights the utility of filter_files_extensions in managing file types within the broader context of file retrieval operations.

**Note**: It is important to ensure that the extensions parameter is provided in a consistent format (e.g., including the leading dot) to achieve the desired filtering results. If the extensions list is empty, all files will be returned.

**Output Example**: Given a list of files `["document.pdf", "image.jpg", "script.py"]` and extensions `[".jpg", ".pdf"]`, the function would return `["document.pdf", "image.jpg"]`.
## FunctionDef get_full_path(folder_name, filename)
**get_full_path**: The function of get_full_path is to retrieve the full file path of a specified file within a given folder, if it exists.

**parameters**: The parameters of this Function.
· folder_name: A string representing the name of the folder where the file is expected to be located.
· filename: A string representing the name of the file for which the full path is being requested.

**Code Description**: The get_full_path function is designed to locate and return the full path of a specified file within a specified folder. It begins by checking if the provided folder_name exists in the global variable folder_names_and_paths. If the folder_name is not found, the function returns None, indicating that the folder does not exist.

If the folder_name is valid, the function retrieves the list of paths associated with that folder. It then constructs a relative path for the filename by joining it with a root directory ("/"). The function iterates through the list of folder paths, creating a full path for each folder by combining the folder path with the filename. It checks if the constructed full path corresponds to an existing file using os.path.isfile. If a valid file path is found, it is returned as the output of the function.

In the context of the project, get_full_path is called by several functions across different modules, primarily to load various types of models and configurations. For example, in the load_checkpoint methods of the CheckpointLoader, CheckpointLoaderSimple, and unCLIPCheckpointLoader classes, get_full_path is used to obtain the paths for configuration files and checkpoint files. Similarly, it is utilized in the load_lora method of the LoraLoader class to find the path for lora files, and in the load_vae method of the VAELoader class to locate VAE files. This demonstrates that get_full_path plays a crucial role in ensuring that the necessary files are correctly located and loaded for various functionalities within the project.

**Note**: It is important to ensure that the folder names and file names provided to this function are accurate and correspond to the expected structure in the project, as any discrepancies will result in a return value of None.

**Output Example**: An example of a possible return value from get_full_path could be "/path/to/project/configs/example_config.json" if the folder "configs" contains the file "example_config.json". If the file does not exist, the function would return None.
## FunctionDef get_filename_list_(folder_name)
**get_filename_list_**: The function of get_filename_list_ is to gather a sorted list of filenames from specified folders while also returning a dictionary of folder paths and their last modified times.

**parameters**: The parameters of this Function.
· parameter1: folder_name - A string representing the name of the folder from which to retrieve the filenames. This name corresponds to a key in the global variable folder_names_and_paths.

**Code Description**: The get_filename_list_ function operates by first accessing a global variable named folder_names_and_paths, which is expected to contain mappings of folder names to their respective paths. It initializes an empty set called output_list to store unique filenames and an empty dictionary output_folders to store folder paths along with their last modified times.

The function retrieves the list of folders associated with the provided folder_name from the global variable. It then iterates over each folder in this list. For each folder, it calls the recursive_search function, which performs a recursive search through the specified directory, returning a list of file paths and a dictionary of directories with their last modified times. The recursive_search function is called with an excluded_dir_names parameter set to exclude any directories named ".git".

The results from recursive_search are processed by the filter_files_extensions function, which filters the files based on specified extensions. The filtered filenames are added to the output_list, ensuring that only the desired file types are included. The output_folders dictionary is updated with the results from the recursive_search function, merging the existing entries with the new ones.

Finally, the function returns a tuple containing a sorted list of unique filenames, the updated dictionary of folder paths and their last modified times, and the elapsed time measured using time.perf_counter().

This function is called by get_filename_list, which serves as a caching layer. If the cached result for the specified folder_name is not found, it invokes get_filename_list_ to perform the actual file retrieval and caching the result for future use.

**Note**: When using this function, ensure that the folder_name provided corresponds to a valid key in the folder_names_and_paths global variable. The function relies on the proper configuration of this variable to function correctly.

**Output Example**: An example of the return value from get_filename_list_ could be:
(
    ['file1.txt', 'file2.jpg', 'script.py'],
    {
        '/path/to/folder1': 1633072800.0,
        '/path/to/folder2': 1633072900.0
    },
    0.123456
)
## FunctionDef cached_filename_list_(folder_name)
**cached_filename_list_**: The function of cached_filename_list_ is to retrieve a cached list of filenames associated with a specified folder name, validating the cache against the current state of the filesystem.

**parameters**: The parameters of this Function.
· folder_name: A string representing the name of the folder for which the cached filename list is requested.

**Code Description**: The cached_filename_list_ function begins by checking if the provided folder_name exists in the global cache, filename_list_cache. If the folder_name is not found in the cache, the function returns None, indicating that there is no cached data available for that folder.

If the folder_name is present in the cache, the function retrieves the associated cached data, which is expected to be a tuple containing two elements: the first element is likely a list of filenames, and the second element is a dictionary mapping filenames to their last modified timestamps.

The function then iterates over the keys of the timestamp dictionary to verify that the current last modified time of each folder matches the cached timestamp. This is done using the os.path.getmtime function. If any folder's last modified time does not match the cached value, the function returns None, indicating that the cache is no longer valid.

Next, the function retrieves the list of folder paths associated with the folder_name from the global folder_names_and_paths dictionary. It checks if each path in the first element of this list is a directory and ensures that all directories are present in the cached output. If any directory is missing from the cache, the function returns None.

If all checks pass, the function returns the cached output, which includes the list of filenames and their corresponding timestamps.

This function is called by get_filename_list, which first attempts to retrieve the cached filename list using cached_filename_list_. If the cached data is not available (i.e., the function returns None), get_filename_list proceeds to call another function, get_filename_list_, to generate a fresh list of filenames, which is then cached for future use. This relationship ensures that filename retrieval is efficient by utilizing caching where possible.

**Note**: It is important to ensure that the global variables filename_list_cache and folder_names_and_paths are properly initialized and maintained, as the function relies on these for its operation.

**Output Example**: An example of the return value of cached_filename_list_ could be:
```
(['file1.txt', 'file2.txt'], {'file1.txt': 1633072800, 'file2.txt': 1633072860})
```
This output indicates that the cached filenames are 'file1.txt' and 'file2.txt', with their respective last modified timestamps.
## FunctionDef get_filename_list(folder_name)
**get_filename_list**: The function of get_filename_list is to retrieve a list of filenames from a specified folder, utilizing caching to improve efficiency.

**parameters**: The parameters of this Function.
· folder_name: A string representing the name of the folder from which to retrieve the filenames. This name corresponds to a key in the global variable folder_names_and_paths.

**Code Description**: The get_filename_list function operates by first attempting to retrieve a cached list of filenames associated with the provided folder_name through the cached_filename_list_ function. If the cached result is not available (i.e., the output is None), it calls the get_filename_list_ function to generate a fresh list of filenames from the specified folder. The results obtained from get_filename_list_ are then stored in a global cache, filename_list_cache, for future retrieval.

The get_filename_list_ function is responsible for gathering a sorted list of filenames from specified folders while also returning a dictionary of folder paths and their last modified times. It accesses the global variable folder_names_and_paths to find the paths associated with the given folder_name and performs a recursive search through these directories to collect the filenames.

The caching mechanism ensures that subsequent calls to get_filename_list with the same folder_name will return the cached results quickly, avoiding the need to repeatedly access the filesystem unless necessary. This enhances performance, particularly when dealing with large directories or frequent requests for the same folder.

The get_filename_list function is called by various components within the project, including the INPUT_TYPES methods of different loaders (e.g., CheckpointLoader, LoraLoader, ControlNetLoader) and functions like vae_list and load_taesd. These components rely on get_filename_list to dynamically retrieve the available filenames from specific folders, ensuring that they operate with the most current data.

**Note**: When using this function, it is essential to ensure that the folder_name provided corresponds to a valid key in the folder_names_and_paths global variable. The function relies on the proper configuration of this variable to function correctly.

**Output Example**: An example of the return value from get_filename_list could be:
```
['config1.json', 'config2.yaml', 'config3.ini']
```
## FunctionDef get_save_image_path(filename_prefix, output_dir, image_width, image_height)
**get_save_image_path**: The function of get_save_image_path is to generate a valid file path for saving images, ensuring that the path adheres to specified constraints and formats.

**parameters**: The parameters of this Function.
· filename_prefix: A string that serves as the base name for the files to be saved.
· output_dir: A string representing the directory where the files will be saved.
· image_width: An optional integer indicating the width of the image (default is 0).
· image_height: An optional integer indicating the height of the image (default is 0).

**Code Description**: The get_save_image_path function is designed to create a structured file path for saving images based on a specified filename prefix and output directory. It begins by defining a helper function, map_filename, which extracts a numeric counter from existing filenames in the target directory. This counter is used to ensure that new files do not overwrite existing ones.

The function also includes compute_vars, which replaces placeholders in the filename prefix with the actual image dimensions. The filename prefix is then processed to determine the subfolder and base filename. The full output folder is constructed by joining the output directory with the subfolder derived from the filename prefix.

A critical check is performed to ensure that the constructed path does not lead to saving files outside the specified output directory. If the path is invalid, an error message is generated, and an exception is raised.

The function attempts to find the highest existing counter for files that match the generated filename pattern. If no such files exist, it initializes the counter to 1. If the output folder does not exist, it creates the necessary directories.

Finally, the function returns a tuple containing the full output folder path, the base filename, the counter for the next file, the subfolder, and the processed filename prefix.

This function is called by various methods within the project, such as SaveLatent.save, SaveImage.save_images, and others. Each of these methods utilizes get_save_image_path to ensure that images or model checkpoints are saved in a structured manner, preventing file conflicts and maintaining organization within the output directory.

**Note**: It is essential to ensure that the output directory is correctly specified to avoid errors related to invalid paths. The function also assumes that the filename prefix is formatted correctly to extract the necessary components.

**Output Example**: A possible return value from the function could be:
('/path/to/output/folder/subfolder', 'ldm_patched', 1, 'subfolder', 'ldm_patched')
### FunctionDef map_filename(filename)
**map_filename**: The function of map_filename is to extract a numerical identifier and a prefix from a given filename.

**parameters**: The parameters of this Function.
· filename: A string representing the full name of the file from which the prefix and numerical identifier will be extracted.

**Code Description**: The map_filename function takes a single parameter, filename, which is expected to be a string. The function first calculates the length of a predefined variable, filename_prefix, which is not defined within the function but is assumed to be accessible in the surrounding scope. It extracts the prefix from the filename by slicing it up to the length of filename_prefix plus one additional character (likely to include a separator). 

Next, the function attempts to extract a numerical identifier from the filename. It does this by taking the substring that follows the prefix and splitting it at underscores ('_'). The first element of this split string is then converted to an integer. If this conversion fails (for example, if the substring does not start with digits), the function catches the exception and assigns a default value of 0 to digits. Finally, the function returns a tuple containing the extracted digits and the prefix.

**Note**: It is important to ensure that the variable filename_prefix is defined in the surrounding scope before calling this function, as its value is critical for the correct operation of the function. Additionally, the function assumes that the filename follows a specific format where the prefix is followed by an underscore and then a numerical identifier.

**Output Example**: If the input filename is "image_1234.png" and filename_prefix is "image", the function would return (1234, "image"). If the input filename is "image_xyz.png", it would return (0, "image").
***
### FunctionDef compute_vars(input, image_width, image_height)
**compute_vars**: The function of compute_vars is to replace placeholder strings in the input with specified image dimensions.

**parameters**: The parameters of this Function.
· parameter1: input - A string that may contain placeholders for width and height.
· parameter2: image_width - An integer representing the width of the image to be inserted into the input string.
· parameter3: image_height - An integer representing the height of the image to be inserted into the input string.

**Code Description**: The compute_vars function takes an input string that may contain the placeholders "%width%" and "%height%". It replaces these placeholders with the actual values of image_width and image_height, respectively. The function first uses the string method `replace()` to substitute "%width%" with the string representation of image_width. It then performs a similar replacement for "%height%" with the string representation of image_height. Finally, the modified input string is returned. This function is useful for dynamically generating strings that include specific image dimensions, which can be particularly helpful in scenarios such as generating HTML or CSS styles where image sizes are required.

**Note**: It is important to ensure that the input string contains the placeholders "%width%" and "%height%" for the replacements to take effect. If these placeholders are not present, the function will return the input string unchanged.

**Output Example**: If the input is "The image size is %width% x %height% pixels." with image_width set to 800 and image_height set to 600, the function will return "The image size is 800 x 600 pixels."
***
