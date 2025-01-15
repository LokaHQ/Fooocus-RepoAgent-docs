## FunctionDef makedirs_with_log(path)
**makedirs_with_log**: The function of makedirs_with_log is to create a directory at the specified path, logging an error message if the directory cannot be created.

**parameters**: The parameters of this Function.
· path: A string representing the directory path that needs to be created.

**Code Description**: The makedirs_with_log function attempts to create a directory at the specified path using the os.makedirs method. The parameter `exist_ok=True` allows the function to not raise an error if the directory already exists. If an OSError occurs during the directory creation process, the function catches this exception and prints a message indicating that the directory could not be created, along with the reason for the failure.

This function is called within the get_dir_or_set_default function located in the modules/config.py file. Specifically, it is invoked when the `make_directory` parameter is set to True. In this context, if the configuration value associated with a given key is a string or a list, makedirs_with_log is called to ensure that the specified directory or directories exist. This ensures that the application can safely proceed with operations that depend on the existence of these directories, thereby enhancing the robustness of the configuration handling process.

**Note**: It is important to ensure that the path provided to makedirs_with_log is valid and that the application has the necessary permissions to create directories in the specified location.
## FunctionDef get_files_from_folder(folder_path, extensions, name_filter)
**get_files_from_folder**: The function of get_files_from_folder is to retrieve a list of file paths from a specified directory, optionally filtering by file extensions and name patterns.

**parameters**: The parameters of this Function.
· folder_path: A string representing the path to the directory from which files are to be retrieved. This must be a valid directory path.
· extensions: An optional list of strings representing the file extensions to filter the retrieved files. If None, all file types are included.
· name_filter: An optional string used to filter the filenames based on a substring match. If None, all filenames are included.

**Code Description**: The get_files_from_folder function begins by validating the provided folder_path to ensure it is a valid directory. If the path is invalid, a ValueError is raised. The function then initializes an empty list called filenames to store the paths of the files that meet the specified criteria.

Using the os.walk method, the function traverses the directory tree starting from folder_path. For each directory it encounters, it constructs a relative path and iterates through the files within that directory. Each filename is sorted in a case-insensitive manner. The function checks if the file's extension is in the provided extensions list (if specified) and if the name_filter (if provided) is present in the directory name. If both conditions are satisfied, the full path of the file is constructed and added to the filenames list.

The function ultimately returns the filenames list, which contains the paths of all files that match the specified criteria.

This function is called by get_model_filenames and update_files in the modules/config.py file. The get_model_filenames function uses get_files_from_folder to gather model files from specified folder paths, applying default extensions if none are provided. The update_files function calls get_files_from_folder to retrieve wildcard filenames from a specific path, ensuring that only text files are included. This demonstrates the utility of get_files_from_folder in managing and organizing file retrieval across different contexts within the project.

**Note**: It is important to ensure that the folder_path provided is valid and accessible. The extensions and name_filter parameters are optional, allowing for flexible usage depending on the specific needs of the file retrieval process.

**Output Example**: An example return value of the function could be:
```
['subfolder1/file1.txt', 'subfolder2/file2.pth', 'file3.ckpt']
```
## FunctionDef try_eval_env_var(value, expected_type)
**try_eval_env_var**: The function of try_eval_env_var is to evaluate a string representation of a value into its corresponding Python data type, based on an optional expected type.

**parameters**: The parameters of this Function.
· parameter1: value (str) - A string that represents the value to be evaluated.
· parameter2: expected_type (type, optional) - The expected Python type that the evaluated value should conform to. If provided, the function will validate the type of the evaluated value against this expected type.

**Code Description**: The try_eval_env_var function attempts to convert a string input into its corresponding Python data type using the `literal_eval` function from the `ast` module. If the expected_type parameter is specified, the function checks whether the evaluated value matches this type. If the conversion is successful and the type matches (if expected_type is provided), the evaluated value is returned. If any exception occurs during the evaluation or if the type does not match, the original string value is returned.

This function is particularly useful in scenarios where configuration values are retrieved from environment variables, as seen in its usage within the get_config_item_or_set_default function in the modules/config.py file. In that context, the function is called to evaluate environment variable values before they are stored in a global configuration dictionary. This ensures that the values are correctly interpreted as their intended types, such as integers, booleans, or lists, rather than remaining as plain strings.

The try_eval_env_var function is also tested in the tests/test_extra_utils.py file, where various test cases are defined to validate its behavior against different input scenarios. These tests confirm that the function correctly evaluates strings representing different data types and handles both successful evaluations and exceptions appropriately.

**Note**: It is important to ensure that the input string is a valid representation of the expected data type, as improper formatting may lead to exceptions during evaluation. Additionally, when using the expected_type parameter, the function will return the original string if the evaluated value does not match the specified type.

**Output Example**: 
- Input: ("1", int) 
- Output: 1
- Input: ("true", bool) 
- Output: True
- Input: ("['a', 'b', 'c']", list) 
- Output: ['a', 'b', 'c']
