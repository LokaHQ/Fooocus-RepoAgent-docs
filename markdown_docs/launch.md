## FunctionDef prepare_environment
**prepare_environment**: The function of prepare_environment is to set up the Python environment by installing necessary packages and verifying their versions.

**parameters**: The parameters of this Function.
· None

**Code Description**: The prepare_environment function is responsible for configuring the Python environment by checking for the installation of essential packages and installing them if they are not present. It begins by retrieving environment variables that specify the index URL for PyTorch, the command to install PyTorch and torchvision, and the path to a requirements file. If these environment variables are not set, default values are used.

The function prints the current Python version and the version of the Fooocus library. It then checks if the packages "torch" and "torchvision" are installed using the is_installed function. If either package is not installed or if the REINSTALL_ALL flag is set to True, it executes a pip command to install the specified versions of these packages using the run function. The run function handles the execution of the command and manages any errors that may arise during installation.

Next, if the TRY_INSTALL_XFORMERS flag is set to True, the function checks for the installation of the "xformers" package. Similar to the previous checks, it uses is_installed to determine if "xformers" is present. If it is not installed or if REINSTALL_ALL is True, it attempts to install the package based on the operating system. For Windows, it checks the Python version and provides instructions for manual installation if the version is unsupported. For Linux, it proceeds to install the package using run_pip.

Finally, the function checks if the requirements specified in the requirements file are met by calling the requirements_met function. If the requirements are not satisfied, it installs the packages listed in the requirements file using run_pip.

This function is integral to the setup process of the application, ensuring that all necessary dependencies are installed and correctly configured before the application is run.

**Note**: It is important to ensure that the environment variables are correctly set and that the requirements file is accessible. The function relies on the presence of the specified packages in the environment to verify their installation status.

**Output Example**: The function does not return a value, but it may produce output such as:
```
Python 3.8.10
Fooocus version: 1.0.0
Installing torch and torchvision
Successfully installed torch-2.1.0 torchvision-0.16.0
```
## FunctionDef ini_args
**ini_args**: The function of ini_args is to retrieve and return the argument settings managed by the args_manager module.

**parameters**: The parameters of this Function.
· There are no parameters for this function.

**Code Description**: The ini_args function is a simple utility designed to import and return the configuration settings related to command-line arguments. When invoked, it imports the 'args' object from the 'args_manager' module. This 'args' object is expected to contain the parsed command-line arguments or configurations that are essential for the application's operation. The function does not take any parameters and directly returns the 'args' object, allowing other parts of the application to access the argument settings without needing to manage the import themselves.

**Note**: It is important to ensure that the 'args_manager' module is correctly implemented and accessible within the project. If the module is not found or if there are issues with the 'args' object, the function may raise an ImportError or return an unexpected result.

**Output Example**: A possible return value of the ini_args function could be an object or dictionary containing command-line arguments, such as:
{
    "verbose": true,
    "output": "results.txt",
    "input": "data.csv"
}
## FunctionDef download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads)
**download_models**: The function of download_models is to manage the downloading of various model files based on specified parameters, ensuring that the necessary models are available for use in the application.

**parameters**: The parameters of this Function.
· default_model: A string representing the name of the default model to be used.
· previous_default_models: A list of strings containing names of previously used default models.
· checkpoint_downloads: A dictionary mapping file names to their respective URLs for checkpoint models to be downloaded.
· embeddings_downloads: A dictionary mapping file names to their respective URLs for embeddings to be downloaded.
· lora_downloads: A dictionary mapping file names to their respective URLs for LoRA models to be downloaded.
· vae_downloads: A dictionary mapping file names to their respective URLs for VAE models to be downloaded.

**Code Description**: The download_models function is responsible for downloading various model files required by the application. It begins by importing the get_file_from_folder_list function from the modules.util module, which is used to locate files within specified directories.

The function first attempts to download VAE approximation files by iterating over a predefined list of file names and URLs. It utilizes the load_file_from_url function to download each file to the specified directory.

Next, the function checks if the model download has been disabled through the args.disable_preset_download flag. If this flag is set, the function prints a message indicating that the model download has been skipped and returns the current default model along with the checkpoint downloads.

If the download is not disabled, the function checks whether the default model file exists in the specified checkpoint paths. If the file is not found, it looks for alternative models in the previous_default_models list. If an alternative model is found, it updates the default model to this alternative and informs the user that they are not using the latest model.

Subsequently, the function proceeds to download the specified checkpoint models, embeddings, LoRA models, and VAE models by iterating over their respective dictionaries and calling load_file_from_url for each file.

Finally, the function returns the updated default model and the checkpoint downloads.

The download_models function is called within the preset_selection_change function in webui.py. This function prepares the necessary parameters for download_models by extracting model information from a preset. It then invokes download_models to ensure that the required models are downloaded and available for use in the application.

**Note**: It is important to ensure that the necessary permissions are set for the directories where models are being downloaded. Additionally, users should be aware of the implications of the args.disable_preset_download and args.always_download_new_model flags, as they control the behavior of model downloading.

**Output Example**: A possible return value of the function could be: ("base_model_name", {"checkpoint_model_name": "checkpoint_url"}), indicating the updated default model and the dictionary of checkpoint downloads.
