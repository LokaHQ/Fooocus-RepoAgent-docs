## FunctionDef get_current_html_path(output_format)
**get_current_html_path**: The function of get_current_html_path is to generate the file path for the current HTML log file.

**parameters**: The parameters of this Function.
· output_format: An optional string that specifies the desired file extension for the HTML log file. If not provided, it defaults to the value defined in the project's configuration.

**Code Description**: The get_current_html_path function is designed to create a path for an HTML log file. It first checks if the output_format parameter is provided; if not, it uses the default output format specified in the project's configuration. The function then calls the generate_temp_filename function from the modules/util.py file, passing in the output directory and the specified file extension. This function generates a unique temporary filename based on the current date and time, ensuring that the filename is unique by appending a random number.

The output of the generate_temp_filename function includes a date string, the full path of the temporary file, and the filename itself. The get_current_html_path function then constructs the path for the HTML log file by joining the directory of the temporary filename with the string 'log.html'. Finally, it returns the complete path to the HTML log file.

This function is utilized in the update_history_link function within the webui.py file. In this context, get_current_html_path is called to retrieve the path of the current HTML log file, which is then embedded in an HTML anchor tag. This allows users to access the log file directly from the user interface, provided that image logging is not disabled by the user.

**Note**: It is important to ensure that the output directory specified in the configuration exists before calling this function, as it does not create the directory structure. The function assumes that the provided folder path is valid and accessible.

**Output Example**: A possible output of the function could be a string like the following:
'/absolute/path/to/outputs/log.html'
## FunctionDef log(img, metadata, metadata_parser, output_format, task, persist_image)
**log**: The function of log is to save an image along with its associated metadata and generate an HTML log file for documentation purposes.

**parameters**: The parameters of this Function.
· img: A NumPy array representing the image to be saved.
· metadata: A list of tuples containing metadata information related to the image.
· metadata_parser: An optional instance of MetadataParser or its subclass used to convert metadata into a string format.
· output_format: An optional string specifying the desired output format for the saved image (e.g., PNG, JPEG, WEBP).
· task: An optional dictionary containing task-related information, such as positive and negative prompts.
· persist_image: A boolean indicating whether to persist the image log or not.

**Code Description**: The log function is designed to handle the logging of images and their associated metadata. It begins by determining the output path based on the configuration settings. If image logging is disabled or if the persist_image parameter is set to False, it uses a temporary path for saving the image. The output format is set to a default if not explicitly provided.

The function then generates a temporary filename using the generate_temp_filename function, which creates a unique filename based on the current date and time. It ensures that the directory for the filename exists before proceeding to save the image.

Next, the function checks if a metadata_parser is provided. If so, it converts the metadata into a string format using the to_string method of the parser. The image is then saved in the specified format, which can be PNG, JPEG, or WEBP. Depending on the format, additional metadata may be embedded into the image using EXIF tags.

If image logging is enabled, the function proceeds to create an HTML log file. It constructs the HTML structure, including CSS styles and JavaScript for clipboard functionality. The metadata is formatted into a table within the HTML, and the image is included as well. The function checks for any existing log content and appends the new log entry accordingly.

Finally, the function writes the complete HTML content to the log file and updates the log cache to ensure that the latest entry is stored. The function returns the path to the saved image, allowing for easy access to the logged image.

The log function is integral to the project as it not only saves images but also documents their associated metadata in a structured format, facilitating better tracking and review of processed images.

**Note**: It is essential to ensure that the metadata_parser provided implements the to_string method to avoid runtime errors. Additionally, the output format should be one of the supported formats (PNG, JPEG, WEBP) to ensure proper image saving.

**Output Example**: An example output of the function could be a string representing the path to the saved image, such as:
'/absolute/path/to/outputs/2023-10-05/2023-10-05_14-30-15_1234.png'
