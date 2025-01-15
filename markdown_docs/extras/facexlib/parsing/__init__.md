## FunctionDef init_parsing_model(model_name, half, device, model_rootpath)
**init_parsing_model**: The function of init_parsing_model is to initialize a parsing model based on the specified model name and configuration parameters.

**parameters**: The parameters of this Function.
· model_name: A string that specifies the name of the model to be initialized. It can be either 'bisenet' or 'parsenet'. The default value is 'bisenet'.
· half: A boolean indicating whether to use half-precision (float16) for the model. The default value is False.
· device: A string that specifies the device on which the model will be loaded, such as 'cuda' for GPU or 'cpu'. The default value is 'cuda'.
· model_rootpath: An optional string that defines the root path where the model weights will be saved. If not provided, the default path will be used.

**Code Description**: The init_parsing_model function is responsible for initializing a semantic segmentation model based on the specified model name. It supports two models: BiSeNet and ParseNet. 

When the model_name is set to 'bisenet', an instance of the BiSeNet class is created with 19 output classes. The function then constructs the URL for the pre-trained weights specific to BiSeNet and calls the load_file_from_url function to download the model weights from the specified URL. The downloaded weights are then loaded into the model using the load_state_dict method, and the model is set to evaluation mode. Finally, the model is transferred to the specified device (CPU or GPU) and returned.

If the model_name is set to 'parsenet', a similar process occurs where an instance of the ParseNet class is created with specified input and output sizes. The function constructs the URL for the pre-trained weights for ParseNet, downloads the weights using load_file_from_url, loads them into the model, sets the model to evaluation mode, and transfers it to the specified device before returning it.

If an unsupported model_name is provided, the function raises a NotImplementedError, indicating that the specified model is not implemented.

The init_parsing_model function is called within the FaceRestoreHelper class in the extras/facexlib/utils/face_restoration_helper.py file. It initializes the face parsing model when the use_parse parameter is set to True, allowing the FaceRestoreHelper to utilize the parsing capabilities of the initialized model for face restoration tasks.

**Note**: It is essential to ensure that the specified model_name is valid ('bisenet' or 'parsenet') to avoid raising a NotImplementedError. Additionally, the model_rootpath should be set appropriately if custom storage for model weights is desired.

**Output Example**: A possible return value of the function could be an instance of the BiSeNet or ParseNet model, ready for inference, such as:
- An instance of BiSeNet: `<BiSeNet instance>`
- An instance of ParseNet: `<ParseNet instance>`
