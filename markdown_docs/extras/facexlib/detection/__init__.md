## FunctionDef init_detection_model(model_name, half, device, model_rootpath)
**init_detection_model**: The function of init_detection_model is to initialize a face detection model based on the specified architecture and load its pre-trained weights.

**parameters**: The parameters of this Function.
· model_name: A string that specifies the name of the model architecture to be used for face detection (e.g., 'retinaface_resnet50' or 'retinaface_mobile0.25').  
· half: A boolean indicating whether to use half-precision inference (default is False).  
· device: A string that specifies the device on which the model will run, either 'cuda' for GPU or 'cpu' (default is 'cuda').  
· model_rootpath: An optional string that defines the root path where the model weights will be saved or loaded from (default is None).

**Code Description**: The init_detection_model function is responsible for setting up a face detection model by selecting the appropriate architecture based on the model_name parameter. It supports two architectures: 'retinaface_resnet50' and 'retinaface_mobile0.25'. Depending on the specified model name, it creates an instance of the RetinaFace class with the corresponding network configuration.

The function first checks the model_name and initializes the RetinaFace model with the specified parameters, including half-precision and device settings. It then constructs the URL for downloading the pre-trained model weights based on the selected architecture. The function utilizes the load_file_from_url function to download the model weights from the specified URL and save them in the designated directory.

Once the model weights are downloaded, the function loads the weights into the RetinaFace model using PyTorch's torch.load method. It ensures that any unnecessary prefixes in the state dictionary (specifically 'module.') are removed before loading the weights. After loading the weights, the model is set to evaluation mode and moved to the specified device (CPU or GPU) for inference.

The init_detection_model function is called within the FaceRestoreHelper class during its initialization. This integration highlights its role in preparing the face detection model that will be used for detecting faces in images. The function is essential for ensuring that the model is correctly configured and ready for use in subsequent processing tasks.

**Note**: It is important to ensure that the model_name provided is valid and that the device specified is available. The function will raise a NotImplementedError if an unsupported model name is provided. Additionally, the model_rootpath should have appropriate permissions for saving the downloaded weights.

**Output Example**: A possible return value of the function could be an instance of the RetinaFace model, ready for inference, such as:
```
<RetinaFace object at 0x7f8c3a1b4d60>
```
