## FunctionDef align_warp_face(self, landmark, border_mode)
**align_warp_face**: The function of align_warp_face is to align and warp a face image based on detected landmarks to fit a predefined face template.

**parameters**: The parameters of this Function.
· landmark: A set of facial landmarks used to estimate the affine transformation for aligning the face.
· border_mode: A string that specifies the border mode to be used when warping the image. It can be 'constant', 'reflect101', or 'reflect'.

**Code Description**: The align_warp_face function takes facial landmarks and a border mode as input to perform an affine transformation on the input image, aligning the detected face to a predefined template. The function first estimates the affine transformation matrix using the cv2.estimateAffinePartial2D method, which computes the transformation based on the provided landmarks and the face template. This matrix is then appended to a list of affine matrices for potential later use.

The function checks the specified border mode and assigns the corresponding OpenCV constant for border handling. The input image, which is expected to contain the face to be aligned, is then warped using the cv2.warpAffine function. This function applies the affine transformation to the input image, resulting in a cropped face that fits the specified dimensions defined by self.face_size. The border mode and a default border value are also applied during this warping process.

This function is called within the crop_image function, which is responsible for detecting faces in an input image. After detecting the facial landmarks, crop_image invokes align_warp_face to align the first detected face based on its landmarks. The result of this function is then returned as the final output of crop_image, which is an aligned and cropped version of the original image containing the face.

**Note**: It is important to ensure that the input image and landmarks are correctly provided to avoid errors during the affine transformation. The border mode should also be chosen based on the desired visual output, as it affects how the edges of the warped image are handled.

**Output Example**: The return value of the align_warp_face function is expected to be a NumPy array representing the aligned and cropped face image, which may look like a rectangular image of a face centered and resized according to the predefined face template.
## FunctionDef crop_image(img_rgb)
**crop_image**: The function of crop_image is to detect faces in an input image and align them based on facial landmarks for further processing.

**parameters**: The parameters of this Function.
· img_rgb: A NumPy array representing the input image in RGB format.

**Code Description**: The crop_image function begins by checking if the global variable faceRestoreHelper is initialized. If it is not, the function imports the FaceRestoreHelper class from the face restoration utilities and creates an instance of it. This instance is configured with an upscale factor of 1, a model root path defined in the configuration, and the device set to 'cpu' to avoid memory management issues.

Once the FaceRestoreHelper instance is ready, the function calls the clean_all method to reset any previous state, ensuring that the processing of the new image starts fresh. The input image, which is expected to be in RGB format, is then converted to BGR format (as required by OpenCV) and passed to the read_image method of the FaceRestoreHelper instance. This method processes the image, preparing it for face detection.

The get_face_landmarks_5 method is subsequently called to detect facial landmarks. This method identifies five key points on the face and stores them in the all_landmarks_5 attribute of the FaceRestoreHelper instance. If no faces are detected, the function prints a message indicating this and returns the original image. If faces are detected, the function prints the number of detected faces.

The first detected face's landmarks are then used to call the align_warp_face function, which aligns and warps the face image based on the detected landmarks. The result of this operation is returned as a NumPy array representing the aligned and cropped face image, converted back to RGB format.

The crop_image function is called within the apply_control_nets function in the async_worker module, specifically when processing tasks related to face images. This integration highlights its role in the broader context of face restoration and manipulation workflows, where accurate face detection and alignment are crucial for subsequent image processing tasks.

**Note**: It is essential to ensure that the input image is in the correct format (RGB) and that the FaceRestoreHelper instance is properly initialized to avoid errors during processing. Users should also be aware that the function will return the original image if no faces are detected.

**Output Example**: The return value of the crop_image function is expected to be a NumPy array representing an aligned and cropped face image, which may appear as a rectangular image of a face centered and resized according to the predefined face template.
