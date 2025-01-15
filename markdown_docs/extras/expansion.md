## FunctionDef safe_str(x)
**safe_str**: The function of safe_str is to sanitize and normalize a given input string by removing excessive whitespace and specific trailing characters.

**parameters**: The parameters of this Function.
· x: The input that needs to be converted to a string and sanitized.

**Code Description**: The safe_str function takes an input parameter x, which can be of any data type. It first converts this input into a string using the str() function. The function then enters a loop that runs 16 times, where it replaces any occurrence of double spaces ('  ') with a single space (' '). This process is repeated to ensure that any excessive whitespace is reduced to a single space. After the loop, the function returns the sanitized string by stripping any leading or trailing characters that are either a comma, period, or whitespace (including newlines and carriage returns) using the strip() method.

The safe_str function is called in multiple places within the project, notably in the __call__ method of the FooocusExpansion class and the process_prompt function in the worker module. In the __call__ method, safe_str is used to sanitize the prompt input before it is tokenized and processed further. This ensures that the prompt is clean and free of unnecessary whitespace, which is crucial for generating accurate results from the model. Similarly, in the process_prompt function, safe_str is utilized to clean both the main prompt and negative prompt inputs, ensuring that they are properly formatted before further processing. This highlights the importance of the safe_str function in maintaining the integrity of input data throughout the project.

**Note**: It is important to ensure that the input to safe_str is of a type that can be converted to a string. If the input is already a string, the function will still operate correctly, but it is designed to handle various data types effectively.

**Output Example**: If the input to safe_str is "  Hello,   World!  ", the output will be "Hello, World!".
## FunctionDef remove_pattern(x, pattern)
**remove_pattern**: The function of remove_pattern is to remove specified patterns from a given string.

**parameters**: The parameters of this Function.
· parameter1: x - A string from which patterns will be removed.
· parameter2: pattern - A list of substrings that need to be removed from the string x.

**Code Description**: The remove_pattern function takes two arguments: a string x and a list of substrings pattern. It iterates through each substring in the pattern list and replaces occurrences of that substring in the string x with an empty string, effectively removing it. The function uses the string method replace, which returns a new string with all occurrences of the specified substring replaced. After processing all patterns, the modified string x is returned. This function is useful for cleaning up strings by removing unwanted substrings, such as specific characters, words, or phrases.

**Note**: It is important to ensure that the patterns provided in the list are correctly specified, as the function will remove all occurrences of each pattern from the string. If a pattern does not exist in the string, it will simply be ignored without causing any errors.

**Output Example**: If the input string is "Hello, world!" and the pattern list is ["Hello", "!"], the function will return " world".
## ClassDef FooocusExpansion
**FooocusExpansion**: The function of FooocusExpansion is to enhance text generation capabilities by applying a bias to the logits based on a predefined set of positive words.

**attributes**: The attributes of this Class.
· tokenizer: An instance of AutoTokenizer used to convert text into token IDs and manage the vocabulary.
· logits_bias: A tensor that applies a bias to the model's output logits, initialized to negative infinity for all tokens except for those specified as positive.
· model: An instance of AutoModelForCausalLM that represents the causal language model used for text generation.
· patcher: An instance of ModelPatcher that manages the model's device allocation and optimizations.

**Code Description**: The FooocusExpansion class is designed to facilitate enhanced text generation by leveraging a language model with specific biases applied to its output. Upon initialization, the class loads a tokenizer and a language model from a specified path. It reads a list of positive words from a file and modifies the logits_bias tensor to ensure that these words are favored during text generation. The logits_bias tensor is initialized to negative infinity for all tokens, effectively suppressing their likelihood, except for the positive words, which are set to zero.

The class includes two main methods: logits_processor and __call__. The logits_processor method is a custom logits processor that modifies the output scores of the model based on the input tokens. It ensures that the logits for the input tokens are set to negative infinity, preventing them from being selected in the generated output, while also maintaining a specific bias for the token with ID 11.

The __call__ method is the primary interface for generating text. It takes a prompt and a seed value, tokenizes the prompt, and generates a response using the language model. The method ensures that the model is loaded on the correct device and applies the logits_processor to influence the generation process. The generated response is then decoded back into text format.

This class is utilized in the refresh_everything function found in the modules/default_pipeline.py file. In this context, FooocusExpansion is instantiated when the final_expansion variable is None, indicating that the expansion model has not yet been loaded. This integration allows the text generation pipeline to leverage the enhanced capabilities provided by the FooocusExpansion class, ensuring that the model can generate text that aligns with the specified positive biases.

**Note**: When using the FooocusExpansion class, ensure that the appropriate model files and vocabulary lists are available at the specified path. Additionally, be aware of the device management for optimal performance, especially when using mixed precision (fp16) settings.

**Output Example**: Given a prompt "The future of AI is", the FooocusExpansion class might generate a response such as "The future of AI is bright, innovative, and transformative," where the words "bright," "innovative," and "transformative" are among the positive words that were favored during the generation process.
### FunctionDef __init__(self)
**__init__**: The function of __init__ is to initialize the FooocusExpansion class, setting up the tokenizer, model, and associated configurations.

**parameters**: The parameters of this Function.
· There are no parameters for this function.

**Code Description**: The __init__ method of the FooocusExpansion class is responsible for setting up the necessary components for the expansion model. Upon instantiation, it performs the following key actions:

1. **Tokenizer Initialization**: The method initializes a tokenizer using the `AutoTokenizer.from_pretrained` function, loading it from a specified path (`path_fooocus_expansion`). This tokenizer is essential for processing text inputs.

2. **Positive Words Loading**: It reads a file named 'positive.txt' located in the same directory as the tokenizer path. The contents of this file are split into lines, and each line is processed to create a list of positive words. Each word is converted to lowercase and prefixed with 'Ġ' (a special character used in tokenization). Empty lines are filtered out.

3. **Logits Bias Initialization**: A tensor named `logits_bias` is created, initialized to zeros with a shape that corresponds to the vocabulary size of the tokenizer. This tensor is used to apply biases to the model's logits during inference.

4. **Logits Bias Adjustment**: The method iterates through the tokenizer's vocabulary. For each word that matches one of the positive words, the corresponding entry in the `logits_bias` tensor is set to zero. This adjustment is intended to influence the model's output positively for these words.

5. **Model Initialization**: The method initializes the model using `AutoModelForCausalLM.from_pretrained`, loading it from the same specified path. The model is set to evaluation mode with `self.model.eval()`.

6. **Device Management**: The method determines the devices for loading and offloading computations by calling `model_management.text_encoder_device()` and `model_management.text_encoder_offload_device()`. It checks if the device is of type MPS (Metal Performance Shaders) and adjusts the load device to CPU if necessary.

7. **FP16 Precision Check**: The method checks whether to use FP16 precision based on the device configuration by calling `model_management.should_use_fp16()`. If FP16 is to be used, the model is converted to half-precision with `self.model.half()`.

8. **Model Patching**: Finally, the method creates an instance of the `ModelPatcher` class, passing the model and the determined devices for loading and offloading. This instance is responsible for managing any patches applied to the model.

The __init__ method is crucial for setting up the FooocusExpansion class, ensuring that all components are properly initialized and configured for subsequent operations. It interacts with several other components in the project, including the tokenizer, model management functions, and the ModelPatcher class, to create a cohesive environment for text expansion tasks.

**Note**: It is important to ensure that the paths and files referenced in this method are correctly set up in the environment to avoid runtime errors during initialization.

**Output Example**: A possible appearance of the initialization process might look like this:
```python
Fooocus V2 Expansion: Vocab with 150 words.
Fooocus Expansion engine loaded for cuda:0, use_fp16 = True.
```
***
### FunctionDef logits_processor(self, input_ids, scores)
**logits_processor**: The function of logits_processor is to modify the input scores by applying a bias based on the provided input IDs.

**parameters**: The parameters of this Function.
· input_ids: A tensor containing the input token IDs for which the logits are being processed.  
· scores: A tensor of shape (1, n) representing the raw logits for the next token predictions.

**Code Description**: The logits_processor function is designed to adjust the logits (raw prediction scores) produced by a model before they are passed to a sampling method during text generation. The function begins by asserting that the scores tensor has two dimensions and that the first dimension equals one, ensuring that it is in the expected shape. It then transfers the logits_bias tensor to the same device as the scores tensor to ensure compatibility.

Next, a clone of the logits_bias is created, which allows for modifications without altering the original bias. The function then sets specific indices in the bias tensor to negative infinity (neg_inf) based on the input_ids. This effectively prevents the model from selecting those token IDs during the sampling process. Additionally, it sets the bias for the token ID 11 to zero, which may indicate that this token should be treated normally without any bias.

Finally, the function returns the modified scores by adding the adjusted bias to the original scores. This process is crucial for controlling the output of the model, particularly in scenarios where certain tokens should be favored or disfavored.

The logits_processor function is called within the __call__ method of the FooocusExpansion class. When a prompt is provided, the __call__ method prepares the input by tokenizing it and determining the maximum number of new tokens to generate. It then invokes the model's generate method, passing in the tokenized input and specifying the logits_processor as part of the generation strategy. This integration allows the model to utilize the logits_processor to refine its predictions based on the specified input IDs, ultimately influencing the generated text.

**Note**: It is important to ensure that the input_ids provided to the logits_processor are valid and correspond to the expected token indices in the model's vocabulary. Misalignment can lead to unexpected behavior during text generation.

**Output Example**: An example of the return value from the logits_processor could be a tensor of modified scores, such as:
```
tensor([[ -2.0, -1.5, -inf, 0.0, 1.2, ... ]])
```
This output indicates the adjusted logits after applying the bias, where certain token scores have been set to negative infinity, effectively removing them from consideration during sampling.
***
### FunctionDef __call__(self, prompt, seed)
**__call__**: The function of __call__ is to generate a response based on a given prompt and seed value, utilizing a machine learning model.

**parameters**: The parameters of this Function.
· prompt: A string input that serves as the basis for generating a response. It should not be empty.
· seed: An integer value used to initialize the random number generator for reproducibility in the generation process.

**Code Description**: The __call__ method is designed to facilitate the generation of text responses based on a provided prompt and seed. Initially, the method checks if the prompt is an empty string; if so, it returns an empty string immediately, ensuring that no further processing occurs with invalid input.

Next, the method verifies if the current device being used by the patcher differs from the device designated for loading the model. If they are different, it indicates that the Fooocus Expansion has been loaded independently, prompting the invocation of the load_model_gpu function. This function is responsible for loading the specified machine learning model onto the GPU, ensuring that the model is ready for inference.

The seed value is then processed to ensure it falls within a predefined limit (SEED_LIMIT_NUMPY), and the set_seed function is called to initialize the random number generator with this seed. This step is crucial for ensuring that the generation process is reproducible.

Following this, the prompt is sanitized using the safe_str function, which cleans the input by removing excessive whitespace and specific trailing characters. The sanitized prompt is then tokenized using the tokenizer, converting it into a format suitable for the model. The tokenized input is transferred to the appropriate device for processing.

The method calculates the current token length and determines the maximum number of new tokens that can be generated, ensuring that the total token length does not exceed a specified limit. If no new tokens can be generated (max_new_tokens equals zero), the method returns the prompt without the trailing comma.

The model's generate method is then called with the tokenized input and various parameters, including top_k and max_new_tokens, to control the generation process. The logits_processor is also passed as part of the generation strategy, allowing for the adjustment of the model's output scores based on specific input IDs.

Finally, the generated features are decoded back into a human-readable format using the tokenizer, and the first response is returned after being sanitized again with safe_str. This ensures that the output is clean and formatted correctly for the user.

The __call__ method is integral to the functionality of the FooocusExpansion class, as it orchestrates the entire process of generating a response from a prompt, managing device compatibility, and ensuring that the model is loaded and utilized effectively.

**Note**: It is essential to provide a valid, non-empty prompt to the __call__ method. Additionally, the seed should be an integer to ensure proper initialization of the random number generator. Users should also be aware of the maximum token length constraints to avoid unexpected behavior during text generation.

**Output Example**: A possible return value from the function could be a string such as "Here is the generated response based on your prompt."
***
