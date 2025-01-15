## FunctionDef build_launcher
**build_launcher**: The function of build_launcher is to create batch files for launching a Windows application based on specified presets.

**parameters**: The parameters of this Function.
· There are no parameters for this function.

**Code Description**: The build_launcher function is designed to facilitate the creation of batch files that execute a command for launching an application in a Windows environment. The function first checks if the current build is a standalone Windows build by evaluating the variable `is_win32_standalone_build`. If this condition is not met, the function exits early without performing any further actions.

If the build is valid, the function initializes a list of presets, which includes `None`, 'anime', and 'realistic'. It then iterates over each preset in this list. For each iteration, the function constructs a command string `win32_cmd_preset` by replacing a placeholder in `win32_cmd` with the appropriate preset option. If the preset is `None`, it omits the preset option from the command.

Next, the function determines the path for the batch file to be created. The path is constructed using `os.path.join`, combining `win32_root` with either 'run.bat' (if the preset is `None`) or 'run_{preset}.bat' (for the other presets). Before creating the batch file, the function checks if the file already exists at the specified path using `os.path.exists`. If the file does not exist, it opens the file in write mode with UTF-8 encoding and writes the constructed command string into the batch file.

This function is called from the `launch.py` module, which suggests that it is part of a larger process aimed at launching applications with specific configurations. The relationship with its caller indicates that `launch.py` likely relies on the successful execution of `build_launcher` to prepare the necessary batch files before proceeding with the application launch.

**Note**: It is important to ensure that the variable `is_win32_standalone_build` is correctly defined and that `win32_cmd` and `win32_root` are properly initialized before calling this function to avoid runtime errors.

**Output Example**: If the function is executed successfully with the 'anime' preset, the content of the created batch file 'run_anime.bat' might look like this:
```
<command_string_with_anime_preset>
```
