## FunctionDef make_progress_html(number, text)
**make_progress_html**: The function of make_progress_html is to generate an HTML string that displays progress information based on the provided number and text.

**parameters**: The parameters of this Function.
· parameter1: number - An integer representing the current progress percentage or step in the process.
· parameter2: text - A string that provides a descriptive message related to the current progress.

**Code Description**: The make_progress_html function takes two parameters: `number` and `text`. It utilizes a predefined HTML template, referred to as `progress_html`, and replaces placeholders within this template with the actual values provided through the parameters. Specifically, it replaces the placeholder '*number*' with the string representation of the `number` parameter and the placeholder '*text*' with the `text` parameter. This results in a dynamic HTML string that can be used to visually represent the progress of a task in a web interface.

The function is called within the `generate_clicked` function located in the webui.py module. In this context, make_progress_html is used to update the user interface with the current progress of an asynchronous task. When the task is initiated, it first displays a message indicating that the task is waiting to start. As the task progresses, the function is called again to update the progress display with the current percentage and title, which are extracted from the task's yields. This integration ensures that users receive real-time feedback on the status of their tasks, enhancing the overall user experience.

**Note**: It is important to ensure that the `progress_html` template is properly defined and contains the placeholders '*number*' and '*text*' for the function to work correctly. Additionally, the values passed to the function should be appropriate for display purposes to maintain clarity in the user interface.

**Output Example**: An example of the output generated by the function might look like this:
```html
<div class="progress">
    <span>1</span>% - Waiting for task to start ...
</div>
``` 
This output indicates that the progress is at 1% and provides a message to the user.
