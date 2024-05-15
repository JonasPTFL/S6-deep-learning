import keras


def model_architecture_to_markdown_table(model: keras.Sequential) -> str:
    """
    Converts the given model architecture to a markdown table
    :param model: architecture of the model
    :return: markdown table string
    """
    summary_string_list = []
    model.summary(print_fn=lambda x: summary_string_list.append(x))
    summary_string = "\n".join(summary_string_list)

    # Split the table text into lines and ignore the first 4 lines, as they only contain the header
    lines = summary_string.strip().split('\n')[4:]

    # Define header and separator for markdown table
    header = ["Layer (type)", "Output Shape", "Param #"]
    separator = ["---", "---", "---"]

    # Initialize the table with header and separator
    table = [header, separator]

    # Process each line to extract the relevant data
    for line in lines:
        if line.startswith('│'):
            columns = line.split('│')[1:-1]  # Remove the first and last borders
            row = [col.strip() for col in columns]
            table.append(row)

    # Convert list of lists to markdown table string
    markdown_table = '\n'.join(['| ' + ' | '.join(row) + ' |' for row in table])

    return markdown_table
