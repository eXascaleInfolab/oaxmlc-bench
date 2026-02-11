def collate_global_int_labels(batched_input_data):
    """Collate function for global dataset for methods requiring integer labels.

    Args:
        batched_input_data (list): A list of tuples containing three elements:
            - A torch.tensor containing the inputs for the model.
            - A list containing the labels.
            - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.

    Returns:
        tuple: A tuple containing three elements:
            - A list of torch.tensor containing the inputs for the model.
            - A list of lists containing the labels.
            - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.
    """

    input_data = []
    labels = []
    column_indices = None
    for sample_data, sample_labels, sample_column_indices in batched_input_data:
        input_data.append(sample_data)
        labels.append(sample_labels)
        # It is always the same
        column_indices = sample_column_indices

    return (
        input_data,
        labels,
        column_indices,
        None
    )

def collate_global_int_labels_hector_tamlec(batched_input_data):
    """Collate function for global dataset for methods requiring integer labels.

    Args:
        batched_input_data (list): A list of tuples containing three elements:
            - A torch.tensor containing the inputs for the model.
            - A list containing the labels.
            - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.

    Returns:
        tuple: A tuple containing three elements:
            - A list of torch.tensor containing the inputs for the model.
            - A list of lists containing the labels.
            - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.
    """

    input_data = []
    labels = []
    column_indices = None
    for sample_data, sample_labels, sample_column_indices in batched_input_data:
        input_data.append(sample_data)
        labels.append(sample_labels)
        # It is always the same
        column_indices = sample_column_indices
        #paths.append(sample_paths)
    return (
        input_data,
        labels,
        column_indices,
        None
    )
