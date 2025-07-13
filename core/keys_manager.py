def load_api_key(file_path: str = "../keys/youtube_data_API_key.txt") -> str:
    """
    Loads the API key from a specified text file.

    Args:
        file_path (str): Path to the file containing the API key. Defaults to 'secrets/api_key.txt'.

    Returns:
        str: The API key as a string.
    """
    with open(file_path, 'r') as f:
        return f.read().strip()
