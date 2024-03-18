import os
import re

def read_and_process_files(directory):
    """
    Reads files matching the pattern 'Template_Acceleration${gesture_number}'
    and stores their data in a dictionary.
    
    Args:
    - directory: The path to the directory containing the files.
    
    Returns:
    A dictionary with gesture numbers as keys and a list of lists of tuples (representing the data) as values.
    """
    data_dict = {}
    
    # Compile a regular expression to match the file names
    pattern = re.compile(r"Template_Acceleration(\d+)")
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            match = pattern.search(filename)
            if match:
                gesture_number = int(match.group(1))  # Extract the gesture number
                filepath = os.path.join(root, filename)
                
                with open(filepath, 'r') as file:
                    # Read lines and split each line into a tuple of floats
                    lines = file.readlines()
                    data = [tuple(map(float, line.strip().split())) for line in lines]
                    
                    # If the gesture number is already in the dictionary, append the data; otherwise, create a new entry
                    if gesture_number in data_dict:
                        data_dict[gesture_number].append(data)
                    else:
                        data_dict[gesture_number] = [data]

    return data_dict
