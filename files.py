import os

# Path to the directory containing the folders
base_path = "/Users/vikramkaushik/Projects/ASLDetect/sign-language-detector-python-master/data"

# Desired mapping: 'A-Z' -> 0-25, 'nothing' -> 26, 'del' -> 27, 'space' -> 28
folder_mapping = {chr(i): str(i - ord('A')) for i in range(ord('A'), ord('Z') + 1)}
folder_mapping.update({
    'nothing': '26',
    'del': '27',
    'space': '28'
})

# Get the list of current folders
current_folders = os.listdir(base_path)

# Rename folders based on the mapping
for folder in current_folders:
    old_path = os.path.join(base_path, folder)
    new_name = folder_mapping.get(folder)
    
    if new_name is not None:  # Only rename folders that are in the mapping
        new_path = os.path.join(base_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")
