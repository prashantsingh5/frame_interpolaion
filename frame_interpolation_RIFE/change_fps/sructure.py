import os

def generate_folder_structure(root_dir):
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        for file in files:
            print(f'{indent}  {file}')

root_dir = input(r'C:\Users\pytorch\Desktop\frame_interpolation')
generate_folder_structure(root_dir)