import shutil
import glob
import os
import re
import argparse
from typing import List

def get_arg():
    parser = argparse.ArgumentParser(description='Arguments for file transfer')
    parser.add_argument('--source', type = str, help = 'Source folder')
    parser.add_argument('--dest', type= str, help = 'Destination folder')
    parser.add_argument('--types', type= str, help= 'ADL/Fall')
    parser.add_argument('--data-type', type = str)
    args = parser.parse_args()
    return args

def find_match_elements(pattern, elements): 
    #compile the regular expression
    try:
        regex_pattern = re.compile(pattern)
        #filtering the elements that match the regex pattern
        matching_elements = [element for element in elements if regex_pattern.search(element)]
        return matching_elements
    except:
        print(f'Error: {e}')
        
    return []

def move_all(file_paths : List, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for path in file_paths:
        shutil.move(path, dest_folder)
    
def move_files(source_folder, destination_folder, pattern):
    try:
        
        # Check if the source folder exists
        if not os.path.exists(source_folder):
            raise FileNotFoundError("Source folder does not exist.")
        
        # Check if the destination folder exists, if not, create it
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Get a list of files in the source folder
        files = os.listdir(source_folder)
        matched_files = find_match_elements(pattern, files)
        
        if not matched_files:
            raise Exception('Couldn\'t find files with the pattern')

        
        for file in matched_files:
            
            source_file_path = os.path.join(source_folder, file)
            destination_file_path = os.path.join(destination_folder, file)

            # Perform the move operation
            shutil.move(source_file_path, destination_file_path)
        print("Files moved successfully.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__" :
    args = get_arg()
    file_paths = glob.glob(f'{args.source}/**/{args.data_type}/{args.types}/**')
    print(file_paths)
    move_all(file_paths, args.dest)