import os
import csv
from pathlib import Path

def list_png_files_to_csv(folder_path, output_csv):
    """
    List all PNG files in a folder and save their names to a CSV file.
    
    Args:
        folder_path (str): Path to the folder containing PNG files
        output_csv (str): Path where the CSV file will be saved
    """
    # Get all PNG files in the folder
    png_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.png'):
            png_files.append(file)
    
    # Sort the files alphabetically
    png_files.sort()
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename'])  # Header
        for png_file in png_files:
            writer.writerow([png_file])

if __name__ == '__main__':
    # Example usage
    folder_path = 'results/figures/heart'  # Change this to your folder path
    output_csv = 'results/figures/png_files_list.csv'    # Change this to your desired output path
    list_png_files_to_csv(folder_path, output_csv)
    print(f"CSV file has been created at: {output_csv}") 