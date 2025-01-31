import pandas as pd
import os
from pathlib import Path

def csv_to_excel(folder_path, output_excel_name='combined_data.xlsx'):
    # Create Excel writer object
    writer = pd.ExcelWriter(os.path.join(folder_path, output_excel_name), engine='xlsxwriter')
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Read each CSV and save to Excel
    for csv_file in csv_files:
        # Read CSV file
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        
        # Get sheet name (remove .csv extension)
        sheet_name = os.path.splitext(csv_file)[0]
        
        # Write to Excel sheet
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Auto-fit column widths
        worksheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns):
            series = df[col]
            max_len = max(
                series.astype(str).map(len).max(),  # length of values
                len(str(series.name))  # length of column name
            ) + 1
            worksheet.set_column(idx, idx, max_len)
    
    # Save the Excel file
    writer.close()
    print(f"Excel file '{output_excel_name}' has been created successfully!")

# Example usage
folder_path = 'results/debug_0123data/head_log'  # Replace with your folder path
csv_to_excel(folder_path)