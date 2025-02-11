import pandas as pd
import glob
import os

# Get all CSV files in results directory
csv_files = glob.glob('results/debug_metrics_v5_newusers1/new_aggregate/*.csv')
site_order = ['heart', 'neck', 'head', 'wrist']  # Define desired order

# Group files by user
user_files = {}
for file in csv_files:
    # Extract username and site from filename
    basename = os.path.basename(file)
    if '_' not in basename:
        continue
    username = basename.split('_')[0]
    if username not in user_files:
        user_files[username] = []
    user_files[username].append(file)

# Create Excel writer
with pd.ExcelWriter('combined_results.xlsx', engine='openpyxl') as writer:
    # Process each user's files
    for username, files in user_files.items():
        # Read and combine all files for this user
        dfs = []
    
        # Sort files according to site order
        sorted_files = []
        for site in site_order:
            for file in files:
                file_site = os.path.basename(file).replace('.csv', '').split('_')[1]
                if file_site == site:
                    sorted_files.append(file)
        
        # Read files in correct order
        for file in sorted_files:
            df = pd.read_csv(file)
            site = os.path.basename(file).replace('.csv', '').split('_')[1]
            df.columns = [col if col == 'fname' else f'{site}_{col}' for col in df.columns]
            dfs.append(df)
        
        # Merge all dataframes on fname
        if dfs:
            combined_df = dfs[0]
            for df in dfs[1:]:
                combined_df = pd.merge(combined_df, df, on='fname', how='outer')
            
            # Write to Excel
            combined_df.to_excel(writer, sheet_name=username, index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets[username]
            for idx, col in enumerate(combined_df.columns):
                max_length = max(
                    combined_df[col].astype(str).apply(len).max(),  # Length of longest value
                    len(str(col))  # Length of column name
                )
                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2  # Add padding

print('Created combined_results.xlsx with sheets for each user.') 