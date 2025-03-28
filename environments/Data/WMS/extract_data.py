import pandas as pd
import os

# Input and output file paths
input_file = r"environments\Data\WMS\camels\pet_hargreaves_day.csv"
precip_file = r"environments\Data\WMS\camels\precip_chirps_day.csv"
output_file = r"environments\Data\WMS\extracted_data.csv"

# Column to extract
column_name = "5748001"

def load_csv(file_path, required_columns=None):
    """Load a CSV file and ensure required columns exist."""
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {file_path} does not exist.")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Error: Missing columns {missing_columns} in {file_path}.")
    
    return data

def process_data(input_file, precip_file, output_file, column_name):
    """Process input and precipitation data, and save the output."""
    # Load input data
    data = load_csv(input_file, required_columns=["date", column_name])
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data.dropna(subset=['date'], inplace=True)  # Drop rows with invalid dates

    # Extract day of year (doy) and ET_0
    data['doy'] = data['date'].dt.dayofyear
    data['ET_0'] = data[column_name]
    data["year"] = data["date"].dt.year

    # Load precipitation data
    precip_data = load_csv(precip_file, required_columns=["date", column_name])
    precip_data['date'] = pd.to_datetime(precip_data['date'], errors='coerce')
    precip_data.dropna(subset=['date'], inplace=True)
    precip_data = precip_data[precip_data['date'].dt.year >= 1981]  # Filter years >= 1981
    precip_data.rename(columns={column_name: 'precipitation'}, inplace=True)
    # Merge precipitation data with input data
    data = pd.merge(data, precip_data[['date', 'precipitation']], on='date', how='inner')
    

    # Select and save the required columns
    output_data = data[['date', 'year', 'doy', 'ET_0', 'precipitation']]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_data.to_csv(output_file, index=False)
    print(f"Data successfully extracted and saved to {output_file}")

# Run the processing function
try:
    process_data(input_file, precip_file, output_file, column_name)
except (FileNotFoundError, ValueError) as e:
    print(e)