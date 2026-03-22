import pandas as pd
import numpy as np

def load_data(file_path):
    #Loads the dataset from a CSV file
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def create_next_day_targets(df):
    
    ## Converts dates, sorts the data by user and date, and creates next-day target variables.

    print("Creating next-day target variables...")
    
    # 1. Convert date to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Sort the data by user_id and date to ensure time continuity
    df = df.sort_values(by=['user_id', 'date']).reset_index(drop=True)
    
    # 3. Use groupby and shift(-1) to create next-day targets
    # This brings tomorrow's value into today's row for each user
    df['next_day_hrv'] = df.groupby('user_id')['hrv'].shift(-1)
    df['next_day_recovery'] = df.groupby('user_id')['recovery_score'].shift(-1)
    
    return df

def validate_targets(df):
    
    #Validates that the shifting was done correctly and that no data leakage occurred between users.
    print("Validating target variables...")
    
    # Validation 1: Get the first user and their first two rows to check alignment
    first_user = df['user_id'].iloc[0]
    user_data = df[df['user_id'] == first_user].head(2)
    
    if len(user_data) < 2:
        print("Error: Not enough data for validation.")
        return False
        
    # Check that current day's target matches the next day's actual value
    hrv_correct = user_data.iloc[0]['next_day_hrv'] == user_data.iloc[1]['hrv']
    rec_correct = user_data.iloc[0]['next_day_recovery'] == user_data.iloc[1]['recovery_score']
    
    # Validation 2: Ensure the last day for a user has a NaN target (no leakage from the next user)
    last_row_user = df[df['user_id'] == first_user].iloc[-1]
    boundary_valid = pd.isna(last_row_user['next_day_hrv'])
    
    # Output results
    print(f" - HRV Shift Alignment: {'Pass' if hrv_correct else 'Fail'}")
    print(f" - Recovery Shift Alignment: {'Pass' if rec_correct else 'Fail'}")
    print(f" - User Boundary Protection: {'Pass' if boundary_valid else 'Fail'}")
    
    return hrv_correct and rec_correct and boundary_valid

def save_processed_data(df, output_path):
    ## Saves the processed dataframe to a CSV file
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("File saved successfully.")

def main():
    ## Main execution block to orchestrate the feature engineering workflow
    input_file = '/Users/nickyl/Documents/GitHub/DAMO-699-5-Group-6-Capstone_project-Whoop/outputs/cleaned_whoop.csv'
    output_file = '/Users/nickyl/Documents/GitHub/DAMO-699-5-Group-6-Capstone_project-Whoop/outputs/whoop_fitness_engineered_functional.csv'
    
    # 1. Load the initial data
    raw_data = load_data(input_file)
    
    # 2. Apply transformations (Feature Engineering)
    data_with_targets = create_next_day_targets(raw_data)
    
    # 3. Validate results for correctness and leakage
    is_valid = validate_targets(data_with_targets)
    print("Output Results : ",is_valid)
    if is_valid:
        # 4. Save the finalized dataset if validation passes
        save_processed_data(data_with_targets, output_file)
    else:
        print("Validation failed. Data not saved to prevent errors in modeling.")

# Run the program
if __name__ == "__main__":
    main()
