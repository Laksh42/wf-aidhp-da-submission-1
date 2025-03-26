"""Script to extract investment data to JSON and CSV files."""

import os
import json
import csv
from pathlib import Path

def main():
    """Save investment data to files."""
    print("Starting to save investment data...")
    
    # Ensure directories exist
    data_dir = Path("./data")
    sample_dir = data_dir / "sample"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Sample investment data
    csv_data = """user_id,investment_id,investment_type,amount,current_value,start_date
4,1,stocks,6068.87,5557.05,2025-03-03
9,2,etfs,4643.9,4377.02,2024-11-06
9,3,mutual_funds,7779.74,8061.87,2020-06-13
9,4,bonds,6566.34,5472.88,2024-04-08
10,5,mutual_funds,3925.32,3811.43,2021-07-01"""
    
    # Write CSV to file
    csv_file = data_dir / "investment_data.csv"
    with open(csv_file, "w") as f:
        f.write(csv_data)
    
    print(f"Saved CSV data to {csv_file}")
    
    # Convert to JSON format
    investments = []
    lines = csv_data.strip().split('\n')
    reader = csv.DictReader(lines)
    
    for row in reader:
        investment = {
            "user_id": row["user_id"],
            "investment_id": int(row["investment_id"]),
            "investment_type": row["investment_type"],
            "amount": float(row["amount"]),
            "current_value": float(row["current_value"]),
            "start_date": row["start_date"]
        }
        investments.append(investment)
    
    # Write JSON to file
    json_file = sample_dir / "investments.json"
    with open(json_file, "w") as f:
        json.dump(investments, f, indent=2)
    
    print(f"Saved JSON data to {json_file}")
    print("Investment data processing completed successfully.")

if __name__ == "__main__":
    main() 