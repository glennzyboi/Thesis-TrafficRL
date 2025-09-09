
import pandas as pd
import os

def generate_dummy_scenarios(master_aggregates_path, output_base_path):
    df = pd.read_csv(master_aggregates_path)

    vehicle_columns = ["car", "motor", "jeepney", "bus", "truck", "tricycle"]

    for index, row in df.iterrows():
        day = row["Day"]
        cycle_num = row["CycleNum"]
        intersection_id = row["IntersectionID"]

        scenario_dir = os.path.join(output_base_path, day)
        os.makedirs(scenario_dir, exist_ok=True)

        # Create a DataFrame for the current scenario
        # Select only the vehicle count columns
        scenario_data = row[vehicle_columns].to_frame().T

        output_file = os.path.join(scenario_dir, f"cycle_{cycle_num}_{intersection_id}.csv")
        scenario_data.to_csv(output_file, index=False)
        print(f"Generated: {output_file}")

if __name__ == "__main__":
    MASTER_AGGREGATES_PATH = "../out/master_aggregates.csv"
    OUTPUT_BASE_PATH = "../out/scenarios/"
    generate_dummy_scenarios(MASTER_AGGREGATES_PATH, OUTPUT_BASE_PATH)


