# D3QN Traffic Signal Control Data Pipeline

A data processing and route generation pipeline for D3QN-based traffic signal control simulation using real traffic data.

## Project Structure

```
D3QN/
├── data/
│   ├── raw/                    # Raw Excel data files
│   ├── processed/              # Processed CSV files
│   └── routes/                 # Generated SUMO route files
├── out/
│   └── scenarios/              # Per-scenario data files
├── scripts/
│   ├── compile_bundles.py      # Process Excel data into CSV bundles
│   └── generate_routes.py      # Generate SUMO route files from bundles
├── network/                    # SUMO network files (.net.xml)
├── models/                     # Trained D3QN models
├── lane_map.json              # Intersection to edge mapping
├── verify_data.py              # Data verification script
└── requirements.txt            # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Process Raw Data

Place your Excel files (format: `INTERSECTION_YYYYMMDD_cycleN.xlsx`) in `data/raw/` and run:

```bash
python scripts/compile_bundles.py
```

This will:
- Read Excel files from `data/raw/`
- Aggregate lane-level data by intersection and vehicle type
- Generate `data/processed/master_bundles.csv` with all data
- Create scenario folders in `out/scenarios/` with per-intersection CSV files

### 3. Generate SUMO Routes

Generate route files for simulation:

```bash
python scripts/generate_routes.py --day 20250828 --cycle 1 --lane-map lane_map.json --mode flow
```

Options:
- `--day`: Day identifier (e.g., 20250828)
- `--cycle`: Cycle number (e.g., 1)
- `--lane-map`: JSON file mapping intersections to edges (optional)
- `--mode`: Either `flow` (aggregate flows) or `explicit` (individual vehicles)

## Data Format

### Input Excel Files

Expected structure:
- **Filename**: `INTERSECTION_YYYYMMDD_cycleN.xlsx` (e.g., `ECOLAND_20250828_cycle1.xlsx`)
- **Sheets**: `Raw_Annotations` (preferred) or `Aggregates`
- **Columns**:
  - `CycleID`: Cycle identifier
  - `LaneID`: Lane identifier (e.g., `EC_N`, `JP_S`)
  - `VehicleType`: Vehicle type (`Car`, `Motorcycle`, `Jeepney`, `Bus`, `Truck`)
  - `Count`: Number of vehicles
  - `CycleTime_s`: Cycle time in seconds
  - `PassengerEquivalent`: Passenger equivalent value
  - `Pass throughput per hr`: Passenger throughput per hour

### Output Files

- **master_bundles.csv**: Combined data for all intersections and cycles
- **scenarios_index.csv**: Index of all scenarios with metadata
- **Individual CSV files**: Per-intersection data in `out/scenarios/DAY/cycle_N/`
- **Route files**: SUMO-compatible `.rou.xml` files in `data/routes/`

## Lane Mapping

Create a `lane_map.json` file to map intersection names to actual SUMO edge IDs:

```json
{
  "ECOLAND": {
    "default": {
      "from": ["EC_N", "EC_S", "EC_E", "EC_W"],
      "to": ["EC_N_out", "EC_S_out", "EC_E_out", "EC_W_out"]
    }
  }
}
```

## Current Data

Your processed data shows:
- **ECOLAND**: 329 vehicles/cycle (55s cycles)
- **JOHNPAUL**: 252 vehicles/cycle (56s cycles)  
- **SANDAWA**: 345 vehicles/cycle (300s cycles)

Vehicle mix includes cars, motorcycles, jeepneys, buses, and trucks with realistic parameters for Philippine traffic conditions.

## Next Steps

1. Update `lane_map.json` with real edge IDs from your SUMO network
2. Integrate with D3QN training pipeline
3. Add more cycles and days to build larger datasets
4. Validate simulation results against real traffic measurements

## Notes

- All scripts work from any directory using absolute paths
- Vehicle flow rates are automatically calculated from counts and cycle times
- Default vehicle parameters are set for Philippine traffic conditions
- Scripts handle missing data gracefully with reasonable defaults
