"""
generate_routes_from_bundle.py

Generates a single combined .rou.xml for a chosen bundle (Day + CycleNum).

- Reads per-intersection minimal CSVs prepared by compile_bundles.py under out/scenarios/<Day>/cycle_<CycleNum>.
- Produces a flow-mode .rou.xml that includes flows for each intersection and vehicle class.
- Optionally accepts a lane-map JSON file mapping intersections -> inbound/outbound edge ids.
  Example lane_map.json structure:
  {
    "ECOLAND": {
       "default": {"from": ["edge_in_1"], "to": ["edge_out_1"]},
       "jeepney": {"from": ["edge_in_1"], "to": ["edge_out_1"]}
    },
    "JOHNPAUL": { ... }
  }

Usage:
  python scripts/generate_routes_from_bundle.py --day 20250828 --cycle 1 [--lane-map lane_map.json] [--mode flow|explicit] [--out data/routes/scenario_20250828_cycle1.rou.xml]

Notes:
- If lane_map is not supplied: the script will write flows using placeholder edge ids (edge_in_X / edge_out_X).
  Replace placeholders with real edges from your .net.xml (use tls_mapping.csv or sumolib to extract edges).
- flow mode uses vehsPerHour computed from counts and CycleTime_s:
    vph = count * (3600 / CycleTime_s)
- explicit mode generates one <vehicle> per count with deterministic/seeded depart times inside the cycle.
"""

import os
import argparse
import json
import math
import random
import pandas as pd
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

# default directories (consistent with compile_bundles output)
# Use absolute paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SCEN_DIR = os.path.join(PROJECT_ROOT, "out", "scenarios")
ROUTE_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "routes")

# default vehicle-to-sumo-type mapping (user can change)
VEH_CLASS_TO_VTYPE = {
    "jeepney": "jeepney",
    "bus": "bus",
    "car": "car",
    "motor": "motor",
    "truck": "truck",
    "tricycle": "tricycle"  
}

DEFAULT_VTYPE_PROPS = {
    "car": {"accel": "2.6", "decel": "4.5", "sigma": "0.5", "length": "4.5", "maxSpeed": "13.9"},
    "motor": {"accel": "3.0", "decel": "5.0", "sigma": "0.5", "length": "2.0", "maxSpeed": "11.0"},
    "jeepney": {"accel": "1.8", "decel": "3.5", "sigma": "0.6", "length": "6.0", "maxSpeed": "12.0"},
    "bus": {"accel": "1.5", "decel": "3.0", "sigma": "0.6", "length": "11.0", "maxSpeed": "11.0"},
    "truck": {"accel": "1.2", "decel": "3.0", "sigma": "0.6", "length": "12.0", "maxSpeed": "10.5"},
    "tricycle": {"accel": "1.5", "decel": "3.0", "sigma": "0.8", "length": "3.0", "maxSpeed": "10.0"}  # <-- added
}

def _pretty_xml(elem):
    return minidom.parseString(tostring(elem)).toprettyxml(indent="  ")


def _read_bundle_minimal(day, cycle):
    """
    Returns a list of per-intersection minimal dicts (as created by compile_bundles)
    """
    bundle_dir = os.path.join(SCEN_DIR, str(day), f"cycle_{cycle}")
    if not os.path.exists(bundle_dir):
        raise FileNotFoundError(f"Bundle folder not found: {bundle_dir}")
    csvs = sorted([os.path.join(bundle_dir, p) for p in os.listdir(bundle_dir) if p.endswith(".csv") and not p.endswith("bundle_meta.csv")])
    if not csvs:
        raise FileNotFoundError(f"No intersection CSVs found in {bundle_dir}")
    rows = []
    for c in csvs:
        df = pd.read_csv(c)
        rows.append(df.iloc[0].to_dict())
    return rows


def build_route_xml(rows, lane_map=None, mode="flow", cycle_time_key="CycleTime_s", seed=42):
    """
    rows: list of per-intersection dicts with keys like IntersectionID, CycleTime_s, jeepney_count, jeepney_pcu, ...
    lane_map: dict mapping intersections -> vehicle class -> {from: [edge], to: [edge]}
    mode: "flow" or "explicit"
    """
    random.seed(seed)

    root = Element("routes")
    # add vType definitions
    for vtype, props in DEFAULT_VTYPE_PROPS.items():
        SubElement(root, "vType", id=vtype, **props)

    # For each intersection, create flows / vehicles
    for info in rows:
        inter = info.get("IntersectionID")
        cycle_s = int(info.get(cycle_time_key, 300))
        scale = 3600.0 / float(cycle_s)

        # lane assignment: from lane_map else use placeholders
        imap = lane_map.get(inter, {}) if lane_map else {}

        # For each vehicle class
        for cls in VEH_CLASS_TO_VTYPE.keys():
            col_count = f"{cls}_count"
            count = int(info.get(col_count, 0))
            if count <= 0:
                continue

            # determine from/to edges
            if cls in imap:
                chosen_from = imap[cls].get("from", imap.get("default", {}).get("from", ["edge_in_1"]))
                chosen_to = imap[cls].get("to", imap.get("default", {}).get("to", ["edge_out_1"]))
            else:
                chosen_from = imap.get("default", {}).get("from", [f"{inter}_in"])
                chosen_to = imap.get("default", {}).get("to", [f"{inter}_out"])

            # pick single from/to (if lists) - simple heuristic; can be extended later
            from_edge = chosen_from[0] if isinstance(chosen_from, (list, tuple)) else chosen_from
            to_edge = chosen_to[0] if isinstance(chosen_to, (list, tuple)) else chosen_to

            vtype = VEH_CLASS_TO_VTYPE[cls]

            if mode == "flow":
                vph = max(1, int(round(count * scale)))
                # create unique id
                flow_id = f"flow_{inter}_{cls}"
                SubElement(root, "flow", id=flow_id, type=vtype, vehsPerHour=str(vph), begin="0", end=str(cycle_s), fromEdge=from_edge, to=to_edge)
            else:
                # explicit mode -> create vehicle entries with depart times spread in 0..cycle_s
                for i in range(count):
                    depart_time = round(random.random() * cycle_s, 3)
                    veh_id = f"veh_{inter}_{cls}_{i}"
                    SubElement(root, "vehicle", id=veh_id, type=vtype, route="r0", depart=str(depart_time))

    return _pretty_xml(root)


def load_lane_map(lane_map_path):
    if not lane_map_path:
        return {}
    with open(lane_map_path, "r") as f:
        lm = json.load(f)
    return lm


def main():
    parser = argparse.ArgumentParser(description="Generate combined .rou.xml from a day+cycle bundle.")
    parser.add_argument("--day", required=True, help="Day identifier (folder name under out/scenarios)")
    parser.add_argument("--cycle", required=True, help="Cycle number (folder cycle_<n>)")
    parser.add_argument("--lane-map", required=False, help="Optional JSON lane map file")
    parser.add_argument("--mode", default="flow", choices=["flow", "explicit"], help="flow: aggregate flows (fast). explicit: one <vehicle> per count.")
    parser.add_argument("--out", default=None, help="Output .rou.xml path (default data/routes/scenario_<day>_<cycle>.rou.xml)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (for explicit mode)")
    args = parser.parse_args()

    rows = _read_bundle_minimal(args.day, args.cycle)
    lane_map = load_lane_map(args.lane_map) if args.lane_map else {}

    xml = build_route_xml(rows, lane_map=lane_map, mode=args.mode, seed=args.seed)

    os.makedirs(ROUTE_OUT_DIR, exist_ok=True)
    out_file = args.out if args.out else os.path.join(ROUTE_OUT_DIR, f"scenario_{args.day}_cycle{args.cycle}.rou.xml")
    with open(out_file, "w") as f:
        f.write(xml)
    print(f"[INFO] Wrote route file: {out_file}")
    print("[INFO] NOTE: If lane_map was not provided, edges are placeholders - replace with real edge IDs from your .net.xml before running SUMO.")

if __name__ == "__main__":
    main()
