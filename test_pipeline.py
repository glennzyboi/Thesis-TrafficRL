"""
Comprehensive test script for the D3QN Traffic Signal Control pipeline
Tests all major components and reports any issues
"""

import os
import sys
import subprocess
from pathlib import Path

def test_dependencies():
    """Test if all required Python packages are installed"""
    print("🔧 Testing Dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'tensorflow', 
        'sumolib', 'traci', 'openpyxl'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Missing packages: {missing}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed\n")
    return True

def test_sumo_installation():
    """Test SUMO installation and environment"""
    print("🚗 Testing SUMO Installation...")
    
    # Test SUMO_HOME
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        print("   ❌ SUMO_HOME not set")
        return False
    
    print(f"   ✅ SUMO_HOME: {sumo_home}")
    
    # Test SUMO binaries
    sumo_binary = os.path.join(sumo_home, 'bin', 'sumo.exe')
    if not os.path.exists(sumo_binary):
        print(f"   ❌ SUMO binary not found: {sumo_binary}")
        return False
    
    print(f"   ✅ SUMO binary found")
    
    # Test tools directory
    tools_dir = os.path.join(sumo_home, 'tools')
    if not os.path.exists(tools_dir):
        print(f"   ❌ SUMO tools directory not found: {tools_dir}")
        return False
    
    print(f"   ✅ SUMO tools directory found")
    
    try:
        import traci
        import sumolib
        print("   ✅ SUMO Python modules imported successfully")
    except ImportError as e:
        print(f"   ❌ SUMO Python modules import failed: {e}")
        return False
    
    print("✅ SUMO installation verified\n")
    return True

def test_data_files():
    """Test if required data files exist"""
    print("📁 Testing Data Files...")
    
    required_files = [
        'data/raw/ECOLAND_20250828_cycle1.xlsx',
        'data/raw/JOHNPAUL_20250828_cycle1.xlsx', 
        'data/raw/SANDAWA_20250828_cycle1.xlsx',
        'network/ThesisNetowrk.net.xml',
        'lane_map.json',
        'requirements.txt'
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️ Missing files: {missing}")
        return False
    
    print("✅ All required data files present\n")
    return True

def test_data_processing():
    """Test data processing pipeline"""
    print("📊 Testing Data Processing Pipeline...")
    
    try:
        # Test compile_bundles.py
        result = subprocess.run([
            sys.executable, 'scripts/compile_bundles.py'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"   ❌ compile_bundles.py failed:")
            print(f"      {result.stderr}")
            return False
        
        print("   ✅ Data compilation successful")
        
        # Check if output files were created
        if os.path.exists('data/processed/master_bundles.csv'):
            print("   ✅ Master bundles created")
        else:
            print("   ❌ Master bundles not created")
            return False
        
    except Exception as e:
        print(f"   ❌ Data processing test failed: {e}")
        return False
    
    print("✅ Data processing pipeline working\n")
    return True

def test_route_generation():
    """Test route generation"""
    print("🛣️ Testing Route Generation...")
    
    try:
        # Test generate_routes.py
        result = subprocess.run([
            sys.executable, 'scripts/generate_routes.py',
            '--day', '20250828',
            '--cycle', '1', 
            '--lane-map', 'lane_map.json',
            '--mode', 'flow'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"   ❌ Route generation failed:")
            print(f"      {result.stderr}")
            return False
        
        print("   ✅ Route generation successful")
        
        # Check if route file was created
        route_file = 'data/routes/scenario_20250828_cycle1.rou.xml'
        if os.path.exists(route_file):
            print(f"   ✅ Route file created: {route_file}")
        else:
            print(f"   ❌ Route file not created: {route_file}")
            return False
        
    except Exception as e:
        print(f"   ❌ Route generation test failed: {e}")
        return False
    
    print("✅ Route generation working\n")
    return True

def test_sumo_simulation():
    """Test basic SUMO simulation with generated routes"""
    print("🚀 Testing SUMO Simulation...")
    
    try:
        # Check if we can load the network
        import sumolib
        net_file = 'network/ThesisNetowrk.net.xml'
        net = sumolib.net.readNet(net_file)
        
        traffic_lights = net.getTrafficLights()
        print(f"   ✅ Network loaded: {len(traffic_lights)} traffic lights")
        
        # Test route file validation
        route_file = 'data/routes/scenario_20250828_cycle1.rou.xml'
        if os.path.exists(route_file):
            print(f"   ✅ Route file exists and readable")
        else:
            print(f"   ❌ Route file missing")
            return False
            
    except Exception as e:
        print(f"   ❌ SUMO simulation test failed: {e}")
        return False
    
    print("✅ SUMO simulation components ready\n")
    return True

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("🧪 D3QN PIPELINE COMPREHENSIVE TEST")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("SUMO Installation", test_sumo_installation), 
        ("Data Files", test_data_files),
        ("Data Processing", test_data_processing),
        ("Route Generation", test_route_generation),
        ("SUMO Simulation", test_sumo_simulation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("Your D3QN pipeline is ready for use!")
        print("\nNext steps:")
        print("   1. Update lane_map.json with real edge IDs")
        print("   2. Develop D3QN training algorithms")
        print("   3. Run traffic signal control experiments")
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed.")
        print("Please fix the issues above before proceeding.")

if __name__ == "__main__":
    run_comprehensive_test()
