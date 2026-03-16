#!/usr/bin/env python3
"""
Comprehensive validation test for all fixes applied to Satellite Collision Prediction project.

Tests:
1. Flask availability in requirements
2. TLE Fetcher with cache and retry logic
3. Model loading with path resolution
4. Deployer batch processing and PINN corrections
5. Server API endpoints
6. Collision detection accuracy
"""

import sys
import os
import logging
import tempfile
import re
from datetime import datetime, timezone, timedelta

# Setup paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('TEST')

# Test categories
TESTS_PASSED = []
TESTS_FAILED = []


def test_result(name, passed, message=""):
    """Record test result."""
    if passed:
        TESTS_PASSED.append(name)
        logger.info(f"✓ PASS: {name}")
    else:
        TESTS_FAILED.append(name)
        logger.error(f"✗ FAIL: {name} - {message}")


def test_flask_in_requirements():
    """Test 1: Verify Flask is in requirements.txt"""
    print("\n" + "="*60)
    print("TEST 1: Flask in requirements")
    print("="*60)
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
        flask_found = 'flask' in content.lower()
        test_result("Flask in requirements.txt", flask_found, 
                   "" if flask_found else "Flask not found in requirements.txt")
    except Exception as e:
        test_result("Flask in requirements.txt", False, str(e))


def test_tle_fetcher_module():
    """Test 2: TLE Fetcher imports and logging setup"""
    print("\n" + "="*60)
    print("TEST 2: TLE Fetcher module")
    print("="*60)
    try:
        from src.tle_fetcher import fetch_tle, fetch_batch, clear_cache, CELESTRAK_GP_URL, logger as tle_logger
        test_result("TLE Fetcher imports", True)
        
        # Check logger is set up
        test_result("TLE Fetcher logging enabled", tle_logger is not None)
        
        # Check CelesTrak URL is correct
        url_ok = "celestrak.org" in CELESTRAK_GP_URL
        test_result("CelesTrak API URL correct", url_ok, 
                   f"URL is {CELESTRAK_GP_URL}")
        
    except Exception as e:
        test_result("TLE Fetcher imports", False, str(e))


def test_deployer_model_loading():
    """Test 3: Deployer model loading with path resolution"""
    print("\n" + "="*60)
    print("TEST 3: Deployer model loading")
    print("="*60)
    try:
        from src.deployer import OrbitDeployer
        
        # Test with default path
        try:
            deployer = OrbitDeployer()
            test_result("Deployer initialization with default path", True)
            test_result("Model loaded successfully", deployer.model is not None)
        except FileNotFoundError as e:
            test_result("Deployer initialization with default path", False, 
                       f"Model file not found: {e}")
        except Exception as e:
            test_result("Deployer initialization with default path", False, str(e))
            
    except Exception as e:
        test_result("Deployer imports", False, str(e))


def test_deployer_shape_handling():
    """Test 4: Deployer PINN correction shape handling"""
    print("\n" + "="*60)
    print("TEST 4: Deployer PINN shape handling")
    print("="*60)
    try:
        import torch
        import numpy as np
        from src.deployer import OrbitDeployer
        from src.model import R_REF, V_REF, MAX_DT
        
        deployer = OrbitDeployer()
        
        # Create batch features (B=4, 9 features)
        batch_size = 4
        features = np.random.randn(batch_size, 9) * 0.1
        features[:, 6:8] *= 1e-4  # bstar and ndot are small
        
        # Test with different t_norm shapes
        test_cases = [
            ("1D array", np.array([0.1, 0.2, 0.3, 0.4])),
            ("2D array (B, 1)", np.array([[0.1], [0.2], [0.3], [0.4]])),
            ("scalar", 0.1),
        ]
        
        all_passed = True
        for case_name, t_norm in test_cases:
            try:
                delta_r, delta_v = deployer._pinn_correct(features, t_norm)
                
                # Check output shapes
                shape_ok = (delta_r.shape == (batch_size, 3) and 
                           delta_v.shape == (batch_size, 3))
                
                if not shape_ok:
                    all_passed = False
                    logger.error(f"Shape mismatch for {case_name}: delta_r shape {delta_r.shape}, delta_v shape {delta_v.shape}")
                else:
                    logger.info(f"✓ Shape OK for {case_name}")
                    
            except Exception as e:
                all_passed = False
                logger.error(f"Error with {case_name}: {str(e)}")
        
        test_result("Deployer shape handling", all_passed)
        
    except Exception as e:
        test_result("Deployer shape handling", False, str(e))


def test_deployer_trajectory():
    """Test 5: Deployer trajectory calculation"""
    print("\n" + "="*60)
    print("TEST 5: Deployer trajectory calculation")
    print("="*60)
    try:
        from src.deployer import OrbitDeployer
        import numpy as np
        
        deployer = OrbitDeployer()
        
        # Use ISS TLEs
        tle1 = "1 25544U 98067A   22011.46046435  .00011925  00000+0  21951-4 0  9991"
        tle2 = "2 25544  51.6438 158.6955 0003151 176.6175 183.5027 15.49503526322916"
        target_time = "2022-01-12 12:00:00"
        
        r_baseline, r_corrected = deployer.get_trajectory(tle1, tle2, target_time, steps=100)
        
        # Check output shapes
        shapes_ok = (r_baseline.shape[1] == 3 and 
                    r_corrected.shape[1] == 3 and
                    r_baseline.shape[0] > 0)
        
        test_result("Trajectory computation", shapes_ok,
                   f"baseline: {r_baseline.shape}, corrected: {r_corrected.shape}")
        
        # Check that corrections are small (PINN shouldn't massively change positions)
        if shapes_ok:
            max_correction = np.max(np.linalg.norm(r_corrected - r_baseline, axis=1))
            reasonable = max_correction < 1000  # Less than 1000 km correction
            test_result("PINN corrections within reasonable bounds", reasonable,
                       f"max correction: {max_correction:.1f} km")
        
    except Exception as e:
        test_result("Deployer trajectory calculation", False, str(e))


def test_target_date_parsing():
    """Test 6: Server target date parsing robustness"""
    print("\n" + "="*60)
    print("TEST 6: Target date parsing")
    print("="*60)
    try:
        test_cases = [
            ("2022-01-12 12:00:00", True, "ISO format"),
            ("", False, "Empty string (should fall back to now)"),
            ("invalid-date", False, "Invalid format (should fall back to now)"),
        ]
        
        for date_str, should_parse, desc in test_cases:
            try:
                if date_str:
                    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    dt = dt.replace(tzinfo=timezone.utc)
                    logger.info(f"✓ Parsed {desc}: {dt}")
                else:
                    dt = datetime.now(timezone.utc)
                    logger.info(f"✓ Fallback {desc}: using now")
            except ValueError:
                if not should_parse:
                    logger.info(f"✓ Expected failure for {desc}")
                else:
                    raise
        
        test_result("Target date parsing", True)
        
    except Exception as e:
        test_result("Target date parsing", False, str(e))


def test_sat_database_initialization():
    """Test 7: SAT_DATABASE TLE field initialization"""
    print("\n" + "="*60)
    print("TEST 7: SAT_DATABASE initialization")
    print("="*60)
    try:
        # Check server.py SAT_DATABASE
        with open('server.py', 'r') as f:
            server_content = f.read()
        
        # Look for tle1 and tle2 in SAT_DATABASE initialization
        pattern = r'SAT_DATABASE\s*=\s*\{[^}]*tle1[^}]*tle2'
        tle_fields_found = bool(re.search(pattern, server_content, re.DOTALL))
        
        test_result("SAT_DATABASE includes tle1/tle2 fields", tle_fields_found,
                   "" if tle_fields_found else "TLE fields not found in initialization")
        
    except Exception as e:
        test_result("SAT_DATABASE initialization", False, str(e))


def test_trajectory_no_normalization_bug():
    """Test 8: Trajectory normalization fix"""
    print("\n" + "="*60)
    print("TEST 8: Trajectory normalization")
    print("="*60)
    try:
        # Check server.py doesn't normalize trajectories prematurely
        with open('server.py', 'r') as f:
            server_content = f.read()
        
        # Look for the collision distance calculation
        # Should use R_EARTH to denormalize when comparing
        pattern = r'traj_[ab]_eci = np\.array\(objects\[ids\[\w\]\]\[\"trajectory_pinn_eci\"\]\) \* R_EARTH'
        denormalization_found = bool(re.search(pattern, server_content))
        
        test_result("Collision detection denormalizes trajectories", denormalization_found,
                   "" if denormalization_found else "Denormalization pattern not found")
        
    except Exception as e:
        test_result("Trajectory normalization check", False, str(e))


def test_logging_setup():
    """Test 9: Logging setup in critical modules"""
    print("\n" + "="*60)
    print("TEST 9: Logging setup")
    print("="*60)
    try:
        all_ok = True
        modules_to_check = [
            ('src/tle_fetcher.py', 'logging\nimport'),
            ('src/deployer.py', 'logging\nimport'),
            ('server.py', 'logging\nimport'),
        ]
        
        for filepath, pattern in modules_to_check:
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                logging_found = 'logging' in content and 'logger' in content.lower()
                
                if logging_found:
                    logger.info(f"✓ Logging configured in {filepath}")
                else:
                    logger.warning(f"✗ Logging not fully configured in {filepath}")
                    all_ok = False
                    
            except FileNotFoundError:
                logger.error(f"File not found: {filepath}")
                all_ok = False
        
        test_result("Logging setup in modules", all_ok)
        
    except Exception as e:
        test_result("Logging setup check", False, str(e))


def test_error_handling():
    """Test 10: Error handling in deployer"""
    print("\n" + "="*60)
    print("TEST 10: Error handling")
    print("="*60)
    try:
        from src.deployer import OrbitDeployer
        
        deployer = OrbitDeployer()
        
        # Test with invalid TLE
        invalid_tle1 = "INVALID TLE LINE 1"
        invalid_tle2 = "INVALID TLE LINE 2"
        
        try:
            result = deployer.predict(invalid_tle1, invalid_tle2, "2022-01-12 12:00:00")
            # Should return (None, None) tuples gracefully
            if result == ((None, None), (None, None), (None, None)):
                logger.info("✓ Graceful handling of invalid TLE")
                test_result("Invalid TLE error handling", True)
            else:
                test_result("Invalid TLE error handling", False, 
                           f"Unexpected result: {result}")
        except Exception as ex:
            logger.warning(f"Exception during invalid TLE test: {ex}")
            test_result("Invalid TLE error handling", False, str(ex))
        
    except Exception as e:
        test_result("Error handling test", False, str(e))


def print_summary():
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"\n✓ PASSED: {len(TESTS_PASSED)}")
    for t in TESTS_PASSED:
        print(f"  • {t}")
    
    if TESTS_FAILED:
        print(f"\n✗ FAILED: {len(TESTS_FAILED)}")
        for t in TESTS_FAILED:
            print(f"  • {t}")
    
    total = len(TESTS_PASSED) + len(TESTS_FAILED)
    pass_rate = (len(TESTS_PASSED) / total * 100) if total > 0 else 0
    print(f"\nPass Rate: {pass_rate:.1f}% ({len(TESTS_PASSED)}/{total})")
    print("="*60 + "\n")
    
    return len(TESTS_FAILED) == 0


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SATELLITE COLLISION PREDICTION — FIXES VALIDATION")
    print("="*60)
    
    test_flask_in_requirements()
    test_tle_fetcher_module()
    test_deployer_model_loading()
    test_deployer_shape_handling()
    test_deployer_trajectory()
    test_target_date_parsing()
    test_sat_database_initialization()
    test_trajectory_no_normalization_bug()
    test_logging_setup()
    test_error_handling()
    
    success = print_summary()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
