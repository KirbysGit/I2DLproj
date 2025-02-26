import unittest
from model.test_backbone import test_backbone
from model.test_fpn import test_fpn
from model.test_detector import test_detector

def run_all_tests():
    """Run all model tests"""
    print("Running all tests...")
    
    try:
        # Test backbone
        test_backbone()
        
        # Test FPN
        test_fpn()
        
        # Test detector
        test_detector()
        
        print("\nAll tests completed successfully!")
        
    except AssertionError as e:
        print(f"\nTest failed: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

if __name__ == "__main__":
    run_all_tests() 