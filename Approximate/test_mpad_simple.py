#!/usr/bin/env python3
"""
简单测试MPAD功能
使用小数据集快速验证MPAD是否正常工作
"""

import os
import sys
import numpy as np
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_mpad_import():
    """测试MPAD类是否能正常导入"""
    print("=== Testing MPAD Import ===")
    try:
        from main_program import MPAD
        print("[OK] MPAD class imported successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to import MPAD: {e}")
        return False

def test_mpad_basic():
    """测试MPAD基本功能"""
    print("\n=== Testing MPAD Basic Functionality ===")
    
    try:
        from main_program import MPAD
        
        # 加载小数据集
        print("Loading Fasttext 1% dataset...")
        X_train = np.load("training_vectors_01pct_Fasttext.npy")
        X_test = np.load("testing_vectors_01pct_Fasttext.npy")
        
        print(f"[OK] Loaded: train={X_train.shape}, test={X_test.shape}")
        
        # 测试参数
        target_dim = 64  # 较小维度，快速测试
        b_percentage = 1.0
        alpha = 0.1
        
        print(f"\nTesting MPAD with: target_dim={target_dim}, b={b_percentage}%, alpha={alpha}")
        
        # 创建MPAD实例（参数在初始化时传入）
        mpad = MPAD(b_percentage=b_percentage, alpha=alpha, target_dim=target_dim)
        
        # 训练
        print("Training MPAD...")
        start_time = time.time()
        X_train_reduced = mpad.fit_transform(X_train)
        train_time = time.time() - start_time
        
        print(f"[OK] Training completed in {train_time:.2f}s")
        print(f"[OK] Reduced train shape: {X_train_reduced.shape}")
        
        # 转换测试数据
        print("Transforming test data...")
        start_time = time.time()
        X_test_reduced = mpad.transform(X_test)
        transform_time = time.time() - start_time
        
        print(f"[OK] Transform completed in {transform_time:.2f}s")
        print(f"[OK] Reduced test shape: {X_test_reduced.shape}")
        
        # 检查输出
        assert X_train_reduced.shape[1] == target_dim, f"Expected dim {target_dim}, got {X_train_reduced.shape[1]}"
        assert X_test_reduced.shape[1] == target_dim, f"Expected dim {target_dim}, got {X_test_reduced.shape[1]}"
        assert not np.any(np.isnan(X_train_reduced)), "Found NaN in train output"
        assert not np.any(np.isnan(X_test_reduced)), "Found NaN in test output"
        
        print("[OK] All checks passed!")
        
        # 显示一些统计信息
        print(f"\nStatistics:")
        print(f"  Train reduced: min={X_train_reduced.min():.4f}, max={X_train_reduced.max():.4f}, mean={X_train_reduced.mean():.4f}")
        print(f"  Test reduced:  min={X_test_reduced.min():.4f}, max={X_test_reduced.max():.4f}, mean={X_test_reduced.mean():.4f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] MPAD test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mpad_with_methods():
    """测试MPAD在main_program中的使用"""
    print("\n=== Testing MPAD in Main Program ===")
    
    try:
        from main_program import main_evaluation
        
        print("Running quick evaluation with MPAD only...")
        print("(This will take a few minutes)")
        
        results, detailed_file, summary_file = main_evaluation(
            dataset_name="Fasttext_test",
            train_file="training_vectors_01pct_Fasttext.npy",
            test_file="testing_vectors_01pct_Fasttext.npy",
            target_dim=64,  # 较小的维度
            b_percentage=1.0,
            alpha=0.1,
            k_values=[1, 10],  # 只测试k=1和k=10
            save_results=True,
            output_dir="Result/test_mpad"
        )
        
        print(f"[OK] Evaluation completed")
        print(f"[OK] Results saved to: {detailed_file}")
        
        # 检查MPAD结果
        if 'MPAD' in results:
            mpad_results = results['MPAD']
            print(f"\nMPAD Results:")
            print(f"  DR time: {mpad_results.get('dr_time', 'N/A'):.2f}s")
            print(f"  DR memory: {mpad_results.get('dr_memory', 'N/A'):.2f}MB")
            
            # 检查各索引方法的结果
            for index_name, index_results in mpad_results.items():
                if index_name not in ['dr_time', 'dr_memory', 'gt_time', 'gt_memory']:
                    print(f"\n  {index_name}:")
                    for k, k_result in index_results.items():
                        if isinstance(k_result, dict):
                            recall = k_result.get('recall', 'N/A')
                            search_time = k_result.get('time', 'N/A')
                            print(f"    k={k}: Recall={recall:.4f}, Time={search_time:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] MPAD evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有MPAD测试"""
    print("Starting MPAD Simple Test Suite")
    print("=" * 60)
    
    tests = [
        ("MPAD Import", test_mpad_import),
        ("MPAD Basic", test_mpad_basic),
        # ("MPAD with Methods", test_mpad_with_methods),  # 注释掉，避免运行太长时间
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n[OK] {test_name} test PASSED\n")
            else:
                print(f"\n[ERROR] {test_name} test FAILED\n")
        except Exception as e:
            print(f"\n[ERROR] {test_name} test ERROR: {e}\n")
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] MPAD is working correctly!")
    else:
        print("[WARNING] Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
