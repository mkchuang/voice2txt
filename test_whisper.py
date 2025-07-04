#!/usr/bin/env python3
"""
Whisper 引擎測試腳本

測試項目：
1. WhisperEngine 基礎功能
2. 模型載入和快取
3. GPU/CPU 自動偵測
4. 音訊轉錄功能
5. 中文識別優化
6. 效能評估

使用方式：
python test_whisper.py [--model tiny] [--device cpu] [--test-audio]
"""

import sys
import os
import time
import numpy as np
import logging
from pathlib import Path

# 將 src 目錄加入 Python 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.core.whisper_engine import WhisperEngine, WhisperModel, EngineState
except ImportError as e:
    print(f"無法載入 WhisperEngine: {e}")
    print("請確保 src/core/whisper_engine.py 存在")
    sys.exit(1)

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_engine_creation():
    """測試引擎建立"""
    print("\n=== 測試引擎建立 ===")
    
    try:
        # 測試不同模型大小
        for model in ["tiny", "base"]:  # 只測試小模型避免下載時間過長
            print(f"\n建立 {model} 模型引擎...")
            engine = WhisperEngine(
                model_name=model,
                language="zh",
                device="cpu"  # 強制使用 CPU 測試
            )
            
            print(f"✓ {model} 引擎建立成功")
            print(f"  狀態: {engine.state.value}")
            print(f"  設備: {engine._device}")
            print(f"  語言: {engine.language}")
            
            # 清理
            engine.cleanup()
            
    except Exception as e:
        print(f"✗ 引擎建立失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ 引擎建立測試通過")
    return True


def test_device_detection():
    """測試設備偵測"""
    print("\n=== 測試設備偵測 ===")
    
    try:
        # 測試自動偵測
        engine = WhisperEngine(model_name="tiny")
        detected_device = engine._device
        print(f"自動偵測設備: {detected_device}")
        
        # 測試手動指定
        engine_cpu = WhisperEngine(model_name="tiny", device="cpu")
        print(f"手動指定 CPU: {engine_cpu._device}")
        
        # 檢查 GPU 可用性
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"✓ GPU 可用: {gpu_name} ({gpu_memory:.1f}GB)")
                print(f"  GPU 數量: {gpu_count}")
            else:
                print("⚠ GPU 不可用，將使用 CPU")
        except ImportError:
            print("⚠ PyTorch 未安裝，無法偵測 GPU")
        
        # 清理
        engine.cleanup()
        engine_cpu.cleanup()
        
    except Exception as e:
        print(f"✗ 設備偵測失敗: {e}")
        return False
    
    print("✓ 設備偵測測試通過")
    return True


def test_model_initialization():
    """測試模型初始化"""
    print("\n=== 測試模型初始化 ===")
    
    try:
        engine = WhisperEngine(model_name="tiny", device="cpu")
        
        # 測試進度回調
        progress_messages = []
        def progress_callback(message, progress):
            progress_messages.append(f"{message} ({progress:.0%})")
            print(f"  進度: {message} ({progress:.0%})")
        
        print("開始初始化模型...")
        start_time = time.time()
        
        success = engine.initialize(progress_callback=progress_callback)
        
        init_time = time.time() - start_time
        
        if success:
            print(f"✓ 模型初始化成功 ({init_time:.2f}s)")
            print(f"  引擎狀態: {engine.state.value}")
            print(f"  是否就緒: {engine.is_ready()}")
            
            # 檢查統計資訊
            stats = engine.get_stats()
            print(f"  統計資訊:")
            print(f"    模型載入時間: {stats['model_load_time']:.2f}s")
            print(f"    設備: {stats['device']}")
            print(f"    模型: {stats['model_name']}")
            
        else:
            print("✗ 模型初始化失敗")
            return False
        
        # 清理
        engine.cleanup()
        
    except Exception as e:
        print(f"✗ 模型初始化測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ 模型初始化測試通過")
    return True


def test_audio_preprocessing():
    """測試音訊預處理"""
    print("\n=== 測試音訊預處理 ===")
    
    try:
        engine = WhisperEngine(model_name="tiny", device="cpu")
        
        # 生成測試音訊數據
        sample_rate = 44100  # 非標準取樣率
        duration = 3.0  # 3 秒
        samples = int(sample_rate * duration)
        
        # 生成正弦波測試信號
        t = np.linspace(0, duration, samples)
        frequency = 440  # A4 音符
        audio_data = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
        
        print(f"原始音訊: {audio_data.dtype}, {len(audio_data)} 樣本, {sample_rate}Hz")
        
        # 測試預處理
        processed = engine._preprocess_audio(audio_data, sample_rate)
        
        print(f"處理後音訊: {processed.dtype}, {len(processed)} 樣本, 16000Hz")
        print(f"  數值範圍: [{processed.min():.3f}, {processed.max():.3f}]")
        
        # 驗證結果
        assert processed.dtype == np.float32, "類型應為 float32"
        assert abs(processed.max()) <= 1.0, "數值應在 [-1, 1] 範圍內"
        assert abs(processed.min()) <= 1.0, "數值應在 [-1, 1] 範圍內"
        
        expected_length = len(audio_data) * 16000 // sample_rate
        assert abs(len(processed) - expected_length) < 100, "重採樣長度正確"
        
        print("✓ 音訊預處理驗證通過")
        
        # 清理
        engine.cleanup()
        
    except Exception as e:
        print(f"✗ 音訊預處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ 音訊預處理測試通過")
    return True


def test_transcription_simulation():
    """測試轉錄功能（模擬）"""
    print("\n=== 測試轉錄功能（模擬）===")
    
    try:
        engine = WhisperEngine(model_name="tiny", device="cpu")
        
        # 初始化引擎
        if not engine.initialize():
            print("✗ 引擎初始化失敗")
            return False
        
        # 生成測試音訊（靜音）
        duration = 2.0  # 2 秒靜音
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio_data = np.zeros(samples, dtype=np.float32)
        
        print(f"測試音訊: {len(audio_data)} 樣本, {duration}s 靜音")
        
        # 執行轉錄
        print("開始轉錄...")
        start_time = time.time()
        
        result = engine.transcribe(audio_data, sample_rate)
        
        transcribe_time = time.time() - start_time
        
        print(f"✓ 轉錄完成 ({transcribe_time:.2f}s)")
        print(f"  轉錄文字: '{result['text']}'")
        print(f"  偵測語言: {result['language']}")
        print(f"  信心度: {result['confidence']:.3f}")
        print(f"  處理時間: {result['processing_time']:.3f}s")
        print(f"  片段數量: {len(result['segments'])}")
        
        # 檢查統計更新
        stats = engine.get_stats()
        print(f"  引擎統計:")
        print(f"    總轉錄次數: {stats['total_transcriptions']}")
        print(f"    平均處理時間: {stats['average_processing_time']:.3f}s")
        
        # 清理
        engine.cleanup()
        
    except Exception as e:
        print(f"✗ 轉錄測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ 轉錄功能測試通過")
    return True


def test_chinese_processing():
    """測試中文處理"""
    print("\n=== 測試中文處理 ===")
    
    try:
        engine = WhisperEngine(model_name="tiny", language="zh", device="cpu")
        
        # 測試中文文本後處理
        test_texts = [
            "你好  世界",  # 多餘空格
            "這是一個測試，包含標點符號。",  # 標點符號
            "   前後空格   ",  # 前後空格
        ]
        
        for text in test_texts:
            processed = engine._process_chinese_text(text)
            print(f"原始: '{text}' -> 處理: '{processed}'")
        
        # 測試轉錄選項
        options = engine.transcribe_options
        print(f"\n中文轉錄選項:")
        for key, value in options.items():
            print(f"  {key}: {value}")
        
        assert options["language"] == "zh", "語言應設為中文"
        assert options["temperature"] == 0.0, "temperature 應為 0"
        
        # 清理
        engine.cleanup()
        
    except Exception as e:
        print(f"✗ 中文處理測試失敗: {e}")
        return False
    
    print("✓ 中文處理測試通過")
    return True


def test_error_handling():
    """測試錯誤處理"""
    print("\n=== 測試錯誤處理 ===")
    
    try:
        # 測試無效模型
        try:
            engine = WhisperEngine(model_name="invalid_model")
            engine.initialize()
            print("✗ 應該拒絕無效模型")
            return False
        except Exception:
            print("✓ 正確拒絕無效模型")
        
        # 測試未初始化轉錄
        engine = WhisperEngine(model_name="tiny", device="cpu")
        audio_data = np.zeros(1000, dtype=np.float32)
        
        try:
            # 第一次轉錄會自動初始化
            result = engine.transcribe(audio_data)
            print("✓ 自動初始化轉錄成功")
        except Exception as e:
            print(f"⚠ 自動初始化失敗: {e}")
        
        # 清理
        engine.cleanup()
        
    except Exception as e:
        print(f"✗ 錯誤處理測試失敗: {e}")
        return False
    
    print("✓ 錯誤處理測試通過")
    return True


def test_performance():
    """測試效能指標"""
    print("\n=== 測試效能指標 ===")
    
    try:
        engine = WhisperEngine(model_name="tiny", device="cpu")
        
        # 初始化並記錄時間
        print("測試初始化效能...")
        start_time = time.time()
        success = engine.initialize()
        init_time = time.time() - start_time
        
        if not success:
            print("✗ 初始化失敗")
            return False
        
        print(f"✓ 初始化時間: {init_time:.2f}s")
        
        # 測試多次轉錄的效能
        print("測試轉錄效能...")
        audio_data = np.zeros(16000, dtype=np.float32)  # 1 秒靜音
        
        times = []
        for i in range(3):
            start_time = time.time()
            result = engine.transcribe(audio_data)
            transcribe_time = time.time() - start_time
            times.append(transcribe_time)
            print(f"  第 {i+1} 次轉錄: {transcribe_time:.3f}s")
        
        avg_time = sum(times) / len(times)
        print(f"✓ 平均轉錄時間: {avg_time:.3f}s")
        
        # 檢查效能目標
        if init_time > 10.0:
            print(f"⚠ 初始化時間過長: {init_time:.2f}s > 10s")
        
        if avg_time > 5.0:
            print(f"⚠ 轉錄時間過長: {avg_time:.3f}s > 5s")
        
        # 檢查統計
        stats = engine.get_stats()
        print(f"\n最終統計:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        # 清理
        engine.cleanup()
        
    except Exception as e:
        print(f"✗ 效能測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ 效能測試通過")
    return True


def main():
    """主測試函數"""
    print("=== Whisper 引擎完整測試 ===")
    print(f"Python 版本: {sys.version}")
    
    # 檢查依賴
    missing_deps = []
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import whisper
        print(f"OpenAI Whisper 可用")
    except ImportError:
        missing_deps.append("openai-whisper")
    
    if missing_deps:
        print(f"\n⚠️  缺少依賴套件: {', '.join(missing_deps)}")
        print("請安裝: pip install torch torchaudio openai-whisper")
        print("將跳過需要這些套件的測試")
    
    # 執行測試
    tests = [
        ("引擎建立", test_engine_creation),
        ("設備偵測", test_device_detection),
        ("音訊預處理", test_audio_preprocessing),
        ("中文處理", test_chinese_processing),
        ("錯誤處理", test_error_handling),
    ]
    
    # 只有在依賴完整時才執行需要模型的測試
    if not missing_deps:
        tests.extend([
            ("模型初始化", test_model_initialization),
            ("轉錄功能", test_transcription_simulation),
            ("效能測試", test_performance),
        ])
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"測試: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通過")
            else:
                print(f"❌ {test_name} 失敗")
        except KeyboardInterrupt:
            print(f"\n用戶中斷測試")
            break
        except Exception as e:
            print(f"❌ {test_name} 異常: {e}")
    
    # 測試總結
    print(f"\n{'='*50}")
    print(f"測試總結: {passed}/{total} 通過")
    print('='*50)
    
    if passed == total:
        print("🎉 所有測試通過！Whisper 引擎已準備就緒")
        return 0
    else:
        print(f"⚠️  {total - passed} 個測試失敗")
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Whisper 引擎測試")
    parser.add_argument("--model", default="tiny", help="測試模型大小")
    parser.add_argument("--device", help="指定計算設備")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細輸出")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    sys.exit(main())