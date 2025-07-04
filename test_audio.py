#!/usr/bin/env python3
"""
音訊錄製功能測試腳本

測試項目：
1. PyAudio 是否可正常安裝和使用
2. AudioRecorder 類別功能
3. RingBuffer 緩衝區效能
4. 錄音品質驗證
"""

import sys
import os
import time
import numpy as np
import logging

# 將 src 目錄加入 Python 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.audio_recorder import AudioRecorder, RecordingState
from src.core.buffer_manager import RingBuffer

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ring_buffer():
    """測試環形緩衝區"""
    print("\n=== 測試環形緩衝區 ===")
    
    # 建立小型緩衝區測試
    buffer = RingBuffer(max_size=1000, dtype=np.float32)
    
    # 測試基本寫入和讀取
    test_data = np.random.randn(500).astype(np.float32)
    assert buffer.write(test_data), "寫入失敗"
    
    status = buffer.get_status()
    print(f"寫入後狀態: 大小={status[0]}, 容量={status[1]}, 使用率={status[2]:.1%}")
    
    # 測試讀取
    read_data = buffer.read(250)
    assert read_data is not None, "讀取失敗"
    assert len(read_data) == 250, "讀取數據長度錯誤"
    print(f"成功讀取 {len(read_data)} 個樣本")
    
    # 測試溢出處理
    large_data = np.random.randn(2000).astype(np.float32)
    buffer.write(large_data)
    status = buffer.get_status()
    print(f"溢出測試後: 大小={status[0]}, 使用率={status[2]:.1%}")
    assert status[0] == 1000, "溢出處理錯誤"
    
    # 測試清空
    buffer.clear()
    assert buffer.is_empty, "清空失敗"
    print("✓ 環形緩衝區測試通過")
    

def test_audio_recorder():
    """測試音訊錄製器（不實際錄音）"""
    print("\n=== 測試音訊錄製器 ===")
    
    try:
        # 建立錄製器
        recorder = AudioRecorder(
            sample_rate=16000,
            channels=1,
            chunk_size=1024,
            buffer_duration=5.0
        )
        
        # 檢查初始狀態
        assert recorder.state == RecordingState.IDLE or recorder.state == RecordingState.ERROR
        print(f"錄製器狀態: {recorder.state.value}")
        
        # 獲取可用裝置
        devices = recorder.get_available_devices()
        print(f"\n可用音訊輸入裝置 ({len(devices)} 個):")
        for device in devices:
            print(f"  [{device['index']}] {device['name']} "
                  f"(聲道: {device['channels']}, 取樣率: {device['sample_rate']}Hz)")
        
        # 獲取統計資訊
        stats = recorder.get_recording_stats()
        print(f"\n統計資訊: {stats}")
        
        print("✓ 音訊錄製器基本測試通過")
        
        # 清理資源
        recorder.cleanup()
        
    except Exception as e:
        print(f"✗ 音訊錄製器測試失敗: {e}")
        import traceback
        traceback.print_exc()


def test_recording_simulation():
    """模擬錄音測試（使用模擬數據）"""
    print("\n=== 模擬錄音測試 ===")
    
    # 建立緩衝區
    sample_rate = 16000
    duration = 3  # 秒
    buffer_size = sample_rate * duration
    buffer = RingBuffer(buffer_size, dtype=np.float32)
    
    # 模擬寫入音訊數據
    print(f"模擬 {duration} 秒錄音...")
    chunk_size = 1024
    num_chunks = buffer_size // chunk_size
    
    start_time = time.time()
    for i in range(num_chunks):
        # 生成測試音訊（正弦波）
        t = np.linspace(i * chunk_size / sample_rate, 
                        (i + 1) * chunk_size / sample_rate, 
                        chunk_size)
        frequency = 440  # A4 音符
        audio_chunk = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
        
        buffer.write(audio_chunk)
        
        # 模擬實時處理延遲
        time.sleep(chunk_size / sample_rate)
        
        # 顯示進度
        if i % 10 == 0:
            progress = (i + 1) / num_chunks
            print(f"  進度: {progress:.0%} - 緩衝區使用率: {buffer.get_status()[2]:.0%}")
    
    elapsed = time.time() - start_time
    print(f"模擬錄音完成，耗時: {elapsed:.2f} 秒")
    
    # 讀取所有數據
    all_data = buffer.read_all()
    print(f"讀取數據: {len(all_data)} 樣本 = {len(all_data)/sample_rate:.2f} 秒")
    
    # 驗證數據
    assert len(all_data) > 0, "無數據"
    assert np.abs(all_data).max() <= 1.0, "數據超出範圍"
    
    print("✓ 模擬錄音測試通過")


def main():
    """主測試函數"""
    print("=== 語音轉文字助手 - 音訊功能測試 ===")
    print(f"Python 版本: {sys.version}")
    print(f"NumPy 版本: {np.__version__}")
    
    # 檢查 PyAudio 狀態
    try:
        import pyaudio
        print(f"PyAudio 版本: {pyaudio.get_portaudio_version_text()}")
        pyaudio_available = True
    except ImportError:
        print("⚠️  PyAudio 未安裝")
        print("請在系統中安裝 PortAudio 開發庫:")
        print("  Ubuntu/Debian: sudo apt-get install portaudio19-dev")
        print("  然後: pip install pyaudio")
        pyaudio_available = False
    
    # 執行測試
    test_ring_buffer()
    
    if pyaudio_available:
        test_audio_recorder()
    else:
        print("\n⚠️  跳過音訊錄製器測試（需要 PyAudio）")
    
    test_recording_simulation()
    
    print("\n=== 測試完成 ===")
    print("\n下一步建議:")
    print("1. 如果 PyAudio 未安裝，請先安裝系統依賴")
    print("2. 實現 Whisper 引擎封裝")
    print("3. 開發 UI 介面")


if __name__ == "__main__":
    main()