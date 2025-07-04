#!/usr/bin/env python3
"""
Windows 環境音訊測試腳本

使用方式：
1. 在 Windows PowerShell 或 CMD 中執行
2. 確保已啟動虛擬環境：venv\Scripts\activate
3. 執行：python test_audio_windows.py
"""

import sys
import time
import numpy as np

def test_pyaudio_windows():
    """測試 Windows 下的 PyAudio"""
    print("=== Windows PyAudio 測試 ===\n")
    
    try:
        import pyaudio
        print(f"✓ PyAudio 載入成功")
        print(f"  版本: {pyaudio.get_portaudio_version_text()}")
        
        # 初始化 PyAudio
        pa = pyaudio.PyAudio()
        print(f"\n音訊裝置總數: {pa.get_device_count()}")
        
        # 列出所有音訊裝置
        print("\n可用音訊裝置:")
        default_input = None
        input_devices = []
        
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            
            # 輸入裝置
            if info['maxInputChannels'] > 0:
                is_default = "(預設)" if i == pa.get_default_input_device_info()['index'] else ""
                print(f"  輸入 [{i}]: {info['name']} - {info['maxInputChannels']}ch {is_default}")
                input_devices.append(i)
                if is_default:
                    default_input = i
            
            # 輸出裝置
            if info['maxOutputChannels'] > 0:
                is_default = "(預設)" if i == pa.get_default_output_device_info()['index'] else ""
                print(f"  輸出 [{i}]: {info['name']} - {info['maxOutputChannels']}ch {is_default}")
        
        pa.terminate()
        
        return True, input_devices, default_input
        
    except Exception as e:
        print(f"✗ PyAudio 測試失敗: {e}")
        return False, [], None


def test_simple_recording(device_index=None, duration=3):
    """簡單錄音測試"""
    print(f"\n=== 簡單錄音測試 ({duration}秒) ===")
    
    try:
        import pyaudio
        
        # 參數設定
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        # 初始化
        pa = pyaudio.PyAudio()
        
        # 開啟音訊串流
        print(f"開始錄音... (裝置: {device_index if device_index else '預設'})")
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        
        # 錄音
        for i in range(0, int(RATE / CHUNK * duration)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # 顯示音量級別
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()
                bar = "█" * int(volume / 1000)
                print(f"\r音量: {bar:<50}", end="")
                
            except Exception as e:
                print(f"\n讀取錯誤: {e}")
        
        print("\n錄音完成！")
        
        # 關閉
        stream.stop_stream()
        stream.close()
        pa.terminate()
        
        # 分析錄音
        all_data = b''.join(frames)
        audio_array = np.frombuffer(all_data, dtype=np.int16)
        
        print(f"\n錄音統計:")
        print(f"  總樣本數: {len(audio_array)}")
        print(f"  持續時間: {len(audio_array)/RATE:.2f} 秒")
        print(f"  最大振幅: {np.abs(audio_array).max()}")
        print(f"  平均振幅: {np.abs(audio_array).mean():.0f}")
        
        # 儲存錄音（可選）
        save = input("\n是否儲存錄音到 test_recording.wav? (y/n): ")
        if save.lower() == 'y':
            import wave
            wf = wave.open('test_recording.wav', 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(all_data)
            wf.close()
            print("✓ 已儲存到 test_recording.wav")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 錄音測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_recorder_class():
    """測試 AudioRecorder 類別"""
    print("\n=== 測試 AudioRecorder 類別 ===")
    
    try:
        # 加入 src 到路徑
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from src.core.audio_recorder import AudioRecorder, RecordingState
        
        # 建立錄製器
        recorder = AudioRecorder(
            sample_rate=16000,
            channels=1,
            chunk_size=1024,
            buffer_duration=5.0
        )
        
        print(f"錄製器狀態: {recorder.state.value}")
        
        # 列出可用裝置
        devices = recorder.get_available_devices()
        if devices:
            print(f"\n透過 AudioRecorder 找到 {len(devices)} 個輸入裝置")
            for dev in devices[:3]:  # 只顯示前3個
                print(f"  [{dev['index']}] {dev['name']}")
        
        # 開始錄音測試
        if recorder.state != RecordingState.ERROR:
            print("\n開始 3 秒錄音測試...")
            recorder.start_recording()
            
            # 監控錄音狀態
            for i in range(30):  # 3秒
                time.sleep(0.1)
                stats = recorder.get_recording_stats()
                if stats['samples_recorded'] > 0:
                    print(f"\r已錄製: {stats['samples_recorded']} 樣本, "
                          f"緩衝區: {stats['buffer_usage']:.0%}", end="")
            
            # 停止錄音
            print("\n停止錄音...")
            audio_data = recorder.stop_recording()
            
            if audio_data is not None and len(audio_data) > 0:
                print(f"✓ 成功錄製 {len(audio_data)} 樣本 ({len(audio_data)/16000:.2f} 秒)")
            else:
                print("✗ 未能取得錄音數據")
        
        # 清理
        recorder.cleanup()
        print("✓ AudioRecorder 測試完成")
        
    except Exception as e:
        print(f"✗ AudioRecorder 測試失敗: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主測試程式"""
    print("=== Voice2Text Windows 音訊測試 ===")
    print(f"Python: {sys.version}")
    print()
    
    # 1. 測試 PyAudio
    success, input_devices, default_input = test_pyaudio_windows()
    
    if not success:
        print("\n⚠️  PyAudio 無法正常運作！")
        print("\n可能的解決方案:")
        print("1. 確保在 Windows 環境執行（不是 WSL）")
        print("2. 安裝 Visual C++ Redistributable")
        print("3. 嘗試重新安裝 PyAudio:")
        print("   pip uninstall pyaudio")
        print("   pip install pyaudio")
        return
    
    # 2. 簡單錄音測試
    if input_devices:
        print("\n準備進行錄音測試...")
        input("請確保麥克風已連接，按 Enter 繼續...")
        
        # 使用預設裝置測試
        if test_simple_recording(default_input, duration=3):
            print("\n✓ 基本錄音功能正常")
        
        # 3. 測試 AudioRecorder 類別
        test_audio_recorder_class()
    else:
        print("\n⚠️  未找到任何輸入裝置！")
        print("請檢查麥克風是否正確連接")
    
    print("\n=== 測試完成 ===")
    print("\n後續步驟:")
    print("1. 如果測試成功，可以繼續開發 Whisper 整合")
    print("2. 如果有問題，請檢查錯誤訊息並修正")


if __name__ == "__main__":
    main()