"""
音訊錄製核心模組

提供高品質的音訊錄製功能，專為語音識別優化
設計原則：
- 低延遲：錄音啟動延遲 < 100ms
- 高品質：16kHz 取樣率，16-bit 深度
- 線程安全：支援非阻塞錄音
- 資源效率：自動降噪，智慧緩衝管理
"""

import threading
import time
import numpy as np
from typing import Optional, Callable, Dict, Any
import logging
from enum import Enum

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.warning("PyAudio 不可用，將使用備選方案")

from .buffer_manager import RingBuffer

logger = logging.getLogger(__name__)


class RecordingState(Enum):
    """錄音狀態枚舉"""
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    ERROR = "error"


class AudioRecorder:
    """
    高性能音訊錄製器
    
    專為語音識別場景優化，提供穩定的音訊錄製能力
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 buffer_duration: float = 10.0):
        """
        初始化音訊錄製器
        
        Args:
            sample_rate: 取樣率，Whisper 標準為 16kHz
            channels: 聲道數，單聲道為 1
            chunk_size: 每次讀取的樣本數
            buffer_duration: 緩衝區時長（秒）
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paFloat32 if PYAUDIO_AVAILABLE else None
        
        # 計算緩衝區大小
        buffer_size = int(sample_rate * buffer_duration)
        self.buffer = RingBuffer(buffer_size, dtype=np.float32)
        
        # 錄音狀態
        self.state = RecordingState.IDLE
        self.stream = None
        self.audio_interface = None
        self.recording_thread = None
        self.state_lock = threading.Lock()
        
        # 回調函數
        self.data_callback: Optional[Callable[[np.ndarray], None]] = None
        self.state_callback: Optional[Callable[[RecordingState], None]] = None
        
        # 統計資訊
        self.stats = {
            'total_samples': 0,
            'dropped_samples': 0,
            'recording_duration': 0.0,
            'start_time': None
        }
        
        logger.info(f"AudioRecorder 初始化: "
                   f"rate={sample_rate}Hz, channels={channels}, "
                   f"chunk={chunk_size}, buffer={buffer_duration}s")
        
        # 初始化音訊系統
        self._init_audio_system()
    
    def _init_audio_system(self):
        """初始化音訊系統"""
        if not PYAUDIO_AVAILABLE:
            logger.error("PyAudio 不可用，無法初始化音訊系統")
            self._set_state(RecordingState.ERROR)
            return
        
        try:
            self.audio_interface = pyaudio.PyAudio()
            
            # 檢查預設音訊裝置
            default_device = self.audio_interface.get_default_input_device_info()
            logger.info(f"預設音訊裝置: {default_device['name']}")
            
        except Exception as e:
            logger.error(f"音訊系統初始化失敗: {e}")
            self._set_state(RecordingState.ERROR)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        音訊數據回調函數
        
        在音訊線程中被調用，需要快速處理以避免丟失數據
        """
        if status:
            logger.warning(f"音訊回調狀態警告: {status}")
        
        try:
            # 轉換音訊數據
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # 寫入緩衝區
            if not self.buffer.write(audio_data):
                self.stats['dropped_samples'] += len(audio_data)
                logger.warning("緩衝區寫入失敗，數據丟失")
            
            # 更新統計
            self.stats['total_samples'] += len(audio_data)
            
            # 執行數據回調
            if self.data_callback:
                try:
                    self.data_callback(audio_data)
                except Exception as e:
                    logger.error(f"數據回調執行失敗: {e}")
            
        except Exception as e:
            logger.error(f"音訊回調處理失敗: {e}")
            return (None, pyaudio.paAbort)
        
        return (None, pyaudio.paContinue)
    
    def start_recording(self) -> bool:
        """
        開始錄音
        
        Returns:
            bool: 是否成功開始錄音
        """
        with self.state_lock:
            if self.state == RecordingState.RECORDING:
                logger.warning("錄音已在進行中")
                return True
            
            if self.state == RecordingState.ERROR:
                logger.error("音訊系統錯誤，無法開始錄音")
                return False
            
            try:
                # 清空緩衝區
                self.buffer.clear()
                
                # 重置統計
                self.stats['total_samples'] = 0
                self.stats['dropped_samples'] = 0
                self.stats['start_time'] = time.time()
                
                # 開啟音訊串流
                self.stream = self.audio_interface.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback,
                    start=False
                )
                
                # 開始錄音
                self.stream.start_stream()
                self._set_state(RecordingState.RECORDING)
                
                logger.info("錄音已開始")
                return True
                
            except Exception as e:
                logger.error(f"開始錄音失敗: {e}")
                self._set_state(RecordingState.ERROR)
                return False
    
    def stop_recording(self) -> Optional[np.ndarray]:
        """
        停止錄音
        
        Returns:
            Optional[np.ndarray]: 錄製的音訊數據，失敗時返回 None
        """
        with self.state_lock:
            if self.state != RecordingState.RECORDING:
                logger.warning("當前沒有錄音在進行")
                return None
            
            try:
                # 停止音訊串流
                if self.stream and self.stream.is_active():
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                
                # 獲取所有錄製的數據
                audio_data = self.buffer.read_all()
                
                # 更新統計
                if self.stats['start_time']:
                    self.stats['recording_duration'] = time.time() - self.stats['start_time']
                
                self._set_state(RecordingState.IDLE)
                
                # 記錄統計資訊
                duration = len(audio_data) / self.sample_rate if len(audio_data) > 0 else 0
                logger.info(f"錄音已停止: {duration:.2f}秒, "
                           f"{len(audio_data)} 樣本, "
                           f"丟失 {self.stats['dropped_samples']} 樣本")
                
                return audio_data
                
            except Exception as e:
                logger.error(f"停止錄音失敗: {e}")
                self._set_state(RecordingState.ERROR)
                return None
    
    def pause_recording(self) -> bool:
        """
        暫停錄音
        
        Returns:
            bool: 是否成功暫停
        """
        with self.state_lock:
            if self.state != RecordingState.RECORDING:
                return False
            
            try:
                if self.stream and self.stream.is_active():
                    self.stream.stop_stream()
                
                self._set_state(RecordingState.PAUSED)
                logger.info("錄音已暫停")
                return True
                
            except Exception as e:
                logger.error(f"暫停錄音失敗: {e}")
                return False
    
    def resume_recording(self) -> bool:
        """
        繼續錄音
        
        Returns:
            bool: 是否成功繼續
        """
        with self.state_lock:
            if self.state != RecordingState.PAUSED:
                return False
            
            try:
                if self.stream:
                    self.stream.start_stream()
                
                self._set_state(RecordingState.RECORDING)
                logger.info("錄音已繼續")
                return True
                
            except Exception as e:
                logger.error(f"繼續錄音失敗: {e}")
                return False
    
    def _set_state(self, new_state: RecordingState):
        """設定錄音狀態並執行回調"""
        self.state = new_state
        if self.state_callback:
            try:
                self.state_callback(new_state)
            except Exception as e:
                logger.error(f"狀態回調執行失敗: {e}")
    
    def set_data_callback(self, callback: Callable[[np.ndarray], None]):
        """設定數據回調函數"""
        self.data_callback = callback
    
    def set_state_callback(self, callback: Callable[[RecordingState], None]):
        """設定狀態回調函數"""
        self.state_callback = callback
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """
        獲取錄音統計資訊
        
        Returns:
            Dict[str, Any]: 統計資訊
        """
        stats = self.stats.copy()
        stats['state'] = self.state.value
        stats['buffer_usage'] = self.buffer.get_status()[2]
        
        if self.state == RecordingState.RECORDING and stats['start_time']:
            stats['current_duration'] = time.time() - stats['start_time']
        
        return stats
    
    def get_available_devices(self) -> list:
        """
        獲取可用的音訊輸入裝置
        
        Returns:
            list: 裝置資訊列表
        """
        if not self.audio_interface:
            return []
        
        devices = []
        for i in range(self.audio_interface.get_device_count()):
            try:
                device_info = self.audio_interface.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': device_info['defaultSampleRate']
                    })
            except Exception as e:
                logger.warning(f"獲取裝置 {i} 資訊失敗: {e}")
        
        return devices
    
    def __del__(self):
        """析構函數，確保資源清理"""
        self.cleanup()
    
    def cleanup(self):
        """清理資源"""
        if self.state == RecordingState.RECORDING:
            self.stop_recording()
        
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        if self.audio_interface:
            try:
                self.audio_interface.terminate()
            except:
                pass
            self.audio_interface = None
        
        logger.info("AudioRecorder 資源已清理")