"""
環形緩衝區管理器

提供線程安全的音訊數據緩衝機制，避免錄音卡頓和記憶體溢出
設計原則：
- 線程安全：支援多線程併發存取
- 自動覆蓋：當緩衝區滿時自動覆蓋舊數據
- 高效能：最小化記憶體分配和複製
"""

import threading
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RingBuffer:
    """
    線程安全的環形緩衝區實現
    
    用於音訊數據的高效緩衝，支援生產者-消費者模式
    """
    
    def __init__(self, max_size: int, dtype=np.float32):
        """
        初始化環形緩衝區
        
        Args:
            max_size: 緩衝區最大容量（樣本數）
            dtype: 數據類型，預設 float32
        """
        self.max_size = max_size
        self.dtype = dtype
        self.buffer = np.zeros(max_size, dtype=dtype)
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        self.lock = threading.RLock()  # 可重入鎖
        
        logger.info(f"RingBuffer 初始化: 容量={max_size}, 類型={dtype}")
    
    def write(self, data: np.ndarray) -> bool:
        """
        寫入數據到緩衝區
        
        Args:
            data: 要寫入的音訊數據
            
        Returns:
            bool: 是否成功寫入
        """
        if len(data) > self.max_size:
            logger.warning(f"數據長度 {len(data)} 超過緩衝區容量 {self.max_size}")
            # 只保留最後的數據
            data = data[-self.max_size:]
        
        with self.lock:
            write_len = len(data)
            
            # 計算寫入位置
            end_pos = (self.write_pos + write_len) % self.max_size
            
            if self.write_pos + write_len <= self.max_size:
                # 一次性寫入
                self.buffer[self.write_pos:self.write_pos + write_len] = data
            else:
                # 分段寫入（環繞）
                first_part = self.max_size - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:end_pos] = data[first_part:]
            
            self.write_pos = end_pos
            
            # 更新大小，處理溢出情況
            if self.size + write_len > self.max_size:
                # 緩衝區溢出，調整讀取位置
                overflow = (self.size + write_len) - self.max_size
                self.read_pos = (self.read_pos + overflow) % self.max_size
                self.size = self.max_size
                logger.debug(f"緩衝區溢出，丟棄 {overflow} 個樣本")
            else:
                self.size += write_len
            
            return True
    
    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """
        從緩衝區讀取數據
        
        Args:
            num_samples: 要讀取的樣本數
            
        Returns:
            Optional[np.ndarray]: 讀取的數據，如果數據不足則返回 None
        """
        with self.lock:
            if num_samples > self.size:
                return None
            
            result = np.zeros(num_samples, dtype=self.dtype)
            
            if self.read_pos + num_samples <= self.max_size:
                # 一次性讀取
                result = self.buffer[self.read_pos:self.read_pos + num_samples].copy()
            else:
                # 分段讀取（環繞）
                first_part = self.max_size - self.read_pos
                result[:first_part] = self.buffer[self.read_pos:]
                result[first_part:] = self.buffer[:num_samples - first_part]
            
            self.read_pos = (self.read_pos + num_samples) % self.max_size
            self.size -= num_samples
            
            return result
    
    def read_all(self) -> np.ndarray:
        """
        讀取緩衝區中所有數據
        
        Returns:
            np.ndarray: 所有可用數據
        """
        with self.lock:
            if self.size == 0:
                return np.array([], dtype=self.dtype)
            
            return self.read(self.size)
    
    def peek(self, num_samples: int) -> Optional[np.ndarray]:
        """
        預覽數據但不從緩衝區移除
        
        Args:
            num_samples: 要預覽的樣本數
            
        Returns:
            Optional[np.ndarray]: 預覽的數據
        """
        with self.lock:
            if num_samples > self.size:
                return None
            
            result = np.zeros(num_samples, dtype=self.dtype)
            
            if self.read_pos + num_samples <= self.max_size:
                result = self.buffer[self.read_pos:self.read_pos + num_samples].copy()
            else:
                first_part = self.max_size - self.read_pos
                result[:first_part] = self.buffer[self.read_pos:]
                result[first_part:] = self.buffer[:num_samples - first_part]
            
            return result
    
    def clear(self):
        """清空緩衝區"""
        with self.lock:
            self.write_pos = 0
            self.read_pos = 0
            self.size = 0
            logger.debug("緩衝區已清空")
    
    def get_status(self) -> Tuple[int, int, float]:
        """
        獲取緩衝區狀態
        
        Returns:
            Tuple[int, int, float]: (當前大小, 最大容量, 使用率)
        """
        with self.lock:
            usage = self.size / self.max_size if self.max_size > 0 else 0
            return self.size, self.max_size, usage
    
    @property
    def available_space(self) -> int:
        """可用空間（樣本數）"""
        return self.max_size - self.size
    
    @property
    def is_full(self) -> bool:
        """緩衝區是否已滿"""
        return self.size >= self.max_size
    
    @property
    def is_empty(self) -> bool:
        """緩衝區是否為空"""
        return self.size == 0