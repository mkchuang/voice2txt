#!/usr/bin/env python3
"""
Whisper 語音識別引擎封裝

負責管理 OpenAI Whisper 模型的載入、快取和轉錄功能。
支援 GPU/CPU 自動切換，針對中文識別進行優化。

作者：Voice2Text 專案
版本：1.0.0
"""

import os
import sys
import time
import threading
import logging
from enum import Enum
from typing import Optional, Dict, Any, Callable, Union
from pathlib import Path
import numpy as np

# 延遲載入 Whisper 相關模組
whisper = None
torch = None

logger = logging.getLogger(__name__)


class WhisperModel(Enum):
    """Whisper 模型大小枚舉"""
    TINY = "tiny"           # 39MB, 快速但準確率較低
    BASE = "base"           # 74MB, 平衡速度與準確率
    SMALL = "small"         # 244MB, 較好準確率
    MEDIUM = "medium"       # 769MB, 高準確率
    LARGE = "large"         # 1550MB, 最高準確率
    LARGE_V2 = "large-v2"   # 1550MB, 改進版本
    LARGE_V3 = "large-v3"   # 1550MB, 最新版本


class EngineState(Enum):
    """引擎狀態"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing" 
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"


class WhisperEngine:
    """
    Whisper 語音識別引擎
    
    特性：
    - 模型懶載入和快取
    - GPU/CPU 自動偵測和切換
    - 中文識別優化
    - 執行緒安全
    - 記憶體管理
    """
    
    def __init__(
        self,
        model_name: str = "small",
        device: Optional[str] = None,
        language: str = "zh",
        cache_dir: Optional[str] = None,
        enable_vad: bool = True
    ):
        """
        初始化 Whisper 引擎
        
        Args:
            model_name: 模型名稱 (tiny, base, small, medium, large, large-v2, large-v3)
            device: 指定設備 ("cuda", "cpu", None=自動偵測)
            language: 語言代碼 ("zh"=中文, "en"=英文, None=自動偵測)
            cache_dir: 模型快取目錄
            enable_vad: 是否啟用語音活動偵測
        """
        self.model_name = model_name
        self.language = language
        self.enable_vad = enable_vad
        
        # 狀態管理
        self.state = EngineState.UNINITIALIZED
        self._model = None
        self._device = None
        self._lock = threading.Lock()
        
        # 快取設定
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "voice2text"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 效能監控
        self.stats = {
            "model_load_time": 0.0,
            "total_transcriptions": 0,
            "average_processing_time": 0.0,
            "last_processing_time": 0.0
        }
        
        # 中文識別優化參數
        self.transcribe_options = {
            "language": self.language,
            "task": "transcribe",
            "temperature": 0.0,        # 降低隨機性
            "beam_size": 5,           # 增加 beam search
            "best_of": 5,             # 多次嘗試取最佳
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
            "condition_on_previous_text": True
        }
        
        # 自動偵測設備
        if device is None:
            self._device = self._detect_device()
        else:
            self._device = device
            
        logger.info(f"WhisperEngine 初始化: model={model_name}, device={self._device}, lang={language}")
    
    def _detect_device(self) -> str:
        """自動偵測最佳計算設備"""
        try:
            import torch
            if torch.cuda.is_available():
                # 檢查 CUDA 記憶體
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory >= 2.0:  # 至少 2GB GPU 記憶體
                    logger.info(f"偵測到 CUDA GPU: {gpu_memory:.1f}GB 記憶體")
                    return "cuda"
                else:
                    logger.warning(f"GPU 記憶體不足: {gpu_memory:.1f}GB < 2GB")
            else:
                logger.info("CUDA 不可用")
        except ImportError:
            logger.warning("PyTorch 未安裝，無法使用 GPU")
        except Exception as e:
            logger.warning(f"GPU 偵測失敗: {e}")
        
        logger.info("使用 CPU 進行推理")
        return "cpu"
    
    def _lazy_import(self):
        """延遲載入 Whisper 模組"""
        global whisper, torch
        
        if whisper is None:
            try:
                import whisper as _whisper
                import torch as _torch
                whisper = _whisper
                torch = _torch
                logger.info("Whisper 模組載入成功")
            except ImportError as e:
                logger.error(f"無法載入 Whisper: {e}")
                raise ImportError(f"請安裝 openai-whisper: pip install openai-whisper") from e
    
    def initialize(self, progress_callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """
        初始化模型
        
        Args:
            progress_callback: 進度回調函數 (message, progress_0_to_1)
            
        Returns:
            bool: 初始化是否成功
        """
        with self._lock:
            if self.state == EngineState.READY:
                return True
                
            if self.state == EngineState.INITIALIZING:
                return False
                
            self.state = EngineState.INITIALIZING
            
        try:
            # 1. 延遲載入模組
            if progress_callback:
                progress_callback("載入 Whisper 模組...", 0.1)
            self._lazy_import()
            
            # 2. 載入模型
            if progress_callback:
                progress_callback(f"載入 {self.model_name} 模型...", 0.3)
                
            start_time = time.time()
            
            # 使用快取載入模型
            self._model = whisper.load_model(
                self.model_name,
                device=self._device,
                download_root=str(self.cache_dir)
            )
            
            load_time = time.time() - start_time
            self.stats["model_load_time"] = load_time
            
            if progress_callback:
                progress_callback("模型載入完成", 1.0)
                
            logger.info(f"模型載入成功: {self.model_name} ({load_time:.2f}s)")
            
            # 3. 驗證模型
            if self._model is None:
                raise RuntimeError("模型載入失敗")
                
            self.state = EngineState.READY
            return True
            
        except Exception as e:
            logger.error(f"模型初始化失敗: {e}")
            self.state = EngineState.ERROR
            return False
    
    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        轉錄音訊為文字
        
        Args:
            audio_data: 音訊數據 (numpy array)
            sample_rate: 取樣率 (預設 16kHz)
            **kwargs: 額外的轉錄參數
            
        Returns:
            Dict: 轉錄結果 {
                "text": str,           # 轉錄文字
                "segments": List[Dict], # 時間戳片段
                "language": str,       # 偵測到的語言
                "processing_time": float, # 處理時間
                "confidence": float    # 信心度
            }
        """
        if self.state != EngineState.READY:
            if not self.initialize():
                raise RuntimeError("Whisper 引擎未就緒")
        
        with self._lock:
            self.state = EngineState.PROCESSING
            
        try:
            start_time = time.time()
            
            # 預處理音訊數據
            audio_processed = self._preprocess_audio(audio_data, sample_rate)
            
            # 合併轉錄選項
            options = {**self.transcribe_options, **kwargs}
            
            # 執行轉錄
            result = self._model.transcribe(audio_processed, **options)
            
            # 後處理結果
            processed_result = self._postprocess_result(result)
            
            # 更新統計
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            processed_result["processing_time"] = processing_time
            
            logger.info(f"轉錄完成: {len(processed_result['text'])} 字元, {processing_time:.2f}s")
            
            return processed_result
            
        except Exception as e:
            logger.error(f"轉錄失敗: {e}")
            raise
        finally:
            self.state = EngineState.READY
    
    def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """預處理音訊數據"""
        # 確保是 float32 格式
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # 正規化到 [-1, 1] 範圍
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # 重採樣到 16kHz (Whisper 要求)
        if sample_rate != 16000:
            # 簡單的重採樣 (生產環境建議使用 librosa)
            target_length = len(audio_data) * 16000 // sample_rate
            audio_data = np.interp(
                np.linspace(0, len(audio_data), target_length),
                np.arange(len(audio_data)),
                audio_data
            )
        
        return audio_data
    
    def _postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """後處理轉錄結果"""
        text = result.get("text", "").strip()
        
        # 中文文本後處理
        if self.language == "zh":
            text = self._process_chinese_text(text)
        
        # 計算平均信心度
        segments = result.get("segments", [])
        confidence = 0.0
        if segments:
            confidence_sum = sum(
                seg.get("avg_logprob", 0.0) for seg in segments
            )
            confidence = max(0.0, min(1.0, (confidence_sum / len(segments) + 1.0) / 2.0))
        
        return {
            "text": text,
            "segments": segments,
            "language": result.get("language", self.language),
            "confidence": confidence
        }
    
    def _process_chinese_text(self, text: str) -> str:
        """中文文本後處理"""
        # 移除多餘空格
        text = " ".join(text.split())
        
        # 中文標點符號正規化
        punctuation_map = {
            "，": "，",
            "。": "。",
            "？": "？",
            "！": "！",
            "：": "：",
            "；": "；"
        }
        
        for old, new in punctuation_map.items():
            text = text.replace(old, new)
        
        return text
    
    def _update_stats(self, processing_time: float):
        """更新效能統計"""
        self.stats["total_transcriptions"] += 1
        self.stats["last_processing_time"] = processing_time
        
        # 計算平均處理時間 (滑動平均)
        alpha = 0.1  # 平滑因子
        if self.stats["average_processing_time"] == 0:
            self.stats["average_processing_time"] = processing_time
        else:
            self.stats["average_processing_time"] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats["average_processing_time"]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取引擎統計資訊"""
        return {
            **self.stats,
            "state": self.state.value,
            "model_name": self.model_name,
            "device": self._device,
            "language": self.language
        }
    
    def is_ready(self) -> bool:
        """檢查引擎是否就緒"""
        return self.state == EngineState.READY
    
    def cleanup(self):
        """清理資源"""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                
            # 清理 GPU 記憶體
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.state = EngineState.UNINITIALIZED
            logger.info("WhisperEngine 資源已清理")
    
    def __del__(self):
        """析構函數"""
        self.cleanup()


# 便利函數
def create_engine(
    model_size: str = "small",
    language: str = "zh",
    device: Optional[str] = None
) -> WhisperEngine:
    """建立預設 Whisper 引擎"""
    return WhisperEngine(
        model_name=model_size,
        language=language,
        device=device
    )


def get_available_models() -> list:
    """獲取可用的模型列表"""
    return [model.value for model in WhisperModel]


if __name__ == "__main__":
    # 測試代碼
    import argparse
    
    parser = argparse.ArgumentParser(description="Whisper 引擎測試")
    parser.add_argument("--model", default="tiny", help="模型大小")
    parser.add_argument("--device", help="計算設備")
    args = parser.parse_args()
    
    # 設定日誌
    logging.basicConfig(level=logging.INFO)
    
    # 建立引擎
    engine = WhisperEngine(
        model_name=args.model,
        device=args.device
    )
    
    print(f"建立引擎: {engine.get_stats()}")
    
    # 初始化
    if engine.initialize():
        print("引擎初始化成功")
        print(f"統計資訊: {engine.get_stats()}")
    else:
        print("引擎初始化失敗")