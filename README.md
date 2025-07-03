# 語音轉文字助手 Voice2Text

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

一款專為隱私保護設計的 Windows 桌面語音轉文字應用程式，整合 OpenAI Whisper 和本地 LLM 技術。

## ✨ 特色功能

- 🔒 **隱私保護**: 所有處理完全本地化，無需上傳雲端
- ⚡ **高效轉錄**: GPU 加速，5-10倍實時轉錄速度
- 🎯 **智慧整理**: AI 自動修正錯誤、整理格式
- 🎪 **無縫整合**: 系統匣常駐，一鍵使用，自動複製

## 🎯 目標用戶

- **主要用戶**: 軟體開發人員、技術文檔撰寫者
- **次要用戶**: 內容創作者、學生、研究人員

## 🏗️ 技術架構

- **開發語言**: Python 3.10+
- **GUI 框架**: PyQt5
- **語音識別**: OpenAI Whisper
- **LLM 引擎**: Ollama
- **音訊處理**: PyAudio + NumPy

## 📋 系統需求

### 最低需求
- **作業系統**: Windows 10/11 (64-bit)
- **CPU**: Intel i5 / AMD Ryzen 5
- **記憶體**: 8GB RAM
- **儲存空間**: 5GB

### 建議配置
- **GPU**: NVIDIA GTX 1060+ (CUDA 支援)
- **記憶體**: 16GB RAM
- **儲存空間**: 10GB (SSD)

## 🚀 快速開始

### 1. 環境準備

```bash
# 建立虛擬環境
python -m venv venv
venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

### 2. 執行應用程式

```bash
# 開發模式
python src/main.py

# 或使用 PyQt5 執行
python -m src.main
```

## 📁 專案結構

```
voice2txt/
├── src/
│   ├── core/              # 核心音訊處理模組
│   │   ├── audio_recorder.py    # 音訊錄製
│   │   ├── buffer_manager.py    # 環形緩衝區
│   │   └── whisper_engine.py    # Whisper 引擎
│   ├── ui/               # 使用者介面
│   │   ├── system_tray.py      # 系統匣
│   │   ├── main_window.py      # 主視窗
│   │   └── floating_button.py  # 懸浮按鈕
│   └── utils/            # 工具模組
│       ├── config.py          # 配置管理
│       └── logger.py          # 日誌系統
├── resources/            # 資源文件
│   ├── icons/           # 圖示
│   └── sounds/          # 音效
├── tests/               # 測試文件
└── .claude/             # AI 助手記憶系統
```

## 🎯 開發階段

### Phase 1: MVP (進行中)
- [x] 專案架構設計
- [x] 音訊錄製核心
- [x] 環形緩衝區管理
- [ ] Whisper 整合
- [ ] 基礎 UI 實現

### Phase 2: 核心功能
- [ ] GPU 加速優化
- [ ] 系統匣功能
- [ ] 熱鍵支援
- [ ] 剪貼簿整合

### Phase 3: 智慧功能
- [ ] Ollama LLM 整合
- [ ] 文字後處理
- [ ] 批次處理
- [ ] 進階設定

### Phase 4: 優化發布
- [ ] 效能優化
- [ ] 打包程式
- [ ] 使用文檔
- [ ] 測試與除錯

## 🔧 開發指南

### 代碼規範
- 使用繁體中文註解和文檔
- 遵循 PEP 8 編碼規範
- 模組化設計，單一職責原則
- 完整的錯誤處理和日誌記錄

### 效能目標
- 啟動時間: < 3 秒
- 轉錄延遲: < 2 秒
- 記憶體使用: < 2GB
- CPU 使用率: < 10% (待機)

## 📄 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 文件

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

1. Fork 本專案
2. 建立功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交變更 (`git commit -m 'feat: 新增驚人功能'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

## 📞 聯絡方式

- 專案主頁: [GitHub Repository](https://github.com/voice2text/voice2txt)
- 問題回報: [Issues](https://github.com/voice2text/voice2txt/issues)

---

⭐ 如果這個專案對您有幫助，請給我們一個星星！