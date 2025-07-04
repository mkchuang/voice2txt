#!/usr/bin/env python3
"""測試虛擬環境和依賴套件是否正常"""

import sys
print(f"Python 版本: {sys.version}")
print(f"Python 路徑: {sys.executable}")
print("\n檢查已安裝套件:")

# 測試各個套件
packages = [
    ("numpy", "NumPy"),
    ("pyaudio", "PyAudio"),
    ("PyQt5", "PyQt5"),
    ("yaml", "PyYAML"),
    ("keyboard", "Keyboard"),
    ("pyperclip", "Pyperclip"),
    ("requests", "Requests"),
    ("pytest", "Pytest")
]

for module_name, display_name in packages:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "版本未知")
        print(f"✓ {display_name}: {version}")
    except ImportError as e:
        print(f"✗ {display_name}: 未安裝或載入失敗 - {e}")

# 測試 PyAudio
print("\n測試 PyAudio:")
try:
    import pyaudio
    pa = pyaudio.PyAudio()
    print(f"✓ PyAudio 初始化成功")
    print(f"  音訊裝置數量: {pa.get_device_count()}")
    
    # 列出音訊裝置
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  輸入裝置 [{i}]: {info['name']}")
    
    pa.terminate()
except Exception as e:
    print(f"✗ PyAudio 測試失敗: {e}")

# 測試 PyQt5
print("\n測試 PyQt5:")
try:
    from PyQt5.QtCore import QT_VERSION_STR
    from PyQt5.Qt import PYQT_VERSION_STR
    print(f"✓ PyQt5 載入成功")
    print(f"  Qt 版本: {QT_VERSION_STR}")
    print(f"  PyQt 版本: {PYQT_VERSION_STR}")
except Exception as e:
    print(f"✗ PyQt5 測試失敗: {e}")