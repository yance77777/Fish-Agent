# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['launcher_app.py'],
    pathex=[],
    binaries=[],
    datas=[('..\\.installer_innounp\\{app}', 'core')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'numpy', 'scipy', 'pandas', 'matplotlib', 'cv2', 'PyQt5', 'PySide6', 'PyQt6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Fish-Agent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\fish_agent_icon.ico'],
    contents_directory='_internal',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Fish-Agent',
)
