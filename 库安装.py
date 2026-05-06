@echo off
chcp 65001 > nul
echo ========================================
echo SH-GNN 依赖库安装脚本
echo ========================================
echo.

:: 检查 Python 是否可用
python --version > nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

echo [信息] 检测到 Python 环境，开始安装依赖...
echo.

:: 升级 pip
python -m pip install --upgrade pip

:: 安装 PyTorch（CPU 版本，如需 CUDA 请自行修改）
:: 若需要 CUDA 11.8，替换为：pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu

:: 安装 PyTorch Geometric 及其依赖（根据 PyTorch 版本自动匹配）
pip install torch_geometric>=2.3.0

:: 安装 torch_scatter（需与 PyTorch 版本匹配，使用官方预编译索引）
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

:: 安装其他基础库
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0
pip install scipy>=1.7.0
pip install onnx>=1.13.0
pip install onnxruntime>=1.14.0
pip install fastapi>=0.95.0
pip install uvicorn>=0.21.0
pip install pydantic>=2.0.0
pip install tensorboard>=2.11.0
pip install tqdm>=4.64.0

echo.
echo ========================================
echo 所有依赖安装完成！
echo 注意：healpy 为可选依赖（CMB数据集需要），Windows下安装较复杂，未自动安装。
echo ========================================
pause