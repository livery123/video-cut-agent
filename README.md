# 鹅鸭杀录屏分局检测

上传鹅鸭杀录屏，自动识别每一局的开始/结束时间，返回 JSON 结果与调试时间轴。

## 功能

- 粗采样（2s）+ 边界精修（0.5s）两阶段抽帧
- 模板匹配 + ROI OCR + 状态机平滑
- REST API + 简易 Web 前端
- 评估脚本（需自备标注数据集）

## 环境要求

- Python 3.10+（推荐 Anaconda + `environment.yml`）
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)（**外部安装**，勾选 **Chinese Simplified / chi_sim**）
- 可选：OpenCV 依赖的系统编解码库（处理 mp4/mkv 等）

## 协作开发：环境配置

仓库只提交**约定 + 示例**；每人本机路径写在 `.env`（不入库）。

### 1. 创建 Conda 环境

```powershell
cd d:\Code_Space\shibie
conda env create -f environment.yml
conda activate shibie
```

或使用已有 Anaconda（本机示例路径 `D:\Anaconda3`）：

```powershell
D:\Anaconda3\Scripts\conda.exe env create -f environment.yml
conda activate shibie
```

### 2. 安装 Tesseract（外部工具，不进 conda）

下载安装包，安装到任意盘符（如 `E:\Tesseract-OCR`），勾选 **chi_sim**。

### 3. 配置本机路径

```powershell
copy .env.example .env
# 编辑 .env，设置 TESSERACT_CMD 与 TESSDATA_PREFIX
```

`.env.example` 示例：

```env
TESSERACT_CMD=E:\Tesseract-OCR\tesseract.exe
TESSDATA_PREFIX=E:\Tesseract-OCR\tessdata
```

### 4. 检查环境

```powershell
python scripts/check_env.py
```

### 5. 启动服务

```powershell
python main.py
# 或双击 start.local.bat（已配置 D:\Anaconda3\envs\shibie）
```

服务默认监听 `http://0.0.0.0:8000`：

| 地址 | 说明 |
|------|------|
| `/` | Web 前端 |
| `/docs` | Swagger API 文档 |
| `POST /api/video/analyze-rounds` | 上传视频并开始分析 |

## 目录结构

```
video_analyzer/     # 核心逻辑（抽帧、OCR、状态机、API）
static/             # 前端页面
templates/          # 界面截图模板（role_chosen_text.png 已入库）
scripts/            # check_env.py 等工具脚本
eval/dataset/       # 评估用视频与标注（需自行添加，不入库）
uploads/            # 运行时上传目录（自动生成，不入库）
tasks/              # 运行时任务数据（自动生成，不入库）
.env.example        # 路径配置模板（提交）
.env                # 本机路径（不提交）
environment.yml     # Conda 环境定义（提交）
```

## 本地启动脚本

- `start.bat` — 通用启动（使用当前 PATH 中的 python）
- `start.local.bat` — 本机 Anaconda 环境（已 gitignore 内容可自定义；当前指向 `D:\Anaconda3\envs\shibie`）

## 评估

在 `eval/dataset/` 放置配对文件后运行：

```powershell
python eval/run_eval.py
```

标注格式见 `eval/dataset/请上传标注数据.txt` 与 `任务目标.ini`。

## 配置

| 文件 | 用途 |
|------|------|
| `.env` | Tesseract 路径（本机） |
| `video_analyzer/config.py` | 识别阈值、关键词、ROI |
| `environment.yml` | Conda + pip 依赖 |

修改 `config.py` 后重启服务生效。

## 许可证

未指定 — 上传前请自行补充。
