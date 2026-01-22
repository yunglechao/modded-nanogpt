# Repository Guidelines

## 项目结构与模块组织
- `train_gpt.py`：短 track 主训练入口；`train_gpt_medium.py`：中等模型入口。
- `triton_kernels.py`：Triton 自定义 kernel 与 fused op 实现。
- `data/`：FineWeb 数据脚本与缓存；`records/`：记录日志与说明；`img/`：图表素材。
- 根目录含 `run.sh`、`requirements.txt`、`Dockerfile`。
 - 训练与记录多集中在脚本与 `records/track_*/`，新增内容优先复用既有路径。

## 构建、测试与本地运行
- `pip install -r requirements.txt`：安装依赖。
- `python data/cached_fineweb10B.py 9`：拉取部分数据（N 为下载分片数）。
- 其他数据脚本：`data/cached_finewebedu10B.py`、`data/cached_fineweb100B.py`。
- `./run.sh` 或 `torchrun --standalone --nproc_per_node=8 train_gpt.py`：8 卡训练。
- Medium track 可用 `torchrun --standalone --nproc_per_node=8 train_gpt_medium.py`。
- Docker：`docker build -t modded-nanogpt .`，再用 `docker run ...` 运行。
- 首次 `torch.compile` 可能较慢，建议预留编译时间。

## 本地开发与排错
- GPU 数不足时可临时改 `run.sh` 的 `--nproc_per_node`，并相应减小 batch/序列长度以防 OOM。
- 训练异常先确认数据已下载、`torchrun` 可执行、以及 CUDA 设备可见。

## 编码风格与命名约定
- Python 4 空格缩进；函数/变量 `snake_case`，类名 `CamelCase`。
- 性能关键路径保持紧凑，Triton/内核改动集中在 `triton_kernels.py`。
- 竞赛规则与允许的编译选项以 `README.md` 为准。
- 非必要不引入新依赖，如需新增请更新 `requirements.txt` 并说明原因。

## 测试与验证
- 仓库无统一单测框架；以训练日志与 loss 变化作为主要验证。
- 变更后至少跑一次小规模训练（可减少数据分片数）确认稳定性。
- 重点关注速度、loss、显存占用与分布式通信是否异常。
- 日志输出建议保留原始 stdout 以便回溯对比。

## 记录与数据规则
- 竞赛规则禁止改动数据流水线；如需变更需在 PR 中明确说明影响。
- 短 track 日志位于 `records/track_1_short/`，中等模型在 `records/track_2_medium/`。
- 记录目录常用 `YYYY-MM-DD_Desc/` 命名，日志多为 `*.txt` 或 `main.log`。

## 提交与 PR 指南
- 历史提交多为简短祈使句，如 `Update README.md`、`New Record: ... (#PR)`。
- 若是新纪录：更新 `README.md` 表格，附 `records/track_*` 日志与说明，注明硬件与计时方式。
- PR 描述建议包含：改动动机、核心差异、测量方法、对 loss/速度的影响。
- 大型改动建议附简短 benchmark 或对比图表。

## 配置与注意事项
- 默认多 GPU，确保 `torchrun` 可用；必要时设置 `CUDA_VISIBLE_DEVICES`。
- 不要修改数据流水线；提交前先核对 `README.md` 的规则限制。
- 分布式环境变量常见 `RANK` / `WORLD_SIZE` / `LOCAL_RANK`，调试时可打印确认。
- CUDA/NCCL 版本不一致时优先使用 Docker 复现，避免环境差异干扰计时。
- 关键实验改动尽量可开关，便于回滚与对比。
