# STAR-CVI

**面向空地视频行人重识别的语义轴路由与反事实视角干预**

[English](README.md) · [复现说明](docs/REPRODUCIBILITY.md) · [发布前检查表](docs/PUBLISHING_CHECKLIST.md)

STAR-CVI 将固定空间位置匹配改写为语义坐标系中的对应关系：CLIP
帧级图像块被动态路由到 24 个身份检索语义轴和 3 个观测上下文轴，随后沿语义轴轨迹进行时序建模。R-CVI 仅替换控制路由的潜在视角条件，保持视觉 token、语义原型和网络参数不变，从而减少平台特定的路由捷径。

![STAR-CVI architecture](assets/overview.png)

## 代码结构

本仓库是 OpenGait 的增量扩展，不重复打包完整 OpenGait：

- `opengait/modeling/models/`：STAR-CVI 主模型与 Human-Basis 语义字典
- `opengait/modeling/losses/`：反事实一致性、原型多样性和跨视角语义轴损失
- `opengait/modeling/model_clip/`：适配 192×96 输入的 CLIP-ReID 组件
- `opengait/data/`：序列一致的数据增强与 CLIP RGB 归一化
- `opengait/evaluation/`：AG-VPReID 四种空地检索协议
- `configs/`：主实验配置与 DeepGaitV2 参考配置
- `scripts/`：安装、数据转换与仓库静态检查脚本

## 快速安装

先根据本机 CUDA 环境安装 PyTorch 和 torchvision，再执行：

```bash
python -m pip install -r requirements.txt

git clone https://github.com/ShiqiYu/OpenGait.git
git -C OpenGait checkout 0efafd4779f127fbce34f22aff301bd82e923da5
bash scripts/install_into_opengait.sh ./OpenGait
```

安装脚本只添加 STAR-CVI 文件，不覆盖 OpenGait 核心文件。已有同名文件时会停止；确认需要更新后可显式使用 `--force`。

## 数据准备

从 [AG-VPReID 官方仓库](https://github.com/agvpreid25/AG-VPReID)获取数据并遵守其使用条款，然后转换为 OpenGait 的 PKL 格式：

```bash
python scripts/prepare_ag_vpreid.py \
  --train-root /path/to/AG-VPReID/train \
  --test-root /path/to/AG-VPReID/test \
  --output-root ./data/AG-VPReID_OpenGait_PKL_192_96 \
  --partition-out ./datasets/AG-VPReID/AG_VPReID.json
```

转换完成后，根据实际路径和训练身份数修改
`OpenGait/configs/star_cvi_ag_vpreid.yaml` 中的 `dataset_root`、
`dataset_partition` 与 `SeparateBNNecks.class_num`。官方相机划分为：
`C0`-`C3` 是地面视角，`C4`-`C5` 是无人机视角。

## 训练与测试

在 OpenGait 根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  opengait/main.py --cfgs ./configs/star_cvi_ag_vpreid.yaml --phase train

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  opengait/main.py --cfgs ./configs/star_cvi_ag_vpreid.yaml \
  --phase test --iter 40000
```

首次运行会自动下载 CLIP ViT-B/16 权重；离线环境可将 `ViT-B-16.pt`
放到 `OpenGait/opengait/modeling/model_clip/`。主配置面向单卡运行，多卡测试时需按 OpenGait 的要求同步调整 `evaluator_cfg.sampler.batch_size`。

## 论文中报告的 AG-VPReID 结果

| 协议 | Rank-1 | Rank-5 | mAP |
| --- | ---: | ---: | ---: |
| 空中 → 地面 | 75.8 | 83.8 | 66.7 |
| 地面 → 空中 | 78.9 | 88.4 | 62.8 |
| 地面 ↔ 地面 | 90.1 | 95.4 | 74.4 |
| 空中 ↔ 空中 | 91.7 | 96.2 | 75.7 |

这些数值来自随附论文，本次代码整理只进行了静态与结构检查，未重新训练验证。

## 发布前注意事项

- 数据集、划分文件、模型权重和训练输出均不会提交到 Git。
- DeepGaitV2 参考配置依赖作者本地的 RGB 适配版本，不能直接视为官方 OpenGait 的可运行基线。
- 当前根目录许可证为“保留所有权利”的临时状态；公开仓库前需要作者确定正式许可证，并更新 `CITATION.cff` 中的作者、期刊和 DOI 信息。
