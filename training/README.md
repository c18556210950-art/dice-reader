# 本地骰子标注与训练

本目录为独立流程，不更改已有 `app.py`、服务器、DeepSeek 配置。标注、训练、预测在本机执行；只有首次下载数据与预训练权重需要联网。

更多公开来源、下载状态、点数标注是否可用及本轮改进实验见项目根目录 [数据集清单](../数据集清单.md)。新版分类器仍为实验分支，未替换默认模型。

## 使用

1. 双击项目根目录的 **启动标注台.cmd**，访问 http://127.0.0.1:8767/。
2. 填写拍摄批次（设备/日期/骰子套装），导入你的照片。在照片上拖动，框住**整颗骰子**。选择类型、填写点数，逐个勾选“人工确认点数”。框出所有骰子后保存。看不清则选择“无法判读”；没有骰子的照片也可以保存为空标注，用作负样本。
3. 双击 **训练骰子模型.cmd**：恢复下载公开数据 → 导出全新数据快照 → YOLO11n 检测训练 → ResNet18 分类训练。默认50轮，使用本机第1块显卡。现有导出、标注和训练结果均保留。

这台电脑已具备 CUDA PyTorch、torchvision、Ultralytics。换电脑时先安装与显卡匹配的 PyTorch，再安装 `requirements-training.txt`。项目固定 Ultralytics 8.3.175，不自动升级已有环境。

## 命令行

在项目根目录执行；使用当前虚拟环境或系统里的 Python，并确保安装了 `requirements-training.txt`。

```powershell
python -m training.fetch_dataset
python -m training.annotate
python -m training.prepare --output .runtime/dice-training/exports/my-first-export
python -m training.train --prepared .runtime/dice-training/exports/my-first-export --name my-first-run --epochs 50 --batch 8
# 先快速检查整个训练链路，产生的模型会标记为 smoke，不能用于正常识别
python -m training.workflow --smoke --skip-download
```

训练结束后，检测权重在 `runs/<名称>/detector/weights/best.pt`，分类权重在 `runs/<名称>/classifier/best.pt`。均相对 `.runtime/dice-training/`。

```powershell
python -m training.predict my-photo.jpg --detector .runtime/dice-training/runs/my-first-run/detector/weights/best.pt --classifier .runtime/dice-training/runs/my-first-run/classifier/best.pt --output result.json
python -m training.evaluate --prepared .runtime/dice-training/exports/my-first-export --detector .runtime/dice-training/runs/my-first-run/detector/weights/best.pt --classifier .runtime/dice-training/runs/my-first-run/classifier/best.pt --output evaluation.json
```

`--device cpu` 可改用 CPU。检测器默认960像素、batch8；显存不足时减小 `--batch` 或 `--imgsz`。训练使用预训练权重微调，`--scratch` 才从随机权重开始。不是从头训练通用视觉大模型。

## 数据来源与限制

- [diecamera-frames](https://huggingface.co/datasets/G-G-Games/diecamera-frames)，G-G-Games / @eschatus / @trivenigandhi，按数据卡标示 **CC BY-SA 4.0**。数据卡同时说明许可措辞处于 provisional 状态；传播衍生数据时保留署名、来源、同许可要求及原始说明。下载器将 README 原文存入 `sources/diecamera/README.md`，锁定 revision，保留元数据及下载失败记录。
- 本次锁定 `b9a08b37d01cac6a6297440a60d81b57008e5424`：351帧、1079颗骰子、587个被数据源标记为可信的点数。这些可信度来自数据源，不代表本项目逐张复核过。
- 所有框可用于检测训练；**只有 `values_trusted=true` 导入的点数，或本地人工逐个确认的点数，才进入分类器**。其余数字不得当成真值。导入的 `human_reviewed=false` 表示尚未由本项目复核，与原数据源是否人工确认分开保存。
- 公开数据含 d4/d6/d8/d10/d12/d20；仅一颗百分骰且无可信数字，不能靠它学会百分骰。检测标签 `d100` 映射本项目的 `d00`。另有 [Roboflow DnD Dice Set](https://universe.roboflow.com/proba-dtou1/dnd-dice-set) 可作为未来框标注补充，本流程未下载、未混入。
- 数据少且与用户骰子材质、字体、角度和光照可能不同。先补充你实际骰盘的照片，跨设备/日期/骰子套装采集；每种骰子的每个点数都应有清晰样本，包含多骰、遮挡、反光和空盘。最终效果以未用于训练的真实拍摄批次为准。

## 防止误导性的准确率

- 按设备和日期的**拍摄批次**整体分配，约70%/15%/15%，不对相邻照片、骰子裁剪随机拆分。数据量按批次决定，比例不保证精确。当前默认导出为246/53/51帧；一个重复帧被去重。
- 完全相同的图片如果跨分区重复，全部排除；同分区重复仅保留一张。没有人工确认“整张照片已框完”的自有照片不参与训练。公开来源的框暂按数据源可用处理，可在标注台修正。
- 导出快照保存 `records.snapshot.jsonl`、每张照片分区和各类别覆盖。补充数据后重新导出，不能拿新分区的结果与旧分区直接比较。
- 检测器生成 mAP/召回率，分类器记录验证集和测试集 top1、逐类计数及混淆。`evaluate` 还报告含漏检的点数准确率、整盘完全正确率、接受覆盖率和接受后的准确率；类别一致且IoU≥0.5才算匹配。
- 默认0.8为候选置信度门槛；概率未经校准，阈值不代表80%准确率。**当前所有预测均需复核**：`needs_review=true`、`total=null`，只有全部通过门槛时才产生 `proposed_total` 候选总和；即使全通过仍可能读错或漏检。分类以 `类型_印刷值` 编码，D10的0只在求和时计作10，d00的00计0，不自动组合百分骰与个位骰。
- 不镜像数字；裁剪为带边距的正方形，保持整颗骰子，不随机切掉数字。D4顶读/底读差异靠正确标注学习；D6点阵和数字字面都需各自样本。6/9方向标记不得忽略。

## 本地文件与恢复

全部图片、锁定来源、JSONL标注、导出、权重、日志保存在 `.runtime/dice-training`，不会上传到云端，也不进入 Git。标注每次保存前保留旧记录到 `annotation-history`；JSONL写入为原子替换，下载器和标注服务使用跨进程文件锁，重新下载不会覆盖人工复核记录。备份请复制整个目录。

标注工具仅绑定127.0.0.1，并校验Host和Origin。它不提供公网服务，不包含训练任务远程执行接口。现有网站仍使用原有识别方式，只有完成独立测试后才考虑切换。

YOLO工具及预训练权重的使用按 [Ultralytics 官方许可说明](https://www.ultralytics.com/license)；分类预训练权重来自 [PyTorch 官方](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。

## 本次实际验证（2026-10-28）

18项测试通过。单轮小样本训练与本地预测已跑通，结果保存在 `runs/pipeline-smoke-v2/`。另完成 **20轮初始基线**：YOLO11n、640像素、batch8；ResNet18、224像素、batch8，均从官方预训练权重微调。本机 RTX4060 8GB，无云端训练。

基线权重：`runs/initial-baseline/detector/weights/best.pt` 和 `runs/initial-baseline/classifier/best.pt`。51张独立测试照片共128颗骰子：

| 指标 | 结果 |
| --- | --- |
| 类别正确且IoU≥0.5的检测召回 | 118/128，92.2% |
| 点数正确（包括漏检，检测→分类全流程） | 42/128，32.8% |
| 整盘骰子数量、类型、点数全部正确 | 8/51，15.7% |
| 分类器单独读取真实框裁剪，70类全局top1 | 37/128，28.9% |
| 通过0.8门槛的匹配骰子中点数正确 | 2/11，18.2% |

**这轮基线不可用于自动判定投骰结果。高置信度也出现错误。** 流程已经可用，但数字分类的跨拍摄批次泛化不足，需要实际骰子、字体、D4读法等更丰富且复核过的点数样本。上述比率仅描述该锁定数据快照，不代表用户实物骰盘的准确率。

完整逐图结果：`runs/initial-baseline/end-to-end-test.json`；分类混淆和每类数量：`classifier/test.json`；检测验证集训练记录：`detector/results.csv`；训练配置：`config.json`。测试集只用于最终评估，没有用来选模型或调整门槛。模型选择依据验证集。

后续 YOLO 运行使用项目独立配置目录，关闭在线版本探测、遥测及云端实验平台集成；预训练权重和公开数据仍通过独立下载器下载，不读取既有 HUB 登录配置。
