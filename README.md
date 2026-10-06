# VenusAlign for IDOLY PRIDE

借助游戏 ADV 脚本生成针对剧情录屏的对帧 ASS 字幕，输入输出视频应通过 **CFR（恒定帧率）** 编码。

注：本项目主要用于例如一些外出卡中含非实时渲染视频的剧情，仅作为本地化无法生效时的备选。

## 安装与运行

建议 Python 3.10+，在项目根目录运行：

```sh
python -m pip install -r requirements-ocr.txt
python main.py \
  --script /path/to/Hoshimi-Adv/Resource/adv_card_kkr_06_02.txt \
  --video /path/to/adv_card_kkr_06_02.mp4 \
  --output adv/ass/result.ass
```

首次运行需要联网下载模型，OCR 在本机执行。也可填写 `config.ini` 中的脚本、视频及目录配置后运行 `python main.py`。

## 常用设置

- `--player-name 牧野`：设置录屏中的玩家名，默认 `マネージャー`。
- `--script-csv /path/to/script.csv`：可选，校验 CSV 与原始 TXT 是否一致。
- `config.ini` 的 `[Hybrid] roi`：字幕区域，格式为 `[左, 上, 右, 下]`，取值为画面宽高的比例。
- `[Arg] need_comment=True`：输出日文注释和空翻译行；设为 `False` 可直接显示日文字幕。
- 双文件剧情可设置 `[Sub] MV_exists=True`、`sub_file_name`；`mv_skip_seconds` 用于跳过已知 MV 空档，默认 0。

其他识别参数见 `config.ini`。旧版绘制模板模式可通过 `--mode template` 启用，仅需安装 `requirements.txt`。

## 非实时渲染视频字幕

默认识别脚本中的全部非实时渲染视频（`video` 引用），根据前后台词的实际帧号扫描对应空档，支持开头、中间、结尾及连续多段；没有台词锚点时扫描到录屏边界。已对齐的脚本字幕会被排除。

- `--movie-ocr off`：关闭补充识别；`--movie-ocr all`：扫描所有未被脚本字幕覆盖的区间。
- `--movie-range 7200:12000`：用明确的帧范围替代自动范围，结束帧不包含在内，可重复指定。
- `[Movie OCR] roi`：非实时渲染视频字幕区域，默认与普通字幕相同；位置或行数不同时需调整。

补充字幕合并进主 ASS，同时单独保存为 `.movies.ass`，角色栏标记为 `OCR`。这些文字没有脚本原文可校验，报告中统一标记 `needs_review`，需校对错字与漏字。

## 输出

生成 ASS 和同名 `.review.json` 报告。报告包含原文、识别结果、起止帧号、未匹配条目及非实时渲染视频补充字幕；未匹配的脚本台词不写入 ASS。两种流程均无结果时仅保存报告并报错。
