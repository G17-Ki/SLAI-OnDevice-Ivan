# LLM 模型权重登记簿（Model Registry）

> 唯一事实来源：哪些权重转过 GGUF、放在哪、手机上跑的是哪个、怎么换。
> **每次换权重/收新权重，必须更新本文件**（§1 当前部署 + §2 库存表加行 + §6 行为记录追加）。
> 识别模型以 **sha256 为准**，不要靠文件名（所有部署版同名 `qwen3_asr_0p6b_text-q4_k_m.gguf`）。

## 1. 当前部署

| 项 | 值 |
|---|---|
| 上线日期 | 2026-10-16 14:27（当天替换 all748；all748 曾替换 10-15 的 90h） |
| 源 checkpoint | `qwen3_asr_0p6b_edge_multiturn_64k_final_20261016` |
| 部署版 sha256 | `197774bd21689fa8b24d7050a819e8b6d4fa11abc6e3895afb285601200c9ba9` |
| 大小 | 396,700,032 bytes（q4_k_m / Q4_K-Medium，596M 参数） |
| 手机路径 | `/data/data/ai.liquid.lfmdemo.dev.runtime/files/text/models/qwen3_asr_0p6b_text-q4_k_m.gguf` |
| 服务 | 端口 6020，alias `Qwen3-ASR-0.6B-Text-Q4_K_M`，ctx 8192（训练上限 40960） |
| 溯源 | `candidate_manifest.json`（同目录，`previous_sha256` 保留 all748 值）；2026-10-16 手机侧 sha 实测一致 |
| 模型身份显示 | dev app `files/model-info.json` sidecar（name/checkpoint/alias/quant/sha256/deployed_at），UI header 徽章显示 `LLM: <name> · <sha8> · <MM-DD>` |
| 验证 | 磁盘 sha 实测 ✅；行为指纹（手机配置 phone-cfg 下 T2）与宿主机同配置直测逐字一致 ✅（见 §6 指纹表） |

换模型时：更新本表全部字段 + §2 状态列 + 写新 `model-info.json` + 重指 §3 符号链接。

## 2. 模型库存

存储根目录：`~/models/lfm-gguf/<checkpoint>/`，每模型一个文件夹：
`qwen3_asr_0p6b_text-bf16.gguf`（母版 1.2GB，重量化用）+ `qwen3_asr_0p6b_text-q4_k_m.gguf`（部署版 379MB）。

| # | 源 checkpoint（lineage / 步数） | 转换日期 | q4 sha256 | bf16 sha256 | 状态 |
|---|---|---|---|---|---|
| 1 | `qwen3_asr_0p6b_spoken_multiturn_90h_20261014`（spoken multiturn 90h，10-14 导出） | 2026-10-15 | `f9c2f25634a7a6424254c955bf80a49ba425ceeb012f0eadfd14b8d16b6d13ce` | `856586e41c4e5c1bcb56418a82d86efe65197236235c826eb741daf394f187d4` | 已撤下（2026-10-16），回退源 |
| 2 | `qwen3_asr_0p6b_edge_multiturn_64k_step648_20261016`（edge 多轮数据，step 648） | 2026-10-16 | `b68acc49618b4814a22559ea55d8482e209d5b99825c2e8fbdf49b67e48e25de` | `69164e2a0452f92d12f731efb9c1f6d27faea65e13e23880d2d6e41bca9a26fb` | 已转换+直测通过，备用 |
| 3 | `qwen3_asr_0p6b_edge_all_64k_step748_20261016`（edge 全量数据，step 748） | 2026-10-16 | `fc4059b693255a6a3b92891c5520299eda46d9070c8c834b6f04950263c56e43` | `22089c5e6b02e1afd7eb05fc843dd74dfaf22102b59d9e4a35478336d237ae6a` | 已撤下（2026-10-16 被 #5 替换），回退源 |
| 4 | `qwen3_asr_0p6b_edge_glasses_all_64k_step352_20261016`（glasses 专用数据，step 352，`prompt_version=glasses_prompt_v1`） | 2026-10-16 | `e0b25991019d3c7b89d034f91f4c4247e337d6326f6f37acc7fad0268df86e63` | `a0e9a41495b18d9ae8eb1b648dcc070940cf0d6ff87d71e48768d6fd0f85263e` | 已转换，**直测记忆召回不稳，不建议部署** |
| 5 | `qwen3_asr_0p6b_edge_multiturn_64k_final_20261016`（edge 多轮数据 final 版，源 checkpoint 全 BF16） | 2026-10-16 | `197774bd21689fa8b24d7050a819e8b6d4fa11abc6e3895afb285601200c9ba9` | `460349f1c48d02fbf80a216427e5b8948fab28625861b1404618a9b7251e3afb` | **部署中**（2026-10-16） |

架构五模型完全一致：28 层 / hidden 1024 / intermediate 3072 / 16 头 / 8 KV / head_dim 128 / vocab 151936 / rope_theta 1e6，q4 均为 396,700,032 bytes（可交叉校验转换产物完整性）。注意：checkpoint 的 config.json 里 `num_hidden_layers` 会出现 18（音频塔子配置），文本层数以 310 张量/28 层断言为准。

不可直接部署的其它产物（不占库存）：
- `~/.cache/lmf-demo/llm-swap/out/qwen3_corpus_text-bf16.gguf`：90h corpus 变体，仅 bf16 未量化，从未部署。
- `~/phone-backup/2026-10-15/data/files/models/`：Qwen3 之前的 LFM2.5-Audio-1.5B 全套（2026-09-07 整包备份），历史回退用。

## 3. 文件布局与符号链接

```
~/models/lfm-gguf/                      ← 正式存储（cache 之外，不会被清）
└── <checkpoint>/
    ├── qwen3_asr_0p6b_text-bf16.gguf   母版
    └── qwen3_asr_0p6b_text-q4_k_m.gguf 部署版

~/.cache/lmf-demo/llm-swap/out/         ← 符号链接（指向"当前部署"模型，换权重时同步重指）
├── qwen3_asr_0p6b_text-q4_k_m.gguf -> ~/models/lfm-gguf/<当前部署 checkpoint>/...
├── qwen3_asr_0p6b_text-bf16.gguf  -> 同上
└── qwen3_text-bf16.gguf           -> 同上（旧名兼容，文档引用）

~/.cache/lmf-demo/android-text-companion/models/
└── qwen3_asr_0p6b_text-q4_k_m.gguf -> 同上（APK 打包输入，install_app_owned_runtime.sh 按 manifest sha 校验；manifest 已随 2026-10-16 换权重更新，旧 sha 存于 previous_sha256）

~/.cache/lmf-demo/llm-swap/out/<tag>_text/   ← 提取中间目录（可再生，留在 cache 无所谓）
```

## 4. 部署/回滚操作手册（手机端零改动）

前提：`adb -H 172.18.0.1 -P 5038`（下称 `AWB`），手机序列号 `S=912615210575`。

```bash
GGUF=~/models/lfm-gguf/<checkpoint>/qwen3_asr_0p6b_text-q4_k_m.gguf
AWB -s $S push "$GGUF" /data/local/tmp/lfm-model.gguf
AWB -s $S shell run-as ai.liquid.lfmdemo.dev.runtime \
  cp /data/local/tmp/lfm-model.gguf files/text/models/qwen3_asr_0p6b_text-q4_k_m.gguf
AWB -s $S shell run-as ai.liquid.lfmdemo.dev.runtime \
  sha256sum files/text/models/qwen3_asr_0p6b_text-q4_k_m.gguf   # 必须与 §2 一致
AWB -s $S shell am broadcast -p ai.liquid.lfmdemo.dev.runtime \
  -a ai.liquid.lfmdemo.dev.runtime.RESTART_TEXT_MODEL            # runtime 自动重生 LLM
# 就绪确认（adb forward tcp:16022 → 手机 tcp:6020 需在位）
curl -s http://172.18.0.1:16022/v1/models
```

- 广播必须 `-p`（package），**不能** `-n` 显式组件（动态 receiver 收不到）。
- 回滚 = 用旧模型 gguf 重复上面 4 步（§1 的 sha 可核对）。
- runtime 进程存活才生效；若 AssistServer 杀过 app，先拉起 dev app。
- **model-info.json sidecar**（2026-10-16 起）：部署后写模型身份到 dev app 的 files 目录，UI header 徽章显示（`NativeAudio.getLlmModelInfo()`）：

  ```bash
  cat > /tmp/model-info.json <<'EOF'
  {
    "name": "<checkpoint 短名>",
    "checkpoint": "<checkpoint 全名>",
    "alias": "Qwen3-ASR-0.6B-Text-Q4_K_M",
    "quant": "q4_k_m",
    "sha256": "<§2 的 q4 sha>",
    "deployed_at": "<ISO8601>"
  }
  EOF
  AWB -s $S push /tmp/model-info.json /data/local/tmp/model-info.json
  AWB -s $S shell run-as ai.liquid.lfmdemo.dev cp /data/local/tmp/model-info.json files/model-info.json
  ```

- 坑（2026-10-16 实遇）：runtime 内 LLM 重启若撞上旧进程未释放 6020，新 llama-server 会 `couldn't bind HTTP server socket` 连败，runtime 卡死后 ASR(6010)/LLM(6020) 均不再 accept（SYN-SENT 堆积）。恢复 = 重启 runtime app（用户手动点 "LFM Offline Runtime DEV"；service 带 signature 权限 adb 拉不起）；bridge 会话自动重连。

## 5. 新增权重转换流程（宿主 Linux）

工作目录 `~/.cache/lmf-demo/llm-swap/`（venv + b10083 工具链在此）。
新 checkpoint 放 `~/models/<name>/`（含 `model.safetensors` + `config.json` + tokenizer）。

```bash
cd ~/.cache/lmf-demo/llm-swap
VENV=.venv/bin/python
TCHLD=$PWD/.venv/lib/python3.12/site-packages/torch/lib
TAG=edge_mt648   # 自定义短标签

# 1) 提取文本解码器（310 张量 thinker.model.*，自动 F32→BF16 降精）
#    注意：v2 脚本；v1 只接受全 BF16 输入
LD_LIBRARY_PATH=$TCHLD $VENV \
  <repo>/scripts/qwen3_asr_llm_swap/extract_text_decoder_v2.py \
  --model-dir ~/models/<name> --out-dir out/${TAG}_text

# 2) HF → GGUF（bf16 母版）
LD_LIBRARY_PATH=$TCHLD $VENV llama.cpp-b10083/convert_hf_to_gguf.py \
  out/${TAG}_text --outtype bf16 --outfile out/${TAG}_text-bf16.gguf

# 3) 量化部署版
LD_LIBRARY_PATH=$TCHLD ./llama-b10083/llama-quantize \
  out/${TAG}_text-bf16.gguf out/${TAG}_text-q4_k_m.gguf q4_k_m

# 4) 宿主机直测（起两个 llama-server 端口 18641/18642，跑 /tmp/opencode/test_new_models.py 场景）

# 5) 归位 + 登记
mkdir -p ~/models/lfm-gguf/<name>
mv out/${TAG}_text-bf16.gguf   ~/models/lfm-gguf/<name>/qwen3_asr_0p6b_text-bf16.gguf
mv out/${TAG}_text-q4_k_m.gguf ~/models/lfm-gguf/<name>/qwen3_asr_0p6b_text-q4_k_m.gguf
sha256sum ~/models/lfm-gguf/<name>/*.gguf   # 写入本文件 §2
```

耗时：提取 ~2 分钟 + 转换 ~10 秒 + 量化 ~3 秒（本机）。
坑位记录：llama-quantize / llama-server 缺 `libgomp.so.1` → 带 `LD_LIBRARY_PATH=$TCHLD` 跑；nohup 后台的 llama-server 会被 shell 会话回收 → 用常驻终端（pty）起。

## 6. 行为记录（按模型）

测试条件：V1 眼镜 prompt（系统+任务+历史 scaffold），temp=0 / top_k=1，宿主机 b10083 直测，2026-10-16。

| 模型 | 记忆召回（"我叫什么名字？"） | 复读吸引子（历史含"请说…内容"对） | 备注 |
|---|---|---|---|
| #1 90h_20261014 | "我叫林悦。"（主客体混淆） | **中招**：逐字复读用户句 | 桥端已加过滤兜底（纯标点跳过 + `isStorableTurn` 拦"请说…内容"对） |
| #2 mt648 | "林悦。" ✅ | **不复读** ✅ | T1 确认句偏简短（"林悦，林芳，陈明。"） |
| #3 all748 | "您叫林悦。" ✅（措辞最准） | **不复读** ✅ | **2026-10-16 已部署**；手机直探与宿主直测逐字一致（temp=0 确定性交叉验证） |
| #4 glasses352 | **不稳**：2 轮 T2 逐字复读问题；3 轮交错 R3 答"我叫助手。"（身份坍缩） | 不复读 ✅（T5 通过） | step 352 欠训（对比 648/748）；翻译/身份正常；虽以 glasses_prompt_v1 微调但记忆召回弱于 #3/#5，不建议部署 |
| #5 mtfinal | 2 轮"林悦。"✅；3 轮交错"我叫林悦。"✅ | **不复读** ✅ | 源 checkpoint 已全 BF16；直测全过，表现与 #3 同档 |

#4/#5 补充场景（2026-10-16，3 轮交错：自我介绍→"中国首都"→"我叫什么名字？"）：
- glasses352：R1"林悦，林芳，陈明。" / R2"中国的首都是北京。" / R3"我叫助手。" ❌
- mtfinal：R1"林悦，林芳，陈明。" / R2"中国的首都是北京。" / R3"我叫林悦。" ✅

**指纹依赖推理配置**（2026-10-16 实测，temp=0）：同一权重在宿主默认配置（cache_prompt on + 默认 batch）与手机配置（`--no-cache-prompt` + batch 128/64，RuntimeService 实际参数）下 T2（"我叫什么名字？"，2 轮林悦 scaffold）输出不同——0.6B 在该问法上是临界 logits，数值分岔放大成整句差异。手机内判别模型要用**手机配置**的指纹：

| 模型 | T2 @ 宿主默认配置 | T2 @ 手机配置（判别用） |
|---|---|---|
| #3 all748 | "林悦。" | "我不知道你的名字，但是你可以叫我"助手"或者"AI助手"。" |
| #5 mtfinal | "林悦。" | "我叫AI助手，不知道你叫什么名字。" |

（两者手机配置指纹可区分；部署 mtfinal 时已用此表 + 磁盘 sha 双验证。）

身份问法敏感（mtfinal，手机配置）："你是谁？"/"你是谁开发的？"→"我是深圳河套学院研发的智能眼镜 AI 助手。"✅；"你叫什么名字？"→"你好，我叫智能助手。"❌ 不提河套。system prompt 本身已确认进入 prompt（token 数吻合），属 0.6B 对问法的泛化不足。

翻译模式 exemplar 覆盖（2026-10-16）：单 exemplar 时 EN2ZH 问句 6 中 5 个被**英文回答**而非翻译（"What's the weather like today?"→"The weather is currently cloudy."）；桥侧 exemplar 扩为 3 条（陈述+问句+自我介绍）后 EN2ZH/ZH2EN 各 8/8 全对（`GlassesBridgeService.kt` BridgeMode.examples，spec prompt 未动）。根治需训练侧补"问句也翻译"的 SFT 样本。

桥侧 TTS 归一化（2026-10-16，`ChineseTtsNormalizer.kt`）：matcha_zh_baker 的 lexicon 无任何 ASCII 数字/字母条目，"1加1等于2"被读成"加等于"、"ABCD"整段静音。zh 模式下数字转汉字（≤4 位数值读/年份逐位/百分号）、字母转读音（常见词表 + 逐字母）后 TTS 正常。

手机实测（2026-10-16，部署后 16022 直探）：mtfinal 行为指纹与宿主机手机配置直测逐字一致，确认线上跑的是新权重。

其它已知共性（与权重无关，桥端已处理）：方言/情绪化发言 ASR 噪音可能引入脏历史 → 眼镜端"清空历史"按钮（`clear_history` 协议，触发 LLM 重启）。

## 7. 维护规则

1. 正式 gguf 只存 `~/models/lfm-gguf/`；cache 里只放符号链接与可再生中间物。
2. 换权重：先 §1 后 §2，sha 必须实测后填写，禁止抄旧值。
3. 新模型先宿主机直测（§5 第 4 步）再上眼镜；上手机前 `sha256sum` 核对。
4. 本文件与 `LLM_SWAP_AND_GLASSES_PLAN.md`（执行方案）配套：方案讲"怎么做的"，本文件讲"现在有什么"。
