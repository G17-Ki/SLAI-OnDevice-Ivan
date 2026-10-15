# LLM 替换 + Rokid 眼镜网关：执行方案与进度

日期：2026-10-15
状态：执行中（阶段 1 T1.8 手机验收 = **NO-GO**，2026-10-15 定案，见 §5.11；阶段 2/3 待 go/no-go 决策后启动）
上游文档：`reports/leader_briefing/LEADER_BRIEFING.md`（leader 汇报）
训练方材料：`scripts/qwen3_asr_llm_swap/PROMPT_ALIGNMENT.md`、`collator.py`、`dialogue_prompt.py`（用户提供，原样复制自 AudioTrain 仓库）

---

## 1. 目标

1. **保住现状**：手机上现有两个 app（`ai.liquid.lfmdemo` / `.runtime`）在开发全程保持可用、可回滚，且确认它们与仓库版本的关系。
2. **LLM 替换**：把 LLM 槽位（:6020）从 Qwen2.5-1.5B-Instruct 换成自训 Qwen3-ASR-0.6B 文本解码器（spoken_multiturn_90h checkpoint），**ASR/VAD/TTS/编排/进程/端口全部不动**。
3. **眼镜网关**：新增局域网服务，手机当端侧服务器，Rokid 眼镜经 LAN 调用模型获得文本+音频。推理仍 100% 在手机（离线属性不变）。

## 2. 硬规则

- 阶段 0 完成前，对 `ai.liquid.lfmdemo*` **永不** uninstall / `pm clear` / 覆盖安装。
- 一切开发只在 `.dev` 包（`ai.liquid.lfmdemo.dev` / `.dev.runtime`，重命名已完成，14 处改动，宿主门全绿）。
- 双 pair 不能同时运行（固定端口 8765/6010/6020/8080）；切换 = force-stop + 安装。
- 本机一切 llama.cpp 调用**必须显式 `-c`**（见 §7 事故 1）。
- 对拍/探针运行时**不使用 torch**（见 §7 事故 2）。
- 大产物放 `~/.cache/lmf-demo/llm-swap/`（持久目录），**不放 `/tmp`**（tmpfs，重启即失）。

## 3. 完整 TODO

### 阶段 0：手机连接与备份（最先做，等待用户连手机）

- [x] T0.1 只读体检：REDMAGIC NX809J（红魔 11S PRO+ 氘锋透明）/ Android 16 (SDK 36) / user 版 ROM；主包 vc=26 `0.6.0-unplugged`（装于 07-16，更 08-05）、runtime vc=1 `0.1.0-unplugged-runtime`（装于 07-29，更 08-05）；**两包均 DEBUGGABLE**；另有 4 个历史 probe 包 + ttsbench（未动，已记录）；无 `.dev` 包
- [x] T0.2 拉取已装 APK + sha256 → `~/phone-backup/2026-10-15/apk/`：**与仓库锚点对不上**（主 `97b28360…` vs 07-29 `a24e203a…`/08-05 `143b9688…`；runtime `081cba99…` vs 08-05 `33bf0d6d…`）→ 结论"版本标签与现仓库一致、构建产物不同"（疑签名 key/构建机不同或脏树；T1.7 新鲜构建后做内容 diff 定论）；两包同一签名 `a85bb3d1`
- [x] T0.3 私有模型清单：`~/phone-backup/2026-10-15/T0.3_provenance.json`（>1MB 文件全部 sha256）；**手机上 LLM = `qwen2.5-1.5b-instruct-q4_k_m.gguf` sha256 `6a1a2eb6…`，与 manifest 条目逐字节一致 → 替换目标确认**
- [x] T0.4 OEM 快照：两包均在 Doze 白名单；主包 appops `RECORD_AUDIO: foreground` 其余 ignore；`dumpsys AutoLaunch` 在 REDMAGIC ROM 不存在（已记录）；RuntimeService 前台运行中
- [x] T0.5 **无条件全量**私有数据备份（因 T0.2 对不上锚点）：main 913MB + runtime 1.3GB tar.gz，`tar -t` 验证完整，sha256 入 `data/SHA256SUMS`
- [x] T0.6 冻结规则生效（`T0.6_freeze.txt`：锚点=APK 哈希+versionCode+数据 tar 哈希；每阶段前后复验）

### 阶段 1：LLM 替换（全部在 `.dev` 包）

- [x] T1.0 prompt 对齐材料入库（3 文件 → `scripts/qwen3_asr_llm_swap/`）
- [x] T1.1 抽取脚本 `extract_text_decoder.py`（纯标准库）+ 运行验证：310 张量、Qwen3 标准布局、config `max_position_embeddings=40960`（原训练导出为 65536，已压限，见 §7 事故 1）
- [x] T1.2 A 侧对拍（分词敏感性，纯 tokenizer 无模型）：每 prompt 仅 1–5 个结构边界 merge（`\n\n` 合并，占 prompt 1–3%，文本相同）；HF 上 seg 与整串输出一致 → **整串 raw prompt 路径放行**（A2 模型级在 T1.8 端到端覆盖）
- [x] T1.3 GGUF 管线（b10083：`convert_hf_to_gguf.py` → bf16 → `llama-quantize` → q4_k_m）+ 产物（见 §6）
- [x] T1.4 B 侧对拍 + **部署路径验证**（llama-server `/v1/completions`，手机实参）：13/13 格式干净，多轮记忆正确（详见 §5.4）
- [x] T1.5 5 个锚点 + JS raw prompt 构造（全部落地，见 §5.6）：
  - `RuntimeService.java:26` 模型文件名常量（→ `qwen3_asr_0p6b_text-q4_k_m.gguf`）+ alias `Qwen3-ASR-0.6B-Text-Q4_K_M`（`--ctx-size 2048` 等启动参数未动，`--jinja` 保留但不再走 chat 端点）
  - `candidate_manifest.json` 新增条目（自训权重无公开 repo：手动放入 `~/.cache/lmf-demo/android-text-companion/models/`，安装器只认 SHA-256 = `f9c2f256…`；安装器 manifest 查询 id 同步改）
  - alias/身份断言：`install_app_owned_runtime.sh:11` + `index.html:577` `QWEN_TEXT_MODEL` + `/v1/models` 校验（列表端点，与 chat/completions 无关）
  - `test_runtime_companion_source.py` 断言更新
  - `index.html`：按 `dialogue_prompt.py` 移植 prompt 构造（`base_prompt` + 逐轮 segment，单条 user 轮折叠历史），请求端点从 `/v1/chat/completions` 改 `/v1/completions`（raw prompt），system 用训练版中文 SYSTEM（`dialogue_prompt.py:6`），**不注入** 多语言 policy（靠 ASR 转录语言自然带出）；采样定 **greedy（temp 0 / top_k 1）**——宿主验证里唯一算术正确且 13/13 格式干净的配置（见 §5.4 发现 3）
  - 遥测标签 `QWEN_RESPONSE_BACKEND` → `qwen3_asr_0p6b_text_q4_k_m_cpu`（验收证据 `response_backend`/`response_model` 字段随真）
- [x] T1.6 宿主门全绿：9/9（8 个原门 + `test_phone_acceptance_evidence.mjs`；`test_adaptive_energy_vad.sh` 本机无 javac 按惯例预期失败）
- [x] T1.7 构建 dev pair + 安装 `.dev`（含 `.dev` 两包 OEM AI-control 配置）——**本机完成**（2026-10-15，用户批准；JDK 17/SDK 35/Gradle 8.9 全装齐，host 缓存从手机字节重建）；主 .dev + runtime .dev 均装好并验证，runtime 自检过（新模型 Qwen3-ASR-0.6B-Text-Q4_K_M 在 6020，ASR 双后端 ready），deviceidle 白名单 + active 桶已配。细节见 §5.8/§5.10（711 DAC 修复、activity 全限定名、push+pm install 通道）
- [ ] T1.8 手机验收：**NO-GO（2026-10-15 定案，详见 §5.11）**。五语言矩阵 4/5（JA 确定性幻觉"ホーニャ"）；28 题扫描 **14/28** vs 基线 23/28（ko 0/3、de/it/ru 1/3、en 2/4、zh 2/3——回退所需的 zh/en 也低于基线）；时延 first_token 中位 272 ms（基线 304）、PSS LLM 侧 ~1.1 GB（基线 ~2 GB，**减半，唯一明确赢面**）；**稳定性硬伤（已修复）**：app 默认 `cache_prompt:true` 路径本会话 5 实例 3 个退化/挂死（空生成→/health 无响应），冷启动首请求即死 2 次；`RuntimeService` 加 `--no-cache-prompt` 后 28 连发（客户端仍发 true）+ 五语言全链路全过、服务存活，且单 slot 前缀增量 prefill 保留（同 prompt 复测 first_token 14 ms）。结论：语言路由回退亦无数据支撑（新模型连母语 zh/en 都低于 1.5B 基线）→ 建议整体回退保留 Qwen2.5-1.5B；0.6B 若只留"内存减半"价值，需换更小的稳定模型或 1.5B 更低量化

### 阶段 2：Rokid 眼镜局域网网关（`.dev` 先行）

- [ ] T2.1 契约规格：`POST /v1/turn`（16 kHz 单声道 PCM16/WAV 音频 = ASR 原生契约免重采样；language、history?、tts:bool → text、分块流式音频、分段时延）、`GET /health`（execution-verified 语义）、Bearer token 鉴权（手机生成/眼镜一次性录入）、LLM 单 slot（`--parallel 1`）→ 全局串行队列；v1 眼镜负责自己的端点 VAD，发完整 utterance（流式/打断 = 二期）
- [ ] T2.2 网关实现（主 `.dev` APK 新增前台服务）：绑 `0.0.0.0:9090`（**只暴露这一个新端口**，6010/6020/8765 保持回环，网关内部代理）、WiFiLock、前台保活、音频流式回传
- [ ] T2.3 眼镜端 stub 客户端（脚本化完整轮，先不依赖眼镜真实固件）
- [ ] T2.4 局域网验收：单轮 text+audio、时延分解（ASR/LLM/TTS/LAN 各段）
- [ ] T2.5 鲁棒性：灭屏 soak、Wi-Fi 漫游/断连重连、50 轮连发稳定性、温度/电池（**从"未声明"变一等验收项**）、手机+眼镜并发排队行为
- [ ] T2.6 文档：契约 + 运维 runbook + 安全说明（LAN 暴露面、token 管理）

### 阶段 3：回迁生产（门：T1.8 + T2.5 全过）

- [ ] T3.1 同改动应用到 `ai.liquid.lfmdemo*` 生产包 → 全新设备验收（现有 app 数据不动，`install -r` 同签名）
- [ ] T3.2 旧 APK 哈希留作回滚锚点；眼镜接入 UX（token 录入）

## 4. 效果区别（新方案 vs 现仓库方案）

**A. 推理侧（换 LLM，手机+眼镜两条路径都受影响）**

| 维度 | 现方案（Qwen2.5-1.5B 通用） | 新方案（Qwen3-0.6B 自训） |
|---|---|---|
| 中文口语对话 | 通用能力兜底 | 预期更好：90h 任务微调，回答为 TTS 调优（简洁、不编造、记忆受训） |
| 首句时延 | 672 ms 中位 | 预期更快（参数少 2.5 倍）；宿主 8 线程实测单轮 0.1–1.1 s（含 prefill） |
| 内存 | PSS ~2 GB | ~1.3 GB，给眼镜网关留余量 |
| 英/俄/日/韩/法 | 24/28 基线 | 预期退化（小模型+中文中心数据），T1.8 定生死，有语言路由回退 |
| 输出风格 | 通用指令遵循 | 固定中文 system + 角色框架，更像"语音助手"；多语言 policy 不再适用 |
| 已知弱点（宿主实测） | — | 算术 17×23 在 phone 采样(temp 0.2)下答 451（greedy 答 391）；珠峰题截断"是8,"（所有配置均现） |

**B. 系统侧（手机从单机 app 变成端侧服务器）**

| 维度 | 现方案 | 新方案 |
|---|---|---|
| 网络依赖 | 飞行模式可用 | 局域网成运行时依赖：Wi-Fi 稳定性/漫游/省电进入关键路径 |
| 安全面 | 全回环无暴露 | 新增 1 个非回环端口 + token 鉴权；6010/6020 仍不暴露 |
| 续航/热 | 从未做 endurance 声明 | 持续服务 = 长时 CPU 负载，热/电池变一等验收项（T2.5） |
| 并发 | 单用户单机 | 单 LLM slot → 手机本地与眼镜串行互斥 |
| 时延预算 | 说完→出声 2 s（纯手机内） | 服务端部分仍 2 s；用户感知 += 眼镜麦克风端点 + LAN RTT + 眼镜播放 |
| 验收范围 | 手机本地+飞行模式已验收 | 局域网服务是全新证据域，T2.4/T2.5 是新增门槛 |

**C. 不变**：离线闭环（眼镜只是客户端）、ASR 模型与 10 s 契约、TTS 引擎与路由、手机本地 UI 流程、双 APK 进程架构。

## 5. 进度记录与证据（2026-10-15）

### 5.1 抽取（T1.1）✅

- `scripts/qwen3_asr_llm_swap/extract_text_decoder.py`：纯标准库读 safetensors 头 + 原始字节切片重写，任意 Python 可跑
- 源模型事实：611 张量 = `thinker.audio_tower.*` 301（丢弃）+ `thinker.model.*` 310（= embed_tokens 1 + 28 层 × 11 + norm 1，标准 Qwen3 命名，无独立 lm_head，tied embeddings）
- 抽取产物（spoken_multiturn_90h）：28 层 / hidden 1024 / 16 头 / 8 KV / head_dim 128 / inter 3072 / 词表 151936 / rope_theta 1e6 / rms 1e-6
- 全模型功能验证（HF，torch）：算术正确、多轮记忆正确（"第三个数字呢" → "是五"）、回答简洁

### 5.2 A 侧对拍（T1.2）✅

- 逐 prompt：seg-wise 编码 vs 整串编码，**仅 1–5 个 merge 差**（`base_prompt|user` 边界的 `\n\n` 合并 + 每多一轮历史 +1），占 prompt 1–3%，文本完全相同；首个分歧恒在 token 77（结构边界）
- HF 上两种编码输出一致 → 整串 raw prompt 可放行；训练方文档亦明确接受整段喂入（`PROMPT_ALIGNMENT.md:28-29`）

### 5.3 GGUF 管线（T1.3）✅

- b10083 的新架构：`convert_hf_to_gguf.py`（bf16/f16/q8_0，不再内置 k-quant）+ `llama-quantize` 二进制分步
- 产物（`~/.cache/lmf-demo/llm-swap/out/`）：
  - `qwen3_asr_0p6b_text-q4_k_m.gguf` — **396,700,032 字节（≈378 MiB），sha256 `f9c2f25634a7a6424254c955bf80a49ba425ceeb012f0eadfd14b8d16b6d13ce`**（T1.5 manifest 用）
  - `qwen3_text-bf16.gguf`（1.2 GB，对拍用）
  - 抽取 HF 目录 `qwen3_text/`（model.safetensors sha256 `5c1b649269167ffaae8cd3d7fb5ed96d8b8b5b753ca33c94ac01335a495ec390`）
- 对照：现手机 Qwen2.5-1.5B q4_k_m ≈ 1.1 GB → 新模型 ≈378 MiB

### 5.4 部署路径验证（T1.4）✅ —— 本阶段最关键发现

**发现 1（已排除的假警报）**：b10083 `llama-cli` 默认是交互 REPL，对 prompt 做会话模式处理 → 两种 checkpoint 在 CLI 下 10/10 落入 ASR 目标格式（`language X<asr_text>…`）+ 多轮复读。

**发现 2（部署路径无此问题）**：llama-server `/v1/completions`（= 手机 JS 的真实调用路径）raw prompt + 训练格式：

| 配置 | 格式干净率（13 题） | 多轮记忆 |
|---|---|---|
| q4 multiturn，phone 采样 (t0.2/p0.8/rp1.08) | **13/13** | "第三个数字呢"→"是五"✓ "明天几点开会"→"明天三点"✓ |
| q4 multiturn，greedy | **13/13** | ✓ |
| bf16 multiturn，两种采样 | **13/13** | ✓ |
| bf16 corpus（对照 checkpoint） | 13/13 | **✗ 幻觉**："下一个就是七"、"上海是上海市的省会" |

→ **checkpoint 确认 spoken_multiturn**（多轮记忆与手机场景匹配，corpus 版多轮弱）。

**发现 3（真实内容弱点，留给 T1.8 定量）**：
- 算术：multiturn 在 phone 采样下 17×23=451（✗），greedy 下 =391（✓）；torch 下 =391。→ 采样参数是本模型的敏感变量（**已定：T1.5 采用 greedy，temp 0 / top_k 1**）
- 珠峰题："珠穆朗玛峰的高度是8," 截断（3 种配置 × 2 种采样均现，0.6B 知识+解码弱点）
- 宿主 8 线程单轮时延 0.1–1.1 s（暖缓存）；手机 4 线程 ARM 会慢，但参数少 2.5 倍

### 5.5 事故记录（全部闭环）

| 时间 | 事件 | 根因 | 处置 |
|---|---|---|---|
| 02:20 | WSL 整机冻结重启 | llama-cli 未钉 `-c`，b10083 默认上下文=模型 max(65536) → KV 7.5 GB → OOM → vsock 死锁 | llama.cpp 调用钉 `-c 2048 --threads 8`；config `max_position_embeddings` 65536→40960（双保险） |
| 02:34 | HF/torch 进程 15.3 GB 被 OOM-kill（VM 存活） | transformers 5.19 + torch 2.14.1+cpu 连续 16 次 generate 病态膨胀 | 对拍运行时彻底移除 torch；B 侧改用 bf16-GGUF vs q4-GGUF（同 runtime 更干净） |
| 02:4x–03:0x | 对拍 B 首跑 30 min 超时 | b10083 `llama-cli` 无 `-e`（现为 escape 开关），one-shot 须 `-st`；REPL 等 stdin | 探针脚本改走 llama-server `/v1/completions`（与手机同路径） |

### 5.6 手机端代码替换（T1.5 + T1.6）✅

改动清单（git diff 22 文件，其中 14 个为此前 `.dev` 重命名）：

- `RuntimeService.java`：`TEXT_MODEL_FILE`/`TEXT_MODEL_ALIAS` 换新模型（server 启动参数未动）
- `index.html`：
  - `TEXT_API_URL` → `http://127.0.0.1:6020/v1/completions`；`QWEN_TEXT_MODEL` → `Qwen3-ASR-0.6B-Text-Q4_K_M`
  - 新增 `QWEN3ASR_SYSTEM` + `qwen3AsrBasePrompt` + `buildQwen3AsrPrompt`（`dialogue_prompt.py` 逐字移植，含中文弯引号“助手”/“我”/“你”）
  - `buildConversationPayload` 重写：raw `prompt` 字段（历史折叠进 prompt 串）、`temperature: 0`、`top_k: 1`、保留 `cache_prompt: true` + `max_tokens` 分级；删除 `messages`/`top_p`/`repeat_penalty`
  - `buildConversationUserText`：去掉语言 policy 行（保留 memory directive + 转录）
  - SSE 解析：`delta.content || choice.text`（兼容 chat 与 completions 流）
  - 遥测标签 `QWEN_RESPONSE_BACKEND` → `qwen3_asr_0p6b_text_q4_k_m_cpu`
  - 死代码保留：`QWEN_CHAT_SYSTEM_PROMPT`/`conversationSystemPrompt`/`RESPONSE_LANGUAGE_POLICIES` 不再被请求路径引用（回滚/语言路由备用）
- `install_app_owned_runtime.sh`：`model_alias` + manifest 查询 id → `qwen3_asr_0p6b_text_q4_k_m`
- `candidate_manifest.json`：新条目置顶（bytes 396700032 / sha256 `f9c2f256…`，无 repo/url，手动投放模型缓存）
- 测试同步（旧行为锚点更新为新契约）：`test_runtime_companion_source.py`（模型常量）、`test_android_native_tts_bridge.py`（端点+alias）、`test_streaming_audio_ui.mjs`（raw prompt 结构断言：prompt 前缀/系统行/历史折叠/结尾段；遥测 backend 标签）、`test_phone_acceptance_evidence.mjs`（fixture 模型名/backend）
- 证据规则适配（`phone_acceptance_evidence.mjs` `responseThinkingDisabled`）：新增"raw prompt 无 thinking 模式"类别（匹配 `qwen3-asr-0.6b`，含 `0p6b` 文件名变体，对 Qwen3-1.7B chat 模板路径无假阳性）——因为验收证据的 `enableThinking` 取自 `chat_template_kwargs.enable_thinking`，raw completions payload 无此字段，旧规则会把新模型误判为"未显式关 thinking"
- 门结果：9/9 PASS（`test_adaptive_energy_vad.sh` 本机无 javac 按惯例预期失败）；全仓扫描无功能性旧锚点残留（`phone_acceptance_plans/*.json` 为 Qwen2.5 时代历史计划文件，保留；`qairt242_colab_*` 无关实验）

### 5.7 手机连接 + 阶段 0 备份 ✅（2026-10-15 上午，全部产物 `~/phone-backup/2026-10-15/`，共 2.3GB）

**传输通道事故与解法（重要环境事实）**：
- 无线调试（adb over Wi-Fi）在此机不可靠：REDMAGIC 省电策略反复杀无线 adbd 会话，**每次会话重启端口都变**；3 次长传输 3 次截断（746MB/84MB/162MB），`svc power stayon` 保屏也压不住
- 解法：USB 通道（手机本就插着电脑）——Windows 管理员 PowerShell：`winget install usbipd` → `usbipd bind --busid 1-1` → `usbipd attach --wsl --busid 1-1`（5.3.0 语法：`--busid`/`--hardware-id`，无 `--vendor`）；WSL 侧 udev 规则 `99-usb-adb.rules`（`SUBSYSTEM==usb,MODE=0660,GROUP=plugdev`，经 `wsl -u root` 写入+手动 chown）
- **坑**：usbipd 挂入后手机节点在 `/dev/bus/usb/001/002`（bus 1），不是 vhci hub 所在的 002 总线；adb 权限检查不认 world 位，必须 group=plugdev 660
- USB 传输速率：~20MB/s，2.5GB 全量备份共 2 分钟；`stay_on_while_plugged_in` 已从 1 恢复

**关键结论**：
1. 手机 app ≠ 仓库证据锚点构建（T0.2），但版本号一致、同一签名对、debuggable → 完整字节级恢复能力 = 已拉 APK + 全量数据 tar（不依赖知道源码是什么）；T1.7 新鲜构建后做内容 diff 定论"差在哪"
2. 替换目标实锤：runtime `files/text/models/` 里就是 `qwen2.5-1.5b-instruct-q4_k_m.gguf`（`6a1a2eb6…`，与 manifest 逐字节一致）
3. 主包 `files/models/` = LFM2.5-Audio-1.5B 直连音频链（4 个 GGUF ~1GB），`files/asr/` = sensevoice + 俄 + QNN runtime（~330MB）——均不在本次改动范围，已哈希存档
4. 4 个 probe 包 + ttsbench 全程未动（版本已记录在 freeze 文件）

### 5.8 T1.7 本机构建环境重建 ✅（2026-10-15，用户批准本机装）

**工具链**（全部在 `~/.cache/lmf-demo/`，磁盘 770GB）：
- JDK 17 = `tools/jdk17`（Temurin 17.0.20.1；Adoptium 官方源 35KB/s → 清华 TUNA 镜像）
- Android SDK = `android-sdk`（cmdline-tools 11076708 + platform-35 + build-tools 35.0.0，licenses 已接受）
- Gradle 8.9 = `~/.gradle/wrapper/dists/gradle-8.9-bin/90cnw93cvbtalezasaz0blq0a/`（官方 sha256 `d725d707…` 验证；services.gradle.org 单连接 39KB/s → 16 路分段 resume 拉取）

**网络限速事实（本机）**：`dl.google.com`/TUNA/aliyun 快；`repo1.maven.apache.org` DNS 被墙 → `~/.gradle/init.d/aliyun-mirror.gradle`（settingsEvaluated 钩子，aliyun google/public/gradle-plugin 前置 + 保留 flatDir sherpa-aar）；GitHub 直连 ~40KB/s → `https://gh-proxy.com/<github-url>` ~1.35MB/s（**曾出现 35KB 截断"成功"，一切下载必须按目标字节数+sha256 双验证，循环 `curl -C -` 续传**）；国内镜像均无 gradle 发行版；archive.org TLS 被墙。

**host 缓存重建（本机原只有 models/，全部从手机字节重建，零新编译）**：
- `android-text-companion/runtime/llama-b10083/`：15/15 文件与 `scripts/android_text_companion/assets.json` 固定规格逐字节一致（14 个负载取自手机 runtime APK `lib/arm64-v8a/`，llama-server=libqwenserver.so；LICENSE 1078B 从 gh-proxy raw@846e991e 下载验证 `94f29bbb…`）
- `npu-multilingual/build/output/sherpa-onnx-1.13.4-android-arm64-qnn/`：sherpa-onnx-offline（=libasrrunner.so）+ libonnxruntime.so 取自 APK；`runner-manifest.json` 重建（target {arm64-v8a,27,qnn} + EXPECTED_ASR_RUNNER_PATCH/PROTOCOL + `sherpa_onnx_commit=14280725…`（prepare_assets.py 硬性要求）+ files 实算哈希）
- `npu-multilingual/bundle/`：`prepare_sensevoice_sm8850_bundle.py --root` 制备（SM8850/QNN 2.40.0.251030/htp v81；三 tarball 经 gh-proxy 下载验证：sensevoice 162MB `ecbc1ffb…`/qnn-libs 51.5MB `1078c083…`/qnn-include 393KB `fafe1c9d…`；test_wavs 唯一来源 = sensevoice tarball）
- `android-asr-companion/`：russian zipformer 60MB tarball `d6a65156…`（downloads/ + 解包 models/ + .archive.sha256 marker）
- `sherpa-aar/sherpa-onnx-1.13.4.aar`：48MB `03f9c4df…`（:app 模块 flatDir 依赖）
- 验证链全绿：`prepare_assets.py --offline` PASS + `prepare_runtime_companion_jni.py` dry-run 21/21 native 负载
- 新模型 GGUF 已投放 `android-text-companion/models/qwen3_asr_0p6b_text-q4_k_m.gguf`（`f9c2f256…`）

**构建**：`./gradlew :app:assembleDebug :runtime-companion:assembleDebug`（applicationId 已是 `ai.liquid.lfmdemo.dev[.runtime]`）；install 主包 = `adb install -r app-debug.apk`，runtime 链 = `install_app_owned_runtime.sh`（它只建/装 runtime companion；`install_main_app_tts_bundle.py` 只装 TTS eval bundle，前提是主包已装）。

### 5.9 T0.2 定论 + 手机侧未合并改动移植（2026-10-15，用户批准移植）

**T0.2 最终结论（"哈希对不上"定论）**：
- runtime APK：新旧 26/26 条目一致，仅 `classes2.dex`(+8B)/`AndroidManifest.xml`(+40B) 字节差异 → 同源码同载荷，差异 = 构建元数据/签名块
- 主 APK：34/34 条目一致，差异 = `assets/demo-ui/index.html`（**+4342B：手机含 174 行仓库没有的改动**）+ dex（Java 常量差异）+ manifest
- 即：手机是从"HEAD + 174 行未提交 UI 改动"的脏树构建的；仓库快照（public-research 提交）落后于手机

**移植（三方合并：base=HEAD / mine=T1.5 / theirs=手机 index.html）**：仅 1 处冲突（buildConversationPayload），保留 T1.5 raw-prompt 契约（phone 侧 temperature 0.2→0 被 T1.5 的 `temperature:0` 吸收；top_p/repeat_penalty 属 T1.5 有意删除的旧 chat payload）。移植的 4 个簇：
1. CJK 首逗号 TTS 分块（`cjkLeadCommaMinChars`/`allowCjkLeadComma`/`nativeTtsLeadQueued`，zh/ja/ko 首块提前）
2. ASR 后端不可用自动回退（`asrBackendForSelection`/`fallBackFromUnavailableAsrRoute` → auto）
3. runtime 健康恢复（`runtimeRecoveryRequestedAt` 10s 冷却 + `ensureRuntimeCompanion()` + `fetchHealth` 超时 + `selectedRouteTransitionBusy` 守卫）
4. chat temperature 0.2→0（已被 T1.5 吸收）
- 合并后：无冲突标记、双方特性 grep 齐全、`node --check` 语法过、**9/9 宿主门绿**（含 test_streaming_audio_ui.mjs）
- 主 .dev APK 重装并验证：installed base.apk 内 index.html sha256 = 仓库 = `dd70941e…`
- **T1.8 实验干净度**：.dev 现 = 手机行为 + T1.5（LLM 槽位替换），对比基线成立

### 5.10 安装脚本 DAC 修复（2026-10-15）

- `install_app_owned_runtime.sh` 原 `chmod 700 "$remote_stage"` 在 user 版构建上必挂：shell-owned 700 目录阻断 app-uid `run-as cp` 穿越（"Permission denied"）；SELinux 实测不拦（run-as 可读 644 的 /data/local/tmp 文件）
- 修复：`chmod 700` → `chmod 711`（仅穿越；目录名含随机 PID，644 暂存文件不可枚举；装完即删）。已验证模型投放成功
- 同批环境事实：本机 `adb install`（streamed install）挂死，`adb push /data/local/tmp + pm install(-r)` 正常（45MB/s）→ 装 APK 一律走 push+pm install
- **activity 短名坑（.dev 重命名漏改点）**：runtime 包 `namespace=ai.liquid.lfmdemo.runtime`（Java 包）与 `applicationId=ai.liquid.lfmdemo.dev.runtime`（设备包名）分离，清单相对组件名 `.RuntimeActivity` 按 namespace 解析 → `am start -n <appId>/.RuntimeActivity` 报 "Activity class does not exist"（Error type 3）。脚本 `activity=` 已改全限定名 `ai.liquid.lfmdemo.dev.runtime/ai.liquid.lfmdemo.runtime.RuntimeActivity`（MainActivity 侧本就用全限定类名 bind，不受影响）
- **端口互斥实操**：生产 pair 与 .dev pair 共用固定端口（6010/6020），生产在跑时 .dev 自检会打到生产 server（identity mismatch：ids=['Qwen2.5-1.5B-Instruct-Q4_K_M']）。验收期间 `am force-stop ai.liquid.lfmdemo[.runtime]`（进程级，包/数据不动，用户开 app 自恢复）；验收结束 force-stop .dev pair 即可交还
- **已定根因：llama-server "空闲挂起"（2026-10-15，.dev 新模型）= 两种独立机制叠加，非 llama.cpp 单一 bug**：
  1. **Android cgroup freezer（系统侧，非模型相关）**：`/sys/fs/cgroup/apps/uid_<uid>/pid_<pid>/cgroup.freeze=1` 直接证实。早晨实例 = runtime 的 active-client wakelock 在 `onUnbind` 释放 → 服务被系统冻结后 rebind 的 `onBind` 永不执行（线程已冻）→ 永久冻结陷阱（dumpsys power 见 wakelock DISABLED + binder 挂起）。下午实例 = bootstrap `RuntimeActivity`（设计不自 finish）盖住 `MainActivity` → 主包 `curProcState=15 (CACHED)` → 被冻（`dumpsys activity processes` isFrozen=true）。修复（未提交）：`RuntimeService` 改 service 生命周期持锁（onCreate 获取/onUnbind 不释放/onDestroy 释放，tag `…runtime:service-alive`）+ `MainActivity` onStart/onStop/onDestroy 持 `ai.liquid.lfmdemo:activity-visible`（manifest 加 WAKE_LOCK）；`test_runtime_companion_source.py` 已同步断言。**操作规则：任何 CDP/验收前最后一步必须 `am start -W -n ai.liquid.lfmdemo.dev/ai.liquid.lfmdemo.MainActivity` 使主包置顶**。另：本 ROM `am set-standby-bucket` 静默无效、`pm install -r` 会重置 standby 桶、deviceidle 白名单不足以豁免。
  2. **LCP prompt 缓存损坏（模型/llama.cpp b10083 侧，T1.8 稳定性硬伤，见 §5.11）：已修复** —— `RuntimeService.java` 启动参数加 `--no-cache-prompt`（服务端关 slot 保存/恢复复用路径，客户端 `cache_prompt:true` 被忽略；单 slot 前缀增量 prefill 保留）。验证：28 连发（app 模式 warmup+间隔，客户端仍发 cache_prompt=1）28/28 零错误零空响应（修复前冷启动首请求即死）；harness v7 五语言全链路 5/5 complete、error=none、服务全程存活。`test_runtime_companion_source.py` 已锁断言；新 runtime APK sha256 前缀 `eac99ee6` 已装（13:59:22）
  - 旁证：`qwen-server.log` 曾因 C stdio 全缓冲（文件重定向）滞后数十 KB，"日志停更"≠"服务停摆"；41 个 startup banner 中 39 次 "couldn't bind 6020" = 死锁孤儿进程跨 force-stop 持端口。tokenizer 两处 warning（`</s>` control-type、special_eos_id）为 Qwen3-ASR 解码器特性，非挂起原因

### 5.11 T1.8 手机验收结果（2026-10-15 下午，.dev pair，NO-GO）

**门（§3）：五语言完整轮 + 28 题扫描 + 时延/PSS 遥测 vs Qwen2.5 基线 → go/no-go；不过则语言路由回退（zh/en 新模型，ru/ja/ko 留 Qwen2.5）**

1. **五语言矩阵 4/5**（harness v6，CDP 全链路 ASR→LLM→TTS，raw prompt 协议）：
   - EN ✓ 2370 ms "The capital of France is Paris."；ZH ✓ 13564 ms（完整北京描述）；RU ✓ 2713 ms（"Москва - столица России," 片段但断言过）；KO ✓ 2841 ms "프랑스의 수도는 파리입니다."
   - **JA ✗ 4235 ms：responseIncludes 失败**，响应 "フランス の 首都 は ホーニャ で すか？"（复述+幻觉"Honya"；temp=0/top_k=1 下 v3/v6 两次逐字相同 = 确定性模型质量失败）
   - 首句时延 firstAudio 632/741/1407/811/632 ms（EN/ZH/RU/JA/KO）、0 underflow → 与基线 672/1253 ms 同档
2. **28 题扫描 14/28 (50%)**（`scripts/sweep_qwen3_asr_0p6b_28cases.py`，raw prompt 逐字对齐 app，`work/t18_qwen3_asr_0p6b/sweep28_nocache.json`）vs 基线 **23/28 (82%)**（`reports/android_phone/streaming_response/qwen2_5_1_5b_q4_k_m_app_owned_phone.json`）：
   - 逐语言：en 2/4（基线 3/4）、es 2/3、**de 1/3**、**it 1/3**、fr 3/3（基线 2/3）、zh 2/3、ja 2/3（基线 2/3）、**ko 0/3**（基线 1/3）、**ru 1/3**（基线 3/3）
   - 门两条全破：≥23/28 ✗；无核心语言 <2/3 ✗（de/it/ko/ru）
   - 14 个失败全为真实模型失败（非 pattern 过严）：算术 3 错（17×23=451✗、9×7="12/7"✗、13+19=22✗；另 5 国算术全对）+ 知识 11 错/截断（富士山"about 1,"、"A presión estándar,"等早停；Sydney 当澳洲首都、Mond 当红行星、Roma 当葡萄牙首都、뉴욕 当加拿大首都、木星当月球、우주선(宇宙飞船)当最大行星、-193°C 当水的冰点…）。注：基线 1.5B 也错 ja/ko 首都（温哥华/慕尼黑）——此类题小模型普遍弱，但 0.6B 多退 10 例
   - 时延（cache_prompt=0）：first_token 中位 272 ms（基线 304）、first_speakable 中位 346 ms（基线 672）、生成 121 tok/s、prefill 388 tok/s
3. **PSS 遥测**：libqwenserver RSS **1.02 GB** + asrrunner 67 MB + runtime 进程 32 MB ≈ **1.1 GB** vs 基线 ~2 GB → **内存减半（唯一明确赢面）**
4. **稳定性硬伤（cache_prompt A/B）**：
   - `cache_prompt=0`：冷启动 28 连发全过、0 崩溃
   - `cache_prompt=1`（app 默认）：本会话 5 个服务实例 3 个坏——27761（5 case app 轮后连发退化：28 条全空 stop@167ms → 挂死）、30833（app 模式 warmup 首请求即 RemoteDisconnected）、32054（monitor 自动重启后首个请求再挂死）；挂死态 = /health 与 /v1/completions 全无响应、cgroup freeze=0（排除系统冻结）。修复后同形态 28 连发 + v7 全链路均正常（b10083 空闲时线程数本就只有 ~5，不能作挂死判据，判据 = 无响应）
   - 定性：llama.cpp b10083 prompt 缓存（slot 保存/恢复复用）在该模型上对连续/冷启动请求间歇性损坏（非逐请求确定，v6 的 5 轮曾成功）；**`--no-cache-prompt` 后同形态负载全过，修复有效且保留前缀增量 prefill（时延不降反升）**
5. **判定：NO-GO（纯质量原因；稳定性缺陷已按用户要求修复，见本节第 4 条与 §5.10 末）**。语言路由回退亦无数据支撑（回退要求 zh/en 新模型 ≥ 基线，实测 en 2/4 < 3/4、zh 2/3 < 3/3）。**建议：整体回退，手机 LLM 槽位保留 Qwen2.5-1.5B**；若内存减半是硬需求，替代路径 = 1.5B 更低量化（q3/q4_0 ≈ 1 GB）或换更小的通用模型，而非本 ASR 微调解码器（其强项是中文口语简洁应答，QA 面全线退化）
6. **副产物**：harness warmup-fetch 过滤（`responseTurnFetches` stream===true）；T1.8 plan 按 .dev 现实改写（EN 路由 = tts/android_native_tts：`NativeAudioBridge.java:40` LFM_RUNTIME_ENABLED=false 为快照固有；.dev 无研究 TTS bundle → 全语言 readyEngine="system"）；sweep 脚本（逐例落盘/健康探测中止/--cache-prompt/--app-warmup/--sleep）；wakelock 修复两 APK（runtime sha256 `9ddc68b3…`、主 `cf577709a40ac6a7…`，已装未提交）

## 6. 环境与复现

- 工作目录：`~/.cache/lmf-demo/llm-swap/`（持久）：`.venv`（uv + Python 3.12 + torch CPU + transformers）、`llama-b10083/`（预编译 ubuntu-x64 二进制，需 `LD_LIBRARY_PATH` 指 torch 的 libgomp）、`llama.cpp-b10083/`（converter 源码）、`out/`（产物）
- 重建命令（~5 分钟）：
  ```bash
  cd ~/.cache/lmf-demo/llm-swap
  python3 <repo>/scripts/qwen3_asr_llm_swap/extract_text_decoder.py \
      --model-dir ~/models/qwen3_asr_0p6b_spoken_multiturn_90h_20261014 \
      --out-dir out/qwen3_text
  LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/torch/lib \
    .venv/bin/python llama.cpp-b10083/convert_hf_to_gguf.py out/qwen3_text \
      --outfile out/qwen3_text-bf16.gguf --outtype bf16
  LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/torch/lib \
    ./llama-b10083/llama-quantize out/qwen3_text-bf16.gguf \
      out/qwen3_asr_0p6b_text-q4_k_m.gguf q4_k_m
  ```
- 探针：`python <repo>/scripts/qwen3_asr_llm_swap/llama_server_probe.py --server http://127.0.0.1:<port>`（llama-server 用持久 PTY 会话起，**不要** nohup 在 bash 工具里起——进程组会被工具回收）
- 本机约束：Python 仅 3.14（无 pip）→ 用 `uv`；无 root（libgomp 借 torch 自带）；`/tmp` 是 tmpfs

## 7. 待办中依赖外部的事项

| 事项 | 依赖 | 阻塞 |
|---|---|---|
| （已完成）阶段 0 全部 | 用户连手机 → USB/usbipd 通道 | — |
| （已完成）T1.7 构建 dev pair | 本机 JDK 17 + Android SDK 35（已装，§5.8） | — |
| （可选，建议）训练团队确认 | 用户联系 AudioTrain 团队：llama.cpp 路径下 dialogue 分支的行为（宿主已验证 server 路径干净，但值得让他们知晓 + 确认采样建议） | 不阻塞 |

## 8. 回滚

- 手机：`.dev` 隔离 → 现有 `ai.liquid.lfmdemo*` 全程未动；dev pair 失败 = force-stop + 卸载 dev 包，现版 app 原样可用
- 宿主：所有产物可重建（§6）；仓库脚本幂等
- 模型：若 T1.8 不过 → 语言路由回退（中/英新模型，俄/日/韩 Qwen2.5）或整体回退（保留 Qwen2.5 GGUF 于 `~/phone-backup/`）
