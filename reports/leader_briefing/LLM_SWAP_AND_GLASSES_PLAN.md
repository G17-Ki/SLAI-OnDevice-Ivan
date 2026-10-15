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

### 5.12 Rokid Glasses ADB 侦察 + 开发者模式解锁（2026-10-15 晚，用户自有 Rokid Glasses，固件 1.26.011）

**目标**：评估能否把麦克风流客户端 APK 装进眼镜（路径 C：眼镜侧 app → 16kHz PCM → 手机级联 → TTS 回传）。

**通道**（本机 = 用户 Windows 机器的 WSL2）：眼镜 USB 只对 Windows 可见 → Windows 侧 `adb -a -P 5038 start-server`（platform-tools 拷至 C:\adb-win），WSL 内经 NAT `adb -H 172.18.0.1 -P 5038` 操作。usbipd 转发 WSL 失败（attach rc=0 但设备未枚举，报错文本被 interop 层破坏为乱码不可读）；注意 **usbipd `bind`（Shared 态）会把设备从 Windows PnP 栈摘除**，排障完需 `unbind` 交还。

**设备画像**（只读侦察，零写入）：
- `Rokid/glasses/glasses:12/SKQ1.240613.001/1.26.011-20260929-150201:user/release-keys` — **Android 12 (SDK 32)、user 构建**（无 root）、SELinux **Enforcing**、高通平台、368 进程
- Wi-Fi radio 存在；/data 19G 空闲 16G；USB 接口 `18d1:4ee7 "RG-glasses-IDP"`，Windows 通用 ADB 驱动自动匹配并授权
- 设备内已有 `com.example.rokidhello`（69MB 语音助手 app：sherpa-onnx ASR 14M 中文 int8 + `librokid-llama.so` + onnxruntime + TTS kit 许可文件，minSdk24/targetSdk37，装于 09-30）——**第三方用 AI 做的社区 app，非 Rokid 官方**（用户更正）；仍是有用数据点：完整端侧语音链路（麦→ASR→LLM→TTS→喇叭）曾在这副眼镜上安装运行

**默认状态厂商锁定（用户开开发者模式前实测，三证据）**：
1. **`adb install` 被显式禁止**：8.5KB 最小探测 APK → `adb: failed to install: error: not allow package install!`（厂商定制报错，非标准 `INSTALL_FAILED_*`）
2. **shell 域一切 binder 服务调用被强制 kill**（会话关闭，非拒绝）：`pm list users`/`cmd package`/`cmd statusbar`/`dumpsys battery`/`service list`/`logcat` 全 `error: closed`；对照 `sleep 10`、50 连发 `getprop` 均正常 → 排除链路抖动，是策略层主动 kill（adbd 日志 tag `RKD--` = Rokid 定制 adbd，锁定在其内实现）
3. **user 构建 + release-keys** → 无 root

**用户开启开发者模式后（乐奇 App 内操作）→ 锁定全解**：
- `pm`/`service list`(203 服务)/`logcat`/`svc` 全部可用
- `adb install` 成功：8.5KB 探测包 `com.probe.lfm` 安装 OK（首两次失败为我方 APK 缺陷：缺 uses-sdk → `INSTALL_FAILED_NEWER_SDK`、缺 classes.dex → "code is missing"，非设备限制）
- **结论更新：路径 C 可用（前提 = 保持开发者模式开启；需确认 OTA 升级是否重置该状态）**
- 注意：开发者模式状态持久性未验证，每次 OTA 后需复检（`pm list packages -3` 能否运行即判据）

**Wi-Fi 实测（shell 全程可控）**：
- `svc wifi enable` 可用；`cmd wifi connect-network <ssid> open` 可连（`add-network` 子命令在该 build 语法失效，connect 自带保存即可）
- 眼镜连 `SLAI` 5GHz（BSSID `60:a6:c5:c5:a2:90` @5300MHz，IP 10.200.123.252/18）；手机连 `SLAI`（BSSID `d8:40:08:81:88:50` @5280MHz，IP 10.200.126.193/18）
- **同子网但双向 ping 全丢 = 同 SSID 多 AP 分 VLAN（6E/5G/2.4 分池）**；`SLAI-Guest`（眼镜原保存网）另属隔离 guest VLAN（同测不通）
- → **Wi-Fi 直连 TCP 暂不可用**（除非改路由器 VLAN/合并频段池）；**传输主选 = BT SPP**（RFCOMM，经典蓝牙，眼镜↔手机本就 BT 配对为通话设备，SPP 吞吐 200–1000 KB/s 对 16kHz PCM16=32KB/s 绰绰有余，且与所处 Wi-Fi 环境无关）；Wi-Fi TCP 留作改好路由后的加速选项

**Rokid 软件栈**（ps，均 system uid）：`com.rokid.os.sprite.assistserver`（+独立 `:tts` 子进程 = "乐奇" AI 助手后端）/ `com.rokid.os.master.screenstream`（眼镜→手机屏幕流，手机为 master）/ `com.rokid.os.sprite.{launcher,live}` / `com.rokid.glass.ota` / `com.rokid.sysconfig` / `com.rokid.cxrservice`；/product/app 含 Alipay/WeixinPay/JdPay（声纹支付）+ RokidOtaUpgrade/RokidScreenRecord

**路线裁定（更新）**：
- 路径 C（侧载）✅ **可用**（开发者模式开启时）：眼镜侧装客户端 app（麦→VAD→SPP/WiFi→手机级联→TTS 回流→喇叭）
- 路径 A（BT HFP 通话桥）降级为**备选**（C 已通；A 的价值 = 不依赖开发者模式，OTA 后若模式被重置可退回 A 的纯手机侧方案）
- 路径 B（官方开发者通道）仍值得并行询（open.rokid.com / `Glasses.BD@rokid.com`）：确认开发者模式是否官方支持、OTA 持久性、未来是否有正式 SDK
- T2.1–T2.5 契约仍有效，客户端形态 = 眼镜侧 app（C 路径，主）；契约按传输无关设计（v1 = BT SPP，后升 Wi-Fi TCP）
- **已知风险**：眼镜麦与 Rokid 自带"乐奇"助手共用麦克风（音频焦点竞争），客户端需按会话态申请/释放焦点

### 5.13 阶段 2 落地：手机桥 + 眼镜客户端 + SPP 连通（2026-10-15 晚，进行中，代码未提交）

**代码（全部编译通过并装机；分支 `t18-qwen3-asr-0p6b-tts-parity`）**：
- 手机侧（`.dev`）：新增 `glasses/GlassesBridgeService.kt`（SPP server，UUID `00001101-...3f83`，名 "LFM-Glasses"；Session = 16k PCM16 麦流→VAD→ASR 6010→LLM 6020 SSE→sherpa TTS→SPP 回写，tts_start/stop 半双工切麦；ClauseChunker/TtsPool/本地 PcmRing）+ `SppProtocol.kt`（帧 `[0x80|type][len:2BE][payload]`；AUDIO_IN=01/AUDIO_OUT=02/CONTROL=03；hello/ready/heard/said/tts_start{rate}/tts_stop/error）；Manifest 加 BLUETOOTH(_ADMIN maxSdk30)/BLUETOOTH_CONNECT + service dataSync；`NativeAudioBridge` 加 JS 钩子 start/stopGlassesBridge；MainActivity 加权限请求；demo-ui 加 "Glasses bridge" 按钮。`app-debug.apk` 20.3MB 装机
- 眼镜侧：新模块 `:glasses-client`（`ai.liquid.lfm.glasses`，minSdk24/target32）：SppProtocol.java 镜像 + `GlassesClientService`（3s 重连、secure→insecure SPP、VOICE_RECOGNITION 16k mono 1280B 帧、AudioTrack 播放、前台通知）+ `MainActivity`（bonded 设备 spinner + 语言 zh/en/ja/ko/ru + START/STOP，prefs `lfm.glasses`）。17.9KB 装机
- 新模块 `:bt-pair-probe`（`ai.liquid.lfm.btpairprobe`，createBond 工具）、`:pair-responder`（`ai.liquid.lfm.pairesponder`，无头配对应答器，见下）

**关键发现：Rokid 眼镜不显示系统配对对话框**：
- 显示层只镜像前台 app 窗口（眼镜 uiautomator 只能见 1×1 `com.rokid.sysconfig` MockWindow；screencap 黑屏=息屏 ~15s 自动睡）→ passkey 确认无人能点，`createBond` 三次全部 30s 超时（BONDING→NONE）
- 解法 = `pair-responder`：manifest receiver 收 `ACTION_PAIRING_REQUEST`，等主机侧把手机显示的 6 位码送达（`am start -n ai.liquid.lfm.pairesponder/.MainActivity --es pin <码>`），反射调 `setPin()` 程序化确认（本项目 android.jar stub 缺 `BluetoothPairingRequest` 类，故全反射 + extras 字符串字面量）→ **17:51:13 一次成 bond（BOND_STATE_BONDED）**
- 副产物：拿到手机 BT MAC `dc:f0:90:68:76:6e`（手机 Android 16 自身 dumpsys 掩码；眼镜 Android 12 侧不掩码）；眼镜 MAC `AC:86:D1:5A:E9:73`（名 Glasses_1491）

**端到端状态（17:53 时刻）**：
- Bond ✅（眼镜 bonded 列表 = 一加12 `BC:E8:FA:8F:F1:C6` / HONOR 90 Pro / 红魔11S PRO+氘锋透明 `DC:F0:90:68:76:6E`）
- SPP 连接 ✅（眼镜 logcat 17:52:20 "connecting to 红魔11S PRO+氘锋透明 → connected"；手机 `GlassesBridgeService` 运行中；6010/6020 LISTEN；demo UI 经 `adb reverse 8765` 正常加载）
- 麦流 ✅（眼镜 PAL 日志 audio-record 流开启，1280B/40ms≈32KB/s）
- **待验收**：对眼镜说一句中文 → VAD endpoint→ASR→LLM→TTS 从眼镜喇叭放出；live 捕获（持久 PTY `logcat -s LfmGlassesClient:V`），预期日志 `bridge heard` → `bridge said` → `speaker up at 24000 Hz`

**本轮新环境坑（沉淀）**：
- 手机 logcat 损坏（main 缓冲全空，OEM 限制）→ 观察靠 dumpsys/dropbox/screencap；眼镜 logcat 可用但 PAL 音频日志冲爆 ring（~9 万行/分钟）→ 必须 `-s` tag 过滤或流式进 PTY
- **碰手机 WebView devtools 端口（`webview_devtools_remote_*` forward + curl）会直接崩 app**（SIGTRAP TRAP_BRKPT in libwebviewchromium，devtools protocol CHECK；dropbox data_app_native_crash）——禁碰
- 手机 Android 16 shell 不能 start 非 exported 服务（"Requires permission not exported"）；activity 启动须全限定 `am start -n ai.liquid.lfmdemo.dev/ai.liquid.lfmdemo.MainActivity`；`cmd bluetooth_manager` 无 scan/pair 子命令；眼镜无 `cmd bluetooth_manager`
- 演示 UI = 主机 `python3 -m http.server 8765 --bind 127.0.0.1`（PTY 起）+ `adb reverse tcp:8765`（手机→主机方向）；主机长驻进程必须 PTY 起（nohup 会被工具会话回收）
- 眼镜 prefs 可 `run-as ai.liquid.lfm.glasses` 改（debuggable）：`shared_prefs/lfm.glasses.xml` 的 address
- Kotlin 移植坑：`Regex` 无 `.test()`（JS 习惯，单字符用 `.matches()`）；`AdaptiveEnergyVad` ENDPOINT 后必须 `rearm()`；`OfflineTts` 是 `release()` 非 `shutdown()`；包私有 `PcmRingBuffer` 跨包子包不可见 → 本地 `PcmRing`；inner class 的 private 成员外层不可访问
- UI 坐标（供复现）：眼镜 480×640，START≈(125,428)；手机 1216×2688，swipe(540,1800→540,600) 后展开 advanced (460,732)、Glasses bridge 按钮 (578,2044)、配对对话框 配对 按钮 (881,2524)

### 5.14 SPP 帧协议 bug 修复 + 握手成功 + 新需求（2026-10-15 18:30–19:00）

**帧级通信打通**：18:46:47 手机 `session started for Glasses_1491` + 眼镜 `bridge ready` / `bridge hello_ack`——SPP 双向控制帧首次真正握手成功，客户端稳定连接。

**根因：SppProtocol 帧头校验 bug（两侧同款，从首版就有）**：Reader 要求首字节 **恰好** `0x80`，而协议实际帧头是 `0x80|type` → 任何带类型的帧（AUDIO_IN 0x81 / AUDIO_OUT 0x82 / CONTROL 0x83）都抛 `desync: bad frame flag`。所以此前"connected"后从未交换过一帧——17:52/18:21 的"链路验证"实际只验证到 RFCOMM 层。修复：`(header & 0x80) == 0x80` 即合法。`SppProtocol.kt` + `glasses-client/SppProtocol.java` 同修。

**同轮修复/加固**：
- 手机桥 `GlassesBridgeService`：构造期 `filesDir` NPE（属性初始化早于 context 注入 → `by lazy`）；BT-off 时静默 `stopBridge()` 无任何日志（三处早退加 `blog()`）；accept 循环单点故障（旧代码 accept 抛 IOException 即 break 永久停服 → 重开 socket 永不 break）
- **手机 logcat 完全锁死**（main/system/crash 全空，OEM 限制）→ 桥新增文件日志 `files/glasses-bridge.log`（run-as 可读），覆盖 onStartCommand/listen 成败/session 启停/VAD endpoint/turn heard/done/FAILED/TTS 引擎加载
- shell 直启钩子（绕开 UI 点击自动化）：手机 `am start -n ai.liquid.lfmdemo.dev/.MainActivity --ez lfm.glasses_bridge_start true`；眼镜 `am start -n ai.liquid.lfm.glasses/.MainActivity --ez lfm.auto_start true`（读 prefs address/lang 直接起服务）
- 手机 `startGlassesBridge` JS 钩子加 1.5s 存活校验（getRunningServices），失败信息指向文件日志

**环境坑（续）**：
- 手机 BT 曾被打到 OFF（原因未明，非我方命令）→ 桥启动时 adapter off 即自杀；排查顺序 = `dumpsys bluetooth_manager | grep '^state'`
- 乐奇 AssistServer（`com.rokid.os.sprite.assistserver`）在场景切换时 `ThirdAppScene -> forceStopPackage` 杀所有第三方 app（已杀眼镜客户端两次）；`pm disable-user --user 0` 被厂商 build 拒绝；短期对策 = 测试前 `am force-stop` + 客户端 auto_start 一条命令即恢复
- 用户一加 12 经 A2DP 往眼镜放音乐（"歌声"）——测试前须关其蓝牙
- 手机侧 SPP 连接曾出现"幽灵 SDP"：旧进程 socket 的 SDP 记录疑似残留，客户端 `connected` 但手机 accept() 永不触发；`svc bluetooth disable/enable` 整栈重置后消失（未 100% 定因，复发即先做此重置）
- 眼镜 USB（Windows adb 5038 隧道）偶发断链（长命令 `error: closed`，短命令可通）→ 拆短命令重试

**用户确认的新需求（排入阶段 2 收尾，E2E 语音验收后做）**：
1. **打开即自动连接**：眼镜 app 打开 = 自动起桥（不再点 Start；prefs 无 address 时才回退手动选择）
2. **文字显示**：眼镜 app 窗口显示用户转写 + 模型回答（控制通道现成，手机发 `heard`/`said` 文本帧 → 客户端 UI 渲染）

**待验收（当前）**：用户戴眼镜说"你好，1加1等于几？"→ 预期 `vad endpoint` → `turn 1 heard: …` → LLM → TTS → 眼镜喇叭出声 + `speaker up at NNNN Hz`。级联 6010/6020 已确认 LISTEN，会话 18:46:46 建立且无断连日志。

### 5.15 阶段 2 E2E 语音验收通过 + 收尾两项落地（2026-10-15 19:20–19:50，代码未提交）

**E2E 语音全链路验收通过（用户在场亲测）**：戴眼镜说"请你给我讲1个30个字的故事。"→ `vad endpoint 3.02s` → `turn g-8 heard` → LLM 出 38 字故事 → TTS `matcha_zh_baker @ 22050Hz` → 眼镜喇叭完整播出（用户确认"效果不错"）。随后**连续 7 轮**问答（g-8…g-13）全部成功：ASR 转写 → LLM → TTS 完整播放 → 麦克风自动恢复 → 下一轮立即可说。无 desync、无麦流中断、无 TTS 截断。

**验收前修复的三个客户端 bug（都在 `glasses-client`）**：
1. **麦线程一轮即死**：`while (micStreaming && running)` 循环，TTS 期间 `pauseMic()` 置 `micStreaming=false` → 线程退出，`resumeMic()` 只置位无线程 → 麦克风永久死（"只能测一次"根因）。改持久线程 `while (running)` + 暂停时 `sleep(20)`，释放才退。
2. **TTS 尾巴被丢**：手机发完音频即发 `tts_stop`，客户端旧逻辑 `stopSpeaker()` 立即 `release()` AudioTrack 并丢队列 → 短回答播 ~2 字即断（"回答两个字就断掉"根因）。改 **drain 模式**：`tts_stop` 只置 `speakerDrain=true`，播放线程把队列排空才真正停。播放改为独立线程 + `LinkedBlockingQueue` 异步，readLoop 不再被 AudioTrack 阻塞（防 RX 停顿→对端缓冲溢出）。
3. **大帧分片**：手机 `send()` 按 `MAX_SEND_PAYLOAD=8192` 分帧，避免单帧过大。
   - 注：`OutputStream.write(byte[],int,int)` 返回 `void`（非计数），无"部分写"回写循环；单次 `write(frame)` 即全量。

**收尾两项（用户批准）已落地**：
1. **打开即自动连接**：`MainActivity.onCreate` → `maybeAutoStart()`（权限齐 + 有配对手机即自动起服务，不再点 Start；`lfm.auto_start` extra 钩子保留兼容）。裸 `am start` 验证 19:49:57 自动连上。
2. **文字显示**：手机 `heard`/`said` 控制帧（带 `text`）→ 客户端 `GlassesClientService.UiListener`（静态 `CopyOnWriteArrayList`）→ `MainActivity` 两个 TextView（我 22sp / 助手 26sp，大字适配眼镜）实时渲染。

**UI 重排 + 屏幕常亮（用户反馈后）**：
- 对话区（我/助手大字）移到屏幕顶部，设置项（设备/语言 Spinner + Start/Stop）压到底部小字（"位置太靠下/上面都是设置"反馈）。
- **屏幕常亮**：`FLAG_KEEP_SCREEN_ON`（onCreate 加到 window）。**关键硬件特性：Rokid 眼镜麦克风增益与屏幕唤醒状态绑定**——`dumpsys power` 显示 `mWakefulness=Asleep` 时麦峰值仅 51-67（VAD 只见 short burst 不出 endpoint，"说话没反应"根因）；Awake 时正常（说话峰值 1385-2177）。排查命令：`dumpsys power | grep mWakefulness`；手动唤醒 `input keyevent 224`（~15s 自睡）。
- **最终验收（用户确认"现在非常好"）**：E2E 多轮语音问答 + 退出 app 重开自动连接识别 + 文字显示 + 常亮，全部通过（19:58 前后）。
- **手机桥打开即自启（用户要求，对称眼镜侧）**：`MainActivity.maybeStartGlassesBridge()` 去掉 `lfm.glasses_bridge_start` extra 条件（extra 保留兼容为 no-op），权限齐即 `startForegroundService`。验证：20:11:18 裸启动 → +4ms `SPP server listening` → +2s 眼镜自动重连。**冷启动完整流程 = 手机打开 app（级联+桥自启）+ 眼镜打开 app（自动连接），顺序无关**。
- **HONOR 90 Pro（48:C1:EE:8D:20:0A）旧 bond 清理尝试**：仅眼镜侧有 bond（手机侧只有 Glasses_1491）。眼镜厂商 BT 栈无 `cmd bluetooth_manager`（"No shell command implementation"）、无 root（/data/misc/bluedroid 拒读）、系统设置 UI 也不显示该设备（只有红魔）→ **shell/UI 均无法删除**。残余影响仅 app 设备选择器里多一行；自动连接用 prefs 存的地址，不会被误选。要彻底清需用户在 HONOR 90 Pro 本机 BT 设置里取消与 Glasses_1491 的配对。

**遗留（非阻塞）**：
- 当前 6020 LLM = `Qwen3-ASR-0.6B-Text`（T1.8 NO-GO 模型，用户决定**暂不换**）：简单问答/短故事可用，复杂指令退化（"讲个笑话"有时输出"一个故事，讲完。"）；日期类事实会幻觉（问"星期几"答"星期日"，实为周四）。要提升回答质量再单独换模型。
- ASR 对数字/量词偶发误转（"三十个字"→"30个字/3"），可接受。
- 阶段 2 全部代码（手机桥 + 眼镜客户端 + 两侧 SppProtocol + shell 钩子 + 文档）**未提交**，待用户确认后 commit。

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
| （阶段 2）Rokid 官方开发者通道确认（路径 B） | 用户已开开发者模式（C 可用）；询 open.rokid.com / `Glasses.BD@rokid.com`：开发者模式是否官方支持 + OTA 持久性 + 正式 SDK 时间表（§5.12） | 不阻塞（C 先行） |
| （阶段 2）眼镜↔手机传输 | v1 = BT SPP（Wi-Fi 直连被同 SSID 多 AP 分 VLAN 隔断，§5.12）；若用户调整路由器 VLAN 可升级 Wi-Fi TCP | 不阻塞 |

## 8. 回滚

- 手机：`.dev` 隔离 → 现有 `ai.liquid.lfmdemo*` 全程未动；dev pair 失败 = force-stop + 卸载 dev 包，现版 app 原样可用
- 宿主：所有产物可重建（§6）；仓库脚本幂等
- 模型：若 T1.8 不过 → 语言路由回退（中/英新模型，俄/日/韩 Qwen2.5）或整体回退（保留 Qwen2.5 GGUF 于 `~/phone-backup/`）

## 9. 以后替换 LLM 模型权重（复用现有眼镜/手机功能的操作手册）

**架构结论：眼镜功能与 LLM 权重完全解耦。** 眼镜桥（`GlassesBridgeService`）只依赖两个回环端点：`127.0.0.1:6010`（ASR）和 `127.0.0.1:6020/v1/completions`（LLM，OpenAI 兼容流式）。SPP 帧协议、VAD、TTS、眼镜客户端全部模型无关。**只要新权重继续以"同端口 6020 + 同名 alias + 兼容的 prompt 格式"提供服务，换权重后眼镜/手机功能零改动直接可用**（眼镜侧一个字节都不用动）。

**权重加载链路**（`runtime-companion` 包 `ai.liquid.lfmdemo.dev.runtime`）：
- `RuntimeService.startTextModel()` 起 `libqwenserver.so`（llama.cpp fork，在 APK nativeLibraryDir）
- 权重文件：`/data/data/ai.liquid.lfmdemo.dev.runtime/files/text/models/<文件名>.gguf`（当前 `qwen3_asr_0p6b_text-q4_k_m.gguf`）
- 关键启动参数：`--model <路径> --alias Qwen3-ASR-0.6B-Text-Q4_K_M --ctx-size 2048 --threads 4 --parallel 1 --n-gpu-layers 0 --no-cache-prompt --host 127.0.0.1 --port 6020 --no-webui`
- **权重不在 APK 里**，装 files 目录 → 重装 APK 不丢权重；只有卸载/清数据才丢（>1MB 模型按 T0.3 惯例留 sha256 清单）

**替换步骤**：
1. 准备 GGUF（架构须被 `libqwenserver.so` 支持——当前已验证 Qwen 系；非 Qwen 架构先验证 server 二进制，必要时重编）
2. 推入设备（runtime 包 debuggable，run-as 可用）：
   ```bash
   adb -s <phone> push new_model.gguf /data/local/tmp/
   adb -s <phone> shell run-as ai.liquid.lfmdemo.dev.runtime sh -c \
     'mkdir -p files/text/models && mv /data/local/tmp/new_model.gguf files/text/models/'
   ```
3. 代码改动——三处硬编码模型名 + 一处文件名：
   | 位置 | 常量 | 作用 |
   |---|---|---|
   | `runtime-companion/.../RuntimeService.java:25-28` | `TEXT_MODEL_FILE` / `TEXT_MODEL_ALIAS` | 服务端加载文件名 + 对外 alias |
   | `app/.../glasses/GlassesBridgeService.kt:98` | `llmModel` | 眼镜桥请求的 model 名 |
   | `app/assets/demo-ui/index.html:582` | `QWEN_TEXT_MODEL` | 手机 chat UI 请求的 model 名 |
   
   **最省事的换法：只改 `TEXT_MODEL_FILE`（文件名），alias 保持原名不变 → 代码零改动**。长期维护建议三处一起改成真实名字（保持服务端 alias 与两个客户端一致即可，客户端之间互相无关）。
4. 按新模型调 server 参数（`RuntimeService.startTextModel()`）：`--ctx-size`（历史长度↔内存）、`--threads`（CPU 核数）；`--no-cache-prompt` 是 b10083 针对现模型的 workaround，保留无害（实测零代价，§5.11）
5. 重建 + 装 runtime 包：`./gradlew :runtime-companion:assembleDebug` → push + `pm install -r` → force-stop 后重新打开主 app（bootstrap 自动拉起）
6. 验证：
   - `adb -s <phone> shell "curl -s http://127.0.0.1:6020/health"` → ok
   - 手机 chat UI 或眼镜问一句（如"今天星期几"）→ 回答合理
7. **prompt 格式兼容（唯一功能性差异）**：眼镜桥 `buildRawPrompt()` 与手机 chat 用"用户：…\n助手回答：…"raw 对话格式（对齐 Qwen3-ASR-0.6B spoken_multiturn_90h 微调，材料见 `scripts/qwen3_asr_llm_swap/PROMPT_ALIGNMENT.md`）。换通用 instruct 模型时其 chat template 不同：
   - 快路径：raw prompt 原样用（可用但质量退化，T1.8 有量化数据）
   - 正路：改 `buildRawPrompt()`（~20 行单函数）对齐新模型模板，或改走 `/v1/chat/completions` + server 端 `--jinja`
8. **眼镜侧零改动**：SPP 协议/帧/客户端 app 都不感知模型；换完后打开眼镜 app 自动连接即用。

**资源参考**：0.6B Q4 ≈ PSS 1.1 GB；Qwen2.5-1.5B Q4 基线 ≈ 2 GB。手机（红魔）内存宽裕，上限看 PSS+ASR+TTS 之和是否触发 swap；ctx 2048 是当前历史长度与内存的折中。
