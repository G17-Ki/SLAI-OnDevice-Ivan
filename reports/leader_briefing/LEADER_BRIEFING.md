# 端侧离线多语言语音助手 — 项目真实状态与级联替换指南

> 汇报对象：项目负责人
> 依据：本仓库当前源码 + `reports/` 内手机实测证据（Nubia NX809J / 高通 SM8850 / Android 16 / arm64）
> 说明：本文区分三类证据——**手机物理实测**、**宿主测试**、**AI Hub 同芯片组参考**。三者不可互相替代。

---

## 1. 项目一句话

在一台零售高通手机上跑通了一个**完全离线**（飞行模式可用）的多语言语音助手闭环：

```
用户说话 → 手机本地 VAD 判停 → 本地 ASR（NPU/CPU）→ 本地 LLM 流式生成 → 本地 TTS 流式合成 → 扬声器
```

不依赖网络、不依赖 ADB、推理进程全部归应用自有 UID。项目核心价值是**"证明可行性 + 可复现的工程底座"**，不是一个可直接发布的成品。

---

## 2. 最真实的项目状态

### 2.1 已物理验证（真机实测，非仿真）

| 项 | 证据位置 |
|---|---|
| 双 APK 应用自有架构：推理进程归 app UID，非 shell | `reports/progress_20260729/ASR_QNN_AUDIT.md` |
| EN/ZH/JA/KO ASR 走高通 NPU（QNN 2.40 / HTP v81），精确转录 + 语言标签匹配 | 同上 |
| RU ASR（Zipformer INT8, CPU）精确转录 | 同上 |
| Qwen2.5-1.5B 28 题手机扫描：24/28（人工修正后），中位首 token 304 ms / P95 475 ms | `reports/progress_20260729/LLM_AUDIT.md` |
| 五语言（EN/ZH/RU/JA/KO）完整对话轮：飞行模式、冷重启后仅开主应用、5/5 通过 | `reports/android_phone/full_turns/POST_SWEEP_FULL_TURN_ACCEPTANCE_20260730.md` |
| TTS：Matcha 中文 118 ms end-to-speaker、RTF≈0.033；Supertonic 俄/日/韩首块 395–577 ms | `reports/android_phone/streaming_response/tts_research_20260726/DEVICE_RESULTS.md` |
| 宿主 ADB 关闭 70 s 四进程存活、零 forward（adbd cgroup 缺陷已消除） | `reports/android_phone/full_turns/final_apk_a24e_33bf_cold_reboot_provenance_20260729.md` |
| 流式播放队列：7 用例（自动路由/手动 pin/多块有序/播放前后取消）物理通过，零 underrun | README "What is physically verified" |

### 2.2 必须对 leader 说清楚的证据边界

1. **当前源码 ≠ 已验收安装对**。已通过验收的是手机上安装的一对 APK（主 `a24e203a...` / runtime `33bf0d6d...`）。当前源码比它新，且**已关闭 LFM 运行时**（`NativeAudioBridge.java:40` `LFM_RUNTIME_ENABLED=false`，`MainActivity` 中无任何代码启动 `LfmService`）。README 的 "Current phone stack" 表描述的是手机上那对旧 APK。当前源码构建若要继承其时延/离线声明，必须重跑一遍设备验收。
2. **AI Hub QRD 数字是同芯片组参考**，不是零售手机时延，更不能外推到眼镜（AR1）。
3. **物理拔 USB 线**（真正的 unplugged）仍是操作员门槛，未完成。现有证据是"宿主 ADB 关闭 70 s 进程存活"，不是"拔线后对话"。
4. 法/意/西/德的 eSpeak 语音标签**不等于**这些语言有健壮的端到端路线。
5. 宿主测试不能证明手机时延、HTP 执行、声学质量、热稳定、拔线存活。

### 2.3 未关闭的门槛

| # | 门槛 | 状态 |
|---|---|---|
| 1 | 物理拔线 + 手机 UI 完整轮（EN/ZH/RU/JA/KO） | 未做（操作员门） |
| 2 | 长时不间断电池/热浸泡 | 未做 |
| 3 | 真实麦克风：噪声/口音/远距/回声；energy VAD 非神经 VAD，barge-in 默认关闭 | 部分（有 fixture，无开放环境） |
| 4 | 日/韩语可听质量人工评审（JA/KO 有精确转录但无人耳确认） | 未做 |
| 5 | 当前源码构建重新验收 | 未做 |

### 2.4 模型质量诚实声明

- Qwen2.5-1.5B 在小扫描（28 题）中 FR/JA/KO 存在事实错误（如 Ottawa 答成 Vancouver）→ **不能作为"通用多语言模型"宣传**，中文/俄语小集合通过。
- SenseVoice 中文在 3 片段诊断上 CER 33%（小样本冒烟，非发表级基准）。
- Matcha 中文音色继承**非商业训练语料限制**，再分发/商用前必须审 TTS 报告。
- ASR 固定 10 秒图（短音频零填充，长音频切 9.5 s 段）；单句 15 s 上限（手动录音 30 s）。
- 全部神经推理在 CPU（除 SenseVoice EN/ZH/JA/KO 在 NPU）；Qwen 进程 PSS 约 1.95 GB。

---

## 3. 端到端流程：每一步的组件与逻辑

### 0) 进程与安全架构（先讲清"东西跑在哪"）

| 组件 | 所在 APK / UID | 说明 |
|---|---|---|
| UI（WebView）+ 编排 JS | 主 `ai.liquid.lfmdemo`（开发版 `.dev`） | 本地 UI server `127.0.0.1:8765` 提供 `demo-ui/index.html`，**JS 是整条流水线的编排器** |
| 麦克风 / VAD / TTS 播放 | 主 APK，`NativeAudioBridge`（Java）+ `AdaptiveEnergyVad` | 原生桥，录音线程逐帧喂 VAD |
| LFM 语音模型 server | 主 APK，`LfmService` → `liblfmserver.so` `:8080` | **当前源码已停用**（开关为 false，无启动代码）；保留为"英文表现力路线"后备 |
| ASR server | runtime `ai.liquid.lfmdemo.runtime`，`AsrHttpServer` `:6010` | 启动 sherpa-onnx 持久 runner 子进程（NPU 或 CPU） |
| LLM server | runtime APK，llama.cpp b10083 `:6020` | `RuntimeService` 启动并**监控/自动重启** Qwen 子进程 |

关键设计：
- 两 APK **同签名**；主应用通过 signature 级权限 `*.permission.START_RUNTIME` + 显式 intent 拉起 runtime 前台服务；另有 ContentProvider / Activity 两级兜底启动路径（应对 OEM 拦截）。
- 所有 HTTP **只绑 127.0.0.1**，不需要 ADB forward。
- 模型权重在**包私有存储**，全部 SHA-256 pin；安装器 fail-closed（字节数+哈希+真实执行验证，文件在 ≠ 就绪）。
- 遥测逐轮写 `streaming-telemetry.jsonl`（时延分段、VAD 事件、TTS 块、路由决策全记录）。

### 1) 收音 + VAD（主 APK，纯 CPU，无模型）

- `AudioRecord` 16 kHz 单声道 PCM16，320 样本（20 ms）帧。
- `AdaptiveEnergyVad`（`AdaptiveEnergyVad.java`）：确定性能量状态机 `IDLE → LISTENING → SPEECH → HANGOVER → ENDPOINT`。
- 关键参数（`vad_mode/CORE_RESULT.md`）：

| 参数 | 值 | 目的 |
|---|---:|---|
| 初始校准 | 300 ms | 先学房间底噪 |
| 激活连续帧 | 60 ms | 拒绝单击/尖峰 |
| 最小有声量 | 160 ms | 拒绝短促非语音 |
| pre-roll | 240 ms | 保住句首辅音 |
| **尾端静音判停** | **420 ms** | 刻意压在 300–500 ms 延迟预算内——这是"说完"到"开始处理"的固定代价 |
| 单句上限 | 15 s | 有界捕获 |
| 激活/释放阈值 | max(-48 dBFS, 噪声+10 dB) / max(-54 dBFS, 噪声+6 dB) | 动态门限+滞回 |

- 噪声底只在非脉冲、非活跃帧学习；确认语音后冻结；**助手播放期间整个 VAD 屏蔽**（barge-in=DISABLED，能量门无法可靠区分用户与合成音）。
- 设计逻辑：宁可多等 420 ms，不可截断尾音或误触发；事件带样本级时间戳，可精确复现"末次语音→端点"延迟。

### 2) ASR（runtime APK，`127.0.0.1:6010`）

- 契约：`POST /asr?language=en|zh|ja|ko|ru|auto|qnn-auto`，输入必须是规范 16 kHz 单声道 PCM16 WAV（≤10 s），**零填充到固定 10 秒图**（320,044 字节）后调 QNN。
- 路由（`AsrHttpServer.java`）：
  - `en/zh/ja/ko` → **SenseVoiceSmall INT8 @ QNN 2.40 / HTP v81（NPU）**，实测推理 RTF≈0.009，返回文本+语言标签（`<|ja|>` 等）；
  - `ru` → **Zipformer INT8（CPU）**；
  - 代码中另存 `parakeet_v3_cpu` 路线（备用）。
- 持久 runner：ASR 引擎以常驻子进程跑（stdin-json 协议），切换语言路线前先 quit 旧进程再分配新路线，避免双份内存。
- **执行感知健康**：`artifacts_ready`（文件在盘）/ `execution_verified`（当前进程出过非空转录）/ `ready`（两者都满足）+ `runner_owned_by_service`（UID 证据）+ QNN 日志须含 `Initializing HtpProvider`。**文件存在永远不算 NPU 证明。**
- 串行代价（明确记录）：对话模式先完成 ASR 再启动 LLM，音频会被处理两遍，LLM 不能与 ASR 并行启动——这是当前 ~2 s 预算里可优化的部分。

### 3) LLM（runtime APK，`127.0.0.1:6020`，CPU）

- 官方 llama.cpp `b10083` server 加载 **Qwen2.5-1.5B-Instruct Q4_K_M**。
- 参数：ctx 2048 / 1 slot / 4 线程 / batch 128 / micro-batch 64 / 0 GPU 层。
- 接口：OpenAI 兼容 `POST /v1/chat/completions`（**流式**）+ `GET /health` + `GET /v1/models`。
- 能力探测：只对 Qwen3 模型发 thinking 控制字段，Qwen2.5 请求刻意不带（防止模板不兼容）。
- 手机实测（28 题短问套件）：首 token 304/475 ms（中位/P95）；首句可读文本 672/1,253 ms；整句完成 707/1,601 ms；prompt/decode 速率 84.4/33.2 tok/s；进程 PSS ≈1.95 GB。
- `RuntimeService` 负责启动、监控、崩溃重启该子进程，并持有 active-client 非引用计数 wakelock。

### 4) TTS（主 APK，语言感知路由，流式）

- 候选**不在 APK 里**：`CandidateRegistry`（`CandidateRegistry.kt`）从外部 bundle 的 `candidates.json` 读取——换音色/加候选不重编译 APK（刻意设计）。
- 自动路由：`zh → matcha_zh_baker`；`ru/ja/ko → supertonic3 s5`（五步生成配置）；其余 → **eSpeak 离线兜底**（Android TTS API）。手动 pin 为实验室覆盖，pin 的候选文件不全必须显式报错，不许静默落到别的模型。
- 流式链路：LLM 吐出**第一句可读分句**（标点/长度边界 flush）即提交 TTS → 每块过 `PcmGate` 实时安全门（非有限 PCM / 近静音 / 削波 / DC 偏移 → 丢弃该块而非播放噪声）→ 进入有序播放队列（支持播放前/后取消、多块排序）。
- 空闲预载：按当前选择 locale 预热线图（独立 token，绝不推进在途轮次）。
- 手机实测：Matcha 中文 end-to-speaker **118 ms**（RTF 0.033，零掉块）；Supertonic 首块 RU 496 / JA 577 / KO 395 ms（RTF 0.23–0.28，零掉块）。

### 5) 播放与半双工

- 播放期间 VAD 屏蔽 + 噪声底冻结（防自激）；规划中的播放 cushion 140 ms（`LANGUAGE_ROUTING.md` 规划算式用值）。
- 实测完整轮：27 块 PCM、零 underflow、零质量告警。

### 6) 时延汇总（手机实测，软件时间戳代理值）

| 阶段 | 中位 | P95 | 备注 |
|---|---:|---:|---|
| VAD 尾端判停 | +420 ms | — | 说完后固定代价 |
| ASR（NPU 路线） | ~百 ms 级 | — | RTF 0.009；CPU 俄语更慢 |
| ASR 完 → 首句文本 | 672 ms | 1,253 ms | LLM 流式 |
| 首句文本 → 出声 | 90–130 ms（中）/ 395–577 ms（俄日韩） | — | TTS 首块 |
| 请求 → 首文本 | 812 ms | — | 离线轮次代理值 |
| **说完 → 出声（总）** | **≈2.16 s** | — | 验收目标：中位 ≤2.0 s / P95 ≤3.0 s，实测贴线 |

完整一整轮（含把回答播完）：KO 3.1 s / JA 4.2 s / EN 4.5 s / RU 4.9 s / ZH 14.5 s（中文那条回答本身长）。
瓶颈排序：VAD 420 ms 判停 > LLM 首句（CPU）> 俄日韩 TTS 首块 > NPU ASR。

---

## 4. 用另一个级联方案替换某个模型：最小最保险的改动方式

### 4.1 核心原则：在"契约缝"上换，不在流程里换

这套架构最大的资产是：**四个模型角色之间只有三条稳定的回环 HTTP 契约**：

| 契约 | 端口 | 协议 |
|---|---|---|
| ASR | 6010 | `POST /asr` → `candidates[i].result.text`（规范 16 kHz WAV in） |
| LLM | 6020 | OpenAI 兼容 `/v1/chat/completions`（流式）+ `/health` + `/v1/models` |
| LFM | 8080 | 同 llama.cpp 风格（当前停用） |

**只要替换后的新方案在同一端口、同一套协议后面，主应用（UI / VAD / 编排 JS / TTS / 播放 / 遥测）一行代码都不用改。** 替换工作的边界被压缩到 runtime companion 内部。

### 4.2 放置位置：必须进 runtime companion APK

- **不能**在 `adb shell` 里跑（adbd cgroup 生命周期缺陷，已被架构淘汰）；
- **不能**走云端（离线是硬需求）；
- companion 已经是现成的"模型运行时"边界：子进程启动/监控/重启、模型私有存储、健康端点、UID 归属验证都在那里。
- 若新方案本身是多模型级联（例如"小模型意图路由 + 大模型生成"），实现为 companion 内的新子进程（或 llama.cpp 多模型/多 slot），`RuntimeService` 按现有模式纳入生命周期管理，对外仍只暴露一个 :6020。

### 4.3 最小 diff 食谱（以替换 LLM 为例，难度最低的一段）

| # | 改什么 | 文件 |
|---|---|---|
| 1 | 新模型 pin（repo/revision/filename/字节数/SHA-256） | `reports/android_phone/streaming_response/model_candidate_followup/candidate_manifest.json` |
| 2 | server 启动命令（模型路径、别名、ctx、线程、batch）；保持 `/health` `/v1/models` 语义 | `runtime-companion/.../RuntimeService.java` |
| 3 | 模型文件校验 + 身份断言（`/v1/models` 返回 id）+ 真实 fixture | `scripts/install_app_owned_runtime.sh` |
| 4 | 源一致性断言（新启动参数、无禁用 flag） | `scripts/test_runtime_companion_source.py` |
| 5 | 主应用 | **零改动** |

调优与预算检查：ctx/线程/batch；首 token 必须守住 672 ms 量级（否则吃掉 2 s 总预算）；响应长度上限在主应用侧，注意同步；PSS 预算（现 ~2 GB，新模型要装得下）。

### 4.4 分段替换难度一览

| 要换的段 | 难度 | 关键点 |
|---|---|---|
| **LLM** | 低（小时级） | 任意 GGUF + OpenAI 兼容 API；只动 manifest pin + 身份检查 + fixture |
| **TTS** | 低（小时级） | 天生热插拔：新 bundle + `candidates.json`，`install_main_app_tts_bundle.sh` 校验安装；引擎须在 `SherpaTtsEngine` 支持集内（matcha/supertonic/piper/kokoro 等 sherpa-onnx 引擎） |
| **ASR（CPU 路线）** | 中（天级） | ONNX 导出 + sherpa-onnx 支持 + 固定 10 s 零填充输入契约 + `AsrHttpServer` 语言路由 + 安装器 fixture 精确转录 |
| **ASR（NPU 路线）** | 高（周级） | QNN context 构建（在 AI Hub 做，固定形状图）、HTP v81 skel、manifest `libcdsprpc.so` 声明、执行感知验证——仓库已完整走过一次（SenseVoice），工具链可复用 |
| **LFM** | 高（周级） | `liblfmserver.so`+`libliquid-audio.so` 是 Liquid Audio 架构专属 fork，换架构=重写 native server；换 Liquid 家族=改 4 个文件名。**但当前源码 LFM 已停用，通常无需动** |

### 4.5 验证阶梯（每级不过，不上下一级）

0. **宿主门**：`test_runtime_companion_source.py` + 协议/单元门（本机无 JDK 时只能到此为止）。
1. **构建 dev pair**：`ai.liquid.lfmdemo.dev` / `.dev.runtime`（本次已把 14 处包名/契约改好并全门通过）——**现有生产 app 零影响**，模型传输自动落到 `.dev` 包自己的私有存储。
2. **dev 包安装 + 执行感知健康**：安装器跑真实 fixture（精确转录+语言标签+QNN provider），不是看文件在不在。
3. **飞行模式完整对话轮**：五语言回归 + 新方案目标语言；UI 绿灯不算数，必须有真实完整轮。
4. **遥测对照**：时延分段（首文本/首 PCM/underrun）、PSS、温度与旧方案逐项对比。
5. **冷重启 + OEM 策略**（Nubia Apps AI-control 对新包名重新配置 + `dumpsys AutoLaunch` 验证）。
6. **回迁生产包**：全部通过后，才把同样改动应用到 `ai.liquid.lfmdemo*` 生产包；保留旧 APK 哈希做回滚锚点。

### 4.6 保险清单（明确不做的事）

- **不动** VAD、ASR 输入契约（16 kHz / 10 s 零填充）、TTS bundle 结构、端口、签名权限——除非新方案本身改变了输入形态。
- **不**在生产包上直接试错：用 `.dev` 隔离；签名不一致的安装会**失败**而非损坏，这是天然保险。
- **不**接受"文件在 = 就绪"：沿用 `execution_verified` 语义。
- **先量时延再谈质量**：流水线是串行的，新段每慢 100 ms 直接吃掉 2 s 预算。
- **许可同级审查**：新模型 license 与 Matcha 语料限制一样，进 `THIRD_PARTY_NOTICES` 与发布说明。
- 双 pair **不能同时运行**（固定回环端口 8765/6010/6020/8080）：切换时 force-stop 旧 pair（不是卸载）。

---

## 5. 风险、许可与存储

| 项 | 说明 |
|---|---|
| 许可 | 一方代码 Apache-2.0；`vendor/`、native 库、模型权重、语音 bundle、Qualcomm 运行时各自条款；**Matcha 中文音色带非商业语料限制**，再分发/商用前必须审 `reports/.../DEVICE_RESULTS.md` 与 TTS 报告 |
| 存储 | 每套 ~3 GiB（LFM 1 G + Qwen 1.1 G + ASR/QNN + TTS 1.4 G）；`.dev` 隔离开发再 +3 GiB |
| 加固 | 当前是 debuggable 研究构建，loopback + signature 权限保护，**未做生产加固** |
| 设备绑定 | QNN context 与 HTP v81 负载绑定 SM8850；安装器拒绝非 arm64/非 SM8850 目标 |
| 证据继承 | 当前源码构建未继承任何已验收时延/离线声明，需重跑设备验收 |

---

*附：开发隔离说明——本 worktree 已将开发版改为独立包 `ai.liquid.lfmdemo.dev` / `ai.liquid.lfmdemo.dev.runtime`（14 处：2 个 build.gradle、2 个 Manifest、4 个 Java 契约文件、3 个安装脚本、5 个运维脚本、1 个源一致性测试），全部宿主门通过。手机上现有 app 的 APK、数据、模型、OEM 策略不受开发版安装影响。*
