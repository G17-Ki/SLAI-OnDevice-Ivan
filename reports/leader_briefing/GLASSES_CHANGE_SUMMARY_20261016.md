# 两日改动总结（2026-10-15 ~ 10-16）· 转交向

> 配套文档：`LLM_SWAP_AND_GLASSES_PLAN.md`（方案+10-15 进度）、`model_candidate_followup/MODEL_REGISTRY.md`（模型唯一事实来源）。
> 本文回答：**这两天真改了哪些代码、为什么、验证到什么程度、接手者要注意什么。**

## 0. 当前系统一句话

Rokid 眼镜 ⇄（SPP 蓝牙）⇄ 手机 bridge ⇄（本地 HTTP 127.0.0.1:6010/6020）⇄ runtime（Qwen3-ASR-0.6B 语音识别 + 自训 Qwen3-0.6B 文本 LLM + Sherpa-ONNX TTS），三任务模式（中文 QA / 中翻英 / 英翻中），手机 app 作配置/调试 UI。

## 1. 提交清单（远程 `origin/t18-qwen3-asr-0p6b-tts-parity`）

| 提交 | 内容 |
|---|---|
| `0fb10ac` | T1.8：手机侧 Qwen3-ASR-0.6B-Text LLM 替换验证（生产 TTS 对拍） |
| `e015fe9` | Rokid SPP 网关：手机 bridge + 眼镜 client，E2E 验收通过 |
| `d8d11d0` | 眼镜桥自愈：BT 栈复位后自动重建 SPP accept（vendor 栈不会唤醒被阻塞的 accept） |
| `2e4e9a2` | 三任务模式 QA/zh2en/en2zh + 统一 V1 prompt |
| `506d09c` | 本日批次：TTS 中文归一化、翻译 3-exemplar、模型徽章、登记簿同步（11 文件） |

## 2. 代码改动（按模块）

### 手机 bridge（`android-app/app/.../glasses/GlassesBridgeService.kt`，核心文件）
- **三任务模式** `BridgeMode`：QA / ZH2EN / EN2ZH，各带 spec prompt + in-context exemplar（翻译各 3 条：陈述句/问句/自我介绍），模式切换触发 LLM 重启并切换 prompt 历史。
- **TTS 回显过滤**：纯标点/超短句不合成；`isStorableTurn` 拦"请说…内容"这类会诱发复读的存史对。
- **TTS 中文归一化钩子**：`speakClause` 对 zh 文本过 `ChineseTtsNormalizer`（见下）。
- **清空历史**协议（`clear_history`）：眼镜端按钮触发，清 prompt 历史 + 重启 LLM（顺带清 KV 复读残留）。

### 新文件 `glasses/ChineseTtsNormalizer.kt`
- 背景：`matcha_zh_baker` 的 lexicon（66375 条）**无任何 ASCII 数字/字母条目**，"1加1等于2"被读成"加等于"、"ABCD"整段静音。
- 规则：≤4 位整数数值读（100→一百）、≥5 位逐位、`2026年`→"二零二六年"、小数点"点"、"百分之"、负号"负"、千分位逗号；字母→常见词表（AI→诶爱、WiFi→歪法、OK、USB、CPU、ID 等 19 词）否则逐字母；字母粘连数字（RTX4090/64K）按型号逐位读，小数（0.5Km）不误伤。
- 仅 zh 模式生效；en 模式走 `supertonic3_en_v4_s5`，不受影响。

### 眼镜 client（`android-app/glasses-client/`）
- `GlassesClientService.java`：`writeMicFrame` NPE 修复（SPP 断连瞬间麦克风帧仍会投递）。
- `MainActivity.java` + 手机 UI：模式选择、"清空历史"按钮、模型徽章入口。

### 手机 app UI（`NativeAudioBridge.java` + `assets/demo-ui/index.html`）
- **模型徽章**：`NativeAudio.getLlmModelInfo()` 读 dev app `files/model-info.json`（name/checkpoint/alias/quant/sha256/deployed_at），header 显示 `LLM: <名> · <sha8> · <MM-DD>`。解决"手机上跑的是哪个权重"的肉眼可辨问题。

### 脚本与登记
- 新 `scripts/qwen3_asr_llm_swap/extract_text_decoder_v2.py`：v1 只收全 BF16 输入，v2 支持 FP32 源 checkpoint 自动降精（mtfinal 就是 v2 提的）。
- 新 `reports/.../MODEL_REGISTRY.md`：模型唯一事实来源（库存/部署/换权重手册/行为指纹/维护规则）。
- `candidate_manifest.json`：sha 跟当前部署走，`previous_sha256` 留上一代；`context_tokens` 1024→8192 修正。

## 3. 模型变更链（两天内两次换权重）

```
90h(10-15部署,有复读) → all748(10-16上午) → mtfinal(10-16 14:27, 当前)
```
- 当前：**mtfinal** `197774bd…`（multiturn 64k final，全 BF16 源），验证 = 磁盘 sha + 手机配置行为指纹 + model-info sidecar 三重。
- all748、mt648、90h 均为现成回退源（`~/models/lfm-gguf/<checkpoint>/`），换权重 5 分钟（`MODEL_REGISTRY.md` §4 手册）。
- 重要经验：**行为指纹依赖推理配置**（temp=0 下 `--no-cache-prompt`+batch 128/64 与宿主默认配置输出不同句），手机内判别模型必须用"手机同款配置"复现，详见登记簿 §6 指纹表。

## 4. 验证状态（接手者基线）

已验证 ✅：
- E2E 语音问答/翻译（10-16 下午，手机+眼镜实机）
- TTS 数字归一化（用户实听确认）
- 翻译 3-exemplar：EN2ZH/ZH2EN 各 8/8（手机配置直探）
- 模型徽章显示、清空历史、BT 断连自愈

待复测 ⏳（截至本文写作，刚部署未听）：
- 字母发音（"ABCD后面字母是什么"应读"诶比西迪…伊"）

## 5. 已知问题与待办（按优先级）

1. **0.6B 长上下文复读/身份坍缩**：8+ 轮后进入复读吸引子（b10083 槽位 KV 累积，与权重无关）；缓解方案"每 N 轮自动重启 LLM"已设计未实现。现有"清空历史"按钮是人工版兜底。
2. **身份问法敏感**："你是谁"→河套学院 ✅，"你叫什么名字"→"智能助手" ❌（prompt 确认已拼入，属 0.6B 泛化不足）→ 训练侧 SFT 补 paraphrase 样本。
3. **翻译 exemplar 是桥侧 workaround**：3 exemplar 实测 8/8，但未见过的问句仍可能走"回答"路径；根治 = 训练侧补"问句也翻译"数据。
4. **AutoLaunch 兜底未修**：`startRuntimeCompanion()` catch SecurityException 路径（AssistServer 周期性杀 app 后的自愈），诊断完未写代码。
5. 环境债：三个 APK 的 debug 构建产物都不在仓库（`android-app/releases/` 仅存 10 月前旧生产 APK）；手机 logcat 锁死；AssistServer 会周期性杀 app（拉活靠手动开两个 app 或 AutoLaunch 修复）。
6. `glasses352` 权重不建议部署（step 352 欠训，记忆召回不稳）。

## 6. 转交清单

| 资产 | 位置 |
|---|---|
| 代码 | `github.com/G17-Ki/SLAI-OnDevice-Ivan` 分支 `t18-qwen3-asr-0p6b-tts-parity` |
| 手机 APK（bridge+UI，包名 `ai.liquid.lfmdemo.dev`） | `android-app/app/build/outputs/apk/debug/app-debug.apk`（构建命令见下） |
| runtime APK（包名 `ai.liquid.lfmdemo.dev.runtime`） | `android-app/runtime-companion/build/outputs/apk/debug/app-debug.apk`（**仓库无备份，需重构建**） |
| 眼镜 APK（`ai.liquid.lfm.glasses`） | `android-app/glasses-client/build/outputs/apk/debug/app-debug.apk` |
| 模型权重 | `~/models/lfm-gguf/`（5 个 checkpoint，登记簿 §2 有全 sha） |
| 手机端已部署态 | 模型 mtfinal + model-info.json + 全部修复（10-16 16:01 版 APK） |
| 构建 | `JAVA_HOME=~/.cache/lmf-demo/tools/jdk17 ANDROID_HOME=~/.cache/lmf-demo/android-sdk SHERPA_AAR_DIR=~/.cache/lmf-demo/sherpa-aar; cd android-app && ./gradlew :app:assembleDebug :runtime-companion:assembleDebug :glasses-client:assembleDebug` |
| 连接 | adb over network：`adb -H 172.18.0.1 -P 5038`，手机序列号 `912615210575` |
| 文档 | 本文件 + `LLM_SWAP_AND_GLASSES_PLAN.md` + `MODEL_REGISTRY.md` + `LEADER_BRIEFING.md` |
