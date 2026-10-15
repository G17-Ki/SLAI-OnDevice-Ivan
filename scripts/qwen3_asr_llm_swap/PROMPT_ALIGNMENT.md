# Prompt 对齐说明（部署用）

来源: /mnt/afs/users/fush/projects/AudioTrain
- `src/audio_agent/models/dialogue_prompt.py`（本目录已原样复制）
- `src/audio_agent/data/collator.py`（本目录已原样复制）
- `src/audio_agent/data/posttraining.py` `compile_examples()`（prompt 调用链，见下）
- prompt_version: `teacher_explicit_roles_v1`

模型包内的 `chat_template.json` 只用于 ASR 分支（音频输入）；文本回复分支**不走 chat template**，走下面的自定义 segment 拼接。

## 1. 文本回复路径（对话 / 翻译 / 问答）

生成第 i 个 assistant 回合时，prompt 由 `messages[:i]`（不含当前目标）构造：

```python
# posttraining.py compile_examples():
segments = prompt_segments(record["messages"][:i], record.get("tools"))
target   = assistant_action(message)          # 即该 assistant 回合的 content

# collator.py SharedDecoderCollator.text():
prefix_ids = [t for seg in segments for t in tokenizer.encode(seg, add_special_tokens=False)]
full_ids   = prefix_ids + tokenizer.encode(target, add_special_tokens=False) + [EOS]
# labels 只在 target+EOS 段有值（prefix 全 -100）
```

**对齐关键点：每个 segment 单独 `encode(add_special_tokens=False)` 再拼接 token 流。**
这不等于把所有 segment 拼成一个大字符串后整体 encode——BPE 在段边界的行为不同。
部署时若要严格对齐训练分布，必须同样按 segment 分块编码后拼接（llama.cpp 的 tokenizer
可对每段单独 encode 再 append ids）。若只能喂整段文本，段边界处可能有少量 token 差异。

- 历史回合之间**不插入 EOS**（prompt_segments 注释明确 "contains no history EOS"）。
- 生成目标 = target 编码 + 一个 EOS token（tokenizer 的 `<|endoftext|>`，id 以 tokenizer_config 为准）。
- 默认 system prompt：`你是一个有帮助的中文语音助手。请根据对话内容直接、简洁、准确地回答用户的问题。`
  （以 dialogue_prompt.py 的 SYSTEM 常量为准，注意是单行原文，不要改动任何标点。）

### segment 具体格式（dialogue_prompt.py）

```
base_prompt(system):
"下面是一段用户和助手之间的真实对话。\n你现在扮演“助手”。\n请站在助手的角色回答最后一个用户问题。\n注意区分用户说的“我”和对助手说的“你”。\n只回答问题，不要继续编写新的用户对话。\n\n系统：{system}\n"

user 回合:      "\n用户：{content}\n助手回答："
assistant 回合: {content}
  （有 tool_calls 时: json.dumps({"tool_calls":[{"name":...,"arguments":{...}}]},
    ensure_ascii=False, sort_keys=True, separators=(",", ":"))）
tool 回合:      "\n工具结果（{name}）：{content}\n助手回答："

有 tools 参数时，system 末尾追加:
"需要调用工具时，只输出 JSON，格式为 {\"tool_calls\":[{\"name\":工具名,\"arguments\":参数对象}]}。收到工具结果后再简洁播报，失败时说明失败。可用工具：" + json.dumps(tools, ensure_ascii=False, sort_keys=True)
```

## 2. ASR 路径（音频输入）

```python
# collator.py SharedDecoderCollator.audio():
prompt = processor.apply_chat_template(
    [{"role": "system", "content": ""},
     {"role": "user", "content": [{"type": "audio", "audio": "<音频占位>"}]}],
    add_generation_prompt=True, tokenize=False)
# 再用 processor(text, audio) 展开音频占位 -> input_features / feature_attention_mask
```

- 走模型包内 `chat_template.json`（已随模型包提供）。
- 音频要求：16 kHz 单声道 float32（load_audio 做了均值降混 + soxr_hq 重采样，只读不改源文件）。
- **ASR 目标格式**：`language {Chinese|English}<asr_text>{转写文本}` + EOS
  （`source_language` 数据里取值就是 `Chinese` / `English` 全拼。）

## 3. 部署检查清单

1. 文本分支：用 dialogue_prompt.py 原样构造 segments，按段分块 tokenize 拼接。
2. 生成后处理：目标以 EOS 结束；spoken 版 serving 用了 `serving_role_and_repeat_stop_v1`
   停止策略（防重复），可参考其逻辑但非 prompt 本身。
3. 两个模型（90h corpus / spoken 多轮）共用同一套 prompt 构造，无差异。
