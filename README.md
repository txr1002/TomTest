# TomTest

统一的 LLM 调用模块，集成了 BenchRAG 和 Crux 的 LLM 客户端功能。

## 功能特性

- **OpenAI 兼容 API**：支持任何兼容 OpenAI 格式的 LLM 服务
- **Mock 响应**：支持测试模式，无需真实 API 调用
- **JSON 输出**：支持结构化 JSON 响应
- **结构化输出**：支持 Pydantic 模型验证
- **批量处理**：支持并行批量调用和自动重试
- **使用统计**：线程安全的 token 使用量和延迟统计

## 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install openai pydantic tqdm
```

## 使用 vLLM serve 本地部署

### 安装 vLLM

```bash
pip install vllm
```

### 启动 vLLM 服务

```bash
# 基础用法
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000

# 指定模型路径
vllm serve /path/to/model \
    --port 8000 \ # 指定端口号
    --tensor-parallel-size 1 \ # 设置模型张量并行度
    --gpu-memory-utilization 0.9 # 设置GPU内存利用率   
    --max-model-len 4096 # 设置最大序列长度 

### 连接 vLLM 服务

```python
from llm import LLMClient

# 连接本地 vLLM 服务
client = LLMClient(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    api_key="not-needed",  # vLLM 不验证密钥
    api_url="http://localhost:8000/v1",
)

# 使用方式与 OpenAI API 完全相同
generations, usage = client.generate(
    prompt="你好",
    instruction="你是一个友好的助手",
)
print(generations[0].text)
```

### vLLM 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 绑定地址 | 0.0.0.0 |
| `--port` | 端口号 | 8000 |
| `--tensor-parallel-size` | GPU 张量并行数 | 1 |
| `--gpu-memory-utilization` | GPU 内存利用率 (0-1) | 0.9 |
| `--max-model-len` | 最大序列长度 | 模型默认 |
| `--api-key` | API 密钥 | 空（无需验证） |

## 快速开始

### 1. LLM 客户端

#### 基础使用

```python
from llm import LLMClient

# 创建客户端
client = LLMClient(
    model_name="gpt-4",
    api_key="your-api-key",
    api_url="https://api.openai.com/v1",
)

# 从配置字典创建
config = {
    "model_name": "gpt-4",
    "api_key": "your-api-key",
    "api_url": "https://api.openai.com/v1",
    "temperature": 0.7,
    "max_tokens": 4096,
}
client = LLMClient.from_config(config)

# 生成文本
prompts, instruction = "你好", "你是一个友好的助手"
generations, usage = client.generate(prompt, instruction)

for gen in generations:
    print(f"输出: {gen.text}")
    if gen.reasoning:
        print(f"推理: {gen.reasoning}")

print(f"使用情况: {usage}")
```

#### Mock 模式（用于测试）

```python
client = LLMClient(
    model_name="gpt-4",
    api_key="test",
    api_url="https://api.openai.com/v1",
    mock_llm=True,  # 启用 mock 模式
)

generations, usage = client.generate("测试", "你是一个助手")
# 返回预设的 mock 响应
```

### 2. JSON 输出

```python
# 单次 JSON 调用
result = client.call_json(
    prompt='{"name": "请返回一个 JSON 对象"}',
    instruction="你是一个 JSON 生成助手",
)
print(result)  # {'status': 'sufficient', ...}

# 批量 JSON 调用
results = client.batch_call_json(
    prompts=["问题1", "问题2"],
    instructions=["指令1", "指令2"],
    max_retry=5,
    max_workers=4,
)
```

### 3. 结构化输出

```python
from pydantic import BaseModel

class Answer(BaseModel):
    summary: str
    confidence: float
    keywords: list[str]

# 生成结构化输出
results, usage = client.generate_structured(
    prompt="总结这篇文章",
    instruction="你是一个文章总结助手",
    schema_model=Answer,
)

for result in results:
    print(f"摘要: {result.summary}")
    print(f"置信度: {result.confidence}")
    print(f"关键词: {result.keywords}")
```

### 4. 批量生成

```python
# 批量文本生成
prompts = ["问题1", "问题2", "问题3"]
instructions = ["指令1", "指令2", "指令3"]

results = client.batch_generate(
    prompts=prompts,
    instructions=instructions,
    n=2,  # 每个提示生成 2 个结果
)

# 批量结构化生成
results = client.batch_generate_structured(
    prompts=prompts,
    instructions=instructions,
    schema_model=Answer,
)
```

### 5. 使用统计

```python
# 获取使用统计
stats = client.get_usage()
print(stats)
# {
#     "total_prompt_tokens": 1000,
#     "total_completion_tokens": 500,
#     "total_tokens": 1500,
#     "total_latency": 2.5,
#     "total_calls": 5,
#     "successful_calls": 5,
#     "failed_calls": 0,
# }

# 重置统计
client.reset_usage()
```

## API 参考

### LLMClient

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_name | str | - | 模型名称 |
| api_key | str | - | API 密钥 |
| api_url | str | - | API 基础 URL |
| temperature | float | 0.6 | 采样温度 |
| max_tokens | int | 32768 | 最大输出 token 数 |
| top_p | float | 0.95 | Nucleus 采样参数 |
| top_k | int | 20 | Top-K 采样参数 |
| presence_penalty | float | 2.0 | 存在惩罚 |
| enable_thinking | bool | True | 启用推理提取 |
| n_return | int | 1 | 每次请求返回的结果数 |
| max_workers | int | 32 | 批处理最大线程数 |
| mock_llm | bool | False | 启用 Mock 模式 |

### 主要方法

| 方法 | 说明 |
|------|------|
| `generate(prompt, instruction, n)` | 生成文本 |
| `batch_generate(prompts, instructions, n)` | 批量生成文本 |
| `call_json(prompt, instruction, model)` | JSON 格式输出 |
| `batch_call_json(prompts, instructions, model)` | 批量 JSON 输出 |
| `generate_structured(prompt, instruction, schema_model, n)` | 结构化输出 |
| `batch_generate_structured(prompts, instructions, schema_model, n)` | 批量结构化输出 |
| `get_usage()` | 获取使用统计 |
| `reset_usage()` | 重置使用统计 |

### Generation

生成结果数据类

| 属性 | 类型 | 说明 |
|------|------|------|
| text | str | 生成文本 |
| reasoning | str | 推理过程（如果有） |

### LLMUsage

使用统计数据类

| 属性 | 类型 | 说明 |
|------|------|------|
| prompt_tokens | int | 输入 token 数 |
| completion_tokens | int | 输出 token 数 |
| total_tokens | int | 总 token 数 |
| latency | float | 延迟时间（秒） |

## 模块结构

```
llm/
├── __init__.py       # 导出所有公共接口
└── client.py         # LLM 客户端
```

## 许可证

MIT License
