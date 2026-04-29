#!/bin/bash
# Run inside WSL to (re)write ~/.pi/agent/models.json without BOM.
# Usage: wsl.exe -d Ubuntu-24.04 -e bash <this-file>
set -e
mkdir -p ~/.pi/agent
cat > ~/.pi/agent/models.json <<'EOF'
{
  "providers": {
    "specula-npu": {
      "baseUrl": "http://host.docker.internal:8081/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "models": [
        {
          "id": "qwen3-4b-npu",
          "name": "Qwen3-4B (NPU, Hexagon HTP)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 2048,
          "maxTokens": 1024,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    },
    "specula-vulkan-4b": {
      "baseUrl": "http://host.docker.internal:8082/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "models": [
        {
          "id": "qwen3-4b",
          "name": "Qwen3-4B (Adreno Vulkan)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 16384,
          "maxTokens": 4096,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    },
    "specula-vulkan-7b": {
      "baseUrl": "http://host.docker.internal:8083/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "models": [
        {
          "id": "qwen2_5-7b",
          "name": "Qwen2.5-7B-Instruct (Adreno Vulkan)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 16384,
          "maxTokens": 4096,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    },
    "specula-daily-driver": {
      "baseUrl": "http://host.docker.internal:8080/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "models": [
        {
          "id": "daily-driver",
          "name": "Qwen3.6-35B-A3B (Vulkan, daily-driver)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 4096,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
EOF
echo "wrote ~/.pi/agent/models.json"
head -3 ~/.pi/agent/models.json
