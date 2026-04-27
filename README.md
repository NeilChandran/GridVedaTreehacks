# GridVeda — NVIDIA-First AI Grid Intelligence

AI-Powered Grid Monitoring with Nemotron Nano 4B on local NVIDIA GPU, Quantum VQC, LSTM Autoencoder, augmented by Cerebras inference and Perplexity web search.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (gridveda-live.html)                          │
│  ├── WebSocket ← Real-time telemetry every 2s           │
│  ├── REST → /api/chat (Nemotron 4B via Ollama)          │
│  ├── REST → /api/search (Perplexity Sonar — 🌐 toggle)  │
│  ├── REST → /api/predict (Quantum + Cerebras + LSTM)    │
│  ├── Anomaly Injector (5 fault types, severity slider)  │
│  └── Voice Assistant (Web Speech API, hands-free)       │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  FastAPI Backend (main.py :8000)                         │
│  ├── WebSocket /ws/telemetry (2s broadcast loop)         │
│  │                                                       │
│  ├── NVIDIA Models (Primary)                             │
│  │   ├── NemotronChat → Ollama :11434 (local GPU)        │
│  │   ├── QuantumVQC (6 qubits, cuQuantum-accelerated)    │
│  │   └── LSTMAutoencoder (45K params, CUDA)              │
│  │                                                       │
│  ├── Sponsor Augmentations                               │
│  │   ├── CerebrasPredictor → Llama 3.3 70B (~2000 tok/s)│
│  │   └── PerplexityChat → Sonar API (web-grounded)       │
│  │                                                       │
│  └── 20 simulated transformers with full DGA profiles    │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Ollama (:11434)                                         │
│  └── nemotron-nano-4b-instruct (NVIDIA open model)       │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install deps & start backend
cd backend && pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# 2. Open gridveda-live.html in your browser

# 3. (Optional) For real Nemotron chat:
ollama pull nemotron-nano-4b-instruct && ollama serve

# 4. (Optional) Enable sponsor APIs:
export CEREBRAS_API_KEY=csk-xxxx      # Free at cloud.cerebras.ai
export PERPLEXITY_API_KEY=pplx-xxxx   # At perplexity.ai/settings/api
```

> Everything works without API keys via intelligent fallbacks.

## AI Pipeline (5 Models)

### NVIDIA Primary (3/5 models)
| Model | Role | Hardware |
|-------|------|----------|
| **Nemotron Nano 4B** | Grid-aware chat (Ollama) | Jetson/RTX/DGX |
| **Quantum VQC** (6 qubits) | DGA fault classification | cuQuantum |
| **LSTM Autoencoder** (45K) | Pattern anomaly detection | CUDA |

### Sponsor Augmentations (2/5 models)
| Model | Role | Speed |
|-------|------|-------|
| **Cerebras Predictor** | Time-series trend analysis | ~2000 tok/s |
| **Perplexity Sonar** | Web-grounded grid research | ~1200 tok/s |

## API Endpoints

| Method | Path | Engine |
|--------|------|--------|
| POST | `/api/chat` | **Nemotron 4B (NVIDIA)** |
| POST | `/api/search` | **Perplexity Sonar** |
| POST | `/api/predict` | **VQC + Cerebras + LSTM** |
| GET | `/api/fleet/metrics` | Fleet health |
| GET | `/api/nvidia/status` | Hardware status |
| POST | `/api/demo/inject-anomaly` | Fault injection |
| WS | `/ws/telemetry` | Live stream (2s) |

Full docs: `http://localhost:8000/docs`

## Demo Features
- **Anomaly Injector** — 5 fault types (thermal/acetylene/ethylene/overload/cascade)
- **Voice Assistant** — Speech-to-text commands, TTS responses
- **🌐 Web Toggle** — Switch chat from Nemotron to Perplexity Sonar

## Hardware Targets (All NVIDIA)
| Tier | Device | Spec |
|------|--------|------|
| Edge | Jetson Orin Nano Super | 67 TOPS, 25W, $249 |
| Demo | RTX 5090 | 32GB VRAM, CUDA 12 |
| DC | DGX Spark | 128GB, Grace Blackwell |

## TreeHacks 2026 Prize Alignment
- ✅ **NVIDIA** — Nemotron 4B + cuQuantum + CUDA (3/5 models, all hardware)
- ✅ **Cerebras** — Fastest inference for real-time grid trend prediction
- ✅ **Perplexity** — Web-grounded grid incident research with citations
