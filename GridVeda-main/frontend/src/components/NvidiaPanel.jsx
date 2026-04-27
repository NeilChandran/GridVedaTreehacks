import React from 'react';

function NvidiaPanel({ status, telemetry }) {
  const gpu = status?.gpu || {};
  const models = status?.models || {};
  const edges = status?.edge_targets || {};

  const hardwareTiers = [
    {
      name: 'DGX Spark',
      icon: '🖥️',
      role: 'Development & Fine-Tuning',
      price: '$3,999',
      specs: [
        'Grace Blackwell GPU',
        '128GB Unified Memory',
        'Run 200B parameter models',
        'Fine-tune Nemotron locally',
      ],
      color: '#76b900',
    },
    {
      name: 'RTX 5090',
      icon: '🎮',
      role: 'Demo & Prototyping',
      price: 'TreeHacks Demo',
      specs: [
        'CUDA 12+ Acceleration',
        '32GB GDDR7 VRAM',
        'Ollama model serving',
        'Full dashboard + inference',
      ],
      color: '#00d4ff',
    },
    {
      name: 'Jetson Orin Nano Super',
      icon: '🔌',
      role: 'Edge / Substation Deploy',
      price: '$249',
      specs: [
        '67 TOPS @ 25W',
        '8GB memory',
        'Nemotron Nano 4B inference',
        'Real-time monitoring',
      ],
      color: '#ffb800',
    },
  ];

  const modelCards = [
    {
      name: 'Nemotron Nano 4B',
      icon: '🤖',
      status: models.nemotron_nano_4b?.status || 'ready',
      backend: 'Ollama',
      role: 'Chat / RAG',
      weight: '—',
      desc: 'Grid-aware conversational AI. Replaces Perplexity — 100% local, offline capable.',
    },
    {
      name: 'Quantum VQC',
      icon: '🔮',
      status: models.quantum_vqc?.status || 'loaded',
      backend: 'cuQuantum',
      role: 'Fault Classification',
      weight: '35%',
      desc: '6-qubit variational circuit for DGA fault classification via Duval Triangle + Rogers Ratio ensemble.',
    },
    {
      name: 'Liquid LTC Network',
      icon: '🧪',
      status: models.liquid_ltc?.status || 'loaded',
      backend: 'PyTorch + CUDA',
      role: 'Time-Series Prediction',
      weight: '35%',
      desc: 'Continuous-time ODE network for degradation trend analysis. ~12K params, edge-optimized.',
    },
    {
      name: 'LSTM Autoencoder',
      icon: '🔍',
      status: models.lstm_autoencoder?.status || 'loaded',
      backend: 'PyTorch + TensorRT',
      role: 'Anomaly Detection',
      weight: '25%',
      desc: 'Sequence-to-sequence autoencoder. Detects unseen failure patterns via reconstruction error.',
    },
  ];

  const stackLayers = [
    { label: 'Application', items: ['GridVeda Dashboard', 'Ask Grid Chat', 'Alert Engine'], color: '#00d4ff' },
    { label: 'AI Models', items: ['Nemotron Nano 4B', 'Quantum VQC', 'Liquid LTC', 'LSTM AE'], color: '#76b900' },
    { label: 'Inference', items: ['Ollama', 'TensorRT-LLM', 'NVIDIA NIM'], color: '#ffb800' },
    { label: 'Compute', items: ['CUDA', 'cuQuantum', 'Tensor Cores'], color: '#ff4757' },
    { label: 'Hardware', items: ['DGX Spark', 'RTX 5090', 'Jetson Orin Nano Super'], color: '#8b5cf6' },
  ];

  return (
    <div className="nvidia-panel">
      {/* Header */}
      <div className="nvidia-header">
        <div className="nvidia-title">
          <span className="nvidia-logo">🟢</span>
          <div>
            <h2>NVIDIA Stack — 100% Open Source</h2>
            <p>From development to edge deployment, powered entirely by NVIDIA open models and hardware.</p>
          </div>
        </div>
      </div>

      {/* Hardware Tiers */}
      <div className="section-label">Hardware Deployment Tiers</div>
      <div className="hardware-grid">
        {hardwareTiers.map((hw, i) => (
          <div key={i} className="hardware-card" style={{ borderTopColor: hw.color }}>
            <div className="hw-header">
              <span className="hw-icon">{hw.icon}</span>
              <div>
                <h3>{hw.name}</h3>
                <span className="hw-role">{hw.role}</span>
              </div>
              <span className="hw-price" style={{ color: hw.color }}>{hw.price}</span>
            </div>
            <ul className="hw-specs">
              {hw.specs.map((spec, j) => (
                <li key={j}>
                  <span className="spec-dot" style={{ background: hw.color }} />
                  {spec}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* AI Model Cards */}
      <div className="section-label">AI Model Pipeline</div>
      <div className="model-grid">
        {modelCards.map((model, i) => (
          <div key={i} className="model-card">
            <div className="model-header">
              <span className="model-icon">{model.icon}</span>
              <div className="model-info">
                <h4>{model.name}</h4>
                <span className="model-role">{model.role}</span>
              </div>
              <div className="model-status-group">
                <span className={`model-status ${model.status}`}>
                  {model.status}
                </span>
                {model.weight !== '—' && (
                  <span className="model-weight">{model.weight}</span>
                )}
              </div>
            </div>
            <p className="model-desc">{model.desc}</p>
            <div className="model-meta">
              <span className="meta-tag">{model.backend}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Full Stack Diagram */}
      <div className="section-label">Full Stack Architecture</div>
      <div className="stack-diagram">
        {stackLayers.map((layer, i) => (
          <div key={i} className="stack-layer" style={{ borderLeftColor: layer.color }}>
            <div className="stack-label" style={{ color: layer.color }}>{layer.label}</div>
            <div className="stack-items">
              {layer.items.map((item, j) => (
                <span key={j} className="stack-chip">{item}</span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* GPU Status */}
      {status && (
        <>
          <div className="section-label">Runtime Status</div>
          <div className="gpu-status-card">
            <div className="gpu-row">
              <span className="gpu-label">GPU</span>
              <span className="gpu-value">{gpu.name || 'NVIDIA GPU'}</span>
            </div>
            <div className="gpu-row">
              <span className="gpu-label">CUDA</span>
              <span className={`gpu-badge ${gpu.cuda_available ? 'active' : ''}`}>
                {gpu.cuda_available ? 'Available' : 'Not Detected'}
              </span>
            </div>
            <div className="gpu-row">
              <span className="gpu-label">TensorRT</span>
              <span className={`gpu-badge ${gpu.tensorrt_ready ? 'active' : ''}`}>
                {gpu.tensorrt_ready ? 'Ready' : 'Not Installed'}
              </span>
            </div>
            <div className="gpu-row">
              <span className="gpu-label">Memory</span>
              <span className="gpu-value">
                {gpu.memory_used_gb || '—'}GB / {gpu.memory_total_gb || '—'}GB
              </span>
            </div>
            <div className="gpu-memory-bar">
              <div
                className="gpu-memory-fill"
                style={{ width: `${((gpu.memory_used_gb || 0) / (gpu.memory_total_gb || 24)) * 100}%` }}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default NvidiaPanel;
