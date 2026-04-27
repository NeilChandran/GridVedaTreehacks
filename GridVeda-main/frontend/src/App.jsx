import React, { useState, useEffect, useRef, useCallback } from 'react';
import Dashboard from './components/Dashboard';
import ChatPanel from './components/ChatPanel';
import NvidiaPanel from './components/NvidiaPanel';
import TransformerDetail from './components/TransformerDetail';
import ChatBubble from './components/ChatBubble';
import './styles/App.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/telemetry';

const MAX_HISTORY = 200; // Keep last N telemetry ticks for graphs

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [telemetry, setTelemetry] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [connected, setConnected] = useState(false);
  const [selectedTransformer, setSelectedTransformer] = useState(null);
  const [nvidiaStatus, setNvidiaStatus] = useState(null);
  const [telemetryHistory, setTelemetryHistory] = useState([]);
  const [transformerHistory, setTransformerHistory] = useState({});
  const wsRef = useRef(null);
  const reconnectRef = useRef(null);

  // WebSocket connection with auto-reconnect
  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      console.log('⚡ WebSocket connected to GridVeda');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'telemetry') {
        setTelemetry(data);

        // Accumulate fleet-level history for dashboard charts
        setTelemetryHistory(prev => [...prev, data].slice(-MAX_HISTORY));

        // Accumulate per-transformer reading history for detail graphs
        if (data.readings) {
          setTransformerHistory(prev => {
            const next = { ...prev };
            for (const [id, reading] of Object.entries(data.readings)) {
              if (!next[id]) next[id] = [];
              next[id] = [...next[id], reading].slice(-MAX_HISTORY);
            }
            return next;
          });
        }

        if (data.alerts?.length > 0) {
          setAlerts(prev => [...data.alerts, ...prev].slice(0, 50));
        }
      }
    };

    ws.onclose = () => {
      setConnected(false);
      reconnectRef.current = setTimeout(connectWS, 3000);
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    connectWS();
    // Fetch NVIDIA status
    fetch(`${API_BASE}/api/nvidia/status`)
      .then(r => r.json())
      .then(setNvidiaStatus)
      .catch(() => {});

    return () => {
      wsRef.current?.close();
      clearTimeout(reconnectRef.current);
    };
  }, [connectWS]);

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: '📊' },
    { id: 'chat', label: 'Ask Grid', icon: '🤖' },
    { id: 'nvidia', label: 'NVIDIA Stack', icon: '🟢' },
  ];

  return (
    <div className="app">
      {/* Top Navigation */}
      <nav className="top-nav">
        <div className="nav-brand">
          <span className="brand-icon">⚡</span>
          <span className="brand-text">GridVeda</span>
          <span className="brand-badge">NVIDIA</span>
        </div>

        <div className="nav-tabs">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => { setActiveTab(tab.id); setSelectedTransformer(null); }}
            >
              <span className="tab-icon">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>

        <div className="nav-status">
          <span className={`status-dot ${connected ? 'online' : 'offline'}`} />
          <span className="status-text">
            {connected ? 'Live' : 'Reconnecting...'}
          </span>
          {telemetry && (
            <span className="tick-counter">T+{telemetry.tick}</span>
          )}
        </div>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        {selectedTransformer ? (
          <TransformerDetail
            transformerId={selectedTransformer}
            telemetry={telemetry}
            history={transformerHistory[selectedTransformer] || []}
            onBack={() => setSelectedTransformer(null)}
            apiBase={API_BASE}
          />
        ) : (
          <>
            {activeTab === 'dashboard' && (
              <Dashboard
                telemetry={telemetry}
                alerts={alerts}
                telemetryHistory={telemetryHistory}
                transformerHistory={transformerHistory}
                onSelectTransformer={setSelectedTransformer}
              />
            )}
            {activeTab === 'chat' && (
              <ChatPanel apiBase={API_BASE} telemetry={telemetry} />
            )}
            {activeTab === 'nvidia' && (
              <NvidiaPanel status={nvidiaStatus} telemetry={telemetry} />
            )}
          </>
        )}
      </main>

      {/* Floating Chat Bubble — always visible */}
      <ChatBubble apiBase={API_BASE} telemetry={telemetry} />
    </div>
  );
}

export default App;
