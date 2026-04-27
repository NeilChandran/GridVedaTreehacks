import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function ChatBubble({ apiBase, telemetry }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: '⚡ **GridVeda AI** powered by **NVIDIA Nemotron Nano 4B** (Ollama)\n\nAsk me about fleet health, DGA analysis, alerts, or the NVIDIA stack.',
      timestamp: new Date().toISOString(),
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const recognitionRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus();
    }
  }, [open]);

  // ─── Web Speech API ───
  const initRecognition = useCallback(() => {
    if (recognitionRef.current) return recognitionRef.current;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return null;

    const recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = (e) => {
      const result = e.results[e.resultIndex];
      const text = result[0].transcript;
      setInput(text);
      if (result.isFinal) {
        setIsListening(false);
      }
    };

    recognition.onend = () => setIsListening(false);
    recognition.onerror = (e) => {
      setIsListening(false);
      if (e.error !== 'no-speech') {
        console.warn('Speech recognition error:', e.error);
      }
    };

    recognitionRef.current = recognition;
    return recognition;
  }, []);

  const toggleMic = useCallback(() => {
    const recognition = initRecognition();
    if (!recognition) return;

    if (isListening) {
      recognition.stop();
      setIsListening(false);
    } else {
      setInput('');
      recognition.start();
      setIsListening(true);
    }
  }, [isListening, initRecognition]);

  // ─── Send message via /api/chat ───
  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMsg = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const resp = await fetch(`${apiBase}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMsg.content,
          context: telemetry ? JSON.stringify(telemetry.fleet_metrics) : null,
        }),
      });

      if (resp.ok) {
        const data = await resp.json();
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.response,
          model: data.model,
          engine: data.engine,
          timestamp: new Date().toISOString(),
        }]);
      } else {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: '⚠ Backend error. Ensure the server is running on port 8000.',
          timestamp: new Date().toISOString(),
          isError: true,
        }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `⚠ Connection failed: ${err.message}`,
        timestamp: new Date().toISOString(),
        isError: true,
      }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const quickActions = [
    { label: 'Fleet Status', msg: 'What is the current fleet health status?' },
    { label: 'DGA Analysis', msg: 'Explain the current DGA readings and any concerns' },
    { label: 'Active Alerts', msg: 'What are the current active alerts and recommendations?' },
    { label: 'Predictions', msg: 'What are the predictive maintenance forecasts?' },
  ];

  return (
    <>
      {/* Floating Action Button */}
      <button
        className={`chat-fab ${open ? 'open' : ''}`}
        onClick={() => setOpen(prev => !prev)}
        title="Ask GridVeda AI"
      >
        <span className="fab-icon">{open ? '✕' : '🤖'}</span>
      </button>

      {/* Chat Window */}
      {open && (
        <div className="chat-bubble-window">
          {/* Header */}
          <div className="cb-header">
            <div className="cb-header-info">
              <span className="cb-avatar">⚡</span>
              <div>
                <div className="cb-title">Ask Grid — Nemotron Nano 4B</div>
                <div className="cb-subtitle">100% Local • NVIDIA Ollama • Grid-Aware</div>
              </div>
            </div>
            <div className="cb-model-badge">
              <span className="cb-nvidia-dot" />
              Nemotron 4B
            </div>
          </div>

          {/* Quick Actions */}
          <div className="cb-quick-actions">
            {quickActions.map((qa, i) => (
              <button
                key={i}
                className="cb-quick-btn"
                onClick={() => { setInput(qa.msg); inputRef.current?.focus(); }}
                disabled={loading}
              >
                {qa.label}
              </button>
            ))}
          </div>

          {/* Messages */}
          <div className="cb-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`cb-msg ${msg.role} ${msg.isError ? 'error' : ''}`}>
                <div className="cb-msg-avatar">
                  {msg.role === 'user' ? '👤' : '⚡'}
                </div>
                <div className="cb-msg-body">
                  <div className="cb-msg-text cb-markdown">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        a: ({ href, children }) => (
                          <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
                        ),
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                  <div className="cb-msg-meta">
                    <span className="cb-msg-time">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </span>
                    {msg.model && (
                      <span className="cb-msg-model">
                        via {msg.model} ({msg.engine})
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {loading && (
              <div className="cb-msg assistant">
                <div className="cb-msg-avatar">⚡</div>
                <div className="cb-msg-body">
                  <div className="cb-typing">
                    <span /><span /><span />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="cb-input-area">
            <button
              className={`cb-mic ${isListening ? 'listening' : ''}`}
              onClick={toggleMic}
              title={isListening ? 'Stop listening' : 'Voice input'}
            >
              🎤
            </button>
            <input
              ref={inputRef}
              type="text"
              className="cb-input"
              placeholder={isListening ? 'Listening...' : 'Ask about grid health, DGA, alerts...'}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && sendMessage()}
              disabled={loading}
            />
            <button
              className="cb-send"
              onClick={sendMessage}
              disabled={!input.trim() || loading}
            >
              {loading ? '⏳' : '→'}
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export default ChatBubble;
