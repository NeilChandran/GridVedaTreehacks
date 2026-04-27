import React, { useState, useEffect } from 'react';
import TempLoadGraph from './TempLoadGraph';
import DGAGraph from './DGAGraph';
import VariablesGrid from './VariablesGrid';

function TransformerDetail({ transformerId, telemetry, history, onBack, apiBase }) {
  const [analysis, setAnalysis] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [legacyAnalysis, setLegacyAnalysis] = useState(null);

  const reading = telemetry?.readings?.[transformerId];

  const runAnalysis = async () => {
    setAnalyzing(true);
    try {
      // Call the new ETT/DGA/Quantum analyze endpoint
      const resp = await fetch(`${apiBase}/api/transformers/${transformerId}/analyze`);
      if (resp.ok) {
        setAnalysis(await resp.json());
      }

      // Also run legacy prediction pipeline (Quantum VQC + Cerebras + LSTM)
      const readings = Array.from({ length: 50 }, (_, i) => ({
        transformer_id: transformerId,
        temperature: (reading?.temperature_c || 65) + Math.random() * 5,
        load_percent: (reading?.load_percent || 55) + Math.random() * 10,
        dga_h2: (reading?.dga?.h2 || 25) + Math.random() * 10,
        dga_ch4: (reading?.dga?.ch4 || 15) + Math.random() * 5,
        dga_c2h2: (reading?.dga?.c2h2 || 1) + Math.random(),
        dga_c2h4: (reading?.dga?.c2h4 || 10) + Math.random() * 5,
        dga_c2h6: (reading?.dga?.c2h6 || 8) + Math.random() * 3,
        moisture_ppm: (reading?.moisture_ppm || 12) + Math.random() * 3,
        vibration_mm_s: (reading?.vibration_mm_s || 2.5) + Math.random(),
      }));

      const predResp = await fetch(`${apiBase}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transformer_id: transformerId,
          readings,
          prediction_horizon_hours: 24,
        }),
      });

      if (predResp.ok) {
        setLegacyAnalysis(await predResp.json());
      }
    } catch (err) {
      console.error('Analysis failed:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  const getStatusColor = (value, thresholds) => {
    if (value >= thresholds.critical) return 'critical';
    if (value >= thresholds.warning) return 'warning';
    return 'healthy';
  };

  const getRiskColor = (score) => {
    if (score > 70) return 'critical';
    if (score > 50) return 'warning';
    return 'healthy';
  };

  if (!reading) {
    return (
      <div className="detail-panel">
        <button className="back-btn" onClick={onBack}>← Back to Fleet</button>
        <div className="empty-state">Waiting for telemetry data for {transformerId}...</div>
      </div>
    );
  }

  return (
    <div className="detail-panel">
      <div className="detail-header">
        <button className="back-btn" onClick={onBack}>← Back to Fleet</button>
        <div className="detail-title">
          <h2>{transformerId}</h2>
          <span className="detail-timestamp">
            Last update: {new Date(reading.timestamp).toLocaleTimeString()}
          </span>
        </div>
        <button
          className={`analyze-btn ${analyzing ? 'loading' : ''}`}
          onClick={runAnalysis}
          disabled={analyzing}
        >
          {analyzing ? '⏳ Running AI Pipeline...' : '🧠 Run Full Analysis'}
        </button>
      </div>

      {/* Sensor Readings */}
      <div className="sensor-grid">
        <SensorCard
          label="Temperature"
          value={`${reading.temperature_c?.toFixed(1)}°C`}
          status={getStatusColor(reading.temperature_c, { warning: 75, critical: 85 })}
          icon="🌡️"
        />
        <SensorCard
          label="Load"
          value={`${reading.load_percent?.toFixed(1)}%`}
          status={getStatusColor(reading.load_percent, { warning: 80, critical: 95 })}
          icon="⚡"
        />
        <SensorCard
          label="Power Factor"
          value={reading.power_factor?.toFixed(4)}
          status={reading.power_factor < 0.92 ? 'warning' : 'healthy'}
          icon="📐"
        />
        <SensorCard
          label="Oil Level"
          value={`${reading.oil_level_percent?.toFixed(1)}%`}
          status={reading.oil_level_percent < 80 ? 'warning' : 'healthy'}
          icon="🛢️"
        />
        <SensorCard
          label="Moisture"
          value={`${reading.moisture_ppm?.toFixed(1)} ppm`}
          status={getStatusColor(reading.moisture_ppm, { warning: 20, critical: 35 })}
          icon="💧"
        />
        <SensorCard
          label="Vibration"
          value={`${reading.vibration_mm_s?.toFixed(2)} mm/s`}
          status={getStatusColor(reading.vibration_mm_s, { warning: 5, critical: 8 })}
          icon="📳"
        />
      </div>

      {/* Time-Series Graphs + Variables Grid */}
      {history && history.length > 2 && (
        <div className="panel detail-graphs-panel">
          <div className="panel-header">
            <h3>📊 Live Telemetry History</h3>
            <span className="alert-count">{history.length} readings</span>
          </div>
          <div className="detail-graphs-layout">
            <div className="detail-graphs-col">
              <TempLoadGraph history={history} height={180} expanded />
              <DGAGraph history={history} height={180} expanded />
            </div>
            <div className="detail-vars-col">
              <VariablesGrid reading={reading} />
            </div>
          </div>
        </div>
      )}

      {/* DGA Panel */}
      <div className="panel dga-panel">
        <div className="panel-header">
          <h3>🧪 Dissolved Gas Analysis (DGA)</h3>
        </div>
        <div className="dga-grid">
          {Object.entries(reading.dga || {}).map(([gas, ppm]) => {
            const thresholds = {
              h2: { warning: 100, critical: 200 },
              ch4: { warning: 50, critical: 100 },
              c2h2: { warning: 2, critical: 10 },
              c2h4: { warning: 20, critical: 50 },
              c2h6: { warning: 15, critical: 40 },
            };
            const t = thresholds[gas] || { warning: 50, critical: 100 };
            const status = getStatusColor(ppm, t);
            const gasLabels = { h2: 'H₂', ch4: 'CH₄', c2h2: 'C₂H₂', c2h4: 'C₂H₄', c2h6: 'C₂H₆' };

            return (
              <div key={gas} className={`dga-card ${status}`}>
                <div className="dga-name">{gasLabels[gas] || gas}</div>
                <div className="dga-value">{ppm?.toFixed(1)} ppm</div>
                <div className="dga-bar">
                  <div
                    className={`dga-bar-fill ${status}`}
                    style={{ width: `${Math.min(100, (ppm / t.critical) * 100)}%` }}
                  />
                </div>
                <div className="dga-thresholds">
                  <span>W: {t.warning}</span>
                  <span>C: {t.critical}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ETT + DGA Ensemble Analysis Results */}
      {analysis && (
        <div className="panel analysis-panel">
          <div className="panel-header">
            <h3>🧠 Pre-Trained Ensemble Analysis (ETT → DGA → Quantum)</h3>
          </div>
          <div className="analysis-grid">
            {/* ETT Anomaly Detector */}
            {analysis.ett_analysis && (
              <div className="analysis-card">
                <div className="analysis-title">
                  <span>📊</span> ETT Anomaly Detector
                  <span className={`analysis-badge ${getRiskColor(analysis.ett_analysis.risk_score)}`}>
                    {analysis.ett_analysis.status}
                  </span>
                </div>
                <div className="analysis-fields">
                  <div className="analysis-field">
                    <span className="field-label">Risk Score</span>
                    <span className={`field-value ${getRiskColor(analysis.ett_analysis.risk_score)}`}>
                      {analysis.ett_analysis.risk_score?.toFixed(1)}%
                    </span>
                  </div>
                  <div className="analysis-field">
                    <span className="field-label">Recommendation</span>
                    <span className="field-value">{analysis.ett_analysis.recommendation}</span>
                  </div>
                  {analysis.ett_analysis.engineered_features && (
                    <>
                      <div className="analysis-field">
                        <span className="field-label">Thermal Stress</span>
                        <span className="field-value">{analysis.ett_analysis.engineered_features.thermal_stress}</span>
                      </div>
                      <div className="analysis-field">
                        <span className="field-label">Arrhenius Factor</span>
                        <span className="field-value">{analysis.ett_analysis.engineered_features.arrhenius_factor}</span>
                      </div>
                      <div className="analysis-field">
                        <span className="field-label">Aging Accel.</span>
                        <span className="field-value">{analysis.ett_analysis.engineered_features.aging_acceleration}h</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* DGA Fault Classifier */}
            {analysis.dga_analysis && (
              <div className="analysis-card">
                <div className="analysis-title">
                  <span>🧪</span> DGA Fault Classifier
                  <span className={`analysis-badge ${
                    analysis.dga_analysis.fault_type === 'Arcing' ? 'critical' :
                    analysis.dga_analysis.fault_type === 'Normal' ? 'healthy' : 'warning'
                  }`}>
                    {analysis.dga_analysis.fault_type}
                  </span>
                </div>
                <div className="analysis-fields">
                  <div className="analysis-field">
                    <span className="field-label">Fault Type</span>
                    <span className="field-value">{analysis.dga_analysis.fault_type}</span>
                  </div>
                  <div className="analysis-field">
                    <span className="field-label">Confidence</span>
                    <span className="field-value">{(analysis.dga_analysis.confidence * 100)?.toFixed(1)}%</span>
                  </div>
                  <div className="analysis-field">
                    <span className="field-label">Rogers Method</span>
                    <span className="field-value">{analysis.dga_analysis.rogers_method}</span>
                  </div>
                  <div className="analysis-field">
                    <span className="field-label">Duval Triangle</span>
                    <span className="field-value">{analysis.dga_analysis.duval_triangle}</span>
                  </div>
                  <div className="analysis-field">
                    <span className="field-label">Recommendation</span>
                    <span className="field-value">{analysis.dga_analysis.recommendation}</span>
                  </div>
                  {analysis.dga_analysis.key_ratios && (
                    <>
                      <div className="analysis-field">
                        <span className="field-label">R1 (CH4/H2)</span>
                        <span className="field-value">{analysis.dga_analysis.key_ratios.R1_CH4_H2}</span>
                      </div>
                      <div className="analysis-field">
                        <span className="field-label">R2 (C2H4/C2H6)</span>
                        <span className="field-value">{analysis.dga_analysis.key_ratios.R2_C2H4_C2H6}</span>
                      </div>
                      <div className="analysis-field">
                        <span className="field-label">R3 (C2H2/C2H4)</span>
                        <span className="field-value">{analysis.dga_analysis.key_ratios.R3_C2H2_C2H4}</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Quantum VQC (from analyze endpoint) */}
            {analysis.dga_analysis?.quantum_analysis && (
              <div className="analysis-card">
                <div className="analysis-title">
                  <span>🔮</span> Quantum VQC
                  <span className="analysis-model">{analysis.dga_analysis.quantum_analysis.qubits}q / {analysis.dga_analysis.quantum_analysis.layers}L</span>
                </div>
                <div className="analysis-fields">
                  <div className="analysis-field">
                    <span className="field-label">Quantum Class</span>
                    <span className="field-value">{analysis.dga_analysis.quantum_analysis.quantum_class}</span>
                  </div>
                  <div className="analysis-field">
                    <span className="field-label">Risk Score</span>
                    <span className="field-value">{(analysis.dga_analysis.quantum_analysis.risk_score * 100)?.toFixed(1)}%</span>
                  </div>
                  <div className="analysis-field">
                    <span className="field-label">Rogers Class</span>
                    <span className="field-value">{analysis.dga_analysis.quantum_analysis.rogers_class}</span>
                  </div>
                  <div className="analysis-field">
                    <span className="field-label">Duval Class</span>
                    <span className="field-value">{analysis.dga_analysis.quantum_analysis.duval_class}</span>
                  </div>
                  <div className="analysis-field">
                    <span className="field-label">Circuit Depth</span>
                    <span className="field-value">{analysis.dga_analysis.quantum_analysis.circuit_depth}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Cerebras Predictor (from analyze endpoint) */}
            {analysis.cerebras_predictor && (
              <div className="analysis-card">
                <div className="analysis-title">
                  <span>🧠</span> Cerebras Predictor
                  <span className="analysis-model">{analysis.cerebras_predictor.model}</span>
                </div>
                <div className="analysis-fields">
                  <div className="analysis-field">
                    <span className="field-label">Trend</span>
                    <span className={`field-value ${analysis.cerebras_predictor.trend === 'deteriorating' ? 'critical' : 'healthy'}`}>
                      {analysis.cerebras_predictor.trend}
                    </span>
                  </div>
                  <div className="analysis-field">
                    <span className="field-label">Risk Score</span>
                    <span className="field-value">{(analysis.cerebras_predictor.risk_score * 100)?.toFixed(1)}%</span>
                  </div>
                  <div className="analysis-field">
                    <span className="field-label">Predicted Hours</span>
                    <span className="field-value">{analysis.cerebras_predictor.predicted_hours}h to threshold</span>
                  </div>
                  <div className="analysis-field">
                    <span className="field-label">Speed</span>
                    <span className="field-value">{analysis.cerebras_predictor.tokens_per_second} tok/s</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Legacy AI Pipeline Results (Quantum VQC + Cerebras + LSTM) */}
      {legacyAnalysis && (
        <div className="panel analysis-panel">
          <div className="panel-header">
            <h3>🔬 NVIDIA AI Pipeline (Quantum VQC + Cerebras + LSTM)</h3>
          </div>
          <div className="analysis-grid">
            {legacyAnalysis.quantum_vqc && (
              <AnalysisCard
                title="Quantum VQC"
                icon="🔮"
                data={legacyAnalysis.quantum_vqc}
                fields={[
                  { label: 'Fault Type', key: 'fault_type' },
                  { label: 'Risk Score', key: 'risk_score' },
                  { label: 'Confidence', key: 'confidence' },
                  { label: 'Rogers Class', key: 'rogers_class' },
                  { label: 'Duval Class', key: 'duval_class' },
                ]}
              />
            )}
            {legacyAnalysis.cerebras_predictor && (
              <AnalysisCard
                title="Cerebras Predictor"
                icon="🧠"
                data={legacyAnalysis.cerebras_predictor}
                fields={[
                  { label: 'Trend', key: 'trend' },
                  { label: 'Risk Score', key: 'risk_score' },
                  { label: 'Rate of Change', key: 'rate_of_change' },
                  { label: 'Hours to Threshold', key: 'predicted_hours' },
                  { label: 'Severity', key: 'severity' },
                ]}
              />
            )}
            {legacyAnalysis.lstm_autoencoder && (
              <AnalysisCard
                title="LSTM Autoencoder"
                icon="🔍"
                data={legacyAnalysis.lstm_autoencoder}
                fields={[
                  { label: 'Anomaly', key: 'is_anomaly' },
                  { label: 'Recon Error', key: 'reconstruction_error' },
                  { label: 'Threshold', key: 'threshold' },
                  { label: 'Severity', key: 'severity' },
                  { label: 'Error Ratio', key: 'error_ratio' },
                ]}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function SensorCard({ label, value, status, icon }) {
  return (
    <div className={`sensor-card ${status}`}>
      <div className="sensor-icon">{icon}</div>
      <div className="sensor-label">{label}</div>
      <div className="sensor-value">{value}</div>
      <div className={`sensor-badge ${status}`}>{status}</div>
    </div>
  );
}

function AnalysisCard({ title, icon, data, fields }) {
  return (
    <div className="analysis-card">
      <div className="analysis-title">
        <span>{icon}</span> {title}
        <span className="analysis-model">{data.model}</span>
      </div>
      <div className="analysis-fields">
        {fields.map(f => (
          <div key={f.key} className="analysis-field">
            <span className="field-label">{f.label}</span>
            <span className="field-value">
              {typeof data[f.key] === 'boolean'
                ? (data[f.key] ? '⚠ YES' : '✓ No')
                : data[f.key] ?? '—'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default TransformerDetail;
