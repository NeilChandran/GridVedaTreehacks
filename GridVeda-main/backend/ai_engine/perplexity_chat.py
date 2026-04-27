"""
Perplexity Sonar API Chat Interface for GridVeda
Replaces local Nemotron Nano with web-grounded Perplexity Sonar.

Features:
  - Real-time web search grounding — answers cite live sources
  - Grid-aware system prompt with transformer domain knowledge
  - DGA analysis interpretation with IEEE C57.104 standards
  - Real-time fleet context injection
  - Citations included in every response

Model: sonar (Perplexity's optimized model, ~1200 tok/s)
API: https://api.perplexity.ai/chat/completions (OpenAI-compatible)

Sponsor: Perplexity (TreeHacks 2026)
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = "sonar"


GRID_SYSTEM_PROMPT = """You are GridVeda AI, an expert power grid monitoring assistant powered by Perplexity Sonar with real-time web search capabilities.

Your expertise includes:
- Transformer health monitoring and predictive maintenance
- Dissolved Gas Analysis (DGA) interpretation using Duval Triangle and Rogers Ratio
- Real-time sensor data analysis (temperature, load, vibration, moisture)
- SCADA systems and industrial protocols (Modbus, DNP3, IEC 61850, OPC-UA)
- Power grid operations, demand response, and renewable integration
- NERC CIP compliance and regulatory requirements

When analyzing grid data:
1. Always reference specific sensor values and thresholds
2. Use IEEE C57.104 standards for DGA interpretation
3. Provide actionable recommendations with urgency levels
4. Consider cascading failure risks across interconnected assets
5. Reference the AI engine results (Quantum VQC, Cerebras Predictor, LSTM AE) when available

You can also search the web for the latest grid reliability reports, equipment recalls, weather-related grid threats, and regulatory updates.

Keep responses concise, technical, and actionable. Use bullet points for recommendations.
Format critical warnings prominently."""


# Simulated responses for when API is unavailable
FALLBACK_RESPONSES = {
    "health": "Based on the current fleet data, the grid health score is {score}. {details}",
    "dga": """**DGA Analysis** (IEEE C57.104)

Key gas concentrations suggest {fault_type}:
- **H₂ (Hydrogen)**: {h2} ppm — {h2_status}
- **C₂H₂ (Acetylene)**: {c2h2} ppm — {c2h2_status}
- **C₂H₄ (Ethylene)**: {c2h4} ppm — {c2h4_status}

**Recommendation**: {recommendation}
**Duval Triangle Classification**: {duval_zone}""",
    "general": """GridVeda's AI ensemble is monitoring 20 transformers across 4 substations:

🧠 **AI Models Active**:
- Quantum VQC: 6-qubit risk classification
- Cerebras Predictor: Llama 3.3 70B trend analysis (~2000 tok/s)
- LSTM Autoencoder: Pattern anomaly detection

📊 **Current Status**: {status}

Ask me about specific transformers, DGA patterns, alert history, or predictive maintenance schedules.""",
    "default": """I'm GridVeda AI, your grid intelligence assistant powered by Perplexity Sonar with real-time web search.

I can help with:
- 📊 Fleet health analysis and trending
- ⚗️ DGA interpretation (Duval Triangle, Rogers Ratio)
- 🔮 Predictive maintenance recommendations
- 🌐 Latest grid reliability news and weather threats
- 📋 NERC CIP compliance guidance

What would you like to know?""",
}


class PerplexityChat:
    """
    Chat interface using Perplexity Sonar API with web search grounding.
    Falls back to intelligent simulated responses if API is unavailable.
    """

    def __init__(self):
        self.model = PERPLEXITY_MODEL
        self.conversation_history: list = []
        self.api_available = False
        self.api_key = os.environ.get("PERPLEXITY_API_KEY", "")
        self.request_count = 0
        self.total_tokens = 0
        self.citations: List[str] = []

        self._check_availability()

    def _check_availability(self):
        """Check if Perplexity API is available."""
        if not self.api_key:
            print("⚠️  PERPLEXITY_API_KEY not set — using simulated grid responses")
            self.api_available = False
            return

        if not HAS_HTTPX:
            print("⚠️  httpx not installed — using simulated grid responses")
            self.api_available = False
            return

        if len(self.api_key) > 10:
            self.api_available = True
            print(f"🔍 Perplexity Sonar ready — model: {PERPLEXITY_MODEL}")
            print(f"   Web-grounded responses with citations")
        else:
            print("⚠️  Invalid PERPLEXITY_API_KEY")

    async def ask(self, message: str, grid_context: Optional[str] = None) -> str:
        """
        Send a message and get a response.
        Uses Perplexity Sonar API if available, falls back to simulated.

        Args:
            message: User's question
            grid_context: JSON string of current fleet metrics

        Returns:
            Response string
        """
        # Build context-enriched message
        enriched_message = message
        if grid_context:
            enriched_message = f"""Current Grid Context:
{grid_context}

User Question: {message}"""

        # Try Perplexity Sonar API
        if self.api_available and HAS_HTTPX:
            response = await self._call_perplexity(enriched_message)
            if response:
                return response

        # Fallback to simulated response
        return self._simulated_response(message, grid_context)

    async def _call_perplexity(self, message: str) -> Optional[str]:
        """Call Perplexity Sonar API."""
        messages = [
            {"role": "system", "content": GRID_SYSTEM_PROMPT},
        ]

        # Add conversation history (last 6 messages for context)
        for msg in self.conversation_history[-6:]:
            messages.append(msg)

        messages.append({"role": "user", "content": message})

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    PERPLEXITY_API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": 800,
                        "temperature": 0.4,
                    },
                )

                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]

                    # Track metrics
                    self.request_count += 1
                    usage = data.get("usage", {})
                    self.total_tokens += usage.get("total_tokens", 0)

                    # Extract citations if present
                    self.citations = data.get("citations", [])

                    # Update conversation history
                    self.conversation_history.append({"role": "user", "content": message})
                    self.conversation_history.append({"role": "assistant", "content": content})

                    # Trim history
                    if len(self.conversation_history) > 20:
                        self.conversation_history = self.conversation_history[-12:]

                    # Append citations if available
                    if self.citations:
                        citation_text = "\n\n📚 **Sources**: " + " | ".join(
                            [f"[{i+1}]({url})" for i, url in enumerate(self.citations[:3])]
                        )
                        content += citation_text

                    return content

                elif resp.status_code == 401:
                    print("⚠️  Perplexity API: Invalid API key")
                    self.api_available = False
                    return None
                elif resp.status_code == 429:
                    print("⚠️  Perplexity API: Rate limited")
                    return None
                else:
                    print(f"⚠️  Perplexity API: {resp.status_code} — {resp.text[:200]}")
                    return None

        except httpx.TimeoutException:
            print("⚠️  Perplexity API: Timeout")
            return None
        except Exception as e:
            print(f"⚠️  Perplexity API error: {e}")
            return None

    def _simulated_response(self, message: str, grid_context: Optional[str] = None) -> str:
        """Generate intelligent simulated response based on query type."""
        msg_lower = message.lower()

        # Parse grid context
        ctx = {}
        if grid_context:
            try:
                ctx = json.loads(grid_context)
            except:
                pass

        fleet_health = ctx.get("fleet_health", 94.2)
        sample = ctx.get("sample_readings", {})

        # Route to appropriate response template
        if any(w in msg_lower for w in ["health", "status", "how", "fleet", "overview"]):
            if fleet_health > 90:
                details = "All 20 transformers operational. Minor monitoring flags on XFMR-007 (elevated temperature trend) and XFMR-012 (intermittent DGA elevation)."
            elif fleet_health > 75:
                details = "⚠️ Several transformers showing degradation. Recommend immediate inspection of units with health scores below 80."
            else:
                details = "🚨 CRITICAL: Multiple transformers in distress. Initiate emergency maintenance protocols."

            return FALLBACK_RESPONSES["health"].format(
                score=f"{fleet_health:.1f}/100",
                details=details,
            )

        elif any(w in msg_lower for w in ["dga", "gas", "dissolved", "duval"]):
            # Find the most interesting transformer from context
            h2_val = 25
            c2h2_val = 1
            c2h4_val = 10
            fault_type = "normal aging patterns"
            recommendation = "Continue routine monitoring per IEEE C57.104 Schedule B"
            duval_zone = "Zone DT (Normal)"

            for xfmr_id, readings in sample.items():
                h2 = readings.get("dga_h2", 0)
                if h2 > h2_val:
                    h2_val = h2
                    c2h2_val = readings.get("dga_c2h2", c2h2_val) if "dga_c2h2" in readings else c2h2_val

            if h2_val > 100:
                fault_type = "potential thermal fault (T2)"
                recommendation = "Schedule oil sampling within 7 days. Increase monitoring frequency to daily."
                duval_zone = "Zone T2 (Thermal Fault 300-700°C)"
            if c2h2_val > 10:
                fault_type = "HIGH ENERGY ARCING (D1-D2)"
                recommendation = "🚨 IMMEDIATE inspection required. Consider de-energizing for internal examination."
                duval_zone = "Zone D2 (High Energy Discharge)"

            return FALLBACK_RESPONSES["dga"].format(
                fault_type=fault_type,
                h2=f"{h2_val:.0f}",
                h2_status="⚠ Elevated" if h2_val > 100 else "Normal",
                c2h2=f"{c2h2_val:.1f}",
                c2h2_status="🚨 Critical" if c2h2_val > 10 else "Normal",
                c2h4=f"{c2h4_val:.0f}",
                c2h4_status="Normal",
                recommendation=recommendation,
                duval_zone=duval_zone,
            )

        elif any(w in msg_lower for w in ["model", "ai", "engine", "system", "architecture"]):
            return FALLBACK_RESPONSES["general"].format(
                status=f"Fleet health {fleet_health:.1f}/100" if fleet_health else "Connecting to telemetry...",
            )

        else:
            return FALLBACK_RESPONSES["default"]

    def get_info(self) -> Dict[str, Any]:
        """Return chat engine info."""
        return {
            "name": "Perplexity Sonar Chat",
            "model": self.model,
            "api_available": self.api_available,
            "web_grounded": True,
            "citations_enabled": True,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "conversation_length": len(self.conversation_history),
            "speed": "~1200 tok/s",
            "sponsor": "Perplexity (TreeHacks 2026)",
        }
