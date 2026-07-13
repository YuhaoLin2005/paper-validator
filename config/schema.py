"""Configuration dataclasses for paper-validator.

Imported by: main.py, engine/api_client.py, layers/*.py, claims/runner.py
No data files read/written — pure dataclass definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """API model configuration."""
    base_url: str = "https://api.deepseek.com/v1"
    model_id: str = "deepseek-chat"
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout_s: int = 60
    max_retries: int = 2
    retry_delay_s: float = 2.0


@dataclass
class LayerConfig:
    """Per-layer configuration thresholds."""
    l0_amendment_consensus: float = 0.6
    l0_cooling_hours: int = 24
    l1_disk_warn_gb: int = 50
    l1_disk_block_gb: int = 15
    l1_config_stale_hours: int = 4
    l2_active_threshold: float = 0.3
    l2_top_logprobs: int = 20
    l3_persona_count: int = 5
    l3_consensus_threshold: float = 0.6
    l4_risk_low: int = 30
    l4_risk_medium: int = 60
    l4_risk_high: int = 80


@dataclass
class ClaimConfig:
    """Configuration for a single claim experiment."""
    name: str = ""
    trials: int = 30
    parallel: bool = False
    output_dir: str = "results"


@dataclass
class AppConfig:
    """Top-level application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    layers: LayerConfig = field(default_factory=LayerConfig)
    state_dir: str = ".paper-validator-state"
    results_dir: str = "results"
