"""
Configuration Module
====================
Handles configuration management for simulation experiments
Supports YAML and JSON formats

Author: AI Infrastructure Research Team
"""

import yaml
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """Hardware specifications for compute nodes"""
    gpu_model: str = "A100"
    compute_tflops: float = 312.0  # FP32 TFLOPS
    memory_gb: float = 80.0
    network_bandwidth_gbps: float = 100.0  # GB/s (e.g., NVLink, InfiniBand)
    num_gpus_per_node: int = 8
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HardwareConfig':
        return cls(**data)
    
    @staticmethod
    def get_preset(preset_name: str) -> 'HardwareConfig':
        """Get predefined hardware configurations"""
        presets = {
            "A100": HardwareConfig(
                gpu_model="A100",
                compute_tflops=312.0,
                memory_gb=80.0,
                network_bandwidth_gbps=100.0,
                num_gpus_per_node=8
            ),
            "H100": HardwareConfig(
                gpu_model="H100",
                compute_tflops=989.0,
                memory_gb=80.0,
                network_bandwidth_gbps=200.0,
                num_gpus_per_node=8
            ),
            "V100": HardwareConfig(
                gpu_model="V100",
                compute_tflops=125.0,
                memory_gb=32.0,
                network_bandwidth_gbps=50.0,
                num_gpus_per_node=8
            ),
            "A6000": HardwareConfig(
                gpu_model="A6000",
                compute_tflops=154.0,
                memory_gb=48.0,
                network_bandwidth_gbps=25.0,
                num_gpus_per_node=4
            )
        }
        
        if preset_name not in presets:
            logger.warning(f"Unknown preset '{preset_name}', using A100")
            return presets["A100"]
        
        return presets[preset_name]


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_type: str = "transformer"  # "transformer", "resnet", "custom"
    num_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12
    sequence_length: int = 512
    vocab_size: int = 30000
    
    # For custom models
    total_parameters: Optional[int] = None
    total_flops: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        return cls(**data)
    
    @staticmethod
    def get_preset(preset_name: str) -> 'ModelConfig':
        """Get predefined model configurations"""
        presets = {
            "gpt2-small": ModelConfig(
                model_type="transformer",
                num_layers=12,
                hidden_size=768,
                num_attention_heads=12,
                sequence_length=1024,
                vocab_size=50257
            ),
            "gpt2-medium": ModelConfig(
                model_type="transformer",
                num_layers=24,
                hidden_size=1024,
                num_attention_heads=16,
                sequence_length=1024,
                vocab_size=50257
            ),
            "gpt2-large": ModelConfig(
                model_type="transformer",
                num_layers=36,
                hidden_size=1280,
                num_attention_heads=20,
                sequence_length=1024,
                vocab_size=50257
            ),
            "bert-base": ModelConfig(
                model_type="transformer",
                num_layers=12,
                hidden_size=768,
                num_attention_heads=12,
                sequence_length=512,
                vocab_size=30522
            ),
            "bert-large": ModelConfig(
                model_type="transformer",
                num_layers=24,
                hidden_size=1024,
                num_attention_heads=16,
                sequence_length=512,
                vocab_size=30522
            ),
            "resnet50": ModelConfig(
                model_type="resnet",
                num_layers=50,
                total_parameters=25_557_032
            )
        }
        
        if preset_name not in presets:
            logger.warning(f"Unknown preset '{preset_name}', using bert-base")
            return presets["bert-base"]
        
        return presets[preset_name]


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings"""
    batch_size_per_gpu: int = 32
    num_iterations: int = 100
    num_nodes: int = 4
    communication_pattern: str = "ring_allreduce"  # ring_allreduce, tree_allreduce, reduce_scatter
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    gradient_checkpointing: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        return cls(**data)


@dataclass
class NetworkConfig:
    """Network topology configuration"""
    topology_type: str = "full_mesh"  # full_mesh, ring, tree, torus_2d
    base_latency_us: float = 5.0  # microseconds
    switch_latency_us: float = 1.0  # additional latency per switch hop
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkConfig':
        return cls(**data)


@dataclass
class SimulationConfig:
    """Complete simulation configuration"""
    name: str = "default_experiment"
    description: str = ""
    
    hardware: HardwareConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    network: NetworkConfig = None
    
    # Analysis settings
    collect_detailed_stats: bool = True
    enable_visualization: bool = True
    output_directory: str = "./results"
    
    def __post_init__(self):
        if self.hardware is None:
            self.hardware = HardwareConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.network is None:
            self.network = NetworkConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'hardware': self.hardware.to_dict(),
            'model': self.model.to_dict(),
            'training': self.training.to_dict(),
            'network': self.network.to_dict(),
            'collect_detailed_stats': self.collect_detailed_stats,
            'enable_visualization': self.enable_visualization,
            'output_directory': self.output_directory
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Create configuration from dictionary"""
        hardware = HardwareConfig.from_dict(data.get('hardware', {}))
        model = ModelConfig.from_dict(data.get('model', {}))
        training = TrainingConfig.from_dict(data.get('training', {}))
        network = NetworkConfig.from_dict(data.get('network', {}))
        
        return cls(
            name=data.get('name', 'default_experiment'),
            description=data.get('description', ''),
            hardware=hardware,
            model=model,
            training=training,
            network=network,
            collect_detailed_stats=data.get('collect_detailed_stats', True),
            enable_visualization=data.get('enable_visualization', True),
            output_directory=data.get('output_directory', './results')
        )
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'SimulationConfig':
        """Load configuration from YAML file"""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {filepath}")
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load YAML config: {e}")
            raise
    
    @classmethod
    def from_json(cls, filepath: str) -> 'SimulationConfig':
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded configuration from {filepath}")
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load JSON config: {e}")
            raise
    
    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            logger.info(f"Saved configuration to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save YAML config: {e}")
            raise
    
    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Saved configuration to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save JSON config: {e}")
            raise
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Hardware validation
        if self.hardware.compute_tflops <= 0:
            errors.append("compute_tflops must be positive")
        if self.hardware.memory_gb <= 0:
            errors.append("memory_gb must be positive")
        if self.hardware.network_bandwidth_gbps <= 0:
            errors.append("network_bandwidth_gbps must be positive")
        
        # Model validation
        if self.model.model_type not in ["transformer", "resnet", "custom"]:
            errors.append(f"Invalid model_type: {self.model.model_type}")
        if self.model.num_layers <= 0:
            errors.append("num_layers must be positive")
        
        # Training validation
        if self.training.batch_size_per_gpu <= 0:
            errors.append("batch_size_per_gpu must be positive")
        if self.training.num_iterations <= 0:
            errors.append("num_iterations must be positive")
        if self.training.num_nodes <= 0:
            errors.append("num_nodes must be positive")
        
        if errors:
            for error in errors:
                logger.error(f"Validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def summary(self) -> str:
        """Generate human-readable configuration summary"""
        lines = [
            f"Simulation Configuration: {self.name}",
            "=" * 60,
            f"Description: {self.description}",
            "",
            "Hardware:",
            f"  GPU Model: {self.hardware.gpu_model}",
            f"  Compute: {self.hardware.compute_tflops} TFLOPS",
            f"  Memory: {self.hardware.memory_gb} GB",
            f"  Network Bandwidth: {self.hardware.network_bandwidth_gbps} GB/s",
            f"  GPUs per Node: {self.hardware.num_gpus_per_node}",
            "",
            "Model:",
            f"  Type: {self.model.model_type}",
            f"  Layers: {self.model.num_layers}",
        ]
        
        if self.model.model_type == "transformer":
            lines.extend([
                f"  Hidden Size: {self.model.hidden_size}",
                f"  Attention Heads: {self.model.num_attention_heads}",
                f"  Sequence Length: {self.model.sequence_length}",
            ])
        
        lines.extend([
            "",
            "Training:",
            f"  Batch Size per GPU: {self.training.batch_size_per_gpu}",
            f"  Number of Nodes: {self.training.num_nodes}",
            f"  Total GPUs: {self.training.num_nodes * self.hardware.num_gpus_per_node}",
            f"  Iterations: {self.training.num_iterations}",
            f"  Communication: {self.training.communication_pattern}",
            f"  Mixed Precision: {self.training.mixed_precision}",
            "",
            "Network:",
            f"  Topology: {self.network.topology_type}",
            f"  Base Latency: {self.network.base_latency_us} μs",
            "=" * 60
        ])
        
        return "\n".join(lines)


# Example configurations for common scenarios
def create_small_scale_config() -> SimulationConfig:
    """Configuration for small-scale experiments (1-4 GPUs)"""
    return SimulationConfig(
        name="small_scale_experiment",
        description="Small-scale training with 4 GPUs",
        hardware=HardwareConfig.get_preset("A100"),
        model=ModelConfig.get_preset("bert-base"),
        training=TrainingConfig(
            batch_size_per_gpu=32,
            num_iterations=100,
            num_nodes=1,
            communication_pattern="ring_allreduce"
        ),
        network=NetworkConfig(topology_type="full_mesh")
    )


def create_medium_scale_config() -> SimulationConfig:
    """Configuration for medium-scale experiments (8-32 GPUs)"""
    return SimulationConfig(
        name="medium_scale_experiment",
        description="Medium-scale training with 32 GPUs (4 nodes)",
        hardware=HardwareConfig.get_preset("A100"),
        model=ModelConfig.get_preset("gpt2-large"),
        training=TrainingConfig(
            batch_size_per_gpu=16,
            num_iterations=100,
            num_nodes=4,
            communication_pattern="ring_allreduce"
        ),
        network=NetworkConfig(topology_type="full_mesh")
    )


def create_large_scale_config() -> SimulationConfig:
    """Configuration for large-scale experiments (64+ GPUs)"""
    return SimulationConfig(
        name="large_scale_experiment",
        description="Large-scale training with 128 GPUs (16 nodes)",
        hardware=HardwareConfig.get_preset("H100"),
        model=ModelConfig(
            model_type="transformer",
            num_layers=48,
            hidden_size=2048,
            num_attention_heads=32,
            sequence_length=2048,
            vocab_size=50257
        ),
        training=TrainingConfig(
            batch_size_per_gpu=8,
            num_iterations=100,
            num_nodes=16,
            communication_pattern="ring_allreduce",
            mixed_precision=True
        ),
        network=NetworkConfig(topology_type="torus_2d")
    )


def create_scaling_study_configs() -> list[SimulationConfig]:
    """Create a series of configurations for scaling studies"""
    configs = []
    
    base_config = SimulationConfig(
        hardware=HardwareConfig.get_preset("A100"),
        model=ModelConfig.get_preset("gpt2-medium"),
        training=TrainingConfig(
            batch_size_per_gpu=16,
            num_iterations=50,
            communication_pattern="ring_allreduce"
        ),
        network=NetworkConfig(topology_type="full_mesh")
    )
    
    # Scale from 1 to 64 nodes (doubling each time)
    for num_nodes in [1, 2, 4, 8, 16, 32, 64]:
        config = SimulationConfig(
            name=f"scaling_study_{num_nodes}_nodes",
            description=f"Scaling study with {num_nodes} nodes",
            hardware=base_config.hardware,
            model=base_config.model,
            training=TrainingConfig(
                batch_size_per_gpu=base_config.training.batch_size_per_gpu,
                num_iterations=base_config.training.num_iterations,
                num_nodes=num_nodes,
                communication_pattern=base_config.training.communication_pattern
            ),
            network=base_config.network
        )
        configs.append(config)
    
    return configs
