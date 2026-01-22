# Distributed AI Workload Simulator

A production-level simulator for modeling distributed AI training workloads, focusing on data-parallel training with detailed analysis of compute, memory, and network bottlenecks.

## 🎯 Features

- **Event-Driven Simulation**: High-performance discrete event simulation engine
- **Distributed Training Models**: Support for data parallelism with various communication patterns
- **Multiple Architectures**: Pre-configured models (ResNet-50, Transformers, GPT-2, BERT)
- **Communication Patterns**: Ring all-reduce, tree all-reduce, reduce-scatter, all-gather
- **Bottleneck Analysis**: Automatic identification of compute, memory, and network bottlenecks
- **Scaling Studies**: Comprehensive scaling efficiency analysis across GPU counts
- **Visualization**: Publication-quality plots for performance analysis
- **Flexible Configuration**: YAML/JSON-based configuration system
- **Hardware Profiles**: Pre-configured profiles for A100, H100, V100, A6000 GPUs

## 📋 Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Seaborn
- PyYAML

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```python
from config import create_small_scale_config
from main_simulator import DistributedAISimulator

# Create configuration
config = create_small_scale_config()

# Run simulation
simulator = DistributedAISimulator(config)
results = simulator.run()

# Analyze results
print(simulator.get_summary())
simulator.analyze()
simulator.visualize()
simulator.export_results()
```

### Command Line Interface

```bash
# Run with preset configuration
python run_simulator.py --preset medium

# Run from config file
python run_simulator.py --config my_config.yaml

# Run scaling study
python run_simulator.py --scaling --nodes 1 2 4 8 16

# Run specific example
python run_simulator.py --example 1
```

## 📊 Examples

### Example 1: Basic Simulation

```python
from config import create_small_scale_config
from main_simulator import DistributedAISimulator

config = create_small_scale_config()
simulator = DistributedAISimulator(config)
results = simulator.run()
simulator.analyze()
```

### Example 2: Custom Configuration

```python
from config import SimulationConfig, HardwareConfig, ModelConfig, TrainingConfig

config = SimulationConfig(
    name="custom_experiment",
    hardware=HardwareConfig(
        gpu_model="H100",
        compute_tflops=989.0,
        memory_gb=80.0,
        network_bandwidth_gbps=200.0
    ),
    model=ModelConfig(
        model_type="transformer",
        num_layers=32,
        hidden_size=4096
    ),
    training=TrainingConfig(
        batch_size_per_gpu=8,
        num_iterations=100,
        num_nodes=8
    )
)

simulator = DistributedAISimulator(config)
results = simulator.run()
```

### Example 3: Scaling Study

```python
from config import create_medium_scale_config
from main_simulator import ScalingStudyRunner

base_config = create_medium_scale_config()
study = ScalingStudyRunner(base_config)

# Test scaling from 1 to 16 nodes
results = study.run_scaling_study([1, 2, 4, 8, 16])
study.visualize("./results/scaling")
```

### Example 4: Configuration from File

Create `config.yaml`:
```yaml
name: "my_experiment"
description: "Large-scale LLM training"

hardware:
  gpu_model: "H100"
  compute_tflops: 989.0
  memory_gb: 80.0
  network_bandwidth_gbps: 200.0
  num_gpus_per_node: 8

model:
  model_type: "transformer"
  num_layers: 48
  hidden_size: 2048
  num_attention_heads: 32
  sequence_length: 2048

training:
  batch_size_per_gpu: 4
  num_iterations: 100
  num_nodes: 16
  communication_pattern: "ring_allreduce"
  mixed_precision: true

network:
  topology_type: "torus_2d"
  base_latency_us: 5.0
```

Load and run:
```python
from config import SimulationConfig
from main_simulator import DistributedAISimulator

config = SimulationConfig.from_yaml("config.yaml")
simulator = DistributedAISimulator(config)
results = simulator.run()
```

## 🏗️ Architecture

### Core Components

1. **SimulationEngine** (`simulator_core.py`)
   - Discrete event simulation with priority queue
   - Event scheduling and time progression
   - Statistics collection

2. **ResourceManager** (`simulator_core.py`)
   - Compute, memory, and network resource tracking
   - Utilization monitoring
   - Capacity constraint enforcement

3. **TopologyManager** (`simulator_core.py`)
   - Network topology modeling (mesh, ring, tree, torus)
   - Routing and latency calculation

4. **ModelArchitecture** (`training_workload.py`)
   - Neural network layer definitions
   - FLOPs and memory calculation
   - Pre-configured model architectures

5. **DistributedTrainingWorkload** (`training_workload.py`)
   - Forward/backward pass scheduling
   - Gradient synchronization
   - Communication pattern implementation

6. **SimulationAnalyzer** (`analysis.py`)
   - Bottleneck detection
   - Performance analysis
   - Recommendation generation

7. **ScalingAnalyzer** (`analysis.py`)
   - Scaling efficiency calculation
   - Breakdown point identification

8. **Visualizer** (`analysis.py`)
   - Performance plotting
   - Resource utilization visualization
   - Scaling efficiency charts

## 📈 Analysis Capabilities

### Bottleneck Detection

The simulator automatically identifies three types of bottlenecks:

1. **Compute Bottleneck**
   - High GPU utilization (>80%)
   - Recommendations: Upgrade GPUs, reduce model size, use mixed precision

2. **Memory Bottleneck**
   - High memory usage (>70%)
   - Recommendations: Reduce batch size, enable gradient checkpointing

3. **Network Bottleneck**
   - High communication overhead (>30%)
   - Recommendations: Gradient compression, increase accumulation steps

### Scaling Efficiency

- Strong scaling analysis
- Efficiency breakdown detection
- Optimal scale recommendations

### Performance Metrics

- Iteration time breakdown
- Throughput (samples/second)
- Communication overhead ratio
- Compute efficiency
- Resource utilization

## 🔧 Configuration

### Hardware Presets

- **A100**: 312 TFLOPS, 80GB memory, 100 GB/s network
- **H100**: 989 TFLOPS, 80GB memory, 200 GB/s network
- **V100**: 125 TFLOPS, 32GB memory, 50 GB/s network
- **A6000**: 154 TFLOPS, 48GB memory, 25 GB/s network

### Model Presets

- **ResNet-50**: 25.5M parameters
- **GPT-2 Small**: 117M parameters, 12 layers
- **GPT-2 Medium**: 345M parameters, 24 layers
- **GPT-2 Large**: 774M parameters, 36 layers
- **BERT Base**: 110M parameters, 12 layers
- **BERT Large**: 340M parameters, 24 layers

### Communication Patterns

- **Ring All-Reduce**: Efficient for large clusters
- **Tree All-Reduce**: Lower latency for small clusters
- **Reduce-Scatter**: Memory-efficient gradient aggregation
- **All-Gather**: Used with reduce-scatter

### Network Topologies

- **Full Mesh**: All-to-all connectivity
- **Ring**: Circular topology
- **Tree**: Hierarchical topology
- **2D Torus**: Grid with wraparound

## 📊 Output Files

The simulator generates several output files:

```
results/
├── {experiment_name}_config.yaml      # Configuration used
├── {experiment_name}_results.json     # Detailed results
├── {experiment_name}_bottlenecks.json # Bottleneck analysis
├── iteration_breakdown.png            # Compute vs communication
├── resource_utilization.png           # GPU/memory/network usage
└── scaling_efficiency.png             # Scaling performance
```

## 🔬 Advanced Usage

### Custom Model Architecture

```python
from training_workload import ModelArchitecture, LayerConfig, LayerType

model = ModelArchitecture("MyCustomModel")

# Add custom layers
model.add_layer(LayerConfig(
    layer_id=0,
    layer_type=LayerType.CONV2D,
    input_shape=(3, 224, 224),
    output_shape=(64, 112, 112),
    num_parameters=9408,
    forward_flops=118e9,
    backward_flops=236e9,
    activation_memory_mb=50,
    gradient_memory_mb=0.04
))
```

### Custom Communication Pattern

Extend `DistributedTrainingWorkload` to implement custom communication:

```python
def _schedule_custom_allreduce(self, engine, resources, start_time, iteration_id):
    # Implement custom communication logic
    pass
```

### Collecting Detailed Events

```python
config.collect_detailed_stats = True
simulator = DistributedAISimulator(config)
results = simulator.run()

# Access event history
for event in simulator.engine.event_history:
    print(f"Event: {event.event_type} at t={event.timestamp}")
```

## 🧪 Testing

Run the included examples to verify installation:

```bash
python run_simulator.py --example 1  # Basic simulation
python run_simulator.py --example 2  # Custom config
python run_simulator.py --example 3  # Scaling study
python run_simulator.py --example 4  # Pattern comparison
python run_simulator.py --example 5  # Config from file
```

## 📚 API Reference

### Main Classes

- `SimulationEngine`: Core event simulator
- `DistributedAISimulator`: High-level simulator interface
- `SimulationConfig`: Configuration container
- `ScalingStudyRunner`: Multi-run scaling analysis

### Key Methods

- `simulator.run()`: Execute simulation
- `simulator.analyze()`: Perform bottleneck analysis
- `simulator.visualize()`: Generate plots
- `simulator.export_results()`: Save results to files
- `simulator.get_summary()`: Get formatted summary

## 🎓 Use Cases

1. **Hardware Planning**: Evaluate GPU requirements for training workloads
2. **Network Design**: Compare interconnect topologies and bandwidths
3. **Scaling Studies**: Determine optimal cluster size for your workload
4. **Optimization**: Identify and address performance bottlenecks
5. **Cost Analysis**: Estimate training time and resource costs
6. **Research**: Experiment with new communication algorithms

## 🤝 Contributing

This is a research/educational tool. Contributions welcome:

- Bug reports and fixes
- New model architectures
- Additional communication patterns
- Analysis features
- Documentation improvements

## 📄 License

MIT License - feel free to use and modify for your needs.

## 🙏 Acknowledgments

Inspired by:
- **Astra-Sim**: Analytical simulator for distributed training
- **ns-3**: Network simulator framework
- Production ML systems at scale

## 📧 Support

For questions and issues:
- Check the examples in `run_simulator.py`
- Review configuration options in `config.py`
- Examine the analysis output and recommendations

## 🗺️ Roadmap

Future enhancements:
- [ ] Pipeline parallelism support
- [ ] Tensor parallelism modeling
- [ ] Heterogeneous clusters
- [ ] Fault tolerance simulation
- [ ] Cost modeling (cloud pricing)
- [ ] Real workload trace replay
- [ ] GPU memory fragmentation
- [ ] Dynamic batch sizing
- [ ] Gradient compression algorithms
- [ ] Multi-job scheduling

---

**Happy Simulating!** 🚀
