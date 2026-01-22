# Quick Start Guide
## Distributed AI Workload Simulator

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python test_simulator.py
```

### Your First Simulation

```bash
# Run a basic example
python run_simulator.py --preset small
```

This will:
- Simulate training on 4 GPUs
- Use BERT-base model
- Run 100 iterations
- Generate performance analysis
- Create visualizations in `./results/`

### View Results

After running, check the `./results/` directory for:
- `*_results.json` - Detailed metrics
- `*_config.yaml` - Configuration used
- `iteration_breakdown.png` - Compute vs communication time
- `resource_utilization.png` - GPU/memory/network usage

### Common Commands

```bash
# Small scale (4 GPUs)
python run_simulator.py --preset small

# Medium scale (32 GPUs)
python run_simulator.py --preset medium

# Large scale (128 GPUs)  
python run_simulator.py --preset large

# Scaling study
python run_simulator.py --scaling --nodes 1 2 4 8 16

# Custom configuration
python run_simulator.py --config example_config.yaml

# All examples
python run_simulator.py --example 1  # Basic
python run_simulator.py --example 2  # Custom config
python run_simulator.py --example 3  # Scaling study
python run_simulator.py --example 4  # Pattern comparison
python run_simulator.py --example 5  # Load from file
```

### Python API

```python
from config import create_small_scale_config
from main_simulator import DistributedAISimulator

# Create and customize config
config = create_small_scale_config()
config.training.num_iterations = 50
config.training.batch_size_per_gpu = 16

# Run simulation
simulator = DistributedAISimulator(config)
results = simulator.run()

# Analyze and visualize
print(simulator.get_summary())
simulator.analyze()
simulator.visualize()
simulator.export_results()
```

### Key Metrics Explained

**Iteration Time**: Total time per training iteration (forward + backward + communication)

**Throughput**: Samples processed per second (higher is better)

**Communication Overhead**: % of time spent in communication (lower is better, <30% is good)

**Scaling Efficiency**: How close to ideal speedup (1.0 = perfect, >0.7 is acceptable)

**Compute Utilization**: % of GPU capacity used (>80% indicates compute-bound)

**Memory Usage**: Peak memory as % of capacity (>90% risks OOM errors)

### Troubleshooting

**Simulation too slow?**
- Reduce `num_iterations` for testing
- Disable `collect_detailed_stats` in config
- Use `--no-visualize` flag

**Want more detail?**
- Enable `collect_detailed_stats: true`
- Use `--verbose` flag
- Check `simulator.log` file

**Need custom hardware?**
```python
from config import HardwareConfig

custom_gpu = HardwareConfig(
    gpu_model="CustomGPU",
    compute_tflops=500.0,
    memory_gb=96.0,
    network_bandwidth_gbps=150.0
)
```

### Next Steps

1. **Read README.md** for comprehensive documentation
2. **Check TECHNICAL_DOCS.md** for implementation details
3. **Run examples** in `run_simulator.py`
4. **Customize** configs for your workload
5. **Extend** with custom models or patterns

### Getting Help

- Run with `--help` for CLI options
- Check example configurations in `example_config.yaml`
- Review code comments for implementation details
- Examine test cases in `test_simulator.py`

### Performance Tips

✅ **DO:**
- Start with small scale for testing
- Use presets for common scenarios
- Enable visualization for insights
- Run scaling studies to find optimal size

❌ **DON'T:**
- Simulate >1000 iterations (use sampling)
- Enable detailed stats for large runs
- Ignore bottleneck warnings

### Example Workflow

```bash
# 1. Start small
python run_simulator.py --preset small

# 2. Customize for your model
# Edit example_config.yaml with your parameters

# 3. Run with custom config
python run_simulator.py --config example_config.yaml

# 4. Analyze bottlenecks
# Check the generated report and recommendations

# 5. Test scaling
python run_simulator.py --scaling --nodes 1 2 4 8 16

# 6. Optimize based on results
# Adjust batch size, communication pattern, etc.

# 7. Export final results
# Results are automatically saved to ./results/
```

Happy simulating! 🚀
