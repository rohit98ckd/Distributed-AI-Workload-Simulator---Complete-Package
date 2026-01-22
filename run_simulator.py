#!/usr/bin/env python3
"""
Example Usage and CLI Entry Point
==================================
Demonstrates how to use the simulator with various scenarios

Author: AI Infrastructure Research Team
"""

import argparse
import logging
import sys
from pathlib import Path

from config import (
    SimulationConfig,
    create_small_scale_config,
    create_medium_scale_config,
    create_large_scale_config,
    create_scaling_study_configs
)
from main_simulator import DistributedAISimulator, ScalingStudyRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('simulator.log')
    ]
)

logger = logging.getLogger(__name__)


def example_basic_simulation():
    """
    Example 1: Basic single simulation
    Runs a simple distributed training simulation with default parameters
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Simulation")
    print("="*80 + "\n")
    
    # Create configuration
    config = create_small_scale_config()
    
    # Initialize and run simulator
    simulator = DistributedAISimulator(config)
    results = simulator.run()
    
    # Print summary
    print(simulator.get_summary())
    
    # Analyze bottlenecks
    simulator.analyze()
    
    # Generate visualizations
    simulator.visualize()
    
    # Export results
    simulator.export_results()
    
    return simulator


def example_custom_configuration():
    """
    Example 2: Custom configuration
    Shows how to create and customize a simulation configuration
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Configuration")
    print("="*80 + "\n")
    
    from config import HardwareConfig, ModelConfig, TrainingConfig, NetworkConfig
    
    # Create custom configuration
    config = SimulationConfig(
        name="custom_llm_training",
        description="Custom LLM training with 64 GPUs",
        hardware=HardwareConfig(
            gpu_model="H100",
            compute_tflops=989.0,
            memory_gb=80.0,
            network_bandwidth_gbps=200.0,
            num_gpus_per_node=8
        ),
        model=ModelConfig(
            model_type="transformer",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            sequence_length=2048,
            vocab_size=50257
        ),
        training=TrainingConfig(
            batch_size_per_gpu=4,
            num_iterations=50,
            num_nodes=8,  # 64 GPUs total
            communication_pattern="ring_allreduce",
            mixed_precision=True
        ),
        network=NetworkConfig(
            topology_type="torus_2d",
            base_latency_us=5.0
        ),
        output_directory="./results/custom_llm"
    )
    
    # Run simulation
    simulator = DistributedAISimulator(config)
    results = simulator.run()
    
    print(simulator.get_summary())
    simulator.analyze()
    simulator.visualize()
    simulator.export_results()
    
    return simulator


def example_scaling_study():
    """
    Example 3: Scaling study
    Runs multiple simulations to analyze scaling behavior
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Scaling Study")
    print("="*80 + "\n")
    
    # Create base configuration
    base_config = create_medium_scale_config()
    base_config.training.num_iterations = 20  # Fewer iterations for speed
    base_config.output_directory = "./results/scaling_study"
    
    # Initialize scaling study
    study = ScalingStudyRunner(base_config)
    
    # Run with different node counts
    node_counts = [1, 2, 4, 8, 16]
    results = study.run_scaling_study(node_counts)
    
    # Generate visualizations
    study.visualize(base_config.output_directory)
    
    return study


def example_bottleneck_comparison():
    """
    Example 4: Compare different communication patterns
    Analyzes how different communication strategies affect performance
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Communication Pattern Comparison")
    print("="*80 + "\n")
    
    patterns = ["ring_allreduce", "reduce_scatter"]
    results = {}
    
    for pattern in patterns:
        config = create_medium_scale_config()
        config.name = f"comparison_{pattern}"
        config.training.communication_pattern = pattern
        config.training.num_iterations = 30
        config.output_directory = f"./results/comparison_{pattern}"
        
        print(f"\nTesting {pattern}...")
        simulator = DistributedAISimulator(config)
        sim_results = simulator.run()
        
        results[pattern] = {
            'avg_iteration_time': sim_results['summary']['avg_iteration_time'],
            'communication_overhead': sim_results['summary']['communication_overhead_ratio'],
            'throughput': sim_results['summary']['throughput_samples_per_sec']
        }
        
        simulator.analyze()
        simulator.visualize()
        simulator.export_results()
    
    # Print comparison
    print("\n" + "="*80)
    print("COMMUNICATION PATTERN COMPARISON")
    print("="*80)
    print(f"{'Pattern':<20} {'Avg Time (s)':<15} {'Comm Overhead':<15} {'Throughput':<15}")
    print("-"*80)
    for pattern, metrics in results.items():
        print(f"{pattern:<20} "
              f"{metrics['avg_iteration_time']:<15.6f} "
              f"{metrics['communication_overhead']:<15.2%} "
              f"{metrics['throughput']:<15.2f}")
    print("="*80)
    
    return results


def example_load_from_config():
    """
    Example 5: Load configuration from file
    Shows how to use YAML/JSON configuration files
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Load from Configuration File")
    print("="*80 + "\n")
    
    # First, create and save a configuration
    config = create_large_scale_config()
    config_path = "./config_example.yaml"
    config.to_yaml(config_path)
    print(f"Created example configuration: {config_path}")
    
    # Load and run from file
    loaded_config = SimulationConfig.from_yaml(config_path)
    loaded_config.training.num_iterations = 10  # Reduce for demo
    
    simulator = DistributedAISimulator(loaded_config)
    results = simulator.run()
    
    print(simulator.get_summary())
    
    return simulator


def main():
    """
    Main CLI entry point
    """
    parser = argparse.ArgumentParser(
        description='Distributed AI Workload Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with preset configuration
  %(prog)s --preset small
  
  # Load from config file
  %(prog)s --config my_config.yaml
  
  # Run scaling study
  %(prog)s --scaling --nodes 1 2 4 8 16
  
  # Run specific example
  %(prog)s --example 1
        """
    )
    
    parser.add_argument(
        '--preset',
        choices=['small', 'medium', 'large'],
        help='Use preset configuration'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML or JSON configuration file'
    )
    
    parser.add_argument(
        '--scaling',
        action='store_true',
        help='Run scaling study'
    )
    
    parser.add_argument(
        '--nodes',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8],
        help='Node counts for scaling study (default: 1 2 4 8)'
    )
    
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run specific example (1-5)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run appropriate mode
    try:
        if args.example:
            # Run specific example
            examples = {
                1: example_basic_simulation,
                2: example_custom_configuration,
                3: example_scaling_study,
                4: example_bottleneck_comparison,
                5: example_load_from_config
            }
            examples[args.example]()
            
        elif args.scaling:
            # Run scaling study
            if args.config:
                config = SimulationConfig.from_yaml(args.config)
            elif args.preset:
                preset_configs = {
                    'small': create_small_scale_config,
                    'medium': create_medium_scale_config,
                    'large': create_large_scale_config
                }
                config = preset_configs[args.preset]()
            else:
                config = create_medium_scale_config()
            
            config.output_directory = args.output
            study = ScalingStudyRunner(config)
            study.run_scaling_study(args.nodes)
            
            if not args.no_visualize:
                study.visualize(args.output)
                
        elif args.config:
            # Load from config file
            config = SimulationConfig.from_yaml(args.config)
            config.output_directory = args.output
            
            simulator = DistributedAISimulator(config)
            simulator.run()
            print(simulator.get_summary())
            simulator.analyze()
            
            if not args.no_visualize:
                simulator.visualize()
            
            simulator.export_results()
            
        elif args.preset:
            # Use preset configuration
            preset_configs = {
                'small': create_small_scale_config,
                'medium': create_medium_scale_config,
                'large': create_large_scale_config
            }
            config = preset_configs[args.preset]()
            config.output_directory = args.output
            
            simulator = DistributedAISimulator(config)
            simulator.run()
            print(simulator.get_summary())
            simulator.analyze()
            
            if not args.no_visualize:
                simulator.visualize()
            
            simulator.export_results()
            
        else:
            # Default: run basic example
            print("No arguments provided. Running basic example...")
            print("Use --help for more options\n")
            example_basic_simulation()
    
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Simulation completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
