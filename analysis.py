"""
Analysis and Visualization Module
==================================
Provides tools for analyzing simulation results, identifying bottlenecks,
and evaluating scaling efficiency

Author: AI Infrastructure Research Team
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import json
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


@dataclass
class BottleneckAnalysis:
    """Results of bottleneck analysis"""
    bottleneck_type: str  # 'compute', 'memory', 'network'
    severity: float  # 0-1, higher is more severe
    affected_layers: List[int]
    recommendations: List[str]
    metrics: Dict[str, float]


class SimulationAnalyzer:
    """
    Analyzes simulation results to identify performance bottlenecks
    and evaluate scaling efficiency
    """
    
    def __init__(self, simulation_results: Dict[str, Any]):
        self.results = simulation_results
        self.bottlenecks: List[BottleneckAnalysis] = []
        
    def analyze_compute_bottleneck(self, resource_stats: List[Dict]) -> BottleneckAnalysis:
        """
        Analyze if compute resources are bottlenecking performance
        """
        compute_utilizations = []
        for node_stats in resource_stats:
            if 'stats' in node_stats:
                compute_time = node_stats['stats'].get('compute_time', 0)
                total_time = (compute_time + 
                             node_stats['stats'].get('idle_time', 0))
                if total_time > 0:
                    compute_utilizations.append(compute_time / total_time)
        
        avg_utilization = np.mean(compute_utilizations) if compute_utilizations else 0
        
        # High utilization indicates compute bottleneck
        severity = avg_utilization
        
        recommendations = []
        if severity > 0.8:
            recommendations.extend([
                "Consider using more powerful GPUs (e.g., H100 vs A100)",
                "Reduce model size or batch size if memory allows",
                "Enable mixed precision training (FP16/BF16)",
                "Optimize kernel implementations for compute-heavy layers"
            ])
        
        return BottleneckAnalysis(
            bottleneck_type="compute",
            severity=severity,
            affected_layers=[],
            recommendations=recommendations,
            metrics={
                'avg_compute_utilization': avg_utilization,
                'max_compute_utilization': max(compute_utilizations) if compute_utilizations else 0
            }
        )
    
    def analyze_memory_bottleneck(self, resource_stats: List[Dict]) -> BottleneckAnalysis:
        """
        Analyze memory usage and potential OOM issues
        """
        peak_memory_usages = []
        memory_capacities = []
        
        for node_stats in resource_stats:
            if 'stats' in node_stats:
                peak = node_stats['stats'].get('peak_memory_usage', 0)
                capacity = node_stats['capacity'].get('memory_gb', 1)
                peak_memory_usages.append(peak)
                memory_capacities.append(capacity)
        
        if memory_capacities:
            avg_usage_ratio = np.mean([p/c for p, c in zip(peak_memory_usages, memory_capacities)])
            max_usage_ratio = max([p/c for p, c in zip(peak_memory_usages, memory_capacities)])
        else:
            avg_usage_ratio = 0
            max_usage_ratio = 0
        
        # High memory usage indicates potential bottleneck
        severity = max_usage_ratio
        
        recommendations = []
        if severity > 0.9:
            recommendations.extend([
                "CRITICAL: Risk of OOM errors",
                "Reduce batch size immediately",
                "Enable gradient checkpointing",
                "Consider model parallelism or pipeline parallelism"
            ])
        elif severity > 0.7:
            recommendations.extend([
                "High memory pressure detected",
                "Consider reducing batch size",
                "Enable gradient checkpointing for large models",
                "Use activation checkpointing for memory-intensive layers"
            ])
        
        return BottleneckAnalysis(
            bottleneck_type="memory",
            severity=severity,
            affected_layers=[],
            recommendations=recommendations,
            metrics={
                'avg_memory_usage_ratio': avg_usage_ratio,
                'max_memory_usage_ratio': max_usage_ratio,
                'peak_memory_gb': max(peak_memory_usages) if peak_memory_usages else 0
            }
        )
    
    def analyze_network_bottleneck(self, resource_stats: List[Dict],
                                   iteration_stats: List[Dict]) -> BottleneckAnalysis:
        """
        Analyze network communication overhead
        """
        communication_ratios = []
        
        for iter_stat in iteration_stats:
            compute_time = iter_stat.get('compute_time', 0)
            comm_time = iter_stat.get('communication_time', 0)
            total_time = compute_time + comm_time
            
            if total_time > 0:
                communication_ratios.append(comm_time / total_time)
        
        avg_comm_ratio = np.mean(communication_ratios) if communication_ratios else 0
        
        # High communication ratio indicates network bottleneck
        severity = avg_comm_ratio
        
        recommendations = []
        if severity > 0.5:
            recommendations.extend([
                "Network communication is bottleneck (>50% of iteration time)",
                "Use gradient compression (e.g., PowerSGD, 1-bit SGD)",
                "Increase gradient accumulation steps to reduce sync frequency",
                "Consider hierarchical all-reduce for multi-node setups",
                "Upgrade network interconnect (e.g., InfiniBand, NVLink)"
            ])
        elif severity > 0.3:
            recommendations.extend([
                "Moderate network overhead detected",
                "Consider gradient compression techniques",
                "Overlap communication with computation using bucketing"
            ])
        
        return BottleneckAnalysis(
            bottleneck_type="network",
            severity=severity,
            affected_layers=[],
            recommendations=recommendations,
            metrics={
                'avg_communication_ratio': avg_comm_ratio,
                'max_communication_ratio': max(communication_ratios) if communication_ratios else 0
            }
        )
    
    def perform_full_analysis(self, resource_stats: List[Dict],
                            iteration_stats: List[Dict]) -> List[BottleneckAnalysis]:
        """
        Perform comprehensive bottleneck analysis
        """
        self.bottlenecks = [
            self.analyze_compute_bottleneck(resource_stats),
            self.analyze_memory_bottleneck(resource_stats),
            self.analyze_network_bottleneck(resource_stats, iteration_stats)
        ]
        
        # Sort by severity
        self.bottlenecks.sort(key=lambda x: x.severity, reverse=True)
        
        return self.bottlenecks
    
    def generate_report(self, bottlenecks: List[BottleneckAnalysis]) -> str:
        """
        Generate a human-readable analysis report
        """
        report = []
        report.append("="*80)
        report.append("DISTRIBUTED TRAINING PERFORMANCE ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        report.append("BOTTLENECK ANALYSIS (sorted by severity):")
        report.append("-"*80)
        
        for i, bottleneck in enumerate(bottlenecks, 1):
            report.append(f"\n{i}. {bottleneck.bottleneck_type.upper()} BOTTLENECK")
            report.append(f"   Severity: {bottleneck.severity:.2%}")
            report.append(f"   Metrics:")
            for key, value in bottleneck.metrics.items():
                if isinstance(value, float):
                    report.append(f"     - {key}: {value:.4f}")
                else:
                    report.append(f"     - {key}: {value}")
            
            if bottleneck.recommendations:
                report.append(f"   Recommendations:")
                for rec in bottleneck.recommendations:
                    report.append(f"     • {rec}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


class ScalingAnalyzer:
    """
    Analyzes scaling efficiency across different GPU/node counts
    """
    
    def __init__(self):
        self.scaling_data: Dict[int, Dict[str, float]] = {}
        
    def add_run(self, num_gpus: int, iteration_time: float,
                throughput: float, efficiency: float) -> None:
        """Add results from a scaling experiment"""
        self.scaling_data[num_gpus] = {
            'iteration_time': iteration_time,
            'throughput': throughput,
            'efficiency': efficiency
        }
    
    def calculate_scaling_efficiency(self, baseline_gpus: int = 1) -> Dict[int, float]:
        """
        Calculate strong scaling efficiency relative to baseline
        
        Efficiency = (T_baseline * N_baseline) / (T_N * N)
        where T is time per iteration, N is number of GPUs
        """
        if baseline_gpus not in self.scaling_data:
            logger.warning(f"Baseline with {baseline_gpus} GPUs not found")
            return {}
        
        baseline_time = self.scaling_data[baseline_gpus]['iteration_time']
        
        efficiencies = {}
        for num_gpus, data in self.scaling_data.items():
            if num_gpus == baseline_gpus:
                efficiencies[num_gpus] = 1.0
            else:
                ideal_time = baseline_time * baseline_gpus / num_gpus
                actual_time = data['iteration_time']
                efficiencies[num_gpus] = ideal_time / actual_time
        
        return efficiencies
    
    def identify_scaling_breakdown(self) -> Dict[str, Any]:
        """
        Identify where scaling efficiency breaks down
        """
        efficiencies = self.calculate_scaling_efficiency()
        
        if not efficiencies:
            return {}
        
        # Find where efficiency drops below 0.7 (70%)
        breakdown_point = None
        for num_gpus in sorted(efficiencies.keys()):
            if efficiencies[num_gpus] < 0.7:
                breakdown_point = num_gpus
                break
        
        return {
            'breakdown_point': breakdown_point,
            'efficiencies': efficiencies,
            'max_efficient_scale': max(
                (k for k, v in efficiencies.items() if v >= 0.7),
                default=None
            )
        }


class Visualizer:
    """
    Creates visualizations for simulation results
    """
    
    @staticmethod
    def plot_iteration_breakdown(iteration_stats: List[Dict], 
                                save_path: Optional[str] = None) -> None:
        """
        Plot breakdown of compute vs communication time per iteration
        """
        iterations = [s.get('iteration', i) for i, s in enumerate(iteration_stats)]
        compute_times = [s.get('compute_time', 0) for s in iteration_stats]
        comm_times = [s.get('communication_time', 0) for s in iteration_stats]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Stacked bar chart
        ax1.bar(iterations, compute_times, label='Compute', color='#2E86AB')
        ax1.bar(iterations, comm_times, bottom=compute_times, 
               label='Communication', color='#A23B72')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Training Iteration Time Breakdown')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Communication ratio over iterations
        total_times = [c + cm for c, cm in zip(compute_times, comm_times)]
        comm_ratios = [cm / t if t > 0 else 0 
                      for cm, t in zip(comm_times, total_times)]
        
        ax2.plot(iterations, comm_ratios, marker='o', linewidth=2, 
                color='#F18F01', markersize=6)
        ax2.axhline(y=0.3, color='r', linestyle='--', 
                   label='30% threshold', alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Communication Ratio')
        ax2.set_title('Communication Overhead Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved iteration breakdown plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_resource_utilization(resource_stats: List[Dict],
                                  save_path: Optional[str] = None) -> None:
        """
        Plot resource utilization across nodes
        """
        node_ids = [s.get('node_id', i) for i, s in enumerate(resource_stats)]
        
        # Extract utilization data
        compute_util = []
        memory_util = []
        network_util = []
        
        for stats in resource_stats:
            util = stats.get('utilization', {})
            compute_util.append(util.get('compute', 0))
            memory_util.append(util.get('memory', 0))
            network_util.append(util.get('network', 0))
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Compute utilization
        axes[0].bar(node_ids, compute_util, color='#2E86AB')
        axes[0].axhline(y=80, color='orange', linestyle='--', 
                       label='80% target', alpha=0.7)
        axes[0].set_xlabel('Node ID')
        axes[0].set_ylabel('Utilization (%)')
        axes[0].set_title('Compute Utilization')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Memory utilization
        axes[1].bar(node_ids, memory_util, color='#A23B72')
        axes[1].axhline(y=90, color='red', linestyle='--', 
                       label='90% warning', alpha=0.7)
        axes[1].set_xlabel('Node ID')
        axes[1].set_ylabel('Utilization (%)')
        axes[1].set_title('Memory Utilization')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Network utilization
        axes[2].bar(node_ids, network_util, color='#F18F01')
        axes[2].set_xlabel('Node ID')
        axes[2].set_ylabel('Utilization (%)')
        axes[2].set_title('Network Utilization')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved resource utilization plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_scaling_efficiency(scaling_data: Dict[int, Dict[str, float]],
                               save_path: Optional[str] = None) -> None:
        """
        Plot strong scaling efficiency
        """
        num_gpus = sorted(scaling_data.keys())
        efficiencies = [scaling_data[n]['efficiency'] for n in num_gpus]
        iteration_times = [scaling_data[n]['iteration_time'] for n in num_gpus]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scaling efficiency
        ax1.plot(num_gpus, efficiencies, marker='o', linewidth=2, 
                markersize=8, color='#2E86AB')
        ax1.plot(num_gpus, [1.0] * len(num_gpus), 'r--', 
                label='Ideal (100%)', alpha=0.7)
        ax1.axhline(y=0.7, color='orange', linestyle='--', 
                   label='70% threshold', alpha=0.7)
        ax1.set_xlabel('Number of GPUs')
        ax1.set_ylabel('Scaling Efficiency')
        ax1.set_title('Strong Scaling Efficiency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # Iteration time speedup
        baseline_time = iteration_times[0]
        speedups = [baseline_time / t for t in iteration_times]
        ideal_speedup = num_gpus
        
        ax2.plot(num_gpus, speedups, marker='o', linewidth=2, 
                markersize=8, color='#A23B72', label='Actual')
        ax2.plot(num_gpus, ideal_speedup, 'r--', 
                label='Ideal (linear)', alpha=0.7)
        ax2.set_xlabel('Number of GPUs')
        ax2.set_ylabel('Speedup')
        ax2.set_title('Training Speedup')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log', base=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved scaling efficiency plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_layer_timeline(layer_events: List[Dict],
                           save_path: Optional[str] = None) -> None:
        """
        Create a Gantt-chart style timeline of layer execution
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = {'forward': '#2E86AB', 'backward': '#A23B72', 
                 'communication': '#F18F01'}
        
        for event in layer_events:
            layer_id = event.get('layer_id', 0)
            phase = event.get('phase', 'unknown')
            start_time = event.get('start_time', 0)
            duration = event.get('duration', 0)
            
            color = colors.get(phase, '#666666')
            ax.barh(layer_id, duration, left=start_time, 
                   height=0.8, color=color, alpha=0.8)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Layer ID')
        ax.set_title('Layer Execution Timeline')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors['forward'], label='Forward Pass'),
            Patch(facecolor=colors['backward'], label='Backward Pass'),
            Patch(facecolor=colors['communication'], label='Communication')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved layer timeline plot to {save_path}")
        else:
            plt.show()
        
        plt.close()


def export_results_to_json(results: Dict[str, Any], 
                           filepath: str) -> None:
    """Export simulation results to JSON file"""
    try:
        # Convert any Enum types to strings for JSON serialization
        def convert_to_serializable(obj):
            if hasattr(obj, '__dict__'):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: (v.value if hasattr(v, 'value') else v) for k, v in obj.items()}
            return str(obj)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=convert_to_serializable)
        logger.info(f"Exported results to {filepath}")
    except Exception as e:
        logger.error(f"Failed to export results: {e}")


def export_bottleneck_analysis(bottlenecks: List[BottleneckAnalysis],
                               filepath: str) -> None:
    """Export bottleneck analysis to JSON file"""
    try:
        data = [asdict(b) for b in bottlenecks]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported bottleneck analysis to {filepath}")
    except Exception as e:
        logger.error(f"Failed to export bottleneck analysis: {e}")