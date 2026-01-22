"""
Distributed Training Workload Module
=====================================
Models neural network training workloads with data parallelism
Includes forward pass, backward pass, and gradient synchronization

Author: AI Infrastructure Research Team
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
from enum import Enum

from simulator_core import (
    Event, EventType, SimulationEngine, ResourceManager,
    CommunicationPattern
)

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Neural network layer types"""
    CONV2D = "conv2d"
    LINEAR = "linear"
    ATTENTION = "attention"
    NORM = "normalization"
    ACTIVATION = "activation"
    POOLING = "pooling"
    EMBEDDING = "embedding"


@dataclass
class LayerConfig:
    """Configuration for a neural network layer"""
    layer_id: int
    layer_type: LayerType
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    num_parameters: int
    forward_flops: float
    backward_flops: float
    activation_memory_mb: float
    gradient_memory_mb: float
    
    def get_parameter_size_gb(self) -> float:
        """Get parameter size in GB (assuming FP32)"""
        return (self.num_parameters * 4) / (1024 ** 3)
    
    def get_gradient_size_gb(self) -> float:
        """Get gradient size in GB"""
        return self.gradient_memory_mb / 1024


class ModelArchitecture:
    """
    Defines a neural network architecture for simulation
    Provides layer-by-layer compute and memory characteristics
    """
    
    def __init__(self, name: str = "CustomModel"):
        self.name = name
        self.layers: List[LayerConfig] = []
        self.total_parameters = 0
        self.total_flops = 0
        
    def add_layer(self, layer: LayerConfig) -> None:
        """Add a layer to the model"""
        self.layers.append(layer)
        self.total_parameters += layer.num_parameters
        self.total_flops += layer.forward_flops + layer.backward_flops
        
    def get_total_parameter_size_gb(self) -> float:
        """Get total model size in GB"""
        return (self.total_parameters * 4) / (1024 ** 3)
    
    def get_layer_by_id(self, layer_id: int) -> Optional[LayerConfig]:
        """Retrieve a specific layer by ID"""
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        return None
    
    @staticmethod
    def create_resnet50() -> 'ModelArchitecture':
        """Create a ResNet-50 model configuration"""
        model = ModelArchitecture("ResNet-50")
        
        # Simplified ResNet-50 with key layers
        layers_config = [
            # Conv1
            (LayerType.CONV2D, (3, 224, 224), (64, 112, 112), 9408, 118e9, 236e9, 50, 0.04),
            # Layer 1
            (LayerType.CONV2D, (64, 56, 56), (256, 56, 56), 4096, 115e9, 230e9, 200, 0.02),
            # Layer 2
            (LayerType.CONV2D, (256, 28, 28), (512, 28, 28), 524288, 115e9, 230e9, 100, 2.0),
            # Layer 3
            (LayerType.CONV2D, (512, 14, 14), (1024, 14, 14), 2097152, 115e9, 230e9, 50, 8.0),
            # Layer 4
            (LayerType.CONV2D, (1024, 7, 7), (2048, 7, 7), 8388608, 115e9, 230e9, 25, 32.0),
            # FC
            (LayerType.LINEAR, (2048,), (1000,), 2048000, 4e9, 8e9, 10, 8.0),
        ]
        
        for i, (ltype, inp, out, params, fflops, bflops, act_mem, grad_mem) in enumerate(layers_config):
            model.add_layer(LayerConfig(
                layer_id=i,
                layer_type=ltype,
                input_shape=inp,
                output_shape=out,
                num_parameters=params,
                forward_flops=fflops,
                backward_flops=bflops,
                activation_memory_mb=act_mem,
                gradient_memory_mb=grad_mem
            ))
        
        return model
    
    @staticmethod
    def create_transformer(num_layers: int = 12, hidden_size: int = 768,
                          num_heads: int = 12, seq_length: int = 512) -> 'ModelArchitecture':
        """Create a Transformer model configuration"""
        model = ModelArchitecture(f"Transformer-{num_layers}L")
        
        # Calculate per-layer parameters and FLOPs
        attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
        ffn_params = 8 * hidden_size * hidden_size  # 2 linear layers with 4x expansion
        layer_params = attention_params + ffn_params + 2 * hidden_size  # + layer norms
        
        # FLOPs estimation (simplified)
        attention_flops = 4 * seq_length * hidden_size * hidden_size + \
                         2 * seq_length * seq_length * hidden_size
        ffn_flops = 16 * seq_length * hidden_size * hidden_size
        layer_flops = attention_flops + ffn_flops
        
        for i in range(num_layers):
            model.add_layer(LayerConfig(
                layer_id=i,
                layer_type=LayerType.ATTENTION,
                input_shape=(seq_length, hidden_size),
                output_shape=(seq_length, hidden_size),
                num_parameters=layer_params,
                forward_flops=layer_flops,
                backward_flops=2 * layer_flops,  # Backward is ~2x forward
                activation_memory_mb=seq_length * hidden_size * 4 / (1024 ** 2),
                gradient_memory_mb=layer_params * 4 / (1024 ** 2)
            ))
        
        # Add embedding and output layers
        vocab_size = 30000
        model.add_layer(LayerConfig(
            layer_id=num_layers,
            layer_type=LayerType.EMBEDDING,
            input_shape=(seq_length,),
            output_shape=(seq_length, hidden_size),
            num_parameters=vocab_size * hidden_size,
            forward_flops=0,  # Lookup operation
            backward_flops=vocab_size * hidden_size * seq_length,
            activation_memory_mb=seq_length * hidden_size * 4 / (1024 ** 2),
            gradient_memory_mb=vocab_size * hidden_size * 4 / (1024 ** 2)
        ))
        
        return model


class DistributedTrainingWorkload:
    """
    Simulates distributed data-parallel training workload
    Handles forward/backward passes and gradient synchronization
    """
    
    def __init__(self, 
                 model: ModelArchitecture,
                 num_nodes: int,
                 batch_size_per_gpu: int,
                 num_iterations: int,
                 communication_pattern: CommunicationPattern = CommunicationPattern.RING_ALLREDUCE):
        self.model = model
        self.num_nodes = num_nodes
        self.batch_size_per_gpu = batch_size_per_gpu
        self.num_iterations = num_iterations
        self.communication_pattern = communication_pattern
        
        # Training state
        self.current_iteration = 0
        self.node_states: Dict[int, Dict] = {}
        
        # Timing statistics
        self.iteration_times: List[float] = []
        self.compute_times: List[float] = []
        self.communication_times: List[float] = []
        self.layer_compute_times: Dict[int, List[float]] = {
            i: [] for i in range(len(model.layers))
        }
        
        # Initialize node states
        for node_id in range(num_nodes):
            self.node_states[node_id] = {
                'current_layer': 0,
                'phase': 'forward',  # 'forward', 'backward', 'communication'
                'iteration_start_time': 0.0,
                'compute_start_time': 0.0,
                'comm_start_time': 0.0
            }
    
    def schedule_iteration(self, engine: SimulationEngine, 
                          resources: Dict[int, ResourceManager],
                          iteration_id: int) -> None:
        """
        Schedule all events for a complete training iteration
        Includes forward pass, backward pass, and gradient synchronization
        """
        start_time = engine.current_time
        
        # Schedule iteration start event for all nodes
        for node_id in range(self.num_nodes):
            event = Event(
                timestamp=start_time,
                event_type=EventType.ITERATION_START,
                node_id=node_id,
                data={'iteration': iteration_id}
            )
            engine.schedule_event(event)
        
        # Schedule forward pass for all layers
        current_time = start_time
        for layer in self.model.layers:
            self._schedule_forward_pass(
                engine, resources, layer, current_time, iteration_id
            )
            # Compute execution time for this layer
            layer_compute_time = resources[0].compute_execution_time(layer.forward_flops)
            current_time += layer_compute_time
        
        # Schedule backward pass (reverse order)
        backward_start = current_time
        for layer in reversed(self.model.layers):
            self._schedule_backward_pass(
                engine, resources, layer, current_time, iteration_id
            )
            layer_compute_time = resources[0].compute_execution_time(layer.backward_flops)
            current_time += layer_compute_time
        
        # Schedule gradient synchronization (all-reduce or reduce-scatter)
        comm_start = current_time
        self._schedule_gradient_synchronization(
            engine, resources, current_time, iteration_id
        )
        
        # Calculate communication time based on total gradient size
        total_gradient_size = sum(layer.get_gradient_size_gb() for layer in self.model.layers)
        comm_time = self._calculate_allreduce_time(total_gradient_size, resources[0])
        current_time += comm_time
        
        # Store iteration timing for analysis
        iteration_stats = {
            'iteration': iteration_id,
            'iteration_time': current_time - start_time,
            'compute_time': comm_start - start_time,
            'communication_time': comm_time
        }
        
        # Schedule iteration end with callback to collect stats
        def collect_stats(event):
            if hasattr(self, '_parent_workload'):
                self._parent_workload.iteration_stats.append(iteration_stats)
        
        for node_id in range(self.num_nodes):
            event = Event(
                timestamp=current_time,
                event_type=EventType.ITERATION_END,
                node_id=node_id,
                data=iteration_stats,
                callback=collect_stats if node_id == 0 else None
            )
            engine.schedule_event(event)
        
        logger.info(
            f"Scheduled iteration {iteration_id}: "
            f"compute={comm_start - start_time:.6f}s, "
            f"comm={comm_time:.6f}s, "
            f"total={current_time - start_time:.6f}s"
        )
    
    def _schedule_forward_pass(self, engine: SimulationEngine,
                               resources: Dict[int, ResourceManager],
                               layer: LayerConfig,
                               start_time: float,
                               iteration_id: int) -> None:
        """Schedule forward pass computation for a layer"""
        for node_id in range(self.num_nodes):
            # Compute start
            event = Event(
                timestamp=start_time,
                event_type=EventType.COMPUTE_START,
                node_id=node_id,
                layer_id=layer.layer_id,
                data={
                    'phase': 'forward',
                    'iteration': iteration_id,
                    'flops': layer.forward_flops
                }
            )
            engine.schedule_event(event)
            
            # Compute end
            compute_time = resources[node_id].compute_execution_time(layer.forward_flops)
            event = Event(
                timestamp=start_time + compute_time,
                event_type=EventType.COMPUTE_END,
                node_id=node_id,
                layer_id=layer.layer_id,
                data={
                    'phase': 'forward',
                    'iteration': iteration_id,
                    'compute_time': compute_time
                }
            )
            engine.schedule_event(event)
    
    def _schedule_backward_pass(self, engine: SimulationEngine,
                                resources: Dict[int, ResourceManager],
                                layer: LayerConfig,
                                start_time: float,
                                iteration_id: int) -> None:
        """Schedule backward pass computation for a layer"""
        for node_id in range(self.num_nodes):
            # Compute start
            event = Event(
                timestamp=start_time,
                event_type=EventType.COMPUTE_START,
                node_id=node_id,
                layer_id=layer.layer_id,
                data={
                    'phase': 'backward',
                    'iteration': iteration_id,
                    'flops': layer.backward_flops
                }
            )
            engine.schedule_event(event)
            
            # Compute end
            compute_time = resources[node_id].compute_execution_time(layer.backward_flops)
            event = Event(
                timestamp=start_time + compute_time,
                event_type=EventType.COMPUTE_END,
                node_id=node_id,
                layer_id=layer.layer_id,
                data={
                    'phase': 'backward',
                    'iteration': iteration_id,
                    'compute_time': compute_time
                }
            )
            engine.schedule_event(event)
    
    def _schedule_gradient_synchronization(self, engine: SimulationEngine,
                                          resources: Dict[int, ResourceManager],
                                          start_time: float,
                                          iteration_id: int) -> None:
        """Schedule gradient synchronization across all nodes"""
        if self.communication_pattern == CommunicationPattern.RING_ALLREDUCE:
            self._schedule_ring_allreduce(engine, resources, start_time, iteration_id)
        elif self.communication_pattern == CommunicationPattern.REDUCE_SCATTER:
            self._schedule_reduce_scatter(engine, resources, start_time, iteration_id)
        else:
            # Default to simple all-reduce
            total_gradient_size = sum(
                layer.get_gradient_size_gb() for layer in self.model.layers
            )
            
            for node_id in range(self.num_nodes):
                event = Event(
                    timestamp=start_time,
                    event_type=EventType.ALLREDUCE_START,
                    node_id=node_id,
                    data={
                        'iteration': iteration_id,
                        'data_size_gb': total_gradient_size
                    }
                )
                engine.schedule_event(event)
                
                comm_time = self._calculate_allreduce_time(total_gradient_size, resources[node_id])
                event = Event(
                    timestamp=start_time + comm_time,
                    event_type=EventType.ALLREDUCE_END,
                    node_id=node_id,
                    data={
                        'iteration': iteration_id,
                        'communication_time': comm_time
                    }
                )
                engine.schedule_event(event)
    
    def _schedule_ring_allreduce(self, engine: SimulationEngine,
                                 resources: Dict[int, ResourceManager],
                                 start_time: float,
                                 iteration_id: int) -> None:
        """
        Schedule ring all-reduce algorithm
        Divides data into chunks and performs reduce-scatter followed by all-gather
        """
        total_gradient_size = sum(layer.get_gradient_size_gb() for layer in self.model.layers)
        chunk_size = total_gradient_size / self.num_nodes
        
        # Reduce-scatter phase
        current_time = start_time
        for step in range(self.num_nodes - 1):
            step_time = resources[0].communication_time(chunk_size)
            for node_id in range(self.num_nodes):
                event = Event(
                    timestamp=current_time,
                    event_type=EventType.COMM_START,
                    node_id=node_id,
                    data={
                        'phase': 'reduce_scatter',
                        'step': step,
                        'chunk_size': chunk_size
                    }
                )
                engine.schedule_event(event)
            current_time += step_time
        
        # All-gather phase
        for step in range(self.num_nodes - 1):
            step_time = resources[0].communication_time(chunk_size)
            for node_id in range(self.num_nodes):
                event = Event(
                    timestamp=current_time,
                    event_type=EventType.COMM_START,
                    node_id=node_id,
                    data={
                        'phase': 'allgather',
                        'step': step,
                        'chunk_size': chunk_size
                    }
                )
                engine.schedule_event(event)
            current_time += step_time
    
    def _schedule_reduce_scatter(self, engine: SimulationEngine,
                                 resources: Dict[int, ResourceManager],
                                 start_time: float,
                                 iteration_id: int) -> None:
        """Schedule reduce-scatter communication pattern"""
        total_gradient_size = sum(layer.get_gradient_size_gb() for layer in self.model.layers)
        
        for node_id in range(self.num_nodes):
            event = Event(
                timestamp=start_time,
                event_type=EventType.REDUCE_SCATTER_START,
                node_id=node_id,
                data={
                    'iteration': iteration_id,
                    'data_size_gb': total_gradient_size
                }
            )
            engine.schedule_event(event)
            
            # Reduce-scatter time is approximately half of all-reduce
            comm_time = self._calculate_allreduce_time(total_gradient_size, resources[node_id]) / 2
            event = Event(
                timestamp=start_time + comm_time,
                event_type=EventType.REDUCE_SCATTER_END,
                node_id=node_id,
                data={
                    'iteration': iteration_id,
                    'communication_time': comm_time
                }
            )
            engine.schedule_event(event)
    
    def _calculate_allreduce_time(self, data_size_gb: float, 
                                 resource: ResourceManager) -> float:
        """
        Calculate all-reduce communication time using bandwidth-latency model
        
        For ring all-reduce: T = 2(n-1)/n * (α + S/B)
        where n = num_nodes, α = latency, S = message size, B = bandwidth
        """
        alpha = 5e-6  # Network latency (5 microseconds)
        bandwidth = resource.network_bandwidth  # GB/s
        
        if self.communication_pattern == CommunicationPattern.RING_ALLREDUCE:
            # Ring all-reduce algorithm
            time = 2 * (self.num_nodes - 1) / self.num_nodes * \
                   (alpha + data_size_gb / bandwidth)
        else:
            # Simple all-reduce
            time = alpha + 2 * (self.num_nodes - 1) / self.num_nodes * data_size_gb / bandwidth
        
        return time