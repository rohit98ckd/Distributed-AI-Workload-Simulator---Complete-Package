"""
Distributed AI Workload Simulator - Core Engine
================================================
Production-level simulator for modeling distributed AI training workloads
Inspired by Astra-Sim and ns-3 architectures

Author: AI Infrastructure Research Team
Version: 1.0.0
License: MIT
"""

import heapq
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
import logging
from collections import defaultdict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Enumeration of all possible event types in the simulation"""
    COMPUTE_START = "compute_start"
    COMPUTE_END = "compute_end"
    COMM_START = "comm_start"
    COMM_END = "comm_end"
    ALLREDUCE_START = "allreduce_start"
    ALLREDUCE_END = "allreduce_end"
    REDUCE_SCATTER_START = "reduce_scatter_start"
    REDUCE_SCATTER_END = "reduce_scatter_end"
    ALLGATHER_START = "allgather_start"
    ALLGATHER_END = "allgather_end"
    MEMORY_ALLOC = "memory_alloc"
    MEMORY_FREE = "memory_free"
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"


class CommunicationPattern(Enum):
    """Supported communication patterns for distributed training"""
    RING_ALLREDUCE = "ring_allreduce"
    TREE_ALLREDUCE = "tree_allreduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALLGATHER = "allgather"
    POINT_TO_POINT = "point_to_point"
    ALLTOALL = "alltoall"


@dataclass(order=True)
class Event:
    """
    Event structure for discrete event simulation
    Events are ordered by timestamp for priority queue processing
    """
    timestamp: float
    event_type: EventType = field(compare=False)
    node_id: int = field(compare=False)
    layer_id: Optional[int] = field(default=None, compare=False)
    data: Optional[Dict[str, Any]] = field(default_factory=dict, compare=False)
    callback: Optional[Callable] = field(default=None, compare=False)
    
    def __repr__(self):
        return (f"Event(t={self.timestamp:.6f}, type={self.event_type.value}, "
                f"node={self.node_id}, layer={self.layer_id})")


class SimulationEngine:
    """
    Core discrete event simulation engine
    Manages event queue and simulation time progression
    """
    
    def __init__(self, max_simulation_time: float = 1e6):
        self.current_time: float = 0.0
        self.max_simulation_time = max_simulation_time
        self.event_queue: List[Event] = []
        self.event_count = 0
        self.processed_events = 0
        self.event_history: List[Event] = []
        self.running = False
        
        # Statistics tracking
        self.stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'simulation_wall_time': 0.0
        }
        
    def schedule_event(self, event: Event) -> None:
        """Schedule a new event in the simulation"""
        if event.timestamp < self.current_time:
            logger.warning(
                f"Attempted to schedule event in the past: "
                f"t={event.timestamp}, current_t={self.current_time}"
            )
            event.timestamp = self.current_time
            
        heapq.heappush(self.event_queue, event)
        self.event_count += 1
        self.stats['events_by_type'][event.event_type] += 1
        
    def run(self, collect_history: bool = False) -> Dict[str, Any]:
        """
        Execute the simulation until completion
        
        Args:
            collect_history: Whether to store all events in memory for analysis
            
        Returns:
            Dictionary containing simulation statistics
        """
        import time
        start_wall_time = time.time()
        
        logger.info(f"Starting simulation with {len(self.event_queue)} initial events")
        self.running = True
        
        while self.event_queue and self.current_time < self.max_simulation_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.timestamp
            
            if collect_history:
                self.event_history.append(event)
            
            # Execute event callback if provided
            if event.callback:
                try:
                    event.callback(event)
                except Exception as e:
                    logger.error(f"Error executing callback for {event}: {e}")
            
            self.processed_events += 1
            
            if self.processed_events % 10000 == 0:
                logger.debug(f"Processed {self.processed_events} events, t={self.current_time:.6f}")
        
        self.running = False
        self.stats['simulation_wall_time'] = time.time() - start_wall_time
        self.stats['total_events'] = self.processed_events
        
        logger.info(
            f"Simulation complete: {self.processed_events} events processed, "
            f"final_t={self.current_time:.6f}s, "
            f"wall_time={self.stats['simulation_wall_time']:.2f}s"
        )
        
        return self.get_statistics()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return simulation statistics"""
        return {
            'total_events': self.stats['total_events'],
            'events_by_type': dict(self.stats['events_by_type']),
            'final_simulation_time': self.current_time,
            'wall_time': self.stats['simulation_wall_time'],
            'events_per_second': (
                self.stats['total_events'] / self.stats['simulation_wall_time']
                if self.stats['simulation_wall_time'] > 0 else 0
            )
        }
    
    def reset(self) -> None:
        """Reset the simulation to initial state"""
        self.current_time = 0.0
        self.event_queue.clear()
        self.event_history.clear()
        self.event_count = 0
        self.processed_events = 0
        self.stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'simulation_wall_time': 0.0
        }
        logger.info("Simulation engine reset")


class ResourceManager:
    """
    Manages compute, memory, and network resources for simulation nodes
    Tracks utilization and enforces capacity constraints
    """
    
    def __init__(self, node_id: int, compute_capacity: float, 
                 memory_capacity: float, network_bandwidth: float):
        self.node_id = node_id
        self.compute_capacity = compute_capacity  # TFLOPS
        self.memory_capacity = memory_capacity    # GB
        self.network_bandwidth = network_bandwidth  # GB/s
        
        # Current utilization
        self.compute_utilization = 0.0
        self.memory_utilization = 0.0
        self.network_utilization = 0.0
        
        # Tracking
        self.memory_allocations: Dict[str, float] = {}
        self.active_operations: List[str] = []
        
        # Statistics
        self.stats = {
            'total_compute_ops': 0,
            'total_memory_allocated': 0.0,
            'total_data_transferred': 0.0,
            'peak_memory_usage': 0.0,
            'peak_network_usage': 0.0,
            'compute_time': 0.0,
            'idle_time': 0.0
        }
        
    def allocate_memory(self, allocation_id: str, size_gb: float) -> bool:
        """
        Attempt to allocate memory
        
        Returns:
            True if allocation successful, False if insufficient memory
        """
        if self.memory_utilization + size_gb <= self.memory_capacity:
            self.memory_allocations[allocation_id] = size_gb
            self.memory_utilization += size_gb
            self.stats['total_memory_allocated'] += size_gb
            self.stats['peak_memory_usage'] = max(
                self.stats['peak_memory_usage'], 
                self.memory_utilization
            )
            return True
        return False
    
    def free_memory(self, allocation_id: str) -> bool:
        """Free previously allocated memory"""
        if allocation_id in self.memory_allocations:
            size = self.memory_allocations.pop(allocation_id)
            self.memory_utilization -= size
            return True
        return False
    
    def can_execute_compute(self, flops: float) -> bool:
        """Check if compute operation can be executed"""
        return self.compute_capacity > 0
    
    def compute_execution_time(self, flops: float) -> float:
        """Calculate execution time for compute operation"""
        if self.compute_capacity == 0:
            return float('inf')
        time_seconds = flops / (self.compute_capacity * 1e12)  # Convert TFLOPS to FLOPS
        self.stats['total_compute_ops'] += 1
        self.stats['compute_time'] += time_seconds
        return time_seconds
    
    def communication_time(self, data_size_gb: float) -> float:
        """Calculate time to transfer data over network"""
        if self.network_bandwidth == 0:
            return float('inf')
        time_seconds = data_size_gb / self.network_bandwidth
        self.stats['total_data_transferred'] += data_size_gb
        self.stats['peak_network_usage'] = max(
            self.stats['peak_network_usage'],
            data_size_gb
        )
        return time_seconds
    
    def get_utilization(self) -> Dict[str, float]:
        """Return current resource utilization percentages"""
        return {
            'compute': self.compute_utilization / self.compute_capacity * 100 
                       if self.compute_capacity > 0 else 0,
            'memory': self.memory_utilization / self.memory_capacity * 100 
                      if self.memory_capacity > 0 else 0,
            'network': self.network_utilization / self.network_bandwidth * 100 
                       if self.network_bandwidth > 0 else 0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return resource usage statistics"""
        return {
            'node_id': self.node_id,
            'capacity': {
                'compute_tflops': self.compute_capacity,
                'memory_gb': self.memory_capacity,
                'network_gbps': self.network_bandwidth
            },
            'utilization': self.get_utilization(),
            'stats': self.stats.copy()
        }


class TopologyManager:
    """
    Manages network topology and routing for distributed communication
    Supports various interconnect topologies
    """
    
    def __init__(self, num_nodes: int, topology_type: str = "full_mesh"):
        self.num_nodes = num_nodes
        self.topology_type = topology_type
        self.adjacency_matrix = self._build_topology()
        self.latency_matrix = np.ones((num_nodes, num_nodes)) * 1e-6  # 1μs default
        
    def _build_topology(self) -> np.ndarray:
        """Build adjacency matrix based on topology type"""
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        
        if self.topology_type == "full_mesh":
            adj = np.ones((self.num_nodes, self.num_nodes), dtype=int)
            np.fill_diagonal(adj, 0)
            
        elif self.topology_type == "ring":
            for i in range(self.num_nodes):
                adj[i][(i + 1) % self.num_nodes] = 1
                adj[i][(i - 1) % self.num_nodes] = 1
                
        elif self.topology_type == "tree":
            # Binary tree topology
            for i in range(self.num_nodes):
                left_child = 2 * i + 1
                right_child = 2 * i + 2
                parent = (i - 1) // 2 if i > 0 else -1
                
                if left_child < self.num_nodes:
                    adj[i][left_child] = 1
                    adj[left_child][i] = 1
                if right_child < self.num_nodes:
                    adj[i][right_child] = 1
                    adj[right_child][i] = 1
                    
        elif self.topology_type == "torus_2d":
            # 2D torus (square grid with wraparound)
            side = int(np.sqrt(self.num_nodes))
            for i in range(side):
                for j in range(side):
                    node = i * side + j
                    # Connect to neighbors with wraparound
                    neighbors = [
                        (i * side + (j + 1) % side),  # right
                        (i * side + (j - 1) % side),  # left
                        (((i + 1) % side) * side + j),  # down
                        (((i - 1) % side) * side + j)   # up
                    ]
                    for neighbor in neighbors:
                        if neighbor < self.num_nodes:
                            adj[node][neighbor] = 1
        
        return adj
    
    def get_route(self, src: int, dst: int) -> List[int]:
        """Get route from source to destination node"""
        if src == dst:
            return [src]
        
        # Simple shortest path using BFS
        from collections import deque
        queue = deque([(src, [src])])
        visited = {src}
        
        while queue:
            node, path = queue.popleft()
            
            for next_node in range(self.num_nodes):
                if self.adjacency_matrix[node][next_node] and next_node not in visited:
                    new_path = path + [next_node]
                    if next_node == dst:
                        return new_path
                    queue.append((next_node, new_path))
                    visited.add(next_node)
        
        return []  # No path found
    
    def get_communication_time(self, src: int, dst: int, data_size_gb: float,
                               bandwidth_gbps: float) -> float:
        """
        Calculate end-to-end communication time including latency and bandwidth
        """
        if src == dst:
            return 0.0
        
        route = self.get_route(src, dst)
        if not route:
            return float('inf')
        
        # Number of hops
        hops = len(route) - 1
        
        # Total latency (sum of link latencies)
        total_latency = sum(
            self.latency_matrix[route[i]][route[i+1]] 
            for i in range(hops)
        )
        
        # Transmission time
        transmission_time = data_size_gb / bandwidth_gbps if bandwidth_gbps > 0 else float('inf')
        
        return total_latency + transmission_time
