"""
Test Suite for Distributed AI Workload Simulator
=================================================
Comprehensive tests for all major components

Author: AI Infrastructure Research Team
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from simulator_core import (
    SimulationEngine, ResourceManager, TopologyManager,
    Event, EventType, CommunicationPattern
)
from training_workload import (
    ModelArchitecture, LayerConfig, LayerType,
    DistributedTrainingWorkload
)
from config import (
    SimulationConfig, HardwareConfig, ModelConfig, TrainingConfig,
    create_small_scale_config
)
from main_simulator import DistributedAISimulator, ScalingStudyRunner
from analysis import SimulationAnalyzer, ScalingAnalyzer


class TestSimulationEngine(unittest.TestCase):
    """Test cases for the core simulation engine"""
    
    def setUp(self):
        self.engine = SimulationEngine()
    
    def test_event_scheduling(self):
        """Test event scheduling and ordering"""
        # Schedule events out of order
        event1 = Event(timestamp=2.0, event_type=EventType.COMPUTE_START, node_id=0)
        event2 = Event(timestamp=1.0, event_type=EventType.COMPUTE_START, node_id=0)
        event3 = Event(timestamp=3.0, event_type=EventType.COMPUTE_START, node_id=0)
        
        self.engine.schedule_event(event1)
        self.engine.schedule_event(event2)
        self.engine.schedule_event(event3)
        
        # Verify events are ordered
        self.assertEqual(len(self.engine.event_queue), 3)
        self.assertEqual(self.engine.event_queue[0].timestamp, 1.0)
    
    def test_simulation_run(self):
        """Test basic simulation execution"""
        # Schedule some test events
        for i in range(10):
            event = Event(
                timestamp=i * 0.1,
                event_type=EventType.COMPUTE_START,
                node_id=0
            )
            self.engine.schedule_event(event)
        
        stats = self.engine.run()
        
        self.assertEqual(stats['total_events'], 10)
        self.assertGreater(stats['final_simulation_time'], 0)
    
    def test_time_progression(self):
        """Test simulation time advances correctly"""
        event1 = Event(timestamp=1.0, event_type=EventType.COMPUTE_START, node_id=0)
        event2 = Event(timestamp=5.0, event_type=EventType.COMPUTE_END, node_id=0)
        
        self.engine.schedule_event(event1)
        self.engine.schedule_event(event2)
        
        self.engine.run()
        
        self.assertEqual(self.engine.current_time, 5.0)


class TestResourceManager(unittest.TestCase):
    """Test cases for resource management"""
    
    def setUp(self):
        self.resource = ResourceManager(
            node_id=0,
            compute_capacity=100.0,
            memory_capacity=80.0,
            network_bandwidth=100.0
        )
    
    def test_memory_allocation(self):
        """Test memory allocation and deallocation"""
        # Allocate memory
        success = self.resource.allocate_memory("alloc1", 40.0)
        self.assertTrue(success)
        self.assertEqual(self.resource.memory_utilization, 40.0)
        
        # Allocate more
        success = self.resource.allocate_memory("alloc2", 30.0)
        self.assertTrue(success)
        self.assertEqual(self.resource.memory_utilization, 70.0)
        
        # Try to over-allocate
        success = self.resource.allocate_memory("alloc3", 20.0)
        self.assertFalse(success)
        
        # Free memory
        success = self.resource.free_memory("alloc1")
        self.assertTrue(success)
        self.assertEqual(self.resource.memory_utilization, 30.0)
    
    def test_compute_execution_time(self):
        """Test compute time calculation"""
        flops = 1e12  # 1 TFLOP
        time = self.resource.compute_execution_time(flops)
        
        # With 100 TFLOPS capacity, should take 0.01 seconds
        self.assertAlmostEqual(time, 0.01, places=4)
    
    def test_communication_time(self):
        """Test communication time calculation"""
        data_size = 10.0  # 10 GB
        time = self.resource.communication_time(data_size)
        
        # With 100 GB/s bandwidth, should take 0.1 seconds
        self.assertAlmostEqual(time, 0.1, places=4)


class TestTopologyManager(unittest.TestCase):
    """Test cases for network topology"""
    
    def test_full_mesh_topology(self):
        """Test full mesh connectivity"""
        topo = TopologyManager(num_nodes=4, topology_type="full_mesh")
        
        # Check all nodes are connected
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertEqual(topo.adjacency_matrix[i][j], 1)
                else:
                    self.assertEqual(topo.adjacency_matrix[i][j], 0)
    
    def test_ring_topology(self):
        """Test ring connectivity"""
        topo = TopologyManager(num_nodes=4, topology_type="ring")
        
        # Each node should have 2 connections (left and right)
        for i in range(4):
            connections = np.sum(topo.adjacency_matrix[i])
            self.assertEqual(connections, 2)
    
    def test_routing(self):
        """Test route finding"""
        topo = TopologyManager(num_nodes=4, topology_type="full_mesh")
        
        # Direct connection
        route = topo.get_route(0, 1)
        self.assertEqual(len(route), 2)
        self.assertEqual(route[0], 0)
        self.assertEqual(route[1], 1)


class TestModelArchitecture(unittest.TestCase):
    """Test cases for model definitions"""
    
    def test_resnet50_creation(self):
        """Test ResNet-50 model creation"""
        model = ModelArchitecture.create_resnet50()
        
        self.assertEqual(model.name, "ResNet-50")
        self.assertGreater(len(model.layers), 0)
        self.assertGreater(model.total_parameters, 0)
    
    def test_transformer_creation(self):
        """Test Transformer model creation"""
        model = ModelArchitecture.create_transformer(
            num_layers=6,
            hidden_size=512,
            num_heads=8
        )
        
        self.assertGreater(len(model.layers), 0)
        self.assertGreater(model.total_parameters, 0)
    
    def test_parameter_size_calculation(self):
        """Test model size calculation"""
        model = ModelArchitecture("TestModel")
        model.add_layer(LayerConfig(
            layer_id=0,
            layer_type=LayerType.LINEAR,
            input_shape=(512,),
            output_shape=(512,),
            num_parameters=512 * 512,  # 262,144 params
            forward_flops=1e9,
            backward_flops=2e9,
            activation_memory_mb=1.0,
            gradient_memory_mb=1.0
        ))
        
        size_gb = model.get_total_parameter_size_gb()
        # 262,144 params * 4 bytes / (1024^3) ≈ 0.00098 GB
        self.assertAlmostEqual(size_gb, 0.00098, places=5)


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration management"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = create_small_scale_config()
        self.assertTrue(config.validate())
        
        # Invalid config
        config.hardware.compute_tflops = -1
        self.assertFalse(config.validate())
    
    def test_yaml_serialization(self):
        """Test YAML save and load"""
        config = create_small_scale_config()
        yaml_path = Path(self.temp_dir) / "test_config.yaml"
        
        # Save
        config.to_yaml(str(yaml_path))
        self.assertTrue(yaml_path.exists())
        
        # Load
        loaded_config = SimulationConfig.from_yaml(str(yaml_path))
        self.assertEqual(config.name, loaded_config.name)
        self.assertEqual(
            config.hardware.compute_tflops,
            loaded_config.hardware.compute_tflops
        )
    
    def test_json_serialization(self):
        """Test JSON save and load"""
        config = create_small_scale_config()
        json_path = Path(self.temp_dir) / "test_config.json"
        
        # Save
        config.to_json(str(json_path))
        self.assertTrue(json_path.exists())
        
        # Load
        loaded_config = SimulationConfig.from_json(str(json_path))
        self.assertEqual(config.name, loaded_config.name)
    
    def test_hardware_presets(self):
        """Test hardware preset loading"""
        a100 = HardwareConfig.get_preset("A100")
        self.assertEqual(a100.gpu_model, "A100")
        self.assertEqual(a100.compute_tflops, 312.0)
        
        h100 = HardwareConfig.get_preset("H100")
        self.assertEqual(h100.gpu_model, "H100")
        self.assertGreater(h100.compute_tflops, a100.compute_tflops)
    
    def test_model_presets(self):
        """Test model preset loading"""
        gpt2_small = ModelConfig.get_preset("gpt2-small")
        self.assertEqual(gpt2_small.num_layers, 12)
        self.assertEqual(gpt2_small.hidden_size, 768)
        
        bert_large = ModelConfig.get_preset("bert-large")
        self.assertEqual(bert_large.num_layers, 24)


class TestDistributedSimulator(unittest.TestCase):
    """Integration tests for the full simulator"""
    
    def test_basic_simulation(self):
        """Test running a basic simulation"""
        config = create_small_scale_config()
        config.training.num_iterations = 5  # Quick test
        
        simulator = DistributedAISimulator(config)
        results = simulator.run()
        
        self.assertIn('summary', results)
        self.assertIn('simulation_stats', results)
        self.assertGreater(results['summary']['throughput_samples_per_sec'], 0)
    
    def test_analysis(self):
        """Test bottleneck analysis"""
        config = create_small_scale_config()
        config.training.num_iterations = 5
        
        simulator = DistributedAISimulator(config)
        results = simulator.run()
        analysis = simulator.analyze()
        
        self.assertIn('bottlenecks', analysis)
        self.assertGreater(len(analysis['bottlenecks']), 0)
    
    def test_export_results(self):
        """Test results export"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = create_small_scale_config()
            config.training.num_iterations = 3
            config.output_directory = temp_dir
            
            simulator = DistributedAISimulator(config)
            simulator.run()
            simulator.export_results()
            
            # Check files were created
            output_path = Path(temp_dir)
            results_file = list(output_path.glob("*_results.json"))
            config_file = list(output_path.glob("*_config.yaml"))
            
            self.assertEqual(len(results_file), 1)
            self.assertEqual(len(config_file), 1)
        
        finally:
            shutil.rmtree(temp_dir)


class TestScalingAnalysis(unittest.TestCase):
    """Test cases for scaling analysis"""
    
    def test_scaling_efficiency_calculation(self):
        """Test scaling efficiency metrics"""
        analyzer = ScalingAnalyzer()
        
        # Add some mock data
        analyzer.add_run(1, iteration_time=1.0, throughput=100, efficiency=1.0)
        analyzer.add_run(2, iteration_time=0.6, throughput=180, efficiency=0.83)
        analyzer.add_run(4, iteration_time=0.35, throughput=320, efficiency=0.71)
        
        efficiencies = analyzer.calculate_scaling_efficiency(baseline_gpus=1)
        
        self.assertEqual(efficiencies[1], 1.0)
        self.assertLess(efficiencies[4], 1.0)
    
    def test_breakdown_detection(self):
        """Test scaling breakdown detection"""
        analyzer = ScalingAnalyzer()
        
        # Simulate poor scaling
        analyzer.add_run(1, iteration_time=1.0, throughput=100, efficiency=1.0)
        analyzer.add_run(2, iteration_time=0.55, throughput=180, efficiency=0.91)
        analyzer.add_run(4, iteration_time=0.35, throughput=320, efficiency=0.71)
        analyzer.add_run(8, iteration_time=0.25, throughput=450, efficiency=0.5)
        
        breakdown = analyzer.identify_scaling_breakdown()
        
        self.assertIsNotNone(breakdown['breakdown_point'])
        self.assertLessEqual(breakdown['breakdown_point'], 8)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metric calculations"""
    
    def test_throughput_calculation(self):
        """Test throughput calculation"""
        config = create_small_scale_config()
        config.training.num_iterations = 10
        config.training.batch_size_per_gpu = 32
        config.training.num_nodes = 1
        config.hardware.num_gpus_per_node = 4
        
        simulator = DistributedAISimulator(config)
        results = simulator.run()
        
        # Global batch size = 32 * 4 = 128
        # Throughput should be samples/sec
        throughput = results['summary']['throughput_samples_per_sec']
        self.assertGreater(throughput, 0)
    
    def test_communication_overhead(self):
        """Test communication overhead calculation"""
        config = create_small_scale_config()
        config.training.num_iterations = 5
        
        simulator = DistributedAISimulator(config)
        results = simulator.run()
        
        overhead = results['summary']['communication_overhead_ratio']
        self.assertGreaterEqual(overhead, 0)
        self.assertLessEqual(overhead, 1)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSimulationEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestResourceManager))
    suite.addTests(loader.loadTestsFromTestCase(TestTopologyManager))
    suite.addTests(loader.loadTestsFromTestCase(TestModelArchitecture))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestDistributedSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestScalingAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
