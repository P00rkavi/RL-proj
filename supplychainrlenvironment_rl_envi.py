import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple, Any
from supplychainsimulator import SupplyChainSimulator, NodeType

class SupplyChainEnv(gym.Env):
    """
    OpenAI Gym environment for supply chain management
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        # Initialize simulator
        self.simulator = SupplyChainSimulator(config)
        
        # Environment parameters
        self.max_episode_steps = config.get('max_episode_steps', 100) if config else 100
        self.current_step = 0
        
        # Get node information
        self.node_ids = list(self.simulator.nodes.keys())
        self.num_nodes = len(self.node_ids)
        
        # Define action space (order quantities for each node)
        # Each node can order 0 to max_order_quantity units
        max_order = 200
        self.action_space = spaces.Box(
            low=0,
            high=max_order,
            shape=(self.num_nodes,),
            dtype=np.int32
        )
        
        # Define observation space
        # For each node: [inventory, demand, capacity, utilization, node_type_encoded]
        # Plus global features: [time_step, total_system_inventory, total_demand]
        obs_dim = self.num_nodes * 5 + 3
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Node type encoding
        self.node_type_encoding = {
            NodeType.SUPPLIER: 0,
            NodeType.MANUFACTURER: 1,
            NodeType.DISTRIBUTOR: 2,
            NodeType.RETAILER: 3
        }
        
        # Reset environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.simulator.reset()
        self.current_step = 0
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Convert action array to action dictionary
        actions = {}
        for i, node_id in enumerate(self.node_ids):
            actions[node_id] = int(action[i])
        
        # Execute simulation step
        step_result = self.simulator.step(actions)
        
        # Calculate reward
        reward = self._calculate_reward(step_result)
        
        # Check if episode is done
        done = self.current_step >= self.max_episode_steps
        
        # Prepare info dictionary
        info = {
            'step_result': step_result,
            'current_step': self.current_step,
            'total_cost': step_result['total_cost'],
            'service_levels': self._calculate_service_levels(step_result),
            'inventory_levels': self._get_inventory_levels()
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        state = self.simulator.get_state()
        
        # Node-specific features
        node_features = []
        total_inventory = 0
        total_demand = 0
        
        for node_id in self.node_ids:
            node_state = state[node_id]
            node = self.simulator.nodes[node_id]
            
            # Features: inventory, demand, capacity, utilization, node_type
            features = [
                node_state['inventory'],
                node_state['demand'],
                node_state['capacity'],
                node_state['utilization'],
                self.node_type_encoding[node.type]
            ]
            
            node_features.extend(features)
            total_inventory += node_state['inventory']
            total_demand += node_state['demand']
        
        # Global features
        global_features = [
            self.current_step / self.max_episode_steps,  # Normalized time
            total_inventory,
            total_demand
        ]
        
        # Combine all features
        observation = np.array(node_features + global_features, dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, step_result: Dict) -> float:
        """Calculate reward based on step results"""
        # Multi-objective reward combining cost minimization and service level
        
        # Cost component (negative because we want to minimize)
        cost_penalty = -step_result['total_cost'] / 1000.0  # Normalized
        
        # Service level reward
        service_levels = self._calculate_service_levels(step_result)
        avg_service_level = np.mean(list(service_levels.values())) if service_levels else 0
        service_reward = avg_service_level * 100  # Scale up service level reward
        
        # Inventory balance reward (penalize excess inventory)
        inventory_penalty = 0
        for node_id, node_state in step_result['state'].items():
            node = self.simulator.nodes[node_id]
            utilization = node_state['utilization']
            
            # Penalize very high or very low utilization
            if utilization > 0.9:
                inventory_penalty -= (utilization - 0.9) * 50
            elif utilization < 0.1:
                inventory_penalty -= (0.1 - utilization) * 30
        
        # Stability reward (penalize large swings in orders)
        stability_reward = 0
        if len(self.simulator.history) > 1:
            current_actions = step_result['info']
            # This is a simplified stability measure
            for node_id in self.node_ids:
                current_order = current_actions.get(node_id, {}).get('order_quantity', 0)
                # Penalize very large orders
                if current_order > 150:
                    stability_reward -= (current_order - 150) * 0.1
        
        # Total reward
        total_reward = cost_penalty + service_reward + inventory_penalty + stability_reward
        
        return total_reward
    
    def _calculate_service_levels(self, step_result: Dict) -> Dict[str, float]:
        """Calculate service levels for retailers"""
        service_levels = {}
        
        for node_id, info in step_result['info'].items():
            node = self.simulator.nodes[node_id]
            if node.type == NodeType.RETAILER:
                demand = node.demand
                if demand > 0:
                    service_level = info['fulfilled_demand'] / demand
                    service_levels[node_id] = min(1.0, service_level)
                else:
                    service_levels[node_id] = 1.0
        
        return service_levels
    
    def _get_inventory_levels(self) -> Dict[str, int]:
        """Get current inventory levels"""
        state = self.simulator.get_state()
        return {node_id: node_state['inventory'] for node_id, node_state in state.items()}
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            state = self.simulator.get_state()
            print(f"\n=== Step {self.current_step} ===")
            
            for node_type in [NodeType.RETAILER, NodeType.DISTRIBUTOR, NodeType.MANUFACTURER, NodeType.SUPPLIER]:
                nodes_of_type = [(nid, node) for nid, node in self.simulator.nodes.items() if node.type == node_type]
                if nodes_of_type:
                    print(f"\n{node_type.value.upper()}S:")
                    for node_id, node in nodes_of_type:
                        node_state = state[node_id]
                        print(f"  {node_id}: Inv={node_state['inventory']}, "
                              f"Demand={node_state['demand']}, "
                              f"Cap={node_state['capacity']}, "
                              f"Util={node_state['utilization']:.2f}")
            
            # Show recent costs if available
            if self.simulator.history:
                last_record = self.simulator.history[-1]
                print(f"\nTotal Cost: {last_record['total_cost']:.2f}")
    
    def get_action_meanings(self) -> Dict[int, str]:
        """Get meanings of actions for interpretability"""
        meanings = {}
        for i, node_id in enumerate(self.node_ids):
            meanings[i] = f"Order quantity for {node_id}"
        return meanings
    
    def get_valid_actions(self) -> np.ndarray:
        """Get valid actions based on current state"""
        # For simplicity, all actions in the defined range are valid
        # In a more complex scenario, this could consider budget constraints, etc.
        return self.action_space.sample()
    
    def get_metrics(self) -> Dict:
        """Get environment performance metrics"""
        return self.simulator.get_metrics()
    
    def seed(self, seed=None):
        """Set random seed"""
        if seed is not None:
            np.random.seed(seed)
            self.simulator.config['random_seed'] = seed
            self.simulator.reset()
        return [seed]

class MultiAgentSupplyChainEnv(SupplyChainEnv):
    """
    Multi-agent version where each node is controlled by a separate agent
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Create separate action spaces for each node
        max_order = 200
        self.action_spaces = {}
        self.observation_spaces = {}
        
        for node_id in self.node_ids:
            # Each agent controls one node
            self.action_spaces[node_id] = spaces.Box(
                low=0, high=max_order, shape=(1,), dtype=np.int32
            )
            
            # Each agent observes local state + some global information
            # Local: [inventory, demand, capacity, utilization, node_type]
            # Global: [time_step, connected_nodes_inventory, total_system_demand]
            obs_dim = 5 + 3  # Simplified for demonstration
            self.observation_spaces[node_id] = spaces.Box(
                low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset and return observations for all agents"""
        super().reset()
        observations = {}
        
        for node_id in self.node_ids:
            observations[node_id] = self._get_agent_observation(node_id)
        
        return observations
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Step with multi-agent actions"""
        self.current_step += 1
        
        # Convert multi-agent actions to single action format
        action_dict = {}
        for node_id, action in actions.items():
            action_dict[node_id] = int(action[0])
        
        # Execute simulation step
        step_result = self.simulator.step(action_dict)
        
        # Calculate rewards for each agent
        rewards = self._calculate_agent_rewards(step_result)
        
        # Check if episode is done
        done = self.current_step >= self.max_episode_steps
        dones = {node_id: done for node_id in self.node_ids}
        dones['__all__'] = done
        
        # Get observations for each agent
        observations = {}
        for node_id in self.node_ids:
            observations[node_id] = self._get_agent_observation(node_id)
        
        # Prepare info
        infos = {node_id: {
            'step_result': step_result,
            'current_step': self.current_step
        } for node_id in self.node_ids}
        
        return observations, rewards, dones, infos
    
    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for a specific agent"""
        state = self.simulator.get_state()
        node_state = state[agent_id]
        node = self.simulator.nodes[agent_id]
        
        # Local features
        local_features = [
            node_state['inventory'],
            node_state['demand'],
            node_state['capacity'],
            node_state['utilization'],
            self.node_type_encoding[node.type]
        ]
        
        # Global/connected features
        connected_inventory = 0
        for connected_id in node.connections:
            if connected_id in state:
                connected_inventory += state[connected_id]['inventory']
        
        total_demand = sum(ns['demand'] for ns in state.values())
        
        global_features = [
            self.current_step / self.max_episode_steps,
            connected_inventory,
            total_demand
        ]
        
        return np.array(local_features + global_features, dtype=np.float32)
    
    def _calculate_agent_rewards(self, step_result: Dict) -> Dict[str, float]:
        """Calculate individual rewards for each agent"""
        rewards = {}
        
        for node_id in self.node_ids:
            node = self.simulator.nodes[node_id]
            
            # Local cost (negative reward)
            local_cost = step_result['costs'][node_id]
            cost_reward = -local_cost / 100.0
            
            # Cooperation reward (based on downstream performance)
            cooperation_reward = 0
            downstream_nodes = self.simulator.network[node_id]['downstream']
            for downstream_id in downstream_nodes:
                downstream_node = self.simulator.nodes[downstream_id]
                if downstream_node.type == NodeType.RETAILER:
                    # Reward for helping downstream retailers meet demand
                    info = step_result['info'][downstream_id]
                    service_level = info['fulfilled_demand'] / max(1, downstream_node.demand)
                    cooperation_reward += service_level * 10
            
            rewards[node_id] = cost_reward + cooperation_reward
        
        return rewards
