import enum
import numpy as np
import pandas as pd

class NodeType(enum.Enum):
    SUPPLIER = 'supplier'
    MANUFACTURER = 'manufacturer'
    DISTRIBUTOR = 'distributor'
    RETAILER = 'retailer'

class Node:
    def __init__(self, id, type, capacity=100, lead_time=1, holding_cost=1.0, shortage_cost=10.0):
        self.id = id
        self.type = type
        self.capacity = capacity
        self.lead_time = lead_time
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.inventory = capacity // 2
        self.demand = 0
        self.utilization = 0.0
        self.connections = []

class SupplyChainSimulator:
    def __init__(self, config):
        self.config = config
        self.nodes = {}
        self.history = []
        self._initialize_network()
    
    def _initialize_network(self):
        # Create nodes based on config
        self.nodes = {}
        node_id = 1
        
        def add_nodes(num, node_type):
            nonlocal node_id
            for _ in range(num):
                nid = f"{node_type.value[0].upper()}{node_id}"
                node = Node(nid, node_type)
                self.nodes[nid] = node
                node_id += 1
        
        add_nodes(self.config.get('num_suppliers', 1), NodeType.SUPPLIER)
        add_nodes(self.config.get('num_manufacturers', 1), NodeType.MANUFACTURER)
        add_nodes(self.config.get('num_distributors', 1), NodeType.DISTRIBUTOR)
        add_nodes(self.config.get('num_retailers', 1), NodeType.RETAILER)
        
        # Setup connections (simple linear chain for demo)
        suppliers = [n for n in self.nodes.values() if n.type == NodeType.SUPPLIER]
        manufacturers = [n for n in self.nodes.values() if n.type == NodeType.MANUFACTURER]
        distributors = [n for n in self.nodes.values() if n.type == NodeType.DISTRIBUTOR]
        retailers = [n for n in self.nodes.values() if n.type == NodeType.RETAILER]
        
        for m in manufacturers:
            m.connections = [s.id for s in suppliers]
        for d in distributors:
            d.connections = [m.id for m in manufacturers]
        for r in retailers:
            r.connections = [d.id for d in distributors]
        
        # Initialize demands randomly for retailers
        max_demand = self.config.get('max_demand', 100)
        demand_variability = self.config.get('demand_variability', 0.3)
        np.random.seed(self.config.get('random_seed', 42))
        for r in retailers:
            r.demand = int(max_demand * (1 + demand_variability * (np.random.rand() - 0.5)))
    
    def reset(self):
        self._initialize_network()
        self.history = []
    
    def step(self, actions):
        # Apply actions (orders) to nodes
        total_cost = 0
        info = {}
        for node_id, order_qty in actions.items():
            node = self.nodes[node_id]
            # Update inventory with order quantity (simplified)
            node.inventory = min(node.capacity, node.inventory + order_qty)
            # Calculate costs
            holding_cost = node.holding_cost * node.inventory
            shortage = max(0, node.demand - node.inventory)
            shortage_cost = node.shortage_cost * shortage
            total_cost += holding_cost + shortage_cost
            utilization = node.inventory / node.capacity if node.capacity > 0 else 0
            node.utilization = utilization
            info[node_id] = {
                'holding_cost': holding_cost,
                'shortage': shortage,
                'shortage_cost': shortage_cost,
                'utilization': utilization,
                'fulfilled_demand': min(node.demand, node.inventory)
            }
            # Update inventory after demand fulfillment
            node.inventory = max(0, node.inventory - node.demand)
        
        self.history.append({
            'time_step': len(self.history) + 1,
            'total_cost': total_cost,
            'info': info,
            'costs': {nid: info[nid]['holding_cost'] + info[nid]['shortage_cost'] for nid in info}
        })
        return {'total_cost': total_cost}
    
    def get_metrics(self):
        if not self.history:
            return {}
        total_costs = [h['total_cost'] for h in self.history]
        average_cost = np.mean(total_costs)
        cost_variance = np.var(total_costs)
        
        # Average service level across retailers
        service_levels = []
        for h in self.history:
            info = h['info']
            for node_id, data in info.items():
                node = self.nodes[node_id]
                if node.type == NodeType.RETAILER:
                    demand = node.demand
                    if demand > 0:
                        service_levels.append(data['fulfilled_demand'] / demand)
        average_service_level = np.mean(service_levels) if service_levels else 0
        
        # Average inventory and variance
        inventories = []
        for node in self.nodes.values():
            inventories.append(node.inventory)
        average_inventory = np.mean(inventories) if inventories else 0
        inventory_variance = np.var(inventories) if inventories else 0
        
        return {
            'average_cost': average_cost,
            'cost_variance': cost_variance,
            'average_service_level': average_service_level,
            'average_inventory': average_inventory,
            'inventory_variance': inventory_variance,
            'total_cost': total_costs[-1] if total_costs else 0
        }
    
    def get_state(self):
        state = {}
        for node_id, node in self.nodes.items():
            state[node_id] = {
                'inventory': node.inventory,
                'demand': node.demand,
                'capacity': node.capacity,
                'utilization': node.utilization
            }
        return state
    
    def get_network_info(self):
        info = {
            'num_nodes': len(self.nodes),
            'nodes_by_type': {nt.value: 0 for nt in NodeType}
        }
        for node in self.nodes.values():
            info['nodes_by_type'][node.type.value] += 1
        return info
    
    def export_history(self):
        if not self.history:
            return pd.DataFrame()
        records = []
        for h in self.history:
            record = {'time_step': h['time_step'], 'total_cost': h['total_cost']}
            for node_id, data in h['info'].items():
                record[f"{node_id}_inventory"] = self.nodes[node_id].inventory
                record[f"{node_id}_demand"] = self.nodes[node_id].demand
            records.append(record)
        return pd.DataFrame(records)

# Removed empty placeholder SupplyChainEnv class to avoid confusion and import errors
