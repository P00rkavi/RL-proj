import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from supplychainsimulator import supplychainsimulator, NodeType
from supplychainsimulator import supplychainenvi_rl_envi
import json

class SupplyChainDashboard:
    """Interactive dashboard for supply chain visualization and simulation"""
    
    def __init__(self):
        st.set_page_config(
            page_title="Supply Chain RL Dashboard",
            page_icon="🏭",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        if 'simulator' not in st.session_state:
            st.session_state.simulator = None
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
        if 'step_count' not in st.session_state:
            st.session_state.step_count = 0
    
    def run(self):
        """Main dashboard interface"""
        st.title("🏭 Supply Chain Reinforcement Learning Dashboard")
        st.markdown("---")
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main content area
        if st.session_state.simulator is None:
            self._render_welcome()
        else:
            self._render_main_dashboard()
    
    def _render_sidebar(self):
        """Render sidebar configuration"""
        st.sidebar.header("⚙️ Configuration")
        
        # Network configuration
        st.sidebar.subheader("Network Structure")
        num_suppliers = st.sidebar.slider("Suppliers", 1, 5, 2)
        num_manufacturers = st.sidebar.slider("Manufacturers", 1, 3, 1)
        num_distributors = st.sidebar.slider("Distributors", 1, 5, 2)
        num_retailers = st.sidebar.slider("Retailers", 1, 10, 3)
        
        # Simulation parameters
        st.sidebar.subheader("Simulation Parameters")
        max_demand = st.sidebar.slider("Max Demand", 50, 200, 100)
        demand_variability = st.sidebar.slider("Demand Variability", 0.1, 0.5, 0.3)
        random_seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)
        
        # Create configuration
        config = {
            'num_suppliers': num_suppliers,
            'num_manufacturers': num_manufacturers,
            'num_distributors': num_distributors,
            'num_retailers': num_retailers,
            'max_demand': max_demand,
            'demand_variability': demand_variability,
            'random_seed': random_seed
        }
        
        # Initialize/Reset simulation
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🚀 Initialize", type="primary"):
                st.session_state.simulator = supplychainsimulator(config)
                st.session_state.step_count = 0
                st.session_state.simulation_running = False
                st.success("Simulator initialized!")
        
        with col2:
            if st.button("🔄 Reset"):
                if st.session_state.simulator:
                    st.session_state.simulator.reset()
                    st.session_state.step_count = 0
                    st.session_state.simulation_running = False
                    st.success("Simulation reset!")
        
        # Simulation controls
        if st.session_state.simulator:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🎮 Simulation Controls")
            
            # Manual step control
            if st.sidebar.button("➡️ Single Step"):
                self._execute_single_step()
            
            # Auto-run controls
            auto_steps = st.sidebar.number_input("Auto Steps", 1, 100, 10)
            if st.sidebar.button("⏩ Auto Run"):
                self._execute_auto_steps(auto_steps)
    
    def _render_welcome(self):
        """Render welcome screen"""
        st.markdown("""
        ## Welcome to the Supply Chain Dashboard! 🎯
        
        This interactive dashboard allows you to:
        - **Configure** supply chain networks with different structures
        - **Simulate** supply chain operations in real-time
        - **Visualize** network topology and performance metrics
        - **Analyze** cost, service levels, and inventory patterns
        - **Test** different ordering strategies
        
        ### Getting Started
        1. Configure your network structure in the sidebar
        2. Set simulation parameters
        3. Click **Initialize** to create your supply chain
        4. Use the simulation controls to run the simulation
        
        ### Features
        - 📊 Real-time performance metrics
        - 🌐 Interactive network visualization
        - 📈 Historical trend analysis
        - 🎛️ Manual order control
        - 🤖 AI agent integration ready
        """)
        
        # Sample network preview
        st.subheader("Sample Network Structure")
        self._render_sample_network()
    
    def _render_sample_network(self):
        """Render a sample network for demonstration"""
        G = nx.DiGraph()
        
        # Add sample nodes
        G.add_node('S1', type='Supplier', pos=(0, 2))
        G.add_node('S2', type='Supplier', pos=(0, 0))
        G.add_node('M1', type='Manufacturer', pos=(2, 1))
        G.add_node('D1', type='Distributor', pos=(4, 2))
        G.add_node('D2', type='Distributor', pos=(4, 0))
        G.add_node('R1', type='Retailer', pos=(6, 2.5))
        G.add_node('R2', type='Retailer', pos=(6, 1.5))
        G.add_node('R3', type='Retailer', pos=(6, 0.5))
        
        # Add edges
        edges = [('S1', 'M1'), ('S2', 'M1'), ('M1', 'D1'), ('M1', 'D2'),
                ('D1', 'R1'), ('D1', 'R2'), ('D2', 'R2'), ('D2', 'R3')]
        G.add_edges_from(edges)
        
        # Create plotly network visualization
        pos = nx.get_node_attributes(G, 'pos')
        
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                       mode='lines', line=dict(width=2, color='gray'),
                                       hoverinfo='none', showlegend=False))
        
        node_trace = go.Scatter(x=[], y=[], mode='markers+text', hoverinfo='text',
                              text=[], textposition="middle center",
                              marker=dict(size=50, color=[], colorscale='Viridis',
                                        line=dict(width=2, color='white')))
        
        colors = {'Supplier': 0, 'Manufacturer': 1, 'Distributor': 2, 'Retailer': 3}
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node])
            node_trace['marker']['color'] += tuple([colors[G.nodes[node]['type']]])
        
        fig = go.Figure(data=edge_trace + [node_trace],
                       layout=go.Layout(title='Sample Supply Chain Network',
                                       showlegend=False, hovermode='closest',
                                       margin=dict(b=20,l=5,r=5,t=40),
                                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_main_dashboard(self):
        """Render main dashboard with simulation"""
        simulator = st.session_state.simulator
        
        # Header with key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Time Step", st.session_state.step_count)
        
        with col2:
            if simulator.history:
                total_cost = simulator.history[-1]['total_cost']
                st.metric("Total Cost", f"${total_cost:.2f}")
            else:
                st.metric("Total Cost", "$0.00")
        
        with col3:
            metrics = simulator.get_metrics()
            service_level = metrics.get('average_service_level', 0)
            st.metric("Service Level", f"{service_level:.1%}")
        
        with col4:
            total_inventory = sum(node.inventory for node in simulator.nodes.values())
            st.metric("Total Inventory", total_inventory)
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🌐 Network View", "📊 Performance", "📈 Trends", "🎛️ Control Panel", "📋 Data"
        ])
        
        with tab1:
            self._render_network_view()
        
        with tab2:
            self._render_performance_view()
        
        with tab3:
            self._render_trends_view()
        
        with tab4:
            self._render_control_panel()
        
        with tab5:
            self._render_data_view()
    
    def _render_network_view(self):
        """Render network topology with current state"""
        simulator = st.session_state.simulator
        
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes with current state
        state = simulator.get_state()
        pos_config = self._calculate_node_positions()
        
        for node_id, node in simulator.nodes.items():
            node_state = state[node_id]
            G.add_node(node_id, 
                      type=node.type.value,
                      inventory=node_state['inventory'],
                      demand=node_state['demand'],
                      capacity=node_state['capacity'],
                      utilization=node_state['utilization'],
                      pos=pos_config[node_id])
        
        # Add edges
        for node_id, node in simulator.nodes.items():
            for connection in node.connections:
                G.add_edge(connection, node_id)
        
        # Create visualization
        fig = self._create_network_plot(G, state)
        st.plotly_chart(fig, use_container_width=True)
        
        # Network statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Network Statistics")
            network_info = simulator.get_network_info()
            
            stats_data = []
            for node_type in NodeType:
                count = sum(1 for n in simulator.nodes.values() if n.type == node_type)
                stats_data.append({"Node Type": node_type.value.title(), "Count": count})
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        
        with col2:
            st.subheader("Current State Summary")
            summary_data = []
            
            for node_type in NodeType:
                nodes_of_type = [n for n in simulator.nodes.values() if n.type == node_type]
                if nodes_of_type:
                    avg_inventory = np.mean([state[n.id]['inventory'] for n in nodes_of_type])
                    avg_utilization = np.mean([state[n.id]['utilization'] for n in nodes_of_type])
                    summary_data.append({
                        "Type": node_type.value.title(),
                        "Avg Inventory": f"{avg_inventory:.1f}",
                        "Avg Utilization": f"{avg_utilization:.1%}"
                    })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    def _render_performance_view(self):
        """Render performance metrics and KPIs"""
        simulator = st.session_state.simulator
        
        if not simulator.history:
            st.info("No simulation data available. Run some simulation steps first.")
            return
        
        metrics = simulator.get_metrics()
        
        # KPI cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Cost Metrics")
            st.metric("Average Cost", f"${metrics.get('average_cost', 0):.2f}")
            st.metric("Total Cost", f"${metrics.get('total_cost', 0):.2f}")
            st.metric("Cost Variance", f"${metrics.get('cost_variance', 0):.2f}")
        
        with col2:
            st.subheader("Service Metrics")
            st.metric("Service Level", f"{metrics.get('average_service_level', 0):.1%}")
            
            # Calculate stockout frequency
            stockouts = 0
            total_retailer_periods = 0
            for record in simulator.history:
                for node_id, info in record['info'].items():
                    if simulator.nodes[node_id].type == NodeType.RETAILER:
                        total_retailer_periods += 1
                        if info['shortage'] > 0:
                            stockouts += 1
            
            stockout_rate = stockouts / max(1, total_retailer_periods)
            st.metric("Stockout Rate", f"{stockout_rate:.1%}")
        
        with col3:
            st.subheader("Inventory Metrics")
            st.metric("Average Inventory", f"{metrics.get('average_inventory', 0):.0f}")
            st.metric("Inventory Variance", f"{metrics.get('inventory_variance', 0):.0f}")
        
        # Performance charts
        st.subheader("Performance Analysis")
        
        # Cost breakdown by node type
        cost_data = []
        latest_costs = simulator.history[-1]['costs']
        
        for node_type in NodeType:
            type_cost = sum(cost for node_id, cost in latest_costs.items() 
                          if simulator.nodes[node_id].type == node_type)
            if type_cost > 0:
                cost_data.append({"Node Type": node_type.value.title(), "Cost": type_cost})
        
        if cost_data:
            fig_cost = px.pie(pd.DataFrame(cost_data), values='Cost', names='Node Type',
                            title='Cost Distribution by Node Type')
            st.plotly_chart(fig_cost, use_container_width=True)
        
        # Service level by retailer
        if len(simulator.history) > 0:
            service_data = []
            for node_id, node in simulator.nodes.items():
                if node.type == NodeType.RETAILER:
                    # Calculate average service level for this retailer
                    service_levels = []
                    for record in simulator.history:
                        info = record['info'][node_id]
                        demand = simulator.nodes[node_id].demand
                        if demand > 0:
                            service_level = info['fulfilled_demand'] / demand
                            service_levels.append(min(1.0, service_level))
                    
                    if service_levels:
                        avg_service = np.mean(service_levels)
                        service_data.append({"Retailer": node_id, "Service Level": avg_service})
            
            if service_data:
                fig_service = px.bar(pd.DataFrame(service_data), x='Retailer', y='Service Level',
                                   title='Average Service Level by Retailer')
                fig_service.update_layout(yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig_service, use_container_width=True)
    
    def _render_trends_view(self):
        """Render historical trends and analysis"""
        simulator = st.session_state.simulator
        
        if not simulator.history:
            st.info("No simulation data available. Run some simulation steps first.")
            return
        
        # Create trends dataframe
        df = simulator.export_history()
        
        # Cost trends
        st.subheader("Cost Trends")
        fig_cost_trend = px.line(df, x='time_step', y='total_cost',
                               title='Total Cost Over Time')
        st.plotly_chart(fig_cost_trend, use_container_width=True)
        
        # Inventory trends by node type
        st.subheader("Inventory Trends")
        
        inventory_cols = [col for col in df.columns if col.endswith('_inventory')]
        inventory_data = []
        
        for col in inventory_cols:
            node_id = col.replace('_inventory', '')
            node_type = simulator.nodes[node_id].type.value
            
            for idx, row in df.iterrows():
                inventory_data.append({
                    'time_step': row['time_step'],
                    'node_id': node_id,
                    'node_type': node_type,
                    'inventory': row[col]
                })
        
        if inventory_data:
            inventory_df = pd.DataFrame(inventory_data)
            
            # Aggregate by node type
            inventory_agg = inventory_df.groupby(['time_step', 'node_type'])['inventory'].sum().reset_index()
            
            fig_inventory = px.line(inventory_agg, x='time_step', y='inventory',
                                  color='node_type', title='Inventory Levels by Node Type')
            st.plotly_chart(fig_inventory, use_container_width=True)
        
        # Demand vs fulfillment for retailers
        st.subheader("Demand vs Fulfillment")
        
        retailer_data = []
        for record in simulator.history:
            for node_id, node in simulator.nodes.items():
                if node.type == NodeType.RETAILER:
                    info = record['info'][node_id]
                    retailer_data.append({
                        'time_step': record['time_step'],
                        'retailer': node_id,
                        'demand': node.demand,
                        'fulfilled': info['fulfilled_demand'],
                        'shortage': info['shortage']
                    })
        
        if retailer_data:
            retailer_df = pd.DataFrame(retailer_data)
            
            # Aggregate demand and fulfillment
            demand_agg = retailer_df.groupby('time_step').agg({
                'demand': 'sum',
                'fulfilled': 'sum',
                'shortage': 'sum'
            }).reset_index()
            
            fig_demand = go.Figure()
            fig_demand.add_trace(go.Scatter(x=demand_agg['time_step'], y=demand_agg['demand'],
                                          mode='lines', name='Total Demand'))
            fig_demand.add_trace(go.Scatter(x=demand_agg['time_step'], y=demand_agg['fulfilled'],
                                          mode='lines', name='Fulfilled Demand'))
            fig_demand.update_layout(title='Demand vs Fulfillment Over Time',
                                   xaxis_title='Time Step', yaxis_title='Quantity')
            
            st.plotly_chart(fig_demand, use_container_width=True)
    
    def _render_control_panel(self):
        """Render manual control panel"""
        simulator = st.session_state.simulator
        
        st.subheader("Manual Order Control")
        st.markdown("Set order quantities for each node and execute a simulation step.")
        
        # Create order input form
        with st.form("order_form"):
            orders = {}
            
            # Group nodes by type for better organization
            for node_type in NodeType:
                nodes_of_type = [(nid, n) for nid, n in simulator.nodes.items() if n.type == node_type]
                
                if nodes_of_type:
                    st.subheader(f"{node_type.value.title()}s")
                    cols = st.columns(min(3, len(nodes_of_type)))
                    
                    for i, (node_id, node) in enumerate(nodes_of_type):
                        with cols[i % 3]:
                            current_inventory = node.inventory
                            suggested_order = max(0, node.capacity - current_inventory) // 2
                            
                            orders[node_id] = st.number_input(
                                f"{node_id}",
                                min_value=0,
                                max_value=200,
                                value=suggested_order,
                                help=f"Current inventory: {current_inventory}, Capacity: {node.capacity}"
                            )
            
            # Submit button
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                execute_step = st.form_submit_button("Execute Step", type="primary")
            
            with col2:
                random_orders = st.form_submit_button("Random Orders")
            
            if execute_step:
                self._execute_manual_step(orders)
            
            if random_orders:
                # Generate random orders and execute
                random_orders_dict = {}
                for node_id in orders.keys():
                    random_orders_dict[node_id] = np.random.randint(0, 100)
                self._execute_manual_step(random_orders_dict)
        
        # Strategy suggestions
        st.subheader("Strategy Suggestions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Optimal Inventory"):
                optimal_orders = self._calculate_optimal_orders()
                st.session_state.suggested_orders = optimal_orders
                st.success("Optimal orders calculated! Check the form above.")
        
        with col2:
            if st.button("⚡ Just-in-Time"):
                jit_orders = self._calculate_jit_orders()
                st.session_state.suggested_orders = jit_orders
                st.success("Just-in-time orders calculated!")
    
    def _render_data_view(self):
        """Render data tables and export options"""
        simulator = st.session_state.simulator
        
        # Current state table
        st.subheader("Current State")
        state_data = []
        state = simulator.get_state()
        
        for node_id, node_state in state.items():
            node = simulator.nodes[node_id]
            state_data.append({
                'Node ID': node_id,
                'Type': node.type.value.title(),
                'Inventory': node_state['inventory'],
                'Demand': node_state['demand'],
                'Capacity': node_state['capacity'],
                'Utilization': f"{node_state['utilization']:.1%}",
                'Lead Time': node.lead_time,
                'Holding Cost': node.holding_cost,
                'Shortage Cost': node.shortage_cost
            })
        
        st.dataframe(pd.DataFrame(state_data), use_container_width=True)
        
        # Historical data
        if simulator.history:
            st.subheader("Historical Data")
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Export History CSV"):
                    df = simulator.export_history()
                    csv = df.to_csv(index=False)
                    st.download_button("Download CSV", csv, "supply_chain_history.csv", "text/csv")
            
            with col2:
                if st.button("📋 Export Configuration"):
                    config = {
                        'simulator_config': simulator.config,
                        'network_info': simulator.get_network_info(),
                        'metrics': simulator.get_metrics()
                    }
                    json_str = json.dumps(config, indent=2, default=str)
                    st.download_button("Download JSON", json_str, "supply_chain_config.json", "application/json")
            
            # Show recent history
            df = simulator.export_history()
            if len(df) > 0:
                st.subheader("Recent History (Last 10 Steps)")
                recent_df = df.tail(10)
                st.dataframe(recent_df, use_container_width=True)
    
    def _execute_single_step(self):
        """Execute a single simulation step with default actions"""
        if not st.session_state.simulator:
            return
        
        # Generate default actions (simple reorder policy)
        actions = {}
        for node_id, node in st.session_state.simulator.nodes.items():
            # Simple reorder policy: order when inventory is below 50% capacity
            if node.inventory < node.capacity * 0.5:
                actions[node_id] = min(50, node.capacity - node.inventory)
            else:
                actions[node_id] = 0
        
        # Execute step
        result = st.session_state.simulator.step(actions)
        st.session_state.step_count += 1
        
        st.success(f"Step {st.session_state.step_count} executed! Total cost: ${result['total_cost']:.2f}")
    
    def _execute_auto_steps(self, num_steps):
        """Execute multiple simulation steps automatically"""
        if not st.session_state.simulator:
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_cost = 0
        
        for i in range(num_steps):
            # Generate actions
            actions = {}
            for node_id, node in st.session_state.simulator.nodes.items():
                # More sophisticated policy for auto-run
                target_inventory = node.capacity * 0.7
                if node.inventory < target_inventory:
                    order_qty = int(target_inventory - node.inventory)
                    actions[node_id] = min(order_qty, 100)
                else:
                    actions[node_id] = 0
            
            # Execute step
            result = st.session_state.simulator.step(actions)
            st.session_state.step_count += 1
            total_cost += result['total_cost']
            
            # Update progress
            progress_bar.progress((i + 1) / num_steps)
            status_text.text(f"Step {st.session_state.step_count}: Cost = ${result['total_cost']:.2f}")
        
        st.success(f"Completed {num_steps} steps! Average cost: ${total_cost/num_steps:.2f}")
        st.rerun()
    
    def _execute_manual_step(self, orders):
        """Execute a single step with manual orders"""
        if not st.session_state.simulator:
            return
        
        result = st.session_state.simulator.step(orders)
        st.session_state.step_count += 1
        
        st.success(f"Manual step executed! Total cost: ${result['total_cost']:.2f}")
        st.rerun()
    
    def _calculate_node_positions(self):
        """Calculate positions for network visualization"""
        simulator = st.session_state.simulator
        positions = {}
        
        # Simple layered layout
        layers = {
            NodeType.SUPPLIER: 0,
            NodeType.MANUFACTURER: 1,
            NodeType.DISTRIBUTOR: 2,
            NodeType.RETAILER: 3
        }
        
        type_counts = {node_type: 0 for node_type in NodeType}
        type_indices = {node_type: 0 for node_type in NodeType}
        
        # Count nodes per type
        for node in simulator.nodes.values():
            type_counts[node.type] += 1
        
        # Assign positions
        for node_id, node in simulator.nodes.items():
            layer = layers[node.type]
            index = type_indices[node.type]
            total_in_layer = type_counts[node.type]
            
            # Spread nodes within layer
            if total_in_layer == 1:
                y_pos = 0
            else:
                y_pos = (index - (total_in_layer - 1) / 2) * 2
            
            positions[node_id] = (layer * 3, y_pos)
            type_indices[node.type] += 1
        
        return positions
    
    def _create_network_plot(self, G, state):
        """Create interactive network plot"""
        pos = nx.get_node_attributes(G, 'pos')
        
        # Edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='rgba(125,125,125,0.5)'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Node trace
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        
        color_map = {
            'supplier': 'red',
            'manufacturer': 'blue', 
            'distributor': 'green',
            'retailer': 'orange'
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            node_text.append(f"{node}<br>Inv: {node_data['inventory']}<br>Util: {node_data['utilization']:.1%}")
            node_color.append(color_map[node_data['type']])
            node_size.append(20 + node_data['utilization'] * 30)  # Size based on utilization
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],
            hovertext=node_text,
            textposition="middle center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title='Supply Chain Network - Current State',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(text="Node size represents inventory utilization", 
                     showarrow=False, xref="paper", yref="paper",
                     x=0.005, y=-0.002, xanchor='left', yanchor='bottom',
                     font=dict(size=12))
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        return fig
    
    def _calculate_optimal_orders(self):
        """Calculate optimal order quantities using simple heuristics"""
        orders = {}
        for node_id, node in st.session_state.simulator.nodes.items():
            # Target 70% capacity utilization
            target_inventory = node.capacity * 0.7
            optimal_order = max(0, int(target_inventory - node.inventory))
            orders[node_id] = min(optimal_order, 100)
        return orders
    
    def _calculate_jit_orders(self):
        """Calculate just-in-time order quantities"""
        orders = {}
        for node_id, node in st.session_state.simulator.nodes.items():
            # Order only what's needed to meet immediate demand
            if node.type == NodeType.RETAILER:
                orders[node_id] = max(0, node.demand - node.inventory)
            else:
                # For upstream nodes, maintain minimal safety stock
                safety_stock = node.capacity * 0.2
                orders[node_id] = max(0, int(safety_stock - node.inventory))
        return orders

def main():
    """Main function to run the dashboard"""
    dashboard = SupplyChainDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
