import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import Counter, deque
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
import networkx as nx

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression

# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- [GLOBAL CONFIG] ---
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- [NEW SIMULATION CONSTANTS] ---
# We will check the miss rate over the last 50 attack packets
ATTACK_WINDOW_SIZE = 50
# If the miss rate exceeds 50%, the system fails
GRIDLOCK_THRESHOLD_RATE = 0.5 


# --- [PART 1: MODEL & ATTACK DEFINITIONS (From Notebook)] ---

class PyTorchLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x)

def fgsm_attack(model, loss_function, x, y, epsilon):
    x.requires_grad = True
    outputs = model(x)
    loss = loss_function(outputs, y)
    model.zero_grad()
    loss.backward()
    x_adv = x + epsilon * x.grad.data.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

def pgd_attack(model, loss_function, x, y, epsilon, alpha, num_iter):
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    x_clean = x.clone().detach()
    for _ in range(num_iter):
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = loss_function(outputs, y)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv.detach() + alpha * x_adv.grad.sign()
        perturbation = torch.clamp(x_adv - x_clean, min=-epsilon, max=epsilon)
        x_adv = x_clean + perturbation
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

def get_pytorch_probs(model, X_tensor):
    model.eval()
    X_tensor = X_tensor.to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs_attack = torch.sigmoid(logits)
        probs_normal = 1 - probs_attack
        return torch.cat((probs_normal, probs_attack), dim=1).cpu().numpy()

# --- [PART 2: CACHED FUNCTIONS (To run only once)] ---

@st.cache_resource
def load_and_preprocess_data():
    st.write("Cache miss: Loading and preprocessing data...")
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'attack', 'level'
    ]
    try:
        train_df = pd.read_csv('KDDTrain+.txt', header=None, names=column_names)
        test_df = pd.read_csv('KDDTest+.txt', header=None, names=column_names)
    except FileNotFoundError:
        return None
    
    train_df['attack_flag'] = train_df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['attack_flag'] = test_df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    train_df, test_df = train_df.drop(columns=['attack', 'level']), test_df.drop(columns=['attack', 'level'])
    
    categorical_features = ['protocol_type', 'service', 'flag']
    combined_df = pd.concat([train_df, test_df], keys=['train', 'test'])
    combined_encoded_df = pd.get_dummies(combined_df, columns=categorical_features)
    
    train_df_encoded = combined_encoded_df.loc['train']
    test_df_encoded = combined_encoded_df.loc['test']
    
    train_cols, test_cols = train_df_encoded.columns.drop('attack_flag'), test_df_encoded.columns.drop('attack_flag')
    missing_in_test = set(train_cols) - set(test_cols); missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_test: test_df_encoded[c] = 0
    for c in missing_in_train: train_df_encoded[c] = 0
    test_df_encoded = test_df_encoded.reindex(columns=train_df_encoded.columns, fill_value=0)
    
    X_train, y_train = train_df_encoded.drop(columns='attack_flag'), train_df_encoded['attack_flag']
    X_test, y_test = test_df_encoded.drop(columns='attack_flag'), test_df_encoded['attack_flag']
    X_test = X_test[X_train.columns]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns); X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    input_dim = X_train_smote.shape[1]
    X_train_tensor = torch.tensor(X_train_smote.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_smote.values, dtype=torch.float32).view(-1, 1)
    train_loader = DataLoader(dataset=TensorDataset(X_train_tensor, y_train_tensor), batch_size=128, shuffle=True)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    return (
        X_train_smote, y_train_smote, X_test.values, y_test.values,
        X_test_tensor, y_test_tensor, train_loader, input_dim, scaler
    )

@st.cache_resource
def train_all_models(_train_loader, _input_dim, _X_train_smote, _y_train_smote):
    st.write("Cache miss: Training all models...")
    
    standard_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    standard_model.fit(_X_train_smote, _y_train_smote)
    st.write("... Standard Model Trained.")
    
    fgsm_robust_model = PyTorchLogisticRegression(_input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(); optimizer = optim.Adam(fgsm_robust_model.parameters(), lr=0.001)
    fgsm_robust_model.train()
    for epoch in range(10):
        for batch_x, batch_y in _train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x_adv = fgsm_attack(fgsm_robust_model, criterion, batch_x, batch_y, epsilon=0.05)
            loss = (0.5 * criterion(fgsm_robust_model(batch_x), batch_y)) + (0.5 * criterion(fgsm_robust_model(batch_x_adv), batch_y))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    st.write("... FGSM-Robust Model Trained.")
    
    pgd_robust_model = PyTorchLogisticRegression(_input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(); optimizer = optim.Adam(pgd_robust_model.parameters(), lr=0.001)
    pgd_robust_model.train()
    for epoch in range(10):
        for batch_x, batch_y in _train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x_adv = pgd_attack(pgd_robust_model, criterion, batch_x, batch_y, epsilon=0.05, alpha=0.01, num_iter=7)
            loss = (0.5 * criterion(pgd_robust_model(batch_x), batch_y)) + (0.5 * criterion(pgd_robust_model(batch_x_adv), batch_y))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    st.write("... PGD-Robust Model Trained.")
            
    return standard_model, fgsm_robust_model, pgd_robust_model

@st.cache_resource
def create_evaluation_testbeds(_input_dim, _train_loader, _X_test_numpy, _X_test_tensor, _y_test_tensor):
    st.write("Cache miss: Generating adversarial testbeds...")
    
    naive_pytorch_model = PyTorchLogisticRegression(_input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(); optimizer = optim.Adam(naive_pytorch_model.parameters(), lr=0.001)
    naive_pytorch_model.train()
    for epoch in range(10):
        for batch_x, batch_y in _train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = criterion(naive_pytorch_model(batch_x), batch_y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    naive_pytorch_model.eval()
    X_test_clean_tensor = _X_test_tensor.to(device); y_test_device_tensor = _y_test_tensor.to(device)
    
    X_test_fgsm_tensor = fgsm_attack(naive_pytorch_model, criterion, X_test_clean_tensor, y_test_device_tensor, epsilon=0.05).to(device)
    X_test_pgd_tensor = pgd_attack(naive_pytorch_model, criterion, X_test_clean_tensor, y_test_device_tensor, epsilon=0.05, alpha=0.01, num_iter=7).to(device)
    st.write("... Testbeds Generated.")

    # This is the original data (mixed)
    testbeds_sklearn = {
        "Clean": _X_test_numpy,
        "FGSM Adv.": X_test_fgsm_tensor.cpu().detach().numpy(),
        "PGD Adv.": X_test_pgd_tensor.cpu().detach().numpy()
    }
    testbeds_pytorch = {
        "Clean": X_test_clean_tensor,
        "FGSM Adv.": X_test_fgsm_tensor,
        "PGD Adv.": X_test_pgd_tensor
    }
    
    return testbeds_sklearn, testbeds_pytorch

# --- [PART 3: VISUALIZATION & SIMULATION STATE] ---
st.set_page_config(layout="wide", page_title="Live IIoT Gateway SOC")

# Initialize session state for the simulation
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.simulation_active = False
    st.session_state.active_model_name = "Standard Model"
    st.session_state.traffic_type = "PGD Adv." # The data to use
    
    st.session_state.packet_index = 0
    st.session_state.correct_count = 0
    st.session_state.total_processed = 0
    
    # --- NEW STATE for Rate-Based Gridlock ---
    st.session_state.attack_packet_window = deque(maxlen=ATTACK_WINDOW_SIZE)
    st.session_state.miss_rate = 0.0
    
    st.session_state.warehouse_status = "OPERATIONAL"
    st.session_state.alert_log = ["Simulation Initialized. Awaiting command."]
    st.session_state.chart_history = pd.DataFrame(columns=['time', 'connections', 'result'])
    st.session_state.models_loaded = False

# Function to create the network graph (purely visual)
def create_network_graph(attack_active=False):
    G = nx.Graph()
    G.add_node("Gateway", size=20, color='blue')
    normal_devices = ['AMR-01', 'AMR-02', 'PLC-Sort', 'PLC-Pack']
    for device in normal_devices:
        G.add_node(device, size=10, color='green')
        G.add_edge("Gateway", device)
        
    if attack_active:
        for i in range(20): # Draw 20 "sybil" nodes
            node_name = f"SYBIL-{i+1}"
            G.add_node(node_name, size=5, color='red')
            G.add_edge("Gateway", node_name)

    pos = nx.spring_layout(G, k=0.5, iterations=20)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    node_x, node_y, node_size, node_color, node_text = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        node_size.append(G.nodes[node]['size']); node_color.append(G.nodes[node]['color']); node_text.append(node)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, marker=dict(showscale=False, size=node_size, color=node_color))
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Live Network Topology', showlegend=False, hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.update_layout(height=400)
    return fig

# --- [PART 4: APP LAYOUT & MAIN LOOP] ---

# --- Sidebar Controls ---
with st.sidebar:
    st.title("🕹️ Simulation Controls")
    
    st.session_state.active_model_name = st.selectbox(
        "Select: Active IDS Model (Defender)",
        ["Standard Model", "FGSM-Robust Model", "PGD-Robust Model", "Hybrid (Avg.) Model"],
        disabled=st.session_state.simulation_active
    )
    
    st.session_state.traffic_type = st.selectbox(
        "Select: Network Traffic",
        ["Clean", "FGSM Adv.", "PGD Adv."], 
        disabled=st.session_state.simulation_active
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ START SIMULATION"):
            if st.session_state.models_loaded and not st.session_state.simulation_active:
                st.session_state.simulation_active = True
                st.session_state.packet_index = 0
                st.session_state.correct_count = 0
                st.session_state.total_processed = 0
                st.session_state.warehouse_status = "OPERATIONAL"
                st.session_state.chart_history = pd.DataFrame(columns=['time', 'connections', 'result'])
                st.session_state.alert_log.insert(0, f"SIMULATION STARTED: {st.session_state.traffic_type}")
                
                # --- RESET NEW STATE ---
                st.session_state.attack_packet_window.clear()
                st.session_state.miss_rate = 0.0
                
                st.rerun()

    with col2:
        if st.button("⏹️ STOP SIMULATION"):
            if st.session_state.simulation_active:
                st.session_state.simulation_active = False
                st.session_state.alert_log.insert(0, "Simulation Halted by user.")
                st.rerun()

    st.subheader("Event Log")
    st.text_area("Log", value="\n".join(st.session_state.alert_log), height=300, disabled=True)

# --- Main Dashboard ---
st.title("🛡️ Live IIoT Gateway - Security Operations Center")

# --- First-time Load: Train models and build testbeds ---
if not st.session_state.models_loaded:
    st.header("Initializing Simulation Backend (First-Time Load)")
    st.info("The application is loading data, training all models, and generating attack sets. This may take 1-2 minutes and only happens once.")
    
    with st.spinner("Loading and preprocessing data..."):
        data_tuple = load_and_preprocess_data()
    
    if data_tuple is None:
        st.error("Error: `KDDTrain+.txt` or `KDDTest+.txt` not found. Please place them in the same folder as the script.")
    else:
        st.success("✅ Data loaded and preprocessed.")
        (
            st.session_state.X_train_smote, st.session_state.y_train_smote, 
            st.session_state.X_test_numpy, st.session_state.y_test_numpy,
            st.session_state.X_test_tensor, st.session_state.y_test_tensor, 
            st.session_state.train_loader, st.session_state.input_dim, st.session_state.scaler
        ) = data_tuple

        with st.spinner("Training Standard, FGSM, and PGD models..."):
            st.session_state.standard_model, st.session_state.fgsm_robust_model, st.session_state.pgd_robust_model = train_all_models(
                st.session_state.train_loader, st.session_state.input_dim, 
                st.session_state.X_train_smote, st.session_state.y_train_smote
            )
        st.success("✅ All models trained.")

        with st.spinner("Generating adversarial test data (FGSM, PGD)..."):
            st.session_state.testbeds_sklearn, st.session_state.testbeds_pytorch = create_evaluation_testbeds(
                st.session_state.input_dim, st.session_state.train_loader, 
                st.session_state.X_test_numpy, st.session_state.X_test_tensor, st.session_state.y_test_tensor
            )
        st.success("✅ Adversarial testbeds generated.")
        
        # We are NOT creating pure streams, just loading all models.
        
        st.session_state.models_loaded = True
        st.balloons()
        st.header("✅ Backend Ready! Select simulation controls from the sidebar.")
        st.rerun()

# --- Main Simulation Loop ---
else:
    # --- Layout ---
    col_graph, col_metrics, col_impact = st.columns([1.5, 2.5, 1.2])
    
    with col_graph:
        st.subheader("Live Network Topology")
        is_attack_data = st.session_state.traffic_type != "Clean"
        st.plotly_chart(create_network_graph(st.session_state.simulation_active and is_attack_data), use_container_width=True)

    with col_metrics:
        st.subheader(f"Active Model: `{st.session_state.active_model_name}`")
        
        st.subheader("Live Traffic Analysis")
        chart_placeholder = st.empty()
        
        m1, m2, m3 = st.columns(3)
        acc_metric = m1.empty()
        det_metric = m2.empty()
        miss_rate_metric = m3.empty() # <-- RENAMED METRIC
        

    with col_impact:
        st.subheader("🏭 Impact Center")
        status_metric = st.empty()
        task_list = st.empty()

    # --- Simulation Logic ---
    if st.session_state.simulation_active:
        # 1. Get the models
        model_name = st.session_state.active_model_name
        std_model = st.session_state.standard_model
        fgsm_model = st.session_state.fgsm_robust_model
        pgd_model = st.session_state.pgd_robust_model

        # 2. Get the correct MIXED data for this packet
        traffic_type = st.session_state.traffic_type
        idx = st.session_state.packet_index
        
        # --- [!!! REVERTED DATA LOGIC !!!] ---
        # We use the original full testbeds
        data_row_sk = st.session_state.testbeds_sklearn[traffic_type][idx]
        data_row_torch = st.session_state.testbeds_pytorch[traffic_type][idx]
        true_label = int(st.session_state.y_test_numpy[idx]) # 0 for normal, 1 for attack
        data_len = len(st.session_state.y_test_numpy)
        # --- [!!! END REVERT !!!] ---
        
        # 3. Run a *real* prediction on the single data packet
        pred = 0
        try:
            if model_name == "Standard Model":
                pred = std_model.predict(data_row_sk.reshape(1, -1))[0]
            elif model_name == "FGSM-Robust Model":
                logit = fgsm_model(data_row_torch.unsqueeze(0))
                pred = (torch.sigmoid(logit) > 0.5).float().item()
            elif model_name == "PGD-Robust Model":
                logit = pgd_model(data_row_torch.unsqueeze(0))
                pred = (torch.sigmoid(logit) > 0.5).float().item()
            elif model_name == "Hybrid (Avg.) Model":
                weights = {'std': 0.2, 'fgsm': 0.3, 'pgd': 0.5}
                probs_std = std_model.predict_proba(data_row_sk.reshape(1, -1))
                probs_fgsm = get_pytorch_probs(fgsm_model, data_row_torch.unsqueeze(0))
                probs_pgd = get_pytorch_probs(pgd_model, data_row_torch.unsqueeze(0))
                final_probs = (weights['std'] * probs_std) + (weights['fgsm'] * probs_fgsm) + (weights['pgd'] * probs_pgd)
                pred = np.argmax(final_probs, axis=1)[0]
        
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.session_state.simulation_active = False

        pred = int(pred)

        # 4. Update counts
        st.session_state.total_processed += 1
        result = "Other"
        
        if pred == true_label:
            st.session_state.correct_count += 1
            if true_label == 1:
                result = "Detected Attack (TP)"
            else:
                result = "Normal (TN)"
        else:
            if true_label == 1 and pred == 0:
                result = "🔥 MISSED ATTACK (FN)"
            elif true_label == 0 and pred == 1:
                result = "False Alarm (FP)"
            
        # 5. --- [NEW GRIDLOCK LOGIC] ---
        if true_label == 1: # Only track the rate for ATTACK packets
            if pred == 0:
                st.session_state.attack_packet_window.append(0) # 0 = miss
            else:
                st.session_state.attack_packet_window.append(1) # 1 = success
        
            # Calculate the new miss rate
            if len(st.session_state.attack_packet_window) > 0:
                misses = st.session_state.attack_packet_window.count(0)
                st.session_state.miss_rate = misses / len(st.session_state.attack_packet_window)

        # Check for gridlock
        # Only check if the window is "full" (or at least 20 packets) to get a stable rate
        is_window_stable = len(st.session_state.attack_packet_window) > 20
        if (st.session_state.miss_rate > GRIDLOCK_THRESHOLD_RATE) and is_window_stable: 
            if st.session_state.warehouse_status == "OPERATIONAL":
                st.session_state.alert_log.insert(0, f"CRITICAL: Miss rate {st.session_state.miss_rate*100:.0f}%! Warehouse is down.")
            st.session_state.warehouse_status = "GRIDLOCK - DOWNTIME"
        
        # 6. Update chart history
        conn_value = data_row_sk[32] * 255 # Use 'dst_host_srv_count' (index 32), un-scaled
        new_row = pd.DataFrame({
            'time': [pd.Timestamp.now()], 
            'connections': [conn_value],
            'result': [result]
        })
        st.session_state.chart_history = pd.concat([st.session_state.chart_history, new_row]).tail(100)
        
        # 7. Increment index
        st.session_state.packet_index = (st.session_state.packet_index + 1) % data_len

    # --- Update all UI elements ---
    
    # Calculate live accuracy
    if st.session_state.total_processed > 0:
        live_accuracy = (st.session_state.correct_count / st.session_state.total_processed) * 100
    else:
        live_accuracy = 100.0

    # Set detection status
    if not st.session_state.simulation_active:
        detection = "IDLE"
    elif st.session_state.traffic_type == "Clean":
        detection = "MONITORING"
    elif st.session_state.miss_rate > GRIDLOCK_THRESHOLD_RATE:
        detection = "🔥 FAILURE"
    elif st.session_state.miss_rate > 0:
        detection = "⚠️ FAILING"
    else:
        detection = "✅ DETECTING"

    acc_metric.metric("Live Accuracy", f"{live_accuracy:.2f}%")
    det_metric.metric("Detection Status", detection)
    
    # --- [NEW METRIC DISPLAY] ---
    miss_rate_metric.metric("Missed Packet Rate (FN)", f"{st.session_state.miss_rate * 100:.1f}%")

    status_metric.metric("Warehouse Status", st.session_state.warehouse_status)
    if st.session_state.warehouse_status == "OPERATIONAL":
        task_list.success("✅ AMR tasks running normally.")
    else:
        task_list.error("🔴 AMR tasks FAILED. System halt.")
    
    # Update chart
    if not st.session_state.chart_history.empty:
        color_map = {
            "🔥 MISSED ATTACK (FN)": "red",
            "Detected Attack (TP)": "orange",
            "False Alarm (FP)": "blue",
            "Normal (TN)": "green",
            "Other": "grey"
        }
        fig = go.Figure()
        for result_type, color in color_map.items():
            plot_data = st.session_state.chart_history[st.session_state.chart_history['result'] == result_type]
            fig.add_trace(go.Scatter(
                x=plot_data['time'], y=plot_data['connections'],
                mode='markers', name=result_type,
                marker=dict(color=color, size=7)
            ))
        
        fig.update_layout(height=400, margin=dict(t=20, b=20, l=0, r=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    # --- Loop ---
    if st.session_state.simulation_active:
        time.sleep(0.05) # Sleep for 50ms to make it run faster
        st.rerun()