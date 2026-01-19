"""
RetentionIQ - Customer Intelligence Platform
Professional Analytics Dashboard with Userpilot Design
Features: Theme Toggle, Export, Notifications, Auto-refresh
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import sys
import io
import base64
from time import sleep

sys.path.insert(0, str(Path(__file__).parent.parent))

# Initialize session state for theme and notifications
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# Page Config
st.set_page_config(
    page_title="RetentionIQ | Customer Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Light Theme CSS - Userpilot Inspired
st.markdown("""
<style>
    /* Fonts & Icons */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css');
    
    /* Light Theme Color System */
    :root {
        --primary: #6366F1;
        --primary-light: #818CF8;
        --primary-dark: #4F46E5;
        --violet: #8B5CF6;
        --pink: #EC4899;
        --cyan: #06B6D4;
        --green: #10B981;
        --yellow: #F59E0B;
        --red: #EF4444;
        --bg-main: #F8FAFC;
        --bg-card: #FFFFFF;
        --bg-card-hover: #F1F5F9;
        --bg-sidebar: #FFFFFF;
        --border: #E2E8F0;
        --border-hover: #CBD5E1;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #94A3B8;
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        --glow: rgba(99, 102, 241, 0.25);
    }
    
    /* Light Theme Base */
    .stApp {
        background: var(--bg-main);
        background-image: 
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.08), transparent),
            radial-gradient(ellipse 60% 40% at 100% 100%, rgba(139, 92, 246, 0.05), transparent);
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--text-primary);
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-4px); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding: 2rem 3rem; max-width: 1700px;}
    
    /* Subtle Grid Background */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image: 
            linear-gradient(rgba(99, 102, 241, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99, 102, 241, 0.03) 1px, transparent 1px);
        background-size: 60px 60px;
        mask-image: radial-gradient(ellipse at center, black 30%, transparent 80%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Sidebar - Clean White */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border-right: 1px solid var(--border);
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.04);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.25rem 1rem;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: var(--text-secondary) !important;
    }
    
    .sidebar-header {
        padding: 1rem 0.5rem 1.5rem 0.5rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .brand-icon {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--violet) 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        color: white;
        box-shadow: 0 4px 12px var(--glow);
    }
    
    .brand-text {
        display: flex;
        flex-direction: column;
    }
    
    .brand-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.5px;
    }
    
    .brand-version {
        font-size: 0.7rem;
        color: var(--primary);
        background: rgba(99, 102, 241, 0.1);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        width: fit-content;
    }
    
    /* Nav Items - Light Theme */
    .nav-section {
        padding: 0 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .nav-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-muted);
        padding: 0 0.75rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        cursor: pointer;
        margin-bottom: 0.25rem;
    }
    
    .nav-item:hover {
        background: rgba(99, 102, 241, 0.06);
        color: var(--text-primary);
    }
    
    .nav-item.active {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.08));
        color: var(--primary);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .nav-item i {
        font-size: 1.1rem;
        width: 24px;
        text-align: center;
    }
    
    .nav-item.active i {
        color: var(--primary);
    }
    
    /* Main Content Area - Light Theme */
    .main-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }
    
    .page-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.5px;
    }
    
    .page-subtitle {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    
    .header-actions {
        display: flex;
        gap: 0.75rem;
    }
    
    .header-btn {
        padding: 0.6rem 1.25rem;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.25s ease;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .header-btn.secondary {
        background: var(--bg-card);
        border: 1px solid var(--border);
        color: var(--text-secondary);
    }
    
    .header-btn.secondary:hover {
        border-color: var(--border-hover);
        box-shadow: var(--shadow-sm);
    }
    
    .header-btn.primary {
        background: linear-gradient(135deg, var(--primary) 0%, var(--violet) 100%);
        border: none;
        color: white;
        box-shadow: 0 4px 12px var(--glow);
    }
    
    .header-btn.primary:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px var(--glow);
    }
    /* Metric Cards - Clean Light Theme */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        box-shadow: var(--shadow-sm);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
        background: linear-gradient(90deg, var(--primary), var(--violet));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }
    
    .metric-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    
    .metric-icon.purple {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.15));
        color: var(--primary);
    }
    
    .metric-icon.green {
        background: rgba(16, 185, 129, 0.12);
        color: var(--green);
    }
    
    .metric-icon.yellow {
        background: rgba(245, 158, 11, 0.12);
        color: var(--yellow);
    }
    
    .metric-icon.red {
        background: rgba(239, 68, 68, 0.12);
        color: var(--red);
    }
    
    .metric-trend {
        font-size: 0.75rem;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .metric-trend.up {
        background: rgba(16, 185, 129, 0.1);
        color: var(--green);
    }
    
    .metric-trend.down {
        background: rgba(239, 68, 68, 0.1);
        color: var(--red);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.5px;
    }
    
    .metric-sparkline {
        height: 30px;
        margin-top: 0.5rem;
    }
    
    /* Chart Cards - Clean Light Theme */
    .chart-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .chart-card:hover {
        border-color: var(--border-hover);
        box-shadow: var(--shadow-md);
    }
    
    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .chart-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .chart-title i {
        color: var(--primary);
    }
    
    .chart-filters {
        display: flex;
        gap: 0.5rem;
    }
    
    .chart-filter-btn {
        padding: 0.4rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        background: var(--bg-main);
        border: 1px solid var(--border);
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .chart-filter-btn.active {
        background: rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.3);
        color: var(--primary);
    }
    
    /* Risk Badges - Pill Style */
    .risk-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-pill.high {
        background: rgba(239, 68, 68, 0.1);
        color: var(--red);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    
    .risk-pill.medium {
        background: rgba(245, 158, 11, 0.1);
        color: var(--yellow);
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
    
    .risk-pill.low {
        background: rgba(16, 185, 129, 0.1);
        color: var(--green);
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    
    /* Prediction UI */
    .prediction-result-container {
        text-align: center;
        padding: 2rem 1rem;
    }
    
    .prediction-percentage {
        font-size: 5rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -3px;
        margin-bottom: 1rem;
    }
    
    .prediction-percentage.high {
        background: linear-gradient(135deg, #EF4444, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .prediction-percentage.medium {
        background: linear-gradient(135deg, #F59E0B, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .prediction-percentage.low {
        background: linear-gradient(135deg, #10B981, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Factor Progress Bars */
    .factor-row {
        display: flex;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid var(--border);
    }
    
    .factor-row:last-child {
        border-bottom: none;
    }
    
    .factor-label {
        width: 140px;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
    
    .factor-bar-track {
        flex: 1;
        height: 6px;
        background: var(--bg-main);
        border-radius: 3px;
        overflow: hidden;
        margin: 0 1rem;
    }
    
    .factor-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .factor-bar-fill.negative {
        background: linear-gradient(90deg, #EF4444, #EC4899);
    }
    
    .factor-bar-fill.positive {
        background: linear-gradient(90deg, #10B981, #06B6D4);
    }
    
    .factor-value {
        width: 50px;
        text-align: right;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* Action Items - Light Theme */
    .action-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 1rem;
        background: var(--bg-card);
        border-radius: 12px;
        margin-bottom: 0.5rem;
        border-left: 3px solid transparent;
        transition: all 0.2s;
        box-shadow: var(--shadow-sm);
    }
    
    .action-item:hover {
        box-shadow: var(--shadow-md);
    }
    
    .action-item.urgent {
        border-left-color: var(--red);
        background: rgba(239, 68, 68, 0.04);
    }
    
    .action-item.warning {
        border-left-color: var(--yellow);
        background: rgba(245, 158, 11, 0.04);
    }
    
    .action-item.info {
        border-left-color: var(--green);
        background: rgba(16, 185, 129, 0.04);
    }
    
    .action-icon-box {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
    }
    
    .action-item.urgent .action-icon-box {
        background: rgba(239, 68, 68, 0.1);
        color: var(--red);
    }
    
    .action-item.warning .action-icon-box {
        background: rgba(245, 158, 11, 0.1);
        color: var(--yellow);
    }
    
    .action-item.info .action-icon-box {
        background: rgba(16, 185, 129, 0.1);
        color: var(--green);
    }
    
    .action-content {
        flex: 1;
    }
    
    .action-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .action-desc {
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.4;
    }
    
    /* Buttons - Clean Style */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--violet) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 1.75rem;
        font-weight: 600;
        font-size: 0.9rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px var(--glow);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px var(--glow);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Inputs - Light Theme */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        padding: 0.75rem 1rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--violet));
    }
    
    /* Status Indicator */
    .status-live {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.8rem;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--green);
    }
    
    .status-live-dot {
        width: 6px;
        height: 6px;
        background: var(--green);
        border-radius: 50%;
        animation: pulse-dot 2s infinite;
    }
    
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }
    
    /* Footer */
    .footer-section {
        text-align: center;
        padding: 2rem 0;
        border-top: 1px solid var(--border);
        margin-top: 2rem;
    }
    
    .footer-brand {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .footer-sub {
        font-size: 0.8rem;
        color: var(--text-muted);
    }
    
    /* Scrollbar - Light Theme */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-main);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }
    
    /* Toast Notifications */
    .toast-notification {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: var(--shadow-lg);
        z-index: 9999;
        animation: slideIn 0.3s ease, fadeOut 0.3s ease 4.7s forwards;
        max-width: 350px;
    }
    
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    
    /* Theme Toggle Button */
    .theme-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        width: 44px;
        height: 44px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        z-index: 1000;
    }
    
    .theme-toggle:hover {
        transform: scale(1.05);
        box-shadow: var(--shadow-lg);
    }
    
    .theme-toggle i {
        font-size: 1.25rem;
        color: var(--text-secondary);
    }
    
    /* Export Dropdown */
    .export-dropdown {
        position: relative;
        display: inline-block;
    }
    
    .export-menu {
        position: absolute;
        top: 100%;
        right: 0;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: var(--shadow-lg);
        min-width: 180px;
        z-index: 100;
    }
    
    .export-menu a {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        color: var(--text-secondary);
        text-decoration: none;
        font-size: 0.85rem;
        transition: all 0.2s;
    }
    
    .export-menu a:hover {
        background: var(--bg-main);
        color: var(--text-primary);
    }
    
    /* Pulse Animation for Alerts */
    .pulse-alert {
        animation: pulseAlert 2s infinite;
    }
    
    @keyframes pulseAlert {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    }
    
    /* Last Updated Badge */
    .last-updated {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.75rem;
        background: rgba(99, 102, 241, 0.08);
        border-radius: 20px;
        font-size: 0.7rem;
        color: var(--text-muted);
    }
    
    .last-updated i {
        font-size: 0.8rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem !important;
        }
        
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr) !important;
        }
        
        .main-header {
            flex-direction: column;
            gap: 1rem;
        }
        
        .page-title {
            font-size: 1.25rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Constants
API_URL = "http://localhost:8080"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data with caching for performance."""
    try:
        processed = Path("data/processed/customers_featured.csv")
        raw = Path("data/raw/customers.csv")
        if processed.exists():
            return pd.read_csv(processed)
        elif raw.exists():
            return pd.read_csv(raw)
    except:
        pass
    
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'customer_id': [f"CUST_{str(i).zfill(6)}" for i in range(1, n+1)],
        'tenure_months': np.random.randint(1, 60, n),
        'monthly_spend': np.random.uniform(50, 1000, n).round(2),
        'total_orders': np.random.randint(1, 100, n),
        'days_since_last_order': np.random.randint(0, 180, n),
        'satisfaction_score': np.random.randint(1, 11, n),
        'churn': np.random.choice([0, 1], n, p=[0.74, 0.26]),
        'churn_probability': np.random.uniform(0, 1, n).round(3)
    })

def load_model_info():
    """Load model info."""
    try:
        with open("models/artifacts/model_info.json", 'r') as f:
            return json.load(f)
    except:
        return {'metrics': {'accuracy': 0.71, 'precision': 0.49, 'recall': 0.24, 'f1': 0.32, 'roc_auc': 0.67}}

def make_prediction(data):
    """Make API prediction."""
    try:
        r = requests.post(f"{API_URL}/predict", json=data, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def check_api():
    """Check API status."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

def export_to_csv(df, filename="export"):
    """Export DataFrame to CSV and return download link."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv" class="header-btn secondary"><i class="bi bi-download"></i> Download CSV</a>'

def export_to_excel(df, filename="export"):
    """Export DataFrame to Excel and return download link."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    b64 = base64.b64encode(output.getvalue()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx" class="header-btn primary"><i class="bi bi-file-earmark-excel"></i> Download Excel</a>'

def add_notification(message, type="info"):
    """Add a notification to the session state."""
    st.session_state.notifications.append({
        'message': message,
        'type': type,
        'time': datetime.now()
    })

def render_notifications():
    """Render toast notifications."""
    if st.session_state.notifications:
        for notif in st.session_state.notifications[-3:]:  # Show last 3
            icon = "bi-check-circle" if notif['type'] == 'success' else "bi-exclamation-triangle" if notif['type'] == 'warning' else "bi-info-circle"
            color = "var(--green)" if notif['type'] == 'success' else "var(--yellow)" if notif['type'] == 'warning' else "var(--primary)"
            st.markdown(f"""
            <div class="toast-notification" style="border-left-color: {color};">
                <i class="bi {icon}" style="color: {color};"></i>
                <span>{notif['message']}</span>
            </div>
            """, unsafe_allow_html=True)

def get_time_greeting():
    """Get appropriate greeting based on time of day."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"


def render_sidebar():
    """Render sidebar."""
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-brand">
                <div class="brand-icon"><i class="bi bi-bullseye"></i></div>
                <div class="brand-text">
                    <span class="brand-name">RetentionIQ</span>
                    <span class="brand-version">PRO 2.0</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown('<div class="nav-label">ANALYTICS</div>', unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["Dashboard", "Predictions", "Customers", "Performance", "Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Status
        api_online = check_api()
        if api_online:
            st.markdown("""
            <div class="status-live">
                <span class="status-live-dot"></span>
                API Connected
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="padding: 0.5rem;">
            <a href="http://localhost:8080/docs" target="_blank" style="color: var(--text-secondary); text-decoration: none; font-size: 0.8rem; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <i class="bi bi-journal-code"></i> API Documentation
            </a>
            <a href="http://localhost:5000" target="_blank" style="color: var(--text-secondary); text-decoration: none; font-size: 0.8rem; display: flex; align-items: center; gap: 0.5rem;">
                <i class="bi bi-bar-chart-line"></i> MLflow Dashboard
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        return page

def render_metric_card(icon, label, value, trend, trend_up, color):
    """Render metric card."""
    trend_class = "up" if trend_up else "down"
    arrow = "bi-arrow-up-right" if trend_up else "bi-arrow-down-right"
    return f"""
    <div class="metric-card">
        <div class="metric-header">
            <div class="metric-icon {color}"><i class="bi {icon}"></i></div>
            <span class="metric-trend {trend_class}"><i class="bi {arrow}"></i> {trend}</span>
        </div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

def render_dashboard(df, model_info):
    """Render dashboard with premium features."""
    
    # Dynamic greeting based on time of day
    greeting = get_time_greeting()
    current_time = datetime.now().strftime("%b %d, %Y • %I:%M %p")
    
    # Header with export functionality
    col_header, col_actions = st.columns([3, 1])
    
    with col_header:
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <div class="page-title">{greeting}, Mohamed 👋</div>
            <div class="page-subtitle">
                Here's your customer retention overview 
                <span class="last-updated"><i class="bi bi-clock"></i> {current_time}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_actions:
        # Export buttons
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📊 CSV",
                data=csv_data,
                file_name=f"retentioniq_export_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with export_col2:
            if st.button("📈 Report", use_container_width=True):
                add_notification("Report generation started!", "success")
                st.rerun()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(render_metric_card(
            "bi-people-fill", "Total Customers", f"{len(df):,}", "5.2%", True, "purple"
        ), unsafe_allow_html=True)
    
    with col2:
        churn_rate = df['churn'].mean() * 100 if 'churn' in df.columns else 28.3
        st.markdown(render_metric_card(
            "bi-graph-down-arrow", "Churn Rate", f"{churn_rate:.1f}%", "2.3%", False, "green"
        ), unsafe_allow_html=True)
    
    with col3:
        auc = model_info.get('metrics', {}).get('roc_auc', 0.67)
        st.markdown(render_metric_card(
            "bi-bullseye", "Model Accuracy", f"{auc:.0%}", "0.5%", True, "yellow"
        ), unsafe_allow_html=True)
    
    with col4:
        high_risk = len(df[df['churn_probability'] > 0.6]) if 'churn_probability' in df.columns else 187
        st.markdown(render_metric_card(
            "bi-exclamation-triangle-fill", "At Risk", f"{high_risk:,}", "23", False, "red"
        ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <span class="chart-title"><i class="bi bi-graph-up"></i> Churn Trend</span>
                <div class="chart-filters">
                    <span class="chart-filter-btn">7D</span>
                    <span class="chart-filter-btn active">30D</span>
                    <span class="chart-filter-btn">90D</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        np.random.seed(42)
        trend = pd.DataFrame({
            'date': dates,
            'rate': 28 + np.cumsum(np.random.randn(30) * 0.3)
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend['date'], y=trend['rate'],
            mode='lines',
            line=dict(color='#6366F1', width=3, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)',
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickfont=dict(color='#64748B'), gridcolor='rgba(255,255,255,0.03)', showgrid=False),
            yaxis=dict(tickfont=dict(color='#64748B'), gridcolor='rgba(255,255,255,0.05)', title=''),
            height=280,
            margin=dict(t=10, b=30, l=40, r=20),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <span class="chart-title"><i class="bi bi-pie-chart-fill"></i> Risk Distribution</span>
            </div>
        """, unsafe_allow_html=True)
        
        if 'churn_probability' in df.columns:
            df['risk'] = pd.cut(df['churn_probability'], bins=[0, 0.3, 0.6, 1], labels=['Low', 'Medium', 'High'])
            risk_counts = df['risk'].value_counts().reindex(['Low', 'Medium', 'High'])
            
            fig = go.Figure(data=[go.Pie(
                values=risk_counts.values,
                labels=risk_counts.index.tolist(),
                hole=0.65,
                marker=dict(colors=['#10B981', '#F59E0B', '#EF4444']),
                textinfo='percent',
                textfont=dict(size=12, color='white'),
            )])
            fig.add_annotation(
                text=f"<b>{len(df):,}</b><br><span style='font-size:10px'>Total</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=18, color='white')
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(font=dict(color='#94A3B8', size=11), orientation='h', y=-0.1),
                height=280,
                margin=dict(t=10, b=40, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # High Risk Table
    st.markdown("""
    <div class="chart-card">
        <div class="chart-header">
            <span class="chart-title"><i class="bi bi-exclamation-diamond"></i> High Risk Customers</span>
            <span class="chart-filter-btn">View All</span>
        </div>
    """, unsafe_allow_html=True)
    
    if 'churn_probability' in df.columns:
        high_risk_df = df.nlargest(5, 'churn_probability')[['customer_id', 'churn_probability', 'monthly_spend', 'days_since_last_order']].copy()
        high_risk_df.columns = ['Customer ID', 'Churn Risk', 'Monthly Spend', 'Days Inactive']
        high_risk_df['Churn Risk'] = high_risk_df['Churn Risk'].apply(lambda x: f"{x:.0%}")
        high_risk_df['Monthly Spend'] = high_risk_df['Monthly Spend'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(high_risk_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_predictions():
    """Render predictions page."""
    
    st.markdown("""
    <div class="main-header">
        <div>
            <div class="page-title">Churn Prediction</div>
            <div class="page-subtitle">Analyze individual customer churn risk</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title"><i class="bi bi-person-lines-fill"></i> Customer Profile</div>', unsafe_allow_html=True)
        
        customer_id = st.text_input("Customer ID", "NEW_001")
        
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.slider("Tenure (months)", 1, 60, 12)
            spend = st.number_input("Monthly Spend ($)", 0, 5000, 250)
            orders = st.slider("Total Orders", 0, 200, 15)
            days_since = st.slider("Days Since Last Order", 0, 180, 30)
            logins = st.slider("Login Frequency", 0, 50, 8)
            products = st.slider("Products Viewed", 0, 100, 15)
        
        with c2:
            avg_order = st.number_input("Avg Order Value ($)", 0, 1000, 75)
            cart = st.slider("Cart Abandonment %", 0, 100, 20)
            tickets = st.slider("Support Tickets", 0, 20, 2)
            discount = st.slider("Discount Usage %", 0, 100, 30)
            satisfaction = st.slider("Satisfaction (1-10)", 1, 10, 7)
            complaints = st.slider("Complaints", 0, 10, 1)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Analyze Churn Risk", use_container_width=True, type="primary"):
            st.session_state.predict_clicked = True
            st.session_state.customer_data = {
                "customer_id": customer_id,
                "tenure_months": tenure,
                "monthly_spend": float(spend),
                "total_orders": orders,
                "avg_order_value": float(avg_order),
                "days_since_last_order": days_since,
                "login_frequency": logins,
                "products_viewed": products,
                "cart_abandonment_rate": cart / 100,
                "support_tickets": tickets,
                "discount_usage_rate": discount / 100,
                "satisfaction_score": satisfaction,
                "complaint_count": complaints
            }
    
    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title"><i class="bi bi-speedometer2"></i> Risk Analysis</div>', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'predict_clicked') and st.session_state.predict_clicked:
            with st.spinner("Analyzing..."):
                result = make_prediction(st.session_state.customer_data)
            
            if result:
                prob = result['churn_probability']
                risk = result['risk_segment']
                
                st.markdown(f"""
                <div class="prediction-result-container">
                    <div class="prediction-percentage {risk}">{prob:.0%}</div>
                    <div class="risk-pill {risk}">
                        <i class="bi bi-{'exclamation-triangle-fill' if risk == 'high' else ('exclamation-circle' if risk == 'medium' else 'check-circle-fill')}"></i>
                        {risk.upper()} RISK
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="chart-title" style="margin-top: 1.5rem;"><i class="bi bi-bar-chart-steps"></i> Key Factors</div>', unsafe_allow_html=True)
                
                for f in result.get('top_churn_factors', [])[:3]:
                    bar_class = "negative" if f['direction'] == 'increases' else "positive"
                    width = int(f['importance'] * 100)
                    st.markdown(f"""
                    <div class="factor-row">
                        <span class="factor-label">{f['feature']}</span>
                        <div class="factor-bar-track">
                            <div class="factor-bar-fill {bar_class}" style="width: {width}%;"></div>
                        </div>
                        <span class="factor-value">{width}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('<div class="chart-title" style="margin-top: 1.5rem;"><i class="bi bi-lightning-charge"></i> Recommended Actions</div>', unsafe_allow_html=True)
                
                if risk == 'high':
                    actions = [
                        ("urgent", "bi-telephone-fill", "Immediate Outreach", "Contact customer within 24 hours"),
                        ("urgent", "bi-gift-fill", "Retention Offer", "Provide 25% loyalty discount"),
                        ("warning", "bi-chat-dots-fill", "Feedback Call", "Schedule to understand concerns")
                    ]
                elif risk == 'medium':
                    actions = [
                        ("warning", "bi-envelope-fill", "Re-engagement Email", "Send personalized campaign"),
                        ("info", "bi-megaphone-fill", "Nurture Campaign", "Add to retention sequence")
                    ]
                else:
                    actions = [
                        ("info", "bi-star-fill", "Loyalty Upgrade", "Consider for VIP program"),
                        ("info", "bi-share-fill", "Referral Invite", "Encourage referrals")
                    ]
                
                for action_type, icon, title, desc in actions:
                    st.markdown(f"""
                    <div class="action-item {action_type}">
                        <div class="action-icon-box"><i class="bi {icon}"></i></div>
                        <div class="action-content">
                            <div class="action-title">{title}</div>
                            <div class="action-desc">{desc}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Could not connect to API")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; color: var(--up-text-dim);">
                <i class="bi bi-lightning-charge" style="font-size: 4rem; color: var(--up-purple);"></i>
                <p style="margin-top: 1rem;">Enter customer details and click <strong>Analyze</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_customers(df):
    """Render customers page."""
    st.markdown("""
    <div class="main-header">
        <div>
            <div class="page-title">Customer Segments</div>
            <div class="page-subtitle">Analyze customer behavior and segments</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title"><i class="bi bi-diagram-3-fill"></i> RFM Analysis</div>', unsafe_allow_html=True)
    
    if 'days_since_last_order' in df.columns and 'total_orders' in df.columns:
        fig = px.scatter(
            df.head(500),
            x='days_since_last_order', y='total_orders',
            color='churn_probability' if 'churn_probability' in df.columns else None,
            size='monthly_spend' if 'monthly_spend' in df.columns else None,
            color_continuous_scale=[[0, '#10B981'], [0.5, '#F59E0B'], [1, '#EF4444']],
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Recency (Days)', tickfont=dict(color='#64748B'), gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title='Frequency (Orders)', tickfont=dict(color='#64748B'), gridcolor='rgba(255,255,255,0.05)'),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_performance(model_info):
    """Render performance page."""
    
    st.markdown("""
    <div class="main-header">
        <div>
            <div class="page-title">Model Performance</div>
            <div class="page-subtitle">Monitor your ML model metrics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    metrics = model_info.get('metrics', {})
    
    cols = st.columns(5)
    for col, (name, key, icon) in zip(cols, [
        ("Accuracy", "accuracy", "bi-check-circle"),
        ("Precision", "precision", "bi-crosshair"),
        ("Recall", "recall", "bi-arrow-repeat"),
        ("F1 Score", "f1", "bi-trophy"),
        ("AUC-ROC", "roc_auc", "bi-graph-up-arrow")
    ]):
        with col:
            val = metrics.get(key, 0.8)
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <div class="metric-icon purple" style="margin: 0 auto 0.75rem auto;"><i class="bi {icon}"></i></div>
                <div class="metric-label">{name}</div>
                <div class="metric-value" style="font-size: 1.5rem;">{val:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="chart-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title"><i class="bi bi-radar"></i> Performance Radar</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[metrics.get(k, 0.8) for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']],
        theta=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC'],
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(color='#6366F1', width=2)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color='#64748B'), gridcolor='rgba(255,255,255,0.08)'),
            angularaxis=dict(tickfont=dict(color='#94A3B8', size=12))
        ),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_settings():
    """Render settings."""
    st.markdown("""
    <div class="main-header">
        <div>
            <div class="page-title">Settings</div>
            <div class="page-subtitle">Configure your dashboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title"><i class="bi bi-gear-fill"></i> Configuration</div>', unsafe_allow_html=True)
    st.text_input("API Endpoint", API_URL)
    st.slider("Risk Threshold", 0.0, 1.0, 0.5)
    st.selectbox("Model", ["Logistic Regression", "Random Forest", "XGBoost"])
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    """Main function."""
    
    df = load_data()
    model_info = load_model_info()
    
    page = render_sidebar()
    
    if page == "Dashboard":
        render_dashboard(df, model_info)
    elif page == "Predictions":
        render_predictions()
    elif page == "Customers":
        render_customers(df)
    elif page == "Performance":
        render_performance(model_info)
    elif page == "Settings":
        render_settings()
    
    # Footer
    st.markdown("""
    <div class="footer-section">
        <div class="footer-brand">RetentionIQ Pro</div>
        <div class="footer-sub">Built with <i class="bi bi-heart-fill" style="color: #EF4444;"></i> by Noor</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
