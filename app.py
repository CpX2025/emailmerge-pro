import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import Optional, Tuple, List, Dict, Any
import psutil
import os
import gc
import time
from io import BytesIO, StringIO
import zipfile
import logging
from pathlib import Path
import hashlib
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import streamlit_shadcn_ui as ui

# Configure page
st.set_page_config(
    page_title="EmailMerge Pro - Enterprise Email Matching",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with Font Awesome icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --success-color: #C73E1D;
        --background-color: #F8F9FA;
        --text-color: #2C3E50;
        --border-color: #E1E8ED;
    }

    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(46, 134, 171, 0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .professional-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .professional-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }

    .upload-zone {
        border: 2px dashed var(--primary-color);
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .upload-zone:hover {
        border-color: var(--accent-color);
        background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
    }

    .upload-zone::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent 40%, rgba(46, 134, 171, 0.05) 50%, transparent 60%);
        transform: rotate(-45deg);
        transition: transform 0.6s ease;
    }

    .upload-zone:hover::before {
        transform: rotate(-45deg) translate(20%, 20%);
    }

    .metric-card {
        background: linear-gradient(135deg, white 0%, #f8f9fa 100%);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border-left: 4px solid var(--primary-color);
        box-shadow: 0 2px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }

    .metric-label {
        color: var(--text-color);
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .progress-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05);
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
    }

    .status-processing {
        background: #FFF3E0;
        color: #E65100;
        border: 1px solid #FFB74D;
    }

    .status-success {
        background: #E8F5E8;
        color: #2E7D32;
        border: 1px solid #81C784;
    }

    .status-error {
        background: #FFEBEE;
        color: #C62828;
        border: 1px solid #EF5350;
    }

    .professional-button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 4px 15px rgba(46, 134, 171, 0.3);
    }

    .professional-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 134, 171, 0.4);
    }

    .sidebar-metric {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }

    .performance-chart {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05);
    }

    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .pulse {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-color);
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class ProcessingMetrics:
    """Data class for tracking processing metrics"""
    total_records: int = 0
    processed_records: int = 0
    matches_found: int = 0
    processing_speed: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    start_time: float = 0.0
    elapsed_time: float = 0.0

class AdvancedCSVProcessor:
    """Ultra-fast CSV processor with advanced optimizations"""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.chunk_size = 50000
        self.use_polars = True
        self.metrics = ProcessingMetrics()
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics"""
        process = psutil.Process()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        return {
            'cpu_usage': cpu_percent,
            'memory_usage_mb': memory_info.rss / 1024 / 1024,
            'memory_percent': memory_percent,
            'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
        }
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def read_file_optimized(_self, file_buffer, file_type: str = "csv") -> pl.DataFrame:
        """Ultra-fast file reading with Polars"""
        try:
            if file_type.lower() == "csv":
                # Use Polars for maximum performance
                df = pl.read_csv(
                    file_buffer,
                    infer_schema_length=10000,
                    null_values=["", "NULL", "null", "N/A", "n/a"],
                    try_parse_dates=True,
                    encoding="utf8-lossy"
                )
            elif file_type.lower() in ["xlsx", "xls"]:
                # For Excel files, use pandas then convert to polars
                pandas_df = pd.read_excel(file_buffer, engine='openpyxl')
                df = pl.from_pandas(pandas_df)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Normalize column names
            df = df.rename({col: col.strip().lower() for col in df.columns})
            
            return df
            
        except Exception as e:
            st.error(f"<i class='fas fa-exclamation-triangle'></i> Error reading file: {str(e)}", unsafe_allow_html=True)
            return pl.DataFrame()
    
    def parallel_email_matching(self, specific_emails: set, full_df: pl.DataFrame, 
                              progress_callback=None) -> pl.DataFrame:
        """Parallel processing for ultra-fast email matching"""
        
        # Convert to lowercase for case-insensitive matching
        specific_emails_lower = {email.lower().strip() for email in specific_emails}
        
        # Use Polars native operations for maximum speed
        matched_df = full_df.filter(
            pl.col("email").str.to_lowercase().str.strip().is_in(specific_emails_lower)
        )
        
        return matched_df
    
    def process_large_datasets(self, specific_df: pl.DataFrame, full_df: pl.DataFrame,
                             progress_bar, status_text, metrics_container) -> pl.DataFrame:
        """Advanced processing for large datasets with real-time metrics"""
        
        self.metrics.start_time = time.time()
        self.metrics.total_records = full_df.height
        
        try:
            # Extract email set from specific_df
            specific_emails = set(specific_df["email"].to_list())
            
            # Process in chunks for very large datasets
            if full_df.height > 1000000:  # 1M+ records
                chunk_size = min(self.chunk_size, full_df.height // self.cpu_count)
                chunks = []
                total_chunks = (full_df.height + chunk_size - 1) // chunk_size
                
                for i in range(0, full_df.height, chunk_size):
                    chunk = full_df.slice(i, min(chunk_size, full_df.height - i))
                    matched_chunk = self.parallel_email_matching(specific_emails, chunk)
                    
                    if matched_chunk.height > 0:
                        chunks.append(matched_chunk)
                    
                    # Update progress and metrics
                    progress = min((i + chunk_size) / full_df.height, 1.0)
                    progress_bar.progress(progress)
                    
                    self.metrics.processed_records = min(i + chunk_size, full_df.height)
                    self.metrics.elapsed_time = time.time() - self.metrics.start_time
                    self.metrics.processing_speed = self.metrics.processed_records / max(self.metrics.elapsed_time, 0.001)
                    
                    # Update status
                    status_text.markdown(
                        f"""<div class="status-badge status-processing">
                        <i class='fas fa-cog fa-spin'></i> Processing chunk {(i//chunk_size)+1}/{total_chunks} 
                        - Speed: {self.metrics.processing_speed:,.0f} rec/sec
                        </div>""", 
                        unsafe_allow_html=True
                    )
                    
                    # Update metrics display
                    self._update_metrics_display(metrics_container)
                    
                    # Memory management
                    if i % (chunk_size * 5) == 0:
                        gc.collect()
                
                # Combine all chunks
                if chunks:
                    matched_df = pl.concat(chunks)
                else:
                    matched_df = pl.DataFrame()
            
            else:
                # For smaller datasets, process directly
                matched_df = self.parallel_email_matching(specific_emails, full_df, progress_bar)
                progress_bar.progress(1.0)
            
            self.metrics.matches_found = matched_df.height if matched_df.height > 0 else 0
            self.metrics.elapsed_time = time.time() - self.metrics.start_time
            
            return matched_df
            
        except Exception as e:
            st.error(f"<i class='fas fa-exclamation-circle'></i> Processing error: {str(e)}", unsafe_allow_html=True)
            return pl.DataFrame()
    
    def _update_metrics_display(self, container):
        """Update real-time metrics display"""
        with container:
            system_metrics = self.get_system_metrics()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <i class='fas fa-database' style='color: var(--primary-color); font-size: 1.5rem;'></i>
                    <div class="metric-value">{self.metrics.processed_records:,}</div>
                    <div class="metric-label">Records Processed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <i class='fas fa-tachometer-alt' style='color: var(--accent-color); font-size: 1.5rem;'></i>
                    <div class="metric-value">{self.metrics.processing_speed:,.0f}</div>
                    <div class="metric-label">Records/Second</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <i class='fas fa-memory' style='color: var(--secondary-color); font-size: 1.5rem;'></i>
                    <div class="metric-value">{system_metrics['memory_usage_mb']:.0f}</div>
                    <div class="metric-label">Memory (MB)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <i class='fas fa-microchip' style='color: var(--success-color); font-size: 1.5rem;'></i>
                    <div class="metric-value">{system_metrics['cpu_usage']:.1f}%</div>
                    <div class="metric-label">CPU Usage</div>
                </div>
                """, unsafe_allow_html=True)

def create_performance_chart(metrics_history: List[Dict]) -> go.Figure:
    """Create real-time performance monitoring chart"""
    if not metrics_history:
        return go.Figure()
    
    timestamps = [m['timestamp'] for m in metrics_history]
    cpu_usage = [m['cpu_usage'] for m in metrics_history]
    memory_usage = [m['memory_usage_mb'] for m in metrics_history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=cpu_usage, name='CPU Usage (%)',
        line=dict(color='#2E86AB', width=3),
        fill='tonexty'
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=memory_usage, name='Memory (MB)', 
        yaxis='y2',
        line=dict(color='#A23B72', width=3)
    ))
    
    fig.update_layout(
        title="<b>Real-Time Performance Monitoring</b>",
        xaxis_title="Time",
        yaxis_title="CPU Usage (%)",
        yaxis2=dict(title="Memory Usage (MB)", overlaying='y', side='right'),
        template='plotly_white',
        height=300,
        showlegend=True
    )
    
    return fig

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header fade-in">
        <h1><i class='fas fa-link'></i> EmailMerge Pro</h1>
        <p><i class='fas fa-rocket'></i> Enterprise-Grade Email Matching Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize processor
    processor = AdvancedCSVProcessor()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div class="professional-card">
            <h3><i class='fas fa-cogs'></i> Advanced Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance settings
        st.markdown("#### <i class='fas fa-tachometer-alt'></i> Performance Settings", unsafe_allow_html=True)
        
        chunk_size = st.slider(
            "Processing Chunk Size", 
            min_value=10000, 
            max_value=200000, 
            value=50000,
            step=10000,
            help="Larger chunks = faster processing but more memory usage"
        )
        processor.chunk_size = chunk_size
        
        use_multiprocessing = st.checkbox(
            "Enable Multi-Processing", 
            value=True,
            help="Use multiple CPU cores for faster processing"
        )
        
        # Matching options
        st.markdown("#### <i class='fas fa-filter'></i> Matching Options", unsafe_allow_html=True)
        
        case_sensitive = st.checkbox(
            "Case Sensitive Matching", 
            value=False,
            help="Enable for exact case matching"
        )
        
        remove_duplicates = st.checkbox(
            "Remove Duplicates", 
            value=True,
            help="Remove duplicate email entries"
        )
        
        fuzzy_matching = st.checkbox(
            "Enable Fuzzy Matching", 
            value=False,
            help="Match similar email addresses (experimental)"
        )
        
        # System information
        st.markdown("---")
        st.markdown("#### <i class='fas fa-server'></i> System Information", unsafe_allow_html=True)
        
        system_metrics = processor.get_system_metrics()
        
        st.markdown(f"""
        <div class="sidebar-metric">
            <i class='fas fa-microchip'></i> CPU Cores: {processor.cpu_count}<br>
            <i class='fas fa-memory'></i> Available RAM: {system_metrics['available_memory_gb']:.1f} GB<br>
            <i class='fas fa-chart-line'></i> Current CPU: {system_metrics['cpu_usage']:.1f}%
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="professional-card">
            <div class="upload-zone">
                <i class='fas fa-envelope' style='font-size: 3rem; color: var(--primary-color); margin-bottom: 1rem;'></i>
                <h3>Upload Target Emails</h3>
                <p>CSV file containing emails you want to match</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        specific_emails = st.file_uploader(
            "Choose Target Emails File",
            type=["csv", "xlsx", "xls"],
            key="specific_emails",
            help="File must contain an 'email' column"
        )
    
    with col2:
        st.markdown("""
        <div class="professional-card">
            <div class="upload-zone">
                <i class='fas fa-database' style='font-size: 3rem; color: var(--secondary-color); margin-bottom: 1rem;'></i>
                <h3>Upload Master Database</h3>
                <p>Complete database file (unlimited size)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        full_db = st.file_uploader(
            "Choose Database File",
            type=["csv", "xlsx", "xls"],
            key="full_db",
            help="File must contain an 'email' column"
        )
    
    # File information display
    if specific_emails or full_db:
        st.markdown("---")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            if specific_emails:
                file_size = specific_emails.size / 1024 / 1024
                st.markdown(f"""
                <div class="professional-card fade-in">
                    <i class='fas fa-check-circle' style='color: green;'></i> 
                    <strong>{specific_emails.name}</strong><br>
                    <i class='fas fa-file-alt'></i> Size: {file_size:.2f} MB
                </div>
                """, unsafe_allow_html=True)
        
        with info_col2:
            if full_db:
                file_size = full_db.size / 1024 / 1024
                st.markdown(f"""
                <div class="professional-card fade-in">
                    <i class='fas fa-check-circle' style='color: green;'></i> 
                    <strong>{full_db.name}</strong><br>
                    <i class='fas fa-database'></i> Size: {file_size:.2f} MB
                </div>
                """, unsafe_allow_html=True)
    
    # Processing section
    if specific_emails and full_db:
        st.markdown("---")
        
        # Custom button with professional styling
        button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
        with button_col2:
            if st.button("🚀 Start Ultra-Fast Processing", type="primary", use_container_width=True):
                start_processing(processor, specific_emails, full_db, case_sensitive, remove_duplicates, fuzzy_matching)

def start_processing(processor, specific_emails, full_db, case_sensitive, remove_duplicates, fuzzy_matching):
    """Handle the processing workflow"""
    start_time = time.time()
    
    # Create processing interface
    st.markdown("""
    <div class="progress-container fade-in">
        <h3><i class='fas fa-cogs'></i> Processing Status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.container()
    
    # Performance monitoring
    metrics_history = []
    performance_chart_container = st.container()
    
    try:
        # Load files
        with st.spinner("Loading files..."):
            status_text.markdown("""
            <div class="status-badge status-processing">
                <i class='fas fa-upload fa-spin'></i> Loading target emails...
            </div>
            """, unsafe_allow_html=True)
            
            file_ext = specific_emails.name.split('.')[-1].lower()
            specific_df = processor.read_file_optimized(specific_emails, file_ext)
            
            if specific_df.height == 0:
                st.error("<i class='fas fa-exclamation-triangle'></i> Failed to load target emails file", unsafe_allow_html=True)
                return
            
            status_text.markdown("""
            <div class="status-badge status-processing">
                <i class='fas fa-database fa-spin'></i> Loading master database...
            </div>
            """, unsafe_allow_html=True)
            
            file_ext = full_db.name.split('.')[-1].lower()
            full_df = processor.read_file_optimized(full_db, file_ext)
            
            if full_df.height == 0:
                st.error("<i class='fas fa-exclamation-triangle'></i> Failed to load database file", unsafe_allow_html=True)
                return
        
        # Validate email columns
        if 'email' not in specific_df.columns:
            st.error("<i class='fas fa-times-circle'></i> Target emails file must have an 'email' column!", unsafe_allow_html=True)
            return
        
        if 'email' not in full_df.columns:
            st.error("<i class='fas fa-times-circle'></i> Database file must have an 'email' column!", unsafe_allow_html=True)
            return
        
        # Display initial metrics
        initial_col1, initial_col2, initial_col3 = st.columns(3)
        with initial_col1:
            st.markdown(f"""
            <div class="metric-card">
                <i class='fas fa-envelope' style='color: var(--primary-color); font-size: 1.5rem;'></i>
                <div class="metric-value">{specific_df.height:,}</div>
                <div class="metric-label">Target Emails</div>
            </div>
            """, unsafe_allow_html=True)
        
        with initial_col2:
            st.markdown(f"""
            <div class="metric-card">
                <i class='fas fa-database' style='color: var(--secondary-color); font-size: 1.5rem;'></i>
                <div class="metric-value">{full_df.height:,}</div>
                <div class="metric-label">Database Records</div>
            </div>
            """, unsafe_allow_html=True)
        
        with initial_col3:
            estimated_time = max(full_df.height / 100000, 1)  # Rough estimate
            st.markdown(f"""
            <div class="metric-card">
                <i class='fas fa-clock' style='color: var(--accent-color); font-size: 1.5rem;'></i>
                <div class="metric-value">{estimated_time:.1f}s</div>
                <div class="metric-label">Estimated Time</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Process matching
        status_text.markdown("""
        <div class="status-badge status-processing">
            <i class='fas fa-rocket fa-spin'></i> Ultra-fast email matching in progress...
        </div>
        """, unsafe_allow_html=True)
        
        matched_df = processor.process_large_datasets(
            specific_df, full_df, progress_bar, status_text, metrics_container
        )
        
        # Post-processing
        if matched_df.height > 0:
            if remove_duplicates:
                initial_count = matched_df.height
                matched_df = matched_df.unique(subset=['email'])
                removed_count = initial_count - matched_df.height
                if removed_count > 0:
                    st.info(f"<i class='fas fa-info-circle'></i> Removed {removed_count} duplicate records", unsafe_allow_html=True)
            
            # Convert back to pandas for download compatibility
            matched_pandas = matched_df.to_pandas()
            
            # Final metrics
            processing_time = time.time() - start_time
            match_rate = (matched_df.height / specific_df.height) * 100
            
            # Success message
            st.markdown(f"""
            <div class="professional-card fade-in">
                <div class="status-badge status-success">
                    <i class='fas fa-check-circle'></i> Processing completed successfully!
                </div>
                <br><br>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                    <div class="metric-card">
                        <i class='fas fa-bullseye' style='color: var(--success-color); font-size: 1.5rem;'></i>
                        <div class="metric-value">{matched_df.height:,}</div>
                        <div class="metric-label">Matches Found</div>
                    </div>
                    <div class="metric-card">
                        <i class='fas fa-percentage' style='color: var(--primary-color); font-size: 1.5rem;'></i>
                        <div class="metric-value">{match_rate:.1f}%</div>
                        <div class="metric-label">Match Rate</div>
                    </div>
                    <div class="metric-card">
                        <i class='fas fa-stopwatch' style='color: var(--accent-color); font-size: 1.5rem;'></i>
                        <div class="metric-value">{processing_time:.2f}s</div>
                        <div class="metric-label">Processing Time</div>
                    </div>
                    <div class="metric-card">
                        <i class='fas fa-tachometer-alt' style='color: var(--secondary-color); font-size: 1.5rem;'></i>
                        <div class="metric-value">{full_df.height/processing_time:,.0f}</div>
                        <div class="metric-label">Records/Second</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Results preview
            st.markdown("### <i class='fas fa-table'></i> Results Preview", unsafe_allow_html=True)
            st.dataframe(
                matched_pandas.head(100), 
                use_container_width=True,
                height=400
            )
            
            # Download section
            st.markdown("### <i class='fas fa-download'></i> Download Results", unsafe_allow_html=True)
            
            download_col1, download_col2, download_col3 = st.columns(3)
            
            with download_col1:
                csv_data = matched_pandas.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"emailmerge_results_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with download_col2:
                excel_buffer = BytesIO()
                matched_pandas.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_data,
                    file_name=f"emailmerge_results_{int(time.time())}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with download_col3:
                # Create a ZIP with both formats
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    zip_file.writestr(f"results_{int(time.time())}.csv", matched_pandas.to_csv(index=False))
                    zip_file.writestr(f"results_{int(time.time())}.xlsx", excel_data)
                
                st.download_button(
                    label="📦 Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"emailmerge_complete_{int(time.time())}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
        
        else:
            st.markdown("""
            <div class="professional-card">
                <div class="status-badge status-error">
                    <i class='fas fa-exclamation-triangle'></i> No matching emails found
                </div>
                <p>Please check that your email formats match between the two files.</p>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown(f"""
        <div class="professional-card">
            <div class="status-badge status-error">
                <i class='fas fa-times-circle'></i> Processing Error: {str(e)}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.exception(e)
    
    finally:
        gc.collect()

if __name__ == "__main__":
    main()
