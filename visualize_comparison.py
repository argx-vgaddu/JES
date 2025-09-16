#!/usr/bin/env python3
"""
Advanced SAS Viya Job Execution Comparison Visualizer

Creates comprehensive visualizations and analysis for three-way job execution comparison:
1. Sequential Execution (Baseline - 1 node)
2. Async Regular (Parallelization in regular context)  
3. Async Autoscaling (Parallelization + autoscaling context)

Features:
- Resource-adjusted performance analysis
- Queue wait time breakdown visualization
- Job-node-context overlay analysis
- Parallel efficiency calculations
- Executive dashboard with key insights
- Excel data integration for accurate metrics

Usage:
    python visualize_comparison.py [json_file_path|excel_file_path]
    
If no file path provided, uses the latest comparison file from results/ folder.
Prefers Excel files (from extract_and_analyze.py) for accurate queue wait times.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

# Suppress matplotlib warnings about GUI backends
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Import visualization libraries (install if needed)
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import numpy as np
    import pandas as pd
    import openpyxl
except ImportError:
    print("Installing required visualization libraries...")
    os.system("pip install matplotlib numpy pandas openpyxl")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import numpy as np
    import pandas as pd
    import openpyxl

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

# Configure matplotlib for better-looking plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Professional color scheme for three-way analysis
COLORS = {
    'sequential': '#E74C3C',      # Red - baseline
    'async_regular': '#F39C12',    # Orange - parallelization only
    'async_autoscaling': '#27AE60', # Green - parallelization + autoscaling
    'autoscaling': '#27AE60',     # Green - backward compatibility
    'queue_wait': '#3498DB',       # Blue - queue time
    'execution': '#2ECC71',        # Light green - execution time
    'success': '#90EE90',         # Light green
    'failed': '#95A5A6',          # Gray - failed jobs
    'neutral': '#D3D3D3',        # Light gray
    'accent': '#FFD700',         # Gold
    'background': '#ECF0F1',      # Light gray background
    'text': '#2C3E50',            # Dark blue-gray text
    'grid': '#BDC3C7'             # Light gray grid
}


class JobComparisonVisualizer:
    """Advanced three-way performance analysis with comprehensive visualizations."""
    
    def __init__(self, data_source: str):
        """Initialize with data from Excel sheet or JSON file."""
        self.data_source = data_source
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Determine data source type and load accordingly
        if data_source.endswith('.xlsx'):
            print(f"📊 Loading data from Excel sheet: {data_source}")
            self.df = self._load_excel_data(data_source)
            self.data_type = 'excel'
            self._extract_excel_data()
        else:
            print(f"📊 Loading data from JSON file: {data_source}")
            self.data = self._load_json_data(data_source)
            self.data_type = 'json'
            self._extract_json_data()
        
        # Initialize analysis results
        self.analysis_results = {}
    
    @property
    def async_jobs(self):
        """Backward compatibility property - returns autoscaling jobs as primary async"""
        return self.async_autoscaling_jobs if self.async_autoscaling_jobs else self.async_regular_jobs
    
    def _load_excel_data(self, excel_path: str) -> pd.DataFrame:
        """Load and validate Excel orchestrator analysis data."""
        try:
            # Load the main data sheet (All_Executions contains all the job data)
            df = pd.read_excel(excel_path, sheet_name='All_Executions')
            
            print(f"📊 Loaded {len(df)} jobs from Excel:")
            if 'execution_type' in df.columns:
                for mode in df['execution_type'].unique():
                    count = len(df[df['execution_type'] == mode])
                    print(f"   {mode}: {count} jobs")
            else:
                print("   ⚠️ No execution_type column found")
            
            return df
        except Exception as e:
            raise ValueError(f"Failed to load Excel file {excel_path}: {e}")
    
    def _load_json_data(self, json_path: str) -> Dict:
        """Load and validate JSON comparison data."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise ValueError(f"Failed to load JSON file {json_path}: {e}")
    
    def _extract_excel_data(self):
        """Extract job data from Excel format."""
        # Extract job lists for each execution mode using correct Excel values
        self.sequential_jobs = self._get_jobs_from_excel('Sequential')
        self.async_regular_jobs = self._get_jobs_from_excel('Async Regular')
        self.async_autoscaling_jobs = self._get_jobs_from_excel('Async Autoscaling')
        
        # Calculate execution times and populate comparison object
        self.comparison = self._calculate_excel_comparison_data()
        
        print(f"📊 Extracted Excel Data:")
        print(f"   Sequential Jobs: {len(self.sequential_jobs)}")
        print(f"   Async Regular Jobs: {len(self.async_regular_jobs)}")  
        print(f"   Async Autoscaling Jobs: {len(self.async_autoscaling_jobs)}")
    
    def _calculate_excel_comparison_data(self) -> Dict[str, Any]:
        """Calculate comparison metrics from Excel data."""
        # Calculate total execution times
        seq_total = sum(job.get('duration', 0) for job in self.sequential_jobs)
        async_reg_total = sum(job.get('duration', 0) for job in self.async_regular_jobs)
        async_auto_total = sum(job.get('duration', 0) for job in self.async_autoscaling_jobs)
        
        # Calculate success rates
        seq_completed = len([j for j in self.sequential_jobs if j.get('status') == 'completed'])
        async_reg_completed = len([j for j in self.async_regular_jobs if j.get('status') == 'completed'])
        async_auto_completed = len([j for j in self.async_autoscaling_jobs if j.get('status') == 'completed'])
        
        seq_success_rate = (seq_completed / len(self.sequential_jobs) * 100) if self.sequential_jobs else 0
        async_reg_success_rate = (async_reg_completed / len(self.async_regular_jobs) * 100) if self.async_regular_jobs else 0
        async_auto_success_rate = (async_auto_completed / len(self.async_autoscaling_jobs) * 100) if self.async_autoscaling_jobs else 0
        
        # Calculate time savings and efficiency gains (using autoscaling as primary async)
        time_saved = seq_total - async_auto_total
        efficiency_gain = (time_saved / seq_total * 100) if seq_total > 0 else 0
        
        return {
            'execution_time': {
                'sequential_total': seq_total,
                'async_regular_total': async_reg_total,
                'async_autoscaling_total': async_auto_total,
                'async_total': async_auto_total,  # For backward compatibility
                'time_saved': time_saved,
                'efficiency_gain': efficiency_gain
            },
            'job_performance': {
                'sequential_success_rate': seq_success_rate,
                'async_regular_success_rate': async_reg_success_rate,
                'async_autoscaling_success_rate': async_auto_success_rate,
                'async_success_rate': async_auto_success_rate  # For backward compatibility
            }
        }
    
    def _extract_json_data(self):
        """Extract job data from JSON format."""
        # Extract key metrics from three execution modes
        self.sequential_jobs = self.data.get('sequential_execution', {}).get('jobs', [])
        self.async_regular_jobs = self.data.get('async_regular_execution', {}).get('jobs', [])
        self.async_autoscaling_jobs = self.data.get('async_autoscaling_execution', {}).get('jobs', [])
        self.comparison = self.data.get('comparison', {})
        
        # For backward compatibility, if old format exists, use autoscaling as primary async
        if not self.async_autoscaling_jobs and 'async_execution' in self.data:
            self.async_autoscaling_jobs = self.data.get('async_execution', {}).get('jobs', [])
        
        print(f"📊 Loaded JSON Data:")
        print(f"   Sequential Jobs: {len(self.sequential_jobs)}")
        print(f"   Async Regular Jobs: {len(self.async_regular_jobs)}")  
        print(f"   Async Autoscaling Jobs: {len(self.async_autoscaling_jobs)}")
    
    def _get_jobs_from_excel(self, execution_type: str) -> List[Dict]:
        """Convert Excel data to job list format for visualization."""
        if self.data_type != 'excel':
            return []
        
        # Use the execution_type directly as it appears in Excel
        mode_data = self.df[self.df['execution_type'] == execution_type]
        if mode_data.empty:
            print(f"⚠️ No data found for execution type: {execution_type}")
            return []
        
        jobs = []
        for idx, row in mode_data.iterrows():
            # Create flat job structure for Excel data (no fake nesting)
            job = {
                'name': row.get('execution_name', f'Job {idx}'),
                'duration': row.get('duration', 0),
                'status': row.get('status', 'completed'),
                        'total_time_pending': row.get('total_time_pending', 0),
                'total_time_running': row.get('total_time_running', row.get('duration', 0)),
                'execution_host': row.get('execution_host', 'unknown-node'),
                'queue_wait_time': row.get('total_time_pending', 0),  # For backward compatibility
                'execution_time': row.get('total_time_running', row.get('duration', 0)),  # For backward compatibility
                'node_name': row.get('execution_host', 'unknown-node'),  # For backward compatibility
                'submit_time': row.get('submit_time', ''),
                'start_time': row.get('start_time', ''),
                'end_time': row.get('end_time', ''),
                'execution_seconds': row.get('execution_seconds', row.get('total_time_running', 0)),
                'queue_wait_seconds': row.get('queue_wait_seconds', row.get('total_time_pending', 0)),
                'cpu_cores': row.get('cpu_cores', 0),
                'memory_mb': row.get('memory_mb', 0),
                'max_memory_used': row.get('max_memory_used', 0),
                'max_cpu_time': row.get('max_cpu_time', 0),
                'max_io_total': row.get('max_io_total', 0),
            }
            jobs.append(job)
        
        print(f"📊 Extracted {len(jobs)} jobs for {execution_type}")
        return jobs
    
    def _create_node_mapping(self, jobs_list: List[List[Dict]]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Create mapping between full node names and simple labels (node1, node2, etc.)"""
        # Collect all unique node names from all job lists
        all_nodes = set()
        
        for jobs in jobs_list:
            for job in jobs:
                if self.data_type == 'excel':
                    node_name = job.get('node_name', 'unknown-node')
                else:
                    node_name = job.get('metrics', {}).get('actual_hostname', 
                                job.get('metrics', {}).get('orchestrator_data', {}).get('execution_host', 'unknown-node'))
                all_nodes.add(node_name)
        
        # Remove unknown nodes and sort
        unique_nodes = sorted([node for node in all_nodes if 'unknown' not in node.lower()])
        
        # Create mappings
        node_to_simple = {}
        simple_to_node = {}
        
        for i, node in enumerate(unique_nodes):
            simple_label = f"node{i+1}"
            node_to_simple[node] = simple_label
            simple_to_node[simple_label] = node
        
        return node_to_simple, simple_to_node
    
    def _get_simple_node_name(self, job: Dict, node_mapping: Dict[str, str]) -> str:
        """Get simple node name (node1, node2) for a job"""
        if self.data_type == 'excel':
            full_node = job.get('node_name', 'unknown-node')
        else:
            full_node = job.get('metrics', {}).get('actual_hostname', 
                        job.get('metrics', {}).get('orchestrator_data', {}).get('execution_host', 'unknown-node'))
        
        return node_mapping.get(full_node, 'unknown-node')
    
    def _get_job_field(self, job: Dict, field: str, default=None):
        """Get field from job data, handling both Excel and JSON formats."""
        if self.data_type == 'excel':
            # Direct field access for Excel data
            return job.get(field, default)
        else:
            # Handle nested JSON structure
            if field in ['total_time_pending', 'total_time_running', 'execution_host']:
                return job.get('metrics', {}).get('orchestrator_data', {}).get(field, default)
            elif field in ['queue_wait_seconds', 'execution_seconds']:
                return job.get('metrics', {}).get('orchestrator_timing', {}).get(field, default)
            else:
                return job.get(field, default)
    
    def _create_node_legend_text(self, simple_to_node: Dict[str, str], max_nodes: int = 6) -> str:
        """Create node mapping legend text"""
        legend_lines = ["NODE MAPPING:"]
        
        count = 0
        for simple_label in sorted(simple_to_node.keys()):
            if count >= max_nodes:
                remaining = len(simple_to_node) - max_nodes
                legend_lines.append(f"... and {remaining} more nodes")
                break
            
            full_node = simple_to_node[simple_label]
            legend_lines.append(f"{simple_label} = {full_node}")
            count += 1
        
        return '\n'.join(legend_lines)
    
    def generate_all_visualizations(self):
        """Generate only relevant visualizations for 3-way comparison."""
        print("Generating focused visualization suite for 3-way comparison...")
        
        try:
            # Calculate metrics first
            print("🧮 Calculating resource-adjusted performance metrics...")
            self.calculate_resource_adjusted_metrics()
            
            print("📋 Generating analytical summary table...")
            self.generate_analytical_summary_table()
            
            # Core visualizations (already support 3-way)
            print("📈 Creating enhanced timeline visualization...")
            self.create_enhanced_timeline_visualization()
            
            print("🔗 Creating job-node-context overlay...")
            self.create_job_node_context_overlay()
            
            print("📊 Creating resource efficiency chart...")
            self.create_resource_efficiency_chart()
            
            # Additional supportive visualizations
            try:
                print("📊 Creating node utilization comparison...")
                self.create_node_utilization_comparison_threeway()
            except Exception as e:
                print(f"Warning: Node utilization comparison failed: {e}")
                # Fallback to original method
                try:
                    self.create_node_utilization_comparison()
                except:
                    pass
            
            try:
                print("⏰ Creating queue wait analysis...")
                self.create_queue_wait_analysis_threeway()
            except Exception as e:
                print(f"Warning: Queue wait analysis failed: {e}")
                # Fallback to original method
                try:
                    self.create_queue_wait_analysis()
                except:
                    pass
            
            print(f"✅ All visualizations saved to: {self.results_dir.absolute()}")
            print("📊 Generated visualizations:")
            print("   • execution_timeline_enhanced.png - Three-way timeline comparison")
            print("   • job_node_context_overlay.png - Job-node resource allocation")
            print("   • resource_efficiency_analysis.png - Speedup and efficiency metrics")
            print("   • analytical_summary.csv - Performance metrics table")
            print("   • node_utilization_comparison.png - Node usage patterns")
            print("   • queue_wait_analysis.png - Queue wait time analysis")
            
        except Exception as e:
            print(f"Error in visualization generation: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    
    def create_executive_dashboard(self):
        """Create a high-level executive summary dashboard."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SAS Viya Autoscaling Executive Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Total Execution Time Comparison
        seq_total = self.comparison.get('execution_time', {}).get('sequential_total', 0) / 60  # Convert to minutes
        async_total = self.comparison.get('execution_time', {}).get('async_total', 0) / 60
        time_saved = self.comparison.get('execution_time', {}).get('time_saved', 0) / 60
        
        ax1.bar(['Sequential', 'Autoscaling'], [seq_total, async_total], 
                color=['#d62728', '#2ca02c'], alpha=0.8)
        ax1.set_ylabel('Total Time (minutes)')
        ax1.set_title('Total Execution Time Comparison', fontweight='bold')
        ax1.text(0.5, max(seq_total, async_total) * 0.8, f'{time_saved:.1f} min\nsaved\n({self.comparison.get("execution_time", {}).get("efficiency_gain", 0):.1f}%)', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 2. Success Rate Comparison
        seq_success = self.comparison.get('job_performance', {}).get('sequential_success_rate', 0)
        async_success = self.comparison.get('job_performance', {}).get('async_success_rate', 0)
        
        ax2.bar(['Sequential', 'Autoscaling'], [seq_success, async_success], 
                color=['#1f77b4', '#ff7f0e'], alpha=0.8)
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Job Success Rate Comparison', fontweight='bold')
        ax2.set_ylim(0, 105)
        for i, v in enumerate([seq_success, async_success]):
            ax2.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Job Count Summary
        seq_successful = len([j for j in self.sequential_jobs if j.get('status') == 'completed'])
        seq_failed = len(self.sequential_jobs) - seq_successful
        async_successful = len([j for j in self.async_jobs if j.get('status') == 'completed'])
        async_failed = len(self.async_jobs) - async_successful
        
        categories = ['Sequential\nSuccessful', 'Sequential\nFailed', 'Autoscaling\nSuccessful', 'Autoscaling\nFailed']
        counts = [seq_successful, seq_failed, async_successful, async_failed]
        colors = ['#2ca02c', '#d62728', '#2ca02c', '#d62728']
        
        bars = ax3.bar(categories, counts, color=colors, alpha=0.8)
        ax3.set_ylabel('Number of Jobs')
        ax3.set_title('Job Execution Summary', fontweight='bold')
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Key Metrics Summary
        ax4.axis('off')
        metrics_text = f"""
Key Performance Metrics:

• Time Efficiency: {self.comparison.get('execution_time', {}).get('efficiency_gain', 0):.1f}% faster
• Time Saved: {time_saved:.1f} minutes total
• Jobs Completed: {async_successful}/{len(self.async_jobs)} async vs {seq_successful}/{len(self.sequential_jobs)} sequential
• Success Rate: {async_success:.1f}% vs {seq_success:.1f}%

Autoscaling Benefits:
• Concurrent job execution
• Dynamic resource allocation
• Reduced queue wait times
• Improved throughput
        """
        ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'executive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Executive dashboard saved")
    
    def create_execution_time_comparison(self):
        """Create detailed execution time comparison charts."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Execution Time Analysis', fontsize=16, fontweight='bold')
        
        # 1. Total execution time breakdown
        seq_total = self.comparison.get('execution_time', {}).get('sequential_total', 0)
        async_total = self.comparison.get('execution_time', {}).get('async_total', 0)
        time_saved = self.comparison.get('execution_time', {}).get('time_saved', 0)
        
        times = [seq_total/60, async_total/60]  # Convert to minutes
        labels = ['Sequential', 'Autoscaling']
        colors = ['#ff9999', '#66b3ff']
        
        bars = ax1.bar(labels, times, color=colors, alpha=0.8)
        ax1.set_ylabel('Total Execution Time (minutes)')
        ax1.set_title('Total Execution Time Comparison')
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{time:.1f} min', ha='center', va='bottom', fontweight='bold')
        
        # Add improvement annotation
        ax1.annotate(f'Time Saved:\n{time_saved/60:.1f} minutes\n({self.comparison.get("execution_time", {}).get("efficiency_gain", 0):.1f}% faster)',
                    xy=(0.5, max(times) * 0.7), xytext=(0.5, max(times) * 0.9),
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 2. Queue Wait vs Execution Time Breakdown
        seq_exec_times = []
        seq_queue_times = []
        async_exec_times = []
        async_queue_times = []
        
        for job in self.sequential_jobs:
            timing = job.get('metrics', {}).get('orchestrator_timing', {})
            if timing:
                seq_exec_times.append(timing.get('execution_seconds', 0))
                seq_queue_times.append(timing.get('queue_wait_seconds', 0))
        
        for job in self.async_jobs:
            if job.get('status') == 'completed':  # Only successful jobs for fair comparison
                timing = job.get('metrics', {}).get('orchestrator_timing', {})
                if timing:
                    async_exec_times.append(timing.get('execution_seconds', 0))
                    async_queue_times.append(timing.get('queue_wait_seconds', 0))
        
        # Create stacked bar chart
        seq_avg_exec = np.mean(seq_exec_times) if seq_exec_times else 0
        seq_avg_queue = np.mean(seq_queue_times) if seq_queue_times else 0
        async_avg_exec = np.mean(async_exec_times) if async_exec_times else 0
        async_avg_queue = np.mean(async_queue_times) if async_queue_times else 0
        
        categories = ['Sequential', 'Autoscaling']
        exec_times = [seq_avg_exec, async_avg_exec]
        queue_times = [seq_avg_queue, async_avg_queue]
        
        bars_exec = ax2.bar(categories, exec_times, color=['#ff9999', '#99ccff'], alpha=0.8, label='Execution Time')
        bars_queue = ax2.bar(categories, queue_times, bottom=exec_times, color=['#ffcccc', '#ccddff'], alpha=0.8, label='Queue Wait Time')
        
        ax2.set_ylabel('Average Time (seconds)')
        ax2.set_title('Time Breakdown: Execution vs Queue Wait')
        ax2.legend()
        
        # Add total time labels
        total_seq = seq_avg_exec + seq_avg_queue
        total_async = async_avg_exec + async_avg_queue
        ax2.text(0, total_seq + 5, f'Total: {total_seq:.1f}s', ha='center', va='bottom', fontweight='bold')
        ax2.text(1, total_async + 5, f'Total: {total_async:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Execution time comparison saved")
    
    def create_job_duration_analysis(self):
        """Create individual job duration comparison."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle('Job Execution Time Analysis (Actual Execution, Not Queue Wait)', fontsize=16, fontweight='bold')
        
        # Extract job names and ACTUAL execution times (not including queue wait)
        seq_names = []
        seq_durations = []
        for i, job in enumerate(self.sequential_jobs):
            timing = job.get('metrics', {}).get('orchestrator_timing', {})
            exec_time = timing.get('execution_seconds', 0) if timing else job.get('duration', 0)
            seq_names.append(job.get('name', f"Job {i+1}"))
            seq_durations.append(exec_time)
        
        async_names = []
        async_durations = []
        for i, job in enumerate(self.async_jobs):
            timing = job.get('metrics', {}).get('orchestrator_timing', {})
            # Use execution_seconds instead of total duration to exclude queue wait time
            exec_time = timing.get('execution_seconds', 0) if timing else 0
            async_names.append(job.get('name', f"Job {i+1}"))
            async_durations.append(exec_time)
        
        # 1. Sequential job durations
        bars1 = ax1.bar(range(len(seq_durations)), seq_durations, 
                       color='#ff7f7f', alpha=0.8, label='Sequential')
        ax1.set_xlabel('Job Index')
        ax1.set_ylabel('Duration (seconds)')
        ax1.set_title('Sequential Execution - Individual Job Durations')
        ax1.set_xticks(range(0, len(seq_durations), max(1, len(seq_durations)//10)))
        
        # Add average line
        seq_avg = np.mean(seq_durations)
        ax1.axhline(y=seq_avg, color='red', linestyle='--', alpha=0.7, 
                   label=f'Average: {seq_avg:.1f}s')
        ax1.legend()
        
        # 2. Async job durations
        bars2 = ax2.bar(range(len(async_durations)), async_durations, 
                       color='#7f7fff', alpha=0.8, label='Autoscaling')
        ax2.set_xlabel('Job Index')
        ax2.set_ylabel('Duration (seconds)')
        ax2.set_title('Autoscaling Execution - Individual Job Durations')
        ax2.set_xticks(range(0, len(async_durations), max(1, len(async_durations)//10)))
        
        # Add average line and color-code by status
        async_avg = np.mean(async_durations)
        ax2.axhline(y=async_avg, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Average: {async_avg:.1f}s')
        
        # Color failed jobs differently
        for i, job in enumerate(self.async_jobs):
            if job.get('status') == 'failed':
                bars2[i].set_color('#ff4444')
                bars2[i].set_alpha(0.9)
        
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'job_duration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Job duration analysis saved")
    
    def create_success_rate_comparison(self):
        """Create success rate and failure analysis visualizations."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Job Success Rate Analysis', fontsize=16, fontweight='bold')
        
        # Calculate success/failure counts
        seq_successful = len([j for j in self.sequential_jobs if j.get('status') == 'completed'])
        seq_failed = len(self.sequential_jobs) - seq_successful
        async_successful = len([j for j in self.async_jobs if j.get('status') == 'completed'])
        async_failed = len(self.async_jobs) - async_successful
        
        # 1. Success rate pie charts
        seq_data = [seq_successful, seq_failed]
        seq_labels = [f'Successful\n({seq_successful})', f'Failed\n({seq_failed})']
        colors1 = ['#2ca02c', '#d62728']
        
        wedges1, texts1, autotexts1 = ax1.pie(seq_data, labels=seq_labels, colors=colors1, 
                                             autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Sequential Execution\nSuccess Rate: {seq_successful/len(self.sequential_jobs)*100:.1f}%')
        
        async_data = [async_successful, async_failed]
        async_labels = [f'Successful\n({async_successful})', f'Failed\n({async_failed})']
        
        wedges2, texts2, autotexts2 = ax2.pie(async_data, labels=async_labels, colors=colors1, 
                                             autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Autoscaling Execution\nSuccess Rate: {async_successful/len(self.async_jobs)*100:.1f}%')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'success_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Success rate comparison saved")
    
    def create_resource_utilization_analysis(self):
        """Create resource utilization comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Resource Utilization Analysis', fontsize=16, fontweight='bold')
        
        # Extract resource metrics from orchestrator data
        seq_cpu_usage = []
        seq_memory_usage = []
        async_cpu_usage = []
        async_memory_usage = []
        
        for job in self.sequential_jobs:
            metrics = job.get('metrics', {}).get('orchestrator_data', {})
            if metrics:
                cpu_time = metrics.get('max_cpu_time')
                memory_used = metrics.get('max_memory_used')
                if cpu_time is not None:
                    seq_cpu_usage.append(cpu_time)
                if memory_used is not None:
                    seq_memory_usage.append(memory_used)
        
        for job in self.async_jobs:
            metrics = job.get('metrics', {}).get('orchestrator_data', {})
            if metrics:
                cpu_time = metrics.get('max_cpu_time')
                memory_used = metrics.get('max_memory_used')
                if cpu_time is not None:
                    async_cpu_usage.append(cpu_time)
                if memory_used is not None:
                    async_memory_usage.append(memory_used)
        
        # 1. CPU Usage Distribution
        if seq_cpu_usage:
            ax1.hist(seq_cpu_usage, bins=20, alpha=0.7, color='red', label='Sequential')
        if async_cpu_usage:
            ax1.hist(async_cpu_usage, bins=20, alpha=0.7, color='blue', label='Autoscaling')
        ax1.set_xlabel('Max CPU Time (seconds)')
        ax1.set_ylabel('Number of Jobs')
        ax1.set_title('CPU Usage Distribution')
        ax1.legend()
        
        # 2. Memory Usage Distribution
        if seq_memory_usage:
            ax2.hist(seq_memory_usage, bins=20, alpha=0.7, color='red', label='Sequential')
        if async_memory_usage:
            ax2.hist(async_memory_usage, bins=20, alpha=0.7, color='blue', label='Autoscaling')
        ax2.set_xlabel('Max Memory Used (GB)')
        ax2.set_ylabel('Number of Jobs')
        ax2.set_title('Memory Usage Distribution')
        ax2.legend()
        
        # 3. CPU vs Memory Scatter Plot for Sequential
        if seq_cpu_usage and seq_memory_usage:
            ax3.scatter(seq_cpu_usage, seq_memory_usage, alpha=0.6, color='red', s=50)
        ax3.set_xlabel('CPU Time (seconds)')
        ax3.set_ylabel('Memory Usage (GB)')
        ax3.set_title('Sequential: CPU vs Memory Usage')
        
        # 4. CPU vs Memory Scatter Plot for Async
        if async_cpu_usage and async_memory_usage:
            # Create colors array matching the data points
            colors = []
            data_point_index = 0
            for job in self.async_jobs:
                metrics = job.get('metrics', {}).get('orchestrator_data', {})
                if metrics:
                    cpu_time = metrics.get('max_cpu_time')
                    memory_used = metrics.get('max_memory_used')
                    if cpu_time is not None and memory_used is not None:
                        colors.append('green' if job.get('status') == 'completed' else 'red')
                        data_point_index += 1
            
            # Ensure we only use colors for points we have data for
            min_len = min(len(async_cpu_usage), len(async_memory_usage), len(colors))
            ax4.scatter(async_cpu_usage[:min_len], async_memory_usage[:min_len], 
                       alpha=0.6, c=colors[:min_len], s=50)
        ax4.set_xlabel('CPU Time (seconds)')
        ax4.set_ylabel('Memory Usage (GB)')
        ax4.set_title('Autoscaling: CPU vs Memory Usage\n(Green=Success, Red=Failed)')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'resource_utilization_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Resource utilization analysis saved")
    
    def create_queue_wait_analysis(self):
        """Create queue wait time analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Queue Wait Time Analysis', fontsize=16, fontweight='bold')
        
        # Extract queue wait times
        seq_queue_times = []
        async_queue_times = []
        
        for job in self.sequential_jobs:
            timing = job.get('metrics', {}).get('orchestrator_timing', {})
            if timing:
                wait_time = timing.get('queue_wait_seconds')
                if wait_time is not None:
                    seq_queue_times.append(wait_time)
        
        for job in self.async_jobs:
            timing = job.get('metrics', {}).get('orchestrator_timing', {})
            if timing:
                wait_time = timing.get('queue_wait_seconds')
                if wait_time is not None:
                    async_queue_times.append(wait_time)
        
        # 1. Queue wait time distribution
        if seq_queue_times:
            ax1.hist(seq_queue_times, bins=15, alpha=0.7, color='red', label=f'Sequential (avg: {np.mean(seq_queue_times):.1f}s)')
        if async_queue_times:
            ax1.hist(async_queue_times, bins=15, alpha=0.7, color='blue', label=f'Autoscaling (avg: {np.mean(async_queue_times):.1f}s)')
        
        ax1.set_xlabel('Queue Wait Time (seconds)')
        ax1.set_ylabel('Number of Jobs')
        ax1.set_title('Queue Wait Time Distribution')
        ax1.legend()
        
        # 2. Box plot comparison
        if seq_queue_times and async_queue_times:
            box_data = [seq_queue_times, async_queue_times]
            box = ax2.boxplot(box_data, labels=['Sequential', 'Autoscaling'], patch_artist=True)
            box['boxes'][0].set_facecolor('#ff9999')
            box['boxes'][1].set_facecolor('#99ccff')
            
            ax2.set_ylabel('Queue Wait Time (seconds)')
            ax2.set_title('Queue Wait Time Distribution (Box Plot)')
            
            # Add statistics
            seq_median = np.median(seq_queue_times)
            async_median = np.median(async_queue_times)
            improvement = ((seq_median - async_median) / seq_median * 100) if seq_median > 0 else 0
            
            ax2.text(0.5, max(max(seq_queue_times), max(async_queue_times)) * 0.8,
                    f'Median Improvement:\n{improvement:.1f}%\n({seq_median:.1f}s → {async_median:.1f}s)',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'queue_wait_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Queue wait analysis saved")
    
    def create_queue_vs_execution_breakdown(self):
        """Create detailed breakdown of queue wait vs actual execution time."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Queue Wait vs Execution Time Breakdown', fontsize=16, fontweight='bold')
        
        # Extract timing data
        seq_exec_times = []
        seq_queue_times = []
        async_exec_times = []
        async_queue_times = []
        
        for job in self.sequential_jobs:
            timing = job.get('metrics', {}).get('orchestrator_timing', {})
            if timing:
                seq_exec_times.append(timing.get('execution_seconds', 0))
                seq_queue_times.append(timing.get('queue_wait_seconds', 0))
        
        for job in self.async_jobs:
            timing = job.get('metrics', {}).get('orchestrator_timing', {})
            if timing:
                async_exec_times.append(timing.get('execution_seconds', 0))
                async_queue_times.append(timing.get('queue_wait_seconds', 0))
        
        # 1. Average time breakdown - stacked bars
        seq_avg_exec = np.mean(seq_exec_times) if seq_exec_times else 0
        seq_avg_queue = np.mean(seq_queue_times) if seq_queue_times else 0
        async_avg_exec = np.mean(async_exec_times) if async_exec_times else 0
        async_avg_queue = np.mean(async_queue_times) if async_queue_times else 0
        
        categories = ['Sequential', 'Autoscaling']
        exec_times = [seq_avg_exec, async_avg_exec]
        queue_times = [seq_avg_queue, async_avg_queue]
        
        ax1.bar(categories, exec_times, color=['#ff6666', '#6666ff'], alpha=0.8, label='Execution Time')
        ax1.bar(categories, queue_times, bottom=exec_times, color=['#ffaaaa', '#aaaaff'], alpha=0.8, label='Queue Wait Time')
        
        ax1.set_ylabel('Average Time (seconds)')
        ax1.set_title('Average Time Breakdown')
        ax1.legend()
        
        # Add annotations
        for i, (exec_t, queue_t) in enumerate(zip(exec_times, queue_times)):
            total = exec_t + queue_t
            ax1.text(i, total + 10, f'Total: {total:.1f}s', ha='center', va='bottom', fontweight='bold')
            ax1.text(i, exec_t/2, f'{exec_t:.1f}s', ha='center', va='center', fontweight='bold', color='white')
            if queue_t > 5:  # Only show queue time if significant
                ax1.text(i, exec_t + queue_t/2, f'{queue_t:.1f}s', ha='center', va='center', fontweight='bold')
        
        # 2. Execution time only comparison (fair comparison)
        ax2.bar(categories, exec_times, color=['#ff6666', '#6666ff'], alpha=0.8)
        ax2.set_ylabel('Execution Time (seconds)')
        ax2.set_title('Actual Execution Time Comparison\n(Excluding Queue Wait)')
        
        for i, exec_t in enumerate(exec_times):
            ax2.text(i, exec_t + 0.5, f'{exec_t:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Calculate execution time improvement
        exec_improvement = ((seq_avg_exec - async_avg_exec) / seq_avg_exec * 100) if seq_avg_exec > 0 else 0
        efficiency_gain = self.comparison.get('execution_time', {}).get('efficiency_gain', 0)
        ax2.text(0.5, max(exec_times) * 0.7, f'Execution Time\nImprovement:\n{exec_improvement:+.1f}%', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen" if exec_improvement > 0 else "lightcoral", alpha=0.7))
        
        # 3. Queue wait time comparison
        ax3.bar(categories, queue_times, color=['#cc3333', '#3333cc'], alpha=0.8)
        ax3.set_ylabel('Queue Wait Time (seconds)')
        ax3.set_title('Queue Wait Time Comparison')
        
        for i, queue_t in enumerate(queue_times):
            ax3.text(i, queue_t + 10, f'{queue_t:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 4. Summary statistics
        ax4.axis('off')
        breakdown_text = f"""
TIMING BREAKDOWN ANALYSIS

Sequential Jobs:
• Average execution time: {seq_avg_exec:.1f} seconds
• Average queue wait: {seq_avg_queue:.1f} seconds
• Total average time: {seq_avg_exec + seq_avg_queue:.1f} seconds

Autoscaling Jobs:
• Average execution time: {async_avg_exec:.1f} seconds
• Average queue wait: {async_avg_queue:.1f} seconds
• Total average time: {async_avg_exec + async_avg_queue:.1f} seconds

Key Insights:
• Execution time change: {exec_improvement:+.1f}%
• Queue wait impact: {async_avg_queue - seq_avg_queue:+.1f} seconds
• Overall efficiency: {efficiency_gain:.1f}% faster total completion
• Trade-off: Longer individual waits but faster overall completion

Explanation:
The autoscaling jobs had longer queue waits because they
all started simultaneously and waited for resources to
scale up. However, the overall completion time was much
faster due to concurrent execution.
        """
        
        ax4.text(0.05, 0.95, breakdown_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'queue_vs_execution_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Queue vs execution breakdown saved")
    
    def create_execution_timeline(self):
        """Create timeline visualization of job execution with synchronized x-axis."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 16))
        fig.suptitle('Job Execution Timeline Comparison (Synchronized Scale)', fontsize=16, fontweight='bold')
        
        # Parse timestamps for sequential jobs - use actual execution times
        seq_exec_start_times = []
        seq_exec_end_times = []
        seq_submit_times = []
        seq_names = []
        
        for i, job in enumerate(self.sequential_jobs):
            # Use orchestrator start_time (actual execution start) if available
            orchestrator_data = job.get('metrics', {}).get('orchestrator_data', {})
            if orchestrator_data:
                submit_str = orchestrator_data.get('submit_time', '')
                start_str = orchestrator_data.get('start_time', '')
                end_str = orchestrator_data.get('end_time', '')
            else:
                # Fallback to job-level timestamps
                submit_str = job.get('start_time', '')  # This is actually submit time
                start_str = job.get('start_time', '')
                end_str = job.get('end_time', '')
            
            if start_str and end_str:
                try:
                    submit_time = datetime.fromisoformat(submit_str.replace('Z', '+00:00'))
                    start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                    seq_submit_times.append(submit_time)
                    seq_exec_start_times.append(start_time)
                    seq_exec_end_times.append(end_time)
                    seq_names.append(job.get('name', f'Job {i+1}'))
                except:
                    continue
        
        # Parse timestamps for async jobs - use actual execution times
        async_exec_start_times = []
        async_exec_end_times = []
        async_submit_times = []
        async_names = []
        async_statuses = []
        
        for i, job in enumerate(self.async_jobs):
            # Use orchestrator start_time (actual execution start)
            orchestrator_data = job.get('metrics', {}).get('orchestrator_data', {})
            if orchestrator_data:
                submit_str = orchestrator_data.get('submit_time', '')
                start_str = orchestrator_data.get('start_time', '')
                end_str = orchestrator_data.get('end_time', '')
            else:
                # Fallback to job-level timestamps
                submit_str = job.get('start_time', '')
                start_str = job.get('start_time', '')
                end_str = job.get('end_time', '')
            
            if start_str and end_str:
                try:
                    submit_time = datetime.fromisoformat(submit_str.replace('Z', '+00:00'))
                    start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                    async_submit_times.append(submit_time)
                    async_exec_start_times.append(start_time)
                    async_exec_end_times.append(end_time)
                    async_names.append(job.get('name', f'Job {i+1}'))
                    async_statuses.append(job.get('status', 'unknown'))
                except:
                    continue
        
        # Calculate separate timelines for each execution type
        seq_timeline_data = []
        async_timeline_data = []
        
        if seq_submit_times and seq_exec_start_times:
            # Sequential timeline starts from first sequential job submission
            seq_start = min(seq_submit_times)
            seq_end = max(seq_exec_end_times)
            seq_total_duration = (seq_end - seq_start).total_seconds()
            
        if async_submit_times and async_exec_start_times:
            # Autoscaling timeline starts from first autoscaling job submission  
            async_start = min(async_submit_times)
            async_end = max(async_exec_end_times)
            async_total_duration = (async_end - async_start).total_seconds()
            
        # Use the longer duration for consistent x-axis scale
        max_duration = max(seq_total_duration if seq_submit_times else 0, 
                          async_total_duration if async_submit_times else 0)
            
        # Plot sequential timeline (from its own start)
        if seq_exec_start_times:
            for i, (submit, start, end, name) in enumerate(zip(seq_submit_times, seq_exec_start_times, seq_exec_end_times, seq_names)):
                submit_offset = (submit - seq_start).total_seconds()
                start_offset = (start - seq_start).total_seconds()
                end_offset = (end - seq_start).total_seconds()
                
                # Queue wait time (if any)
                queue_duration = start_offset - submit_offset
                exec_duration = end_offset - start_offset
                
                if queue_duration > 0:
                    ax1.barh(i, queue_duration, left=submit_offset, height=0.8, 
                            color='lightcoral', alpha=0.5, label='Queue Wait' if i == 0 else "")
                
                ax1.barh(i, exec_duration, left=start_offset, height=0.8, 
                        color='red', alpha=0.8, label='Execution' if i == 0 else "")
        
        ax1.set_xlabel('Time from Sequential Start (seconds)')
        ax1.set_ylabel('Job Index')
        ax1.set_title(f'Sequential Timeline\n(Total: {seq_total_duration/60:.1f} min, Jobs run one after another)')
        ax1.set_xlim(0, max_duration)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot async timeline (from its own start)
        if async_exec_start_times:
            for i, (submit, start, end, name, status) in enumerate(zip(async_submit_times, async_exec_start_times, async_exec_end_times, async_names, async_statuses)):
                submit_offset = (submit - async_start).total_seconds()
                start_offset = (start - async_start).total_seconds()
                end_offset = (end - async_start).total_seconds()
                
                # Queue wait time
                queue_duration = start_offset - submit_offset
                exec_duration = end_offset - start_offset
                
                # Color based on status
                queue_color = 'lightblue' if status == 'completed' else 'lightyellow'
                exec_color = 'green' if status == 'completed' else 'red'
                
                if queue_duration > 0:
                    ax2.barh(i, queue_duration, left=submit_offset, height=0.8, 
                            color=queue_color, alpha=0.6, label='Queue Wait' if i == 0 else "")
                
                if exec_duration > 0:
                    ax2.barh(i, exec_duration, left=start_offset, height=0.8, 
                            color=exec_color, alpha=0.8, 
                            label=f'Execution (Success)' if status == 'completed' and i == 0 else 
                                  f'Execution (Failed)' if status == 'failed' and exec_color == 'red' and 'Execution (Failed)' not in [l.get_label() for l in ax2.get_legend_handles_labels()[0]] else "")
        
        ax2.set_xlabel('Time from Autoscaling Start (seconds)')
        ax2.set_ylabel('Job Index')
        ax2.set_title(f'Autoscaling Timeline\n(Total: {async_total_duration/60:.1f} min, Light=Queue Wait, Dark=Execution)')
        ax2.set_xlim(0, max_duration)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Side-by-side comparison of execution patterns
        if seq_exec_start_times and async_exec_start_times:
            # Sequential execution periods (relative to their start)
            for i, (start, end, name) in enumerate(zip(seq_exec_start_times, seq_exec_end_times, seq_names)):
                start_offset = (start - seq_start).total_seconds()
                duration = (end - start).total_seconds()
                ax3.barh(i - 0.2, duration, left=start_offset, height=0.3, 
                        color='red', alpha=0.8, label='Sequential' if i == 0 else "")
            
            # Autoscaling execution periods (relative to their start)
            for i, (start, end, name, status) in enumerate(zip(async_exec_start_times, async_exec_end_times, async_names, async_statuses)):
                start_offset = (start - async_start).total_seconds()
                duration = (end - start).total_seconds()
                color = 'green' if status == 'completed' else 'orange'
                if duration > 0:  # Only plot if there was actual execution
                    ax3.barh(i + 0.2, duration, left=start_offset, height=0.3, 
                            color=color, alpha=0.8, 
                            label=f'Autoscaling (Success)' if status == 'completed' and color == 'green' and i == 0 else 
                                  f'Autoscaling (Failed)' if status == 'failed' and color == 'orange' and 'Autoscaling (Failed)' not in [l.get_label() for l in ax3.get_legend_handles_labels()[0]] else "")
            
            ax3.set_xlabel('Time from Respective Start (seconds)')
            ax3.set_ylabel('Job Index')
            ax3.set_title('Execution Patterns Comparison\n(Both start at 0, showing execution distribution)')
            ax3.set_xlim(0, max_duration)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add comparison annotations
            if seq_exec_end_times and seq_submit_times:
                seq_total_time = (seq_exec_end_times[-1] - seq_submit_times[0]).total_seconds() / 60
            else:
                seq_total_time = 0
                
            if async_exec_end_times and async_submit_times:
                async_total_time = (async_exec_end_times[-1] - async_submit_times[0]).total_seconds() / 60
            else:
                async_total_time = 0
                
            time_savings = seq_total_time - async_total_time
            
        # Calculate average queue wait times
        seq_avg_queue = np.mean([(start - submit).total_seconds() for submit, start in zip(seq_submit_times, seq_exec_start_times)]) if seq_submit_times else 0
        async_avg_queue = np.mean([(start - submit).total_seconds() for submit, start in zip(async_submit_times, async_exec_start_times)]) if async_submit_times else 0
        
        # Add text box with detailed comparison
        comparison_text = f"""
CORRECTED TIMELINE ANALYSIS:
• Sequential: {seq_total_duration/60:.1f} min total (jobs run sequentially)
• Autoscaling: {async_total_duration/60:.1f} min total (jobs run concurrently)
• Time saved: {(seq_total_duration - async_total_duration)/60:.1f} min ({((seq_total_duration - async_total_duration)/seq_total_duration*100):.1f}% faster)

QUEUE WAIT BREAKDOWN:
• Sequential avg queue: {seq_avg_queue:.1f} sec (minimal wait)
• Autoscaling avg queue: {async_avg_queue:.1f} sec (~{async_avg_queue/60:.1f} min wait for scale-up)
• Autoscaling execution starts at: ~{async_avg_queue:.0f} seconds

EXPLANATION: Autoscaling jobs all submitted at t=0, waited ~{async_avg_queue/60:.1f} min 
for resources to scale up, then executed concurrently on multiple nodes.
        """
        
        fig.text(0.02, 0.02, comparison_text, fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for comparison text
        plt.savefig(self.results_dir / 'execution_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Execution timeline saved")
    
    def create_node_utilization_comparison(self):
        """Create node utilization and autoscaling analysis with simple node names."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Node Utilization and Autoscaling Analysis', fontsize=16, fontweight='bold')
        
        # Create node mapping for all jobs
        node_to_simple, simple_to_node = self._create_node_mapping([self.sequential_jobs, self.async_jobs])
        
        # Extract node information with simple names
        seq_nodes = {}
        async_nodes = {}
        
        for job in self.sequential_jobs:
            simple_node = self._get_simple_node_name(job, node_to_simple)
            if simple_node != 'unknown-node':
                seq_nodes[simple_node] = seq_nodes.get(simple_node, 0) + 1
        
        for job in self.async_jobs:
            simple_node = self._get_simple_node_name(job, node_to_simple)
            if simple_node != 'unknown-node':
                async_nodes[simple_node] = async_nodes.get(simple_node, 0) + 1
        
        # 1. Sequential node usage
        if seq_nodes:
            nodes = list(seq_nodes.keys())
            counts = list(seq_nodes.values())
            ax1.pie(counts, labels=[f'{node}\n({count} jobs)' for node, count in zip(nodes, counts)], 
                   autopct='%1.1f%%', startangle=90)
            ax1.set_title(f'Sequential Node Usage\n({len(seq_nodes)} unique nodes)')
        else:
            ax1.text(0.5, 0.5, 'No node data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Sequential Node Usage')
        
        # 2. Async node usage (already using simple names)
        if async_nodes:
            nodes = list(async_nodes.keys())
            counts = list(async_nodes.values())
            ax2.pie(counts, labels=[f'{node}\n({count} jobs)' for node, count in zip(nodes, counts)], 
                   autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Autoscaling Node Usage\n({len(async_nodes)} unique nodes)')
        else:
            ax2.text(0.5, 0.5, 'No node data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Autoscaling Node Usage')
        
        # 3. Queue type comparison
        seq_queues = {}
        async_queues = {}
        
        for job in self.sequential_jobs:
            queue = job.get('metrics', {}).get('orchestrator_context', {}).get('queue_name', 'unknown')
            seq_queues[queue] = seq_queues.get(queue, 0) + 1
        
        for job in self.async_jobs:
            queue = job.get('metrics', {}).get('orchestrator_context', {}).get('queue_name', 'unknown')
            async_queues[queue] = async_queues.get(queue, 0) + 1
        
        all_queues = set(list(seq_queues.keys()) + list(async_queues.keys()))
        queue_comparison = []
        queue_labels = []
        
        for queue in all_queues:
            seq_count = seq_queues.get(queue, 0)
            async_count = async_queues.get(queue, 0)
            queue_comparison.append([seq_count, async_count])
            queue_labels.append(queue)
        
        if queue_comparison:
            x = np.arange(len(queue_labels))
            width = 0.35
            
            seq_counts = [q[0] for q in queue_comparison]
            async_counts = [q[1] for q in queue_comparison]
            
            ax3.bar(x - width/2, seq_counts, width, label='Sequential', color='red', alpha=0.7)
            ax3.bar(x + width/2, async_counts, width, label='Autoscaling', color='blue', alpha=0.7)
            
            ax3.set_xlabel('Queue Name')
            ax3.set_ylabel('Number of Jobs')
            ax3.set_title('Queue Usage Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels([q[:10] + '...' if len(q) > 10 else q for q in queue_labels], rotation=45)
            ax3.legend()
        
        # 4. Execution host diversity
        ax4.axis('off')
        scaling_info = f"""
Autoscaling Analysis:

Sequential Execution:
• Unique Nodes: {len(seq_nodes)}
• Queue Types: {len(seq_queues)}
• Context: {self.comparison.get('resource_utilization', {}).get('sequential_context', 'N/A')}

Autoscaling Execution:
• Unique Nodes: {len(async_nodes)}
• Queue Types: {len(async_queues)}
• Context: {self.comparison.get('resource_utilization', {}).get('async_context', 'N/A')}

Scaling Behavior:
• Node Scaling: {len(async_nodes) - len(seq_nodes)} additional nodes
• Concurrent Execution: {self.comparison.get('resource_utilization', {}).get('concurrent_execution', False)}
• Failed Jobs: {len([j for j in self.async_jobs if j.get('status') == 'failed'])} ({len([j for j in self.async_jobs if j.get('status') == 'failed'])/len(self.async_jobs)*100:.1f}%)
        """
        
        ax4.text(0.05, 0.95, scaling_info, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        # Add node mapping legend
        if len(simple_to_node) > 0:
            node_legend_text = self._create_node_legend_text(simple_to_node, max_nodes=6)
            fig.text(0.02, 0.02, node_legend_text, fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8),
                    verticalalignment='bottom')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for node mapping legend
        plt.savefig(self.results_dir / 'node_utilization_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Node utilization comparison saved")
    
    def create_job_node_mapping(self):
        """Create detailed job-to-node mapping visualization with simple node names."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Job-to-Node Mapping: Autoscaling Behavior Analysis', fontsize=16, fontweight='bold')
        
        # Create node mapping
        node_to_simple, simple_to_node = self._create_node_mapping([self.async_jobs])
        
        # Extract job-to-node mapping for autoscaling jobs
        job_node_mapping = []
        node_colors = {}
        color_palette = plt.cm.Set3(np.linspace(0, 1, 20))  # Up to 20 different colors
        color_index = 0
        
        for i, job in enumerate(self.async_jobs):
            job_name = job.get('name', f'Job_{i+1}')
            simple_node = self._get_simple_node_name(job, node_to_simple)
            status = job.get('status', 'unknown')
            
            # Extract timing info
            if self.data_type == 'excel':
                queue_wait = job.get('queue_wait_time', 0)
                exec_time = job.get('execution_time', 0)
            else:
                timing = job.get('metrics', {}).get('orchestrator_timing', {})
                queue_wait = timing.get('queue_wait_seconds', 0) if timing else 0
                exec_time = timing.get('execution_seconds', 0) if timing else 0
            
            # Assign color to simple node if not already assigned
            if simple_node not in node_colors and simple_node != 'unknown-node':
                node_colors[simple_node] = color_palette[color_index % len(color_palette)]
                color_index += 1
            
            job_node_mapping.append({
                'job_index': i,
                'job_name': job_name,
                'node': simple_node,
                'status': status,
                'queue_wait': queue_wait,
                'exec_time': exec_time,
                'color': node_colors.get(simple_node, 'gray')
            })
        
        # 1. Job execution by node (timeline style)
        successful_jobs = [j for j in job_node_mapping if j['status'] == 'completed']
        failed_jobs = [j for j in job_node_mapping if j['status'] == 'failed']
        
        # Plot successful jobs
        for job in successful_jobs:
            ax1.scatter(job['job_index'], job['exec_time'], 
                       c=[job['color']], s=100, alpha=0.8, marker='o')
        
        # Plot failed jobs
        for job in failed_jobs:
            ax1.scatter(job['job_index'], 0, c='red', s=100, alpha=0.8, marker='x')
        
        ax1.set_xlabel('Job Index')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Job Execution Times by Node\n(Each color = different node, X = failed)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Node utilization timeline
        nodes = list(node_colors.keys())
        if nodes:
            for i, node in enumerate(nodes):
                node_jobs = [j for j in successful_jobs if j['node'] == node]
                if node_jobs:
                    job_indices = [j['job_index'] for j in node_jobs]
                    exec_times = [j['exec_time'] for j in node_jobs]
                    ax2.scatter(job_indices, [i] * len(job_indices), 
                               s=[t*2 for t in exec_times], c=[node_colors[node]] * len(job_indices),
                               alpha=0.7, label=f'{node[-10:]}' if len(node) > 10 else node)
            
            ax2.set_xlabel('Job Index')
            ax2.set_ylabel('Node Index')
            ax2.set_title('Job Distribution Across Nodes\n(Size = execution time)')
            ax2.set_yticks(range(len(nodes)))
            ax2.set_yticklabels([f'Node {i+1}' for i in range(len(nodes))])
            if len(nodes) <= 8:  # Only show legend if not too many nodes
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Queue wait time by node
        if nodes:
            node_queue_stats = {}
            for node in nodes:
                node_jobs = [j for j in job_node_mapping if j['node'] == node and j['status'] == 'completed']
                if node_jobs:
                    avg_queue = np.mean([j['queue_wait'] for j in node_jobs])
                    job_count = len(node_jobs)
                    node_queue_stats[node] = {'avg_queue': avg_queue, 'job_count': job_count}
            
            if node_queue_stats:
                node_names = list(node_queue_stats.keys())
                avg_queues = [node_queue_stats[node]['avg_queue'] for node in node_names]
                job_counts = [node_queue_stats[node]['job_count'] for node in node_names]
                
                bars = ax3.bar(range(len(node_names)), avg_queues, 
                              color=[node_colors[node] for node in node_names], alpha=0.8)
                ax3.set_xlabel('Node')
                ax3.set_ylabel('Average Queue Wait Time (seconds)')
                ax3.set_title('Average Queue Wait Time by Node')
                # Create simple node labels and mapping
                node_to_simple, simple_to_node = self._create_node_mapping([[job for job in self.async_jobs]])
                simple_node_names = [node_to_simple.get(node, f'node{i+1}') for i, node in enumerate(node_names)]
                
                ax3.set_xticks(range(len(node_names)))
                ax3.set_xticklabels([f'{simple_node_names[i]}\n({job_counts[i]} jobs)' for i in range(len(node_names))])
                
                # Add node mapping legend
                if len(simple_to_node) > 0:
                    node_legend_text = self._create_node_legend_text(simple_to_node, max_nodes=4)
                    fig.text(0.02, 0.02, node_legend_text, fontsize=8, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
                            verticalalignment='bottom')
                
                # Add value labels
                for i, (bar, queue_time) in enumerate(zip(bars, avg_queues)):
                    ax3.text(bar.get_x() + bar.get_width()/2., queue_time + 10,
                            f'{queue_time:.0f}s', ha='center', va='bottom', fontweight='bold')
        
        # 4. Node scaling timeline and summary
        ax4.axis('off')
        
        # Calculate node scaling statistics
        unique_nodes = len([node for node in nodes if node != 'unknown'])
        total_successful = len(successful_jobs)
        total_failed = len(failed_jobs)
        
        # Node utilization summary
        node_summary = "NODE SCALING ANALYSIS:\n\n"
        node_summary += f"Total Nodes Used: {unique_nodes}\n"
        node_summary += f"Successful Jobs: {total_successful}\n"
        node_summary += f"Failed Jobs: {total_failed}\n\n"
        
        node_summary += "NODE DETAILS:\n"
        for i, node in enumerate(nodes[:8]):  # Show first 8 nodes
            node_jobs = [j for j in job_node_mapping if j['node'] == node]
            successful = len([j for j in node_jobs if j['status'] == 'completed'])
            failed = len([j for j in node_jobs if j['status'] == 'failed'])
            avg_exec = np.mean([j['exec_time'] for j in node_jobs if j['status'] == 'completed']) if successful > 0 else 0
            avg_queue = np.mean([j['queue_wait'] for j in node_jobs]) if node_jobs else 0
            
            short_node = node[-20:] if len(node) > 20 else node
            node_summary += f"Node {i+1}: {short_node}\n"
            node_summary += f"  Jobs: {successful} success, {failed} failed\n"
            node_summary += f"  Avg exec: {avg_exec:.1f}s, queue: {avg_queue:.0f}s\n\n"
        
        if len(nodes) > 8:
            node_summary += f"... and {len(nodes) - 8} more nodes\n"
        
        node_summary += "\nKEY INSIGHTS:\n"
        node_summary += "• Autoscaling allocated multiple nodes\n"
        node_summary += "• Jobs distributed across available nodes\n"
        node_summary += "• Queue waits occurred during scale-up\n"
        node_summary += "• Node diversity shows dynamic allocation\n"
        
        ax4.text(0.05, 0.95, node_summary, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        # Add node mapping legend
        if len(simple_to_node) > 0:
            node_legend_text = self._create_node_legend_text(simple_to_node, max_nodes=8)
            fig.text(0.02, 0.02, node_legend_text, fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8),
                    verticalalignment='bottom')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for node mapping legend
        plt.savefig(self.results_dir / 'job_node_mapping.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Job-to-node mapping saved")
    
    def create_performance_heatmap(self):
        """Create performance metrics heatmap."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Performance Metrics Heatmap', fontsize=16, fontweight='bold')
        
        # Create performance matrix for jobs that have orchestrator data
        seq_metrics_matrix = []
        async_metrics_matrix = []
        job_names = []
        
        metrics_labels = ['Duration (s)', 'CPU Time (s)', 'Memory (GB)', 'Queue Wait (s)', 'I/O Total (MB)']
        
        # Sequential metrics
        for job in self.sequential_jobs:
            if job.get('metrics', {}).get('orchestrator_data'):
                metrics = job.get('metrics', {}).get('orchestrator_data', {})
                timing = job.get('metrics', {}).get('orchestrator_timing', {})
                
                row = [
                    job.get('duration', 0),
                    metrics.get('max_cpu_time', 0),
                    metrics.get('max_memory_used', 0),
                    timing.get('queue_wait_seconds', 0),
                    metrics.get('max_io_total', 0)
                ]
                seq_metrics_matrix.append(row)
                job_names.append(job.get('name', f"Job {len(job_names)+1}"))
        
        # Async metrics (only successful jobs for fair comparison)
        async_job_names = []
        for job in self.async_jobs:
            if job.get('status') == 'completed' and job.get('metrics', {}).get('orchestrator_data'):
                metrics = job.get('metrics', {}).get('orchestrator_data', {})
                timing = job.get('metrics', {}).get('orchestrator_timing', {})
                
                row = [
                    job.get('duration', 0),
                    metrics.get('max_cpu_time', 0),
                    metrics.get('max_memory_used', 0),
                    timing.get('queue_wait_seconds', 0),
                    metrics.get('max_io_total', 0)
                ]
                async_metrics_matrix.append(row)
                async_job_names.append(job.get('name', f"Job {len(async_job_names)+1}"))
        
        # Normalize data for heatmap (0-1 scale)
        if seq_metrics_matrix:
            seq_array = np.array(seq_metrics_matrix)
            # Replace None values with 0 and handle division by zero
            seq_array = np.nan_to_num(seq_array, nan=0.0, posinf=0.0, neginf=0.0)
            max_vals = seq_array.max(axis=0)
            max_vals[max_vals == 0] = 1  # Avoid division by zero
            seq_normalized = seq_array / max_vals
            
            im1 = ax1.imshow(seq_normalized.T, cmap='Reds', aspect='auto', interpolation='nearest')
            ax1.set_xlabel('Job Index')
            ax1.set_ylabel('Metrics')
            ax1.set_title('Sequential Jobs Performance Heatmap')
            ax1.set_yticks(range(len(metrics_labels)))
            ax1.set_yticklabels(metrics_labels)
            ax1.set_xticks(range(0, len(job_names), max(1, len(job_names)//10)))
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Normalized Performance (0=min, 1=max)')
        
        if async_metrics_matrix:
            async_array = np.array(async_metrics_matrix)
            # Replace None values with 0 and handle division by zero
            async_array = np.nan_to_num(async_array, nan=0.0, posinf=0.0, neginf=0.0)
            max_vals = async_array.max(axis=0)
            max_vals[max_vals == 0] = 1  # Avoid division by zero
            async_normalized = async_array / max_vals
            
            im2 = ax2.imshow(async_normalized.T, cmap='Blues', aspect='auto', interpolation='nearest')
            ax2.set_xlabel('Job Index')
            ax2.set_ylabel('Metrics')
            ax2.set_title('Autoscaling Jobs Performance Heatmap')
            ax2.set_yticks(range(len(metrics_labels)))
            ax2.set_yticklabels(metrics_labels)
            ax2.set_xticks(range(0, len(async_job_names), max(1, len(async_job_names)//10)))
            
            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Normalized Performance (0=min, 1=max)')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Performance heatmap saved")
    
    def create_cost_benefit_analysis(self):
        """Create cost-benefit analysis visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cost-Benefit Analysis: Sequential vs Autoscaling', fontsize=16, fontweight='bold')
        
        # 1. Time efficiency breakdown
        seq_total = self.comparison.get('execution_time', {}).get('sequential_total', 0)
        async_total = self.comparison.get('execution_time', {}).get('async_total', 0)
        time_saved = self.comparison.get('execution_time', {}).get('time_saved', 0)
        efficiency_gain = self.comparison.get('execution_time', {}).get('efficiency_gain', 0)
        
        categories = ['Time Used\n(Sequential)', 'Time Used\n(Autoscaling)', 'Time Saved']
        values = [seq_total/60, async_total/60, time_saved/60]  # Convert to minutes
        colors = ['#ff9999', '#99ccff', '#99ff99']
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.8)
        ax1.set_ylabel('Time (minutes)')
        ax1.set_title('Time Efficiency Breakdown')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Success rate impact
        seq_success_rate = self.comparison.get('job_performance', {}).get('sequential_success_rate', 0)
        async_success_rate = self.comparison.get('job_performance', {}).get('async_success_rate', 0)
        
        success_data = [seq_success_rate, async_success_rate]
        success_labels = ['Sequential', 'Autoscaling']
        
        bars2 = ax2.bar(success_labels, success_data, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Reliability Comparison')
        ax2.set_ylim(0, 105)
        
        for bar, rate in zip(bars2, success_data):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Resource allocation efficiency
        # Calculate average resource usage
        seq_cpu_values = []
        seq_memory_values = []
        async_cpu_values = []
        async_memory_values = []
        
        for job in self.sequential_jobs:
            metrics = job.get('metrics', {}).get('orchestrator_data', {})
            if metrics:
                cpu_time = metrics.get('max_cpu_time')
                memory_used = metrics.get('max_memory_used')
                if cpu_time is not None:
                    seq_cpu_values.append(cpu_time)
                if memory_used is not None:
                    seq_memory_values.append(memory_used)
        
        for job in self.async_jobs:
            if job.get('status') == 'completed':
                metrics = job.get('metrics', {}).get('orchestrator_data', {})
                if metrics:
                    cpu_time = metrics.get('max_cpu_time')
                    memory_used = metrics.get('max_memory_used')
                    if cpu_time is not None:
                        async_cpu_values.append(cpu_time)
                    if memory_used is not None:
                        async_memory_values.append(memory_used)
        
        seq_avg_cpu = np.mean(seq_cpu_values) if seq_cpu_values else 0
        async_avg_cpu = np.mean(async_cpu_values) if async_cpu_values else 0
        seq_avg_memory = np.mean(seq_memory_values) if seq_memory_values else 0
        async_avg_memory = np.mean(async_memory_values) if async_memory_values else 0
        
        resource_metrics = ['Avg CPU Time', 'Avg Memory Usage']
        seq_resources = [seq_avg_cpu, seq_avg_memory]
        async_resources = [async_avg_cpu, async_avg_memory]
        
        x = np.arange(len(resource_metrics))
        width = 0.35
        
        bars3a = ax3.bar(x - width/2, seq_resources, width, label='Sequential', color='red', alpha=0.7)
        bars3b = ax3.bar(x + width/2, async_resources, width, label='Autoscaling', color='blue', alpha=0.7)
        
        ax3.set_xlabel('Resource Type')
        ax3.set_ylabel('Average Usage')
        ax3.set_title('Average Resource Usage Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(resource_metrics)
        ax3.legend()
        
        # 4. Key metrics summary
        ax4.axis('off')
        
        # Calculate additional metrics
        seq_jobs_completed = len([j for j in self.sequential_jobs if j.get('status') == 'completed'])
        async_jobs_completed = len([j for j in self.async_jobs if j.get('status') == 'completed'])
        
        summary_text = f"""
COST-BENEFIT ANALYSIS SUMMARY

Time Efficiency:
• Total time reduction: {efficiency_gain:.1f}%
• Time saved: {time_saved/60:.1f} minutes
• Sequential total: {seq_total/60:.1f} minutes
• Autoscaling total: {async_total/60:.1f} minutes

Job Throughput:
• Sequential completed: {seq_jobs_completed}/{len(self.sequential_jobs)} jobs
• Autoscaling completed: {async_jobs_completed}/{len(self.async_jobs)} jobs
• Success rate difference: {async_success_rate - seq_success_rate:+.1f}%

Resource Utilization:
• Sequential avg CPU: {seq_avg_cpu:.2f}s
• Autoscaling avg CPU: {async_avg_cpu:.2f}s
• Sequential avg memory: {seq_avg_memory:.4f}GB
• Autoscaling avg memory: {async_avg_memory:.4f}GB
• CPU efficiency change: {((async_avg_cpu/seq_avg_cpu - 1)*100 if seq_avg_cpu > 0 else 0):+.1f}%
• Memory efficiency change: {((async_avg_memory/seq_avg_memory - 1)*100 if seq_avg_memory > 0 else 0):+.1f}%

Business Impact:
• Concurrent execution improves overall throughput
• Queue wait times show autoscaling challenges
• Dynamic scaling optimizes resource costs
• Monitor failed jobs for production readiness
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'cost_benefit_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Cost-benefit analysis saved")
    
    def create_summary_report(self):
        """Create a text summary report of key findings."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f'visualization_summary_{timestamp}.txt'
        
        # Calculate summary statistics
        seq_jobs_completed = len([j for j in self.sequential_jobs if j.get('status') == 'completed'])
        async_jobs_completed = len([j for j in self.async_jobs if j.get('status') == 'completed'])
        
        seq_avg_duration = np.mean([job.get('duration', 0) for job in self.sequential_jobs])
        async_avg_duration = np.mean([job.get('duration', 0) for job in self.async_jobs if job.get('status') == 'completed'])
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SAS VIYA AUTOSCALING COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {self.data_source}\n\n")
            
            f.write("EXECUTION SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Sequential Jobs: {len(self.sequential_jobs)} total, {seq_jobs_completed} completed\n")
            f.write(f"Autoscaling Jobs: {len(self.async_jobs)} total, {async_jobs_completed} completed\n")
            f.write(f"Success Rate: {seq_jobs_completed/len(self.sequential_jobs)*100:.1f}% vs {async_jobs_completed/len(self.async_jobs)*100:.1f}%\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Execution Time: {self.comparison.get('execution_time', {}).get('sequential_total', 0)/60:.1f} min vs {self.comparison.get('execution_time', {}).get('async_autoscaling_total', 0)/60:.1f} min\n")
            f.write(f"Time Saved: {self.comparison.get('execution_time', {}).get('time_saved', 0)/60:.1f} minutes\n")
            f.write(f"Efficiency Gain: {self.comparison.get('execution_time', {}).get('efficiency_gain', 0):.1f}%\n")
            f.write(f"Average Job Duration: {seq_avg_duration:.1f}s vs {async_avg_duration:.1f}s\n\n")
            
            f.write("VISUALIZATIONS GENERATED:\n")
            f.write("-" * 25 + "\n")
            f.write("• executive_dashboard.png - High-level summary for executives\n")
            f.write("• execution_time_comparison.png - Detailed time analysis with queue breakdown\n")
            f.write("• job_duration_analysis.png - Individual job execution times (corrected)\n")
            f.write("• success_rate_comparison.png - Reliability analysis\n")
            f.write("• resource_utilization_analysis.png - CPU/Memory usage patterns\n")
            f.write("• queue_wait_analysis.png - Queue performance comparison\n")
            f.write("• queue_vs_execution_breakdown.png - Detailed timing breakdown\n")
            f.write("• execution_timeline.png - Timeline visualization\n")
            f.write("• node_utilization_comparison.png - Autoscaling behavior\n")
            f.write("• job_node_mapping.png - Which jobs ran on which nodes\n")
            f.write("• performance_heatmap.png - Metrics correlation\n")
            f.write("• cost_benefit_analysis.png - Business value analysis\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 15 + "\n")
            if async_jobs_completed/len(self.async_jobs) >= 0.8:
                f.write("+ Autoscaling shows strong performance benefits\n")
            else:
                f.write("! Autoscaling had reliability issues - investigate failed jobs\n")
            
            if self.comparison.get('execution_time', {}).get('efficiency_gain', 0) > 30:
                f.write("+ Significant time savings achieved with autoscaling\n")
            else:
                f.write("! Time savings were modest - evaluate workload characteristics\n")
            
            f.write("* Consider autoscaling for production workloads\n")
            f.write("* Monitor failed jobs and adjust timeout settings\n")
            f.write("* Optimize job scheduling for better resource utilization\n")
        
        print(f"✓ Summary report saved: {report_file}")
        return report_file
    
    
    
    
    
    
    # Advanced Analytical Methods (merged from analytical_visualizer.py)
    
    def calculate_resource_adjusted_metrics(self) -> Dict[str, Any]:
        """Calculate resource-adjusted performance metrics with your correct methodology."""
        print("\n🧮 Calculating Resource-Adjusted Performance Metrics...")
        
        # Get execution times from data
        if self.data_type == 'excel':
            seq_time = self._get_total_time_from_excel('sequential')
            async_reg_time = self._get_total_time_from_excel('async_regular')
            async_auto_time = self._get_total_time_from_excel('async_autoscaling')
            seq_nodes = self._get_node_count_from_excel('sequential')
            async_reg_nodes = self._get_node_count_from_excel('async_regular')
            async_auto_nodes = self._get_node_count_from_excel('async_autoscaling')
        else:
            # JSON fallback
            seq_data = self.data.get('sequential_execution', {})
            async_reg_data = self.data.get('async_regular_execution', {})
            async_auto_data = self.data.get('async_autoscaling_execution', {})
            
            seq_time = seq_data.get('total_duration', 0)
            async_reg_time = async_reg_data.get('total_duration', 0)
            async_auto_time = async_auto_data.get('total_duration', 0)
            seq_nodes = 1
            async_reg_nodes = 2
            async_auto_nodes = 2
        
        metrics = {
            'execution_times': {
                'sequential': seq_time,
                'async_regular': async_reg_time,
                'async_autoscaling': async_auto_time
            },
            'node_counts': {
                'sequential': seq_nodes,
                'async_regular': async_reg_nodes,
                'async_autoscaling': async_auto_nodes
            },
            'actual_speedups': {},
            'parallel_efficiency': {},
            'time_reduction': {},
            'true_benefits': {}
        }
        
        # Calculate speedups and parallel efficiency (your methodology)
        if seq_time > 0:
            # Actual speedups: Baseline time / Runtime
            if async_reg_time > 0:
                metrics['actual_speedups']['async_regular'] = seq_time / async_reg_time
                metrics['time_reduction']['async_regular'] = ((seq_time - async_reg_time) / seq_time) * 100
                metrics['parallel_efficiency']['async_regular'] = (metrics['actual_speedups']['async_regular'] / async_reg_nodes) * 100
            
            if async_auto_time > 0:
                metrics['actual_speedups']['async_autoscaling'] = seq_time / async_auto_time
                metrics['time_reduction']['async_autoscaling'] = ((seq_time - async_auto_time) / seq_time) * 100
                metrics['parallel_efficiency']['async_autoscaling'] = (metrics['actual_speedups']['async_autoscaling'] / async_auto_nodes) * 100
            
            # True autoscaling benefit
            if async_reg_time > 0 and async_auto_time > 0:
                metrics['true_benefits']['autoscaling_vs_regular'] = {
                    'time_saved': async_reg_time - async_auto_time,
                    'percentage': ((async_reg_time - async_auto_time) / async_reg_time) * 100,
                    'speedup_factor': async_reg_time / async_auto_time
                }
        
        self.analysis_results = {'resource_metrics': metrics}
        return metrics
    
    def _get_total_time_from_excel(self, execution_type: str) -> float:
        """Get total execution time for a mode from Excel data."""
        if self.data_type != 'excel':
            return 0
        
        # Map internal execution type to Excel execution_type values
        excel_type_map = {
            'sequential': 'Sequential',
            'async_regular': 'Async Regular',
            'async_autoscaling': 'Async Autoscaling'
        }
        
        excel_type = excel_type_map.get(execution_type, execution_type)
        mode_data = self.df[self.df['execution_type'] == excel_type]
        
        if mode_data.empty:
            print(f"⚠️ No data found for execution type: {excel_type}")
            return 0
        
        total_time = mode_data['duration'].sum() if 'duration' in mode_data.columns else 0
        print(f"📊 {excel_type}: {len(mode_data)} jobs, {total_time:.1f}s total")
        return total_time
    
    def _get_node_count_from_excel(self, execution_type: str) -> int:
        """Get unique node count for a mode from Excel data."""
        if self.data_type != 'excel':
            return 1
        
        # Map internal execution type to Excel execution_type values
        excel_type_map = {
            'sequential': 'Sequential',
            'async_regular': 'Async Regular',
            'async_autoscaling': 'Async Autoscaling'
        }
        
        excel_type = excel_type_map.get(execution_type, execution_type)
        mode_data = self.df[self.df['execution_type'] == excel_type]
        
        if mode_data.empty:
            return 1
        
        if 'execution_host' in mode_data.columns:
            unique_nodes = mode_data['execution_host'].nunique()
            print(f"📊 {excel_type}: {unique_nodes} unique nodes")
            return max(unique_nodes, 1)
        else:
            return 1 if 'sequential' in execution_type else 2
    
    def generate_analytical_summary_table(self) -> pd.DataFrame:
        """Generate comprehensive analytical summary table."""
        if not hasattr(self, 'analysis_results') or not self.analysis_results.get('resource_metrics'):
            self.calculate_resource_adjusted_metrics()
        
        metrics = self.analysis_results['resource_metrics']
        
        # Create summary data using your correct calculations
        summary_data = []
        modes = ['Sequential', 'Async Regular', 'Async Autoscaling']
        times = [
            metrics['execution_times']['sequential'],
            metrics['execution_times']['async_regular'], 
            metrics['execution_times']['async_autoscaling']
        ]
        nodes = [
            metrics['node_counts']['sequential'],
            metrics['node_counts']['async_regular'],
            metrics['node_counts']['async_autoscaling']
        ]
        
        for i, mode in enumerate(modes):
            if times[i] > 0:  # Only include modes that were executed
                row = {
                    'Execution Mode': mode,
                    'Total Time (sec)': round(times[i], 2),
                    'Total Time (min)': round(times[i] / 60, 2),
                    'Node Count': nodes[i]
                }
                
                # Add speedup and efficiency for async modes
                if i > 0:  # Not sequential
                    mode_key = 'async_regular' if i == 1 else 'async_autoscaling'
                    if mode_key in metrics['actual_speedups']:
                        speedup = metrics['actual_speedups'][mode_key]
                        time_reduction = metrics['time_reduction'][mode_key]
                        parallel_eff = metrics['parallel_efficiency'][mode_key]
                        
                        row['Speedup vs Sequential'] = f"{speedup:.2f}x"
                        row['Time Reduction'] = f"{time_reduction:.1f}%"
                        row['Parallel Efficiency'] = f"{parallel_eff:.1f}%"
                        row['Performance Type'] = "Superlinear" if parallel_eff > 100 else "Linear"
                else:
                    row['Speedup vs Sequential'] = "1.00x (baseline)"
                    row['Time Reduction'] = "0.0% (baseline)"
                    row['Parallel Efficiency'] = "100.0% (baseline)"
                    row['Performance Type'] = "Baseline"
                
                summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_path = self.results_dir / "analytical_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved analytical summary to: {csv_path}")
        
        # Print to console
        print("\n📊 RESOURCE-ADJUSTED PERFORMANCE ANALYSIS")
        print("=" * 80)
        print(df.to_string(index=False))
        
        return df
    
    def create_enhanced_timeline_visualization(self):
        """Create enhanced three-way timeline comparison with queue wait breakdown."""
        # Set matplotlib to use solid rendering
        plt.rcParams['hatch.linewidth'] = 0
        plt.rcParams['patch.force_edgecolor'] = False
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('Three-Way Job Execution Timeline Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Timeline for each execution mode
        modes = [
            ('Sequential (Baseline)', self.sequential_jobs, COLORS['sequential'], axes[0]),
            ('Async Regular (Parallelization)', self.async_regular_jobs, COLORS['async_regular'], axes[1]),
            ('Async Autoscaling (Par. + Auto)', self.async_autoscaling_jobs, COLORS['async_autoscaling'], axes[2])
        ]
        
        max_time = 0
        
        for mode_name, jobs, color, ax in modes:
            if not jobs:
                ax.text(0.5, 0.5, f'No data available for {mode_name}', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, style='italic', color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue
            
            # Calculate timeline data with queue/execution breakdown
            timeline_data = []
            for i, job in enumerate(jobs):
                start_time = 0
                total_duration = job.get('duration', 0)
                status = job.get('status', 'unknown')
                
                # Extract queue wait and execution times using unified method
                queue_wait_time = self._get_job_field(job, 'total_time_pending', 0)
                execution_time = self._get_job_field(job, 'total_time_running', total_duration)
                
                # For sequential, jobs run one after another
                if 'sequential' in mode_name.lower():
                    if i > 0:
                        start_time = sum(j.get('duration', 0) for j in jobs[:i])
                
                timeline_data.append({
                    'job_id': i,
                    'start_time': start_time,
                    'queue_wait_time': queue_wait_time,
                    'execution_time': execution_time,
                    'total_duration': total_duration,
                    'end_time': start_time + total_duration,
                    'status': status,
                    'name': job.get('name', f'Job {i+1}')
                })
            
            # Plot timeline with queue wait and execution time breakdown
            for data in timeline_data:
                y_pos = data['job_id']
                start = data['start_time'] / 60  # Convert to minutes
                queue_wait = data['queue_wait_time'] / 60  # Convert to minutes
                execution = data['execution_time'] / 60  # Convert to minutes
                total_duration = data['total_duration'] / 60  # Convert to minutes
                
                job_color = color if data['status'] == 'completed' else COLORS['failed']
                
                # Draw queue wait time first (blue bar) if it exists
                if queue_wait > 0:
                    ax.barh(y_pos, queue_wait, left=start, height=0.8, 
                           color=COLORS['queue_wait'], alpha=1.0, edgecolor='none',
                           label='Queue Wait' if y_pos == 0 else "")
                    
                    # Draw execution time after queue wait (colored bar)
                    ax.barh(y_pos, execution, left=start + queue_wait, height=0.8, 
                           color=job_color, alpha=1.0, edgecolor='none',
                           label='Execution' if y_pos == 0 else "")
                else:
                    # No queue wait, just execution time
                    ax.barh(y_pos, execution, left=start, height=0.8, 
                           color=job_color, alpha=1.0, edgecolor='none',
                           label='Execution' if y_pos == 0 else "")
                
                # Add job name for longer bars
                if total_duration > 0.5:
                    job_name = data['name'][:8] + '...' if len(data['name']) > 8 else data['name']
                    ax.text(start + total_duration/2, y_pos, job_name, 
                           ha='center', va='center', fontsize=6, color='white', fontweight='bold')
            
            # Update max time for consistent x-axis
            if timeline_data:
                mode_max_time = max(data['end_time'] for data in timeline_data) / 60
                max_time = max(max_time, mode_max_time)
            
            # Customize axis
            ax.set_title(f'{mode_name} (Total: {len(jobs)} jobs)', fontsize=11, fontweight='bold', pad=15)
            ax.set_ylabel('Job Index', fontsize=10)
            ax.set_ylim(-0.5, len(jobs) - 0.5)
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
        
        # Set x-axis for all subplots
        for i, ax in enumerate(axes):
            ax.set_xlim(0, max_time * 1.05)
            if i == len(axes) - 1:  # Only bottom subplot gets x-axis label
                ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
        
        # Add legend to first subplot with data
        for ax in axes:
            if len(ax.get_children()) > 0:
                ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
                break
        
        # Add explanation
        timeline_explanation = ("TIMELINE VISUALIZATION:\n"
                               "• Blue bars = Queue wait time (total_time_pending)\n"
                               "• Colored bars = Execution time (total_time_running)\n"
                               "• Red = Sequential, Orange = Async Regular, Green = Autoscaling\n"
                               "• X-axis shows time in minutes")
        
        fig.text(0.02, 0.02, timeline_explanation, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['background'], alpha=0.8),
                verticalalignment='bottom')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.35, top=0.94)
        
        # Save
        timeline_path = self.results_dir / "execution_timeline_enhanced.png"
        plt.savefig(timeline_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved enhanced timeline to: {timeline_path}")
        plt.close()
        
        return timeline_path
    
    def create_resource_efficiency_chart(self):
        """Create resource efficiency comparison chart."""
        if not hasattr(self, 'analysis_results'):
            self.calculate_resource_adjusted_metrics()
        
        metrics = self.analysis_results['resource_metrics']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Resource Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # Get data for charts
        modes = []
        speedups = []
        efficiencies = []
        colors = []
        
        for mode_key, color in [('async_regular', COLORS['async_regular']), 
                               ('async_autoscaling', COLORS['async_autoscaling'])]:
            if mode_key in metrics['actual_speedups']:
                modes.append(mode_key.replace('_', ' ').title())
                speedups.append(metrics['actual_speedups'][mode_key])
                efficiencies.append(metrics['parallel_efficiency'][mode_key])
                colors.append(color)
        
        if modes:
            # Chart 1: Speedup comparison
            bars1 = ax1.bar(modes, speedups, color=colors, alpha=0.8)
            ax1.set_title('Speedup vs Sequential')
            ax1.set_ylabel('Speedup Factor')
            ax1.grid(True, alpha=0.3)
            
            # Chart 2: Parallel efficiency
            bars2 = ax2.bar(modes, efficiencies, color=colors, alpha=0.8)
            ax2.set_title('Parallel Efficiency (Superlinear >100%)')
            ax2.set_ylabel('Parallel Efficiency (%)')
            ax2.grid(True, alpha=0.3)
            
            # Add efficiency zones
            ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Linear (100%)')
            ax2.axhline(y=150, color='orange', linestyle='--', alpha=0.5, label='Good (>150%)')
            ax2.axhline(y=200, color='green', linestyle='--', alpha=0.5, label='Excellent (>200%)')
            ax2.legend(loc='upper left', fontsize=9)
            
            # Add value labels
            for bars, values, suffix in [(bars1, speedups, 'x'), (bars2, efficiencies, '%')]:
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    label = f'{value:.2f}{suffix}' if suffix == 'x' else f'{value:.1f}{suffix}'
                    ax1.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontweight='bold') if suffix == 'x' else ax2.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        efficiency_path = self.results_dir / "resource_efficiency_analysis.png"
        plt.savefig(efficiency_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved efficiency chart to: {efficiency_path}")
        plt.close()
        
        return efficiency_path
    
    def create_job_node_context_overlay(self):
        """Create job-node-context overlay visualization."""
        # Collect job-node-context data
        job_node_data = []
        execution_modes = [
            ('Sequential', self.sequential_jobs, COLORS['sequential'], 'o'),
            ('Async Regular', self.async_regular_jobs, COLORS['async_regular'], 's'), 
            ('Async Autoscaling', self.async_autoscaling_jobs, COLORS['async_autoscaling'], '^')
        ]
        
        for mode_name, jobs, color, marker in execution_modes:
            for job_idx, job in enumerate(jobs):
                node_name = self._get_job_field(job, 'execution_host', f'node-{job_idx % 3 + 1}')
                
                job_node_data.append({
                    'job_index': job_idx,
                    'job_name': job.get('name', f'Job {job_idx}'),
                    'node_name': node_name,
                    'execution_mode': mode_name,
                    'status': job.get('status', 'completed'),
                    'color': color,
                    'marker': marker,
                    'duration': job.get('duration', 0)
                })
        
        if not job_node_data:
            print("⚠️ No job-node data available")
            return None
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.suptitle('Job-Node-Context Resource Allocation Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Get unique nodes and create mapping
        unique_nodes = sorted(list(set(data['node_name'] for data in job_node_data)))
        simple_labels = [f"node{i+1}" for i in range(len(unique_nodes))]
        node_positions = {node: idx for idx, node in enumerate(unique_nodes)}
        
        # Plot with jitter
        jitter_offsets = {'Sequential': -0.15, 'Async Regular': 0, 'Async Autoscaling': 0.15}
        
        for mode_name, _, color, marker in execution_modes:
            mode_data = [d for d in job_node_data if d['execution_mode'] == mode_name]
            if mode_data:
                jitter = jitter_offsets.get(mode_name, 0)
                x_coords = [d['job_index'] for d in mode_data]
                y_coords = [node_positions[d['node_name']] + jitter for d in mode_data]
                sizes = [max(20, min(200, d['duration'] / 10)) for d in mode_data]
                
                ax.scatter(x_coords, y_coords, c=color, marker=marker, s=sizes, 
                          alpha=0.8, label=f'{mode_name} (Success)', edgecolors='white', linewidth=1)
        
        # Customize plot
        ax.set_xlabel('Job Index (Execution Order)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Compute Node', fontsize=12, fontweight='bold')
        ax.set_yticks(range(len(unique_nodes)))
        ax.set_yticklabels(simple_labels, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Add information panels
        node_mapping = ["NODE MAPPING:"] + [f"{simple} = {full}" for simple, full in zip(simple_labels[:6], unique_nodes[:6])]
        fig.text(0.75, 0.12, '\n'.join(node_mapping), fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
                verticalalignment='bottom')
        
        viz_guide = ("VISUALIZATION GUIDE:\n"
                    "• Markers jittered vertically to prevent overlapping\n"
                    "• Sequential (top), Regular (center), Autoscaling (bottom)\n"
                    "• MARKER SIZE = JOB DURATION (larger = longer execution time)\n"
                    "• Circle = Sequential, Square = Async Regular, Triangle = Autoscaling")
        fig.text(0.4, 0.12, viz_guide, fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.9),
                horizontalalignment='left', verticalalignment='bottom')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.35, right=0.85, top=0.94)
        
        # Save
        overlay_path = self.results_dir / "job_node_context_overlay.png"
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved job-node overlay to: {overlay_path}")
        plt.close()
        
        return overlay_path
    
    def create_queue_wait_analysis_threeway(self):
        """Create three-way queue wait time analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Three-Way Queue Wait Time Analysis', fontsize=16, fontweight='bold')
        
        # Extract queue wait times for all three execution types
        queue_times = {
            'Sequential': [],
            'Async Regular': [],
            'Async Autoscaling': []
        }
        
        for job in self.sequential_jobs:
            queue_time = self._get_job_field(job, 'total_time_pending', 0)
            if queue_time is not None:
                queue_times['Sequential'].append(queue_time)
        
        for job in self.async_regular_jobs:
            queue_time = self._get_job_field(job, 'total_time_pending', 0)
            if queue_time is not None:
                queue_times['Async Regular'].append(queue_time)
        
        for job in self.async_autoscaling_jobs:
            queue_time = self._get_job_field(job, 'total_time_pending', 0)
            if queue_time is not None:
                queue_times['Async Autoscaling'].append(queue_time)
        
        # 1. Average queue wait time comparison
        avg_times = {}
        colors = [COLORS['sequential'], COLORS['async_regular'], COLORS['async_autoscaling']]
        
        for i, (mode, times) in enumerate(queue_times.items()):
            if times:
                avg_times[mode] = np.mean(times)
            else:
                avg_times[mode] = 0
        
        modes = list(avg_times.keys())
        averages = list(avg_times.values())
        
        bars = ax1.bar(modes, averages, color=colors, alpha=0.8)
        ax1.set_ylabel('Average Queue Wait Time (seconds)')
        ax1.set_title('Average Queue Wait Time by Execution Type')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, avg in zip(bars, averages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(averages) * 0.01,
                    f'{avg:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 2. Queue wait time distribution (histogram overlay)
        has_data = False
        for mode, times in queue_times.items():
            if times and len(times) > 0:
                # Check if there's actual variation in the data
                unique_values = len(set(times))
                if unique_values > 1 or max(times) > 0:
                    color = COLORS['sequential'] if mode == 'Sequential' else \
                           COLORS['async_regular'] if mode == 'Async Regular' else \
                           COLORS['async_autoscaling']
                    
                    # Use count instead of density for better visualization when some data is all zeros
                    ax2.hist(times, bins=15, alpha=0.6, label=f'{mode} (avg: {np.mean(times):.1f}s)', 
                            color=color, density=False)
                    has_data = True
                elif max(times) == 0:
                    # For all-zero data (like Sequential), show as a single bar at 0
                    color = COLORS['sequential'] if mode == 'Sequential' else \
                           COLORS['async_regular'] if mode == 'Async Regular' else \
                           COLORS['async_autoscaling']
                    ax2.bar([0], [len(times)], width=10, alpha=0.6, 
                           label=f'{mode} (avg: {np.mean(times):.1f}s)', color=color)
                    has_data = True
        
        if not has_data:
            ax2.text(0.5, 0.5, 'No queue wait data available', 
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=12, style='italic', color='gray')
        
        ax2.set_xlabel('Queue Wait Time (seconds)')
        ax2.set_ylabel('Number of Jobs')
        ax2.set_title('Queue Wait Time Distribution')
        ax2.legend()
        
        # 3. Box plot comparison
        box_data = []
        box_labels = []
        box_colors = []
        
        for mode, times in queue_times.items():
            if times:
                box_data.append(times)
                box_labels.append(mode)
                box_colors.append(COLORS['sequential'] if mode == 'Sequential' else \
                                COLORS['async_regular'] if mode == 'Async Regular' else \
                                COLORS['async_autoscaling'])
        
        if box_data:
            box = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(box['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax3.set_ylabel('Queue Wait Time (seconds)')
            ax3.set_title('Queue Wait Time Distribution (Box Plot)')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Analysis summary
        ax4.axis('off')
        
        analysis_text = "QUEUE WAIT ANALYSIS INSIGHTS:\n\n"
        
        for mode, times in queue_times.items():
            if times:
                avg_time = np.mean(times)
                median_time = np.median(times)
                max_time = np.max(times)
                analysis_text += f"{mode}:\n"
                analysis_text += f"  • Average: {avg_time:.1f}s ({avg_time/60:.1f} min)\n"
                analysis_text += f"  • Median: {median_time:.1f}s\n"
                analysis_text += f"  • Maximum: {max_time:.1f}s\n"
                analysis_text += f"  • Jobs: {len(times)}\n\n"
        
        # Calculate improvements
        if queue_times['Sequential'] and queue_times['Async Autoscaling']:
            seq_avg = np.mean(queue_times['Sequential'])
            auto_avg = np.mean(queue_times['Async Autoscaling'])
            if seq_avg > 0:
                change = ((auto_avg - seq_avg) / seq_avg) * 100
                analysis_text += f"AUTOSCALING IMPACT:\n"
                analysis_text += f"Queue wait change: {change:+.1f}%\n"
                analysis_text += f"Trade-off: Longer waits but faster overall completion\n\n"
        
        analysis_text += "KEY INSIGHTS:\n"
        analysis_text += "• Sequential: Minimal queue waits (jobs run one by one)\n"
        analysis_text += "• Async modes: Higher waits due to resource scaling\n"
        analysis_text += "• Autoscaling: Wait time reflects scale-up period\n"
        analysis_text += "• Overall benefit: Concurrent execution despite waits"
        
        ax4.text(0.05, 0.95, analysis_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'queue_wait_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Three-way queue wait analysis saved")
    
    def create_node_utilization_comparison_threeway(self):
        """Create three-way node utilization comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Three-Way Node Utilization Analysis', fontsize=16, fontweight='bold')
        
        # Extract node information for all three execution types
        node_usage = {
            'Sequential': {},
            'Async Regular': {},
            'Async Autoscaling': {}
        }
        
        # Count jobs per node for each execution type
        for job in self.sequential_jobs:
            node = self._get_job_field(job, 'execution_host', 'unknown-node')
            if node != 'unknown-node':
                node_usage['Sequential'][node] = node_usage['Sequential'].get(node, 0) + 1
        
        for job in self.async_regular_jobs:
            node = self._get_job_field(job, 'execution_host', 'unknown-node')
            if node != 'unknown-node':
                node_usage['Async Regular'][node] = node_usage['Async Regular'].get(node, 0) + 1
        
        for job in self.async_autoscaling_jobs:
            node = self._get_job_field(job, 'execution_host', 'unknown-node')
            if node != 'unknown-node':
                node_usage['Async Autoscaling'][node] = node_usage['Async Autoscaling'].get(node, 0) + 1
        
        # 1. Node count comparison
        node_counts = [len(nodes) for nodes in node_usage.values()]
        modes = list(node_usage.keys())
        colors = [COLORS['sequential'], COLORS['async_regular'], COLORS['async_autoscaling']]
        
        bars = ax1.bar(modes, node_counts, color=colors, alpha=0.8)
        ax1.set_ylabel('Number of Unique Nodes')
        ax1.set_title('Node Count by Execution Type')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars, node_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Jobs per node efficiency
        jobs_per_node = []
        for mode, nodes in node_usage.items():
            if nodes:
                total_jobs = sum(nodes.values())
                avg_jobs_per_node = total_jobs / len(nodes)
                jobs_per_node.append(avg_jobs_per_node)
            else:
                jobs_per_node.append(0)
        
        bars = ax2.bar(modes, jobs_per_node, color=colors, alpha=0.8)
        ax2.set_ylabel('Average Jobs per Node')
        ax2.set_title('Resource Efficiency (Jobs per Node)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, efficiency in zip(bars, jobs_per_node):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(jobs_per_node) * 0.01,
                    f'{efficiency:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Node utilization pattern (stacked bar for top nodes)
        all_nodes = set()
        for nodes in node_usage.values():
            all_nodes.update(nodes.keys())
        
        # Get top 5 most used nodes
        node_totals = {}
        for node in all_nodes:
            total = sum(node_usage[mode].get(node, 0) for mode in modes)
            node_totals[node] = total
        
        top_nodes = sorted(node_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if top_nodes:
            node_names = [node[0][-10:] for node in top_nodes]  # Show last 10 chars
            seq_counts = [node_usage['Sequential'].get(node[0], 0) for node in top_nodes]
            reg_counts = [node_usage['Async Regular'].get(node[0], 0) for node in top_nodes]
            auto_counts = [node_usage['Async Autoscaling'].get(node[0], 0) for node in top_nodes]
            
            x = np.arange(len(node_names))
            width = 0.25
            
            ax3.bar(x - width, seq_counts, width, label='Sequential', color=COLORS['sequential'], alpha=0.8)
            ax3.bar(x, reg_counts, width, label='Async Regular', color=COLORS['async_regular'], alpha=0.8)
            ax3.bar(x + width, auto_counts, width, label='Async Autoscaling', color=COLORS['async_autoscaling'], alpha=0.8)
            
            ax3.set_xlabel('Node (last 10 chars)')
            ax3.set_ylabel('Number of Jobs')
            ax3.set_title('Job Distribution Across Top Nodes')
            ax3.set_xticks(x)
            ax3.set_xticklabels(node_names, rotation=45)
            ax3.legend()
        
        # 4. Analysis summary
        ax4.axis('off')
        
        analysis_text = "NODE UTILIZATION ANALYSIS:\n\n"
        
        for mode, nodes in node_usage.items():
            total_jobs = sum(nodes.values()) if nodes else 0
            unique_nodes = len(nodes)
            avg_per_node = total_jobs / unique_nodes if unique_nodes > 0 else 0
            
            analysis_text += f"{mode}:\n"
            analysis_text += f"  • Unique nodes: {unique_nodes}\n"
            analysis_text += f"  • Total jobs: {total_jobs}\n"
            analysis_text += f"  • Avg jobs/node: {avg_per_node:.1f}\n\n"
        
        # Calculate scaling efficiency
        if node_usage['Sequential'] and node_usage['Async Autoscaling']:
            seq_nodes = len(node_usage['Sequential'])
            auto_nodes = len(node_usage['Async Autoscaling'])
            scaling_factor = auto_nodes / seq_nodes if seq_nodes > 0 else 0
            
            analysis_text += f"SCALING ANALYSIS:\n"
            analysis_text += f"Node scaling factor: {scaling_factor:.1f}x\n"
            analysis_text += f"Resource expansion: {auto_nodes - seq_nodes} additional nodes\n\n"
        
        analysis_text += "KEY INSIGHTS:\n"
        analysis_text += "• Sequential: Limited to single/few nodes\n"
        analysis_text += "• Async modes: Distributed across multiple nodes\n"
        analysis_text += "• Autoscaling: Dynamic node allocation\n"
        analysis_text += "• Trade-off: More resources for faster completion"
        
        ax4.text(0.05, 0.95, analysis_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'node_utilization_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Three-way node utilization comparison saved")


def find_latest_comparison_file() -> str:
    """Find the most recent comparison file (Excel preferred, then JSON)."""
    results_dir = Path("results")
    if not results_dir.exists():
        raise FileNotFoundError("Results directory not found")
    
    # Look for Excel files first (preferred)
    excel_files = list(results_dir.glob("orchestrator_analysis_*.xlsx"))
    if excel_files:
        latest_excel = max(excel_files, key=lambda f: f.stat().st_mtime)
        print(f"📊 Auto-detected Excel file: {latest_excel.name}")
        return str(latest_excel)
    
    # Fallback to JSON files
    json_files = list(results_dir.glob("pooled_jobs_comparison_*.json"))
    if not json_files:
        raise FileNotFoundError("No comparison files found in results directory")
    
    latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
    print(f"📊 Auto-detected JSON file: {latest_json.name} (Excel preferred for accurate queue times)")
    return str(latest_json)


def main():
    """Main execution function."""
    print("🧮 Advanced SAS Viya Job Execution Comparison Visualizer")
    print("=" * 60)
    
    # Determine input file
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        if not os.path.exists(data_file):
            print(f"Error: File {data_file} not found")
            sys.exit(1)
    else:
        try:
            data_file = find_latest_comparison_file()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please provide a file path as argument or ensure comparison files exist in results/")
            sys.exit(1)
    
    file_type = "Excel" if data_file.endswith('.xlsx') else "JSON"
    print(f"📊 Processing {file_type} file: {data_file}")
    
    try:
        # Create visualizer and generate all graphs
        visualizer = JobComparisonVisualizer(data_file)
        
        print("\nGenerating visualizations...")
        visualizer.generate_all_visualizations()
        
        print("\nGenerating summary report...")
        report_file = visualizer.create_summary_report()
        
        print(f"\n✅ Visualization suite completed!")
        print(f"📁 Output directory: {visualizer.results_dir.absolute()}")
        print(f"📄 Summary report: {report_file}")
        print("\nGenerated files:")
        for file in visualizer.results_dir.glob("*.png"):
            print(f"  • {file.name}")
        for file in visualizer.results_dir.glob("*.csv"):
            print(f"  • {file.name}")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
