#!/usr/bin/env python3
"""
Extract and Analyze - JES (Job Execution System) Performance Analyzer

This script extracts comprehensive orchestrator data from pooled jobs comparison JSON files,
performs advanced performance calculations, and creates structured Excel output with analytical insights.

Features:
- Extracts data from all three execution types (Sequential, Async Regular, Async Autoscaling)
- Performs advanced performance calculations (speedup, efficiency, time savings)
- Creates multiple Excel sheets for different analysis views
- Handles all orchestrator metrics including queue wait times, resource usage, and performance data
- Provides summary statistics, performance comparisons, and analytical insights
- Calculates per-node efficiency, perfect split estimates, and autoscaling gains

Usage:
    python extract_and_analyze.py [json_file_path]
    
    Examples:
    # Use latest file automatically
    python extract_and_analyze.py
    
    # Use specific file
    python extract_and_analyze.py results/pooled_jobs_comparison_20250915_171747.json
    
    If no file path is provided, it will look for the latest pooled_jobs_comparison_*.json file
    in the results/ directory.

Output:
    Excel file with multiple sheets:
    - All_Executions: Complete dataset with all job executions
    - Job_Summary: Aggregated statistics by job
    - Performance_Metrics: Focus on orchestrator performance data
    - Timing_Analysis: Detailed timing information
    - Execution_Type_Summary: Comparison across execution types
    - Performance_Summary: Speedup, efficiency, and time savings analysis
    - Performance_Comparison: Autoscaling vs regular async comparison
"""

import json
import pandas as pd
from datetime import datetime
import os
import sys
import glob
import re
from typing import Dict, List, Any, Optional

def extract_timestamp_from_filename(file_path: str) -> Optional[str]:
    """
    Extract timestamp from JSON filename.
    
    Expected format: pooled_jobs_comparison_YYYYMMDD_HHMMSS.json
    Returns: YYYYMMDD_HHMMSS or None if not found
    """
    filename = os.path.basename(file_path)
    
    # Pattern to match: pooled_jobs_comparison_20250915_171747.json
    pattern = r'pooled_jobs_comparison_(\d{8}_\d{6})\.json'
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1)
    
    # Fallback: try to extract any timestamp-like pattern YYYYMMDD_HHMMSS
    pattern = r'(\d{8}_\d{6})'
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1)
    
    return None

def find_latest_json_file() -> Optional[str]:
    """Find the latest pooled_jobs_comparison JSON file in results directory."""
    pattern = "results/pooled_jobs_comparison_*.json"
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No pooled_jobs_comparison_*.json files found in results/ directory")
        return None
    
    # Sort by modification time, most recent first
    latest_file = max(files, key=os.path.getmtime)
    print(f"📁 Using latest file: {latest_file}")
    return latest_file

def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load and parse the JSON file."""
    print(f"📂 Loading data from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ JSON data loaded successfully")
    return data

def extract_orchestrator_metrics(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract orchestrator metrics from a job execution."""
    orchestrator_data = job_data.get('metrics', {}).get('orchestrator_data', {})
    
    if not orchestrator_data:
        return {}
    
    # Extract key orchestrator metrics
    metrics = {
        'workload_job_id': orchestrator_data.get('workload_job_id'),
        'workload_job_name': orchestrator_data.get('workload_job_name'),
        'queue_name': orchestrator_data.get('queue_name'),
        'context_type': orchestrator_data.get('context_type'),
        'submit_time': orchestrator_data.get('submit_time'),
        'start_time': orchestrator_data.get('start_time'),
        'end_time': orchestrator_data.get('end_time'),
        'queue_wait_seconds': orchestrator_data.get('queue_wait_seconds'),
        'execution_seconds': orchestrator_data.get('execution_seconds'),
        'total_seconds': orchestrator_data.get('total_seconds'),
        
        # Critical queue wait data (as documented in README)
        'total_time_running': orchestrator_data.get('total_time_running'),
        'total_time_pending': orchestrator_data.get('total_time_pending'),  # Real queue wait time
        'total_time_starting': orchestrator_data.get('total_time_starting'),
        'total_time_susp_admin': orchestrator_data.get('total_time_susp_admin'),
        'total_time_susp_thresh': orchestrator_data.get('total_time_susp_thresh'),
        'total_time_susp_preempt': orchestrator_data.get('total_time_susp_preempt'),
        
        # Resource allocation
        'cpu_cores': orchestrator_data.get('cpu_cores'),
        'memory_mb': orchestrator_data.get('memory_mb'),
        
        # Performance metrics
        'max_memory_used': orchestrator_data.get('max_memory_used'),
        'max_cpu_time': orchestrator_data.get('max_cpu_time'),
        'max_io_total': orchestrator_data.get('max_io_total'),
        
        # Infrastructure
        'execution_host': orchestrator_data.get('execution_host'),
        'execution_ip': orchestrator_data.get('execution_ip'),
    }
    
    # Extract consumed resources
    consumed_resources = orchestrator_data.get('consumed_resources', [])
    for resource in consumed_resources:
        resource_name = resource.get('name')
        resource_value = resource.get('value')
        if resource_name:
            metrics[f'consumed_{resource_name}'] = resource_value
    
    # Extract limit values
    limit_values = orchestrator_data.get('limit_values', [])
    for limit in limit_values:
        limit_name = limit.get('name')
        limit_value = limit.get('value')
        if limit_name:
            metrics[f'limit_{limit_name}'] = limit_value
    
    return metrics

def extract_job_metrics(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract general job metrics."""
    job_metrics = job_data.get('job_metrics', {})
    
    metrics = {
        'job_id': job_metrics.get('job_id'),
        'job_name': job_metrics.get('job_name'),
        'state': job_metrics.get('state'),
        'submission_time': job_metrics.get('submission_time'),
        'completion_time': job_metrics.get('completion_time'),
        'elapsed_time_ms': job_metrics.get('elapsed_time_ms'),
        'elapsed_time_seconds': job_metrics.get('elapsed_time_seconds'),
        'submitted_by': job_metrics.get('submitted_by'),
        'created_by': job_metrics.get('created_by'),
        'job_type': job_metrics.get('job_type'),
        'context_name': job_metrics.get('context_name'),
        'results_count': job_metrics.get('results_count'),
        'has_log': job_metrics.get('has_log'),
        'compute_job_id': job_metrics.get('compute_job_id'),
        'compute_session': job_metrics.get('compute_session'),
        'compute_context': job_metrics.get('compute_context'),
    }
    
    return metrics

def process_job_executions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process all job executions from three execution types and extract relevant data."""
    all_executions = []
    
    # Get job definitions for reference
    job_definitions = {job['definition_id']: job for job in data.get('jobs', [])}
    
    # Store comparison data for later use
    global comparison_data
    comparison_data = data.get('comparison', {})
    
    # Process executions from all three execution types
    execution_types = [
        ('sequential_execution', 'Sequential'),
        ('async_regular_execution', 'Async Regular'),
        ('async_autoscaling_execution', 'Async Autoscaling')
    ]
    
    executions = []
    for exec_type, exec_label in execution_types:
        exec_section = data.get(exec_type, {})
        jobs_in_section = exec_section.get('jobs', [])
        
        print(f"📊 Found {len(jobs_in_section)} jobs in {exec_label} execution")
        
        # Add execution type label to each job
        for job in jobs_in_section:
            job['execution_type'] = exec_label
            job['execution_section'] = exec_type
            executions.append(job)
    
    print(f"📈 Processing {len(executions)} total job executions...")
    
    for execution in executions:
        # Basic execution info
        base_info = {
            'execution_name': execution.get('name'),
            'execution_type': execution.get('execution_type'),
            'execution_section': execution.get('execution_section'),
            'definition_id': execution.get('definition_id'),
            'sequence_number': execution.get('sequence_number'),
            'start_time': execution.get('start_time'),
            'end_time': execution.get('end_time'),
            'status': execution.get('status'),
            'duration': execution.get('duration'),
            'unique_job_name': execution.get('unique_job_name'),
            'job_instance_id': execution.get('job_instance_id'),
        }
        
        # Add job definition info
        job_def = job_definitions.get(execution.get('definition_id'), {})
        base_info.update({
            'job_definition_name': job_def.get('name'),
            'job_description': job_def.get('description'),
            'context_name_def': job_def.get('context_name'),
            'enabled': job_def.get('enabled'),
        })
        
        # Extract job metrics
        job_metrics = extract_job_metrics(execution)
        base_info.update(job_metrics)
        
        # Extract orchestrator metrics
        orchestrator_metrics = extract_orchestrator_metrics(execution)
        base_info.update(orchestrator_metrics)
        
        all_executions.append(base_info)
    
    return all_executions

def calculate_performance_metrics(exec_summary: pd.DataFrame) -> tuple:
    """
    Calculate advanced performance metrics based on execution type summary.
    
    Returns tuple of (performance_summary_df, performance_comparison_df)
    """
    
    # Use the comparison data from JSON which has the correct wall-clock execution times
    global comparison_data
    if not comparison_data or 'execution_times' not in comparison_data:
        print("⚠️  Warning: No comparison execution times found in JSON data")
        return pd.DataFrame(), pd.DataFrame()
    
    # Get the correct wall-clock execution times from JSON comparison section
    execution_times = comparison_data['execution_times']
    baseline_total_seconds = execution_times.get('sequential_total', 0)
    async_regular_total_seconds = execution_times.get('async_regular_total', 0)
    async_autoscaling_total_seconds = execution_times.get('async_autoscaling_total', 0)
    
    baseline_runtime_min = baseline_total_seconds / 60.0  # Convert to minutes
    baseline_nodes = 1  # Sequential always uses 1 node
    
    # Performance Summary Data (like job_performance_summary_ascii_headers.csv)
    performance_summary = []
    
    # Create performance summary using the correct execution times from JSON
    execution_type_times = {
        'Sequential': baseline_total_seconds,
        'Async Regular': async_regular_total_seconds,
        'Async Autoscaling': async_autoscaling_total_seconds
    }
    
    for exec_type, total_runtime_sec in execution_type_times.items():
        if total_runtime_sec <= 0:
            continue
            
        runtime_min = total_runtime_sec / 60.0
        
        # Determine node count based on execution type
        if exec_type == 'Sequential':
            nodes = 1
            mode_name = 'Baseline_sequential'
        elif exec_type == 'Async Regular':
            nodes = 2  # Based on your data showing 2 nodes
            mode_name = 'Two_nodes_async_no_autoscaling'
        elif exec_type == 'Async Autoscaling':
            nodes = 2  # Based on your data showing 2 nodes  
            mode_name = 'Two_nodes_async_autoscaling'
        else:
            continue
        
        # Calculate metrics using your formulas
        perfect_split_est_min = baseline_runtime_min / nodes if nodes > 1 else None
        beats_perfect_split_min = (baseline_runtime_min / nodes) - runtime_min if nodes > 1 else None
        beats_perfect_split_percent = (((baseline_runtime_min / nodes) - runtime_min) / (baseline_runtime_min / nodes) * 100) if nodes > 1 else None
        speedup_vs_baseline = baseline_runtime_min / runtime_min
        time_saved_vs_baseline_min = baseline_runtime_min - runtime_min
        time_saved_vs_baseline_percent = (baseline_runtime_min - runtime_min) / baseline_runtime_min * 100
        per_node_efficiency_percent = (baseline_runtime_min / runtime_min) / nodes * 100
        
        performance_summary.append({
            'Mode': mode_name,
            'Nodes': nodes,
            'Runtime_min': round(runtime_min, 1),
            'Perfect_split_est_min': round(perfect_split_est_min, 1) if perfect_split_est_min else None,
            'Beats_perfect_split_min': round(beats_perfect_split_min, 1) if beats_perfect_split_min else None,
            'Beats_perfect_split_percent': round(beats_perfect_split_percent, 1) if beats_perfect_split_percent else None,
            'Speedup_vs_baseline_x': round(speedup_vs_baseline, 2),
            'Time_saved_vs_baseline_min': round(time_saved_vs_baseline_min, 1),
            'Time_saved_vs_baseline_percent': round(time_saved_vs_baseline_percent, 1),
            'Per_node_efficiency_percent': round(per_node_efficiency_percent, 1)
        })
    
    performance_summary_df = pd.DataFrame(performance_summary)
    
    # Performance Comparison Data (like job_performance_comparison_ascii_headers.csv)
    async_regular = exec_summary[exec_summary['execution_type'] == 'Async Regular']
    async_autoscaling = exec_summary[exec_summary['execution_type'] == 'Async Autoscaling']
    
    performance_comparison = []
    
    if async_regular_total_seconds > 0 and async_autoscaling_total_seconds > 0:
        # Use the correct execution times from JSON comparison section
        t_noauto = async_regular_total_seconds / 60.0  # Async Regular in minutes
        t_auto = async_autoscaling_total_seconds / 60.0  # Async Autoscaling in minutes
        nodes = 2  # Both use 2 nodes
        
        # Calculate comparison metrics using your formulas
        minutes_faster = t_noauto - t_auto
        percent_faster = (t_noauto - t_auto) / t_noauto * 100
        times_faster = t_noauto / t_auto
        efficiency_autoscaling = (baseline_runtime_min / t_auto) / nodes * 100
        efficiency_no_autoscaling = (baseline_runtime_min / t_noauto) / nodes * 100
        efficiency_gain = efficiency_autoscaling - efficiency_no_autoscaling
        
        performance_comparison.append({
            'Comparison': 'Autoscaling_vs_No_autoscaling_both_two_nodes',
            'Minutes_faster': round(minutes_faster, 1),
            'Percent_faster_percent': round(percent_faster, 1),
            'Times_faster_x': round(times_faster, 2),
            'Efficiency_autoscaling_percent': round(efficiency_autoscaling, 1),
            'Efficiency_no_autoscaling_percent': round(efficiency_no_autoscaling, 1),
            'Efficiency_gain_pp': round(efficiency_gain, 1)
        })
    
    performance_comparison_df = pd.DataFrame(performance_comparison)
    
    return performance_summary_df, performance_comparison_df

def generate_takeaway_markdown(exec_summary: pd.DataFrame, performance_summary: pd.DataFrame, 
                              performance_comparison: pd.DataFrame, output_dir: str = "results") -> str:
    """
    Generate a takeaway.md file with performance analysis insights.
    
    Args:
        exec_summary: DataFrame with execution type summary
        performance_summary: DataFrame with performance metrics
        performance_comparison: DataFrame with autoscaling vs regular comparison
        output_dir: Directory to save the takeaway file
    
    Returns:
        Path to the generated takeaway.md file
    """
    
    # Get baseline data
    sequential_data = exec_summary[exec_summary['execution_type'] == 'Sequential']
    async_regular_data = exec_summary[exec_summary['execution_type'] == 'Async Regular']
    async_autoscaling_data = exec_summary[exec_summary['execution_type'] == 'Async Autoscaling']
    
    if sequential_data.empty:
        print("⚠️ Warning: No Sequential data found for takeaway generation")
        return None
    
    # Extract key metrics - use the correct execution times from JSON comparison
    global comparison_data
    if not comparison_data or 'execution_times' not in comparison_data:
        print("⚠️ Warning: No comparison data found for takeaway generation")
        return None
    
    execution_times = comparison_data['execution_times']
    baseline_min = execution_times.get('sequential_total', 0) / 60.0
    async_regular_min = execution_times.get('async_regular_total', 0) / 60.0
    async_autoscaling_min = execution_times.get('async_autoscaling_total', 0) / 60.0
    baseline_nodes = 1
    
    # Get performance data from performance_summary
    seq_perf = performance_summary[performance_summary['Mode'] == 'Baseline_sequential']
    reg_perf = performance_summary[performance_summary['Mode'] == 'Two_nodes_async_no_autoscaling']
    auto_perf = performance_summary[performance_summary['Mode'] == 'Two_nodes_async_autoscaling']
    
    # Create takeaway content
    takeaway_content = "# Performance Analysis Takeaways\n\n"
    
    # Section (a): Autoscaling Analysis
    if not auto_perf.empty:
        auto_runtime = auto_perf['Runtime_min'].iloc[0]
        auto_perfect_split = auto_perf['Perfect_split_est_min'].iloc[0] if not pd.isna(auto_perf['Perfect_split_est_min'].iloc[0]) else baseline_min / 2
        auto_beats_perfect = auto_perf['Beats_perfect_split_min'].iloc[0] if not pd.isna(auto_perf['Beats_perfect_split_min'].iloc[0]) else auto_perfect_split - auto_runtime
        auto_beats_percent = auto_perf['Beats_perfect_split_percent'].iloc[0] if not pd.isna(auto_perf['Beats_perfect_split_percent'].iloc[0]) else (auto_beats_perfect / auto_perfect_split * 100)
        auto_speedup = auto_perf['Speedup_vs_baseline_x'].iloc[0]
        auto_time_saved = auto_perf['Time_saved_vs_baseline_min'].iloc[0]
        auto_percent_reduction = auto_perf['Time_saved_vs_baseline_percent'].iloc[0]
        
        takeaway_content += "## (a) 2 Nodes with Autoscaling\n\n"
        takeaway_content += "**Baseline Performance:**\n"
        takeaway_content += f"- 1 node: {baseline_min:.1f} min\n"
        takeaway_content += f"- \"Perfect split\" estimate for 2 nodes: {baseline_min:.1f} ÷ 2 = {auto_perfect_split:.1f} min\n"
        takeaway_content += f"- **Actual time: {auto_runtime:.1f} min**\n\n"
        
        takeaway_content += "### Analysis\n\n"
        takeaway_content += "**Performance vs Perfect Split:**\n"
        takeaway_content += f"- Beat the \"perfect split\" guess by: {auto_perfect_split:.1f} − {auto_runtime:.1f} = **{auto_beats_perfect:.1f} min faster**\n"
        takeaway_content += f"- How much better: {auto_beats_perfect:.1f} ÷ {auto_perfect_split:.1f} = **{auto_beats_percent:.1f}% better** than the simple half-time estimate\n\n"
        
        takeaway_content += "**Performance vs Single Node:**\n"
        takeaway_content += f"- Times faster than 1 node: {baseline_min:.1f} ÷ {auto_runtime:.1f} = **{auto_speedup:.2f}×**\n"
        takeaway_content += f"- Total time saved: {baseline_min:.1f} − {auto_runtime:.1f} = {auto_time_saved:.1f} min\n"
        takeaway_content += f"- Percentage reduction: {auto_time_saved:.1f} ÷ {baseline_min:.1f} = **{auto_percent_reduction:.1f}% less time**\n\n"
        
        takeaway_content += "### 📊 Key Takeaway\n"
        takeaway_content += f"> Instead of the expected ~{auto_perfect_split:.1f} min on 2 nodes, you finished in {auto_runtime:.1f} min—that's ~{auto_speedup/2:.1f}× faster than even a perfect 2-way split and ~{auto_speedup:.1f}× faster than 1 node (saving {auto_percent_reduction:.0f}% of the time).\n\n"
    
    # Section (b): Regular Async Analysis
    if not reg_perf.empty:
        reg_runtime = reg_perf['Runtime_min'].iloc[0]
        reg_perfect_split = reg_perf['Perfect_split_est_min'].iloc[0] if not pd.isna(reg_perf['Perfect_split_est_min'].iloc[0]) else baseline_min / 2
        reg_beats_perfect = reg_perf['Beats_perfect_split_min'].iloc[0] if not pd.isna(reg_perf['Beats_perfect_split_min'].iloc[0]) else reg_perfect_split - reg_runtime
        reg_beats_percent = reg_perf['Beats_perfect_split_percent'].iloc[0] if not pd.isna(reg_perf['Beats_perfect_split_percent'].iloc[0]) else (reg_beats_perfect / reg_perfect_split * 100)
        reg_speedup = reg_perf['Speedup_vs_baseline_x'].iloc[0]
        reg_time_saved = reg_perf['Time_saved_vs_baseline_min'].iloc[0]
        reg_percent_reduction = reg_perf['Time_saved_vs_baseline_percent'].iloc[0]
        
        takeaway_content += "## (b) 2 Nodes without Autoscaling\n\n"
        takeaway_content += "**Baseline Performance:**\n"
        takeaway_content += f"- 1 node: {baseline_min:.1f} min\n"
        takeaway_content += f"- \"Perfect split\" estimate for 2 nodes: {baseline_min:.1f} ÷ 2 = {reg_perfect_split:.1f} min\n"
        takeaway_content += f"- **Actual time: {reg_runtime:.1f} min**\n\n"
        
        takeaway_content += "### Analysis\n\n"
        takeaway_content += "**Performance vs Perfect Split:**\n"
        takeaway_content += f"- Beat the \"perfect split\" guess by: {reg_perfect_split:.1f} − {reg_runtime:.1f} = **{reg_beats_perfect:.1f} min faster**\n"
        takeaway_content += f"- How much better: {reg_beats_perfect:.1f} ÷ {reg_perfect_split:.1f} = **{reg_beats_percent:.1f}%** better than the simple half-time estimate\n\n"
        
        takeaway_content += "**Performance vs Single Node:**\n"
        takeaway_content += f"- Times faster than 1 node: {baseline_min:.1f} ÷ {reg_runtime:.1f} = **{reg_speedup:.2f}×**\n"
        takeaway_content += f"- Total time saved: {baseline_min:.1f} − {reg_runtime:.1f} = {reg_time_saved:.1f} min\n"
        takeaway_content += f"- Percentage reduction: {reg_time_saved:.1f} ÷ {baseline_min:.1f} = **{reg_percent_reduction:.1f}% less time**\n\n"
        
        takeaway_content += "### 📊 Key Takeaway\n"
        takeaway_content += f"> Instead of the expected ~{reg_perfect_split:.1f} min on 2 nodes, you finished in {reg_runtime:.1f} min—that's ~{reg_beats_percent:.0f}% better than a perfect 2-way split and ~{reg_speedup:.1f}× faster than 1 node (saving {reg_percent_reduction:.0f}% of the time).\n\n"
    
    # Section (c): Autoscaling vs Regular Comparison
    if not auto_perf.empty and not reg_perf.empty and not performance_comparison.empty:
        auto_runtime = auto_perf['Runtime_min'].iloc[0]
        reg_runtime = reg_perf['Runtime_min'].iloc[0]
        
        comp_data = performance_comparison.iloc[0]
        minutes_faster = comp_data['Minutes_faster']
        percent_faster = comp_data['Percent_faster_percent']
        times_faster = comp_data['Times_faster_x']
        
        auto_beats_percent = auto_perf['Beats_perfect_split_percent'].iloc[0] if not pd.isna(auto_perf['Beats_perfect_split_percent'].iloc[0]) else 0
        reg_beats_percent = reg_perf['Beats_perfect_split_percent'].iloc[0] if not pd.isna(reg_perf['Beats_perfect_split_percent'].iloc[0]) else 0
        
        takeaway_content += "## (c) Autoscaling vs. No Autoscaling Comparison\n\n"
        takeaway_content += "**Both configurations use 2 nodes:**\n"
        takeaway_content += f"- No autoscaling: {reg_runtime:.1f} min\n"
        takeaway_content += f"- **Autoscaling: {auto_runtime:.1f} min**\n\n"
        
        takeaway_content += "### Analysis\n\n"
        takeaway_content += "**Direct Comparison:**\n"
        takeaway_content += f"- Minutes faster: {reg_runtime:.1f} − {auto_runtime:.1f} = **{minutes_faster:.1f} min**\n"
        takeaway_content += f"- Percent faster: {minutes_faster:.1f} ÷ {reg_runtime:.1f} = **{percent_faster:.1f}%**\n"
        takeaway_content += f"- Times faster: {reg_runtime:.1f} ÷ {auto_runtime:.1f} = **{times_faster:.2f}×**\n\n"
        
        perfect_split = baseline_min / 2
        takeaway_content += f"**Context vs Perfect Split ({perfect_split:.1f} min):**\n"
        takeaway_content += f"- Autoscaling beats it by **{auto_beats_percent:.1f}%**\n"
        takeaway_content += f"- No autoscaling beats it by **{reg_beats_percent:.1f}%**\n"
        takeaway_content += f"- Difference: ~**{auto_beats_percent - reg_beats_percent:.1f} percentage points**\n\n"
        
        takeaway_content += "### 📊 Key Takeaway\n"
        takeaway_content += f"> With the same 2 nodes, autoscaling finishes ~{times_faster:.2f}× faster than no-autoscaling (~{percent_faster:.0f}% less time), saving {minutes_faster:.1f} minutes on this workload.\n\n"
    
    # Summary table
    takeaway_content += "---\n\n## Summary\n\n"
    takeaway_content += "| Configuration | Time (min) | vs 1 Node | vs Perfect Split |\n"
    takeaway_content += "|---------------|------------|-----------|------------------|\n"
    takeaway_content += f"| 1 Node (baseline) | {baseline_min:.1f} | - | - |\n"
    
    if not reg_perf.empty:
        reg_runtime = reg_perf['Runtime_min'].iloc[0]
        reg_speedup = reg_perf['Speedup_vs_baseline_x'].iloc[0]
        reg_beats_percent = reg_perf['Beats_perfect_split_percent'].iloc[0] if not pd.isna(reg_perf['Beats_perfect_split_percent'].iloc[0]) else 0
        takeaway_content += f"| 2 Nodes (no autoscaling) | {reg_runtime:.1f} | {reg_speedup:.2f}× faster | {reg_beats_percent:.1f}% better |\n"
    
    if not auto_perf.empty:
        auto_runtime = auto_perf['Runtime_min'].iloc[0]
        auto_speedup = auto_perf['Speedup_vs_baseline_x'].iloc[0]
        auto_beats_percent = auto_perf['Beats_perfect_split_percent'].iloc[0] if not pd.isna(auto_perf['Beats_perfect_split_percent'].iloc[0]) else 0
        takeaway_content += f"| 2 Nodes (with autoscaling) | {auto_runtime:.1f} | {auto_speedup:.2f}× faster | {auto_beats_percent:.1f}% better |\n"
    
    # Save to file
    import os
    os.makedirs(output_dir, exist_ok=True)
    takeaway_path = os.path.join(output_dir, "takeaway.md")
    
    with open(takeaway_path, 'w', encoding='utf-8') as f:
        f.write(takeaway_content)
    
    print(f"📝 Generated takeaway analysis: {takeaway_path}")
    return takeaway_path

def create_execution_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics by execution type."""
    summary_data = []
    
    if 'execution_type' not in df.columns:
        return pd.DataFrame()
    
    for exec_type in df['execution_type'].unique():
        type_data = df[df['execution_type'] == exec_type]
        
        summary = {
            'execution_type': exec_type,
            'total_jobs': len(type_data),
            'avg_duration': type_data['duration'].mean() if 'duration' in type_data.columns else None,
            'avg_execution_seconds': type_data['execution_seconds'].mean() if 'execution_seconds' in type_data.columns else None,
            'avg_queue_wait_pending': type_data['total_time_pending'].mean() if 'total_time_pending' in type_data.columns else None,
            'avg_total_seconds': type_data['total_seconds'].mean() if 'total_seconds' in type_data.columns else None,
            'avg_cpu_cores': type_data['cpu_cores'].mean() if 'cpu_cores' in type_data.columns else None,
            'avg_memory_mb': type_data['memory_mb'].mean() if 'memory_mb' in type_data.columns else None,
            'avg_max_memory_used': type_data['max_memory_used'].mean() if 'max_memory_used' in type_data.columns else None,
            'avg_max_cpu_time': type_data['max_cpu_time'].mean() if 'max_cpu_time' in type_data.columns else None,
            'success_rate': (type_data['status'] == 'completed').sum() / len(type_data) * 100 if 'status' in type_data.columns else None,
            'unique_hosts': type_data['execution_host'].nunique() if 'execution_host' in type_data.columns else None,
        }
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)

def create_excel_output(executions: List[Dict[str, Any]], output_path: str):
    """Create Excel file with multiple sheets for different views of the data."""
    
    # Create DataFrame
    df = pd.DataFrame(executions)
    
    # Sort by execution type and job name for better organization
    sort_columns = []
    if 'execution_type' in df.columns:
        sort_columns.append('execution_type')
    if 'job_definition_name' in df.columns:
        sort_columns.append('job_definition_name')
    elif 'execution_name' in df.columns:
        sort_columns.append('execution_name')
    if 'sequence_number' in df.columns:
        sort_columns.append('sequence_number')
    
    if sort_columns:
        df = df.sort_values(sort_columns)
    
    print(f"📊 Creating Excel file with {len(df)} executions...")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main data sheet
        df.to_excel(writer, sheet_name='All_Executions', index=False)
        print("✅ Created 'All_Executions' sheet")
        
        # Execution type summary sheet
        exec_summary = create_execution_type_summary(df)
        if not exec_summary.empty:
            exec_summary.to_excel(writer, sheet_name='Execution_Type_Summary', index=False)
            print("✅ Created 'Execution_Type_Summary' sheet")
            
            # Performance analysis sheets
            performance_summary, performance_comparison = calculate_performance_metrics(exec_summary)
            
            if not performance_summary.empty:
                performance_summary.to_excel(writer, sheet_name='Performance_Summary', index=False)
                print("✅ Created 'Performance_Summary' sheet (speedup, efficiency, time savings)")
            
            if not performance_comparison.empty:
                performance_comparison.to_excel(writer, sheet_name='Performance_Comparison', index=False)
                print("✅ Created 'Performance_Comparison' sheet (autoscaling vs regular async)")
            
            # Generate takeaway markdown file
            try:
                takeaway_path = generate_takeaway_markdown(exec_summary, performance_summary, performance_comparison, "results")
                if takeaway_path:
                    print("✅ Generated takeaway.md with performance insights")
            except Exception as e:
                print(f"⚠️ Warning: Could not generate takeaway.md: {e}")
        
        # Job summary sheet - aggregate statistics by job
        summary_data = []
        job_name_column = None
        
        # Find the correct job name column
        for col in ['job_definition_name', 'execution_name', 'unique_job_name']:
            if col in df.columns:
                job_name_column = col
                break
        
        if job_name_column and not df.empty:
            for job_name in df[job_name_column].unique():
                job_data = df[df[job_name_column] == job_name]
                
                summary = {
                    'job_name': job_name,
                    'total_executions': len(job_data),
                    'execution_types': ', '.join(job_data['execution_type'].unique()) if 'execution_type' in job_data.columns else None,
                    'avg_execution_seconds': job_data['execution_seconds'].mean() if 'execution_seconds' in job_data.columns else None,
                    'avg_queue_wait_pending': job_data['total_time_pending'].mean() if 'total_time_pending' in job_data.columns else None,
                    'avg_total_seconds': job_data['total_seconds'].mean() if 'total_seconds' in job_data.columns else None,
                    'avg_cpu_cores': job_data['cpu_cores'].mean() if 'cpu_cores' in job_data.columns else None,
                    'avg_memory_mb': job_data['memory_mb'].mean() if 'memory_mb' in job_data.columns else None,
                    'avg_max_memory_used': job_data['max_memory_used'].mean() if 'max_memory_used' in job_data.columns else None,
                    'avg_max_cpu_time': job_data['max_cpu_time'].mean() if 'max_cpu_time' in job_data.columns else None,
                    'success_rate': (job_data['status'] == 'completed').sum() / len(job_data) * 100 if 'status' in job_data.columns else None,
                }
                summary_data.append(summary)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Job_Summary', index=False)
            print("✅ Created 'Job_Summary' sheet")
        
        # Performance metrics sheet - focus on orchestrator data
        performance_cols = [
            'execution_name', 'execution_type', 'job_definition_name', 'sequence_number', 'workload_job_id', 
            'execution_seconds', 'total_time_pending', 'total_time_running', 'queue_wait_seconds', 'total_seconds',
            'total_time_starting', 'cpu_cores', 'memory_mb', 'max_memory_used', 'max_cpu_time', 
            'max_io_total', 'execution_host', 'context_type', 'queue_name'
        ]
        
        # Filter to only include columns that exist in the data
        available_cols = [col for col in performance_cols if col in df.columns]
        if available_cols:
            performance_df = df[available_cols]
            performance_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
            print("✅ Created 'Performance_Metrics' sheet")
        
        # Timing analysis sheet
        timing_cols = [
            'execution_name', 'execution_type', 'job_definition_name', 'sequence_number', 
            'submit_time', 'start_time', 'end_time', 'duration',
            'execution_seconds', 'total_time_pending', 'queue_wait_seconds', 'total_seconds',
            'total_time_running', 'total_time_starting'
        ]
        available_timing_cols = [col for col in timing_cols if col in df.columns]
        if available_timing_cols:
            timing_df = df[available_timing_cols]
            timing_df.to_excel(writer, sheet_name='Timing_Analysis', index=False)
            print("✅ Created 'Timing_Analysis' sheet")
    
    print(f"\n🎉 Excel file created: {output_path}")
    print(f"📊 Total executions processed: {len(executions)}")
    
    # Count unique jobs using available column
    unique_jobs = 0
    for col in ['job_definition_name', 'execution_name', 'unique_job_name']:
        if col in df.columns:
            unique_jobs = df[col].nunique()
            break
    print(f"📋 Unique jobs: {unique_jobs}")
    
    # Show execution type breakdown
    if 'execution_type' in df.columns:
        print(f"🔄 Execution type breakdown:")
        for exec_type, count in df['execution_type'].value_counts().items():
            print(f"   • {exec_type}: {count} executions")

def main():
    """Main execution function."""
    
    # Determine input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            print(f"❌ Error: File '{input_file}' not found")
            sys.exit(1)
    else:
        input_file = find_latest_json_file()
        if not input_file:
            print("❌ Please provide a JSON file path or ensure pooled_jobs_comparison_*.json files exist in results/")
            print("Usage: python extract_and_analyze.py [json_file_path]")
            sys.exit(1)
    
    # Generate output filename using input file timestamp
    input_timestamp = extract_timestamp_from_filename(input_file)
    
    if input_timestamp:
        output_file = f"results/orchestrator_analysis_{input_timestamp}.xlsx"
        print(f"📅 Using timestamp from input file: {input_timestamp}")
    else:
        # Fallback to current timestamp if we can't extract from filename
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/orchestrator_analysis_{current_timestamp}.xlsx"
        print(f"⚠️  Could not extract timestamp from filename, using current time: {current_timestamp}")
    
    try:
        print("🚀 Starting orchestrator data extraction...")
        print("=" * 60)
        
        # Load and process data
        data = load_json_data(input_file)
        
        print("\n📈 Processing job executions...")
        executions = process_job_executions(data)
        
        if not executions:
            print("⚠️  Warning: No job executions found in the JSON file")
            print("   Make sure the file contains sequential_execution, async_regular_execution, or async_autoscaling_execution sections")
            sys.exit(1)
        
        print(f"\n📊 Creating Excel output...")
        create_excel_output(executions, output_file)
        
        print("\n" + "=" * 60)
        print(f"✅ Successfully extracted orchestrator data!")
        print(f"📁 Input file: {input_file}")
        print(f"📊 Output file: {output_file}")
        print(f"📈 Total executions: {len(executions)}")
        
        # Show timestamp correlation
        input_ts = extract_timestamp_from_filename(input_file)
        output_ts = extract_timestamp_from_filename(output_file)
        if input_ts and output_ts and input_ts == output_ts:
            print(f"🔗 Timestamp correlation: Both files use {input_ts} for easy matching")
        
        # Print summary of jobs found
        job_names = set(exec.get('job_definition_name') or exec.get('execution_name', 'Unknown') for exec in executions)
        job_names.discard('Unknown')
        if job_names:
            print(f"📋 Jobs processed: {len(job_names)} unique jobs")
            print(f"   Sample jobs: {', '.join(sorted(list(job_names))[:5])}{'...' if len(job_names) > 5 else ''}")
        
        print("\n🎯 Excel file contains multiple sheets:")
        print("   • All_Executions: Complete dataset with all job executions")
        print("   • Execution_Type_Summary: Comparison across execution types")
        print("   • Performance_Summary: Speedup, efficiency, and time savings analysis")
        print("   • Performance_Comparison: Autoscaling vs regular async comparison")
        print("   • Job_Summary: Aggregated statistics by job")
        print("   • Performance_Metrics: Focus on orchestrator performance data")
        print("   • Timing_Analysis: Detailed timing information")
        print("\n📝 Additional output:")
        print("   • takeaway.md: Executive summary with key performance insights")
        
        return output_file
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
