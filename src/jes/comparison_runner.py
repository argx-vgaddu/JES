#!/usr/bin/env python3
"""
Streamlined Pooled Jobs Comparison Runner

This script runs the same set of pooled jobs in two different modes:
1. Sequential execution in Compute Context (no autoscaling)
2. Asynchronous execution in Autoscaling Context (with autoscaling)

Captures precise timing metrics and comprehensive session/job data for comparison.
"""

import json
import time
import asyncio
import argparse
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import os

from .job_execution import SASJobExecutionClient, load_access_token


class StreamlinedComparisonRunner:
    """Streamlined runner for comparing sequential vs async job execution"""
    
    def __init__(self, base_url: str, access_token: str):
        """Initialize the comparison runner"""
        self.client = SASJobExecutionClient(base_url, access_token)
        self.base_url = base_url
        self.results = {
            'metadata': {
                'execution_timestamp': datetime.now(timezone.utc).isoformat(),
                'base_url': base_url
            },
            'jobs': [],
            'sequential_execution': {
                'context_name': 'SAS Job Execution compute context',
                'mode': 'sequential',
                'jobs': [],
                'summary': {}
            },
            'async_execution': {
                'context_name': 'Autoscaling POC Context',
                'mode': 'asynchronous',
                'jobs': [],
                'summary': {}
            },
            'comparison': {}
        }
    
    def load_pooled_jobs(self, config_file: str = 'config/job_configs.json') -> List[Dict[str, Any]]:
        """Load pooled jobs from configuration file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            enabled_jobs = [job for job in config['jobs'] if job.get('enabled', True)]
            print(f"📋 Loaded {len(enabled_jobs)} enabled pooled jobs from {config_file}")
            
            for job in enabled_jobs:
                print(f"  • {job['name']} ({job['definition_id']})")
            
            self.results['jobs'] = enabled_jobs
            return enabled_jobs
            
        except FileNotFoundError:
            print(f"❌ Configuration file {config_file} not found")
            raise
        except Exception as e:
            print(f"❌ Error loading job configuration: {e}")
            raise
    
    def run_sequential_jobs(self, jobs: List[Dict[str, Any]], 
                          compute_context: str = "SAS Job Execution compute context") -> Dict[str, Any]:
        """
        Run jobs sequentially in compute context (no autoscaling)
        
        Args:
            jobs: List of job configurations
            compute_context: Name of compute context to use
            
        Returns:
            Execution results with timing and metrics
        """
        print(f"\n🔄 Starting Sequential Execution in '{compute_context}'")
        print("=" * 60)
        
        execution_start = time.time()
        execution_results = {
            'start_time': datetime.now(timezone.utc).isoformat(),
            'context_name': compute_context,
            'jobs': [],
            'total_duration': 0,
            'successful_jobs': 0,
            'failed_jobs': 0
        }
        
        for i, job_config in enumerate(jobs, 1):
            job_name = job_config['name']
            job_id = job_config['definition_id']
            
            print(f"\n📝 Job {i}/{len(jobs)}: {job_name}")
            print("-" * 40)
            
            job_start = time.time()
            job_result = {
                'name': job_name,
                'definition_id': job_id,
                'sequence_number': i,
                'start_time': datetime.now(timezone.utc).isoformat(),
                'status': 'pending',
                'duration': 0,
                'session_metrics': {},
                'job_metrics': {}
            }
            
            try:
                # Submit job (sequential mode - specify context in arguments to avoid routing issues)
                print(f"🚀 Submitting job (sequential mode)...")
                job_definition_uri = f"/jobDefinitions/definitions/{job_id}"
                seq_arguments = job_config.get('arguments', {}).copy()
                seq_arguments["_contextName"] = "SAS Job Execution compute context"
                
                # Create unique job name for deterministic orchestrator correlation
                unique_job_name = f"{job_name}_sequential_{i}_{int(time.time())}"
                
                job_submission = self.client.submit_job(
                    job_definition_uri=job_definition_uri,
                    arguments=seq_arguments,
                    name=unique_job_name
                )
                
                job_instance_id = job_submission.get('id')
                session_id = job_submission.get('sessionId')
                
                job_result['job_instance_id'] = job_instance_id
                job_result['session_id'] = session_id
                job_result['unique_job_name'] = unique_job_name
                
                print(f"✅ Job submitted - ID: {job_instance_id}")
                print(f"📊 Session ID: {session_id}")
                print(f"📊 Unique job name: {unique_job_name}")
                
                # No need to capture COMPUTE_JOB ID - we use unique timestamps in names
                
                # Wait for completion and collect resource metrics
                print("⏳ Waiting for job completion...")
                final_status = self.wait_for_completion_with_metrics(
                    job_instance_id, 
                    timeout=3600  # 60 minutes in seconds
                )
                
                job_end = time.time()
                job_duration = job_end - job_start
                
                job_result['end_time'] = datetime.now(timezone.utc).isoformat()
                job_result['duration'] = job_duration
                job_result['status'] = final_status.get('state', 'unknown')
                
                # Collect detailed metrics
                try:
                    job_details = self.client.get_job_details(job_instance_id)
                    
                    # No need for COMPUTE_JOB ID - we use unique timestamps in names
                    
                    job_result['job_metrics'] = self.extract_job_metrics(job_details)
                    
                    # Note: Session details collection removed as method doesn't exist
                    job_result['session_metrics'] = {'note': 'Session details not available'}
                        
                except Exception as metrics_error:
                    print(f"⚠️ Could not collect detailed metrics: {metrics_error}")
                    job_result['metrics_error'] = str(metrics_error)
                
                if final_status.get('state') == 'completed':
                    execution_results['successful_jobs'] += 1
                    print(f"✅ Job completed in {job_duration:.2f} seconds")
                else:
                    execution_results['failed_jobs'] += 1
                    print(f"❌ Job failed with state: {final_status.get('state')}")
                    
            except Exception as e:
                job_end = time.time()
                job_duration = job_end - job_start
                
                job_result['end_time'] = datetime.now(timezone.utc).isoformat()
                job_result['duration'] = job_duration
                job_result['status'] = 'error'
                job_result['error'] = str(e)
                
                execution_results['failed_jobs'] += 1
                print(f"❌ Job failed: {e}")
            
            execution_results['jobs'].append(job_result)
        
        execution_end = time.time()
        execution_results['end_time'] = datetime.now(timezone.utc).isoformat()
        execution_results['total_duration'] = execution_end - execution_start
        
        print(f"\n📊 Sequential Execution Summary:")
        print(f"   Total time: {execution_results['total_duration']:.2f} seconds")
        print(f"   Successful jobs: {execution_results['successful_jobs']}")
        print(f"   Failed jobs: {execution_results['failed_jobs']}")
        
        return execution_results
    
    async def run_async_jobs(self, jobs: List[Dict[str, Any]], 
                           autoscale_context: str = "Autoscaling POC Context",
                           max_concurrent: int = 5,
                           execution_mode: str = "batch") -> Dict[str, Any]:
        """
        Run jobs asynchronously in autoscaling context
        
        Args:
            jobs: List of job configurations
            autoscale_context: Name of autoscaling context to use
            max_concurrent: Maximum number of concurrent jobs (used in batch mode)
            execution_mode: Either "batch" (process in batches) or "all" (submit all at once)
            
        Returns:
            Execution results with timing and metrics
        """
        print(f"\n🚀 Starting Asynchronous Execution in '{autoscale_context}'")
        print(f"📊 Execution Mode: {execution_mode.upper()}")
        if execution_mode == "batch":
            print(f"📦 Batch Size: {max_concurrent}")
        else:
            print(f"🚀 Submitting all {len(jobs)} jobs simultaneously")
        print("=" * 60)
        
        execution_start = time.time()
        execution_results = {
            'start_time': datetime.now(timezone.utc).isoformat(),
            'context_name': autoscale_context,
            'execution_mode': execution_mode,
            'max_concurrent': max_concurrent if execution_mode == "batch" else len(jobs),
            'jobs': [],
            'total_duration': 0,
            'successful_jobs': 0,
            'failed_jobs': 0
        }
        
        all_job_results = []
        
        if execution_mode == "all":
            # Submit ALL jobs at once
            print(f"🚀 Submitting all {len(jobs)} jobs simultaneously...")
            
            # Create tasks for all jobs
            job_tasks = []
            for i, job_config in enumerate(jobs):
                print(f"📝 Creating async task for job {i+1}/{len(jobs)}: {job_config['name']}")
                task = self.submit_async_job(job_config, i+1, len(jobs), autoscale_context)
                job_tasks.append(task)
            
            # Wait for ALL jobs to complete
            print(f"⏳ Waiting for all {len(job_tasks)} jobs to complete...")
            try:
                all_job_results = await asyncio.gather(*job_tasks, return_exceptions=True)
            except Exception as e:
                print(f"❌ All-at-once execution failed: {e}")
                # Add error results for all jobs
                for _ in jobs:
                    all_job_results.append(Exception(f"All-at-once processing error: {e}"))
        
        else:
            # Process jobs in batches (original behavior)
            print(f"🚀 Submitting {len(jobs)} jobs in batches of {max_concurrent}...")
            
            # Process jobs in batches
            for batch_start in range(0, len(jobs), max_concurrent):
                batch_end = min(batch_start + max_concurrent, len(jobs))
                batch_jobs = jobs[batch_start:batch_end]
                
                print(f"\n📦 Batch {batch_start//max_concurrent + 1}: Jobs {batch_start+1}-{batch_end}")
                
                # Create tasks for this batch
                job_tasks = []
                for i, job_config in enumerate(batch_jobs):
                    global_index = batch_start + i + 1
                    print(f"📝 Creating async task for job {global_index}/{len(jobs)}: {job_config['name']}")
                    task = self.submit_async_job(job_config, global_index, len(jobs), autoscale_context)
                    job_tasks.append(task)
                
                # Wait for this batch to complete
                print(f"⏳ Waiting for batch of {len(job_tasks)} jobs to complete...")
                try:
                    batch_results = await asyncio.gather(*job_tasks, return_exceptions=True)
                    all_job_results.extend(batch_results)
                    
                    # Small delay between batches to avoid overwhelming the API
                    if batch_end < len(jobs):
                        print(f"⏸️  Brief pause before next batch...")
                        await asyncio.sleep(5)
                        
                except Exception as e:
                    print(f"❌ Batch failed: {e}")
                    # Add error results for failed batch
                    for _ in batch_jobs:
                        all_job_results.append(Exception(f"Batch processing error: {e}"))
        
        job_results = all_job_results
        
        execution_end = time.time()
        execution_results['end_time'] = datetime.now(timezone.utc).isoformat()
        execution_results['total_duration'] = execution_end - execution_start
        
        # Process results
        for result in job_results:
            if isinstance(result, Exception):
                execution_results['failed_jobs'] += 1
                execution_results['jobs'].append({
                    'status': 'error',
                    'error': str(result),
                    'duration': 0
                })
            else:
                execution_results['jobs'].append(result)
                if result.get('status') == 'completed':
                    execution_results['successful_jobs'] += 1
                else:
                    execution_results['failed_jobs'] += 1
        
        print(f"\n📊 Asynchronous Execution Summary:")
        print(f"   Execution Mode: {execution_mode.upper()}")
        print(f"   Total time: {execution_results['total_duration']:.2f} seconds")
        print(f"   Successful jobs: {execution_results['successful_jobs']}")
        print(f"   Failed jobs: {execution_results['failed_jobs']}")
        if execution_mode == "batch":
            print(f"   Batch size used: {max_concurrent}")
        else:
            print(f"   All {len(jobs)} jobs submitted simultaneously")
        
        return execution_results
    
    async def submit_async_job(self, job_config: Dict[str, Any], sequence_num: int, 
                              total_jobs: int, context_name: str) -> Dict[str, Any]:
        """Submit and monitor a single job asynchronously"""
        job_name = job_config['name']
        job_id = job_config['definition_id']
        
        job_start = time.time()
        job_result = {
            'name': job_name,
            'definition_id': job_id,
            'sequence_number': sequence_num,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'status': 'pending',
            'duration': 0,
            'session_metrics': {},
            'job_metrics': {}
        }
        
        try:
            print(f"🚀 [{sequence_num}/{total_jobs}] Submitting {job_name}...")
            
            # Submit job (async mode - specify context in arguments to avoid routing conflict)
            job_definition_uri = f"/jobDefinitions/definitions/{job_id}"
            async_arguments = job_config.get('arguments', {}).copy()
            async_arguments["_contextName"] = context_name
            
            # Create unique job name for deterministic orchestrator correlation
            unique_job_name = f"{job_name}_async_{sequence_num}_{int(time.time())}"
            
            job_submission = self.client.submit_job(
                job_definition_uri=job_definition_uri,
                arguments=async_arguments,
                name=unique_job_name
            )
            
            job_instance_id = job_submission.get('id')
            session_id = job_submission.get('sessionId')
            
            job_result['job_instance_id'] = job_instance_id
            job_result['session_id'] = session_id
            job_result['unique_job_name'] = unique_job_name
            
            print(f"✅ [{sequence_num}/{total_jobs}] {job_name} submitted - ID: {job_instance_id}")
            print(f"📊 Unique job name: {unique_job_name}")
            
            # No need to capture COMPUTE_JOB ID - we use unique timestamps in names
            
            # Wait for completion asynchronously
            final_status = await self.wait_for_completion_async(job_instance_id, timeout=3600)
            
            job_end = time.time()
            job_duration = job_end - job_start
            
            job_result['end_time'] = datetime.now(timezone.utc).isoformat()
            job_result['duration'] = job_duration
            job_result['status'] = final_status.get('state', 'unknown')
            
            # Collect detailed metrics
            try:
                job_details = self.client.get_job_details(job_instance_id)
                
                # Ensure we have COMPUTE_JOB ID for correlation
                if 'compute_job_id' not in job_result:
                    compute_job_id = job_details.get('results', {}).get('COMPUTE_JOB')
                    if compute_job_id:
                        job_result['compute_job_id'] = compute_job_id
                        print(f"   ✅ COMPUTE_JOB ID captured: {compute_job_id}")
                
                job_result['job_metrics'] = self.extract_job_metrics(job_details)
                
                # Note: Session details collection removed as method doesn't exist
                job_result['session_metrics'] = {'note': 'Session details not available'}
                    
            except Exception as metrics_error:
                print(f"⚠️ [{sequence_num}/{total_jobs}] Could not collect metrics for {job_name}: {metrics_error}")
                job_result['metrics_error'] = str(metrics_error)
            
            if final_status.get('state') == 'completed':
                print(f"✅ [{sequence_num}/{total_jobs}] {job_name} completed in {job_duration:.2f}s")
            else:
                print(f"❌ [{sequence_num}/{total_jobs}] {job_name} failed: {final_status.get('state')}")
                
        except Exception as e:
            job_end = time.time()
            job_duration = job_end - job_start
            
            job_result['end_time'] = datetime.now(timezone.utc).isoformat()
            job_result['duration'] = job_duration
            job_result['status'] = 'error'
            job_result['error'] = str(e)
            
            print(f"❌ [{sequence_num}/{total_jobs}] {job_name} failed: {e}")
        
        return job_result
    
    async def wait_for_completion_async(self, job_id: str, timeout: int = 3600, 
                                       poll_interval: int = 10) -> Dict[str, Any]:
        """
        Asynchronously wait for job completion without blocking other jobs
        Also attempts to collect resource metrics during execution
        
        Args:
            job_id: Job instance ID to monitor
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check job status in seconds
            
        Returns:
            Final job status dictionary with resource metrics if available
        """
        import asyncio
        
        start_time = time.time()
        resource_samples = []
        
        while time.time() - start_time < timeout:
            try:
                job_details = self.client.get_job_details(job_id)
                state = job_details.get('state', 'unknown')
                
                # Try to collect resource metrics while job is running
                if state == 'running':
                    resource_metrics = await self.collect_runtime_metrics(job_details)
                    if resource_metrics:
                        resource_samples.append({
                            'timestamp': time.time(),
                            'metrics': resource_metrics
                        })
                
                if state in ['completed', 'failed', 'canceled']:
                    # Add collected resource samples to job details
                    if resource_samples:
                        job_details['resource_samples'] = resource_samples
                        job_details['avg_resource_usage'] = self.calculate_avg_resources(resource_samples)
                    return job_details
                
                # Sleep asynchronously so other jobs can run
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                print(f"⚠️ Error checking job {job_id}: {e}")
                await asyncio.sleep(poll_interval)
        
        # Timeout reached
        try:
            job_details = self.client.get_job_details(job_id)
            job_details['timeout'] = True
            if resource_samples:
                job_details['resource_samples'] = resource_samples
                job_details['avg_resource_usage'] = self.calculate_avg_resources(resource_samples)
            return job_details
        except:
            return {'state': 'timeout', 'timeout': True, 'id': job_id}
    
    async def collect_runtime_metrics(self, job_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to collect CPU/memory metrics while job is running
        
        Args:
            job_details: Current job details
            
        Returns:
            Dictionary of resource metrics if available
        """
        metrics = {}
        
        try:
            # Try to get compute session info if available
            results = job_details.get('results', {})
            compute_session = results.get('COMPUTE_SESSION')
            
            if compute_session and 'ended' not in compute_session.lower():
                session_id = compute_session.split()[0] if ' ' in compute_session else compute_session
                
                # Try to get session details while it's active
                import requests
                session_url = f"{self.base_url}/compute/sessions/{session_id}"
                headers = {
                    'Authorization': f'Bearer {self.client.access_token}',
                    'Accept': 'application/json'
                }
                
                response = requests.get(session_url, headers=headers, timeout=5)
                if response.status_code == 200:
                    session_data = response.json()
                    
                    # Extract resource information from session
                    metrics.update({
                        'session_state': session_data.get('state'),
                        'node_count': session_data.get('nodeCount'),
                        'cpu_count': session_data.get('cpuCount'), 
                        'memory_size': session_data.get('memorySize'),
                        'session_attributes': session_data.get('attributes', {}),
                        'session_id': session_data.get('id'),
                        'context_id': session_data.get('contextId'),
                        'context_name': session_data.get('contextName')
                    })
                    
                    # Try to get detailed node information
                    node_info = await self.get_session_nodes(session_id, headers)
                    if node_info:
                        metrics['node_details'] = node_info
        
        except Exception as e:
            # Silent failure - resource metrics are optional
            pass
        
        return metrics
    
    async def get_session_nodes(self, session_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Get detailed node information for a compute session (async version)
        
        Args:
            session_id: Session ID to query
            headers: HTTP headers for authentication
            
        Returns:
            Node details if available
        """
        try:
            import requests
            
            # Try different endpoints for node information
            endpoints_to_try = [
                f"{self.base_url}/compute/sessions/{session_id}/nodes",
                f"{self.base_url}/compute/sessions/{session_id}/state",
                f"{self.base_url}/compute/sessions/{session_id}/logs"
            ]
            
            node_info = {}
            
            for endpoint in endpoints_to_try:
                try:
                    response = requests.get(endpoint, headers=headers, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Extract node-related information
                        if 'nodes' in data:
                            node_info['nodes'] = data['nodes']
                        elif 'items' in data:
                            # Check if items contain node information
                            for item in data['items']:
                                if any(key in item for key in ['nodeId', 'nodeName', 'hostname', 'host']):
                                    if 'nodes' not in node_info:
                                        node_info['nodes'] = []
                                    node_info['nodes'].append(item)
                        
                        # Look for any field that might contain node/host information
                        for key, value in data.items():
                            if any(node_key in key.lower() for node_key in ['node', 'host', 'worker', 'executor']):
                                node_info[key] = value
                                
                except Exception:
                    continue
            
            return node_info if node_info else None
            
        except Exception:
            return None
    
    def get_session_nodes_sync(self, session_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Get detailed node information for a compute session (sync version)
        
        Args:
            session_id: Session ID to query
            headers: HTTP headers for authentication
            
        Returns:
            Node details if available
        """
        try:
            import requests
            
            # Try different endpoints for node information
            endpoints_to_try = [
                f"{self.base_url}/compute/sessions/{session_id}/nodes",
                f"{self.base_url}/compute/sessions/{session_id}/state", 
                f"{self.base_url}/compute/sessions/{session_id}/logs"
            ]
            
            node_info = {}
            
            for endpoint in endpoints_to_try:
                try:
                    response = requests.get(endpoint, headers=headers, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Extract node-related information
                        if 'nodes' in data:
                            node_info['nodes'] = data['nodes']
                        elif 'items' in data:
                            # Check if items contain node information
                            for item in data['items']:
                                if any(key in item for key in ['nodeId', 'nodeName', 'hostname', 'host']):
                                    if 'nodes' not in node_info:
                                        node_info['nodes'] = []
                                    node_info['nodes'].append(item)
                        
                        # Look for any field that might contain node/host information
                        for key, value in data.items():
                            if any(node_key in key.lower() for node_key in ['node', 'host', 'worker', 'executor']):
                                node_info[key] = value
                                
                except Exception:
                    continue
            
            return node_info if node_info else None
            
        except Exception:
            return None
    
    def calculate_avg_resources(self, resource_samples: List[Dict]) -> Dict[str, Any]:
        """
        Calculate average resource usage from samples
        
        Args:
            resource_samples: List of resource metric samples
            
        Returns:
            Average resource usage metrics
        """
        if not resource_samples:
            return {}
        
        # Aggregate numeric metrics
        totals = {}
        counts = {}
        
        for sample in resource_samples:
            metrics = sample['metrics']
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0) + value
                    counts[key] = counts.get(key, 0) + 1
        
        # Calculate averages
        averages = {}
        for key in totals:
            if counts[key] > 0:
                averages[f'avg_{key}'] = totals[key] / counts[key]
        
        averages['sample_count'] = len(resource_samples)
        return averages
    
    def wait_for_completion_with_metrics(self, job_id: str, timeout: int = 3600, 
                                        poll_interval: int = 10) -> Dict[str, Any]:
        """
        Wait for job completion while collecting resource metrics (synchronous version)
        
        Args:
            job_id: Job instance ID to monitor
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check job status in seconds
            
        Returns:
            Final job status dictionary with resource metrics if available
        """
        import requests
        
        start_time = time.time()
        resource_samples = []
        
        while time.time() - start_time < timeout:
            try:
                job_details = self.client.get_job_details(job_id)
                state = job_details.get('state', 'unknown')
                
                # Try to collect resource metrics while job is running
                if state == 'running':
                    resource_metrics = self.collect_runtime_metrics_sync(job_details)
                    if resource_metrics:
                        resource_samples.append({
                            'timestamp': time.time(),
                            'metrics': resource_metrics
                        })
                
                if state in ['completed', 'failed', 'canceled']:
                    # Add collected resource samples to job details
                    if resource_samples:
                        job_details['resource_samples'] = resource_samples
                        job_details['avg_resource_usage'] = self.calculate_avg_resources(resource_samples)
                    return job_details
                
                time.sleep(poll_interval)
                
            except Exception as e:
                print(f"⚠️ Error checking job {job_id}: {e}")
                time.sleep(poll_interval)
        
        # Timeout reached
        try:
            job_details = self.client.get_job_details(job_id)
            job_details['timeout'] = True
            if resource_samples:
                job_details['resource_samples'] = resource_samples
                job_details['avg_resource_usage'] = self.calculate_avg_resources(resource_samples)
            return job_details
        except:
            return {'state': 'timeout', 'timeout': True, 'id': job_id}
    
    def collect_runtime_metrics_sync(self, job_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous version of collect_runtime_metrics
        """
        metrics = {}
        
        try:
            # Try to get compute session info if available
            results = job_details.get('results', {})
            compute_session = results.get('COMPUTE_SESSION')
            
            if compute_session and 'ended' not in compute_session.lower():
                session_id = compute_session.split()[0] if ' ' in compute_session else compute_session
                
                # Try to get session details while it's active
                import requests
                session_url = f"{self.base_url}/compute/sessions/{session_id}"
                headers = {
                    'Authorization': f'Bearer {self.client.access_token}',
                    'Accept': 'application/json'
                }
                
                response = requests.get(session_url, headers=headers, timeout=5)
                if response.status_code == 200:
                    session_data = response.json()
                    
                    # Extract resource information from session
                    metrics.update({
                        'session_state': session_data.get('state'),
                        'node_count': session_data.get('nodeCount'),
                        'cpu_count': session_data.get('cpuCount'), 
                        'memory_size': session_data.get('memorySize'),
                        'session_attributes': session_data.get('attributes', {}),
                        'session_id': session_data.get('id'),
                        'context_id': session_data.get('contextId'),
                        'context_name': session_data.get('contextName')
                    })
                    
                    # Try to get detailed node information (sync version)
                    node_info = self.get_session_nodes_sync(session_id, headers)
                    if node_info:
                        metrics['node_details'] = node_info
        
        except Exception as e:
            # Silent failure - resource metrics are optional
            pass
        
        return metrics
    
    def extract_job_metrics(self, job_details: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metrics from job execution details using correct field names"""
        # Use the working field names from the SAS client's extract_execution_metrics method
        metrics = {
            'job_id': job_details.get('id'),
            'job_name': job_details.get('jobRequest', {}).get('name', 'Unnamed'),
            'state': job_details.get('state'),
            'submission_time': job_details.get('creationTimeStamp'),
            'completion_time': job_details.get('endTimeStamp'), 
            'elapsed_time_ms': job_details.get('elapsedTime'),
            'submitted_by': job_details.get('submittedByApplication'),
            'created_by': job_details.get('createdBy'),
            'job_type': job_details.get('jobRequest', {}).get('jobDefinition', {}).get('type'),
            'context_name': job_details.get('jobRequest', {}).get('arguments', {}).get('_contextName'),
            'results_count': len(job_details.get('results', {})),
            'has_log': bool(job_details.get('logLocation')),
            'compute_job_id': job_details.get('results', {}).get('COMPUTE_JOB'),  # CRITICAL for correlation
        }
        
        # Calculate elapsed time in seconds if available
        if metrics['elapsed_time_ms']:
            metrics['elapsed_time_seconds'] = metrics['elapsed_time_ms'] / 1000
        
        # Calculate total execution time if timestamps available
        if metrics['submission_time'] and metrics['completion_time']:
            try:
                from datetime import datetime
                start = datetime.fromisoformat(metrics['submission_time'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(metrics['completion_time'].replace('Z', '+00:00'))
                total_duration = end - start
                metrics['total_duration_seconds'] = total_duration.total_seconds()
            except Exception as e:
                print(f"⚠️ Could not calculate total duration: {e}")
        
        # Extract result file information
        results = job_details.get('results', {})
        if results:
            metrics['result_files'] = list(results.keys())
            metrics['compute_job_id'] = results.get('COMPUTE_JOB')
            metrics['compute_session'] = results.get('COMPUTE_SESSION')
            metrics['compute_context'] = results.get('COMPUTE_CONTEXT')
        
        # Include runtime resource metrics if available
        if 'resource_samples' in job_details:
            metrics['resource_samples'] = job_details['resource_samples']
        
        if 'avg_resource_usage' in job_details:
            metrics['avg_resource_usage'] = job_details['avg_resource_usage']
            
            # Add summary metrics for easy access
            avg_usage = job_details['avg_resource_usage']
            if 'avg_cpu_count' in avg_usage:
                metrics['avg_cpu_count'] = avg_usage['avg_cpu_count']
            if 'avg_memory_size' in avg_usage:
                metrics['avg_memory_size'] = avg_usage['avg_memory_size']
            if 'avg_node_count' in avg_usage:
                metrics['avg_node_count'] = avg_usage['avg_node_count']
        
        # Extract unique nodes used during execution
        if 'resource_samples' in job_details:
            unique_nodes = set()
            node_usage_timeline = []
            
            for sample in job_details['resource_samples']:
                sample_metrics = sample['metrics']
                
                # Extract node information from this sample
                if 'node_details' in sample_metrics:
                    node_details = sample_metrics['node_details']
                    
                    if 'nodes' in node_details:
                        for node in node_details['nodes']:
                            node_id = node.get('nodeId') or node.get('id') or node.get('hostname') or node.get('host')
                            if node_id:
                                unique_nodes.add(str(node_id))
                                
                    # Track node usage over time
                    node_usage_timeline.append({
                        'timestamp': sample['timestamp'],
                        'session_id': sample_metrics.get('session_id'),
                        'node_details': node_details
                    })
            
            if unique_nodes:
                metrics['unique_nodes_used'] = list(unique_nodes)
                metrics['total_unique_nodes'] = len(unique_nodes)
                metrics['node_usage_timeline'] = node_usage_timeline
        
        # NOTE: Hostname extraction from job logs gives Kubernetes pod names, not real nodes
        # Real node hostnames come from orchestrator data (executionHost)
        # We skip log-based hostname extraction to avoid misleading pod-based node counts
        
        # NOTE: Orchestrator data will be fetched at the end to avoid affecting timing measurements
        metrics['orchestrator_data'] = None  # Will be populated later
        metrics['orchestrator_timing'] = None
        metrics['orchestrator_resources'] = None
        metrics['orchestrator_context'] = None
        
        return metrics
    
    def extract_session_metrics(self, session_details: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metrics from session details"""
        return {
            'state': session_details.get('state'),
            'creation_time': session_details.get('creationTimeStamp'),
            'start_time': session_details.get('startTimeStamp'),
            'end_time': session_details.get('endTimeStamp'),
            'attributes': session_details.get('attributes', {}),
            'context_name': session_details.get('contextName'),
            'node_count': session_details.get('nodeCount'),
            'cpu_count': session_details.get('cpuCount'),
            'memory_size': session_details.get('memorySize')
        }
    
    def calculate_comparison_metrics(self) -> Dict[str, Any]:
        """Calculate comparison metrics between sequential and async execution"""
        seq_results = self.results['sequential_execution']
        async_results = self.results['async_execution']
        
        comparison = {
            'execution_time': {
                'sequential_total': seq_results.get('total_duration', 0),
                'async_total': async_results.get('total_duration', 0),
                'time_saved': 0,
                'efficiency_gain': 0
            },
            'job_performance': {
                'sequential_success_rate': 0,
                'async_success_rate': 0,
                'sequential_avg_job_time': 0,
                'async_avg_job_time': 0
            },
            'resource_utilization': {
                'sequential_context': seq_results.get('context_name'),
                'async_context': async_results.get('context_name'),
                'concurrent_execution': True
            }
        }
        
        # Calculate time efficiency
        seq_time = seq_results.get('total_duration', 0)
        async_time = async_results.get('total_duration', 0)
        
        if seq_time > 0 and async_time > 0:
            comparison['execution_time']['time_saved'] = seq_time - async_time
            comparison['execution_time']['efficiency_gain'] = ((seq_time - async_time) / seq_time) * 100
        
        # Calculate success rates
        seq_total = len(seq_results.get('jobs', []))
        async_total = len(async_results.get('jobs', []))
        
        if seq_total > 0:
            comparison['job_performance']['sequential_success_rate'] = (
                seq_results.get('successful_jobs', 0) / seq_total
            ) * 100
        
        if async_total > 0:
            comparison['job_performance']['async_success_rate'] = (
                async_results.get('successful_jobs', 0) / async_total
            ) * 100
        
        # Calculate average job times
        seq_job_times = [job.get('duration', 0) for job in seq_results.get('jobs', [])]
        async_job_times = [job.get('duration', 0) for job in async_results.get('jobs', [])]
        
        if seq_job_times:
            comparison['job_performance']['sequential_avg_job_time'] = sum(seq_job_times) / len(seq_job_times)
        
        if async_job_times:
            comparison['job_performance']['async_avg_job_time'] = sum(async_job_times) / len(async_job_times)
        
        # Analyze node distribution and autoscaling behavior
        comparison['node_analysis'] = self.analyze_node_distribution()
        
        return comparison
    
    def analyze_node_distribution(self) -> Dict[str, Any]:
        """
        Analyze how jobs were distributed across nodes in sequential vs async execution
        
        Returns:
            Dictionary containing node distribution analysis
        """
        analysis = {
            'sequential_nodes': {},
            'async_nodes': {},
            'autoscaling_behavior': {}
        }
        
        try:
            # Analyze sequential execution node usage
            seq_jobs = self.results.get('sequential_execution', {}).get('jobs', [])
            seq_unique_nodes = set()
            
            for job in seq_jobs:
                job_metrics = job.get('job_metrics', {})
                
                # Use actual hostname if available, otherwise fall back to session-based nodes
                if 'actual_hostname' in job_metrics:
                    hostname = job_metrics['actual_hostname']
                    seq_unique_nodes.add(hostname)
                    analysis['sequential_nodes'][job['name']] = {
                        'hostname': hostname,
                        'node_count': 1
                    }
                elif 'unique_nodes_used' in job_metrics:
                    nodes = job_metrics['unique_nodes_used']
                    seq_unique_nodes.update(nodes)
                    analysis['sequential_nodes'][job['name']] = {
                        'nodes_used': nodes,
                        'node_count': len(nodes) if nodes else 0
                    }
            
            analysis['sequential_nodes']['total_unique_nodes'] = len(seq_unique_nodes)
            analysis['sequential_nodes']['all_nodes'] = list(seq_unique_nodes)
            
            # Analyze async execution node usage
            async_jobs = self.results.get('async_execution', {}).get('jobs', [])
            async_unique_nodes = set()
            concurrent_node_usage = {}
            
            for job in async_jobs:
                job_metrics = job.get('job_metrics', {})
                
                # Use actual hostname if available, otherwise fall back to session-based nodes
                if 'actual_hostname' in job_metrics:
                    hostname = job_metrics['actual_hostname']
                    async_unique_nodes.add(hostname)
                    analysis['async_nodes'][job['name']] = {
                        'hostname': hostname,
                        'node_count': 1
                    }
                    
                    # Track concurrent usage
                    if hostname not in concurrent_node_usage:
                        concurrent_node_usage[hostname] = []
                    concurrent_node_usage[hostname].append(job['name'])
                    
                elif 'unique_nodes_used' in job_metrics:
                    nodes = job_metrics['unique_nodes_used']
                    async_unique_nodes.update(nodes)
                    analysis['async_nodes'][job['name']] = {
                        'nodes_used': nodes,
                        'node_count': len(nodes) if nodes else 0
                    }
                    
                    # Track concurrent usage
                    for node in nodes:
                        if node not in concurrent_node_usage:
                            concurrent_node_usage[node] = []
                        concurrent_node_usage[node].append(job['name'])
            
            analysis['async_nodes']['total_unique_nodes'] = len(async_unique_nodes)
            analysis['async_nodes']['all_nodes'] = list(async_unique_nodes)
            analysis['async_nodes']['concurrent_node_usage'] = concurrent_node_usage
            
            # Analyze autoscaling behavior
            analysis['autoscaling_behavior'] = {
                'sequential_vs_async_nodes': {
                    'sequential_total': len(seq_unique_nodes),
                    'async_total': len(async_unique_nodes),
                    'additional_nodes_used': len(async_unique_nodes - seq_unique_nodes),
                    'node_scaling_factor': len(async_unique_nodes) / len(seq_unique_nodes) if seq_unique_nodes else 0
                },
                'concurrent_job_distribution': len([node for node, jobs in concurrent_node_usage.items() if len(jobs) > 1]),
                'nodes_with_multiple_jobs': {node: jobs for node, jobs in concurrent_node_usage.items() if len(jobs) > 1}
            }
            
            # Determine autoscaling effectiveness
            if len(async_unique_nodes) > len(seq_unique_nodes):
                analysis['autoscaling_behavior']['scaling_observed'] = True
                analysis['autoscaling_behavior']['scaling_type'] = 'scale_out'
                analysis['autoscaling_behavior']['description'] = f"Autoscaling scaled out from {len(seq_unique_nodes)} to {len(async_unique_nodes)} nodes"
            elif len(async_unique_nodes) == len(seq_unique_nodes):
                analysis['autoscaling_behavior']['scaling_observed'] = False
                analysis['autoscaling_behavior']['scaling_type'] = 'no_scaling'
                analysis['autoscaling_behavior']['description'] = "No additional nodes were allocated for concurrent execution"
            else:
                analysis['autoscaling_behavior']['scaling_observed'] = True
                analysis['autoscaling_behavior']['scaling_type'] = 'consolidation'
                analysis['autoscaling_behavior']['description'] = "Jobs were consolidated on fewer nodes"
        
        except Exception as e:
            analysis['error'] = f"Could not analyze node distribution: {e}"
        
        return analysis
    
    def extract_hostname_from_log(self, log_content: str) -> str:
        """
        Extract hostname from SAS job log that contains %put RUNNING ON HOST: &SYSHOSTNAME; output
        
        Args:
            log_content: Raw job log content
            
        Returns:
            Hostname if found, None otherwise
        """
        import re
        import json
        
        try:
            # Try to parse as JSON first (SAS Viya format)
            log_data = json.loads(log_content)
            items = log_data.get('items', [])
            
            for item in items:
                line = item.get('line', '')
                
                # Look for our specific hostname message
                if 'RUNNING ON HOST:' in line:
                    match = re.search(r'RUNNING ON HOST:\s*([^\s,;]+)', line)
                    if match:
                        return match.group(1).strip()
                
                # Also check for other hostname patterns
                hostname_patterns = [
                    r'SYSHOSTNAME[=:\s]+([a-zA-Z0-9\-\.]+)',
                    r'hostname[=:\s]+([a-zA-Z0-9\-\.]+)',
                ]
                
                for pattern in hostname_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
        
        except json.JSONDecodeError:
            # If not JSON, try plain text parsing
            lines = log_content.split('\n')
            for line in lines:
                if 'RUNNING ON HOST:' in line:
                    match = re.search(r'RUNNING ON HOST:\s*([^\s,;]+)', line)
                    if match:
                        return match.group(1).strip()
        
        return None
    
    def get_workload_orchestrator_data(self, job_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive workload orchestrator data for accurate job analysis
        Using the discovered naming pattern for better correlation
        
        Args:
            job_details: Job details from SAS Job Execution API
            
        Returns:
            Comprehensive workload orchestrator data with timing, resources, and node info
        """
        import requests
        import time as time_module
        
        try:
            headers = {
                'Authorization': f'Bearer {self.client.access_token}',
                'Accept': 'application/json'  # Critical for orchestrator API (no Content-Type needed for GET)
            }
            
            # Get job identifiers
            job_name = job_details.get('name', '')  # This is the unique job name
            job_creation_time = job_details.get('creationTimeStamp')
            
            # Try to use the naming pattern for pooled jobs
            expected_prefix = None
            if '_pooled' in job_name:
                # Handle both regular and unique job names
                if '_sequential_' in job_name or '_async_' in job_name:
                    # This is a unique job name like cm_pooled_sequential_1_1757476678
                    # It becomes cmpooledsequential11757476678-[UUID] in orchestrator
                    transformed = job_name.replace('_', '').lower()
                    expected_prefix = transformed
                    print(f"🔍 DEBUG - Unique job name transformation:")
                    print(f"   Submitted job name: {job_name}")
                    print(f"   Transformed prefix: {transformed}")
                    print(f"   Looking for orchestrator names starting with: {transformed}")
                else:
                    # Regular pooled job name like cm_pooled
                    # It becomes cmpooled-[UUID] in orchestrator
                    base_name = job_name.split('_')[0] + '_pooled'
                    expected_prefix = base_name.replace('_', '').lower()
                    print(f"🔍 DEBUG - Regular job name transformation:")
                    print(f"   Original job name: {job_name}")
                    print(f"   Base name: {base_name}")
                    print(f"   Expected prefix: {expected_prefix}")
            
            if not expected_prefix:
                print(f"⚠️ No naming pattern found for job: {job_name}")
                print(f"   This job doesn't appear to be a pooled job")
                return None
            
            # Add small delay to allow orchestrator data to be available
            # The orchestrator API might have a slight delay after job completion
            time_module.sleep(2)
            
            # Add retry logic for orchestrator API calls during high concurrency
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Get recent workload orchestrator jobs with larger limit for concurrent scenarios
                    list_url = f"{self.base_url}/workloadOrchestrator/jobs"
                    list_params = {
                        'limit': 500, 
                        'start': 0,
                        'sortBy': 'submitTime:descending'  # Get most recent jobs first
                    }
                    
                    response = requests.get(list_url, headers=headers, params=list_params, timeout=30)
                    if response.status_code != 200:
                        print(f"⚠️ Orchestrator API returned status {response.status_code}, attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            time_module.sleep(2 * (attempt + 1))  # Exponential backoff
                            continue
                        return None
                    
                    jobs_list = response.json()
                    print(f"   📊 Retrieved {len(jobs_list.get('items', []))} orchestrator jobs from API")
                    break
                    
                except requests.exceptions.Timeout as e:
                    print(f"⚠️ Orchestrator API timeout on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time_module.sleep(2 * (attempt + 1))
                        continue
                    return None
                except Exception as e:
                    print(f"⚠️ Orchestrator API error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time_module.sleep(2 * (attempt + 1))
                        continue
                    return None
            
            # Get the Job Execution job ID for deterministic correlation
            job_execution_id = job_details.get('id')
            job_execution_name = job_details.get('name', '')
            
            print(f"🔍 Looking for orchestrator match:")
            print(f"   Job Execution ID: {job_execution_id}")
            print(f"   Job Name: {job_execution_name}")
            
            # Find matching workload job with deterministic matching
            # The key insight: orchestrator job names contain the COMPUTE_JOB ID, not the Job Execution ID
            
            # First pass: Try to match by naming pattern for pooled jobs
            # Since we use timestamps in job names, each submission has a unique name
            if expected_prefix:
                print(f"🔍 Searching orchestrator jobs for prefix: {expected_prefix}")
                
                # Check in batches if needed
                total_checked = 0
                page_start = 0
                max_pages = 3  # Check up to 3 pages (1500 jobs total)
                
                for page in range(max_pages):
                    if page > 0:
                        # Fetch next page
                        print(f"   📄 Checking page {page + 1}...")
                        list_params['start'] = page * 500
                        
                        try:
                            response = requests.get(list_url, headers=headers, params=list_params, timeout=30)
                            if response.status_code != 200:
                                break
                            jobs_list = response.json()
                        except:
                            break
                    
                    items = jobs_list.get('items', [])
                    if not items:
                        break
                    
                    for i, workload_job in enumerate(items):
                        workload_name = workload_job.get('request', {}).get('name', '')
                        total_checked += 1
                        
                        # Debug: Show first few orchestrator job names from first page
                        if page == 0 and i < 5:
                            print(f"   Job {i+1}: {workload_name[:60]}...")
                        
                        # Check if this matches our naming pattern (should be unique due to timestamp)
                        if workload_name.lower().startswith(expected_prefix):
                            print(f"✅ Found unique naming pattern match at position {total_checked}!")
                            print(f"   Orchestrator job: {workload_name}")
                            print(f"   Queue: {workload_job.get('request', {}).get('queue', '')}")
                            print(f"   Submit time: {workload_job.get('processingInfo', {}).get('submitTime')}")
                            
                            # Since timestamps make names unique, we can return immediately
                            return self.extract_comprehensive_orchestrator_metrics(workload_job)
                    
                    print(f"   Checked {len(items)} jobs on page {page + 1}, total checked: {total_checked}")
                
                print(f"❌ No orchestrator jobs found with prefix: {expected_prefix} after checking {total_checked} jobs")
                return None
            
            # If we get here without finding a match, something went wrong
            print(f"❌ Orchestrator correlation failed for job: {job_name}")
            return None
            
        except Exception as e:
            print(f"⚠️ Could not fetch workload orchestrator data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_time_diff(self, time1: str, time2: str) -> float:
        """Calculate absolute time difference in seconds between two ISO timestamps"""
        try:
            from datetime import datetime
            if not time1 or not time2:
                return float('inf')
            t1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
            t2 = datetime.fromisoformat(time2.replace('Z', '+00:00'))
            return abs((t1 - t2).total_seconds())
        except:
            return float('inf')
    
    def extract_comprehensive_orchestrator_metrics(self, workload_job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive metrics from workload orchestrator job data
        
        Args:
            workload_job: Complete workload orchestrator job object
            
        Returns:
            Comprehensive metrics including timing, resources, and node information
        """
        processing_info = workload_job.get('processingInfo', {})
        request_info = workload_job.get('request', {})
        
        # Extract timing information
        submit_time = processing_info.get('submitTime')
        start_time = processing_info.get('startTime')
        end_time = processing_info.get('endTime')
        
        # Calculate timing metrics
        queue_wait_seconds = None
        execution_seconds = None
        total_seconds = None
        
        if submit_time and start_time:
            queue_wait_seconds = self.calculate_time_diff(submit_time, start_time)
        
        if start_time and end_time:
            execution_seconds = self.calculate_time_diff(start_time, end_time)
            
        if submit_time and end_time:
            total_seconds = self.calculate_time_diff(submit_time, end_time)
        
        # Extract resource information
        consumed_resources = processing_info.get('consumedResources', [])
        cpu_cores = None
        memory_mb = None
        
        for resource in consumed_resources:
            if resource.get('name') == 'cores':
                cpu_cores = resource.get('value')
            elif resource.get('name') == 'memory':
                memory_mb = resource.get('value')
        
        # Extract node and queue information
        execution_host = processing_info.get('executionHost')
        queue_name = request_info.get('queue', 'unknown')
        
        # Identify execution context based on queue
        context_type = 'synchronous' if queue_name == 'default' else 'asynchronous'
        
        # Host scheduling information
        host_scheduling_info = processing_info.get('hostSchedulingInfo', [])
        scheduled_hosts = []
        for host_info in host_scheduling_info:
            scheduled_hosts.append({
                'hostname': host_info.get('hostName'),
                'queue': host_info.get('queueName'),
                'state': host_info.get('state')
            })
        
        return {
            # Identification
            'workload_job_id': workload_job.get('id'),
            'workload_job_name': request_info.get('name'),
            'queue_name': queue_name,
            'context_type': context_type,
            
            # Timing metrics (all in seconds)
            'submit_time': submit_time,
            'start_time': start_time,
            'end_time': end_time,
            'queue_wait_seconds': queue_wait_seconds,
            'execution_seconds': execution_seconds,
            'total_seconds': total_seconds,
            
            # Detailed timing breakdown (milliseconds from orchestrator)
            'total_time_running': processing_info.get('totalTimeRunning'),
            'total_time_pending': processing_info.get('totalTimePending'),
            'total_time_starting': processing_info.get('totalTimeStarting'),
            'total_time_susp_admin': processing_info.get('totalTimeSuspAdmin'),
            'total_time_susp_thresh': processing_info.get('totalTimeSuspThresh'),
            'total_time_susp_preempt': processing_info.get('totalTimeSuspPreempt'),
            
            # Resource utilization
            'cpu_cores': cpu_cores,
            'memory_mb': memory_mb,
            'consumed_resources': consumed_resources,
            
            # Resource limits and peak usage
            'limit_values': processing_info.get('limitValues', []),
            'max_memory_used': next((lv['value'] for lv in processing_info.get('limitValues', []) if lv.get('name') == 'maxMemory'), None),
            'max_cpu_time': next((lv['value'] for lv in processing_info.get('limitValues', []) if lv.get('name') == 'maxCpuTime'), None),
            'max_io_total': next((lv['value'] for lv in processing_info.get('limitValues', []) if lv.get('name') == 'maxIoTotal'), None),
            
            # Node allocation and infrastructure
            'execution_host': execution_host,
            'execution_ip': processing_info.get('executionIP'),
            'scheduled_hosts': scheduled_hosts,
            'k8s_object_name': processing_info.get('k8sObjName'),
            
            # Autoscaling and reliability indicators
            'requeue_count': processing_info.get('requeueCount', 0),
            'launch_status': processing_info.get('launchStatus'),
            'queue_priority': processing_info.get('queuePriority'),
            'suspend_flags': processing_info.get('suspendFlags', 0),
            'exit_code': processing_info.get('exitCode'),
            
            # Job status
            'status': processing_info.get('status', 'unknown'),
            'state': processing_info.get('state'),
            'pending_cause': processing_info.get('pendingCause', ''),
            
            # Additional orchestrator data
            'orchestrator_data_available': True,
            'orchestrator_fetch_time': datetime.now(timezone.utc).isoformat()
        }
    
    def extract_node_info_from_workload_data(self, workload_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract node/hostname information from workload orchestrator data"""
        node_info = {}
        
        # Common fields that might contain node information
        node_fields = ['hostname', 'host', 'node', 'nodeName', 'workerNode', 'executionHost', 'server', 'machine']
        
        def search_for_nodes(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check if this key might contain node info
                    if any(field.lower() in key.lower() for field in node_fields):
                        if isinstance(value, str) and value.strip():
                            node_info[current_path] = value
                    
                    # Recurse into nested objects
                    if isinstance(value, (dict, list)):
                        search_for_nodes(value, current_path)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_for_nodes(item, f"{path}[{i}]")
        
        search_for_nodes(workload_data)
        
        return node_info
    
    def extract_workload_node_info(self, workload_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract node information from workload orchestrator data"""
        processing_info = workload_data.get('processingInfo', {})
        
        return {
            'execution_host': processing_info.get('executionHost'),
            'execution_ip': processing_info.get('executionIP'),
            'host_scheduling_info': processing_info.get('hostSchedulingInfo', []),
            'k8s_object_name': processing_info.get('k8sObjName'),
            'execution_host_aliases': processing_info.get('executionHostAliases', [])
        }
    
    def extract_workload_resources(self, workload_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract resource usage from workload orchestrator data"""
        processing_info = workload_data.get('processingInfo', {})
        request_info = workload_data.get('request', {})
        
        # Extract consumed resources
        consumed = {}
        for resource in processing_info.get('consumedResources', []):
            consumed[resource.get('name')] = resource.get('value')
        
        # Extract limit values (actual usage metrics)
        limits = {}
        for limit in processing_info.get('limitValues', []):
            limits[limit.get('name')] = limit.get('value')
        
        return {
            'consumed_resources': consumed,
            'limit_values': limits,
            'timing': {
                'total_time_pending': processing_info.get('totalTimePending'),
                'total_time_starting': processing_info.get('totalTimeStarting'),
                'total_time_running': processing_info.get('totalTimeRunning'),
                'submit_time': processing_info.get('submitTime'),
                'start_time': processing_info.get('startTime'),
                'end_time': processing_info.get('endTime')
            },
            'state': processing_info.get('state'),
            'exit_code': processing_info.get('exitCode')
        }
    
    def correlate_all_orchestrator_data(self) -> None:
        """
        Correlate all orchestrator data at the very end of execution.
        This ensures orchestrator querying doesn't affect job timing measurements.
        """
        print(f"\n🔄 Correlating all orchestrator data (post-execution)...")
        print("=" * 60)
        
        # Collect all jobs from both sequential and async executions
        all_jobs = []
        
        if 'sequential_execution' in self.results and self.results['sequential_execution']:
            seq_jobs = self.results['sequential_execution'].get('jobs', [])
            if seq_jobs:
                print(f"📊 Processing {len(seq_jobs)} sequential jobs...")
                all_jobs.extend([(job, 'sequential') for job in seq_jobs])
        
        if 'async_execution' in self.results and self.results['async_execution']:
            async_jobs = self.results['async_execution'].get('jobs', [])
            if async_jobs:
                print(f"📊 Processing {len(async_jobs)} async jobs...")
                all_jobs.extend([(job, 'async') for job in async_jobs])
        
        if not all_jobs:
            print("⚠️ No jobs to correlate")
            return
        
        print(f"📊 Total jobs to correlate: {len(all_jobs)}")
        
        # Wait a bit to ensure all orchestrator jobs are created
        print("⏳ Waiting 15 seconds for orchestrator data to be available...")
        time.sleep(15)
        
        # Process each job
        successful_correlations = 0
        failed_correlations = 0
        
        for job_data, execution_type in all_jobs:
            try:
                # The job_data structure has these fields from the execution
                job_name = job_data.get('unique_job_name') or job_data.get('name', 'unknown')
                
                # We need to reconstruct job_details for the orchestrator correlation
                # Since we use unique timestamps, we don't need COMPUTE_JOB ID
                job_details = {
                    'id': job_data.get('job_instance_id'),
                    'name': job_name,
                    'results': {}  # No COMPUTE_JOB needed
                }
                
                if not job_details:
                    print(f"⚠️ No job details for {job_name}")
                    failed_correlations += 1
                    continue
                
                # Get orchestrator data for this job
                orchestrator_data = self.get_workload_orchestrator_data(job_details)
                
                if orchestrator_data:
                    # Update the job's metrics with orchestrator data
                    if 'metrics' not in job_data:
                        job_data['metrics'] = {}
                    
                    job_data['metrics']['orchestrator_data'] = orchestrator_data
                    
                    # Override hostname with correct orchestrator hostname
                    if orchestrator_data.get('execution_host'):
                        job_data['metrics']['actual_hostname'] = orchestrator_data['execution_host']
                    
                    # Add orchestrator timing metrics
                    job_data['metrics']['orchestrator_timing'] = {
                        'queue_wait_seconds': orchestrator_data.get('queue_wait_seconds'),
                        'execution_seconds': orchestrator_data.get('execution_seconds'),
                        'total_seconds': orchestrator_data.get('total_seconds')
                    }
                    
                    # Add resource utilization
                    job_data['metrics']['orchestrator_resources'] = {
                        'cpu_cores': orchestrator_data.get('cpu_cores'),
                        'memory_mb': orchestrator_data.get('memory_mb')
                    }
                    
                    # Add context identification
                    job_data['metrics']['orchestrator_context'] = {
                        'queue_name': orchestrator_data.get('queue_name'),
                        'context_type': orchestrator_data.get('context_type')
                    }
                    
                    successful_correlations += 1
                    print(f"✅ Correlated {execution_type} job: {job_name}")
                else:
                    failed_correlations += 1
                    print(f"⚠️ No orchestrator data found for {execution_type} job: {job_name}")
                    
            except Exception as e:
                failed_correlations += 1
                print(f"❌ Error correlating job: {e}")
        
        print("=" * 60)
        print(f"📊 Orchestrator Correlation Summary:")
        print(f"   ✅ Successful: {successful_correlations}")
        print(f"   ❌ Failed: {failed_correlations}")
        print(f"   📈 Success Rate: {successful_correlations / len(all_jobs) * 100:.1f}%")
    
    def batch_correlate_orchestrator_data(self, execution_results: Dict[str, Any]) -> None:
        """
        DEPRECATED: This method is kept for backward compatibility but does nothing.
        Orchestrator correlation is now done at the very end via correlate_all_orchestrator_data().
        
        Args:
            execution_results: Results from sequential or async execution
        """
        # This method intentionally does nothing - orchestrator correlation moved to end
        pass
        
        # Fetch all orchestrator jobs at once
        print("📊 Fetching orchestrator jobs...")
        try:
            import requests
            headers = {
                'Authorization': f'Bearer {self.client.access_token}',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            # Get a large batch of orchestrator jobs
            orch_url = f"{self.base_url}/workloadOrchestrator/jobs"
            orch_params = {'limit': 500, 'start': 0}
            
            response = requests.get(orch_url, headers=headers, params=orch_params, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ Failed to fetch orchestrator jobs: {response.status_code}")
                return
            
            orchestrator_jobs = response.json().get('items', [])
            print(f"✅ Retrieved {len(orchestrator_jobs)} orchestrator jobs")
            
            # Create a lookup map for faster correlation (case-insensitive)
            orch_lookup = {}
            for orch_job in orchestrator_jobs:
                orch_name = orch_job.get('request', {}).get('name', '')
                # Extract UUID from orchestrator job name (after 'sas-compute-server-')
                if 'sas-compute-server-' in orch_name.lower():
                    uuid_part = orch_name.lower().split('sas-compute-server-')[1]
                    orch_lookup[uuid_part] = orch_job
                    # Also store uppercase variant
                    orch_lookup[uuid_part.upper()] = orch_job
            
            print(f"📊 Found {len(orch_lookup)//2} compute server jobs in orchestrator")
            
            # Correlate each job
            successful_correlations = 0
            failed_correlations = 0
            
            for job in jobs_to_correlate:
                job_name = job.get('name', 'Unknown')
                compute_job_id = job.get('compute_job_id')
                
                if not compute_job_id:
                    # Try to get it from job_metrics if not in top level
                    compute_job_id = job.get('job_metrics', {}).get('compute_job_id')
                
                if compute_job_id:
                    # Try both original case and lowercase
                    found = False
                    for variant in [compute_job_id, compute_job_id.lower(), compute_job_id.upper()]:
                        if variant in orch_lookup:
                            orch_job = orch_lookup[variant]
                            successful_correlations += 1
                            found = True
                            
                            # Extract and add orchestrator data to job
                            orchestrator_data = self.extract_comprehensive_orchestrator_metrics(orch_job)
                            
                            # Update job metrics with orchestrator data
                            if 'job_metrics' not in job:
                                job['job_metrics'] = {}
                            
                            job['job_metrics']['orchestrator_data'] = orchestrator_data
                            job['job_metrics']['orchestrator_correlation'] = 'batch_success'
                            
                            print(f"✅ Correlated: {job_name} -> {orch_job.get('request', {}).get('name')[:50]}...")
                            break
                    
                    if not found:
                        failed_correlations += 1
                        print(f"❌ Not found: {job_name} (ID: {compute_job_id})")
                else:
                    failed_correlations += 1
                    print(f"⚠️ No COMPUTE_JOB ID for: {job_name}")
            
            print(f"\n📊 Batch Correlation Results:")
            print(f"   ✅ Successful: {successful_correlations}/{len(jobs_to_correlate)}")
            print(f"   ❌ Failed: {failed_correlations}/{len(jobs_to_correlate)}")
            
            if failed_correlations > 0:
                print(f"\n💡 Troubleshooting failed correlations:")
                print(f"   • Orchestrator jobs might use different case UUIDs")
                print(f"   • Some jobs might not have created orchestrator entries")
                print(f"   • Try running batch_correlate.py separately with longer delay")
            
        except Exception as e:
            print(f"❌ Batch correlation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def save_results(self, output_file: str = None) -> str:
        """Save comprehensive results to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/pooled_jobs_comparison_{timestamp}.json"
        
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
        
        # Calculate comparison metrics
        self.results['comparison'] = self.calculate_comparison_metrics()
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        return output_file
    
    def print_summary(self):
        """Print execution summary"""
        comparison = self.results.get('comparison', {})
        
        print(f"\n" + "=" * 60)
        print(f"📊 POOLED JOBS COMPARISON SUMMARY")
        print(f"=" * 60)
        
        exec_time = comparison.get('execution_time', {})
        seq_time = exec_time.get('sequential_total', 0)
        async_time = exec_time.get('async_total', 0)
        time_saved = exec_time.get('time_saved', 0)
        efficiency = exec_time.get('efficiency_gain', 0)
        
        print(f"⏱️  Execution Times:")
        print(f"   Sequential (Compute Context): {seq_time:.2f} seconds")
        print(f"   Asynchronous (Autoscaling):   {async_time:.2f} seconds")
        print(f"   Time Saved: {time_saved:.2f} seconds")
        print(f"   Efficiency Gain: {efficiency:.1f}%")
        
        job_perf = comparison.get('job_performance', {})
        print(f"\n📈 Job Performance:")
        print(f"   Sequential Success Rate: {job_perf.get('sequential_success_rate', 0):.1f}%")
        print(f"   Async Success Rate: {job_perf.get('async_success_rate', 0):.1f}%")
        print(f"   Sequential Avg Job Time: {job_perf.get('sequential_avg_job_time', 0):.2f}s")
        print(f"   Async Avg Job Time: {job_perf.get('async_avg_job_time', 0):.2f}s")
        
        seq_jobs = len(self.results.get('sequential_execution', {}).get('jobs', []))
        async_jobs = len(self.results.get('async_execution', {}).get('jobs', []))
        
        print(f"\n📋 Jobs Executed:")
        print(f"   Total Jobs: {seq_jobs}")
        print(f"   Sequential Successful: {self.results.get('sequential_execution', {}).get('successful_jobs', 0)}")
        print(f"   Async Successful: {self.results.get('async_execution', {}).get('successful_jobs', 0)}")
        
        # Show node distribution analysis
        node_analysis = comparison.get('node_analysis', {})
        if node_analysis:
            autoscaling_behavior = node_analysis.get('autoscaling_behavior', {})
            
            print(f"\n🖥️  Node Distribution:")
            seq_nodes = node_analysis.get('sequential_nodes', {}).get('total_unique_nodes', 0)
            async_nodes = node_analysis.get('async_nodes', {}).get('total_unique_nodes', 0)
            
            print(f"   Sequential Nodes Used: {seq_nodes}")
            print(f"   Async Nodes Used: {async_nodes}")
            
            if autoscaling_behavior.get('scaling_observed'):
                scaling_type = autoscaling_behavior.get('scaling_type', 'unknown')
                description = autoscaling_behavior.get('description', '')
                print(f"   🚀 Autoscaling: {scaling_type.upper()} - {description}")
            else:
                print(f"   ⚪ Autoscaling: No scaling observed")
            
            # Show concurrent job distribution
            concurrent_nodes = autoscaling_behavior.get('concurrent_job_distribution', 0)
            if concurrent_nodes > 0:
                print(f"   🔄 Nodes with Multiple Concurrent Jobs: {concurrent_nodes}")
                
                multiple_jobs = autoscaling_behavior.get('nodes_with_multiple_jobs', {})
                for node, jobs in list(multiple_jobs.items())[:3]:  # Show first 3
                    print(f"      • Node {node}: {', '.join(jobs)}")
                if len(multiple_jobs) > 3:
                    print(f"      • ... and {len(multiple_jobs) - 3} more nodes")
        
        # Add orchestrator insights
        self.print_orchestrator_summary()
    
    def print_orchestrator_summary(self):
        """Print workload orchestrator insights"""
        print(f"\n🎯 WORKLOAD ORCHESTRATOR INSIGHTS")
        print(f"=" * 60)
        
        seq_exec = self.results.get('sequential_execution', {})
        async_exec = self.results.get('async_execution', {})
        
        orchestrator_data_found = False
        
        # Analyze orchestrator data for each phase
        for phase_name, phase_data in [('Sequential', seq_exec), ('Asynchronous', async_exec)]:
            jobs = phase_data.get('jobs', [])
            # Check both possible locations for orchestrator data
            orchestrator_jobs = [
                job for job in jobs 
                if (job.get('metrics', {}).get('orchestrator_data') or 
                    job.get('job_metrics', {}).get('orchestrator_data'))
            ]
            
            if orchestrator_jobs:
                orchestrator_data_found = True
                print(f"\n📊 {phase_name} Execution:")
                
                # Queue analysis
                queues = {}
                total_wait_time = 0
                total_exec_time = 0
                resource_usage = {'cpu': 0, 'memory': 0}
                hosts = set()
                
                for job in orchestrator_jobs:
                    # Get orchestrator data from either location
                    orch_data = (job.get('metrics', {}).get('orchestrator_data') or 
                                job.get('job_metrics', {}).get('orchestrator_data'))
                    
                    # Queue information
                    queue_name = orch_data.get('queue_name', 'unknown')
                    queues[queue_name] = queues.get(queue_name, 0) + 1
                    
                    # Timing
                    wait_time = orch_data.get('queue_wait_seconds')
                    exec_time = orch_data.get('execution_seconds')
                    if wait_time is not None:
                        total_wait_time += wait_time
                    if exec_time is not None:
                        total_exec_time += exec_time
                    
                    # Resources
                    cpu = orch_data.get('cpu_cores')
                    memory = orch_data.get('memory_mb')
                    if cpu:
                        resource_usage['cpu'] += cpu
                    if memory:
                        resource_usage['memory'] += memory
                    
                    # Hosts
                    host = orch_data.get('execution_host')
                    if host:
                        hosts.add(host)
                
                # Display insights
                print(f"   📤 Queues Used:")
                for queue, count in queues.items():
                    context_type = "Autoscaling" if queue == "autoscaling-poc-queue" else "Standard"
                    print(f"      • {queue} ({context_type}): {count} jobs")
                
                print(f"   🖥️  Compute Nodes: {len(hosts)} unique")
                for host in list(hosts)[:2]:
                    print(f"      • {host}")
                if len(hosts) > 2:
                    print(f"      • ... and {len(hosts) - 2} more")
                
                if orchestrator_jobs:
                    avg_wait = total_wait_time / len(orchestrator_jobs)
                    avg_exec = total_exec_time / len(orchestrator_jobs)
                    avg_cpu = resource_usage['cpu'] / len(orchestrator_jobs)
                    avg_memory = resource_usage['memory'] / len(orchestrator_jobs)
                    
                    print(f"   ⏱️  Average Timing:")
                    print(f"      • Queue Wait: {avg_wait:.2f} seconds")
                    print(f"      • Execution: {avg_exec:.2f} seconds")
                    
                    print(f"   💾 Average Resources:")
                    print(f"      • CPU: {avg_cpu:.3f} cores")
                    print(f"      • Memory: {avg_memory:.1f} MB")
                    
                    # Calculate autoscaling-specific metrics
                    total_requeues = sum(job.get('metrics', {}).get('orchestrator_data', {}).get('requeue_count', 0) for job in orchestrator_jobs)
                    total_suspensions = sum(
                        (job.get('metrics', {}).get('orchestrator_data', {}).get('total_time_susp_admin', 0) or 0) +
                        (job.get('metrics', {}).get('orchestrator_data', {}).get('total_time_susp_thresh', 0) or 0) +
                        (job.get('metrics', {}).get('orchestrator_data', {}).get('total_time_susp_preempt', 0) or 0)
                        for job in orchestrator_jobs
                    )
                    avg_starting_time = sum(job.get('metrics', {}).get('orchestrator_data', {}).get('total_time_starting', 0) or 0 for job in orchestrator_jobs) / len(orchestrator_jobs)
                    
                    if phase_name == "Asynchronous":
                        print(f"   🚀 Autoscaling Metrics:")
                        print(f"      • Total Requeues: {total_requeues}")
                        print(f"      • Total Suspension Time: {total_suspensions:.1f}ms")
                        print(f"      • Avg Startup Time: {avg_starting_time:.1f}ms")
                        
                        # Node distribution analysis
                        unique_hosts = set(job.get('metrics', {}).get('orchestrator_data', {}).get('execution_host') 
                                         for job in orchestrator_jobs 
                                         if job.get('metrics', {}).get('orchestrator_data', {}).get('execution_host'))
                        if unique_hosts:
                            print(f"      • Nodes Allocated: {len(unique_hosts)}")
                            print(f"      • Jobs per Node: {len(orchestrator_jobs) / len(unique_hosts):.1f}")
                    else:
                        print(f"   📊 Sequential Metrics:")
                        print(f"      • Startup Overhead: {avg_starting_time:.1f}ms")
                        print(f"      • Reliability: {100 - (total_requeues / len(orchestrator_jobs) * 100):.1f}%")
        
        if not orchestrator_data_found:
            print(f"\n⚠️  No workload orchestrator data found in results")
            print(f"   This may indicate:")
            print(f"   • Jobs were not correlated with orchestrator data")
            print(f"   • Orchestrator API was not accessible during execution")
            print(f"   • Job timing correlation failed")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Streamlined Pooled Jobs Comparison Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Execution Modes:
  batch    Run jobs in batches (safer, default behavior)
  all      Submit all jobs simultaneously (requires higher SAS Viya limits)

Examples:
  python streamlined_comparison_runner.py                    # Default: batch mode, size 5
  python streamlined_comparison_runner.py --mode batch --concurrent 10
  python streamlined_comparison_runner.py --mode all        # Submit all 42 jobs at once
  python streamlined_comparison_runner.py --async-only --mode all  # Skip sequential, run all async
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["batch", "all"], 
        default="batch",
        help="Async execution mode: 'batch' for batched execution, 'all' for all-at-once (default: batch)"
    )
    
    parser.add_argument(
        "--concurrent", 
        type=int, 
        default=5,
        help="Maximum concurrent jobs in batch mode (default: 5)"
    )
    
    parser.add_argument(
        "--async-only", 
        action="store_true",
        help="Skip sequential execution and run only asynchronous execution"
    )
    
    parser.add_argument(
        "--sequential-only", 
        action="store_true",
        help="Run only sequential execution (skip asynchronous)"
    )
    
    parser.add_argument(
        "--limit-jobs", 
        type=int,
        help="Limit the number of jobs to process (useful for testing)"
    )
    
    return parser.parse_args()


async def main():
    """Main execution function"""
    args = parse_arguments()
    
    print("🚀 Streamlined Pooled Jobs Comparison Runner")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   Async Mode: {args.mode.upper()}")
    if args.mode == "batch":
        print(f"   Batch Size: {args.concurrent}")
    print(f"   Sequential Only: {args.sequential_only}")
    print(f"   Async Only: {args.async_only}")
    if args.limit_jobs:
        print(f"   Job Limit: {args.limit_jobs}")
    print("=" * 60)
    
    BASE_URL = "https://xarprodviya.ondemand.sas.com"
    
    try:
        # Initialize runner
        print("🔑 Loading access token...")
        access_token = load_access_token()
        runner = StreamlinedComparisonRunner(BASE_URL, access_token)
        
        # Load pooled jobs
        print("📋 Loading pooled job configurations...")
        jobs = runner.load_pooled_jobs()
        
        if not jobs:
            print("❌ No jobs to run. Exiting.")
            return
        
        # Apply job limit if specified
        if args.limit_jobs and args.limit_jobs > 0:
            original_count = len(jobs)
            jobs = jobs[:args.limit_jobs]
            print(f"📋 Limited to {len(jobs)} jobs (from {original_count} total)")
        else:
            print(f"📋 Found {len(jobs)} jobs to execute")
        
        # Run sequential execution (unless skipped)
        if not args.async_only:
            print(f"\n🔄 Phase 1: Sequential Execution")
            sequential_results = runner.run_sequential_jobs(jobs, "SAS Job Execution compute context")
            
            runner.results['sequential_execution'] = sequential_results
            
            if not args.sequential_only:
                print(f"\n⏸️  Pausing 30 seconds between execution phases...")
                time.sleep(30)
        
        # Run asynchronous execution (unless skipped)
        if not args.sequential_only:
            print(f"\n🚀 Phase 2: Asynchronous Execution")
            async_results = await runner.run_async_jobs(
                jobs, 
                "Autoscaling POC Context",
                max_concurrent=args.concurrent,
                execution_mode=args.mode
            )
            
            runner.results['async_execution'] = async_results
        
        # Correlate all orchestrator data at the very end
        # This ensures orchestrator API calls don't affect job timing measurements
        runner.correlate_all_orchestrator_data()
        
        # Save results and print summary
        output_file = runner.save_results()
        runner.print_summary()
        
        print(f"\n✅ Execution complete! Results saved to: {output_file}")
        
        # Print helpful information about the mode used
        if args.mode == "all":
            print(f"\n💡 You used ALL-AT-ONCE mode:")
            print(f"   • All {len(jobs)} jobs were submitted simultaneously")
            print(f"   • This requires higher concurrent job limits in SAS Viya")
            print(f"   • If jobs failed to submit, try --mode batch --concurrent 10")
        else:
            print(f"\n💡 You used BATCH mode:")
            print(f"   • Jobs were processed in batches of {args.concurrent}")
            print(f"   • To try all-at-once: --mode all")
            print(f"   • To increase batch size: --concurrent 10")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
