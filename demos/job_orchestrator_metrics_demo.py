#!/usr/bin/env python3
"""
Job Submission and Orchestrator Metrics Demo

This demo script:
1. Submits a job to SAS Viya
2. Waits for job execution to complete
3. Queries the workload orchestrator for job details
4. Captures and displays comprehensive metrics

Author: JES Development Team
Date: 2025-09-09
"""

import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any
import sys

# Configuration
BASE_URL = "https://xarprodviya.ondemand.sas.com"
TOKEN_FILE = Path("../data/sas_tokens.json")

class JobOrchestratorDemo:
    """Demo class for job submission and orchestrator metrics capture."""
    
    def __init__(self):
        """Initialize the demo with authentication."""
        self.base_url = BASE_URL
        self.session = self._setup_session()
        self.job_id = None
        self.orchestrator_job_name = None
        self.orchestrator_job_id = None
        
    def _setup_session(self) -> requests.Session:
        """Setup authenticated session."""
        # Load tokens
        if not TOKEN_FILE.exists():
            print(f"❌ Token file not found: {TOKEN_FILE}")
            sys.exit(1)
            
        with open(TOKEN_FILE, 'r') as f:
            tokens = json.load(f)
            
        access_token = tokens.get('access_token')
        if not access_token:
            print("❌ No access token found in token file")
            sys.exit(1)
            
        # Create session
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {access_token}"
        })
        
        return session
    
    def submit_job(self, job_name: str, definition_id: str) -> Dict[str, Any]:
        """
        Submit a job for execution.
        
        Args:
            job_name: Name of the job
            definition_id: Job definition ID
            
        Returns:
            Job submission response
        """
        print(f"\n📤 Submitting job: {job_name}")
        print(f"   Definition ID: {definition_id}")
        
        url = f"{self.base_url}/jobExecution/jobs"
        
        job_request = {
            "jobDefinitionUri": f"/jobDefinitions/definitions/{definition_id}",
            "name": job_name,
            "description": f"Orchestrator metrics demo - {datetime.now().isoformat()}",
            "arguments": {
                "_contextName": "SAS Job Execution compute context"
            },
            "createdByApplication": "OrchestratorMetricsDemo"
        }
        
        headers = {
            "Content-Type": "application/vnd.sas.job.execution.job.request+json",
            "Accept": "application/vnd.sas.job.execution.job+json"
        }
        
        try:
            response = self.session.post(url, json=job_request, headers=headers)
            response.raise_for_status()
            
            job_data = response.json()
            self.job_id = job_data.get('id')
            
            print(f"✅ Job submitted successfully!")
            print(f"   Job ID: {self.job_id}")
            print(f"   Initial State: {job_data.get('state')}")
            
            return job_data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error submitting job: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"   Response: {e.response.text}")
            raise
    
    def wait_for_completion(self, timeout: int = 300) -> Dict[str, Any]:
        """
        Wait for job to complete execution.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            Final job status
        """
        if not self.job_id:
            raise ValueError("No job ID available. Submit a job first.")
            
        print(f"\n⏳ Waiting for job completion (timeout: {timeout}s)...")
        
        url = f"{self.base_url}/jobExecution/jobs/{self.job_id}"
        headers = {
            "Accept": "application/vnd.sas.job.execution.job+json"
        }
        
        start_time = time.time()
        poll_interval = 2  # Start with 2 second intervals
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(url, headers=headers)
                response.raise_for_status()
                
                job_status = response.json()
                state = job_status.get('state', 'unknown')
                
                # Print status update
                elapsed = int(time.time() - start_time)
                print(f"   [{elapsed:3d}s] State: {state}", end='\r')
                
                if state in ['completed', 'failed', 'canceled', 'error']:
                    print(f"\n✅ Job finished with state: {state}")
                    return job_status
                
                # Adaptive polling - increase interval over time
                if elapsed > 30:
                    poll_interval = min(10, poll_interval + 1)
                    
                time.sleep(poll_interval)
                
            except requests.exceptions.RequestException as e:
                print(f"\n⚠️ Error checking job status: {e}")
                time.sleep(poll_interval)
        
        print(f"\n⚠️ Timeout reached after {timeout} seconds")
        return self.get_job_status()
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get current job status."""
        if not self.job_id:
            raise ValueError("No job ID available")
            
        url = f"{self.base_url}/jobExecution/jobs/{self.job_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def find_orchestrator_job(self, job_name: str, submission_time: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Find the orchestrator job corresponding to our submission.
        Always returns the LATEST matching job to handle multiple submissions.
        
        Args:
            job_name: Original job name
            submission_time: Time when job was submitted (for correlation)
            
        Returns:
            Orchestrator job details or None
        """
        print(f"\n🔍 Searching for orchestrator job...")
        
        # Transform job name to expected orchestrator pattern
        expected_prefix = job_name.replace('_', '').lower()
        
        url = f"{self.base_url}/workloadOrchestrator/jobs"
        params = {
            "limit": 20,  # Get more jobs to find the right one
            "sortBy": "submitTime:descending"  # LATEST FIRST
        }
        headers = {
            "Accept": "application/json"
        }
        
        try:
            response = self.session.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            matching_jobs = []
            
            if 'items' in data:
                for item in data['items']:
                    if 'request' in item and 'name' in item['request']:
                        orch_name = item['request']['name']
                        
                        # Check if this matches our pattern
                        if orch_name.lower().startswith(expected_prefix):
                            # Add submit time for sorting
                            submit_time_str = item.get('processingInfo', {}).get('submitTime')
                            item['_parsed_submit_time'] = None
                            
                            if submit_time_str:
                                try:
                                    # Parse the submit time
                                    item['_parsed_submit_time'] = datetime.fromisoformat(
                                        submit_time_str.replace('Z', '+00:00')
                                    )
                                except:
                                    pass
                            
                            matching_jobs.append(item)
            
            if not matching_jobs:
                print("⚠️ No matching orchestrator jobs found")
                return None
            
            # Sort by submit time (newest first) - double check the sorting
            matching_jobs.sort(
                key=lambda x: x.get('_parsed_submit_time') or datetime.min,
                reverse=True
            )
            
            # Take the LATEST job
            latest_job = matching_jobs[0]
            
            self.orchestrator_job_name = latest_job['request']['name']
            self.orchestrator_job_id = latest_job.get('id')
            
            print(f"✅ Found latest orchestrator job!")
            print(f"   Name: {self.orchestrator_job_name}")
            print(f"   ID: {self.orchestrator_job_id}")
            
            # If we have a submission time, validate it's recent enough
            if submission_time and latest_job.get('_parsed_submit_time'):
                time_diff = latest_job['_parsed_submit_time'] - submission_time
                if abs(time_diff.total_seconds()) > 60:  # More than 60 seconds difference
                    print(f"   ⚠️ Warning: Job submit time differs by {abs(time_diff.total_seconds()):.1f} seconds")
            
            # Show if there were multiple matches
            if len(matching_jobs) > 1:
                print(f"   ℹ️ Found {len(matching_jobs)} matching jobs, selected the latest")
                print(f"   Other matches:")
                for job in matching_jobs[1:4]:  # Show up to 3 other matches
                    print(f"     - {job['request']['name']} (ID: {job.get('id')})")
            
            return latest_job
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error querying orchestrator: {e}")
            return None
    
    def get_orchestrator_metrics(self, orchestrator_job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive metrics from orchestrator job data.
        
        Args:
            orchestrator_job: Orchestrator job details
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n📊 Extracting orchestrator metrics...")
        
        metrics = {
            "job_info": {},
            "timing": {},
            "resources": {},
            "execution": {},
            "status": {}
        }
        
        # Basic job information
        if 'request' in orchestrator_job:
            req = orchestrator_job['request']
            metrics['job_info'] = {
                'name': req.get('name'),
                'queue': req.get('queue'),
                'user': req.get('user'),
                'tenant': req.get('tenant'),
                'type': req.get('type'),
                'project': req.get('project')
            }
        
        # Processing information and timing
        if 'processingInfo' in orchestrator_job:
            proc = orchestrator_job['processingInfo']
            
            metrics['status'] = {
                'state': proc.get('state'),
                'exit_code': proc.get('exitCode'),
                'requeue_count': proc.get('requeueCount', 0)
            }
            
            # Parse timing information
            submit_time = proc.get('submitTime')
            start_time = proc.get('startTime')
            end_time = proc.get('endTime')
            
            metrics['timing'] = {
                'submit_time': submit_time,
                'start_time': start_time,
                'end_time': end_time,
                'total_time_pending': proc.get('totalTimePending', 0),
                'total_time_starting': proc.get('totalTimeStarting', 0),
                'total_time_running': proc.get('totalTimeRunning', 0)
            }
            
            # Calculate durations if times are available
            if submit_time and start_time:
                try:
                    submit_dt = datetime.fromisoformat(submit_time.replace('Z', '+00:00'))
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    metrics['timing']['queue_time_seconds'] = (start_dt - submit_dt).total_seconds()
                except:
                    pass
            
            if start_time and end_time:
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    metrics['timing']['execution_time_seconds'] = (end_dt - start_dt).total_seconds()
                except:
                    pass
            
            # Execution details
            metrics['execution'] = {
                'execution_host': proc.get('executionHost'),
                'execution_ip': proc.get('executionIP'),
                'k8s_object_name': proc.get('k8sObjName'),
                'queue_priority': proc.get('queuePriority')
            }
            
            # Resource consumption
            consumed = proc.get('consumedResources', [])
            for resource in consumed:
                if resource.get('name') == 'cores':
                    metrics['resources']['cores_consumed'] = resource.get('value')
                elif resource.get('name') == 'memory':
                    metrics['resources']['memory_consumed_mb'] = resource.get('value')
        
        return metrics
    
    def display_metrics(self, metrics: Dict[str, Any]):
        """
        Display metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics to display
        """
        print("\n" + "=" * 70)
        print("ORCHESTRATOR METRICS REPORT")
        print("=" * 70)
        
        # Job Information
        print("\n📋 JOB INFORMATION")
        print("-" * 40)
        for key, value in metrics['job_info'].items():
            if value:
                print(f"  {key:15}: {value}")
        
        # Status
        print("\n✅ STATUS")
        print("-" * 40)
        for key, value in metrics['status'].items():
            if value is not None:
                print(f"  {key:15}: {value}")
        
        # Timing Metrics
        print("\n⏱️ TIMING METRICS")
        print("-" * 40)
        timing = metrics['timing']
        
        if timing.get('submit_time'):
            print(f"  Submit Time    : {timing['submit_time']}")
        if timing.get('start_time'):
            print(f"  Start Time     : {timing['start_time']}")
        if timing.get('end_time'):
            print(f"  End Time       : {timing['end_time']}")
        
        if timing.get('queue_time_seconds') is not None:
            print(f"  Queue Time     : {timing['queue_time_seconds']:.2f} seconds")
        if timing.get('execution_time_seconds') is not None:
            print(f"  Execution Time : {timing['execution_time_seconds']:.2f} seconds")
        
        print(f"  Time Pending   : {timing.get('total_time_pending', 0)} ms")
        print(f"  Time Starting  : {timing.get('total_time_starting', 0)} ms")
        print(f"  Time Running   : {timing.get('total_time_running', 0)} ms")
        
        # Resource Metrics
        print("\n💻 RESOURCE CONSUMPTION")
        print("-" * 40)
        if metrics['resources'].get('cores_consumed') is not None:
            print(f"  CPU Cores      : {metrics['resources']['cores_consumed']}")
        if metrics['resources'].get('memory_consumed_mb') is not None:
            print(f"  Memory (MB)    : {metrics['resources']['memory_consumed_mb']}")
        
        # Execution Details
        print("\n🖥️ EXECUTION DETAILS")
        print("-" * 40)
        for key, value in metrics['execution'].items():
            if value:
                print(f"  {key:15}: {value}")
        
        print("\n" + "=" * 70)
    
    def save_metrics(self, metrics: Dict[str, Any], orchestrator_job: Dict[str, Any]):
        """
        Save metrics to a JSON file.
        
        Args:
            metrics: Processed metrics
            orchestrator_job: Raw orchestrator job data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"orchestrator_metrics_{timestamp}.json")
        
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "job_execution_id": self.job_id,
            "orchestrator_job_id": self.orchestrator_job_id,
            "orchestrator_job_name": self.orchestrator_job_name,
            "processed_metrics": metrics,
            "raw_orchestrator_data": orchestrator_job
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Metrics saved to: {output_file}")
    
    def run_demo(self, job_name: str = "cm_pooled", 
                 definition_id: str = "b102ac5e-7d8e-4594-96ed-b68bd7fcc5dd"):
        """
        Run the complete demo.
        
        Args:
            job_name: Name of job to submit
            definition_id: Job definition ID
        """
        print("\n" + "=" * 70)
        print("JOB SUBMISSION AND ORCHESTRATOR METRICS DEMO")
        print("=" * 70)
        
        try:
            # Step 1: Submit the job
            submission_time = datetime.now()
            job_result = self.submit_job(job_name, definition_id)
            
            # Step 2: Wait for completion
            final_status = self.wait_for_completion(timeout=300)
            
            print(f"\n📋 Final Job Status:")
            print(f"   State: {final_status.get('state')}")
            print(f"   Job ID: {final_status.get('id')}")
            
            # Step 3: Find orchestrator job (pass submission time for better correlation)
            time.sleep(2)  # Give orchestrator time to update
            orchestrator_job = self.find_orchestrator_job(job_name, submission_time)
            
            if orchestrator_job:
                # Step 4: Extract metrics
                metrics = self.get_orchestrator_metrics(orchestrator_job)
                
                # Step 5: Display metrics
                self.display_metrics(metrics)
                
                # Step 6: Save metrics
                self.save_metrics(metrics, orchestrator_job)
                
                print("\n✅ Demo completed successfully!")
                
                return metrics
            else:
                print("\n⚠️ Could not retrieve orchestrator metrics")
                print("   The job may still be processing or may have been cleaned up")
                
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            raise


def main():
    """Main function to run the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Job Submission and Orchestrator Metrics Demo"
    )
    parser.add_argument(
        '--job-name',
        default='cm_pooled',
        help='Name of the job to submit (default: cm_pooled)'
    )
    parser.add_argument(
        '--definition-id',
        default='b102ac5e-7d8e-4594-96ed-b68bd7fcc5dd',
        help='Job definition ID'
    )
    parser.add_argument(
        '--config-file',
        default='../config/job_configs.json',
        help='Path to job configurations file'
    )
    parser.add_argument(
        '--use-config',
        action='store_true',
        help='Use job configuration from config file'
    )
    
    args = parser.parse_args()
    
    # If using config file, load job details from there
    if args.use_config:
        config_path = Path(args.config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Find the job in config
            for job in config.get('jobs', []):
                if job['name'] == args.job_name:
                    args.definition_id = job['definition_id']
                    print(f"📋 Loaded job config for: {args.job_name}")
                    break
            else:
                print(f"⚠️ Job '{args.job_name}' not found in config file")
    
    # Run the demo
    demo = JobOrchestratorDemo()
    demo.run_demo(
        job_name=args.job_name,
        definition_id=args.definition_id
    )


if __name__ == "__main__":
    main()
