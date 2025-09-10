#!/usr/bin/env python3
"""
SAS Viya Job Execution API Client

This module provides a client for interacting with the SAS Viya Job Execution API.
It allows you to submit jobs, monitor their execution, and retrieve results and metrics.
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SASJobExecutionClient:
    """Client for SAS Viya Job Execution API"""
    
    def __init__(self, base_url: str, access_token: str):
        """
        Initialize the Job Execution client
        
        Args:
            base_url: Base URL of SAS Viya environment
            access_token: Valid access token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.access_token = access_token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # API endpoints
        self.job_execution_base = f"{self.base_url}/jobExecution"
        self.jobs_endpoint = f"{self.job_execution_base}/jobs"
        self.job_requests_endpoint = f"{self.job_execution_base}/jobRequests"
        self.job_definitions_endpoint = f"{self.base_url}/jobDefinitions/definitions"
        self.compute_contexts_endpoint = f"{self.base_url}/compute/contexts"
    
    def discover_api_links(self) -> Dict[str, Any]:
        """
        Discover top-level API links using HATEOAS approach
        
        Returns:
            Dictionary containing available API links
        """
        logger.info("Discovering Job Execution API links...")
        
        try:
            response = self.session.get(
                self.job_execution_base,
                headers={'Accept': 'application/vnd.sas.api+json'}
            )
            response.raise_for_status()
            
            api_info = response.json()
            logger.info("✅ API links discovered successfully")
            return api_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to discover API links: {e}")
            raise
    
    def list_compute_contexts(self, limit: int = 100, start: int = 0) -> Dict[str, Any]:
        """
        List available compute contexts
        
        Args:
            limit: Maximum number of contexts to return
            start: Starting index for pagination
            
        Returns:
            Dictionary containing compute contexts and metadata
        """
        logger.info("Retrieving compute contexts list...")
        
        params = {
            'limit': limit,
            'start': start
        }
        
        try:
            response = self.session.get(
                self.compute_contexts_endpoint,
                params=params,
                headers={'Accept': 'application/vnd.sas.collection+json'}
            )
            response.raise_for_status()
            
            contexts_data = response.json()
            logger.info(f"✅ Retrieved {len(contexts_data.get('items', []))} compute contexts")
            return contexts_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to list compute contexts: {e}")
            raise
    
    def get_compute_context(self, context_id: str) -> Dict[str, Any]:
        """
        Get details of a specific compute context
        
        Args:
            context_id: ID of the compute context
            
        Returns:
            Compute context details
        """
        logger.info(f"Retrieving compute context: {context_id}")
        
        try:
            response = self.session.get(
                f"{self.compute_contexts_endpoint}/{context_id}",
                headers={'Accept': 'application/vnd.sas.compute.context+json'}
            )
            response.raise_for_status()
            
            context = response.json()
            logger.info(f"✅ Retrieved compute context: {context.get('name', 'Unknown')}")
            return context
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get compute context {context_id}: {e}")
            raise
    
    def find_context_by_name(self, context_name: str) -> Optional[Dict[str, Any]]:
        """
        Find a compute context by name
        
        Args:
            context_name: Name of the compute context to find
            
        Returns:
            Compute context details if found, None otherwise
        """
        logger.info(f"Searching for compute context: {context_name}")
        
        try:
            contexts_data = self.list_compute_contexts()
            
            for context in contexts_data.get('items', []):
                if context.get('name', '').lower() == context_name.lower():
                    logger.info(f"✅ Found context: {context['name']} (ID: {context['id']})")
                    return context
            
            logger.warning(f"⚠️ Context '{context_name}' not found")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to search for context '{context_name}': {e}")
            return None
    
    
    def create_job_definition(self, name: str, code: str, job_type: str = "Compute", 
                            parameters: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Create a job definition
        
        Args:
            name: Name of the job definition
            code: SAS code to execute
            job_type: Type of job (default: "Compute")
            parameters: Optional list of parameter definitions
            
        Returns:
            Created job definition
        """
        logger.info(f"Creating job definition: {name}")
        
        if parameters is None:
            parameters = []
        
        job_definition = {
            "version": 1,
            "name": name,
            "type": job_type,
            "parameters": parameters,
            "code": code
        }
        
        try:
            response = self.session.post(
                self.job_definitions_endpoint,
                json=job_definition,
                headers={'Content-Type': 'application/vnd.sas.job.definition+json',
                        'Accept': 'application/vnd.sas.job.definition+json'}
            )
            response.raise_for_status()
            
            created_definition = response.json()
            logger.info(f"✅ Job definition created with ID: {created_definition.get('id')}")
            return created_definition
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to create job definition: {e}")
            raise
    
    def list_job_definitions(self, limit: int = 100, start: int = 0, 
                           filter_text: Optional[str] = None) -> Dict[str, Any]:
        """
        List job definitions available on the server
        
        Args:
            limit: Maximum number of definitions to return
            start: Starting index for pagination
            filter_text: Optional text filter for definition names
            
        Returns:
            Dictionary containing job definitions and metadata
        """
        logger.info("Retrieving job definitions list...")
        
        params = {
            'limit': limit,
            'start': start
        }
        
        if filter_text:
            params['filter'] = f"contains(name,'{filter_text}')"
        
        try:
            response = self.session.get(
                self.job_definitions_endpoint,
                params=params,
                headers={'Accept': 'application/vnd.sas.collection+json'}
            )
            response.raise_for_status()
            
            definitions_data = response.json()
            logger.info(f"✅ Retrieved {len(definitions_data.get('items', []))} job definitions")
            return definitions_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to list job definitions: {e}")
            raise
    
    def get_job_definition(self, definition_id: str) -> Dict[str, Any]:
        """
        Get details of a specific job definition
        
        Args:
            definition_id: ID of the job definition
            
        Returns:
            Job definition details
        """
        logger.info(f"Retrieving job definition: {definition_id}")
        
        try:
            response = self.session.get(
                f"{self.job_definitions_endpoint}/{definition_id}",
                headers={'Accept': 'application/vnd.sas.job.definition+json'}
            )
            response.raise_for_status()
            
            definition = response.json()
            logger.info(f"✅ Retrieved job definition: {definition.get('name', 'Unknown')}")
            return definition
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get job definition {definition_id}: {e}")
            raise
    
    def delete_job_definition(self, definition_id: str) -> bool:
        """
        Delete a job definition
        
        Args:
            definition_id: ID of the job definition to delete
            
        Returns:
            True if successful
        """
        logger.info(f"Deleting job definition: {definition_id}")
        
        try:
            # First check if the definition exists and we have access to it
            try:
                definition = self.get_job_definition(definition_id)
                logger.info(f"Found definition to delete: {definition.get('name', 'Unknown')}")
            except requests.exceptions.RequestException as check_error:
                if hasattr(check_error, 'response') and check_error.response.status_code == 404:
                    logger.warning(f"Job definition {definition_id} not found (may already be deleted)")
                    return True  # Consider it successful if already gone
                elif hasattr(check_error, 'response') and check_error.response.status_code == 401:
                    logger.error(f"❌ Access denied to job definition {definition_id}")
                    logger.error("This may be because:")
                    logger.error("1. The definition was created by another user")
                    logger.error("2. Your token doesn't have sufficient permissions")
                    logger.error("3. The definition is being used by active jobs")
                    raise PermissionError(f"Access denied to delete job definition {definition_id}")
                else:
                    raise
            
            # Attempt to delete
            response = self.session.delete(
                f"{self.job_definitions_endpoint}/{definition_id}",
                headers={'Accept': 'application/json'}
            )
            
            if response.status_code == 204:
                logger.info(f"✅ Job definition {definition_id} deleted successfully")
                return True
            elif response.status_code == 404:
                logger.warning(f"Job definition {definition_id} not found (may already be deleted)")
                return True
            elif response.status_code == 401:
                logger.error(f"❌ Access denied when deleting job definition {definition_id}")
                logger.error("Possible reasons:")
                logger.error("• Definition was created by another user")
                logger.error("• Insufficient permissions on your access token")
                logger.error("• Definition may be in use by active jobs")
                raise PermissionError(f"Access denied to delete job definition {definition_id}")
            elif response.status_code == 409:
                logger.error(f"❌ Cannot delete job definition {definition_id} - it may be in use")
                logger.error("Try waiting for any running jobs using this definition to complete")
                raise ValueError(f"Job definition {definition_id} is in use and cannot be deleted")
            else:
                response.raise_for_status()
            
        except PermissionError:
            raise  # Re-raise permission errors as-is
        except ValueError:
            raise  # Re-raise conflict errors as-is
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to delete job definition {definition_id}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            raise
    
    def execute_job_by_id(self, definition_id: str, name: Optional[str] = None,
                         description: Optional[str] = None, 
                         arguments: Optional[Dict] = None,
                         context_name: Optional[str] = None,
                         wait_for_completion: bool = False,
                         timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a job using an existing job definition ID
        
        Args:
            definition_id: ID of the job definition to execute
            name: Optional name for the job execution
            description: Optional description for the job execution
            arguments: Optional arguments to pass to the job
            context_name: Optional compute context name to use
            wait_for_completion: Whether to wait for job completion
            timeout: Timeout in seconds if waiting for completion (None for no timeout)
            
        Returns:
            Job execution details
        """
        logger.info(f"Executing job using definition ID: {definition_id}")
        
        # Build the job definition URI
        job_definition_uri = f"/jobDefinitions/definitions/{definition_id}"
        
        # Submit the job using the existing submit_job method
        # Context handling is done in submit_job method
        job = self.submit_job(
            job_definition_uri=job_definition_uri,
            name=name or f"Execution of {definition_id}",
            description=description,
            arguments=arguments,
            context_name=context_name
        )
        
        if wait_for_completion:
            logger.info("Waiting for job completion...")
            if timeout is None:
                # No timeout - wait indefinitely with longer poll intervals
                job = self.wait_for_completion_no_timeout(job['id'])
            else:
                job = self.wait_for_completion(job['id'], timeout=timeout)
        
        return job
    
    def submit_job(self, job_definition_uri: Optional[str] = None, 
                   job_definition: Optional[Dict] = None,
                   name: Optional[str] = None,
                   description: Optional[str] = None,
                   arguments: Optional[Dict] = None,
                   context_name: Optional[str] = None,
                   submitter: Optional[str] = None) -> Dict[str, Any]:
        """
        Submit a job for execution
        
        Args:
            job_definition_uri: URI of existing job definition
            job_definition: Embedded job definition
            name: Optional job name
            description: Optional job description  
            arguments: Optional arguments to override job definition parameters
            context_name: Optional compute context name to use
            submitter: Optional submitter identifier
            
        Returns:
            Created job execution object
        """
        logger.info("Submitting job for execution...")
        
        # Build job request
        job_request = {}
        
        if name:
            job_request["name"] = name
        if description:
            job_request["description"] = description
        
        # Handle context selection and ensure we have required context arguments
        job_arguments = arguments.copy() if arguments else {}
        
        # Check if job arguments already have context parameters
        has_context_params = any(key in job_arguments for key in ["_contextName", "_contextId", "_sessionId"])
        
        if context_name and not has_context_params:
            # Find the context by name
            context = self.find_context_by_name(context_name)
            if context:
                logger.info(f"Using compute context: {context['name']} (ID: {context['id']})")
                # Use only _contextId to avoid parameter conflict
                job_arguments["_contextId"] = context['id']
                # Remove any existing _contextName to avoid conflicts
                job_arguments.pop("_contextName", None)
            else:
                logger.warning(f"Context '{context_name}' not found, using default")
        elif context_name and has_context_params:
            logger.warning(f"Job definition already has context parameters, ignoring context_name '{context_name}'")
        
        # Only set default context if no context parameters are present
        if not any(key in job_arguments for key in ["_contextName", "_contextId", "_sessionId"]):
            job_arguments["_contextName"] = "SAS Job Execution compute context"
        
        job_request["arguments"] = job_arguments
        
        # Either use job definition URI or embed definition
        if job_definition_uri:
            job_request["jobDefinitionUri"] = job_definition_uri
        elif job_definition:
            job_request["jobDefinition"] = job_definition
        else:
            raise ValueError("Must provide either job_definition_uri or job_definition")
        
        # Build URL with optional submitter parameter
        url = self.jobs_endpoint
        if submitter:
            url += f"?submitter={submitter}"
        
        try:
            response = self.session.post(
                url,
                json=job_request,
                headers={
                    'Content-Type': 'application/vnd.sas.job.execution.job.request+json',
                    'Accept': 'application/vnd.sas.job.execution.job+json'
                }
            )
            
            logger.info(f"Job submission response status: {response.status_code}")
            
            if response.status_code != 201 and response.status_code != 200:
                logger.error(f"Job submission failed with status {response.status_code}")
                logger.error(f"Response text: {response.text}")
                response.raise_for_status()
            
            job = response.json()
            job_id = job.get('id')
            logger.info(f"✅ Job submitted successfully with ID: {job_id}")
            logger.info(f"Job state: {job.get('state')}")
            
            # Log any immediate errors or warnings
            if job.get('state') == 'failed':
                logger.warning("Job submitted but immediately failed")
                if 'error' in job:
                    logger.error(f"Job error: {job['error']}")
            
            return job
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to submit job: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            raise
    
    def get_job_state(self, job_id: str) -> str:
        """
        Get the current state of a job
        
        Args:
            job_id: Job ID to check
            
        Returns:
            Current job state as string
        """
        try:
            response = self.session.get(
                f"{self.jobs_endpoint}/{job_id}/state",
                headers={'Accept': 'text/plain'}
            )
            response.raise_for_status()
            
            state = response.text.strip()
            return state
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get job state: {e}")
            raise
    
    def get_job_details(self, job_id: str) -> Dict[str, Any]:
        """
        Get complete job details
        
        Args:
            job_id: Job ID to retrieve
            
        Returns:
            Complete job object
        """
        try:
            response = self.session.get(
                f"{self.jobs_endpoint}/{job_id}",
                headers={'Accept': 'application/vnd.sas.job.execution.job+json'}
            )
            response.raise_for_status()
            
            job = response.json()
            return job
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get job details: {e}")
            raise
    
    def wait_for_completion(self, job_id: str, timeout: int = 300, 
                          poll_interval: int = 5) -> Dict[str, Any]:
        """
        Wait for job completion with polling
        
        Args:
            job_id: Job ID to monitor
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks in seconds
            
        Returns:
            Final job object when completed
        """
        logger.info(f"Monitoring job {job_id} for completion...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                state = self.get_job_state(job_id)
                logger.info(f"Job {job_id} state: {state}")
                
                if state in ['completed', 'failed', 'cancelled']:
                    # Get final job details
                    job = self.get_job_details(job_id)
                    
                    if state == 'completed':
                        logger.info(f"✅ Job {job_id} completed successfully")
                    elif state == 'failed':
                        logger.error(f"❌ Job {job_id} failed")
                    else:
                        logger.warning(f"⚠️ Job {job_id} was cancelled")
                    
                    return job
                
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error checking job status: {e}")
                time.sleep(poll_interval)
        
        # Timeout reached
        logger.error(f"❌ Timeout waiting for job {job_id} to complete")
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
    
    def wait_for_completion_no_timeout(self, job_id: str, poll_interval: int = 10) -> Dict[str, Any]:
        """
        Wait for job completion without timeout (for long-running jobs)
        
        Args:
            job_id: Job ID to monitor
            poll_interval: Time between status checks in seconds (default: 10)
            
        Returns:
            Final job object when completed
        """
        logger.info(f"Monitoring job {job_id} for completion (no timeout)...")
        logger.info("This will wait indefinitely until the job completes, fails, or is cancelled")
        
        poll_count = 0
        
        while True:
            try:
                state = self.get_job_state(job_id)
                poll_count += 1
                elapsed_minutes = (poll_count * poll_interval) / 60
                
                # Log status every few polls to show progress
                if poll_count % 6 == 0:  # Every minute with 10s intervals
                    logger.info(f"Job {job_id} state: {state} (running for {elapsed_minutes:.1f} minutes)")
                
                if state in ['completed', 'failed', 'cancelled']:
                    # Get final job details
                    job = self.get_job_details(job_id)
                    
                    if state == 'completed':
                        logger.info(f"✅ Job {job_id} completed successfully after {elapsed_minutes:.1f} minutes")
                    elif state == 'failed':
                        logger.error(f"❌ Job {job_id} failed after {elapsed_minutes:.1f} minutes")
                    else:
                        logger.warning(f"⚠️ Job {job_id} was cancelled after {elapsed_minutes:.1f} minutes")
                    
                    return job
                
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error checking job status: {e}")
                logger.info("Continuing to monitor job...")
                time.sleep(poll_interval)
    
    def get_job_log(self, job: Dict[str, Any]) -> Optional[str]:
        """
        Retrieve job log if available
        
        Args:
            job: Job object containing log location
            
        Returns:
            Log content as string, or None if not available
        """
        log_location = job.get('logLocation')
        if not log_location:
            logger.warning("No log location found in job")
            return None
        
        try:
            # Log location might be relative, so make it absolute if needed
            if log_location.startswith('/'):
                log_url = f"{self.base_url}{log_location}"
            else:
                log_url = log_location
            
            # Try to get the actual log content using /content endpoint
            if '/files/files/' in log_url and not log_url.endswith('/content'):
                log_url = f"{log_url}/content"
            
            response = self.session.get(log_url)
            response.raise_for_status()
            
            return response.text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to retrieve job log: {e}")
            return None
    
    def extract_execution_metrics(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract execution metrics from completed job
        
        Args:
            job: Completed job object
            
        Returns:
            Dictionary containing execution metrics
        """
        metrics = {
            'job_id': job.get('id'),
            'job_name': job.get('jobRequest', {}).get('name', 'Unnamed'),
            'state': job.get('state'),
            'submission_time': job.get('creationTimeStamp'),
            'completion_time': job.get('endTimeStamp'),
            'elapsed_time_ms': job.get('elapsedTime'),
            'submitted_by': job.get('submittedByApplication'),
            'created_by': job.get('createdBy'),
            'job_type': job.get('jobRequest', {}).get('jobDefinition', {}).get('type'),
            'context_name': job.get('jobRequest', {}).get('arguments', {}).get('_contextName'),
            'results_count': len(job.get('results', {})),
            'has_log': bool(job.get('logLocation')),
        }
        
        # Calculate elapsed time in seconds if available
        if metrics['elapsed_time_ms']:
            metrics['elapsed_time_seconds'] = metrics['elapsed_time_ms'] / 1000
        
        # Calculate total execution time if timestamps available
        if metrics['submission_time'] and metrics['completion_time']:
            try:
                start = datetime.fromisoformat(metrics['submission_time'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(metrics['completion_time'].replace('Z', '+00:00'))
                total_duration = end - start
                metrics['total_duration_seconds'] = total_duration.total_seconds()
            except Exception as e:
                logger.warning(f"Could not calculate total duration: {e}")
        
        # Extract result file information
        results = job.get('results', {})
        if results:
            metrics['result_files'] = list(results.keys())
            # No longer extracting COMPUTE_JOB as it's not used for correlation
            metrics['compute_session'] = results.get('COMPUTE_SESSION')
            metrics['compute_context'] = results.get('COMPUTE_CONTEXT')
        
        return metrics
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job (cleanup)
        
        Args:
            job_id: Job ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.delete(f"{self.jobs_endpoint}/{job_id}")
            response.raise_for_status()
            
            logger.info(f"✅ Job {job_id} deleted successfully")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to delete job {job_id}: {e}")
            return False
    
    def list_jobs(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        List all jobs
        
        Args:
            limit: Optional limit on number of jobs to return
            
        Returns:
            Collection of jobs
        """
        try:
            url = self.jobs_endpoint
            params = {}
            if limit:
                params['limit'] = limit
            
            response = self.session.get(
                url,
                params=params,
                headers={'Accept': 'application/vnd.sas.collection+json'}
            )
            response.raise_for_status()
            
            jobs = response.json()
            logger.info(f"Retrieved {len(jobs.get('items', []))} jobs")
            return jobs
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to list jobs: {e}")
            raise

def load_access_token(token_file: str = 'data/sas_tokens.json') -> str:
    """
    Load access token from token file
    
    Args:
        token_file: Path to token file
        
    Returns:
        Access token string
    """
    try:
        with open(token_file, 'r') as f:
            tokens = json.load(f)
        
        access_token = tokens.get('access_token')
        if not access_token:
            raise ValueError("No access token found in token file")
        
        # Check if token is expired
        expires_at = tokens.get('expires_at')
        if expires_at:
            exp_time = datetime.fromisoformat(expires_at)
            if datetime.now() >= exp_time:
                logger.warning("⚠️ Access token appears to be expired")
        
        return access_token
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Token file {token_file} not found. Please run authentication first.")
    except Exception as e:
        raise ValueError(f"Error loading access token: {e}")
