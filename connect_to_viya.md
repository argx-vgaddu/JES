# SAS Viya Connection and Program Submission

This guide provides a **skinny version** of the SAS Viya connection and job execution code extracted from the JES (Job Execution System) project. This code can be used as a foundation for connecting to SAS Viya and submitting programs in other applications.

## Overview

The code consists of four main components:
1. **Authentication** - OAuth flow with automatic token refresh
2. **Job Execution** - Submit and monitor SAS programs
3. **Configuration** - Environment-based settings
4. **Orchestrator Metrics** - Capture detailed performance data from SAS Viya's workload orchestrator

## Prerequisites

- Python 3.7+
- SAS Viya environment with OAuth client configured
- Required packages: `requests`, `python-dotenv`, `urllib3`

## Quick Start

### 1. Environment Configuration

Create a `.env` file with your SAS Viya credentials:

```bash
# SAS Viya Server Configuration
SAS_BASE_URL=https://your-viya-server.com
SAS_CLIENT_ID=your_oauth_client_id

# User Credentials (keep these secure!)
SAS_USERNAME=your_username@domain.com
SAS_PASSWORD=your_password

# OAuth Settings
SAS_SCOPE=openid

# Token Storage
SAS_TOKEN_FILE=data/sas_tokens.json
```

### 2. Basic Authentication and Job Submission

Here's the complete working example:

```python
#!/usr/bin/env python3
"""
SAS Viya Connection and Job Submission - Skinny Version

This is a minimal implementation for connecting to SAS Viya and submitting programs.
Extracted from the JES project for use in other applications.
"""

import requests
import json
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SASViyaAuth:
    """Minimal SAS Viya authentication with automatic token management"""

    def __init__(self, base_url, client_id, username, password):
        self.base_url = base_url.rstrip('/')
        self.client_id = client_id
        self.username = username
        self.password = password
        self.tokens = {}

        # OAuth endpoints
        self.auth_url = f"{self.base_url}/SASLogon/oauth/authorize"
        self.token_url = f"{self.base_url}/SASLogon/oauth/token"

    def authenticate(self, save_tokens=True):
        """Complete OAuth flow and return access token"""
        try:
            # Get authorization code via browser
            auth_code = self._get_authorization_code()

            # Exchange for tokens
            tokens = self._exchange_code_for_tokens(auth_code)

            # Save tokens if requested
            if save_tokens:
                self._save_tokens()

            return tokens['access_token']

        except Exception as e:
            print(f"Authentication failed: {e}")
            raise

    def _get_authorization_code(self):
        """Get authorization code via browser (manual URL capture)"""
        import urllib.parse
        import webbrowser

        print("🔐 Starting OAuth authorization flow...")

        # Build authorization URL
        auth_params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'scope': 'openid'
        }

        auth_url = f"{self.auth_url}?{urllib.parse.urlencode(auth_params)}"

        print(f"📋 Authorization URL: {auth_url}")
        print("🌐 Opening browser for authorization...")
        webbrowser.open(auth_url)

        print("\n" + "="*60)
        print("📌 MANUAL STEP REQUIRED:")
        print("1. Complete the login process in your browser")
        print("2. After login, you'll be redirected to an error page")
        print("3. Look at the URL in your browser - it will contain 'code=...'")
        print("4. Copy the authorization code from the URL")
        print("="*60 + "\n")

        # Prompt user to enter the authorization code manually
        while True:
            try:
                auth_code = input("🔑 Please paste the authorization code from the URL: ").strip()

                if not auth_code:
                    print("❌ Please enter a valid authorization code")
                    continue

                # Basic validation
                if len(auth_code) < 10:
                    print("❌ Authorization code seems too short. Please check and try again.")
                    continue

                print(f"✅ Authorization code received: {auth_code[:20]}...")
                return auth_code

            except KeyboardInterrupt:
                print("\n❌ Authorization cancelled by user")
                raise
            except Exception as e:
                print(f"❌ Error reading authorization code: {e}")
                continue

    def _exchange_code_for_tokens(self, authorization_code):
        """Exchange authorization code for tokens"""
        print("🔄 Exchanging authorization code for tokens...")

        token_data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'client_id': self.client_id
        }

        auth = (self.username, self.password)

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }

        try:
            response = requests.post(
                self.token_url,
                data=token_data,
                auth=auth,
                headers=headers,
                timeout=30
            )

            print(f"📡 Token request status: {response.status_code}")

            if response.status_code == 200:
                self.tokens = response.json()

                # Add expiration times
                now = datetime.now()
                if 'expires_in' in self.tokens:
                    self.tokens['expires_at'] = (now + timedelta(seconds=self.tokens['expires_in'])).isoformat()
                if 'refresh_expires_in' in self.tokens:
                    self.tokens['refresh_expires_at'] = (now + timedelta(seconds=self.tokens['refresh_expires_in'])).isoformat()

                print("✅ Tokens received successfully!")
                return self.tokens
            else:
                print(f"❌ Token request failed: {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            print(f"❌ Error during token exchange: {e}")
            raise

    def get_valid_access_token(self):
        """Get a valid access token, refreshing if necessary"""
        if not self.tokens:
            return None

        # Check if token is expired
        if 'expires_at' in self.tokens:
            expires_at = datetime.fromisoformat(self.tokens['expires_at'])
            if datetime.now() >= expires_at - timedelta(minutes=5):  # Refresh 5 minutes before expiry
                try:
                    self._refresh_access_token()
                except Exception as e:
                    print(f"⚠️ Failed to refresh token: {e}")
                    return None

        return self.tokens.get('access_token')

    def _refresh_access_token(self):
        """Refresh the access token using refresh token"""
        if not self.tokens or 'refresh_token' not in self.tokens:
            raise ValueError("No refresh token available")

        print("🔄 Refreshing access token...")

        refresh_data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.tokens['refresh_token'],
            'client_id': self.client_id
        }

        auth = (self.username, self.password)

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }

        try:
            response = requests.post(
                self.token_url,
                data=refresh_data,
                auth=auth,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                new_tokens = response.json()

                # Update tokens
                self.tokens.update(new_tokens)

                # Update expiration times
                now = datetime.now()
                if 'expires_in' in new_tokens:
                    self.tokens['expires_at'] = (now + timedelta(seconds=new_tokens['expires_in'])).isoformat()

                print("✅ Access token refreshed successfully!")
                self._save_tokens()  # Save the refreshed tokens to file
                return self.tokens
            else:
                print(f"❌ Token refresh failed: {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            print(f"❌ Error during token refresh: {e}")
            raise

    def _save_tokens(self, filename='data/sas_tokens.json'):
        """Save tokens to file"""
        # Ensure data directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if self.tokens:
            with open(filename, 'w') as f:
                json.dump(self.tokens, f, indent=2)
            print(f"💾 Tokens saved to {filename}")

    def load_tokens(self, filename='data/sas_tokens.json'):
        """Load tokens from file"""
        try:
            with open(filename, 'r') as f:
                self.tokens = json.load(f)
            print(f"📂 Tokens loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"📂 No saved tokens found at {filename}")
            return False
        except Exception as e:
            print(f"❌ Error loading tokens: {e}")
            return False


class SASJobExecutionClient:
    """Minimal client for SAS Viya Job Execution API"""

    def __init__(self, base_url: str, access_token: str):
        """Initialize the Job Execution client"""
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

    def create_job_definition(self, name: str, code: str, job_type: str = "Compute",
                            parameters: list = None) -> dict:
        """Create a job definition"""
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
                headers={
                    'Content-Type': 'application/vnd.sas.job.definition+json',
                    'Accept': 'application/vnd.sas.job.definition+json'
                }
            )
            response.raise_for_status()

            created_definition = response.json()
            print(f"✅ Job definition created with ID: {created_definition.get('id')}")
            return created_definition

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to create job definition: {e}")
            raise

    def submit_job(self, job_definition_uri: str = None, job_definition: dict = None,
                   name: str = None, arguments: dict = None) -> dict:
        """Submit a job for execution"""
        # Build job request
        job_request = {}

        if name:
            job_request["name"] = name

        if arguments:
            job_request["arguments"] = arguments

        # Either use job definition URI or embed definition
        if job_definition_uri:
            job_request["jobDefinitionUri"] = job_definition_uri
        elif job_definition:
            job_request["jobDefinition"] = job_definition
        else:
            raise ValueError("Must provide either job_definition_uri or job_definition")

        try:
            response = self.session.post(
                self.jobs_endpoint,
                json=job_request,
                headers={
                    'Content-Type': 'application/vnd.sas.job.execution.job.request+json',
                    'Accept': 'application/vnd.sas.job.execution.job+json'
                }
            )

            if response.status_code not in [201, 200]:
                print(f"Job submission failed with status {response.status_code}")
                print(f"Response text: {response.text}")
                response.raise_for_status()

            job = response.json()
            job_id = job.get('id')
            print(f"✅ Job submitted successfully with ID: {job_id}")
            print(f"Job state: {job.get('state')}")

            return job

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to submit job: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            raise

    def wait_for_completion(self, job_id: str, timeout: int = 300, poll_interval: int = 5) -> dict:
        """Wait for job completion with polling"""
        print(f"⏳ Monitoring job {job_id} for completion...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                state = self.get_job_state(job_id)
                print(f"📊 Job {job_id} state: {state}")

                if state in ['completed', 'failed', 'cancelled']:
                    # Get final job details
                    job = self.get_job_details(job_id)

                    if state == 'completed':
                        print(f"✅ Job {job_id} completed successfully")
                    elif state == 'failed':
                        print(f"❌ Job {job_id} failed")
                    else:
                        print(f"⚠️ Job {job_id} was cancelled")

                    return job

                time.sleep(poll_interval)

            except Exception as e:
                print(f"⚠️ Error checking job status: {e}")
                time.sleep(poll_interval)

        # Timeout reached
        print(f"❌ Timeout waiting for job {job_id} to complete")
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

    def get_job_state(self, job_id: str) -> str:
        """Get the current state of a job"""
        try:
            response = self.session.get(
                f"{self.jobs_endpoint}/{job_id}/state",
                headers={'Accept': 'text/plain'}
            )
            response.raise_for_status()

            state = response.text.strip()
            return state

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get job state: {e}")
            raise

    def get_job_details(self, job_id: str) -> dict:
        """Get complete job details"""
        try:
            response = self.session.get(
                f"{self.jobs_endpoint}/{job_id}",
                headers={'Accept': 'application/vnd.sas.job.execution.job+json'}
            )
            response.raise_for_status()

            job = response.json()
            return job

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get job details: {e}")
            raise


def get_config():
    """Get SAS configuration from environment variables"""
    return {
        "base_url": os.getenv("SAS_BASE_URL", "https://xarprodviya.ondemand.sas.com"),
        "client_id": os.getenv("SAS_CLIENT_ID", "xar_api"),
        "username": os.getenv("SAS_USERNAME"),
        "password": os.getenv("SAS_PASSWORD"),
        "scope": os.getenv("SAS_SCOPE", "openid"),
        "token_file": os.getenv("SAS_TOKEN_FILE", "data/sas_tokens.json")
    }


def get_sas_tokens():
    """Get SAS tokens with automatic refresh - main entry point"""
    config = get_config()

    auth_client = SASViyaAuth(
        base_url=config["base_url"],
        client_id=config["client_id"],
        username=config["username"],
        password=config["password"]
    )

    # Try to load existing tokens first
    if auth_client.load_tokens():
        access_token = auth_client.get_valid_access_token()
        if access_token:
            return auth_client.tokens

    # If no valid tokens, start fresh authentication
    return auth_client.authenticate()


# Example usage
if __name__ == "__main__":
    print("🚀 SAS Viya Connection Example")
    print("=" * 40)

    try:
        # Get authenticated session
        tokens = get_sas_tokens()
        access_token = tokens['access_token']

        print(f"✅ Authentication successful!")
        print(f"Access Token: {access_token[:50]}...")

        # Create job execution client
        config = get_config()
        client = SASJobExecutionClient(config["base_url"], access_token)

        # Example SAS program
        sas_program = """
        %put RUNNING ON HOST: &SYSHOSTNAME;
        %put Current Date: %sysfunc(date(), yymmdd10.);
        %put Current Time: %sysfunc(time(), time8.);

        data test;
            do i = 1 to 1000000;
                output;
            end;
        run;

        proc means data=test;
            var i;
        run;

        %put Job completed successfully!;
        """

        # Create job definition
        job_def = client.create_job_definition(
            name="Example SAS Program",
            code=sas_program,
            job_type="Compute"
        )

        # Submit job
        job = client.submit_job(
            job_definition_uri=f"/jobDefinitions/definitions/{job_def['id']}",
            name="Test SAS Job"
        )

        # Wait for completion
        final_job = client.wait_for_completion(job['id'])

        print("
📊 Job Results:"        print(f"Job ID: {final_job['id']}")
        print(f"Final State: {final_job['state']}")
        print(f"Elapsed Time: {final_job.get('elapsedTime', 'N/A')} ms")

        if final_job.get('results'):
            print(f"Results: {list(final_job['results'].keys())}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
```

## Detailed Usage Guide

### 1. Authentication Flow

The authentication uses the OAuth 2.0 Authorization Code flow:

1. **Authorization URL Generation**: Creates a URL for user authentication
2. **Manual Browser Step**: User completes login in browser
3. **Code Exchange**: Exchanges authorization code for access/refresh tokens
4. **Token Management**: Automatically refreshes tokens before expiry
5. **Token Persistence**: Saves tokens to file for reuse

### 2. Job Execution

The job execution follows this pattern:

1. **Create Job Definition**: Define the SAS program and parameters
2. **Submit Job**: Send job to SAS Viya for execution
3. **Monitor Progress**: Poll for job status updates
4. **Retrieve Results**: Get final job details and outputs

### 3. Configuration Management

Configuration is handled through environment variables:

- `SAS_BASE_URL`: Your SAS Viya server URL
- `SAS_CLIENT_ID`: OAuth client identifier
- `SAS_USERNAME`: Your username
- `SAS_PASSWORD`: Your password
- `SAS_SCOPE`: OAuth scope (default: `openid`)

## Advanced Usage

### Custom Job Arguments

You can pass custom arguments to your SAS programs:

```python
# Submit job with custom arguments
job = client.submit_job(
    job_definition_uri=f"/jobDefinitions/definitions/{job_def['id']}",
    name="Custom Arguments Job",
    arguments={
        "_contextName": "SAS Job Execution compute context",
        "MY_PARAM": "custom_value"
    }
)
```

### Compute Context Selection

Specify which compute context to use:

```python
# Use specific compute context
job = client.submit_job(
    job_definition_uri=f"/jobDefinitions/definitions/{job_def['id']}",
    arguments={
        "_contextName": "Your Custom Context Name"
    }
)
```

### Error Handling

The code includes comprehensive error handling:

```python
try:
    job = client.submit_job(...)
    final_job = client.wait_for_completion(job['id'])
except requests.exceptions.RequestException as e:
    print(f"Job execution failed: {e}")
    # Handle specific error cases
```

## Integration Tips

1. **Environment Variables**: Use a `.env` file for configuration in development
2. **Token Persistence**: Tokens are automatically saved and reused
3. **Error Logging**: Add logging for production use
4. **Timeout Handling**: Adjust timeouts based on expected job duration
5. **Batch Processing**: Submit multiple jobs concurrently for better performance

## Security Notes

- Store credentials securely (environment variables, key vault, etc.)
- Use HTTPS for all SAS Viya connections
- Rotate tokens regularly
- Implement proper access controls for your application

## Troubleshooting

### Common Issues

1. **Authentication Failed**: Check credentials and OAuth client configuration
2. **Job Submission Failed**: Verify job definition syntax and permissions
3. **Timeout Errors**: Increase timeout values for long-running jobs
4. **Token Expired**: Tokens auto-refresh, but check system clock

### Debug Information

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## Orchestrator Metrics Capture

For comparing performance between Base SAS and CASL approaches, you need detailed metrics from SAS Viya's workload orchestrator. This provides insights into queue wait times, resource allocation, node usage, and execution efficiency.

### Key Orchestrator Metrics

The orchestrator provides critical performance data:

- **Timing Metrics**: Queue wait time, execution time, total job duration
- **Resource Allocation**: CPU cores, memory usage, node assignments
- **Queue Information**: Which queue the job ran in, priority levels
- **Node Information**: Which nodes were used, execution host details
- **Suspension/Requeue Data**: Jobs that were suspended or requeued

### Complete Orchestrator Metrics Client

Here's the enhanced job execution client with orchestrator metrics capture:

```python
class SASJobExecutionClientWithOrchestrator:
    """Enhanced client for SAS Viya Job Execution with orchestrator metrics"""

    def __init__(self, base_url: str, access_token: str):
        """Initialize the Job Execution client with orchestrator support"""
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
        self.orchestrator_endpoint = f"{self.base_url}/workloadOrchestrator/jobs"

    def get_workload_orchestrator_data(self, job_details: dict) -> dict:
        """
        Get comprehensive workload orchestrator data for accurate job analysis.
        Uses job naming patterns to correlate with orchestrator jobs.

        Args:
            job_details: Job details from SAS Job Execution API

        Returns:
            Comprehensive workload orchestrator data with timing, resources, and node info
        """
        import requests
        import time as time_module
        from datetime import datetime

        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }

            # Get job identifiers
            job_name = job_details.get('name', '')  # This is the unique job name
            job_creation_time = job_details.get('creationTimeStamp')

            # Transform job name to match orchestrator naming pattern
            # Example: "cm_pooled_sequential_1_1757476678" becomes "cmpooledsequential11757476678"
            expected_prefix = None
            if '_pooled' in job_name:
                if any(pattern in job_name for pattern in ['_sequential_', '_async_reg_', '_async_auto_', '_async_']):
                    # Unique job name with timestamp
                    transformed = job_name.replace('_', '').lower()
                    expected_prefix = transformed
                else:
                    # Regular pooled job name
                    base_name = job_name.split('_')[0] + '_pooled'
                    expected_prefix = base_name.replace('_', '').lower()

            if not expected_prefix:
                print(f"⚠️ No naming pattern found for job: {job_name}")
                return None

            # Add delay to allow orchestrator data to be available
            time_module.sleep(2)

            # Retry logic for high concurrency scenarios
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Get recent workload orchestrator jobs
                    list_url = self.orchestrator_endpoint
                    list_params = {
                        'limit': 500,
                        'start': 0,
                        'sortBy': 'submitTime:descending'
                    }

                    response = requests.get(list_url, headers=headers, params=list_params, timeout=30)
                    if response.status_code != 200:
                        if attempt < max_retries - 1:
                            time_module.sleep(2 * (attempt + 1))
                            continue
                        return None

                    jobs_list = response.json()
                    break

                except requests.exceptions.Timeout as e:
                    if attempt < max_retries - 1:
                        time_module.sleep(2 * (attempt + 1))
                        continue
                    return None
                except Exception as e:
                    if attempt < max_retries - 1:
                        time_module.sleep(2 * (attempt + 1))
                        continue
                    return None

            # Find matching workload job by naming pattern
            if expected_prefix:
                # Search through orchestrator jobs
                for workload_job in jobs_list.get('items', []):
                    workload_name = workload_job.get('request', {}).get('name', '')

                    # Check if this matches our naming pattern
                    if workload_name.lower().startswith(expected_prefix):
                        print(f"✅ Found orchestrator match: {workload_name}")
                        return self.extract_comprehensive_orchestrator_metrics(workload_job)

            return None

        except Exception as e:
            print(f"⚠️ Could not fetch workload orchestrator data: {e}")
            return None

    def extract_comprehensive_orchestrator_metrics(self, workload_job: dict) -> dict:
        """
        Extract comprehensive metrics from workload orchestrator job data

        Args:
            workload_job: Complete workload orchestrator job object

        Returns:
            Comprehensive metrics including timing, resources, and node information
        """
        from datetime import datetime
        import math

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
            'total_time_pending': processing_info.get('totalTimePending'),  # Real queue wait time
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
            'orchestrator_fetch_time': datetime.now().isoformat()
        }

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

    def execute_job_with_orchestrator_metrics(self, job_definition_uri: str = None,
                                           job_definition: dict = None,
                                           name: str = None,
                                           arguments: dict = None,
                                           wait_for_completion: bool = True,
                                           timeout: int = 300) -> dict:
        """
        Execute a job and capture both Job Execution and Orchestrator metrics

        Args:
            job_definition_uri: URI of existing job definition
            job_definition: Embedded job definition
            name: Optional job name
            arguments: Optional arguments to pass to the job
            wait_for_completion: Whether to wait for job completion
            timeout: Timeout in seconds if waiting for completion

        Returns:
            Enhanced job object with orchestrator metrics
        """
        print(f"🚀 Submitting job: {name}")

        # Submit the job
        job = self.submit_job(
            job_definition_uri=job_definition_uri,
            job_definition=job_definition,
            name=name,
            arguments=arguments
        )

        job_id = job.get('id')
        print(f"✅ Job submitted with ID: {job_id}")

        if wait_for_completion:
            print("⏳ Waiting for job completion...")
            final_job = self.wait_for_completion(job_id, timeout=timeout)

            # Get orchestrator metrics
            print("📊 Fetching orchestrator metrics...")
            orchestrator_data = self.get_workload_orchestrator_data(final_job)

            if orchestrator_data:
                # Add orchestrator data to job metrics
                if 'metrics' not in final_job:
                    final_job['metrics'] = {}

                final_job['metrics']['orchestrator_data'] = orchestrator_data

                # Add orchestrator timing to job metrics
                final_job['metrics']['orchestrator_timing'] = {
                    'queue_wait_seconds': orchestrator_data.get('queue_wait_seconds'),
                    'execution_seconds': orchestrator_data.get('execution_seconds'),
                    'total_seconds': orchestrator_data.get('total_seconds')
                }

                # Add resource utilization
                final_job['metrics']['orchestrator_resources'] = {
                    'cpu_cores': orchestrator_data.get('cpu_cores'),
                    'memory_mb': orchestrator_data.get('memory_mb'),
                    'execution_host': orchestrator_data.get('execution_host')
                }

                # Add context identification
                final_job['metrics']['orchestrator_context'] = {
                    'queue_name': orchestrator_data.get('queue_name'),
                    'context_type': orchestrator_data.get('context_type')
                }

                print(f"✅ Orchestrator metrics captured for job {job_id}")
            else:
                print(f"⚠️ No orchestrator data found for job {job_id}")

            return final_job
        else:
            return job

    # Include all the basic job execution methods from the original client
    def submit_job(self, job_definition_uri: str = None, job_definition: dict = None,
                   name: str = None, arguments: dict = None) -> dict:
        """Submit a job for execution (same as original client)"""
        # Build job request
        job_request = {}

        if name:
            job_request["name"] = name

        if arguments:
            job_request["arguments"] = arguments

        # Either use job definition URI or embed definition
        if job_definition_uri:
            job_request["jobDefinitionUri"] = job_definition_uri
        elif job_definition:
            job_request["jobDefinition"] = job_definition
        else:
            raise ValueError("Must provide either job_definition_uri or job_definition")

        try:
            response = self.session.post(
                self.jobs_endpoint,
                json=job_request,
                headers={
                    'Content-Type': 'application/vnd.sas.job.execution.job.request+json',
                    'Accept': 'application/vnd.sas.job.execution.job+json'
                }
            )

            if response.status_code not in [201, 200]:
                print(f"Job submission failed with status {response.status_code}")
                print(f"Response text: {response.text}")
                response.raise_for_status()

            job = response.json()
            job_id = job.get('id')
            print(f"✅ Job submitted successfully with ID: {job_id}")
            print(f"Job state: {job.get('state')}")

            return job

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to submit job: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            raise

    def wait_for_completion(self, job_id: str, timeout: int = 300, poll_interval: int = 5) -> dict:
        """Wait for job completion with polling (same as original client)"""
        print(f"⏳ Monitoring job {job_id} for completion...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                state = self.get_job_state(job_id)
                print(f"📊 Job {job_id} state: {state}")

                if state in ['completed', 'failed', 'cancelled']:
                    # Get final job details
                    job = self.get_job_details(job_id)

                    if state == 'completed':
                        print(f"✅ Job {job_id} completed successfully")
                    elif state == 'failed':
                        print(f"❌ Job {job_id} failed")
                    else:
                        print(f"⚠️ Job {job_id} was cancelled")

                    return job

                time.sleep(poll_interval)

            except Exception as e:
                print(f"⚠️ Error checking job status: {e}")
                time.sleep(poll_interval)

        # Timeout reached
        print(f"❌ Timeout waiting for job {job_id} to complete")
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

    def get_job_state(self, job_id: str) -> str:
        """Get the current state of a job"""
        try:
            response = self.session.get(
                f"{self.jobs_endpoint}/{job_id}/state",
                headers={'Accept': 'text/plain'}
            )
            response.raise_for_status()

            state = response.text.strip()
            return state

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get job state: {e}")
            raise

    def get_job_details(self, job_id: str) -> dict:
        """Get complete job details"""
        try:
            response = self.session.get(
                f"{self.jobs_endpoint}/{job_id}",
                headers={'Accept': 'application/vnd.sas.job.execution.job+json'}
            )
            response.raise_for_status()

            job = response.json()
            return job

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get job details: {e}")
            raise
```

### Usage Example with Orchestrator Metrics

```python
#!/usr/bin/env python3
"""
Example of using the enhanced client with orchestrator metrics
"""

# Get authenticated session
tokens = get_sas_tokens()
access_token = tokens['access_token']

# Create enhanced client
config = get_config()
client = SASJobExecutionClientWithOrchestrator(config["base_url"], access_token)

# Example SAS program for comparison
sas_program = """
%put RUNNING ON HOST: &SYSHOSTNAME;
%put Current Date: %sysfunc(date(), yymmdd10.);
%put Current Time: %sysfunc(time(), time8.);

data test;
    do i = 1 to 1000000;
        output;
    end;
run;

proc means data=test;
    var i;
run;

%put Job completed successfully!;
"""

# Create job definition
job_def = client.create_job_definition(
    name="Performance Comparison SAS Program",
    code=sas_program,
    job_type="Compute"
)

# Execute job with orchestrator metrics capture
final_job = client.execute_job_with_orchestrator_metrics(
    job_definition_uri=f"/jobDefinitions/definitions/{job_def['id']}",
    name="perf_test_sequential_1_1757476678",  # Unique name for orchestrator correlation
    wait_for_completion=True,
    timeout=600
)

# Extract metrics for comparison
orchestrator_data = final_job['metrics']['orchestrator_data']
job_metrics = final_job['metrics']['orchestrator_timing']

print("
📊 PERFORMANCE METRICS FOR COMPARISON:"print(f"Queue Wait Time: {orchestrator_data.get('queue_wait_seconds', 'N/A')} seconds")
print(f"Execution Time: {orchestrator_data.get('execution_seconds', 'N/A')} seconds")
print(f"Total Time: {orchestrator_data.get('total_seconds', 'N/A')} seconds")
print(f"CPU Cores Used: {orchestrator_data.get('cpu_cores', 'N/A')}")
print(f"Memory Used: {orchestrator_data.get('memory_mb', 'N/A')} MB")
print(f"Execution Host: {orchestrator_data.get('execution_host', 'N/A')}")
print(f"Queue Type: {orchestrator_data.get('queue_name', 'N/A')}")

# Calculate efficiency metrics
if orchestrator_data.get('total_seconds') and orchestrator_data.get('execution_seconds'):
    queue_overhead_percent = (orchestrator_data['queue_wait_seconds'] / orchestrator_data['total_seconds']) * 100
    print(f"Queue Overhead: {queue_overhead_percent:.1f}%")

# Compare with Base SAS/CASL metrics here
```

### Key Performance Comparison Metrics

When comparing Base SAS and CASL approaches, focus on these metrics:

1. **Queue Wait Time** (`queue_wait_seconds`): Time spent waiting in queue before execution
2. **Execution Time** (`execution_seconds`): Actual processing time
3. **Total Time** (`total_seconds`): End-to-end duration
4. **Resource Utilization**: CPU cores and memory usage
5. **Node Allocation**: Which nodes were used for execution
6. **Queue Type**: Standard vs. autoscaling queues

### Performance Analysis

Use the captured metrics to compare approaches:

```python
def analyze_performance_metrics(orchestrator_data):
    """Analyze orchestrator metrics for performance comparison"""

    metrics = {
        'queue_efficiency': 0,
        'resource_efficiency': 0,
        'overall_efficiency': 0,
        'autoscaling_benefit': 0
    }

    # Calculate queue efficiency
    if orchestrator_data.get('total_seconds') and orchestrator_data.get('execution_seconds'):
        total_time = orchestrator_data['total_seconds']
        execution_time = orchestrator_data['execution_seconds']
        queue_time = orchestrator_data.get('queue_wait_seconds', 0)

        metrics['queue_efficiency'] = (execution_time / total_time) * 100
        metrics['resource_efficiency'] = (orchestrator_data.get('cpu_cores', 1) * execution_time) / total_time

    # Determine if autoscaling was beneficial
    if orchestrator_data.get('context_type') == 'asynchronous':
        # Compare with synchronous execution metrics
        metrics['autoscaling_benefit'] = 1  # Calculate based on your comparison data

    return metrics

# Use this for your comparison analysis
performance_metrics = analyze_performance_metrics(orchestrator_data)
```

## Comparison with Base SAS and CASL

This Python-based approach with orchestrator metrics provides several advantages over traditional Base SAS and CASL:

### Base SAS Comparison
- **Python Integration**: Easy integration with modern data pipelines
- **Asynchronous Processing**: Non-blocking job submission and monitoring
- **Rich API**: RESTful API with comprehensive job management
- **Token Management**: Automatic token refresh and persistence

### CASL Comparison
- **Programmatic Control**: Full programmatic control over job lifecycle
- **Scalability**: Better handling of multiple concurrent jobs
- **Monitoring**: Real-time job status and progress tracking
- **Integration**: Seamless integration with Python-based workflows
- **Detailed Metrics**: Comprehensive orchestrator data for performance analysis
- **Queue Analysis**: Insights into queue wait times and resource allocation
- **Node Tracking**: Detailed information about which nodes executed jobs

### Orchestrator Metrics for Performance Comparison

The orchestrator metrics provide critical data for comparing Base SAS and CASL:

**Timing Analysis:**
- Base SAS: Manual timing, limited queue visibility
- CASL: Automatic queue time tracking, detailed execution phases
- **Orchestrator Advantage**: Real queue wait times, startup overhead, suspension data

**Resource Efficiency:**
- Base SAS: Limited resource tracking
- CASL: Basic resource monitoring
- **Orchestrator Advantage**: Detailed CPU/memory usage, node allocation patterns

**Scalability Insights:**
- Base SAS: Single-threaded execution
- CASL: Multi-threaded processing
- **Orchestrator Advantage**: Node distribution analysis, autoscaling effectiveness

**Queue Management:**
- Base SAS: No queue management
- CASL: Basic queue handling
- **Orchestrator Advantage**: Queue type identification, priority analysis, requeue tracking

## Performance Comparison Framework

Use the orchestrator metrics to create comprehensive performance comparisons:

```python
def create_performance_comparison(orchestrator_data_list):
    """
    Create a comprehensive performance comparison using orchestrator metrics

    Args:
        orchestrator_data_list: List of orchestrator data from different executions

    Returns:
        Dictionary with comparison metrics
    """

    comparison = {
        'queue_efficiency_comparison': {},
        'resource_utilization_comparison': {},
        'autoscaling_analysis': {},
        'execution_patterns': {}
    }

    # Group data by approach (Base SAS, CASL, Python with orchestrator)
    approaches = {}
    for data in orchestrator_data_list:
        approach = data.get('approach_type', 'unknown')
        if approach not in approaches:
            approaches[approach] = []
        approaches[approach].append(data)

    # Calculate average metrics for each approach
    for approach, data_list in approaches.items():
        avg_queue_wait = sum(d.get('queue_wait_seconds', 0) for d in data_list) / len(data_list)
        avg_execution_time = sum(d.get('execution_seconds', 0) for d in data_list) / len(data_list)
        avg_total_time = sum(d.get('total_seconds', 0) for d in data_list) / len(data_list)

        comparison['queue_efficiency_comparison'][approach] = {
            'avg_queue_wait_seconds': avg_queue_wait,
            'avg_execution_seconds': avg_execution_time,
            'avg_total_seconds': avg_total_time,
            'queue_efficiency_percent': (avg_execution_time / avg_total_time) * 100 if avg_total_time > 0 else 0
        }

    return comparison

# Example usage
orchestrator_metrics = [
    {'approach_type': 'base_sas', 'queue_wait_seconds': 5, 'execution_seconds': 120, 'total_seconds': 125},
    {'approach_type': 'casl', 'queue_wait_seconds': 2, 'execution_seconds': 110, 'total_seconds': 112},
    {'approach_type': 'python_orchestrator', 'queue_wait_seconds': 1, 'execution_seconds': 108, 'total_seconds': 109}
]

performance_comparison = create_performance_comparison(orchestrator_metrics)
```

This skinny version maintains all the essential functionality while being easily adaptable for use in other applications that need to compare SAS Viya capabilities with Base SAS and CASL implementations. The orchestrator metrics capture provides the foundation for comprehensive performance analysis and comparison.
