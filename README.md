# JES - Job Execution System for SAS Viya

## 🚀 Overview

A streamlined system for comparing job execution performance in SAS Viya between:
- **Sequential execution** in Compute Context (no autoscaling)
- **Asynchronous execution** in Autoscaling POC Context (with autoscaling)

### Key Features
- **Orchestrator Integration**: Automatic correlation with SAS Viya Workload Orchestrator for real metrics
- **Unique Job Naming**: Timestamp-based naming ensures reliable job tracking
- **Comprehensive Metrics**: Queue times, resource usage, execution hosts, and autoscaling behavior
- **Production Ready**: Clean, maintainable code with proper error handling

## 🏃‍♂️ Quick Start

### Prerequisites
1. Python 3.8+ with virtual environment
2. Access to SAS Viya environment
3. Valid authentication tokens in `data/sas_tokens.json`

### Installation
```powershell
# Clone the repository
git clone <repository-url>
cd JES

# Install dependencies
pip install -r requirements.txt
```

### Authentication Setup
The system uses OAuth2 authentication with SAS Viya. Before running jobs, you need to set up authentication:

#### 1. Environment Configuration
```powershell
# Copy the environment template
copy env.template .env

# Edit .env with your SAS Viya credentials:
# - SAS_VIYA_BASE_URL=https://your-viya-server.com
# - SAS_VIYA_CLIENT_ID=your_client_id
# - SAS_VIYA_USERNAME=your_username
# - SAS_VIYA_PASSWORD=your_password
```

#### 2. Get Authentication Tokens
```powershell
# Run the authentication script
.\.venv\Scripts\python.exe src\jes\auth.py
```

**What happens during authentication:**
1. **Browser Authorization**: Opens your browser to SAS Viya login page
2. **Manual Code Capture**: After login, copy the authorization code from the redirect URL
3. **Token Exchange**: System exchanges the code for access and refresh tokens
4. **Auto-Save**: Tokens are saved to `data/sas_tokens.json` for future use

#### 3. Authentication Flow Details
- **OAuth2 Authorization Code Flow**: Industry-standard security protocol
- **Automatic Token Refresh**: System automatically refreshes expired tokens
- **Token Persistence**: Tokens are saved locally and reused across sessions
- **Smart Management**: Only re-authenticates when tokens are invalid or expired

#### 4. Token Management
```powershell
# Tokens are automatically managed, but you can manually refresh if needed:
.\.venv\Scripts\python.exe src\jes\auth.py

# Check token status in data/sas_tokens.json:
# - access_token: Used for API calls (expires in ~24 hours)
# - refresh_token: Used to get new access tokens (expires in ~14 days)
# - expires_at: When access token expires
# - refresh_expires_at: When refresh token expires
```

## 🎮 Running the System

### Basic Usage
```powershell
# Quick test with 1 job (sequential only)
.\.venv\Scripts\python.exe main.py --limit-jobs 1 --sequential-only

# Full comparison (all 54 jobs, both sequential and async)
.\.venv\Scripts\python.exe main.py

# Custom batch size for async execution
.\.venv\Scripts\python.exe main.py --mode batch --concurrent 10
```

### Command Examples

#### **Testing & Development**
```powershell
# Test with a single job (fastest for development)
.\.venv\Scripts\python.exe main.py --limit-jobs 1 --sequential-only

# Test with 3 jobs, sequential only
.\.venv\Scripts\python.exe main.py --limit-jobs 3 --sequential-only

# Test async execution only with 5 jobs
.\.venv\Scripts\python.exe main.py --limit-jobs 5 --async-only

# Test small batch processing
.\.venv\Scripts\python.exe main.py --limit-jobs 10 --mode batch --concurrent 3
```

#### **Production Runs**
```powershell
# Default production run (all jobs, batch mode)
.\.venv\Scripts\python.exe main.py

# Large batch processing (if your SAS Viya supports it)
.\.venv\Scripts\python.exe main.py --mode batch --concurrent 15

# All-at-once mode (requires high SAS Viya concurrent limits)
.\.venv\Scripts\python.exe main.py --mode all

# Sequential execution only (for baseline measurements)
.\.venv\Scripts\python.exe main.py --sequential-only
```

#### **Performance Testing**
```powershell
# Test autoscaling with all jobs at once
.\.venv\Scripts\python.exe main.py --async-only --mode all

# Compare different batch sizes
.\.venv\Scripts\python.exe main.py --mode batch --concurrent 5
.\.venv\Scripts\python.exe main.py --mode batch --concurrent 10
.\.venv\Scripts\python.exe main.py --mode batch --concurrent 20

# Stress test with maximum concurrency
.\.venv\Scripts\python.exe main.py --mode all
```

#### **Troubleshooting**
```powershell
# Debug with minimal jobs
.\.venv\Scripts\python.exe main.py --limit-jobs 1 --sequential-only

# Check orchestrator correlation with 2 jobs
.\.venv\Scripts\python.exe main.py --limit-jobs 2 --sequential-only

# Test async execution only (skip sequential if it's failing)
.\.venv\Scripts\python.exe main.py --limit-jobs 3 --async-only --mode batch --concurrent 1
```

#### **Help & Options**
```powershell
# Show all available options
.\.venv\Scripts\python.exe main.py --help

# Show version and configuration
.\.venv\Scripts\python.exe main.py --limit-jobs 0
```

### Command Options
| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Async execution mode: `batch` or `all` | `batch` |
| `--concurrent` | Batch size for batch mode | `5` |
| `--async-only` | Skip sequential execution | `False` |
| `--sequential-only` | Skip async execution | `False` |
| `--limit-jobs` | Limit number of jobs to process | `None` (all) |

## 🔐 Authentication Architecture

### OAuth2 Authorization Code Flow
The system implements the industry-standard OAuth2 authorization code flow for secure SAS Viya authentication:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Browser   │    │     JES     │    │ SAS Viya    │
│             │    │   System    │    │   Server    │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        │  1. Auth Request  │                   │
        │<──────────────────│                   │
        │                   │                   │
        │  2. User Login    │                   │
        │──────────────────────────────────────>│
        │                   │                   │
        │  3. Auth Code     │                   │
        │<──────────────────────────────────────│
        │                   │                   │
        │  4. Manual Copy   │                   │
        │──────────────────>│                   │
        │                   │                   │
        │                   │  5. Token Exchange│
        │                   │──────────────────>│
        │                   │                   │
        │                   │  6. Access Token  │
        │                   │<──────────────────│
```

### Token Lifecycle Management
- **Access Token**: Valid for ~24 hours, used for all API calls
- **Refresh Token**: Valid for ~14 days, used to get new access tokens
- **Automatic Refresh**: System automatically refreshes tokens 5 minutes before expiry
- **Persistent Storage**: Tokens saved in `data/sas_tokens.json` for session reuse

### Security Features
- **No Password Storage**: Only OAuth2 tokens are stored locally
- **Token Validation**: System tests tokens before use
- **Automatic Cleanup**: Expired tokens are automatically refreshed or re-authenticated
- **Secure Transmission**: All authentication uses HTTPS

## 📊 How It Works

### Job Naming Pattern
The system uses unique timestamps in job names to ensure reliable orchestrator correlation:
```
Submitted: cm_pooled_sequential_1_1757479582
Transformed: cmpooledsequential11757479582
Orchestrator: cmpooledsequential11757479582-b9deca7d-7056-438c-99c5-5df90ee2a3fb
```

### Orchestrator Correlation
1. **Job Submission**: Each job gets a unique name with timestamp
2. **Post-Execution**: System waits 15 seconds for orchestrator registration
3. **Pattern Matching**: Direct prefix match finds the exact orchestrator job
4. **Metrics Extraction**: Comprehensive metrics from orchestrator API

### Metrics Captured
- **Timing**: Queue wait time, execution time, total duration
- **Resources**: CPU cores used, memory consumption
- **Infrastructure**: Actual compute node hostnames
- **Autoscaling**: Scaling events and node allocation

## 📁 Project Structure

```
JES/
├── main.py                    # Main entry point
├── src/jes/                   # Core package
│   ├── auth.py               # OAuth2 authentication & token management
│   ├── comparison_runner.py  # Main comparison logic
│   └── job_execution.py      # SAS Viya API client
├── config/                    # Configuration files
│   └── job_configs.json      # Job definitions (54 pooled jobs)
├── data/                      # Runtime data
│   └── sas_tokens.json       # OAuth2 tokens (auto-generated)
├── .env                       # Environment configuration (create from env.template)
├── demos/                     # Demo scripts
│   └── job_orchestrator_metrics_demo.py
└── results/                   # Output JSON files
```

## 📈 Output Format

Results are saved to `results/pooled_jobs_comparison_YYYYMMDD_HHMMSS.json`:

```json
{
  "metadata": {
    "execution_timestamp": "2025-09-10T...",
    "base_url": "https://xarprodviya.ondemand.sas.com"
  },
  "sequential_execution": {
    "context_name": "SAS Job Execution compute context",
    "total_duration": 123.45,
    "successful_jobs": 54,
    "failed_jobs": 0,
    "jobs": [
      {
        "name": "cm_pooled",
        "unique_job_name": "cm_pooled_sequential_1_1757479582",
        "duration": 22.81,
        "metrics": {
          "orchestrator_data": {
            "execution_host": "aks-compute-36279683-vmss000001",
            "queue_wait_seconds": 0.5,
            "execution_seconds": 22.3,
            "cpu_cores": 0.15,
            "memory_mb": 250
          }
        }
      }
    ]
  },
  "async_execution": {
    "context_name": "Autoscaling POC Context",
    "total_duration": 45.67,
    "successful_jobs": 54,
    "failed_jobs": 0
  },
  "comparison": {
    "time_saved": 77.78,
    "efficiency_gain": 62.5
  }
}
```

## 🔧 Configuration

### Job Configuration (`config/job_configs.json`)
- 54 pooled jobs available
- Enable/disable jobs with `"enabled": true/false`
- Each job has a unique `definition_id` and `name`

### Available Pooled Jobs (Sample)
- `cm_pooled` - Core metrics pooled job
- `ae_pooled` - Adverse events pooled job
- `dm_pooled` - Demographics pooled job
- `ds_pooled` - Disposition pooled job
- ... and 50 more

## 🔍 Troubleshooting

### Common Issues

**No orchestrator data found:**
- Increase wait time in `correlate_all_orchestrator_data()` if needed
- Verify orchestrator API access with proper permissions
- Check that jobs are appearing in `/workloadOrchestrator/jobs`

**Authentication errors:**

*Token expired or invalid:*
```powershell
# Re-run authentication to get fresh tokens
.\.venv\Scripts\python.exe src\jes\auth.py
```

*Browser authorization issues:*
- Ensure you can access your SAS Viya server in the browser
- Check that your username/password are correct in `.env`
- Verify the client_id is properly configured for your SAS Viya instance
- Make sure you're copying the complete authorization code from the redirect URL

*Environment configuration problems:*
```powershell
# Verify your .env file exists and has correct values
type .env

# Check if environment variables are loading properly
.\.venv\Scripts\python.exe -c "from src.jes.config import get_config; print(get_config())"

# If you get import errors, make sure you're in the project root directory
cd C:\Python\JES
```

*Token refresh failures:*
- If refresh token is expired (>14 days), you'll need to re-authenticate completely
- Check `refresh_expires_at` in `data/sas_tokens.json`
- Delete `data/sas_tokens.json` and re-authenticate if tokens are corrupted

*Network/connectivity issues:*
- Verify you can reach your SAS Viya server: `ping your-viya-server.com`
- Check if you're behind a corporate firewall that blocks OAuth flows
- Ensure your SAS Viya server supports the OAuth2 authorization code flow

**Job submission failures:**
- Check compute context availability
- Verify job definition IDs are correct
- Ensure sufficient permissions for job execution

## 🚀 Advanced Features

### Demos
```powershell
# Run orchestrator metrics demo
.\.venv\Scripts\python.exe demos\job_orchestrator_metrics_demo.py
```

### Testing
The main system includes comprehensive testing through:
```powershell
# Test with a single job
.\.venv\Scripts\python.exe main.py --limit-jobs 1 --sequential-only

# Test batch processing
.\.venv\Scripts\python.exe main.py --limit-jobs 5 --mode batch --concurrent 3
```

## 📝 Key Improvements

### Recent Updates
- ✅ **Simplified Correlation**: No longer depends on COMPUTE_JOB IDs
- ✅ **Unique Timestamps**: Guarantees one-to-one job mapping
- ✅ **Efficient Querying**: Sorts by most recent, finds jobs quickly
- ✅ **Clean Codebase**: Removed dead code and temporary files
- ✅ **Post-Execution Correlation**: Doesn't impact job timing measurements

### Performance
- Jobs typically found at position #1 in orchestrator (with proper sorting)
- 100% correlation success rate when jobs exist
- Supports up to 1500 jobs with pagination
- 15-second wait ensures orchestrator registration

## 📊 Visualization and Analysis

### PowerPoint-Ready Slides
Generate presentation-ready slides with embedded analysis:

```powershell
# Generate PowerPoint slides only
.\.venv\Scripts\python.exe create_powerpoint_slides.py

# Generate both standard visualizations and PowerPoint slides
.\.venv\Scripts\python.exe visualize_comparison.py
```

**PowerPoint slides include:**
- **Slide 1: Executive Summary** - High-level overview with key metrics and recommendations
- **Slide 2: Performance Comparison** - Technical analysis with timing breakdowns  
- **Slide 3: Timeline Analysis** - Visual execution patterns with queue wait times
- **Slide 4: Resource Analysis** - Node utilization and autoscaling behavior
- **Slide 5: Business Impact** - ROI analysis and strategic recommendations

All slides use consistent, professional color schemes and include embedded analysis descriptions. Generated slides are saved to `results/powerpoint_slides/` and can be directly inserted into PowerPoint presentations.

### Standard Visualizations
The system also generates comprehensive standard visualizations:
- **Executive Dashboard**: High-level summary for decision makers
- **Execution Time Comparison**: Detailed timing analysis
- **Job Duration Analysis**: Individual job performance
- **Success Rate Comparison**: Reliability metrics
- **Resource Utilization**: CPU/Memory usage patterns
- **Queue Wait Analysis**: Queue performance comparison
- **Timeline Visualization**: Execution patterns over time
- **Node Utilization**: Autoscaling behavior analysis
- **Performance Heatmap**: Metrics correlation
- **Cost-Benefit Analysis**: Business value assessment

### Generate Visualizations
```powershell
# Generate all visualizations from latest comparison data
.\.venv\Scripts\python.exe visualize_comparison.py

# Or specify a specific comparison file
.\.venv\Scripts\python.exe visualize_comparison.py results/pooled_jobs_comparison_20240101_120000.json
```

All visualizations are saved to the `results/` directory with timestamps.

## 📊 Metrics Insights

The system provides comprehensive insights including:
- **Execution Comparison**: Sequential vs parallel performance
- **Resource Utilization**: CPU and memory usage patterns
- **Queue Analysis**: Wait times and queue distribution
- **Autoscaling Behavior**: Node allocation and scaling events
- **Host Distribution**: Actual compute nodes used

## 🎯 Next Steps

1. **Run a test**: `python main.py --limit-jobs 1 --sequential-only`
2. **Review results**: Check `results/` folder for JSON output
3. **Analyze metrics**: Compare sequential vs async performance
4. **Scale up**: Remove `--limit-jobs` to run all 54 jobs
5. **Optimize**: Adjust batch sizes based on your SAS Viya limits

## 📄 License

[Your License Here]

## 👥 Contributors

[Your Team/Contributors Here]