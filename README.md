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

### Authentication
```powershell
# Get/refresh your SAS Viya access tokens
.\.venv\Scripts\python.exe src\jes\auth.py
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
│   ├── auth.py               # Authentication handling
│   ├── comparison_runner.py  # Main comparison logic
│   └── job_execution.py      # SAS Viya API client
├── config/                    # Configuration files
│   └── job_configs.json      # Job definitions (54 pooled jobs)
├── data/                      # Runtime data
│   └── sas_tokens.json       # Access tokens (auto-generated)
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
- Run `python src\jes\auth.py` to refresh tokens
- Check token expiration in `data/sas_tokens.json`
- Verify environment access permissions

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