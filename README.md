# JES - Job Execution System for SAS Viya

## 🚀 Overview

An advanced system for **three-way performance comparison** of job execution in SAS Viya:

1. **Sequential Execution** (Baseline) - Jobs run one at a time in regular compute context
2. **Async Regular** (Parallelization) - Jobs run in parallel in regular compute context  
3. **Async Autoscaling** (Parallelization + Autoscaling) - Jobs run in parallel in autoscaling context

### Key Features
- **🧮 Resource-Adjusted Analysis**: Accounts for node count differences to isolate true autoscaling benefits
- **📊 Three-Way Comparison**: Separates parallelization gains from autoscaling gains
- **🔗 Orchestrator Integration**: Automatic correlation with SAS Viya Workload Orchestrator for real metrics
- **🏷️ Smart Job Naming**: Context-aware naming (async_reg vs async_auto) ensures reliable tracking
- **📈 Advanced Visualizations**: Timeline analysis, efficiency charts, and executive dashboards
- **🎯 Analytical Insights**: Automated efficiency analysis and performance recommendations

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
# Quick test with 3 jobs (all three execution modes)
python -m src.jes.comparison_runner --limit-jobs 3

# Full three-way comparison (all 54 jobs)
python -m src.jes.comparison_runner

# Test only autoscaling vs sequential (skip async regular)
python -m src.jes.comparison_runner --skip-async-regular --limit-jobs 5

# Custom batch size for async execution
python -m src.jes.comparison_runner --mode batch --concurrent 10
```

### Command Examples

#### **Testing & Development**
```powershell
# Test with minimal jobs (fastest for development)
python -m src.jes.comparison_runner --limit-jobs 2

# Test only sequential (baseline measurement)
python -m src.jes.comparison_runner --sequential-only --limit-jobs 3

# Test only async regular (parallelization without autoscaling)
python -m src.jes.comparison_runner --async-regular-only --limit-jobs 3

# Test only autoscaling (parallelization + autoscaling)
python -m src.jes.comparison_runner --async-autoscaling-only --limit-jobs 3

# Compare just async modes (skip sequential)
python -m src.jes.comparison_runner --skip-sequential --limit-jobs 5
```

#### **Production Runs**
```powershell
# Full three-way comparison (all jobs, batch mode)
python -m src.jes.comparison_runner

# Large batch processing (if your SAS Viya supports it)
python -m src.jes.comparison_runner --mode batch --concurrent 15

# All-at-once mode (requires high SAS Viya concurrent limits)
python -m src.jes.comparison_runner --mode all

# Skip async regular to focus on sequential vs autoscaling
python -m src.jes.comparison_runner --skip-async-regular
```

#### **Analytical Focus**
```powershell
# Isolate pure autoscaling benefits (skip sequential baseline)
python -m src.jes.comparison_runner --skip-sequential --limit-jobs 10

# Focus on resource efficiency (regular vs autoscaling async)
python -m src.jes.comparison_runner --skip-sequential

# Compare different batch sizes for autoscaling
python -m src.jes.comparison_runner --async-autoscaling-only --mode batch --concurrent 5
python -m src.jes.comparison_runner --async-autoscaling-only --mode batch --concurrent 10
```

#### **Troubleshooting**
```powershell
# Debug with single mode
python -m src.jes.comparison_runner --sequential-only --limit-jobs 1

# Test orchestrator correlation
python -m src.jes.comparison_runner --async-autoscaling-only --limit-jobs 2

# Conservative async testing
python -m src.jes.comparison_runner --skip-sequential --mode batch --concurrent 1
```

#### **Help & Options**
```powershell
# Show all available options
python -m src.jes.comparison_runner --help

# View execution plan without running
python -m src.jes.comparison_runner --limit-jobs 0
```

### Command Options
| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Async execution mode: `batch` or `all` | `batch` |
| `--concurrent` | Batch size for batch mode | `5` |
| `--sequential-only` | Run only sequential execution | `False` |
| `--async-regular-only` | Run only async in regular context | `False` |
| `--async-autoscaling-only` | Run only async in autoscaling context | `False` |
| `--skip-sequential` | Skip sequential execution | `False` |
| `--skip-async-regular` | Skip async regular execution | `False` |
| `--skip-async-autoscaling` | Skip async autoscaling execution | `False` |
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

### Smart Job Naming Pattern
The system uses context-aware naming with unique timestamps to ensure reliable orchestrator correlation:

**Sequential Jobs:**
```
Submitted: cm_pooled_sequential_1_1757479582
Transformed: cmpooledsequential11757479582
Orchestrator: cmpooledsequential11757479582-b9deca7d-7056-438c-99c5-5df90ee2a3fb
```

**Async Regular Jobs:**
```
Submitted: cm_pooled_async_reg_1_1757479645
Transformed: cmpooled asyncreg11757479645
Orchestrator: cmpooledasyncreg11757479645-c2f8a1b3-8d42-4e9f-a7c6-9d5e2f4b8a1c
```

**Async Autoscaling Jobs:**
```
Submitted: cm_pooled_async_auto_1_1757479712
Transformed: cmpooled asyncauto11757479712
Orchestrator: cmpooledasyncauto11757479712-f1e9b4d7-6a83-4c2e-b8f5-3a7c9e2d1b4f
```

### Orchestrator Correlation
1. **Job Submission**: Each job gets a unique name with timestamp
2. **Post-Execution**: System waits 15 seconds for orchestrator registration
3. **Pattern Matching**: Direct prefix match finds the exact orchestrator job
4. **Metrics Extraction**: Comprehensive metrics from orchestrator API

### Orchestrator Data Structure (Important!)

**Queue Wait Times**: The real queue wait data is stored in:
```
job['metrics']['orchestrator_data']['total_time_pending']  # Queue wait in seconds
job['metrics']['orchestrator_data']['total_time_running']  # Execution in seconds
```

**NOT in**: `orchestrator_timing.queue_wait_seconds` (this is always 0.0)

**Typical Queue Wait Times Found**:
- **Sequential Jobs**: ~0 seconds (immediate execution)
- **Async Regular Jobs**: ~218 seconds (~3.6 minutes queue wait)
- **Async Autoscaling Jobs**: ~119 seconds (~2.0 minutes queue wait)

**Key Insight**: Autoscaling reduces queue wait time by ~45% AND improves execution efficiency!

### Metrics Captured
- **Timing**: Queue wait time, execution time, total duration
- **Resources**: CPU cores used, memory consumption, node count
- **Infrastructure**: Actual compute node hostnames
- **Autoscaling**: Scaling events and node allocation
- **Performance**: Resource-adjusted speedup calculations
- **Efficiency**: Actual vs theoretical performance ratios

## 📁 Project Structure

```
JES/
├── src/jes/                          # Core package
│   ├── auth.py                      # OAuth2 authentication & token management
│   ├── comparison_runner.py         # Three-way comparison logic
│   └── job_execution.py             # SAS Viya API client
├── extract_and_analyze.py           # Data extraction (JSON → Excel) + Auto takeaway generation
├── visualize_comparison.py          # Three-way comparison visualizations (Excel-optimized)
├── config/                          # Configuration files
│   └── job_configs.json             # Job definitions (54 pooled jobs)
├── data/                            # Runtime data
│   └── sas_tokens.json              # OAuth2 tokens (auto-generated)
├── archive/                         # Archived documentation
│   ├── ASYNC_JOBS_README.md         # Legacy documentation
│   └── AUTOSCALING_ANALYSIS_GUIDE.md # Analysis guide
├── .env                             # Environment configuration (create from env.template)
├── demos/                           # Demo scripts
└── results/                         # Output files
    ├── pooled_jobs_comparison_*.json # Three-way comparison results
    ├── orchestrator_analysis_*.xlsx  # Excel data with performance metrics
    ├── takeaway.md                   # Auto-generated executive summary
    ├── analytical_summary.csv       # Resource-adjusted analysis table
    ├── execution_timeline_enhanced.png  # Three-way timeline comparison
    ├── resource_efficiency_analysis.png # Speedup and parallel efficiency
    ├── job_node_context_overlay.png     # Job-node resource allocation
    ├── queue_wait_analysis.png          # Three-way queue wait analysis
    ├── node_utilization_comparison.png  # Node scaling behavior
```

## 📈 Three-Way Analysis & Visualization

### Running Analysis

**Complete Workflow (Recommended)**:
```powershell
# Step 1: Run three-way comparison
python -m src.jes.comparison_runner --limit-jobs 5

# Step 2: Extract data to Excel + Generate takeaway.md
python extract_and_analyze.py

# Step 3: Generate focused visualizations (uses Excel data automatically)
python visualize_comparison.py
```

**What Gets Generated**:
- **Excel File**: `orchestrator_analysis_YYYYMMDD_HHMMSS.xlsx` with 7 sheets of analysis
- **Takeaway File**: `takeaway.md` with executive summary and key insights
- **6 Key Visualizations**: Timeline, node overlay, efficiency, queue analysis, etc.

**Manual File Selection**:
```powershell
# Use specific Excel file (preferred - accurate queue wait times)
python visualize_comparison.py results/orchestrator_analysis_20250915_171747.xlsx

# Use specific JSON file (fallback - legacy support)
python visualize_comparison.py results/pooled_jobs_comparison_20250915_171747.json

# Use latest file automatically (recommended)
python visualize_comparison.py
```

### Data Pipeline Workflow

**1. Job Execution** → **2. Data Extraction & Visualization**

```
comparison_runner.py  →  extract_and_analyze.py  →  visualize_comparison.py
     (JSON output)    →     (Excel output)      →     (PNG charts)
```

**Why This Pipeline?**:
- **JSON**: Contains raw job execution data with complex nested structures
- **Excel**: Pre-processed, clean data with proper queue wait times extracted
- **Visualizations**: Use clean Excel data for accurate, fast chart generation

**Data Freshness Validation**:
- Automatically checks if Excel timestamp matches latest JSON
- Prevents using outdated Excel data with newer JSON results
- Guides user to run extraction step when needed

**Simplified Workflow**: All visualization needs are now handled by `analytical_visualizer.py` - no need for multiple visualization tools.

### Resource-Adjusted Analysis Framework

The system provides sophisticated analysis that accounts for resource differences:

**Key Insight**: If autoscaling uses 2 nodes vs sequential's 1 node, theoretical maximum speedup should be 2x. Any performance beyond that is **true autoscaling benefit**.

**Analysis Components:**
- **Parallelization Benefit**: Sequential vs Async Regular = Pure parallelization gains
- **True Autoscaling Benefit**: Async Regular vs Async Autoscaling = Pure autoscaling gains  
- **Resource Efficiency**: Actual speedup vs theoretical speedup based on node count
- **Performance Decomposition**: Separates infrastructure gains from autoscaling intelligence

### Generated Outputs

**📊 Analytical Summary Table** (`analytical_summary.csv`):
```
Execution Mode          Total Time (sec)  Node Count  Actual Speedup  Efficiency Ratio
Sequential (Baseline)   2789.45          1           1.00x           100.0%
Async Regular          1456.23          2           1.92x           96.0%
Async Autoscaling      1123.67          2           2.48x           124.0%
```

**All visualizations are generated by `visualize_comparison.py`**:
- **📈 Enhanced Timeline**: Three-panel comparison with queue wait breakdown
- **📊 Resource Efficiency**: Speedup and parallel efficiency analysis  
- **📋 Executive Dashboard**: Key insights and performance metrics
- **🔗 Job-Node Overlay**: Resource allocation patterns across compute nodes
- **📊 Analytical Summary**: CSV table with performance calculations

### Real-World Performance Insights

**Critical Data Discovery**: Queue wait times are in `metrics.orchestrator_data.total_time_pending`, not `orchestrator_timing.queue_wait_seconds` (always 0.0).

**Actual Performance Results**:
- **Sequential**: 47.0 min (1 node, ~0s queue wait) - Baseline
- **Async Regular**: 15.1 min (2 nodes, ~218s queue wait) - 155.6% parallel efficiency  
- **Async Autoscaling**: 10.2 min (2 nodes, ~119s queue wait) - 230.4% parallel efficiency

**Autoscaling Double Benefit**:
1. **Queue Wait Reduction**: 218s → 119s (45% improvement)
2. **Execution Optimization**: Better resource utilization and scheduling
3. **Combined Impact**: 32.5% faster than regular async (74.8 efficiency points gain)

**Efficiency Calculations**:
- **Speedup**: `Baseline time ÷ Runtime` (e.g., 47.0 ÷ 10.2 = 4.61×)
- **Parallel Efficiency**: `(Speedup ÷ Node count) × 100%` (e.g., 4.61 ÷ 2 = 230.4%)
- **Superlinear (>100%)**: Indicates optimizations beyond simple parallelization

## 📈 JSON Output Format

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
  "async_regular_execution": {
    "context_name": "SAS Job Execution compute context",
    "total_duration": 78.23,
    "successful_jobs": 54,
    "failed_jobs": 0
  },
  "async_autoscaling_execution": {
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

### Analytical Insights

The system provides automated performance insights:

**🎯 Efficiency Analysis:**
- **Excellent (>80%)**: Parallelization/autoscaling working optimally
- **Good (60-80%)**: Reasonable performance with some overhead
- **Poor (<60%)**: Investigate bottlenecks or configuration issues

**📊 Key Metrics:**
- **Pure Autoscaling Benefit**: Performance gain beyond parallelization
- **Resource Efficiency**: How well resources are utilized vs theoretical maximum
- **Scaling Factor**: Actual nodes used in each execution mode

## 🔍 Troubleshooting

### Common Issues

**Three-way comparison not working:**
- Ensure you're using the new command: `python -m src.jes.comparison_runner`
- Check that all three execution modes are enabled (not skipped)
- Verify job configurations include proper context names

**Excel data outdated error:**
- Error message: "Excel data is outdated. Run extract_analyze.py first"
- **Solution**: `.\.venv\Scripts\python.exe extract_analyze.py`
- **Cause**: JSON file is newer than Excel file (timestamps don't match)
- **Prevention**: Always run extract_analyze.py after comparison_runner.py

**No orchestrator data found:**
- Increase wait time in `correlate_all_orchestrator_data()` if needed
- Verify orchestrator API access with proper permissions
- Check that jobs are appearing in `/workloadOrchestrator/jobs`
- Ensure job naming patterns are correctly transformed

**Visualization errors:**
- Install required packages: `pip install matplotlib pandas numpy`
- Ensure JSON results file exists and contains three-way data
- Check that all execution modes have data (not empty job lists)

**Missing queue wait times in visualizations:**
- **Correct Path**: `job['metrics']['orchestrator_data']['total_time_pending']`
- **Wrong Path**: `orchestrator_timing.queue_wait_seconds` (always 0.0)
- **Expected Values**: Async Regular ~218s, Async Autoscaling ~119s, Sequential ~0s
- **Validation**: Check that blue bars appear in timeline for async jobs
- **Debug Command**: `py -c "import json; data=json.load(open('results/latest.json')); print(data['async_regular_execution']['jobs'][10]['metrics']['orchestrator_data']['total_time_pending'])"`

**Takeaway.md not generated:**
- **Check**: Ensure `extract_and_analyze.py` completed successfully
- **Location**: File should be in `results/takeaway.md`
- **Requirements**: Needs Sequential and at least one async execution type
- **Debug**: Check console output for "✅ Generated takeaway.md with performance insights"
- **Manual Generation**: Run `python extract_and_analyze.py` after job execution

**Blank histogram in queue wait analysis:**
- **Issue**: Sequential jobs have 0s queue wait, causing blank histogram
- **Fix**: Updated visualization handles zero-value data properly
- **Expected**: Red bar at 0 for Sequential, histograms for async modes
- **Validation**: All three execution types should appear in the chart

**Authentication errors:**

*Token expired or invalid:*
```powershell
# Re-run authentication to get fresh tokens
python src\jes\auth.py
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

### Major Updates (Latest)
- ✅ **Excel-First Pipeline**: Visualizations now use clean Excel data instead of complex JSON
- ✅ **Three-Way Visualization Support**: All charts properly show Sequential, Async Regular, and Async Autoscaling
- ✅ **Automatic Takeaway Generation**: `extract_and_analyze.py` now creates executive summary markdown
- ✅ **Focused Visualization Suite**: Reduced from 15+ charts to 6 meaningful visualizations
- ✅ **Fixed Data Access Issues**: Unified data handling for both Excel and JSON formats
- ✅ **Queue Wait Histogram Fix**: Properly displays zero-value data (Sequential jobs)
- ✅ **PowerPoint Slide Deprecation**: Removed outdated 2-way comparison slides

### Previous Updates
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

## 📊 Advanced Analytical Visualizations

The enhanced `visualize_comparison.py` provides comprehensive analysis and visualization capabilities:

### Generated Visualizations (6 Key Charts)
- **📈 execution_timeline_enhanced.png**: Three-way timeline with queue wait breakdown
- **🔗 job_node_context_overlay.png**: Job-node resource allocation patterns  
- **📊 resource_efficiency_analysis.png**: Speedup and parallel efficiency analysis
- **⏰ queue_wait_analysis.png**: Three-way queue wait time comparison
- **📊 node_utilization_comparison.png**: Node scaling behavior analysis
- **📋 analytical_summary.csv**: Resource-adjusted performance metrics table

### Auto-Generated Executive Summary
- **📝 takeaway.md**: Executive summary with key performance insights
  - Performance vs perfect split analysis
  - Autoscaling vs regular async comparison  
  - Business impact summary table
  - Real-time calculations using actual data

### Visualization Features
- **Resource-Adjusted Analysis**: Accounts for node count differences
- **Queue Wait Breakdown**: Shows real queue times vs execution times
- **Superlinear Efficiency**: Identifies performance beyond theoretical limits
- **Node Allocation Patterns**: Visualizes how jobs distribute across compute nodes
- **Professional Charts**: Publication-ready visualizations with embedded insights

All visualizations are automatically saved to the `results/` directory with proper timestamps and data validation.

## 📊 Metrics Insights

The system provides comprehensive insights including:
- **Execution Comparison**: Sequential vs parallel performance
- **Resource Utilization**: CPU and memory usage patterns
- **Queue Analysis**: Wait times and queue distribution
- **Autoscaling Behavior**: Node allocation and scaling events
- **Host Distribution**: Actual compute nodes used

## 🎯 Next Steps

1. **Run a test**: `python -m src.jes.comparison_runner --limit-jobs 3`
2. **Extract data + Generate takeaway**: `python extract_and_analyze.py`
3. **Generate visualizations**: `python visualize_comparison.py`
4. **Review results**: Check `results/` folder for:
   - `orchestrator_analysis_*.xlsx` - Excel data with 7 analysis sheets
   - `takeaway.md` - Executive summary with key insights
   - 6 PNG visualization files (timeline, efficiency, node analysis, etc.)
5. **Scale up**: Remove `--limit-jobs` to run all 54 jobs for full analysis
6. **Share insights**: Use `takeaway.md` for executive presentations
7. **Optimize**: Adjust batch sizes based on your SAS Viya limits

## 📄 License

[Your License Here]

## 👥 Contributors

[Your Team/Contributors Here]