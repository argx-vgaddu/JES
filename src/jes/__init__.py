"""
JES - Job Execution System for SAS Viya

A streamlined solution to compare job execution performance between
sequential and asynchronous execution contexts in SAS Viya.
"""

from .auth import SASViyaAuth
from .job_execution import SASJobExecutionClient
from .comparison_runner import StreamlinedComparisonRunner

__version__ = "1.0.0"
__all__ = ["SASViyaAuth", "SASJobExecutionClient", "StreamlinedComparisonRunner"]
