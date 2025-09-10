#!/usr/bin/env python3
"""
JES - Job Execution System for SAS Viya
Main Entry Point

This script provides the main interface for running job comparisons
using the JES package.
"""

import sys
import os
import asyncio

# Add src to path so we can import the jes package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


if __name__ == "__main__":
    # Import and run the comparison runner's main function
    from jes.comparison_runner import main as comparison_main
    asyncio.run(comparison_main())
