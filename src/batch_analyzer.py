"""
Batch Analysis Module.
Enables running multiple simulations in sequence (#17).
"""
import os
import json
import time
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


@dataclass
class BatchAnalysisJob:
    """Single batch analysis job configuration."""
    job_id: str
    earthquake_file: Optional[str] = None
    max_acc: float = 400.0
    duration: float = 10.0
    scale_factor: float = 1.0
    description: str = ""


@dataclass
class BatchAnalysisResult:
    """Result from a single batch job."""
    job_id: str
    success: bool
    max_drift: float = 0.0
    max_disp: float = 0.0
    max_accel: float = 0.0
    elapsed_time: float = 0.0
    error_message: str = ""
    output_file: str = ""


class BatchAnalyzer:
    """
    Runs multiple earthquake simulations in batch mode.
    
    Features:
    - Sequential or parallel execution
    - Progress tracking
    - Result aggregation
    - Export to CSV/JSON
    """
    
    def __init__(self, output_dir: str = "batch_results"):
        self.output_dir = output_dir
        self.jobs: List[BatchAnalysisJob] = []
        self.results: List[BatchAnalysisResult] = []
        
        os.makedirs(output_dir, exist_ok=True)
        
    def add_job(self, job: BatchAnalysisJob):
        """Add a job to the batch queue."""
        self.jobs.append(job)
        
    def add_scaled_series(
        self, 
        earthquake_file: str,
        scale_factors: List[float],
        base_description: str = ""
    ):
        """Add a series of jobs with different scale factors."""
        for i, sf in enumerate(scale_factors):
            job = BatchAnalysisJob(
                job_id=f"scaled_{i+1}",
                earthquake_file=earthquake_file,
                scale_factor=sf,
                description=f"{base_description} (SF={sf})"
            )
            self.jobs.append(job)
            
    def add_wave_suite(
        self,
        earthquake_files: List[str],
        max_acc: float = 400.0,
        duration: float = 10.0
    ):
        """Add a suite of different earthquake waves."""
        for i, filepath in enumerate(earthquake_files):
            filename = os.path.basename(filepath)
            job = BatchAnalysisJob(
                job_id=f"wave_{i+1}",
                earthquake_file=filepath,
                max_acc=max_acc,
                duration=duration,
                description=filename
            )
            self.jobs.append(job)
            
    def run_sequential(
        self,
        run_function: Callable,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[BatchAnalysisResult]:
        """
        Run all jobs sequentially.
        
        Args:
            run_function: Function that takes a job and returns results
            progress_callback: Optional callback (current, total, job_id)
            
        Returns:
            List of results
        """
        self.results = []
        total = len(self.jobs)
        
        for i, job in enumerate(self.jobs):
            if progress_callback:
                progress_callback(i + 1, total, job.job_id)
                
            start_time = time.time()
            
            try:
                result_data = run_function(job)
                elapsed = time.time() - start_time
                
                result = BatchAnalysisResult(
                    job_id=job.job_id,
                    success=True,
                    max_drift=result_data.get('max_drift', 0),
                    max_disp=result_data.get('max_disp', 0),
                    max_accel=result_data.get('max_accel', 0),
                    elapsed_time=elapsed,
                    output_file=result_data.get('output_file', '')
                )
            except Exception as e:
                result = BatchAnalysisResult(
                    job_id=job.job_id,
                    success=False,
                    elapsed_time=time.time() - start_time,
                    error_message=str(e)
                )
                
            self.results.append(result)
            
        return self.results
    
    def run_parallel(
        self,
        run_function: Callable,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[BatchAnalysisResult]:
        """
        Run jobs in parallel using thread pool.
        
        Args:
            run_function: Function that takes a job and returns results
            max_workers: Maximum number of parallel workers
            progress_callback: Optional callback
            
        Returns:
            List of results
        """
        self.results = []
        total = len(self.jobs)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(self._run_single_job, run_function, job): job
                for job in self.jobs
            }
            
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total, job.job_id)
                    
                try:
                    result = future.result()
                    self.results.append(result)
                except Exception as e:
                    self.results.append(BatchAnalysisResult(
                        job_id=job.job_id,
                        success=False,
                        error_message=str(e)
                    ))
                    
        return self.results
    
    def _run_single_job(
        self,
        run_function: Callable,
        job: BatchAnalysisJob
    ) -> BatchAnalysisResult:
        """Run a single job and return result."""
        start_time = time.time()
        
        try:
            result_data = run_function(job)
            elapsed = time.time() - start_time
            
            return BatchAnalysisResult(
                job_id=job.job_id,
                success=True,
                max_drift=result_data.get('max_drift', 0),
                max_disp=result_data.get('max_disp', 0),
                max_accel=result_data.get('max_accel', 0),
                elapsed_time=elapsed,
                output_file=result_data.get('output_file', '')
            )
        except Exception as e:
            return BatchAnalysisResult(
                job_id=job.job_id,
                success=False,
                elapsed_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def export_results_csv(self, filepath: str = None):
        """Export results to CSV file."""
        if filepath is None:
            filepath = os.path.join(self.output_dir, "batch_results.csv")
            
        with open(filepath, 'w') as f:
            # Header
            f.write("job_id,success,max_drift,max_disp,max_accel,elapsed_time,error\n")
            
            for r in self.results:
                f.write(f"{r.job_id},{r.success},{r.max_drift:.6f},{r.max_disp:.6f},"
                       f"{r.max_accel:.3f},{r.elapsed_time:.2f},{r.error_message}\n")
                
        return filepath
    
    def export_results_json(self, filepath: str = None):
        """Export results to JSON file."""
        if filepath is None:
            filepath = os.path.join(self.output_dir, "batch_results.json")
            
        data = {
            'jobs': [asdict(j) for j in self.jobs],
            'results': [asdict(r) for r in self.results],
            'summary': self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filepath
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of batch results."""
        if not self.results:
            return {}
            
        successful = [r for r in self.results if r.success]
        
        if not successful:
            return {
                'total_jobs': len(self.results),
                'successful': 0,
                'failed': len(self.results)
            }
            
        drifts = [r.max_drift for r in successful]
        disps = [r.max_disp for r in successful]
        times = [r.elapsed_time for r in successful]
        
        return {
            'total_jobs': len(self.results),
            'successful': len(successful),
            'failed': len(self.results) - len(successful),
            'max_drift_mean': np.mean(drifts),
            'max_drift_std': np.std(drifts),
            'max_drift_max': np.max(drifts),
            'max_disp_mean': np.mean(disps),
            'max_disp_max': np.max(disps),
            'total_time': sum(times),
            'avg_time_per_job': np.mean(times)
        }
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print(" BATCH ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total Jobs: {summary.get('total_jobs', 0)}")
        print(f"Successful: {summary.get('successful', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        
        if summary.get('successful', 0) > 0:
            print(f"\nMax Drift: {summary.get('max_drift_mean', 0):.4f} ± "
                  f"{summary.get('max_drift_std', 0):.4f} (max: {summary.get('max_drift_max', 0):.4f})")
            print(f"Max Displacement: {summary.get('max_disp_mean', 0):.4f} m (max: {summary.get('max_disp_max', 0):.4f})")
            print(f"Total Time: {summary.get('total_time', 0):.1f} s")
            print(f"Avg Time/Job: {summary.get('avg_time_per_job', 0):.1f} s")
        print("="*50)


def create_ida_batch(
    earthquake_file: str,
    scale_factors: List[float] = None
) -> BatchAnalyzer:
    """
    Create a batch analyzer configured for IDA (Incremental Dynamic Analysis).
    
    Args:
        earthquake_file: Base earthquake file
        scale_factors: List of scale factors (default: 0.5 to 3.0)
        
    Returns:
        Configured BatchAnalyzer
    """
    if scale_factors is None:
        scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
        
    analyzer = BatchAnalyzer(output_dir="ida_results")
    analyzer.add_scaled_series(earthquake_file, scale_factors, "IDA")
    
    return analyzer
