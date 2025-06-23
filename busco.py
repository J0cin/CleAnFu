import subprocess
import logging
import re
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
from src.config_loader import Config
from src.error_utils import PipelineError, handle_graceful_exit

logger = logging.getLogger(__name__)


@handle_graceful_exit
def parse_busco_summary(output_dir: Path, lineage: str) -> Dict[str, Any]:
    """
    Parse BUSCO summary file to extract metrics.
    
    Args:
        output_dir: Directory containing BUSCO output files
        lineage: BUSCO lineage used for analysis
        
    Returns:
        Dictionary containing parsed BUSCO metrics
    """
    logger.info(f"Searching for summary files in: {output_dir}")
    all_files = list(output_dir.glob("**/*.txt"))
    logger.info(f"Files found: {[f.name for f in all_files]}")
    
    # Initialize summary_file as None
    summary_file = None
    
    # Try to find summary files with broader pattern
    summary_files = list(output_dir.glob("**/short_summary*.txt"))
    
    if summary_files:
        summary_file = summary_files[0]
        logger.info(f"Using BUSCO summary file: {summary_file}")
    else:
        logger.warning(f"Could not find 'short_summary*.txt' in {output_dir}")
        # Check alternate locations - look in runs directory
        summary_files = list(output_dir.glob("**/run_*/short_summary*.txt"))
        if summary_files:
            summary_file = summary_files[0]
        else:
            logger.info("Searching in all files for summary")
            for file in all_files:
                if "summary" in file.name.lower():
                    summary_file = file
                    break
    
    # Final check if summary_file was assigned
    if not summary_file:
        logger.error(f"BUSCO summary file not found in {output_dir}")
        logger.warning("Using default results: 0.0")
        species_name = output_dir.parts[-3] if len(output_dir.parts) >= 3 else output_dir.name
        return {
            "file_name": species_name,
            "species": species_name,
            "busco_score": "0.0",
            "single_copy": "0.0",
            "duplicated": "0.0",
            "fragmented": "0.0",
            "missing": "0.0",
            "total_buscos": "0",
            "lineage": lineage
        }
            
    logger.info(f"Using file {summary_file} for BUSCO summary parsing")   
    
    # Create species name for results
    species_name = output_dir.parts[-3] if len(output_dir.parts) >= 3 else output_dir.name
    logger.info(f"Species name generated: {species_name}")
    
    # Variable to store results
    results = {
        "file_name": species_name,
        "species": species_name
    }
    
    try:
        with summary_file.open('r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for the summary line with percentages
            summary_line = None
            for line in content.split('\n'):
                if 'C:' in line and 'S:' in line and 'F:' in line and 'M:' in line and 'n:' in line:
                    summary_line = line.strip()
                    break
            
            if not summary_line:
                logger.error(f"BUSCO summary line not found in: {summary_file}")
                logger.error("File format not recognized")
                return _get_default_busco_results(species_name, lineage, "format_not_recognized")
            
            logger.info(f"BUSCO summary line found: {summary_line}")
            
            # Extract completion percentage
            c_match = re.search(r'C:(\d+\.\d+)%', summary_line)
            if c_match:
                results["busco_score"] = c_match.group(1)
                logger.info(f"BUSCO score found for {species_name}")
            else:
                results["busco_score"] = "0.0"
                logger.error(f"BUSCO score not found for {species_name}")
            
            # Extract single-copy percentage
            s_match = re.search(r'S:(\d+\.\d+)%', summary_line)
            if s_match:
                results["single_copy"] = s_match.group(1)
                logger.info(f"Single-copy found for {species_name}")
            else:
                results["single_copy"] = "0.0"
                logger.error(f"Single-copy not found for {species_name}")
            
            # Extract duplicated percentage
            d_match = re.search(r'D:(\d+\.\d+)%', summary_line)
            if d_match:
                results["duplicated"] = d_match.group(1)
                logger.info(f"Duplicated found for {species_name}")
            else:
                results["duplicated"] = "0.0"
                logger.error(f"Duplicated not found for {species_name}")
            
            # Extract fragmented percentage
            f_match = re.search(r'F:(\d+\.\d+)%', summary_line)
            if f_match:
                results["fragmented"] = f_match.group(1)
                logger.info(f"Fragmented found for {species_name}")
            else:
                results["fragmented"] = "0.0"
                logger.error(f"Fragmented not found for {species_name}")
            
            # Extract missing percentage
            m_match = re.search(r'M:(\d+\.\d+)%', summary_line)
            if m_match:
                results["missing"] = m_match.group(1)
                logger.info(f"Missing found for {species_name}")
            else:
                results["missing"] = "0.0"
                logger.error(f"Missing not found for {species_name}")
            
            # Extract total BUSCOs
            n_match = re.search(r'n:(\d+)', summary_line)
            results["total_buscos"] = n_match.group(1) if n_match else "0"
            
            if lineage:
                results["lineage"] = lineage
                logger.info(f"Lineage: {lineage} found for {species_name}")
            else:
                results["lineage"] = "Unknown"
                logger.warning(f"Lineage not found for {species_name}")
            
            logger.info(f"Data extracted successfully: {results}")
            return results
            
    except Exception as e:
        logger.error(f"BUSCO summary parsing failed for {summary_file}: {e}")
        return _get_default_busco_results(species_name, lineage, str(e))


def _get_default_busco_results(species_name: str, lineage: str, error: str = "") -> Dict[str, Any]:
    """
    Generate default BUSCO results dictionary for failed analyses.
    
    Args:
        species_name: Name of the species
        lineage: BUSCO lineage used
        error: Error message to include
        
    Returns:
        Dictionary with default BUSCO results
    """
    results = {
        "file_name": species_name,
        "species": species_name,
        "busco_score": "0.0",
        "single_copy": "0.0",
        "duplicated": "0.0",
        "fragmented": "0.0",
        "missing": "0.0",
        "total_buscos": "0",
        "lineage": lineage
    }
    
    if error:
        results["error"] = error
        
    return results


@handle_graceful_exit    
def process_busco_file(dir_path: Path, file_path: Path, config: Config, lineage: str) -> Dict[str, Any]:
    """
    Process a single file through BUSCO analysis.
    
    Args:
        dir_path: Directory path containing the file
        file_path: Path to the input FASTA file
        config: Pipeline configuration
        lineage: BUSCO lineage database to use
        
    Returns:
        Dictionary with BUSCO analysis results or None if processing failed
    """
    logger.info(f"Processing file: {file_path}")
    species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
    
    # Use the current input directory to nest output directory
    out_path = file_path.parent
    # Create name for output directory
    out_name = f'{file_path.stem}_busco_{lineage}_odb10'

    cmd = [
        config.busco.busco_cmd,
        '-i', str(file_path),
        '-l', lineage,
        '-m', config.busco.mode,
        '--out_path', str(out_path),  # Specify parent directory
        '-o', str(out_name),  # Specify output directory for results
        '--cpu', str(config.busco.cpu),
        '-f'  # Force overwrite
    ]

    logger.info(f'Running BUSCO command: {" ".join(str(arg) for arg in cmd)}')
    
    # Create variable for output path
    output_dir = Path(out_path) / out_name
    logger.info(f'BUSCO output directory: {output_dir}')
    
    try:
        result = subprocess.run(
            cmd, 
            check=False,  # Don't raise exception immediately
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=config.busco.timeout
        )

        if result.returncode != 0:
            logger.error(f'BUSCO command failed for {file_path.name}: {result.stderr}')
            logger.error(f'BUSCO stdout: {result.stdout}')
            
            # Check if any output was generated despite error
            busco_output = output_dir
            if busco_output.exists() and list(busco_output.glob("**/*.txt")):
                logger.info("BUSCO failed but generated some files, attempting to recover results")
                return parse_busco_summary(busco_output, lineage)
            else:
                logger.error(f"BUSCO analysis failed for {file_path.name}: {result.stderr}")
                return _get_default_busco_results(species_name, lineage, "busco_command_failed")
        
        # Look in the correct location for output
        busco_output = output_dir
        logger.info(f"Searching for BUSCO results in: {busco_output}")
        return parse_busco_summary(busco_output, lineage)

    except subprocess.TimeoutExpired:
        logger.error(f"BUSCO timeout for {file_path}")
        return _get_default_busco_results(species_name, lineage, "timeout")
    except FileNotFoundError as e:
        logger.error(f"BUSCO results not found for {file_path}: {e}")
        return _get_default_busco_results(species_name, lineage, "file_not_found")
    except Exception as e:
        logger.error(f"Generic BUSCO error for {file_path}: {e}")
        return _get_default_busco_results(species_name, lineage, str(e))


@handle_graceful_exit
def run_busco_analysis(config: Config, input_files: List[Tuple[Path, Path, str]]) -> None:
    """
    Run BUSCO analysis pipeline with parallel processing.
    Uses ProcessPoolExecutor with a maximum of 4 concurrent processes.
    
    Args:
        config: Pipeline configuration
        input_files: List of tuples containing (directory_path, file_path, lineage) for analysis
    """
    logger.info("=" * 80)
    logger.info("Starting parallel BUSCO analysis pipeline")
    logger.info("=" * 80)
    
    # Check input files
    if not input_files:
        error_msg = "No input files provided for BUSCO analysis"
        logger.error(error_msg)
        raise PipelineError(error_msg)
    
    results = []
    
    # Execute in serial or parallel mode based on number of files
    if len(input_files) == 1:
        dir_path, file_path, lineage = input_files[0]
        species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
        logger.info(f'BUSCO analysis for {species_name} with lineage {lineage}')
        try:
            result = process_busco_file(dir_path, file_path, config, lineage)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.error(f"BUSCO analysis error: {e}")
    else:
        # For parallelism between files with maximum 4 workers
        max_workers = min(len(input_files), 4)
        logger.info(f"Using ProcessPoolExecutor with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_busco_file, dir_path, file_path, config, lineage): (dir_path, file_path, lineage)
                for dir_path, file_path, lineage in input_files
            }

            for future in as_completed(future_to_file):
                dir_path, file_path, lineage = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"File processing error {file_path.name}: {e}")

    # Generate summary report if there are results
    if results:
        # Filter valid results
        valid_results = [r for r in results if r is not None]
        if valid_results:
            # Ensure output directory exists
            results_dir = Path("./busco_results")
            results_dir.mkdir(exist_ok=True, parents=True)
            
            # Create results dataframe and save to CSV
            output_path = results_dir / "busco_analysis_results.csv"
            pd.DataFrame(valid_results).to_csv(output_path, index=False)
            logger.info(f"BUSCO results saved to {output_path}")
            
            # Generate summary statistics
            total_species = len(valid_results)
            avg_busco_score = sum(float(r.get("busco_score", 0)) for r in valid_results) / total_species
            total_complete = sum(float(r.get("busco_score", 0)) >= 90.0 for r in valid_results)
            
            logger.info("=" * 80)
            logger.info("BUSCO Analysis Summary")
            logger.info(f"Total species analyzed: {total_species}")
            logger.info(f"Average BUSCO score: {avg_busco_score:.2f}%")
            logger.info(f"Species with >90% completeness: {total_complete}")
            logger.info("=" * 80)
        else:
            logger.warning("No valid BUSCO results found")
    else:
        logger.warning("No BUSCO results obtained")


@handle_graceful_exit
def run_busco_summary(config: Config, input_files: List[Path]) -> None:
    """
    Generate a summary of existing BUSCO analyses.
    
    Args:
        config: Pipeline configuration
        input_files: List of files to include in the summary
    """
    logger.info("Generating BUSCO results summary")
    
    # Search for BUSCO result directories
    busco_dirs = list(config.busco.output_dir.glob("*/*/short_summary.txt"))
    
    if not busco_dirs:
        error_msg = "No BUSCO results found for summary generation"
        logger.error(error_msg)
        raise PipelineError(error_msg)
    
    results = []
    for summary_file in busco_dirs:
        try:
            # Extract lineage from path if possible
            lineage = "unknown"
            if "_odb10" in str(summary_file):
                lineage_match = re.search(r'_([^_]+)_odb10', str(summary_file))
                if lineage_match:
                    lineage = lineage_match.group(1)
            
            result = parse_busco_summary(summary_file.parent, lineage)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {summary_file}: {e}")
    
    if results:
        output_path = config.output_dir / "busco_summary.csv"
        pd.DataFrame(results).to_csv(output_path, index=False)
        logger.info(f"BUSCO summary saved to {output_path}")
    else:
        logger.warning("No results obtained for BUSCO summary")
