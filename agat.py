import subprocess
import logging
import pandas as pd
import re
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
from src.config_loader import Config
from src.error_utils import PipelineError, handle_graceful_exit

logger = logging.getLogger(__name__)


@handle_graceful_exit
def diagnose_config_loading(config: Config) -> None:
    """
    Comprehensive diagnostic function to identify configuration loading issues.
    
    Args:
        config: Configuration object to diagnose
    
    Raises:
        RuntimeError: If configuration loading issues are detected
    """
    # Detailed diagnostic checks
    diagnostics = {
        "Config Type": type(config),
        "Config Attributes": dir(config),
        "AGAT Config Exists": hasattr(config, 'agat'),
        "AGAT Attribute Type": type(config.agat) if hasattr(config, 'agat') else "N/A",
        "AGAT Config Attributes": dir(config.agat) if hasattr(config, 'agat') else "No AGAT config",
        "Patterns Attribute Check": hasattr(config.agat, 'patterns') if hasattr(config, 'agat') else "No AGAT config"
    }

    # Prepare detailed error message
    error_details = "\n".join([f"{k}: {v}" for k, v in diagnostics.items()])
    
    # Raise an explicit error with full diagnostic information
    raise RuntimeError(f"""
    CONFIGURATION LOADING DIAGNOSTIC
    -------------------------------
    CRITICAL: Unable to access configuration patterns

    Detailed Diagnostic Information:
    {error_details}

    YAML Configuration Check:
    - Is the YAML correctly formatted?
    - Are the attributes exactly matching the defined Config class?
    - Verify the import and config loading mechanism
    """)


@handle_graceful_exit
def parse_agat_summary(file_path: Path, config: Config) -> Dict[str, Any]:
    """
    Parse AGAT statistics file to extract metrics.
    
    Args:
        file_path: Path to AGAT statistics output file
        config: Pipeline configuration containing extraction patterns
        
    Returns:
        Dictionary containing parsed AGAT statistics
    """
    species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
    results = {
        "file_name": species_name,
        "species": species_name
    }
    
    try:
        with file_path.open('r', encoding='utf-8') as f:
            content = f.read()
            
            # Search for the mRNA section
            mrna_section_match = re.search(r"-+\s*mrna\s*-+", content)
            if not mrna_section_match:
                logger.error(f'mRNA section not found in AGAT statistics file: {file_path.stem}')
                return results
            else:
                logger.info(f'mRNA section found in {file_path.stem}')
            
            # Extract mRNA section content
            mrna_section_start = mrna_section_match.end()
            mrna_section = content[mrna_section_start:]
            
            # Dynamic extraction based on config patterns
            for key, pattern in config.agat.patterns.items():
                try:
                    match = re.search(pattern, mrna_section, re.IGNORECASE)
                    results[key] = match.group(1).strip() if match and match.groups() else None
                except Exception as e:
                    logger.warning(f"Could not extract {key} for {species_name}: {e}")
                    results[key] = None
                    
            # Validate required statistics
            for stat in config.agat.required_stats:
                if results.get(stat) is None:
                    logger.warning(f'Required statistic "{stat}" is missing for {species_name}')
            
            return results

    except Exception as e:
        logger.error(f"Error parsing AGAT summary {file_path}: {e}")
        raise PipelineError(f"AGAT parsing error: {e}")


@handle_graceful_exit
def process_agat_file(dir_path: Path, file_path: Path, config: Config) -> Dict[str, Any]:
    """
    Process a single file through AGAT statistics analysis.
    
    Args:
        dir_path: Directory path containing the file
        file_path: Path to the GFF file
        config: Pipeline configuration
        
    Returns:
        Dictionary with AGAT analysis results or None if processing failed
    """
    logger.info(f"Processing file: {file_path}")
    species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
    
    out_name = f'{file_path.stem}_agat_stats.txt'
    out_path = file_path.parent
    output_file = out_path / out_name

    # Construct command with explicit long-form parameters
    cmd = [
        'agat_sp_statistics.pl',
        '--gff', str(file_path),  # Use full --gff parameter
        '-o', str(output_file),
        '-v'
    ]
        
    logger.info("=" * 80)
    logger.info("STEP 1: Running AGAT statistics command")
    logger.info("-" * 80)
    logger.info(f'Command: {" ".join(str(arg) for arg in cmd)}')
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True,  # Capture both stdout and stderr
            text=True,
            check=False  # Don't raise exception immediately
        )

        # Log command output for debugging
        logger.debug(f"AGAT STDOUT: {result.stdout}")
        logger.debug(f"AGAT STDERR: {result.stderr}")

        # Check return code explicitly
        if result.returncode != 0:
            logger.error(f"AGAT command failed with return code {result.returncode}")
            logger.error(f"AGAT STDOUT: {result.stdout}")
            logger.error(f"AGAT STDERR: {result.stderr}")
            return None
    
        logger.info("=" * 80)
        logger.info("STEP 2: Parsing AGAT summary results")
        logger.info("-" * 80)
        
        # Verify output file exists
        if output_file.exists():
            return parse_agat_summary(output_file, config)
        else:
            logger.error("=" * 80)
            logger.error("STEP 2 ERROR: AGAT output file not generated")
            logger.error("-" * 80)
            logger.error(f"No AGAT output generated for {species_name}")
            return None

    except Exception as e:
        logger.error("=" * 80)
        logger.error("STEP 1 ERROR: AGAT command execution failed")
        logger.error("-" * 80)
        logger.error(f"Error running AGAT command for {species_name}: {e}")
        
        # Additional debugging information
        logger.error(f"AGAT executable path: {shutil.which('agat_sp_statistics.pl')}")
        logger.error(f'Failed command: {" ".join(str(arg) for arg in cmd)}')
        return None


@handle_graceful_exit
def run_agat_analysis(config: Config, input_files: List[Tuple[Path, Path]]) -> None:
    """
    Run AGAT statistics analysis pipeline with parallel processing.
    Uses ProcessPoolExecutor with a maximum of 4 concurrent processes.
    
    Args:
        config: Pipeline configuration
        input_files: List of tuples containing (directory_path, file_path) for GFF files
    """
    logger.info("=" * 80)
    logger.info("Starting parallel AGAT statistics analysis pipeline")
    logger.info("=" * 80)
    
    # Check input files
    if not input_files:
        error_msg = "No input files provided for AGAT analysis"
        logger.error(error_msg)
        raise PipelineError(error_msg)
    
    results = []
    
    # Execute in serial or parallel mode based on number of files
    if len(input_files) == 1:
        dir_path, file_path = input_files[0]
        species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
        logger.info(f'AGAT analysis for {species_name}')
        try:
            result = process_agat_file(dir_path, file_path, config)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.error(f"AGAT analysis error: {e}")
    else:
        # For parallelism between files with maximum 4 workers
        max_workers = min(len(input_files), 4)
        logger.info(f"Using ProcessPoolExecutor with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_agat_file, dir_path, file_path, config): (dir_path, file_path)
                for dir_path, file_path in input_files
            }

            for future in as_completed(future_to_file):
                dir_path, file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"File processing error {file_path.name}: {e}")

    # Generate summary report if there are results
    if results:
        # Ensure output directory exists
        results_dir = Path("./agat_results")
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Create results dataframe and save to CSV
        output_path = results_dir / "agat_statistics_results.csv"
        pd.DataFrame(results).to_csv(output_path, index=False)
        logger.info(f"AGAT results saved to {output_path}")
        
        # Generate summary statistics
        total_species = len(results)
        successful_analyses = sum(1 for r in results if r.get("file_name"))
        
        logger.info("=" * 80)
        logger.info("AGAT Analysis Summary")
        logger.info(f"Total species analyzed: {total_species}")
        logger.info(f"Successful analyses: {successful_analyses}")
        logger.info("=" * 80)
    else:
        logger.warning("No valid AGAT results obtained")


@handle_graceful_exit
def run_agat_keep_longest(config: Config, input_files: List[Tuple[Path, Path]]) -> None:
    """
    Run AGAT keep longest isoform analysis for input GFF files.
    
    Args:
        config: Pipeline configuration
        input_files: List of tuples containing (directory_path, file_path) for GFF files
    """
    logger.info("=" * 80)
    logger.info("Starting AGAT keep longest isoform analysis")
    logger.info("=" * 80)
    logger.info("AGAT process: keep_longest_isoform")

    # Filter for valid GFF files
    gff_files = []
    for dir_path, file_path in input_files:
        species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
        logger.info(f'Processing species: {species_name}')
        if file_path.suffix.lower() in ('.gff', '.gff3'):
            gff_files.append((dir_path, file_path))

    if not gff_files:
        error_msg = f"No valid GFF files found in input files: {input_files}"
        logger.error(error_msg)
        raise PipelineError(error_msg)

    successful_processes = 0
    failed_processes = 0

    for dir_path, file_path in gff_files:
        species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
        output_file = file_path.with_name(f"{file_path.stem}{config.agat.output_suffix}{file_path.suffix}")
        
        cmd = [
            config.agat.keep_longest_cmd,
            "-g", str(file_path),
            "-o", str(dir_path / output_file.name)
        ]

        logger.info(f'Running command: {" ".join(str(arg) for arg in cmd)} for {species_name}')
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                stderr=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                text=True
            )

            # Verify output file was created
            expected_output = dir_path / output_file.name
            if not expected_output.exists():
                logger.error(f"Output file not generated: {expected_output}")
                failed_processes += 1
                continue

            logger.info(f"Successfully generated file: {expected_output.name}")
            successful_processes += 1

        except subprocess.CalledProcessError as e:
            logger.error(f"AGAT execution error for {species_name}: {e.stderr}")
            failed_processes += 1
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing {species_name}: {e}")
            failed_processes += 1
            continue

    # Generate summary report
    logger.info("=" * 80)
    logger.info("AGAT Keep Longest Isoform Summary")
    logger.info(f"Total files processed: {len(gff_files)}")
    logger.info(f"Successful processes: {successful_processes}")
    logger.info(f"Failed processes: {failed_processes}")
    logger.info("=" * 80)

    if failed_processes > 0:
        logger.warning(f"Some AGAT processes failed ({failed_processes}/{len(gff_files)})")
