import logging
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.Cleaning_utils import get_unique_seq_treated
from src.error_utils import handle_graceful_exit, PipelineError
from src.config_loader import Config

logger = logging.getLogger(__name__)


@handle_graceful_exit
def run_sequence_cleaner(config: Config, input_files: List[Tuple[Path, Path]]) -> None:
    """
    Run the complete sequence cleaning pipeline with parallel processing.
    Uses ProcessPoolExecutor with a maximum of 3 concurrent processes.
    
    Args:
        config: Pipeline configuration
        input_files: List of tuples containing (directory_path, file_path) of FASTA files
    """
    logger.info("=" * 80)
    logger.info("Starting sequence cleaning pipeline")
    logger.info("=" * 80)
    
    # Check input files
    if not input_files:
        error_msg = "No input files provided for sequence cleaning"
        logger.error(error_msg)
        raise PipelineError(error_msg)
    
    # Filter FASTA files
    fasta_files = _filter_fasta_files(input_files)
    
    if not fasta_files:
        error_msg = "No FASTA files found in input"
        logger.error(error_msg)
        raise PipelineError(error_msg)
    
    results = []
    
    # Process files in serial or parallel mode based on file count
    if len(fasta_files) == 1:
        dir_path, file_path = fasta_files[0]
        species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
        logger.info(f"Sequence cleaning for {species_name}")
        try:
            result = process_cleaning_file(dir_path, file_path, config)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.error(f"Sequence cleaning error: {e}")
    else:
        # Use parallel processing with maximum 3 workers
        max_workers = min(len(fasta_files), 3)
        logger.info(f"Using ProcessPoolExecutor with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_cleaning_file, dir_path, file_path, config): (dir_path, file_path)
                for dir_path, file_path in fasta_files
            }

            for future in as_completed(future_to_file):
                dir_path, file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"File cleaning error {file_path.name}: {e}")
    
    # Generate summary report if there are results
    if results:
        _generate_cleaning_summary(results)
    else:
        logger.warning("No valid sequence cleaning results obtained")
    
    logger.info("=" * 80)
    logger.info("Sequence cleaning pipeline completed")
    logger.info("=" * 80)


def process_cleaning_file(dir_path: Path, file_path: Path, config: Config) -> Dict[str, Any]:
    """
    Process a single FASTA file through the sequence cleaning pipeline.
    
    Args:
        dir_path: Directory path containing the file
        file_path: Path to the FASTA file
        config: Pipeline configuration
        
    Returns:
        Dictionary with cleaning results or None if processing failed
    """
    logger.info(f"Processing file: {file_path}")
    species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
    
    try:
        # Generate output file path
        output_file = dir_path / file_path.with_name(f"{file_path.stem}_clean{file_path.suffix}")
        
        # Initialize counters
        cleaning_stats = {
            "sequences_removed": 0,
            "sequences_added_with_limit": 0,
            "sequences_added_no_limit": 0,
            "total_sequences_processed": 0
        }
        
        # Check if length filtering is configured
        length_filter_enabled = _is_length_filter_enabled(config)
        max_length = getattr(config.cleaning, 'max_length', None) if length_filter_enabled else None
        
        # Process sequences
        cleaning_stats = _process_sequences(
            file_path, output_file, length_filter_enabled, max_length, cleaning_stats
        )
        
        # Validate output and generate result
        if output_file.exists():
            result = _create_cleaning_result(
                species_name, file_path, output_file, cleaning_stats, length_filter_enabled
            )
            logger.info(f"Successfully cleaned file for {species_name}")
            return result
        else:
            error_msg = f"Output file not generated: {output_file}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return None


def _filter_fasta_files(input_files: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
    """
    Filter input files to include only FASTA files.
    
    Args:
        input_files: List of (directory_path, file_path) tuples
        
    Returns:
        List of FASTA file tuples
    """
    fasta_extensions = {'.fa', '.fasta', '.faa'}
    fasta_files = []
    
    for dir_path, file_path in input_files:
        if file_path.suffix.lower() in fasta_extensions:
            fasta_files.append((dir_path, file_path))
    
    logger.info(f"Found {len(fasta_files)} FASTA files for processing")
    return fasta_files


def _is_length_filter_enabled(config: Config) -> bool:
    """
    Check if sequence length filtering is enabled in configuration.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        True if length filtering is enabled, False otherwise
    """
    return (hasattr(config, 'cleaning') and 
            hasattr(config.cleaning, 'max_length') and 
            config.cleaning.max_length is not None)


def _process_sequences(
    input_file: Path, 
    output_file: Path, 
    length_filter_enabled: bool, 
    max_length: int, 
    cleaning_stats: Dict[str, int]
) -> Dict[str, int]:
    """
    Process sequences from input file and write cleaned sequences to output file.
    
    Args:
        input_file: Path to input FASTA file
        output_file: Path to output cleaned FASTA file
        length_filter_enabled: Whether length filtering is enabled
        max_length: Maximum allowed sequence length
        cleaning_stats: Statistics dictionary to update
        
    Returns:
        Updated cleaning statistics dictionary
    """
    try:
        with open(output_file, 'w') as out:
            for header, sequences in get_unique_seq_treated(input_file):
                cleaning_stats["total_sequences_processed"] += 1
                seq_length = len(sequences.replace("\n", ""))
                
                if length_filter_enabled:
                    # Apply length filtering
                    if seq_length < max_length:
                        out.write(f'>{header}\n{sequences}\n')
                        cleaning_stats["sequences_added_with_limit"] += 1
                    else:
                        cleaning_stats["sequences_removed"] += 1
                else:
                    # No length filtering
                    out.write(f'>{header}\n{sequences}\n')
                    cleaning_stats["sequences_added_no_limit"] += 1
        
        return cleaning_stats
        
    except Exception as e:
        logger.error(f"Error processing sequences: {e}")
        raise


def _create_cleaning_result(
    species_name: str, 
    input_file: Path, 
    output_file: Path, 
    cleaning_stats: Dict[str, int], 
    length_filter_enabled: bool
) -> Dict[str, Any]:
    """
    Create a comprehensive result dictionary for the cleaning operation.
    
    Args:
        species_name: Name of the species
        input_file: Path to input file
        output_file: Path to output file
        cleaning_stats: Cleaning statistics
        length_filter_enabled: Whether length filtering was enabled
        
    Returns:
        Dictionary containing cleaning results
    """
    # Log cleaning statistics
    _log_cleaning_statistics(species_name, cleaning_stats, length_filter_enabled)
    
    # Create result dictionary
    result = {
        "species": species_name,
        "input_file": input_file.name,
        "output_file": output_file.name,
        "total_sequences_processed": cleaning_stats["total_sequences_processed"],
        "sequences_removed": cleaning_stats["sequences_removed"],
        "sequences_retained": (cleaning_stats["sequences_added_with_limit"] + 
                              cleaning_stats["sequences_added_no_limit"]),
        "length_filter_applied": length_filter_enabled,
        "cleaning_successful": True
    }
    
    # Add filter-specific statistics
    if length_filter_enabled:
        result["sequences_added_with_limit"] = cleaning_stats["sequences_added_with_limit"]
        result["removal_rate"] = (cleaning_stats["sequences_removed"] / 
                                 cleaning_stats["total_sequences_processed"] * 100) if cleaning_stats["total_sequences_processed"] > 0 else 0
    else:
        result["sequences_added_no_limit"] = cleaning_stats["sequences_added_no_limit"]
        result["removal_rate"] = 0
    
    return result


def _log_cleaning_statistics(
    species_name: str, 
    cleaning_stats: Dict[str, int], 
    length_filter_enabled: bool
) -> None:
    """
    Log cleaning statistics for a species.
    
    Args:
        species_name: Name of the species
        cleaning_stats: Cleaning statistics dictionary
        length_filter_enabled: Whether length filtering was enabled
    """
    logger.info("-" * 80)
    logger.info(f"Cleaning statistics for {species_name}")
    
    if length_filter_enabled:
        if cleaning_stats["sequences_added_with_limit"] > 0:
            logger.info(f"Sequences retained with length filter: {cleaning_stats['sequences_added_with_limit']}")
        
        if cleaning_stats["sequences_removed"] > 0:
            logger.info(f"Sequences removed by length filter: {cleaning_stats['sequences_removed']}")
        else:
            logger.info("No sequences removed by length filter")
    else:
        if cleaning_stats["sequences_added_no_limit"] > 0:
            logger.info(f"Sequences retained (no length filter): {cleaning_stats['sequences_added_no_limit']}")
    
    logger.info(f"Total sequences processed: {cleaning_stats['total_sequences_processed']}")
    logger.info("-" * 80)


def _generate_cleaning_summary(results: List[Dict[str, Any]]) -> None:
    """
    Generate summary report for all cleaning operations.
    
    Args:
        results: List of cleaning result dictionaries
    """
    logger.info("=" * 80)
    logger.info("Generating sequence cleaning summary")
    
    # Ensure output directory exists
    results_dir = Path("./sequence_cleaning")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Create results dataframe and save to CSV
    output_path = results_dir / "sequence_cleaning_statistics.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    logger.info(f"Sequence cleaning results saved to {output_path}")
    
    # Generate summary statistics
    total_species = len(results)
    total_sequences_processed = sum(r.get("total_sequences_processed", 0) for r in results)
    total_sequences_removed = sum(r.get("sequences_removed", 0) for r in results)
    total_sequences_retained = sum(r.get("sequences_retained", 0) for r in results)
    
    # Calculate species with/without length filtering
    species_with_filter = len([r for r in results if r.get("length_filter_applied", False)])
    species_without_filter = total_species - species_with_filter
    
    # Calculate average removal rate for species with filtering
    species_with_removal = [r for r in results if r.get("sequences_removed", 0) > 0]
    avg_removal_rate = (sum(r.get("removal_rate", 0) for r in species_with_removal) / 
                       len(species_with_removal)) if species_with_removal else 0
    
    logger.info("=" * 80)
    logger.info("Sequence Cleaning Analysis Summary")
    logger.info(f"Total species processed: {total_species}")
    logger.info(f"Species with length filtering: {species_with_filter}")
    logger.info(f"Species without length filtering: {species_without_filter}")
    logger.info(f"Total sequences processed: {total_sequences_processed}")
    logger.info(f"Total sequences removed: {total_sequences_removed}")
    logger.info(f"Total sequences retained: {total_sequences_retained}")
    if avg_removal_rate > 0:
        logger.info(f"Average removal rate: {avg_removal_rate:.2f}%")
    logger.info("=" * 80)
