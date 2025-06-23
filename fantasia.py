import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

from src.fantasia_utils import run_prepare_files, run_fantasia_command, get_fantasia_GO, parse_go_terms
from src.error_utils import handle_graceful_exit, PipelineError
from src.config_loader import Config

logger = logging.getLogger(__name__)

@handle_graceful_exit
def run_fantasia_analysis(config: Config, input_files: List[Tuple[Path, Path]]) -> None:
    """
    Run the complete FANTASIA analysis pipeline.
    
    Args:
        config: Pipeline configuration
        input_files: List of tuples containing (directory_path, file_path) of protein FASTA files
    """
    logger.info("=" * 80)
    logger.info("Starting FANTASIA analysis pipeline")
    logger.info("=" * 80)
    
    # Check input files
    if not input_files:
        error_msg = "No input files provided for FANTASIA analysis"
        logger.error(error_msg)
        raise PipelineError(error_msg)
    
    results = []
    
    # Process each input file through the FANTASIA pipeline
    for dir_path, file_path in input_files:
        species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
        logger.info(f'FANTASIA analysis for {species_name}')
        try:
            result = process_fantasia_file(dir_path, file_path, config)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.error(f"run_fantasia_analysis, FANTASIA error: {e} for {species_name}")
    
    # Generate summary report if there are results
    if results:
        # Ensure output directory exists
        results_dir = Path(f'./{config.fantasia.fantasia_results_dir}')
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Create results dataframe and save to CSV
        output_path = results_dir / "fantasia_go_statistics.csv"
        pd.DataFrame(results).to_csv(output_path, index=False)
        logger.info(f"FANTASIA results saved to {output_path}")
        
        # Generate summary statistics
        total_species = len(results)
        total_go_terms = sum(r.get("total_go_terms", 0) for r in results)
        total_genes = sum(r.get("genes_with_go", 0) for r in results)
        
        logger.info("=" * 80)
        logger.info("FANTASIA Analysis Summary")
        logger.info(f"Total species analyzed: {total_species}")
        logger.info(f"Total GO terms assigned: {total_go_terms}")
        logger.info(f"Total genes with GO terms: {total_genes}")
        logger.info("=" * 80)
    else:
        logger.warning("No valid FANTASIA results obtained")


def process_fantasia_file(dir_path: Path, file_path: Path, config: Config) -> Dict[str, Any]:
    """
    Process a single file through the FANTASIA pipeline.
    
    Args:
        dir_path: Directory path containing the file
        file_path: Path to the protein FASTA file
        config: Pipeline configuration
        
    Returns:
        Dictionary with analysis results or None if processing failed
    """
    logger.info(f"Processing file: {file_path}")
    species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
    
    try:
        # Step 1: Prepare files
        prep_result = run_prepare_files(dir_path, file_path, config)
        
        if not prep_result["status"]:
            logger.warning(f"Skipping FANTASIA analysis for {species_name} due to failed preparation")
            return None
        
        # Extract values from result
        specie = prep_result["specie"]
        prefix = prep_result["prefix"]
        fantasia_dir = prep_result["fantasia_dir"]
        
        # Step 2: Run FANTASIA analysis
        analysis_result = run_fantasia_command(fantasia_dir, prefix, specie, prep_result["status"], config)
        
        if not analysis_result["status"]:
            logger.warning(f"FANTASIA analysis failed for {species_name}")
            return None
        
        # Step 3: Extract GO terms
        go_result = get_fantasia_GO(config, prefix, file_path, specie, analysis_result["status"], fantasia_dir)
        
        if not go_result["status"]:
            logger.warning(f"Failed to extract GO terms for {species_name}")
            return None
        
        # Step 4: Parse GO terms
        logger.info(f'FANTASIA GO terms txt generated')
        go_stats = parse_go_terms(go_result)
        
        if go_stats:
            logger.info(f"Successfully processed {species_name}")
            # Add file information to results
            go_stats["file_name"] = file_path.name
            go_stats["species"] = species_name
            return go_stats
        else:
            logger.warning(f"Failed to parse GO terms for {species_name}")
            return None
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None