import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.diamond_utils import run_diamond_blastp, create_ahrd_config, run_ahrd, parse_ahrd_go_terms
from src.error_utils import handle_graceful_exit, PipelineError
from src.config_loader import Config

logger = logging.getLogger(__name__)

@handle_graceful_exit
def run_diamond_ahrd_analysis(config: Config, input_files: List[Tuple[Path, Path]]) -> None:
    """
    Run the complete DIAMOND-AHRD analysis pipeline with parallel processing.
    Uses ProcessPoolExecutor with a maximum of 3 concurrent processes.
    
    Args:
        config: Pipeline configuration
        input_files: List of tuples containing (directory_path, file_path, species_name) of protein FASTA files
    """
    logger.info("=" * 80)
    logger.info("Starting parallel DIAMOND-AHRD analysis pipeline")
    logger.info("=" * 80)
    
    # Check input files
    if not input_files:
        error_msg = "No input files provided for DIAMOND-AHRD analysis"
        logger.error(error_msg)
        raise PipelineError(error_msg)
    
    results = []
    
    # Ejecutar en modo serial o paralelo según cantidad de archivos
    if len(input_files) == 1:
        dir_path, file_path = input_files[0]
        species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
        logger.info(f'DIAMOND-AHRD analysis for {species_name}')
        try:
            result = process_diamond_file(dir_path, file_path, config)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.error(f"run_diamond_ahrd_analysis, DIAMOND-AHRD error: {e}")
    elif len(input_files) > 1 and str(config.diamond.db_type) == "Trembl":
        for dir_path, file_path in input_files:
            species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
            logger.info(f'DIAMOND-AHRD analysis for {species_name}')
            try:
                result = process_diamond_file(dir_path, file_path, config)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.error(f"run_diamond_ahrd_analysis, DIAMOND-AHRD error: {e} for {species_name}")
            
    else:
        # Para paralelismo entre archivos con máximo 3 trabajadores
        max_workers = min(len(input_files), 3)
        logger.info(f"Using ProcessPoolExecutor with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_diamond_file, dir_path, file_path, config): (dir_path, file_path)
                for dir_path, file_path in input_files
            }

            for future in as_completed(future_to_file):
                dir_path, file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"run_diamond_ahrd_analysis multiple, File error {file_path.name}: {e}")
    
    # Generate summary report if there are results
    if results:
        # Ensure output directory exists
        results_dir = Path("./diamond_ahrd")
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Create results dataframe and save to CSV
        output_path = results_dir / "diamond_ahrd_go_statistics.csv"
        pd.DataFrame(results).to_csv(output_path, index=False)
        logger.info(f"DIAMOND-AHRD results saved to {output_path}")
        
        # Generate summary statistics
        total_species = len(results)
        total_go_terms = sum(r.get("total_go_terms", 0) for r in results)
        total_genes = sum(r.get("genes_with_go", 0) for r in results)
        
        logger.info("=" * 80)
        logger.info("DIAMOND-AHRD Analysis Summary")
        logger.info(f"Total species analyzed: {total_species}")
        logger.info(f"Total GO terms assigned: {total_go_terms}")
        logger.info(f"Total genes with GO terms: {total_genes}")
        logger.info("=" * 80)
    else:
        logger.warning("No valid DIAMOND-AHRD results obtained")


def process_diamond_file(dir_path: Path, file_path: Path, config: Config) -> Dict[str, Any]:
    """
    Process a single file through the DIAMOND-AHRD pipeline.
    
    Args:
        dir_path: Directory path containing the file
        file_path: Path to the protein FASTA file
        config: Pipeline configuration
        species_name: Name of the species being analyzed
        
    Returns:
        Dictionary with analysis results or None if processing failed
    """
    logger.info(f"Processing file: {file_path}")
    species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
    try:
        # Step 1: Run DIAMOND BLASTP
        diamond_result = run_diamond_blastp(file_path, config)
        
        if not diamond_result["status"]:
            logger.warning(f"Skipping AHRD analysis for {species_name} due to failed DIAMOND search")
            return None
            
        # Step 2: Create AHRD configuration
        ahrd_config_result = create_ahrd_config(file_path, diamond_result, config)
        
        if not ahrd_config_result["status"]:
            logger.warning(f"Skipping AHRD analysis for {species_name} due to failed configuration creation")
            return None
            
        # Step 3: Run AHRD analysis
        ahrd_result = run_ahrd(ahrd_config_result, config)
        
        if not ahrd_result["status"]:
            logger.warning(f"AHRD analysis failed for {species_name}")
            return None
        
        # Step 4: Parse GO terms
        go_stats = parse_ahrd_go_terms(ahrd_result)
        
        if go_stats:
            logger.info(f"Successfully processed {species_name}")
            # Add file information to results
            go_stats["file_name"] = file_path.name
            go_stats["species"] = species_name
            return go_stats
        else:
            logger.warning(f"Failed to extract GO terms for {species_name}")
            return None
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None
