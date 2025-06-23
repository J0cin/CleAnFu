import logging
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config_loader import Config
from src.comparison_stats_utils import merge_two_ahrd_files, parse_go_terms, compare_go_sets
from src.error_utils import handle_graceful_exit, PipelineError

logger = logging.getLogger(__name__)


@handle_graceful_exit
def run_comparison_analysis(config: Config, input_files: List[Tuple[Path, Path]], min_list_threshold: int) -> None:
    """
    Run the complete AHRD file merging and GO term comparison analysis pipeline.
    Compares merged AHRD results (SwissProt + TrEMBL) against FANTASIA results.
    
    Args:
        config: Pipeline configuration object
        input_files: List of tuples containing (directory_path, file_path) for analysis
        min_list_threshold: Minimum threshold for GO term list processing
    """
    logger.info("=" * 80)
    logger.info("Starting merged AHRD analysis (SwissProt + TrEMBL) with FANTASIA comparison")
    logger.info("=" * 80)
    
    # Check input files
    if not input_files:
        error_msg = "No input files provided for comparison analysis"
        logger.error(error_msg)
        raise PipelineError(error_msg)
    
    # Initialize result collections
    ahrd_results = []
    fantasia_results = []
    comparison_results = []
    
    # Get file identifiers from configuration
    try:
        sws_identifier = config.stats.Files_SWS_identifier
        tre_identifier = config.stats.Files_TrE_identifier
        fantasia_identifier = config.stats.Files_FAN_identifier
        logger.info(f"File identifiers - SWS: '{sws_identifier}', TrE: '{tre_identifier}', FANTASIA: '{fantasia_identifier}'")
    except AttributeError as e:
        error_msg = f"Configuration error reading file identifiers: {e}"
        logger.error(error_msg)
        raise PipelineError(error_msg)
    
    # Process each directory
    for dir_path, file_path in input_files:
        species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
        logger.info(f"Processing species: {species_name}")
        
        try:
            result = process_comparison_file(
                dir_path, file_path, config, species_name, 
                sws_identifier, tre_identifier, fantasia_identifier, 
                min_list_threshold
            )
            
            if result is not None:
                ahrd_results.append(result["ahrd_stats"])
                fantasia_results.append(result["fantasia_stats"])
                comparison_results.append(result["comparison_stats"])
            
        except Exception as e:
            logger.error(f"Error processing {species_name}: {e}")
            # Add failure records
            ahrd_results.append({"species": species_name, "status": "processing_failed"})
            fantasia_results.append({"species": species_name, "status": "processing_failed"})
            comparison_results.append({"species": species_name, "status": "processing_failed"})
    
    # Generate summary reports
    if ahrd_results or fantasia_results or comparison_results:
        _generate_summary_reports(ahrd_results, fantasia_results, comparison_results)
    else:
        logger.warning("No valid results obtained from any species")
    
    logger.info("=" * 80)
    logger.info("Merged AHRD analysis with FANTASIA comparison completed successfully")
    logger.info("=" * 80)


def process_comparison_file(
    dir_path: Path, 
    file_path: Path, 
    config: Config, 
    species_name: str,
    sws_identifier: str,
    tre_identifier: str, 
    fantasia_identifier: str,
    min_list_threshold: int
) -> Dict[str, Any]:
    """
    Process a single species directory through the comparison analysis pipeline.
    
    Args:
        dir_path: Directory path containing the analysis files
        file_path: Path to the protein FASTA file
        config: Pipeline configuration
        species_name: Name of the species being analyzed
        sws_identifier: SwissProt file identifier
        tre_identifier: TrEMBL file identifier
        fantasia_identifier: FANTASIA file identifier
        min_list_threshold: Minimum threshold for GO term processing
        
    Returns:
        Dictionary containing AHRD, FANTASIA, and comparison statistics or None if processing failed
    """
    logger.info(f"Processing comparison analysis for: {species_name}")
    
    try:
        # Validate directories
        fantasia_dir_path = dir_path / config.fantasia.fantasia_dir
        
        if not dir_path.is_dir():
            logger.warning(f"Analysis directory not found, skipping: {dir_path}")
            return None
            
        if not fantasia_dir_path.is_dir():
            logger.warning(f"FANTASIA directory not found, skipping: {fantasia_dir_path}")
            return None
        
        # Find required files
        sws_files = list(dir_path.glob(f"*{sws_identifier}*"))
        tre_files = list(dir_path.glob(f"*{tre_identifier}*"))
        fantasia_files = list(fantasia_dir_path.glob(f"*{fantasia_identifier}*"))
        
        # Validate file discovery
        if not _validate_files(sws_files, tre_files, fantasia_files, species_name, 
                              sws_identifier, tre_identifier, fantasia_identifier):
            return None
        
        sws_file_path = sws_files[0]
        tre_file_path = tre_files[0]
        fantasia_file_path = fantasia_files[0]
        
        # Step 1: Process merged AHRD analysis
        ahrd_stats, ahrd_unique_go = _process_merged_ahrd(
            sws_file_path, tre_file_path, dir_path, species_name, config, min_list_threshold
        )
        
        # Step 2: Process FANTASIA analysis
        fantasia_stats, fantasia_unique_go = _process_fantasia_analysis(
            fantasia_file_path, species_name, min_list_threshold
        )
        
        # Step 3: Compare GO term sets
        comparison_stats = _compare_go_term_sets(
            ahrd_unique_go, fantasia_unique_go, species_name, dir_path
        )
        
        # Step 4: Combine results
        final_result = _combine_analysis_results(
            species_name, ahrd_stats, fantasia_stats, comparison_stats
        )
        
        logger.info(f"Successfully completed comparison analysis for {species_name}")
        return final_result
        
    except Exception as e:
        logger.error(f"Error in comparison analysis for {species_name}: {e}")
        return None


def _validate_files(
    sws_files: List[Path], 
    tre_files: List[Path], 
    fantasia_files: List[Path],
    species_name: str,
    sws_identifier: str,
    tre_identifier: str,
    fantasia_identifier: str
) -> bool:
    """
    Validate that required files are found and handle multiple file warnings.
    
    Returns:
        True if validation passes, False otherwise
    """
    if not sws_files:
        logger.warning(f"[{species_name}] No SwissProt file found matching '*{sws_identifier}*'")
        return False
        
    if not tre_files:
        logger.warning(f"[{species_name}] No TrEMBL file found matching '*{tre_identifier}*'")
        return False
        
    if not fantasia_files:
        logger.warning(f"[{species_name}] No FANTASIA file found matching '*{fantasia_identifier}*'")
        return False
    
    # Handle multiple files
    if len(sws_files) > 1:
        logger.warning(f"[{species_name}] Multiple SwissProt files found, using: {sws_files[0]}")
        
    if len(tre_files) > 1:
        logger.warning(f"[{species_name}] Multiple TrEMBL files found, using: {tre_files[0]}")
        
    if len(fantasia_files) > 1:
        logger.warning(f"[{species_name}] Multiple FANTASIA files found, using: {fantasia_files[0]}")
    
    return True


def _process_merged_ahrd(
    sws_file_path: Path, 
    tre_file_path: Path, 
    dir_path: Path, 
    species_name: str, 
    config: Config, 
    min_list_threshold: int
) -> Tuple[Dict[str, Any], set]:
    """
    Process merged AHRD analysis (SwissProt + TrEMBL).
    
    Returns:
        Tuple of (statistics_dict, unique_go_terms_set)
    """
    logger.info(f"[{species_name}] Running merged AHRD analysis")
    logger.info(f"[{species_name}] Files - SWS: {sws_file_path.name}, TrE: {tre_file_path.name}")
    
    # Run merge operation
    merge_result = merge_two_ahrd_files(sws_file_path, tre_file_path, dir_path, species_name, config)
    
    if not merge_result.get("status"):
        logger.error(f"[{species_name}] Merge operation failed: {merge_result.get('error')}")
        return {"species": species_name, "status": "merge_failed"}, set()
    
    # Parse GO terms from merged result
    try:
        go_stats, unique_go_terms = parse_go_terms(merge_result, species_name, min_list_threshold)
        
        if go_stats:
            logger.info(f"[{species_name}] Successfully parsed merged AHRD GO terms")
            go_stats["species"] = species_name
            return go_stats, unique_go_terms
        else:
            logger.warning(f"[{species_name}] Failed to extract GO terms from merged AHRD")
            return {"species": species_name, "status": "go_parse_failed"}, set()
            
    except Exception as e:
        logger.error(f"[{species_name}] Error parsing merged AHRD GO terms: {e}")
        return {"species": species_name, "status": "go_parse_exception"}, set()


def _process_fantasia_analysis(
    fantasia_file_path: Path, 
    species_name: str, 
    min_list_threshold: int
) -> Tuple[Dict[str, Any], set]:
    """
    Process FANTASIA analysis results.
    
    Returns:
        Tuple of (statistics_dict, unique_go_terms_set)
    """
    logger.info(f"[{species_name}] Processing FANTASIA analysis")
    logger.info(f"[{species_name}] File: {fantasia_file_path.name}")
    
    if not fantasia_file_path.exists():
        logger.error(f"[{species_name}] FANTASIA file does not exist: {fantasia_file_path}")
        return {"species": species_name, "status": "file_not_found"}, set()
    
    fantasia_result = {
        "output_path": fantasia_file_path,
        "status": True
    }
    
    try:
        go_stats, unique_go_terms = parse_go_terms(fantasia_result, species_name, min_list_threshold)
        
        if go_stats:
            logger.info(f"[{species_name}] Successfully parsed FANTASIA GO terms")
            go_stats["species"] = species_name
            return go_stats, unique_go_terms
        else:
            logger.warning(f"[{species_name}] Failed to extract GO terms from FANTASIA")
            return {"species": species_name, "status": "go_parse_failed"}, set()
            
    except Exception as e:
        logger.error(f"[{species_name}] Error parsing FANTASIA GO terms: {e}")
        return {"species": species_name, "status": "go_parse_exception"}, set()


def _compare_go_term_sets(
    ahrd_go_terms: set, 
    fantasia_go_terms: set, 
    species_name: str, 
    dir_path: Path
) -> Dict[str, Any]:
    """
    Compare GO term sets between merged AHRD and FANTASIA results.
    
    Returns:
        Dictionary containing comparison statistics
    """
    logger.info(f"[{species_name}] Comparing GO term sets")
    
    try:
        comparison_stats = compare_go_sets(
            ahrd_go_terms, fantasia_go_terms, 
            "Merged_AHRD", "FANTASIA", dir_path
        )
        logger.info(f"[{species_name}] GO term comparison completed")
        return comparison_stats
        
    except Exception as e:
        logger.error(f"[{species_name}] Error in GO term comparison: {e}")
        return {"species": species_name, "status": "comparison_failed"}


def _combine_analysis_results(
    species_name: str,
    ahrd_stats: Dict[str, Any],
    fantasia_stats: Dict[str, Any],
    comparison_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine all analysis results into a final result dictionary.
    
    Returns:
        Dictionary containing all combined results
    """
    # Create comprehensive result structure
    final_result = {
        "ahrd_stats": ahrd_stats,
        "fantasia_stats": fantasia_stats,
        "comparison_stats": {
            "species": species_name,
            **comparison_stats
        }
    }
    
    # Add selected statistics to comparison results for easy access
    if ahrd_stats and "status" not in ahrd_stats:
        final_result["comparison_stats"]["ahrd_mean_go"] = ahrd_stats.get("mean_go_per_annotated_gene", 0)
    
    if fantasia_stats and "status" not in fantasia_stats:
        final_result["comparison_stats"]["fantasia_mean_go"] = fantasia_stats.get("mean_go_per_annotated_gene", 0)
    
    return final_result


def _generate_summary_reports(
    ahrd_results: List[Dict[str, Any]], 
    fantasia_results: List[Dict[str, Any]], 
    comparison_results: List[Dict[str, Any]]
) -> None:
    """
    Generate summary reports for all analysis results.
    """
    logger.info("=" * 80)
    logger.info("Generating analysis summary reports")
    
    # Generate AHRD summary
    if ahrd_results:
        _generate_ahrd_summary(ahrd_results)
    
    # Generate FANTASIA summary  
    if fantasia_results:
        _generate_fantasia_summary(fantasia_results)
    
    # Generate comparison summary
    if comparison_results:
        _generate_comparison_summary(comparison_results)


def _generate_ahrd_summary(ahrd_results: List[Dict[str, Any]]) -> None:
    """Generate merged AHRD analysis summary."""
    logger.info("Generating merged AHRD analysis summary")
    
    try:
        results_df = pd.DataFrame(ahrd_results)
        
        # Save to CSV
        output_dir = Path("./analysis_ahrd_summary")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "merged_ahrd_go_statistics.csv"
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"Merged AHRD GO statistics saved to: {output_path}")
        
        # Generate console summary
        successful_results = [r for r in ahrd_results if isinstance(r.get("total_go_assignments"), (int, float))]
        
        if successful_results:
            total_species = len(results_df['species'].unique())
            total_go_terms = sum(r.get("total_go_assignments", 0) for r in successful_results)
            total_genes_with_go = sum(r.get("proteins_with_go", 0) for r in successful_results)
            
            logger.info("-" * 80)
            logger.info("Merged AHRD Analysis Summary")
            logger.info(f"Total species processed: {total_species}")
            logger.info(f"Total GO terms assigned: {total_go_terms}")
            logger.info(f"Total genes with GO terms: {total_genes_with_go}")
            logger.info("-" * 80)
        
    except Exception as e:
        logger.error(f"Error generating AHRD summary: {e}")


def _generate_fantasia_summary(fantasia_results: List[Dict[str, Any]]) -> None:
    """Generate FANTASIA analysis summary."""
    logger.info("Generating FANTASIA analysis summary")
    
    try:
        results_df = pd.DataFrame(fantasia_results)
        
        # Save to CSV
        output_dir = Path("./analysis_fantasia_summary")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "fantasia_go_statistics.csv"
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"FANTASIA GO statistics saved to: {output_path}")
        
        # Generate console summary
        successful_results = [r for r in fantasia_results if isinstance(r.get("total_go_assignments"), (int, float))]
        
        if successful_results:
            total_species = len(results_df['species'].unique())
            total_go_terms = sum(r.get("total_go_assignments", 0) for r in successful_results)
            total_genes_with_go = sum(r.get("proteins_with_go", 0) for r in successful_results)
            
            logger.info("-" * 80)
            logger.info("FANTASIA Analysis Summary")
            logger.info(f"Total species processed: {total_species}")
            logger.info(f"Total GO terms assigned: {total_go_terms}")
            logger.info(f"Total genes with GO terms: {total_genes_with_go}")
            logger.info("-" * 80)
        
    except Exception as e:
        logger.error(f"Error generating FANTASIA summary: {e}")


def _generate_comparison_summary(comparison_results: List[Dict[str, Any]]) -> None:
    """Generate comparison analysis summary."""
    logger.info("Generating comparison analysis summary")
    
    try:
        results_df = pd.DataFrame(comparison_results)
        
        # Save to CSV
        output_dir = Path("./analysis_comparison_summary")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "comparison_unique_go_statistics.csv"
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"Comparison GO statistics saved to: {output_path}")
        
        # Generate console summary
        successful_results = [r for r in comparison_results if isinstance(r.get("common_go_terms"), (int, float))]
        
        if successful_results:
            total_species = len(results_df['species'].unique())
            total_ahrd_mean_go = sum(r.get("ahrd_mean_go", 0) for r in successful_results)
            total_fantasia_mean_go = sum(r.get("fantasia_mean_go", 0) for r in successful_results)
            
            logger.info("-" * 80)
            logger.info("Comparison Analysis Summary")
            logger.info(f"Total species processed: {total_species}")
            logger.info(f"Total mean GO terms AHRD: {total_ahrd_mean_go}")
            logger.info(f"Total mean GO terms FANTASIA: {total_fantasia_mean_go}")
            logger.info("-" * 80)
        
    except Exception as e:
        logger.error(f"Error generating comparison summary: {e}")
