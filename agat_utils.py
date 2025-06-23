import re
import logging
from pathlib import Path
from typing import Dict, Optional, List

from src.error_utils import PipelineError, handle_graceful_exit
from src.config_loader import Config

logger = logging.getLogger(__name__)

# Pattern cache to avoid regex recompilation in bioinformatics workflows
_pattern_cache: Dict[str, re.Pattern] = {}


def get_compiled_pattern(pattern: str) -> re.Pattern:
    """
    Retrieve or compile regex pattern with caching for performance optimization.
    
    Implements pattern caching to improve performance in large-scale bioinformatics
    analyses where the same patterns are used repeatedly across multiple files.
    
    Args:
        pattern: Regular expression pattern string
        
    Returns:
        Compiled regex pattern object with case-insensitive matching
    """
    if pattern not in _pattern_cache:
        _pattern_cache[pattern] = re.compile(pattern, re.IGNORECASE)
    return _pattern_cache[pattern]


@handle_graceful_exit
def extract_agat_statistics(file_path: Path, config: Config) -> Dict[str, Optional[str]]:
    """
    Extract genome annotation statistics from AGAT output using configured regex patterns.
    
    Parses AGAT (Another Genome Annotation Toolkit) output files to extract
    key genomic statistics such as gene counts, feature distributions, and
    annotation quality metrics. Uses configurable regex patterns for flexibility
    across different AGAT output formats.
    
    Args:
        file_path: Path to AGAT statistics output file
        config: Pipeline configuration containing regex patterns for extraction
        
    Returns:
        Dictionary containing extracted statistics with original file path
        
    Raises:
        FileNotFoundError: If the AGAT output file does not exist
        PipelineError: For parsing errors or configuration issues
    """
    logger.info(f"Extracting AGAT statistics from: {file_path}")
    
    statistics = {'file': str(file_path)}
    
    # Validate file existence
    if not file_path.exists():
        error_msg = f"AGAT output file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Retrieve extraction patterns from configuration
    extraction_patterns = config.agat.patterns
    if not extraction_patterns:
        logger.warning("No extraction patterns defined in configuration for AGAT analysis")
        return statistics
    
    try:
        # Read AGAT output file content
        file_content = file_path.read_text(encoding='utf-8')
        logger.debug(f"Successfully read AGAT file content ({len(file_content)} characters)")
        
        # Extract statistics using configured patterns
        for statistic_key, pattern_string in extraction_patterns.items():
            compiled_pattern = get_compiled_pattern(pattern_string)
            pattern_match = compiled_pattern.search(file_content)
            
            if pattern_match and pattern_match.groups():
                extracted_value = pattern_match.group(1).strip()
                statistics[statistic_key] = extracted_value
                logger.debug(f"Extracted {statistic_key}: {extracted_value}")
            else:
                statistics[statistic_key] = None
                logger.warning(f"No match found for pattern '{statistic_key}': {pattern_string}")
                
        logger.info(f"Successfully extracted {len([v for v in statistics.values() if v is not None]) - 1} "
                   f"statistics from AGAT output")
        
    except Exception as e:
        error_msg = f"Error processing AGAT file {file_path}: {e}"
        logger.error(error_msg)
        raise PipelineError(error_msg, context="extract_agat_statistics")
        
    return statistics


def validate_agat_statistics(statistics: Dict[str, Optional[str]], required_keys: List[str]) -> bool:
    """
    Validate that extracted AGAT statistics contain all required annotation metrics.
    
    Performs quality control validation to ensure that essential genomic
    statistics have been successfully extracted from AGAT output. This is
    critical for downstream comparative genomics and annotation quality assessment.
    
    Args:
        statistics: Dictionary of extracted AGAT statistics
        required_keys: List of required statistic keys for validation
        
    Returns:
        True if all required statistics are present and non-null, False otherwise
    """
    logger.debug(f"Validating AGAT statistics for {len(required_keys)} required keys")
    
    missing_keys = []
    for required_key in required_keys:
        if statistics.get(required_key) is None:
            missing_keys.append(required_key)
    
    if missing_keys:
        logger.warning(f"Missing required AGAT statistics: {', '.join(missing_keys)}")
        return False
    
    logger.info("All required AGAT statistics successfully validated")
    return True


def parse_agat_feature_counts(statistics: Dict[str, Optional[str]]) -> Dict[str, int]:
    """
    Parse and convert AGAT feature count statistics to integer values.
    
    Converts string-based feature counts from AGAT output to integers
    for numerical analysis and comparative genomics studies. Handles
    common formatting issues in bioinformatics output files.
    
    Args:
        statistics: Dictionary containing raw AGAT statistics
        
    Returns:
        Dictionary with parsed integer feature counts
    """
    feature_counts = {}
    count_keys = ['gene_count', 'mrna_count', 'exon_count', 'cds_count']
    
    logger.debug("Parsing AGAT feature counts to integers")
    
    for count_key in count_keys:
        raw_value = statistics.get(count_key)
        if raw_value is not None:
            try:
                # Handle common formatting: remove commas, whitespace
                cleaned_value = re.sub(r'[,\s]', '', str(raw_value))
                feature_counts[count_key] = int(cleaned_value)
                logger.debug(f"Parsed {count_key}: {feature_counts[count_key]}")
            except ValueError as e:
                logger.warning(f"Could not parse {count_key} value '{raw_value}': {e}")
                feature_counts[count_key] = 0
        else:
            logger.warning(f"Missing value for {count_key}")
            feature_counts[count_key] = 0
    
    return feature_counts


def generate_agat_summary(statistics_list: List[Dict[str, Optional[str]]]) -> Dict[str, any]:
    """
    Generate comprehensive summary statistics from multiple AGAT analyses.
    
    Aggregates statistics from multiple genome annotations to provide
    comparative genomics insights and quality metrics across datasets.
    
    Args:
        statistics_list: List of AGAT statistics dictionaries from multiple files
        
    Returns:
        Dictionary containing summary statistics and metrics
    """
    logger.info(f"Generating AGAT summary for {len(statistics_list)} annotation files")
    
    if not statistics_list:
        logger.warning("No AGAT statistics provided for summary generation")
        return {}
    
    # Extract all feature counts
    all_feature_counts = []
    for stats in statistics_list:
        feature_counts = parse_agat_feature_counts(stats)
        if any(count > 0 for count in feature_counts.values()):
            all_feature_counts.append(feature_counts)
    
    if not all_feature_counts:
        logger.warning("No valid feature counts found in AGAT statistics")
        return {}
    
    # Calculate summary metrics
    summary = {
        "total_files_processed": len(statistics_list),
        "files_with_valid_counts": len(all_feature_counts),
        "average_gene_count": sum(fc.get('gene_count', 0) for fc in all_feature_counts) / len(all_feature_counts),
        "total_genes_across_all": sum(fc.get('gene_count', 0) for fc in all_feature_counts),
        "average_mrna_count": sum(fc.get('mrna_count', 0) for fc in all_feature_counts) / len(all_feature_counts),
        "total_mrnas_across_all": sum(fc.get('mrna_count', 0) for fc in all_feature_counts)
    }
    
    logger.info(f"AGAT summary: {summary['files_with_valid_counts']} valid files, "
               f"average {summary['average_gene_count']:.0f} genes per genome")
    
    return summary