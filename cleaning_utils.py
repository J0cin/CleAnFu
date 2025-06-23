import logging
from pathlib import Path
from typing import Dict, List, Tuple, Generator, Set, Any

from src.error_utils import handle_graceful_exit, PipelineError
from src.config_loader import Config

logger = logging.getLogger(__name__)


@handle_graceful_exit
def get_unique_seq_treated(input_file: Path) -> Dict[str, Any]:
    """
    Process protein sequences from FASTA file with deduplication and quality control.
    
    Extracts unique protein sequences from FASTA files with comprehensive quality
    control including duplicate removal, non-standard amino acid normalization,
    and sequence validation. Optimized for bioinformatics workflows requiring
    high-quality protein datasets for downstream analysis.
    
    Args:
        input_file: Path to input protein FASTA file for processing
        
    Returns:
        Dictionary containing processed sequences and processing statistics
        
    Yields:
        Generator of tuples containing (header, sequence) for unique sequences
    """
    logger.info("=" * 80)
    logger.info("Processing protein sequences with quality control and deduplication")
    logger.info("-" * 80)
    
    try:
        # Validate input file existence
        if not input_file.exists():
            logger.error(f"Input FASTA file not found: {input_file}")
            return {
                "status": False,
                "error": f"Input file does not exist: {input_file}"
            }
        
        logger.info(f"Processing sequences from: {input_file}")
        
        # Initialize processing variables
        header = None
        processed_names: Set[str] = set()
        is_unique = False
        sequence_lines = []
        valid_amino_acids = set('ARNDCEQGHILKMFPSTWYVX')  # Standard amino acid alphabet
        
        # Processing statistics
        total_sequences = 0
        unique_sequences = 0
        duplicate_sequences = 0
        non_standard_aa_replacements = 0
        
        def sequence_generator():
            nonlocal header, is_unique, sequence_lines, total_sequences
            nonlocal unique_sequences, duplicate_sequences, non_standard_aa_replacements
            
            with open(input_file, 'r', encoding='utf-8') as fasta_file:
                for line in fasta_file:
                    line = line.strip()
                    
                    # Process FASTA header lines
                    if line.startswith('>'):
                        # Yield previous sequence if it was unique
                        if header is not None and is_unique:
                            yield (header, '\n'.join(sequence_lines))
                            unique_sequences += 1
                        
                        # Extract sequence identifier
                        header = line.split('>')[1]
                        total_sequences += 1
                        
                        # Check for duplicate sequences
                        if header not in processed_names:
                            is_unique = True
                            processed_names.add(header)
                            logger.debug(f"Processing unique sequence: {header}")
                        else:
                            logger.warning(f"Duplicate sequence identifier detected: {header}")
                            is_unique = False
                            duplicate_sequences += 1
                            
                        sequence_lines = []
                        
                    # Process sequence data lines
                    else:
                        if is_unique:
                            # Normalize amino acid sequence
                            normalized_sequence = ''
                            for amino_acid in line:
                                if amino_acid in valid_amino_acids:
                                    normalized_sequence += amino_acid
                                else:
                                    # Replace non-standard amino acids with 'X'
                                    logger.debug(f"Replacing non-standard amino acid '{amino_acid}' with 'X' in {header}")
                                    normalized_sequence += 'X'
                                    non_standard_aa_replacements += 1
                            sequence_lines.append(normalized_sequence)
                
                # Process final sequence in file
                if header is not None and is_unique:
                    yield (header, '\n'.join(sequence_lines))
                    unique_sequences += 1
        
        # Generate processed sequences
        processed_sequences = list(sequence_generator())
        
        # Log processing statistics
        logger.info(f"Sequence processing completed successfully")
        logger.info(f"Total sequences processed: {total_sequences}")
        logger.info(f"Unique sequences retained: {unique_sequences}")
        logger.info(f"Duplicate sequences removed: {duplicate_sequences}")
        logger.info(f"Non-standard amino acids normalized: {non_standard_aa_replacements}")
        
        return {
            "status": True,
            "sequences": processed_sequences,
            "statistics": {
                "total_sequences": total_sequences,
                "unique_sequences": unique_sequences,
                "duplicate_sequences": duplicate_sequences,
                "non_standard_aa_replacements": non_standard_aa_replacements
            },
            "input_file": input_file
        }
        
    except Exception as e:
        logger.error(f"Error processing protein sequences: {e}")
        
        return {
            "status": False,
            "error": str(e),
            "input_file": input_file
        }


@handle_graceful_exit
def get_seq_size(unique_sequences: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Calculate sequence length statistics for processed protein sequences.
    
    Analyzes protein sequence lengths to generate comprehensive statistics
    including individual sequence sizes, distribution metrics, and quality
    assessment data for bioinformatics pipeline validation and reporting.
    
    Args:
        unique_sequences: List of tuples containing (header, sequence) pairs
        
    Returns:
        Dictionary containing sequence size mapping and statistical analysis
    """
    logger.info("=" * 80)
    logger.info("Calculating protein sequence length statistics")
    logger.info("-" * 80)
    
    try:
        # Validate input data
        if not unique_sequences:
            logger.warning("No sequences provided for size analysis")
            return {
                "status": True,
                "sequence_sizes": {},
                "statistics": {
                    "total_sequences": 0,
                    "min_length": 0,
                    "max_length": 0,
                    "average_length": 0
                }
            }
        
        logger.info(f"Analyzing sequence lengths for {len(unique_sequences)} proteins")
        
        # Calculate individual sequence sizes
        sequence_sizes = {}
        sequence_lengths = []
        
        for header, sequence in unique_sequences:
            # Remove newlines and calculate actual sequence length
            clean_sequence = sequence.replace('\n', '').replace('\r', '')
            sequence_length = len(clean_sequence)
            
            # Store sequence size information
            if header not in sequence_sizes:
                sequence_sizes[header] = sequence_length
                sequence_lengths.append(sequence_length)
                logger.debug(f"Sequence {header}: {sequence_length} amino acids")
            else:
                logger.warning(f"Duplicate sequence header in size analysis: {header}")
        
        # Calculate statistical metrics
        total_sequences = len(sequence_lengths)
        min_length = min(sequence_lengths) if sequence_lengths else 0
        max_length = max(sequence_lengths) if sequence_lengths else 0
        average_length = sum(sequence_lengths) / total_sequences if sequence_lengths else 0
        
        # Log statistical summary
        logger.info(f"Sequence length analysis completed successfully")
        logger.info(f"Total sequences analyzed: {total_sequences}")
        logger.info(f"Minimum sequence length: {min_length} amino acids")
        logger.info(f"Maximum sequence length: {max_length} amino acids")
        logger.info(f"Average sequence length: {average_length:.2f} amino acids")
        
        return {
            "status": True,
            "sequence_sizes": sequence_sizes,
            "statistics": {
                "total_sequences": total_sequences,
                "min_length": min_length,
                "max_length": max_length,
                "average_length": round(average_length, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating sequence length statistics: {e}")
        
        return {
            "status": False,
            "error": str(e)
        }