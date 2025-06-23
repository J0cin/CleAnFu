import logging
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any

from src.error_utils import handle_graceful_exit, PipelineError
from src.config_loader import Config

logger = logging.getLogger(__name__)

@handle_graceful_exit
def run_gffread_analysis(config: Config, input_files: List[Tuple[Path, Path]]) -> None:
    """
    Run the complete GFFread analysis pipeline to convert GFF files to protein FASTA.
    
    Args:
        config: Pipeline configuration
        input_files: List of tuples containing (directory_path, file_path) of GFF files
    """
    logger.info("=" * 80)
    logger.info("Starting GFFread analysis pipeline")
    logger.info("=" * 80)
    
    # Check input files
    if not input_files:
        error_msg = "No input files provided for GFFread analysis"
        logger.error(error_msg)
        raise PipelineError(error_msg)
    
    # Filter GFF files and validate suffixes
    gff_files = []
    for dir_path, file_path in input_files:
        if file_path.suffix.lower() in ('.gff', '.gff3'):
            logger.info(f'GFF file found: {file_path.name}')
            gff_files.append((dir_path, file_path))
        else:
            logger.warning(f'File with non-GFF suffix skipped: {file_path.name}')
    
    if not gff_files:
        error_msg = "No GFF files found in input files"
        logger.error(error_msg)
        raise PipelineError(error_msg)
    
    results = []
    
    # Process each GFF file through the GFFread pipeline
    for dir_path, file_path in gff_files:
        species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
        logger.info(f'GFFread analysis for {species_name}')
        try:
            result = process_gffread_file(dir_path, file_path, config)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.error(f"run_gffread_analysis, GFFread error: {e} for {species_name}")
    
    # Generate summary report if there are results
    if results:
        # Ensure output directory exists
        results_dir = Path("./gffread_results")
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Create results dataframe and save to CSV
        output_path = results_dir / "gffread_conversion_statistics.csv"
        pd.DataFrame(results).to_csv(output_path, index=False)
        logger.info(f"GFFread results saved to {output_path}")
        
        # Generate summary statistics
        total_files = len(results)
        successful_conversions = sum(1 for r in results if r.get("status", False))
        total_sequences = sum(r.get("sequence_count", 0) for r in results)
        
        logger.info("=" * 80)
        logger.info("GFFread Analysis Summary")
        logger.info(f"Total files processed: {total_files}")
        logger.info(f"Successful conversions: {successful_conversions}")
        logger.info(f"Total sequences generated: {total_sequences}")
        logger.info("=" * 80)
    else:
        logger.warning("No valid GFFread results obtained")


def process_gffread_file(dir_path: Path, file_path: Path, config: Config) -> Dict[str, Any]:
    """
    Process a single GFF file through the GFFread pipeline.
    
    Args:
        dir_path: Directory path containing the file
        file_path: Path to the GFF file
        config: Pipeline configuration
        
    Returns:
        Dictionary with analysis results or None if processing failed
    """
    logger.info(f"Processing file: {file_path}")
    species_name = file_path.parts[-3] if len(file_path.parts) >= 3 else file_path.stem
    
    try:
        # Step 1: Find corresponding genome FASTA file
        genome_file = find_genome_file(dir_path, config)
        
        if genome_file is None:
            logger.warning(f"No genome FASTA file found for {species_name}")
            return None
        
        # Step 2: Set up output file path
        output_file = file_path.with_name(f'{file_path.stem}{config.gffread.suffix}')
        
        # Step 3: Run GFFread command
        gffread_result = run_gffread_command(file_path, genome_file, output_file, config)
        
        if not gffread_result["status"]:
            logger.warning(f"GFFread conversion failed for {species_name}")
            return None
        
        # Step 4: Count sequences in output file
        sequence_count = count_sequences_in_fasta(output_file)
        
        logger.info(f"Successfully processed {species_name}: {sequence_count} sequences generated")
        
        # Return results dictionary
        return {
            "file_name": file_path.name,
            "species": species_name,
            "genome_file": genome_file.name,
            "output_file": output_file.name,
            "sequence_count": sequence_count,
            "status": True
        }
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def find_genome_file(dir_path: Path, config: Config) -> Path:
    """
    Find the corresponding genome FASTA file in the directory.
    
    Args:
        dir_path: Directory to search in
        config: Pipeline configuration containing genome patterns
        
    Returns:
        Path to genome file or None if not found
    """
    genome_pattern = config.gffread.genome_pattern
    
    for pattern in genome_pattern:
        genome_files = list(dir_path.glob(f"*{pattern}"))
        if genome_files:
            logger.info(f"Found genome file: {genome_files[0].name}")
            return genome_files[0]
    
    logger.error(f"No genome FASTA files found in {dir_path} with patterns: {genome_pattern}")
    return None


def run_gffread_command(gff_file: Path, genome_file: Path, output_file: Path, config: Config) -> Dict[str, Any]:
    """
    Execute the GFFread command to convert GFF to protein FASTA.
    
    Args:
        gff_file: Path to input GFF file
        genome_file: Path to genome FASTA file
        output_file: Path to output protein FASTA file
        config: Pipeline configuration
        
    Returns:
        Dictionary with command execution results
    """
    cmd = [
        config.gffread.cmd,
        "-y", str(output_file),
        "-g", str(genome_file),
        str(gff_file)
    ]
    
    logger.info(f'Running: {" ".join(cmd)}')
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            stderr=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            text=True
        )
        
        # Verify output file was created
        if not output_file.exists():
            error_msg = f'Output file not generated: {output_file}'
            logger.error(error_msg)
            return {"status": False, "error": error_msg}
        
        logger.info(f'File successfully generated: {output_file.name}')
        return {
            "status": True,
            "output_file": output_file,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        error_msg = f'GFFread command failed: {e.stderr}'
        logger.error(error_msg)
        return {"status": False, "error": error_msg}


def count_sequences_in_fasta(fasta_file: Path) -> int:
    """
    Count the number of sequences in a FASTA file.
    
    Args:
        fasta_file: Path to FASTA file
        
    Returns:
        Number of sequences in the file
    """
    try:
        with open(fasta_file, 'r') as f:
            return sum(1 for line in f if line.startswith('>'))
    except Exception as e:
        logger.warning(f"Could not count sequences in {fasta_file}: {e}")
        return 0