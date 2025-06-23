import subprocess
import logging
import yaml
import re
import traceback
from pathlib import Path
from typing import Dict, Any, Set, List

from src.error_utils import handle_graceful_exit
from src.config_loader import Config

logger = logging.getLogger(__name__)

@handle_graceful_exit
def run_diamond_blastp(file_path: Path, config: Config) -> Dict[str, Any]:
    """
    Execute DIAMOND BLASTP search against reference protein database.
    
    Performs high-throughput protein sequence alignment using DIAMOND BLASTP
    algorithm against the specified reference database (SwissProt/TrEMBL).
    Optimized for bioinformatics workflows with sensitive search parameters.
    
    Args:
        file_path: Path to input protein FASTA file
        config: Pipeline configuration object containing database paths and parameters
        
    Returns:
        Dictionary containing execution status, output file path, and error information
    """
    logger.info("=" * 80)
    logger.info(f"STEP 1: Running DIAMOND BLASTP against {str(config.diamond.db_type)}")
    logger.info("-" * 80)
    
    file_name = file_path.stem
    output_dir = file_path.parent
    output_file = output_dir / f"{file_name}.dmd.{str(config.diamond.db_type)}.o6.txt"
    
    logger.info(f"Processing file: {file_name}")
    
    # Construct DIAMOND command with bioinformatics best practices
    diamond_cmd = [
        "diamond",
        "blastp",
        "--query", str(file_path),
        "--db", str(config.diamond.db_path),
        "--outfmt", config.diamond.output_format,
        "--sensitive",
        "--max-target-seqs", str(config.diamond.max_target_seqs),
        "--evalue", str(config.diamond.evalue),
        "--out", str(output_file),
        "--threads", str(config.diamond.threads)
    ]
    
    logger.info(f"Executing DIAMOND command: {' '.join(str(arg) for arg in diamond_cmd)}")
    
    try:
        result = subprocess.run(
            diamond_cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Log command output for debugging
        logger.debug(f"DIAMOND STDOUT: {result.stdout}")
        logger.debug(f"DIAMOND STDERR: {result.stderr}")
        
        # Validate execution success
        if result.returncode != 0:
            logger.error(f"DIAMOND BLASTP failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            
            return {
                "status": False,
                "error": f"DIAMOND execution failed: {result.stderr}",
                "file_name": file_name
            }
        
        logger.info("DIAMOND BLASTP completed successfully")
        
        return {
            "status": True,
            "output_file": output_file,
            "file_name": file_name
        }
        
    except Exception as e:
        logger.error(f"Error executing DIAMOND BLASTP: {e}")
        
        return {
            "status": False,
            "error": str(e),
            "file_name": file_name
        }


@handle_graceful_exit
def create_ahrd_config(file_path: Path, diamond_result: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """
    Generate AHRD configuration file for functional annotation analysis.
    
    Creates a YAML configuration file containing all necessary parameters
    for AHRD (Automated Human Readable Descriptions) functional annotation,
    including database references, scoring weights, and output specifications.
    
    Args:
        file_path: Path to original protein FASTA file
        diamond_result: Dictionary containing DIAMOND BLASTP results
        config: Pipeline configuration object
        
    Returns:
        Dictionary containing configuration file path, output path, and execution status
    """
    if not diamond_result["status"]:
        return {
            "status": False,
            "error": "DIAMOND analysis failed, cannot create AHRD configuration"
        }
    
    logger.info("=" * 80)
    logger.info("STEP 2: Creating AHRD configuration file")
    logger.info("-" * 80)
    
    try:
        output_dir = file_path.parent
        file_name = file_path.stem
        diamond_output = diamond_result["output_file"]
        ahrd_output = output_dir / f"{file_name}.{str(config.diamond.db_type)}_funct_ahrd.txt"
        ahrd_config_path = output_dir / f"ahrd_configuration_{str(config.diamond.db_type)}.yml"
        
        # Define AHRD configuration with optimized bioinformatics parameters
        ahrd_configuration = {
            "proteins_fasta": str(file_path),
            "token_score_bit_score_weight": 0.468,
            "token_score_database_score_weight": 0.2098,
            "token_score_overlap_score_weight": 0.3221,
            "gene_ontology_result": "/data/shared_dbs/swissprot/goa_uniprot_all.gaf",
            "reference_go_regex": "^UniProtKB\\t(?<shortAccession>[^\\t]+)\\t[^\\t]+\\t(?!NOT\\|)[^\\t]*\\t(?<goTerm>GO:\\d{7})",
            "prefer_reference_with_go_annos": True,
            "output": str(ahrd_output),
            "blast_dbs": {
                "swissprot": {
                    "weight": 653,
                    "description_score_bit_score_weight": 2.717061,
                    "file": str(diamond_output),
                    "database": str(config.diamond.db_path),
                    "blacklist": "/data/software/AHRD/test/resources/blacklist_descline.txt",
                    "filter": "/data/software/AHRD/test/resources/filter_descline_sprot.txt",
                    "token_blacklist": "/data/software/AHRD/test/resources/blacklist_token.txt"
                }
            }
        }
        
        # Write AHRD configuration to YAML file
        with open(ahrd_config_path, 'w', encoding='utf-8') as config_file:
            yaml.dump(ahrd_configuration, config_file, default_flow_style=False)
        
        logger.info(f"AHRD configuration written to: {ahrd_config_path}")
        
        return {
            "status": True,
            "config_path": ahrd_config_path,
            "output_path": ahrd_output
        }
    
    except Exception as e:
        logger.error(f"Error creating AHRD configuration: {e}")
        
        return {
            "status": False,
            "error": str(e)
        }


@handle_graceful_exit
def run_ahrd(ahrd_config_result: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """
    Execute AHRD functional annotation analysis.
    
    Runs the AHRD (Automated Human Readable Descriptions) Java application
    to generate functional annotations for protein sequences based on
    homology search results and Gene Ontology information.
    
    Args:
        ahrd_config_result: Dictionary containing AHRD configuration file path
        config: Pipeline configuration object
        
    Returns:
        Dictionary containing execution status and output file path
    """
    if not ahrd_config_result["status"]:
        return {
            "status": False,
            "error": "AHRD configuration failed, cannot execute AHRD analysis"
        }
    
    logger.info("=" * 80)
    logger.info("STEP 3: Running AHRD functional annotation analysis")
    logger.info("-" * 80)
    
    ahrd_config_path = ahrd_config_result["config_path"]
    output_path = ahrd_config_result["output_path"]
    
    # Construct AHRD Java command with memory optimization
    ahrd_cmd = [
        "java",  
        "-Xmx64g",  # Allocate 64GB memory for large-scale analysis
        "-jar",
        config.diamond.cmd_ahrd,
        str(ahrd_config_path)
    ]
    
    logger.info(f"Executing AHRD command: {' '.join(ahrd_cmd)}")
    
    try:
        result = subprocess.run(
            ahrd_cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Log command output for debugging
        logger.debug(f"AHRD STDOUT: {result.stdout}")
        logger.debug(f"AHRD STDERR: {result.stderr}")
        
        # Validate execution success
        if result.returncode != 0:
            logger.error(f"AHRD analysis failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            
            return {
                "status": False,
                "error": f"AHRD execution failed: {result.stderr}"
            }
        
        logger.info("AHRD functional annotation analysis completed successfully")
        
        return {
            "status": True,
            "output_path": output_path
        }
        
    except Exception as e:
        logger.error(f"Error executing AHRD analysis: {e}")
        
        return {
            "status": False,
            "error": str(e)
        }


@handle_graceful_exit
def parse_ahrd_go_terms(ahrd_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse Gene Ontology terms from AHRD output file and generate statistics.
    
    Extracts GO terms from AHRD functional annotation results, handling
    various output formats and wrapped lines. Calculates comprehensive
    annotation statistics for downstream analysis.
    
    Args:
        ahrd_result: Dictionary containing AHRD output file path
        
    Returns:
        Dictionary containing GO term statistics and annotation metrics
    """
    output_path = ahrd_result.get("output_path")
    if not output_path or not Path(output_path).exists():
        logger.error(f"AHRD output file not found: {output_path}")
        return None

    try:
        all_go_terms = []
        genes_with_go = set()
        total_proteins = 0
        proteins_with_hits = 0

        logger.info(f"Parsing GO terms from AHRD output: {output_path}")

        with open(output_path, 'r', encoding='utf-8') as output_file:
            # Parse header to identify column structure
            header = next(output_file, "").strip().split('\t')

            # Determine column indices for GO terms
            protein_col = 0  # First column typically contains protein ID
            go_col = None

            for i, column_name in enumerate(header):
                if "Gene-Ontology" in column_name:
                    go_col = i
                    break

            # Process entries with special handling for multi-line entries
            current_entry = None

            for line in output_file:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')

                # Identify new entry (starts with protein ID)
                if parts[0] and not parts[0].startswith(' '):
                    # Process previous entry if exists
                    if current_entry:
                        _process_ahrd_entry(current_entry, genes_with_go, all_go_terms)

                    # Initialize new entry
                    current_entry = line
                    total_proteins += 1

                    # Check for BLAST hit presence
                    if len(parts) > 1 and parts[1].strip():
                        proteins_with_hits += 1
                else:
                    # Handle continuation line (wrapped entry)
                    if current_entry:
                        current_entry += " " + line

            # Process final entry
            if current_entry:
                _process_ahrd_entry(current_entry, genes_with_go, all_go_terms)

        # Calculate comprehensive statistics
        total_go_terms = len(all_go_terms)
        unique_go_terms = len(set(all_go_terms))
        annotation_rate = (proteins_with_hits / total_proteins * 100) if total_proteins > 0 else 0
        go_annotation_rate = (len(genes_with_go) / total_proteins * 100) if total_proteins > 0 else 0

        species_name = Path(output_path).stem.split(".funct_ahrd")[0]

        statistics_result = {
            "species": species_name,
            "total_proteins": total_proteins,
            "proteins_with_hits": proteins_with_hits,
            "annotation_rate": round(annotation_rate, 2),
            "total_go_terms": total_go_terms,
            "unique_go_terms": unique_go_terms,
            "genes_with_go": len(genes_with_go),
            "go_annotation_rate": round(go_annotation_rate, 2)
        }

        logger.info(f"Successfully parsed {total_go_terms} GO terms ({unique_go_terms} unique) "
                   f"for {len(genes_with_go)} genes from {species_name}")

        return statistics_result

    except Exception as e:
        logger.error(f"Error parsing GO terms from AHRD output: {e}")
        logger.error(traceback.format_exc())
        return None


def _process_ahrd_entry(entry_text: str, genes_with_go: Set[str], all_go_terms: List[str]) -> None:
    """
    Process a single AHRD entry to extract GO terms and update statistics.
    
    Helper function to parse individual AHRD entries, extract GO terms using
    regular expressions, and update the global statistics collections.
    
    Args:
        entry_text: Complete AHRD entry text (may span multiple lines)
        genes_with_go: Set to store genes with GO annotations
        all_go_terms: List to store all extracted GO terms
    """
    # Extract protein ID from first column
    parts = entry_text.split('\t')
    protein_id = parts[0].strip()

    # Extract GO terms using bioinformatics-standard regex pattern
    go_terms = re.findall(r'GO:\d{7}', entry_text)

    if go_terms:
        genes_with_go.add(protein_id)
        all_go_terms.extend(go_terms)