from pathlib import Path
from pydantic import BaseModel, ValidationError, Field, field_validator
from typing import Literal, List, Dict
import yaml
import logging

logger = logging.getLogger(__name__)


class InputConfig(BaseModel):
    """
    Configuration for input data processing parameters.
    
    Defines input data source type, file paths, and pattern matching
    for bioinformatics pipeline data ingestion. Supports multiple
    input formats including AGAT, BUSCO, and cleaned sequence data.
    """
    type: Literal['directory_agat', 'directory_busco', 'directory_clean']
    path: Path
    file_pattern: List[str]
    
    @field_validator('path')
    def validate_path_exists(cls, v):
        """Validate that the specified input path exists in the filesystem."""
        if not v.exists():
            raise ValueError(f"Input path does not exist: {v}")
        return v


class BUSCOConfig(BaseModel):
    """
    Configuration parameters for BUSCO phylogenomic analysis.
    
    BUSCO (Benchmarking Universal Single-Copy Orthologs) assessment
    configuration including execution parameters, resource allocation,
    and output specifications for genome completeness evaluation.
    """
    busco_cmd: str = "busco"
    mode: Literal['geno', 'prot', 'tran'] = "prot"
    threads: int = Field(4, ge=1, le=32)
    cpu: int = 24
    timeout: int = Field(600, description="Maximum execution time per analysis (seconds)")
    output_dir: Path = Path("busco_results")


class AGATConfig(BaseModel):
    """
    Configuration for AGAT (Another Gtf/Gff Analysis Toolkit) processing.
    
    Parameters for genome annotation processing including isoform selection,
    statistical analysis, and output formatting. Optimized for genomic
    feature extraction and annotation quality assessment.
    """
    keep_longest_cmd: str = "agat_sp_keep_longest_isoform.pl"
    agat_stats_cmd: str = "agat_sp_statistics.pl"
    output_suffix: str 
    required_stats: List[str] 
    patterns: Dict[str, str]
    output_dir: Path 


class CLEANINGConfig(BaseModel):
    """
    Configuration for sequence data cleaning and preprocessing.
    
    Parameters for quality control and preprocessing of biological
    sequences including length filtering and pattern matching for
    downstream bioinformatics analysis workflows.
    """
    input_pattern: List[str]
    max_length: int


class GFFREADConfig(BaseModel):
    """
    Configuration for GFFRead sequence extraction utility.
    
    Parameters for extracting genomic sequences from GFF/GTF annotation
    files, including output format specification and genome reference
    pattern matching for accurate sequence retrieval.
    """
    cmd: str = "gffread"
    output_suffix: str = "fasta"
    genome_pattern: List[str]


class STATSConfig(BaseModel):
    """
    Configuration for statistical analysis and results aggregation.
    
    Parameters for merging and analyzing results from multiple database
    searches (SwissProt, TrEMBL) and functional annotation pipelines,
    including output formatting and statistical reporting.
    """
    files_sws_identifier: str
    files_tre_identifier: str
    files_fan_identifier: str
    output_suffix: str = "merged.txt"
    run_get_stats: bool = True
    outfile_name: str = "SWP_TREMBL_GO"
    merge_column: str = "Protein-Accession"


class FANTASIAConfig(BaseModel):
    """
    Configuration for FANTASIA functional annotation pipeline.
    
    FANTASIA (Functional Annotation using Neural networks for Transcript
    Analysis and Sequence Interpretation Applications) configuration
    including deep learning model parameters and execution settings.
    """
    cmd: str = "/data/software/FANTASIA/FANTASIA/launch_gopredsim_pipeline.sh"
    prepare_cmd: str = "/data/software/FANTASIA/FANTASIA/generate_gopredsim_input_files.sh"
    go_cmd: str = "/data/software/FANTASIA/FANTASIA/convert_topgo_format.py"
    fantasia_results_dir: Path = Path("FANTASIA_results")
    fantasia_dir: Path = Path("fantasia_run")
    model: Literal["prott5", "seqvec"] = "prott5"
    model_path: Path = None
    batch_size: int = 64
    device: Literal['cpu', 'GPU'] = "GPU"
    

class DIAMONDConfig(BaseModel):
    """
    Configuration for DIAMOND sequence alignment parameters.
    
    High-performance sequence aligner configuration including database
    paths, search sensitivity, threading parameters, and output formatting
    for protein homology searches against reference databases.
    """
    cmd_blastp: str = "diamond blastp"
    cmd_ahrd: str = "/data/software/AHRD/dist/ahrd.jar"
    db_path: Path
    db_type: Literal["SwissProt", "Trembl"] = "SwissProt"
    threads: int = Field(24, ge=1, le=32)
    evalue: float = 1e-20
    output_format: str = "6"
    max_target_seqs: int = 1
    
    @field_validator('db_path')
    def validate_database_exists(cls, v):
        """Validate that the DIAMOND database exists and is accessible."""
        if not v.exists():
            raise ValueError(f"DIAMOND database does not exist: {v}")
        return v


class Config(BaseModel):
    """
    Master configuration object for bioinformatics pipeline execution.
    
    Comprehensive configuration container that aggregates all pipeline
    component configurations including input processing, sequence analysis,
    functional annotation, and output generation parameters.
    """
    input: InputConfig
    busco: BUSCOConfig
    agat: AGATConfig
    cleaning: CLEANINGConfig
    gffread: GFFREADConfig
    diamond: DIAMONDConfig
    fantasia: FANTASIAConfig
    stats: STATSConfig
    output_dir: Path = Path("results")
    logging_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = "INFO"


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load and validate bioinformatics pipeline configuration from YAML file.
    
    Parses configuration file containing pipeline parameters, validates
    all settings including file paths and bioinformatics tool parameters,
    and returns a validated configuration object for pipeline execution.
    
    Args:
        config_path: Path to YAML configuration file containing pipeline parameters
        
    Returns:
        Config: Validated configuration object with all pipeline settings
        
    Raises:
        FileNotFoundError: If configuration file does not exist
        ValidationError: If configuration parameters are invalid
        RuntimeError: For configuration parsing or validation errors
    """
    logger.info("=" * 80)
    logger.info("Loading bioinformatics pipeline configuration")
    logger.info("-" * 80)
    
    try:
        config_file = Path(config_path)
        
        # Validate configuration file existence
        if not config_file.exists():
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Reading configuration from: {config_path}")
        
        # Load and parse YAML configuration
        with open(config_path, 'r', encoding='utf-8') as config_file_handle:
            config_data = yaml.safe_load(config_file_handle)
            
        # Validate configuration parameters
        validated_config = Config(**config_data)
        
        logger.info("Configuration loaded and validated successfully")
        logger.info(f"Output directory: {validated_config.output_dir}")
        logger.info(f"Logging level: {validated_config.logging_level}")
        
        return validated_config
        
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise RuntimeError(f"Configuration validation error: {str(e)}") from e
        
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise RuntimeError(f"Configuration loading error: {str(e)}") from e