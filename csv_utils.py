import csv
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.config_loader import Config

logger = logging.getLogger(__name__)


def write_csv(data: List[Dict[str, Any]], file_path: Path, create_dirs: bool = True) -> Dict[str, Any]:
    """
    Write structured data to CSV file with robust validation and error handling.
    
    Performs comprehensive data validation including schema consistency checks
    and creates output directories as needed. Optimized for bioinformatics
    pipeline data export with UTF-8 encoding and proper quoting for downstream analysis.
    
    Args:
        data: List of dictionaries containing structured data for CSV export
        file_path: Path object specifying the output CSV file location
        create_dirs: Boolean flag to create parent directories if they don't exist
        
    Returns:
        Dictionary containing execution status and file path information
        
    Raises:
        ValueError: If data is empty or has inconsistent schema
        PermissionError: If file system permissions prevent writing
        OSError: For other file system related errors
    """
    logger.info("=" * 80)
    logger.info("Writing structured data to CSV file")
    logger.info("-" * 80)
    
    try:
        # Validate input data structure
        if not data:
            logger.error("Cannot write empty dataset to CSV file")
            raise ValueError("Data must not be empty for CSV export")
        
        # Validate schema consistency across all records
        headers = set(data[0].keys())
        for i, row in enumerate(data):
            if set(row.keys()) != headers:
                logger.error(f"Schema mismatch detected at row {i}")
                raise ValueError("All rows must have identical column structure for CSV export")
        
        logger.info(f"Validated {len(data)} records with {len(headers)} columns")
        logger.info(f"Output file: {file_path}")
        
        # Create output directory structure if needed
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created output directory: {file_path.parent}")
        
        # Write CSV data with bioinformatics-optimized settings
        with file_path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(
                csv_file, 
                fieldnames=sorted(headers),  # Consistent column ordering
                quoting=csv.QUOTE_NONNUMERIC
            )
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"Successfully wrote {len(data)} records to CSV file")
        
        return {
            "status": True,
            "output_file": file_path,
            "records_written": len(data),
            "columns": len(headers)
        }
        
    except (PermissionError, OSError) as e:
        logger.error(f"File system operation failed: {e}")
        
        return {
            "status": False,
            "error": f"File operation failed: {str(e)}",
            "output_file": file_path
        }
        
    except Exception as e:
        logger.error(f"Error writing CSV data: {e}")
        
        return {
            "status": False,
            "error": str(e),
            "output_file": file_path
        }


def write_csv_output(data: List[Dict[str, Any]], config: Config) -> Dict[str, Any]:
    """
    Write bioinformatics analysis results to CSV using pipeline configuration.
    
    Exports structured analysis results to CSV format using paths and parameters
    defined in the pipeline configuration. Handles output directory creation
    and provides comprehensive error reporting for pipeline integration.
    
    Args:
        data: List of dictionaries containing analysis results for export
        config: Pipeline configuration object containing output specifications
        
    Returns:
        Dictionary containing execution status and output file information
    """
    logger.info("=" * 80)
    logger.info("Exporting analysis results using pipeline configuration")
    logger.info("-" * 80)
    
    try:
        # Validate configuration parameters
        if not config.output_dir:
            logger.error("Output directory not specified in pipeline configuration")
            raise ValueError("Output directory not specified in pipeline configuration")
        
        # Construct output file path from configuration
        output_file = Path(config.output_dir) / "analysis_results.csv"
        logger.info(f"Using configured output directory: {config.output_dir}")
        
        # Execute CSV write operation
        result = write_csv(data, output_file, create_dirs=True)
        
        if result["status"]:
            logger.info("Analysis results exported successfully using pipeline configuration")
        
        return result
        
    except Exception as e:
        logger.error(f"Error exporting analysis results: {e}")
        
        return {
            "status": False,
            "error": str(e)
        }


def read_csv_input(file_path: str) -> Dict[str, Any]:
    """
    Read and parse CSV file containing bioinformatics data for pipeline processing.
    
    Loads CSV data with comprehensive error handling and validation suitable
    for bioinformatics workflows. Returns structured data ready for downstream
    analysis with UTF-8 encoding support for international annotations.
    
    Args:
        file_path: String path to the input CSV file containing analysis data
        
    Returns:
        Dictionary containing parsed CSV data and file information
        
    Raises:
        FileNotFoundError: If the specified CSV file does not exist
        OSError: For file system access or permission errors
    """
    logger.info("=" * 80)
    logger.info("Reading CSV input data for pipeline processing")
    logger.info("-" * 80)
    
    try:
        # Validate input file existence
        input_path = Path(file_path)
        if not input_path.exists():
            logger.error(f"Input CSV file not found: {file_path}")
            raise FileNotFoundError(f"Input CSV file does not exist: {file_path}")
        
        logger.info(f"Reading CSV data from: {file_path}")
        
        # Parse CSV data with bioinformatics-optimized settings
        with input_path.open('r', newline='', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            csv_data = list(reader)
            
            # Extract column information for validation
            fieldnames = reader.fieldnames if reader.fieldnames else []
        
        logger.info(f"Successfully loaded {len(csv_data)} records with {len(fieldnames)} columns")
        logger.debug(f"Column headers: {fieldnames}")
        
        return {
            "status": True,
            "data": csv_data,
            "records_count": len(csv_data),
            "columns": fieldnames,
            "file_path": input_path
        }
        
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {e}")
        
        return {
            "status": False,
            "error": f"File not found: {str(e)}",
            "file_path": file_path
        }
        
    except OSError as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        
        return {
            "status": False,
            "error": f"File reading error: {str(e)}",
            "file_path": file_path
        }
        
    except Exception as e:
        logger.error(f"Error parsing CSV data: {e}")
        
        return {
            "status": False,
            "error": str(e),
            "file_path": file_path
        }