import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.config_loader import load_config, Config
from src.file_utils import get_input_files 
from src.error_utils import handle_graceful_exit
from src.logger_config import setup_logging

from src.analysis.agat import run_agat_analysis, run_agat_keep_longest
from src.analysis.busco import run_busco_analysis
from src.analysis.cleaning import run_sequence_cleaner
from src.analysis.gfftool import run_gffread_tool
from src.analysis.diamond_v2 import run_diamond_ahrd_analysis
from src.analysis.fantasia import run_fantasia_analysis
from src.analysis.comparison_stats_v2 import run_go_analysis_suite
from src.analysis.visualization_v3 import run_visualization_and_analysis

logger = logging.getLogger(__name__)

@handle_graceful_exit
def parse_command_line_arguments() -> Dict[str, Any]:
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Bioinformatics Pipeline for Functional Genomic Annotation",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the main YAML configuration file.")
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available tools and analysis steps")

    # --- Comandos de Pre-procesamiento ---
    agat_parser = subparsers.add_parser('agat', help="AGAT toolkit for GFF processing")
    agat_parser.add_argument('--mode', choices=['keep_longest', 'statistics'], required=True, help="AGAT execution mode")
    subparsers.add_parser('busco', help="BUSCO for genome completeness assessment")
    subparsers.add_parser('cleaning', help="Clean protein FASTA files")
    subparsers.add_parser('gffread', help="Convert GFF to protein FASTA")
    
    # --- Comandos de Anotación ---
    diamond_parser = subparsers.add_parser('diamond', help="Run DIAMOND + AHRD for homology annotation")
    diamond_parser.add_argument('--skip_diamond', action='store_true', help="Skip DIAMOND, run AHRD on existing results")
    subparsers.add_parser('fantasia', help="Run FANTASIA for PLM-based annotation")

    # --- Comandos de Análisis Comparativo ---
    stats_parser = subparsers.add_parser('stats', help="Generate comparative statistics from annotation files")
    stats_parser.add_argument('--lineage_file', type=str, required=True, help="Path to the CSV file with species lineages.")
    stats_parser.add_argument('--taxon_map_file', type=str, required=True, help="Path to the TSV file mapping GO terms to taxa.")
    stats_parser.add_argument('--obo_file', type=str, required=True, help="Path to the go-basic.obo file.")
    stats_parser.add_argument('--go_slim_file', type=str, required=True, help="Path to the generic GO slim OBO file.")
    

    viz_parser = subparsers.add_parser('visualize', help="Generate plots from the 'stats' analysis results.")
    viz_parser.add_argument(
        '--results_dir', 
        type=str, 
        required=True, 
        help="Path to the directory containing the results from the 'stats' command (e.g., 'output/stats_results')."
    )
    viz_parser.add_argument(
        '--only_figure', 
        type=str, 
        required=False,
        
        choices=['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'fig7', 'fig8', 'fig10','fig11','fig12','fig13', 'excel'],
        help="Optional: Generate only a specific figure or the final Excel summary."
    )


    args = parser.parse_args()
    return {"args": args, "command": args.command}

@handle_graceful_exit
def setup_directories(config: Config, command: str):
    """Crea los directorios de salida necesarios para cada comando."""
    config_map = {
        'agat': config.agat, 'busco': config.busco, 'fantasia': config.fantasia, 'stats': config.stats
    }
    if command in config_map:
        command_config = config_map[command]
        output_subdir_name = getattr(command_config, 'output_dir', None) or getattr(command_config, 'fantasia_results_dir', None)
        if output_subdir_name:
            target_dir = Path(config.output_dir) / output_subdir_name
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory for '{command}': {target_dir}")

@handle_graceful_exit
def main() -> int:
    """Punto de entrada principal del pipeline."""
    args_result = parse_command_line_arguments()
    args, command = args_result["args"], args_result["command"]
    
    config = load_config(args.config)
    
    setup_logging(config.logging_level)
    
    logger.info(f"PIPELINE INITIALIZED. COMMAND: {command.upper()}")
    
    setup_directories(config, command)


    if command == 'validate':
        output_validation_dir = Path(config.output_dir) / "validation_results"
        proteomes_parent_dir = config.input.path.parent if config.input.path.is_file() else config.input.path
        diamond_db_path = config.diamond.databases['TrEMBL'].db_path
        proteome_pattern = config.cleaning.input_pattern[0]

        cmd_to_run = (
            f"python src/analysis/validate_anomalies.py \\\n"
            f"    -i {args.anomalies_file} \\\n"
            f"    -p {proteomes_parent_dir} \\\n"
            f"    -d \"{diamond_db_path}\" \\\n"
            f"    -t {args.threshold} \\\n"
            f"    -o {output_validation_dir} \\\n"
            f"    --proteome_pattern \"{proteome_pattern}\""
        )
        logger.info("="*80)
        logger.info("VALIDATION STEP: Copy and run the following command in your terminal:")
        print("\n" + cmd_to_run + "\n")
        logger.info("="*80)
        return 0


    logger.info("Identifying input files...")
    input_files = get_input_files(config) 
    if not input_files:
        logger.error("No input files found. Check 'input.type' and 'input.path' in config.yml."); return 1
    logger.info(f"Found {len(input_files)} items to process.")

    # --- Dispatcher de Comandos ---
    if command == 'agat':
        if args.mode == 'keep_longest': run_agat_keep_longest(config, input_files)
        else: run_agat_analysis(config, input_files)
    elif command == 'busco':
        run_busco_analysis(config, input_files)
    elif command == 'cleaning':
        run_sequence_cleaner(config, input_files)
    elif command == 'gffread':
        run_gffread_tool(config, input_files)
    elif command == 'diamond':
        run_diamond_ahrd_analysis(config, input_files, args.skip_diamond)
    elif command == 'fantasia':
        run_fantasia_analysis(config, input_files)
    elif command == 'stats':
        run_go_analysis_suite(config, input_files, Path(args.lineage_file), Path(args.taxon_map_file), Path(args.obo_file), Path(args.go_slim_file))
   
    elif command == 'visualize':
        # El comando visualize necesita una ruta de salida diferente
        output_viz_dir = Path(config.output_dir) / "visualization_outputs_v2"
        output_viz_dir.mkdir(parents=True, exist_ok=True) # Asegurarse de que el directorio existe
        
        run_visualization_and_analysis(
            results_dir=Path(args.results_dir), 
            output_dir=output_viz_dir, 
            only_figure=args.only_figure
        )
    else:
        raise ValueError(f"Unknown command: {command}")

    logger.info(f"--- COMMAND {command.upper()} COMPLETED SUCCESSFULLY ---")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.getLogger(__name__).critical(f"A critical, unhandled error occurred: {e}", exc_info=True)
        sys.exit(1)
