import logging
import pandas as pd
from pathlib import Path
import pickle
from typing import List, Tuple, Dict, Any
import math
import subprocess
from collections import Counter
import numpy as np
import itertools
import random
import tempfile 
import sys 
import importlib.util 
import shutil
from pronto import Ontology

# --- Dependencias científicas ---
from goatools.obo_parser import GODag
from goatools.goea.go_enrichment_ns import GOEnrichmentStudy
from goatools.semantic import get_info_content, TermCounts
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pygosemsim import term_set, annotation, graph, similarity
# --- Módulos internos ---
from src.config_loader import Config
from src.go_analyzer import parse_go_annotations, compare_go_sets
from src.error_utils import handle_graceful_exit

logger = logging.getLogger(__name__)

# #############################################################################
# --- DEFINICIONES CENTRALIZADAS PARA ANÁLISIS DE ANOMALÍAS (JERÁRQUICO) ---
# #############################################################################

# Listas de palabras clave (en minúsculas para comparación insensible)
GENERAL_NOISE_KEYWORDS = ['root', 'cellular organisms', 'eukaryota', 'opisthokonta']
PLANT_KEYWORDS = [
    'viridiplantae', 'embryophyta', 'streptophyta', 'magnoliopsida', 'rosids', 
    'asterids', 'poaceae', 'arabidopsis', 'oryza', 'zea mays', 'solanum'
]
ANIMAL_KEYWORDS = [
    'metazoa', 'amniota', 'bilateria', 'euteleostomi', 'euarchontoglires', 
    'gnathostomata', 'eumetazoa', 'vertebrata', 'tetrapoda', 'boreoeutheria',
    'mammalia', 'eutheria', 'theria', 'chordata', 'deuterostomia', 'drosophila',
    'homo sapiens', 'mus musculus', 'danio rerio', 'caenorhabditis'
]
FUNGI_KEYWORDS = ['fungi', 'ascomycota', 'saccharomyces', 'schizosaccharomyces', 'aspergillus']
BACTERIA_KEYWORDS = ['bacteria', 'pseudomonadota', 'escherichia coli', 'mycobacterium']
ARCHAEA_KEYWORDS = ['archaea']
VIRUS_KEYWORDS = ['viruses', 'caudoviricetes', 'orthornavirae']

def _classify_taxon(taxon: str) -> str:
    """
    Clasifica un taxón en una categoría de anomalía usando una lógica jerárquica por prioridad.
    El orden de las comprobaciones es crucial.
    """
    if not isinstance(taxon, str):
        return 'Valid_Non_Specific'
    
    taxon_lower = taxon.lower()

    # 1. Ignorar ruido general
    if any(keyword in taxon_lower for keyword in GENERAL_NOISE_KEYWORDS):
        return 'General_Noise'
    if taxon == 'Non taxon-specific':
        return 'Valid_Non_Specific'

    # 2. Identificar taxones válidos (Plantas)
    if any(keyword in taxon_lower for keyword in PLANT_KEYWORDS):
        return 'Valid_Plant'

    # 3. Identificar Alucinaciones (Animales y Hongos)
    if any(keyword in taxon_lower for keyword in ANIMAL_KEYWORDS):
        return 'Hallucination_Metazoa'
    if any(keyword in taxon_lower for keyword in FUNGI_KEYWORDS):
        return 'Hallucination_Fungi'

    # 4. Identificar Contaminación
    if any(keyword in taxon_lower for keyword in BACTERIA_KEYWORDS):
        return 'Contamination_Bacteria'
    if any(keyword in taxon_lower for keyword in ARCHAEA_KEYWORDS):
        return 'Contamination_Archaea'
    if any(keyword in taxon_lower for keyword in VIRUS_KEYWORDS):
        return 'Contamination_Viruses'
        
    # 5. Si no coincide con nada, es una anomalía no clasificada
    return 'Other_Anomaly'

def _run_outlier_validation_analysis(
    all_fantasia_dfs: list, 
    all_homology_dfs: list, 
    taxon_map_df: pd.DataFrame, 
    pca_results: dict, 
    base_output_dir: Path
):
    """
    Valida cuantitativamente los outliers para CADA método (FANTASIA y Homología).
    1. Identifica outliers programáticamente usando los resultados del PCA (distancia al centroide).
    2. Identifica outliers de cobertura (especies exclusivas a un método).
    3. Para cada outlier, compara su perfil de contaminación con el promedio de su linaje.
    """
    logger.info("--- Starting Comprehensive Quantitative Outlier Validation Analysis (for both methods) ---")
    
    all_validation_data = []
    
    data_sources = {
        'FANTASIA': {'dfs': all_fantasia_dfs, 'pca': pca_results.get('fantasia_pca')},
        'Homology': {'dfs': all_homology_dfs, 'pca': pca_results.get('homology_pca')}
    }

    species_in_fan = set(pd.concat(all_fantasia_dfs)['species']) if all_fantasia_dfs else set()
    species_in_hom = set(pd.concat(all_homology_dfs)['species']) if all_homology_dfs else set()

    for method_name, sources in data_sources.items():
        logger.info(f"--- Analyzing outliers for method: {method_name} ---")
        
        if not sources['dfs'] or sources['pca'] is None or sources['pca'].empty:
            logger.warning(f"No data or PCA results for method '{method_name}'. Skipping.")
            continue

        pc_df = sources['pca'].copy()
        centroids = pc_df.groupby('lineage')[['PC1', 'PC2', 'PC3']].transform('mean')
        pc_df['distance'] = np.sqrt(np.sum((pc_df[['PC1', 'PC2', 'PC3']] - centroids)**2, axis=1))
        
        distance_stats = pc_df.groupby('lineage')['distance'].agg(['mean', 'std']).fillna(0)
        pc_df = pc_df.join(distance_stats, on='lineage')
        pc_df['outlier_threshold'] = pc_df['mean'] + 2.0 * pc_df['std']
        
        functional_outliers = set(pc_df[pc_df['distance'] > pc_df['outlier_threshold']].index)
        logger.info(f"Found {len(functional_outliers)} functional outliers for {method_name}: {functional_outliers}")

        if method_name == 'FANTASIA':
            coverage_outliers = species_in_fan - species_in_hom
        else:
            coverage_outliers = species_in_hom - species_in_fan
        logger.info(f"Found {len(coverage_outliers)} coverage outliers for {method_name}: {coverage_outliers}")
        
        total_outliers_to_check = functional_outliers.union(coverage_outliers)
        if not total_outliers_to_check:
            logger.info(f"No outliers to analyze for {method_name}.")
            continue

        df_method = pd.concat(sources['dfs'], ignore_index=True)
        merged_df = pd.merge(df_method, taxon_map_df, on='GO_term', how='left')
        merged_df['category'] = merged_df['taxon'].apply(_classify_taxon)
        
        species_profiles = merged_df.groupby(['species', 'lineage', 'category']).size().unstack(fill_value=0)
        species_profiles_perc = species_profiles.apply(lambda x: 100 * x / x.sum(), axis=1)

        for outlier in total_outliers_to_check:
            # --- INICIO DE LA CORRECCIÓN ---
            
            # Usar indexado booleano para seleccionar de forma segura, esto siempre devuelve un DataFrame.
            outlier_profile_df = species_profiles_perc[species_profiles_perc.index.get_level_values('species') == outlier]

            # Si la especie no se encuentra en los perfiles (caso raro), la saltamos.
            if outlier_profile_df.empty:
                continue

            # Convertimos la primera (y única) fila de este DataFrame a una Serie.
            # Esta Serie SÍ tendrá el atributo .name que necesitamos.
            outlier_profile = outlier_profile_df.iloc[0]
            
            # Ahora, esta línea funcionará correctamente porque outlier_profile es una Serie.
            # El atributo .name de la Serie contiene el índice original, que es una tupla (especie, linaje).
            outlier_lineage = outlier_profile.name[1]
            
            # --- FIN DE LA CORRECCIÓN ---

            lineage_peers = species_profiles_perc[
                (species_profiles_perc.index.get_level_values('lineage') == outlier_lineage) &
                (~species_profiles_perc.index.get_level_values('species').isin(total_outliers_to_check))
            ]
            
            reference_profile = lineage_peers.mean() if not lineage_peers.empty else pd.Series(0, index=outlier_profile.index)
            
            comparison_df = pd.DataFrame({'Outlier_Percentage': outlier_profile, 'Lineage_Avg_Percentage': reference_profile}).reset_index()
            comparison_df['method'] = method_name
            comparison_df['outlier_species'] = outlier
            comparison_df['lineage'] = outlier_lineage
            comparison_df['outlier_type'] = 'Functional' if outlier in functional_outliers else 'Coverage'
            if outlier in functional_outliers and outlier in coverage_outliers:
                comparison_df['outlier_type'] = 'Functional & Coverage'

            all_validation_data.append(comparison_df)

    if not all_validation_data:
        logger.info("No outlier validation data was generated for any method.")
        return

    final_report = pd.concat(all_validation_data, ignore_index=True)
    output_file = base_output_dir / "outlier_quantitative_validation_comprehensive.csv"
    
    final_report = final_report[['method', 'outlier_species', 'outlier_type', 'lineage', 'category', 'Outlier_Percentage', 'Lineage_Avg_Percentage']]
    final_report.to_csv(output_file, index=False, float_format='%.2f')
    logger.info(f"Comprehensive quantitative outlier validation report saved to {output_file}")
   
def _run_lineage_profile_comparison(
    all_fantasia_dfs: list, 
    all_homology_dfs: list, 
    taxon_map_df: pd.DataFrame, 
    base_output_dir: Path
):
    """
    Para cada especie, compara su perfil de anomalías con el perfil promedio de su propio linaje.
    Este análisis se realiza para TODAS las especies, no solo para los outliers.
    VERSIÓN OPTIMIZADA: Usa operaciones vectorizadas en lugar de bucles para mayor eficiencia.
    """
    logger.info("--- Starting Comprehensive Lineage Profile Comparison Analysis (Optimized) ---")
    
    all_comparison_data = []
    data_sources = {'FANTASIA': all_fantasia_dfs, 'Homology': all_homology_dfs}

    for method_name, dfs in data_sources.items():
        if not dfs:
            logger.warning(f"No data for method '{method_name}'. Skipping lineage profile comparison.")
            continue

        df_method = pd.concat(dfs, ignore_index=True)
        merged_df = pd.merge(df_method, taxon_map_df, on='GO_term', how='left')
        merged_df['category'] = merged_df['taxon'].apply(_classify_taxon)
        
        species_profiles = merged_df.groupby(['species', 'lineage', 'category']).size().unstack(fill_value=0)
        species_profiles_perc = species_profiles.apply(lambda x: 100 * x / x.sum() if x.sum() > 0 else 0, axis=1)

        # --- INICIO DE LA CORRECCIÓN Y OPTIMIZACIÓN ---
        
        # 1. Calcular la suma y el tamaño de cada grupo de linaje
        lineage_sums = species_profiles_perc.groupby('lineage').transform('sum')
        lineage_counts = species_profiles_perc.groupby('lineage').transform('size')

        # 2. Calcular la suma y el tamaño de los "pares" (peers) para cada fila
        # Suma de los pares = Suma total del linaje - valor de la fila actual
        peer_sums = lineage_sums - species_profiles_perc
        # Conteo de los pares = Conteo total del linaje - 1
        peer_counts = lineage_counts - 1

        # 3. Calcular el promedio de los pares.
        # Usamos .div() y fill_value=0 para manejar de forma segura la división por cero
        # (esto ocurre en linajes con una sola especie, donde peer_counts es 0).
        lineage_avg_profile = peer_sums.div(peer_counts, axis=0).fillna(0)

        # 4. Ensamblar el DataFrame final
        # Renombramos las columnas para que coincidan con el formato deseado
        species_profiles_perc.columns = pd.MultiIndex.from_product([['Species_Percentage'], species_profiles_perc.columns])
        lineage_avg_profile.columns = pd.MultiIndex.from_product([['Lineage_Avg_Percentage'], lineage_avg_profile.columns])
        
        # Concatenamos los resultados
        final_df = pd.concat([species_profiles_perc, lineage_avg_profile], axis=1)
        
        # Reestructuramos el DataFrame a un formato "largo" para el CSV final
        report = final_df.stack(level=1).reset_index()
        report['method'] = method_name
        all_comparison_data.append(report)

        # --- FIN DE LA CORRECCIÓN Y OPTIMIZACIÓN ---

    if not all_comparison_data:
        logger.info("No lineage profile comparison data was generated.")
        return

    final_report = pd.concat(all_comparison_data, ignore_index=True)
    final_report = final_report[['method', 'lineage', 'species', 'category', 'Species_Percentage', 'Lineage_Avg_Percentage']]
    output_file = base_output_dir / "full_lineage_profile_comparison.csv"
    final_report.to_csv(output_file, index=False, float_format='%.2f')
    logger.info(f"Comprehensive lineage profile comparison report saved to {output_file}")
    
# #############################################################################
# --- FUNCIONES DE ANÁLISIS ESPECÍFICAS (CORREGIDAS Y NUEVAS) ---
# #############################################################################

def calculate_effect_size(series1: pd.Series, series2: pd.Series, test_used: str) -> float:
    """Calcula el tamaño del efecto apropiado (Cohen's d o Correlación Rango-Biserial)."""
    try:
        if test_used == 'T-test':
            n1, n2 = len(series1), len(series2)
            if n1 < 2 or n2 < 2: return 0.0
            s1, s2 = series1.std(), series2.std()
            pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            return (series1.mean() - series2.mean()) / pooled_std if pooled_std > 0 else 0
        elif test_used == 'Mann-Whitney U':
            n1, n2 = len(series1), len(series2)
            if n1 == 0 or n2 == 0: return 0.0
            u_statistic, _ = mannwhitneyu(series1, series2, alternative='two-sided')
            return 1 - (2 * u_statistic) / (n1 * n2)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return 0.0

def _run_statistical_comparison(series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
    """Función helper refactorizada para una comparación estadística robusta."""
    results = {}
    s1_clean = series1.dropna()
    s2_clean = series2.dropna()

    if len(s1_clean) < 10 or len(s2_clean) < 10:
        return {'error': f"Insufficient data (sizes: {len(s1_clean)}, {len(s2_clean)})"}

    results['fantasia_stats'] = s1_clean.describe().round(3).to_dict()
    results['homology_stats'] = s2_clean.describe().round(3).to_dict()

    use_non_parametric = False
    if len(s1_clean) > 5000 or len(s2_clean) > 5000:
        use_non_parametric = True
        results['normality_assumption'] = 'Assumed non-normal (large sample)'
    else:
        shapiro_p1 = shapiro(s1_clean).pvalue
        shapiro_p2 = shapiro(s2_clean).pvalue
        if shapiro_p1 < 0.05 or shapiro_p2 < 0.05:
            use_non_parametric = True
            results['normality_assumption'] = 'Not normal (Shapiro)'
        else:
            results['normality_assumption'] = 'Normal (Shapiro)'

    if not use_non_parametric:
        statistic, p_value = ttest_ind(s1_clean, s2_clean, alternative='two-sided', equal_var=False)
        results['comparison_test_used'] = 'T-test'
    else:
        statistic, p_value = mannwhitneyu(s1_clean, s2_clean, alternative='two-sided')
        results['comparison_test_used'] = 'Mann-Whitney U'

    results['comparison_p_value'] = p_value
    results['comparison_statistic'] = statistic
    results['effect_size'] = calculate_effect_size(s1_clean, s2_clean, results['comparison_test_used'])

    return results

def _calculate_per_protein_ic_scores(df: pd.DataFrame, ic_weights: dict) -> pd.Series:
    """Calcula la puntuación MÁXIMA de IC por proteína."""
    if df.empty:
        return pd.Series(dtype=float)
    df_copy = df.copy()
    df_copy['score'] = get_go_ic(df_copy['GO_term'], ic_weights)
    per_protein_scores = df_copy.dropna(subset=['score']).groupby('protein')['score'].max()
    return per_protein_scores

def get_go_ic(go_series: pd.Series, ic_weights: Dict[str, float]) -> pd.Series:
    """Calcula el IC para una serie de términos GO usando pesos pre-calculados."""
    if go_series.empty:
        return pd.Series(dtype=float)
    return go_series.map(ic_weights).dropna()

def _create_master_summary_stats(df_fan: pd.DataFrame, df_hom: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabla de resumen maestra comparando FANTASIA y Homología de forma clara."""
    prots_fan = set(df_fan['protein'])
    prots_hom = set(df_hom['protein'])
    gos_fan_unique = set(df_fan['GO_term'])
    gos_hom_unique = set(df_hom['GO_term'])
    
    fan_go_counts_per_prot = df_fan.groupby('protein')['GO_term'].nunique()
    hom_go_counts_per_prot = df_hom.groupby('protein')['GO_term'].nunique()
    
    stat, p_val = mannwhitneyu(fan_go_counts_per_prot, hom_go_counts_per_prot, alternative='two-sided')
    
    summary_data = {
        'Metric': [
            "Total Unique Proteins Annotated", "Total Unique GO Terms (Vocabulary)",
            "Proteins in Common", "Unique GO Terms in Common", "Total Proteins Annotated",
            "Total GO Assignments (Inflated)", "Proteins Exclusive to Method",
            "Unique GO Terms Exclusive to Method", "Mean GOs per Protein",
            "Median GOs per Protein", "Mann-Whitney U p-value (Quantity)"
        ],
        'FANTASIA': [
            len(prots_fan.union(prots_hom)), len(gos_fan_unique.union(gos_hom_unique)),
            len(prots_fan.intersection(prots_hom)), len(gos_fan_unique.intersection(gos_hom_unique)),
            len(prots_fan), len(df_fan), len(prots_fan - prots_hom),
            len(gos_fan_unique - gos_hom_unique), fan_go_counts_per_prot.mean(),
            fan_go_counts_per_prot.median(), p_val
        ],
        'Homology': [
            len(prots_fan.union(prots_hom)), len(gos_fan_unique.union(gos_hom_unique)),
            len(prots_fan.intersection(prots_hom)), len(gos_fan_unique.intersection(gos_hom_unique)),
            len(prots_hom), len(df_hom), len(prots_hom - prots_fan),
            len(gos_hom_unique - gos_fan_unique), hom_go_counts_per_prot.mean(),
            hom_go_counts_per_prot.median(), None
        ]
    }
    return pd.DataFrame(summary_data).round(4)

def _create_vocabulary_comparison_summary(df_fan: pd.DataFrame, df_hom: pd.DataFrame, godag: GODag, base_output_dir: Path):
    """Usa compare_go_sets para crear un resumen detallado de la comparación de vocabularios."""
    logger.info("--- Starting GO Vocabulary Comparison Analysis ---")
    gos_fan_total = set(df_fan['GO_term'])
    gos_hom_total = set(df_hom['GO_term'])
    ns_map = {go: godag[go].namespace for go in godag}
    
    fan_sets_by_ns = {ns: {go for go in gos_fan_total if ns_map.get(go) == name} for ns, name in [('BP', 'biological_process'), ('MF', 'molecular_function'), ('CC', 'cellular_component')]}
    hom_sets_by_ns = {ns: {go for go in gos_hom_total if ns_map.get(go) == name} for ns, name in [('BP', 'biological_process'), ('MF', 'molecular_function'), ('CC', 'cellular_component')]}

    results = []
    total_comp = compare_go_sets(gos_fan_total, gos_hom_total); total_comp['ontology'] = 'Total'; results.append(total_comp)
    for ns in ['BP', 'MF', 'CC']:
        ns_comp = compare_go_sets(fan_sets_by_ns[ns], hom_sets_by_ns[ns]); ns_comp['ontology'] = ns; results.append(ns_comp)

    results_df = pd.DataFrame(results).rename(columns={
        'only_in_set1': 'exclusive_to_fantasia', 'only_in_set2': 'exclusive_to_homology',
        'total_unique_set1': 'total_unique_fantasia', 'total_unique_set2': 'total_unique_homology'
    })
    ordered_cols = ['ontology', 'total_unique_fantasia', 'total_unique_homology', 'common_go_terms', 'exclusive_to_fantasia', 'exclusive_to_homology', 'jaccard_index', 'dice_coefficient', 'overlap_coefficient']
    results_df[ordered_cols].to_csv(base_output_dir / "go_vocabulary_comparison.csv", index=False, float_format='%.4f')
    logger.info(f"GO vocabulary comparison results saved to {base_output_dir / 'go_vocabulary_comparison.csv'}")

@handle_graceful_exit

@handle_graceful_exit
def _calculate_external_ic_with_r(all_annotations_df: pd.DataFrame, obo_file: Path, base_dir: Path, r_script_path: Path) -> Dict[str, float]:
    """Calcula el IC usando un script de R externo, con caché."""
    external_ic_file = base_dir / "external_ic_weights.tsv"
    if external_ic_file.exists():
        logger.info(f"Loading cached external IC weights from: {external_ic_file}")
        ic_df = pd.read_csv(external_ic_file, sep='\t', names=['GO_term', 'IC'])
        return pd.Series(ic_df.IC.values, index=ic_df.GO_term).to_dict()

    logger.info("External IC weights not found. Calculating with R script...")
    temp_annotations_file = base_dir / "temp_all_annotations.tsv"
    all_annotations_df.rename(columns={'protein': 'PROTEIN_ID', 'GO_term': 'GO_ID'}).to_csv(temp_annotations_file, sep='\t', index=False)
    command = ["/usr/bin/Rscript", str(r_script_path), str(obo_file), str(temp_annotations_file), str(external_ic_file)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("R script executed successfully.")
        ic_df = pd.read_csv(external_ic_file, sep='\t', names=['GO_term', 'IC'])
        return pd.Series(ic_df.IC.values, index=ic_df.GO_term).to_dict()
    except Exception as e:
        logger.error(f"ERROR: The R script failed to execute. Stderr: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
        return {}
    finally:
        if temp_annotations_file.exists(): temp_annotations_file.unlink()

@handle_graceful_exit
def _perform_and_cache_pca(df_fan: pd.DataFrame, df_hom: pd.DataFrame, output_path: Path):
    """Realiza el análisis PCA y guarda los resultados en caché."""
    if output_path.exists():
        logger.info(f"PCA results already exist. Skipping calculation. Found at: {output_path}"); return
    logger.info("PCA results not found. Performing analysis...")
    pca_results = {}
    for method_name, df in [('fantasia', df_fan), ('homology', df_hom)]:
        if df.empty: continue
        count_matrix = pd.crosstab(df['species'], df['GO_term'])
        if count_matrix.shape[0] < 4 or count_matrix.shape[1] < 3: continue
        scaled_data = StandardScaler().fit_transform(count_matrix)
        pca = PCA(n_components=3)
        pc_coords = pca.fit_transform(scaled_data)
        pc_df = pd.DataFrame(pc_coords, columns=['PC1', 'PC2', 'PC3'], index=count_matrix.index)
        lineage_map = df.drop_duplicates('species').set_index('species')['lineage']
        pc_df = pc_df.join(lineage_map)
        pca_results[f'{method_name}_pca'] = pc_df
        pca_results[f'{method_name}_variance'] = pca.explained_variance_ratio_
    with open(output_path, 'wb') as f: pickle.dump(pca_results, f)
    logger.info(f"PCA results successfully calculated and cached to {output_path}")

def _parse_lineages(lineage_file: Path) -> Dict[str, List[str]]:
    """Parsea el archivo de linajes."""
    try:
        groups = {}
        df = pd.read_csv(lineage_file, header=0)
        for lineage, group_df in df.groupby('lineage'): groups[lineage] = group_df['species'].tolist()
        return groups
    except Exception as e:
        logger.error(f"Failed to parse lineage file {lineage_file}: {e}"); return {}

# --- Funciones de Análisis de Anomalías (Refinadas) ---

def _run_anomaly_summary(unique_gos: set, taxon_map_df: pd.DataFrame) -> Dict[str, Any]:
    """Clasifica un conjunto de GOs únicos usando la lógica jerárquica."""
    if not unique_gos or taxon_map_df.empty: return {"total_unique_gos": 0}
    go_df = pd.DataFrame(list(unique_gos), columns=['GO_term'])
    merged = pd.merge(go_df, taxon_map_df, on='GO_term', how='left')
    merged['category'] = merged['taxon'].apply(_classify_taxon)
    counts = merged['category'].value_counts().to_dict()
    counts['total_unique_gos'] = len(unique_gos)
    return counts

def _calculate_total_anomaly_counts(df: pd.DataFrame, taxon_map_df: pd.DataFrame) -> pd.Series:
    """Calcula el conteo total de asignaciones anómalas, agrupadas por clasificación jerárquica."""
    if df.empty or taxon_map_df.empty: return pd.Series(dtype=int)
    merged = pd.merge(df, taxon_map_df, on='GO_term', how='left')
    merged['category'] = merged['taxon'].apply(_classify_taxon)
    anomalies_df = merged[~merged['category'].isin(['Valid_Plant', 'Valid_Non_Specific', 'General_Noise'])]
    return anomalies_df['category'].value_counts()

def _run_per_species_contamination_diagnosis(all_fantasia_dfs_global: list, all_homology_dfs_global: list, taxon_map_df: pd.DataFrame, base_output_dir: Path):
    """Genera un perfil detallado del origen de las anomalías para cada especie."""
    logger.info("--- Starting Per-Species Contamination Diagnosis (Hierarchical) ---")
    df_fan = pd.concat(all_fantasia_dfs_global, ignore_index=True) if all_fantasia_dfs_global else pd.DataFrame()
    df_hom = pd.concat(all_homology_dfs_global, ignore_index=True) if all_homology_dfs_global else pd.DataFrame()
    all_profiles = []
    for method_name, df_method in [('FANTASIA', df_fan), ('Homology', df_hom)]:
        if df_method.empty: continue
        merged = pd.merge(df_method, taxon_map_df, on='GO_term', how='left')
        merged['category'] = merged['taxon'].apply(_classify_taxon)
        anomalies_df = merged[~merged['category'].isin(['Valid_Plant', 'Valid_Non_Specific', 'General_Noise'])]
        if anomalies_df.empty: continue
        profile = anomalies_df.groupby(['species', 'category']).size().reset_index(name='anomaly_count')
        profile['method'] = method_name; all_profiles.append(profile)
    if not all_profiles: logger.info("No anomalous annotations found."); return
    pd.concat(all_profiles, ignore_index=True).to_csv(base_output_dir / "per_species_contamination_profile.csv", index=False)
    logger.info(f"Per-species contamination diagnosis saved to {base_output_dir / 'per_species_contamination_profile.csv'}")

# --- Funciones de Análisis de Enriquecimiento (Simétricas) ---

def _perform_goea(study: set, pop: set, assoc: dict, godag: GODag, alpha=0.05) -> pd.DataFrame:
    """Función helper para ejecutar el análisis de enriquecimiento GO."""
    if not study or not pop or not assoc: return pd.DataFrame()
    goeaobj = GOEnrichmentStudy(pop, assoc, godag, alpha=alpha, methods=['fdr_bh'])
    goea_results = goeaobj.run_study(study)
    significant_results = [r for r in goea_results if r.p_fdr_bh < alpha]
    return pd.DataFrame([r.__dict__ for r in significant_results]) if significant_results else pd.DataFrame()

def _run_common_proteins_enrichment(df_summary: pd.DataFrame, godag: GODag, base_output_dir: Path):
    """
    Realiza un análisis de enriquecimiento sobre las proteínas comunes a ambos métodos.
    Esta versión corregida usa el nombre de columna correcto ('p_uncorrected').
    """
    logger.info("--- Starting Enrichment Analysis for Common Proteins (Corrected Logic) ---")

    population_prots = set(df_summary['protein'])
    
    is_in_fan = df_summary['fantasia_gos'].str.len() > 0
    is_in_hom = df_summary['homology_gos'].str.len() > 0
    common_prots_df = df_summary[is_in_fan & is_in_hom]
    
    if common_prots_df.empty:
        logger.warning("No common proteins found. Skipping common enrichment analysis.")
        return

    study_prots = set(common_prots_df['protein'])
    
    # --- Ejecutar GOEA para FANTASIA ---
    assoc_fan = df_summary.set_index('protein')['fantasia_gos'].to_dict()
    enrichment_fan = _perform_goea(study_prots, population_prots, assoc_fan, godag)
    if not enrichment_fan.empty:
        # --- INICIO DE LA CORRECCIÓN ---
        # Usar 'p_uncorrected' en lugar de 'p_value'
        enrichment_fan = enrichment_fan[['GO', 'name', 'NS', 'p_fdr_bh', 'p_uncorrected']]
        enrichment_fan.rename(columns={'p_fdr_bh': 'p_fdr_bh_fantasia', 'p_uncorrected': 'p_value_fantasia'}, inplace=True)
        # --- FIN DE LA CORRECCIÓN ---
    else:
        logger.warning("GOEA for FANTASIA on common proteins yielded no results.")
        enrichment_fan = pd.DataFrame()

    # --- Ejecutar GOEA para Homología ---
    assoc_hom = df_summary.set_index('protein')['homology_gos'].to_dict()
    enrichment_hom = _perform_goea(study_prots, population_prots, assoc_hom, godag)
    if not enrichment_hom.empty:
        # --- INICIO DE LA CORRECCIÓN ---
        # Usar 'p_uncorrected' en lugar de 'p_value'
        enrichment_hom = enrichment_hom[['GO', 'p_fdr_bh', 'p_uncorrected']]
        enrichment_hom.rename(columns={'p_fdr_bh': 'p_fdr_bh_homology', 'p_uncorrected': 'p_value_homology'}, inplace=True)
        # --- FIN DE LA CORRECCIÓN ---
    else:
        logger.warning("GOEA for Homology on common proteins yielded no results.")
        enrichment_hom = pd.DataFrame()

    # --- Fusionar los resultados y guardar ---
    if enrichment_fan.empty and enrichment_hom.empty:
        logger.error("Enrichment analysis for common proteins failed for both methods.")
        return
        
    merged_results = pd.merge(enrichment_fan, enrichment_hom, on='GO', how='outer')
    
    merged_results['depth'] = merged_results['GO'].apply(lambda go_id: godag[go_id].depth if go_id in godag else None)
    
    # Esta parte ya es correcta porque opera sobre las columnas renombradas
    merged_results['min_p_value'] = merged_results[['p_value_fantasia', 'p_value_homology']].min(axis=1)
    merged_results.sort_values('min_p_value', inplace=True)
    
    final_cols = ['GO', 'name', 'NS', 'depth', 'p_value_fantasia', 'p_fdr_bh_fantasia', 'p_value_homology', 'p_fdr_bh_homology']
    final_results = merged_results[[col for col in final_cols if col in merged_results.columns]]

    output_file = base_output_dir / "common_proteins_enrichment_comparison.csv"
    final_results.head(1000).to_csv(output_file, index=False, float_format='%.4g')
    logger.info(f"Common proteins enrichment comparison (Top 1000) saved to {output_file}")
    
def _run_fantasia_exclusive_enrichment(df_summary: pd.DataFrame, godag: GODag, base_output_dir: Path):
    """
    Identifica funciones enriquecidas en proteínas anotadas exclusivamente por FANTASIA
    usando un universo de fondo común para una comparación justa y simétrica.
    """
    logger.info("--- Starting FANTASIA-Exclusive Function Enrichment Analysis (Symmetric Logic) ---")
    
    # --- CORRECCIÓN LÓGICA CLAVE ---
    # El universo (pop) debe ser TODAS las proteínas anotadas por CUALQUIER método.
    pop = set(df_summary[(df_summary['fantasia_gos'].str.len() > 0) | (df_summary['homology_gos'].str.len() > 0)]['protein'])
    
    # El grupo de estudio son las proteínas exclusivas de FANTASIA.
    study = set(df_summary[(df_summary['fantasia_gos'].str.len() > 0) & (df_summary['homology_gos'].str.len() == 0)]['protein'])
    
    # La asociación se hace con los términos de FANTASIA para las proteínas del universo.
    assoc = df_summary[df_summary['protein'].isin(pop)].set_index('protein')['fantasia_gos'].to_dict()
    
    if not study:
        logger.warning("No proteins found exclusively in FANTASIA."); return
        
    enrichment_results_df = _perform_goea(study, pop, assoc, godag)
    
    if enrichment_results_df.empty:
        logger.info("No significant enrichment for FANTASIA-exclusive terms."); return
        
    output_file = base_output_dir / "fantasia_exclusive_enrichment.csv"
    enrichment_results_df.sort_values('p_fdr_bh').to_csv(output_file, index=False, float_format='%.4g')
    logger.info(f"FANTASIA-exclusive enrichment results saved to {output_file}")

def _run_homology_exclusive_enrichment(df_summary: pd.DataFrame, godag: GODag, base_output_dir: Path):
    """
    Identifica funciones enriquecidas en proteínas anotadas exclusivamente por Homología
    usando un universo de fondo común para una comparación justa y simétrica.
    """
    logger.info("--- Starting Homology-Exclusive Function Enrichment Analysis (Symmetric Logic) ---")
    
    # --- CORRECCIÓN LÓGICA CLAVE ---
    # El universo (pop) es el mismo que en el análisis de FANTASIA para asegurar una comparación justa.
    pop = set(df_summary[(df_summary['fantasia_gos'].str.len() > 0) | (df_summary['homology_gos'].str.len() > 0)]['protein'])
    
    # El grupo de estudio son las proteínas exclusivas de Homología.
    study = set(df_summary[(df_summary['homology_gos'].str.len() > 0) & (df_summary['fantasia_gos'].str.len() == 0)]['protein'])
    
    # La asociación se hace con los términos de Homología para las proteínas del universo.
    assoc = df_summary[df_summary['protein'].isin(pop)].set_index('protein')['homology_gos'].to_dict()
    
    if not study:
        logger.warning("No proteins found exclusively in Homology."); return
        
    enrichment_results_df = _perform_goea(study, pop, assoc, godag)
    
    if enrichment_results_df.empty:
        logger.info("No significant enrichment for Homology-exclusive terms."); return
        
    output_file = base_output_dir / "homology_exclusive_enrichment.csv"
    enrichment_results_df.sort_values('p_fdr_bh').to_csv(output_file, index=False, float_format='%.4g')
    logger.info(f"Homology-exclusive enrichment results saved to {output_file}")

def _run_brassica_napus_focused_enrichment(df_summary: pd.DataFrame, godag: GODag, base_output_dir: Path):
    """
    Realiza un análisis de enriquecimiento enfocado únicamente en Brassica napus,
    comparando las anotaciones de FANTASIA vs. Homología.
    """
    SPECIES_OF_INTEREST = 'Anisodus_tanguticus01'
    logger.info(f"--- Starting Focused Enrichment Analysis for {SPECIES_OF_INTEREST} ---")

    # 1. Definir el grupo de estudio (proteínas de B. napus) y el de fondo (proteínas de su linaje)
    try:
        b_napus_lineage = df_summary[df_summary['species'] == SPECIES_OF_INTEREST]['lineage'].iloc[0]
        study_prots = set(df_summary[df_summary['species'] == SPECIES_OF_INTEREST]['protein'])
        pop_prots = set(df_summary[df_summary['lineage'] == b_napus_lineage]['protein'])
        logger.info(f"Study group: {len(study_prots)} proteins from {SPECIES_OF_INTEREST}. Population: {len(pop_prots)} proteins from lineage '{b_napus_lineage}'.")
    except IndexError:
        logger.error(f"No se encontró la especie '{SPECIES_OF_INTEREST}' en los datos. Omitiendo análisis.")
        return

    if not study_prots:
        logger.warning(f"No proteins found for {SPECIES_OF_INTEREST}. Skipping enrichment."); return

    # 2. Realizar enriquecimiento para FANTASIA
    assoc_fan = df_summary.set_index('protein')['fantasia_gos'].to_dict()
    enrichment_fan = _perform_goea(study_prots, pop_prots, assoc_fan, godag)
    if not enrichment_fan.empty:
        output_file_fan = base_output_dir / "brassica_napus_fantasia_enrichment.csv"
        enrichment_fan.sort_values('p_fdr_bh').to_csv(output_file_fan, index=False, float_format='%.4g')
        logger.info(f"FANTASIA enrichment for {SPECIES_OF_INTEREST} saved to {output_file_fan}")

    # 3. Realizar enriquecimiento para Homología
    assoc_hom = df_summary.set_index('protein')['homology_gos'].to_dict()
    enrichment_hom = _perform_goea(study_prots, pop_prots, assoc_hom, godag)
    if not enrichment_hom.empty:
        output_file_hom = base_output_dir / "brassica_napus_homology_enrichment.csv"
        enrichment_hom.sort_values('p_fdr_bh').to_csv(output_file_hom, index=False, float_format='%.4g')
        logger.info(f"Homology enrichment for {SPECIES_OF_INTEREST} saved to {output_file_hom}")
# --- Otras Funciones de Análisis Global ---

def _run_global_ic_scenarios_with_depth_filter(df_fan: pd.DataFrame, df_hom: pd.DataFrame, ic_sources: Dict[str, Dict[str, float]], godag: GODag, base_output_dir: Path):
    """
    Realiza un análisis de IC usando un filtro de profundidad de GO como alternativa
    al filtro por puntuación.
    Produce salidas .pkl (para plots) y .csv (para Excel).
    """
    logger.info("--- Starting Global IC Scenarios Analysis (with Depth Filter) ---")

    DEPTH_THRESHOLD = 5
    logger.info(f"Using GO depth threshold >= {DEPTH_THRESHOLD} for filtering.")

    full_distributions = {}
    summary_stats = []

    depth_map = {go_id: godag[go_id].depth for go_id in godag}

    for ic_source_name, ic_weights in ic_sources.items():
        if not ic_weights: continue
        
        full_distributions[ic_source_name] = {}
        
        df_fan['ic'] = df_fan['GO_term'].map(ic_weights)
        df_fan['depth'] = df_fan['GO_term'].map(depth_map)
        df_hom['ic'] = df_hom['GO_term'].map(ic_weights)
        df_hom['depth'] = df_hom['GO_term'].map(depth_map)

        df_fan_clean = df_fan.dropna(subset=['ic', 'depth'])
        df_hom_clean = df_hom.dropna(subset=['ic', 'depth'])

        for method_name, df in [('FANTASIA', df_fan_clean), ('Homology', df_hom_clean)]:
            if df.empty: continue

            # --- Escenario 1: Sin Filtro ---
            scenario_name = 'No_Filter'
            ic_series_no_filter = df['ic']
            
            if scenario_name not in full_distributions[ic_source_name]: full_distributions[ic_source_name][scenario_name] = {}
            full_distributions[ic_source_name][scenario_name][method_name] = ic_series_no_filter
            
            stats = ic_series_no_filter.describe().to_dict()
            # --- CORRECCIÓN AQUÍ ---
            stats.update({'ic_source': ic_source_name, 'method': method_name, 'scenario': scenario_name, 'group': 'all'})
            summary_stats.append(stats)

            # --- Escenario 2: Filtro por Profundidad ---
            scenario_name = 'Depth_Filter'
            df_filtered = df[df['depth'] >= DEPTH_THRESHOLD]
            ic_series_filtered = df_filtered['ic']

            if not ic_series_filtered.empty:
                if scenario_name not in full_distributions[ic_source_name]: full_distributions[ic_source_name][scenario_name] = {}
                full_distributions[ic_source_name][scenario_name][method_name] = ic_series_filtered
                
                stats = ic_series_filtered.describe().to_dict()
                # --- CORRECCIÓN AQUÍ ---
                stats.update({'ic_source': ic_source_name, 'method': method_name, 'scenario': scenario_name, 'group': 'all'})
                summary_stats.append(stats)

    if full_distributions:
        output_file_pkl = base_output_dir / "global_ic_scenarios_distributions.pkl"
        with open(output_file_pkl, 'wb') as f: pickle.dump(full_distributions, f)
        logger.info(f"Full IC distributions for plotting saved to {output_file_pkl}")

    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_df = summary_df.rename(columns={'count': 'annotation_count', 'mean': 'ic_mean', 'std': 'ic_std', 'min': 'ic_min', '25%': 'ic_q1', '50%': 'ic_median', '75%': 'ic_q3', 'max': 'ic_max'})
        # La columna 'group' ahora existirá
        ordered_cols = ['ic_source', 'scenario', 'group', 'method', 'annotation_count', 'ic_mean', 'ic_median', 'ic_std', 'ic_min', 'ic_q1', 'ic_q3', 'ic_max']
        summary_df = summary_df[ordered_cols]
        
        output_file_csv = base_output_dir / "global_ic_scenarios_summary.csv"
        summary_df.to_csv(output_file_csv, index=False, float_format='%.3f')
        logger.info(f"Summary of IC scenarios for Excel saved to {output_file_csv}")

def _run_functional_bias_analysis(df_summary: pd.DataFrame, go_slim_file: Path, godag: GODag, base_output_dir: Path):
    """
    Calcula el índice de sesgo funcional por linaje y categoría GO Slim.
    """
    logger.info("--- Starting Functional Bias Analysis (for Heatmap) ---")
    try:
        with open(go_slim_file, 'r') as f:
            slim_terms_ids = {line.strip().split(' ')[1] for line in f if line.strip().startswith('id: GO:')}
        bp_slim_terms_ids = {go_id for go_id in slim_terms_ids if go_id in godag and godag[go_id].namespace == 'biological_process'}
    except Exception as e:
        logger.error(f"No se pudo procesar el archivo GO Slim {go_slim_file}: {e}"); return

    def _get_slim_counts(go_sets: pd.Series, slim_ids: set, godag: GODag) -> dict:
        all_gos = set.union(*go_sets.dropna())
        counts = {slim_id: 0 for slim_id in slim_ids}
        for go_id in all_gos:
            if go_id in godag:
                for parent_id in godag[go_id].get_all_parents().union({go_id}):
                    if parent_id in slim_ids:
                        counts[parent_id] += 1
        return counts

    all_bias_data = []
    for lineage, group_df in df_summary.groupby('lineage'):
        fan_counts = pd.Series(_get_slim_counts(group_df['fantasia_gos'], bp_slim_terms_ids, godag))
        hom_counts = pd.Series(_get_slim_counts(group_df['homology_gos'], bp_slim_terms_ids, godag))
        bias_df = pd.DataFrame({'FANTASIA': fan_counts, 'Homology': hom_counts}).fillna(0)
        bias_df['bias_index'] = (bias_df['FANTASIA'] - bias_df['Homology']) / (bias_df['FANTASIA'] + bias_df['Homology'] + 1)
        bias_df['lineage'] = lineage
        all_bias_data.append(bias_df.reset_index().rename(columns={'index': 'slim_id'}))
    
    if not all_bias_data:
        logger.warning("No data for functional bias analysis."); return

    heatmap_data = pd.concat(all_bias_data).pivot(index='slim_id', columns='lineage', values='bias_index').fillna(0)
    heatmap_data = heatmap_data.loc[(heatmap_data != 0).any(axis=1)]
    
    # Mapear GO IDs a nombres para que el archivo sea legible
    heatmap_data.index = heatmap_data.index.map(lambda go_id: f"{godag[go_id].name} ({go_id})" if go_id in godag else go_id)
    
    output_file = base_output_dir / "functional_bias_heatmap_data.csv"
    heatmap_data.to_csv(output_file)
    logger.info(f"Functional bias data for heatmap saved to {output_file}")
    
def _calculate_semantic_coherence_with_python(df_summary: pd.DataFrame, base_dir: Path, project_ic_weights: Dict[str, float], godag: GODag):
    """Calcula la coherencia semántica (similitud media intra-grupo) para los grupos de proteínas."""
    logger.info("--- Starting Semantic Coherence Analysis (Pure Python, Optimized) ---")
    semantic_results_file = base_dir / "semantic_coherence_results.csv"
    if semantic_results_file.exists(): logger.info("Semantic coherence results already exist. Skipping."); return
    df_summary['group'] = 'Common'
    df_summary.loc[(df_summary['fantasia_gos'].str.len() > 0) & (df_summary['homology_gos'].str.len() == 0), 'group'] = 'FANTASIA_Only'
    df_summary.loc[(df_summary['homology_gos'].str.len() > 0) & (df_summary['fantasia_gos'].str.len() == 0), 'group'] = 'Homology_Only'
    term_sets = {f"{row['group']}_{row['protein']}": {go for go in (row['fantasia_gos'] if row['group'] != 'Homology_Only' else row['homology_gos']) if go in godag and go in project_ic_weights} for _, row in df_summary.iterrows() if (row['fantasia_gos'] if row['group'] != 'Homology_Only' else row['homology_gos'])}
    def get_mica(g1, g2, godag, ic):
        common_ancestors = (godag[g1].get_all_parents() | {g1}).intersection(godag[g2].get_all_parents() | {g2})
        return max(ic.get(t, 0.0) for t in common_ancestors) if common_ancestors else 0.0
    def resnik_bma(s1, s2, godag, ic):
        if not s1 or not s2: return 0.0
        term_pairs = list(itertools.product(s1, s2)); mica_scores = {pair: get_mica(pair[0], pair[1], godag, ic) for pair in term_pairs}
        best_s1 = sum(max(mica_scores.get((t1, t2), 0.0) for t2 in s2) for t1 in s1) / len(s1)
        best_s2 = sum(max(mica_scores.get((t1, t2), 0.0) for t1 in s1) for t2 in s2) / len(s2)
        return (best_s1 + best_s2) / 2
    results = []
    protein_groups = {group: [p for p in term_sets if p.startswith(group)] for group in ['FANTASIA_Only', 'Homology_Only', 'Common']}
    SAMPLING_THRESHOLD, SAMPLE_SIZE = 2000, 50000
    for group_name, members in protein_groups.items():
        num_members = len(members)
       
        if num_members < 2: 
            results.append({'Group': group_name, 'Mean_Semantic_Similarity': float('nan'), 'Protein_Count': num_members})
            continue
        if num_members > SAMPLING_THRESHOLD:
            p1_indices = [random.randint(0, num_members - 1) for _ in range(SAMPLE_SIZE)]; p2_indices = [random.randint(0, num_members - 1) for _ in range(SAMPLE_SIZE)]
            pair_iterator = ((members[i1], members[i2]) for i1, i2 in zip(p1_indices, p2_indices) if i1 != i2)
        else: pair_iterator = itertools.combinations(members, 2)
        similarities = [resnik_bma(term_sets[p1], term_sets[p2], godag, project_ic_weights) for p1, p2 in pair_iterator]
     
        results.append({'Group': group_name, 'Mean_Semantic_Similarity': sum(similarities) / len(similarities) if similarities else 0.0, 'Protein_Count': num_members})
    pd.DataFrame(results).to_csv(semantic_results_file, index=False, float_format='%.5f')
    logger.info(f"Semantic coherence analysis completed. Results in: {semantic_results_file}")



def _calculate_ic_with_computeic_script(corpus_df: pd.DataFrame, base_dir: Path, compute_ic_script_path: Path) -> Dict[str, float]:
    """
    Calcula el IC usando el script compute_IC.py y extrae el valor de IC
    de la lista de resultados que genera.
    """
    logger.info("Calculating corpus-based IC using 'compute_IC.py' script (precompute mode)...")
    cache_file = base_dir / "project_ic_computeic_script.pkl"
    if cache_file.exists():
        logger.info(f"Loading cached IC weights from: {cache_file}")
        with open(cache_file, 'rb') as f: return pickle.load(f)

    temp_annotations_file = base_dir / "temp_corpus_annotations_for_computeic.tsv"
    temp_output_pickle = base_dir / "temp_ic_from_computeic_script.pickle"
    
    script_working_dir = compute_ic_script_path.parent
    tool_obo_path = script_working_dir / "data" / "go.obo"

    try:
        logger.info(f"Loading tool's OBO file ({tool_obo_path}) to create a valid GO term filter...")
        tool_godag = Ontology(tool_obo_path)
        unique_go_terms = corpus_df['GO_term'].unique()
        valid_go_terms = [go for go in unique_go_terms if go in tool_godag]
        logger.info(f"Pre-filtering complete: {len(valid_go_terms)}/{len(unique_go_terms)} GO terms are valid for the tool.")
        pd.Series(valid_go_terms).to_csv(temp_annotations_file, index=False, header=False)
        
        command = [
            sys.executable, str(compute_ic_script_path),
            str(temp_annotations_file),
            '--precompute',
            '--outputpath', str(temp_output_pickle)
        ]
        logger.info(f"Executing command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=script_working_dir)
        
        logger.info("compute_IC.py script (precompute mode) executed successfully.")
        
        if not temp_output_pickle.exists() or temp_output_pickle.stat().st_size < 50:
            logger.error("compute_IC.py script ran but produced an empty or invalid output file.")
            return {}
            
        with open(temp_output_pickle, 'rb') as f:
            ic_data_from_tool = pickle.load(f)

        # --- ESTA ES LA PARTE IMPORTANTE QUE NO SE ESTÁ EJECUTANDO ---
        ic_weights = {go_id: values[2] for go_id, values in ic_data_from_tool.items() if isinstance(values, list) and len(values) == 3}
        logger.info(f"Successfully extracted {len(ic_weights)} IC values from the tool's output.")

        with open(cache_file, 'wb') as f:
            pickle.dump(ic_weights, f)
            
        return ic_weights

    except Exception as e:
        logger.error(f"An unexpected error occurred while running compute_IC.py script: {e}", exc_info=True)
        return {}
    finally:
        if temp_annotations_file.exists(): temp_annotations_file.unlink()
        if temp_output_pickle.exists(): temp_output_pickle.unlink()
        

# --- NUEVA FUNCIÓN PARA SIMILITUD SEMÁNTICA CON PYGOSEMSIM ---
def _run_semantic_similarity_analysis(df_summary: pd.DataFrame, obo_file: Path, base_dir: Path, method: str = 'resnik'):
    """
    Calcula la similitud semántica entre los conjuntos de anotaciones de FANTASIA y Homología.
    Es flexible y puede usar diferentes métodos como 'resnik', 'lin', o 'wang'.
    """
    logger.info(f"--- Starting Semantic Similarity Analysis (Method: {method.upper()}) ---")
    output_file = base_dir / f"semantic_similarity_scores_{method}.csv"
    if output_file.exists():
        logger.info(f"Semantic similarity scores for method '{method}' already exist. Skipping."); return

    logger.info(f"Loading GO graph from: {obo_file}")
    G = graph.from_obo(str(obo_file))
    
    # El pre-cálculo de 'lower_bounds' es esencial para que Resnik pueda calcular el IC internamente.
    similarity.precalc_lower_bounds(G)
    
    try:
        sim_func = getattr(similarity, method)
    except AttributeError:
        logger.error(f"Similarity method '{method}' not found in pygosemsim. Skipping.")
        return

    pairwise_similarity = lambda t1, t2: sim_func(G, t1, t2)
    
    results = []
    df_ref = df_summary[df_summary['homology_gos'].str.len() > 0]
    df_comp = df_summary[df_summary['fantasia_gos'].str.len() > 0]
    common_prots = set(df_ref['protein']).intersection(set(df_comp['protein']))
    df_subset = df_summary[df_summary['protein'].isin(common_prots)].set_index('protein')

    for _, row in df_subset.iterrows():
        hom_gos = row['homology_gos']
        comp_gos = row['fantasia_gos']
        
        for ns, name in [('BP', 'biological_process'), ('MF', 'molecular_function'), ('CC', 'cellular_component')]:
            hom_gos_ns = {go for go in hom_gos if go in G.nodes and G.nodes[go].get('namespace') == name}
            comp_gos_ns = {go for go in comp_gos if go in G.nodes and G.nodes[go].get('namespace') == name}
            
            if not hom_gos_ns or not comp_gos_ns: continue
            
            sim_score = term_set.sim_bma(hom_gos_ns, comp_gos_ns, pairwise_similarity)
            
            if sim_score is not None and np.isfinite(sim_score):
                results.append({
                    'comparison': f"Homology\nvs\nFANTASIA",
                    'namespace': name,
                    'similarity_score': sim_score
                })

    pd.DataFrame(results).to_csv(output_file, index=False)
    logger.info(f"Semantic similarity scores (Method: {method.upper()}) saved to {output_file}")
    

# --- NUEVA FUNCIÓN PARA GUARDAR DISTRIBUCIONES DE IC ---
def _save_ic_distributions_for_plotting(all_dfs: Dict[str, pd.DataFrame], ic_sources: Dict[str, Dict[str, float]], godag: GODag, base_dir: Path):
    """Prepara y guarda las distribuciones de IC para la figura de densidades."""
    logger.info("--- Saving IC distributions for plotting ---")
    output_file = base_dir / "ic_distributions_long_format.csv"
    if output_file.exists():
        logger.info("IC distributions file already exists. Skipping."); return
        
    results = []
    for method_name, df in all_dfs.items():
        for ic_source_name, ic_weights in ic_sources.items():
            if not ic_weights: continue
            
            df_copy = df.copy()
            df_copy['ic'] = df_copy['GO_term'].map(ic_weights)
            df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_copy.dropna(subset=['ic'], inplace=True)
            
            # --- CORRECCIÓN CLAVE ---
            # Accedemos al namespace como un atributo (.namespace) en lugar de usar .get()
            # También añadimos una comprobación para asegurarnos de que el término GO existe en el godag.
            df_copy['namespace'] = df_copy['GO_term'].apply(
                lambda go: godag[go].namespace if go in godag else None
            )
            df_copy.dropna(subset=['namespace'], inplace=True) # Eliminamos filas donde no se encontró el namespace

            # Crear el formato largo
            for _, row in df_copy.iterrows():
                results.append({
                    'method': method_name,
                    'ic_source': ic_source_name.replace('_', ' '),
                    'ic_value': row['ic'],
                    'namespace': row['namespace']
                })
    
    pd.DataFrame(results).to_csv(output_file, index=False)
    logger.info(f"IC distributions saved to {output_file}")


def _run_stratified_specificity_analysis(df_summary: pd.DataFrame, ic_sources: Dict[str, Dict[str, float]], base_output_dir: Path):
    """
    Realiza un análisis de especificidad (IC) estratificado en tres grupos:
    1. Proteínas exclusivas de FANTASIA.
    2. Proteínas exclusivas de Homología.
    3. Proteínas comunes a ambos, con una comparación pareada para evitar sesgos de dilución.
    """
    logger.info("--- Starting Stratified Specificity Analysis (Corrected Logic) ---")

    # Definir los tres grupos de proteínas
    is_in_fan = df_summary['fantasia_gos'].str.len() > 0
    is_in_hom = df_summary['homology_gos'].str.len() > 0

    df_fan_only = df_summary[is_in_fan & ~is_in_hom]
    df_hom_only = df_summary[~is_in_fan & is_in_hom]
    df_common = df_summary[is_in_fan & is_in_hom]

    logger.info(f"Found {len(df_fan_only)} proteins exclusive to FANTASIA.")
    logger.info(f"Found {len(df_hom_only)} proteins exclusive to Homology.")
    logger.info(f"Found {len(df_common)} proteins common to both methods.")

    exclusive_results = []
    for ic_source_name, ic_weights in ic_sources.items():
        if not ic_weights: continue
        ic_weights_clean = {go: ic for go, ic in ic_weights.items() if np.isfinite(ic)}

        # --- 1. Análisis de Proteínas Exclusivas de FANTASIA ---
        if not df_fan_only.empty:
            all_fan_only_gos = [go for gos_set in df_fan_only['fantasia_gos'] for go in gos_set]
            fan_only_ics = pd.Series([ic_weights_clean.get(go) for go in all_fan_only_gos if ic_weights_clean.get(go) is not None])
            if not fan_only_ics.empty:
                stats = fan_only_ics.describe().to_dict()
                stats.update({'group': 'FANTASIA_Only', 'ic_source': ic_source_name})
                exclusive_results.append(stats)

        # --- 2. Análisis de Proteínas Exclusivas de Homología ---
        if not df_hom_only.empty:
            all_hom_only_gos = [go for gos_set in df_hom_only['homology_gos'] for go in gos_set]
            hom_only_ics = pd.Series([ic_weights_clean.get(go) for go in all_hom_only_gos if ic_weights_clean.get(go) is not None])
            if not hom_only_ics.empty:
                stats = hom_only_ics.describe().to_dict()
                stats.update({'group': 'Homology_Only', 'ic_source': ic_source_name})
                exclusive_results.append(stats)

    if exclusive_results:
        summary_exclusive_df = pd.DataFrame(exclusive_results)
        
        # --- INICIO DE LA CORRECCIÓN ---
        # Renombrar TODAS las columnas de .describe() para que sean legibles y consistentes
        summary_exclusive_df = summary_exclusive_df.rename(columns={
            'count': 'annotation_count',
            'mean': 'ic_mean',
            'std': 'ic_std',
            'min': 'ic_min',
            '25%': 'ic_q1',
            '50%': 'ic_median',  # <-- Esta es la corrección clave que soluciona el error
            '75%': 'ic_q3',
            'max': 'ic_max'
        })
        summary_exclusive_df.to_csv(base_output_dir / "specificity_summary_exclusive_groups.csv", index=False, float_format='%.4f')
        logger.info("Exclusive groups specificity summary saved.")

    # --- 3. Análisis Pareado de Proteínas Comunes (por Linaje) ---
    common_results_by_lineage = []
    if not df_common.empty:
        for lineage, group_df in df_common.groupby('lineage'):
            for ic_source_name, ic_weights in ic_sources.items():
                if not ic_weights: continue
                ic_weights_clean = {go: ic for go, ic in ic_weights.items() if np.isfinite(ic)}

                max_ic_fan = group_df['fantasia_gos'].apply(lambda s: max([ic_weights_clean.get(g, 0) for g in s], default=0))
                max_ic_hom = group_df['homology_gos'].apply(lambda s: max([ic_weights_clean.get(g, 0) for g in s], default=0))

                valid_indices = (max_ic_fan > 0) & (max_ic_hom > 0)
                if valid_indices.sum() < 10: continue

                max_ic_fan_paired = max_ic_fan[valid_indices]
                max_ic_hom_paired = max_ic_hom[valid_indices]

                stat, p_val = wilcoxon(max_ic_fan_paired, max_ic_hom_paired, alternative='two-sided')

                common_results_by_lineage.append({
                    'lineage': lineage,
                    'ic_source': ic_source_name,
                    'protein_pairs_count': len(max_ic_fan_paired),
                    'fantasia_mean_max_ic': max_ic_fan_paired.mean(),
                    'homology_mean_max_ic': max_ic_hom_paired.mean(),
                    'fantasia_median_max_ic': max_ic_fan_paired.median(),
                    'homology_median_max_ic': max_ic_hom_paired.median(),
                    'wilcoxon_p_value': p_val,
                    'wilcoxon_statistic': stat
                })

    if common_results_by_lineage:
        summary_common_df = pd.DataFrame(common_results_by_lineage)
        summary_common_df.to_csv(base_output_dir / "specificity_summary_common_proteins_paired.csv", index=False, float_format='%.4f')
        logger.info("Paired comparison summary for common proteins saved.")

def _run_ic_method_concordance_analysis(ic_sources: Dict[str, Dict[str, float]], base_dir: Path):
    """
    Compara los diferentes métodos de cálculo de IC entre sí.
    1. Guarda un dataframe con los valores de IC de cada método para cada GO term.
    2. Calcula y guarda una matriz de correlación (concordancia) entre los métodos.
    """
    logger.info("--- Running IC Method Concordance Analysis ---")
    
    # Crear un DataFrame con GO terms como índice y métodos como columnas
    df_ic_comparison = pd.DataFrame.from_dict(ic_sources).dropna()
    df_ic_comparison.index.name = 'GO_term'
    
    # Guardar los valores brutos para los boxplots
    output_raw_file = base_dir / "ic_values_by_method.csv"
    df_ic_comparison.reset_index().to_csv(output_raw_file, index=False)
    logger.info(f"Raw IC values by method saved to {output_raw_file}")

    # Calcular la matriz de correlación de Spearman (concordancia)
    # Usamos Spearman porque es no paramétrico y robusto a outliers.
    concordance_matrix = df_ic_comparison.corr(method='spearman')
    
    # Guardar la matriz de concordancia para el heatmap
    output_concordance_file = base_dir / "ic_method_concordance.csv"
    concordance_matrix.to_csv(output_concordance_file)
    logger.info(f"IC method concordance matrix saved to {output_concordance_file}")


# --- Funcion Depth ---
def _run_common_protein_depth_comparison(df_summary: pd.DataFrame, godag: GODag, base_output_dir: Path):
    """
    Para las proteínas comunes, compara la profundidad de sus anotaciones entre FANTASIA y Homología.
    VERSIÓN CORREGIDA: Genera un resumen agregado por linaje en lugar de una tabla masiva por proteína.
    """
    logger.info("--- Starting Common Protein Annotation Depth Comparison (Aggregated by Lineage) ---")
    
    is_in_fan = df_summary['fantasia_gos'].str.len() > 0
    is_in_hom = df_summary['homology_gos'].str.len() > 0
    df_common = df_summary[is_in_fan & is_in_hom].copy()

    if df_common.empty:
        logger.warning("No common proteins found for depth comparison. Skipping.")
        return

    depth_map = {go_id: godag[go_id].depth for go_id in godag}

    def get_max_depth(go_set):
        if not go_set: return 0
        depths = [depth_map.get(go, 0) for go in go_set]
        return max(depths) if depths else 0

    # Calcular la profundidad MÁXIMA por proteína para cada método
    df_common['fan_max_depth'] = df_common['fantasia_gos'].apply(get_max_depth)
    df_common['hom_max_depth'] = df_common['homology_gos'].apply(get_max_depth)

    # --- Lógica de Agregación por Linaje ---
    lineage_summary = []
    for lineage, group_df in df_common.groupby('lineage'):
        
        # Realizar una prueba estadística pareada (Wilcoxon) para ver si hay una diferencia significativa
        # entre las profundidades máximas de los dos métodos dentro de ese linaje.
        try:
            stat, p_val = wilcoxon(group_df['fan_max_depth'], group_df['hom_max_depth'])
        except ValueError:
            # Esto puede ocurrir si todos los valores son idénticos (ej. todos cero)
            stat, p_val = None, None

        lineage_summary.append({
            'lineage': lineage,
            'protein_count': len(group_df),
            'fantasia_mean_max_depth': group_df['fan_max_depth'].mean(),
            'homology_mean_max_depth': group_df['hom_max_depth'].mean(),
            'fantasia_median_max_depth': group_df['fan_max_depth'].median(),
            'homology_median_max_depth': group_df['hom_max_depth'].median(),
            'wilcoxon_p_value': p_val,
            'wilcoxon_statistic': stat
        })

    if not lineage_summary:
        logger.warning("Could not generate lineage summary for depth comparison.")
        return

    # Crear el DataFrame final, que será pequeño y manejable
    summary_df = pd.DataFrame(lineage_summary)
    
    output_file = base_output_dir / "common_proteins_depth_comparison.csv"
    summary_df.to_csv(output_file, index=False, float_format='%.3f')
    logger.info(f"Aggregated depth comparison summary by lineage saved to {output_file}")

# #############################################################################
# --- ORQUESTADOR PRINCIPAL DEL MÓDULO DE ANÁLISIS ---
# #############################################################################

@handle_graceful_exit
@handle_graceful_exit
def run_go_analysis_suite(config: Config, input_files: List[Tuple[Path, Dict[str, Path]]], lineage_file: Path, taxon_map_file: Path, obo_file: Path, go_slim_file: Path) -> None:
    """
    Orquestador principal y completo del pipeline de análisis comparativo.
    Esta función carga los datos, calcula métricas de múltiples maneras (incluyendo
    herramientas estándar como Compute_IC y pygosemsim), y ejecuta una suite
    completa de análisis estratificados y globales para comparar los métodos de
    anotación FANTASIA y Homología.
    """
    # --- FASE 1: CONFIGURACIÓN E INICIALIZACIÓN ---
    logger.info("=" * 80)
    logger.info("Starting Comparative GO Analysis Suite (v7.0 - Comprehensive & Standardized)")
    logger.info("=" * 80)
    base_output_dir = Path(config.output_dir) / config.stats.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading auxiliary data files (OBO, lineages, taxon map)...")
    godag = GODag(str(obo_file))
    lineage_dict = _parse_lineages(lineage_file)
    taxon_map_df = pd.read_csv(taxon_map_file, sep="\t", names=['GO_term', 'taxon'])
    
    # --- FASE 2: CARGA Y AGREGACIÓN DE DATOS DE ANOTACIÓN ---
    logger.info("Parsing all annotation files from input directories...")
    species_files = {dir_path.parent.name: files_dict for dir_path, files_dict in input_files}
    
    # Cargar todas las anotaciones en un único DataFrame para el corpus
    all_dfs_raw = [df for _, files_dict in input_files for file in files_dict.values() if (df := parse_go_annotations(file)) is not None]
    if not all_dfs_raw:
        logger.error("FATAL: No valid annotation files found. Stopping analysis.")
        return
    corpus_df = pd.concat(all_dfs_raw, ignore_index=True).drop_duplicates(subset=['protein', 'GO_term'])
    logger.info(f"Corpus created with {len(corpus_df)} unique protein-GO assignments.")

    # --- FASE 3: CÁLCULO DE PESOS DE IC DESDE MÚLTIPLES FUENTES ---
    logger.info("Calculating Information Content (IC) weights from multiple sources...")
    
    ic_sources = {}
    
    # Método 1: Basado en la frecuencia de términos en nuestro corpus (usando goatools)
    term_counts = TermCounts(godag, corpus_df.groupby('protein')['GO_term'].apply(set).to_dict())
    ic_sources['Project_goatools'] = {go: get_info_content(go, term_counts) for go in godag if get_info_content(go, term_counts) is not None}
    

    # Método 2: Usando el SCRIPT compute_IC.py del paper, llamado como proceso externo
    if config.compute_ic_script and config.compute_ic_script.exists():
        # --- CORRECCIÓN: Eliminamos el argumento 'obo_file' de la llamada ---
        ic_sources['Project_ComputeIC_Script'] = _calculate_ic_with_computeic_script(
            corpus_df, base_output_dir, config.compute_ic_script
        )
    else:
        logger.warning("Path to 'compute_ic.py' script not provided or not found in config. Skipping this IC source.")
    
    
    # Limpiar las fuentes que podrían haber fallado y devuelto un diccionario vacío
    ic_sources = {k: v for k, v in ic_sources.items() if v}
    logger.info(f"Successfully calculated IC weights from {len(ic_sources)} sources: {list(ic_sources.keys())}")

    # --- AÑADE ESTA LLAMADA ---
    if len(ic_sources) > 1:
        logger.info(f"Starting corcondance analysis {len(ic_sources)} sources: {list(ic_sources.keys())}")
        _run_ic_method_concordance_analysis(ic_sources, base_output_dir)
        
    # --- FASE 4: PROCESAMIENTO POR LINAJE Y CREACIÓN DE DATAFRAMES GLOBALES ---
    all_fantasia_dfs_global, all_homology_dfs_global = [], []
    for lineage, species_in_lineage in lineage_dict.items():
        logger.info(f"--- Processing Lineage: {lineage} ---")
        lineage_fantasia_dfs, lineage_homology_dfs = [], []
        for s in species_in_lineage:
            if s in species_files:
                if (df_fan := parse_go_annotations(species_files[s].get('fantasia'))) is not None:
                    df_fan['species'], df_fan['lineage'] = s, lineage
                    lineage_fantasia_dfs.append(df_fan)
                if (df_hom := parse_go_annotations(species_files[s].get('homology'))) is not None:
                    df_hom['species'], df_hom['lineage'] = s, lineage
                    lineage_homology_dfs.append(df_hom)
        
        if lineage_fantasia_dfs:
            all_fantasia_dfs_global.append(pd.concat(lineage_fantasia_dfs, ignore_index=True))
        if lineage_homology_dfs:
            all_homology_dfs_global.append(pd.concat(lineage_homology_dfs, ignore_index=True))

    # --- FASE 5: GENERACIÓN DE RESUMEN GLOBAL DE PROTEÍNAS ---
    logger.info("Aggregating all lineage data into global dataframes...")
    global_fantasia_df = pd.concat(all_fantasia_dfs_global, ignore_index=True) if all_fantasia_dfs_global else pd.DataFrame()
    global_homology_df = pd.concat(all_homology_dfs_global, ignore_index=True) if all_homology_dfs_global else pd.DataFrame()

    logger.info("Creating the final protein summary dataframe (for stratified analysis)...")
    fan_summary = global_fantasia_df.groupby('protein').agg(fantasia_gos=('GO_term', set), species=('species', 'first'), lineage=('lineage', 'first')) if not global_fantasia_df.empty else pd.DataFrame()
    hom_summary = global_homology_df.groupby('protein').agg(homology_gos=('GO_term', set)) if not global_homology_df.empty else pd.DataFrame()
    
    final_summary = fan_summary.join(hom_summary, how='outer').reset_index()
    final_summary['fantasia_gos'] = final_summary['fantasia_gos'].apply(lambda d: d if isinstance(d, set) else set())
    final_summary['homology_gos'] = final_summary['homology_gos'].apply(lambda d: d if isinstance(d, set) else set())
    final_summary.to_pickle(base_output_dir / "global_protein_summaries.pkl")
    logger.info("Global protein summary saved. This file is key for subsequent analyses.")

    # --- FASE 6: EJECUCIÓN DE LA SUITE COMPLETA DE ANÁLISIS ---
    
    logger.info("--- Running: Master Summary Statistics ---")
    _create_master_summary_stats(global_fantasia_df, global_homology_df).to_csv(base_output_dir / "master_summary_statistics.csv", index=False)
    
    logger.info("--- Running: GO Vocabulary Comparison ---")
    _create_vocabulary_comparison_summary(global_fantasia_df, global_homology_df, godag, base_output_dir)
    
    logger.info("--- Running: PCA Analysis ---")
    _perform_and_cache_pca(global_fantasia_df, global_homology_df, base_output_dir / "pca_results.pkl")

    logger.info("--- Running: Stratified Specificity Analysis  ---")
    _run_stratified_specificity_analysis(final_summary, ic_sources, base_output_dir)
    
    
    logger.info("--- Running: Semantic Similarity Analysis ---")
    
    # Cambiamos 'wang' por 'resnik'.
    _run_semantic_similarity_analysis(final_summary, obo_file, base_output_dir, method='resnik')
    
    logger.info("--- Running: Symmetric Exclusive Enrichment Analysis ---")
    _run_fantasia_exclusive_enrichment(final_summary, godag, base_output_dir)
    _run_homology_exclusive_enrichment(final_summary, godag, base_output_dir)
    
    logger.info("--- Running:  Common Enrichment Analysis  ---")
    _run_common_proteins_enrichment(final_summary, godag, base_output_dir)
    
    logger.info("--- Running:  Common Enrichment Analysis  ---")
    _run_common_proteins_enrichment(final_summary, godag, base_output_dir)
    
    # --- NUEVA LLAMADA ---
    logger.info("--- Running: Focused Enrichment for Brassica napus ---")
    _run_brassica_napus_focused_enrichment(final_summary, godag, base_output_dir)
    
    logger.info("--- Running: Per-Species Contamination Diagnosis ---")
    _run_per_species_contamination_diagnosis(all_fantasia_dfs_global, all_homology_dfs_global, taxon_map_df, base_output_dir)
    
    logger.info("--- Running: Functional Bias Analysis (for Heatmap) ---")
    _run_functional_bias_analysis(final_summary, go_slim_file, godag, base_output_dir)
    
    logger.info("--- Running: Common Protein Annotation Depth Comparison ---")
    _run_common_protein_depth_comparison(final_summary, godag, base_output_dir)

    # --- FASE 7: PREPARACIÓN DE DATOS PARA GRÁFICAS ESPECÍFICAS ---
    
    all_method_dfs = {"FANTASIA": global_fantasia_df, "Homology": global_homology_df}
    
    logger.info("--- Running: Save IC Distributions for Plotting ---")
    _save_ic_distributions_for_plotting(all_method_dfs, ic_sources, godag, base_output_dir)
    
   

    
    
    
    logger.info("--- Running: Quantitative Outlier Validation ---")
    # Necesitamos los resultados del PCA para saber qué método analizar
    pca_results_path = base_output_dir / "pca_results.pkl"
    if pca_results_path.exists():
        with open(pca_results_path, 'rb') as f:
            pca_results = pickle.load(f)
        _run_outlier_validation_analysis(
            all_fantasia_dfs_global, 
            all_homology_dfs_global, 
            taxon_map_df, 
            pca_results, 
            base_output_dir
        )
        
    logger.info("--- Running: Comprehensive Lineage Profile Comparison ---")
    _run_lineage_profile_comparison(
        all_fantasia_dfs_global,
        all_homology_dfs_global,
        taxon_map_df,
        base_output_dir
    )
        
    logger.info("=" * 80)
    logger.info("GO Analysis Suite completed successfully.")
    logger.info("=" * 80)