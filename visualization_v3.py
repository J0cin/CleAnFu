import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import textwrap
import numpy as np
import pickle
import ast
from typing import Dict
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D


from goatools.obo_parser import GODag
from scipy.stats import mannwhitneyu
from matplotlib_venn import venn2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from goatools.goea.go_enrichment_ns import GOEnrichmentStudy
from mpl_toolkits.mplot3d import Axes3D


logger = logging.getLogger(__name__)
FANTASIA_COLOR = '#ff8c00'
HOMOLOGY_COLOR = '#1e90ff'
PALETTE = {'FANTASIA': FANTASIA_COLOR, 'Homology': HOMOLOGY_COLOR}
CONTAMINATION_COLOR = '#d62728' # Rojo
HALLUCINATION_COLOR = '#8c564b' # Marrón

def _aggregate_taxons_for_radar(df_counts: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los conteos de taxones en categorías principales para los gráficos de radar.
    Define qué es contaminación y qué se ignora.
    """
    if df_counts.empty:
        return pd.DataFrame()

    # Definir las categorías principales y los taxones que pertenecen a cada una
    # Esto es personalizable
    category_map = {
        'Bacteria': 'Bacteria',
        'Archaea': 'Bacteria', # Agrupamos Archaea con Bacteria
        'Fungi': 'Fungi',
        'Metazoa': 'Metazoa',
        'Viruses': 'Viruses',
        # Añade más mapeos si es necesario
    }
    
    # Taxones a ignorar (no son contaminación)
    taxons_to_ignore = [
        'Eukaryota', 'root', 'cellular organisms', 'Opisthokonta',
        'Viridiplantae', 'Non taxon-specific'
    ]

    df_filtered = df_counts[~df_counts['taxon'].isin(taxons_to_ignore)].copy()
    df_filtered['category'] = df_filtered['taxon'].map(category_map).fillna('Other Eukaryotes')
    
    # Agrupar por la nueva categoría y sumar los conteos
    agg_df = df_filtered.groupby('category')[['FANTASIA', 'Homology']].sum().reset_index()
    
    return agg_df.rename(columns={'category': 'taxon'})

def _get_significance_stars(p_value: float) -> str:
    """Convierte un p-valor en una cadena de asteriscos según la convención estándar."""
    if p_value <= 0.0001:
        return '****'
    elif p_value <= 0.001:
        return '***'
    elif p_value <= 0.01:
        return '**'
    elif p_value <= 0.05:
        return '*'
    else:
        return 'ns' # No Significativo


def plot_figure_1a_coverage_venn(df_master_summary: pd.DataFrame, proteins_fan: set, proteins_hom: set, output_dir: Path):
    """Genera la Figura 1A: Paneles de Venn para cobertura de proteínas y GOs."""
    logger.info("Generando Figura 1A: Cobertura (Venn Diagrams)...")
    
    fig, (ax_venn_prot, ax_venn_go) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Panel A: Cobertura de Proteínas
    v1 = venn2([proteins_fan, proteins_hom], 
          ax=ax_venn_prot, 
          set_colors=(FANTASIA_COLOR, HOMOLOGY_COLOR), 
          alpha=0.7, 
          set_labels=('FANTASIA', 'Homology'))
    for text in v1.set_labels: text.set_fontsize(18)
    for text in v1.subset_labels: text.set_fontsize(16)
    ax_venn_prot.set_title("A) Cobertura de Proteínas Anotadas", fontsize=20, weight='bold')

    # Panel B: Universo de Términos GO
    try:
        exclusive_fan = df_master_summary.query("Metric == 'Unique GO Terms Exclusive to Method'")['FANTASIA'].iloc[0]
        exclusive_hom = df_master_summary.query("Metric == 'Unique GO Terms Exclusive to Method'")['Homology'].iloc[0]
        common_gos = df_master_summary.query("Metric == 'Unique GO Terms in Common'")['FANTASIA'].iloc[0]
        venn_gos_subsets = (exclusive_fan, exclusive_hom, common_gos)
    except (IndexError, KeyError):
        logger.error("No se pudieron extraer los datos para el Venn de GOs. Omitiendo.")
        venn_gos_subsets = (0, 0, 0)
        
    v2 = venn2(subsets=venn_gos_subsets, 
          ax=ax_venn_go, 
          set_colors=(FANTASIA_COLOR, HOMOLOGY_COLOR), 
          alpha=0.7, 
          set_labels=('FANTASIA', 'Homology'))
    for text in v2.set_labels: text.set_fontsize(18)
    for text in v2.subset_labels: text.set_fontsize(16)
    ax_venn_go.set_title("B) Universo de Términos GO Anotados", fontsize=20, weight='bold')
    
    # --- MEJORA: Aumentar tamaño de la leyenda ---
    fig.legend(handles=[mpatches.Patch(color=FANTASIA_COLOR, label='FANTASIA'), 
                       mpatches.Patch(color=HOMOLOGY_COLOR, label='Homología')], 
               loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=18, frameon=True)
               
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / "Figure_1a_Coverage_Venn.png", dpi=600)
    plt.close()

def plot_figure_1b_quantity_by_lineage(df_summary: pd.DataFrame, output_dir: Path):
    """Genera la Figura 1B: Boxplot de cantidad de anotaciones por linaje y método."""
    logger.info("Generando Figura 1B: Cantidad de Anotaciones por Linaje (Boxplot)...")

    df_summary['fantasia_go_count'] = df_summary['fantasia_gos'].str.len()
    df_summary['homology_go_count'] = df_summary['homology_gos'].str.len()
    
    df_fan = df_summary[df_summary['fantasia_go_count'] > 0][['lineage', 'fantasia_go_count']].rename(columns={'fantasia_go_count': 'go_count'})
    df_fan['source'] = 'FANTASIA'
    
    df_hom = df_summary[df_summary['homology_go_count'] > 0][['lineage', 'homology_go_count']].rename(columns={'homology_go_count': 'go_count'})
    df_hom['source'] = 'Homology'
    
    df_plot_data = pd.concat([df_fan, df_hom], ignore_index=True)
    
    plt.figure(figsize=(20, 12)) # --- MEJORA: Aumentar altura para mejor espaciado ---
    ax = sns.boxplot(data=df_plot_data, x='lineage', y='go_count', hue='source', palette=PALETTE, showfliers=False)
    
    ax.set_xlabel("Linaje", fontsize=20)
    ax.set_ylabel("Número de Términos GO / Proteína", fontsize=20)
    
    ax.tick_params(axis='x', rotation=45, labelsize=16)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(title='Método', fontsize=18, title_fontsize=20)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / "Figure_1b_Quantity_by_Lineage.png", dpi=600)
    plt.close()

def plot_figure_2_exclusive_specificity(df_exclusive: pd.DataFrame, output_dir: Path):
    """Genera un gráfico de barras comparando la especificidad mediana de los vocabularios exclusivos."""
    logger.info("Generando Figura 2: Especificidad de Vocabularios Exclusivos...")
    if df_exclusive.empty: return

    available_sources = df_exclusive['ic_source'].unique()
    ic_source_to_plot = 'Project_ComputeIC_Script' if 'Project_ComputeIC_Script' in available_sources else available_sources[0]
    df_plot = df_exclusive[df_exclusive['ic_source'] == ic_source_to_plot].copy()
    if df_plot.empty: return

    df_plot['group_label'] = df_plot['group'].map({
        'FANTASIA_Only': 'Exclusivas FANTASIA',
        'Homology_Only': 'Exclusivas Homología'
    })

    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        data=df_plot, 
        x='group_label', 
        y='ic_median', 
        hue='group_label',
        palette={'Exclusivas FANTASIA': FANTASIA_COLOR, 'Exclusivas Homología': HOMOLOGY_COLOR},
        legend=False
    )
    
    
    ax.set_xlabel("Grupo de Proteínas", fontsize=18)
    ax.set_ylabel("Especificidad Mediana (IC Mediano)", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # --- MEJORA: Aumentar tamaño de anotaciones ---
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points',
                    fontsize=16, weight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "Figure_2_Exclusive_Specificity.png", dpi=600)
    plt.close()

def plot_figure_3_enrichment_panels(df_fan_enrich: pd.DataFrame, df_hom_enrich: pd.DataFrame, output_dir: Path):
    """
    Muestra los paneles de enriquecimiento en formato vertical (2x1) para A4,
    con títulos forzados a la izquierda para garantizar que no se corten.
    """
    logger.info("Generando Figura 3: Paneles de Enriquecimiento (Vertical)...")
    if df_fan_enrich.empty or df_hom_enrich.empty: return
    
    fig, (ax_enrich_fan, ax_enrich_hom) = plt.subplots(2, 1, figsize=(18, 30))
    
    _plot_enrichment_panel(df_fan_enrich, ax_enrich_fan, FANTASIA_COLOR)
    _plot_enrichment_panel(df_hom_enrich, ax_enrich_hom, HOMOLOGY_COLOR)
    
    # --- CORRECCIÓN DEFINITIVA: Usar coordenadas negativas para forzar el título a la izquierda ---
    # Esto mueve el inicio del texto al margen izquierdo, fuera del área del gráfico.
    # `tight_layout` y `savefig` expandirán la figura para incluirlo.
    
    # Título para el panel A
    ax_enrich_fan.text(-0.35, 1.02, "A) Enriquecimiento en Proteínas Exclusivas de FANTASIA", 
                       transform=ax_enrich_fan.transAxes,
                       fontsize=30, 
                       fontweight='bold', 
                       va='bottom',
                       ha='left')

    # Título para el panel B
    ax_enrich_hom.text(-0.35, 1.02, "B) Enriquecimiento en Proteínas Exclusivas de Homología", 
                       transform=ax_enrich_hom.transAxes,
                       fontsize=30, 
                       fontweight='bold', 
                       va='bottom',
                       ha='left')

    # Ajustar el espaciado para dar cabida a los nuevos títulos
    plt.tight_layout(pad=5.0, h_pad=8.0)
    plt.savefig(output_dir / "Figure_3_Enrichment_Panels_Vertical.png", dpi=600)
    plt.close()
    
def plot_figure_3b_heatmap(heatmap_data: pd.DataFrame, output_dir: Path):
    """Muestra el heatmap de sesgo funcional como un CLUSTERMAP."""
    logger.info("Generando Figura 3b: Clustermap de Sesgo Funcional...")
    if heatmap_data.empty: return
    heatmap_data = heatmap_data.loc[(heatmap_data != 0).any(axis=1), (heatmap_data != 0).any(axis=0)]
    if heatmap_data.empty: return

    clustergrid = sns.clustermap(
        heatmap_data,
        cmap='coolwarm',
        center=0,
        annot=False,
        linewidths=.75,
        figsize=(18, max(12, len(heatmap_data) * 0.7)),
        cbar_kws={'label': 'Índice de Sesgo (Positivo=FANTASIA, Negativo=Homología)'}
    )
    
    plt.setp(clustergrid.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=14)
    plt.setp(clustergrid.ax_heatmap.get_yticklabels(), fontsize=14)
    
    # --- CORRECCIÓN: Acceder de forma segura al eje de la barra de color ---
    if hasattr(clustergrid, 'cbar_ax'):
        cbar_label = 'Índice de Sesgo\n(Positivo=FANTASIA, Negativo=Homología)'
        clustergrid.cbar_ax.set_ylabel(cbar_label, fontsize=16)
        clustergrid.cbar_ax.tick_params(labelsize=14)
    
    output_path = output_dir / "Figure_3b_Functional_Bias_Clustermap.png"
    clustergrid.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(clustergrid.fig)


    
def plot_figure_4_pca_composite(pca_results: dict, output_dir: Path):
    logger.info("Generando Figura 4: Panel Compuesto de Análisis PCA (desde caché)...")
    fig = plt.figure(figsize=(30, 18))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.5, 1], height_ratios=[1, 1])
    ax_3d_fan = fig.add_subplot(gs[0, 0], projection='3d')
    ax_3d_hom = fig.add_subplot(gs[1, 0], projection='3d')
    ax_2d_fan = fig.add_subplot(gs[0, 1])
    ax_2d_hom = fig.add_subplot(gs[1, 1])
    axes_map = {
        'FANTASIA': {'3d': ax_3d_fan, '2d': ax_2d_fan, 'data': pca_results.get('fantasia_pca'), 'var': pca_results.get('fantasia_variance', [0,0,0])},
        'Homología': {'3d': ax_3d_hom, '2d': ax_2d_hom, 'data': pca_results.get('homology_pca'), 'var': pca_results.get('homology_variance', [0,0,0])}
    }

    # --- CORRECCIÓN LEYENDA: Preparar handles manualmente fuera del loop ---
    all_lineages = pd.concat([axes_map['FANTASIA']['data'], axes_map['Homología']['data']])['lineage'].unique()
    palette_legend = sns.color_palette("husl", n_colors=len(all_lineages))
    color_map_legend = dict(zip(all_lineages, palette_legend))
    legend_handles = [mpatches.Patch(color=color, label=name) for name, color in color_map_legend.items()]

    for method_name, axes in axes_map.items():
        pc_df = axes['data']
        if pc_df is None or pc_df.empty: continue
        ax3d, ax2d, variance = axes['3d'], axes['2d'], axes['var']
        
        for lineage_name, group in pc_df.groupby('lineage'):
            color = color_map_legend[lineage_name]
            centroid = group[['PC1', 'PC2', 'PC3']].mean()
            distances = np.sqrt(np.sum((group[['PC1', 'PC2', 'PC3']] - centroid)**2, axis=1))
            outlier_threshold = distances.mean() + 2.5 * distances.std()
            
            # --- MEJORA: Puntos y centroides más grandes ---
            ax3d.scatter(group['PC1'], group['PC2'], group['PC3'], s=120, alpha=0.7, c=[color])
            ax3d.scatter(centroid['PC1'], centroid['PC2'], centroid['PC3'], s=350, marker='o', c=[color], depthshade=False, edgecolor='black', linewidth=2)
            ax2d.scatter(group['PC1'], group['PC2'], s=120, alpha=0.7, c=[color])
            ax2d.scatter(centroid['PC1'], centroid['PC2'], s=350, marker='o', c=[color], edgecolor='black', linewidth=2)
            
            if len(group) > 2:
                try:
                    points = group[['PC1', 'PC2']].values
                    hull = ConvexHull(points)
                    poly = Polygon(points[hull.vertices,:], facecolor=color, alpha=0.2)
                    ax2d.add_patch(poly)
                except Exception:
                    pass

            for idx, row in group.iterrows():
                ax3d.plot([row['PC1'], centroid['PC1']], [row['PC2'], centroid['PC2']], [row['PC3'], centroid['PC3']], c=color, linestyle='--', linewidth=1.5, alpha=0.6)
                ax2d.plot([row['PC1'], centroid['PC1']], [row['PC2'], centroid['PC2']], c=color, linestyle='--', linewidth=1.5, alpha=0.6)
                
                # --- CORRECCIÓN: Se eliminan las etiquetas de texto de los outliers para no saturar el gráfico ---
                # if distances.get(idx, 0) > outlier_threshold and len(group) > 1:
                #     ax2d.text(row['PC1'], row['PC2'], idx, fontsize=12, color='black', weight='bold')
                #     ax3d.text(row['PC1'], row['PC2'], row['PC3'], idx, fontsize=12, color='black', weight='bold')
        
        # --- MEJORA: Textos aún más grandes ---
        ax3d.set_title(f"PCA 3D - {method_name}", fontsize=22)
        ax3d.set_xlabel(f"PC1 ({variance[0]:.1%})", fontsize=20); ax3d.set_ylabel(f"PC2 ({variance[1]:.1%})", fontsize=20); ax3d.set_zlabel(f"PC3 ({variance[2]:.1%})", fontsize=20)
        ax3d.tick_params(axis='both', labelsize=16)
        ax2d.set_title(f"PCA 2D - {method_name}", fontsize=22)
        ax2d.set_xlabel(f"PC1 ({variance[0]:.1%})", fontsize=20); ax2d.set_ylabel(f"PC2 ({variance[1]:.1%})", fontsize=20)
        ax2d.tick_params(axis='both', labelsize=18)
        ax2d.grid(True, linestyle='--', alpha=0.6)

    # --- CORRECCIÓN LEYENDA: Usar los handles manuales ---
    fig.legend(handles=legend_handles, loc='center right', title='Linaje', bbox_to_anchor=(1.08, 0.5), fontsize=20, title_fontsize=22)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_dir / "Figure_4_PCA_Composite_Final.png", dpi=600)
    plt.close()


def plot_figure_4_pca_diagnostic(pca_results: dict, output_dir: Path):
    logger.info("Generando Figura de Diagnóstico: PCA 2D con todas las etiquetas...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 28))
    # fig.suptitle("Figura de Diagnóstico: PCA 2D con todas las etiquetas de especies", fontsize=20)
    for ax, method_name in zip(axes, ['fantasia', 'homology']):
        pc_df = pca_results.get(f'{method_name}_pca')
        variance = pca_results.get(f'{method_name}_variance', [0,0])
        if pc_df is None or pc_df.empty: continue
        sns.scatterplot(x='PC1', y='PC2', hue='lineage', data=pc_df, s=100, ax=ax, legend='full')
        for idx, row in pc_df.iterrows():
            ax.text(row['PC1']+0.01, row['PC2']+0.01, idx, fontsize=8)
        ax.set_title(f"PCA 2D - {method_name.capitalize()}")
        ax.set_xlabel(f"PC1 ({variance[0]:.1%})")
        ax.set_ylabel(f"PC2 ({variance[1]:.1%})")
        ax.grid(True, linestyle='--')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_dir / "Supp_Figure_PCA_Diagnostic_Labels.png", dpi=300)
    plt.close()

def plot_figure_4_pca_2d(pca_results: dict, output_dir: Path):
    """Genera una figura comparativa de los PCA 2D en paneles verticales con leyendas en español."""
    logger.info("Generando Figura 4 (Principal): Comparación de PCA 2D en paneles verticales...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 30), sharex=True)
    
    all_lineages = pd.concat([
        pca_results.get('fantasia_pca', pd.DataFrame()), 
        pca_results.get('homology_pca', pd.DataFrame())
    ])['lineage'].unique()
    palette = sns.color_palette("husl", n_colors=len(all_lineages))
    color_map = dict(zip(all_lineages, palette))

    translation_map_lineages = {
        'chlorophyta': 'Clorofitas',
        'embryophyta': 'Embriófitas',
        'eudicots': 'Eudicotiledóneas',
        'fabales': 'Fabales',
        'liliopsida': 'Liliopsida',
        'magnoliids': 'Magnólidas'
    }
    
    legend_handles = [mpatches.Patch(color=color, label=translation_map_lineages.get(name, name)) 
                      for name, color in color_map.items()]

    panel_labels = 'AB'
    display_names = {
        'fantasia': 'FANTASIA',
        'homology': 'Homología'
    }

    for i, (ax, method_name_key) in enumerate(zip(axes, ['fantasia', 'homology'])):
        pc_df = pca_results.get(f'{method_name_key}_pca')
        variance = pca_results.get(f'{method_name_key}_variance', [0,0,0])
        
        if pc_df is None or pc_df.empty:
            ax.text(0.5, 0.5, f"No hay datos de PCA para\n{display_names[method_name_key]}", ha='center', va='center', fontsize=24)
            continue

        # El bucle interno sigue usando los nombres originales ('chlorophyta', etc.) para agrupar los datos
        for lineage_name, group in pc_df.groupby('lineage'):
            color = color_map[lineage_name]
            centroid = group[['PC1', 'PC2']].mean()
            
            ax.scatter(group['PC1'], group['PC2'], s=180, alpha=0.7, c=[color])
            ax.scatter(centroid['PC1'], centroid['PC2'], s=500, marker='o', c=[color], edgecolor='black', linewidth=2.5)
            
            if len(group) > 2:
                try:
                    points = group[['PC1', 'PC2']].values
                    hull = ConvexHull(points)
                    poly = Polygon(points[hull.vertices,:], facecolor=color, alpha=0.2)
                    ax.add_patch(poly)
                except Exception:
                    pass

            for _, row in group.iterrows():
                ax.plot([row['PC1'], centroid['PC1']], [row['PC2'], centroid['PC2']], c=color, linestyle='--', linewidth=1.5, alpha=0.6)
        
        title_text = f"{panel_labels[i]}) PCA {display_names[method_name_key]}"
        ax.set_title(title_text, fontsize=28, weight='bold')
        
        ax.set_ylabel(f"PC2 ({variance[1]:.1%})", fontsize=26)
        ax.tick_params(axis='both', labelsize=22)
        ax.grid(True, linestyle='--', alpha=0.7)

    axes[1].set_xlabel(f"PC1 ({variance[0]:.1%})", fontsize=26)

    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=min(len(legend_handles), 3), fontsize=24, title="Linaje", title_fontsize=26, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_dir / "Figure_4_PCA_2D_Comparison_Vertical.png", dpi=600, bbox_inches='tight')
    plt.close()

def plot_figure_4_pca_3d(pca_results: dict, output_dir: Path):
    """Genera una figura comparativa de los PCA 3D como figura suplementaria."""
    logger.info("Generando Figura Suplementaria: Comparación de PCA 3D...")
    
    fig = plt.figure(figsize=(26, 13))
    
    # Preparar leyenda unificada
    all_lineages = pd.concat([
        pca_results.get('fantasia_pca', pd.DataFrame()), 
        pca_results.get('homology_pca', pd.DataFrame())
    ])['lineage'].unique()
    palette = sns.color_palette("husl", n_colors=len(all_lineages))
    color_map = dict(zip(all_lineages, palette))
    legend_handles = [mpatches.Patch(color=color, label=name) for name, color in color_map.items()]

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    axes = [ax1, ax2]

    for ax, method_name_key in zip(axes, ['fantasia', 'homology']):
        method_name_display = method_name_key.capitalize()
        pc_df = pca_results.get(f'{method_name_key}_pca')
        variance = pca_results.get(f'{method_name_key}_variance', [0,0,0])
        
        if pc_df is None or pc_df.empty:
            ax.text2D(0.5, 0.5, f"No hay datos de PCA para\n{method_name_display}", ha='center', va='center', fontsize=20)
            continue

        for lineage_name, group in pc_df.groupby('lineage'):
            color = color_map[lineage_name]
            centroid = group[['PC1', 'PC2', 'PC3']].mean()
            
            ax.scatter(group['PC1'], group['PC2'], group['PC3'], s=120, alpha=0.7, c=[color])
            ax.scatter(centroid['PC1'], centroid['PC2'], centroid['PC3'], s=400, marker='o', c=[color], edgecolor='black', linewidth=2, depthshade=False)
            
            for _, row in group.iterrows():
                ax.plot([row['PC1'], centroid['PC1']], [row['PC2'], centroid['PC2']], [row['PC3'], centroid['PC3']], c=color, linestyle='--', linewidth=1.2, alpha=0.5)
        
        ax.set_title(f"PCA 3D - {method_name_display}", fontsize=24, weight='bold')
        ax.set_xlabel(f"PC1 ({variance[0]:.1%})", fontsize=20, labelpad=15)
        ax.set_ylabel(f"PC2 ({variance[1]:.1%})", fontsize=20, labelpad=15)
        ax.set_zlabel(f"PC3 ({variance[2]:.1%})", fontsize=20, labelpad=15)
        ax.tick_params(axis='both', labelsize=16)

    fig.legend(handles=legend_handles, loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize=20, title="Linaje", title_fontsize=22)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_dir / "Supp_Figure_PCA_3D_Comparison.png", dpi=600)
    plt.close()
    
def _get_slim_counts(go_sets: pd.Series, slim_ids: set, godag: GODag) -> Dict[str, int]:
    all_gos = set.union(*go_sets.dropna())
    counts = {slim_id: 0 for slim_id in slim_ids}
    for go_id in all_gos:
        if go_id in godag:
            for parent_id in godag[go_id].get_all_parents().union({go_id}):
                if parent_id in slim_ids:
                    counts[parent_id] += 1
    return counts

def _perform_goea(study: set, pop: set, assoc: dict, godag: GODag) -> pd.DataFrame:
    if not study: return pd.DataFrame()
    goeaobj = GOEnrichmentStudy(pop, assoc, godag, alpha=0.05, methods=['fdr_bh'])
    goea_results = goeaobj.run_study(study)
    results_list = [r for r in goea_results if r.p_fdr_bh < 0.05]
    return pd.DataFrame([r.__dict__ for r in results_list])

def _plot_enrichment_panel(df_res: pd.DataFrame, ax, base_color: str):
    """
    Función interna para plotear un panel de enriquecimiento con color sólido.
    MODIFICADA para aumentar el tamaño de todas las fuentes.
    """
    if df_res.empty:
        ax.text(0.5, 0.5, "No se encontró enriquecimiento significativo", ha='center', va='center', fontsize=24)
        ax.set_xticks([]); ax.set_yticks([])
        return
    
    df_plot = df_res[df_res.NS == 'BP'].sort_values('p_fdr_bh').head(15)
    if df_plot.empty:
        ax.text(0.5, 0.5, "No se encontraron términos BP significativos", ha='center', va='center', fontsize=24)
        return
        
    df_plot['log_p'] = -np.log10(df_plot['p_fdr_bh'].replace(0, 1e-300))
    df_plot['name_wrapped'] = df_plot['name'].apply(lambda x: textwrap.fill(x, 40))
    
    sns.barplot(data=df_plot, x='log_p', y='name_wrapped', color=base_color, ax=ax)
    
    # --- MEJORA: Fuentes mucho más grandes para nitidez ---
    ax.set_xlabel("-log10(p-valor ajustado)", fontsize=26)
    ax.set_ylabel("")
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)

def plot_figure_5_semantic_coherence(df_sem_sim: pd.DataFrame, output_dir: Path):
    """Genera un gráfico de barras para la coherencia funcional de los grupos."""
    logger.info("Generando Figura 5: Coherencia Funcional (Similitud Semántica)...")
    
    plt.figure(figsize=(14, 10))
    ax = sns.barplot(data=df_sem_sim, x='Group', y='Mean_Semantic_Similarity', palette=['#ff8c00', '#1e90ff', '#7f7f7f'])
    
    # --- MEJORA: Textos aún más grandes ---
    ax.set_xlabel("Grupo de Proteínas", fontsize=22)
    ax.set_ylabel("Similitud Semántica Media (Resnik BMA)", fontsize=22)
    ax.set_xticklabels(['Exclusivas FANTASIA', 'Exclusivas Homología', 'Comunes'], rotation=0, fontsize=20)
    ax.tick_params(axis='y', labelsize=18)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 12), 
                    textcoords='offset points',
                    fontsize=20, weight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "Figure_5_Semantic_Coherence.png", dpi=600)
    plt.close()

def plot_figure_6_revised_anomaly_profile_by_lineage(df_profile: pd.DataFrame, df_lineages: pd.DataFrame, output_dir: Path):
    """
    Genera un panel de radar vertical (2x1) mostrando el perfil de anomalías para cada linaje,
    optimizado para formato A4 con textos grandes y en español.
    """
    logger.info("Generando Figura 6 (Revisada): Perfil de Anomalías por Linaje (Vertical)...")
    if df_profile.empty or df_lineages.empty:
        logger.warning("Faltan datos para Figura 6.")
        return
    df_merged = pd.merge(df_profile, df_lineages, on='species')
    df_agg = df_merged.groupby(['method', 'lineage', 'category'])['anomaly_count'].sum().reset_index()
    if df_agg.empty:
        logger.warning("No hay datos de anomalías agregados para Figura 6.")
        return

    translation_map_categories = {
        'Contamination_Bacteria': 'Bacterias',
        'Hallucination_Fungi': 'Hongos',
        'Hallucination_Metazoa': 'Metazoos',
        'Contamination_Viruses': 'Virus',
        'Contamination_Archaea': 'Arqueas',
        'Other_Anomaly': 'Otras Anomalías'
    }
    translation_map_lineages = {
        'chlorophyta': 'Clorofitas',
        'embryophyta': 'Embriófitas',
        'eudicots': 'Eudicotiledóneas',
        'fabales': 'Fabales',
        'liliopsida': 'Liliopsida',
        'magnoliids': 'Magnólidas'
    }

    categories = sorted(df_agg['category'].unique())
    translated_categories = [translation_map_categories.get(c, c) for c in categories]
    
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 1, figsize=(16, 28), subplot_kw=dict(polar=True))
    
    lineages = sorted(df_agg['lineage'].unique())
    palette = plt.cm.get_cmap('tab10', len(lineages))
    panel_labels = 'AB'
    
    # --- MODIFICADO: Traducir 'Homology' directamente aquí ---
    method_names_display = ['FANTASIA', 'Homología']
    method_names_internal = ['FANTASIA', 'Homology']

    for i, (ax, method_name_internal) in enumerate(zip(axes, method_names_internal)):
        # --- MODIFICADO: Título con etiqueta de panel y traducción ---
        title_text = f"{panel_labels[i]}) {method_names_display[i]}"
        ax.set_title(title_text, size=30, weight='bold', y=1.1)
        
        ax.set_thetagrids(np.degrees(angles[:-1]), translated_categories, size=24)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_rscale('log')
        ax.set_rlabel_position(30)

        for j, lineage_name in enumerate(lineages):
            lineage_data = df_agg[(df_agg['method'] == method_name_internal) & (df_agg['lineage'] == lineage_name)]
            if lineage_data.empty: continue
            
            lineage_data = lineage_data.set_index('category').reindex(categories, fill_value=0)
            stats = lineage_data['anomaly_count'].values.tolist()
            stats += stats[:1]
            
            ax.plot(angles, stats, color=palette(j), linewidth=3.5, linestyle='solid', label=lineage_name)
            ax.fill(angles, stats, color=palette(j), alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    translated_labels = [translation_map_lineages.get(lbl, lbl) for lbl in labels]
    
    fig.legend(handles, translated_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
               ncol=min(len(labels), 3), fontsize=26, title="Linaje", title_fontsize=28, frameon=False)
               
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_dir / "Figure_6_Revised_Anomaly_Radar_by_Lineage_Vertical.png", dpi=600)
    plt.close()

def plot_figure_7_outlier_anomaly_radar(df_profile: pd.DataFrame, output_dir: Path):
    logger.info("Generando Figura 7: Diagnóstico de Anomalías en Especies Outlier...")
    if df_profile.empty: return
    outlier_species = ['Anisodus_tanguticus01']
    df_plot_data = df_profile[df_profile['species'].isin(outlier_species)]
    if df_plot_data.empty: return
    df_pivot = df_plot_data.pivot_table(index=['species', 'category'], columns='method', values='anomaly_count', fill_value=0).reset_index()
    df_pivot.to_csv(output_dir / "data_figure_7_outlier_anomaly_diagnostics.csv", index=False)
    
    all_categories = sorted(df_pivot['category'].unique()); num_vars = len(all_categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist(); angles += angles[:1]
    fig, axes = plt.subplots(figsize=(16, 16), nrows=2, ncols=2, subplot_kw=dict(polar=True)); axes = axes.flatten()
    for i, species_name in enumerate(outlier_species):
        if i >= len(axes): break
        ax = axes[i]; species_data = df_pivot[df_pivot['species'] == species_name].set_index('category').reindex(all_categories, fill_value=0)
        # ... (código de ploteo interno) ...
        # OMITIDO: Título de subpanel
        ax.set_thetagrids(np.degrees(angles[:-1]), all_categories, size=10)
        ax.set_rscale('log'); ax.set_rlabel_position(0)
    # OMITIDO: Título
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.97, 0.97), fontsize='x-large')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / "Figure_7_Outlier_Anomaly_Diagnostics_Radar.png", dpi=600)
    plt.close()
    
def plot_figure_8_paired_specificity(df_paired: pd.DataFrame, output_dir: Path):
    """
    Genera un gráfico de barras comparando la especificidad máxima por proteína (comparación pareada)
    para cada linaje, con corchetes de significancia y leyendas horizontales en la parte inferior.
    """
    logger.info("Generando Figura 8 (Final): Comparación Pareada de Especificidad por Linaje...")

    if df_paired.empty:
        logger.warning("No hay datos de comparación pareada para generar la Figura 8.")
        return

    df_plot = df_paired[df_paired['ic_source'] == 'Project_ComputeIC_Script']
    if df_plot.empty:
        logger.warning("No se encontraron datos de IC de 'Project_ComputeIC_Script' para la Figura 8.")
        return

    df_melted = df_plot.melt(
        id_vars=['lineage', 'wilcoxon_p_value'],
        value_vars=['fantasia_mean_max_ic', 'homology_mean_max_ic'],
        var_name='Method',
        value_name='Mean_Max_IC'
    )
    df_melted['Method'] = df_melted['Method'].replace({
        'fantasia_mean_max_ic': 'FANTASIA',
        'homology_mean_max_ic': 'Homology'
    })

    # --- AÑADIDO: Mapa de traducción para los linajes del eje X ---
    translation_map = {
        'chlorophyta': 'Clorofitas',
        'embryophyta': 'Embriófitas',
        'eudicots': 'Eudicotiledóneas',
        'fabales': 'Fabales',
        'liliopsida': 'Liliopsida',
        'magnoliids': 'Magnólidas'
    }
    # Aplicar la traducción. Si un linaje no está en el mapa, se mantiene el original.
    df_melted['lineage'] = df_melted['lineage'].map(translation_map).fillna(df_melted['lineage'])
    # Asegurarse de que el dataframe para los p-valores también esté traducido
    df_plot['lineage'] = df_plot['lineage'].map(translation_map).fillna(df_plot['lineage'])


    fig, ax = plt.subplots(figsize=(22, 15))
    
    # Usar el orden del dataframe traducido para el plot
    lineage_order = df_plot['lineage'].unique()
    sns.barplot(data=df_melted, x='lineage', y='Mean_Max_IC', hue='Method', palette=PALETTE, ax=ax, legend=False, order=lineage_order)
    
    # --- MEJORA: Textos y elementos visuales aún más grandes y nítidos ---
    ax.set_xlabel("Linaje", fontsize=28)
    ax.set_ylabel("Especificidad Media de la Mejor Anotación (IC)", fontsize=28)
    plt.xticks(rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_ylim(bottom=6)

    y_max_total = df_melted['Mean_Max_IC'].max()
    ax.set_ylim(top=y_max_total * 1.30)

    # Usar los xticklabels actuales que ya están en el orden correcto
    for i, lineage_label in enumerate(ax.get_xticklabels()):
        lineage_name = lineage_label.get_text()
        
        # Obtener el p-valor del dataframe original traducido
        p_value_series = df_plot[df_plot['lineage'] == lineage_name]['wilcoxon_p_value']
        if p_value_series.empty:
            continue
        p_value = p_value_series.iloc[0]
        
        stars = _get_significance_stars(p_value)
        
        if stars != 'ns':
            y = df_melted[df_melted['lineage'] == lineage_name]['Mean_Max_IC'].max()
            h = y * 0.03
            x1, x2 = i - 0.2, i + 0.2
            
            # Corchetes más gruesos y estrellas más grandes
            ax.plot([x1, x1, x2, x2], [y + h, y + 2*h, y + 2*h, y + h], lw=3, c='black')
            ax.text((x1 + x2) * .5, y + 2.2*h, stars, ha='center', va='bottom', color='black', fontsize=30)

    # Leyendas más grandes
    legend_handles = [
        mpatches.Patch(color=FANTASIA_COLOR, label='FANTASIA'),
        mpatches.Patch(color=HOMOLOGY_COLOR, label='Homología')
    ]
    fig.legend(handles=legend_handles, loc='lower right', bbox_to_anchor=(0.95, 0.01), ncol=2, title="Método", frameon=False, fontsize=22, title_fontsize=24)

    significance_text = 'Significancia:  ns: p > 0.05      * p ≤ 0.05      ** p ≤ 0.01      *** p ≤ 0.001      **** p ≤ 0.0001'
    fig.text(0.05, 0.02, significance_text, ha='left', va='center', fontsize=20)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_dir / "Figure_8_Paired_Specificity_Comparison_ComputeIC.png", dpi=600)
    plt.close()


def plot_figure_9_global_ic_comparison(df_global_ic: pd.DataFrame, output_dir: Path):
    """
    Crea un barplot comparando la especificidad media/mediana global de IC.
    """
    logger.info("Generando Figura 9: Comparación Global de Especificidad de IC...")
    if df_global_ic.empty: return

    df_melted = df_global_ic.melt(
        id_vars=['ic_source', 'p_value_adjusted'],
        value_vars=['fantasia_mean_ic', 'homology_mean_ic', 'fantasia_median_ic', 'homology_median_ic'],
        var_name='Metric',
        value_name='IC_Value'
    )
    split_df = df_melted['Metric'].str.split('_', expand=True)
    df_melted[['Method', 'Stat']] = split_df[[0, 1]] 
    df_melted['Method'] = df_melted['Method'].str.capitalize()

    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharey=False)
    
    # --- MEJORA: Textos aún más grandes ---
    sns.barplot(data=df_melted[df_melted['Stat'] == 'mean'], x='ic_source', y='IC_Value', hue='Method', ax=axes[0], palette=PALETTE)
    axes[0].set_title('Comparación de IC Medio Global', fontsize=22, weight='bold')
    axes[0].set_xlabel('Método de Cálculo de IC', fontsize=20)
    axes[0].set_ylabel('IC Medio', fontsize=20)
    axes[0].tick_params(axis='x', rotation=20, labelsize=18, ha='right')
    axes[0].tick_params(axis='y', labelsize=18)
    axes[0].legend(fontsize=18, title_fontsize=20)

    sns.barplot(data=df_melted[df_melted['Stat'] == 'median'], x='ic_source', y='IC_Value', hue='Method', ax=axes[1], palette=PALETTE)
    axes[1].set_title('Comparación de IC Mediano Global', fontsize=22, weight='bold')
    axes[1].set_xlabel('Método de Cálculo de IC', fontsize=20)
    axes[1].set_ylabel('IC Mediano', fontsize=20)
    axes[1].tick_params(axis='x', rotation=20, labelsize=18, ha='right')
    axes[1].tick_params(axis='y', labelsize=18)
    axes[1].legend(fontsize=18, title_fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "Figure_9_Global_IC_Comparison.png", dpi=600)
    plt.close()

def plot_figure_10_common_enrichment_barchart(df_common_enrich: pd.DataFrame, output_dir: Path):
    logger.info("Generando Figura 10: Gráfico de Barras de Enriquecimiento en Proteínas Comunes...")
    if df_common_enrich.empty: return

    df_plot = df_common_enrich.copy()
    df_plot = df_plot[df_plot['NS'] == 'BP']
    
    df_plot['logp_fantasia'] = -np.log10(df_plot['p_fdr_bh_fantasia'].replace(0, 1e-100))
    df_plot['logp_homology'] = -np.log10(df_plot['p_fdr_bh_homology'].replace(0, 1e-100))

    df_plot['min_logp'] = df_plot[['logp_fantasia', 'logp_homology']].max(axis=1)
    df_plot = df_plot.sort_values('min_logp', ascending=False).head(15)
    df_plot['name_wrapped'] = df_plot['name'].apply(lambda x: textwrap.fill(x, 60))

    df_melted = df_plot.melt(
        id_vars=['name_wrapped'], value_vars=['logp_fantasia', 'logp_homology'],
        var_name='Method', value_name='-log10(p-adj)'
    )
    df_melted['Method'] = df_melted['Method'].replace({
        'logp_fantasia': 'FANTASIA',
        'logp_homology': 'Homology'
    })

    plt.figure(figsize=(18, 14))
    ax = sns.barplot(data=df_melted, y='name_wrapped', x='-log10(p-adj)', hue='Method', palette=PALETTE)

    # --- MEJORA: Textos aún más grandes ---
    ax.set_xlabel("-log10(p-valor ajustado)", fontsize=22)
    ax.set_ylabel("Término Gene Ontology (BP)", fontsize=22)
    ax.legend(title="Método", fontsize=20, title_fontsize=22)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / "Figure_10_Common_Enrichment_BarChart.png", dpi=600)
    plt.close()

def plot_figure_11_ic_distributions(df_ic: pd.DataFrame, output_dir: Path):
    """
    Genera una cuadrícula de gráficos de densidad 2x3 para comparar las distribuciones de IC,
    con un filtrado robusto para evitar paneles vacíos.
    """
    logger.info("Generando Figura 11 (Revisada y Corregida): Comparación Dual de Distribuciones de IC...")
    if df_ic.empty:
        logger.warning("No hay datos de distribución de IC para generar la Figura 11.")
        return

    # --- CORRECCIÓN: Filtrado robusto para evitar errores por nombres exactos ---
    # En lugar de buscar una coincidencia exacta, buscamos subcadenas clave.
    # Esto hace que el código sea resistente a variaciones como 'Project_ComputeIC' vs 'Project ComputeIC'.
    
    # 1. Crear una columna estandarizada para un filtrado seguro
    conditions = [
        df_ic['ic_source'].str.contains("ComputeIC", case=False, na=False),
        df_ic['ic_source'].str.contains("goatools", case=False, na=False)
    ]
    choices = ['ComputeIC', 'Goatools']
    df_ic['ic_source_standardized'] = np.select(conditions, choices, default='Other')

    # 2. Filtrar el DataFrame usando la nueva columna estandarizada
    df_plot = df_ic[df_ic['ic_source_standardized'].isin(choices)].copy()
    
    # 3. Traducir los namespaces para los títulos de los paneles
    df_plot['namespace'] = df_plot['namespace'].replace({
        'biological_process': 'Proceso Biológico', 
        'molecular_function': 'Función Molecular', 
        'cellular_component': 'Componente Celular'
    })
    
    if df_plot.empty:
        logger.error("No se encontraron datos para las fuentes de IC requeridas (ComputeIC, goatools) después del filtrado. Omitiendo Figura 11.")
        return

    # Definir el orden y los nombres para los bucles
    ic_sources_to_plot = ['ComputeIC', 'Goatools']
    namespaces_to_plot = ['Proceso Biológico', 'Función Molecular', 'Componente Celular']

    fig, axes = plt.subplots(2, 3, figsize=(26, 16), sharex=True, sharey='col')
    
    panel_labels = 'ABCDEF'
    panel_idx = 0
    
    for i, ic_source_std in enumerate(ic_sources_to_plot):
        for j, namespace in enumerate(namespaces_to_plot):
            ax = axes[i, j]
            
            # 4. Filtrar para cada panel usando las columnas estandarizadas y traducidas
            data_panel = df_plot[
                (df_plot['ic_source_standardized'] == ic_source_std) & 
                (df_plot['namespace'] == namespace)
            ]
            
            if not data_panel.empty:
                sns.kdeplot(data=data_panel, x='ic_value', hue='method', palette=PALETTE, ax=ax, cut=0, legend=False, linewidth=4)

            if i == 0: ax.set_title(namespace, fontsize=28, weight='bold')
            if i == len(ic_sources_to_plot) - 1: ax.set_xlabel("Contenido de Información (IC)", fontsize=26)
            else: ax.set_xlabel("")
            
            ax.tick_params(axis='both', labelsize=22)
            
            if j == 0:
                source_label = "Compute_IC" if ic_source_std == "ComputeIC" else "Goatools"
                ax.set_ylabel(f"Densidad\n(IC: {source_label})", fontsize=26)
            else: ax.set_ylabel("")
            
            ax.text(0.05, 0.95, panel_labels[panel_idx], transform=ax.transAxes,
                    fontsize=30, fontweight='bold', va='top', ha='left')
            panel_idx += 1

    legend_handles = [
        Line2D([0], [0], color=FANTASIA_COLOR, lw=5, label='FANTASIA'),
        Line2D([0], [0], color=HOMOLOGY_COLOR, lw=5, label='Homología')
    ]
    
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=26,
        frameon=False
    )
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(output_dir / "Figure_11_IC_Distributions_Dual_Comparison.png", dpi=600, bbox_inches='tight')
    plt.close()


def plot_figure_12_semantic_similarity(df_sim: pd.DataFrame, output_dir: Path, method_name: str = "Resnik"):
    """
    Genera gráficos de violín para la similitud semántica de FANTASIA vs. Homología.
    """
    logger.info(f"Generando Figura 12 (Definitiva): Similitud Semántica (Método: {method_name})...")
    if df_sim.empty: return
        
    df_plot = df_sim.copy()
    df_plot['namespace'] = df_plot['namespace'].replace({
        'biological_process': 'Proceso Biológico', 
        'molecular_function': 'Función Molecular', 
        'cellular_component': 'Componente Celular'
    })
    
    plt.figure(figsize=(18, 12))
    
    ax = sns.violinplot(
        data=df_plot,
        x='namespace',
        y='similarity_score',
        palette=[FANTASIA_COLOR, FANTASIA_COLOR, FANTASIA_COLOR],
        inner='box',
        cut=0,
        linewidth=2.5
    )

    # --- MEJORA: Textos aún más grandes ---
    ax.set_xlabel("Dominio Gene Ontology", fontsize=24)
    ax.set_ylabel(f"Similitud Semántica ({method_name} BMA)", fontsize=24)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=20)
    
    if method_name == "Lin": ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / f"Figure_12_Semantic_Similarity_{method_name}.png", dpi=600)
    plt.close()

def plot_figure_13_ic_concordance(df_raw_ic: pd.DataFrame, df_concordance: pd.DataFrame, output_dir: Path):
    """
    Genera una figura compuesta que compara los métodos de cálculo de IC.
    """
    logger.info("Generando Figura 13 (Mejorada): Comparación y Concordancia de Métodos de IC...")
    if df_raw_ic.empty or df_concordance.empty: return

    # --- Panel A: Boxplots ---
    df_plot_box = df_raw_ic.melt(id_vars=['GO_term'], var_name='IC Method', value_name='IC Value')
    df_plot_box['IC Method'] = df_plot_box['IC Method'].str.replace('_', ' ').str.replace('Project ', '')

    fig, ax_box = plt.subplots(figsize=(14, 10))
    
    sns.boxplot(data=df_plot_box, x='IC Method', y='IC Value', hue='IC Method', ax=ax_box, palette='viridis', legend=False)
    
    ax_box.set_xlabel("Método de Cálculo de IC", fontsize=22)
    ax_box.set_ylabel("Valor de Contenido de Información (IC)", fontsize=22)
    
    ax_box.tick_params(axis='x', rotation=25, labelsize=18)
    plt.setp(ax_box.get_xticklabels(), ha='right', rotation_mode='anchor')
    
    ax_box.tick_params(axis='y', labelsize=18)
    ax_box.set_ylim(bottom=5)
    plt.tight_layout()
    plt.savefig(output_dir / "Figure_13_IC_Method_Distribution.png", dpi=600)
    plt.close(fig)

    # --- Panel B: Clustermap de Concordancia ---
    df_concordance.columns = df_concordance.columns.str.replace('_', ' ').str.replace('Project ', '')
    df_concordance.index = df_concordance.index.str.replace('_', ' ').str.replace('Project ', '')

    clustergrid = sns.clustermap(
        df_concordance,
        cmap='coolwarm',
        annot=True,
        annot_kws={"size": 18},
        fmt=".2f",
        linewidths=.75,
        vmin=0,
        vmax=1,
        figsize=(12, 11),
        # Etiqueta de cbar simplificada
        cbar_kws={'label': 'Concordancia'}
    )
    plt.setp(clustergrid.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=18)
    plt.setp(clustergrid.ax_heatmap.get_yticklabels(), rotation=0, fontsize=18)
    
    if hasattr(clustergrid, 'cbar_ax'):
        # Letra de la etiqueta de cbar mucho más grande
        clustergrid.cbar_ax.set_ylabel('Concordancia', fontsize=28)
        clustergrid.cbar_ax.tick_params(labelsize=18)
    
    clustergrid.savefig(output_dir / "Figure_13_IC_Method_Concordance.png", dpi=600, bbox_inches='tight')
    plt.close(clustergrid.fig)
    logger.info("Figura 13 (Distribución y Concordancia) guardada en dos archivos separados.")



def plot_figure_14_common_enrichment_dotplot(df_fan_top1000: pd.DataFrame, df_hom_top1000: pd.DataFrame, output_dir: Path):
    """
    Crea un 'dumbbell plot' para visualizar los términos GO comunes.
    """
    logger.info("Generando Figura 14: Comparación de Enriquecimiento Común (Dumbbell Plot)...")
    if df_fan_top1000.empty or df_hom_top1000.empty: return

    df_fan = df_fan_top1000[['GO', 'name', 'p_value']].rename(columns={'p_value': 'p_value_fan'})
    df_hom = df_hom_top1000[['GO', 'p_value']].rename(columns={'p_value': 'p_value_hom'})
    common_gos = set(df_fan['GO']).intersection(set(df_hom['GO']))
    if not common_gos: return
        
    df_merged = pd.merge(df_fan[df_fan['GO'].isin(common_gos)], df_hom, on='GO')
    df_merged['logp_fan'] = -np.log10(df_merged['p_value_fan'].replace(0, 1e-300))
    df_merged['logp_hom'] = -np.log10(df_merged['p_value_hom'].replace(0, 1e-300))
    df_merged['logp_diff'] = abs(df_merged['logp_fan'] - df_merged['logp_hom'])

    df_plot = df_merged.sort_values('logp_diff', ascending=False).head(25)
    df_plot = df_plot.sort_values('logp_fan')
    df_plot['name_wrapped'] = df_plot['name'].apply(lambda x: textwrap.fill(x, 50))

    fig, ax = plt.subplots(figsize=(16, 20))

    # --- MEJORA: Elementos más grandes ---
    ax.hlines(y=df_plot['name_wrapped'], xmin=df_plot['logp_fan'], xmax=df_plot['logp_hom'],
              color='grey', alpha=0.7, linewidth=3)
    ax.scatter(df_plot['logp_fan'], df_plot['name_wrapped'], color=FANTASIA_COLOR, s=250, label='FANTASIA', zorder=3)
    ax.scatter(df_plot['logp_hom'], df_plot['name_wrapped'], color=HOMOLOGY_COLOR, s=250, label='Homología', zorder=3)

    # --- MEJORA: Textos aún más grandes ---
    ax.set_xlabel("-log10(p-valor)", fontsize=22)
    ax.set_ylabel("Término Gene Ontology (BP)", fontsize=22)
    ax.legend(title="Método", fontsize=20, title_fontsize=22)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "Figure_14_Common_Enrichment_Dumbbell.png", dpi=600)
    plt.close()

 
    
# --- FUNCIÓN DE EXPORTACIÓN A EXCEL ---    
def export_summary_tables_to_excel(data_dict: dict, output_file: Path):
    """
    Exporta todas las tablas de resumen generadas a un único y bien organizado archivo Excel.
    """
    logger.info(f"Exportando todas las tablas de análisis al archivo Excel: {output_file}...")
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Diccionario para mapear claves de datos a nombres de hojas descriptivos
            sheet_map = {
                'master_summary': 'Resumen_Maestro_Global',
                'go_vocabulary_comparison': 'Comparacion_Vocabulario_GO',
                'fantasia_exclusive_enrichment': 'Enriquecimiento_Exclusivo_FAN',
                'homology_exclusive_enrichment': 'Enriquecimiento_Exclusivo_HOM',
                'common_proteins_enrichment': 'Enriquecimiento_Proteinas',
                'specificity_exclusive': 'Especificidad_Grupos_Exclusivos',
                'specificity_common_paired': 'Especificidad_Proteinas',
                'common_proteins_depth': 'Profundidad_Proteinas_Comunes',
                'semantic_coherence': 'Coherencia_Semantica_Interna',
                # 'semantic_similarity': 'Similitud_Semantica_vs_Homologia',
                'ic_method_concordance': 'Concordancia_Metodos_IC',
                # 'ic_distributions': 'Distribuciones_Globales_IC',
                # 'functional_bias_heatmap': 'Heatmap_Sesgo_Funcional',
                'per_species_contamination_profile': 'Contaminacion_Por_Especie',
                'outlier_validation': 'Validacion_Cuantitativa_Outliers',
                'contamination_lineage': 'full_lineage_profile_comparison.csv'
            }

            # Iterar sobre el mapa para exportar cada tabla si existe
            for key, sheet_name in sheet_map.items():
                if key in data_dict and not data_dict[key].empty:
                    logger.info(f"  [Ok] -> Escribiendo hoja: {sheet_name}")
                    # El heatmap necesita que su índice se guarde, los demás no.
                    index_to_save = True if key == 'functional_bias_heatmap' else False
                    data_dict[key].to_excel(writer, sheet_name=sheet_name, index=index_to_save)
                else:
                    logger.warning(f"  -> Omitiendo hoja '{sheet_name}' (datos no encontrados para la clave '{key}').")
                 
        logger.info(f"Todas las tablas se han exportado correctamente a {output_file}")
    except Exception as e:
        logger.error(f"Falló la exportación de tablas a Excel: {e}", exc_info=True)
    
# #############################################################################
# --- ORQUESTADOR PRINCIPAL (ACTUALIZADO) ---
# #############################################################################

def run_visualization_and_analysis(results_dir: Path, output_dir: Path, only_figure: str = None):
    """Orquestador principal del pipeline de visualización."""
    logger.info("=== Iniciando Pipeline de Visualización (v6.2 ===")

    if only_figure: logger.info(f"--- EJECUTANDO EN MODO DE FIGURA ÚNICA: '{only_figure}' ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Cargando todos los resultados de análisis requeridos...")
    # El diccionario de archivos ahora está perfectamente alineado con las nuevas funciones de análisis
    data_files = {
        'protein_summaries': "global_protein_summaries.pkl",
        'master_summary': "master_summary_statistics.csv",
        'semantic_coherence': "semantic_coherence_results.csv",
        'go_vocabulary_comparison': "go_vocabulary_comparison.csv",
        'fantasia_exclusive_enrichment': "fantasia_exclusive_enrichment.csv",
        'homology_exclusive_enrichment': "homology_exclusive_enrichment.csv",
        'common_proteins_enrichment': "common_proteins_enrichment_comparison.csv",
        'per_species_contamination_profile': "per_species_contamination_profile.csv",
        "functional_bias_heatmap": "functional_bias_heatmap_data.csv",
        "pca_results": "pca_results.pkl",
        "specificity_exclusive": "specificity_summary_exclusive_groups.csv",
        "specificity_common_paired": "specificity_summary_common_proteins_paired.csv",
        "common_proteins_depth": "common_proteins_depth_comparison.csv",
        "ic_distributions": "ic_distributions_long_format.csv",
        "semantic_similarity": "semantic_similarity_scores_resnik.csv",
        "ic_values_by_method": "ic_values_by_method.csv",
        "ic_method_concordance": "ic_method_concordance.csv",
        'outlier_validation': "outlier_quantitative_validation_comprehensive.csv",
        "brassica_fan_enrich": "brassica_napus_fantasia_enrichment.csv",
        "brassica_hom_enrich": "brassica_napus_homology_enrichment.csv",
        'contamination_lineage': 'full_lineage_profile_comparison.csv'
    }
    
    data = {}
    for key, filename in data_files.items():
        file_path = results_dir / filename
        try:
            if filename.endswith('.pkl'):
                with open(file_path, 'rb') as f: data[key] = pickle.load(f)
            else:
                # Cargar el heatmap con su índice, los demás sin él.
                index_col = 0 if key == 'functional_bias_heatmap' else None
                data[key] = pd.read_csv(file_path, index_col=index_col)
            logger.info(f"  [OK] Archivo cargado: {filename}")
        except FileNotFoundError:
            logger.warning(f"  [AVISO] Archivo de datos '{filename}' no encontrado."); data[key] = pd.DataFrame() if not filename.endswith('.pkl') else {}
            
    if not data.get('protein_summaries', pd.DataFrame()).empty:
        df_summary = data['protein_summaries']
        df_summary['fantasia_gos'] = df_summary['fantasia_gos'].apply(lambda x: x if isinstance(x, set) else set())
        df_summary['homology_gos'] = df_summary['homology_gos'].apply(lambda x: x if isinstance(x, set) else set())
        data['protein_summaries'] = df_summary
    logger.info("Todos los datos se cargaron y procesaron preliminarmente de forma correcta.")

    # --- Lógica de ejecución selectiva de figuras (CORREGIDA) ---
    if not only_figure or only_figure == 'fig1':
        if not data['protein_summaries'].empty and not data['master_summary'].empty:
            # Extraer los sets de proteínas aquí para pasarlos a la función de Venn
            proteins_fan = set(data['protein_summaries'][data['protein_summaries']['fantasia_gos'].str.len() > 0]['protein'])
            proteins_hom = set(data['protein_summaries'][data['protein_summaries']['homology_gos'].str.len() > 0]['protein'])
            
            plot_figure_1a_coverage_venn(data['master_summary'], proteins_fan, proteins_hom, output_dir)
            plot_figure_1b_quantity_by_lineage(data['protein_summaries'], output_dir)
            
    if not only_figure or only_figure == 'fig2':
        if 'specificity_exclusive' in data and not data['specificity_exclusive'].empty: plot_figure_2_exclusive_specificity(data['specificity_exclusive'], output_dir)
    
    # --- CORRECCIÓN DE ERROR: Llamamos a las dos nuevas funciones ---
    if not only_figure or only_figure == 'fig3':
        plot_figure_3_enrichment_panels(data['fantasia_exclusive_enrichment'], data['homology_exclusive_enrichment'], output_dir)
        plot_figure_3b_heatmap(data['functional_bias_heatmap'], output_dir)
    
    if not only_figure or only_figure == 'fig4':
        if data.get('pca_results'):
            plot_figure_4_pca_2d(data['pca_results'], output_dir)
            plot_figure_4_pca_3d(data['pca_results'], output_dir)
    
    if not only_figure or only_figure == 'fig5':
        if not data['semantic_coherence'].empty:
            plot_figure_5_semantic_coherence(data['semantic_coherence'], output_dir)

    if not only_figure or only_figure == 'fig6':
        if not data['protein_summaries'].empty and not data['per_species_contamination_profile'].empty:
            logger.info("Generando mapa de linajes desde 'protein_summaries' para la Figura 6.")
            df_lineages = data['protein_summaries'][['species', 'lineage']].drop_duplicates().dropna()
            plot_figure_6_revised_anomaly_profile_by_lineage(data['per_species_contamination_profile'], df_lineages, output_dir)
        else:
            logger.warning("No se generará la Figura 6 por falta de 'protein_summaries' o 'per_species_contamination_profile'.")
        
    if not only_figure or only_figure == 'fig7':
        # Esta figura depende del mismo perfil de contaminación, pero filtra por outliers específicos.
        if not data['per_species_contamination_profile'].empty:
            plot_figure_7_outlier_anomaly_radar(data['per_species_contamination_profile'], output_dir)
    
    if not only_figure or only_figure == 'fig8':
        if 'specificity_common_paired' in data and not data['specificity_common_paired'].empty:
            plot_figure_8_paired_specificity(data['specificity_common_paired'], output_dir)
        
    # La Figura 9 se considera redundante con los datos más detallados mostrados en la Figura 10
    # y las tablas de Excel, por lo que se omite su generación.
    
    if not only_figure or only_figure == 'fig10':
        if 'common_proteins_enrichment' in data and not data['common_proteins_enrichment'].empty:
            plot_figure_10_common_enrichment_barchart(data['common_proteins_enrichment'], output_dir)
            
    if not only_figure or only_figure == 'fig11':
        if 'ic_distributions' in data and not data['ic_distributions'].empty: plot_figure_11_ic_distributions(data['ic_distributions'], output_dir)
    if not only_figure or only_figure == 'fig12':
        if 'semantic_similarity' in data and not data['semantic_similarity'].empty: plot_figure_12_semantic_similarity(data['semantic_similarity'], output_dir, method_name="resnik")
    if not only_figure or only_figure == 'fig13':
        if 'ic_values_by_method' in data and 'ic_method_concordance' in data:
            df_concordance = data['ic_method_concordance'].set_index(data['ic_method_concordance'].columns[0])
            plot_figure_13_ic_concordance(data['ic_values_by_method'], df_concordance, output_dir)       
        
    if not only_figure or only_figure == 'excel':
        excel_output_path = output_dir / "Tablas_Analisis_Comprensivo_Final.xlsx"
        export_summary_tables_to_excel(data, excel_output_path)
        
    logger.info(f"=== Pipeline de Visualización Completado. Resultados en: {output_dir} ===")
    