#!/usr/bin/env python3
"""
Dataset Analyzer for AMR Studies
=================================

Professional tool for analyzing antimicrobial resistance (AMR) datasets with
comprehensive statistics on data distribution, species composition, and resistance patterns
across different data partitions.

Features:
- Partition-wise data distribution analysis
- Species composition with taxonomic flexibility (GTDB/NCBI)
- Resistance pattern statistics (resistant/sensitive ratios)
- Cross-partition comparison capabilities
- Detailed reporting with output logging

Author: Evo-AMR Team
License: MIT
"""

import pandas as pd
import numpy as np
import os
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TeeOutput:
    """Class to write output to both stdout and file simultaneously."""
    
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


class DatasetAnalyzer:
    """
    Main class for analyzing AMR datasets with comprehensive statistics.
    """
    
    def __init__(self, 
                 kover_csv_path: str,
                 metadata_tsv_path: str,
                 data_directory: str,
                 use_gtdb: bool = True,
                 dataset_name: str = "AMR Dataset"):
        """
        Initialize the dataset analyzer.
        
        Args:
            kover_csv_path: Path to the kover CSV file with labels and partitions
            metadata_tsv_path: Path to the metadata TSV file with taxonomic information
            data_directory: Directory containing processed data files
            use_gtdb: Whether to use GTDB taxonomy (True) or NCBI taxonomy (False)
            dataset_name: Name of the dataset for display purposes
        """
        self.kover_csv_path = Path(kover_csv_path)
        self.metadata_tsv_path = Path(metadata_tsv_path)
        self.data_directory = Path(data_directory)
        self.use_gtdb = use_gtdb
        self.dataset_name = dataset_name
        
        # Validate input files
        self._validate_inputs()
        
        # Load data
        self.kover_df = None
        self.metadata_df = None
        self.analysis_results = {}
        
    def _validate_inputs(self):
        """Validate that input files and directories exist."""
        if not self.kover_csv_path.exists():
            raise FileNotFoundError(f"Kover CSV file not found: {self.kover_csv_path}")
        
        if not self.metadata_tsv_path.exists():
            raise FileNotFoundError(f"Metadata TSV file not found: {self.metadata_tsv_path}")
        
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")
    
    def load_data(self) -> None:
        """Load and preprocess the dataset files."""
        logger.info("Loading dataset files...")
        
        # Load kover CSV with labels and partitions
        try:
            self.kover_df = pd.read_csv(self.kover_csv_path)
            logger.info(f"Loaded kover CSV: {len(self.kover_df)} records")
        except Exception as e:
            raise RuntimeError(f"Error loading kover CSV: {e}")
        
        # Load metadata with taxonomic information
        try:
            self.metadata_df = pd.read_csv(self.metadata_tsv_path, sep='\t', low_memory=False)
            logger.info(f"Loaded metadata TSV: {len(self.metadata_df)} records")
        except Exception as e:
            raise RuntimeError(f"Error loading metadata TSV: {e}")
        
        # Validate required columns
        self._validate_columns()
        
        # Standardize genome IDs
        self.kover_df['genome_id_str'] = self.kover_df['genome_id'].astype(str)
        self.metadata_df['genome_id_str'] = self.metadata_df['genome_id'].astype(str)
        
        logger.info("Data loading completed successfully")
    
    def _validate_columns(self) -> None:
        """Validate that required columns exist in the datasets."""
        # Required columns in kover CSV
        kover_required = ['genome_id', 'partition_label']
        missing_kover = [col for col in kover_required if col not in self.kover_df.columns]
        if missing_kover:
            raise ValueError(f"Missing required columns in kover CSV: {missing_kover}")
        
        # Detect label column (could be 'label', 'antibiotic_label', etc.)
        label_candidates = ['label', 'antibiotic_label', 'resistance_label', 'amr_label']
        label_col = None
        for candidate in label_candidates:
            if candidate in self.kover_df.columns:
                label_col = candidate
                break
        
        if label_col is None:
            raise ValueError(f"No label column found. Expected one of: {label_candidates}")
        
        self.label_column = label_col
        logger.info(f"Using label column: {self.label_column}")
        
        # Required columns in metadata
        species_col = 'gtdb_species' if self.use_gtdb else 'ncbi_species_name'
        if species_col not in self.metadata_df.columns:
            available_cols = [col for col in self.metadata_df.columns if 'species' in col.lower()]
            raise ValueError(f"Species column '{species_col}' not found. Available species columns: {available_cols}")
        
        self.species_column = species_col
        self.taxonomy_type = 'GTDB' if self.use_gtdb else 'NCBI'
        logger.info(f"Using species column: {self.species_column} ({self.taxonomy_type})")
    
    def extract_genome_ids_from_directory(self) -> set:
        """
        Extract genome IDs from files in the data directory.
        
        Returns:
            Set of genome IDs found in the directory
        """
        genome_ids = set()
        
        # Common file patterns for different data types
        patterns = [
            r'^([^_]+)_pca_stats_dim\d+\.(npz|csv)$',  # PCA files
            r'^([^_]+)_mr_win\d+_s\d+\.npz$',          # MiniRocket files
            r'^([^_]+)_pooled_embeddings\.(npz|csv)$', # Pooled embeddings
            r'^([^_]+)_embeddings\.(h5|hdf5)$',        # Raw embeddings
            r'^([^_]+)\.npz$',                         # Generic NPZ files
            r'^([^_]+)\.csv$',                         # Generic CSV files
            r'^([^_]+)\.h5$',                          # Generic H5 files
        ]
        
        files = list(self.data_directory.iterdir())
        logger.info(f"Scanning {len(files)} files in data directory")
        
        for file_path in files:
            if file_path.is_file():
                filename = file_path.name
                
                for pattern in patterns:
                    match = re.match(pattern, filename)
                    if match:
                        genome_id = match.group(1)
                        genome_ids.add(genome_id)
                        break
        
        logger.info(f"Extracted {len(genome_ids)} unique genome IDs from file names")
        return genome_ids
    
    def filter_data_by_availability(self) -> pd.DataFrame:
        """
        Filter kover data to include only genomes with available processed data.
        
        Returns:
            Filtered DataFrame with available genomes only
        """
        # Get genome IDs from data directory
        available_genome_ids = self.extract_genome_ids_from_directory()
        
        # Find intersection with kover data
        kover_genome_ids = set(self.kover_df['genome_id_str'])
        matching_genomes = available_genome_ids.intersection(kover_genome_ids)
        
        logger.info(f"Genome IDs in kover CSV: {len(kover_genome_ids)}")
        logger.info(f"Genome IDs in data directory: {len(available_genome_ids)}")
        logger.info(f"Matching genome IDs: {len(matching_genomes)}")
        
        if len(matching_genomes) == 0:
            raise ValueError("No matching genome IDs found between kover CSV and data directory")
        
        # Filter kover data
        filtered_df = self.kover_df[self.kover_df['genome_id_str'].isin(matching_genomes)]
        
        logger.info(f"Filtered dataset size: {len(filtered_df)} samples")
        return filtered_df
    
    def analyze_partitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze data distribution across partitions with resistance statistics.
        
        Returns:
            Dictionary with partition-wise analysis results
        """
        logger.info("Starting partition analysis...")
        
        # Filter data to available genomes
        filtered_kover = self.filter_data_by_availability()
        
        # Merge with metadata for species information
        merged_df = filtered_kover.merge(
            self.metadata_df[['genome_id_str', self.species_column]], 
            on='genome_id_str', 
            how='left'
        )
        
        # Get unique partitions
        partitions = sorted(merged_df['partition_label'].unique())
        logger.info(f"Found partitions: {partitions}")
        
        results = {}
        
        for partition in partitions:
            logger.info(f"Analyzing partition: {partition}")
            partition_data = merged_df[merged_df['partition_label'] == partition]
            
            # Basic statistics
            total_samples = len(partition_data)
            
            # Resistance statistics
            if self.label_column in partition_data.columns:
                resistance_counts = partition_data[self.label_column].value_counts()
                resistant_count = resistance_counts.get(1, 0)
                sensitive_count = resistance_counts.get(0, 0)
                resistant_rate = (resistant_count / total_samples * 100) if total_samples > 0 else 0
                sensitive_rate = (sensitive_count / total_samples * 100) if total_samples > 0 else 0
            else:
                resistant_count = sensitive_count = 0
                resistant_rate = sensitive_rate = 0
            
            # Species analysis
            species_counts = partition_data[self.species_column].value_counts()
            species_analysis = self._analyze_species_in_partition(partition_data, species_counts)
            
            # Store results
            results[partition] = {
                'total_samples': total_samples,
                'resistant_count': resistant_count,
                'sensitive_count': sensitive_count,
                'resistant_rate': resistant_rate,
                'sensitive_rate': sensitive_rate,
                'unique_species': len(species_counts),
                'species_counts': species_counts.to_dict() if not species_counts.empty else {},
                'species_analysis': species_analysis
            }
        
        self.analysis_results = results
        logger.info("Partition analysis completed")
        return results
    
    def _analyze_species_in_partition(self, partition_data: pd.DataFrame, 
                                    species_counts: pd.Series) -> List[Dict]:
        """
        Analyze species-specific resistance patterns within a partition.
        
        Args:
            partition_data: Data for the specific partition
            species_counts: Species count series
            
        Returns:
            List of species analysis dictionaries
        """
        species_analysis = []
        
        for species, count in species_counts.items():
            if pd.isna(species):
                species_name = '[Unknown/NA]'
                species_resistant = species_sensitive = 0
                species_resistant_rate = 0
            else:
                species_name = str(species)
                species_data = partition_data[partition_data[self.species_column] == species]
                
                if self.label_column in species_data.columns:
                    species_resistant = (species_data[self.label_column] == 1).sum()
                    species_sensitive = (species_data[self.label_column] == 0).sum()
                    species_resistant_rate = (species_resistant / count * 100) if count > 0 else 0
                else:
                    species_resistant = species_sensitive = 0
                    species_resistant_rate = 0
            
            species_analysis.append({
                'species': species_name,
                'total_count': count,
                'resistant_count': species_resistant,
                'sensitive_count': species_sensitive,
                'resistant_rate': species_resistant_rate
            })
        
        # Sort by total count (descending)
        species_analysis.sort(key=lambda x: x['total_count'], reverse=True)
        return species_analysis
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate overall summary statistics across all partitions.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analyze_partitions() first.")
        
        # Calculate totals
        total_samples = sum(r['total_samples'] for r in self.analysis_results.values())
        total_resistant = sum(r['resistant_count'] for r in self.analysis_results.values())
        total_sensitive = sum(r['sensitive_count'] for r in self.analysis_results.values())
        
        # Get all unique species across all partitions
        all_species = set()
        species_global_counts = defaultdict(int)
        
        for partition_data in self.analysis_results.values():
            for species_info in partition_data['species_analysis']:
                species_name = species_info['species']
                if species_name != '[Unknown/NA]':
                    all_species.add(species_name)
                    species_global_counts[species_name] += species_info['total_count']
        
        # Top species globally
        top_species = sorted(species_global_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        summary = {
            'total_samples': total_samples,
            'total_resistant': total_resistant,
            'total_sensitive': total_sensitive,
            'overall_resistant_rate': (total_resistant / total_samples * 100) if total_samples > 0 else 0,
            'overall_sensitive_rate': (total_sensitive / total_samples * 100) if total_samples > 0 else 0,
            'total_unique_species': len(all_species),
            'top_species': top_species,
            'partition_count': len(self.analysis_results)
        }
        
        return summary
    
    def print_analysis_report(self) -> None:
        """Print comprehensive analysis report to stdout."""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analyze_partitions() first.")
        
        print('=' * 100)
        print(f'DATASET ANALYSIS REPORT')
        print(f'Dataset: {self.dataset_name}')
        print(f'Taxonomy: {self.taxonomy_type}')
        print(f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print('=' * 100)
        
        # Configuration summary
        print('\nCONFIGURATION:')
        print(f'  Kover CSV: {self.kover_csv_path}')
        print(f'  Metadata TSV: {self.metadata_tsv_path}')
        print(f'  Data Directory: {self.data_directory}')
        print(f'  Species Column: {self.species_column}')
        print(f'  Label Column: {self.label_column}')
        
        # Summary statistics
        summary = self.generate_summary_statistics()
        print('\nOVERALL SUMMARY:')
        print(f'  Total Samples: {summary["total_samples"]:,}')
        print(f'  Total Resistant: {summary["total_resistant"]:,} ({summary["overall_resistant_rate"]:.1f}%)')
        print(f'  Total Sensitive: {summary["total_sensitive"]:,} ({summary["overall_sensitive_rate"]:.1f}%)')
        print(f'  Total Unique Species: {summary["total_unique_species"]:,}')
        print(f'  Partitions: {summary["partition_count"]}')
        
        # Top species globally
        print('\nTOP 10 SPECIES (GLOBALLY):')
        print(f'{"Species":<50} {"Count":<10}')
        print('-' * 60)
        for species, count in summary['top_species']:
            species_display = species[:47] + '...' if len(species) > 50 else species
            print(f'{species_display:<50} {count:<10}')
        
        # Partition summary table
        print('\nPARTITION SUMMARY:')
        print(f'{"Partition":<20} {"Samples":<10} {"Species":<10} {"Resistant":<12} {"Sensitive":<12} {"R Rate %":<10}')
        print('-' * 85)
        
        for partition in sorted(self.analysis_results.keys()):
            data = self.analysis_results[partition]
            print(f'{partition:<20} {data["total_samples"]:<10} {data["unique_species"]:<10} '
                  f'{data["resistant_count"]:<12} {data["sensitive_count"]:<12} '
                  f'{data["resistant_rate"]:<10.1f}')
        
        # Detailed partition analysis
        print('\n' + '=' * 100)
        print('DETAILED PARTITION ANALYSIS')
        print('=' * 100)
        
        for partition in sorted(self.analysis_results.keys()):
            self._print_partition_details(partition, self.analysis_results[partition])
    
    def _print_partition_details(self, partition: str, data: Dict[str, Any]) -> None:
        """Print detailed analysis for a specific partition."""
        print(f'\n{"=" * 80}')
        print(f'PARTITION: {partition}')
        print(f'{"=" * 80}')
        print(f'Total Samples: {data["total_samples"]:,}')
        print(f'Resistant: {data["resistant_count"]:,} ({data["resistant_rate"]:.1f}%)')
        print(f'Sensitive: {data["sensitive_count"]:,} ({data["sensitive_rate"]:.1f}%)')
        print(f'Unique Species: {data["unique_species"]:,}')
        
        if data['species_analysis']:
            print(f'\nSPECIES BREAKDOWN:')
            print(f'{"Species":<50} {"Total":<8} {"R":<6} {"S":<6} {"R%":<8}')
            print('-' * 78)
            
            # Show top 20 species for readability
            species_to_show = data['species_analysis'][:20]
            for species_info in species_to_show:
                species_name = species_info['species']
                species_display = species_name[:47] + '...' if len(species_name) > 50 else species_name
                
                print(f'{species_display:<50} {species_info["total_count"]:<8} '
                      f'{species_info["resistant_count"]:<6} {species_info["sensitive_count"]:<6} '
                      f'{species_info["resistant_rate"]:<8.1f}')
            
            if len(data['species_analysis']) > 20:
                remaining = len(data['species_analysis']) - 20
                print(f'... and {remaining} more species')
        else:
            print('\nNo species data available for this partition.')
    
    def save_detailed_report(self, output_path: str) -> None:
        """
        Save detailed analysis report to file.
        
        Args:
            output_path: Path where to save the report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Redirect stdout to file
            original_stdout = sys.stdout
            sys.stdout = f
            
            try:
                self.print_analysis_report()
            finally:
                sys.stdout = original_stdout
        
        logger.info(f"Detailed report saved to: {output_path}")
    
    def save_summary_csv(self, output_path: str) -> None:
        """
        Save partition summary as CSV file.
        
        Args:
            output_path: Path where to save the CSV
        """
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analyze_partitions() first.")
        
        # Create summary DataFrame
        summary_data = []
        for partition, data in self.analysis_results.items():
            summary_data.append({
                'partition': partition,
                'total_samples': data['total_samples'],
                'unique_species': data['unique_species'],
                'resistant_count': data['resistant_count'],
                'sensitive_count': data['sensitive_count'],
                'resistant_rate': data['resistant_rate'],
                'sensitive_rate': data['sensitive_rate']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)
        logger.info(f"Summary CSV saved to: {output_path}")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Analyze AMR dataset distribution with species and resistance statistics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--kover-csv', type=str, required=True,
                        help='Path to the kover CSV file with labels and partitions')
    parser.add_argument('--metadata-tsv', type=str, required=True,
                        help='Path to the metadata TSV file with taxonomic information')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing processed data files')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                        help='Directory for output files')
    parser.add_argument('--dataset-name', type=str, default='AMR Dataset',
                        help='Name of the dataset for reporting')
    parser.add_argument('--use-ncbi', action='store_true',
                        help='Use NCBI taxonomy instead of GTDB')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save summary as CSV file')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize analyzer
        analyzer = DatasetAnalyzer(
            kover_csv_path=args.kover_csv,
            metadata_tsv_path=args.metadata_tsv,
            data_directory=args.data_dir,
            use_gtdb=not args.use_ncbi,
            dataset_name=args.dataset_name
        )
        
        # Load data and run analysis
        analyzer.load_data()
        analyzer.analyze_partitions()
        
        # Generate timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Print to console and save detailed report
        analyzer.print_analysis_report()
        
        # Save detailed report
        report_path = output_dir / f'dataset_analysis_{timestamp}.txt'
        analyzer.save_detailed_report(report_path)
        
        # Save CSV summary if requested
        if args.save_csv:
            csv_path = output_dir / f'partition_summary_{timestamp}.csv'
            analyzer.save_summary_csv(csv_path)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Detailed report: {report_path}")
        if args.save_csv:
            print(f"CSV summary: {csv_path}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()