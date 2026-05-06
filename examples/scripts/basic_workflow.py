#!/usr/bin/env python3
"""
Basic Evo-AMR Workflow Example
=============================

This script demonstrates a complete workflow from genomic sequences
to AMR predictions using the Evo-AMR pipeline.

Usage:
    python basic_workflow.py --input sequences.csv --output results/

Requirements:
    - Input CSV with columns: sequence_id, sequence, label (optional)
    - Configured Evo-AMR environment
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Evo-AMR imports
from config.config_manager import ConfigManager
from embedding_pipeline.embedding_generator import EmbeddingGenerator
from minirocket.minirocket_pipeline.core.minirocket_combined_pipeline import MiniRocketPipeline
from utils.validation import validate_input_file, ValidationError
from utils.logging import setup_logger


def main():
    """Main workflow function."""
    parser = argparse.ArgumentParser(description='Basic Evo-AMR workflow')
    parser.add_argument('--input', required=True, type=Path,
                       help='Input CSV file with sequences')
    parser.add_argument('--output', required=True, type=Path,
                       help='Output directory for results')
    parser.add_argument('--model', default='evo-1-8k-base',
                       help='Model to use for embeddings')
    parser.add_argument('--limit', type=int,
                       help='Limit number of sequences (for testing)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(__name__, level=log_level)
    
    try:
        # Step 1: Setup configuration
        logger.info("Setting up configuration...")
        config = ConfigManager()
        
        # Step 2: Validate input
        logger.info(f"Validating input file: {args.input}")
        validate_input_file(args.input, 'csv', ['sequence_id', 'sequence'])
        
        # Step 3: Load sequences
        logger.info("Loading sequences...")
        df = pd.read_csv(args.input)
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Limited to {len(df)} sequences")
        
        sequences = df['sequence'].tolist()
        sequence_ids = df['sequence_id'].tolist()
        
        # Step 4: Generate embeddings
        logger.info(f"Generating embeddings using {args.model}...")
        embedding_generator = EmbeddingGenerator(
            model_name=args.model,
            config=config
        )
        
        embeddings = embedding_generator.generate_embeddings(
            sequences=sequences,
            sequence_ids=sequence_ids,
            batch_size=8
        )
        
        # Step 5: Save embeddings
        embeddings_file = args.output / "embeddings.h5"
        args.output.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving embeddings to {embeddings_file}")
        embedding_generator.save_embeddings(embeddings, embeddings_file)
        
        # Step 6: Feature extraction with MiniRocket (if labels available)
        if 'label' in df.columns:
            logger.info("Running MiniRocket classification...")
            
            # Prepare data for MiniRocket
            labels = df['label'].values
            
            # Initialize MiniRocket pipeline
            mr_pipeline = MiniRocketPipeline(config=config)
            
            # Train MiniRocket features
            features = mr_pipeline.fit_transform(embeddings)
            
            # Train classifier
            accuracy = mr_pipeline.train_classifier(features, labels)
            
            logger.info(f"MiniRocket accuracy: {accuracy:.3f}")
            
            # Save results
            results = {
                'sequence_id': sequence_ids,
                'true_label': labels,
                'prediction': mr_pipeline.predict(features),
                'accuracy': accuracy
            }
            
            results_df = pd.DataFrame(results)
            results_file = args.output / "predictions.csv"
            results_df.to_csv(results_file, index=False)
            
            logger.info(f"Results saved to {results_file}")
        
        # Step 7: Generate summary report
        logger.info("Generating summary report...")
        generate_summary_report(
            input_file=args.input,
            output_dir=args.output,
            num_sequences=len(sequences),
            model_used=args.model,
            has_labels='label' in df.columns
        )
        
        logger.info("Workflow completed successfully!")
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return 1
    
    return 0


def generate_summary_report(input_file: Path, 
                          output_dir: Path,
                          num_sequences: int,
                          model_used: str,
                          has_labels: bool):
    """Generate a summary report of the workflow."""
    
    report_content = f"""
# Evo-AMR Workflow Summary

## Input
- **File**: {input_file}
- **Sequences processed**: {num_sequences}
- **Model used**: {model_used}

## Outputs
- **Embeddings**: {output_dir}/embeddings.h5
{"- **Predictions**: " + str(output_dir) + "/predictions.csv" if has_labels else ""}

## Processing Steps
1. ✅ Input validation
2. ✅ Sequence loading
3. ✅ Embedding generation
4. ✅ Feature extraction{"" if not has_labels else " and classification"}

## Files Generated
"""
    
    # List all files in output directory
    for file_path in output_dir.glob("*"):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            report_content += f"- `{file_path.name}` ({size_mb:.2f} MB)\n"
    
    report_content += f"""
## Next Steps
- View embeddings: Load `embeddings.h5` with h5py or pandas
{"- Analyze predictions: Check `predictions.csv` for classification results" if has_labels else ""}
- Advanced analysis: Use embeddings for clustering, visualization, or custom modeling

Generated by Evo-AMR basic workflow
"""
    
    report_file = output_dir / "workflow_summary.md"
    with open(report_file, 'w') as f:
        f.write(report_content)


if __name__ == "__main__":
    exit(main())