import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple
import seaborn as sns
import re
import argparse
from pathlib import Path

class TorchProfileAnalyzer:
    QUANT_METHODS = {
        'int8': 'INT8',
        'int4': 'INT4',
        'awq': 'AWQ',
        'gptq': 'GPTQ',
        'sqq': 'SQQ',
    }

    def __init__(self):
        self.kernel_stats = defaultdict(list)
        self.summary_stats = {}
        
        # Define layer operation patterns
        self.layer_patterns = {
            'linear': [
                r'gemm', r'mm_', r'linear', r'dot', r'addmm', r'cutlass'
            ],
            'attention': [
                r'attention', r'softmax', r'scaled_dot_product'
            ],
            'normalization': [
                r'layer_norm', r'batch_norm', r'norm', r'mean', r'var'
            ],
            'activation': [
                r'relu', r'gelu', r'silu', r'swish', r'tanh', r'sigmoid'
            ],
            'elementwise': [
                r'add_', r'mul_', r'div_', r'sub_', r'elementwise'
            ],
            'memory': [
                r'copy', r'transpose', r'permute', r'view', r'reshape'
            ],
            'embedding': [
                r'embed', r'gather', r'index_select'
            ],
            'dropout': [
                r'dropout'
            ]
        }

    def identify_layer_type(self, kernel_name: str) -> str:
        """Identify the type of layer operation based on kernel name."""
        kernel_lower = kernel_name.lower()
        
        for layer_type, patterns in self.layer_patterns.items():
            if any(re.search(pattern, kernel_lower) for pattern in patterns):
                return layer_type
                
        return 'other'

    def load_profile_data(self, filepath: str) -> None:
        """Load and parse the PyTorch profiler JSON file with validation."""
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
                # Extract traceEvents from the JSON structure
                profile_data = data.get('traceEvents', [])
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file format: {filepath}\nError: {e}")

        # Validate that traceEvents is a list
        if not isinstance(profile_data, list):
            raise ValueError(
                f"Expected 'traceEvents' to be a list in: {filepath}. "
                f"Got type {type(profile_data)} instead."
            )

        # Filter for kernel events
        kernel_events = [event for event in profile_data if event.get('cat') == 'kernel']

        if not kernel_events:
            raise ValueError(
                f"No kernel events found in: {filepath}. "
                "Check if the profiler output is correct or if 'cat' field is present."
            )

        # Process each kernel event
        for event in kernel_events:
            self.kernel_stats['name'].append(event['name'])
            self.kernel_stats['duration'].append(event['dur'])
            self.kernel_stats['layer_type'].append(
                self.identify_layer_type(event['name']))
            self.kernel_stats['occupancy'].append(
                event['args'].get('est. achieved occupancy %', 0))
            self.kernel_stats['blocks_per_sm'].append(
                event['args'].get('blocks per SM', 0))
            self.kernel_stats['warps_per_sm'].append(
                event['args'].get('warps per SM', 0))
            self.kernel_stats['shared_memory'].append(
                event['args'].get('shared memory', 0))
    def analyze_layer_types(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze performance grouped by layer types."""
        df = pd.DataFrame(self.kernel_stats)
        
        # Layer type summary
        layer_summary = df.groupby('layer_type').agg({
            'duration': ['count', 'mean', 'sum', 'std'],
            'occupancy': 'mean',
            'shared_memory': 'mean'
        }).round(2)
        
        # Calculate percentage of total time
        total_time = df['duration'].sum()
        layer_summary['time_percentage'] = (
            layer_summary[('duration', 'sum')] / total_time * 100
        ).round(2)
        
        # Detailed kernel analysis within each layer type
        kernel_analysis = df.groupby(['layer_type', 'name']).agg({
            'duration': ['count', 'mean', 'sum', 'std'],
            'occupancy': 'mean',
            'shared_memory': 'mean'
        }).round(2)
        
        kernel_analysis['time_percentage'] = (
            kernel_analysis[('duration', 'sum')] / total_time * 100
        ).round(2)
        
        return layer_summary, kernel_analysis

    def compare_quantization_by_layer(self, fp32_file: str, quant_files: Dict[str, str]) -> Dict:
        """
        Compare layer type performance across different quantization methods.
        
        Args:
            fp32_file: Path to FP32 profile data
            quant_files: Dictionary mapping quantization method to profile data file
        """
        results = {}
        
        # Process FP32 (baseline) first
        self.__init__()
        self.load_profile_data(fp32_file)
        layer_summary, _ = self.analyze_layer_types()
        results['FP32'] = {
            'layer_summary': layer_summary,
            'total_time': layer_summary[('duration', 'sum')].sum()
        }
        
        # Process each quantization method
        for quant_method, filepath in quant_files.items():
            self.__init__()
            self.load_profile_data(filepath)
            layer_summary, _ = self.analyze_layer_types()
            
            results[quant_method] = {
                'layer_summary': layer_summary,
                'total_time': layer_summary[('duration', 'sum')].sum()
            }
            
        # Calculate speedup ratios for each layer type
        speedup_analysis = {}
        for layer_type in self.layer_patterns.keys():
            speedup_analysis[layer_type] = {}
            for quant_method in quant_files.keys():
                if layer_type in results['FP32']['layer_summary'].index:
                    fp32_time = results['FP32']['layer_summary'].loc[layer_type, ('duration', 'sum')]
                    quant_time = results[quant_method]['layer_summary'].loc[layer_type, ('duration', 'sum')] \
                        if layer_type in results[quant_method]['layer_summary'].index else fp32_time
                    speedup_analysis[layer_type][f'{quant_method}_speedup'] = fp32_time / quant_time
                else:
                    speedup_analysis[layer_type][f'{quant_method}_speedup'] = 1.0
            
        return {'quantization_results': results, 'speedup_analysis': speedup_analysis}

    def plot_layer_analysis(self, comparison_results: Dict, quant_methods: List[str], save_path: str = None):
        """Create visualizations for layer-wise analysis with multiple quantization methods."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Time distribution across layer types for each quantization
        plt.subplot(2, 1, 1)
        data = []
        for quant_type, result in comparison_results['quantization_results'].items():
            summary = result['layer_summary']
            for layer_type in summary.index:
                # Ensure time_percentage is accessed correctly from the summary DataFrame
                time_pct = summary.loc[layer_type, 'time_percentage']
                if isinstance(time_pct, pd.Series):
                    time_pct = time_pct.iloc[0]
                data.append({
                    'Quantization': quant_type,
                    'Layer Type': layer_type,
                    'Time %': float(time_pct)  # Convert to float to avoid any ambiguity
                })
        
        plot_df = pd.DataFrame(data)
        # Ensure the data types are correct
        plot_df['Time %'] = plot_df['Time %'].astype(float)
        plot_df['Layer Type'] = plot_df['Layer Type'].astype(str)
        plot_df['Quantization'] = plot_df['Quantization'].astype(str)
        
        # Create the barplot with explicit categorical data
        sns.barplot(data=plot_df, x='Layer Type', y='Time %', hue='Quantization')
        plt.title('Time Distribution Across Layer Types')
        plt.xticks(rotation=45)
        
        # Plot 2: Speedup comparison
        plt.subplot(2, 1, 2)
        speedup_data = []
        for layer_type, speedups in comparison_results['speedup_analysis'].items():
            speedup_dict = {'Layer Type': layer_type}
            for quant_method in quant_methods:
                speedup_dict[f'{quant_method} Speedup'] = float(speedups[f'{quant_method}_speedup'])
            speedup_data.append(speedup_dict)
        
        speedup_df = pd.DataFrame(speedup_data)
        # Convert speedup columns to float
        for method in quant_methods:
            speedup_df[f'{method} Speedup'] = speedup_df[f'{method} Speedup'].astype(float)
        
        speedup_df.plot(x='Layer Type', 
                    y=[f'{method} Speedup' for method in quant_methods], 
                    kind='bar', width=0.8)
        plt.title('Speedup by Layer Type')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze PyTorch profiler results across quantization methods')
    parser.add_argument('--fp32', required=True, help='Path to FP32 profile data')
    parser.add_argument('--quant-methods', nargs='+', choices=TorchProfileAnalyzer.QUANT_METHODS.keys(),
                      help='Quantization methods to analyze')
    parser.add_argument('--profile-dir', type=str, default='.',
                      help='Directory containing profile data files')
    parser.add_argument('--output-dir', type=str, default='.',
                      help='Directory for output files')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup analyzer
    analyzer = TorchProfileAnalyzer()
    
    # Create dictionary of quantization files
    quant_files = {}
    for method in args.quant_methods:
        profile_path = Path(args.profile_dir) / f'{method}_profile.json'
        if profile_path.exists():
            quant_files[analyzer.QUANT_METHODS[method]] = str(profile_path)
        else:
            print(f"Warning: Profile data for {method} not found at {profile_path}")
    
    if not quant_files:
        print("Error: No valid quantization profile data found")
        return
    
    # Run comparison
    comparison_results = analyzer.compare_quantization_by_layer(
        args.fp32,
        quant_files
    )
    
    # Print results
    print("\nFP32 Layer Type Analysis:")
    print(comparison_results['quantization_results']['FP32']['layer_summary'])
    
    print("\nSpeedup Analysis by Layer Type:")
    for layer_type, speedups in comparison_results['speedup_analysis'].items():
        print(f"\n{layer_type}:")
        for quant_method in quant_files.keys():
            speedup = speedups[f'{quant_method}_speedup']
            print(f"  {quant_method} Speedup: {speedup:.2f}x")
    
    # Generate visualization
    analyzer.plot_layer_analysis(
        comparison_results,
        list(quant_files.keys()),
        output_dir / 'layer_analysis.png'
    )

if __name__ == "__main__":
    main()