import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from nano_atom.analysis import (
    compute_transmittance, 
    compute_polarization_conversion, 
    compute_far_field_phase
)
from nano_atom.material import get_n


def discover_wavelength_dirs(results_dir):
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    wavelength_dirs = []
    for item in results_path.iterdir():
        if item.is_dir() and item.name.startswith("wavelength_"):
            wavelength_dirs.append(item)
    
    wavelength_dirs.sort(key=lambda x: int(x.name.split("_")[1].replace("nm", "")))
    return wavelength_dirs


def discover_angle_dirs(wavelength_dir):
    angle_dirs = []
    for item in wavelength_dir.iterdir():
        if item.is_dir() and item.name.startswith("angle_"):
            angle_dirs.append(item)
    
    angle_dirs.sort(key=lambda x: int(x.name.split("_")[1].replace("deg", "")))
    return angle_dirs


def load_simulation_data(angle_dir):
    try:
        solution_path = angle_dir / "solution.pt"
        if not solution_path.exists():
            return None
        
        solution = torch.load(solution_path, map_location='cpu')
        
        metrics_path = angle_dir / "metrics.yaml"
        metrics = {}
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = yaml.safe_load(f)
        
        eps_path = angle_dir / "eps.pt"
        src_path = angle_dir / "src.pt"
        
        eps = None
        src = None
        if eps_path.exists():
            eps = torch.load(eps_path, map_location='cpu')
        if src_path.exists():
            src = torch.load(src_path, map_location='cpu')
        
        return {
            'solution': solution,
            'eps': eps,
            'src': src,
            'metrics': metrics,
            'angle_dir': angle_dir
        }
    except Exception as e:
        print(f"Error loading data from {angle_dir}: {e}")
        return None


def analyze_wavelength_sweep(wavelength_dir, wavelength, dL, kwargs, angle=None):
    if angle is not None:
        print(f"Analyzing wavelength {wavelength}nm at angle {angle}°...")
    else:
        print(f"Analyzing wavelength {wavelength}nm across all angles...")
    
    angle_dirs = discover_angle_dirs(wavelength_dir)
    if not angle_dirs:
        print(f"  No angle directories found in {wavelength_dir}")
        return None
    
    inc_flux = None
    
    if angle is not None:
        target_angle_dir = None
        for angle_dir in angle_dirs:
            angle_from_dir = int(angle_dir.name.split("_")[1].replace("deg", ""))
            if angle_from_dir == angle:
                target_angle_dir = angle_dir
                break
        
        if target_angle_dir is None:
            print(f"  Angle {angle}° not found in {wavelength_dir}")
            return None
        
        angle_dirs = [target_angle_dir]
    
    results = {
        'wavelength': wavelength,
        'angles': [],
        'transmittances': [],
        'conversions': [],
        'efficiencies': [],
        'phases': [],
        'successful_runs': 0,
        'failed_runs': 0
    }

    monitor_z_idx = 110
    
    for angle_dir in angle_dirs:
        angle = int(angle_dir.name.split("_")[1].replace("deg", ""))
        substrate_material = kwargs.get("substrate_material")
        top_medium_material = kwargs.get("top_medium_material")
        meta_atom_material = kwargs.get("meta_atom_material")
        n_substrate = get_n(substrate_material, wavelength * 1e-3)
        n_top_medium = get_n(top_medium_material, wavelength * 1e-3)
        n_meta_atom = get_n(meta_atom_material, wavelength * 1e-3)
        
        data = load_simulation_data(angle_dir)
        if data is None:
            print(f"  Failed to load data for angle {angle}°")
            results['failed_runs'] += 1
            continue
        
        try:
            transmittance = compute_transmittance(
                data['solution'], 
                monitor_z_idx=monitor_z_idx,
                n_monitor=n_top_medium,
                n_source=n_substrate,
                wvln=wavelength,
                dL=dL,
                inc_flux=inc_flux,
                src=data['src'],
            )
            print('wavelength: ', wavelength)
            
            conversion = compute_polarization_conversion(
                data['solution'], 
                monitor_z_idx=monitor_z_idx,
                n_monitor=n_top_medium,
                pol='lcp',
                wvln=wavelength,
                dL=dL
            )
            
            phase = compute_far_field_phase(
                data['solution'], 
                monitor_z_idx=monitor_z_idx,
                pol='lcp',
            )

            if inc_flux is None:
                inc_flux = transmittance['incident_flux']
            results['angles'].append(angle)
            results['transmittances'].append(transmittance['transmittance'])
            results['conversions'].append(conversion['conversion_efficiency'])
            results['efficiencies'].append(transmittance['transmittance'] * conversion['conversion_efficiency'])
            results['phases'].append(phase)
            results['successful_runs'] += 1
            
            print(f"  Angle {angle}°: T={transmittance['transmittance']:.4f}, C={conversion['conversion_efficiency']:.4f}, E={transmittance['transmittance']*conversion['conversion_efficiency']:.4f}")
            
        except Exception as e:
            print(f"  Error analyzing angle {angle}°: {e}")
            results['failed_runs'] += 1
    
    if results['successful_runs'] > 0:
        results['avg_transmittance'] = np.mean(results['transmittances'])
        results['avg_conversion'] = np.mean(results['conversions'])
        results['avg_efficiency'] = np.mean(results['efficiencies'])
        results['std_transmittance'] = np.std(results['transmittances'])
        results['std_conversion'] = np.std(results['conversions'])
        results['std_efficiency'] = np.std(results['efficiencies'])
    else:
        results['avg_transmittance'] = 0.0
        results['avg_conversion'] = 0.0
        results['avg_efficiency'] = 0.0
        results['std_transmittance'] = 0.0
        results['std_conversion'] = 0.0
        results['std_efficiency'] = 0.0
    
    print(f"  Average T={results['avg_transmittance']:.4f}, C={results['avg_conversion']:.4f}, E={results['avg_efficiency']:.4f}")
    print(f"  Successful: {results['successful_runs']}, Failed: {results['failed_runs']}")
    
    return results


def plot_2d_analysis_results(analysis_results, output_dir):
    if not analysis_results:
        print("No analysis results to plot")
        return
    
    wavelengths = [r['wavelength'] for r in analysis_results]
    avg_transmittances = [r['avg_transmittance'] for r in analysis_results]
    avg_conversions = [r['avg_conversion'] for r in analysis_results]
    avg_efficiencies = [r['avg_efficiency'] for r in analysis_results]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('2D Nanofin Sweep Analysis: Average Performance vs Wavelength', fontsize=16, fontweight='bold')
    
    axes[0].plot(wavelengths, avg_transmittances, marker='o', linewidth=2, markersize=8,
                 label='Average Transmittance', color='tab:blue', alpha=0.8)
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Transmittance')
    axes[0].set_title('Average Transmittance vs Wavelength')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(wavelengths, avg_conversions, marker='s', linewidth=2, markersize=8,
                 label='Average Conversion Efficiency', color='tab:orange', alpha=0.8)
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Conversion Efficiency')
    axes[1].set_title('Average Polarization Conversion vs Wavelength')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(wavelengths, avg_efficiencies, marker='^', linewidth=2, markersize=8,
                 label='Average Overall Efficiency', color='tab:green', alpha=0.8)
    axes[2].set_xlabel('Wavelength (nm)')
    axes[2].set_ylabel('Overall Efficiency')
    axes[2].set_title('Average Overall Efficiency vs Wavelength')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    plot_path = output_dir / "nanofin_2d_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis plot saved to: {plot_path}")


def print_summary_statistics(analysis_results):
    if not analysis_results:
        print("No analysis results to summarize")
        return
    
    print("\n" + "="*70)
    print("2D SWEEP ANALYSIS SUMMARY")
    print("="*70)
    
    all_transmittances = []
    all_conversions = []
    all_efficiencies = []
    total_successful = 0
    total_failed = 0
    
    for result in analysis_results:
        all_transmittances.extend(result['transmittances'])
        all_conversions.extend(result['conversions'])
        all_efficiencies.extend(result['efficiencies'])
        total_successful += result['successful_runs']
        total_failed += result['failed_runs']
    
    print(f"Total simulations analyzed: {total_successful}")
    print(f"Failed simulations: {total_failed}")
    print(f"Success rate: {total_successful/(total_successful+total_failed)*100:.1f}%")
    
    if all_transmittances:
        print(f"\nTransmittance range: {min(all_transmittances):.4f} - {max(all_transmittances):.4f}")
        print(f"Mean transmittance: {np.mean(all_transmittances):.4f} ± {np.std(all_transmittances):.4f}")
    
    if all_conversions:
        print(f"Conversion efficiency range: {min(all_conversions):.4f} - {max(all_conversions):.4f}")
        print(f"Mean conversion efficiency: {np.mean(all_conversions):.4f} ± {np.std(all_conversions):.4f}")
    
    if all_efficiencies:
        print(f"Overall efficiency range: {min(all_efficiencies):.4f} - {max(all_efficiencies):.4f}")
        print(f"Mean overall efficiency: {np.mean(all_efficiencies):.4f} ± {np.std(all_efficiencies):.4f}")
    
    if analysis_results:
        best_wavelength_idx = np.argmax([r['avg_efficiency'] for r in analysis_results])
        best_result = analysis_results[best_wavelength_idx]
        print(f"\nBest performing wavelength: {best_result['wavelength']}nm")
        print(f"  Average transmittance: {best_result['avg_transmittance']:.4f}")
        print(f"  Average conversion: {best_result['avg_conversion']:.4f}")
        print(f"  Average efficiency: {best_result['avg_efficiency']:.4f}")


def load_config(config_path="nano_atom/configs/nanofin_g.yaml"):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {
            "sim_shape": [32, 32, 180],
            "dL": 10.1562,
            "kwargs": {
                "substrate_material": "SiO2",
                "meta_atom_material": "TiO2", 
                "top_medium_material": "Air"
            }
        }


def main(angle=None, config_path="nano_atom/configs/nanofin_g.yaml"):
    config = load_config(config_path)
    dL = config.get("dL", 10.1562)
    sim_shape = config.get("sim_shape", [32, 32, 180])
    kwargs = config.get("kwargs", {})
    
    results_dir = "nano_atom/results/nanofin_2d_sweep_b"
    
    print("="*70)
    if angle is not None:
        print(f"2D NANOFIN SWEEP ANALYSIS - ANGLE {angle}°")
    else:
        print("2D NANOFIN SWEEP ANALYSIS - ALL ANGLES")
    print("="*70)
    print(f"Results directory: {results_dir}")
    print(f"Grid size: {dL} nm")
    print(f"Simulation shape: {sim_shape}")
    print(f"Substrate material: {kwargs.get('substrate_material', 'Unknown')}")
    print(f"Meta-atom material: {kwargs.get('meta_atom_material', 'Unknown')}")
    print(f"Top medium material: {kwargs.get('top_medium_material', 'Unknown')}")
    if angle is not None:
        print(f"Analyzing angle: {angle}°")
    print()
    
    # Discover wavelength directories
    try:
        wavelength_dirs = discover_wavelength_dirs(results_dir)
        print(f"Found {len(wavelength_dirs)} wavelength directories")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    if not wavelength_dirs:
        print("No wavelength directories found")
        return
    
    analysis_results = []
    for wavelength_dir in wavelength_dirs:
        wavelength = int(wavelength_dir.name.split("_")[1].replace("nm", ""))
        
        result = analyze_wavelength_sweep(wavelength_dir, wavelength, dL, kwargs, angle)
        if result is not None:
            analysis_results.append(result)
    
    if not analysis_results:
        print("No successful analysis results")
        return
    
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_2d_analysis_results(analysis_results, output_dir)
    
    print_summary_statistics(analysis_results)
    
    print("\n2D sweep analysis completed successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze 2D nanofin sweep results")
    parser.add_argument("--angle", type=int, default=None, 
                       help="Specific angle to analyze (in degrees). If not provided, analyzes all angles.")
    parser.add_argument("--config", type=str, default="nano_atom/configs/nanofin_b.yaml",
                       help="Path to the YAML configuration file.")
    
    args = parser.parse_args()
    main(angle=None, config_path=args.config)
