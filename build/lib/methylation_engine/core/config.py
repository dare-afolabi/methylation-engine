#!/usr/bin/env python
# coding: utf-8

"""
Study Planner Configuration Database
Centralized configuration for platforms, designs, and cost models
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


# ============================================================================
# DEFAULT CONFIGURATION DATABASE
# ============================================================================

DEFAULT_PLATFORMS = {
    '27K': {
        'name': 'HumanMethylation27',
        'manufacturer': 'Illumina',
        'n_cpgs': 27578,
        'cost_per_sample': 250,
        'processing_days': 5,
        'dna_required_ng': 500,
        'coverage': 'Gene promoters',
        'release_year': 2008,
        'status': 'Discontinued',
        'recommended': False,
        'notes': 'Legacy platform, no longer available'
    },
    '450K': {
        'name': 'HumanMethylation450',
        'manufacturer': 'Illumina',
        'n_cpgs': 485512,
        'cost_per_sample': 400,
        'processing_days': 7,
        'dna_required_ng': 500,
        'coverage': 'Genome-wide',
        'release_year': 2011,
        'status': 'Mature',
        'recommended': True,
        'notes': 'Widely used, extensive literature'
    },
    'EPIC': {
        'name': 'MethylationEPIC',
        'manufacturer': 'Illumina',
        'n_cpgs': 866895,
        'cost_per_sample': 500,
        'processing_days': 7,
        'dna_required_ng': 500,
        'coverage': 'Enhanced',
        'release_year': 2016,
        'status': 'Current standard',
        'recommended': True,
        'notes': 'Most popular for new studies'
    },
    'EPICv2': {
        'name': 'MethylationEPIC v2',
        'manufacturer': 'Illumina',
        'n_cpgs': 935000,
        'cost_per_sample': 550,
        'processing_days': 7,
        'dna_required_ng': 200,
        'coverage': 'Improved',
        'release_year': 2022,
        'status': 'Latest',
        'recommended': True,
        'notes': 'Improved content, reduced DNA input'
    },
    'WGBS': {
        'name': 'Whole Genome BS-Seq',
        'manufacturer': 'Multiple',
        'n_cpgs': 28000000,
        'cost_per_sample': 1200,
        'processing_days': 14,
        'dna_required_ng': 1000,
        'coverage': 'High resolution',
        'release_year': None,
        'status': 'Gold standard',
        'recommended': False,
        'notes': 'For targeted applications, high cost'
    },
    'RRBS': {
        'name': 'Reduced Representation BS-Seq',
        'manufacturer': 'Multiple',
        'n_cpgs': 2000000,
        'cost_per_sample': 400,
        'processing_days': 10,
        'dna_required_ng': 500,
        'coverage': 'CpG islands',
        'release_year': None,
        'status': 'Specialized',
        'recommended': False,
        'notes': 'Cost-effective for CpG islands'
    },
    'Nanopore': {
        'name': 'Oxford Nanopore Methylation',
        'manufacturer': 'Oxford Nanopore',
        'n_cpgs': 28000000,
        'cost_per_sample': 800,
        'processing_days': 7,
        'dna_required_ng': 1000,
        'coverage': 'Native',
        'release_year': 2020,
        'status': 'Emerging',
        'recommended': False,
        'notes': 'Long reads, direct detection'
    }
}

DEFAULT_DESIGNS = {
    'two_group': {
        'name': 'Two-Group Comparison',
        'description': 'Compare two independent groups (e.g., case vs control)',
        'n_groups': 2,
        'paired': False,
        'complexity': 'Simple',
        'min_n_recommended': 12,
        'power_adjustment': 1.0,
        'analysis_method': 'Two-sample t-test / Linear model',
        'example_uses': ['Case-control', 'Tumor vs normal', 'Treatment vs control']
    },
    'paired': {
        'name': 'Paired Design',
        'description': 'Compare matched samples (e.g., before/after treatment)',
        'n_groups': 2,
        'paired': True,
        'complexity': 'Simple',
        'min_n_recommended': 10,
        'power_adjustment': 0.71,
        'analysis_method': 'Paired t-test / Linear mixed model',
        'example_uses': ['Before-after treatment', 'Matched case-control', 'Twin studies']
    },
    'multi_group': {
        'name': 'Multi-Group Comparison',
        'description': 'Compare 3+ groups (e.g., control, low, high dose)',
        'n_groups': 3,
        'paired': False,
        'complexity': 'Moderate',
        'min_n_recommended': 15,
        'power_adjustment': 1.2,
        'analysis_method': 'ANOVA / Linear model with contrasts',
        'example_uses': ['Dose response', 'Disease subtypes', 'Multiple conditions']
    },
    'time_series': {
        'name': 'Time Series',
        'description': 'Multiple time points (e.g., 0h, 24h, 48h)',
        'n_groups': 3,
        'paired': True,
        'complexity': 'Complex',
        'min_n_recommended': 8,
        'power_adjustment': 0.85,
        'analysis_method': 'Repeated measures / Mixed effects model',
        'example_uses': ['Time course', 'Development', 'Disease progression']
    },
    'factorial': {
        'name': 'Factorial Design',
        'description': '2+ factors (e.g., treatment × genotype)',
        'n_groups': 4,
        'paired': False,
        'complexity': 'Complex',
        'min_n_recommended': 20,
        'power_adjustment': 1.5,
        'analysis_method': 'Two-way ANOVA / Linear model with interactions',
        'example_uses': ['Gene-environment', 'Treatment-genotype', 'Multi-factor']
    },
    'longitudinal': {
        'name': 'Longitudinal',
        'description': 'Multiple subjects, multiple time points',
        'n_groups': 2,
        'paired': True,
        'complexity': 'Complex',
        'min_n_recommended': 15,
        'power_adjustment': 0.80,
        'analysis_method': 'Linear mixed effects model',
        'example_uses': ['Clinical trials', 'Aging studies', 'Treatment response']
    }
}

DEFAULT_COST_COMPONENTS = {
    'dna_extraction': {
        'cost': 50,
        'unit': 'per_sample',
        'description': 'DNA extraction from tissue/blood',
        'optional': False
    },
    'quality_control': {
        'cost': 30,
        'unit': 'per_sample',
        'description': 'DNA QC (NanoDrop, Qubit, gel)',
        'optional': False
    },
    'bisulfite_conversion': {
        'cost': 40,
        'unit': 'per_sample',
        'description': 'Bisulfite conversion (for BS-seq only)',
        'optional': True,
        'applies_to': ['WGBS', 'RRBS']
    },
    'library_prep': {
        'cost': 80,
        'unit': 'per_sample',
        'description': 'Library preparation (for sequencing)',
        'optional': True,
        'applies_to': ['WGBS', 'RRBS', 'Nanopore']
    },
    'data_storage': {
        'cost': 20,
        'unit': 'per_sample',
        'description': 'Cloud storage and backup',
        'optional': False
    },
    'bioinformatics': {
        'cost': 100,
        'unit': 'per_sample',
        'description': 'Data processing and analysis',
        'optional': False
    },
    'project_management': {
        'cost': 50,
        'unit': 'per_sample',
        'description': 'Project coordination and reporting',
        'optional': True
    },
    'validation': {
        'cost': 150,
        'unit': 'per_cpg',
        'description': 'Pyrosequencing validation',
        'optional': True
    }
}

DEFAULT_TIMELINE_PHASES = {
    'planning_irb': {
        'name': 'Planning & IRB',
        'base_duration_days': 30,
        'scaling_factor': 0,
        'description': 'Study design, protocol writing, IRB approval',
        'critical': True
    },
    'sample_collection': {
        'name': 'Sample Collection',
        'base_duration_days': 14,
        'scaling_factor': 0.5,
        'description': 'Collect biological samples',
        'critical': True
    },
    'dna_extraction': {
        'name': 'DNA Extraction',
        'base_duration_days': 5,
        'scaling_factor': 0.2,
        'description': 'Extract DNA, QC, quantification',
        'critical': False
    },
    'array_processing': {
        'name': 'Array Processing',
        'base_duration_days': 7,
        'scaling_factor': 0,
        'batch_adjustment': 2,
        'description': 'Array hybridization and scanning',
        'critical': False
    },
    'data_generation': {
        'name': 'Data Generation',
        'base_duration_days': 3,
        'scaling_factor': 0,
        'description': 'Scanning, data export, QC',
        'critical': False
    },
    'quality_control': {
        'name': 'Quality Control',
        'base_duration_days': 5,
        'scaling_factor': 0,
        'description': 'Sample QC, data preprocessing',
        'critical': False
    },
    'analysis': {
        'name': 'Analysis',
        'base_duration_days': 14,
        'scaling_factor': 0,
        'description': 'Differential methylation analysis',
        'critical': True
    },
    'validation': {
        'name': 'Validation',
        'base_duration_days': 21,
        'scaling_factor': 0,
        'description': 'Result validation, follow-up experiments',
        'critical': False,
        'optional': True
    },
    'manuscript': {
        'name': 'Manuscript',
        'base_duration_days': 60,
        'scaling_factor': 0,
        'description': 'Writing, revision, submission',
        'critical': False,
        'optional': True
    }
}


# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class PlannerConfig:
    """
    Configuration manager for study planner.
    
    Handles loading/saving configurations from files and provides
    centralized access to platform, design, and cost information.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Parameters
        ----------
        config_file : str, optional
            Path to JSON configuration file
        """
        self.platforms = DEFAULT_PLATFORMS.copy()
        self.designs = DEFAULT_DESIGNS.copy()
        self.cost_components = DEFAULT_COST_COMPONENTS.copy()
        self.timeline_phases = DEFAULT_TIMELINE_PHASES.copy()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, filepath: str):
        """
        Load configuration from JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to JSON configuration file
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        if 'platforms' in config:
            self.platforms.update(config['platforms'])
        if 'designs' in config:
            self.designs.update(config['designs'])
        if 'cost_components' in config:
            self.cost_components.update(config['cost_components'])
        if 'timeline_phases' in config:
            self.timeline_phases.update(config['timeline_phases'])
    
    def save_to_file(self, filepath: str):
        """
        Save current configuration to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save JSON configuration
        """
        config = {
            'platforms': self.platforms,
            'designs': self.designs,
            'cost_components': self.cost_components,
            'timeline_phases': self.timeline_phases,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_from_excel(self, filepath: str):
        """
        Load configuration from Excel file.
        
        Parameters
        ----------
        filepath : str
            Path to Excel file with sheets: Platforms, Designs, Costs, Timeline
        """
        # Load platforms
        if 'Platforms' in pd.ExcelFile(filepath).sheet_names:
            df = pd.read_excel(filepath, sheet_name='Platforms')
            for _, row in df.iterrows():
                platform_id = row['platform_id']
                self.platforms[platform_id] = row.to_dict()
        
        # Load designs
        if 'Designs' in pd.ExcelFile(filepath).sheet_names:
            df = pd.read_excel(filepath, sheet_name='Designs')
            for _, row in df.iterrows():
                design_id = row['design_id']
                self.designs[design_id] = row.to_dict()
        
        # Load costs
        if 'Costs' in pd.ExcelFile(filepath).sheet_names:
            df = pd.read_excel(filepath, sheet_name='Costs')
            for _, row in df.iterrows():
                cost_id = row['component_id']
                self.cost_components[cost_id] = row.to_dict()
    
    def export_to_excel(self, filepath: str):
        """
        Export configuration to Excel file.
        
        Parameters
        ----------
        filepath : str
            Path to save Excel file
        """
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Platforms
            pd.DataFrame(self.platforms).T.to_excel(
                writer, sheet_name='Platforms'
            )
            
            # Designs
            pd.DataFrame(self.designs).T.to_excel(
                writer, sheet_name='Designs'
            )
            
            # Costs
            pd.DataFrame(self.cost_components).T.to_excel(
                writer, sheet_name='Costs'
            )
            
            # Timeline
            pd.DataFrame(self.timeline_phases).T.to_excel(
                writer, sheet_name='Timeline'
            )
    
    def get_platform(self, platform_id: str) -> Dict[str, Any]:
        """Get platform information."""
        return self.platforms.get(platform_id, self.platforms['EPIC'])
    
    def get_design(self, design_id: str) -> Dict[str, Any]:
        """Get design information."""
        return self.designs.get(design_id, self.designs['two_group'])
    
    def get_cost_components(
        self, 
        platform: Optional[str] = None,
        include_optional: bool = True
    ) -> Dict[str, Dict]:
        """
        Get applicable cost components.
        
        Parameters
        ----------
        platform : str, optional
            Platform ID to filter platform-specific costs
        include_optional : bool
            Include optional costs
        
        Returns
        -------
        Dict
            Cost components dictionary
        """
        components = {}
        
        for comp_id, comp in self.cost_components.items():
            # Skip optional if not requested
            if not include_optional and comp.get('optional', False):
                continue
            
            # Filter by platform if specified
            if platform and 'applies_to' in comp:
                if platform not in comp['applies_to']:
                    continue
            
            components[comp_id] = comp
        
        return components
    
    def update_platform_cost(self, platform_id: str, new_cost: float):
        """
        Update platform cost (useful for regional pricing).
        
        Parameters
        ----------
        platform_id : str
            Platform identifier
        new_cost : float
            New cost per sample
        """
        if platform_id in self.platforms:
            self.platforms[platform_id]['cost_per_sample'] = new_cost
    
    def add_custom_platform(self, platform_id: str, platform_info: Dict[str, Any]):
        """
        Add custom platform.
        
        Parameters
        ----------
        platform_id : str
            Unique platform identifier
        platform_info : Dict
            Platform specifications
        """
        required_fields = [
            'name', 'n_cpgs', 'cost_per_sample',
            'processing_days', 'coverage'
        ]
        
        for field in required_fields:
            if field not in platform_info:
                raise ValueError(f"Missing required field: {field}")
        
        self.platforms[platform_id] = platform_info
    
    def list_platforms(self, recommended_only: bool = False) -> pd.DataFrame:
        """
        List all available platforms.
        
        Parameters
        ----------
        recommended_only : bool
            Only show recommended platforms
        
        Returns
        -------
        pd.DataFrame
            Platform summary table
        """
        platforms = self.platforms
        
        if recommended_only:
            platforms = {
                k: v for k, v in platforms.items()
                if v.get('recommended', False)
            }
        
        df = pd.DataFrame(platforms).T
        columns = ['name', 'n_cpgs', 'cost_per_sample', 
                  'processing_days', 'coverage', 'recommended']
        
        return df[[c for c in columns if c in df.columns]]
    
    def list_designs(self) -> pd.DataFrame:
        """
        List all available study designs.
        
        Returns
        -------
        pd.DataFrame
            Design summary table
        """
        df = pd.DataFrame(self.designs).T
        columns = ['name', 'description', 'complexity', 
                  'min_n_recommended', 'paired']
        
        return df[[c for c in columns if c in df.columns]]


# ============================================================================
# GLOBAL CONFIGURATION INSTANCE
# ============================================================================

# Create global config instance
_global_config = PlannerConfig()


def get_config() -> PlannerConfig:
    """
    Get global configuration instance.
    
    Returns
    -------
    PlannerConfig
        Global configuration manager
    
    Examples
    --------
    >>> config = get_config()
    >>> epic = config.get_platform('EPIC')
    >>> print(epic['cost_per_sample'])
    500
    """
    return _global_config


def load_config(filepath: str):
    """
    Load configuration from file into global instance.
    
    Parameters
    ----------
    filepath : str
        Path to configuration file (JSON or Excel)
    """
    global _global_config
    
    if filepath.endswith('.json'):
        _global_config.load_from_file(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        _global_config.load_from_excel(filepath)
    else:
        raise ValueError("Config file must be JSON or Excel format")


def export_default_config(filepath: str):
    """
    Export default configuration template.
    
    Parameters
    ----------
    filepath : str
        Path to save configuration
    
    Examples
    --------
    >>> export_default_config('my_config.xlsx')
    >>> # Edit in Excel, then load
    >>> load_config('my_config.xlsx')
    """
    config = PlannerConfig()
    
    if filepath.endswith('.json'):
        config.save_to_file(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        config.export_to_excel(filepath)
    else:
        raise ValueError("Filepath must end with .json or .xlsx")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def update_regional_pricing(region: str = 'US', multiplier: float = 1.0):
    """
    Adjust all prices for regional differences.
    
    Parameters
    ----------
    region : str
        Region code (for documentation)
    multiplier : float
        Price adjustment factor (e.g., 1.2 for 20% higher)
    
    Examples
    --------
    >>> # Prices 30% higher in Europe
    >>> update_regional_pricing('EU', multiplier=1.3)
    """
    config = get_config()
    
    for platform_id in config.platforms:
        current_cost = config.platforms[platform_id]['cost_per_sample']
        config.platforms[platform_id]['cost_per_sample'] = int(current_cost * multiplier)
    
    for comp_id in config.cost_components:
        current_cost = config.cost_components[comp_id]['cost']
        config.cost_components[comp_id]['cost'] = int(current_cost * multiplier)


def get_platform_by_budget(max_cost_per_sample: float) -> pd.DataFrame:
    """
    Find platforms within budget.
    
    Parameters
    ----------
    max_cost_per_sample : float
        Maximum cost per sample
    
    Returns
    -------
    pd.DataFrame
        Platforms within budget
    """
    config = get_config()
    
    platforms = {
        k: v for k, v in config.platforms.items()
        if v['cost_per_sample'] <= max_cost_per_sample
    }
    
    return pd.DataFrame(platforms).T[['name', 'cost_per_sample', 'n_cpgs', 'coverage']]