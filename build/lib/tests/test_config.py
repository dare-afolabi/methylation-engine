#!/usr/bin/env python
# coding: utf-8

"""
Test suite for the `config` module.

This collection of tests validates:
- Loading and parsing of configuration values
- Default values and expected constants
- Error handling for malformed or missing configuration
- Environment or runtime overrides (if applicable)

The tests are designed to achieve ≥ 80% coverage of `core.config`.

Usage:
    pytest -v --cov=core.config --cov-report=html
"""

import json
import os
from datetime import datetime

import pandas as pd
import pytest

from methylation_engine.core.config import (
    _global_config
    get_config
    DEFAULT_PLATFORMS
    PlannerConfig
    DEFAULT_DESIGNS
    load_config
    export_default_config
    update_regional_pricing
    get_platform_by_budget
)


@pytest.fixture(autouse=True)
def restore_global_config():
    """
    Make a copy of the global config before each test and restore after test.
    This prevents tests from interfering with each other by mutating the singleton.
    """
    original = cfg._global_config
    # Make a shallow clone of important dicts to restore reliably
    saved = {
        'platforms': dict(original.platforms),
        'designs': dict(original.designs),
        'cost_components': dict(original.cost_components),
        'timeline_phases': dict(original.timeline_phases)
    }
    yield
    # Restore
    original.platforms = dict(saved['platforms'])
    original.designs = dict(saved['designs'])
    original.cost_components = dict(saved['cost_components'])
    original.timeline_phases = dict(saved['timeline_phases'])


def test_get_config_returns_singleton():
    g1 = cfg.get_config()
    g2 = cfg.get_config()
    assert g1 is g2
    # EPIC default present
    assert 'EPIC' in g1.platforms
    assert g1.get_platform('EPIC')['cost_per_sample'] == cfg.DEFAULT_PLATFORMS['EPIC']['cost_per_sample']


def test_plannerconfig_save_and_load_json(tmp_path):
    p = tmp_path / "conf.json"
    conf = cfg.PlannerConfig()
    # change something
    conf.update_platform_cost('EPIC', 999)
    conf.save_to_file(str(p))
    # read back
    new = cfg.PlannerConfig()
    new.load_from_file(str(p))
    assert new.platforms['EPIC']['cost_per_sample'] == 999
    # last_updated present in file
    with open(str(p), 'r') as f:
        d = json.load(f)
    assert 'last_updated' in d
    # last_updated is iso-format parseable
    datetime.fromisoformat(d['last_updated'])


def test_load_from_file_updates_only_present_keys(tmp_path):
    # create partial config JSON
    p = tmp_path / "partial.json"
    partial = {'platforms': {'NEWPLAT': {'name': 'X', 'n_cpgs': 10, 'cost_per_sample': 10, 'processing_days': 1, 'coverage': 'X'}}}
    with open(p, 'w') as f:
        json.dump(partial, f)
    conf = cfg.PlannerConfig()
    assert 'NEWPLAT' not in conf.platforms
    conf.load_from_file(str(p))
    assert 'NEWPLAT' in conf.platforms
    assert conf.platforms['NEWPLAT']['name'] == 'X'


def test_load_from_excel_and_export_excel_roundtrip(tmp_path):
    # Build simple DataFrames for the expected sheets
    platforms_df = pd.DataFrame([{
        'platform_id': 'XPLAT',
        'name': 'XPlatform',
        'n_cpgs': 123
    }])
    designs_df = pd.DataFrame([{
        'design_id': 'XDES',
        'name': 'XDesign',
        'description': 'desc'
    }])
    costs_df = pd.DataFrame([{
        'component_id': 'XCOMP',
        'cost': 77,
        'unit': 'per_sample'
    }])
    timeline_df = pd.DataFrame([{
        'phase_id': 'XPH',
        'name': 'XPhase',
        'base_duration_days': 2
    }])

    excel_path = tmp_path / "cfg.xlsx"
    with pd.ExcelWriter(str(excel_path), engine='openpyxl') as w:
        platforms_df.to_excel(w, index=False, sheet_name='Platforms')
        designs_df.to_excel(w, index=False, sheet_name='Designs')
        costs_df.to_excel(w, index=False, sheet_name='Costs')
        timeline_df.to_excel(w, index=False, sheet_name='Timeline')

    conf = cfg.PlannerConfig()
    # ensure doesn't exist yet
    assert 'XPLAT' not in conf.platforms
    conf.load_from_excel(str(excel_path))
    assert 'XPLAT' in conf.platforms
    assert conf.platforms['XPLAT']['name'] == 'XPlatform'
    assert 'XDES' in conf.designs
    assert 'XCOMP' in conf.cost_components

    # export to excel and ensure file written and contains sheets
    out_path = tmp_path / "out.xlsx"
    conf.export_to_excel(str(out_path))
    assert out_path.exists()
    # confirm sheets exist by reading them
    xl = pd.ExcelFile(str(out_path))
    for s in ['Platforms', 'Designs', 'Costs', 'Timeline']:
        assert s in xl.sheet_names


def test_get_platform_default_when_missing():
    conf = cfg.PlannerConfig()
    # request missing platform
    result = conf.get_platform('DOESNOTEXIST')
    assert isinstance(result, dict)
    # default should be EPIC
    assert result['name'] == cfg.DEFAULT_PLATFORMS['EPIC']['name']


def test_get_design_default_when_missing():
    conf = cfg.PlannerConfig()
    result = conf.get_design('NOPE')
    assert isinstance(result, dict)
    assert result['name'] == cfg.DEFAULT_DESIGNS['two_group']['name']


def test_get_cost_components_filters_optional_and_applies_to():
    conf = cfg.PlannerConfig()
    # ensure there is a component that has applies_to (bisulfite_conversion)
    comps_all = conf.get_cost_components(platform=None, include_optional=True)
    assert 'bisulfite_conversion' in comps_all
    # when excluding optional, bisulfite_conversion should be missing
    comps_no_opt = conf.get_cost_components(include_optional=False)
    assert 'bisulfite_conversion' not in comps_no_opt
    # when filtering by platform not in applies_to, bisulfite_conversion should not be included
    comps_for_epic = conf.get_cost_components(platform='EPIC', include_optional=True)
    assert 'bisulfite_conversion' not in comps_for_epic
    # WGBS should include bisulfite_conversion
    comps_for_wgbs = conf.get_cost_components(platform='WGBS', include_optional=True)
    assert 'bisulfite_conversion' in comps_for_wgbs


def test_update_platform_cost_and_add_custom_platform_and_missing_field():
    conf = cfg.PlannerConfig()
    # update an existing platform
    conf.update_platform_cost('EPIC', 1234.5)
    assert conf.platforms['EPIC']['cost_per_sample'] == 1234.5

    # adding a valid custom platform
    new_info = {
        'name': 'CUSTOM',
        'n_cpgs': 10,
        'cost_per_sample': 5,
        'processing_days': 1,
        'coverage': 'test'
    }
    conf.add_custom_platform('CUSTOM1', new_info)
    assert 'CUSTOM1' in conf.platforms
    assert conf.platforms['CUSTOM1']['name'] == 'CUSTOM'

    # adding with missing required field should raise ValueError
    bad_info = {'name': 'BAD'}
    with pytest.raises(ValueError):
        conf.add_custom_platform('BAD1', bad_info)


def test_list_platforms_and_list_designs_formatting():
    conf = cfg.PlannerConfig()
    df = conf.list_platforms()
    # contains expected columns
    for c in ['name', 'n_cpgs', 'cost_per_sample']:
        assert c in df.columns
    # recommended_only filter
    df_rec = conf.list_platforms(recommended_only=True)
    # at least EPIC is recommended in defaults
    assert 'EPIC' in df_rec.index

    # designs
    ddf = conf.list_designs()
    for c in ['name', 'description', 'complexity', 'min_n_recommended', 'paired']:
        assert c in ddf.columns


def test_load_config_and_export_default_config_filetype_errors(tmp_path):
    # invalid extension should raise
    with pytest.raises(ValueError):
        cfg.load_config("badfile.txt")

    # create a json file and load
    conffile = tmp_path / "some.json"
    # write minimal json
    with open(conffile, 'w') as f:
        json.dump({'platforms': {}}, f)
    # should run without error
    cfg.load_config(str(conffile))

    # export_default_config invalid ext
    with pytest.raises(ValueError):
        cfg.export_default_config("badname.txt")

    # export_default_config JSON and XLSX should succeed
    out_json = tmp_path / "def.json"
    out_xlsx = tmp_path / "def.xlsx"
    cfg.export_default_config(str(out_json))
    assert out_json.exists()
    cfg.export_default_config(str(out_xlsx))
    assert out_xlsx.exists()


def test_update_regional_pricing_multiplies_and_int_casts():
    conf = cfg.get_config()
    # set known values
    conf.platforms['EPIC']['cost_per_sample'] = 200
    conf.cost_components['dna_extraction']['cost'] = 50
    cfg.update_regional_pricing(region='TEST', multiplier=1.3)
    # int cast expected: int(200*1.3) == 260
    assert conf.platforms['EPIC']['cost_per_sample'] == int(200 * 1.3)
    assert conf.cost_components['dna_extraction']['cost'] == int(50 * 1.3)


def test_get_platform_by_budget_filters_and_columns():
    conf = cfg.PlannerConfig()
    # ensure there is at least one cheap platform and one expensive
    conf.platforms['CHEAP'] = {
        'name': 'Cheapie',
        'cost_per_sample': 10,
        'n_cpgs': 1,
        'coverage': 'low'
    }
    df = cfg.get_platform_by_budget(50)
    assert 'Cheapie' in df['name'].values or 'CHEAP' in df.index
    # check columns
    for c in ['name', 'cost_per_sample', 'n_cpgs', 'coverage']:
        assert c in df.columns