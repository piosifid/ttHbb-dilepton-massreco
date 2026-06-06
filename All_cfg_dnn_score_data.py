from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import (
    get_nObj_eq, get_nObj_min, get_nObj_less, get_HLTsel,
    get_nBtagMin, get_nElectron, get_nMuon, get_nPVgood,
    goldenJson, eventFlags
)
from pocket_coffea.parameters.histograms import *
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.weights.common import common
from math import pi
import os
import mass_reco_functions_mc
from mass_reco_functions_mc import *
import workflow_data
from workflow_data import ttHbb_Run3

import custom_function
import custom_cut
import event_shapes
import mass_reco_histograms_base
import mass_reco_histograms_dr_study

from custom_function import *
from custom_cut import *
from event_shapes import compute_event_shapes
from mass_reco_histograms_base import mass_reco_histograms_base as base_hists
from mass_reco_histograms_dr_study import mass_reco_histograms_dr_study as dr_hists

import custom_weights_DL
from custom_weights_DL import *

# NOTE: mass_reco_histograms_truth intentionally NOT imported for data

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

categories_dict = {
    "baseline_all":               [passthrough],
    "baseline_all_atleast2bjets": [get_nObj_min(2, 15., "BJetGood")],
    "baseline_all_atleast3bjets": [get_nObj_min(3, 15., "BJetGood")],
    "baseline_all_atleast4bjets": [get_nObj_min(4, 15., "BJetGood")],
}

year = "2024"
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/hlt_triggers_Run3_DL.yaml",
    f"{localdir}/params/variations.yaml",
    f"{localdir}/params/jets_calibration_Run3_DL.yaml",
    f"{localdir}/params/btagging_fixedWP_Run3_DL.yaml",
    f"{localdir}/params/btagSF_calibration.yaml",
    f"{localdir}/params/lepton_scale_factors_Run3_DL.yaml",
    update=True
)
parameters["run_period"] = "Run3"

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets_24/Run3_DATA_2024.json",
            f"{localdir}/datasets_24/Run3_MC_2024_Ttbar.json",
        ],
        "filter": {
            "samples": [
                "DATA_EGamma",
                "DATA_Muon",
                "DATA_MuonEG",
            ],
            "year": [year],
        },
        "subsamples": {
            "DATA_MuonEG": {"rmOverlap": [
                get_HLTsel(primaryDatasets=["MuonEG"])
            ]},
            "DATA_Muon": {"rmOverlap": [
                get_HLTsel(primaryDatasets=["DoubleMuon", "SingleMuon"]),
                get_HLTsel(primaryDatasets=["MuonEG"], invert=True)
            ]},
            "DATA_EGamma": {"rmOverlap": [
                get_HLTsel(primaryDatasets=["DoubleEle", "SingleEle"]),
                get_HLTsel(primaryDatasets=["MuonEG", "DoubleMuon", "SingleMuon"], invert=True)
            ]},
        },
    },

    workflow=ttHbb_Run3,
    workflow_options={
        "output": "parquet",
        "dump_columns_as_arrays_per_chunk": "./"
    },

    skim=[
        get_nPVgood(1),
        eventFlags,
        goldenJson,
        get_nObj_min(2, 15., "Jet"),
        get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon", "MuonEG", "DoubleEle", "DoubleMuon"])
    ],

    preselections=[
        dileptonic_presel,
        get_nObj_min(4, 15., "JetGood"),
        get_nObj_min(4, 15., "BJetGood")
    ],

    categories={**categories_dict},

    weights_classes=common.common_weights,
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
                "pileup",
            ],
            "bycategory": {}
        },
        "bysample": {
            "TTTo2L2Nu": {"inclusive": []},
            "TTToLNu2Q": {"inclusive": []},
            "TTTo4Q":    {"inclusive": []},
        }
    },

    variations={
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory": {}
            },
            "bysample": {
                "TTTo2L2Nu": {"inclusive": []},
                "TTToLNu2Q": {"inclusive": []},
                "TTTo4Q":    {"inclusive": []},
            }
        },
        "shape": {
            "common": {"inclusive": []}
        }
    },

    # =========================================================================
    # Variables — base + DR study only (no truth)
    # =========================================================================
    variables={
        **base_hists,
        **dr_hists,
        # Standard jet/lepton/MET hists from PocketCoffea
        **count_hist(name="nJets",    coll="JetGood",    bins=10, start=2,  stop=12),
        **count_hist(name="nBJets",   coll="BJetGood",   bins=14, start=0,  stop=14),
        **count_hist(name="nLeptons", coll="LeptonGood", bins=3,  start=0,  stop=3),
        **lepton_hists(coll="LeptonGood", pos=0),
        **lepton_hists(coll="LeptonGood", pos=1),
        **jet_hists(name="bjet", coll="BJetGood", pos=0),
        **jet_hists(name="bjet", coll="BJetGood", pos=1),
        **jet_hists(name="jet",  coll="JetGood",  pos=0),
        **jet_hists(name="jet",  coll="JetGood",  pos=1),
        **met_hists(coll="PuppiMET"),
        "npvsGood": HistConf([
            Axis(coll="PV", field="npvsGood", bins=100, start=0, stop=100,
                 label=r"$N_{\mathrm{good\,vertices}}$")
        ]),
        "PuppyMET": HistConf([
            Axis(coll="PuppiMET", field="pt",
                 bins=[20, 30, 40, 60, 80, 100, 125, 150, 175, 200],
                 start=20, stop=200, label=r"$MET$ [GeV]", lim=(20, 200))
        ]),
    }
)


run_options = {
    "executor":        "dask/lxplus",
    "env":             "singularity",
    "workers":         350,
    "scaleout":        100,
    "worker_image":    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest",
    "queue":           "longlunch",
    "walltime":        "02:00:00",
    "mem_per_worker":  "4GB",
    "disk_per_worker": "1GB",
    "exclusive":       False,
    "chunk":           200000,
    "retries":         50,
    "treereduction":   20,
    "adapt":           False,
}

if "dask" in run_options["executor"]:
    import cloudpickle
    cloudpickle.register_pickle_by_value(workflow_data)
    cloudpickle.register_pickle_by_value(custom_function)
    cloudpickle.register_pickle_by_value(custom_cut)
    cloudpickle.register_pickle_by_value(event_shapes)
    cloudpickle.register_pickle_by_value(custom_weights_DL)
    cloudpickle.register_pickle_by_value(mass_reco_histograms_base)
    cloudpickle.register_pickle_by_value(mass_reco_histograms_dr_study)
    cloudpickle.register_pickle_by_value(mass_reco_functions_mc)
    # NOTE: mass_reco_functions_mc imported via * in workflow_data — no separate registration needed