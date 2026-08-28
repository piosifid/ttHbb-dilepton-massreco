from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_nObj_eq, get_nObj_min, get_nObj_less, get_HLTsel, get_nBtagMin, get_nElectron, get_nMuon, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.histograms import *
#from params.custom_histograms import *
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.weights.common import common
import workflow_mc
from workflow_mc import ttHbb_Run3
from math import pi
import custom_function
import custom_cut
import mass_reco_functions_mc
from custom_function import *
from custom_cut import *
import event_shapes
# ~ from params.binning import bins
# ~ from params.axis_settings import axis_settings
import os
localdir = os.path.dirname(os.path.abspath(__file__))
from pocket_coffea.lib.columns_manager import ColOut
from mass_reco_functions_mc import *
# Loading default parameters
import custom_weights_DL
from custom_weights_DL import *
from pocket_coffea.parameters import defaults
import mass_reco_histograms_base
import mass_reco_histograms_truth
import mass_reco_histograms_dr_study
from mass_reco_histograms_base import mass_reco_histograms_base as base_hists
from mass_reco_histograms_truth import mass_reco_histograms_truth as truth_hists
from mass_reco_histograms_dr_study import mass_reco_histograms_dr_study as dr_hists
import dileptonic_presel_functions

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

categories_dict = {
    "baseline_all_atleast4bjets" :[get_nObj_min(4, 15., "BJetGood")],
}

year = "2023_preBPix"
parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection.yaml",
                                                  f"{localdir}/params/hlt_triggers_Run3_DL.yaml",
                                                  f"{localdir}/params/variations.yaml",
                                                  f"{localdir}/params/jets_calibration_Run3_DL.yaml",
                                                  f"{localdir}/params/btagging_fixedWP_Run3_DLPOL.yaml",
                                                  f"{localdir}/params/btagSF_calibration.yaml",
                                                  f"{localdir}/params/lepton_scale_factors_Run3_DL.yaml",
                                                  update=True)
parameters["has_higgs_truth_samples"] = ["TTH_Hto2B"]
parameters["run_period"] = "Run3"

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons":[
			f"{localdir}/datasets_24/files_skim_22_23_DL.json",
            f"{localdir}/datasets_24/files_skim_22_23_DL.json",
            #f"{localdir}/datasets/Run3_MC_ttBkg.json", #notDesy
			#f"{localdir}/datasets/Run3_MC_otherBKG.json", #notDesy
		],
        "filter" : {
            "samples": [
          #      "TTTo2L2Nu",
                "TTH_Hto2B",
            ],
       #     "samples_exclude" : [],
            "year": [year],
           
            # ~ "year": ['2022_preEE','2022_postEE','2023_preBPix','2023_postBPix']
        },
    

     "subsamples": {
            'DATA_MuonEG'  : {'rmOverlap' : [
                get_HLTsel(primaryDatasets=["MuonEG"])] #i.e. HLT_eleXX_muXX
            },
            'DATA_Muon'  : {'rmOverlap' : [
                get_HLTsel(primaryDatasets=["DoubleMuon", "SingleMuon"]), #i.e. HLT_muXX_muXX OR HLT_muXX
                # ~ get_HLTsel(primaryDatasets=["DoubleMuon"]), #i.e. HLT_muXX_muXX OR HLT_muXX
                get_HLTsel(primaryDatasets=["MuonEG"], invert=True)] #i.e. NOT selected by HLT_eleXX_muXX
            },
            'DATA_EGamma'  : {'rmOverlap' : [
                get_HLTsel(primaryDatasets=["DoubleEle", "SingleEle"]), #i.e. HLT_eleXX_eleXX OR HLT_eleXX
                # ~ get_HLTsel(primaryDatasets=["DoubleEle"]), #i.e. HLT_eleXX_eleXX OR HLT_eleXX
                get_HLTsel(primaryDatasets=["MuonEG", "DoubleMuon", "SingleMuon"], invert=True)] #i.e. NOT selected by one above
                # ~ get_HLTsel(primaryDatasets=["MuonEG", "DoubleMuon"], invert=True)] #i.e. NOT selected by one above
            },
            
        }
    },   
        
    workflow = ttHbb_Run3,
    workflow_options = {
            "output": "parquet",
            "dump_columns_as_arrays_per_chunk": "./"
    },
     # Skimming and categorization
    skim = [ get_nPVgood(1), eventFlags, goldenJson, get_nObj_min(2, 15., "Jet"),
             get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon", "MuonEG", "DoubleEle", "DoubleMuon"])
             ],
             
    preselections = [dileptonic_presel, get_nObj_min(4, 15., "JetGood"), get_nObj_min(4, 15., "BJetGood")
                    ],
    
    categories = {
     **categories_dict,
        
    },
       weights_classes = common.common_weights + [SF_btag_fixed_wp, SF_top_pt, SF_trigger_DL, SF_calibration_only_ttsplit_FixedWp],
       weights = {
        "common": {
            "inclusive": [# "genWeight",
                          # "lumi",
                         #  "XS",
                        #   "pileup",
                       #    "sf_ele_reco",
			#	  "sf_ele_id",
			#	  "sf_mu_id",
			#	  "sf_mu_iso",
            #      "sf_trigger_DL",
            #      "sf_btag_fixed_wp"

                          ],
            "bycategory" : {
            }
        },
         "bysample": {
	        "TTTo2L2Nu": {
        	    "inclusive": ["sf_top_pt"],
            },
            "TTToLNu2Q": {
                "inclusive": [],
            },
            "TTTo4Q": {
                "inclusive": [],
            }
        }
    },
    variations = {
        "weights": {
            "common": {
                "inclusive": [ 

                ],
                "bycategory" : {
                }
            },
            "bysample": {
            "TTTo2L2Nu": {
                "inclusive": [],
            },
            "TTToLNu2Q": {
                "inclusive": [],
            },
            "TTTo4Q": {
                "inclusive": [],
            }
          }
        },

        "shape": {
            "common":{
                "inclusive": [ ]
            }
        }
    },
    
    
variables = {
    **base_hists,
    **truth_hists,
#    **dr_hists,
}
)
 
 
run_options = {
        "executor"       : "dask/lxplus",
        "env"            : "singularity",
        "workers"        : 350,
        "scaleout"       : 100,
        "worker_image"   : "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest",
        "queue"          : "longlunch",
        "walltime"       : "02:00:00",
        "mem_per_worker" : "4GB", # GB
        "disk_per_worker" : "1GB", # GB
        "exclusive"      : False,
        "chunk"          : 10000,
        "retries"        : 50,
        "treereduction"  : 20,
        "adapt"          : False,
        
    }
    
    
    
if "dask"  in run_options["executor"]:
    import cloudpickle
    cloudpickle.register_pickle_by_value(workflow_mc)
    cloudpickle.register_pickle_by_value(custom_function)
    cloudpickle.register_pickle_by_value(custom_cut)    
    cloudpickle.register_pickle_by_value(mass_reco_functions_mc)
    cloudpickle.register_pickle_by_value(event_shapes)
    cloudpickle.register_pickle_by_value(custom_weights_DL)
    cloudpickle.register_pickle_by_value(mass_reco_histograms_base)
    cloudpickle.register_pickle_by_value(mass_reco_histograms_truth)
    cloudpickle.register_pickle_by_value(mass_reco_histograms_dr_study)
    cloudpickle.register_pickle_by_value(dileptonic_presel_functions)