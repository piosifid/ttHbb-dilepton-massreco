from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_nObj_eq, get_nObj_min, get_nObj_less, get_HLTsel, get_nBtagMin, get_nElectron, get_nMuon, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.histograms import *
#from params.custom_histograms import *
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.weights.common import common
import workflow
from workflow import ttHbb_Run3
from math import pi
import custom_function
import custom_cut
import mass_reco_functions
from custom_function import *
from custom_cut import *
import event_shapes
# ~ from params.binning import bins
# ~ from params.axis_settings import axis_settings
import os
localdir = os.path.dirname(os.path.abspath(__file__))
from pocket_coffea.lib.columns_manager import ColOut
from mass_reco_functions import *
# Loading default parameters
import custom_weights_DL
from custom_weights_DL import *
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

categories_dict = {
    "baseline_all" :[passthrough],
    "baseline_all_atleast3bjets" :[get_nObj_min(3, 15., "BJetGood")],
    "baseline_all_atleast4bjets" :[get_nObj_min(4, 15., "BJetGood")],
    "baseline_all_atleast2bjets": [get_nObj_min(2, 15., "BJetGood")],
}



year = "2024"
parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection.yaml",
                                                  f"{localdir}/params/hlt_triggers_Run3_DL.yaml",
                                                  f"{localdir}/params/variations.yaml",
                                                  f"{localdir}/params/jets_calibration_Run3_DL.yaml",
                                                  f"{localdir}/params/btagging_fixedWP_Run3_DL.yaml",
                                                  f"{localdir}/params/btagSF_calibration.yaml",
                                                  f"{localdir}/params/lepton_scale_factors_Run3_DL.yaml",
                                                  update=True)
parameters["run_period"] = "Run3"

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons":[
			f"{localdir}/datasets_24/Run3_DATA_2024.json",
            f"{localdir}/datasets_24/Run3_MC_2024_Ttbar.json",
            #f"{localdir}/datasets/Run3_MC_ttBkg.json", #notDesy
			#f"{localdir}/datasets/Run3_MC_otherBKG.json", #notDesy
		],
        "filter" : {
            "samples": [
				"DATA_EGamma",
				"DATA_Muon",
	  	        "DATA_MuonEG",
                
			#	"TTTo2L2Nu",
		    #
            #    "TTH_Hto2B",
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
             
    preselections = [dileptonic_presel, get_nObj_min(4, 15., "JetGood"), get_nObj_min(2, 15., "BJetGood")
                    ],
    
    categories = {
     **categories_dict,
        
    },
       weights_classes = common.common_weights,
       weights = {
        "common": {
            "inclusive": [ "genWeight",
                           "lumi",
                           "XS",
                           "pileup",

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
        "deltaRbb_min" : HistConf(
            [Axis(coll="events", field="deltaRbb_min", bins=50, start=0, stop=5,
                  label="$\Delta R_{bb}$", overflow=True, underflow=True)]
        ), 
        "mbb" : HistConf(
            [Axis(coll="events", field="mbb", bins=100, start=0, stop=1000,
                  label="mbb")]
        ),
        "npvsGood" : HistConf(
            [Axis(coll="PV", field="npvsGood", bins=100, start=0, stop=100,
                  label="$NGood_{vertices}$")]
        ),
        **count_hist(name="nJets", coll="JetGood",bins=10, start=2, stop=12),
        **count_hist(name="nBJets", coll="BJetGood",bins=14, start=0, stop=14),
        **count_hist(name="nLeptons", coll="LeptonGood",bins=3, start=0, stop=3),
        **lepton_hists(coll="LeptonGood",pos=0),
        **lepton_hists(coll="LeptonGood",pos=1),
        **jet_hists(name="bjet",coll="BJetGood", pos=0),
        **jet_hists(name="bjet",coll="BJetGood", pos=1),
        **jet_hists(name="jet",coll="JetGood", pos=0),
        **jet_hists(name="jet",coll="JetGood", pos=1),
        **met_hists(coll="PuppiMET"),
        **met_hists(coll="MET"),
        "PuppyMET" : HistConf(
            [
                Axis(coll="PuppiMET", field="pt", bins=[ 20, 30, 40, 60, 80, 100, 125, 150, 175, 200], start = 20, stop=200, label="$MET$ [GeV]", lim=(20,200))
            ]
        ),      
        "ht" : HistConf(
            [
                Axis(coll="events", field="JetGood_Ht", bins= 120, start = 0, stop=1200, label="$H_T$ [GeV]")
            ]
        ),
        "mbb_min" : HistConf(
            [Axis(coll="events", field="mbb_min", bins=100, start=0, stop=1000,
                  label="mbb_min", overflow=True, underflow=True)]
        ),
        "Max_Weight": HistConf(
            [Axis(coll="events", field="max_weight", pos=None, bins=100, start=0, stop=10000,
                  label="Maximum Weight per Event", overflow=True, underflow=True)]
        ),
        "Max_Weight_Higgs_Mass": HistConf(
            [Axis(coll="events", field="max_weight_higgs_mass", pos=None, bins=[0,20,40,60,80,100,120,140,160,180,200,230,260,300,350,425,500], start=0, stop=1000,
                  label="Max Weight Higgs Mass [GeV]", overflow=True, underflow=True)]
        ),   
        "Massreco_chosen_pair_higgs_mass": HistConf(
            [Axis(coll="events", field="massreco_chosen_pair_higgs_mass", pos=None, bins=100, start=0, stop=1000,
                  label="Max Weight Higgs Mass [GeV]", overflow=True, underflow=True)]
        ),
        "Massreco_chosen_pair_higgs_mass_1": HistConf(
            [Axis(coll="events", field="massreco_chosen_pair_higgs_mass", pos=None, bins=30, start=0, stop=500,
                  label="Max Weight Higgs Mass [GeV]", overflow=True, underflow=True)]
        ),
        "Massreco_chosen_pair_higgs_mass_2": HistConf(
            [Axis(coll="events", field="massreco_chosen_pair_higgs_mass", pos=None, bins=[0,20,40,60,80,100,120,140,160,180,210,240,270,310,350,390,430,500], start=0, stop=500,
                  label="Max Weight Higgs Mass [GeV]", overflow=True, underflow=True)]
        ),
        "Massreco_chosen_pair_higgs_mass_3": HistConf(
            [Axis(coll="events", field="massreco_chosen_pair_higgs_mass", pos=None, bins=[0,20,40,60,80,100,120,140,160,180,200,230,260,290,320,360,400,450,500], start=0, stop=500,
                  label="Max Weight Higgs Mass [GeV]", overflow=True, underflow=True)]
        ),
        "Massreco_chosen_pair_higgs_mass_4": HistConf(
            [Axis(coll="events", field="massreco_chosen_pair_higgs_mass", pos=None, bins=[0,20,40,60,80,100,120,140,160,180,200,230,260,300,350,425,500], start=0, stop=500,
                  label="Max Weight Higgs Mass [GeV]", overflow=True, underflow=True)]
        ),
        "Solutions_Number": HistConf(
        [Axis(coll="events", field="filtered_solutions_number", pos=None, bins=10, start=0, stop=10,
              label="Number of Solutions per Event", overflow=True, underflow=True)]
        ),
        "pt_jet2": HistConf(
            [Axis(coll="events", field="pt_jet2", bins=50, start=0, stop=500, label=r"$p_T(\mathrm{jet\,2})$ [GeV]")]
        ),
        "HT_events": HistConf(   # separate from your existing "ht"/JetGood_Ht
            [Axis(coll="events", field="HT", bins=60, start=0, stop=1500, label=r"$H_T$ [GeV]")]
        ),
        "dRjj_min": HistConf(
            [Axis(coll="events", field="dRjj_min", bins=30, start=0, stop=3, label=r"$\min(\Delta R_{jj})$")]
        ),
        "deta_max": HistConf(
            [Axis(coll="events", field="deta_max", bins=20, start=0, stop=5, label=r"$\max(|\Delta\eta|)$")]
        ),
        "m_higgs_like_jj": HistConf(
            [Axis(coll="events", field="m_higgs_like_jj", bins=60, start=0, stop=300, label=r"$m_{jj}^{\mathrm{H\text{-}like}}$ [GeV]")]
        ),
        "ptjj_at_dRmin": HistConf(
            [Axis(coll="events", field="ptjj_at_dRmin", bins=60, start=0, stop=600, label=r"$p_T^{jj}(\Delta R_{\min})$ [GeV]")]
        ),
        "Njj_higgs_like": HistConf(
            [Axis(coll="events", field="Njj_higgs_like", bins=10, start=0, stop=10, label=r"$N_{jj}^{100<m<140}$")]
        ),
        "m_jjj_maxpT": HistConf(
            [Axis(coll="events", field="m_jjj_maxpT", bins=35, start=0, stop=700, label=r"$m_{jjj}^{\max \sum p_T}$ [GeV]")]
        ), 
        "sum4_btag": HistConf(
            [Axis(coll="events", field="sum4_btag",
                  bins=80, start=0.0, stop=4.0,
                  label=r"$\sum_{\text{top 4}} b\text{-tag scores}$")]
        ),
        "dnn_score": HistConf(
            [Axis(coll="events", field="dnn_score", bins=100, start=0.0, stop=1.0, label="DNN score")]
        ),
        "C_jet": HistConf(
            [Axis(coll="events", field="C_jet",
                  bins=50, start=0.0, stop=1.0,
                  label=r"$C_{\mathrm{jet}}$")]
        ),
        "D_jet": HistConf(
            [Axis(coll="events", field="D_jet",
                  bins=50, start=0.0, stop=1.0,
                  label=r"$D_{\mathrm{jet}}$")]
        ),
        "Aplanarity": HistConf(
            [Axis(coll="events", field="Aplanarity",
                  bins=40, start=0.0, stop=0.5,
                  label=r"Aplanarity")]
        ),
        "H4": HistConf(
            [Axis(coll="events", field="H4",
                  bins=50, start=0.0, stop=1.0,
                  label=r"$H_4$ (Fox–Wolfram)")]
        ),
        "sum_pt_b1b2": HistConf(
            [Axis(coll="events", field="sum_pt_b1b2",
                  bins=100, start=0.0, stop=1000.0,
                  label=r"$p_{T}^{b_1} + p_{T}^{b_2}$ [GeV]")]
        ),

        "pt_ratio_b1_b2": HistConf(
            [Axis(coll="events", field="pt_ratio_b1_b2",
                  bins=80, start=0.0, stop=8.0,
                  label=r"$p_{T}^{b_1}/p_{T}^{b_2}$")]
        ),
        "mbb_btag_top2": HistConf(
            [Axis(coll="events", field="mbb_btag_top2",
                  bins=60, start=0.0, stop=300.0,
                  label=r"$m_{bb}^{\text{(top 2 b-tag)}}$ [GeV]")]
        ),
        "dR_b1_b2": HistConf(
            [Axis(coll="events", field="dR_b1_b2",
                  bins=50, start=0.0, stop=5.0,
                  label=r"$\Delta R(b_1, b_2)$")]
        ),

        "sum_m_jets": HistConf(
            [Axis(coll="events", field="sum_m_jets",
                  bins=60, start=0.0, stop=1200.0,
                  label=r"$\sum m_{\mathrm{jets}}$ [GeV]")]
        ),

        "centrality": HistConf(
            [Axis(coll="events", field="centrality",
                  bins=50, start=0.0, stop=1.5,
                  label=r"Event Centrality $\left(\frac{\sum p_T}{\sum E}\right)$")]
        ),

        "pT_higgs_like": HistConf(
            [Axis(coll="events", field="pT_higgs_like",
                  bins=60, start=0.0, stop=600.0,
                  label=r"$p_T^{\mathrm{H\text{-}like}}$ [GeV]")]
        ),

        "sum_pt_b1b2_over_HT": HistConf(
            [Axis(coll="events", field="sum_pt_b1b2_over_HT",
                  bins=40, start=0.0, stop=1.2,
                  label=r"$\frac{p_{T}^{b_1}+p_{T}^{b_2}}{H_T}$")]
        ),

        "MET_pt": HistConf(
            [Axis(coll="events", field="MET_pt",
                  bins=60, start=0.0, stop=300.0,
                  label=r"$MET_{\mathrm{Puppi}}$ [GeV]")]
        ),

        "MET_phi": HistConf(
            [Axis(coll="events", field="MET_phi",
                  bins=64, start=-3.2, stop=3.2,
                  label=r"$\phi(MET)$")]
        ),

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
        "chunk"          : 200000,
        "retries"        : 50,
        "treereduction"  : 20,
        "adapt"          : False,
        
    }
    
    
    
if "dask"  in run_options["executor"]:
    import cloudpickle
    cloudpickle.register_pickle_by_value(workflow)
    cloudpickle.register_pickle_by_value(custom_function)
    cloudpickle.register_pickle_by_value(custom_cut)    
    cloudpickle.register_pickle_by_value(mass_reco_functions)
    cloudpickle.register_pickle_by_value(event_shapes)
    cloudpickle.register_pickle_by_value(custom_weights_DL)