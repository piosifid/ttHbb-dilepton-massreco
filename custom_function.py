from collections.abc import Iterable
import awkward as ak
import numpy as np
import correctionlib
from pocket_coffea.lib.triggers import get_trigger_mask_byprimarydataset
from pocket_coffea.lib.cut_functions import get_JetVetoMap_Mask
from pocket_coffea.lib.cut_definition import Cut
import math

import copy
import importlib
import gzip
import cloudpickle
from coffea.jetmet_tools import  CorrectedMETFactory
from pocket_coffea.lib.deltaR_matching import get_matching_pairs_indices, object_matching
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from parton import mkPDF
from numba import njit, float64
from mass_reco_functions import *



def get_dnn_score_min(threshold: float):
    """Return a Cut that keeps events with dnn_score >= threshold."""
    def _fn(events, params, **kwargs):
        if "dnn_score" not in ak.fields(events):
            raise RuntimeError(
                "[DNN] 'dnn_score' is missing — run the inference step in the workflow before categories."
            )
        thr = float(params["threshold"])
        return events["dnn_score"] >= thr

    return Cut(
        name=f"dnn_ge_{threshold:.2f}",
        params={"threshold": float(threshold)},
        function=_fn,
    )


def get_dnn_score_in(lo: float, hi: float):
    """Keep events with lo < dnn_score ≤ hi."""
    def _fn(events, params, **kwargs):
        if "dnn_score" not in ak.fields(events):
            raise RuntimeError(
                "[DNN] 'dnn_score' is missing — run the inference step in the workflow before categories."
            )
        s = events["dnn_score"]
        a = ak.to_numpy(s)
        if not np.isfinite(a).all():
            raise RuntimeError("[DNN] Non-finite values found in 'dnn_score'.")
        lo_v = float(params["lo"]); hi_v = float(params["hi"])
        return (s > lo_v) & (s <= hi_v)

    return Cut(
        name=f"dnn_in_{lo:.2f}_{hi:.2f}",
        params={"lo": float(lo), "hi": float(hi)},
        function=_fn,
    )
def dileptonic(events, params, year, processor_params, sample, isMC, **kwargs):
    # testing(processor_params, events, year)
    lepIds = abs(ak.mask(events.LeptonGood.pdgId, events.nLeptonGood == 2))

    is_em = ak.where(lepIds[:,0] + lepIds[:,1] == 24, True, False) #11 + 13 = 24
    is_em = is_em & get_trigger_mask_byprimarydataset(events, processor_params.HLT_triggers, year, isMC, 
                                     primaryDatasets=["MuonEG", "SingleMuon", "SingleEle"])
    
    is_mm = ak.where(lepIds[:,0] + lepIds[:,1] == 26, True, False) #13 + 13 = 26
    is_mm = is_mm & get_trigger_mask_byprimarydataset(events, processor_params.HLT_triggers, year, isMC, 
                                     primaryDatasets=["DoubleMuon", "SingleMuon"])
    
    is_ee = ak.where(lepIds[:,0] + lepIds[:,1] == 22, True, False) #11 + 11 = 22
    is_ee = is_ee & get_trigger_mask_byprimarydataset(events, processor_params.HLT_triggers, year, isMC, 
                                     primaryDatasets=["DoubleEle", "SingleEle"])
    
    if "dy_window" in params.keys():
        channel_cut = (is_em  
            | (is_mm & (
                (events.ll.mass > params["dy_window"]["stop"]) 
                | ((events.ll.mass > params["m_ee_mumu_min"])
                    & (events.ll.mass < params["dy_window"]["start"]))
            ))
            | (is_ee & (
                (events.ll.mass > params["dy_window"]["stop"]) 
                | ((events.ll.mass > params["m_ee_mumu_min"])
                    & (events.ll.mass < params["dy_window"]["start"]))
            ))
        )
    else:
        channel_cut = (is_em | is_mm | is_ee)    
    
    if "met" in params.keys():
        met_cut = ak.where((is_ee | is_mm), events.PuppiMET.pt > params["met"], True)
    else:
        met_cut = True

    if year in ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"]:
        mask_jetVetoMap = get_JetVetoMap_Mask(events, params, year, processor_params, sample, isMC, **kwargs)
    else:
        mask_jetVetoMap = True
    mask = (
        (events.nLeptonGood == 2)
        & (ak.firsts(events.LeptonGood.pt) >= params["pt_leading_lepton"])
        & (ak.mask(events.LeptonGood.pt>=params["pt_subleading_lepton"], ak.num(events.LeptonGood.pt)>=2)[:,1])
        & (ak.sum(events.LeptonGood.charge, axis=1) == 0)
        & (events.nJetGood >= params["njet"])
        & (events.nBJetGood >= params["nbjet"])
        & met_cut
        & mask_jetVetoMap
        & channel_cut
#        & ak.all(events.JetGood.btagRobustParTAK4B >= 0, axis=1)  # <-- Exclude events with any jet tag value < 0
    )
    return ak.where(ak.is_none(mask), False, mask)

def dileptonic_MaskNegTagVal(events, params, year, processor_params, sample, isMC, **kwargs):
    # testing(processor_params, events, year)
    lepIds = abs(ak.mask(events.LeptonGood.pdgId, events.nLeptonGood == 2))

    is_em = ak.where(lepIds[:,0] + lepIds[:,1] == 24, True, False) #11 + 13 = 24
    is_em = is_em & get_trigger_mask_byprimarydataset(events, processor_params.HLT_triggers, year, isMC, 
                                     primaryDatasets=["MuonEG", "SingleMuon", "SingleEle"])
    
    is_mm = ak.where(lepIds[:,0] + lepIds[:,1] == 26, True, False) #13 + 13 = 26
    is_mm = is_mm & get_trigger_mask_byprimarydataset(events, processor_params.HLT_triggers, year, isMC, 
                                     primaryDatasets=["DoubleMuon", "SingleMuon"])
    
    is_ee = ak.where(lepIds[:,0] + lepIds[:,1] == 22, True, False) #11 + 11 = 22
    is_ee = is_ee & get_trigger_mask_byprimarydataset(events, processor_params.HLT_triggers, year, isMC, 
                                     primaryDatasets=["DoubleEle", "SingleEle"])
    
    if "dy_window" in params.keys():
        channel_cut = (is_em  
            | (is_mm & (
                (events.ll.mass > params["dy_window"]["stop"]) 
                | ((events.ll.mass > params["m_ee_mumu_min"])
                    & (events.ll.mass < params["dy_window"]["start"]))
            ))
            | (is_ee & (
                (events.ll.mass > params["dy_window"]["stop"]) 
                | ((events.ll.mass > params["m_ee_mumu_min"])
                    & (events.ll.mass < params["dy_window"]["start"]))
            ))
        )
    else:
        channel_cut = (is_em | is_mm | is_ee)    
    
    if "met" in params.keys():
        met_cut = ak.where((is_ee | is_mm), events.PuppiMET.pt > params["met"], True)
    else:
        met_cut = True

    if year in ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"]:
        mask_jetVetoMap = get_JetVetoMap_Mask(events, params, year, processor_params, sample, isMC, **kwargs)
    else:
        mask_jetVetoMap = True
    mask = (
        (events.nLeptonGood == 2)
        & (ak.firsts(events.LeptonGood.pt) >= params["pt_leading_lepton"])
        & (ak.mask(events.LeptonGood.pt>=params["pt_subleading_lepton"], ak.num(events.LeptonGood.pt)>=2)[:,1])
        & (ak.sum(events.LeptonGood.charge, axis=1) == 0)
        & (events.nJetGood >= params["njet"])
        & (events.nBJetGood >= params["nbjet"])
        & met_cut
        & mask_jetVetoMap
        & channel_cut
        & ak.all(events.JetGood.btagRobustParTAK4B >= 0, axis=1)  # <-- Exclude events with any jet tag value < 0
    )
    return ak.where(ak.is_none(mask), False, mask)


##################  Custom preselection functions
def custom_dilepton(events, params, year, processor_params, **kwargs):
	if params["selec"] == "ee":
		SF = ((events.nMuonGood == 0) & (events.nElectronGood == 2))
		OS = events.ll.charge == 0

		mask = (
			(events.nLeptonGood == 2)
			& OS & SF
		)
		return ak.where(ak.is_none(mask), False, mask)
	elif params["selec"] == "em":
		SF = ((events.nMuonGood == 1) & (events.nElectronGood == 1))
		OS = events.ll.charge == 0

		mask = (
			(events.nLeptonGood == 2)
			& OS & SF
		)
		return ak.where(ak.is_none(mask), False, mask)
	elif params["selec"] == "mm":
		SF = ((events.nMuonGood == 2) & (events.nElectronGood == 0))
		OS = events.ll.charge == 0

		mask = (
			(events.nLeptonGood == 2)
			& OS & SF
		)
		return ak.where(ak.is_none(mask), False, mask)
	else:
		raise Exception("selection name not valid") 


def req_nLep_min(events, params, processor_params, year, isMC, **kwargs):
    leptons = ak.with_name(
        ak.concatenate((events.Muon, events.Electron), axis=1),
        name='PtEtaPhiMCandidate',
    )
    mask = ak.num(leptons.pt)>=params["nLep"]
    return ak.where(ak.is_none(mask), False, mask)
	

def ttJets_mask(events, params, processor_params, year, isMC, **kwargs):
	""" Categorization of ttbar events according to genTtbar ID, see reference here:
	 https://twiki.cern.ch/twiki/bin/view/CMSPublic/GenHFHadronMatcher#Event_categorization_example_1 """ 
	
	genTtbarId = events["genTtbarId"]
	if params["ttBId"] == "ttb":
		return ((abs(genTtbarId) % 100) == 51)
	elif params["ttBId"] == "tt2b":
		return ((abs(genTtbarId) % 100) == 52)
	elif params["ttBId"] == "ttbb":
		return (
			((abs(genTtbarId) % 100) == 53)
			| ((abs(genTtbarId) % 100) == 54)
			| ((abs(genTtbarId) % 100) == 55))
	elif params["ttBId"] == "ttB":
		return (((abs(genTtbarId) % 100) == 51)
			| ((abs(genTtbarId) % 100) == 52)
			| ((abs(genTtbarId) % 100) == 53)
			| ((abs(genTtbarId) % 100) == 54)
			| ((abs(genTtbarId) % 100) == 55))
	elif params["ttBId"] == "ttC":
		return (((abs(genTtbarId) % 100) == 41)
		    | ((abs(genTtbarId) % 100) == 42)
			| ((abs(genTtbarId) % 100) == 43)
			| ((abs(genTtbarId) % 100) == 44)
			| ((abs(genTtbarId) % 100) == 45))
	elif params["ttBId"] == "ttLF":
		return ((abs(genTtbarId) % 100) ==0)


# def ttJets_mask(events, params, processor_params, year, isMC, **kwargs):
# 	""" Categorization of ttbar events according to genTtbar ID, see reference here:
# 	 https://twiki.cern.ch/twiki/bin/view/CMSPublic/GenHFHadronMatcher#Event_categorization_example_1 """ 
	
# 	genTtbarId = events["genTtbarId"]
# 	if params["ttBId"] == "ttb":
# 		return ((abs(genTtbarId) % 100) == 51)
# 	elif params["ttBId"] == "tt2b":
# 		return ((abs(genTtbarId) % 100) == 52)
# 	elif params["ttBId"] == "ttbb":
# 		return (
# 			((abs(genTtbarId) % 100) == 53)
# 			| ((abs(genTtbarId) % 100) == 54)
# 			| ((abs(genTtbarId) % 100) == 55))
# 	elif params["ttBId"] == "ttC":
# 		return (
# 			((abs(genTtbarId) % 100) == 41)
# 		    | ((abs(genTtbarId) % 100) == 42)
# 			| ((abs(genTtbarId) % 100) == 43)
# 			| ((abs(genTtbarId) % 100) == 44)
# 			| ((abs(genTtbarId) % 100) == 45))
# 	elif params["ttBId"] == "ttLF":
# 		return ~(
# 			((abs(genTtbarId) % 100) == 41)
# 			| ((abs(genTtbarId) % 100) == 42)
# 			| ((abs(genTtbarId) % 100) == 43)
# 			| ((abs(genTtbarId) % 100) == 44)
# 			| ((abs(genTtbarId) % 100) == 45)
# 			| ((abs(genTtbarId) % 100) == 51)
# 			| ((abs(genTtbarId) % 100) == 52)
# 			| ((abs(genTtbarId) % 100) == 53)
# 			| ((abs(genTtbarId) % 100) == 54)
# 			| ((abs(genTtbarId) % 100) == 55))

def eq_genTtbarId_100(events, params, year, sample, **kwargs):
    """
    This function returns a mask for events where genTtbarId % 100 == params["genTtbarId"]
    or the logic OR of masks in the case in which params["genTtbarId"] is an iterable.
    The dictionary for genTtbarId % 100 is the following
    (taken from https://twiki.cern.ch/twiki/bin/view/CMSPublic/GenHFHadronMatcher, visited on 23.11.2023):
    0  : "tt+LF",
    41 : "tt+c",
    42 : "tt+2c",
    43 : "tt+cc",
    44 : "tt+c2c",
    45 : "tt+2c2c",
    46 : "tt+C",
    51 : "tt+b",
    52 : "tt+2b",
    53 : "tt+bb",
    54 : "tt+b2b",
    55 : "tt+2b2b",
    56 : "tt+B",
    """
    allowed_ids = [0, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56]
    if type(params["genTtbarId"]) == int:
        if params["genTtbarId"] in allowed_ids:
            return events.genTtbarId % 100 == params["genTtbarId"]
        else:
            raise Exception(f"The cut on genTtbarId % 100 must be an integer between 0 and 56.\nPossible choices:{allowed_ids}")
    elif isinstance(params["genTtbarId"], Iterable):
        mask = ak.zeros_like(events.event, dtype=bool)
        for _id in params["genTtbarId"]:
            if _id in allowed_ids:
                mask = mask | (events.genTtbarId % 100 == _id)
            else:
                raise Exception(f"The cut on genTtbarId % 100 must be an integer between 0 and 56.\nPossible choices:{allowed_ids}")
        return mask
    else:
        raise Exception(f'params["genTtbarId"] must be an integer or an iterable of integers between 0 and 56.\nPossible choices:{allowed_ids}')



def getJet_pt_cut(events, params, year, processor_params, **kwargs):
    mask = (
        ak.firsts(events.JetGood.pt>=70.)
        & 
        ak.firsts(events.JetGood.pt<=90.)
        )
    return ak.where(ak.is_none(mask), False, mask)
    
def getJet_eta_cut(events, params, year, processor_params, **kwargs):
    mask = (
        ak.firsts(events.JetGood.abseta>=0.2)
        & ak.firsts(events.JetGood.abseta<=0.4)
        )
    return ak.where(ak.is_none(mask), False, mask)



def custom_btagging(Jet, btag_algo, wp, year, params):
    #print('year: ',year)
    wp_value = params['btagging']['working_point'][year]['other_btagging_WPs'][btag_algo][wp]
    #print('wp value: ',wp_value)
    return Jet[Jet[btag_algo] > wp_value]
