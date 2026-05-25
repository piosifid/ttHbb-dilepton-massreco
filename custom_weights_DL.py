from pocket_coffea.lib.weights.weights import WeightWrapper, WeightLambda, WeightData, WeightDataMultiVariation
import numpy as np
import awkward as ak
import correctionlib
from dileptonic_presel_functions import eq_genTtbarId_100


SF_trigger_DL = WeightLambda.wrap_func(
    name="sf_trigger_DL",
    function=lambda params, metadata, events, size, shape_variations:
        get_sf_trigger_DL(events, params, metadata["year"]),
    has_variations=True
)

def get_sf_trigger_DL(events, params, year):
    leps = events["LeptonGood"]
    lep_flav = events.nMuonGood
    lep_leading_pt = leps[:, 0]["pt"]
    lep_trailing_pt = leps[:, 1]["pt"]

    params_trig_sf = params["lepton_scale_factors"]["trigger_sf_DL"]
    trig_sf_file = params_trig_sf["JSONfiles"][year]["file"]

    trig_sf_cset = correctionlib.CorrectionSet.from_file(trig_sf_file)

    trig_sfs_DL = trig_sf_cset[params_trig_sf["name"]].evaluate(lep_flav, lep_leading_pt, lep_trailing_pt)
    trig_sfs_DLUp = trig_sf_cset[params_trig_sf["name"] + "_totalUp"].evaluate(lep_flav, lep_leading_pt, lep_trailing_pt)
    trig_sfs_DLDown = trig_sf_cset[params_trig_sf["name"] + "_totalDown"].evaluate(lep_flav, lep_leading_pt, lep_trailing_pt)

    return trig_sfs_DL, trig_sfs_DLUp, trig_sfs_DLDown


SF_top_pt = WeightLambda.wrap_func(
    name="sf_top_pt",
    function=lambda params, metadata, events, size, shape_variations:
            get_sf_top_pt(events["GenPart"]),
    has_variations=True
    )

def get_sf_top_pt(part):
    part = part[~ak.is_none(part.parent, axis=1)]
    part = part[part.hasFlags("isLastCopy")]
    part = part[abs(part.pdgId) == 6]
    part = part[ak.argsort(part.pdgId, ascending=False)]
    arg = {
        "a": 0.103,
        "b": -0.0118, 
        "c": -0.000134,
        "d": 0.973
    }
    top_weight = arg["a"] * np.exp(arg["b"] * part.pt[:,0]) + arg["c"] * part.pt[:,0] + arg["d"]
    antitop_weight = arg["a"] * np.exp(arg["b"] * part.pt[:,1]) + arg["c"] * part.pt[:,1] + arg["d"]
    weight = np.sqrt(ak.prod([top_weight, antitop_weight], axis=0))
    return weight, np.ones(np.shape(weight)), ak.copy(weight)


class SF_btag_fixed_wp(WeightWrapper):
    name = "sf_btag_fixed_wp"
    has_variations = True
    
    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        self.params = params
        self.metadata = metadata
        self._variations = params["systematic_variations"]["weight_variations"]["sf_btag_fixed_wp"][metadata["year"]]["btag_variations"]
        
    def compute(self, events, size, shape_variation):
        
        if shape_variation == "nominal":
            nominal, variations, up_var, down_var = get_sf_btag_fixed_wp(
                self.params, 
                events.JetGood, 
                self.metadata["year"], 
                self.metadata["sample"], 
                return_variations=True
            )
            return WeightDataMultiVariation(
                name = self.name,
                nominal = nominal,
                variations = variations,
                up = up_var,
                down = down_var
            )
        else:
            return WeightData(
                name = self.name, 
                nominal = get_sf_btag_fixed_wp(
                    self.params, 
                    events.JetGood, 
                    self.metadata["year"], 
                    self.metadata["sample"], 
                    return_variations=False
                )
            )


def get_sf_btag_fixed_wp(params, Jets, year, sample, return_variations=True):

    sampleGroups = {
        "ttbarLike" :  [
            "TTTo2L2Nu",
            "TTToLNu2Q",
            "TTTo4Q",
            "TTH_Hto2B",
            "TTHtoNon2B",
            "TTLL",
            "TTNuNu",
            "TTLNu",
            "TTZ-ZtoQQ",
            "TWto2L2Nu"
        ],
        "dibosonLike" : [
            "ZZto2L2Nu",
            "WWto2L2Nu",
            "WZto3LNu"
        ],
        "VjetsLike" : [
            "DYJetsToLL",
            "WtoLNu+jets"
        ]
    }

    btag_effi_sample_group = ""
    for sampleGroupName, sampleGroup in sampleGroups.items():
        if sample in sampleGroup:
            btag_effi_sample_group = sampleGroupName
    if btag_effi_sample_group == "":
        print("WARNING: Sample does not correspond to one of the given sample groupings!")
    
    paramsBtagSf = params["jet_scale_factors"]["btagSF"][year]
    btag_algo = params["btagging"]["working_point"][year]["btagging_algorithm"]
    btag_wps = params["btagging"]["working_point"][year]["btagging_WP"]
    sf_file = paramsBtagSf["file"]
    btag_effi_file = paramsBtagSf["btagEfficiencyFile"][btag_algo]
    #Jets = events["JetGood"]

    btag_effi_corr_set = correctionlib.CorrectionSet.from_file(btag_effi_file)
    btag_sf_corr_set = correctionlib.CorrectionSet.from_file(sf_file)

    jetpt = ak.flatten(Jets["pt"])
    jeteta = ak.flatten(Jets["eta"])
    jetflav = ak.flatten(Jets["hadronFlavour"])
    jetcounts = ak.num(Jets["pt"])

    jetpt_heavy = ak.flatten(Jets["pt"][Jets["hadronFlavour"]>3])
    jeteta_heavy = ak.flatten(Jets["eta"][Jets["hadronFlavour"]>3])
    jetflav_heavy = ak.flatten(Jets["hadronFlavour"][Jets["hadronFlavour"]>3])

    jetpt_light = ak.flatten(Jets["pt"][Jets["hadronFlavour"]<4])
    jeteta_light = ak.flatten(Jets["eta"][Jets["hadronFlavour"]<4])
    jetflav_light = ak.flatten(Jets["hadronFlavour"][Jets["hadronFlavour"]<4])

    wp = params.object_preselection["Jet"]["btag"]["wp"]
    print('btag algo: ',btag_algo)
    print('wp: ',wp)
    effi_MC = ak.unflatten(
        btag_effi_corr_set[btag_effi_sample_group + "_wp_" + wp].evaluate(jetpt, np.abs(jeteta), jetflav), 
        counts=jetcounts
    )

    paramsBtagVar = params["systematic_variations"]["weight_variations"]["sf_btag_fixed_wp"][year]
    sf_btag_collection = paramsBtagVar["sf_collection_to_use"]
    sf_btag_lightFlavorName = paramsBtagVar["light_flavor_name"]
    btag_sf_name = paramsBtagSf["fixed_wp_name_map"][btag_algo]
    if return_variations:

        if sf_btag_collection not in ["comb", "mujets"]:
            print(
                "ERROR: type_sf_btag has to be either comb or mujets, "+\
                "see documentation here https://btv-wiki.docs.cern.ch"+\
                "/PerformanceCalibration/SFUncertaintiesAndCorrelations"+\
                "/#working-point-based-sfs-fixedwp-sfs")
        



        heavy_variations = paramsBtagVar[sf_btag_collection]
        light_variations = paramsBtagVar[sf_btag_lightFlavorName]

        variation_names = []
        if heavy_variations == None:
            heavyVarUp,  heavyVarDown = [], []
        if len(heavy_variations)<=1:
            heavyVarUp = ["up"]
            heavyVarDown = ["down"]
            variation_names.append("heavyFlavor")
        else: 
            heavyVarUp = ["up_"+var for var in heavy_variations]
            heavyVarDown = ["down_"+var for var in heavy_variations]
            variation_names += list(heavy_variations)

        if light_variations == None:
            lightVarUp,  lightVarDown = [], []
        if len(light_variations)<=1:
            lightVarUp = ["up"]
            lightVarDown = ["down"]
            variation_names.append("lightFlavor")
        else: 
            lightVarUp = ["up_"+var for var in light_variations]
            lightVarDown = ["down_"+var for var in light_variations]
            variation_names += ["lightFlavor_"+var for var in light_variations]
        
        variationsDict = {
            "central": {
                "light": ["central"],
                "heavy": ["central"],
                "btag_sf": []
            },
            "up": {
                "light": ["central" for var in heavyVarUp]+lightVarUp,
                "heavy": heavyVarUp+["central" for var in lightVarUp],
                "btag_sf": []
            },
            "down": {
                "light": ["central" for var in heavyVarDown]+lightVarDown,
                "heavy": heavyVarDown+["central" for var in lightVarDown],
                "btag_sf": []
            }
        }
    else:
        variationsDict = {
            "central": {
                "light": ["central"],
                "heavy": ["central"],
                "btag_sf": []
            }
        }

    for variationType, variationColl in variationsDict.items():
        for variation_light, variation_heavy in zip(variationColl["light"], variationColl["heavy"]):
            sf_light_flat = btag_sf_corr_set[btag_sf_name+"_"+sf_btag_lightFlavorName].evaluate(
                variation_light, 
                wp, 
                jetflav_light,
                np.abs(jeteta_light), 
                jetpt_light
            )
            sf_heavy_flat = btag_sf_corr_set[btag_sf_name+"_"+sf_btag_collection].evaluate(
                variation_heavy, 
                wp, 
                jetflav_heavy, 
                np.abs(jeteta_heavy), 
                jetpt_heavy
            )
            sf_flat = ak.to_numpy(ak.copy(jetpt))
            sf_flat[jetflav>3] = sf_heavy_flat
            sf_flat[jetflav<4] = sf_light_flat
            sf_DATA_MC = ak.unflatten(sf_flat, jetcounts)
            

            effi_DATA = ak.prod([sf_DATA_MC, effi_MC], axis=0)

            Jets = ak.with_field(Jets, effi_MC, "effi_MC_"+wp)
            Jets = ak.with_field(Jets, effi_DATA, "effi_DATA_"+wp)

            Jets_not_tagged = Jets[Jets[btag_algo] <= btag_wps[wp]]
            Jets_tagged = Jets[Jets[btag_algo] > btag_wps[wp]]

            p_MC_1 = 1 - Jets_not_tagged["effi_MC_"+wp]
            p_MC_2 = Jets_tagged["effi_MC_"+wp]

            p_DATA_1 = 1 - Jets_not_tagged["effi_DATA_"+wp]
            p_DATA_2 = Jets_tagged["effi_DATA_"+wp]

            p_MC = ak.concatenate([p_MC_1, p_MC_2], axis=1)
            p_MC = ak.prod(p_MC, axis=-1)
            p_DATA = ak.concatenate([p_DATA_1, p_DATA_2], axis=1)
            p_DATA = ak.prod(p_DATA, axis=-1)

            btag_sf_fixed_wp = np.divide(p_DATA, p_MC)

            variationsDict[variationType]["btag_sf"].append(btag_sf_fixed_wp)
    
    if return_variations:
        return (variationsDict["central"]["btag_sf"][0], 
                variation_names, 
                variationsDict["up"]["btag_sf"], 
                variationsDict["down"]["btag_sf"])
    return variationsDict["central"]["btag_sf"][0]
    
class SF_btag_fixed_wp_kinfit(WeightWrapper):
    name = "sf_btag_fixed_wp_kinfit"
    has_variations = True
    
    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        self.params = params
        self.metadata = metadata
        self._variations = params["systematic_variations"]["weight_variations"]["sf_btag_fixed_wp"][metadata["year"]]["btag_variations"]
        
    def compute(self, events, size, shape_variation):
        
        if shape_variation == "nominal":
            nominal, variations, up_var, down_var = get_sf_btag_fixed_wp_kinfit(
                self.params, 
                events.JetGood, 
                self.metadata["year"], 
                self.metadata["sample"], 
                return_variations=True
            )
            return WeightDataMultiVariation(
                name = self.name,
                nominal = nominal,
                variations = variations,
                up = up_var,
                down = down_var
            )
        else:
            return WeightData(
                name = self.name, 
                nominal = get_sf_btag_fixed_wp_kinfit(
                    self.params, 
                    events.JetGood, 
                    self.metadata["year"], 
                    self.metadata["sample"], 
                    return_variations=False
                )
            )


def get_sf_btag_fixed_wp_kinfit(params, Jets, year, sample, return_variations=True):
    
    sampleGroups = {
        "ttbarLike" :  [
            "TTTo2L2Nu",
            "TTToLNu2Q",
            "TTTo4Q",
            "TTH_Hto2B",
            "TTHtoNon2B",
            "TTLL",
            "TTNuNu",
            "TTLNu",
            "TTZ-ZtoQQ",
            "TWto2L2Nu"
        ],
        "dibosonLike" : [
            "ZZto2L2Nu",
            "WWto2L2Nu",
            "WZto3LNu"
        ],
        "VjetsLike" : [
            "DYJetsToLL",
            "WtoLNu+jets"
        ]
    }
    
    paramsBtagSf = params["jet_scale_factors"]["btagSF"][year]
    btag_algo = params["btagging"]["working_point"][year]["btagging_algorithm"]
    btag_wps = params["btagging"]["working_point"][year]["btagging_WP"]
    
    btag_effi_sample_group = ""
    for sampleGroupName, sampleGroup in sampleGroups.items():
        if sample in sampleGroup:
            btag_effi_sample_group = sampleGroupName
    if btag_effi_sample_group == "":
        raise Exception("WARNING: Sample does not correspond to one of the given sample groupings!")

    btag_effi_corr_set = correctionlib.CorrectionSet.from_file(paramsBtagSf["btagEfficiencyFile"][btag_algo])
    btag_sf_corr_set = correctionlib.CorrectionSet.from_file(paramsBtagSf["file"])
    # PRELIMINARY
    # CORRECTION ONLY APPLIED TO b JETS, NO c AND udsg!!!!!!!!!!!
    Jets = Jets[Jets["hadronFlavour"]>4]
    jetpt = ak.flatten(Jets["pt"])
    jeteta = ak.flatten(Jets["eta"])
    jetflav = ak.flatten(Jets["hadronFlavour"])
    jetcounts = ak.num(Jets["pt"])

    jetpt_heavy = ak.flatten(Jets["pt"][Jets["hadronFlavour"]>3])
    jeteta_heavy = ak.flatten(Jets["eta"][Jets["hadronFlavour"]>3])
    jetflav_heavy = ak.flatten(Jets["hadronFlavour"][Jets["hadronFlavour"]>3])


    wp = params.object_preselection["Jet"]["btag"]["wp"]
    effi_MC = ak.unflatten(
        btag_effi_corr_set[btag_effi_sample_group + "_wp_" + wp ].evaluate(jetpt, jeteta, jetflav), 
        counts=jetcounts
    )

    paramsBtagVar = params["systematic_variations"]["weight_variations"]["sf_btag_fixed_wp"][year]
    sf_btag_collection = paramsBtagVar["sf_collection_to_use"]
    if sf_btag_collection not in ["kinfit"]:
        raise Exception(
            "ERROR: type_sf_btag has to be kinfit, for this implementation. Otherwise use different CustomWeights object")
    
    btag_sf_name = paramsBtagSf["fixed_wp_name_map"][btag_algo]


    heavy_variations = paramsBtagVar[sf_btag_collection]

    if return_variations:
        variation_names = []
        if heavy_variations == None:
            heavyVarUp,  heavyVarDown = [], []
        if len(heavy_variations)<=1:
            heavyVarUp = ["up"]
            heavyVarDown = ["down"]
            variation_names.append("heavyFlavor")
        else: 
            heavyVarUp = ["up_"+var for var in heavy_variations]
            heavyVarDown = ["down_"+var for var in heavy_variations]
            variation_names += list(heavy_variations)

        
        variationsDict = {
            "central": {
                "light": ["central"],
                "heavy": ["central"],
                "btag_sf": []
            },
            "up": {
                "heavy": heavyVarUp,
                "btag_sf": []
            },
            "down": {
                "heavy": heavyVarDown,
                "btag_sf": []
            }
        }
    else:
        variationsDict = {
            "central": {
                "light": ["central"],
                "heavy": ["central"],
                "btag_sf": []
            }
        }

    for variationType, variationColl in variationsDict.items():
        for variation_heavy in variationColl["heavy"]:
            sf_heavy_flat = btag_sf_corr_set[btag_sf_name+"_"+sf_btag_collection].evaluate(
                variation_heavy, 
                wp, 
                jetflav_heavy, 
                abs(jeteta_heavy), 
                jetpt_heavy
            )
            sf_flat = ak.to_numpy(ak.copy(jetpt))
            sf_flat[jetflav>3] = sf_heavy_flat
            sf_flat[jetflav<4] = 1
            sf_DATA_MC = ak.unflatten(sf_flat, jetcounts)
            

            effi_DATA = ak.prod([sf_DATA_MC, effi_MC], axis=0)

            Jets = ak.with_field(Jets, effi_MC, "effi_MC_"+wp)
            Jets = ak.with_field(Jets, effi_DATA, "effi_DATA_"+wp)

            Jets_not_tagged = Jets[Jets[btag_algo] <= btag_wps[wp]]
            Jets_tagged = Jets[Jets[btag_algo] > btag_wps[wp]]

            p_MC_1 = 1 - Jets_not_tagged["effi_MC_"+wp]
            p_MC_2 = Jets_tagged["effi_MC_"+wp]

            p_DATA_1 = 1 - Jets_not_tagged["effi_DATA_"+wp]
            p_DATA_2 = Jets_tagged["effi_DATA_"+wp]

            p_MC = ak.concatenate([p_MC_1, p_MC_2], axis=1)
            p_MC = ak.prod(p_MC, axis=-1)
            p_DATA = ak.concatenate([p_DATA_1, p_DATA_2], axis=1)
            p_DATA = ak.prod(p_DATA, axis=-1)

            btag_sf_fixed_wp = np.divide(p_DATA, p_MC)

            variationsDict[variationType]["btag_sf"].append(btag_sf_fixed_wp)
    
    if return_variations:
        return (variationsDict["central"]["btag_sf"][0], 
                variation_names, 
                variationsDict["up"]["btag_sf"], 
                variationsDict["down"]["btag_sf"])
    return variationsDict["central"]["btag_sf"][0]


class SF_calibration_only_ttsplit_FixedWp(WeightWrapper):
    name = "sf_calibration_only_ttsplit_fixed_wp"
    has_variations = False  # No variations now
    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        self.jet_coll = params.jet_scale_factors.jet_collection.btag
    def compute(self, events, size, shape_variation=None):
        jetsHt = ak.sum(events[self.jet_coll].pt, axis=1)
        njets = events[f"n{self.jet_coll}"]

        out = sf_btag_calibration_only_ttsplit(
            events,
            self._params,
            sample=self._metadata["sample"],
            year=self._metadata["year"],
            njets=njets,
            jetsHt=jetsHt,
            SFmethod='FixedWp',
            variations=["central"]
        )
        return WeightData(
            name=self.name,
            nominal=out["central"][0]
        )

class SF_calibration_only_ttsplit_Shape(WeightWrapper):
    name = "sf_calibration_only_ttsplit_shape"
    has_variations = False  # No variations now
    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        self.jet_coll = params.jet_scale_factors.jet_collection.btag
    def compute(self, events, size, shape_variation=None):
        jetsHt = ak.sum(events[self.jet_coll].pt, axis=1)
        njets = events[f"n{self.jet_coll}"]

        out = sf_btag_calibration_only_ttsplit(
            events,
            self._params,
            sample=self._metadata["sample"],
            year=self._metadata["year"],
            njets=njets,
            jetsHt=jetsHt,
            SFmethod='Shape',
            variations=["central"]
        )
        return WeightData(
            name=self.name,
            nominal=out["central"][0]
        )
        
def sf_btag_calibration_only_ttsplit(events, params, sample, year, njets, jetsHt, SFmethod, variations=["central"]):
    from collections import defaultdict
    cset_calib = correctionlib.CorrectionSet.from_file(
        params.btagSF_calibration_ttsplit[year]["file" + SFmethod]
    )
    corr_calib = cset_calib[params.btagSF_calibration_ttsplit[year]["name"]]
    # Define masks for splitting ttbar into subcategories
    lf_mask = eq_genTtbarId_100(events, params={"genTtbarId": [0]}, year=year, sample=sample)
    cc_mask = eq_genTtbarId_100(events, params={"genTtbarId": [41, 42, 43, 44, 45, 46]}, year=year, sample=sample)
    bb_mask = eq_genTtbarId_100(events, params={"genTtbarId": [51, 52, 53, 54, 55, 56]}, year=year, sample=sample)

    subsamples = {"ttLF": lf_mask, "ttC": cc_mask, "ttB": bb_mask}
    output = {}
    for var in variations:
        if var == "central":
            output[var] = [
                sum(
                    mask * corr_calib.evaluate(f"{sample}__{subsample}", njets, jetsHt)
                    for subsample, mask in subsamples.items()
                )
            ]
    return output

