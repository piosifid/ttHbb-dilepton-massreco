import awkward as ak
import numpy as np
import custom_function
from pocket_coffea.workflows.tthbb_base_processor import ttHbbBaseProcessor
from pocket_coffea.lib.deltaR_matching import metric_eta, metric_phi
from pocket_coffea.lib.deltaR_matching import object_matching
from pocket_coffea.lib.parton_provenance import get_partons_provenance_ttHbb, get_partons_provenance_ttbb4F, get_partons_provenance_tt5F
from pocket_coffea.lib.objects import (
    getGenLeptons,
    getGenJets,
    get_dilepton,
    get_dijet
)
#from custom_function import solve_ttbar_dilepton
from pocket_coffea.lib.objects import (
    btagging,
)
from custom_function import *
from mass_reco_functions import *
import event_shapes
from event_shapes import compute_event_shapes
import vector
vector.register_awkward()
import csv

def debug_bscore_raise(bscore):
    import awkward as ak
    import numpy as np

    mask_bad = ~ak.is_finite(bscore)

    # If all values are finite, nothing to debug
    if not ak.any(mask_bad):
        return

    # Find first problematic event
    evt = ak.where(ak.any(mask_bad, axis=1))[0][0]

    # Find indices of jets with NaN or inf
    bad_jets = ak.where(mask_bad[evt])[0]

    # Build debug message
    msg = []
    msg.append("\n[DEBUG] Non-finite bscore detected!")
    msg.append(f"Event index: {evt}")
    msg.append(f"Jet indices with NaN/inf: {bad_jets.tolist()}")
    msg.append(f"bscore values for event:\n{bscore[evt].tolist()}")

    # Sorted values
    idx_evt = ak.argsort(bscore[evt], ascending=False)
    sorted_evt = bscore[evt][idx_evt]

    msg.append(f"Sorted bscore values:\n{sorted_evt.tolist()}")
    msg.append(f"Top-4 sorted:\n{sorted_evt[:4].tolist()}")

    # Compute sum (will be nan)
    sum4 = ak.sum(sorted_evt[:4])
    msg.append(f"Sum(top-4): {sum4}")

    full_msg = "\n".join(msg)

    # Raise an exception so Dask returns this to your terminal
    raise RuntimeError(full_msg)

# ==== DNN (Step 1: constants + loaders) ======================================
# EXACT order must match what you used to train the network
DNN_FEATURES = [
    "pt_jet2",
    "HT",
    "dRjj_min",
    "deta_max",
    "m_higgs_like_jj",
    "ptjj_at_dRmin",
    "Njj_higgs_like",
    "m_jjj_maxpT",
    "sum4_btag",
    "dR_top2_btag",
    "ptjj_at_mH",
    "C_jet",
    "D_jet",
    "Aplanarity",
    "H4",
    "sum_pt_b1b2",
    "pt_ratio_b1_b2",
    "mbb_btag_top2",
    "sum_m_jets",
    "centrality",
   # "pT_higgs_like",
    "MET_pt",
]



# Clear, easy-to-edit paths
DNN_NORMS_PATH = "/eos/user/p/piosifid/PocketCoffea/ANN_newCoffea/DNN_new/output/the_one_for_the_Analysis_Note/variable_norm.csv"
DNN_MODEL_PATH = "/eos/user/p/piosifid/PocketCoffea/ANN_newCoffea/DNN_new/output/the_one_for_the_Analysis_Note/model_best.keras"

def _load_norms_csv(csv_path):
    """
    Return (mu, sigma) numpy arrays aligned with DNN_FEATURES.
    Accepts CSV 'var' either as exact name or with an 'events_' prefix.
    Still crashes if anything is missing or std==0.
    """

    # read CSV; normalize keys by stripping leading 'events_' if present
    lut = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            raw = row["var"]
            key = raw[7:] if raw.startswith("events_") else raw
            mu = float(row["mu"])
            sd = float(row["std"])
            lut[key] = (mu, sd)

    # build in EXACT DNN_FEATURES order
    mu, sg = [], []
    missing = [n for n in DNN_FEATURES if n not in lut]
    if missing:
        raise RuntimeError(f"[DNN] variable_norm.csv is missing features: {missing}")
    for name in DNN_FEATURES:
        m, s = lut[name]
        if s == 0.0:
            raise RuntimeError(f"[DNN] std==0 for feature '{name}' in variable_norm.csv (cannot standardize).")
        mu.append(m); sg.append(s)

    return np.array(mu, dtype=np.float64), np.array(sg, dtype=np.float64)

def _load_tf_model(model_path):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)
# =============================================================================



class ttHbb_Run3(ttHbbBaseProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.isRun3 = True if self.params["run_period"]=='Run3' else False
         # --- Step 1: load model + norms once per worker (will raise if broken) ---
        self._dnn_mu, self._dnn_sigma = _load_norms_csv(DNN_NORMS_PATH)
        self._dnn_model = _load_tf_model(DNN_MODEL_PATH)

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation=variation)
        self.events["ll"] = get_dilepton(
            self.events.ElectronGood, self.events.MuonGood
        )
        ##self.events["MyGenLeptons"] = getGenLeptons(
          #  self.events, "Both", self.params
        #)
        #self.events["MyGenLeptons","charge"] = np.sign(self.events["MyGenLeptons"]["pdgId"])

        #self.events["ll"] = get_dilepton(self.events.MyGenLeptons, None)

        #self.events["MyGenJets"] =  getGenJets(self.events, "MyGenLeptons", self.params)
        #self.events["MyGenJets", "pdgId"] = self.events.MyGenJets.partonFlavour
        
        #self.events["dijet"] = get_dijet(self.events.MyGenJets, tagger=None)
    def get_partons_provenance_ttHbb_dileptonic(self, pdgIds, array_builder):
      """
      This function assigns particle provenance (origin) for b-quarks in a dileptonic ttH -> bb process,
      where the Higgs decays into two b-quarks, and both the top and anti-top quarks decay, producing
      additional b-quarks.

      1 = higgs bquarks,
      2 = top bquark,
      3 = antitop bquark,
      4 = additional radiation (if present)
      """

      for ids in pdgIds:
        from_part = [-1] * max(4, len(ids))
        if len(ids) == 5:
            offset = 1
            from_part[0] = 4
        else:
            offset = 0

        if len(ids) == 4 or len(ids) == 5:
            if ids[0 + offset] == 5:
                from_part[0 + offset] = 2
            if ids[1 + offset] == -5:
                from_part[1 + offset] = 3

            from_part[2 + offset] = 1
            from_part[3 + offset] = 1
        else:

            from_part[0 + offset] = 2
            from_part[1 + offset] = 3
            from_part[2 + offset] = 1
            from_part[3 + offset] = 1

        array_builder.begin_list()
        for i in from_part:
            array_builder.append(i)
        array_builder.end_list()
      return array_builder

    def define_common_variables_after_presel(self, variation):
        super().define_common_variables_before_presel(variation=variation)
        # ~ print(self.events["BJetGood_L"]["btagDeepFlavB"][1][0:50])
        # ~ print(ak.to_list(self.events["BJetGood_M"]["btagDeepFlavB"][:50]))
        # ~ print(ak.to_list(self.events["BJetGood_L"]["btagDeepFlavB"][:50]))
        # ~ print(self.events["BJetGood_L"])
        # List all keys in the self.events array
        #available_keys = ak.fields(self.events["event"])
        
        #genpart_fields = ak.fields(self.events["GenPart"])
        #print("Available fields in GenPart:", genpart_fields)
        # Compute deltaR(b, b) of all possible b-jet pairs.
        # We require deltaR > 0 to exclude the deltaR between the jets with themselves
        # print(ak.fields(self.events))
        # print(ak.to_list(self.events["event"]))
        deltaR = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"]), axis=2)
        deltaEta = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"], metric=metric_eta), axis=2)
        deltaPhi = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"], metric=metric_phi), axis=2)
        deltaR = deltaR[deltaR > 0.]
        deltaEta = deltaEta[deltaEta > 0.]
        deltaPhi = deltaPhi[deltaPhi > 0.]
        
        # Get the deltaR with no possibility of repetition of identical b-jet pairs

        # Get all the possible combinations of b-jet pairs
        pairs = ak.argcombinations(self.events["BJetGood"], 2, axis=1)
        b1 = self.events["BJetGood"][pairs.slot0]
        b2 = self.events["BJetGood"][pairs.slot1]

        # Compute deltaR between the pairs
        deltaR_unique = b1.delta_r(b2)
        idx_pairs_sorted = ak.argsort(deltaR_unique, axis=1)
        pairs_sorted = pairs[idx_pairs_sorted]

        # Compute the minimum deltaR(b, b), deltaEta(b, b), deltaPhi(b, b) and the invariant mass of the closest b-jet pair
        self.events["deltaRbb_min"] = ak.min(deltaR, axis=1)
        self.events["deltaEtabb_min"] = ak.min(deltaEta, axis=1)
        self.events["deltaPhibb_min"] = ak.min(deltaPhi, axis=1)
        # Map numeric provenance labels to names for clarity
        self.events["mbb"] = (self.events["BJetGood"][pairs_sorted.slot0] + self.events["BJetGood"][pairs_sorted.slot1]).mass
        self.events["mbb_min"] = ak.firsts(self.events["mbb"])
        self.events["mass_w"] = 80
        self.events["mass_top"] = 170
        self.events["mass_w_minus"] = self.events["mass_w"]
        self.events["mass_antitop"] = self.events["mass_top"]
        # Assign the first b-jet to "1st_jet"
        '''
        self.events["whatever"] = ak.firsts(self.events["JetGood"])
        
        self.events["1st_jet"] = ak.firsts(self.events["JetGood"])

        # Assign the second b-jet to "2nd_jet", if it exists
        self.events["2nd_jet"] = ak.firsts(self.events["JetGood"][:, 1:])

        # Assign the third b-jet to "3rd_jet", if it exists
        self.events["3rd_jet"] = ak.firsts(self.events["JetGood"][:, 2:])

        # Assign the fourth b-jet to "4th_jet", if it exists
        self.events["4th_jet"] = ak.firsts(self.events["JetGood"][:, 3:])
        self.events["5th_jet"] = ak.firsts(self.events["JetGood"][:, 4:])
        
        '''        
        # Compute raw pt
        jet_pt_raw = self.events["JetGood"]["pt"] * (1 - self.events["JetGood"]["rawFactor"])

        # Compute regressed pt
        jet_pt_regressed = (
            jet_pt_raw *
            self.events["JetGood"]["PNetRegPtRawCorr"] *
            self.events["JetGood"]["PNetRegPtRawCorrNeutrino"]
        )
        # Now store this as a new field in JetGood (this will be a "virtual" field inside awkward array)
        self.events["JetGood"] = ak.with_field(self.events["JetGood"], jet_pt_regressed, "pt_regressed")
        # Get the indices that would sort JetGood by btag score (descending)

        # Get the indices that would sort JetGood by btag score (descending)
        btag_sorted_idx = ak.argsort(self.events["JetGood"]["btagUParTAK4B"], ascending=False)

        # Use these indices to get the sorted jets
        sorted_jets = self.events["JetGood"][btag_sorted_idx]

        # Store the original indices of the sorted jets
        sorted_jet_orig_idx = btag_sorted_idx

        # Assign top 5 jets and their original indices
        self.events["1st_jet"] = ak.firsts(sorted_jets[:, 0:1])
        self.events["2nd_jet"] = ak.firsts(sorted_jets[:, 1:2])
        self.events["3rd_jet"] = ak.firsts(sorted_jets[:, 2:3])
        self.events["4th_jet"] = ak.firsts(sorted_jets[:, 3:4])
        self.events["5th_jet"] = ak.firsts(sorted_jets[:, 4:5])

        # Also keep original indices
        self.events["1st_jet_idx"] = ak.firsts(sorted_jet_orig_idx[:, 0:1])
        self.events["2nd_jet_idx"] = ak.firsts(sorted_jet_orig_idx[:, 1:2])
        self.events["3rd_jet_idx"] = ak.firsts(sorted_jet_orig_idx[:, 2:3])
        self.events["4th_jet_idx"] = ak.firsts(sorted_jet_orig_idx[:, 3:4])
        self.events["5th_jet_idx"] = ak.firsts(sorted_jet_orig_idx[:, 4:5])
        
        import vector
        vector.register_awkward()
        bjet1 = ak.zip({
             "pt": self.events["1st_jet"]["pt_regressed"],
             "eta": self.events["1st_jet"]["eta"],
             "phi": self.events["1st_jet"]["phi"],
             "mass": self.events["1st_jet"]["mass"]
        }, with_name="Momentum4D")
        #print(bjet1.behavior)
        # Store px, py, pz, and E into the self.events structure
        self.events["1st_jet"] = ak.with_field(self.events["1st_jet"], bjet1.px, "px")
        self.events["1st_jet"] = ak.with_field(self.events["1st_jet"], bjet1.py, "py")
        self.events["1st_jet"] = ak.with_field(self.events["1st_jet"], bjet1.pz, "pz")
        self.events["1st_jet"] = ak.with_field(self.events["1st_jet"], bjet1.E, "E")

      #  print("1st_jet components:")
     #   print("E:", ak.to_list(self.events["1st_jet"]["E"]))


        # Similarly for the 2nd b-jet
        bjet2 = ak.zip({
             "pt": self.events["2nd_jet"]["pt_regressed"],
             "eta": self.events["2nd_jet"]["eta"],
             "phi": self.events["2nd_jet"]["phi"],
             "mass": self.events["2nd_jet"]["mass"]
        }, with_name="Momentum4D")

        self.events["2nd_jet"] = ak.with_field(self.events["2nd_jet"], bjet2.px, "px")
        self.events["2nd_jet"] = ak.with_field(self.events["2nd_jet"], bjet2.py, "py")
        self.events["2nd_jet"] = ak.with_field(self.events["2nd_jet"], bjet2.pz, "pz")
        self.events["2nd_jet"] = ak.with_field(self.events["2nd_jet"], bjet2.E, "E")


        
        bjet3 = ak.zip({
             "pt": self.events["3rd_jet"]["pt_regressed"],
             "eta": self.events["3rd_jet"]["eta"],
             "phi": self.events["3rd_jet"]["phi"],
             "mass": self.events["3rd_jet"]["mass"]
        }, with_name="Momentum4D")

        self.events["3rd_jet"] = ak.with_field(self.events["3rd_jet"], bjet3.px, "px")
        self.events["3rd_jet"] = ak.with_field(self.events["3rd_jet"], bjet3.py, "py")
        self.events["3rd_jet"] = ak.with_field(self.events["3rd_jet"], bjet3.pz, "pz")
        self.events["3rd_jet"] = ak.with_field(self.events["3rd_jet"], bjet3.E, "E")
        # Convert and store the four-momentum subfields for the 4th b-jet (2nd other b-jet)
        bjet4 = ak.zip({
             "pt": self.events["4th_jet"]["pt_regressed"],
             "eta": self.events["4th_jet"]["eta"],
             "phi": self.events["4th_jet"]["phi"],
             "mass": self.events["4th_jet"]["mass"]
        }, with_name="Momentum4D")

        self.events["4th_jet"] = ak.with_field(self.events["4th_jet"], bjet4.px, "px")
        self.events["4th_jet"] = ak.with_field(self.events["4th_jet"], bjet4.py, "py")
        self.events["4th_jet"] = ak.with_field(self.events["4th_jet"], bjet4.pz, "pz")
        self.events["4th_jet"] = ak.with_field(self.events["4th_jet"], bjet4.E, "E")

        bjet5 = ak.zip({
             "pt": self.events["5th_jet"]["pt_regressed"],
             "eta": self.events["5th_jet"]["eta"],
             "phi": self.events["5th_jet"]["phi"],
             "mass": self.events["5th_jet"]["mass"]
        }, with_name="Momentum4D")

        self.events["5th_jet"] = ak.with_field(self.events["5th_jet"], bjet5.px, "px")
        self.events["5th_jet"] = ak.with_field(self.events["5th_jet"], bjet5.py, "py")
        self.events["5th_jet"] = ak.with_field(self.events["5th_jet"], bjet5.pz, "pz")
        self.events["5th_jet"] = ak.with_field(self.events["5th_jet"], bjet5.E, "E")
        
        # Filter the positive leptons (charge == +1)
        lepton_pos_mask = self.events["LeptonGood"]["charge"] == 1
        self.events["lepton_pos"] = self.events["LeptonGood"][lepton_pos_mask]

        # Filter the negative leptons (charge == -1)
        lepton_neg_mask = self.events["LeptonGood"]["charge"] == -1
        self.events["lepton_neg"] = self.events["LeptonGood"][lepton_neg_mask]
        
        # Convert and store four-momentum for the positively charged leptons
        lepton_pos = ak.zip({
             "pt": self.events["lepton_pos"]["pt"],
             "eta": self.events["lepton_pos"]["eta"],
             "phi": self.events["lepton_pos"]["phi"],
             "mass": self.events["lepton_pos"]["mass"]
        }, with_name="Momentum4D")

        self.events["lepton_pos"] = ak.with_field(self.events["lepton_pos"], lepton_pos.px, "px")
        self.events["lepton_pos"] = ak.with_field(self.events["lepton_pos"], lepton_pos.py, "py")
        self.events["lepton_pos"] = ak.with_field(self.events["lepton_pos"], lepton_pos.pz, "pz")
        self.events["lepton_pos"] = ak.with_field(self.events["lepton_pos"], lepton_pos.E, "E")

        # Convert and store four-momentum for the negatively charged leptons
        lepton_neg = ak.zip({
             "pt": self.events["lepton_neg"]["pt"],
             "eta": self.events["lepton_neg"]["eta"],
             "phi": self.events["lepton_neg"]["phi"],
             "mass": self.events["lepton_neg"]["mass"]
        }, with_name="Momentum4D")

        self.events["lepton_neg"] = ak.with_field(self.events["lepton_neg"], lepton_neg.px, "px")
        self.events["lepton_neg"] = ak.with_field(self.events["lepton_neg"], lepton_neg.py, "py")
        self.events["lepton_neg"] = ak.with_field(self.events["lepton_neg"], lepton_neg.pz, "pz")
        self.events["lepton_neg"] = ak.with_field(self.events["lepton_neg"], lepton_neg.E, "E")
        #print(ak.to_list(self.events["lepton_neg"]["px"]))
        self.events["MET_few"] = ak.Array({
                             "pt": self.events["PuppiMET"]["pt"],
                             "phi": self.events["PuppiMET"]["phi"]
                             })
        '''
        # Create an array of the four leading jets
        leading_jets = ak.zip({
            "pt": ak.concatenate([
                self.events["1st_jet"]["pt"],
                self.events["2nd_jet"]["pt"],
                self.events["3rd_jet"]["pt"],
                self.events["4th_jet"]["pt"]
            ], axis=1),
            "eta": ak.concatenate([
                self.events["1st_jet"]["eta"],
                self.events["2nd_jet"]["eta"],
                self.events["3rd_jet"]["eta"],
                self.events["4th_jet"]["eta"]
            ], axis=1),
            "phi": ak.concatenate([
                self.events["1st_jet"]["phi"],
                self.events["2nd_jet"]["phi"],
                self.events["3rd_jet"]["phi"],
                self.events["4th_jet"]["phi"]
            ], axis=1),
            "mass": ak.concatenate([
                self.events["1st_jet"]["mass"],
                self.events["2nd_jet"]["mass"],
                self.events["3rd_jet"]["mass"],
                self.events["4th_jet"]["mass"]
            ], axis=1),
        }, with_name="Momentum4D")
        '''
        # Get index combinations of lepton_pos and BJetGood
        lep_bjet_idx_pos = ak.argcartesian([self.events["lepton_pos"], self.events["BJetGood"]], axis=1)
        lep_pos = self.events["lepton_pos"][lep_bjet_idx_pos["0"]]
        bjet = self.events["BJetGood"][lep_bjet_idx_pos["1"]]
        dr_pos = lep_pos.delta_r(bjet)

        # Min ΔR for pos leptons
        min_dr_pos = ak.min(dr_pos, axis=1)

        # Same for negative leptons
        lep_bjet_idx_neg = ak.argcartesian([self.events["lepton_neg"], self.events["BJetGood"]], axis=1)
        lep_neg = self.events["lepton_neg"][lep_bjet_idx_neg["0"]]
        bjet_neg = self.events["BJetGood"][lep_bjet_idx_neg["1"]]
        dr_neg = lep_neg.delta_r(bjet_neg)

        # Min ΔR for neg leptons
        min_dr_neg = ak.min(dr_neg, axis=1)

        # Save to self.events
        self.events["min_dr_lepton_pos_bjets"] = min_dr_pos
        self.events["min_dr_lepton_neg_bjets"] = min_dr_neg

        # Combine both
        self.events["min_dr_any_lepton_bjets"] = ak.min(
            ak.concatenate([min_dr_pos[..., None], min_dr_neg[..., None]], axis=1),
            axis=1
        )
        # --- Easy & Medium ANN variables (Awkward-safe version) ---
        jets = self.events["JetGood"]
        
        shapes = compute_event_shapes(jets)

        self.events["C_jet"] = shapes["C_jet"]
        self.events["D_jet"] = shapes["D_jet"]
        self.events["Aplanarity"] = shapes["Aplanarity"]
        self.events["H4"] = shapes["H4"]


        # pick which pT to use (regressed if available)
        if "pt_regressed" in ak.fields(jets):
            ptj = ak.values_astype(jets.pt_regressed, np.float64)
        else:
            ptj = ak.values_astype(jets.pt, np.float64)

        # 1) pT(jet 2) — second-leading jet pT
        idx_ptdesc = ak.argsort(ptj, axis=1, ascending=False)
        jets_ptsorted = jets[idx_ptdesc]
        pt_sorted = ptj[idx_ptdesc]
        self.events["pt_jet2"] = ak.fill_none(ak.firsts(pt_sorted[:, 1:2]), np.nan)

        # 2) HT — scalar sum of jet pT
        self.events["HT"] = ak.values_astype(ak.sum(ptj, axis=1), np.float64)

        # Build all jet pairs (i < j)
        pairs = ak.combinations(jets, 2, fields=["j1", "j2"])
        has_ge2 = ak.num(jets) >= 2

        # 3) ΔRjj(min) and 4) Δη(max)
        dR_pairs = pairs.j1.delta_r(pairs.j2)
        dEta_pairs = abs(pairs.j1.eta - pairs.j2.eta)

        self.events["dRjj_min"] = ak.values_astype(
            ak.where(has_ge2, ak.min(dR_pairs, axis=1), np.nan), np.float64
        )
        self.events["deta_max"] = ak.values_astype(
            ak.where(has_ge2, ak.max(dEta_pairs, axis=1), np.nan), np.float64
        )

        # Compute mjj and ptjj for pairs
        mjj = (pairs.j1 + pairs.j2).mass
        ptjj = ak.values_astype(pairs.j1.pt + pairs.j2.pt, np.float64)

        # 5) m_higgs_like_jj — mass of pair closest to 125 GeV
        idx_closest = ak.argmin(abs(mjj - 125.0), axis=1, keepdims=True)
        self.events["m_higgs_like_jj"] = ak.values_astype(
            ak.where(has_ge2, ak.firsts(mjj[idx_closest]), np.nan), np.float64
        )

        # 6) pT_jj at ΔR(min) — sum pT of closest-ΔR pair
        idx_drmin = ak.argmin(dR_pairs, axis=1, keepdims=True)
        self.events["ptjj_at_dRmin"] = ak.values_astype(
            ak.where(has_ge2, ak.firsts(ptjj[idx_drmin]), np.nan), np.float64
        )

        # 7) N_higgs_like_jj — number of dijet pairs in 100 < m_jj < 140 GeV window
        in_win = (mjj > 100.0) & (mjj < 140.0)
        self.events["Njj_higgs_like"] = ak.values_astype(
            ak.where(has_ge2, ak.sum(in_win, axis=1), 0), np.int32
        )

        # 8) m_jjj^(max pT) — tri-jet mass of combination with largest scalar pT
        has_ge3 = ak.num(jets) >= 3
        trips = ak.combinations(jets, 3, fields=["j1", "j2", "j3"])
        ptsum3 = ak.values_astype(
            trips.j1.pt + trips.j2.pt + trips.j3.pt, np.float64
        )
        mjjj = (trips.j1 + trips.j2 + trips.j3).mass
        idx_maxpt3 = ak.argmax(ptsum3, axis=1, keepdims=True)
        self.events["m_jjj_maxpT"] = ak.values_astype(
            ak.where(has_ge3, ak.firsts(mjjj[idx_maxpt3]), np.nan), np.float64
        )
        bscore = ak.values_astype(self.events["JetGood"].btagUParTAK4B, np.float64)        
        # max b-tag score in event
        self.events["max_btag"] = ak.values_astype(ak.max(bscore, axis=1), np.float64)

        # sum of top-4 b-tag scores
        
        idx_bdesc = ak.argsort(bscore, ascending=False)
        self.events["nBJets"] = ak.num(self.events["BJetGood"])
        # ΔR between the two most b-like jets
        jets_bsorted = self.events["JetGood"][idx_bdesc]
        self.events["dR_top2_btag"] = ak.values_astype(
            ak.firsts(jets_bsorted[:, 0:1].delta_r(jets_bsorted[:, 1:2])), np.float64
        )

        # pT(jj) for the pair whose mass is closest to 125 GeV
        self.events["ptjj_at_mH"] = ak.values_astype(
            ak.where(has_ge2, ak.firsts(ptjj[idx_closest]), np.nan), np.float64
        )

        #bscore = ak.values_astype(self.events["JetGood"].btagUParTAK4B, np.float64)
        idx_bdesc = ak.argsort(bscore, ascending=False)
        b_sorted  = bscore[idx_bdesc]

        # Fill missing ranks with 0.0 when an event has <4 jets
        self.events["btag_rank1"] = ak.fill_none(ak.firsts(b_sorted[:, 0:1]), 0.0)
        self.events["btag_rank2"] = ak.fill_none(ak.firsts(b_sorted[:, 1:2]), 0.0)
        self.events["btag_rank3"] = ak.fill_none(ak.firsts(b_sorted[:, 2:2+1]), 0.0)
        self.events["btag_rank4"] = ak.fill_none(ak.firsts(b_sorted[:, 3:3+1]), 0.0)

        # sum of top-4 b-tag scores
        idx_bdesc = ak.argsort(bscore, ascending=False)
        b_sorted  = bscore[idx_bdesc]
        self.events["sum4_btag"] = ak.values_astype(ak.sum(b_sorted[:, :4], axis=1), np.float64)

        # ΔR between the two most b-like jets
        jets_bsorted = self.events["JetGood"][idx_bdesc]
        self.events["dR_top2_btag"] = ak.values_astype(
            ak.firsts(jets_bsorted[:, 0:1].delta_r(jets_bsorted[:, 1:2])), np.float64
        )

        # pT(jj) for the pair whose mass is closest to 125 GeV
        self.events["ptjj_at_mH"] = ak.values_astype(
            ak.where(has_ge2, ak.firsts(ptjj[idx_closest]), np.nan), np.float64
        )

        # pT of the two most b-tag-like jets
        pt_b1 = jets_bsorted[:, 0:1].pt
        pt_b2 = jets_bsorted[:, 1:2].pt

        pt_b1_f = ak.fill_none(ak.firsts(pt_b1), 0.0)
        pt_b2_f = ak.fill_none(ak.firsts(pt_b2), 0.0)

        # 1) Sum pT of the two leading b-tag jets
        self.events["sum_pt_b1b2"] = pt_b1_f + pt_b2_f
        
        # Normalised version: (pT_b1 + pT_b2) / HT
        self.events["sum_pt_b1b2_over_HT"] = ak.values_astype(
            (pt_b1_f + pt_b2_f) / ak.where(self.events["HT"] > 0, self.events["HT"], np.nan),
            np.float64
        )
        # 2) Ratio pT(b1) / pT(b2)
        self.events["pt_ratio_b1_b2"] = ak.values_astype(
            pt_b1_f / ak.where(pt_b2_f > 0, pt_b2_f, np.nan),
            np.float64
        )
        # Mass of the two most b-tag-like jets
        mbb_btag = (jets_bsorted[:, 0:1] + jets_bsorted[:, 1:2]).mass
        self.events["mbb_btag_top2"] = ak.fill_none(ak.firsts(mbb_btag), np.nan)

        # ΔR(b1, b2) using the two most b-tag-like jets
        self.events["dR_b1_b2"] = ak.values_astype(
            ak.firsts(jets_bsorted[:, 0:1].delta_r(jets_bsorted[:, 1:2])),
            np.float64
        )
        # Sum of all jet masses = Σ m(jet_i)
        self.events["sum_m_jets"] = ak.values_astype(
            ak.sum(self.events["JetGood"].mass, axis=1),
            np.float64
        )
        # Event centrality = (sum pT) / (sum Energy)
        sum_pt = ak.sum(self.events["JetGood"].pt, axis=1)
        sum_E  = ak.sum(self.events["JetGood"].energy, axis=1)

        self.events["centrality"] = ak.values_astype(
            sum_pt / ak.where(sum_E > 0, sum_E, np.nan),
            np.float64
        )

        # pT of the Higgs-like dijet: |pT_j1 + pT_j2|
        higgsj1 = pairs.j1[idx_closest]
        higgsj2 = pairs.j2[idx_closest]

        self.events["pT_higgs_like"] = ak.values_astype(
            ak.firsts((higgsj1 + higgsj2).pt),
            np.float64
        )

        self.events["MET_pt"]  = ak.values_astype(self.events["PuppiMET"]["pt"], np.float64)
        self.events["MET_phi"]  = ak.values_astype(self.events["PuppiMET"]["phi"], np.float64)



        # 1) Collect features in the EXACT training order
        feat_names = DNN_FEATURES
        cols = []
        ev = self.events
        for name in feat_names:
            if name not in ak.fields(ev):
                raise RuntimeError(
                    f"[DNN] Missing feature '{name}' in events. "
                    "Order/names must match training exactly."
                )
            arr = ev[name]  # <-- top-level, NOT ev['events'][name]
            # per your preference: crash if NaN/Inf present
            a_np = ak.to_numpy(arr)
            if not np.isfinite(a_np).all():
                bad = np.argwhere(~np.isfinite(a_np)).ravel()[:5]
                raise RuntimeError(
                    f"[DNN] Non-finite values in feature '{name}'. "
                    f"Examples of bad indices: {bad} …"
                )
            cols.append(a_np.astype(np.float64))

        # 2) Stack -> (N_events, N_features)
        X_raw = np.column_stack(cols)
        if X_raw.ndim != 2 or X_raw.shape[1] != len(feat_names):
            raise RuntimeError(
                f"[DNN] Feature matrix shape mismatch: got {X_raw.shape}, "
                f"expected (N, {len(feat_names)})."
            )

        # 3) Standardize and 4) predict, as you already have:
        Xz = (X_raw - self._dnn_mu) / self._dnn_sigma
        scores = self._dnn_model.predict(Xz, batch_size=4096, verbose=0).ravel()
        self.events["dnn_score"] = ak.Array(scores)

        results = custom_function.solve_ttbar_dilepton(self.events)  # Assuming solve_ttbar_dilepton returns a dictionary
        
        # Store all_higgs_masses_per_event in self.events to persist across files
        self.events["solutions_number"] = ak.Array(results["solutions_number"])
        self.events["all_higgs_masses_per_event"] = ak.Array(results["all_higgs_masses_per_event"])
        self.events["all_weights_per_event"] = ak.Array(results["all_weights_per_event"])
        self.events["max_weight"] = ak.Array(results["max_weight_per_event"])
        self.events["max_weight_higgs_mass"] = ak.Array(results["max_weight_higgs_mass_per_event"])
        self.events["max_weight_ttH_mass"] = ak.Array(results["max_weight_ttH_mass_per_event"])
        self.events["max_sum_weight_ttH_mass"] = ak.Array(results["max_sum_weight_ttH_mass_per_event"])
        self.events["max_mean_weight_ttH_mass"] = ak.Array(results["max_mean_weight_ttH_mass_per_event"])
        self.events["max_weight_top_mass"] = ak.Array(results["max_weight_top_mass_per_event"])
        self.events["max_weight_W_mass"] = ak.Array(results["max_weight_W_mass_per_event"])
        self.events["max_weight_combination"] = ak.Array(results["max_weight_combination_per_event"])
        self.events["second_max_weight"] = ak.Array(results["second_max_weight_per_event"])
        self.events["second_max_weight_higgs_mass"] = ak.Array(results["second_max_weight_higgs_mass_per_event"])
        self.events["second_max_weight_ttH_mass"] = ak.Array(results["second_max_weight_ttH_mass_per_event"])
        self.events["second_max_weight_top_mass"] = ak.Array(results["second_max_weight_top_mass_per_event"])
        self.events["second_max_weight_W_mass"] = ak.Array(results["second_max_weight_W_mass_per_event"])
        self.events["second_max_weight_combination"] = ak.Array(results["second_max_weight_combination_per_event"])
        self.events["third_max_weight"] = ak.Array(results["third_max_weight_per_event"])
        self.events["third_max_weight_higgs_mass"] = ak.Array(results["third_max_weight_higgs_mass_per_event"])
        self.events["third_max_weight_ttH_mass"] = ak.Array(results["third_max_weight_ttH_mass_per_event"])
        self.events["third_max_weight_top_mass"] = ak.Array(results["third_max_weight_top_mass_per_event"])
        self.events["third_max_weight_W_mass"] = ak.Array(results["third_max_weight_W_mass_per_event"])
        self.events["third_max_weight_combination"] = ak.Array(results["third_max_weight_combination_per_event"])
        self.events["fourth_max_weight"] = ak.Array(results["fourth_max_weight_per_event"])
        self.events["fourth_max_weight_higgs_mass"] = ak.Array(results["fourth_max_weight_higgs_mass_per_event"])
        self.events["fourth_max_weight_ttH_mass"] = ak.Array(results["fourth_max_weight_ttH_mass_per_event"])
        self.events["fourth_max_weight_top_mass"] = ak.Array(results["fourth_max_weight_top_mass_per_event"])
        self.events["fourth_max_weight_W_mass"] = ak.Array(results["fourth_max_weight_W_mass_per_event"])
        self.events["fourth_max_weight_combination"] = ak.Array(results["fourth_max_weight_combination_per_event"])
        self.events["max_mean_weight"] = ak.Array(results["max_mean_weight_per_event"])
        self.events["max_mean_weight_higgs_mass"] = ak.Array(results["max_mean_weight_higgs_mass_per_event"])
        self.events["max_mean_weight_top_mass"] = ak.Array(results["max_mean_weight_top_mass_per_event"])
        self.events["max_mean_weight_W_mass"] = ak.Array(results["max_mean_weight_W_mass_per_event"])
        self.events["max_mean_weight_combination"] = ak.Array(results["max_mean_weight_combination_per_event"])
        self.events["max_sum_weight"] = ak.Array(results["max_sum_weight_per_event"])
        self.events["max_sum_weight_higgs_mass"] = ak.Array(results["max_sum_weight_higgs_mass_per_event"])
        self.events["max_sum_weight_top_mass"] = ak.Array(results["max_sum_weight_top_mass_per_event"])
        self.events["max_sum_weight_W_mass"] = ak.Array(results["max_sum_weight_W_mass_per_event"])
        self.events["max_sum_weight_combination"] = ak.Array(results["max_sum_weight_combination_per_event"])
        self.events["filtered_solutions_number"] = self.events["solutions_number"]

        res = custom_function.dr_criterion_for_max_weight(self.events)
        self.events["massreco_chosen_pair_higgs_mass"] = ak.Array(res["massreco_chosen_pair_higgs_mass"])
        self.events["massreco_chosen_pair_top_mass"] = ak.Array(res["massreco_chosen_pair_top_mass"])
        self.events["massreco_chosen_pair_W_mass"] = ak.Array(res["massreco_chosen_pair_W_mass"])
        self.events["massreco_chosen_pair_ttH_mass"] = ak.Array(res["massreco_chosen_pair_ttH_mass"])
        self.events["massreco_chosen_pair_combination"] = ak.Array(res["massreco_chosen_pair_combination"])