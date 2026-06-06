import awkward as ak
import numpy as np
from pocket_coffea.workflows.tthbb_base_processor import ttHbbBaseProcessor
from pocket_coffea.lib.deltaR_matching import metric_eta, metric_phi
from pocket_coffea.lib.objects import (
    get_dilepton,
)
from pocket_coffea.lib.objects import (
    btagging,
)
from custom_function import *
from mass_reco_functions_mc import *
import event_shapes
from event_shapes import compute_event_shapes
import vector
vector.register_awkward()
import csv


def debug_bscore_raise(bscore):
    import awkward as ak
    import numpy as np

    mask_bad = ~ak.is_finite(bscore)
    if not ak.any(mask_bad):
        return

    evt = ak.where(ak.any(mask_bad, axis=1))[0][0]
    bad_jets = ak.where(mask_bad[evt])[0]
    msg = []
    msg.append("\n[DEBUG] Non-finite bscore detected!")
    msg.append(f"Event index: {evt}")
    msg.append(f"Jet indices with NaN/inf: {bad_jets.tolist()}")
    msg.append(f"bscore values for event:\n{bscore[evt].tolist()}")
    idx_evt = ak.argsort(bscore[evt], ascending=False)
    sorted_evt = bscore[evt][idx_evt]
    msg.append(f"Sorted bscore values:\n{sorted_evt.tolist()}")
    msg.append(f"Top-4 sorted:\n{sorted_evt[:4].tolist()}")
    sum4 = ak.sum(sorted_evt[:4])
    msg.append(f"Sum(top-4): {sum4}")
    full_msg = "\n".join(msg)
    raise RuntimeError(full_msg)


# ==== DNN (Step 1: constants + loaders) ======================================
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
    "MET_pt",
]

DNN_NORMS_PATH = "/eos/user/p/piosifid/PocketCoffea/ANN_newCoffea/DNN_new/output/the_one_for_the_Analysis_Note/variable_norm.csv"
DNN_MODEL_PATH = "/eos/user/p/piosifid/PocketCoffea/ANN_newCoffea/DNN_new/output/the_one_for_the_Analysis_Note/model_best.keras"

def _load_norms_csv(csv_path):
    lut = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            raw = row["var"]
            key = raw[7:] if raw.startswith("events_") else raw
            mu = float(row["mu"])
            sd = float(row["std"])
            lut[key] = (mu, sd)

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
        self.isRun3 = True if self.params["run_period"] == 'Run3' else False
        self._dnn_mu, self._dnn_sigma = _load_norms_csv(DNN_NORMS_PATH)
        self._dnn_model = _load_tf_model(DNN_MODEL_PATH)

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation=variation)
        self.events["ll"] = get_dilepton(
            self.events.ElectronGood, self.events.MuonGood
        )

    def define_common_variables_after_presel(self, variation):
        # FIX: was calling before_presel — now correctly calls after_presel
        super().define_common_variables_after_presel(variation=variation)

        # =====================================================================
        # b-jet pair kinematics
        # =====================================================================
        deltaR   = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"]), axis=2)
        deltaEta = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"], metric=metric_eta), axis=2)
        deltaPhi = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"], metric=metric_phi), axis=2)
        deltaR   = deltaR[deltaR > 0.]
        deltaEta = deltaEta[deltaEta > 0.]
        deltaPhi = deltaPhi[deltaPhi > 0.]

        pairs = ak.argcombinations(self.events["BJetGood"], 2, axis=1)
        b1 = self.events["BJetGood"][pairs.slot0]
        b2 = self.events["BJetGood"][pairs.slot1]
        deltaR_unique  = b1.delta_r(b2)
        idx_pairs_sorted = ak.argsort(deltaR_unique, axis=1)
        pairs_sorted   = pairs[idx_pairs_sorted]

        self.events["deltaRbb_min"]   = ak.min(deltaR, axis=1)
        self.events["deltaEtabb_min"] = ak.min(deltaEta, axis=1)
        self.events["deltaPhibb_min"] = ak.min(deltaPhi, axis=1)
        self.events["mbb"]            = (self.events["BJetGood"][pairs_sorted.slot0] + self.events["BJetGood"][pairs_sorted.slot1]).mass
        self.events["mbb_min"]        = ak.firsts(self.events["mbb"])
        self.events["mass_w"]         = 80
        self.events["mass_top"]       = 170
        self.events["mass_w_minus"]   = self.events["mass_w"]
        self.events["mass_antitop"]   = self.events["mass_top"]

        # =====================================================================
        # Jet pt regression and btag sorting
        # =====================================================================
        jet_pt_raw = self.events["JetGood"]["pt"] * (1 - self.events["JetGood"]["rawFactor"])
        jet_pt_regressed = (
            jet_pt_raw *
            self.events["JetGood"]["PNetRegPtRawCorr"] *
            self.events["JetGood"]["PNetRegPtRawCorrNeutrino"]
        )
        self.events["JetGood"] = ak.with_field(self.events["JetGood"], jet_pt_regressed, "pt_regressed")

        btag_sorted_idx  = ak.argsort(self.events["JetGood"]["btagUParTAK4B"], ascending=False)
        sorted_jets      = self.events["JetGood"][btag_sorted_idx]
        sorted_jet_orig_idx = btag_sorted_idx

        self.events["1st_jet"]     = ak.firsts(sorted_jets[:, 0:1])
        self.events["2nd_jet"]     = ak.firsts(sorted_jets[:, 1:2])
        self.events["3rd_jet"]     = ak.firsts(sorted_jets[:, 2:3])
        self.events["4th_jet"]     = ak.firsts(sorted_jets[:, 3:4])
        self.events["5th_jet"]     = ak.firsts(sorted_jets[:, 4:5])
        self.events["1st_jet_idx"] = ak.firsts(sorted_jet_orig_idx[:, 0:1])
        self.events["2nd_jet_idx"] = ak.firsts(sorted_jet_orig_idx[:, 1:2])
        self.events["3rd_jet_idx"] = ak.firsts(sorted_jet_orig_idx[:, 2:3])
        self.events["4th_jet_idx"] = ak.firsts(sorted_jet_orig_idx[:, 3:4])
        self.events["5th_jet_idx"] = ak.firsts(sorted_jet_orig_idx[:, 4:5])

        import vector
        vector.register_awkward()

        for jet_key, jet_field in [
            ("1st_jet", "1st_jet"), ("2nd_jet", "2nd_jet"),
            ("3rd_jet", "3rd_jet"), ("4th_jet", "4th_jet"),
            ("5th_jet", "5th_jet"),
        ]:
            bjet = ak.zip({
                "pt":   self.events[jet_field]["pt_regressed"],
                "eta":  self.events[jet_field]["eta"],
                "phi":  self.events[jet_field]["phi"],
                "mass": self.events[jet_field]["mass"],
            }, with_name="Momentum4D")
            self.events[jet_key] = ak.with_field(self.events[jet_key], bjet.px, "px")
            self.events[jet_key] = ak.with_field(self.events[jet_key], bjet.py, "py")
            self.events[jet_key] = ak.with_field(self.events[jet_key], bjet.pz, "pz")
            self.events[jet_key] = ak.with_field(self.events[jet_key], bjet.E,  "E")

        # =====================================================================
        # Lepton 4-vectors
        # =====================================================================
        lepton_pos_mask = self.events["LeptonGood"]["charge"] == 1
        self.events["lepton_pos"] = self.events["LeptonGood"][lepton_pos_mask]
        lepton_neg_mask = self.events["LeptonGood"]["charge"] == -1
        self.events["lepton_neg"] = self.events["LeptonGood"][lepton_neg_mask]

        for lep_key in ["lepton_pos", "lepton_neg"]:
            lep = ak.zip({
                "pt":   self.events[lep_key]["pt"],
                "eta":  self.events[lep_key]["eta"],
                "phi":  self.events[lep_key]["phi"],
                "mass": self.events[lep_key]["mass"],
            }, with_name="Momentum4D")
            self.events[lep_key] = ak.with_field(self.events[lep_key], lep.px, "px")
            self.events[lep_key] = ak.with_field(self.events[lep_key], lep.py, "py")
            self.events[lep_key] = ak.with_field(self.events[lep_key], lep.pz, "pz")
            self.events[lep_key] = ak.with_field(self.events[lep_key], lep.E,  "E")

        self.events["MET_few"] = ak.Array({
            "pt":  self.events["PuppiMET"]["pt"],
            "phi": self.events["PuppiMET"]["phi"],
        })

        # =====================================================================
        # Lepton-bjet min DR
        # =====================================================================
        lep_bjet_idx_pos = ak.argcartesian([self.events["lepton_pos"], self.events["BJetGood"]], axis=1)
        lep_pos_cart     = self.events["lepton_pos"][lep_bjet_idx_pos["0"]]
        bjet_pos         = self.events["BJetGood"][lep_bjet_idx_pos["1"]]
        min_dr_pos       = ak.min(lep_pos_cart.delta_r(bjet_pos), axis=1)

        lep_bjet_idx_neg = ak.argcartesian([self.events["lepton_neg"], self.events["BJetGood"]], axis=1)
        lep_neg_cart     = self.events["lepton_neg"][lep_bjet_idx_neg["0"]]
        bjet_neg         = self.events["BJetGood"][lep_bjet_idx_neg["1"]]
        min_dr_neg       = ak.min(lep_neg_cart.delta_r(bjet_neg), axis=1)

        self.events["min_dr_lepton_pos_bjets"] = min_dr_pos
        self.events["min_dr_lepton_neg_bjets"] = min_dr_neg
        self.events["min_dr_any_lepton_bjets"] = ak.min(
            ak.concatenate([min_dr_pos[..., None], min_dr_neg[..., None]], axis=1), axis=1
        )

        # =====================================================================
        # Event shape and DNN input variables
        # =====================================================================
        jets   = self.events["JetGood"]
        shapes = compute_event_shapes(jets)
        self.events["C_jet"]      = shapes["C_jet"]
        self.events["D_jet"]      = shapes["D_jet"]
        self.events["Aplanarity"] = shapes["Aplanarity"]
        self.events["H4"]         = shapes["H4"]

        ptj = ak.values_astype(
            jets.pt_regressed if "pt_regressed" in ak.fields(jets) else jets.pt,
            np.float64
        )

        idx_ptdesc = ak.argsort(ptj, axis=1, ascending=False)
        pt_sorted  = ptj[idx_ptdesc]
        self.events["pt_jet2"] = ak.fill_none(ak.firsts(pt_sorted[:, 1:2]), np.nan)
        self.events["HT"]      = ak.values_astype(ak.sum(ptj, axis=1), np.float64)

        pairs   = ak.combinations(jets, 2, fields=["j1", "j2"])
        has_ge2 = ak.num(jets) >= 2
        dR_pairs    = pairs.j1.delta_r(pairs.j2)
        dEta_pairs  = abs(pairs.j1.eta - pairs.j2.eta)
        mjj         = (pairs.j1 + pairs.j2).mass
        ptjj        = ak.values_astype(pairs.j1.pt + pairs.j2.pt, np.float64)

        self.events["dRjj_min"] = ak.values_astype(ak.where(has_ge2, ak.min(dR_pairs, axis=1), np.nan), np.float64)
        self.events["deta_max"] = ak.values_astype(ak.where(has_ge2, ak.max(dEta_pairs, axis=1), np.nan), np.float64)

        idx_closest = ak.argmin(abs(mjj - 125.0), axis=1, keepdims=True)
        self.events["m_higgs_like_jj"] = ak.values_astype(ak.where(has_ge2, ak.firsts(mjj[idx_closest]), np.nan), np.float64)

        idx_drmin = ak.argmin(dR_pairs, axis=1, keepdims=True)
        self.events["ptjj_at_dRmin"] = ak.values_astype(ak.where(has_ge2, ak.firsts(ptjj[idx_drmin]), np.nan), np.float64)

        in_win = (mjj > 100.0) & (mjj < 140.0)
        self.events["Njj_higgs_like"] = ak.values_astype(ak.where(has_ge2, ak.sum(in_win, axis=1), 0), np.int32)

        has_ge3   = ak.num(jets) >= 3
        trips     = ak.combinations(jets, 3, fields=["j1", "j2", "j3"])
        ptsum3    = ak.values_astype(trips.j1.pt + trips.j2.pt + trips.j3.pt, np.float64)
        mjjj      = (trips.j1 + trips.j2 + trips.j3).mass
        idx_maxpt3 = ak.argmax(ptsum3, axis=1, keepdims=True)
        self.events["m_jjj_maxpT"] = ak.values_astype(ak.where(has_ge3, ak.firsts(mjjj[idx_maxpt3]), np.nan), np.float64)

        bscore    = ak.values_astype(self.events["JetGood"].btagUParTAK4B, np.float64)
        idx_bdesc = ak.argsort(bscore, ascending=False)
        b_sorted  = bscore[idx_bdesc]
        jets_bsorted = self.events["JetGood"][idx_bdesc]

        self.events["max_btag"]    = ak.values_astype(ak.max(bscore, axis=1), np.float64)
        self.events["nBJets"]      = ak.num(self.events["BJetGood"])
        self.events["btag_rank1"]  = ak.fill_none(ak.firsts(b_sorted[:, 0:1]), 0.0)
        self.events["btag_rank2"]  = ak.fill_none(ak.firsts(b_sorted[:, 1:2]), 0.0)
        self.events["btag_rank3"]  = ak.fill_none(ak.firsts(b_sorted[:, 2:3]), 0.0)
        self.events["btag_rank4"]  = ak.fill_none(ak.firsts(b_sorted[:, 3:4]), 0.0)
        self.events["sum4_btag"]   = ak.values_astype(ak.sum(b_sorted[:, :4], axis=1), np.float64)
        self.events["dR_top2_btag"] = ak.values_astype(
            ak.firsts(jets_bsorted[:, 0:1].delta_r(jets_bsorted[:, 1:2])), np.float64
        )
        self.events["ptjj_at_mH"] = ak.values_astype(ak.where(has_ge2, ak.firsts(ptjj[idx_closest]), np.nan), np.float64)

        pt_b1_f = ak.fill_none(ak.firsts(jets_bsorted[:, 0:1].pt), 0.0)
        pt_b2_f = ak.fill_none(ak.firsts(jets_bsorted[:, 1:2].pt), 0.0)
        self.events["sum_pt_b1b2"]         = pt_b1_f + pt_b2_f
        self.events["sum_pt_b1b2_over_HT"] = ak.values_astype(
            (pt_b1_f + pt_b2_f) / ak.where(self.events["HT"] > 0, self.events["HT"], np.nan), np.float64
        )
        self.events["pt_ratio_b1_b2"] = ak.values_astype(
            pt_b1_f / ak.where(pt_b2_f > 0, pt_b2_f, np.nan), np.float64
        )
        mbb_btag = (jets_bsorted[:, 0:1] + jets_bsorted[:, 1:2]).mass
        self.events["mbb_btag_top2"] = ak.fill_none(ak.firsts(mbb_btag), np.nan)
        self.events["dR_b1_b2"]      = ak.values_astype(
            ak.firsts(jets_bsorted[:, 0:1].delta_r(jets_bsorted[:, 1:2])), np.float64
        )
        self.events["sum_m_jets"] = ak.values_astype(ak.sum(self.events["JetGood"].mass, axis=1), np.float64)

        sum_pt = ak.sum(self.events["JetGood"].pt, axis=1)
        sum_E  = ak.sum(self.events["JetGood"].energy, axis=1)
        self.events["centrality"] = ak.values_astype(sum_pt / ak.where(sum_E > 0, sum_E, np.nan), np.float64)

        higgsj1 = pairs.j1[idx_closest]
        higgsj2 = pairs.j2[idx_closest]
        self.events["pT_higgs_like"] = ak.values_astype(ak.firsts((higgsj1 + higgsj2).pt), np.float64)

        self.events["MET_pt"]  = ak.values_astype(self.events["PuppiMET"]["pt"],  np.float64)
        self.events["MET_phi"] = ak.values_astype(self.events["PuppiMET"]["phi"], np.float64)

        # =====================================================================
        # DNN inference
        # =====================================================================
        feat_names = DNN_FEATURES
        cols = []
        ev   = self.events
        for name in feat_names:
            if name not in ak.fields(ev):
                raise RuntimeError(f"[DNN] Missing feature '{name}' in events.")
            a_np = ak.to_numpy(ev[name]).astype(np.float64)
            if not np.isfinite(a_np).all():
                bad = np.argwhere(~np.isfinite(a_np)).ravel()[:5]
                raise RuntimeError(f"[DNN] Non-finite values in feature '{name}'. Bad indices: {bad}")
            cols.append(a_np)

        X_raw = np.column_stack(cols)
        if X_raw.ndim != 2 or X_raw.shape[1] != len(feat_names):
            raise RuntimeError(f"[DNN] Feature matrix shape mismatch: got {X_raw.shape}, expected (N, {len(feat_names)}).")

        Xz     = (X_raw - self._dnn_mu) / self._dnn_sigma
        scores = self._dnn_model.predict(Xz, batch_size=4096, verbose=0).ravel()
        self.events["dnn_score"] = ak.Array(scores)

        # =====================================================================
        # Mass reconstruction
        # =====================================================================
        results = solve_ttbar_dilepton(self.events)

        self.events["solutions_number"]           = ak.Array(results["solutions_number"])
        self.events["all_higgs_masses_per_event"] = ak.Array(results["all_higgs_masses_per_event"])
        self.events["all_weights_per_event"]      = ak.Array(results["all_weights_per_event"])
        self.events["max_weight"]                 = ak.Array(results["max_weight_per_event"])
        self.events["max_weight_higgs_mass"]      = ak.Array(results["max_weight_higgs_mass_per_event"])
        self.events["max_weight_ttH_mass"]        = ak.Array(results["max_weight_ttH_mass_per_event"])
        self.events["max_sum_weight_ttH_mass"]    = ak.Array(results["max_sum_weight_ttH_mass_per_event"])
        self.events["max_mean_weight_ttH_mass"]   = ak.Array(results["max_mean_weight_ttH_mass_per_event"])
        self.events["max_weight_top_mass"]        = ak.Array(results["max_weight_top_mass_per_event"])
        self.events["max_weight_W_mass"]          = ak.Array(results["max_weight_W_mass_per_event"])
        self.events["max_weight_combination"]     = ak.Array(results["max_weight_combination_per_event"])
        self.events["second_max_weight"]              = ak.Array(results["second_max_weight_per_event"])
        self.events["second_max_weight_higgs_mass"]   = ak.Array(results["second_max_weight_higgs_mass_per_event"])
        self.events["second_max_weight_ttH_mass"]     = ak.Array(results["second_max_weight_ttH_mass_per_event"])
        self.events["second_max_weight_top_mass"]     = ak.Array(results["second_max_weight_top_mass_per_event"])
        self.events["second_max_weight_W_mass"]       = ak.Array(results["second_max_weight_W_mass_per_event"])
        self.events["second_max_weight_combination"]  = ak.Array(results["second_max_weight_combination_per_event"])
        self.events["third_max_weight"]               = ak.Array(results["third_max_weight_per_event"])
        self.events["third_max_weight_higgs_mass"]    = ak.Array(results["third_max_weight_higgs_mass_per_event"])
        self.events["third_max_weight_ttH_mass"]      = ak.Array(results["third_max_weight_ttH_mass_per_event"])
        self.events["third_max_weight_top_mass"]      = ak.Array(results["third_max_weight_top_mass_per_event"])
        self.events["third_max_weight_W_mass"]        = ak.Array(results["third_max_weight_W_mass_per_event"])
        self.events["third_max_weight_combination"]   = ak.Array(results["third_max_weight_combination_per_event"])
        self.events["fourth_max_weight"]              = ak.Array(results["fourth_max_weight_per_event"])
        self.events["fourth_max_weight_higgs_mass"]   = ak.Array(results["fourth_max_weight_higgs_mass_per_event"])
        self.events["fourth_max_weight_ttH_mass"]     = ak.Array(results["fourth_max_weight_ttH_mass_per_event"])
        self.events["fourth_max_weight_top_mass"]     = ak.Array(results["fourth_max_weight_top_mass_per_event"])
        self.events["fourth_max_weight_W_mass"]       = ak.Array(results["fourth_max_weight_W_mass_per_event"])
        self.events["fourth_max_weight_combination"]  = ak.Array(results["fourth_max_weight_combination_per_event"])
        self.events["max_mean_weight"]                = ak.Array(results["max_mean_weight_per_event"])
        self.events["max_mean_weight_higgs_mass"]     = ak.Array(results["max_mean_weight_higgs_mass_per_event"])
        self.events["max_mean_weight_top_mass"]       = ak.Array(results["max_mean_weight_top_mass_per_event"])
        self.events["max_mean_weight_W_mass"]         = ak.Array(results["max_mean_weight_W_mass_per_event"])
        self.events["max_mean_weight_combination"]    = ak.Array(results["max_mean_weight_combination_per_event"])
        self.events["max_sum_weight"]                 = ak.Array(results["max_sum_weight_per_event"])
        self.events["max_sum_weight_higgs_mass"]      = ak.Array(results["max_sum_weight_higgs_mass_per_event"])
        self.events["max_sum_weight_top_mass"]        = ak.Array(results["max_sum_weight_top_mass_per_event"])
        self.events["max_sum_weight_W_mass"]          = ak.Array(results["max_sum_weight_W_mass_per_event"])
        self.events["max_sum_weight_combination"]     = ak.Array(results["max_sum_weight_combination_per_event"])
        self.events["filtered_solutions_number"]      = self.events["solutions_number"]

        # =====================================================================
        # DR criterion — selects best combination, stores all per-rank variables
        # =====================================================================
        dr_res = dr_criterion_for_max_weight(self.events)

        # Original outputs
        self.events["massreco_chosen_pair_higgs_mass"]  = ak.Array(dr_res["massreco_chosen_pair_higgs_mass"])
        self.events["massreco_chosen_pair_top_mass"]    = ak.Array(dr_res["massreco_chosen_pair_top_mass"])
        self.events["massreco_chosen_pair_W_mass"]      = ak.Array(dr_res["massreco_chosen_pair_W_mass"])
        self.events["massreco_chosen_pair_ttH_mass"]    = ak.Array(dr_res["massreco_chosen_pair_ttH_mass"])
        self.events["massreco_chosen_pair_combination"] = ak.Array(dr_res["massreco_chosen_pair_combination"])

        # All-events per-rank
        self.events["rank_used"]          = ak.Array(dr_res["rank_used"])
        self.events["dr_rank1"]           = ak.Array(dr_res["dr_rank1"])
        self.events["dr_rank2"]           = ak.Array(dr_res["dr_rank2"])
        self.events["dr_rank3"]           = ak.Array(dr_res["dr_rank3"])
        self.events["dr_rank4"]           = ak.Array(dr_res["dr_rank4"])
        self.events["dr_chosen"]          = ak.Array(dr_res["dr_chosen"])
        self.events["higgs_mass_rank1"]   = ak.Array(dr_res["higgs_mass_rank1"])
        self.events["higgs_mass_rank2"]   = ak.Array(dr_res["higgs_mass_rank2"])
        self.events["higgs_mass_rank3"]   = ak.Array(dr_res["higgs_mass_rank3"])
        self.events["higgs_mass_rank4"]   = ak.Array(dr_res["higgs_mass_rank4"])
        self.events["higgs_mass_chosen"]  = ak.Array(dr_res["higgs_mass_chosen"])

        # Selected-from-rank
        self.events["higgs_mass_chosen_from_rank1"]    = ak.Array(dr_res["higgs_mass_chosen_from_rank1"])
        self.events["higgs_mass_chosen_from_rank2"]    = ak.Array(dr_res["higgs_mass_chosen_from_rank2"])
        self.events["higgs_mass_chosen_from_rank3"]    = ak.Array(dr_res["higgs_mass_chosen_from_rank3"])
        self.events["higgs_mass_chosen_from_rank4"]    = ak.Array(dr_res["higgs_mass_chosen_from_rank4"])
        self.events["higgs_mass_chosen_from_fallback"] = ak.Array(dr_res["higgs_mass_chosen_from_fallback"])
        self.events["dr_chosen_from_rank1"]            = ak.Array(dr_res["dr_chosen_from_rank1"])
        self.events["dr_chosen_from_rank2"]            = ak.Array(dr_res["dr_chosen_from_rank2"])
        self.events["dr_chosen_from_rank3"]            = ak.Array(dr_res["dr_chosen_from_rank3"])
        self.events["dr_chosen_from_rank4"]            = ak.Array(dr_res["dr_chosen_from_rank4"])
        self.events["dr_chosen_from_fallback"]         = ak.Array(dr_res["dr_chosen_from_fallback"])

        # Rejected-from-rank
        self.events["higgs_mass_rank1_rejected"] = ak.Array(dr_res["higgs_mass_rank1_rejected"])
        self.events["higgs_mass_rank2_rejected"] = ak.Array(dr_res["higgs_mass_rank2_rejected"])
        self.events["higgs_mass_rank3_rejected"] = ak.Array(dr_res["higgs_mass_rank3_rejected"])
        self.events["higgs_mass_rank4_rejected"] = ak.Array(dr_res["higgs_mass_rank4_rejected"])
        self.events["dr_rank1_rejected"]         = ak.Array(dr_res["dr_rank1_rejected"])
        self.events["dr_rank2_rejected"]         = ak.Array(dr_res["dr_rank2_rejected"])
        self.events["dr_rank3_rejected"]         = ak.Array(dr_res["dr_rank3_rejected"])
        self.events["dr_rank4_rejected"]         = ak.Array(dr_res["dr_rank4_rejected"])

        # DR combination fields for downstream use
        self.events["dr_combination"] = ak.Array(dr_res["massreco_chosen_pair_combination"])
        self.events["dr_higgs_mass"]  = ak.Array(dr_res["massreco_chosen_pair_higgs_mass"])
        self.events["dr_top_mass"]    = ak.Array(dr_res["massreco_chosen_pair_top_mass"])
        self.events["dr_W_mass"]      = ak.Array(dr_res["massreco_chosen_pair_W_mass"])
        self.events["dr_ttH_mass"]    = ak.Array(dr_res["massreco_chosen_pair_ttH_mass"])