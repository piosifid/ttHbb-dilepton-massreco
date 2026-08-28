import awkward as ak
import numpy as np
import custom_function
from pocket_coffea.workflows.tthbb_base_processor import ttHbbBaseProcessor
from pocket_coffea.lib.deltaR_matching import metric_eta, metric_phi
from pocket_coffea.lib.deltaR_matching import object_matching
from pocket_coffea.lib.objects import get_dilepton
from custom_function import *
from mass_reco_functions_mc import *
import event_shapes
from event_shapes import compute_event_shapes
import vector
vector.register_awkward()
import csv

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
    def compute_pull_vectors_drmatched(self, genjets, genparts, dr_threshold=0.4):

        # Keep only relevant GenParts (quarks & gluons, final state)
        is_qg = ((abs(genparts.pdgId) <= 6) | (genparts.pdgId == 21)) & genparts.hasFlags("isLastCopy")
        partons = genparts[is_qg]

        # Match GenParts to GenJets
        matched_parts, matched_jets, dR = object_matching(partons, genjets, dr_min=dr_threshold)

        pull_y_all = []
        pull_phi_all = []

        for iev in range(len(genjets)):
            jets_evt = genjets[iev]
            parts_evt = matched_parts[iev]
            jets_matched_evt = matched_jets[iev]

            pull_y_evt = []
            pull_phi_evt = []

            for ijet, jet in enumerate(jets_evt):
                # Get GenParts matched to this GenJet
                mask = (
                    (jets_matched_evt.pt == jet.pt)
                    & (jets_matched_evt.eta == jet.eta)
                    & (jets_matched_evt.phi == jet.phi)
                    & (jets_matched_evt.mass == jet.mass)
                )

                daughters = parts_evt[mask]

                if len(daughters) == 0:
                    pull_y_evt.append(np.nan)
                    pull_phi_evt.append(np.nan)
                    continue

                deta = daughters.eta - jet.eta
                dphi = (daughters.phi - jet.phi + np.pi) % (2 * np.pi) - np.pi
                weight = daughters.pt / jet.pt

                ty = ak.sum(weight * deta * abs(deta))
                tphi = ak.sum(weight * deta * dphi)
    
                pull_y_evt.append(ty)
                pull_phi_evt.append(tphi)

            pull_y_all.append(pull_y_evt)
            pull_phi_all.append(pull_phi_evt)

        pull_y = ak.Array(pull_y_all)
        pull_phi = ak.Array(pull_phi_all)
        pull_magnitude = np.sqrt(pull_y ** 2 + pull_phi ** 2)
        pull_angle = np.arctan2(pull_phi, pull_y)

        return {
            "pull_y": pull_y,
            "pull_phi": pull_phi,
            "pull_magnitude": pull_magnitude,
            "pull_angle": pull_angle,
        }

    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.isRun3 = True if self.params["run_period"]=='Run3' else False
        # List of sample names (as they appear in events.metadata["sample"])
        # that actually have a true Higgs boson to match against, e.g.
        # ["TTH_Hto2B"]. Set via `has_higgs_truth_samples` in the config
        # params. Any sample NOT in this list gets has_higgs_truth = False,
        # including the case where the param isn't set at all (defaults to
        # an empty list -- nobody gets Higgs truth unless explicitly listed).
        # This can't be resolved to a single self.has_higgs_truth here in
        # __init__, because __init__ runs once for the whole job, before any
        # chunk (and therefore any per-sample metadata) exists --
        # self.has_higgs_truth is instead set per-chunk, from the sample
        # name, at the top of define_common_variables_after_presel.
        self._signal_samples = self.params.get("has_higgs_truth_samples", [])
         # --- Step 1: load model + norms once per worker (will raise if broken) ---
        self._dnn_mu, self._dnn_sigma = _load_norms_csv(DNN_NORMS_PATH)
        self._dnn_model = _load_tf_model(DNN_MODEL_PATH)

        # --- Era-dependent btag algorithm ---
        # btagUParTAK4B (UParT) is only correct for 2024. For 2022/2023 eras
        # the right branch is btagRobustParTAK4B, per the working_point block
        # in btagging_fixedWP_Run3_DL.yaml (merged into self.params under
        # self.params["btagging"]["working_point"][<era>]["btagging_algorithm"]).
        # Resolved once here (not per-chunk): a single Configurator run covers
        # one era, set via `parameters["year"] = ...` in config.py, e.g. "2024",
        # "2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix".
        self._year = self.params.get("year", "2024")
        try:
            self.btag_branch = self.params["btagging"]["working_point"][self._year]["btagging_algorithm"]
        except (KeyError, TypeError) as e:
            raise KeyError(
                f"Could not resolve the btag algorithm for year='{self._year}' from "
                f"self.params['btagging']['working_point'][\"{self._year}\"]['btagging_algorithm']. "
                f"Make sure `parameters[\"year\"] = <era>` is set in config.py (matching a "
                f"key in btagging_fixedWP_Run3_DL.yaml, e.g. '2024', '2022_preEE', "
                f"'2022_postEE', '2023_preBPix', '2023_postBPix') and that file is included "
                f"in defaults.merge_parameters_from_files(...)."
            ) from e

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation=variation)
        self.events["ll"] = get_dilepton(
            self.events.ElectronGood, self.events.MuonGood
        )

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
        n = len(ids)
        if n == 5:
            # 2 top b-quarks + 2 Higgs b-quarks + 1 extra ISR/FSR parton
            offset = 1
            from_part = [-1] * 5
            from_part[0] = 4
        elif n == 4:
            # 2 top b-quarks + 2 Higgs b-quarks
            offset = 0
            from_part = [-1] * 4
        else:
            # No ttH-shaped Higgs pair present (e.g. a background sample
            # with has_higgs_truth=False, so `ids` is just the LHE
            # b/bbar partons -- no padding to a Higgs-sized array, or the
            # provenance array would end up longer than `ids` itself and
            # break ak.with_field(quarks, prov, "provenance") below.
            offset = 0
            from_part = [-1] * n

        if n in (4, 5):
            if ids[0 + offset] == 5:
                from_part[0 + offset] = 2
            if ids[1 + offset] == -5:
                from_part[1 + offset] = 3

            from_part[2 + offset] = 1
            from_part[3 + offset] = 1
        else:
            # No Higgs quarks to label here -- just tag the top/antitop
            # b-quarks wherever they show up among the n partons.
            for i, pid in enumerate(ids):
                if pid == 5:
                    from_part[i] = 2
                elif pid == -5:
                    from_part[i] = 3

        array_builder.begin_list()
        for i in from_part:
            array_builder.append(i)
        array_builder.end_list()
      return array_builder

    def define_common_variables_after_presel(self, variation):
        super().define_common_variables_before_presel(variation=variation)
        # Resolve has_higgs_truth for THIS chunk's sample. Runs per chunk
        # (not just once in __init__) because different samples in the same
        # job (e.g. TTH_Hto2B signal + TT2L2Nu background processed
        # together) need different answers here. Anything not explicitly
        # listed in has_higgs_truth_samples (config params) is False.
        sample_name = self.events.metadata.get("sample", None)
        self.has_higgs_truth = sample_name in self._signal_samples

        # Compute deltaR(b, b) of all possible b-jet pairs.
        # We require deltaR > 0 to exclude the deltaR between the jets with themselves
        deltaR = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"]), axis=2)
        deltaEta = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"], metric=metric_eta), axis=2)
        deltaPhi = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"], metric=metric_phi), axis=2)
        deltaR = deltaR[deltaR > 0.]
        deltaEta = deltaEta[deltaEta > 0.]
        deltaPhi = deltaPhi[deltaPhi > 0.]
        
        # Get the deltaR with no possibility of repetition of identical b-jet pairs
        # (unique combinations only, sorted by ascending deltaR)
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
        provenance_map = {
            1: "Higgs",
            2: "Top",
            3: "AntiTop",
            4: "ISR/FSR",
            -1: "Unknown"
        }
        isOutgoing = self.events.LHEPart.status == 1
        isParton = abs(self.events.LHEPart.pdgId) <= 6
        quarks = self.events.LHEPart[isOutgoing & isParton]

        # Naming the quark array in this particular way will make more functions accessible
        # (Similar as to vector.register_awkward() and "Momentum4D")
        if self.has_higgs_truth:
            higgs = self.events.GenPart[
                    (self.events.GenPart.pdgId == 25)
                    & (self.events.GenPart.hasFlags(['fromHardProcess']))
            ]

            higgs = higgs[ak.num(higgs.childrenIdxG, axis=2) == 2]
            higgs_partons = ak.flatten(higgs.children, axis=2)
            quarks = ak.with_name(
                    ak.concatenate((quarks, higgs_partons), axis=1),
                    name='PtEtaPhiMCandidate',
            )
        else:
            # Background sample with no true Higgs (has_higgs_truth=False):
            # only the LHE top/antitop b-quarks go into `quarks`. Every
            # downstream truth-matching function already handles "no
            # provenance==1 (Higgs) jets found" per event gracefully, so
            # nothing else needs to change -- correct_match, higgs_mass_truth_jets,
            # etc. will just come back empty/None for every event, as expected.
            quarks = ak.with_name(quarks, name='PtEtaPhiMCandidate')

        # Obtain parton provenance and match partons to RECO-level jets
        prov = self.get_partons_provenance_ttHbb_dileptonic(
                quarks.pdgId, ak.ArrayBuilder()).snapshot()

        quarks = ak.with_field(quarks, prov, "provenance")
        q_matched, j_matched, dR = object_matching(
                quarks, self.events.JetGood, dr_min=.4
            )
        # === Extract pT quantities
        self.events["matched_genparton_pt"] = q_matched.pt
        self.events["matched_recojet_pt_jec"] = j_matched.pt

        jet_pt_raw = j_matched.pt * (1 - j_matched.rawFactor)
        self.events["matched_recojet_pt_regressed"] = (
            jet_pt_raw *
            j_matched.PNetRegPtRawCorr *
            j_matched.PNetRegPtRawCorrNeutrino
        )

        # === Compute responses (jagged, one entry per matched jet per event)
        self.events["matched_response_jec"] = self.events["matched_recojet_pt_jec"] / self.events["matched_genparton_pt"]
        self.events["matched_response_regressed"] = self.events["matched_recojet_pt_regressed"]/self.events["matched_genparton_pt"]

        # === Object collection for histogramming (coll="MatchedJets" in HistConf,
        # NOT coll="events" -- q_matched/j_matched are jagged, 0-N matches per event,
        # so a coll="events" axis breaks the SHAPE GUARD and PocketCoffea's per-event
        # weight alignment. Same pattern as workflow_mc_v3.py / CorrectChosenPairJets
        # below: keep the per-event jagged structure and expose it as a proper object
        # collection so HistManager auto-broadcasts weights and flattens per-object.
        self.events["MatchedJets"] = ak.zip({
            "pt": j_matched.pt,
            "genparton_pt": self.events["matched_genparton_pt"],
            "response_jec": self.events["matched_response_jec"],
            "response_regressed": self.events["matched_response_regressed"],
            "provenance": q_matched.provenance,
        }, with_name="PtEtaPhiMCandidate")

        # === Optional: save provenance and ΔR for further filtering
        self.events["matched_provenance"] = q_matched.provenance
        self.events["matched_dR"] = dR
        # j_matched[0] holds the reco jet matched to the first quark, if needed for inspection.

        self.events["mbb"] = (self.events["BJetGood"][pairs_sorted.slot0] + self.events["BJetGood"][pairs_sorted.slot1]).mass
        self.events["mbb_min"] = ak.firsts(self.events["mbb"])
        self.events["mass_w"] = 80
        self.events["mass_top"] = 170
        self.events["mass_w_minus"] = self.events["mass_w"]
        self.events["mass_antitop"] = self.events["mass_top"]

        # Compute GenJet pull vectors from GenPart
        pull_vectors = self.compute_pull_vectors_drmatched(self.events.GenJet, self.events.GenPart)

        # Map GenJet pull values to reco jets using Jet.genJetIdx
        genJetIdx = self.events["JetGood"]["genJetIdx"]
        valid_mask = genJetIdx >= 0
        clipped_idx = ak.where(valid_mask, genJetIdx, 0)

        # Assign pull vector components using Awkward-safe syntax
        self.events["JetGood"] = ak.with_field(
            self.events["JetGood"],
            ak.where(valid_mask, pull_vectors["pull_y"][clipped_idx], -999.),
            "pull_y"
        )
        self.events["JetGood"] = ak.with_field(
            self.events["JetGood"],
            ak.where(valid_mask, pull_vectors["pull_phi"][clipped_idx], -999.),
            "pull_phi"
        )
        self.events["JetGood"] = ak.with_field(
            self.events["JetGood"],
            ak.where(valid_mask, pull_vectors["pull_magnitude"][clipped_idx], -999.),
            "pull_magnitude"
        )
        self.events["JetGood"] = ak.with_field(
            self.events["JetGood"],
            ak.where(valid_mask, pull_vectors["pull_angle"][clipped_idx], -999.),
            "pull_angle"
        )
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

        # Get the indices that would sort JetGood by btag score (descending),
        # using the era-appropriate branch (self.btag_branch, resolved in
        # __init__), and use them to pick the leading 5 jets
        # ("1st_jet" ... "5th_jet") by btag ordering.
        btag_sorted_idx = ak.argsort(self.events["JetGood"][self.btag_branch], ascending=False)

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

        # =====================================================================
        # DEBUG: jet-selection sanity check (first 6 events)
        # Purely diagnostic — prints JetGood pt/eta/phi/mass/btag score
        # (in original NanoAOD order) alongside the btag-ordered selection
        # (using the era-appropriate self.btag_branch) that becomes
        # "1st_jet".."5th_jet" below. Does not touch any stored branch or
        # the physics logic.
        # =====================================================================
        n_debug = min(6, len(self.events))
        if n_debug > 0:
            jets_dbg = self.events["JetGood"][:n_debug]
            sel_pt_dbg = sorted_jets[:n_debug]
            sel_idx_dbg = sorted_jet_orig_idx[:n_debug]
            print(f"\n[DEBUG] Jet selection check (btag-ordered, branch={self.btag_branch}, year={self._year}) — first {n_debug} event(s)")
            for iev in range(n_debug):
                jev = jets_dbg[iev]
                print(f"  Event {iev}: nJetGood={len(jev)}")
                print(f"    pt (orig order)   : {ak.to_list(jev.pt)}")
                print(f"    eta               : {ak.to_list(jev.eta)}")
                print(f"    phi               : {ak.to_list(jev.phi)}")
                print(f"    mass              : {ak.to_list(jev.mass)}")
                print(f"    btag ({self.btag_branch}): {ak.to_list(jev[self.btag_branch])}")
                sel = sel_pt_dbg[iev][:5]
                print(f"    -> selected (btag-ordered) idx  : {ak.to_list(sel_idx_dbg[iev][:5])}")
                print(f"    -> selected (btag-ordered) pt   : {ak.to_list(sel.pt)}")
                print(f"    -> selected (btag-ordered) mass : {ak.to_list(sel.mass)}")

        bjet1 = vector.zip({
             "pt": self.events["1st_jet"]["pt"],
             "eta": self.events["1st_jet"]["eta"],
             "phi": self.events["1st_jet"]["phi"],
             "mass": self.events["1st_jet"]["mass"]
        })
        # Store px, py, pz, and E into the self.events structure
        self.events["1st_jet"] = ak.with_field(self.events["1st_jet"], bjet1.px, "px")
        self.events["1st_jet"] = ak.with_field(self.events["1st_jet"], bjet1.py, "py")
        self.events["1st_jet"] = ak.with_field(self.events["1st_jet"], bjet1.pz, "pz")
        self.events["1st_jet"] = ak.with_field(self.events["1st_jet"], bjet1.E, "E")

        # Similarly for the 2nd b-jet
        bjet2 = vector.zip({
             "pt": self.events["2nd_jet"]["pt"],
             "eta": self.events["2nd_jet"]["eta"],
             "phi": self.events["2nd_jet"]["phi"],
             "mass": self.events["2nd_jet"]["mass"]
        })

        self.events["2nd_jet"] = ak.with_field(self.events["2nd_jet"], bjet2.px, "px")
        self.events["2nd_jet"] = ak.with_field(self.events["2nd_jet"], bjet2.py, "py")
        self.events["2nd_jet"] = ak.with_field(self.events["2nd_jet"], bjet2.pz, "pz")
        self.events["2nd_jet"] = ak.with_field(self.events["2nd_jet"], bjet2.E, "E")


        
        bjet3 = vector.zip({
             "pt": self.events["3rd_jet"]["pt"],
             "eta": self.events["3rd_jet"]["eta"],
             "phi": self.events["3rd_jet"]["phi"],
             "mass": self.events["3rd_jet"]["mass"]
        })

        self.events["3rd_jet"] = ak.with_field(self.events["3rd_jet"], bjet3.px, "px")
        self.events["3rd_jet"] = ak.with_field(self.events["3rd_jet"], bjet3.py, "py")
        self.events["3rd_jet"] = ak.with_field(self.events["3rd_jet"], bjet3.pz, "pz")
        self.events["3rd_jet"] = ak.with_field(self.events["3rd_jet"], bjet3.E, "E")
        # Convert and store the four-momentum subfields for the 4th b-jet (2nd other b-jet)
        bjet4 = vector.zip({
             "pt": self.events["4th_jet"]["pt"],
             "eta": self.events["4th_jet"]["eta"],
             "phi": self.events["4th_jet"]["phi"],
             "mass": self.events["4th_jet"]["mass"]
        })

        self.events["4th_jet"] = ak.with_field(self.events["4th_jet"], bjet4.px, "px")
        self.events["4th_jet"] = ak.with_field(self.events["4th_jet"], bjet4.py, "py")
        self.events["4th_jet"] = ak.with_field(self.events["4th_jet"], bjet4.pz, "pz")
        self.events["4th_jet"] = ak.with_field(self.events["4th_jet"], bjet4.E, "E")

        bjet5 = vector.zip({
             "pt": self.events["5th_jet"]["pt"],
             "eta": self.events["5th_jet"]["eta"],
             "phi": self.events["5th_jet"]["phi"],
             "mass": self.events["5th_jet"]["mass"]
        })

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
        lepton_pos = vector.zip({
             "pt": self.events["lepton_pos"]["pt"],
             "eta": self.events["lepton_pos"]["eta"],
             "phi": self.events["lepton_pos"]["phi"],
             "mass": self.events["lepton_pos"]["mass"]
        })

        self.events["lepton_pos"] = ak.with_field(self.events["lepton_pos"], lepton_pos.px, "px")
        self.events["lepton_pos"] = ak.with_field(self.events["lepton_pos"], lepton_pos.py, "py")
        self.events["lepton_pos"] = ak.with_field(self.events["lepton_pos"], lepton_pos.pz, "pz")
        self.events["lepton_pos"] = ak.with_field(self.events["lepton_pos"], lepton_pos.E, "E")

        # Convert and store four-momentum for the negatively charged leptons
        lepton_neg = vector.zip({
             "pt": self.events["lepton_neg"]["pt"],
             "eta": self.events["lepton_neg"]["eta"],
             "phi": self.events["lepton_neg"]["phi"],
             "mass": self.events["lepton_neg"]["mass"]
        })

        self.events["lepton_neg"] = ak.with_field(self.events["lepton_neg"], lepton_neg.px, "px")
        self.events["lepton_neg"] = ak.with_field(self.events["lepton_neg"], lepton_neg.py, "py")
        self.events["lepton_neg"] = ak.with_field(self.events["lepton_neg"], lepton_neg.pz, "pz")
        self.events["lepton_neg"] = ak.with_field(self.events["lepton_neg"], lepton_neg.E, "E")
        self.events["MET_few"] = ak.Array({
                             "pt": self.events["PuppiMET"]["pt"],
                             "phi": self.events["PuppiMET"]["phi"]
                             })

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
        bscore = ak.values_astype(self.events["JetGood"][self.btag_branch], np.float64)
        # max b-tag score in event
        self.events["max_btag"] = ak.values_astype(ak.max(bscore, axis=1), np.float64)
        self.events["nBJets"] = ak.num(self.events["BJetGood"])

        # Single btag-descending ordering, reused below for all btag-rank
        # and "two most b-like jets" quantities (DNN features).
        idx_bdesc = ak.argsort(bscore, ascending=False)
        b_sorted = bscore[idx_bdesc]
        jets_bsorted = self.events["JetGood"][idx_bdesc]

        # ΔR between the two most b-like jets
        self.events["dR_top2_btag"] = ak.values_astype(
            ak.firsts(jets_bsorted[:, 0:1].delta_r(jets_bsorted[:, 1:2])), np.float64
        )

        # pT(jj) for the pair whose mass is closest to 125 GeV
        self.events["ptjj_at_mH"] = ak.values_astype(
            ak.where(has_ge2, ak.firsts(ptjj[idx_closest]), np.nan), np.float64
        )

        # Fill missing ranks with 0.0 when an event has <4 jets
        self.events["btag_rank1"] = ak.fill_none(ak.firsts(b_sorted[:, 0:1]), 0.0)
        self.events["btag_rank2"] = ak.fill_none(ak.firsts(b_sorted[:, 1:2]), 0.0)
        self.events["btag_rank3"] = ak.fill_none(ak.firsts(b_sorted[:, 2:2+1]), 0.0)
        self.events["btag_rank4"] = ak.fill_none(ak.firsts(b_sorted[:, 3:3+1]), 0.0)

        # sum of top-4 b-tag scores
        self.events["sum4_btag"] = ak.values_astype(ak.sum(b_sorted[:, :4], axis=1), np.float64)

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

        # "solutions_number" is per-combination (jagged: one entry per jet-pair
        # hypothesis tried), not one value per event. "filtered_solutions_number"
        # is the histogram-safe, reduced version — total number of solutions
        # found across all combinations in the event (consistent with the
        # total length of all_higgs_masses_per_event / all_weights_per_event
        # per event). It was previously a bare alias of the jagged array,
        # which is what crashed the "Solutions_Number" histogram (coll="events"
        # axis expects exactly one value per event).
        self.events["filtered_solutions_number"] = ak.values_astype(
            ak.fill_none(ak.sum(self.events["solutions_number"], axis=1), 0),
            np.int64,
        )

        
        props_maxweight = compute_jet_pair_properties_all_events(
            self.events, j_matched, q_matched, prefix="max_weight", btag_branch=self.btag_branch
        )
        self.events["correct_match_maxweight"]             = ak.firsts(ak.Array(props_maxweight["correct_matches"]))
        # NOTE: compute_jet_pair_properties_all_events() returns these as a
        # 0-or-1-entry list per event (one candidate "chosen pair" hit, or
        # none). "correct_matches" above is reduced with ak.firsts() -- the
        # rest were being stored bare, which is what the SHAPE GUARD below
        # catches (var * float64 instead of one value/event). Reduced the
        # same way, with NaN for events with no hit, consistent with the
        # existing pt_jet2 / mbb_btag_top2 pattern in this file.
        self.events["massreco_maxweight_correct_higgs_mass"] = ak.fill_none(ak.firsts(ak.Array(props_maxweight["massreco_chosen_pair_correct_higgs_mass"])), np.nan)
        self.events["massreco_maxweight_higgs_mass"]       = ak.fill_none(ak.firsts(ak.Array(props_maxweight["massreco_chosen_pair_all_higgs_mass"])), np.nan)
        self.events["massreco_maxweight_wrong_higgs_mass"] = ak.fill_none(ak.firsts(ak.Array(props_maxweight["massreco_chosen_pair_wrong_higgs_mass"])), np.nan)
        self.events["massreco_maxweight_correct_dr"]       = ak.fill_none(ak.firsts(ak.Array(props_maxweight["massreco_chosen_pair_correct_dr"])), np.nan)
        self.events["massreco_maxweight_wrong_dr"]         = ak.fill_none(ak.firsts(ak.Array(props_maxweight["massreco_chosen_pair_wrong_dr"])), np.nan)
        self.events["massreco_maxweight_wrong_higgs_mass_2_jets"] = ak.fill_none(ak.firsts(ak.Array(props_maxweight["massreco_chosen_pair_wrong_higgs_mass_2_jets"])), np.nan)
        self.events["massreco_maxweight_wrong_higgs_mass_1_jets"] = ak.fill_none(ak.firsts(ak.Array(props_maxweight["massreco_chosen_pair_wrong_higgs_mass_1_jets"])), np.nan)
        # =====================================================================
        # Step 2: Run DR criterion — selects best combination per event
        #         Stores all per-rank DR and Higgs mass variables for plots
        # =====================================================================
        dr_res = dr_criterion_for_max_weight(self.events)
        # --- Rejected-from-rank variables ---
        self.events["higgs_mass_rank1_rejected"] = ak.Array(dr_res["higgs_mass_rank1_rejected"])
        self.events["higgs_mass_rank2_rejected"] = ak.Array(dr_res["higgs_mass_rank2_rejected"])
        self.events["higgs_mass_rank3_rejected"] = ak.Array(dr_res["higgs_mass_rank3_rejected"])
        self.events["higgs_mass_rank4_rejected"] = ak.Array(dr_res["higgs_mass_rank4_rejected"])
        self.events["dr_rank1_rejected"]         = ak.Array(dr_res["dr_rank1_rejected"])
        self.events["dr_rank2_rejected"]         = ak.Array(dr_res["dr_rank2_rejected"])
        self.events["dr_rank3_rejected"]         = ak.Array(dr_res["dr_rank3_rejected"])
        self.events["dr_rank4_rejected"]         = ak.Array(dr_res["dr_rank4_rejected"])

        # Original DR criterion outputs (used downstream as the chosen combination)
        self.events["massreco_chosen_pair_higgs_mass"] = ak.Array(dr_res["massreco_chosen_pair_higgs_mass"])
        self.events["massreco_chosen_pair_top_mass"]   = ak.Array(dr_res["massreco_chosen_pair_top_mass"])
        self.events["massreco_chosen_pair_W_mass"]     = ak.Array(dr_res["massreco_chosen_pair_W_mass"])
        self.events["massreco_chosen_pair_ttH_mass"]   = ak.Array(dr_res["massreco_chosen_pair_ttH_mass"])
        self.events["massreco_chosen_pair_combination"] = ak.Array(dr_res["massreco_chosen_pair_combination"])

        # New variables for DR plots
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
        # New: chosen-from-rank variables (filled only for events where that rank was selected)
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

        # Store dr combination fields so compute_jet_pair_properties_all_events
        # can be called with prefix="dr" — mirrors the max_weight_* fields
        self.events["dr_combination"] = ak.Array(dr_res["massreco_chosen_pair_combination"])
        self.events["dr_higgs_mass"]  = ak.Array(dr_res["massreco_chosen_pair_higgs_mass"])
        self.events["dr_top_mass"]    = ak.Array(dr_res["massreco_chosen_pair_top_mass"])
        self.events["dr_W_mass"]      = ak.Array(dr_res["massreco_chosen_pair_W_mass"])
        self.events["dr_ttH_mass"]    = ak.Array(dr_res["massreco_chosen_pair_ttH_mass"])

        # =====================================================================
        # Step 3: Run compute_jet_pair_properties_all_events with DR combination
        #         Gives correct_match_dr for plots + all the main analysis quantities
        # =====================================================================
        props = compute_jet_pair_properties_all_events(
            self.events, j_matched, q_matched, prefix="dr", btag_branch=self.btag_branch
        )

        # correct_match_dr for DR plot comparison
        self.events["correct_match_dr"] = ak.firsts(ak.Array(props["correct_matches"]))
        # Main analysis quantities (all based on DR-criterion combination)
        self.events["numerator"]           = ak.Array(props["numerators"])
        self.events["denominator"]         = ak.Array(props["denominators"])
        self.events["correct_match"]       = ak.Array(props["correct_matches"])
        self.events["correct_agreement"]   = ak.Array(props["correct_agreements"])
        # Same reduction issue as props_maxweight above -- these come back as
        # 0-or-1-entry lists per event from compute_jet_pair_properties_all_events().
        # Only the fields actually bound to a coll="events" histogram axis
        # (checked by the SHAPE GUARD below) are fixed here; the remaining
        # props[...] fields further down (top_mass/W_mass/ttH_mass/jet_phi/
        # jet_eta/jet_mult/delta_angle/delta_magn/pull_*/dr_top/lep_*/
        # pt_assymetry, plus correct_match/numerator/denominator/
        # correct_agreement above) are left as-is since nothing histograms
        # them yet -- apply the same ak.fill_none(ak.firsts(...), ...)
        # treatment to any of those before adding them to a histogram too.
        self.events["higgs_mass_truth_jets"] = ak.fill_none(ak.firsts(ak.Array(props["higgs_mass_truth_jets"])), np.nan)

        self.events["massreco_chosen_pair_all_higgs_mass"]          = ak.fill_none(ak.firsts(ak.Array(props["massreco_chosen_pair_all_higgs_mass"])), np.nan)
        self.events["massreco_chosen_pair_correct_higgs_mass"]      = ak.fill_none(ak.firsts(ak.Array(props["massreco_chosen_pair_correct_higgs_mass"])), np.nan)
        self.events["massreco_chosen_pair_wrong_higgs_mass"]        = ak.fill_none(ak.firsts(ak.Array(props["massreco_chosen_pair_wrong_higgs_mass"])), np.nan)
        self.events["massreco_chosen_pair_wrong_higgs_mass_2_jets"] = ak.fill_none(ak.firsts(ak.Array(props["massreco_chosen_pair_wrong_higgs_mass_2_jets"])), np.nan)
        self.events["massreco_chosen_pair_wrong_higgs_mass_1_jets"] = ak.fill_none(ak.firsts(ak.Array(props["massreco_chosen_pair_wrong_higgs_mass_1_jets"])), np.nan)
        self.events["massreco_chosen_pair_correct_top_mass"]        = ak.Array(props["massreco_chosen_pair_correct_top_mass"])
        self.events["massreco_chosen_pair_wrong_top_mass"]          = ak.Array(props["massreco_chosen_pair_wrong_top_mass"])
        self.events["massreco_chosen_pair_correct_W_mass"]          = ak.Array(props["massreco_chosen_pair_correct_W_mass"])
        self.events["massreco_chosen_pair_wrong_W_mass"]            = ak.Array(props["massreco_chosen_pair_wrong_W_mass"])
        self.events["massreco_chosen_pair_correct_ttH_mass"]        = ak.Array(props["massreco_chosen_pair_correct_ttH_mass"])
        self.events["massreco_chosen_pair_wrong_ttH_mass"]          = ak.Array(props["massreco_chosen_pair_wrong_ttH_mass"])
        # massreco_chosen_pair_correct_jet_pt is NOT assigned as a flat
        # coll="events" field -- it's a genuine 2-jet-per-event quantity
        # (see CorrectChosenPairJets.pt below), so it's exposed there instead.
        self.events["massreco_chosen_pair_wrong_jet_pt"]            = ak.Array(props["massreco_chosen_pair_wrong_jet_pt"])
        self.events["massreco_chosen_pair_correct_jet_phi"]         = ak.Array(props["massreco_chosen_pair_correct_jet_phi"])
        self.events["massreco_chosen_pair_wrong_jet_phi"]           = ak.Array(props["massreco_chosen_pair_wrong_jet_phi"])
        self.events["massreco_chosen_pair_correct_jet_eta"]         = ak.Array(props["massreco_chosen_pair_correct_jet_eta"])
        self.events["massreco_chosen_pair_wrong_jet_eta"]           = ak.Array(props["massreco_chosen_pair_wrong_jet_eta"])
        self.events["massreco_chosen_pair_correct_jet_mult"]        = ak.Array(props["massreco_chosen_pair_correct_jet_mult"])
        self.events["massreco_chosen_pair_wrong_jet_mult"]          = ak.Array(props["massreco_chosen_pair_wrong_jet_mult"])
        self.events["massreco_chosen_pair_correct_dr"]              = ak.fill_none(ak.firsts(ak.Array(props["massreco_chosen_pair_correct_dr"])), np.nan)
        self.events["massreco_chosen_pair_wrong_dr"]                = ak.fill_none(ak.firsts(ak.Array(props["massreco_chosen_pair_wrong_dr"])), np.nan)
        self.events["massreco_chosen_pair_correct_delta_angle"]     = ak.Array(props["massreco_chosen_pair_correct_delta_angle"])
        self.events["massreco_chosen_pair_wrong_delta_angle"]       = ak.Array(props["massreco_chosen_pair_wrong_delta_angle"])
        self.events["massreco_chosen_pair_correct_delta_magn"]      = ak.Array(props["massreco_chosen_pair_correct_delta_magn"])
        self.events["massreco_chosen_pair_wrong_delta_magn"]        = ak.Array(props["massreco_chosen_pair_wrong_delta_magn"])
        self.events["massreco_chosen_pair_correct_pull_magn"]       = ak.Array(props["massreco_chosen_pair_correct_pull_magn"])
        self.events["massreco_chosen_pair_wrong_pull_magn"]         = ak.Array(props["massreco_chosen_pair_wrong_pull_magn"])
        self.events["massreco_chosen_pair_correct_pull_angle"]      = ak.Array(props["massreco_chosen_pair_correct_pull_angle"])
        self.events["massreco_chosen_pair_wrong_pull_angle"]        = ak.Array(props["massreco_chosen_pair_wrong_pull_angle"])
        self.events["massreco_chosen_pair_correct_dr_top"]          = ak.Array(props["massreco_chosen_pair_correct_dr_top"])
        self.events["massreco_chosen_pair_wrong_dr_top"]            = ak.Array(props["massreco_chosen_pair_wrong_dr_top"])
        self.events["massreco_chosen_pair_correct_lep_neg"]         = ak.Array(props["massreco_chosen_pair_correct_lep_neg"])
        self.events["massreco_chosen_pair_wrong_lep_neg"]           = ak.Array(props["massreco_chosen_pair_wrong_lep_neg"])
        self.events["massreco_chosen_pair_correct_lep_pos"]         = ak.Array(props["massreco_chosen_pair_correct_lep_pos"])
        self.events["massreco_chosen_pair_wrong_lep_pos"]           = ak.Array(props["massreco_chosen_pair_wrong_lep_pos"])
        self.events["massreco_chosen_pair_correct_pt_assymetry"]    = ak.Array(props["massreco_chosen_pair_correct_pt_assymetry"])
        self.events["massreco_chosen_pair_wrong_pt_assymetry"]      = ak.Array(props["massreco_chosen_pair_wrong_pt_assymetry"])



        # =====================================================================
        # Step 4: Per-rank truth matching for all 4 ranks independently
        #         Gives unconditional efficiency per rank and rejected rank correctness
        # =====================================================================
        props_per_rank = compute_jet_pair_properties_per_rank(
            self.events, j_matched, q_matched
        )
        self.events["correct_match_rank1"] = ak.Array(props_per_rank["correct_match_rank1"])
        self.events["correct_match_rank2"] = ak.Array(props_per_rank["correct_match_rank2"])
        self.events["correct_match_rank3"] = ak.Array(props_per_rank["correct_match_rank3"])
        self.events["correct_match_rank4"] = ak.Array(props_per_rank["correct_match_rank4"])
        self.events["higgs_mass_rank1_correct"] = ak.Array(props_per_rank["higgs_mass_rank1_correct"])
        self.events["higgs_mass_rank2_correct"] = ak.Array(props_per_rank["higgs_mass_rank2_correct"])
        self.events["higgs_mass_rank3_correct"] = ak.Array(props_per_rank["higgs_mass_rank3_correct"])
        self.events["higgs_mass_rank4_correct"] = ak.Array(props_per_rank["higgs_mass_rank4_correct"])
        self.events["higgs_mass_rank1_wrong"]   = ak.Array(props_per_rank["higgs_mass_rank1_wrong"])
        self.events["higgs_mass_rank2_wrong"]   = ak.Array(props_per_rank["higgs_mass_rank2_wrong"])
        self.events["higgs_mass_rank3_wrong"]   = ak.Array(props_per_rank["higgs_mass_rank3_wrong"])
        self.events["higgs_mass_rank4_wrong"]   = ak.Array(props_per_rank["higgs_mass_rank4_wrong"])
        # Pull vector collections for histogramming
        # NOTE on "pt": massreco_chosen_pair_correct_jet_pt is a genuine
        # 2-entry-per-event list [pt(jet1), pt(jet2)] (single-nested, unlike
        # pull_angle/pull_magn which are double-nested [[v1, v2]]). Do NOT
        # run it through ak.firsts() -- that strips the 2-jet dimension
        # itself and silently keeps only jet1's pt. Feeding it straight into
        # this per-event 2-object collection (instead of a flat coll="events"
        # scalar) is what preserves both jets.
        self.events["CorrectChosenPairJets"] = ak.zip({
            "pull_angle": ak.fill_none(
                ak.firsts(props["massreco_chosen_pair_correct_pull_angle"]), np.nan
            ),
            "pull_magn": ak.fill_none(
                ak.firsts(props["massreco_chosen_pair_correct_pull_magn"]), np.nan
            ),
            "pt": ak.Array(props["massreco_chosen_pair_correct_jet_pt"]),
        }, with_name="PtEtaPhiMCandidate")

        self.events["WrongChosenPairJets"] = ak.zip({
            "pull_angle": ak.fill_none(
                ak.firsts(props["massreco_chosen_pair_wrong_pull_angle"]), np.nan
            ),
            "pull_magn": ak.fill_none(
                ak.firsts(props["massreco_chosen_pair_wrong_pull_magn"]), np.nan
            ),
        }, with_name="PtEtaPhiMCandidate")

        print(ak.type(self.events["CorrectChosenPairJets"].pull_angle))

        # =====================================================================
        # Step 5: Acceptance and classification study
        # =====================================================================
        param = compute_jet_pair_properties_all_events_4(self.events, j_matched, q_matched, btag_branch=self.btag_branch)

        # Same 0-or-1-entry-per-event issue as compute_jet_pair_properties_all_events()
        # above. "pair_in_the_4leading*" are the categorical bins=4 axes
        # (values 0-3); reduced to -1 for events with no truth pair so they
        # fall outside the [0,4) range and are simply not counted, since
        # overflow=underflow=False on those histogram axes.
        self.events["mass_reco_had_a_solution"]      = ak.Array(param["mass_reco_had_a_solution"])
        self.events["pair_in_the_4leading"]          = ak.fill_none(ak.firsts(ak.Array(param["pair_in_the_4leading"])), -1)
        self.events["pair_in_the_4leadingbtagscore"] = ak.fill_none(ak.firsts(ak.Array(param["pair_in_the_4leadingbtagscore"])), -1)
        self.events["pair_phi_truth"]                = ak.Array(param["pair_phi_truth"])
        self.events["pair_eta_truth"]                = ak.Array(param["pair_eta_truth"])
        self.events["pair_mass_truth"]               = ak.fill_none(ak.firsts(ak.Array(param["pair_mass_truth"])), np.nan)
        self.events["pair_dr_truth"]                 = ak.fill_none(ak.firsts(ak.Array(param["pair_dr_truth"])), np.nan)
        # Debug: quick sanity check on how many events had no combinatoric solution
        n_mw_none = sum(1 for x in ak.to_list(self.events["max_weight_combination"]) if x is None)
        n_dr_none = sum(1 for x in ak.to_list(self.events["dr_combination"]) if x is None)
        print(f"max_weight_combination None: {n_mw_none}")
        print(f"dr_combination None: {n_dr_none}")

        # =====================================================================
        # Step 6: Selected/rejected × correct/wrong mass per rank
        # =====================================================================
        props_srcw = compute_selected_rejected_correct_wrong(
            self.events, props, props_per_rank
        )
        for r in [1, 2, 3, 4]:
            self.events[f"higgs_mass_rank{r}_selected_correct"] = ak.Array(props_srcw[f"higgs_mass_rank{r}_selected_correct"])
            self.events[f"higgs_mass_rank{r}_selected_wrong"]   = ak.Array(props_srcw[f"higgs_mass_rank{r}_selected_wrong"])
            self.events[f"higgs_mass_rank{r}_rejected_correct"] = ak.Array(props_srcw[f"higgs_mass_rank{r}_rejected_correct"])
            self.events[f"higgs_mass_rank{r}_rejected_wrong"]   = ak.Array(props_srcw[f"higgs_mass_rank{r}_rejected_wrong"])
        self.events["higgs_mass_fallback_selected_correct"] = ak.Array(props_srcw["higgs_mass_fallback_selected_correct"])
        self.events["higgs_mass_fallback_selected_wrong"]   = ak.Array(props_srcw["higgs_mass_fallback_selected_wrong"])

        # =====================================================================
        # DEBUG: mass-reconstruction result check (first 6 events)
        # Purely diagnostic — for each event shows which JetGood jets are
        # truth-matched to the Higgs (provenance == 1), the jet pair chosen
        # by the DR criterion and by the max-weight criterion, their
        # reconstructed Higgs masses, and whether each was correct/wrong
        # relative to truth. Does not touch any stored branch or the
        # physics logic.
        # =====================================================================
        n_debug2 = min(6, len(self.events))
        if n_debug2 > 0:
            print(f"\n[DEBUG] Mass-reco result check — first {n_debug2} event(s)")
            prov_dbg = q_matched.provenance[:n_debug2]
            jpt_dbg = j_matched.pt[:n_debug2]
            jeta_dbg = j_matched.eta[:n_debug2]
            for iev in range(n_debug2):
                prov_iev = ak.to_list(prov_dbg[iev])
                pt_iev = ak.to_list(jpt_dbg[iev])
                eta_iev = ak.to_list(jeta_dbg[iev])
                higgs_truth_jets = [
                    (round(pt, 1), round(eta, 2))
                    for p, pt, eta in zip(prov_iev, pt_iev, eta_iev)
                    if p == 1
                ]
                dr_combo = ak.to_list(self.events["dr_combination"][iev])
                mw_combo = ak.to_list(self.events["max_weight_combination"][iev])
                print(f"  Event {iev}:")
                print(f"    truth Higgs-jet(s) [pt, eta] (provenance==1) : {higgs_truth_jets}")
                print(f"    DR criterion  -> combination                 : {dr_combo}")
                print(f"    DR criterion  -> reco Higgs mass              : {self.events['dr_higgs_mass'][iev]}")
                print(f"    DR criterion  -> correct match?               : {bool(self.events['correct_match_dr'][iev])}")
                print(f"    DR criterion  -> rank used                    : {self.events['rank_used'][iev]}")
                print(f"    max-weight    -> combination                 : {mw_combo}")
                print(f"    max-weight    -> reco Higgs mass              : {self.events['max_weight_higgs_mass'][iev]}")
                print(f"    max-weight    -> correct match?               : {bool(self.events['correct_match_maxweight'][iev])}")

        # =====================================================================
        # SHAPE GUARD: fail fast, with the real field name, instead of the
        # cryptic PocketCoffea crash in hist_manager.fill_histograms
        # ("boolean index did not match indexed array along dimension 0").
        # That crash happens when a "coll=events" histogram axis is fed a
        # jagged (>1 entry per event) array instead of one value per event.
        # Every field below is used as a coll="events" axis in
        # mass_reco_histograms_base — each MUST be exactly one entry per
        # event (a None is fine, a list is not). solve_ttbar_dilepton() and
        # the DR-criterion helpers (custom_function.py /
        # mass_reco_functions_mc.py) run over jets ordered by btag score
        # (self.btag_branch, era-dependent) here; this check pinpoints
        # exactly which field and how many events are affected if any of
        # them returns a nested/jagged result, without changing any values.
        # =====================================================================
        _histogram_event_fields = ["Aplanarity", "C_jet", "D_jet", "H4", "HT", "MET_phi", "MET_pt", "Njj_higgs_like", "centrality", "correct_match_dr", "correct_match_maxweight", "correct_match_rank1", "correct_match_rank2", "correct_match_rank3", "correct_match_rank4", "dR_b1_b2", "dR_top2_btag", "dRjj_min", "deta_max", "dnn_score", "filtered_solutions_number", "higgs_mass_fallback_selected_correct", "higgs_mass_fallback_selected_wrong", "higgs_mass_rank1_correct", "higgs_mass_rank1_rejected_correct", "higgs_mass_rank1_rejected_wrong", "higgs_mass_rank1_selected_correct", "higgs_mass_rank1_selected_wrong", "higgs_mass_rank1_wrong", "higgs_mass_rank2_correct", "higgs_mass_rank2_rejected_correct", "higgs_mass_rank2_rejected_wrong", "higgs_mass_rank2_selected_correct", "higgs_mass_rank2_selected_wrong", "higgs_mass_rank2_wrong", "higgs_mass_rank3_correct", "higgs_mass_rank3_rejected_correct", "higgs_mass_rank3_rejected_wrong", "higgs_mass_rank3_selected_correct", "higgs_mass_rank3_selected_wrong", "higgs_mass_rank3_wrong", "higgs_mass_rank4_correct", "higgs_mass_rank4_rejected_correct", "higgs_mass_rank4_rejected_wrong", "higgs_mass_rank4_selected_correct", "higgs_mass_rank4_selected_wrong", "higgs_mass_rank4_wrong", "higgs_mass_truth_jets", "m_higgs_like_jj", "m_jjj_maxpT", "massreco_chosen_pair_all_higgs_mass", "massreco_chosen_pair_correct_dr", "massreco_chosen_pair_correct_higgs_mass", "massreco_chosen_pair_higgs_mass", "massreco_chosen_pair_wrong_dr", "massreco_chosen_pair_wrong_higgs_mass", "massreco_chosen_pair_wrong_higgs_mass_1_jets", "massreco_chosen_pair_wrong_higgs_mass_2_jets", "massreco_maxweight_correct_dr", "massreco_maxweight_correct_higgs_mass", "massreco_maxweight_higgs_mass", "massreco_maxweight_wrong_dr", "massreco_maxweight_wrong_higgs_mass", "massreco_maxweight_wrong_higgs_mass_1_jets", "massreco_maxweight_wrong_higgs_mass_2_jets", "max_weight", "max_weight_higgs_mass", "mbb_btag_top2", "pT_higgs_like", "pair_dr_truth", "pair_in_the_4leading", "pair_in_the_4leadingbtagscore", "pair_mass_truth", "pt_jet2", "pt_ratio_b1_b2", "ptjj_at_dRmin", "ptjj_at_mH", "rank_used", "sum4_btag", "sum_m_jets", "sum_pt_b1b2", "sum_pt_b1b2_over_HT"]
        n_events = len(self.events)
        bad_fields = []
        for fname in _histogram_event_fields:
            if fname not in ak.fields(self.events):
                continue
            farr = self.events[fname]
            ftype = str(ak.type(farr))
            is_jagged = "var *" in ftype
            wrong_len = len(farr) != n_events
            if is_jagged or wrong_len:
                bad_fields.append((fname, ftype, len(farr)))

        if bad_fields:
            details = "\n".join(
                f"    - '{name}': type={ftype}, len={flen} (expected {n_events}, "
                f"flat one-value-per-event)"
                for name, ftype, flen in bad_fields
            )
            raise RuntimeError(
                "[SHAPE GUARD] The following field(s) bound for coll='events' "
                "histogram axes are NOT flat/one-value-per-event, which is what "
                "crashes PocketCoffea's fill_histograms downstream:\n"
                f"{details}\n"
                "These come from solve_ttbar_dilepton()/DR-criterion helpers in "
                "custom_function.py / mass_reco_functions_mc.py. Check those "
                "functions for an assumption tied to the previous btag-ordered "
                "jet selection that no longer holds under pT ordering."
            )