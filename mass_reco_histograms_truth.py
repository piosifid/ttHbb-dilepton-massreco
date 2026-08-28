from pocket_coffea.parameters.histograms import Axis, HistConf

# Remove this entire import in config when running on data
# variables = {**mass_reco_histograms_base, **mass_reco_histograms_dr_study}
# remove: **mass_reco_histograms_truth

_MASS_BINS = [60,80,90,100,110,120,130,140,150,170,210,300,450,600]
_DR_BINS   = 50
_DR_START  = 0
_DR_STOP   = 5

mass_reco_histograms_truth = {
    # =========================================================================
    # Truth matching — correct/wrong/all Higgs mass
    # =========================================================================
    # === A7b — max weight wrong higgs mass ===
    "massreco_maxweight_wrong_higgs_mass": HistConf([
        Axis(coll="events", field="massreco_maxweight_wrong_higgs_mass", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass max weight -- wrong match [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_maxweight_correct_higgs_mass": HistConf([
        Axis(coll="events", field="massreco_maxweight_correct_higgs_mass", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Reco Higgs Mass -- any wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_maxweight_wrong_higgs_mass_2_jets": HistConf([
        Axis(coll="events", field="massreco_maxweight_wrong_higgs_mass_2_jets", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Reco Higgs Mass -- both jets wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_maxweight_wrong_higgs_mass_1_jets": HistConf([
        Axis(coll="events", field="massreco_maxweight_wrong_higgs_mass_1_jets", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Reco Higgs Mass -- one jet wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "massreco_maxweight_wrong_higgs_mass_1": HistConf([
        Axis(coll="events", field="massreco_maxweight_wrong_higgs_mass", pos=None,
             bins=60, start=0, stop=600,
             label="Higgs mass max weight -- wrong match [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_maxweight_correct_higgs_mass_1": HistConf([
        Axis(coll="events", field="massreco_maxweight_correct_higgs_mass", pos=None,
             bins=60, start=0, stop=600,
             label="Reco Higgs Mass -- any wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_maxweight_wrong_higgs_mass_2_jets_1": HistConf([
        Axis(coll="events", field="massreco_maxweight_wrong_higgs_mass_2_jets", pos=None,
             bins=60, start=0, stop=600,
             label="Reco Higgs Mass -- both jets wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_maxweight_wrong_higgs_mass_1_jets_1": HistConf([
        Axis(coll="events", field="massreco_maxweight_wrong_higgs_mass_1_jets", pos=None,
             bins=60, start=0, stop=600,
             label="Reco Higgs Mass -- one jet wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    # === A7 — max weight correct/wrong DR ===
    "massreco_maxweight_correct_dr": HistConf([
        Axis(coll="events", field="massreco_maxweight_correct_dr",
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R_{bb}$ max weight -- correct match",
             overflow=True, underflow=True)
    ]),
    "massreco_maxweight_wrong_dr": HistConf([
        Axis(coll="events", field="massreco_maxweight_wrong_dr",
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R_{bb}$ max weight -- wrong match",
             overflow=True, underflow=True)
    ]),
    # === A8 — per-rank correct match and higgs mass ===
    "correct_match_rank1": HistConf([
        Axis(coll="events", field="correct_match_rank1", pos=None,
             bins=4, start=0, stop=4,
             label="Correct match rank 1", overflow=False, underflow=False)
    ]),
    "correct_match_rank2": HistConf([
        Axis(coll="events", field="correct_match_rank2", pos=None,
             bins=4, start=0, stop=4,
             label="Correct match rank 2", overflow=False, underflow=False)
    ]),
    "correct_match_rank3": HistConf([
        Axis(coll="events", field="correct_match_rank3", pos=None,
             bins=4, start=0, stop=4,
             label="Correct match rank 3", overflow=False, underflow=False)
    ]),
    "correct_match_rank4": HistConf([
        Axis(coll="events", field="correct_match_rank4", pos=None,
             bins=4, start=0, stop=4,
             label="Correct match rank 4", overflow=False, underflow=False)
    ]),
    "higgs_mass_rank1_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 1 -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank2_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 2 -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank3_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 3 -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank4_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 4 -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank1_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 1 -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank2_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 2 -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank3_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 3 -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank4_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 4 -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_correct_higgs_mass": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_correct_higgs_mass", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Reco Higgs Mass -- correct match [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_correct_higgs_mass_1": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_correct_higgs_mass", pos=None,
             bins=60, start=0, stop=600,
             label="Reco Higgs Mass -- correct match [GeV]",
             overflow=True, underflow=True)
    ]),
    "massreco_maxweight_correct_dr": HistConf([
        Axis(coll="events", field="massreco_maxweight_correct_dr",
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R_{bb}$ max weight -- correct match",
             overflow=True, underflow=True)
    ]),
    "massreco_maxweight_wrong_dr": HistConf([
        Axis(coll="events", field="massreco_maxweight_wrong_dr",
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R_{bb}$ max weight -- wrong match",
             overflow=True, underflow=True)
    ]),
    "rank_vs_correct_match": HistConf([
        Axis(coll="events", field="rank_used",
             bins=6, start=-1, stop=5,
             label="Rank used by DR criterion"),
        Axis(coll="events", field="correct_match_dr",
             bins=4, start=0, stop=4,
             label="Correct match (DR criterion)"),
    ]),
    "Massreco_chosen_pair_all_higgs_mass": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_all_higgs_mass", pos=None,
             bins=50, start=0, stop=600,
             label="Reco Higgs Mass -- all [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_wrong_higgs_mass": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_wrong_higgs_mass", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Reco Higgs Mass -- any wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_wrong_higgs_mass_2_jets": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_wrong_higgs_mass_2_jets", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Reco Higgs Mass -- both jets wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_wrong_higgs_mass_1_jets": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_wrong_higgs_mass_1_jets", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Reco Higgs Mass -- one jet wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_wrong_higgs_mass_1": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_wrong_higgs_mass", pos=None,
             bins=60, start=0, stop=600,
             label="Reco Higgs Mass -- any wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_wrong_higgs_mass_2_jets_1": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_wrong_higgs_mass_2_jets", pos=None,
             bins=60, start=0, stop=600,
             label="Reco Higgs Mass -- both jets wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_wrong_higgs_mass_1_jets_1": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_wrong_higgs_mass_1_jets", pos=None,
             bins=60, start=0, stop=600,
             label="Reco Higgs Mass -- one jet wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    # NOTE: coll="CorrectChosenPairJets" (not "events") -- this field holds
    # BOTH jets of the chosen pair (2 entries/event, built in workflow_mc.py
    # from massreco_chosen_pair_correct_jet_pt). A flat coll="events" axis
    # can't represent that; see CorrectChosenPairJets in workflow_mc.py.
    "Massreco_chosen_pair_correct_jet_pt": HistConf([
        Axis(coll="CorrectChosenPairJets", field="pt", pos=None,
             bins=50, start=0, stop=600,
             label="Correct jet pT [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_correct_dr": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_correct_dr",
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R_{bb}$ -- correct match",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_wrong_dr": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_wrong_dr",
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R_{bb}$ -- wrong match",
             overflow=True, underflow=True)
    ]),
    # =========================================================================
    # Max weight truth matching (for comparison with DR criterion)
    # =========================================================================
    "massreco_maxweight_higgs_mass": HistConf([
        Axis(coll="events", field="massreco_maxweight_higgs_mass", pos=None,
             bins=50, start=0, stop=600,
             label="Higgs mass max weight -- all [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_maxweight_correct_higgs_mass": HistConf([
        Axis(coll="events", field="massreco_maxweight_correct_higgs_mass", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass max weight -- correct match [GeV]",
             overflow=True, underflow=True)
    ]),
    "correct_match_maxweight": HistConf([
        Axis(coll="events", field="correct_match_maxweight", pos=None,
             bins=4, start=0, stop=4,
             label="Correct match (max weight only)",
             overflow=False, underflow=False)
    ]),
    "correct_match_dr": HistConf([
        Axis(coll="events", field="correct_match_dr", pos=None,
             bins=4, start=0, stop=4,
             label="Correct match (DR criterion)",
             overflow=False, underflow=False)
    ]),
    # =========================================================================
    # Acceptance study (from compute_jet_pair_properties_all_events_4)
    # =========================================================================
    "pair_in_the_4leadingbtagscore": HistConf([
        Axis(coll="events", field="pair_in_the_4leadingbtagscore", pos=None,
             bins=4, start=0, stop=4,
             label="Truth pair in top-4 btag",
             overflow=False, underflow=False)
    ]),
    "pair_in_the_4leading": HistConf([
        Axis(coll="events", field="pair_in_the_4leading", pos=None,
             bins=4, start=0, stop=4,
             label="Truth pair in top-4 pT",
             overflow=False, underflow=False)
    ]),
    "pair_mass_truth": HistConf([
        Axis(coll="events", field="pair_mass_truth", pos=None,
             bins=50, start=0, stop=600,
             label="Truth Higgs pair mass [GeV]",
             overflow=True, underflow=True)
    ]),
    "pair_dr_truth": HistConf([
        Axis(coll="events", field="pair_dr_truth", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ truth Higgs pair",
             overflow=True, underflow=True)
    ]),
    # =========================================================================
    # 2D histograms — truth vs reco Higgs mass
    # =========================================================================
    "truth_vs_reco_higgs_mass_dr": HistConf([
        Axis(coll="events", field="higgs_mass_truth_jets",
             bins=_MASS_BINS, start=0, stop=600,
             label="Truth Higgs mass [GeV]"),
        Axis(coll="events", field="massreco_chosen_pair_correct_higgs_mass",
             bins=_MASS_BINS, start=0, stop=600,
             label="Reco Higgs mass DR -- correct match [GeV]"),
    ]),
    "truth_vs_reco_higgs_mass_maxweight": HistConf([
        Axis(coll="events", field="higgs_mass_truth_jets",
             bins=_MASS_BINS, start=0, stop=600,
             label="Truth Higgs mass [GeV]"),
        Axis(coll="events", field="massreco_maxweight_correct_higgs_mass",
             bins=_MASS_BINS, start=0, stop=600,
             label="Reco Higgs mass max weight -- correct match [GeV]"),
    ]),
    # =========================================================================
    # A9 — Selected/rejected × correct/wrong Higgs mass per rank
    # To add to mass_reco_histograms_truth.py
    # =========================================================================
    # --- Rank 1 ---
    "higgs_mass_rank1_selected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_selected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 1 selected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank1_selected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_selected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 1 selected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank1_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_rejected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 1 rejected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank1_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_rejected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 1 rejected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    # --- Rank 2 ---
    "higgs_mass_rank2_selected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_selected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 2 selected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank2_selected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_selected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 2 selected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank2_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_rejected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 2 rejected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank2_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_rejected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 2 rejected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    # --- Rank 3 ---
    "higgs_mass_rank3_selected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_selected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 3 selected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank3_selected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_selected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 3 selected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank3_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_rejected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 3 rejected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank3_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_rejected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 3 rejected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    # --- Rank 4 ---
    "higgs_mass_rank4_selected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_selected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 4 selected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank4_selected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_selected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 4 selected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank4_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_rejected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 4 rejected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank4_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_rejected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass rank 4 rejected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    # --- Fallback ---
    "higgs_mass_fallback_selected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_fallback_selected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass fallback selected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_fallback_selected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_fallback_selected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass fallback selected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    # =========================================================================
    # A9 — 2D histograms: selected correct vs rejected correct per rank
    # Shows mass of correct combinations when selected vs when rejected
    # =========================================================================
    "rank1_selected_correct_vs_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_selected_correct",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 1 selected correct [GeV]"),
        Axis(coll="events", field="higgs_mass_rank1_rejected_correct",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 1 rejected correct [GeV]"),
    ]),
    "rank2_selected_correct_vs_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_selected_correct",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 2 selected correct [GeV]"),
        Axis(coll="events", field="higgs_mass_rank2_rejected_correct",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 2 rejected correct [GeV]"),
    ]),
    "rank3_selected_correct_vs_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_selected_correct",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 3 selected correct [GeV]"),
        Axis(coll="events", field="higgs_mass_rank3_rejected_correct",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 3 rejected correct [GeV]"),
    ]),
    "rank4_selected_correct_vs_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_selected_correct",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 4 selected correct [GeV]"),
        Axis(coll="events", field="higgs_mass_rank4_rejected_correct",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 4 rejected correct [GeV]"),
    ]),
    # =========================================================================
    # A9 — 2D histograms: selected wrong vs rejected wrong per rank
    # Shows mass of wrong combinations when selected vs when rejected
    # =========================================================================
    "rank1_selected_wrong_vs_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_selected_wrong",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 1 selected wrong [GeV]"),
        Axis(coll="events", field="higgs_mass_rank1_rejected_wrong",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 1 rejected wrong [GeV]"),
    ]),
    "rank2_selected_wrong_vs_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_selected_wrong",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 2 selected wrong [GeV]"),
        Axis(coll="events", field="higgs_mass_rank2_rejected_wrong",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 2 rejected wrong [GeV]"),
    ]),
    "rank3_selected_wrong_vs_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_selected_wrong",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 3 selected wrong [GeV]"),
        Axis(coll="events", field="higgs_mass_rank3_rejected_wrong",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 3 rejected wrong [GeV]"),
    ]),
    "rank4_selected_wrong_vs_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_selected_wrong",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 4 selected wrong [GeV]"),
        Axis(coll="events", field="higgs_mass_rank4_rejected_wrong",
             bins=_MASS_BINS, start=0, stop=600,
             label="Rank 4 rejected wrong [GeV]"),
    ]),
}

# =========================================================================
# Generic event-level fields — kept separate from mass_reco_histograms_truth
# NOTE: bins/ranges below are reasonable defaults inferred from variable
# name/type -- please check/adjust against your actual field ranges.
# =========================================================================
_histogram_event_fields = {
    "HT": HistConf([
        Axis(coll="events", field="HT", pos=None,
             bins=50, start=0, stop=3000,
             label="$H_T$ [GeV]",
             overflow=True, underflow=True)
    ]),
    "MET_pt": HistConf([
        Axis(coll="events", field="MET_pt", pos=None,
             bins=50, start=0, stop=600,
             label="MET $p_T$ [GeV]",
             overflow=True, underflow=True)
    ]),
    "MET_phi": HistConf([
        Axis(coll="events", field="MET_phi", pos=None,
             bins=50, start=-3.1416, stop=3.1416,
             label="MET $\\phi$",
             overflow=False, underflow=False)
    ]),
    "dnn_score": HistConf([
        Axis(coll="events", field="dnn_score", pos=None,
             bins=50, start=0, stop=1,
             label="DNN score",
             overflow=False, underflow=False)
    ]),
    "C_jet": HistConf([
        Axis(coll="events", field="C_jet", pos=None,
             bins=50, start=0, stop=1,
             label="Event shape $C$",
             overflow=False, underflow=False)
    ]),
    "D_jet": HistConf([
        Axis(coll="events", field="D_jet", pos=None,
             bins=50, start=0, stop=1,
             label="Event shape $D$",
             overflow=False, underflow=False)
    ]),
    "Aplanarity": HistConf([
        Axis(coll="events", field="Aplanarity", pos=None,
             bins=50, start=0, stop=0.5,
             label="Aplanarity",
             overflow=False, underflow=False)
    ]),
    "H4": HistConf([
        Axis(coll="events", field="H4", pos=None,
             bins=50, start=0, stop=1,
             label="Fox-Wolfram moment $H_4$",
             overflow=False, underflow=False)
    ]),
    "pt_jet2": HistConf([
        Axis(coll="events", field="pt_jet2", pos=None,
             bins=50, start=0, stop=600,
             label="Jet 2 $p_T$ [GeV]",
             overflow=True, underflow=True)
    ]),
    "dRjj_min": HistConf([
        Axis(coll="events", field="dRjj_min", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"min $\Delta R_{jj}$",
             overflow=True, underflow=True)
    ]),
    "deta_max": HistConf([
        Axis(coll="events", field="deta_max", pos=None,
             bins=50, start=0, stop=6,
             label=r"max $|\Delta\eta_{jj}|$",
             overflow=True, underflow=True)
    ]),
    "m_higgs_like_jj": HistConf([
        Axis(coll="events", field="m_higgs_like_jj", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs-like $m_{jj}$ [GeV]",
             overflow=True, underflow=True)
    ]),
    "ptjj_at_dRmin": HistConf([
        Axis(coll="events", field="ptjj_at_dRmin", pos=None,
             bins=50, start=0, stop=600,
             label="$p_{T,jj}$ at min $\\Delta R$ [GeV]",
             overflow=True, underflow=True)
    ]),
    "Njj_higgs_like": HistConf([
        Axis(coll="events", field="Njj_higgs_like", pos=None,
             bins=10, start=0, stop=10,
             label="N Higgs-like jj pairs",
             overflow=True, underflow=False)
    ]),
    "m_jjj_maxpT": HistConf([
        Axis(coll="events", field="m_jjj_maxpT", pos=None,
             bins=50, start=0, stop=1000,
             label="Trijet mass at max $p_T$ [GeV]",
             overflow=True, underflow=True)
    ]),
    "sum4_btag": HistConf([
        Axis(coll="events", field="sum4_btag", pos=None,
             bins=50, start=0, stop=4,
             label="Sum of top-4 b-tag scores",
             overflow=False, underflow=False)
    ]),
    "sum_pt_b1b2": HistConf([
        Axis(coll="events", field="sum_pt_b1b2", pos=None,
             bins=50, start=0, stop=1000,
             label="$p_T(b_1)+p_T(b_2)$ [GeV]",
             overflow=True, underflow=True)
    ]),
    "pt_ratio_b1_b2": HistConf([
        Axis(coll="events", field="pt_ratio_b1_b2", pos=None,
             bins=50, start=0, stop=1,
             label="$p_T(b_2)/p_T(b_1)$",
             overflow=False, underflow=False)
    ]),
    "mbb_btag_top2": HistConf([
        Axis(coll="events", field="mbb_btag_top2", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="$m_{bb}$ top-2 b-tag [GeV]",
             overflow=True, underflow=True)
    ]),
    "dR_b1_b2": HistConf([
        Axis(coll="events", field="dR_b1_b2", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R(b_1,b_2)$",
             overflow=True, underflow=True)
    ]),
    "sum_m_jets": HistConf([
        Axis(coll="events", field="sum_m_jets", pos=None,
             bins=50, start=0, stop=1500,
             label="Sum of jet masses [GeV]",
             overflow=True, underflow=True)
    ]),
    "centrality": HistConf([
        Axis(coll="events", field="centrality", pos=None,
             bins=50, start=0, stop=1,
             label="Event centrality",
             overflow=False, underflow=False)
    ]),
    "pT_higgs_like": HistConf([
        Axis(coll="events", field="pT_higgs_like", pos=None,
             bins=50, start=0, stop=600,
             label="Higgs-like $p_T$ [GeV]",
             overflow=True, underflow=True)
    ]),
    "sum_pt_b1b2_over_HT": HistConf([
        Axis(coll="events", field="sum_pt_b1b2_over_HT", pos=None,
             bins=50, start=0, stop=1,
             label="$(p_T(b_1)+p_T(b_2))/H_T$",
             overflow=True, underflow=False)
    ]),
    "dR_top2_btag": HistConf([
        Axis(coll="events", field="dR_top2_btag", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ top-2 b-tag jets",
             overflow=True, underflow=True)
    ]),
    "ptjj_at_mH": HistConf([
        Axis(coll="events", field="ptjj_at_mH", pos=None,
             bins=50, start=0, stop=600,
             label="$p_{T,jj}$ at $m_H$ [GeV]",
             overflow=True, underflow=True)
    ]),
    "max_weight": HistConf([
        Axis(coll="events", field="max_weight", pos=None,
             bins=50, start=0, stop=1,
             label="Max combinatorial weight",
             overflow=False, underflow=False)
    ]),
    "max_weight_higgs_mass": HistConf([
        Axis(coll="events", field="max_weight_higgs_mass", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Higgs mass at max weight [GeV]",
             overflow=True, underflow=True)
    ]),
    "filtered_solutions_number": HistConf([
        Axis(coll="events", field="filtered_solutions_number", pos=None,
             bins=20, start=0, stop=20,
             label="N filtered solutions",
             overflow=True, underflow=False)
    ]),
    "massreco_chosen_pair_higgs_mass": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_higgs_mass", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Reco Higgs mass -- chosen pair [GeV]",
             overflow=True, underflow=True)
    ]),
}


def get_events_axis_fields(*hist_dicts):
    """
    Collect every unique field name used by an Axis with coll="events"
    across any number of {name: HistConf} dictionaries.

    Use this instead of a hand-maintained field list so the shape guard
    below always covers every histogram actually defined here.
    """
    fields = set()
    for hd in hist_dicts:
        for histconf in hd.values():
            for axis in histconf.axes:
                if axis.coll == "events":
                    fields.add(axis.field)
    return sorted(fields)


# All coll="events" fields referenced by any histogram in this file
ALL_EVENTS_HISTOGRAM_FIELDS = get_events_axis_fields(
    mass_reco_histograms_truth, _histogram_event_fields
)