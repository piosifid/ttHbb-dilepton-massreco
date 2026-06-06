from pocket_coffea.parameters.histograms import Axis, HistConf

# Remove this entire import in config when running on data
# variables = {**mass_reco_histograms_base, **mass_reco_histograms_dr_study}
# remove: **mass_reco_histograms_truth

_MASS_BINS = [0,20,40,60,80,100,120,140,160,180,200,230,260,300,350,425,500]
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
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass max weight -- wrong match [GeV]",
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
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 1 -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank2_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 2 -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank3_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 3 -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank4_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 4 -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank1_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 1 -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank2_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 2 -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank3_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 3 -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank4_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 4 -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_correct_higgs_mass": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_correct_higgs_mass", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
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
             bins=50, start=0, stop=500,
             label="Reco Higgs Mass -- all [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_wrong_higgs_mass": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_wrong_higgs_mass", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Reco Higgs Mass -- any wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_wrong_higgs_mass_2_jets": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_wrong_higgs_mass_2_jets", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Reco Higgs Mass -- both jets wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_wrong_higgs_mass_1_jets": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_wrong_higgs_mass_1_jets", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Reco Higgs Mass -- one jet wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_correct_jet_pt": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_correct_jet_pt", pos=None,
             bins=50, start=0, stop=500,
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
             bins=50, start=0, stop=500,
             label="Higgs mass max weight -- all [GeV]",
             overflow=True, underflow=True)
    ]),
    "Massreco_maxweight_correct_higgs_mass": HistConf([
        Axis(coll="events", field="massreco_maxweight_correct_higgs_mass", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
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
    "mass_reco_had_a_solution": HistConf([
        Axis(coll="events", field="mass_reco_had_a_solution", pos=None,
             bins=8, start=0, stop=8,
             label="Mass reco outcome code",
             overflow=False, underflow=False)
    ]),
    "pair_mass_truth": HistConf([
        Axis(coll="events", field="pair_mass_truth", pos=None,
             bins=50, start=0, stop=500,
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
             bins=_MASS_BINS, start=0, stop=500,
             label="Truth Higgs mass [GeV]"),
        Axis(coll="events", field="massreco_chosen_pair_correct_higgs_mass",
             bins=_MASS_BINS, start=0, stop=500,
             label="Reco Higgs mass DR -- correct match [GeV]"),
    ]),
    "truth_vs_reco_higgs_mass_maxweight": HistConf([
        Axis(coll="events", field="higgs_mass_truth_jets",
             bins=_MASS_BINS, start=0, stop=500,
             label="Truth Higgs mass [GeV]"),
        Axis(coll="events", field="massreco_maxweight_correct_higgs_mass",
             bins=_MASS_BINS, start=0, stop=500,
             label="Reco Higgs mass max weight -- correct match [GeV]"),
    ]),

    # =========================================================================
    # A9 — Selected/rejected × correct/wrong Higgs mass per rank
    # To add to mass_reco_histograms_truth.py
    # =========================================================================

    # --- Rank 1 ---
    "higgs_mass_rank1_selected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_selected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 1 selected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank1_selected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_selected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 1 selected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank1_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_rejected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 1 rejected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank1_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_rejected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 1 rejected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),

    # --- Rank 2 ---
    "higgs_mass_rank2_selected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_selected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 2 selected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank2_selected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_selected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 2 selected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank2_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_rejected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 2 rejected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank2_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_rejected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 2 rejected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),

    # --- Rank 3 ---
    "higgs_mass_rank3_selected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_selected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 3 selected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank3_selected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_selected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 3 selected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank3_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_rejected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 3 rejected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank3_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_rejected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 3 rejected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),

    # --- Rank 4 ---
    "higgs_mass_rank4_selected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_selected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 4 selected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank4_selected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_selected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 4 selected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank4_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_rejected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 4 rejected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_rank4_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_rejected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 4 rejected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),

    # --- Fallback ---
    "higgs_mass_fallback_selected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_fallback_selected_correct", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass fallback selected -- correct [GeV]",
             overflow=True, underflow=True)
    ]),
    "higgs_mass_fallback_selected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_fallback_selected_wrong", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass fallback selected -- wrong [GeV]",
             overflow=True, underflow=True)
    ]),

    # =========================================================================
    # A9 — 2D histograms: selected correct vs rejected correct per rank
    # Shows mass of correct combinations when selected vs when rejected
    # =========================================================================
    "rank1_selected_correct_vs_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_selected_correct",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 1 selected correct [GeV]"),
        Axis(coll="events", field="higgs_mass_rank1_rejected_correct",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 1 rejected correct [GeV]"),
    ]),
    "rank2_selected_correct_vs_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_selected_correct",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 2 selected correct [GeV]"),
        Axis(coll="events", field="higgs_mass_rank2_rejected_correct",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 2 rejected correct [GeV]"),
    ]),
    "rank3_selected_correct_vs_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_selected_correct",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 3 selected correct [GeV]"),
        Axis(coll="events", field="higgs_mass_rank3_rejected_correct",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 3 rejected correct [GeV]"),
    ]),
    "rank4_selected_correct_vs_rejected_correct": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_selected_correct",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 4 selected correct [GeV]"),
        Axis(coll="events", field="higgs_mass_rank4_rejected_correct",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 4 rejected correct [GeV]"),
    ]),

    # =========================================================================
    # A9 — 2D histograms: selected wrong vs rejected wrong per rank
    # Shows mass of wrong combinations when selected vs when rejected
    # =========================================================================
    "rank1_selected_wrong_vs_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_selected_wrong",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 1 selected wrong [GeV]"),
        Axis(coll="events", field="higgs_mass_rank1_rejected_wrong",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 1 rejected wrong [GeV]"),
    ]),
    "rank2_selected_wrong_vs_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_selected_wrong",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 2 selected wrong [GeV]"),
        Axis(coll="events", field="higgs_mass_rank2_rejected_wrong",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 2 rejected wrong [GeV]"),
    ]),
    "rank3_selected_wrong_vs_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_selected_wrong",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 3 selected wrong [GeV]"),
        Axis(coll="events", field="higgs_mass_rank3_rejected_wrong",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 3 rejected wrong [GeV]"),
    ]),
    "rank4_selected_wrong_vs_rejected_wrong": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_selected_wrong",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 4 selected wrong [GeV]"),
        Axis(coll="events", field="higgs_mass_rank4_rejected_wrong",
             bins=_MASS_BINS, start=0, stop=500,
             label="Rank 4 rejected wrong [GeV]"),
    ]),
}
