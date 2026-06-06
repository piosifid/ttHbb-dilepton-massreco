from pocket_coffea.parameters.histograms import Axis, HistConf

_MASS_BINS = [0,20,40,60,80,100,120,140,160,180,200,230,260,300,350,425,500]
_DR_BINS   = 50
_DR_START  = 0
_DR_STOP   = 5

mass_reco_histograms_dr_study = {

    # =========================================================================
    # DR criterion activity
    # =========================================================================
    "rank_used": HistConf([
        Axis(coll="events", field="rank_used", pos=None,
             bins=6, start=-1, stop=5,
             label="Rank used by DR criterion",
             overflow=True, underflow=True)
    ]),

    # =========================================================================
    # All-events per-rank mass and DR
    # =========================================================================
    "higgs_mass_rank1": HistConf([
        Axis(coll="events", field="higgs_mass_rank1", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 1 [GeV]", overflow=True, underflow=True)
    ]),
    "higgs_mass_rank2": HistConf([
        Axis(coll="events", field="higgs_mass_rank2", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 2 [GeV]", overflow=True, underflow=True)
    ]),
    "higgs_mass_rank3": HistConf([
        Axis(coll="events", field="higgs_mass_rank3", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 3 [GeV]", overflow=True, underflow=True)
    ]),
    "higgs_mass_rank4": HistConf([
        Axis(coll="events", field="higgs_mass_rank4", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 4 [GeV]", overflow=True, underflow=True)
    ]),
    "higgs_mass_chosen_binned": HistConf([
        Axis(coll="events", field="higgs_mass_chosen", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass DR chosen [GeV]", overflow=True, underflow=True)
    ]),
    "dr_rank1": HistConf([
        Axis(coll="events", field="dr_rank1", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ rank 1", overflow=True, underflow=True)
    ]),
    "dr_rank2": HistConf([
        Axis(coll="events", field="dr_rank2", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ rank 2", overflow=True, underflow=True)
    ]),
    "dr_rank3": HistConf([
        Axis(coll="events", field="dr_rank3", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ rank 3", overflow=True, underflow=True)
    ]),
    "dr_rank4": HistConf([
        Axis(coll="events", field="dr_rank4", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ rank 4", overflow=True, underflow=True)
    ]),
    "dr_chosen": HistConf([
        Axis(coll="events", field="dr_chosen", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ chosen pair", overflow=True, underflow=True)
    ]),

    # =========================================================================
    # Selected-from-rank mass and DR
    # =========================================================================
    "higgs_mass_chosen_from_rank1": HistConf([
        Axis(coll="events", field="higgs_mass_chosen_from_rank1", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass -- rank 1 selected [GeV]", overflow=True, underflow=True)
    ]),
    "higgs_mass_chosen_from_rank2": HistConf([
        Axis(coll="events", field="higgs_mass_chosen_from_rank2", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass -- rank 2 selected [GeV]", overflow=True, underflow=True)
    ]),
    "higgs_mass_chosen_from_rank3": HistConf([
        Axis(coll="events", field="higgs_mass_chosen_from_rank3", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass -- rank 3 selected [GeV]", overflow=True, underflow=True)
    ]),
    "higgs_mass_chosen_from_rank4": HistConf([
        Axis(coll="events", field="higgs_mass_chosen_from_rank4", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass -- rank 4 selected [GeV]", overflow=True, underflow=True)
    ]),
    "higgs_mass_chosen_from_fallback": HistConf([
        Axis(coll="events", field="higgs_mass_chosen_from_fallback", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass -- fallback selected [GeV]", overflow=True, underflow=True)
    ]),
    "dr_chosen_from_rank1": HistConf([
        Axis(coll="events", field="dr_chosen_from_rank1", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ -- rank 1 selected", overflow=True, underflow=True)
    ]),
    "dr_chosen_from_rank2": HistConf([
        Axis(coll="events", field="dr_chosen_from_rank2", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ -- rank 2 selected", overflow=True, underflow=True)
    ]),
    "dr_chosen_from_rank3": HistConf([
        Axis(coll="events", field="dr_chosen_from_rank3", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ -- rank 3 selected", overflow=True, underflow=True)
    ]),
    "dr_chosen_from_rank4": HistConf([
        Axis(coll="events", field="dr_chosen_from_rank4", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ -- rank 4 selected", overflow=True, underflow=True)
    ]),
    "dr_chosen_from_fallback": HistConf([
        Axis(coll="events", field="dr_chosen_from_fallback", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ -- fallback selected", overflow=True, underflow=True)
    ]),

    # =========================================================================
    # Rejected-from-rank mass and DR
    # =========================================================================
    "higgs_mass_rank1_rejected": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_rejected", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass -- rank 1 rejected [GeV]", overflow=True, underflow=True)
    ]),
    "higgs_mass_rank2_rejected": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_rejected", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass -- rank 2 rejected [GeV]", overflow=True, underflow=True)
    ]),
    "higgs_mass_rank3_rejected": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_rejected", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass -- rank 3 rejected [GeV]", overflow=True, underflow=True)
    ]),
    "higgs_mass_rank4_rejected": HistConf([
        Axis(coll="events", field="higgs_mass_rank4_rejected", pos=None,
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass -- rank 4 rejected (fallback) [GeV]",
             overflow=True, underflow=True)
    ]),
    "dr_rank1_rejected": HistConf([
        Axis(coll="events", field="dr_rank1_rejected", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ -- rank 1 rejected", overflow=True, underflow=True)
    ]),
    "dr_rank2_rejected": HistConf([
        Axis(coll="events", field="dr_rank2_rejected", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ -- rank 2 rejected", overflow=True, underflow=True)
    ]),
    "dr_rank3_rejected": HistConf([
        Axis(coll="events", field="dr_rank3_rejected", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ -- rank 3 rejected", overflow=True, underflow=True)
    ]),
    "dr_rank4_rejected": HistConf([
        Axis(coll="events", field="dr_rank4_rejected", pos=None,
             bins=_DR_BINS, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ -- rank 4 rejected (fallback)",
             overflow=True, underflow=True)
    ]),

    # =========================================================================
    # 2D histograms — mass vs mass (rank transitions)
    # =========================================================================
    "higgs_mass_rank1_selected_vs_rejected": HistConf([
        Axis(coll="events", field="higgs_mass_chosen_from_rank1",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 1 selected [GeV]"),
        Axis(coll="events", field="higgs_mass_rank1_rejected",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 1 rejected [GeV]"),
    ]),
    "higgs_mass_rank1_rejected_vs_rank2_selected": HistConf([
        Axis(coll="events", field="higgs_mass_rank1_rejected",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 1 rejected [GeV]"),
        Axis(coll="events", field="higgs_mass_chosen_from_rank2",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 2 selected [GeV]"),
    ]),
    "higgs_mass_rank2_rejected_vs_rank3_selected": HistConf([
        Axis(coll="events", field="higgs_mass_rank2_rejected",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 2 rejected [GeV]"),
        Axis(coll="events", field="higgs_mass_chosen_from_rank3",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 3 selected [GeV]"),
    ]),
    "higgs_mass_rank3_rejected_vs_rank4_selected": HistConf([
        Axis(coll="events", field="higgs_mass_rank3_rejected",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 3 rejected [GeV]"),
        Axis(coll="events", field="higgs_mass_chosen_from_rank4",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 4 selected [GeV]"),
    ]),

    # =========================================================================
    # 2D histograms — mass vs DR per rank (selected and rejected)
    # =========================================================================
    "dr_vs_mass_rank1_selected": HistConf([
        Axis(coll="events", field="dr_chosen_from_rank1",
             bins=25, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ rank 1 selected"),
        Axis(coll="events", field="higgs_mass_chosen_from_rank1",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 1 selected [GeV]"),
    ]),
    "dr_vs_mass_rank1_rejected": HistConf([
        Axis(coll="events", field="dr_rank1_rejected",
             bins=25, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ rank 1 rejected"),
        Axis(coll="events", field="higgs_mass_rank1_rejected",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 1 rejected [GeV]"),
    ]),
    "dr_vs_mass_rank2_selected": HistConf([
        Axis(coll="events", field="dr_chosen_from_rank2",
             bins=25, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ rank 2 selected"),
        Axis(coll="events", field="higgs_mass_chosen_from_rank2",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 2 selected [GeV]"),
    ]),
    "dr_vs_mass_rank2_rejected": HistConf([
        Axis(coll="events", field="dr_rank2_rejected",
             bins=25, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ rank 2 rejected"),
        Axis(coll="events", field="higgs_mass_rank2_rejected",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 2 rejected [GeV]"),
    ]),
    "dr_vs_mass_rank3_selected": HistConf([
        Axis(coll="events", field="dr_chosen_from_rank3",
             bins=25, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ rank 3 selected"),
        Axis(coll="events", field="higgs_mass_chosen_from_rank3",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 3 selected [GeV]"),
    ]),
    "dr_vs_mass_rank3_rejected": HistConf([
        Axis(coll="events", field="dr_rank3_rejected",
             bins=25, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ rank 3 rejected"),
        Axis(coll="events", field="higgs_mass_rank3_rejected",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 3 rejected [GeV]"),
    ]),
    "dr_vs_mass_rank4_selected": HistConf([
        Axis(coll="events", field="dr_chosen_from_rank4",
             bins=25, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ rank 4 selected"),
        Axis(coll="events", field="higgs_mass_chosen_from_rank4",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 4 selected [GeV]"),
    ]),
    "dr_vs_mass_rank4_rejected": HistConf([
        Axis(coll="events", field="dr_rank4_rejected",
             bins=25, start=_DR_START, stop=_DR_STOP,
             label=r"$\Delta R$ rank 4 rejected (fallback)"),
        Axis(coll="events", field="higgs_mass_rank4_rejected",
             bins=_MASS_BINS, start=0, stop=500,
             label="Higgs mass rank 4 rejected [GeV]"),
    ]),
}
