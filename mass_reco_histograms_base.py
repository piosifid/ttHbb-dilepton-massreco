from pocket_coffea.parameters.histograms import Axis, HistConf, count_hist

_MASS_BINS = [60,80,90,100,110,120,130,140,150,170,210,300,450,600]
mass_reco_histograms_base = {

    # =========================================================================
    # Standard kinematic histograms
    # =========================================================================
    "deltaRbb_min": HistConf([
        Axis(coll="events", field="deltaRbb_min", bins=50, start=0, stop=5,
             label=r"$\min(\Delta R_{bb})$", overflow=True, underflow=True)
    ]),
    "mbb": HistConf([
        Axis(coll="events", field="mbb", bins=100, start=0, stop=1000,
             label=r"$m_{bb}$ [GeV]")
    ]),
    "mbb_min": HistConf([
        Axis(coll="events", field="mbb_min", bins=100, start=0, stop=1000,
             label=r"$m_{bb}^{\min}$ [GeV]", overflow=True, underflow=True)
    ]),
    "npvsGood": HistConf([
        Axis(coll="PV", field="npvsGood", bins=100, start=0, stop=100,
             label=r"$N_{\mathrm{good\,vertices}}$")
    ]),
    **count_hist(name="nJets",    coll="JetGood",    bins=10, start=2,  stop=12),
    **count_hist(name="nBJets",   coll="BJetGood",   bins=14, start=0,  stop=14),
    **count_hist(name="nLeptons", coll="LeptonGood", bins=3,  start=0,  stop=3),
    "ht": HistConf([
        Axis(coll="events", field="JetGood_Ht", bins=120, start=0, stop=1200,
             label=r"$H_T$ [GeV]")
    ]),
    "HT_events": HistConf([
        Axis(coll="events", field="HT", bins=60, start=0, stop=1500,
             label=r"$H_T$ [GeV]")
    ]),
    "MET_pt": HistConf([
        Axis(coll="events", field="MET_pt", bins=60, start=0.0, stop=300.0,
             label=r"$MET_{\mathrm{Puppi}}$ [GeV]")
    ]),
    "MET_phi": HistConf([
        Axis(coll="events", field="MET_phi", bins=64, start=-3.2, stop=3.2,
             label=r"$\phi(MET)$")
    ]),
    "PuppyMET": HistConf([
        Axis(coll="PuppiMET", field="pt",
             bins=[20, 30, 40, 60, 80, 100, 125, 150, 175, 200],
             start=20, stop=200, label=r"$MET$ [GeV]", lim=(20, 200))
    ]),

    # =========================================================================
    # DNN and event shape variables
    # =========================================================================
    "dnn_score": HistConf([
        Axis(coll="events", field="dnn_score", bins=100, start=0.0, stop=1.0,
             label="DNN score")
    ]),
    "C_jet": HistConf([
        Axis(coll="events", field="C_jet", bins=50, start=0.0, stop=1.0,
             label=r"$C_{\mathrm{jet}}$")
    ]),
    "D_jet": HistConf([
        Axis(coll="events", field="D_jet", bins=50, start=0.0, stop=1.0,
             label=r"$D_{\mathrm{jet}}$")
    ]),
    "Aplanarity": HistConf([
        Axis(coll="events", field="Aplanarity", bins=40, start=0.0, stop=0.5,
             label=r"Aplanarity")
    ]),
    "H4": HistConf([
        Axis(coll="events", field="H4", bins=50, start=0.0, stop=1.0,
             label=r"$H_4$ (Fox-Wolfram)")
    ]),
    "pt_jet2": HistConf([
        Axis(coll="events", field="pt_jet2", bins=50, start=0, stop=500,
             label=r"$p_T(\mathrm{jet\,2})$ [GeV]")
    ]),
    "dRjj_min": HistConf([
        Axis(coll="events", field="dRjj_min", bins=30, start=0, stop=3,
             label=r"$\min(\Delta R_{jj})$")
    ]),
    "deta_max": HistConf([
        Axis(coll="events", field="deta_max", bins=20, start=0, stop=5,
             label=r"$\max(|\Delta\eta|)$")
    ]),
    "m_higgs_like_jj": HistConf([
        Axis(coll="events", field="m_higgs_like_jj", bins=60, start=0, stop=300,
             label=r"$m_{jj}^{\mathrm{H-like}}$ [GeV]")
    ]),
    "ptjj_at_dRmin": HistConf([
        Axis(coll="events", field="ptjj_at_dRmin", bins=60, start=0, stop=600,
             label=r"$p_T^{jj}(\Delta R_{\min})$ [GeV]")
    ]),
    "Njj_higgs_like": HistConf([
        Axis(coll="events", field="Njj_higgs_like", bins=10, start=0, stop=10,
             label=r"$N_{jj}^{100<m<140}$")
    ]),
    "m_jjj_maxpT": HistConf([
        Axis(coll="events", field="m_jjj_maxpT", bins=35, start=0, stop=700,
             label=r"$m_{jjj}^{\max \sum p_T}$ [GeV]")
    ]),
    "sum4_btag": HistConf([
        Axis(coll="events", field="sum4_btag", bins=80, start=0.0, stop=4.0,
             label=r"$\sum_{\text{top 4}} b\text{-tag scores}$")
    ]),
    "sum_pt_b1b2": HistConf([
        Axis(coll="events", field="sum_pt_b1b2", bins=100, start=0.0, stop=1000.0,
             label=r"$p_{T}^{b_1} + p_{T}^{b_2}$ [GeV]")
    ]),
    "pt_ratio_b1_b2": HistConf([
        Axis(coll="events", field="pt_ratio_b1_b2", bins=80, start=0.0, stop=8.0,
             label=r"$p_{T}^{b_1}/p_{T}^{b_2}$")
    ]),
    "mbb_btag_top2": HistConf([
        Axis(coll="events", field="mbb_btag_top2", bins=60, start=0.0, stop=300.0,
             label=r"$m_{bb}^{\text{(top 2 b-tag)}}$ [GeV]")
    ]),
    "dR_b1_b2": HistConf([
        Axis(coll="events", field="dR_b1_b2", bins=50, start=0.0, stop=5.0,
             label=r"$\Delta R(b_1, b_2)$")
    ]),
    "sum_m_jets": HistConf([
        Axis(coll="events", field="sum_m_jets", bins=60, start=0.0, stop=1200.0,
             label=r"$\sum m_{\mathrm{jets}}$ [GeV]")
    ]),
    "centrality": HistConf([
        Axis(coll="events", field="centrality", bins=50, start=0.0, stop=1.5,
             label=r"Event Centrality")
    ]),
    "pT_higgs_like": HistConf([
        Axis(coll="events", field="pT_higgs_like", bins=60, start=0.0, stop=600.0,
             label=r"$p_T^{\mathrm{H-like}}$ [GeV]")
    ]),
    "sum_pt_b1b2_over_HT": HistConf([
        Axis(coll="events", field="sum_pt_b1b2_over_HT", bins=40, start=0.0, stop=1.2,
             label=r"$(p_{T}^{b_1}+p_{T}^{b_2})/H_T$")
    ]),
    "dR_top2_btag": HistConf([
        Axis(coll="events", field="dR_top2_btag", bins=50, start=0.0, stop=5.0,
             label=r"$\Delta R(\text{top 2 b-tag})$")
    ]),
    "ptjj_at_mH": HistConf([
        Axis(coll="events", field="ptjj_at_mH", bins=60, start=0.0, stop=600.0,
             label=r"$p_T^{jj}$ at $m_H$ [GeV]")
    ]),

    # =========================================================================
    # Mass reco — core outputs (data-safe)
    # =========================================================================
    "Max_Weight": HistConf([
        Axis(coll="events", field="max_weight", pos=None, bins=100, start=0, stop=10000,
             label="Maximum Weight per Event", overflow=True, underflow=True)
    ]),
    "Max_Weight_Higgs_Mass": HistConf([
        Axis(coll="events", field="max_weight_higgs_mass", pos=None,
             bins=100, start=0, stop=1000,
             label="Max Weight Higgs Mass [GeV]", overflow=True, underflow=True)
    ]),
    "Max_Weight_Higgs_Mass_1": HistConf([
        Axis(coll="events", field="max_weight_higgs_mass", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="Max Weight Higgs Mass [GeV]", overflow=True, underflow=True)
    ]),
    "Solutions_Number": HistConf([
        Axis(coll="events", field="filtered_solutions_number", pos=None,
             bins=10, start=0, stop=10,
             label="Number of Solutions per Event", overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_higgs_mass": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_higgs_mass", pos=None,
             bins=100, start=0, stop=1000,
             label="DR Criterion Higgs Mass [GeV]", overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_higgs_mass_1": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_higgs_mass", pos=None,
             bins=_MASS_BINS, start=0, stop=600,
             label="DR Criterion Higgs Mass [GeV]", overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_higgs_mass_2": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_higgs_mass", pos=None,
             bins=[0,20,40,60,80,100,120,140,160,180,210,240,270,310,350,390,430,500],
             start=0, stop=500,
             label="DR Criterion Higgs Mass [GeV]", overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_higgs_mass_3": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_higgs_mass", pos=None,
             bins=[0,20,40,60,80,100,120,140,160,180,200,230,260,290,320,360,400,450,500],
             start=0, stop=500,
             label="DR Criterion Higgs Mass [GeV]", overflow=True, underflow=True)
    ]),
    "Massreco_chosen_pair_higgs_mass_4": HistConf([
        Axis(coll="events", field="massreco_chosen_pair_higgs_mass", pos=None,
             bins=[0,20,40,60,80,100,120,140,160,180,200,230,260,300,350,425,500],
             start=0, stop=500,
             label="DR Criterion Higgs Mass [GeV]", overflow=True, underflow=True)
    ]),
}
