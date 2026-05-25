import awkward as ak
from coffea.util import load
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
import os
import argparse
import matplotlib.gridspec as gridspec
from pocket_coffea.utils.plot_utils import Shape
from matplotlib import gridspec
import sys
from hist import Hist
import correctionlib, rich
import correctionlib.convert
import boost_histogram as bh
import hist


var_argNames = {
	"MuonGood_eta": "Muons_eta",
	"MuonGood_pt": "Muons_pt",
	"MuonGood_phi": "Muons_phi",
	"MuonGood_eta_1": "leading_Muon_eta",
	"MuonGood_pt_1": "leading_Muon_pt",
	"MuonGood_phi_1": "leading_Muon_phi",
	"ElectronGood_pt": "Electrons_pt",
	"ElectronGood_etaSC": "Electrons_etaSC",
	"ElectronGood_phi": "Electrons_phi",
	"ElectronGood_pt_1": "leading_Electron_pt",
	"ElectronGood_etaSC_1": "leading_Electron_etaSC",
	"ElectronGood_phi_1": "leading_Electron_phi",
	"JetGood_pt_1": "leading_Jet_pt",
	"JetGood_eta": "Jets_eta",
	"JetGood_pt": "Jets_pt",
	"JetGood_phi": "Jets_phi",
	"JetGood_btagDeepFlavB": "btagDeepFlavB",
	"JetGood_btagDeepFlavCvL": "btagDeepFlavCvL",
	"JetGood_btagDeepFlavCvB": "btagDeepFlavCvB",
	"nMuons": "nMuons",
	"nElectrons": "nElectrons",
	"nLeptons": "nLeptons",
	"nJets": "nJets",
#	"nBJets": "nBJets",
	"ht": "HT",
	"mjj": "mjj",
        "deltaRjj_min":"deltaRjj_min",
	"mll":"mll"
}

varsss=['deltaRbb_min', 'Max_Weight', 'npvsGood', 'nJets', 'nBJets', 'nLeptons', 'LeptonGood_eta_1', 'LeptonGood_pt_1', 'LeptonGood_phi_1', 'LeptonGood_pdgId_1', 'LeptonGood_eta_2', 'LeptonGood_pt_2', 'LeptonGood_phi_2', 'LeptonGood_pdgId_2', 'bjet_eta_1', 'bjet_pt_1', 'bjet_phi_1', 'bjet_eta_2', 'bjet_pt_2', 'bjet_phi_2', 'jet_eta_1', 'jet_pt_1', 'jet_phi_1', 'MET_pt', 'MET_phi']

parser = argparse.ArgumentParser()
parser.add_argument("--var", "-v", default="all", help=f"variable to be plotted \n options: {var_argNames.keys()} \n default: \"all\"" )
parser.add_argument("--logscale", "-log", action="store_true", help="apply logscale in upper pad, default: False" )
parser.add_argument("--input-dir", "-i", default=False, help="choose directory to save plots" )
parser.add_argument("--ext","-e", default="", help="add a prefix at the output directory: XXX_trf_closure")
parser.add_argument("--doFit", "-f", action="store_true",  help="perform 0th & 1st degree pol fit in the ratio, default: False")
parser.add_argument("--isConditional", "-c", action="store_true",  help="conditional probabilities used, default: False")

args = parser.parse_args()
var_args = [args.var]
log = args.logscale
Output = args.input_dir
Prefix = args.ext
do_fit = args.doFit
cond_flag = args.isConditional


filename = Output + "/output_all.coffea"
inputFile = load(filename)
categories = [
]

for step in inputFile['sumw'].keys():
	categories.append(step)
categories.sort()
print("\nCategories:\n",categories)


print("\033[94m\nCreating plots for all vars - with logscale:", str(log), "- in directory - with fit:", str(do_fit), "- for conditional prob.:", str(cond_flag), "\033[0m")


def sum_over_hists(file):
	var = load(file)["variables"]
	outdict = {}
	for H in var.keys():
		concdict = {}
		samples = var[H]
		outdict.update({f"{str(H)}": 0})
		for s in samples.keys():
			datasets = samples[s]
			concdict.update({f"{str(s)}": 0})
			for d in datasets.keys():
				hist_obj = datasets[d]
				#print(hist_obj)
				if concdict[str(s)] == 0:
					concdict.update({f"{str(s)}": hist_obj})
				else:
					concdict[str(s)] += hist_obj
			outdict.update({f"{str(H)}": concdict})
	return outdict

output = sum_over_hists(filename)
#print('@@@output_sum_of_histos: ',output)

print('\nVariables:\n',output.keys())


def fill_3d_hist_from_2d_histo(hist_3d, hist_2d, flavour_bin):
    for i in range(hist_2d.shape[0]):
        for j in range(hist_2d.shape[1]):
            value = hist_2d[i, j]
            hist_3d[i, j, flavour_bin] += value


for var in varsss:
	print(output[var])
	if var=='Max_Weight':
		histoo=output[var]['TTH_Hto2B'][{"cat": 'baseline'}][{'variation':'nominal'}]
		plt.hist(histoo, edgecolor='black')
		plt.savefig('maxWeight.png')
