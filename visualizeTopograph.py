import ROOT
import uproot
import joblib
from ROOT import TFile
import numpy as np
import matplotlib.pyplot as plt
from functions import tools
from functions import functions

# List of filenames
filenames = [
    "RDF_BprimeBprimeto2B4Tau_MB-1000_MXi-2000_TuneCP5_13p6TeV-madgraph-pythia8_2022_0.root",
    "RDF_BprimeBprimeto2B4Tau_MB-1000_MXi-2000_TuneCP5_13p6TeV-madgraph-pythia8_2022EE_0.root",
    "RDF_BprimeBprimeto2B4Tau_MB-1000_MXi-2000_TuneCP5_13p6TeV-madgraph-pythia8_2023_0.root",
    "RDF_BprimeBprimeto2B4Tau_MB-1600_MXi-2000_TuneCP5_13p6TeV-madgraph-pythia8_2023_41.root",
    "RDF_BprimeBprimeto2B4Tau_MB-400_MXi-2000_TuneCP5_13p6TeV-madgraph-pythia8_2023_61.root",
    #"RDF_BprimeBprimeto2B4Tau_MB-1600_MXi-2000_TuneCP5_13p6TeV-madgraph-pythia8_2023_71.root", This one errored out :(
    "RDF_BprimeBprimeto2B4Tau_MB-700_MXi-2000_TuneCP5_13p6TeV-madgraph-pythia8_2023_75.root",
    "RDF_BprimeBprimeto2B4Tau_MB-1000_MXi-2000_TuneCP5_13p6TeV-madgraph-pythia8_2023BPix_0.root",
]


# Initialize lists
GenPt = []
GenEta = []
GenPhi = []
GenPDGID = []
GenPartIdxMother = []
GenB1Pt = []
GenB2Pt = []
GenB1Eta = []
GenB2Eta = []
GenB1Phi = []
GenB2Phi = []
GenB1Px = []
GenB1Py = []
GenB1Pz = []
GenB2Px = []
GenB2Py = []
GenB2Pz = []
MB1Px = []
MB1Py = []
MB1Pz = []
MB2Px = []
MB2Py = []
MB2Pz = []
TB1Pt = []
TB2Pt = []

for filename in filenames:
    print(f"Opening...{filename}")
    f = uproot.open(filename)
    events = f['Events_Nominal;1']
    nevents = events.num_entries
    print(f"{nevents = }")
    'B1finalPx', 'B1finalPy', 'B1finalPz', 'B2finalPx', 'B2finalPy', 'B2finalPz', 'GenPart_pdgId', 'GenPart_mass', 'GenPart_pt', 'GenPart_phi', 'GenPart_eta'
    # Extend master lists with data from current file
    GenPt.extend(events['GenPart_pt'].array())
    GenEta.extend(events['GenPart_eta'].array())
    GenPhi.extend(events['GenPart_phi'].array())
    GenPDGID.extend(events['GenPart_pdgId'].array())
    GenPartIdxMother.extend(events['GenPart_genPartIdxMother'].array())
    MB1Px.extend(events['B1finalPx'].array())
    MB1Py.extend(events['B1finalPy'].array())
    MB1Pz.extend(events['B1finalPz'].array())
    MB2Px.extend(events['B2finalPx'].array())
    MB2Py.extend(events['B2finalPy'].array())
    MB2Pz.extend(events['B2finalPz'].array())
    
#array1 = np.load('Outputs_part2/preds_tops.npy')
loaded_arr = np.load('Outputs/preds_tops.npy')
#loaded_arr = np.concatenate((array1, array2))
TB1Pt = loaded_arr[:, 0, :]
TB2Pt = loaded_arr[:, 1, :]
TPt = (TB1Pt + TB2Pt)/2
print("TPt: ", TPt.shape)

#array1 = np.load('Outputs_part2/preds_tops_initial.npy')
loaded_arr = np.load('Outputs/preds_tops_initial.npy')
#loaded_arr = np.concatenate((array1, array2))
TGenB1Pt = loaded_arr[:, 0, :]
TGenB2Pt = loaded_arr[:, 1, :]
TGenPt = (TGenB1Pt + TGenB2Pt)/2
print("TGenPt: ", TGenPt.shape)

print("Finished Reading Files")
    
for i in range(0, len(GenPDGID)):
    for j in range(0, len(GenPDGID[i])):
        # Find index of PDGID 9000005
        if GenPDGID[i][j] == 15 and GenPDGID[i][GenPartIdxMother[i][j]] == 9000005:
            GenB1Pt.append(GenPt[i][GenPartIdxMother[i][j]])
            GenB1Eta.append(GenEta[i][GenPartIdxMother[i][j]])
            GenB1Phi.append(GenPhi[i][GenPartIdxMother[i][j]])
            
        if GenPDGID[i][j] == 15 and GenPDGID[i][GenPartIdxMother[i][j]] == -9000005:
            GenB2Pt.append(GenPt[i][GenPartIdxMother[i][j]])
            GenB2Eta.append(GenEta[i][GenPartIdxMother[i][j]])
            GenB2Phi.append(GenPhi[i][GenPartIdxMother[i][j]])
         
GenPt = [(a + b) / 2 for a, b in zip(GenB1Pt, GenB2Pt)]
GenPhi = [(a + b) / 2 for a, b in zip(GenB1Phi, GenB2Phi)] 
GenEta = [(a + b) / 2 for a, b in zip(GenB1Eta, GenB2Eta)] 
MPx = [(a + b) / 2 for a, b in zip(MB1Px, MB2Px)]
MPy = [(a + b) / 2 for a, b in zip(MB1Py, MB2Py)] 
MPz = [(a + b) / 2 for a, b in zip(MB1Pz, MB2Pz)] 
        
print("Averaged Values")
        
GenPx = [pt * np.cos(phi) for pt, phi in zip(GenPt, GenPhi)]
GenPy = [pt * np.sin(phi) for pt, phi in zip(GenPt, GenPhi)]
GenPz = [pt * np.sinh(eta) for pt, eta in zip(GenPt, GenEta)]

GenPt = np.array([GenPx, GenPy, GenPz]).T
MPt = np.array([MPx, MPy, MPz]).T
    
Tscaler = joblib.load('Outputs_part2/scaler.Tops')
print("Scaler Loaded")

GenPt = Tscaler.transform(GenPt)
MPt = Tscaler.transform(MPt)
print("Values Transformed")

ManualPtCalc = np.where(GenPt != 0, (MPt - GenPt) / GenPt, np.nan)

# Compute transverse momentum (pt) = sqrt(px^2 + py^2) for ManualB1PtCalc and GenB1Pt
pt_ManualPtCalc = np.sqrt(ManualPtCalc[:, 0]**2 + ManualPtCalc[:, 1]**2) 
pt_GenPt = np.sqrt(GenPt[:, 0]**2 + GenPt[:, 1]**2)

pt_TPtCalc = np.sqrt(TPt[:, 0]**2 + TPt[:, 1]**2) 
pt_TGenPt = np.sqrt(TGenPt[:, 0]**2 + TGenPt[:, 1]**2) 


# Calculate (pt_ManualB1PtCalc - pt_GenB1Pt) / pt_GenB1Pt, handling division by zero
pt_relative_diff_MB = np.where(pt_GenPt != 0, (pt_ManualPtCalc - pt_GenPt) / pt_GenPt, np.nan)
pt_relative_diff_TB = np.where(pt_TGenPt != 0, (pt_TPtCalc - pt_TGenPt) / pt_TGenPt, np.nan)

# Set up the plot
plt.figure(figsize=(10, 6))

# Histogram parameters
bins = 50
hist_range = (-2, 8)

# Compute histograms for pt_relative_diff_TB and pt_relative_diff_MB
hist_TB, bin_edges = np.histogram(pt_relative_diff_TB[~np.isnan(pt_relative_diff_TB)], bins=bins, range=hist_range, density=False)
hist_MB, _ = np.histogram(pt_relative_diff_MB[~np.isnan(pt_relative_diff_MB)], bins=bins, range=hist_range, density=False)

# Normalize histograms to have the same peak height
hist_TB_normalized = hist_TB / np.max(hist_TB)
hist_MB_normalized = hist_MB / np.max(hist_MB)

# Compute bin edges for step plot (including the last edge for step style)
step_edges = bin_edges

# Plot the normalized histograms as step plots
plt.step(step_edges[:-1], hist_TB_normalized, label='Topograph', color='blue', linewidth=1.5, where='post')
plt.step(step_edges[:-1], hist_MB_normalized, label='Manual', color='red', linewidth=1.5, where='post')

# Customize plot
plt.title('Relative Error in Average $p_T$')
plt.xlabel('(PtCalc - GenPt) / GenPt')
plt.ylabel('Normalized Count')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('Momentum.png')
print("Momentum.png Saved")