import pandas as pd
import numpy as np
import scipy.io as scio
from scipy.sparse import coo_matrix

#############start-load_data##############
openness=pd.read_csv("data/openness2.bed",sep='\t', header=None)
Element_name = openness.iloc[:,0].values
Opn = openness.iloc[:,1].values
Opn_median = openness.iloc[:,2].values
MotifMatchMat = scio.loadmat('data/MotifMatch_mouse_rmdup.mat')
Match2 = MotifMatchMat['Match2']
motifName = MotifMatchMat['motifName']
motifWeight = MotifMatchMat['motifWeight']
TFTG_corr = scio.loadmat('data/TFTG_corr_mouse.mat')
Exp_median = TFTG_corr['Exp_median']
List = TFTG_corr['List']
R2 = TFTG_corr['R2']
TFExp_median = TFTG_corr['TFExp_median']
TFName = TFTG_corr['TFName']
############end-load_data###############
TF_binding=mfbs(TFName,Element_name,motifName,motifWeight,Match2)
C = pd.read_csv('data/RAd4.txt',header=None,sep='\t')
Symbol = C.iloc[:,0].values
G = C.iloc[:,1].values
CRInfo = scio.loadmat('data/CRInfo_mouse.mat')
CRName = CRInfo['CRName']
CRName = [crn[0][0] for crn in CRName]
CR_TF = CRInfo['CR_TF']
C_TFName = CRInfo['C_TFName']
TFS = CRInfo['TFS']
eita0=-30.4395
eita1=0.8759
d, f = ismember([c_tfn[0] for c_tfn in C_TFName.flatten()],[tfn[0] for tfn in TFName.flatten()])
C_TFName = C_TFName[d == 1,:]
TFS = TFS[d==1,:]
CR_TF = CR_TF[:, d == 1]
TFB = TF_binding[f[d==1],:]
C_TFExp = np.zeros((len(C_TFName),1))
d, f = ismember([c_tfn[0] for c_tfn in C_TFName.flatten()], Symbol)
C_TFExp[d==1] = np.log(1+G[f[d==1]][:,np.newaxis])
TFBO = (np.tile(C_TFExp,(1,len(Opn)))*np.tile(C_TFExp/TFS,(1,len(Opn)))*TF_binding*np.tile(Opn.T,(len(TFName),1)))**0.25
CRB=eita0+eita1*np.dot(CR_TF,TFBO)
CRB_P=1-np.exp(CRB.T)/(1+np.exp(CRB.T))
file_out = open('result/CR_binding_pval.txt', 'w')
for i in range(len(CRName)-1):
    file_out.write(str(CRName[i])+'\t')
file_out.write(str(CRName[i+1])+'\n')
df = pd.DataFrame(CRB_P)
df.to_csv(file_out, mode='a', header=False)  

####################

alhfa = 0.5
Opn_median = np.log(1+Opn_median)
Opn1 = np.log(1+Opn)
Opn = Opn1*(Opn1/(Opn_median+0.5))
geneName = np.intersect1d(Symbol,List).tolist()
geneName = [gn[0] if type(gn) == np.ndarray else gn for gn in geneName]
d, f = ismember(geneName, [li[0][0] for li in List])   
R2 = R2[:,f]
Exp_median = Exp_median[f]
d, f = ismember(geneName, Symbol)
G = G[f]
f1 = np.argsort(G)
d2 = np.sort(Exp_median)
G1 = np.array([0]*len(f1))
G1[f1[:,np.newaxis]] = d2
G = (G1**alhfa)*(G1/(Exp_median+0.5))
d, f = ismember([tfn[0] for tfn in TFName.flatten()], geneName)
TFName = TFName[d==1,:]
TF_binding = TF_binding[d==1,:]
TFExp = G[f[d==1]]
R2 = R2[d==1,:]
#######################################################
fileID = pd.read_csv('data/knownResults_TFrank.txt',header=None, sep='\t')
knowTF = fileID.iloc[:,0].values
knowTFScore = fileID.iloc[:,1].values
d, f = ismember([tfn[0] for tfn in TFName.flatten()],knowTF) 
TF_motif = np.zeros(len(TFName))
TF_motif[d==1] = knowTFScore[f[d==1]]
TFExp = TFExp*(TF_motif.T[:,np.newaxis])
#######################################################
fileID = pd.read_csv('data/peak_gene_100k_corr.bed',header=None, sep ='\t')
C_1 = fileID.iloc[:,0].values
C_2 = fileID.iloc[:,1].values
C_3 = fileID.iloc[:,2].values
C_4 = fileID.iloc[:,3].values
d, f = ismember(C_1, Element_name)
d1, f1 = ismember(C_2, geneName)
f2, ia, ic = np.unique(np.vstack((f[d*d1==1],f1[d*d1==1])), return_index=True, return_inverse=True,axis=1)
c3 = accumarray_Min(ic, C_3[d*d1==1])
c4 = accumarray_Min(ic, C_4[d*d1==1])
c4[c4<0.2] = 0
d0 = 500000
c=np.exp(-1*c3/d0)*c4
H1=coo_matrix((c,(f2[1,:],f2[0,:])),shape=((len(geneName),len(Element_name))))
TFO=TF_binding*np.tile(Opn.T,(np.size(TF_binding,0),1))
BOH=np.dot(TFO,H1.T.toarray())
Score=(np.dot(TFExp,G.T))*(2**np.abs(R2))*BOH
Score[np.isnan(Score)]=0
############start-output###############
df = pd.DataFrame(Score)
df.to_csv('result/TFTG_regulationScore.txt',sep='\t',index=False,header=False)
filename='data/TFName.txt'
fid = open(filename,'w')
for i in range(len(TFName)):
    fid.write(str(TFName[i])+'\n')
fid.close()
filename='data/TGName.txt';
fid = open(filename,'w');
for i in range(len(geneName)):
    fid.write(str(geneName[i])+'\n')
fid.close()
############end-output#################
#################################
TFTG_mouse=scio.loadmat('data/TFTG_mouse_nagetriveControl.mat')
Back_net=TFTG_mouse['Back_net']
d, f = ismember([bn[0] for bn in Back_net[:,0]], [tfn[0] for tfn in TFName.flatten()])
d1, f1 = ismember([bn[0] for bn in Back_net[:,1]], geneName)
f2 = np.vstack((f[d*d1==1], f1[d*d1==1]))
Back_score=Score.flatten('F')[np.array(f2[1,:])*np.size(Score,0)+f2[0,:]]
Cut = np.percentile(Back_score, 99, interpolation='midpoint')
b, a = np.where(Score.T>Cut)
c=[np.size(Score,0)*j+i for j in range(np.size(Score,1)) for i in range(np.size(Score,0)) if Score[i,j]>Cut ]
c1=Score.flatten('F')[c]
Net=np.vstack((np.array([f[0][0] for f in TFName])[a], np.array(geneName)[b])).T
H1 = H1.toarray()
a1 = np.sort(H1.T,axis=0)[::-1]
a2 = np.argsort(H1.T,axis=0)[::-1] 
a1=a1[range(10),:]
a2=a2[range(10),:]
TFTG_RE=np.array([";".join(Element_name[a2[(TFO[a[i],a2[:,b[i]]]>0)*(a1[:,b[i]]>0) ==1,b[i]]]) for i in range(len(a))])
d = np.sort(c1,axis=0)[::-1]
f = np.argsort(c1,axis=0)[::-1] 

Net=np.hstack((Net[f], d.reshape(-1,1), TFTG_RE[f].reshape(-1,1)))
filename='result/RAd4_network.txt';
fid=open(filename,'w');
fid.write('%s\t'%('TF'))
fid.write('%s\t'%('TG'))
fid.write('%s\t'%('Score'))
fid.write('%s\t'%('FDR'))
fid.write('%s\n'%('REs'))
for i in range(np.size(Net,0)):
    fid.write('%s\t'%(Net[i,0]))
    fid.write('%s\t'%(Net[i,1]))
    fid.write('%g\t'%(float(Net[i,2])))
    fid.write('%g\t'%((np.sum(Back_score>d[i])+1)/len(Back_score)))
    fid.write('%s\n'%(Net[i,3]))
fid.close()

def mfbs(TFName,Element_name,motifName,motifWeight,Match2):
    MotifTarget = pd.read_csv('data/MotifTarget.txt',header=None,sep ='\t')
    f3 = MotifTarget.iloc[:,2]
    d1, f1 = ismember(MotifTarget.iloc[:,0],Element_name)
    d2, f2 = ismember(MotifTarget.iloc[:,1],[mn[0][0] for mn in motifName])
    t1 = np.setdiff1d(range(len(motifName)), np.unique(f2))
    f2 = np.hstack((f2[d1*d2 ==1], t1)) 
    f1 = np.hstack((f1[d1*d2 ==1],np.ones(len(t1))))
    f3 = np.hstack((f3[d1*d2 ==1],np.zeros(len(t1))))
    t1 = np.setdiff1d(range(len(Element_name)), np.unique(f1))
    f1 = np.hstack((f1,t1))
    f2 = np.hstack((f2, np.ones(len(t1)))) 
    f3 = np.hstack((f3, np.zeros(len(t1))))
    Motif_binding = coo_matrix((f3,(f2,f1)),shape=(len(motifName), len(Element_name)))
    Motif_binding = np.diag(1/(motifWeight.flatten()+0.1))*Motif_binding
    Motif_binding = np.log(1+Motif_binding)
    TF_binding = np.zeros((len(TFName),len(Element_name)))
    d, f1 = ismember([m2[0] for m2 in Match2[:,0]],[mn[0][0] for mn in motifName])
    d, f2 = ismember([m2[0] for m2 in Match2[:,1]],[tf[0][0] for tf in TFName])
    for i in range(len(TFName)):
        a = [index for index,tmp in enumerate(f2) if tmp ==i]
        if len(a)>1:
            TF_binding[i,:] = np.max(Motif_binding[f1[a],:],axis=0)
        elif len(a)==1:
            TF_binding[i,:] = Motif_binding[f1[a],:]
        else:
            TF_binding[i,:] = np.zeros(len(Element_name))
    #TF_binding = coo_matrix(TF_binding)
    return TF_binding


def ismember(a,b):
    b_dict = {b[i]:i for i in range(0,len(b))}
    indices = [b_dict.get(x, -1) for x in a] 
    booleans = np.in1d(a,b)
    return booleans ,np.array(indices, dtype=np.int)


def accumarray_Min(subs, weight):
    return (np.array([np.min(weight[index]) if index != [] else 0 for index in [list(np.where(subs==z)) 
            if z in subs else list() for z in np.arange(min(subs),max(subs)+1)]]))


