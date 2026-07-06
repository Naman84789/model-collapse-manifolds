import numpy as np

data = {
 ('unf',0):[20.3693,20.7590,20.9482,24.1950,22.4064,21.5588,23.6090,21.6922],
 ('unf',1):[20.5738,22.4452,22.3693,23.7238,22.8924,25.3032,25.2588,23.8071],
 ('unf',2):[22.0095,22.8561,22.3604,22.4769,23.9381,22.0913,21.5545,21.6605],
 ('fix',0):[18.4863,17.7967,16.1853,16.9317,16.9281,16.2968,16.1844,16.3205],
 ('fix',1):[15.9864,15.9193,14.8740,15.6840,15.5338,16.3904,16.0388,15.5606],
 ('fix',2):[16.7197,15.9353,15.8137,17.4464,16.6250,16.9882,16.4627,16.3018],
}
TRUE=20.1490
g=np.arange(8)

print("=== TREND ANALYSIS (positive slope = compounding degradation = collapse) ===")
hdr = "{:10s} {:>6s} {:>6s} {:>6s} {:>5s} {:>10s} {:>7s}".format(
    "arm/seed","g0","g7","mean","std","slope/gen","g0->g7")
print(hdr)
for arm in ["unf","fix"]:
  slopes=[]
  for s in [0,1,2]:
    y=np.array(data[(arm,s)])
    slope=np.polyfit(g,y,1)[0]; slopes.append(slope)
    print("{:s} s{:<6d} {:6.2f} {:6.2f} {:6.2f} {:5.2f} {:+10.3f} {:+7.2f}".format(
        arm,s,y[0],y[-1],y.mean(),y.std(),slope,y[-1]-y[0]))
  print("  -> {:s} mean slope across seeds = {:+.3f}/gen (std {:.3f})\n".format(
      arm,np.mean(slopes),np.std(slopes)))

print("=== EXCLUDING g0 (g0 = first gen, trained mostly on real data) ===")
for arm in ["unf","fix"]:
  vals=np.array([data[(arm,s)][1:] for s in [0,1,2]]).flatten()
  print("  {:s}: g1-g7 pooled = {:.2f} +/- {:.2f}  (TRUE baseline {:.2f})".format(
      arm,vals.mean(),vals.std(),TRUE))

print("\n=== GAP (unfixed - fixed), g1-g7 per seed ===")
gaps=[]
for s in [0,1,2]:
  u=np.array(data[("unf",s)][1:]); f=np.array(data[("fix",s)][1:])
  gp=u.mean()-f.mean(); gaps.append(gp)
  print("  seed{:d}: gap={:+.2f}  (unf {:.2f} vs fix {:.2f})".format(s,gp,u.mean(),f.mean()))
print("  across seeds: {:.2f} +/- {:.2f}  (min {:.2f}, all positive: {})".format(
    np.mean(gaps),np.std(gaps),min(gaps),all(x>0 for x in gaps)))

print("\n=== KEY QUESTION: is unfixed ABOVE true baseline, fixed BELOW? ===")
for arm in ["unf","fix"]:
  vals=np.array([data[(arm,s)][1:] for s in [0,1,2]]).flatten()
  rel = vals.mean()-TRUE
  print("  {:s}: {:+.2f} vs true baseline ({} true data)".format(
      arm, rel, "WORSE than" if rel>0 else "closer to ref than"))

print("\n=== VARIANCE within arm (stability) ===")
for arm in ["unf","fix"]:
  allstd=[np.std(data[(arm,s)]) for s in [0,1,2]]
  print("  {:s}: per-seed within-run std = {}  mean {:.2f}".format(
      arm,[round(x,2) for x in allstd],np.mean(allstd)))
