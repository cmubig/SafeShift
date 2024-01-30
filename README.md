# SafeShift: Safety-Informed Distribution Shift for Robust Trajectory Prediction in Autonomous Driving 

As autonomous driving technology matures, safety and robustness of its key components, including trajectory prediction, is vital. Though real-world datasets, such as Waymo Open Motion, provide realistic recorded scenarios for model development, they often lack truly safety-critical situations. Rather than utilizing unrealistic simulation or dangerous real-world testing, we instead propose a framework to characterize such datasets and find hidden safety-relevant scenarios within. Our approach expands the spectrum of safety-relevance, allowing us to study trajectory prediction models under a safety-informed, distribution shift setting. We contribute a generalized
scenario characterization method, a novel scoring scheme to find subtly-avoided risky scenarios, and an evaluation of trajectory prediction models in this setting. We further contribute a remediation strategy, achieving a 10% average reduction in prediction collision rates. To facilitate future research, we release our code to the public: github.com/cmubig/SafeShift

Repository based off of MTR: https://github.com/sshaoshuai/MTR  
