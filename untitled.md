Marble meeting 4/2

Next steps
1. make plots in viz_find_peaks including all oscillators from somata.
2. include results somata for 'damping a' and 'sigma2' for initParams SSPE model
3. build average oscilator framework
4. implement the rules below

for somata results: 
-use knee_osc, if knee_osc + 1 is hg include that too
-if knee_osc = 0, look ahead if there is other model with a hg osc. if not, let knee_osc be 1 or 2

rule 1
for channels that don't have a HG oscillator: 
1.1 substitude all oscillators with a set of average oscillaters from channels that do have hg
or 
1.2 add an average hg oscillator set constructed from channels that have hg

rule 2
for all chanels, use set of average oscillaters from channels that do have hg

rule 3
find with a multivariate approach with a ML model which channels are usefull for the decoder (with bandpass feature data)
for usefull channels: use rule 1.2
for useless channels: use rule 2

using correlation with melspec is not a good way. This is because some channels are only usefull in combination with other channels. those can be VERY informative together, but not on their own. so you need a more sophisticated method for that which evaluates the channels as informative as a set

---------------------------