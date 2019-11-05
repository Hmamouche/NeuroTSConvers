import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from celluloid import Camera
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

colors = ["darkblue", "brown", "slategrey", "darkorange", "red", "grey","blue", "indigo", "darkgreen"]

# READ PREDICTIONS
df = pd. read_csv ("demo/predictions.csv", sep = ';', header = 0, index_col = 0)
regions = df. columns
index = df. index
title = 'Brain activity predictions'


# SAVE LEGENDS SEPARATELY
fig = plt.figure()
fig_legend = plt.figure(figsize=(3, 1.5))
ax = fig.add_subplot(111)
bars = ax.bar(range(4), range(4), color=colors[0:len (regions)], label=regions)
fig_legend.legend(bars.get_children(), regions, loc='center', frameon=False)
fig_legend. savefig ("demo/legend.png")
plt.clf ()
plt. cla ()
plt. close ()


# SAVE PREDICTIONS AS A VIDEO
fig, ax = plt.subplots (nrows = len (regions), ncols = 1, figsize=(10,6),  sharex=True)
fig.text(0.5, 0.04, 'Time (s)', ha='center')
fig.text(0.04, 0.5, title, va='center', rotation='vertical')

camera = Camera(fig)
legend_image = plt. figure (figsize = (3,5))


for j in range (len (regions)):
	ax [j]. set_xlim (np.min (index), np. max (index) + 1)
	#ax [j]. set_xticks (index)
	#ax [j]. set_xticklabels (index, rotation=-10)
	ax [j]. xaxis.set_minor_locator(MultipleLocator(5))
	ax [j]. set_ylim (0, 1.1)

for i in range (1,len (index)):
	for j in range (len (regions)):
		ax[j]. plot (index [:i], df. iloc [:i, j], linewidth = 2, color = colors [j])
	camera.snap()

animation = camera.animate (repeat = False, interval = 1200)

animation.save('demo/predictions_video.mp4')
#plt. show ()
