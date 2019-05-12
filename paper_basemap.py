import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd

# m = Basemap(projection='lcc', lat_0=37, lon_0=137, width=1500000, height=2000000, resolution="h")
# m.shadedrelief()
# m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')
# m.drawmapboundary(fill_color="#DDEEFF")  # 绘制边界
# m.drawstates()        # 绘制州
# m.drawcoastlines()    # 绘制海岸线
# m.drawcountries()     # 绘制国家
# # m.drawcounties()      # 绘制县
# parallels = np.arange(30., 44, 4)
# m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10, linewidth=0.5)  # 绘制纬线
# meridians = np.arange(128., 146., 4)
# m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10, linewidth=0.5)  # 绘制经线
# plt.show()


latlon1 = np.load('event_latlon.npy')
latlon2 = np.load('event_latlon57.npy')
latlon = np.concatenate((latlon1, latlon2), axis=0)

event_label1 = np.load('event_magn.npy')
event_label2 = np.load('event_magn57.npy')
event_label = np.concatenate((event_label1, event_label2), axis=0)
df = pd.DataFrame({'event': latlon[:, 2], 'lat': latlon[:, 0], 'lon': latlon[:, 1], 'magn': event_label[:, 0]})
df['event'] = df['event'].apply(lambda x: x[:-1] if len(x) == 17 else x)
df['event'] = df['event'].apply(lambda x: x[-10:])
df['lat'] = df['lat'].apply(lambda x: np.float(x))
df['lon'] = df['lon'].apply(lambda x: np.float(x))
df['magn'] = df['magn'].apply(lambda x: np.float(x))
df['s'] = df['magn'].apply(lambda x: (x-2)*(x-2)*(x-2))

lat = list(df.groupby(['event'])['lat'].mean())
lon = list(df.groupby(['event'])['lon'].mean())
magn = list(df.groupby(['event'])['magn'].mean())
s = list(df.groupby(['event'])['s'].mean())
print(len(lat))


m = Basemap(projection='stere', lat_0=37, lon_0=137, width=1500000, height=2000000, resolution="h")
# m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')
parallels = np.arange(30., 44, 4)
m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10, linewidth=0.5)  # 绘制纬线
meridians = np.arange(128., 146., 4)
m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10, linewidth=0.5)  # 绘制经线

m.scatter(lat, lon, latlon=True, c=magn, s=s, cmap='Reds', alpha=0.5)# 3. 构建色彩轴与图例
plt.colorbar(label=r'$M_W$')

for a in [4, 5, 7]:
    show_a = a
    a = (a-2)*(a-2)*(a-2)
    plt.scatter([], [], c='k', alpha=0.5, s=a, label='$M_W$' + "  " + str(show_a))
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='lower right')

plt.clim(3, 7)
plt.show()
