# -*- coding: utf-8 -*-
#!/usr/bin/python
# Tempco plot demo app, requires matplotlib and Python 3.9
from colorama import just_fix_windows_console
just_fix_windows_console()
print ("\033[32;1mRevision 0.2 JAN.13.2025 | xDevs.com TCKIT plot tool")
print ("  (C) Illya Tsemenko | https://xdevs.com\033[39;0m")

import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import math
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import scipy.ndimage.filters
import scipy.interpolate
import os
from datetime import datetime
import matplotlib.dates as mdates
cwd = os.getcwd()

if (len(sys.argv) <= 1):
    fn = "mi6010rfm_rs_t5685_10r_rx_xg9330_100_tcr_3458a_sr1010_r2_jan2025.dsv"   # Hardcoded filename
else:
    fn = sys.argv[1]                               # Read data input filename from command line arg as alternative

fnout = cwd + "\\" + fn.split(".dsv")[0] + "_sorted.dsv"
print ("\033[44;33m-i- Input file: %s  \033[49;0m" % fn)
## ------------------ Configuration variables --------------------------------------------------- ##

start_sample = 500                          # Start sample for analysis
end_sample   = 54500                        # End sample for analysis
datespan = 2                                # Horizontal timescale resolution , hours/div

enable_detrending = True  # Apply data linear detrending if set to True
min_temp = 17.5                               # Minimum temperature for scale
max_temp = 28.5                               # Maximum temperature for scale
reftemp  = 23                               # Reference temperature for calculations, typically 23 C
rs = 100.003368                             # Reference RS resistance for ratio calculation (DCC method)
ref_range1   = 10
ref_val1   = 10
ref_val2   = 10
ref_val3   = 10
ref_val4   = 10
ref_val5   = 10.0
ref_val6   = 100
ref_val7   = 10
ref_val8   = 10

title_label = r'DUT: Tinsley 5685A S/N 262283, Transfer from G9330-100, 31.6 mA, 6s' # µA 
axis2_x_label = "Timescale"
axis4_x_label = "DUT Airbath temperature, °C"
stats_units = "Ω"

axis2_y_label = ("Sensor temperatures, °C" )
axis3_y_label = "Ambient temperature, °C"
axis3_yb_label = "Ambient humidity, %RH"
axis4_y_label = "DUT1 Deviation relative to temperature, µΩ/Ω"
axis5_y_label = "DUT2 Deviation relative to temperature, µΩ/Ω"
axis6_y_label = ("RT, µΩ/Ω of nominal")

chart_name  = ("DUT TCR sweep with DCC, µΩ/Ω to nominal %g Ω" % round(ref_val5, 3)) 
chart2_name = "Airbath parameters, DUT temperature"
chart3_name = "Environment conditions"
chart4_name = "Tinsley T5685A2 Temperature coefficient relative to airbath"
chart5_name = "SR1010-100 R2 Temperature coefficient relative to airbath"
chart6_name = "TCR sweep with SR1010-100 R2, µΩ/Ω to nominal 100 Ω"

axis_y_max   = 5
axis_y_min   = -5
axis_y_step  = 1.0
axis_y_label = "Relative error"
axis_x_max   = 10
axis_x_min   = 0
axis_x_step  = 0.1
axis_x_label = "Timescale"

axis_y_label = axis_y_label + (", µ%s/%s vs calibrated R23 value" % (stats_units, stats_units))

ax2_legend_labels = ['PT100 in DUT, °C', 'NTC in test air-bath, °C', 'Programmed temperature, °C']

dataset_name1  =  "Tinsley 5685A2 S/N 262283"
dataset_name2  =  "HP 3458B (Reference 2)"
dataset_name3  =  "HP 3459 X9D (Reference 3)"
dataset_name4  =  "HP 34420A, INL fit"
dataset_name5  =  "Tinsley 5685A2"
dataset_name6  =  "ESI SR1010-100, R2"
dataset_name7  =  "Keithley 2002+1801"
dataset_name8  =  "1σ error "
dataset_name9  =  "Datron 1281 20V, RESL8, FAST_OFF"
dataset_name10 =  "3458D (peak)"
dataset_name11 =  "8508A (peak)"
dataset_name12 =  "K182-M (peak)"

# Various matplotlib settings
img_resolution_w  = 2800  # Output plot resolution in pixels, full-size
img_resolution_h  = 3400  # Output plot resolution in pixels, full-size
img_resolution_ws = 1400  # Output plot resolution in pixels for thumbnail
img_resolution_hs = 1800  # Output plot resolution in pixels for thumbnail
dateformat_parse  = "%d/%m/%Y-%H:%M:%S"
show_plot         = False

dataset_color1  = '#4bd7fb'
dataset_color2  = '#7e04e9'
dataset_color3  = '#0700f8'
dataset_color4  = '#0fb6b5'
dataset_color5  = '#f01f69'
dataset_color6  = '#ff9900'
dataset_color5l = '#d21ae8'
dataset_color6l = '#ffc60c'
dataset_color7  = '#f8c400'
dataset_color8  = '#f749ff'
dataset_color9  = '#0000ff'
dataset_color10 = '#ff0000'
dataset_color11 = '#ffcc22'
dataset_color12 = '#0000ff'
xdevs_bg_color  = '#eeeeff'
## ----------------------------------------------------------------------------------------------- ##

# Resort data from input random DSV
#df = pd.read_csv(fn , skiprows=0, sep=';', index_col = 0)
#print ("-i- Read file into DF [A]")
#print ("\033[31;1m", df.head(8))
#df_out = df.sort_values(by=['duta'])
#print ("\033[31;1m", df_out.head(8))
#print (fnout)
#df_out.to_csv(fnout, sep=';')# header=None)

idxr = []
cntr = []

val1 = []
val2 = []
val3 = []
val4 = []
val5 = []
val6 = []
val7 = []
val8 = []
tmp116 = []
tmp117 = []
tmp119 = []
raw1 = []
raw2 = []
raw3 = []
raw4 = []
raw5 = []
raw6 = []
raw7 = []
raw8 = []

chdiff = []
gain1diff = []
gain2diff = []
gainhdiff = []
error1 = []
error2 = []
error3 = []
error4 = []
error5 = []
error6 = []
ambient = []
rh = []
pressure = []
count = 0
count_cut = 0
stamp = []

ofs1 = 0.0
ofs2 = 0.0
ofs3 = 0.0
ofs4 = 0.0
ofs5 = 0.0
ofs6 = 0.0
ofs7 = 0.0
ofs8 = 0.0
sv_temp = []
ysi_a = 1.032e-3
ysi_b = 3891
ysi_c = 1.58e-7

with open(fn) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
    is_first = True
    for row in spamreader:
        if is_first:
            is_first = False
            continue
        count_cut = count_cut + 1
        if (start_sample < count_cut < end_sample):
            idxr.extend([row[0]]) #3458
            stamp.extend([datetime.strptime(row[0], dateformat_parse)])
            count = count + 1
            cntr.extend([count])
            val1.extend([ ofs1 + ((float(row[11])  / ref_val1) - 1) * 1e6 ])   # vbl1 : 3458A
            val2.extend([ ofs2 + ((float(row[2]) / ref_val2) - 1) * 1e6 ])  # vbl2 : 3458B
            val3.extend([ ofs3 + ((float(row[3]) / ref_val3) - 1) * 1e6 ])  # vbl3 : 3459X
            val4.extend([ ofs4 + ((float(row[4]) / ref_val4) - 1) * 1e6 ])  # vbl4 : na
            val5.extend([ ofs5 + (( rs / float(row[4]) / ref_val5) - 1) * 1e6 ])  # vbl5 : 34420-ch1
            val6.extend([ ofs6 + ((float(row[10]) / ref_val6) - 1) * 1e6 ])  # vbl6 : 34420-ch2
            val7.extend([ ofs7 + ((float(row[7]) / ref_val7) - 1) * 1e6 ])  # vbl7 : 
            val8.extend([ ofs8 + ((float(row[8]) / ref_val8) - 1) * 1e6 ])  # vbl8 :
            
            raw1.extend([ float(row[11])  ])   # vbl1 : 3458A
            raw2.extend([ float(row[2]) ])  # vbl2 : 3458B
            raw3.extend([ float(row[3]) ])  # vbl3 : 3459X
            raw4.extend([ float(row[4]) ])  # vbl4 : na
            raw5.extend([ rs / float(row[4]) ])  # vbl5 : 34420-ch1
            raw6.extend([ float(row[10]) ])  # vbl6 : 34420-ch2
            raw7.extend([ float(row[7]) ])  # vbl7 : 
            raw8.extend([ float(row[8]) ])  # vbl8 : 
            
            #tmp116.extend([ ((float(row[14]) - 10000.76) / 10000.76) * 1e3 + 23 ]) # ch1
            sv_temp.extend([ float(row[31]) ]) # ch1
            tmp117.extend([ float(row[28]) ])
            #tmp117.extend([(1 / ( ( math.log(float(row[11]) / 10000) / ysi_b + 1 / (23 + 273.15) ) ) ) - 273.15 ]) # ch1
            tmp119.extend([float(row[29]) - 0.4 ]) # ch1
            
            chdiff.extend([ (((float(row[10]) - 10000.76) / 10000.76) * 1e3 + 23) - (float(row[29]) - 0.4) ]) # ch-ch difference nV
            #gain1diff.extend([ (( float(row[5]) / float(row[1])) - 1) * 1e6 ])
            #gain2diff.extend([ (( float(row[6]) / float(row[1])) - 1) * 1e6 ])
            #gainhdiff.extend([ (( knv / float(row[6])) - 1) * 1e6 ])
            
            ambient.extend([float(row[25])]) # 3458
            rh.extend([float(row[26])]) # 2002
            pressure.extend([float(row[27])]) # 5720
        else:
            continue
        
        #real5.extend([float(row[6])])
        #real6.extend([float(row[7])])

print ("-i-\033[34;1m Analysis samples: %d\033[39;1m " % cntr[-1])
first_stamp = stamp[0]
last_stamp = stamp[-1]
diffstp = (stamp[-1] - stamp[0]) / 32
print ("\033[31;2m-i-", "First sample TS: ",first_stamp, "| Last sample TS: ", last_stamp, "\r\nData duration", diffstp, "\033[39;0m\r\n")

val5_detrended = []
val6_detrended = []
raw5_detrended = []
raw6_detrended = []

amb_max = np.amax(ambient)
amb_min = np.amin(ambient)
amb_span = amb_max - amb_min

press_max = np.amax(pressure)
press_min = np.amin(pressure)
press_span = press_max - press_min

tmp119_max = np.amax(tmp119)
tmp119_min = np.amin(tmp119)
tmp119_span = tmp119_max - tmp119_min

rh_max = np.amax(rh)
rh_min = np.amin(rh)
rh_span = rh_max - rh_min

print ("\033[35;1m-d- Median relative start for val5 = %.5f µ%s/%s" % (np.median(val5[:32]) , stats_units, stats_units) )
print ("\033[35;1m-d- Median relative start for val6 = %.5f µ%s/%s" % (np.median(val6[:32]) , stats_units, stats_units) )
print ("\033[36;1m-d- Median relative end for val5 = %.5f µ%s/%s" % (np.median(val5[-32:]), stats_units, stats_units) )
print ("\033[36;1m-d- Median relative end for val6 = %.5f µ%s/%s" % (np.median(val6[-32:]), stats_units, stats_units) )
print ("\033[33;2m-d- Median RAW for val5 = %.9f %s" % (np.median(raw5), stats_units) )
print ("\033[33;2m-d- Median RAW for val6 = %.9f %s" % (np.median(raw6), stats_units) )

start_ch5 = np.median(val5[:48])
end_ch5 = np.median(val5[-48:])
print ("start ch5 : ", start_ch5, "end ch5: ", end_ch5)
drift_ch5 = (start_ch5 - end_ch5)
if (enable_detrending == True):
    drift_rate_ch5 = (drift_ch5 / cntr[-1])
else:
    drift_rate_ch5 = 0
print ("drift ch5: ", drift_ch5)
print ("drift rate ch5: ", drift_rate_ch5 )

start_ch6 = np.median(val6[:48])
end_ch6 = np.median(val6[-48:])
print ("start ch6 : ", start_ch6, "end ch6: ", end_ch6)
drift_ch6 = (start_ch6 - end_ch6)
if (enable_detrending == True):
    drift_rate_ch6 = (drift_ch6 / cntr[-1])
else:
    drift_rate_ch6 = 0
print ("drift rate ch6: ", drift_rate_ch6 )

start_ch5r = np.median(raw5[:32])
end_ch5r = np.median(raw5[-32:])
print ("start ch5r : ", start_ch5r, "end ch5r: ", end_ch5r)
drift_ch5r = (start_ch5r - end_ch5r)
if (enable_detrending == True):
    drift_rate_ch5r = (drift_ch5r / cntr[-1])
else:
    drift_rate_ch5r = 0
print ("drift ch5r: ", drift_ch5r)
print ("drift rate ch5r: ", drift_rate_ch5r )

start_ch6r = np.median(raw6[:32])
end_ch6r = np.median(raw6[-32:])
print ("start ch6r : ", start_ch6r, "end ch6r: ", end_ch6r)
drift_ch6r = (start_ch6r - end_ch6r)
if (enable_detrending == True):
    drift_rate_ch6r = (drift_ch6r / cntr[-1])
else:
    drift_rate_ch6r = 0
print ("drift ch6r: ", drift_ch6r)
print ("drift rate ch6r: ", drift_rate_ch6r )

for ditr in range(0, cntr[-1]):
    val5_detrended.extend([ val5[ditr] + drift_rate_ch5 * ditr ])
    val6_detrended.extend([ val6[ditr] + drift_rate_ch6 * ditr ])
    raw5_detrended.extend([ raw5[ditr] + drift_rate_ch5r * ditr ])
    raw6_detrended.extend([ raw6[ditr] + drift_rate_ch6r * ditr ])

residual_ch6 = np.median(val6_detrended[:32]) - np.median(val6_detrended[-32:])

filtered1 = scipy.ndimage.gaussian_filter1d(val1, sigma=1)
filtered2 = scipy.ndimage.gaussian_filter1d(val2, sigma=1)
filtered3 = scipy.ndimage.gaussian_filter1d(val3, sigma=1)
filtered5 = scipy.ndimage.gaussian_filter1d(val5, sigma=1)
filtered6 = scipy.ndimage.gaussian_filter1d(val6, sigma=1)

filtered5_dt = scipy.ndimage.gaussian_filter1d(val5_detrended, sigma=1)
filtered6_dt = scipy.ndimage.gaussian_filter1d(val6_detrended, sigma=1)

filtered5_min = np.min(filtered5)
filtered5_max = np.max(filtered5)
#ax.yaxis.set_ticks(np.arange(round(filtered5_min,0) - 0.1, round(filtered5_max,0) + 0.1, 0.1))
print (filtered5_min, filtered5_max)

# Calculate polyfits
polych5 = np.polyfit(tmp119, raw5_detrended, 2)
polych6 = np.polyfit(tmp119, raw6_detrended, 2)
polych5_rel = np.polyfit(tmp119, filtered5_dt, 2) # Use detrended data
polych6_rel = np.polyfit(tmp119, filtered6_dt, 2) # Use detrended data
#polych5_rel = np.polyfit(tmp119, filtered5, 2)
#polych6_rel = np.polyfit(tmp119, filtered6, 2)

c2   = polych5[0]
c1   = polych5[1]
gain = polych5[2]

c2b   = polych6[0]
c1b   = polych6[1]
gainb = polych6[2]

tvalue5 = (c2 * reftemp*reftemp + c1*reftemp + gain)
alpha5 = ((c1 + 2*reftemp*c2) / tvalue5*1e6 )
beta5 = (c2 / tvalue5*1e6 )
zerotctemp5 = (-alpha5 / 2 / beta5 + reftemp)

tvalue6 = (c2b * reftemp*reftemp + c1b*reftemp + gainb)
alpha6 = ((c1b + 2*reftemp*c2b) / tvalue6*1e6 )
beta6 = (c2b / tvalue6*1e6 )
zerotctemp6 = (-alpha6 / 2 / beta6 + reftemp)

print ("\033[33;2m\r\n-P- Polyfit ch5 coefficients (B^2,A,V): ", polych5)
print ("\033[34;1m-P- Polyfit ch6 coefficients (B^2,A,V): ", polych6)
print ("\033[33;2m-P- Value[5]: %.9f" % tvalue5, "Alpha 5: %.3f" % alpha5, "Beta 5: %.3f" % beta5, "Zero TC temp 5: %.3f C" % zerotctemp5)
print ("\033[34;1m-P- Value[6]: %.9f" % tvalue6, "Alpha 6: %.3f" % alpha6, "Beta 6: %.3f" % beta6, "Zero TC temp 6: %.3f C\033[39;0m" % zerotctemp6)

print ("TMP119 temp channel data: ", tmp119_min, tmp119_max, tmp119_span)
chdiff_filt = scipy.ndimage.gaussian_filter1d(chdiff, sigma=1)


###------------------------------------------ PLOTTING Section code -------------------------###

fig, ((ax, ax6), (ax4, ax5), (ax2, ax3)) = plt.subplots(3,2)
fig.suptitle(title_label + (", %d samples" % count), size=18)

ax.patch.set_color('#ffffff')
fig.patch.set_facecolor(xdevs_bg_color)
ax.set_clip_on(False)

w = img_resolution_w / 100
h = img_resolution_h / 100
ws = img_resolution_ws / 100
hs = img_resolution_hs / 100
fig.set_size_inches(w,h)
plt.tight_layout(rect=[0.03, 0.03, 0.982, 0.982], w_pad=0.1, h_pad=0.08)                                # Reduce margins around the graph
plt.subplots_adjust(hspace=0.1, wspace=0.08)

ax3b = ax3.twinx()
# Optional pressure scale
#ax3c = ax3.twinx()
#ax3c.spines['right'].set_position(('axes', 1.125))   # Offset Y3 axis 

#ax6.plot(stamp, raw1, dataset_color1, label='Data %i' % (1), alpha=0.7, linewidth=0.2, marker="p", markersize=5, markerfacecolor=dataset_color1)
#ax6.plot(stamp, filtered2, dataset_color2, label='Data %i' % (1), alpha=0.7, linewidth=0.2, marker="p", markersize=5, markerfacecolor=dataset_color2)
#ax6.plot(stamp, filtered3, dataset_color3, label='Data %i' % (1), alpha=0.7, linewidth=0.2, marker="p", markersize=5, markerfacecolor=dataset_color3)
ax.plot(stamp, filtered5   ,dataset_color5l, alpha=0.8, linewidth=1, marker=" ", markersize=3, markerfacecolor=dataset_color5l)
ax.plot(stamp, filtered5_dt,dataset_color5 , alpha=0.8, linewidth=2, marker="p", markersize=3, markerfacecolor=dataset_color5)
ax6.plot(stamp,filtered6   ,dataset_color6l, alpha=0.8, linewidth=1, marker=" ", markersize=3, markerfacecolor=dataset_color6l)
ax6.plot(stamp,filtered6_dt,dataset_color6 , alpha=0.8, linewidth=2, marker="p", markersize=3, markerfacecolor=dataset_color6)

ax4.plot(tmp119, filtered5_dt, dataset_color5, alpha=0.7, linewidth=1, marker="p", markersize=5, markerfacecolor=dataset_color5)
ax4.plot(tmp119, filtered5, dataset_color5l, alpha=0.5, linewidth=1, marker=" ", markersize=2, markerfacecolor=dataset_color5l)
ax4.plot(tmp119, np.polyval(polych5_rel, tmp119), '#000000', linestyle='dotted', alpha=1, linewidth=2, marker=" ")

print (filtered5)
print (np.polyval(polych5_rel, tmp119))
print ("---- CH6 \r\n ", filtered6_dt)
print (np.polyval(polych6_rel, tmp119))

ax5.plot(tmp119, filtered6_dt, dataset_color6 , alpha=0.7, linewidth=1, marker="p", markersize=5, markerfacecolor=dataset_color6)
ax5.plot(tmp119, filtered6   , dataset_color6l, alpha=0.5, linewidth=1, marker=" ", markersize=5, markerfacecolor=dataset_color6l)
ax5.plot(tmp119, np.polyval(polych6_rel, tmp119), '#000000', linestyle='dotted', alpha=1, linewidth=2, marker=" ")

#ax5.plot(stamp, chdiff_filt, dataset_color12, label='Data %i' % (1), alpha=0.7, linewidth=0.2, marker="p", markersize=5, markerfacecolor=dataset_color12)

#ax2.plot(real4, filtered4b, dataset_color4, label='Data %i' % (4), alpha=0.8, linewidth=0.2, marker="X", markerfacecolor=dataset_color4)

#ax2.errorbar(real, filtered1b, yerr=error1, fmt=' ', alpha=0.3, linewidth=0.5, capsize=3, color=dataset_color1)

#ax.errorbar(ideal, filtered1, yerr=error1, fmt=' ', linewidth=0.5, alpha=0.9, capsize=3, color=dataset_color1)

#ax2.plot(stamp, tmp116, "#0055aa", label='Data %i' % (4), alpha=0.8, linewidth=0, marker="^", markersize=2, markerfacecolor="#88aaaa")
ax2.plot(stamp, tmp119, "#50e991", label='Data %i' % (4), alpha=0.8, linewidth=2, marker="+", markersize=2, markerfacecolor="#8be04e")
ax2.plot(stamp, tmp117, "#fd7f6f", alpha=1, linewidth=2, marker=".", markersize=1, markerfacecolor="#e50049")
ax2.plot(stamp, sv_temp, "#111111", alpha=1, linewidth=1, linestyle='dashed', marker=" ", markersize=1, markerfacecolor="#000000")

ax3.plot(stamp, ambient, "#0000ee", label='Ambient T, °C', alpha=1, linewidth=1, marker="^", markersize=1, markerfacecolor="#0000ee")
ax3b.plot(stamp, rh, "#009900", label='Humidity, %%RH', alpha=1, linewidth=1, marker="s", markersize=1, markerfacecolor="#009900")
#Optional pressure data
#ax3b.plot(stamp, pressure, "#ee0000", label='Pressure, hPa', alpha=1, linewidth=1, marker="o", markersize=1, markerfacecolor="#ff0000")

ax3.axis([axis_x_min, axis_x_max, amb_min * 0.99, amb_max * 1.01])
ax3b.axis([axis_x_min, axis_x_max, rh_min * 0.99, rh_max * 1.01])
#ax4.axis([tmp119_min, tmp119_max, -10, 10])
ax4.xaxis.set_ticks(np.arange(round(tmp119_min,0) - 1, round(tmp119_max,0) + 1, 1))
ax5.xaxis.set_ticks(np.arange(round(tmp119_min,0) - 1, round(tmp119_max,0) + 1, 1))

#ax5.axis([axis_x_min, axis_x_max, -1, 1])

#ax.axis([axis_x_min, axis_x_max, axis_y_min, axis_y_max])
#ax.axis([21, 25, axis_y_min, axis_y_max])
#ax2.axis([axis_x_min, axis_x_max, axis_y_min, axis_y_max])
#start2, end2 = ax2.get_ylim()
#ax2.axis([0, count, axis_y_min, axis_y_max])
#ax4.axis([0, count, np.amin(filtered1),np.amax(filtered1)])

ax.legend([dataset_name5, dataset_name5 + ", detrended"] , loc='upper center')#, dataset_name5, dataset_name6])
ax6.legend([dataset_name6, dataset_name6 + ", detrended"], loc='upper center')#, dataset_name5, dataset_name6])
ax2.legend(ax2_legend_labels)
ax3.legend(["Ambient temperature, °C"], loc='upper right')
ax3b.legend(["Ambient humidity, %%RH"], loc='upper left')
#ax4.legend(['Interchannel difference (CH1 - CH2), nV'])
ax5.legend([dataset_name6, dataset_name6 + " samples", "Polyfit vs PT100"])
ax4.legend([dataset_name5, dataset_name5 + " samples", "Polyfit vs PT100"])
#, dataset_name7, dataset_name8, dataset_name9, dataset_name10, dataset_name11, dataset_name12

ax.text(0.02, 0.03,("DUT end points: %.3f µ%s/%s" % (drift_ch5, stats_units, stats_units)), horizontalalignment='left', verticalalignment='center', transform = ax.transAxes, bbox=dict(facecolor="#eeffff", alpha=0.55))
ax6.text(0.02, 0.03,("Residual end points: %.3f µ%s/%s" % (residual_ch6, stats_units, stats_units)), horizontalalignment='left', verticalalignment='center', transform = ax6.transAxes, bbox=dict(facecolor="#eeffff", alpha=0.55))

ax4.text(0.02, 0.2,("R23: %9f %s" % (tvalue5, stats_units)), fontsize=12, horizontalalignment='left', verticalalignment='center', transform = ax4.transAxes, bbox=dict(facecolor="#eeeeff", alpha=0.45))
ax4.text(0.02, 0.13,("Alpha R: %.3f µ%s/%s/K" % (alpha5, stats_units, stats_units)), horizontalalignment='left', verticalalignment='center', transform = ax4.transAxes, bbox=dict(facecolor="#eeeeff", alpha=0.45))
ax4.text(0.02, 0.08,("Beta R: %.3f µ%s/%s/K^2" % (beta5, stats_units, stats_units)), horizontalalignment='left', verticalalignment='center', transform = ax4.transAxes, bbox=dict(facecolor="#eeeeff", alpha=0.45))
ax4.text(0.02, 0.01,("Zero TC: %.1f °C" % (zerotctemp5)), horizontalalignment='left', verticalalignment='bottom', transform = ax4.transAxes, bbox=dict(facecolor="#eeeeff", alpha=0.45))

ax5.text(0.02, 0.2,("R23: %9f %s" % (tvalue6, stats_units)), fontsize=12, horizontalalignment='left', verticalalignment='center', transform = ax5.transAxes, bbox=dict(facecolor="#eeeeff", alpha=0.45))
ax5.text(0.02, 0.13,("Alpha R: %.3f µ%s/%s/K" % (alpha6, stats_units, stats_units)), horizontalalignment='left', verticalalignment='center', transform = ax5.transAxes, bbox=dict(facecolor="#eeeeff", alpha=0.45))
ax5.text(0.02, 0.08,("Beta R: %.3f µ%s/%s/K^2" % (beta6, stats_units, stats_units)), horizontalalignment='left', verticalalignment='center', transform = ax5.transAxes, bbox=dict(facecolor="#eeeeff", alpha=0.45))
ax5.text(0.02, 0.01,("Zero TC: %.1f °C" % (zerotctemp6)), horizontalalignment='left', verticalalignment='bottom', transform = ax5.transAxes, bbox=dict(facecolor="#eeeeff", alpha=0.45))

#ax4.text(0.7, 0.15,("Alpha CH2: %.3f %s/V/K" % (alpha6, stats_units)), horizontalalignment='left', verticalalignment='center', transform = ax4.transAxes, bbox=dict(facecolor="#eeeeff", alpha=0.65))
#ax4.text(0.7, 0.09,("Beta CH2: %.3f %s/V/K^2" % (beta6, stats_units)), horizontalalignment='left', verticalalignment='center', transform = ax4.transAxes, bbox=dict(facecolor="#eeeeff", alpha=0.65))
#ax4.text(0.7, 0.03,("Zero TC: %.1f °C" % (zerotctemp6)), horizontalalignment='left', verticalalignment='center', transform = ax4.transAxes, bbox=dict(facecolor="#eeeeff", alpha=0.65))

#ax.text(0.3, 0.05, (" INL Max: %.3f %s, Min %.3f %s (34420 CH1)" % (inl_max ,stats_units , inl_min, stats_units)), horizontalalignment='left', verticalalignment='center', transform = ax.transAxes, bbox=dict(facecolor='white', alpha=0.65))
#ax.text(0.6, 0.05, (" INL span +/-%.3f %s (34420 CH1)" % (inl_span/2, stats_units)), horizontalalignment='left', verticalalignment='center', transform = ax.transAxes, bbox=dict(facecolor='white', alpha=0.65))
#
#ax2.text(0.1, 0.05, (" SDEV CH1 Max: %.4f %s, Min %.4f %s" % (err1_max , stats_units,  err1_min, stats_units)), horizontalalignment='left', verticalalignment='center', transform = ax2.transAxes, bbox=dict(facecolor='white', alpha=0.65))
#ax2.text(0.6, 0.05, (" SDEV CH1 span +/-%.4f %s, Median %.4f %s" % (err1_span/2, stats_units,  np.median(error4), stats_units)), horizontalalignment='left', verticalalignment='center', transform = ax2.transAxes, bbox=dict(facecolor='white', alpha=0.65))

if (amb_span < 0.25):
    temp_box_color = "#cff800"
elif (0.25 >= amb_span < 0.5):
    temp_box_color = "#ffbf65"
elif (0.5 >= amb_span < 1.0):
    temp_box_color = "#ffec59"
else:
    temp_box_color = "#ff5c77"
    
if (amb_span < 15):
    rh_box_color = "#00f8ee"
elif (15 >= amb_span < 60):
    rh_box_color = "#2fff25"
elif (60 >= amb_span):
    rh_box_color = "#ff5c00"

ax3.text(0.02, 0.90,("T Max: %.2f °C, Min %.2f °C " % (amb_max , amb_min)), horizontalalignment='left', verticalalignment='center', transform = ax3.transAxes, bbox=dict(facecolor=temp_box_color, alpha=0.65))
ax3.text(0.6, 0.90, ("T span +/-%.2f °C peak" % (amb_span/2)), horizontalalignment='left', verticalalignment='center', transform = ax3.transAxes, bbox=dict(facecolor=temp_box_color, alpha=0.65))
ax3.text(0.02, 0.11,("H Max: %.2f %%RH, Min %.2f %%RH " % (rh_max , rh_min)), horizontalalignment='left', verticalalignment='center', transform = ax3.transAxes, bbox=dict(facecolor=rh_box_color, alpha=0.65))
ax3.text(0.6, 0.11, ("H span +/-%.2f %%RH peak" % (rh_span/2)), horizontalalignment='left', verticalalignment='center', transform = ax3.transAxes, bbox=dict(facecolor=rh_box_color, alpha=0.65))
ax3.text(0.02, 0.05,("P Max: %.1f hPa, Min %.1f hPa" % (press_max , press_min)), horizontalalignment='left', verticalalignment='center', transform = ax3.transAxes, bbox=dict(facecolor=rh_box_color, alpha=0.65))
ax3.text(0.6, 0.05, ("P span +/-%.1f hPa peak" % (rh_span/2)), horizontalalignment='left', verticalalignment='center', transform = ax3.transAxes, bbox=dict(facecolor=rh_box_color, alpha=0.65))

start, end = ax.get_ylim()

#ax.yaxis.set_ticks(np.arange(start, end, axis_y_step))
#ax.yaxis.set_ticks (np.arange(start, end, axis_y_step * 2))
#ax2.yaxis.set_ticks(np.arange(start2, end2, axis_y_step * 2))

startx, endx = ax.get_xlim()

#ax.xaxis.set_ticks (np.arange(first_stamp, last_stamp, axis_x_step))
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, datespan)))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H h'))
ax.set_xlim(first_stamp, last_stamp)
ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))

ax6.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, datespan)))
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H h'))
ax6.set_xlim(first_stamp, last_stamp)
ax6.yaxis.set_major_locator(plt.MultipleLocator(1.0))

ax2.axis([axis_x_min, axis_x_max, min_temp, max_temp])
#ax2.yaxis.set_ticks(np.arange(round(tmp119_min,0) - 1, round(tmp119_max,0) + 1, 0.5))
#ax2.axis([axis_x_min, axis_x_max, start2, end2])
ax2.yaxis.set_major_locator(plt.MultipleLocator(1.0))
ax2.set_ylim(min_temp, max_temp)
ax2.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, datespan)))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H h'))
ax2.set_xlim(first_stamp, last_stamp)
#ax2.set_ylim(min_temp, max_temp)

ax3.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, datespan)))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H h'))
ax3.set_xlim(first_stamp, last_stamp)
ax3.tick_params(axis='y', colors='#0000ee')
ax3b.tick_params(axis='y', colors='#009900')

#ax5.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, datespan)))
#ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H h'))
#ax5.set_xlim(first_stamp, last_stamp)

ax4.set_xlim(min_temp, max_temp)
ax4.xaxis.set_major_locator(plt.MultipleLocator(1.0))
ax4.yaxis.set_major_locator(plt.MultipleLocator(1.0))

ax5.set_xlim(min_temp, max_temp)
ax5.xaxis.set_major_locator(plt.MultipleLocator(1.0))
ax5.yaxis.set_major_locator(plt.MultipleLocator(1.0))
#plt.xticks(pd.date_range(start="2023-01-01", end="2023-12-31", freq="M"))
#ax2.xaxis.set_ticks(np.arange(0, count, axis_x_step))
#ax3.xaxis.set_ticks(np.arange(0, count, axis_x_step))
#ax5.xaxis.set_ticks(np.arange(0, count, axis_x_step))

ax.set (xlabel=axis_x_label, ylabel=axis_y_label,   title='%s' % (chart_name))
ax6.set(xlabel=axis_x_label, ylabel=axis6_y_label,   title='%s' % (chart6_name))
ax2.set(xlabel=axis2_x_label,ylabel=axis2_y_label,  title='%s' % (chart2_name))
ax3.set(xlabel=axis_x_label, title='%s' % (chart3_name))
ax3.set_ylabel(ylabel=axis3_y_label, color='#0000ee')
ax3b.set_ylabel(ylabel=axis3_yb_label, color='#009900')
ax4.set(xlabel=axis4_x_label, ylabel=axis4_y_label, title='%s' % (chart4_name))
ax5.set(xlabel=axis4_x_label, ylabel=axis5_y_label, title='%s' % (chart5_name))

#ax2.xaxis.set_ticks(np.arange(0, count, 2))

#start4, end4 = ax4.get_ylim()
#ax4.yaxis.set_ticks(np.arange(start4, end4, axis_y_step))
#ax4.xaxis.set_ticks(np.arange(0, count, 20))

# Plot grids on each of the subplots
ax.grid ()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()

png_fn_large = fn.split(".dsv")[0] + ".png"
png_fn_small = fn.split(".dsv")[0] + "_1.png"

xcsfont = {'fontname':'HandelGothic BT'}
xbsfont = {'fontname':'Cambria'}

ax5.text(1.025, 0.5, '© xDevs.com | https://github.com/tin-/tckit', **xcsfont, rotation=90, fontsize=14, horizontalalignment='left', verticalalignment='center', transform = ax5.transAxes)
ax6.text(1.025, 0.5, "Start: %s        End: %s" % (first_stamp, last_stamp), **xbsfont, rotation=90, fontsize=16, horizontalalignment='left', verticalalignment  ='center', transform = ax6.transAxes)

fig.savefig(png_fn_large, facecolor=fig.get_facecolor(), transparent=False)
print ("\033[44;33m-i- Saved PNG large: %s  \033[49;0m" % png_fn_large)
#ax3.yaxis.set_ticks (np.arange(start3, end3, axis_y_step * 2))
plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.97], w_pad=0.1, h_pad=0.1)                                # Reduce margins around the graph
plt.subplots_adjust(hspace=0.15, wspace=0.12)
figs = fig
figs.set_size_inches(ws,hs)
figs.savefig(png_fn_small, facecolor=figs.get_facecolor(), transparent=False)
print ("\033[44;33m-i- Saved PNG small: %s  \033[49;0m" % png_fn_small)
print ("\033[49;32m-i- Job done  \033[49;39m")

if (show_plot == True):
    plt.show()
