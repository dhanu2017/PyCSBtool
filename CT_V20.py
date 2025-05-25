from tkinter import *
import tkinter.font
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import messagebox 
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import animation
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import serial
import serial.tools.list_ports
import control as co
from scipy.optimize import curve_fit
from PIL import Image,ImageTk
import pandas as pd
import sympy  as sp

#--- exit btn pressed-----
def EXIT():
    global com_select,ser
    plt.close()
    if(com_select == True):
        ser.close()
        com_select = False
    CTGUI.destroy()

#--- com port refresh
def COM_REFRESH():
    global com_select,ser, co_port
    if(com_select == True):
        ser.close()
        com_select = False

    if (len(serial.tools.list_ports.comports())>0):
        com_ports_combo.set(serial.tools.list_ports.comports()[0])
        co_port = com_ports_combo.get()
        com_port = co_port[co_port.find("(")+1:co_port.find(")")]
        #if(len(com_port) !=0):
        ser = serial.Serial(com_port,baud_rate[baud_rate_combo.current()])
        com_select = True
        com_read_btn.config(state=NORMAL)
        com_data_save_btn.config(state=DISABLED)
        print(com_port, com_select)
    else:
        com_ports_combo.set(value = "None")
        com_select = False
        print("Com port select : ", com_select)
        com_read_btn.config(state=DISABLED)
        com_data_save_btn.config(state=DISABLED)

    


#--- com port read and plot
def comPort_Read():
    global com_select,ser,start_time,CTGUI_canvas,CTGUI_plot,plot_type
    
    if(com_select == True):
        ser.close()
        com_select = False
      
    co_port = com_ports_combo.get()
    if(co_port == "None"):
        com_read_btn.config(state=DISABLED)
        com_data_save_btn.config(state=DISABLED)
    if(co_port != "None"):
        com_port = co_port[co_port.find("(")+1:co_port.find(")")]
        if(len(com_port) !=0):
            ser = serial.Serial(com_port,baud_rate[baud_rate_combo.current()])
            com_select = True
            plot_type = 'c'
            com_read_btn.config(state=DISABLED)
            com_data_save_btn.config(state=NORMAL)
            print("Com Port : ",com_port," select :",com_select )
            #print(ser.readline().decode("ASCII"))
            start_time = time.time()
            CTGUI_plot.set(title = com_port +" device - System Responce")
            CTGUI_plot.grid(True)
# just to check Dhanushka sys        
#            for i in range(0, 10,1):
#               d = ser.readline().decode("ASCII")
                
# remove above                
            ani = animation.FuncAnimation(fig=CTGUI_fig, frames=1000,func=animate, interval=0,blit = False)
            CTGUI_canvas.draw()
                  


def animate(i):
    global CTGUI_plot,com_xdata,com_ydata,start_time,line_color,plot_back_color
    d = float(ser.readline().decode("ASCII"))
    com_ydata=np.append(com_ydata,d)
    #print(ydata)
    #xdata=np.append(xdata,time.time()-start_time)
    com_xdata=np.append(com_xdata,i)
    #print(xdata)
    #com_plot.set_offsets(np.c_[xdata,ydata])
    CTGUI_plot.plot(com_xdata, com_ydata,color = line_color,linewidth=0.8)
    CTGUI_plot.set_xlim(min(com_xdata),max(com_xdata)+1)
    CTGUI_plot.set_ylim(min(com_ydata)-1,max(com_ydata)+2)
    

#--- com port data save
def com_data_save():
    global com_xdata,com_ydata
    data = np.transpose([com_xdata,com_ydata])
    np.savetxt('data.csv', data, delimiter=',')
    com_data_save_btn.config(state=DISABLED)
    plt.close()
    

#--- com port select
def com_ports():
    return serial.tools.list_ports.comports()

#--- color them select
def colortheme_select(event=None):
    global line_color,plot_back_color,CTGUI_plot 
    #color_theme = ['wh-bk','bk-gr','wh-bu','bk-yl']
    col = color_theme_combo.get()
    if(col =='wh-bk'):
        line_color = 'k'
        plot_back_color = 'white'
    elif(col =='bk-gr'):
        line_color = 'g'
        plot_back_color = 'black'
    elif(col =='wh-bu'):
        line_color = 'b'
        plot_back_color = 'white'
    elif(col =='bk-yl'):
        line_color = 'y'
        plot_back_color = 'black'
        
    CTGUI_plot.cla()
    CTGUI_plot.set_facecolor(plot_back_color)
    
    if(plot_type == 'p'):
        pzplot()
    elif(plot_type == 's'):
        stepplot()
    elif(plot_type == 'i'):
        impulseplot()
    elif(plot_type == 'c'):
        CTGUI_plot.grid(True)
        CTGUI_plot.set(title = com_port +" device - System Responce")
        CTGUI_plot.set_xlabel('time(s)', fontsize=10)
        CTGUI_plot.set_ylabel('Amplitude', fontsize=10)
        
        
        

#--- about control system basic csb
def about_CSB():
    messagebox.showinfo("showinfo", "Control System Basic tool was developed by the Department of Physics, University of Sri Jayewardenepura, Sri Lanka.")

#TF numerater focus in
def TF_num_entry_temp_text(e):
    TF_num_entry.delete(0,"end")

#TF denominator focus in
def TF_den_entry_temp_text(e):
    TF_den_entry.delete(0,"end")

#SS_stateMatrixA select
def SS_stateMatrix_entry_temp_text(e):
    SS_stateMatrix_entry.delete(0,"end")

#SS_inputMatrixB select
def SS_inputMatrix_entry_temp_text(e):
    SS_inputMatrix_entry.delete(0,"end")

#SS_outputMatrixC select
def SS_outputMatrix_entry_temp_text(e):
    SS_outputMatrix_entry.delete(0,"end")

#SS_feedMatrixD select
def SS_feedthroughMatrix_entry_temp_text(e):
    SS_feedthroughMatrix_entry.delete(0,"end")    


#-- radio button selected
def TFRadio1():
    TF_den_entry.delete(0,"end")
    TF_num_entry.delete(0,"end")
    SS_stateMatrix_entry.delete(0,"end")
    SS_inputMatrix_entry.delete(0,"end")
    SS_outputMatrix_entry.delete(0,"end")
    SS_feedthroughMatrix_entry.delete(0,"end")
    TF_num_entry.insert(0, "3,1")
    TF_den_entry.insert(0, "1,10,20")
    #SS_stateMatrix_entry.insert(0,"-10,-20;1,0")
    #SS_inputMatrix_entry.insert(0,"1;0")
    #SS_outputMatrix_entry.insert(0,"3,1")
    #SS_feedthroughMatrix_entry.insert(0,"0")
    
    
def SSRadio2():
    TF_den_entry.delete(0,"end")
    TF_num_entry.delete(0,"end")
    SS_stateMatrix_entry.delete(0,"end")
    SS_inputMatrix_entry.delete(0,"end")
    SS_outputMatrix_entry.delete(0,"end")
    SS_feedthroughMatrix_entry.delete(0,"end")
    #TF_num_entry.insert(0, "3,1")
    #TF_den_entry.insert(0, "1,10,20")
    SS_stateMatrix_entry.insert(0,"-10,-20;1,0")
    SS_inputMatrix_entry.insert(0,"1;0")
    SS_outputMatrix_entry.insert(0,"3,1")
    SS_feedthroughMatrix_entry.insert(0,"0")
    
#get numerator and denominator
def num_den():
    if(sys_mod_radiobtn_var.get() == 1):
        n = TF_num_entry.get()
        d = TF_den_entry.get()

        #print(nu.split(','))
        open_num = np.zeros(len(n.split(',')))
        open_den = np.zeros(len(d.split(',')))
        i=0
        for j in n.split(','):
            open_num[i] = float(j)
            i=i+1

        i=0
        for j in d.split(','):
            open_den[i] = float(j)
            i=i+1
        closed_num = []
        closed_den = []
        
    else:
        aa = SS_stateMatrix_entry.get()
        bb = SS_inputMatrix_entry.get()
        cc = SS_outputMatrix_entry.get()
        dd = SS_feedthroughMatrix_entry.get()
        aaa = np.array(np.asmatrix(aa))
        bbb = np.array(np.asmatrix(bb))
        ccc = np.array(np.asmatrix(cc))
        ddd = np.array(np.asmatrix(dd))
        TF_G = co.ss2tf(aaa,bbb,ccc,ddd)
        #print(TF_G)
        n,d = co.tfdata(TF_G)
        open_num = n[0][0]
        open_den = d[0][0]
        closed_num = []
        closed_den = []
    print("Open Loop State-Space")
    print(co.tf2ss(open_num,open_den))
    G = co.tf(open_num,open_den)
    kp = cal_kp()
    ki = cal_ki()
    kd = cal_kd()
    if(kp>0 or ki>0 or kd>0):
        s = co.tf('s')
        C = kp+ki/s+kd*s
        negfeed = co.feedback(C*G,1,sign=-1)
        n,d = co.tfdata(negfeed)
        closed_num = n[0][0]
        closed_den = d[0][0]
        print("Closed Loop State-Space")
        print(co.tf2ss(closed_num,closed_den))

    return open_num,open_den,closed_num,closed_den
    

# cal kp val
def cal_kp(event = None):
    #global kp
    kp = kp_sl.get()*float(kp_res_entry.get())
    kp = "{:.{}f}".format(kp,2)
    kpVal_lbl["text"]=str(kp)
    kp = float(kp)
    return kp

# cal ki val    
def cal_ki(event = None):
    #global ki
    ki = ki_sl.get()*float(ki_res_entry.get())
    ki = "{:.{}f}".format(ki,2)
    kiVal_lbl["text"]=str(ki)
    ki = float(ki)
    return ki

# cal kd val
def cal_kd(event = None):
    #global kd
    kd = kd_sl.get()*float(kd_res_entry.get())
    kd = "{:.{}f}".format(kd,2)
    kdVal_lbl["text"]=str(kd)
    kd = float(kd)
    return kd
    
#display transferFunction
def display_TF(n,d,kp,ki,kd):
    G = co.tf(n,d)
    print("Open Loop Transfer Function")
    print(G)
    if(kp>0 or ki>0 or kd>0):
        s = co.tf('s')
        C = kp+ki/s+kd*s
        print("  ")
        print("Kp + Ki/s + Kd*s")
        print(C)

        negfeed = co.feedback(C*G,1,sign=-1)
        print("Closed Loop Transfer Function")
        print(negfeed)
    
#pz plot
def pzplot():
    global plot_type 
    plot_type = 'p'
    PZplot_btn.focus_set()
    o_n,o_d,c_n,c_d = num_den()
    kp = cal_kp()
    ki = cal_ki()
    kd = cal_kd()
    display_TF(o_n,o_d,kp,ki,kd)
    if(kp>0 or ki>0 or kd>0):
        num = c_n
        den = c_d
    else:
        num = o_n
        den = o_d

    #PZ    
    z = np.roots(num)
    p = np.roots(den)
    print("Pole-Zero Map")
    print("zeros ", z)
    print("poles ", p)

    re_p = np.real(p)
    im_p =np.imag(p)
    re_z = np.real(z)
    im_z =np.imag(z)
    CTGUI_plot.cla()
    #CTGUI_plot.scatter([], [])
    CTGUI_plot.scatter(re_p, im_p,marker = 'x',s = 80, color = line_color)
    CTGUI_plot.scatter(re_z, im_z,marker ='o',s =80, facecolors='none', edgecolors=line_color)
    CTGUI_plot.grid(True)
    CTGUI_plot.axhline(0, color='black')
    CTGUI_plot.axvline(0, color='black')
    CTGUI_plot.set_xlabel('Real ($s^{-1}$)', fontsize=10)
    CTGUI_plot.set_ylabel('Imaginary ($s^{-1}$)', fontsize=10)
    CTGUI_plot.set(title = "Pole-Zero Map")
    #CTGUI_plot.tick_params(axis='x', labelsize=8)
    #CTGUI_plot.tick_params(axis='y', labelsize=8)
    vals = np.array([])
    vals=np.append(vals,re_p)
    vals=np.append(vals,re_z)
    CTGUI_plot.set_xlim(min(vals)-10,max(vals)+10)
    vals = np.array([])
    vals=np.append(vals,im_p)
    vals=np.append(vals,im_z)
    CTGUI_plot.set_ylim(min(vals)-20,max(vals)+20)
    CTGUI_canvas.draw()

#--- step responce
def stepplot():
    global plot_type
    plot_type = 's'
    Stepplot_btn.focus_set()
    o_n,o_d,c_n,c_d = num_den()
    kp = cal_kp()
    ki = cal_ki()
    kd = cal_kd()
    display_TF(o_n,o_d,kp,ki,kd)
    if(kp>0 or ki>0 or kd>0):
        num = c_n
        den = c_d
    else:
        num = o_n
        den = o_d
    CTGUI_plot.cla()
    traF = co.tf(num,den)
    t = float(time_entry.get())
    t = np.linspace(0,t,int(t/0.01)) #0s to ts 100 samples per sec
    t,y=co.step_response(traF,t)#calculate the step responce of traF
    CTGUI_plot.plot(t,y,color = line_color,linewidth=0.8)
    CTGUI_plot.grid(True)
    CTGUI_plot.set_xlabel('time(s)', fontsize=10)
    CTGUI_plot.set_ylabel('Amplitude', fontsize=10)
    CTGUI_plot.set(title = "Step Response")
    CTGUI_plot.set_xlim(0,max(t)+max(t)/10)
    CTGUI_plot.set_ylim(min(y),max(y)+max(y)/10)
    CTGUI_canvas.draw()
    data = np.transpose([t,y])
    np.savetxt('step_data.csv', data, delimiter=',')

#impulse response
def impulseplot():
    global plot_type
    plot_type = 'i'
    Impulseplot_btn.focus_set()
    o_n,o_d,c_n,c_d = num_den()
    kp = cal_kp()
    ki = cal_ki()
    kd = cal_kd()
    display_TF(o_n,o_d,kp,ki,kd)
    if(kp>0 or ki>0 or kd>0):
        num = c_n
        den = c_d
    else:
        num = o_n
        den = o_d
    CTGUI_plot.cla()
    traF = co.tf(num,den)
    t = float(time_entry.get())
    t = np.linspace(0,t,int(t/0.01)) #0s to 3s 300 samples
    t,y=co.impulse_response(traF,t)#calculate the impulse responce of TF
    CTGUI_plot.plot(t,y,color = line_color,linewidth=1)
    CTGUI_plot.grid(True)
    CTGUI_plot.set_xlabel('time(s)', fontsize=10)
    CTGUI_plot.set_ylabel('Amplitude', fontsize=10)
    CTGUI_plot.set(title = "Impulse Response")
    CTGUI_plot.set_xlim(0,max(t)+max(t)/10)
    CTGUI_plot.set_ylim(min(y),max(y)+max(y)/10)
    CTGUI_canvas.draw()

# RootLocus plot
def rootlocus_plot():
    global plot_type,CTGUI_plot 
    RLplot_btn.focus_set()
    plot_type = 'r'
    o_n,o_d,c_n,c_d = num_den()
    kp = cal_kp()
    ki = cal_ki()
    kd = cal_kd()
    display_TF(o_n,o_d,kp,ki,kd)
    if(kp>0 or ki>0 or kd>0):
        num = c_n
        den = c_d
    else:
        num = o_n
        den = o_d
    CTGUI_plot.cla()
    
    traF = co.tf(num,den)
    #CTGUI_plot = CTGUI_fig.add_subplot(111)
    #rlist, klist = co.rlocus(traF)
    co.root_locus(traF)
    plt.show()
    #CTGUI_canvas.draw()
    

#--- parameter sweep
def sweep():
    #make pid controller 0
    kp_sl.set(0)
    kp_res_entry.delete(0,'end')
    kp_res_entry.insert(0,'1')
    kpVal_lbl["text"]='0.0'
    ki_sl.set(0)
    ki_res_entry.delete(0,'end')
    ki_res_entry.insert(0,'1')
    kiVal_lbl["text"]='0.0'
    kd_sl.set(0)
    kd_res_entry.delete(0,'end')
    kd_res_entry.insert(0,'1')
    kdVal_lbl["text"]='0.0'

    kp_sw = kp_sw_entry.get()
    kps = np.zeros(len(kp_sw.split(',')))
    n_kps = len(kp_sw.split(',')) 
    i=0
    for j in kp_sw.split(','):
        kps[i] = float(j)
        i=i+1
    #print(kps)
    ki_sw = ki_sw_entry.get()
    kis = np.zeros(len(ki_sw.split(',')))
    n_kis = len(ki_sw.split(','))
    i=0
    for j in ki_sw.split(','):
        kis[i] = float(j)
        i=i+1
    #print(kis)
    kd_sw = kd_sw_entry.get()
    kds = np.zeros(len(kd_sw.split(',')))
    n_kds = len(kd_sw.split(','))
    i=0
    for j in kd_sw.split(','):
        kds[i] = float(j)
        i=i+1
    #print(kds)
    CTGUI_plot.cla()
    o_n,o_d,c_n,c_d = num_den()
    G = co.tf(o_n,o_d)
    s = co.tf('s')
    print("----- parameter sweep -----")
    print("Open Loop Transfer Function")
    print(G)
    t = float(time_entry.get())
    #t = np.linspace(0,3,300)
    
    if(n_kps > n_kis and n_kps > n_kds):
        ki = kis[0]
        kd = kds[0]
        for kp in kps:
            print("Kp = ",kp, " Ki = ", ki," Kd = ",kd)
            print("PID controller")
            C = kp+ki/s+kd*s
            print(C)
            negfeed = co.feedback(C*G,1,sign=-1)
            print("Closed Loop Transfer Function ")
            print(negfeed)
            print("------------------------------")
            t,y=co.step_response(negfeed,t)#calculate the step responce of traF
            CTGUI_plot.plot(t,y,linewidth=0.8,
                            label = "kp = {0:.1f}, ki = {1:.1f},kd = {2:.1f}".format(kp,ki,kd)) 
                
        

    elif(n_kis > n_kps and n_kis > n_kds):
        kp = kps[0]
        kd = kds[0]
        for ki in kis:
            print("Kp = ",kp, " Ki = ", ki," Kd = ",kd)
            print("PID controller")
            C = kp+ki/s+kd*s
            print(C)
            negfeed = co.feedback(C*G,1,sign=-1)
            print("Closed Loop Transfer Function ")
            print(negfeed)
            print("------------------------------")
            t,y=co.step_response(negfeed,t)#calculate the step responce of traF
            CTGUI_plot.plot(t,y,linewidth=0.8,
                            label = "kp = {0:.1f}, ki = {1:.1f},kd = {2:.1f}".format(kp,ki,kd)) 
                
             
    else:
        kp=kps[0]
        ki=kis[0]
        for kd in kds:
            print("Kp = ",kp, " Ki = ", ki," Kd = ",kd)
            print("PID controller")
            C = kp+ki/s+kd*s
            print(C)
            negfeed = co.feedback(C*G,1,sign=-1)
            print("Closed Loop Transfer Function ")
            print(negfeed)
            print("------------------------------")
            t,y=co.step_response(negfeed,t)#calculate the step responce of traF
            CTGUI_plot.plot(t,y,linewidth=0.8,
                            label = "kp = {0:.1f}, ki = {1:.1f},kd = {2:.1f}".format(kp,ki,kd)) 
                
    CTGUI_plot.legend()
    CTGUI_plot.set_xlim(0,max(t)+max(t)/10)
    CTGUI_plot.set_ylim(min(y),max(y)+max(y)/10)
    CTGUI_plot.grid(True)
    CTGUI_plot.set_xlabel('time(s)', fontsize=10)
    CTGUI_plot.set_ylabel('Amplitude', fontsize=10)
    CTGUI_plot.set(title = "Step Response")
    CTGUI_canvas.draw()

#---load data
def load_file():
    global time_data, response_data
    f_path = filedialog.askopenfilename(initialdir="D:",
      title="Select a File", filetypes=(("CSV file","*.csv*"),("Text files","*.txt*")))
    #print(f_path)  
    
    # using loadtxt()
    data = np.loadtxt(f_path,delimiter=",")
    #print(np.shape(data))
    t_data = data[:,0]
    y_data = data[:,1]
    time_data = t_data
    response_data = y_data
    CTGUI_plot.cla()
    CTGUI_plot.plot(t_data,y_data,linewidth=0.8, color = line_color) 
    CTGUI_plot.set_xlim(0,max(t_data)+max(t_data)/10)
    CTGUI_plot.set_ylim(min(y_data),max(y_data)+max(y_data)/10)
    CTGUI_plot.grid(True)
    CTGUI_plot.set_xlabel('time(s)', fontsize=10)
    CTGUI_plot.set_ylabel('Amplitude', fontsize=10)
    CTGUI_plot.set(title = "System Step Response")
    CTGUI_canvas.draw()
    

#---System Identification
def sys_id():
    # Fit the model to the data
    initial_guess = [1.0, 1.0, 0.5]  # Initial guesses for K, wn, zeta
    try:
        params, covariance = curve_fit(second_order_model, time_data, response_data, p0=initial_guess)
        K_fit, wn_fit, zeta_fit = params

        # Generate the fitted response
        fitted_response = second_order_model(time_data, K_fit, wn_fit, zeta_fit)
                
        # Plot the original and fitted response
        CTGUI_plot.cla()
        CTGUI_plot.plot(time_data, response_data, label="Original Data")
        CTGUI_plot.plot(time_data, fitted_response, '--', label=f"Fitted Model\nK={K_fit:.2f}, wn={wn_fit:.2f}, ζ={zeta_fit:.2f}")
        CTGUI_plot.set_xlim(0,max(time_data)+max(time_data)/10)
        CTGUI_plot.set_ylim(min(response_data),max(response_data)+max(response_data)/10)
        CTGUI_plot.grid(True)
        CTGUI_plot.set_xlabel('time(s)', fontsize=10)
        CTGUI_plot.set_ylabel('Amplitude', fontsize=10)
        CTGUI_plot.set(title = "Step Response with Fitted Second-Order Model")
        CTGUI_plot.legend()
        CTGUI_canvas.draw()

        hkp = float(Hkp_entry.get())
        hki = float(Hki_entry.get())
        hkd = float(Hkd_entry.get())
        s = co.tf('s')
        H = hkp+hki/s+hkd*s
        TT = co.tf([K_fit * wn_fit**2], [1, 2 * zeta_fit * wn_fit, wn_fit**2])
        print("------------------------")
        print("sys identification: TF" , TT)
        print("Controller TF: ", H)
        GGn = (TT)/(1-TT)
#        sim = co.minreal(GGn)
        GG = GGn / H
        GGsim = co.minreal(GG)
        print("Plant TF: ", GGsim)
##        print("Plant TF: ", GGn)
##        print("Plant TF: ", GG)
##        print("Plant sim: ", sim)
##        print("Plant sim: ", GGsim)

    except Exception as e:
        print(f"Error fitting the model: {e}")       


# Define a second-order transfer function model for fitting
def second_order_model(t, K, wn, zeta):
    """
    K: Gain
    wn: Natural frequency
    zeta: Damping ratio
    """
    # Calculate the step response of the system
    trans_f = co.tf([K * wn**2], [1, 2 * zeta * wn, wn**2])
    t_out, y_out = co.step_response(trans_f, t)
    return np.interp(t, t_out, y_out)

def transient():
    # Load the uploaded file to inspect its structure
    file_path = 'step_data.csv'
    data = pd.read_csv(file_path)
    # Rename the columns for better clarity
    data.columns = ["Time", "Response"]
    # Convert data to numeric types (if necessary) and preview the cleaned data
    data = data.apply(pd.to_numeric, errors='coerce')
    # Extract steady-state value
    steady_state_value = data["Response"].iloc[-1]
    # Calculate 10% and 90% of the steady-state value for rise time
    ten_percent = 0.1 * steady_state_value
    ninety_percent = 0.9 * steady_state_value
    # Find rise time
    rise_time_indices = data[(data["Response"] >= ten_percent) & (data["Response"] <= ninety_percent)].index
    rise_time = data["Time"].iloc[rise_time_indices[-1]] - data["Time"].iloc[rise_time_indices[0]]

    # Find peak time and percent overshoot
    peak_value = data["Response"].max()
    peak_time = data["Time"].iloc[data["Response"].idxmax()]
    percent_overshoot = ((peak_value - steady_state_value) / steady_state_value) * 100

    # Find settling time (±2% band of steady-state value)
    settling_band = 0.02 * steady_state_value
    settling_time_indices = data[
    (data["Response"] >= steady_state_value - settling_band) &
    (data["Response"] <= steady_state_value + settling_band)
    ].index

    settling_time = data["Time"].iloc[settling_time_indices[0]] if not settling_time_indices.empty else np.nan
    
    
    print("---Trancient Response---")
    print("Steady State Value : ","%.2f" % steady_state_value)
    print("Rise Time(10% to 90% of steady-state) : ","%.2f" %  rise_time, "sec")
    print("Peak Time : ", "%.2f" % peak_time, "sec")
    print("Percent Overshoot : ", "%.2f" % percent_overshoot)
    print("Settling Time (within ±2% of steady-state): ", "%.2f" %settling_time, "sec")

    steady_state_xdata = [data["Time"].iloc[0],data["Time"].iloc[-1]]
    steady_state_ydata = [steady_state_value,steady_state_value]
    ten_percent_xdata = [data["Time"].iloc[0],data["Time"].iloc[rise_time_indices[0]],data["Time"].iloc[rise_time_indices[0]]]
    ten_percent_ydata = [ten_percent,ten_percent,0]
    ninety_percent_xdata = [data["Time"].iloc[0],data["Time"].iloc[rise_time_indices[-1]],data["Time"].iloc[rise_time_indices[-1]]]
    ninety_percent_ydata = [ninety_percent,ninety_percent,0]

    
    CTGUI_plot.cla()
    CTGUI_plot.plot(data["Time"],data["Response"])
    CTGUI_plot.plot(steady_state_xdata,steady_state_ydata,color='k',linestyle='dashed',linewidth=0.8)
    CTGUI_plot.plot(ten_percent_xdata,ten_percent_ydata ,color='k',linestyle='dashed',linewidth=0.8)
    CTGUI_plot.plot(ninety_percent_xdata,ninety_percent_ydata ,color='k',linestyle='dashed',linewidth=0.8)
    CTGUI_plot.set_xlabel('time(s)', fontsize=10)
    CTGUI_plot.set_ylabel('Amplitude', fontsize=10)
    CTGUI_plot.set(title = "transient and steady-state responses")
    CTGUI_plot.grid(True)
    CTGUI_canvas.draw()
    
    

    
#-----------main program-------------------
#---creating tkinter window 
CTGUI = Tk()
#---read screen size
screen_width = CTGUI.winfo_screenwidth()
screen_height = CTGUI.winfo_screenheight()
#print(screen_width,screen_height)
#---setting window geometry
f_ratio = 0.85
window_width = int(screen_width*f_ratio)  #1161
window_height= int(screen_height*f_ratio) #652
#print(window_width,window_height)
win_geom = str(window_width)+'x'+str(window_height)+'+50+20'
#print(win_geom)

CTGUI.title("Control System Basic")
CTGUI.geometry(win_geom)
CTGUI.resizable(0, 0)#Don't allow resizing in the x or y direction



#----- Font types
font_btn = tkinter.font.Font(family='Helvetica', size =13, weight= "bold")
font_main_lbl = tkinter.font.Font(family='Helvetica', size =13, weight= "bold")
font_Large_lbl = tkinter.font.Font(family='Helvetica', size =13)
font_sub_lbl = tkinter.font.Font(family='Helvetica', size =11)
font_sub2_lbl = tkinter.font.Font(family='Helvetica', size =8)
font_Large2_lbl = tkinter.font.Font(family='Helvetica', size =14,weight= "bold")

#---------the figure that will contain the plot 
CTGUI_fig = Figure(figsize = (7,4) ,dpi = 100,facecolor='#cacad5')#700x400
CTGUI_plot = CTGUI_fig.add_subplot(111)
CTGUI_plot.set_xlim(0,10)
CTGUI_plot.set_ylim(0,10)
CTGUI_plot.tick_params(axis='x', labelsize=8)
CTGUI_plot.tick_params(axis='y', labelsize=8)
CTGUI_plot.set_xlabel('sample', fontsize=10)
CTGUI_plot.set_ylabel('Amp', fontsize=10)
CTGUI_plot.set(title = "System Response")
CTGUI_plot.set_facecolor("white")
#--- plot the data
com_xdata = []
com_ydata = []
com_xdata = np.array(com_xdata)
com_ydata = np.array(com_ydata)
CTGUI_plot.plot(com_xdata, com_ydata)
CTGUI_plot.grid()
CTGUI_canvas = FigureCanvasTkAgg(CTGUI_fig,master = CTGUI)   
CTGUI_canvas.draw()
CTGUI_fig_size = CTGUI_fig.get_size_inches()*CTGUI_fig.dpi
#print(CTGUI_fig_size) #CTGUI_fig window size 700x400
CTGUI_canvas.get_tk_widget().place(x = window_width-CTGUI_fig_size[0]-int(window_width/58), y= int(window_height/32))
#---creating the Matplotlib toolbar 
CTGUI_fig_toolbar = NavigationToolbar2Tk(CTGUI_canvas,CTGUI)
CTGUI_fig_toolbar.update() 
CTGUI_fig_toolbar.place(x = window_width-CTGUI_fig_size[0]-int(window_width/58),y= CTGUI_fig_size[1]+int(window_height/32))
#for curve fitting
time_data = np.array([])
response_data = np.array([])






#-----------buttons label entry combo ---------

#----Transfer function entry
TF_G_lbl = Label(CTGUI,text="G = ", font = font_main_lbl,fg='black') 
TF_G_lbl.place(x= 30, y = 90 )
TF_num_entry = Entry(CTGUI,fg="blue", bg="white", width=15)
TF_num_entry.place(x = 70, y = 80)
TF_den_entry = Entry(CTGUI,fg="blue", bg="white", width=15)
TF_den_entry.place(x = 70, y = 105)
TF_num_entry.insert(0, "3,1")
TF_num_entry.bind("<FocusIn>", TF_num_entry_temp_text)
TF_den_entry.insert(0, "1,10,20")
TF_den_entry.bind("<FocusIn>", TF_den_entry_temp_text)
#----state space entry
SS_stateMatrix_lbl = Label(CTGUI,text="State matrix A", font = font_sub_lbl,fg='black')
SS_stateMatrix_lbl.place(x = 30, y= 175 )
SS_stateMatrix_entry = Entry(CTGUI,fg="blue", bg="white", width=20) 
SS_stateMatrix_entry.place(x = 140,y = 175)
SS_inputMatrix_lbl = Label(CTGUI,text="Input matrix B", font = font_sub_lbl,fg='black')
SS_inputMatrix_lbl.place(x = 30, y= 210 )
SS_inputMatrix_entry = Entry(CTGUI,fg="blue", bg="white", width=20) 
SS_inputMatrix_entry.place(x = 140,y = 210)
SS_outputMatrix_lbl = Label(CTGUI,text="Output matrix C", font = font_sub_lbl,fg='black')
SS_outputMatrix_lbl.place(x = 30, y= 245 )
SS_outputMatrix_entry = Entry(CTGUI,fg="blue", bg="white", width=20) 
SS_outputMatrix_entry.place(x = 140,y = 245)
SS_feedthroughMatrix_lbl = Label(CTGUI,text="Feedthrough matrix D", font = font_sub_lbl,fg='black')
SS_feedthroughMatrix_lbl.place(x = 30, y= 275 )
SS_feedthroughMatrix_entry = Entry(CTGUI,fg="blue", bg="white", width=10) 
SS_feedthroughMatrix_entry.place(x = 200,y = 275)

SS_stateMatrix_entry.bind("<FocusIn>", SS_stateMatrix_entry_temp_text)
SS_inputMatrix_entry.bind("<FocusIn>", SS_inputMatrix_entry_temp_text)
SS_outputMatrix_entry.bind("<FocusIn>",SS_outputMatrix_entry_temp_text)
SS_feedthroughMatrix_entry.bind("<FocusIn>", SS_feedthroughMatrix_entry_temp_text)

SS_statedim_lbl = Label(CTGUI,text="nxn", font = font_sub2_lbl,fg='black')
SS_statedim_lbl.place(x = 265, y= 175 )
SS_inputdim_lbl = Label(CTGUI,text="nx1", font = font_sub2_lbl,fg='black')
SS_inputdim_lbl.place(x = 265, y= 210 )
SS_outputdim_lbl = Label(CTGUI,text="1xn", font = font_sub2_lbl,fg='black')
SS_outputdim_lbl.place(x = 265, y= 245 )
SS_feeddim_lbl = Label(CTGUI,text="1x1", font = font_sub2_lbl,fg='black')
SS_feeddim_lbl.place(x = 265, y= 275 )

#----system modeling
sys_mod_lbl = Label(CTGUI,text="System Modeling", font = font_main_lbl,fg='black')
sys_mod_lbl.place(x = 10,y = 10)
sys_mod_radiobtn_var = IntVar()
sys_mod_radiobtn1 = Radiobutton(CTGUI, text="Frequency Domain Transfer Function",
                                font = font_sub_lbl,variable=sys_mod_radiobtn_var, value=1,command = TFRadio1)
sys_mod_radiobtn1.place(x=15,y=40)
sys_mod_radiobtn1.select()
sys_mod_radiobtn2 = Radiobutton(CTGUI, text="Time Domain State-Space Representation",
                                font = font_sub_lbl,variable=sys_mod_radiobtn_var, value=2,command = SSRadio2)
sys_mod_radiobtn2.place(x = 15, y = 140)
sys_mod_radiobtn2.deselect()

#--- PID controller
pid_lbl = Label(CTGUI,text="PID Controller", font = font_main_lbl,fg='black') 
pid_lbl.place(x= 10, y = 325 )
kp_lbl = Label(CTGUI,text="Kp", font = font_Large_lbl,fg='black') 
kp_lbl.place(x= 30, y = 360 )
ki_lbl = Label(CTGUI,text="Ki", font = font_Large_lbl,fg='black') 
ki_lbl.place(x= 30, y = 400 )
kd_lbl = Label(CTGUI,text="Kd", font = font_Large_lbl,fg='black') 
kd_lbl.place(x= 30, y = 440 )
kp_sl = Scale(CTGUI, from_=0, to=10,orient=HORIZONTAL, length = 150,command = cal_kp)
kp_sl.set(0)
kp_sl.place(x = 70, y = 350)
ki_sl = Scale(CTGUI, from_=0, to=10,orient=HORIZONTAL, length = 150,command = cal_ki)
ki_sl.set(0)
ki_sl.place(x = 70, y = 390)
kd_sl = Scale(CTGUI, from_=0, to=10,orient=HORIZONTAL, length = 150,command = cal_kd)
kd_sl.set(0)
kd_sl.place(x = 70, y = 430)

kp_res_entry = Entry(CTGUI,fg="blue", bg="white", width=5,textvariable = cal_kp) 
kp_res_entry.place(x = 240,y = 370)
kp_res_entry.insert(0,'1')
ki_res_entry = Entry(CTGUI,fg="blue", bg="white", width=5,textvariable = cal_ki) 
ki_res_entry.place(x = 240,y = 410)
ki_res_entry.insert(0,'1')
kd_res_entry = Entry(CTGUI,fg="blue", bg="white", width=5,textvariable = cal_kd) 
kd_res_entry.place(x = 240,y = 450)
kd_res_entry.insert(0,'1')

x1_lbl = Label(CTGUI,text="x", font = font_sub_lbl,fg='black') 
x1_lbl.place(x= 222, y = 370 )
x2_lbl = Label(CTGUI,text="x", font = font_sub_lbl,fg='black') 
x2_lbl.place(x= 222, y = 410 )
x3_lbl = Label(CTGUI,text="x", font = font_sub_lbl,fg='black') 
x3_lbl.place(x= 222, y = 450 )
eq1_lbl = Label(CTGUI,text="=", font = font_sub_lbl,fg='black') 
eq1_lbl.place(x= 275, y = 370 )
eq2_lbl = Label(CTGUI,text="=", font = font_sub_lbl,fg='black') 
eq2_lbl.place(x= 275, y = 410 )
eq3_lbl = Label(CTGUI,text="=", font = font_sub_lbl,fg='black') 
eq3_lbl.place(x= 275, y = 450 )

kp = kp_sl.get()*float(kp_res_entry.get())
ki = ki_sl.get()*float(ki_res_entry.get())
kd = kd_sl.get()*float(kd_res_entry.get())
kpVal_lbl = Label(CTGUI,text=str(kp), font = font_Large_lbl,fg='black') 
kpVal_lbl.place(x= 290, y = 370 )
kiVal_lbl = Label(CTGUI,text=str(kp), font = font_Large_lbl,fg='black') 
kiVal_lbl.place(x= 290, y = 410 )
kdVal_lbl = Label(CTGUI,text=str(kp), font = font_Large_lbl,fg='black') 
kdVal_lbl.place(x= 290, y = 450 )

para_sweep_lbl = Label(CTGUI,text="Parameter Sweep", font = font_main_lbl,fg='black') 
para_sweep_lbl.place(x= 10, y = 480 )

kp_sw_lbl = Label(CTGUI,text="Kp", font = font_Large_lbl,fg='black') 
kp_sw_lbl.place(x= 30, y = 510 )
ki_sw_lbl = Label(CTGUI,text="Ki", font = font_Large_lbl,fg='black') 
ki_sw_lbl.place(x= 30, y = 550 )
kd_sw_lbl = Label(CTGUI,text="Kd", font = font_Large_lbl,fg='black') 
kd_sw_lbl.place(x= 30, y = 590 )

kp_sw_entry = Entry(CTGUI,fg="blue", bg="white", width=10) 
kp_sw_entry.place(x = 80,y = 512)
kp_sw_entry.insert(0,'1,10,100')
ki_sw_entry = Entry(CTGUI,fg="blue", bg="white", width=10)
ki_sw_entry.place(x = 80,y = 552)
ki_sw_entry.insert(0,'0')
kd_sw_entry = Entry(CTGUI,fg="blue", bg="white", width=10) 
kd_sw_entry.place(x = 80,y = 592)
kd_sw_entry.insert(0,'0')
#time axis
time_lbl = Label(CTGUI,text="Time Axis", font = font_Large_lbl,fg='black') 
time_lbl .place(x = 350, y=400)
time_entry = Entry(CTGUI,fg="blue", bg="white", width=5) 
time_entry.place(x = 360,y = 425)
time_entry.insert(0,'3')
time_unit_lbl = Label(CTGUI,text="sec.", font = font_sub_lbl,fg='black') 
time_unit_lbl.place(x = 390, y=425)
t = 3
#--- plot btns
PZplot_btn =Button(CTGUI, text="Pole\nZero",fg="black", height = 2, width = 7,font = font_btn,
                   state= ACTIVE,command=pzplot)
PZplot_btn.place(x = 350, y=50)
RLplot_btn =Button(CTGUI, text="Root\nLocus",fg="black", height = 2, width = 7,font = font_btn,
                   state= ACTIVE,command=rootlocus_plot)
RLplot_btn.place(x = 350, y=120)
Stepplot_btn =Button(CTGUI, text="Step",fg="black", height = 2, width = 7,font = font_btn,
                     state= ACTIVE,command=stepplot)
Stepplot_btn.place(x = 350, y=190)
Impulseplot_btn =Button(CTGUI, text="Impulse",fg="black", height = 2, width = 7,font = font_btn,
                        state= ACTIVE,command=impulseplot)
Impulseplot_btn.place(x = 350, y=260)

transient_btn =Button(CTGUI, text="Transient",fg="black", height = 2, width = 7,font = font_btn,
                        state= ACTIVE,command=transient)
transient_btn.place(x = 350, y=330)

sweep_btn =Button(CTGUI, text = "Sweep",fg="black", height = 2, width = 7,font = font_btn,state= ACTIVE,command=sweep)
sweep_btn.place(x = 170, y=560)


#----close btn------
exit_btn =Button(CTGUI, text="Exit",fg="black",  pady = 5,height = 1, width = 8,font = font_btn,state= ACTIVE,command=EXIT)
exit_btn.place(x = 1050,y=595)
#-- about Control System Basic btn
about_CSB_btn =Button(CTGUI, text="About",fg="black", pady = 5,height = 1, width = 8,font = font_btn,state= ACTIVE,command=about_CSB) 
about_CSB_btn.place(x = 1050,y=500)
#---- Serial communication------
com_select = False
plot_type = 'n'
com_slect_lbl = Label(CTGUI,text="Connect COM Device", font = font_main_lbl,fg='black')
com_slect_lbl.place(x = 558,y= 472 )
com_ports_combo_lbl = Label(CTGUI,text="ComPort ", font = font_sub_lbl,fg='black')
com_ports_combo_lbl.place(x = window_width-CTGUI_fig_size[0],y= CTGUI_fig_size[1]+int(window_height/6))
com_ports_combo = ttk.Combobox(CTGUI, values=com_ports())
com_ports_combo.place(x = window_width-CTGUI_fig_size[0]+100,y= CTGUI_fig_size[1]+int(window_height/6))
baud_rate_combo_lbl = Label(CTGUI,text="Baud Rate ", font = font_sub_lbl,fg='black')
baud_rate_combo_lbl.place(x = window_width-CTGUI_fig_size[0],y= CTGUI_fig_size[1]+int(window_height/4.4))
baud_rate = [9600, 19200, 38400, 57600, 115200]
baud_rate_combo = ttk.Combobox(CTGUI, state="readonly",values=baud_rate)
baud_rate_combo.set(baud_rate[0])
baud_rate_combo.place(x = window_width-CTGUI_fig_size[0]+100,y= CTGUI_fig_size[1]+int(window_height/4.5))
#---- System Identification ----
sysID_lbl = Label(CTGUI,text="System Identification", font = font_main_lbl,fg='black')
sysID_lbl.place(x = 820,y= 472 )
sysID_co_lbl =Label(CTGUI,text="Con. H(s)", font = font_sub_lbl,fg='black')
sysID_co_lbl.place(x = 770,y= 520 )
Hkp_entry = Entry(CTGUI,fg="blue", bg="white", width=5) 
Hkp_entry.place(x = 850,y = 520)
Hkp_entry.insert(0,'0')
Hp_lbl = Label(CTGUI,text="+", font = font_sub_lbl,fg='black')
Hp_lbl.place(x=880, y = 520)

Hki_entry = Entry(CTGUI,fg="blue", bg="white", width=5) 
Hki_entry.place(x = 900,y = 520)
Hki_entry.insert(0,'0')
Hpp_lbl = Label(CTGUI,text="+", font = font_sub_lbl,fg='black')
Hpp_lbl.place(x=930, y = 520)
Hps_lbl = Label(CTGUI,text="s", font = font_sub_lbl,fg='black')
Hps_lbl.place(x=910, y = 538)

Hkd_entry = Entry(CTGUI,fg="blue", bg="white", width=5) 
Hkd_entry.place(x = 950,y = 520)
Hkd_entry.insert(0,'0')
Hds_lbl = Label(CTGUI,text="s", font = font_sub_lbl,fg='black')
Hds_lbl.place(x=990, y = 515)

Hkpid_lbl= Label(CTGUI,text="kp        ki         kd", font = font_sub_lbl,fg='black')
Hkpid_lbl.place(x = 855,y = 495)


#-- refresh button
refresh_btn =Button(CTGUI, text="Refresh",fg="black", pady = 5,height = 1, width = 8,font = font_btn,state= ACTIVE,command=COM_REFRESH)
#refresh_btn.place(x = window_width-CTGUI_fig_size[0],y=window_height-int(window_height*0.09))
refresh_btn.place(x = 460,y=595)
#--Serial Port Read and plot
com_read_btn =Button(CTGUI, text="Read",fg="black", pady = 5,height = 1, width = 8,font = font_btn,state= ACTIVE,command=comPort_Read)
com_read_btn.place(x = 570,y=595)
#--Com Port data save in a csv
com_data_save_btn =Button(CTGUI, text="Save",fg="black", pady = 5,height = 1, width = 8,font = font_btn,state= ACTIVE,command=com_data_save) 
com_data_save_btn.place(x = 680,y=595)
#--- load data
load_data_btn =Button(CTGUI, text="Load",fg="black", pady = 5,height = 1, width = 8,font = font_btn,state= ACTIVE,command=load_file) 
load_data_btn.place(x = 790,y=595)
#--- systen identification btn
sys_ide_btn =Button(CTGUI, text="SysId",fg="black", pady = 5,height = 1, width = 8,
                    font = font_btn,state= ACTIVE,command=sys_id) 
sys_ide_btn.place(x = 900,y=595)




#plot color theme----
plot_color_lbl = Label(CTGUI,text="ColorTheme", font = font_sub_lbl,fg='black')
plot_color_lbl.place(x = window_width-int(window_width*0.22),y= CTGUI_fig_size[1]+int(window_height/25))
line_color = 'k'
plot_back_color = 'white'
color_theme = ['wh-bk','bk-gr','wh-bu','bk-yl']
color_theme_combo = ttk.Combobox(CTGUI, state="readonly",values=color_theme,width = 10)
color_theme_combo.set(color_theme[0])
color_theme_combo.bind('<<ComboboxSelected>>', colortheme_select)
color_theme_combo.place(x = window_width-int(window_width*0.12),y= CTGUI_fig_size[1]+int(window_height/25))

#--- transfer function  state-space ----





#--- initializing-------
#serial com port
if (len(serial.tools.list_ports.comports())>0):
    com_ports_combo.set(serial.tools.list_ports.comports()[0])
    #com_select = True
else:
    com_ports_combo.set(value = "None")
    #com_select = False
   

co_port = com_ports_combo.get()
if(co_port == "None"):
    com_read_btn.config(state=DISABLED)
    com_data_save_btn.config(state=DISABLED)
    com_select = False
   
if(co_port != "None"):
    com_port = co_port[co_port.find("(")+1:co_port.find(")")]
    if(len(com_port) !=0):
        ser = serial.Serial(com_port,baud_rate[baud_rate_combo.current()])
        com_select = True
        com_read_btn.config(state=NORMAL)
        com_data_save_btn.config(state=DISABLED)
print(" Com port select state : ",com_select)


CSB_lbl = Label(CTGUI,text="Control System Basic tool", font = font_sub_lbl,fg='black')
CSB_lbl.place(x=260, y = 585)
pyCSB_lbl = Label(CTGUI,text="PyCSBtool", font = font_Large2_lbl,fg='black')
pyCSB_lbl.place(x=300, y = 560)

"""
#Create a canvas
canvas= Canvas(CTGUI, width= 150, height= 200)
#canvas.pack()
canvas.place(x= 250, y = 500)
#Load an image in the script
img = Image.open("CSBtool.png")
#img=img.resize((150,100), Image.ANTIALIAS)
img=img.resize((150,10), Image.Resampling.LANCZOS)
pic= ImageTk.PhotoImage(img)
#Add image to the Canvas Items
canvas.create_image(150,100,image=pic)
"""
CTGUI.mainloop()
