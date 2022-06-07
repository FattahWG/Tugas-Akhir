#====================================================================================
# NAMA              : FATTAH WIDJAYA GANDHI
# NIM               : 535170089
# FAKULTAS/PRODI    : FTI/TI 2017
# Pengenalan jenis masker menggunakan metode color histogram dan Euclidean Distance
#====================================================================================
import cv2
import shutil
import glob
import os
import sys
import tkinter as tk
import numpy as np
import glob
import matplotlib
import csv
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import *
from tkinter import ttk
from pathlib import Path as path
from PIL import ImageTk
from PIL import Image
from tkinter import filedialog
from tkinter.filedialog import askopenfilenames
from numpy.lib import  median
from scipy.spatial import distance as dist



#-------------------------
# untuk mengatur perpindahan frame
class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Color Histogram RGB - Face Mask")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (Utama, Pengenalan, Hasil, Help):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(Utama)
    
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class Utama(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self,bg='#849974', padx= 150, pady=150)

        button1 = tk.Button(self,pady=10,padx=10,text="Pengenalan", borderwidth=0.5, bg='#FEB47B',height=3,width=30, command=lambda: controller.show_frame(Pengenalan))
        button1.pack(padx=10, pady=10)
        button2 = tk.Button(self,pady=10,padx=10, text="Hasil", borderwidth=0.5, bg='#FEB47B',height=3,width=30, command=lambda: controller.show_frame(Hasil))
        button2.pack(padx=10, pady=10)
        button3 = tk.Button(self,pady=10,padx=10, text="Help", borderwidth=0.5, bg='#FEB47B',height=3,width=30, command=lambda: controller.show_frame(Help))
        button3.pack(padx=10, pady=10)

#----------------------------------
class Pengenalan(tk.Frame):
    global test,datahist, minHist, min2hist  

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self,bg='#849974')

        global  ujihist, x2, hist2, loct, test, datahist, minHist, min2hist  
        dlhisttrain = []
        ujihist = {}
        datahist = {}
        minHist = []
        min2hist = []
        #--------------------------------------------------
        def insertfilesTraining():
            global olah, x2, hist2, loct
            listTraining.delete(0, tk.END)
            listTraining.insert(tk.END, "===================")        
            listTraining.insert(tk.END, "Data Training")  
            listTraining.insert(tk.END, "===================")  
            for filename in glob.glob('dataset1/*.png'):
                listTraining.insert(tk.END, filename)
            for file in glob.glob('dataset1/datacrop/*.png'):
                foto=cv2.imread(file)
                foto = cv2.convertScaleAbs(foto, alpha=1, beta=3)
                channels = cv2.split(foto) 
                colors = ("b", "g", "r")  
                for (i, col) in zip(channels, colors): 
                    hist = cv2.calcHist([i], [0], None, [256], [0, 256])
                    histt = cv2.normalize(hist, hist).flatten()
                dlhisttrain.append(histt)
                datahist[file]=histt
            np.savetxt('datahistogramtrainCrop.text', dlhisttrain, delimiter='|')
            
            for file in glob.glob('dataset1/*.png'):
                foto=cv2.imread(file)
                channels = cv2.split(foto) 
                colors = ("b", "g", "r") 
                for (i, col) in zip(channels, colors): 
                    hist = cv2.calcHist([i], [0], None, [256], [0, 256])
                    hist3 = cv2.normalize(hist, hist).flatten()
                minHist.append(hist3)
                ujihist[file]=hist3 
                np.savetxt('datahistogramtrain.csv', minHist, delimiter='|')

        def insertfilesUji():
            listUji.delete(0, tk.END)
            listUji.insert(tk.END, "===================")        
            listUji.insert(tk.END, "Data Uji")  
            listUji.insert(tk.END, "===================") 
            sumber = "C:\\final\\dataset2"
            fotomedis = "C:\\final\\Pilihan"
            lista = filedialog.askopenfilenames(parent=self, initialdir= "C:/final/dataset2", title='Please select a directory')
 
            for filename in glob.glob('Pilihan/*.png'):
                os.remove(filename)
 
            for filename in lista:
                shutil.copy(os.path.join(sumber, filename), fotomedis)

            for file in glob.glob('Pilihan/*.png'):
                listUji.insert(tk.END, file)
        #--------------------------------------------------
        def showimgTrain(event):
            n = listTraining.curselection()
            filename = listTraining.get(n)
            img = tk.PhotoImage(file=filename)
            w, h = img.width(), img.height()
            print(filename)
            canvas.image = img
            canvas.create_image(0, 0, image=img, anchor=tk.NW)
        #--------------------------------------------------
        def showimgUji(event):
            n = listUji.curselection()
            filename = listUji.get(n)
            img = tk.PhotoImage(file=filename)
            w, h = img.width(), img.height()
            print(filename)
            canvas.image = img
            canvas.create_image(0, 0, image=img, anchor=tk.NW)
        #-----------------------------------------------------
        def capture():
            cam = cv2.VideoCapture(0)
            img_counter = 0
            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                cv2.imshow("tekan space bar untuk menangkap foto", frame)
                k = cv2.waitKey(1)
                if k%106 == 27:
                    cv2.destroyAllWindows()
                    break
                elif k%106 == 32:
                    path = "C:\\final\\Pilihan"
                    img_name = "capture{}.png".format(img_counter)
                    cv2.imwrite(os.path.join(path, img_name), frame)
                    print("{} berhasil disimpan!".format(img_name))
                    img_counter += 1
            cam.release()
            
            for file in glob.glob('Pilihan/*.png'):
                listUji.insert(tk.END, file)
        #--------------------------------------------------------
        canvas = tk.Canvas(self,background='#F4E7D7')    
        canvas.config(width=600, height=600)
        canvas.pack(side="right",padx=10, pady=10)
       #---------------------------------------------------------    
        listUji = tk.Listbox(self, width=20, background='#F4E7D7',activestyle='none')
        listUji.insert(tk.END, "===================")        
        listUji.insert(tk.END, "Data Uji")  
        listUji.insert(tk.END, "===================") 
        listUji.pack(side="right", fill=tk.BOTH,  expand=0, padx=10, pady=10)
        listUji.bind("<<ListboxSelect>>", showimgUji)
        
        listTraining = tk.Listbox(self, width=20,background='#F4E7D7',activestyle='none')
        listTraining.pack(side="right", fill=tk.BOTH,  expand=0, padx=10, pady=10)
        listTraining.insert(tk.END, "===================")        
        listTraining.insert(tk.END, "Data Training")  
        listTraining.insert(tk.END, "===================")      
        listTraining.bind("<<ListboxSelect>>", showimgTrain)
        #---------------------------------------------------------
        button1 = tk.Button(self, text="Home", borderwidth=0.5, bg='#FEB47B',height=3,width=30, command=lambda: controller.show_frame(Utama))
        button1.pack(side="top",pady=10)
        buttonTraining = tk.Button(self, text="Data Training", borderwidth=0.5, bg='#FEB47B',height=3,width=30, command=insertfilesTraining)
        buttonTraining.pack(side="top",pady=10)
        button3 = tk.Button(self, text="Capture", borderwidth=0.5, bg='#FEB47B',height=3,width=30, command=capture)
        button3.pack(side="top",pady=10)
        buttonUji = tk.Button(self, text="Data Uji", borderwidth=0.5, bg='#FEB47B',height=3,width=30, command=insertfilesUji)
        buttonUji.pack(side="top",pady=10)
        buttonHasil = tk.Button(self, text="Hasil", borderwidth=0.5, bg='#FEB47B',height=3,width=30, command=lambda: controller.show_frame(Hasil))
        buttonHasil.pack(side="top", pady=10)

#-----------------------------------
class Hasil(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self,bg='#849974')
        #-----------------------------------------------
        def banding():
            pilihanhist= {}
            edis=[]   
            dhistuji=[]
            global hasil2
            Jnon=0
            Jmed=0 
            for filename in glob.glob('Hasil/medis/*.png'):
                os.remove(filename)
            for filename in glob.glob('Hasil/nonmedis/*.png'):
                os.remove(filename)

            sumber = "C:\\final\\"
            fotomedis = "C:\\final\\Hasil\\medis"
            fotononmedis = "C:\\final\\Hasil\\nonmedis"
            for file in glob.glob('Pilihan/*.png'):
                foto=cv2.imread(file)
                foto = cv2.resize(foto, (500, 600))
                foto = cv2.convertScaleAbs(foto, alpha=1, beta=15)
                channels = cv2.split(foto) 
                colors = ("b", "g", "r") 
                for (i, col) in zip(channels, colors): 
                    hist2 = cv2.calcHist([i], [0], None, [256], [0, 256])
                    hist2 = cv2.normalize(hist2, hist2).flatten()
                pilihanhist[file]=hist2
                dhistuji.append(hist2)
                np.savetxt('datahistogramUji.csv', dhistuji, delimiter='|')    
                
                for (k, hist) in ujihist.items():
                    for (v, wa) in ujihist.items():
                        yy = dist.euclidean(wa, hist)
                        min2hist.append(yy)
                    break
                minn = min(i for i in min2hist if i > 0)
                maxx = median(min2hist)

                for (k, hist) in datahist.items():
                    a = dist.euclidean(hist2, hist)
                    edis.append(a)
                    if a < minn: 
                        shutil.copy(os.path.join(sumber, file), fotononmedis)
                        Jnon=Jnon+1
                        break
                    if a <= maxx :
                        shutil.copy(os.path.join(sumber, file), fotomedis)
                        Jmed=Jmed+1
                        break
                    if a > maxx:
                        shutil.copy(os.path.join(sumber, file), fotononmedis)
                        Jnon=Jnon+1
                        break  
                np.savetxt('HasilEdistance.csv', edis, delimiter='|')
            print(Jmed)
            print(Jnon)
            label1.configure(text=Jmed)
            label2.configure(text=Jnon)
            
            listmedis.delete(0, tk.END)
            listmedis.insert(tk.END, "===================")        
            listmedis.insert(tk.END, "Masker medis")  
            listmedis.insert(tk.END, "===================") 
            for filename in glob.glob("Hasil/medis/*.png"):
                listmedis.insert(tk.END, filename)
            
            listnon.delete(0, tk.END)
            listnon.insert(tk.END, "===================")        
            listnon.insert(tk.END, "Masker Non medis")  
            listnon.insert(tk.END, "===================") 
            for file in glob.glob("Hasil/nonmedis/*.png"):
                listnon.insert(tk.END, file)

        def show1(event):
            a.clear()
            n = listmedis.curselection()
            filename = listmedis.get(n)
            img = tk.PhotoImage(file=filename)
            w, h = img.width(), img.height()
            #print(filename)
            canvas.image = img
            canvas.create_image(0, 0, image=img, anchor=tk.NW)
            foto=cv2.imread(filename)
            m = cv2.resize(foto, (500, 600))
            channels = cv2.split(m)      
            colors = ("b", "g", "r") 
            for (i, col) in zip(channels, colors):       
                hist = cv2.calcHist([i], [0], None, [256], [0, 256])
                hist3 = cv2.normalize(hist, hist).flatten()
                a.plot( hist3, color=col)
            canvahist.draw()
            for (k, hist) in datahist.items():
                 b = dist.euclidean(hist3, hist)
                 label3.configure(text=b)


        def show2(event):
            a.clear()
            n = listnon.curselection()
            filename = listnon.get(n)
            img = tk.PhotoImage(file=filename)
            w, h = img.width(), img.height()
            canvas.image = img
            canvas.create_image(0, 0, image=img, anchor=tk.NW)
            foto=cv2.imread(filename)
            m = cv2.resize(foto, (500, 600))
            channels = cv2.split(m)      
            colors = ("b", "g", "r") 
            for (i, col) in zip(channels, colors):      
                hist = cv2.calcHist([i], [0], None, [256], [0, 256])
                hist3 = cv2.normalize(hist, hist).flatten()
                a.plot( hist3, color=col)
            canvahist.draw()
            for (k, hist) in datahist.items():
                 b = dist.euclidean(hist3, hist)
                 label3.configure(text=b)
        #----------------------------------------------
        f= Figure(figsize=(3,3), dpi=100)
        a= f.add_subplot(111)
        canvahist = FigureCanvasTkAgg(f, self)
        canvahist.draw()
        canvahist.get_tk_widget().pack(side="right",padx=10)
        #---------------------------------------------
        canvas = tk.Canvas(self,background='#F4E7D7')
        canvas.config(width=500, height=500)
        canvas.pack(side="right", padx=10)
        #---------------------------------------------
        listmedis = tk.Listbox(self, width=20,background='#F4E7D7',activestyle='none')
        listmedis.pack(side="right", fill=tk.BOTH, expand=0, padx=10, pady=10)
        listmedis.bind("<<ListboxSelect>>", show1)
        listmedis.insert(tk.END, "===================")        
        listmedis.insert(tk.END, "Masker medis")  
        listmedis.insert(tk.END, "===================") 
        
        listnon = tk.Listbox(self, width=20,background='#F4E7D7',activestyle='none')
        listnon.pack(side="right", fill=tk.BOTH, expand=0, padx=10, pady=10)
        listnon.bind("<<ListboxSelect>>", show2)
        listnon.insert(tk.END, "===================")        
        listnon.insert(tk.END, "Masker medis")  
        listnon.insert(tk.END, "===================") 

        #----------------------------------------------
        buttonn = tk.Button(self, text="Hasil", borderwidth=0.5, bg='#FEB47B',height=3,width=15, command=banding)
        buttonn.pack(side="top",padx=10, pady=10)
        button1 = tk.Button(self, text="Pengenalan", borderwidth=0.5, bg='#FEB47B',height=3,width=15, command=lambda: controller.show_frame(Pengenalan))
        button1.pack(side="top",padx=10, pady=10)
        button2 = tk.Button(self, text="Home", borderwidth=0.5, bg='#FEB47B',height=3,width=15, command=lambda: controller.show_frame(Utama))
        button2.pack(side="top",padx=10, pady=10)

        tk.Label(self,text="Jumlah M.Medis:", borderwidth=0.5, bg='#FEB47B',height=3,width=15).pack(side="top", padx=10, pady=10)
        label1=tk.Label(self, borderwidth=0.5, bg='#FEB47B',height=3,width=15)
        label1.pack(side="top", padx=10, pady=10)
        tk.Label(self,text="Jumlah M.non Medis:", borderwidth=0.5, bg='#FEB47B',height=3,width=15).pack(side="top", padx=10, pady=10)
        label2=tk.Label(self, borderwidth=0.5, bg='#FEB47B',height=3,width=15)
        label2.pack(side="top", padx=10, pady=10)
        tk.Label(self,text="Jarak Euclidean foto:", borderwidth=0.5, bg='#FEB47B',height=3,width=15).pack(side="top", padx=10, pady=10)
        label3=tk.Label(self, borderwidth=0.5, bg='#FEB47B',height=3,width=15)
        label3.pack(side="top", padx=10, pady=10)
#------------------------------------
class Help(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self,bg='#849974')
        dimg= []
        for filename in glob.glob('helpIMG/*.png'):
            img = tk.PhotoImage(file=filename)
            dimg.append(img)
        global klik
        klik=0
        def tutor():
            global klik
            if klik ==1:
                a = glob.glob('helpIMG/help-2.png')
                img = tk.PhotoImage(file=a)
                canvas.image = img
                canvas.create_image(0, 0, image=img, anchor=tk.NW)
                klik = klik-1
            else:
                a = glob.glob('helpIMG/help-1.png')
                img = tk.PhotoImage(file=a)
                canvas.image = img
                canvas.create_image(0, 0, image=img, anchor=tk.NW)
                klik = klik+1



        button1 = tk.Button(self, text="Home", borderwidth=0.5, bg='#FEB47B',height=3,width=10, command=lambda: controller.show_frame(Utama))
        button1.pack(side="top")
        button1 = tk.Button(self, text="next", borderwidth=0.5, bg='#FEB47B',height=3,width=10, command= tutor)
        button1.pack(side="top")    
        canvas = tk.Canvas(self)
        canvas.image = img
        canvas.create_image(0, 0, image=img, anchor=tk.NW)
        a = glob.glob('helpIMG/help-2.png')
        img = tk.PhotoImage(file=a)
        canvas.config(width=500, height=500)
        canvas.pack(side="top", padx=50, pady=10)

app = App()
app.mainloop()