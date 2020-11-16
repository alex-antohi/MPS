# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import math
import pandas as pd
from datetime import datetime
from pandas import read_csv
import re
import numpy as np
import sklearn

from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import mean_absolute_error as auto_mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report

class Patient:
    def __init__(self, sex, varsta, data_simpt, simpt_decl,
                 data_internare, simpt_rap, diagnostic, calatorii, mij_trans,
                 contact, data_rez, rez):
        self.sex = sex
        self.varsta = varsta
        self. data_simpt = data_simpt
        self.simpt_decl = simpt_decl
        self.data_internare = data_internare
        self.simpt_rap = simpt_rap
        self.diagnostic = diagnostic
        self.calatorii = calatorii
        self.mij_trans = mij_trans
        self.contact = contact
        self.data_rez = data_rez
        self.rez = rez

    if __name__ == '__main__':
        df = pd.read_excel('mps.dataset.xlsx')
        df = pd.DataFrame(df, columns=df.columns)
        
        # replacing nan
        df['vârstă'] = df['vârstă'].replace(np.nan, -1)
        df['dată debut simptome declarate'] = df['dată debut simptome declarate'].replace(np.nan, -1)
        df['dată internare'] = df['dată internare'].replace(np.nan, -1)
        df['data rezultat testare'] = df['data rezultat testare'].replace(np.nan, -1)
        df['istoric de călătorie'] = df['istoric de călătorie'].replace(np.nan, 0)
        df['mijloace de transport folosite'] = df['mijloace de transport folosite'].replace(np.nan, 0)
        df['confirmare contact cu o persoană infectată'] = df['confirmare contact cu o persoană infectată'].replace(np.nan, 0)
        df['simptome declarate'] = df['simptome declarate'].replace(np.nan, 0)
        df['simptome raportate la internare'] = df['simptome raportate la internare'].replace(np.nan, 0)
        df['diagnostic și semne de internare'] = df['diagnostic și semne de internare'].replace(np.nan, 0)
        df['rezultat testare'] = df['rezultat testare'].replace(np.nan, 0)      
        
        
        # df.index
        nrRows = df.index.stop
        for i in range(nrRows):
            # codificare instituția sursă
            if isinstance(df['instituția sursă'][i], str):
                if 'X' in df['instituția sursă'][i].upper():
                    df['instituția sursă'][i] = 0
                elif 'Y' in df['instituția sursă'][i].upper():
                    df['instituția sursă'][i] = 1
                elif 'Z' in df['instituția sursă'][i].upper():
                    df['instituția sursă'][i] = 2
            else:
                df['instituția sursă'][i] = -1


            # codificare sex
            if isinstance(df['sex'][i], str):
                if 'MASCULIN' in df['sex'][i].upper() or 'M' in df['sex'][i].upper():
                    df['sex'][i] = 0
                elif 'FEMININ' in df['sex'][i].upper() or 'F' in df['sex'][i].upper():
                    df['sex'][i] = 1
                else:
                    # exceptii
                    print(df['sex'][i], i)
            else:
                df['sex'][i] = -1


            # codificare varsta
            if isinstance(df['vârstă'][i], int):
                df['vârstă'][i] = int(df['vârstă'][i])
            elif isinstance(df['vârstă'][i], str):
                if df['vârstă'][i].isdecimal():
                    df['vârstă'][i] = int(df['vârstă'][i])
                else:
                    if 'ANI' in df['vârstă'][i].upper():
                        df['vârstă'][i] = df['vârstă'][i].upper().replace(' ANI', '')
                        if df['vârstă'][i].isdecimal():
                            df['vârstă'][i] = int(df['vârstă'][i])
                        elif 'LUN' in df['vârstă'][i].upper():
                            df['vârstă'][i] = df['vârstă'][i].upper().replace(' LUNA', '')
                            df['vârstă'][i] = df['vârstă'][i].upper().replace(' LUNI', '')
                            df['vârstă'][i] = df['vârstă'][i].upper().replace('LUNA', '')
                            df['vârstă'][i] = df['vârstă'][i].upper().replace('LUNI', '')
                            df['vârstă'][i] = int(df['vârstă'][i].split(' ')[0])
                            #   + int(df['vârstă'][i].split(' ')[1]) / 12
                    elif 'LUN' in df['vârstă'][i].upper():
                        df['vârstă'][i] = df['vârstă'][i].upper().replace(' LUNA', '')
                        df['vârstă'][i] = df['vârstă'][i].upper().replace(' LUNI', '')
                        df['vârstă'][i] = df['vârstă'][i].upper().replace('LUNA', '')
                        df['vârstă'][i] = df['vârstă'][i].upper().replace('LUNI', '')
                        if df['vârstă'][i].isdecimal():
                            df['vârstă'][i] = int(df['vârstă'][i])
                        else:
                            df['vârstă'][i] = df['vârstă'][i].upper().replace(' ', '')
                            if df['vârstă'][i].isdecimal():
                                df['vârstă'][i] = int(df['vârstă'][i])
                    else:
                        # exceptii
                        df['vârstă'][i] = 0


            # codificare dată debut simptome declarate ca numarul zilei din an
            if isinstance(df['dată debut simptome declarate'][i], datetime):
                df['dată debut simptome declarate'][i] = int(df['dată debut simptome declarate'][i].strftime("%j"))
            elif isinstance(df['dată debut simptome declarate'][i], str):
                if 'NU' in df['dată debut simptome declarate'][i].upper() or\
                        '-' in df['dată debut simptome declarate'][i]:
                    df['dată debut simptome declarate'][i] = -1
                else:
                    df['dată debut simptome declarate'][i] = df['dată debut simptome declarate'][i].replace(' ', '-')
                    df['dată debut simptome declarate'][i] = df['dată debut simptome declarate'][i].replace('.', '-')
                    df['dată debut simptome declarate'][i] = df['dată debut simptome declarate'][i].replace(',', '-')

                    if re.match(r"(\d+)-(\d+)-(\d\d\d\d)$", df['dată debut simptome declarate'][i]):
                        df['dată debut simptome declarate'][i] = \
                            datetime.strptime(df['dată debut simptome declarate'][i], '%d-%m-%Y')
                        df['dată debut simptome declarate'][i] = \
                            int(df['dată debut simptome declarate'][i].strftime("%j"))
                    else:
                        # exceptii nerezolvate
                        df['dată debut simptome declarate'][i] = 0


            # codificare dată internare ca numarul zilei din an
            if isinstance(df['dată internare'][i], datetime):
                df['dată internare'][i] = int(df['dată internare'][i].strftime("%j"))
            elif isinstance(df['dată internare'][i], str):
                if 'NU' in df['dată internare'][i].upper() or\
                        '-' in df['dată internare'][i]:
                    df['dată internare'][i] = -1
                else:
                    df['dată internare'][i] = df['dată internare'][i].replace(' ', '-')
                    df['dată internare'][i] = df['dată internare'][i].replace('.', '-')
                    df['dată internare'][i] = df['dată internare'][i].replace(',', '-')

                    if re.match(r"(\d+)-(\d+)-(\d\d\d\d)$", df['dată internare'][i]):
                        df['dată internare'][i] = datetime.strptime(df['dată internare'][i], '%d-%m-%Y')
                        df['dată internare'][i] = int(df['dată internare'][i].strftime("%j"))
                    else:
                        # exceptii nerezolvate
                        df['dată internare'][i] = 0


            # codificare dată rezultat testare ca numarul zilei din an
            
            if isinstance(df['data rezultat testare'][i], datetime):
                df['data rezultat testare'][i] = int(df['data rezultat testare'][i].strftime("%j"))
            elif isinstance(df['data rezultat testare'][i], str):
                if 'NU' in df['data rezultat testare'][i].upper() or\
                        '-' in df['data rezultat testare'][i]:
                    df['data rezultat testare'][i] = -1
                else:
                    df['data rezultat testare'][i] = df['data rezultat testare'][i].replace(' ', '-')
                    df['data rezultat testare'][i] = df['data rezultat testare'][i].replace('.', '-')
                    df['data rezultat testare'][i] = df['data rezultat testare'][i].replace(',', '-')

                    if re.match(r"(\d+)-(\d+)-(\d\d\d\d)$", df['data rezultat testare'][i]):
                        df['data rezultat testare'][i] = datetime.strptime(df['data rezultat testare'][i], '%d-%m-%Y')
                        df['data rezultat testare'][i] = int(df['data rezultat testare'][i].strftime("%j"))
                    else:
                        # exceptii nerezolvate
                        df['data rezultat testare'][i] = 0
                    
            
            # codificare istoric calatorii
            if isinstance(df['istoric de călătorie'][i], str):
                if re.match("( *)NU( *)", df['istoric de călătorie'][i].upper()) or \
                        re.match("( *)NEAG", df['istoric de călătorie'][i].upper()) or \
                        re.match("( *)FARA( *)", df['istoric de călătorie'][i].upper()) or \
                        re.match("( *)0( *)", df['istoric de călătorie'][i]):
                    df['istoric de călătorie'][i] = 0
                elif re.match("( *)DA( *)", df['istoric de călătorie'][i].upper()) or \
                        re.match("( *)1( *)", df['istoric de călătorie'][i]):
                    df['istoric de călătorie'][i] = 1
                else:
                    df['istoric de călătorie'][i] = 1
            elif df['istoric de călătorie'][i] == 0:
                df['istoric de călătorie'][i] = 0
            elif df['istoric de călătorie'][i] == 1:
                df['istoric de călătorie'][i] = 1


            # codificare mijloace de transport folosite
            if isinstance(df['mijloace de transport folosite'][i], str):
                if re.match("( *)NU( *)", df['mijloace de transport folosite'][i].upper()) or \
                        re.match("( *)NEAG", df['mijloace de transport folosite'][i].upper()) or \
                        re.match("( *)FARA( *)", df['mijloace de transport folosite'][i].upper()) or \
                        re.match("( *)0( *)", df['mijloace de transport folosite'][i]):
                    df['mijloace de transport folosite'][i] = 0
                elif re.match("( *)DA( *)", df['mijloace de transport folosite'][i].upper()) or \
                        re.match("( *)1( *)", df['mijloace de transport folosite'][i]):
                    df['mijloace de transport folosite'][i] = 1
                else:
                    df['mijloace de transport folosite'][i] = 1
            elif df['mijloace de transport folosite'][i] == 0:
                df['mijloace de transport folosite'][i] = 0
            elif df['mijloace de transport folosite'][i] == 1:
                df['mijloace de transport folosite'][i] = 1


            # codificare confirmare contact cu o persoană infectată
            if isinstance(df['confirmare contact cu o persoană infectată'][i], str):
                if re.match("( *)NU( *)", df['confirmare contact cu o persoană infectată'][i].upper()) or \
                        re.match("( *)NEAG", df['confirmare contact cu o persoană infectată'][i].upper()) or \
                        re.match("( *)FARA( *)", df['confirmare contact cu o persoană infectată'][i].upper()) or \
                        re.match("( *)0( *)", df['confirmare contact cu o persoană infectată'][i]):
                    df['confirmare contact cu o persoană infectată'][i] = 0
                elif re.match("( *)DA( *)", df['confirmare contact cu o persoană infectată'][i].upper()) or \
                        re.match("( *)1( *)", df['confirmare contact cu o persoană infectată'][i]):
                    df['confirmare contact cu o persoană infectată'][i] = 1
                else:
                    df['confirmare contact cu o persoană infectată'][i] = 1
            elif df['confirmare contact cu o persoană infectată'][i] == 0:
                df['confirmare contact cu o persoană infectată'][i] = 0
            elif df['confirmare contact cu o persoană infectată'][i] == 1:
                df['confirmare contact cu o persoană infectată'][i] = 1


            # codificare simptome declarate
            if isinstance(df['simptome declarate'][i], str):
                if re.match("( *)ASIMPTOMATIC( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)NU( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)FARA( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)ABSENTE( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)AFEBRIL( *)", df['simptome declarate'][i].upper()):
                    df['simptome declarate'][i] = 0
                elif re.match("( *)TUS( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)DIAREE( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)OBOSEALA( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)DISP( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)RESPIRAT( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)FEBRA( *)", df['simptome declarate'][i]):
                    df['simptome declarate'][i] = 1
                else:
                    df['simptome declarate'][i] = -1
            else:
                df['simptome declarate'][i] = -1


            # codificare simptome raportate la internare
            if isinstance(df['simptome raportate la internare'][i], str):
                if re.match("( *)ASIM( *)", df['simptome raportate la internare'][i].upper()) or \
                        re.match("( *)NU( *)", df['simptome raportate la internare'][i].upper()) or \
                        re.match("( *)FARA( *)", df['simptome raportate la internare'][i].upper()) or \
                        re.match("( *)ABSENTE( *)", df['simptome raportate la internare'][i].upper()) or \
                        re.match("( *)AFEBRIL( *)", df['simptome raportate la internare'][i].upper()):
                    df['simptome raportate la internare'][i] = 0
                elif re.match("( *)TUS( *)", df['simptome raportate la internare'][i].upper()) or \
                        re.match("( *)DIAREE( *)", df['simptome raportate la internare'][i].upper()) or \
                        re.match("( *)OBOSEALA( *)", df['simptome raportate la internare'][i].upper()) or \
                        re.match("( *)DISP( *)", df['simptome raportate la internare'][i].upper()) or \
                        re.match("( *)INSUF( *)", df['simptome raportate la internare'][i].upper()) or \
                        re.match("( *)PIERDERE( *)", df['simptome raportate la internare'][i].upper()) or \
                        re.match("( *)RESPIRAT( *)", df['simptome raportate la internare'][i].upper()) or \
                        re.match("( *)FEBRA( *)", df['simptome raportate la internare'][i]):
                    df['simptome raportate la internare'][i] = 1
                else:
                    df['simptome raportate la internare'][i] = -1


            # codificare diagnostic și semne de internare
            if isinstance(df['diagnostic și semne de internare'][i], str):
                if re.match("( *)ASIM( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)NU( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)FARA( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)ABSENTE( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)AFEBRIL( *)", df['diagnostic și semne de internare'][i].upper()):
                    df['diagnostic și semne de internare'][i] = 0
                elif re.match("( *)TUS( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)DIAREE( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)OBOSEALA( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)DISP( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)INSUF( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)PIERDERE( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)RESP( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)COV( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)SARS( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)SUSP( *)", df['diagnostic și semne de internare'][i].upper()) or \
                        re.match("( *)FEBRA( *)", df['diagnostic și semne de internare'][i]):
                    df['diagnostic și semne de internare'][i] = 1
                else:
                    df['diagnostic și semne de internare'][i] = -1


            
            # codificare coloana rezultate
            if isinstance(df['rezultat testare'][i], str):
                if re.match("( *)NEGATIV( *)", df['rezultat testare'][i].upper()):
                    df['rezultat testare'][i] = 0
                elif re.match("( *)POZITIV( *)", df['rezultat testare'][i].upper()):
                    df['rezultat testare'][i] = 1
                else:
                    df['rezultat testare'][i] = -1

        
        # verific ca toate valorile sa fie int-uri
        for i in range(nrRows):
            if not isinstance(df['instituția sursă'][i], int) or\
                    not isinstance(df['sex'][i], int) or\
                    not isinstance(df['vârstă'][i], int) or\
                    not isinstance(df['dată debut simptome declarate'][i], int) or\
                    not isinstance(df['dată internare'][i], int) or\
                    not isinstance(df['data rezultat testare'][i], int) or\
                    not isinstance(df['istoric de călătorie'][i], int) or\
                    not isinstance(df['mijloace de transport folosite'][i], int) or\
                    not isinstance(df['confirmare contact cu o persoană infectată'][i], int) or\
                    not isinstance(df['simptome declarate'][i], int) or \
                    not isinstance(df['simptome raportate la internare'][i], int) or \
                    not isinstance(df['diagnostic și semne de internare'][i], int) or \
                    not isinstance(df['rezultat testare'][i], int):
                print(i, df['instituția sursă'][i], 
                    df['sex'][i],
                    df['vârstă'][i],
                    df['dată debut simptome declarate'][i],
                    df['dată internare'][i],
                    df['data rezultat testare'][i],
                    df['istoric de călătorie'][i],
                    df['mijloace de transport folosite'][i],
                    df['confirmare contact cu o persoană infectată'][i],
                    df['simptome declarate'][i],
                    df['simptome raportate la internare'][i],
                    df['diagnostic și semne de internare'][i],
                    df['rezultat testare'][i])

    
        df.to_csv('dfexport.csv', encoding='utf-8', index=False, header=None)
        
        cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12]
        newdf = read_csv('dfexport.csv', header=None, usecols=cols, nrows=nrRows)
        newdf = newdf.values
        newdf = newdf.astype('int')
        
            
        poz = np.empty((0, 13), int)
        neg = np.empty((0, 13), int)
        for i in range(nrRows):
            if newdf[i][12] == 1:
                poz = np.concatenate((poz, [newdf[i]]), axis=0)
            else:
                neg = np.concatenate((neg, [newdf[i]]), axis=0)

        X_train_p, X_test_p =  np.array_split(poz, [int(len(poz) * 0.8)])
        X_train_n, X_test_n =  np.array_split(neg, [int(len(neg) * 0.8)])

        X_test = np.concatenate((X_test_p, X_test_n))
        
        X_train_p = np.delete(X_train_p, 12, 1)
        X_train_n = np.delete(X_train_n, 12, 1)
        X_test = np.delete(X_test, 12, 1)
        
        X_validare_p = np.array_split(X_train_p, [int(len(X_train_p) * 0.7)])[0]
        X_validare_n = np.array_split(X_train_n, [int(len(X_train_n) * 0.7)])[0]
        X_validare = np.concatenate((X_validare_p, X_validare_n))
        
        
        X_antrenare_p = np.array_split(X_train_p, [int(len(X_train_p) * 0.3)])[1]
        X_antrenare_n = np.array_split(X_train_n, [int(len(X_train_n) * 0.3)])[1]
        X_antrenare = np.concatenate((X_antrenare_p, X_antrenare_n))
        
        y_train_p, y_test_p =  np.array_split(poz, [int(len(poz) * 0.8)])
        y_train_n, y_test_n =  np.array_split(neg, [int(len(neg) * 0.8)])

        y_test = np.concatenate((y_test_p, y_test_n))

        for i in range(0,12,1):
            y_train_p = np.delete(y_train_p, 0, 1)
            y_train_n = np.delete(y_train_n, 0, 1)
            y_test = np.delete(y_test, 0, 1)
        y_train_p = y_train_p.flatten()
        y_train_n = y_train_n.flatten()
        y_test = y_test.flatten()

        y_validare_p = np.array_split(y_train_p, [int(len(y_train_p) * 0.7)])[0]
        y_validare_n = np.array_split(y_train_n, [int(len(y_train_n) * 0.7)])[0]
        y_validare = np.concatenate((y_validare_p, y_validare_n))
        
        y_antrenare_p = np.array_split(y_train_p, [int(len(y_train_p) * 0.3)])[1]
        y_antrenare_n = np.array_split(y_train_n, [int(len(y_train_n) * 0.3)])[1]
        y_antrenare = np.concatenate((y_antrenare_p, y_antrenare_n))
        
        
        model = OneVsRestClassifier(svm.SVC(class_weight='balanced')).fit(X_antrenare, y_antrenare)
        out_validare = model.predict(X_validare)
        
        
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(out_validare.size):
            if out_validare[i] == y_validare[i]:
                if (out_validare[i] == 0):
                        TN += 1
                if (out_validare[i] == 1):
                        TP += 1
            else:
                if (out_validare[i] == 0):
                        FN += 1
                if (out_validare[i] == 1):
                        FP += 1
        
        acuratete = (TP + TN) / (TP + FP + TN + FN)
        print('Acuratete_v = ', "{:.1f}".format(acuratete))
        
        precizie = TP / (TP + FP)
        print('Precizie_v = ', "{:.1f}".format(precizie))
        
        rapel = TP / (TP + FN)
        print('Rapel_v = ', "{:.1f}".format(rapel))
        
        scorF1 = (2 * precizie * rapel) / (precizie + rapel)
        print('ScorF1_v = ', "{:.1f}".format(scorF1))
        
        mat_conf = [[TN, FP], [FN, TP]]
        print(mat_conf)
        
         
        out_test = model.predict(X_test)
        
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(out_test.size):
            if out_test[i] == y_test[i]:
                if (out_test[i] == 0):
                        TN += 1
                if (out_test[i] == 1):
                        TP += 1
            else:
                if (out_test[i] == 0):
                        FN += 1
                if (out_test[i] == 1):
                        FP += 1

        acuratete = (TP + TN) / (TP + FP + TN + FN)
        print()
        print('Acuratete_t = ', "{:.1f}".format(acuratete))
        
        precizie = TP / (TP + FP)
        print('Precizie_t = ', "{:.1f}".format(precizie))

        rapel = TP / (TP + FN)
        print('Rapel_t = ', "{:.1f}".format(rapel))
        
        scorF1 = (2 * precizie * rapel) / (precizie + rapel)
        print('ScorF1_t = ', "{:.1f}".format(scorF1))
        
        mat_conf = [[TN, FP], [FN, TP]]
        print(mat_conf)
        #print(classification_report(y_validare, out_validare))

