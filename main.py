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
        
        # df.index
        nrRows = 1000
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
                        # print(df['vârstă'][i], i)
                        df['vârstă'][i] = 0
            else:
                if math.isnan(df['vârstă'][i]):
                    df['vârstă'][i] = -1

            # codificare dată debut simptome declarate ca numarul zilei din an
            if isinstance(df['dată debut simptome declarate'][i], datetime):
                # print(df['dată debut simptome declarate'][i])
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
                        # print(df['dată debut simptome declarate'][i])
            else:
                if math.isnan(df['dată debut simptome declarate'][i]):
                    df['dată debut simptome declarate'][i] = -1

            # codificare dată internare ca numarul zilei din an
            if isinstance(df['dată internare'][i], datetime):
                # print(df['dată internare'][i])
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
                        # print(df['dată internare'][i])
            else:
                if math.isnan(df['dată internare'][i]):
                    df['dată internare'][i] = -1

            # codificare dată rezultat testare ca numarul zilei din an
            if isinstance(df['data rezultat testare'][i], datetime):
                # print(df['data rezultat testare'][i])
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
                        # print(df['data rezultat testare'][i])
                        df['data rezultat testare'][i] = 0
            else:
                if math.isnan(df['data rezultat testare'][i]):
                    df['data rezultat testare'][i] = -1
            # print(df['instituția sursă'][i], df['sex'][i], df['vârstă'][i],
            # df['dată debut simptome declarate'][i], df['dată internare'][i], df['data rezultat testare'][i])

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
            elif math.isnan(df['istoric de călătorie'][i]):
                df['istoric de călătorie'][i] = 0
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

            elif math.isnan(df['mijloace de transport folosite'][i]):
                df['mijloace de transport folosite'][i] = 0
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

            elif math.isnan(df['confirmare contact cu o persoană infectată'][i]):
                df['confirmare contact cu o persoană infectată'][i] = 0
            elif df['confirmare contact cu o persoană infectată'][i] == 0:
                df['confirmare contact cu o persoană infectată'][i] = 0
            elif df['confirmare contact cu o persoană infectată'][i] == 1:
                df['confirmare contact cu o persoană infectată'][i] = 1

            # codificare simptome declarate
            df['simptome declarate'][i] = 69

            # codificare simptome raportate la internare
            df['simptome raportate la internare'][i] = 69

            # codificare diagnostic și semne de internare
            df['diagnostic și semne de internare'][i] = 69

            # codificare coloana rezultate
            if isinstance(df['rezultat testare'][i], str):
                if re.match("( *)NEGATIV( *)", df['rezultat testare'][i].upper()):
                    df['rezultat testare'][i] = 0
                elif re.match("( *)POZITIV( *)", df['rezultat testare'][i].upper()):
                    df['rezultat testare'][i] = 1
                else:
                    df['rezultat testare'][i] = -1

            elif math.isnan(df['rezultat testare'][i]):
                df['rezultat testare'][i] = 0

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
                print(i, df['instituția sursă'][i], df['sex'][i], df['vârstă'][i],
                      df['dată debut simptome declarate'][i],
                      df['dată internare'][i], df['data rezultat testare'][i],
                      df['istoric de călătorie'][i],
                      df['mijloace de transport folosite'][i],
                      df['confirmare contact cu o persoană infectată'][i],
                      df['simptome declarate'][i],
                      df['simptome raportate la internare'][i],
                      df['diagnostic și semne de internare'][i],
                      df['rezultat testare'][i])
                      
        df.to_csv('dfexport.csv', encoding='utf-8', index=False, header=None)

        cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11]
        Xframe = read_csv('dfexport.csv', header=None, usecols=cols, nrows=nrRows)
        X = Xframe.values
        X = X.astype('int')
        
        cols = [12]
        yframe = read_csv('dfexport.csv', header=None, usecols=cols, nrows=nrRows)
        y = yframe.values
        y = y.astype('int')
        y = np.concatenate(y, axis=0)
        
        
        X_train, X_test = np.array_split(X, [int(df.index.stop * 0.8)])
        X_antrenare = np.array_split(X_train, [int(X_train.size/12 * 0.7)])[0]
        X_validare = np.array_split(X_train, [int(X_train.size/12 * 0.3)])[1]
        
        y_train, y_test = np.array_split(y, [int(df.index.stop * 0.8)])
        y_antrenare = np.array_split(y_train, [int(y_train.size * 0.7)])[0]
        y_validare = np.array_split(y_train, [int(y_train.size * 0.3)])[1]
        
             
        model = OneVsRestClassifier(svm.SVC()).fit(X_antrenare, y_antrenare)
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
        print('ACURATETE = ', "{:.1f}".format(acuratete))
        
        precizie = TP / (TP + FP)
        print('PRECIZIE = ', "{:.1f}".format(precizie))
        
        rapel = TP / (TP + FN)
        print('RAPEL = ', "{:.1f}".format(rapel))
        
        scorF1 = (2 * precizie * rapel) / (precizie + rapel)
        print('SCORUL F1 = ', "{:.1f}".format(scorF1))
        
        mat_conf = [[TN, FP], [FN, TP]]
        print(mat_conf)

