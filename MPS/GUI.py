#!/usr/bin/python3

from tkinter import *
from tkinter.ttk import *
import tkinter.filedialog as tkdiag
from main import *
import threading
import queue

window = Tk()
window.geometry('540x540')
window.configure(bg='#fdcf60')
window.title('Machine learning Covid')

que = queue.Queue()

frameCnt = 6
frames = [PhotoImage(file='./covid_g.gif', format='gif -index %i' % i) for i in range(frameCnt)]


def update(ind):
    frame = frames[ind]
    ind += 1
    if ind == frameCnt:
        ind = 0

    label.configure(image=frame)
    window.after(100, update, ind)


label = Label(window, background='#fdcf60')
label.place(relx=0.48, rely=0)


button_label = Label(window, text='Choose a set', font=
('calibri', 10, 'bold'), background='#fdcf60')
button_label.place(relx=0.105, rely=0.15, anchor='center')

result = Label()


def changeLabelText():
    button_label.configure(text='Choose a set')
    button_label.place(relx=0.105, rely=0.15, anchor='center')


def getSet():
    file = tkdiag.askopenfile(parent=window, mode='r', title='Choose a file')
    if file:
        if str(file.name).endswith('.xlsx') or str(file.name).endswith('.xls'):
            result.configure(text="Processing... This little maneuver is gonna cost us 51 years.", justify='left',
                             background='#fdcf60', padding=10, font=('calibri', 10, 'bold'), wraplength=250)
            result.place(rely=0.8, anchor='w')
            window.update()
            thread = threading.Thread(target=lambda q, arg1: q.put(encode(arg1)), args=(que, str(file.name)))
            thread.start()
            while thread.is_alive():
                window.update()
            thread.join()
            out = que.get()

            result.configure(text=out)
        else:
            result.configure(text='')
            button_label.configure(text='File is not supported', font=
            ('calibri', 10, 'bold'))
            button_label.place(relx=0.165, rely=0.15, anchor='center')
            button_label.after(5000, changeLabelText)


style = Style()

style.configure('W.TButton', font=
('calibri', 10, 'bold'),
                foreground='red', background='#DCDCDC')

button = Button(window,
                text='Browse',
                style='W.TButton',
                command=getSet
                )

button.place(relx=0.1, rely=0.2, anchor='center')
window.after(0, update, 0)
window.mainloop()
