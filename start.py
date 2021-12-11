#!/usr/bin/env python

from tkinter import *
import glob
import os

from videoanalysis import analyze


if __name__ == "__main__":
    root = Tk()

    root.geometry('750x500')
    bottom_frame = Frame(root, width=500, height=50)
    bottom_frame.pack(side='bottom')
    btn_exit = Button(bottom_frame, text = 'Exit', bd = '5',
                              command = root.destroy)
    btn_exit.pack(side='right')
    btn_analyze = Button(bottom_frame, text = 'Analyze', bd = '5',
            command = lambda: analyze.analyze(v.get(), f.get(), webcam=w.get()))
    btn_analyze.pack(side='left')

    ## Create part for video selection
    video_frame = Frame(root, width=200, height=800)
    video_frame.pack(side='left')
    videos = sorted(glob.glob("res/*.mp4"))
    videos.append("live")
    videos.append("webcam")
    v = StringVar(video_frame, videos[0])

    for video in videos:
        Radiobutton(video_frame, text = os.path.basename(video), variable = v,
            value = video).pack(side = TOP, ipady = 5)

    Label(video_frame, text="Webcam: ").pack(side=LEFT)
    w = Entry(video_frame)
    w.pack(side=RIGHT)

    ## Create part for filter selection
    filter_frame = Frame(root, width=300, height=800)
    filter_frame.pack(side='right')
    filters = analyze.filters

    f = StringVar(filter_frame, list(filters.keys())[0])
    for filt in filters:
        Radiobutton(filter_frame, text=filt, variable = f,
                value=filt).pack(side=TOP, ipady=5)


    root.mainloop()
