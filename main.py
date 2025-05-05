from scipy.io.wavfile import read as wav_r
from scipy import fft
import scipy as sc
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import scipy.signal as sign


def remove_duplicates(array):
    seen = set()
    new_array = []
    for element in array:
        if element not in seen:
            seen.add(element)
            new_array.append(element)
    return new_array


def find_note(frequency):
    notes = [' A', ' A#', ' B', 'C', 'C#', ' D', ' D#', ' E', ' F', ' F#', ' G', ' G#']

    note_number = round(12 * np.log2(frequency / 440) + 49)
    note = notes[(note_number - 1) % len(notes)]
    octave = (note_number + 8) // len(notes)
    if octave > 0:
        return note, octave
    else:
        return "none", "none"


def find_lilipond_note(frequency):
    lily_notes = ['a', ' ais', 'b', 'c', 'cis', 'd', ' dis', 'e', 'f', ' fis', 'g', 'gis']

    note_number = round(12 * np.log2(frequency / 440) + 49)
    note = lily_notes[(note_number - 1) % len(lily_notes)]
    octave = (note_number + 8) // len(lily_notes)

    if octave > 2:
        lily_octave = ""
        for i in range(octave-3):
            lily_octave += "'"
        return note + lily_octave
    else:
        return "none", "none"


def find_and_label_peaks(signal, freqs, lowest, highest):
    peaks = sc.signal.find_peaks(np.round(signal), threshold=1, prominence=3, width=1, distance=5)
    peaks = peaks[0]

    peak_freq = []
    peak_value = []
    peak_note = []

    for peak_i in peaks:
        new_freq = freqs[peak_i]
        if new_freq < lowest or new_freq < highest:
            new_peak = signal[peak_i]
            peak_value.append(new_peak)

            peak_freq.append(new_freq)

            new_note, new_octave = find_note(new_freq)
            peak_note.append(new_note + str(new_octave))

            print(round(new_freq, 2), "&", round(10 * np.log10(new_peak), 2), "&", new_note, "&", new_octave, "\\\\")
            print(find_lilipond_note(new_freq))
    return peak_freq, peak_value, peak_note


filename = "flute3.wav"
wave = wav_r(filename)
sample_rate = wave[0]
print("srate = ", sample_rate)
wave = wave[1]
print("len = ", len(wave))
wave = wave[0:700000]

# _______________________ find duration ___________________________________
tempo = 120
cutoff_low, cutoff_high = 100, 800
SEGMENT_MS = 8
duration = 8
song = AudioSegment.from_wav(filename)

volume = [segment.dBFS for segment in song[::SEGMENT_MS]]
volume = volume[0:int(duration*1000/SEGMENT_MS+100/SEGMENT_MS)]

# Filter parameters
fs = 2000  # Sampling rate in Hz
order = 1  # Filter order

# Design the low-pass filter
nyq = 0.5 * fs
normal_cutoff = cutoff_low / nyq
b, a = sign.butter(order, normal_cutoff, btype='highpass')
filtered_data = sign.filtfilt(b, a, volume)

pauses = sc.signal.find_peaks(np.abs(filtered_data), threshold=0, prominence=1.05, width=0.5, distance=15)
pauses = pauses[0] * SEGMENT_MS / 1000
pause0 = pauses[0]
pauses = pauses - pause0
print("pauses(",len(pauses),") = ", pauses)
round_pauses = np.round(pauses*20)/20
print("round_pauses = ", round_pauses )


interval = int(tempo/15)
notedurs = []
for i in range(len(pauses)):
    if i:
        notedurs.append(pauses[i] - pauses[i - 1])

notedur_names = []
lily_notedurs_names = []
# names = ["eigth", "quarter", "half", "whole"]
names = [ r"$\frac{1}{16}$", r"$\frac{1}{8}$",  r"$\frac{1}{4}$", r"$\frac{1}{2}$", "1"]
lily_names = [ "16", "8",  "4", "2", "1"]

for i in range(len(notedurs)):
    notedur_names.append(names[int(np.log2(np.round(interval * np.asarray(notedurs[i]))))])
    lily_notedurs_names.append(lily_names[int(np.log2(np.round(interval * np.asarray(notedurs[i]))))])
print(notedur_names)



# _______________________ find frequencies ___________________________________

start = 0
step = 2205
last_freq = 0
last_time = 0

iteration = int((len(wave)) / step)

notefreqs = []
lilynotes = []

for i in range(iteration):
    wave_section = wave[start + step * i:start + step * (i + 1)]

    tfreq = fft.fftfreq(wave_section.size, 1 / sample_rate)
    twave = fft.fft(wave_section)
    psd = 2 / wave.size * np.abs(twave[round(twave.size / 2):])
    freqs = np.abs(tfreq[round(twave.size / 2):])

    t = np.round((start + step * i) / sample_rate,4)
    print("( t=", t, "s)_____________________________________________________________")
    peak_freq, peak_value, peak_note = find_and_label_peaks(psd, freqs, cutoff_low, cutoff_high)

    if peak_freq:
        try:
            max = max(peak_value)
        except:
            max = peak_value[0]
        for freq, val in zip(peak_freq, peak_value):
            if val == max:
                notefreqs.append(freq)
                if last_freq != freq:
                    last_freq = freq
        for round_pause in round_pauses:
            # print("pause and time)",round_pause, t)
            if t == round_pause+0.1:
                lilynotes.append(find_lilipond_note(last_freq))
    else:
        # print("array empty")
        notefreqs.append(last_freq)
        for round_pause in round_pauses:
            # print("pause and time)",round_pause, t)
            if t == round_pause and i:
                lilynotes.append(find_lilipond_note(last_freq))

notefreqs.remove(0)
no_dup = remove_duplicates(notefreqs)
print("lilynotes(", len(lilynotes),") =", lilynotes)

# _______________________ plotting _____________________________________________

fig3 = plt.figure("Note height detection")
ax = fig3.add_subplot(211)
ax.set_title("Frequency changes in time")
plt.ylabel("f (Hz)")
plt.xlabel("t (s)")

plt.plot(np.arange(0, len(notefreqs)) * step / sample_rate, notefreqs, 'k', linewidth=2)

# plotting frequencies
for i in range(len(pauses)):
    plt.axvline(x=pauses[i], color='b', linewidth=0.5, linestyle='--')
plt.axis([0, np.max(pauses) + 0.1, np.min(no_dup) * 0.9, np.max(no_dup) * 1.1])

for i in range(len(no_dup)):
    if i == 0:
        plt.axhline(y=no_dup[i], color='b', linewidth=0.5, linestyle='-', label="Note height")
        ax.text(0.1, no_dup[i] + 5, "  " + find_note(no_dup[i])[0] + str(find_note(no_dup[i])[1]))
    else:
        plt.axhline(y=no_dup[i], color='b', linewidth=0.5, linestyle='-')
        ax.text(0.1, no_dup[i] + 5, "  " + find_note(no_dup[i])[0] + str(find_note(no_dup[i])[1]))
ax.legend(loc="lower right")

# plotting duration
ax = fig3.add_subplot(212)
ax.set_title("Volume difference with butterworth highpass filter applied")
plt.ylabel("volume (dB)")
plt.xlabel("t (s)")
plt.plot(np.arange(len(volume)) * SEGMENT_MS / 1000 - pause0, filtered_data, 'r', linewidth=1)

lensum =0
for i in range(len(pauses)):
    if i == 0:
        plt.axvline(x=pauses[i], color='b', linewidth=0.5, linestyle=':', label="Note boundaries")
    else:
        plt.axvline(x=pauses[i], color='b', linewidth=0.5, linestyle=':')
        ax.text(lensum + notedurs[i - 1] / 16 * 7, np.max(filtered_data) * 0.9, notedur_names[i - 1])
        lensum += notedurs[i - 1]

plt.axis([0, np.max(pauses) + 0.1, np.min(filtered_data) * 1.1, np.max(filtered_data) * 1.1])
plt.legend(loc="lower right")

print(notefreqs)

lily_out = ""
for note, duration in zip(lilynotes, lily_notedurs_names):
    lily_out+=note+str(duration)+"  "

print(lily_out)
plt.show()
