import soundfile as sf
import matplotlib.pyplot as plt

audio, sample_rate = sf.read("input.wav")
print("Sample Rate:", sample_rate)

plt.plot(audio)
plt.title("Waveform")
plt.show()