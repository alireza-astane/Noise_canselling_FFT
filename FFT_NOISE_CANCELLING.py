import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from pydub import AudioSegment
import numpy as np
from scipy.fft import fft, ifft

# Import the .wav audio
f = "file.wav"
s, a = wavfile.read(f)
print("Sampling Rate:", s)
print("Audio Shape:", np.shape(a))


# comapring two channels
# number of samples
na = a.shape[0]
# audio time duration
la = na / s

# plot signal versus time
t = np.linspace(0, la, na)
plt.subplot(2, 1, 1)
plt.title("two channel audio file")
plt.plot(t, a[:, 0], "b-")
plt.ylabel("Left")
plt.subplot(2, 1, 2)
plt.plot(t, a[:, 1], "r-")
plt.ylabel("Right")
plt.xlabel("Time (s)")

plt.savefig("w1.png")
plt.show()

# converting into mono channel audio file
sound = AudioSegment.from_wav(f)
sound = sound.set_channels(1)
fm = f[:-4] + "_mono.wav"
sound.export(fm, format="wav")

# Import the mono channel .wav audio
s, a = wavfile.read(fm)
print("Sampling Rate:", s)
print("Audio Shape:", np.shape(a))


# plotting the mono channel audio
na = a.shape[0]
la = na / s
t = np.linspace(0, la, na)
plt.title("mono channel audio file")
plt.plot(t, a, "k-", color="purple")
plt.xlabel("Time (s)")
plt.savefig("w2.png")
plt.show()

# analyze entire audio clip using FFT
na = len(a)
a_k = np.fft.fft(a)[0 : int(na / 2)] / na  # FFT function from numpy
a_k[1:] = 2 * a_k[1:]  # single-sided spectrum only
Pxx = np.abs(a_k)  # remove imaginary part
f = s * np.arange((na / 2)) / na  # frequency vector

# plotting
fig, ax = plt.subplots()
plt.title("Audio File Fourier Transform")
plt.plot(f, Pxx, "b-", label="Audio Signal")
plt.plot(
    [20, 20000], [0.1, 0.1], "r-", alpha=0.7, linewidth=10, label="Audible (Humans)"
)
ax.set_xscale("log")
ax.set_yscale("log")
plt.grid()
plt.legend()
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")
plt.savefig("w3.png")
plt.show()

# frequency vs time spectrom
fr, tm, spgram = signal.spectrogram(a, s)
lspg = np.log(spgram)
plt.pcolormesh(tm, fr, lspg, shading="auto")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (sec)")
plt.show()


# Generate random noise
noise = 200 * np.random.uniform(-1, 1, len(a))

# Add noise to the audio signal
noisy_audio2 = a + noise

# Save the noisy audio as a new file
noisy_fm = fm[:-4] + "_noisy5.wav"
wavfile.write(noisy_fm, s, noisy_audio)


plt.title("noisy audio file")
plt.plot(t, noisy_audio, "k-", color="purple")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.savefig("w4.png")
plt.show()


plt.title("Noise reduced audio file")
plt.plot(t, noisy_audio, "r-", label="Noisy Audio")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig("w9.png")
plt.show()

# comparing FFT of mono file and noisier file
# na_noisy = len(noisy_audio)
a_k_noisy = np.fft.fft(noisy_audio)[0 : int(na_noisy / 2)] / na_noisy
a_k_noisy[1:] = 2 * a_k_noisy[1:]
Pxx_noisy = np.abs(a_k_noisy)
f_noisy = s * np.arange((na_noisy / 2)) / na_noisy

# plotting
fig, ax = plt.subplots()
plt.plot(f_noisy, Pxx_noisy, "b-", label="Noisy Audio Signal")
plt.plot(
    [20, 20000], [0.1, 0.1], "r-", alpha=0.7, linewidth=10, label="Audible (Humans)"
)
ax.set_xscale("log")
ax.set_yscale("log")
plt.grid()
plt.legend()
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")

plt.show()
na = len(a)
a_k = np.fft.fft(a)[0 : int(na / 2)] / na  # FFT function from numpy
a_k[1:] = 2 * a_k[1:]  # single-sided spectrum only
Pxx = np.abs(a_k)  # remove imaginary part
f = s * np.arange((na / 2)) / na  # frequency vector

# plotting
fig, ax = plt.subplots()
plt.plot(f, Pxx, "b-", label="Audio Signal")
plt.plot(
    [20, 20000], [0.1, 0.1], "r-", alpha=0.7, linewidth=10, label="Audible (Humans)"
)
ax.set_xscale("log")
ax.set_yscale("log")
plt.grid()
plt.legend()
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")

plt.show()

# Create a single ax
fig, ax = plt.subplots(figsize=(8, 8))

# Plotting the first subplot
ax.plot(f_noisy, Pxx_noisy, "b-")
ax.plot([20, 20000], [0.1, 0.1], "r-", alpha=0.7, linewidth=10)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid()
ax.set_ylabel("Amplitude")
ax.set_xlabel("Frequency (Hz)")

# Plotting the second subplot
ax.plot(f, Pxx, "g-")  # Change the color to green
ax.legend(["Noisy Audio Signal", "Audible (Humans)", "Audio Signal"])  # Add legend
plt.show()


# frequency vs time spectrom for noisy audio
fr, tm, spgram = signal.spectrogram(noisy_audio, s)
lspg = np.log(spgram)
plt.pcolormesh(tm, fr, lspg, shading="auto")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (sec)")
plt.show()


# noise reduction algorithem using filtering frequencies and Windowing
def noise_reduction(signal, window_size, threshold):
    # Apply Fourier transform to the signal
    spectrum = fft(signal)

    # Normalize the spectrum
    normalized_spectrum = spectrum / len(spectrum)

    # Apply sliding window frequency threshold noise filtering
    filtered_spectrum = np.copy(normalized_spectrum)
    for i in range(window_size, len(filtered_spectrum) - window_size):
        window = filtered_spectrum[i - window_size : i + window_size]
        window_mean = np.mean(window)
        if window_mean < threshold:
            filtered_spectrum[i] = 0

    # Apply inverse Fourier transform to get the denoised signal
    denoised_signal = ifft(len(spectrum) * filtered_spectrum)

    return np.real(denoised_signal)


# Read the audio file
s, audio = wavfile.read(fm)

# Create the time array
t = np.linspace(0, len(audio) / s, len(audio))

# Plot the audio
plt.plot(t, audio, "b-")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# Set the parameters for noise reduction
window_size = 1000
threshold = 1000

# Perform noise reduction
denoised_audio = noise_reduction(noisy_audio, window_size, threshold)

# Save the denoised audio as a new file
denoised_file = "denoised_audio.wav"
wavfile.write(denoised_file, s, 4 * denoised_audio)


t = np.linspace(0, len(denoised_audio) / s, len(denoised_audio))
plt.plot(t, denoised_audio, "b-")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()


plt.title("comparing audio files")

plt.plot(t, audio / 4, "g-", label="Noise Reduced Audio")

# Plotting the noisy audio
plt.plot(t, noisy_audio, "r-", label="Noisy Audio")

# Plotting the mono file
plt.plot(t, a, "b-", label="Mono Audio")

# Plotting the denoised file


plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.savefig("w9.png")
plt.show()


# Compute the FFT of the mono audio
mono_fft = np.fft.fft(a)
mono_fft = mono_fft[: len(mono_fft) // 2]

# Compute the FFT of the noisy audio
noisy_fft = np.fft.fft(noisy_audio)
noisy_fft = np.abs(noisy_fft[: len(noisy_fft) // 2])

# Compute the FFT of the denoised audio
denoised_fft = np.fft.fft(audio)
denoised_fft = np.abs(denoised_fft[: len(denoised_fft) // 2])

# Plotting

plt.title("Fourier Transform Comparison: Noise reduced vs Mono Channel")
plt.plot(f, denoised_fft / 5, "g-", label="Noise reduced Audio")
plt.plot(f, mono_fft, "b-", label="Mono Audio")
# plt.plot(f, noisy_fft, 'r-', label='Noisy Audio')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig("w10.png")
plt.show()


# Compute the FFT of the mono audio
mono_fft = np.fft.fft(a)
mono_fft = np.abs(mono_fft[: len(mono_fft) // 2])


# Compute the FFT of the denoised audio
denoised_fft = np.fft.fft(noisy_audio2)
denoised_fft = np.abs(denoised_fft[: len(noisy_audio2) // 2])

# Plotting
plt.plot(f, denoised_fft, "g-", label="Denoised Audio")
plt.plot(f, mono_fft, "b-", label="Mono Audio")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Compute the FFT of the mono audio
mono_fft = np.fft.fft(a)
mono_fft = np.abs(mono_fft[: len(mono_fft) // 2])

# Compute the FFT of the noisy audio
noisy_fft = np.fft.fft(noisy_audio)
noisy_fft = np.abs(noisy_fft[: len(noisy_fft) // 2])

# Plotting
plt.plot(f, noisy_fft, "r-", label="Noisy Audio")
plt.plot(f, mono_fft, "b-", label="Mono Audio")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


# filtering based on the frequencies
def filter_audio(audio, lower_threshold, upper_threshold):
    # Perform FFT on the audio array
    spectrum = fft(audio)

    # Get the frequency vector
    filtered_spectrum = np.copy(spectrum)
    filtered_spectrum[:lower_threshold] = 0
    filtered_spectrum[upper_threshold:] = 0

    # Perform inverse FFT to get the filtered audio array
    filtered_audio = ifft(filtered_spectrum)

    # Return the real part of the filtered audio array
    return np.real(filtered_audio)


# Set the file path for the filtered audio
filtered_file = "filtered_audio.wav"

# Save the filtered audio as a new file

# Set the lower and upper frequency thresholds
lower_threshold = 1
upper_threshold = 10000

# Apply the filter to the noisy audio
filtered_audio = filter_audio(noisy_audio, lower_threshold, upper_threshold)
wavfile.write(filtered_file, s, filtered_audio)
# Plot the filtered audio
t = np.linspace(0, len(filtered_audio) / s, len(filtered_audio))
plt.plot(t, filtered_audio, "b-")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# Compute the FFT of the mono audio
mono_fft = np.fft.fft(a)
mono_fft = np.abs(mono_fft[: len(mono_fft) // 2])

# Compute the FFT of the noisy audio
noisy_fft = np.fft.fft(noisy_audio)
noisy_fft = np.abs(noisy_fft[: len(noisy_fft) // 2])

# Compute the FFT of the filtered audio
filtered_fft = np.fft.fft(filtered_audio)
filtered_fft = np.abs(filtered_fft[: len(filtered_fft) // 2])

# Plotting

plt.plot(f, mono_fft, "b-", label="Mono Audio")
plt.plot(f, filtered_fft, "g-", label="Filtered Audio")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


# Compute the FFT of the mono audio
mono_fft = np.fft.fft(a)
mono_fft = np.abs(mono_fft[: len(mono_fft) // 2])

# Compute the FFT of the noisy audio
noisy_fft = np.fft.fft(noisy_audio)
noisy_fft = np.abs(noisy_fft[: len(noisy_fft) // 2])

# Compute the FFT of the filtered audio
filtered_fft = np.fft.fft(filtered_audio)
filtered_fft = np.abs(filtered_fft[: len(filtered_fft) // 2])

# Plotting
plt.plot(f, noisy_fft, "r-", label="Noisy Audio")
plt.plot(f, mono_fft, "b-", label="Mono Audio")
plt.plot(f, filtered_fft, "g-", label="Filtered Audio")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


na_noisy = len(noisy_audio)
a_k_noisy = np.fft.fft(noisy_audio)[0 : int(na_noisy / 2)] / na_noisy
a_k_noisy[1:] = 2 * a_k_noisy[1:]
Pxx_noisy = np.abs(a_k_noisy)
f_noisy = s * np.arange((na_noisy / 2)) / na_noisy


na_nr = len(audio)
a_k_nr = np.fft.fft(audio)[0 : int(na_nr / 2)] / na_nr
a_k_nr[1:] = 2 * a_k_nr[1:]
Pxx_nr = np.abs(a_k_nr)
f_nr = s * np.arange((na_nr / 2)) / na_nr


# Create a single ax
fig, ax = plt.subplots(figsize=(8, 8))

# Plotting the first subplot
# ax.plot(f_noisy, Pxx_noisy, 'b-',label="Noisy Audio")
ax.plot(f_nr, Pxx_nr, "r-", label="Noised Reduced Audio")
# ax.plot(f, Pxx, 'g-',label="Mono Channel Audio")  # Change the color to green

ax.set_xscale("log")
ax.set_yscale("log")
ax.grid()
ax.set_ylabel("Amplitude")
ax.set_xlabel("Frequency (Hz)")

# Plotting the second subplot
ax.legend()  # Add legend


plt.title("Noise Reduced Audio Fourier Transform ")
plt.savefig("w16.png")
plt.show()
