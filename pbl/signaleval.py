# Tanpura Drone Amplifier - signals & systems verification
# Needs tran.csv (V(vin1), V(vin2), V(vout)) and optionally ac.csv exported from KiCad.
# Run:  python signal_eval_tanpura.py   -> saves the graph PNG and 3 WAV audio files.
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# ---------------- settings ----------------
TRAN_CSV = "tran.csv"
AC_CSV = "ac.csv"
FS = 50_000            # uniform resample rate (Hz) = 1/20u step
SETTLE_TIME = 0.5      # s, ignore start-up growth of the oscillators for the FFT
SA_HZ, PA_HZ = 221.0, 332.0   # design frequencies from the schematic
OUT_PNG = "signals_system_validation.png"


# ---------------- helpers ----------------
def load_csv(path):
    """Read a KiCad/ngspice CSV export (any encoding/separator). Returns (header, 2-D array)."""
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = raw.decode(enc)
            if "\x00" not in text:
                break
        except UnicodeError:
            continue
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    first = lines[0]
    delim = max((",", ";", "\t"), key=first.count)
    if first.count(delim) == 0:
        delim = None                      # whitespace separated

    def split(ln):
        cells = ln.split(delim) if delim else ln.split()
        return [c.strip().strip('"').strip() for c in cells if c.strip() != ""]

    def num(c):
        try:
            return float(c)
        except ValueError:
            return float(c.replace(",", "."))   # decimal comma

    header, rows = None, []
    for ln in lines:
        cells = split(ln)
        try:
            rows.append([num(c) for c in cells])
        except ValueError:
            if not rows:
                header = cells            # last text line before the data = header
    if not rows:
        print(f"[!] Could not find numeric data in {path}. First lines of the file:")
        for ln in lines[:4]:
            print("    ", ln[:150])
        raise SystemExit(1)
    width = max(set(map(len, rows)), key=list(map(len, rows)).count)
    rows = [r for r in rows if len(r) == width]
    data = np.array(rows, dtype=float)
    if header is None or len(header) != data.shape[1]:
        header = [f"col{i}" for i in range(data.shape[1])]
    return header, data


def find_col(header, name):
    """Column index whose header contains `name` (e.g. 'vin1'), case-insensitive."""
    for i, h in enumerate(header):
        if name.lower() in h.lower():
            return i
    raise KeyError(f"No column containing '{name}' in {header}")


def spectrum_db(t, v, fs, t_start):
    """Hann-windowed magnitude spectrum in dB (0 dB = strongest peak)."""
    m = t >= t_start
    seg = v[m] - np.mean(v[m])
    w = np.hanning(len(seg))
    X = np.abs(rfft(seg * w))
    f = rfftfreq(len(seg), 1.0 / fs)
    return f, 20 * np.log10(X / np.max(X) + 1e-9)


def top_peaks(f, db, fmin=50, fmax=3000, n=3):
    m = (f >= fmin) & (f <= fmax)
    fs_, ds_ = f[m], db[m]
    idx = [i for i in range(1, len(ds_) - 1) if ds_[i] > ds_[i - 1] and ds_[i] >= ds_[i + 1]]
    idx = sorted(idx, key=lambda i: ds_[i], reverse=True)[:n]
    return sorted(fs_[i] for i in idx)


def save_wavs(t, signals, t_start, loop_seconds=6.0):
    """Write each simulated signal to a 16-bit WAV (settled part, looped to ~6 s
    so it is long enough to listen to). Each file is normalised to its own peak."""
    from scipy.io import wavfile
    m = t >= t_start
    for name, v in signals.items():
        seg = v[m] - np.mean(v[m])
        seg = seg / (np.max(np.abs(seg)) + 1e-12)
        reps = max(1, int(np.ceil(loop_seconds * FS / len(seg))))
        # fade the loop seam: trim to a whole number of samples, simple tiling
        audio = np.tile(seg, reps)
        wavfile.write(f"{name}.wav", FS, (audio * 0.8 * 32767).astype(np.int16))
        print(f"[+] Audio saved: {name}.wav  ({len(audio) / FS:.1f} s)")


# ---------------- main ----------------
def main(tran_csv=TRAN_CSV, ac_csv=AC_CSV):
    # ---- TRAN data ----
    hdr, d = load_csv(tran_csv)
    t_raw = d[:, 0]
    c1, c2, c3 = find_col(hdr, "vin1"), find_col(hdr, "vin2"), find_col(hdr, "vout")
    # ngspice time steps are not uniform -> resample onto a uniform grid for the FFT
    t = np.arange(t_raw[0], t_raw[-1], 1.0 / FS)
    vin1 = np.interp(t, t_raw, d[:, c1])
    vin2 = np.interp(t, t_raw, d[:, c2])
    vout = np.interp(t, t_raw, d[:, c3])

    settle = SETTLE_TIME if t[-1] > SETTLE_TIME + 0.2 else 0.0
    f1, S1 = spectrum_db(t, vin1, FS, settle)
    f2, S2 = spectrum_db(t, vin2, FS, settle)
    fo, SO = spectrum_db(t, vout, FS, settle)

    # ---- Audio export: the circuit's own signals as WAV files ----
    save_wavs(t, {"input_Sa_vin1": vin1, "input_Pa_vin2": vin2, "output_vout": vout}, settle)

    print("[+] Peak frequencies in V(vout) (Hz):", [round(p, 1) for p in top_peaks(fo, SO)])
    print(f"    design: Sa = {SA_HZ} Hz, Pa = {PA_HZ} Hz")
    print(f"    V(vin1) peak ~ {f1[np.argmax(S1)]:.1f} Hz, V(vin2) peak ~ {f2[np.argmax(S2)]:.1f} Hz")

    # ---- AC data (system transfer function |H(f)|) ----
    try:
        ah, a = load_csv(ac_csv)
        fa = a[:, 0]
        mag = a[:, 1]
        is_db = ("db" in ah[1].lower()) or (np.any(mag < 0) and np.max(np.abs(mag)) < 200)
        H_dB = mag if is_db else 20 * np.log10(np.abs(mag) + 1e-12)
        have_ac = True
    except Exception as e:
        print(f"[!] AC file not used ({e}); plotting only TRAN results.")
        have_ac = False

    # ---- Plot ----
    rows = 3 if have_ac else 2
    plt.figure(figsize=(11, 4 * rows))

    # 1) time domain: last ~5 cycles of the lower tone
    plt.subplot(rows, 1, 1)
    t_end = t[-1]
    win = (t >= t_end - 5.0 / SA_HZ) & (t <= t_end)
    plt.plot(t[win], vin1[win], label="V(vin1)  Sa oscillator", alpha=0.8)
    plt.plot(t[win], vin2[win], label="V(vin2)  Pa oscillator", alpha=0.8)
    plt.plot(t[win], vout[win], label="V(vout)  speaker output", alpha=0.9)
    plt.title("Time-domain: oscillator signals and output  x(t) -> y(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.grid(True)
    plt.legend(loc="upper right")

    # 2) spectra
    plt.subplot(rows, 1, 2)
    plt.semilogx(f1, S1, label="|Vin1(f)|", alpha=0.6)
    plt.semilogx(f2, S2, label="|Vin2(f)|", alpha=0.6)
    plt.semilogx(fo, SO, label="|Vout(f)|", alpha=0.9)
    for fx, nm in ((SA_HZ, "Sa 221 Hz"), (PA_HZ, "Pa 332 Hz")):
        plt.axvline(fx, color="k", ls="--", lw=0.8)
        plt.text(fx * 1.03, -5, nm, rotation=90, va="top", fontsize=8)
    plt.title("Frequency-domain (FFT): input tones vs output")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.xlim(20, 20000)
    plt.ylim(-100, 5)
    plt.grid(True, which="both")
    plt.legend()

    # 3) AC analysis
    if have_ac:
        plt.subplot(rows, 1, 3)
        plt.semilogx(fa, H_dB, label="V(vout) AC gain", color="tab:red")
        plt.axvline(6000, color="k", ls="--", lw=0.8)
        plt.text(6000 * 1.05, np.min(H_dB) + 3, "fc = 6 kHz", fontsize=8)
        plt.title("AC analysis: |H(f)| of mixer + low-pass + amplifier chain")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Gain [dB]")
        plt.grid(True, which="both")
        plt.legend()

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    print(f"[+] Plot saved: {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*(args[:2]))
