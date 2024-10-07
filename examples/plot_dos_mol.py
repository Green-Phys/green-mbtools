import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
from scipy.signal import find_peaks

# Constants
AU2EV = 27.21139

# Set numpy print options
np.set_printoptions(suppress=True, precision=6, linewidth=400)

# Argument parser
parser = argparse.ArgumentParser(
    description=(
        "Plot density of states (DOS) for a molecular system using RHF reference. "
        "Requires the output of Nevanlinna. For UHF references, the alpha DOS will be plotted instead."
    )
parser.add_argument(
    "--ac_out", type=str, default="ac_out.h5",
    help="Output of UGF2 code, i.e., sim.h5"
)
args = parser.parse_args()

# Load data from HDF5 file
with h5py.File(args.ac_out, 'r') as file:
    it = file["iter"][()]
    freqs = file['/nevanlinna/freqs'][()]
    dos = file['/nevanlinna/dos'][()]
    mu = file[f'/iter{it}/mu'][()]
    nel_cell = file["/HF/nelec"][()]
    print(f"Number of electrons in the cell: {nel_cell}")
    print("WARNING: RHF nocc used")
    homo = nel_cell // 2 - 1

# Convert frequencies and density of states
freqs_ev = freqs * AU2EV
dos_positive = dos / AU2EV

# Get Ionization Potentials (IP)
n_soln_ip = 4
print(f"\nLowest {n_soln_ip} Ionization Potentials:")
occ_ip = []

for i in range(n_soln_ip):
    try:
        peaks_scgw, _ = find_peaks(dos_positive[:, 0, 0, homo - i], 0.1)
        ip_values = [freqs_ev[peak_index] + mu * AU2EV for peak_index in peaks_scgw]
        if ip_values:
            print(f"GW IP {i + 1}: {ip_values[0]:8.3f} eV")
            occ_ip.append(ip_values[0])  # Store the first peak value
    except Exception:
        continue  # Ignore errors silently

# Get Electron Affinities (EA)
n_soln_ea = 4
print(f"\nLowest {n_soln_ea} Electron Affinities:")
occ_ea = []

for i in range(n_soln_ea):
    try:
        peaks_scgw, _ = find_peaks(dos_positive[:, 0, 0, homo + 1 + i], 0.1)
        ea_values = [freqs_ev[peak_index] + mu * AU2EV for peak_index in peaks_scgw]
        if ea_values:
            print(f"GW EA {i + 1}: {ea_values[0]:8.3f} eV")
            occ_ea.append(ea_values[-1])  # Store the final peak
    except Exception:
        continue  # Ignore errors silently

# Total DOS
total_dos = np.einsum('tska->tsk', dos_positive)

# Save results to HDF5 file
with h5py.File('Aw.h5', 'w') as file:
    file['freqs'] = freqs_ev + mu * AU2EV
    file['dos'] = total_dos[:, 0, 0]
    file['qp_mo'] = occ_ip + occ_ea  # Combining IP and EA for output


# Plotting
plt.figure(figsize=(8, 3))
plt.rcParams.update({'font.size': 16})
plt.plot(
    freqs_ev + mu * AU2EV, total_dos[:, 0, 0], label='scGW',
    linestyle='-', color='tab:red', alpha=1, linewidth=2.0
)
#plt.minorticks_on()
plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")

plt.xlabel('eV')
plt.ylabel(r'A($\omega$')
plt.axis(xmin=-20, xmax=20, ymax=4, ymin=-0.1)
plt.legend()
plt.savefig("spectra.pdf", format="pdf", bbox_inches="tight")

