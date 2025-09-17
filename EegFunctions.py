import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch
from autoreject import AutoReject
from mne.viz import plot_epochs as plot_mne_epochs
from scipy.io import loadmat

# from scipy.stats import zscore
import os
import seaborn as sns

def create_mne_raw(path, unit_conversion=1e-6,sampling_rate=250):
    """Convert DataFrame to MNE Raw object"""
    print("=== CREATING MNE OBJECT ===")
    
    extension = path.split('.')[-1]
    df = None
    if extension =='csv':
        df = pd.read_csv(path)
        CHANNELS = df.columns[1:-1].to_list()
        eeg_data = df[CHANNELS].values.T * unit_conversion 
        print(CHANNELS)
            # Create MNE info object
        info = mne.create_info(
            ch_names=CHANNELS, 
            sfreq=sampling_rate, 
            ch_types=['eeg'] * len(CHANNELS)
        )
        
        # Create Raw object
        raw = mne.io.RawArray(eeg_data, info)
    elif extension in ('set','fdt'):
        if _is_epoched_set_file(path):
            print("Detected epoched data - using mne.io.read_epochs_eeglab()")
            epochs = mne.io.read_epochs_eeglab(path, verbose=False)
            return epochs,df
        else:
            raw = mne.io.read_raw_eeglab(path, preload=True)
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
    except:
        print("Warning: Could not set electrode positions")
    return raw,df

def _is_epoched_set_file(path):
    
    try:
       
        set_data = loadmat(path, struct_as_record=False, squeeze_me=True)
        if 'EEG' in set_data:
            eeg = set_data['EEG']
        else:
            eeg = set_data
            
        # Check if trials field exists and is greater than 1
        if hasattr(eeg, 'trials'):
            trials = eeg.trials
            print(f"Number of trials detected: {trials}")
            return trials > 1
        elif 'trials' in eeg:
            trials = eeg['trials']
            print(f"Number of trials detected: {trials}")
            return trials > 1
        else:
            print("No trials field found - assuming continuous data")
            return False
            
    except Exception as e:
        print(f"Error reading .set file structure: {e}")
        print("Defaulting to continuous data reader")
        return False

def extract_events(df=None,raw=None):
    """Extract stimulus events from DataFrame"""
    print("=== EXTRACTING EVENTS ===")
    if df is not None and df.shape:
        # Find stimulus events
        
        events_from_stim = (df['stim'] != 0.0).values
        event_samples = np.where(events_from_stim)[0]
        event_ids = df['stim'][events_from_stim].values.astype(int)
        events = np.vstack([event_samples, np.zeros_like(event_samples), event_ids]).T
        
        print(f"Found {len(events)} total events")
        print(f"Event types: {np.unique(event_ids)}")
        
        # Count events by type
        stimulus_counts = df['stim'].value_counts()
        print(f"Standard trials (stim=1): {stimulus_counts.get(1.0, 0)}")
        print(f"Deviant trials (stim=2): {stimulus_counts.get(2.0, 0)}")
    elif raw:
        if raw.annotations:
            print(f"\nFound {len(raw.annotations)} annotations:")
            unique_descriptions = set(raw.annotations.description)
            stimulus_counts = 0
            for desc in unique_descriptions:
                count = sum(1 for d in raw.annotations.description if d == desc)
                stimulus_counts+=count
                print(f"  {desc}: {count} occurrences")
            
            # Convert annotations to events
            events, event_ids = mne.events_from_annotations(raw)
        else:
            try:
                events = mne.find_events(raw)
                unique_event_ids = np.unique(events[:, 2])
                event_ids = {f'event_{i}': int(eid) for i, eid in enumerate(unique_event_ids)}
                print(f"Found events: {event_ids}")
            except:
                print("No events found!")
    else:
        raise ValueError("Either df or raw must be provided")
    
    return events,event_ids, stimulus_counts

def create_epochs(raw, events,Config):
    """Create epochs from raw data and events"""
    print("=== CREATING EPOCHS ===")
    
    epochs = mne.Epochs(
        raw, events, 
        event_id=Config.EVENT_DICT, 
        tmin=Config.EPOCH_TMIN, 
        tmax=Config.EPOCH_TMAX,
        baseline=Config.BASELINE, 
        preload=True, 
        picks='all'
        # reject={'eeg':0.0001}  # We'll do manual artifact rejection
    )
    
    # Print counts for all event types dynamically
    for event_type in epochs.event_id.keys():
        print(f"{event_type.capitalize()}: {len(epochs[event_type])}")

    # Or in one line:
    event_counts = [f"{event_type.capitalize()}: {len(epochs[event_type])}" for event_type in epochs.event_id.keys()]
    print(f"Created epochs - {', '.join(event_counts)}")
    return epochs

def create_bins(events,event_codes=None):
    """Create bins based on stimulus sequence"""
    bin_1_indices = [] 
    bin_2_indices = []
    if event_codes is None:
        
        for i, event in enumerate(events[1:], 1):  # Skip first event
            current_event = event[2]
            previous_event = events[i-1][2]
            
            # Bin 1: Current is deviant (70), previous is standard (80)
            if current_event == 3 and previous_event == 4:
                bin_1_indices.append(i-1)  # Adjust for epoch indexing
                
            # Bin 2: Current is standard (80), previous is standard (80)
            elif current_event == 4 and previous_event == 4:
                bin_2_indices.append(i-1)  # Adjust for epoch indexing
    else:
        print("Using provided bin indices")
        condition_1 = event_codes.get('condition_1', [])
        condition_2 = event_codes.get('condition_2', [])
        for i, event in enumerate(events):
            if event[2] in condition_1:
                bin_1_indices.append(i)
            elif event[2] in condition_2:
                bin_2_indices.append(i)
        
    return bin_1_indices, bin_2_indices

def compute_sme_erplab(epoch_list, time_window):
    """Compute SME per bin like ERPLAB"""
    bins = {f'Bin_{i}': v for i, v in enumerate(epoch_list, 1)}
    sme_results = {}
    
    for bin_name, epochs in bins.items():
        epochs_cropped = epochs.copy().crop(tmin=time_window[0], tmax=time_window[1])
        data = epochs_cropped.get_data()  # (n_epochs, n_channels, n_times)
        trial_means = np.mean(data, axis=2)  
        sme_vals = np.std(trial_means, axis=0) / np.sqrt(trial_means.shape[0])
        
        sme_results[bin_name] = {
            'sme_values': sme_vals,
            'channel_names': epochs.ch_names,
            'n_epochs': len(epochs)
        }
    
    return sme_results

def moving_window_step_artifact_detection(epochs, channel_name="VEOG-lower",window_size=0.2,
                                           step_size=0.01, threshold=100,
                                           tmin=None, tmax=None ):
    sfreq = epochs.info['sfreq']
    win_samp = int(window_size * sfreq)
    step_samp = int(step_size * sfreq)

    # Pick an EOG channel (e.g., VEOG-lower)
    eog_data = epochs.copy().pick([channel_name]).get_data() * 1e6  # µV

    if tmin is not None and tmax is not None:
        times = epochs.times  # relative time in seconds
        
        # Indices for test period
        mask = (times >= tmin) & (times <= tmax)
        test_indices = np.where(mask)[0]
    else:
        test_indices [0,epochs[0].shape[1]]

    n_epochs = eog_data.shape[0]
    bad_epochs = [None] * n_epochs

    for i, epoch in enumerate(eog_data):
        max_diff = 0
        for start in range(test_indices[0], test_indices[-1] - win_samp, step_samp):
            half = win_samp // 2
            mean1 = epoch[0, start:start+half].mean()
            mean2 = epoch[0, start+half:start+win_samp].mean()
            diff = abs(mean2 - mean1)
            max_diff = max(max_diff, diff)
        if max_diff > threshold:
            bad_epochs[i] = i 
    return bad_epochs

def plot_bad_epochs(epochs_interpolated,bad_epochs,channel_name,n_epochs_to_plot,title="Blink Artifact Detection"):
    epoch_colors = []

    n_channels = len(epochs_interpolated.info['ch_names'])

    for epoch_idx in range(len(epochs_interpolated)):
        epoch_colors.append(['k'] * n_channels)
        if bad_epochs[epoch_idx]:
            epoch_colors[epoch_idx][epochs_interpolated.ch_names.index(channel_name)] = 'r'
            
    plot_mne_epochs(
        epochs=epochs_interpolated,
        picks='eog',
        epoch_colors=epoch_colors,
        scalings={  'eeg': 100e-6, 'eog': 100e-6},
        title=title,
        n_epochs = n_epochs_to_plot
    )