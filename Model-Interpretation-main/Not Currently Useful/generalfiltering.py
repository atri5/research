import timeit
import pandas as pd
import os


t0 = timeit.default_timer()
# Load data


#Specify file name
file = '/Users/ayushtripathi/Lab Research/Model-Interpretation/HN_P2C_files/HN_P2c_20240105.csv'


checkpoints = 1
df = pd.read_csv(file)
t1 = timeit.default_timer()
print("Opened file in: "+ str(t1 - t0) + " seconds.")
print(df)

#shows first few dataplots in df
#print(df.info())

t1 = timeit.default_timer()
print("Opened file in: "+ str(t1 - t0) + " seconds.")

#removes In Vivo filtering 
filtered_data = df[df['ScanContext'] == 'In Vivo']

t1 = timeit.default_timer()
print("Checkpoint #1 at: "+str(t1-t0)+ " seconds.")


#drops columns 13-28
df.drop(df.columns[12:28], axis=1)

t1 = timeit.default_timer()
print("Checkpoint #2 at: "+str(t1-t0)+ " seconds.")

# #Remove certain patients (Case not found in the dataframe!)
# df = df[df['Case'] != 124]
# df = df[df['Case'] < 160]


#removes CH4 data
df = df[df['DataChannelsUsed'] != 'CH4']


#SNR Thresholding
SNR_Thr_V4 = 30
SNR_Thr_FB = 50

mask = (
    ((df['snr_ch1_1'] < SNR_Thr_V4) & (df['Instrument'] == 'V4')) |
    ((df['snr_ch2_1'] < SNR_Thr_V4) & (df['Instrument'] == 'V4')) |
    ((df['snr_ch3_1'] < SNR_Thr_V4) & (df['Instrument'] == 'V4'))
)

df = df[~mask]

mask2 = (
    ((df['snr_ch1_1'] < SNR_Thr_FB) & (df['Instrument'] == 'FLImBrush')) |
    ((df['snr_ch2_1'] < SNR_Thr_FB) & (df['Instrument'] == 'FLImBrush')) |
    ((df['snr_ch3_1'] < SNR_Thr_FB) & (df['Instrument'] == 'FLImBrush'))
)

df = df[~mask2]

t1 = timeit.default_timer()
print("Checkpoint #3 at: "+str(t1-t0)+ " seconds.")

# Remove rows with missing entries in specific columns (NaN values)
df = df.dropna(subset=['lifet_avg_ch1_1', 'lifet_avg_ch2_1', 'lifet_avg_ch3_1'])
df = df.dropna(subset=['int_ratio_ch1_1', 'int_ratio_ch2_1', 'int_ratio_ch3_1'])

#currently there is no Label column in the df
#df = df[df['Label'] != 0]

t1 = timeit.default_timer()
print("Checkpoint #4 at: "+str(t1-t0)+ " seconds.")

#Remove data with negative int values
df = df[~((df['lifet_avg_ch1_1'] < 0) | (df['lifet_avg_ch2_1'] < 0) | (df['lifet_avg_ch3_1'] < 0))]
df = df[~((df['spec_int_ch1_1'] < 0) | (df['spec_int_ch2_1'] < 0) | (df['spec_int_ch3_1'] < 0))]

t1 = timeit.default_timer()
print("Checkpoint #5 at: "+str(t1-t0)+ " seconds.")

# Normalizing intensity ratios for V4 to be comparable to FLImBrush

total_spec_int = df['spec_int_ch1_1'] + df['spec_int_ch2_1'] + df['spec_int_ch3_1']
IR_ch1 = 0.1877 * (df['spec_int_ch1_1'] / total_spec_int)
IR_ch2 = 0.8108 * (df['spec_int_ch2_1'] / total_spec_int)
IR_ch3 = 1.4116 * (df['spec_int_ch3_1'] / total_spec_int)

## Calculate Int_ratio_ch1, Int_ratio_ch2, and Int_ratio_ch3
total_IR = IR_ch1 + IR_ch2 + IR_ch3
Int_ratio_ch1 = IR_ch1 / total_IR
Int_ratio_ch2 = IR_ch2 / total_IR
Int_ratio_ch3 = IR_ch3 / total_IR

condition = df['Instrument'] == 'V4'
df.loc[condition, 'int_ratio_ch1_1'] = Int_ratio_ch1[condition]
df.loc[condition, 'int_ratio_ch2_1'] = Int_ratio_ch2[condition]
df.loc[condition, 'int_ratio_ch3_1'] = Int_ratio_ch3[condition]

t1 = timeit.default_timer()
print("Checkpoint #6 at: "+str(t1-t0)+ " seconds.")

print("Saving to file...")
# Gets dir and saves to local folder 
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(script_dir, 'modified_' + file)
df.to_csv(output_file_path, index=False)


# #deletes old csv(comment out if not needed)
# file_to_delete = os.path.join(script_dir, file)
# if os.path.exists(file_to_delete):

t1 = timeit.default_timer()
print("Saved to " + output_file_path + ". Finished in " + str(t1-t0) + " seconds.")
