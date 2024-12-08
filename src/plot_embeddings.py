import h5py

# Open the HDF5 file in read mode
file_path = "embeddings.h5"
with h5py.File(file_path, "r") as h5_file:
    # List all groups and datasets in the file
    print("Keys:", list(h5_file.keys()))
    
    # Access a specific dataset
    dataset_name = "PigAirwayPressure_train"  # Replace with the actual dataset name
    data = h5_file[dataset_name][:]
    print("Data shape:", data.shape)
    print("Data:", data)
