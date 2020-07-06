import numpy as np 

def get_bounds(input_data):
    """
    Return minimum and maximum bounds of each feature/dimension in an array. 

    Args:
        input_data (ndarray): an array where each row is an event or observation.
    
    Returns:
        bounds (list): a list of bounrds for each dimension
    """
    bounds = []
    for i in input_data.T:
        i_min = np.amin(i)
        i_max = np.amax(i)
        bounds.append([i_min, i_max])
    return bounds

def clean_data(textfile, outfile):
    """
    Cleans sample data acquired from the Franti & Sieranoja datasets and saves to csv. 

    Args:
        textfile (str): location of a txt file containing clustering benchmark dataset. 
        outfile (str): location to save output csv 

    Returns: 
        None
    """
    with open(textfile, 'r') as f:
        lines_list = [line[5:] for line in f.readlines()]
        lines_list = [line[:-1] for line in lines_list]
        lines_list = [line.replace('     ', ',') for line in lines_list]

    with open(outfile, 'w') as f_out:
        for line in lines_list:
            f_out.write(f"{line}\n")