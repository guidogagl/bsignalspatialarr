import numpy as np
#from loguru import logger

def build_10_20_matrix_19_electrodes_( values : np.ndarray = np.ones(19) ) -> np.ndarray:
    assert len(values.shape) == 1, "Values dimension mismatch"
    assert len(values) == 19, "Values dimension mismatch, recquired dimension is 19, found %d" % len(values)    
    
    matrix = np.zeros((7, 9))

    matrix[0, 3] = values[0]
    matrix[0, 5] = values[1]

    matrix[2, 1] = values[2]
    matrix[2, 3] = values[3]
    matrix[2, 4] = values[4]
    matrix[2, 5] = values[5]
    matrix[2, 7] = values[6] 

    matrix[3, 0] = values[7]
    matrix[3, 2] = values[8]
    matrix[3, 4] = values[9]
    matrix[3, 6] = values[10]
    matrix[3, 8] = values[11]

    matrix[4, 1] = values[12]
    matrix[4, 3] = values[13]
    matrix[4, 4] = values[14]
    matrix[4, 5] = values[15]
    matrix[4, 7] = values[16] 

    matrix[6, 3] = values[17]
    matrix[6, 5] = values[18]

    return matrix


def build_10_20_matrix_32_electrodes_( values : np.ndarray = np.ones(32) ) -> np.ndarray:
    assert len(values.shape) == 1, "Values dimension mismatch"
    assert len(values) == 32, "Values dimension mismatch, recquired dimension is 32, found %d" % len(values)    
    
    matrix = np.zeros((11, 9))

    # First row
    matrix[0,3] = values[0]
    matrix[0,5] = values[1]

    # Third row
    matrix[2,3] = values[2]
    matrix[2,5] = values[3]

    # Fourth row
    matrix[3,0] = values[4]
    matrix[3,2] = values[5]
    matrix[3,4] = values[6]
    matrix[3,6] = values[7]
    matrix[3,8] = values[8]

    # Fifth row
    matrix[4,1] = values[9]
    matrix[4,3] = values[10]
    matrix[4,5] = values[11]
    matrix[4,7] = values[12]

    # Sixth row
    matrix[5,0] = values[13]
    matrix[5,2] = values[14]
    matrix[5,4] = values[15]
    matrix[5,6] = values[16]
    matrix[5,8] = values[17]

    # Seventh row
    matrix[6,1] = values[18]
    matrix[6,3] = values[19]
    matrix[6,5] = values[20]
    matrix[6,7] = values[21]

    # Eighth row
    matrix[7,0] = values[22]
    matrix[7,2] = values[23]
    matrix[7,4] = values[24]
    matrix[7,6] = values[25]
    matrix[7,8] = values[26]

    # Ninth row
    matrix[8,3] = values[27]
    matrix[8,5] = values[28]

    # Eleventh row
    matrix[10,3] = values[29]
    matrix[10,4] = values[30]
    matrix[10,5] = values[31]

    return matrix

# number of supported electrodes = 19, 32
class BSignalSpatialArr:
    def __init__(self, v_pad : int = 3, o_pad : int = 3):
        self.v_pad = v_pad
        self.o_pad = o_pad
        
        return

    def build_feature_matrix_row_(self, features_by_bands : np.ndarray = np.ones((4, 32)))  -> np.ndarray:
        # construct one row of the multifeatures matrix structure
        # the input should consist in bands x num_electrodes
        assert len(features_by_bands.shape) == 2, "Features values dimension mismatch"
        
        num_electrodes, num_bands = features_by_bands.shape[1], features_by_bands.shape[0]        

        assert (num_electrodes == 19) or (num_electrodes == 32), "Supported number of electrodes are 19 and 32"
        
        for i in range(num_bands):
            if num_electrodes == 19:       
                tmp_matrix = build_10_20_matrix_19_electrodes_( features_by_bands[i] )
                space = np.zeros((7, self.o_pad)) 
            else:
                tmp_matrix = build_10_20_matrix_32_electrodes_( features_by_bands[i] )
                space = np.zeros((11, self.o_pad))    
            
            if i == 0:
                matrix = tmp_matrix
            else:
                matrix = np.concatenate( (matrix,  space, tmp_matrix), axis = 1)

        
        return matrix
 
    def build_multifeatures_matrix_rearrangement_( self, features_values : np.ndarray = np.ones((4, 4, 32))) -> np.ndarray:
        # the input should consist in num_extracted_features x bands x num_electrodes
        assert len(features_values.shape) == 3, "Features values dimension mismatch"
        
        num_electrodes, num_bands, num_features = features_values.shape[2], features_values.shape[1], features_values.shape[0]        
        assert (num_electrodes == 19) or (num_electrodes == 32), "Supported number of electrodes are 19 and 32"

        for i in range(num_features):
            tmp_matrix = self.build_feature_matrix_row_( features_values[i] )
            space = np.zeros((self.v_pad, tmp_matrix.shape[1])) 
            
            if i == 0:
                matrix = tmp_matrix
            else:
                matrix = np.concatenate( (matrix, space, tmp_matrix), axis = 0) 

        return matrix
    

    def build_single_feature_matrix_rearrangement_( self, 
                                                    features_values : np.ndarray = np.ones((1, 4, 32)),
                                                    cols : int = 2,
                                                    rows : int = 2 ) -> np.ndarray:
        assert len(features_values.shape) == 3, "Features values dimension mismatch"
        
        num_electrodes, num_bands, num_features = features_values.shape[2], features_values.shape[1], features_values.shape[0]        
        assert (num_electrodes == 19) or (num_electrodes == 32), "Supported number of electrodes are 19 and 32"
        assert num_features == 1, "For single features rearrangement number of features should be: 1"
        assert num_bands == 4, "For single features rearrangement number of supported bands: 4"
        
        matrix = np.reshape(features_values, (rows, cols, num_electrodes) )
        matrix = self.build_multifeatures_matrix_rearrangement_( matrix )

        return matrix
    
def rearrange_features(values : np.ndarray, v_pad : int = 3, o_pad : int = 3, mode = "multi", cols : int = 2, rows : int = 2 ) -> np.ndarray:
    
    assert len(values.shape) == 4, "Features values dimension mismatch"
    assert mode == "multi" or mode == "mono", "Mode should be either multi or mono"
    num_samples, num_features, num_bands, num_electrodes  = values.shape[0], values.shape[1], values.shape[2], values.shape[3]        

    assert (num_electrodes == 19) or (num_electrodes == 32), "Supported number of electrodes are 19 and 32, found %d" % num_electrodes
    
    bssa = BSignalSpatialArr(v_pad, o_pad)
    
    if mode == "mono":
        assert num_features == 1, "For single features rearrangement number of features should be: 1"
        assert num_bands == 4, "For single features rearrangement number of supported bands: 4"

        tmp = bssa.build_single_feature_matrix_rearrangement_(np.ones((num_features, num_bands, num_electrodes)), cols, rows )

    else:
        tmp = bssa.build_multifeatures_matrix_rearrangement_( np.ones((num_features, num_bands, num_electrodes)))
    
    values_matrix = np.zeros((num_samples, tmp.shape[0], tmp.shape[1]))
    
    for i in range( num_samples ):
        if mode == "mono":
            values_matrix[i] = bssa.build_single_feature_matrix_rearrangement_( values[i], cols, rows )
        else:
            values_matrix[i] = bssa.build_multifeatures_matrix_rearrangement_( values[i] )
            
    return values_matrix