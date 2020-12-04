import numpy as np
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    """
    Motion dataset. 
    Prepares conditioning information (previous poses + control signal) and the corresponding next poses"""

    def __init__(self, control_data, joint_data, past_context_len, future_context_len, dropout):
        """
        Args:
        control_data: The control input
        joint_data: body pose input 
        Both with shape (samples, time-slices, features)
        past_context_len: number of autoregressive body poses and previous control values
        future_context_len: number of future control-values
        dropout: (0-1) dropout probability for previous poses
        """
        self.control_data = control_data
        self.joint_data = joint_data
        self.past_context_len = past_context_len
        self.future_context_len = future_context_len
        self.dropout = dropout
        seqlen_control = past_context_len + future_context_len + 1
        self.control_seqlen = seqlen_control
        #For LSTM network
        n_frames = joint_data.shape[1]
                    
        # Joint positions for n previous frames
        ## autoreg = self.concat_sequence(self.past_context_len, joint_data[:,:n_frames-future_context_len-1,:])
                    
        # Control for n previous frames + current frame
        ## control = self.concat_sequence(seqlen_control, control_data)


        ## print("autoreg:" + str(autoreg.shape))        
        ## print("control:" + str(control.shape))        
        ## new_cond = np.concatenate((autoreg,control),axis=2)

        # Joint positions for the current frame
        self.x = joint_data[:, past_context_len:n_frames-future_context_len, :]
        ## self.cond = new_cond
        
        #TODO TEMP swap C and T axis to match existing implementation
        self.x = np.swapaxes(self.x, 1, 2)
        ## self.cond = np.swapaxes(self.cond, 1, 2)
        
        print("self.x:" + str(self.x.shape))        
        ## print("self.cond:" + str(self.cond.shape))
        self.cond_dim = self.get_conditioning(0).shape[0]

    def n_channels(self):
        return self.x.shape[1], self.cond_dim

    def get_conditioning(self, batch_idx):
        n_frames = self.joint_data.shape[1]
        
        autoreg = self.concat_sequence(
            seqlen = self.past_context_len, 
            data = self.joint_data[batch_idx,:n_frames-self.future_context_len-1,:][np.newaxis, :]
        )

        control = self.concat_sequence(
            seqlen = self.control_seqlen,
            data = self.control_data[batch_idx][np.newaxis, :]
        )

        conditioning = np.concatenate((autoreg, control), axis=2)
        #TODO TEMP swap C and T axis to match existing implementation
        conditioning = np.swapaxes(conditioning, 1, 2)
        return conditioning.squeeze(axis=0)

    def concat_sequence(self, seqlen, data):
        """ 
        Concatenates a sequence of features to one.
        """
        nn,n_timesteps,n_feats = data.shape
        L = n_timesteps-(seqlen-1)
        inds = np.zeros((L, seqlen)).astype(int)

        #create indices for the sequences we want
        rng = np.arange(0, n_timesteps)
        for ii in range(0,seqlen):  
            inds[:, ii] = np.transpose(rng[ii:(n_timesteps-(seqlen-ii-1))])  

        #slice each sample into L sequences and store as new samples 
        cc=data[:,inds,:].copy()
        
        #print ("cc: " + str(cc.shape))

        #reshape all timesteps and features into one dimention per sample
        dd = cc.reshape((nn, L, seqlen*n_feats))
        #print ("dd: " + str(dd.shape))
        return dd
                                                                                                                               
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Returns poses and conditioning.
        If data-dropout sould be applied, a random selection of the previous poses is masked.
        The control is not masked
        """
        
        if self.dropout>0.:
            n_feats, tt = self.x[idx,:,:].shape
            cond_masked = self.get_conditioning(idx)
            keep_pose = np.random.rand(self.past_context_len, tt)<(1-self.dropout)

            #print(keep_pose)
            n_cond = cond_masked.shape[0]-(n_feats*self.past_context_len)
            mask_cond = np.full((n_cond, tt), True)

            mask = np.repeat(keep_pose, n_feats, axis = 0)
            mask = np.concatenate((mask, mask_cond), axis=0)
            #print(mask)

            cond_masked = cond_masked*mask
            sample = {'x': self.x[idx,:,:], 'cond': cond_masked}
        else:
            sample = {'x': self.x[idx,:,:], 'cond': self.cond[idx,:,:]}
            
        return sample

class TestDataset(Dataset):
    """Test dataset."""

    def __init__(self, control_data, joint_data):
        """
        Args:
        control_data: The control input
        joint_data: body pose input 
        Both with shape (samples, time-slices, features)
        """        
        # Joint positions
        self.autoreg = joint_data

        # Control
        self.control = control_data
        
    def __len__(self):
        return self.autoreg.shape[0]

    def __getitem__(self, idx):
        sample = {'autoreg': self.autoreg[idx,:], 'control': self.control[idx,:]}
        return sample
