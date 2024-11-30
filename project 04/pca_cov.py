'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
Ghailan Fadah
CS 251 Data Analysis Visualization
Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset.

        (No changes should be needed)
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

        self.mean = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''

        Ac = data - data.mean( axis = 0)

        cov_matrix = 1/ (data.shape[0] -1) * Ac.transpose() @ Ac

        return cov_matrix

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        list = []

        sum = e_vals.sum()

        for val in e_vals:
            c = val / sum
            list.append(c)

     
        return list


        pass

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''

        list = []
        c= 0
        for i in prop_var:
            c += i
            list.append(c)

        return list


    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
            - Before normalizing, create instance variables containing information that would be
            needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''

        self.vars = vars

        df = self.data[vars]

        self.A = df.to_numpy()
        self.mean = self.A.mean(axis = 0)

        Ah = np.hstack((self.A, np.ones((self.A.shape[0], 1)))).squeeze()

        self.l_min = np.min(self.A, axis=0)
        self.l_max = np.max(self.A, axis=0)

        if normalize:
            self.normalized = True
            list_min = self.l_min.tolist()
            list_max = self.l_max.tolist()

            scale_list = []


            for i in range(len(list_max)):
                scale = 1 / (list_max[i] - list_min[i])
                scale_list.append(scale)

            scale_list.append(1)

   

            for i in range(len(list_min)):
                list_min[i] *= -1

          

            t_matrix = np.eye(len(list_min) + 1)


            for i in range(len(list_min)):
                t_matrix[i, -1] = list_min[i]

            s_matrix = np.eye(len(scale_list))
            np.fill_diagonal(s_matrix, scale_list)

          
            C = s_matrix @ t_matrix
            matrix = C @ Ah.transpose()
            norm = matrix.transpose()[:, :-1]
            self.A = norm
            self.mean = self.A.mean(axis = 0)
        

        e_vals , v = np.linalg.eig( self.covariance_matrix(self.A) )

        sort_inds = np.argsort(e_vals)[::-1]
        self.e_vals = e_vals[sort_inds]
        self.e_vecs = v[:, sort_inds]
        self.prop_var = self.compute_prop_var(e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)

        pass

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        list = []

        list2 = []

        list3 = []

        for i in  range(len(self.prop_var)):
            list.append(i)

        if num_pcs_to_keep is None:
            plt.scatter(list, self.cum_var, s = 100)
            plt.xlabel("Each PC")
            plt.ylabel("Precent of Var")
        else:
            for x in range(num_pcs_to_keep):
                list2.append(x)
                list3.append(self.cum_var[x])
            plt.scatter(list2, list3, s = 100)
            plt.xlabel("Each PC")
            plt.ylabel("Precent of Var")
        pass

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''
        list = []
        for i in pcs_to_keep:
            list.append(self.e_vecs[:, i])


        matrix = np.array(list).transpose()

        self.A_proj = (self.A - self.mean) @ matrix 

        return self.A_proj

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        - If you normalized, remember to rescale the data projected back to the original data space.
        '''

        list = []

        for i in range(top_k):
            list.append(i)

        self.pca_project(list)

        v = self.e_vecs[:, 0: top_k]

        Ar = self.A_proj @ v.transpose() + self.mean

        if self.normalized:
            Ar = Ar * (self.l_max - self.l_min) + self.l_min
            return Ar
        else:
            return Ar

        

    


