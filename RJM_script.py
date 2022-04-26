from RJM_functions import*

# for multi processing: run one simulation
# apart from the simu ID and the data, all arguments are defined locally inside the function
# if no data is provided, synthetic data is generated instead
def run_one_simulation(simu, X_full=None, labels=None):
    """
    run one simulation
   
    Arguments:
        simu: An integer. ID of this simulation. Determines the seed of the simulation.
        X_full: A pandas DataFrame of shape: n x p. Observed data: the index is the observation ID, the columns are the features. If None, synthetic data is generated.
        labels: A pandas Series of length n. Class-labels of the observed data.
    """
    
    # set the global seed here to have each simulation be different
    np.random.seed(simu)
    
    # just print when the job starts and ends
    print(f'simu {simu} started, with random ID: {np.random.rand()}')
    # check cpus available to this job
    if platform.system()=='Linux':
        cpu_avail = os.sched_getaffinity(0)
        print(f'simu {simu} can access the following CPUs:{cpu_avail}')
    print()
    # print nothing else (to avoid confusion)
    #if True:
    with HiddenPrints(): 
        # by default, the true parameters do not exist
        mu = None
        Omega = None
        Sigma = None
        tau_true = None
        
        # if the data is not provided, then we are in synthetic data mode
        # remark: if generated here, each job (individual simu) will work with its own unique data set
        if (X_full is None) or (labels is None):
            # these parameters are now chosen
            # number of classes
            K = 2
            # total number of features
            p_total = 2000
            #p_total = 200
            # total number of observations
            n_total = 2000
            #n_total = 100
            # does the synthetic data have the same covariance matrix across all classes?
            same_covariance_synthetic = False
            # distribution of the synthetic data
            synthetic_data_distribution = 'normal'

            # generate synthetic mixture parameters
            mu, Omega, Sigma, tau_true = generate_synthetic_parameters(K, p_total, same_covariance_synthetic)
            # generate synthetic data from mixture parameters
            X_full, labels = generate_synthetic_data(mu, Sigma, tau_true, n_total, synthetic_data_distribution)
        
        # if the data is provided, get the dimensions there 
        else:
            # total number of observations and features
            n_total, p_total = X_full.shape
            # to be efficient, we will estimate the true parameters from the labels only after p is subsampled

        # recover number of classes
        K = labels.nunique()
        
        ############################################################# experiment_parameters #############################################################

        # version number
        results_version = 'V1'
        
        ##################  projections and metrics

        # list of projections 
        projection_tried = [
            'mixture-PCA projection', 
            #'shrunk mixture-PCA projection', 
            #'random projection',
            #'sparse random projection'
                           ]
        # list of clustering algorithms compared
        clustering_algorithms_list = ['KMeans', 
                                      'GaussianMixture', 
                                      'MixtureOfExperts',
                                     ]
        

        # do we evaluate the oracle error with the true parameters?
        evaluate_oracle_error = True
        # split the data to evaluate the OoS error?
        evaluate_OoS_error = False
        # OoS set size
        train_set_size = 0.7
        
        # do we remove the tempering to compute the final weights after the EM is done?
        remove_final_tempering = False
        # do we estimate the final likelihood with a UNIFORMELY (1/q) tempered/normalised bloc X?
        uniformely_temper_metrics = False
        
        # do we run an EM (with shrinkage) in the ambient space?
        run_ambient_space_EM = True
        
        # do we compute other clustering algorithms (KMeans) as a benchmark?
        benchmark_clustering = True
        # do we compute these clustering algorithms in the embedding space as well?
        embedding_clustering = False
        # do we cluster on [X, Y] as well as X?
        XY_clustering = True
        # do we use the model based oracle labels as reference for the other algorithms?
        use_oracle_as_reference = False
        
        # do we evaluate the ROC curve of the sparsistency in beta and Omega?
        evaluate_sparsistency_roc = False
        
        # do we estimate the final graph (for ROC sparsitency) with a gaussian estimator?
        run_gaussian_estimator = True
        # do we estimate the final graph (for ROC sparsitency) with a nonparanormal estimator?
        run_nonparanormal_estimator = False
        # do we estimate the final graph (for ROC sparsitency) with a mixed data (binary+normal) estimator?
        run_binary_estimator = False
        
        # Do we evaluate the stability of the clustering over several folds?
        evaluate_stability = False
        
        # do we raise the exception when a method fails or just move on?
        raise_exceptions = False

        ################## sliders
        
        # simu_id provided by the user
        simu_list = [simu]

        # ambient space size provided by the user
        p_list = [100, 316, 1000, 3162, 10000]
        p_list = [100]
        
        # number of rows in the initial dataset
        n_list = [100, 200, 500, 1000, 2000]
        n_list = [500]

        # number of classes tried, in JCR only (all other methods use the real K)
        K_list = [1, 2, 3, 4, 5, 10]
        K_list = [K]
        
        # embedding space sizes
        q_list = [1, 2, 5, 10, 20, 50]
        q_list = [5]

        # list of imposed class-norm differences in the empirical mu of the data
        delta_mu_list = [0, 0.1, 0.125, 0.15, 0.2]
        delta_mu_list = [0, 0.1, 0.2]
        #delta_mu_list = [0, None]

        # proportion of equal coefficients between classes among the non-zero ones in beta
        overlap_beta_list = [0, 0.1, 0.5, 0.9, 1]
        overlap_beta_list = [0]
        
        # Absolute value of all non zero coefficient
        amplitude_beta_list = [0, 0.5, 1.5, 3., 5.]
        amplitude_beta_list = [0, 2.5, 5]
        amplitude_beta_list = [0, 0.1]
        #amplitude_beta_list = [0.1]
        
        # tempering of the bloc X
        log_tmp_exponent_list = [-2, -1, -0.75, -0.5, -0.1, 0, 0.1, 0.5, 0.75, 1, 2]
        log_tmp_exponent_list = [2, 1, 0.75, 0.5, 0.1, 0, -0.1]
        log_tmp_exponent_list = [1, 0]
        #log_tmp_exponent_list = [0]
        
        # the tempering exponent tried have the form: T = 1/q^(alpha); alpha is scaleless this way
        exponent_list = [f"1/(q^{log})" for log in log_tmp_exponent_list]

        # shrinkage on Sigma_X in the M step of the projected_EM
        shrinkage_list = [False,  'oas']
        shrinkage_list = [False]
            
        # list of all followed metrics, given the provided simulation settings        
        metric_list = get_metrics_list(evaluate_oracle_error)
        
        # the final tempering: once we have theta_hat at the end of the EM, do we temper the model based estimation?
        # three possibilities: same as the EM, no tmp (T=1) or uniform at 1/x_dim
        final_tempering_list = ["same_as_EM"]+["none"]*remove_final_tempering+["1/x_dim"]*uniformely_temper_metrics
        
        # data set used, train or validation
        dataset_type_list = ['IS']+['OoS']*evaluate_OoS_error
        
        # algorithms tried
        algorithm_list = ['JCR_EM']+['ambient_JCR_EM']*run_ambient_space_EM
        
        # the clustering alorithms can be ran on X or XY
        clustering_data_type = ['X']+['XY']*XY_clustering
        # add all possiblie variation of the clustering algorithms
        if benchmark_clustering:
            for algorithm in clustering_algorithms_list:
                for algorithm_suffixe in clustering_data_type:
                    # we do not do MoE with just X
                    if (f'{algorithm}_{algorithm_suffixe}'=='MixtureOfExperts_X'):
                        continue
                    algorithm_list += [f'{algorithm}_{algorithm_suffixe}']+[f'projected_{algorithm}_{algorithm_suffixe}']*embedding_clustering
        
        ## stability
        
        # number of sub-sampled data sets used in the stability measure
        n_stability_folds = 5
        # number of distinct pairs of subsampling folds
        n_stability_fold_pairs = int(n_stability_folds*(n_stability_folds-1)/2)
        
        # metrics measured over several stability folds 
        stability_metric_list=['rand_index']
        
        ## ROC metrics
                
        # nb solutions with different penalty intensities to compute
        n_solutions = 10
        
        # confusion metrics measured on the parameter estimation
        roc_metric_list = ['P', 'N', 'TP', 'TN', 'l0_sign', 'l1_sign', 'l1', 'l2']
    
        # sparse parameters to estimate
        parameter_list = ['beta']+['Omega glasso']*run_gaussian_estimator+\
            ['Omega non_paranormal']*run_nonparanormal_estimator+['Omega mixed_binary']*run_binary_estimator

        # are we looking at the estimated parameters with the estimated or the true labels?
        label_type_list = ['estimated', 'true']

        
        ################## initialise data array

        # initialise local results dataarray
        results_da = xr.DataArray(
            coords = [
                algorithm_list,
                projection_tried, 
                p_list,
                q_list,
                n_list,
                K_list,
                overlap_beta_list,
                amplitude_beta_list,
                delta_mu_list,
                simu_list,
                metric_list,
                exponent_list,
                shrinkage_list,
                final_tempering_list,
                dataset_type_list
            ], 
            dims = [
                'algorithm',
                'projection', 
                'p', 
                'q', 
                'n',
                'K_tried_JCR',
                'overlap_beta',
                'amplitude_beta',
                'delta_mu',
                'simu', 
                'metric', 
                'tempering_exponent',
                'shrink_model_covariance',
                'final_tempering',
                'dataset_type'
            ])

        # in this data array, we save the metrics measured over several stability folds 
        stability_da = xr.DataArray(
            coords = [
                projection_tried, 
                p_list,
                q_list,
                n_list,
                K_list,
                overlap_beta_list,
                amplitude_beta_list,
                delta_mu_list,
                simu_list,
                stability_metric_list,
                exponent_list,
                shrinkage_list,
                range(n_stability_fold_pairs), 
            ], 
            dims = [
                'projection', 
                'p', 
                'q', 
                'n',
                'K_tried_JCR',
                'overlap_beta',
                'amplitude_beta',
                'delta_mu',
                'simu', 
                'metric', 
                'tempering_exponent',
                'shrink_model_covariance',
                'subsample_folds_pair'
            ])
        
        # in this data array, we save the confusion metrics of the estimated sparse parameters to plot ROC curves
        # no "K_list" here, since the ROC metrics make no sense if K_tried=/=K_real anyway
        sparsistency_roc_da = xr.DataArray(
            coords = [
                algorithm_list,
                projection_tried, 
                p_list,
                q_list,
                n_list,
                overlap_beta_list,
                amplitude_beta_list,
                delta_mu_list,
                simu_list,
                exponent_list,
                shrinkage_list,
                roc_metric_list,
                parameter_list,
                range(n_solutions),
                label_type_list,
                dataset_type_list,
                final_tempering_list
            ], 
            dims = [
                'algorithm',
                'projection', 
                'p', 
                'q', 
                'n',
                'overlap_beta',
                'amplitude_beta',
                'delta_mu',
                'simu', 
                'tempering_exponent',
                'shrink_model_covariance',
                'metric', 
                'parameter', 
                'solution',
                'label_type',
                'dataset_type',
                'final_tempering'
            ])
        
        ################## fixed parameters

        ## data generation

        # beta generation method
        beta_method = 'overlap'
        #beta_method = 'flipped_signs' 
        #beta_method = 'simpsons_paradox'
        
        # beta slider-parameters grid
        beta_parameters_list = list(itertools.product(overlap_beta_list, amplitude_beta_list))

        # sparse beta or dense beta?
        sparse_beta = True
        # are all the non zero coefficient in the same place, even if they are not equal?
        common_support = False
        # are the (absolute) value of the non zero coefficients fixed or random?
        fixed_non_zero_value = True
        # nb of non-zero coeff in each beta, not used if fraction_non_zeros is specified as something other than "None"
        n_non_zeros = 100
        n_non_zeros = 10
        #n_non_zeros = 1
        # fraction (of p) of non-zero coeff in each beta,
        fraction_non_zeros = 0.01
        fraction_non_zeros = None

        # if using "flipped signs" beta or fixed_non_zero_value = True for 'overlap' beta: absolute value of all non zero coefficient
        #non_zero_abs_value=3.
        # it is a slider now
    
        # lower absolute value of the non-zero coef
        threshold = 1.
        # gaussian parameters for the dense beta
        zero_scale = 0.2
        non_zero_mean = 2.
        non_zero_scale = 0.7

        # variance of Y|X
        sigma = 0.5

        ## penalties

        # pen prop to n of p: to big -> vanishing clusters all the time
        # prop to q or 1: ok!
        penalty_sigma = 5.

        # penalty on the class weights tau, such that tau>n_low_threshold/n
        n_low_threshold = 5
        # the formula is dependent on n: penalty_tau = n * n_low_threshold / (n - K * n_low_threshold)

        # parameter lambda in the lasso on beta 
        # the formula is dependent on n: penalty_lasso = xr.DataArray(data=np.ones(K)*n/K, coords=[range(K)], dims=['label'])

        ## stability

        # fraction of the total data in each subsample data set
        subsampled_fraction = 0.75

        ## procedure parameters

        # EM stopping criteria
        neg_ll_ratio_threshold = 1e-5
        E_step_difference_threshold = 1e-3
        max_steps = 100
        min_steps = 5

        ## clustering algorithms parameters
    
        clustering_algorithm_parameters = {}
        
        # KMeans parameters
        clustering_algorithm_parameters['KMeans'] = dict(
            # clusters to estimate
            n_clusters = K,
            # we keep the default parameters for the rest, but to change that, just change below:
            
            # init type
            init='k-means++',
            # number of init tried
            n_init=10,
            # max nb of iterations
            max_iter=300,
            # tolerance threshold for convergence
            tol=0.0001,
            # KMeans alogorithm
            algorithm='auto')
        
        # SpectralClustering parameters
        clustering_algorithm_parameters['SpectralClustering'] = dict(n_clusters=K, eigen_solver = 'lobpcg')
        
        # GaussianMixture parameters
        clustering_algorithm_parameters['GaussianMixture'] = dict(n_components=K,
                                                                  # regularization added to the diagonal of covariance
                                                                  reg_covar = 1e-3,
                                                                  # number of init tried
                                                                  n_init=10)
        # MixtureOfExperts parameters
        clustering_algorithm_parameters['MixtureOfExperts'] = dict(n_clusters=K,
                                                                  # R MoE library to use 'flexmix' or 'mixtools'
                                                                  library = 'flexmix')
        
        ## EM ambient space parameters
        
        # shrinkage method in the M step of the ambient RJM-EM
        shrinkage_ambient_EM = 'oas'
        
        ## Oracle Model-based Classifier parameters
        
        # data array version of sigma
        sigma_true = xr.DataArray(sigma*np.ones(K), coords=[range(K)], dims=['label']) 
        

        ################## loop
        for p in p_list:
            print(f'-------------------------------------------- p = {p} -------------------------------------------- ')
            # sampling here = nested q, fix the seed so that different real data runs have the sample columns (diff simu and p should still have diff columns)
            X = X_full.sample(p, replace=False, axis='columns', random_state = simu + p).copy()  
            
            # sample the mean mu, the precision Omega and the covariance Sigma accordingly
            columns_sampled = X.columns
            
            # if the data is real, we must estimate the true parameters from the labels
            if mu is None:
                # create temporary data arrays, needed for the function
                X_full_da = xr.DataArray(
                    data = X.copy(), 
                    coords = [range(n_total) , range(p)], 
                    dims = ['observation', 'component_p'])
                # convert labels to dummies, and put in data array form
                labels_full_da = convert_label_to_dummies(labels)

                # estimate the true parameters
                mu_base, Sigma_true, tau_true = get_true_parameters(X_full_da, labels_full_da, shrinkage='oas')
                # we do not need those anymore
                del X_full_da, labels_full_da
                # Omega: precision matrix
                Omega_true = xr.DataArray(data = np.linalg.inv(Sigma_true), coords = Sigma_true.coords, dims = Sigma_true.dims)
            
            # if the data is synthetic, slice the true parameters according to the columns subsample
            else:
                mu_base = mu.sel(component_p = columns_sampled).assign_coords({'component_p':range(p)})
                Omega_true = Omega.sel(component_p = columns_sampled, component_pT = columns_sampled).assign_coords({'component_p':range(p), 'component_pT':range(p)})
                Sigma_true = Sigma.sel(component_p = columns_sampled, component_pT = columns_sampled).assign_coords({'component_p':range(p), 'component_pT':range(p)})
            
            for n in n_list:
                print(f'-------------------------------------------- n = {n} -------------------------------------------- ')
                # "n" in the penalty is the size of the training set: n if no split, n*train_set_size otherwise
                n_penalty = int(n*train_set_size**evaluate_OoS_error)
                # set the values of the n-dependent penalties here
                #penalty_tau = n_penalty * n_low_threshold / (n_penalty - K * n_low_threshold)
                #penalty_lasso = xr.DataArray(data=np.ones(K)*n_penalty/K, coords=[range(K)], dims=['label'])
                # these are used for all algo using the real K
                # when JCR tries other K_tried, the penalty are recomputed for K_tried

                # subsample n rows here
                X_subsampled, labels_subsampled = subsample_rows(X, labels, n)
                # convert to Data array
                X_da = xr.DataArray(
                    data = X_subsampled.copy(), 
                    coords = [range(n) , range(p)], 
                    dims = ['observation', 'component_p'])

                labels_da =  xr.DataArray(
                    data = labels_subsampled.copy(), 
                    coords = [range(n)], 
                    dims = ['observation'])

                for delta_mu in delta_mu_list:
                    print(f'-------------------------------------------- delta_mu = {delta_mu} -------------------------------------------- ')
                    # Treat (shift) mu, the class averages (keeping the native delta_mu: use delta_mu = None in the function)
                    X_processed_da, mu_translation = fix_empirical_mu(X_da, labels_da, delta_mu) 
                        
                    # get the correct mu_true
                    mu_true = mu_base + mu_translation
                        
                    # beta parameters grid
                    for (overlap_beta, non_zero_abs_value) in beta_parameters_list:
                        print(f'-------------------------------------------- overlap_beta = {overlap_beta} -------------------------------------------- ')
                        print(f'-------------------------------------------- non_zero_abs_value = {non_zero_abs_value} -------------------------------------------- ')
                        # is the fraction of non-zero coeff in beta fixed for all p?
                        if fraction_non_zeros:
                            # nb of non-zero coeff in each beta
                            n_non_zeros = int(p*fraction_non_zeros)
                        # nb of common non-zero coefficients between classess
                        n_common = int(overlap_beta*n_non_zeros)
                        # if beta_method = flipped_signs, this is the parameter used:
                        sign_flip_window = (n_non_zeros-n_common)//2
                        # careful: if n_non_zeros-n_common =/= 0 [2], the provided overlap fraction will only be approximated, not exact
                        
                        # Generate beta with controlled signal, according to the selected method
                        beta = generate_beta_wrapper(p, K, 
                                                     # name of the method
                                                     beta_method = beta_method,
                                                     n_non_zeros = n_non_zeros, n_common = n_common, 
                                                     # sparse or dense beta
                                                     sparse_beta=sparse_beta,
                                                     # are all the non zero coefficient in the same place, even if they are not equal?
                                                     common_support=common_support,
                                                     # are the (absolute) value of the non zero coefficients fixed or random?
                                                     fixed_non_zero_value = fixed_non_zero_value,
                                                     # minimum threshold for the gaussian behind the sparse beta
                                                     threshold=threshold, 
                                                     # gaussian parameters for the dense beta
                                                     zero_scale = zero_scale,
                                                     non_zero_mean = non_zero_mean,
                                                     non_zero_scale = non_zero_scale,

                                                     # specific to 'flipped_signs':

                                                     # number of coefficients with flipped signs from the reference
                                                     # each pair of beta are different by 2*sign_flip_window coefficients
                                                     sign_flip_window = sign_flip_window,  
                                                     # absolute value of all non zero coefficient
                                                     non_zero_abs_value = non_zero_abs_value)                        

                        # normalise beta by the number of non-zeros
                        # no impact on the relative scale of the coefficent in beta
                        # simply here to balance the impact of beta and the impact of the noise sigma in the value of Y
                        beta = 10*beta/n_non_zeros
                        
                        # beta_true is beta, but, by convention, we need an intercept (= 0 here) on the last (p+1th) component
                        beta_true = xr.DataArray(data=np.zeros((K, p+1)), coords = [range(K), range(p+1)], dims = ['label', 'component_p'])
                        # put beta on the first p components, let 0 on the last one
                        beta_true.loc[dict(component_p = range(p))] = beta.copy()

                        # generate Y
                        Y_da = generate_Y(X_processed_da, labels_da, beta, sigma)
                        
                        # if we want to evaluate OoS criteria: split train/val set
                        # Doing the split here, after the mean of X is set empirically can be source of controversy
                        if evaluate_OoS_error:
                            Y_train, X_train, labels_train, Y_val, X_val, labels_val = split_train_val_sets(Y_da, X_processed_da, labels_da, train_set_size)
                            # convert label from R^n with entries in {1, ..., K} to R^{n x K} with entries in {0, 1}
                            labels_dummies_val = convert_label_to_dummies(labels_val)
                        else:
                            # change name of labels for consistency with the "evaluate_OoS_error" case
                            Y_train, X_train, labels_train = Y_da.copy(), X_processed_da.copy(), labels_da.copy()
                            # no validation sets
                            Y_val, X_val, labels_val, labels_dummies_val = None, None, None, None

                        # convert label from R^n with entries in {1, ..., K} to R^{n x K} with entries in {0, 1}
                        labels_dummies_train = convert_label_to_dummies(labels_train)
                        
                        # compute the clustering methods from the ambient space data (with and without Y in the clustering)
                        if benchmark_clustering:
                            print('ambient space clustering')
                            
                            algorithm_prefix=''

                            for algorithm in clustering_algorithms_list:
                                for algorithm_suffixe in clustering_data_type:
                                    # name of the algo
                                    algorithm_name = f'{algorithm}_{algorithm_suffixe}'
                                    # parameters for this algo
                                    algorithm_parameters = clustering_algorithm_parameters[algorithm]
                                    # we do not do MoE with just X
                                    if (algorithm_name=='MixtureOfExperts_X'):
                                        continue

                                    # we try to run the method
                                    try:
                                        # run the algo and get the results
                                        output = evaluate_algorithm(
                                            # train data
                                            Y_train, X_train, labels_dummies_train, 
                                            # algorithm used
                                            algorithm=algorithm_name,
                                            # parameters for the algorithm
                                            algorithm_parameters = algorithm_parameters,
                                            # projection
                                            projection_da = None,
                                            # real beta, for sparsistency
                                            beta_true = beta_true,  
                                            # real precision matrix, for graph sparsistency
                                            Omega_true = Omega_true,
                                            # validation data (if desired)
                                            Y_val = Y_val, X_val = X_val, labels_val = labels_dummies_val,
                                            # if provided, the labels estimated by an oracle method to compare oneself to
                                            labels_train_oracle = None, 
                                            labels_val_oracle = None,
                                            # do we evaluate the oracle error with the true parameters?
                                            use_oracle_as_reference=use_oracle_as_reference, 
                                            # do we evaluate the ROC curve of the sparsistency in beta and Omega?
                                            evaluate_sparsistency_roc=evaluate_sparsistency_roc,
                                            # sparse parameters to estimate
                                            parameter_list=parameter_list,
                                            # number of solutions to compute for the ROC curve
                                            n_solutions = n_solutions)

                                        # the current settings for this run, used to slice the data array in the update
                                        settings_dictionary=dict(p=p, n=n, simu=simu, delta_mu=delta_mu, overlap_beta=overlap_beta, amplitude_beta = non_zero_abs_value,
                                                                 algorithm=f'{algorithm_prefix}{algorithm_name}')
                                        # update the data arrays 
                                        update_results_da(results_da, sparsistency_roc_da, output, settings_dictionary, evaluate_sparsistency_roc)
                                    
                                    # if the method glitched for some reason:
                                    except Exception as e:
                                        # raise the exception if we are in a reaserch/debug mode
                                        if raise_exceptions:
                                            raise e
                                        # otherwise, ignore, let the data array empty and move on
                                        
                                # 'for algorithm_suffixe in clustering_data_type' over
                            # 'for algorithm in clustering_algorithms_list' over
                        # 'if benchmark_clustering' over
                        
                        # run an EM in the ambient space
                        if run_ambient_space_EM and (p<5000):                            
                            for exponent_string, log_tmp_exponent in zip(exponent_list, log_tmp_exponent_list):
                                print(f'-------------------------------------------- exponent_string = {exponent_string} -------------------------------------------- ')
                                # get the exponent numerical value as well as the descriptive string
                                tempering_exponent = 1./p**log_tmp_exponent

                                # the different class numbers K (= models) tried
                                for K_tried in K_list:
                                    print(f'-------------------------------------------- K tried = {K_tried} -------------------------------------------- ')
                                    # set the values of the (n, K)-dependent penalties here
                                    penalty_tau = n_penalty * n_low_threshold / (n_penalty - K_tried * n_low_threshold)
                                    penalty_lasso = xr.DataArray(data=np.ones(K_tried)*n_penalty/K_tried, coords=[range(K_tried)], dims=['label'])

                                    # if K_tried is not the real K, we can only compute optim metrics, no reconstruction or ROC metrics
                                    if K_tried == K:
                                        only_optimisation_metrics = False
                                    else:
                                        only_optimisation_metrics = True

                                    # we try to run the method
                                    try:
                                        ###### run EM and compute metrics
                                        output = evaluate_EM_JCR(
                                            # train data
                                            Y_train, X_train, labels_dummies_train, 
                                            # projection
                                            projection_da = None,
                                            # or directly the encoded data (not necesarily linear encoding),
                                            WT_X_da = None,
                                            # initialisation
                                            beta_0 = None, sigma_0 = None, mu_0 = None, Sigma_0 = None, tau_0 = None,
                                            # real values of the simulated parameters (for Oracle loss)
                                            beta_true = beta_true, sigma_true = sigma_true, mu_true = mu_true, Sigma_true = Sigma_true, tau_true = tau_true, 
                                            # real precision matrix, for graph sparsistency
                                            Omega_true = Omega_true,
                                            # validation data (if desired)
                                            Y_val = Y_val, X_val = X_val, labels_val = labels_dummies_val,
                                            # EM stopping criteria
                                            neg_ll_ratio_threshold = neg_ll_ratio_threshold,
                                            E_step_difference_threshold = E_step_difference_threshold,
                                            max_steps = max_steps, 
                                            min_steps = min_steps,
                                            # shrinkaged method in the M step
                                            shrinkage = shrinkage_ambient_EM,
                                            # exponent of bloc X
                                            tempering_exponent = tempering_exponent,
                                            # penalty intensities
                                            penalty_lasso = penalty_lasso, 
                                            penalty_sigma = penalty_sigma,
                                            penalty_tau = penalty_tau,
                                            # should we remove the tempering on X in the final class weigh estimation as well?
                                            remove_final_tempering = remove_final_tempering,
                                            # should we estimate the final likelihood with a UNIFORMELY (1/q) tempered/normalised bloc X?
                                            uniformely_temper_metrics=uniformely_temper_metrics,
                                            # do we evaluate the oracle error with the true parameters?
                                            evaluate_oracle_error=evaluate_oracle_error, 
                                            # do we evaluate the ROC curve of the sparsistency in beta and Omega?
                                            evaluate_sparsistency_roc=evaluate_sparsistency_roc, 
                                            # sparse parameters to estimate
                                            parameter_list=parameter_list,
                                            # number of solutions to compute for the ROC curve
                                            n_solutions = n_solutions,
                                            # do we skip all metrics except the optimisiation ones? (used when K=/=K_true)
                                            only_optimisation_metrics = only_optimisation_metrics, 
                                            # desired K, if different from the real one
                                            K = K_tried)

                                        # the current settings for this run to slice the data array with in the update
                                        settings_dictionary=dict(p=p, n=n, K_tried_JCR=K_tried, simu=simu, delta_mu=delta_mu,
                                                                 overlap_beta=overlap_beta, amplitude_beta=non_zero_abs_value,
                                                                 tempering_exponent=exponent_string, algorithm='ambient_JCR_EM')
                                        # update the data arrays 
                                        update_results_da(results_da, sparsistency_roc_da, output, settings_dictionary, evaluate_sparsistency_roc)

                                    # if the method glitched for some reason:
                                    except Exception as e:
                                        # raise the exception if we are in a reaserch/debug mode
                                        if raise_exceptions:
                                            raise e
                                        # otherwise, ignore, let the data array empty and move on

                                # 'for K_tried in K_list' over    
                            # 'for exponent_string in exponent_list' over
                        # 'if run_ambient_space_EM and (p<5000)' over
                                    
                        #### projection part
                        for q in q_list:
                            if (q<=p):
                                print(f'-------------------------------------------- q = {q} -------------------------------------------- ')
                                # where we do the initialisation depends on whether we are just using the real K or a list of K_tried
                                if (len(K_list) == 1) and (K_list[0] == K):
                                    # get the initialisation here, so that it is the same for all projections
                                    # (except potentially JCR if JCR tries different K_tried)
                                    print()
                                    print('get a random initialisation')
                                    beta_0, sigma_0, mu_0, Sigma_0, tau_0 = initialise_parameters(K, p, q)

                                for projection_name in projection_tried:
                                    ###### learn projection 
                                    start_time = time.time()

                                    # random projection
                                    projection = get_projection(pd.DataFrame(X_train.values), q, projection_name)

                                    end_time = time.time()
                                    # save the duration of the projection learning step
                                    projection_learning_duration = end_time - start_time
                                    
                                    # convert to da
                                    projection_da = xr.DataArray(
                                        data = projection, 
                                        coords = [range(p), range(q) ], 
                                        dims = ['component_p', 'component_q' ])

                                    # compute the KMeans from the Embedding space data (with and without Y in the clustering)
                                    if benchmark_clustering & embedding_clustering:
                                        print('embedding space clustering')

                                        algorithm_prefix='projected_'
                                        
                                        for algorithm in clustering_algorithms_list:
                                            for algorithm_suffixe in clustering_data_type:
                                                # name of the algo
                                                algorithm_name = f'{algorithm}_{algorithm_suffixe}'
                                                # parameters for this algo
                                                algorithm_parameters = clustering_algorithm_parameters[algorithm]
                                                # we do not do MoE with just X
                                                if (algorithm_name=='MixtureOfExperts_X'):
                                                    continue

                                                # try to run the method
                                                try:
                                                    # run the algo and get the results
                                                    output = evaluate_algorithm(
                                                        # train data
                                                        Y_train, X_train, labels_dummies_train, 
                                                        # algorithm used
                                                        algorithm=algorithm_name,
                                                        # parameters for the algorithm
                                                        algorithm_parameters = algorithm_parameters,
                                                        # projection
                                                        projection_da = projection_da,
                                                        # real beta, for sparsistency
                                                        beta_true = beta_true,  
                                                        # real precision matrix, for graph sparsistency
                                                        Omega_true = Omega_true,
                                                        # validation data (if desired)
                                                        Y_val = Y_val, X_val = X_val, labels_val = labels_dummies_val,
                                                        # if provided, the labels estimated by an oracle method to compare oneself to
                                                        labels_train_oracle = None, 
                                                        labels_val_oracle = None,
                                                        # do we evaluate the oracle error with the true parameters?
                                                        use_oracle_as_reference=use_oracle_as_reference, 
                                                        # do we evaluate the ROC curve of the sparsistency in beta and Omega?
                                                        evaluate_sparsistency_roc=evaluate_sparsistency_roc,                                             
                                                        # sparse parameters to estimate
                                                        parameter_list=parameter_list,
                                                        # number of solutions to compute for the ROC curve
                                                        n_solutions = n_solutions)

                                                    # the current settings for this run to slice the data array with in the update
                                                    settings_dictionary=dict(p=p, n=n, simu=simu, q=q, projection=projection_name,
                                                                             delta_mu=delta_mu, overlap_beta=overlap_beta, amplitude_beta = non_zero_abs_value,
                                                                             algorithm=f'{algorithm_prefix}{algorithm_name}')
                                                    # update the data arrays 
                                                    update_results_da(results_da, sparsistency_roc_da, output, settings_dictionary, evaluate_sparsistency_roc)
                                                
                                                # if the method glitched for some reason:
                                                except Exception as e:
                                                    # raise the exception if we are in a reaserch/debug mode
                                                    if raise_exceptions:
                                                        raise e
                                                    # otherwise, ignore, let the data array empty and move on
                                            
                                            # 'for algorithm_suffixe in clustering_data_type' over
                                        # 'for algorithm in clustering_algorithms_list' over
                                    # 'if benchmark_clustering' over   
                                    
                                    # we do not shrink in this experiment, to save on computation costs
                                    for shrinkage in shrinkage_list:
                                        for exponent_string, log_tmp_exponent in zip(exponent_list, log_tmp_exponent_list):
                                            print(f'-------------------------------------------- exponent_string = {exponent_string} -------------------------------------------- ')
                                            # get the exponent numerical value as well as the descriptive string
                                            tempering_exponent = 1./q**log_tmp_exponent

                                            # the different class numbers K (= models) tried
                                            for K_tried in K_list:
                                                print(f'-------------------------------------------- K tried = {K_tried} -------------------------------------------- ')
                                                # set the values of the (n, K)-dependent penalties here
                                                penalty_tau = n_penalty * n_low_threshold / (n_penalty - K_tried * n_low_threshold)
                                                penalty_lasso = xr.DataArray(data=np.ones(K_tried)*n_penalty/K_tried, coords=[range(K_tried)], dims=['label'])
                                            
                                            
                                                # if we do not try just K_tried == K_real, then each K has its own initialisation
                                                if (len(K_list) != 1) or (K_tried != K):
                                                    # get the initialisation here, it is not the same for every projection anymore
                                                    # but having each K tried with the exact same embedded data WtX is much more important
                                                    print()
                                                    print('get a random initialisation')
                                                    beta_0, sigma_0, mu_0, Sigma_0, tau_0= initialise_parameters(K_tried, p, q)
                                                
                                                # if K_tried is not the real K, we can only compute optim metrics, no reconstruction or ROC metrics
                                                if K_tried == K:
                                                    only_optimisation_metrics = False
                                                else:
                                                    only_optimisation_metrics = True


                                                print()
                                                print(p, simu, n, delta_mu, overlap_beta, non_zero_abs_value, q, projection_name, K_tried, shrinkage, exponent_string)

                                                # try to run the method
                                                try:
                                                    ###### run EM and compute metrics
                                                    output = evaluate_EM_JCR(
                                                        # train data
                                                        Y_train, X_train, labels_dummies_train, 
                                                        # projection
                                                        projection_da,
                                                        # or directly the encoded data (not necesarily linear encoding),
                                                        WT_X_da=None,
                                                        # initialisation
                                                        beta_0=beta_0, sigma_0=sigma_0, mu_0=mu_0, Sigma_0=Sigma_0, tau_0=tau_0,
                                                        # real values of the simulated parameters (for Oracle loss)
                                                        beta_true = beta_true, sigma_true = sigma_true, 
                                                        mu_true = mu_true, Sigma_true = Sigma_true, tau_true = tau_true, 
                                                        # real precision matrix, for graph sparsistency
                                                        Omega_true = Omega_true,
                                                        # validation data (if desired)
                                                        Y_val = Y_val, X_val = X_val, labels_val = labels_dummies_val,
                                                        # EM stopping criteria
                                                        neg_ll_ratio_threshold = neg_ll_ratio_threshold,
                                                        E_step_difference_threshold = E_step_difference_threshold,
                                                        max_steps = max_steps, 
                                                        min_steps = min_steps,
                                                        # shrinkaged method in the M step
                                                        shrinkage = shrinkage,
                                                        # exponent of bloc X
                                                        tempering_exponent = tempering_exponent,
                                                        # penalty intensities
                                                        penalty_lasso = penalty_lasso, 
                                                        penalty_sigma = penalty_sigma,
                                                        penalty_tau = penalty_tau,
                                                        # should we remove the tempering on X in the final class weigh estimation as well?
                                                        remove_final_tempering = remove_final_tempering,
                                                        # should we estimate the final likelihood with a UNIFORMELY (1/q) tempered/normalised bloc X?
                                                        uniformely_temper_metrics=uniformely_temper_metrics,
                                                        # do we evaluate the oracle error with the true parameters?
                                                        evaluate_oracle_error=evaluate_oracle_error, 
                                                        # do we evaluate the ROC curve of the sparsistency in beta and Omega?
                                                        evaluate_sparsistency_roc=evaluate_sparsistency_roc,
                                                        # sparse parameters to estimate
                                                        parameter_list=parameter_list,
                                                        # number of solutions to compute for the ROC curve
                                                        n_solutions = n_solutions,
                                                        # do we skip all metrics except the optimisiation ones? (used when K=/=K_true)
                                                        only_optimisation_metrics = only_optimisation_metrics, 
                                                        # desired K, if different from the real one
                                                        K = K_tried)

                                                    # the current settings for this run to slice the data array with in the update
                                                    settings_dictionary=dict(p=p, n=n,  K_tried_JCR=K_tried, simu=simu, delta_mu=delta_mu, 
                                                                             overlap_beta=overlap_beta, amplitude_beta=non_zero_abs_value,
                                                                             tempering_exponent=exponent_string, shrink_model_covariance=shrinkage, 
                                                                             q=q, projection=projection_name, algorithm='JCR_EM')
                                                    # update the data arrays 
                                                    update_results_da(results_da, sparsistency_roc_da, output, settings_dictionary, evaluate_sparsistency_roc)

                                                # if the method glitched for some reason:
                                                except Exception as e:
                                                    # raise the exception if we are in a reaserch/debug mode
                                                    if raise_exceptions:
                                                        raise e
                                                    # otherwise, ignore, let the data array empty and move on


                                                # evaluate the stability of the clustering over several folds
                                                if evaluate_stability:
                                                    # size of each subsampled data set
                                                    size_subsample_fold = int(subsampled_fraction*n)
                                                    # set the values of the n-dependent penalties
                                                    penalty_tau_subsample = size_subsample_fold * n_low_threshold / (size_subsample_fold - K_tried * n_low_threshold)
                                                    penalty_lasso_subsample = xr.DataArray(data=np.ones(K_tried)*size_subsample_fold/K_tried, coords=[range(K_tried)], dims=['label'])

                                                    # try to compute the stability of proj. JCR EM
                                                    try:
                                                        # get the estimated class assignment of several subsampling folds
                                                        hard_assignment_da, subsampled_indices_da = get_several_subsampled_class_assignments(
                                                            Y_train, X_train, projection_da, 
                                                            # number of folds
                                                            n_stability_folds=n_stability_folds,
                                                            # fraction of the total data in each subsample data set
                                                            subsampled_fraction = subsampled_fraction, 
                                                            ## EM parameters
                                                            # initialisation parameters
                                                            beta_0=beta_0, sigma_0=sigma_0, mu_0=mu_0, Sigma_0=Sigma_0, tau_0=tau_0, 
                                                            # stopping criteria
                                                            neg_ll_ratio_threshold = neg_ll_ratio_threshold,
                                                            E_step_difference_threshold = E_step_difference_threshold,
                                                            max_steps = max_steps, 
                                                            min_steps = min_steps,
                                                            # do we regularise the estimated covariance matrices?
                                                            shrinkage = shrinkage,
                                                            # exponent of the X-bloc in the E step
                                                            tempering_exponent=tempering_exponent, 
                                                            # penalty intensities
                                                            penalty_lasso = penalty_lasso_subsample, 
                                                            penalty_sigma = penalty_sigma,
                                                            penalty_tau = penalty_tau_subsample)

                                                        # get the pairwise rand indices between these class assignments
                                                        rand_index_list = compute_pairwise_rand_index(hard_assignment_da, subsampled_indices_da)

                                                        # update the stability results data array
                                                        stability_da.loc[dict(p=p, n=n, K_tried_JCR=K_tried, simu=simu, delta_mu=delta_mu, 
                                                                              overlap_beta=overlap_beta, amplitude_beta = non_zero_abs_value,
                                                                              q=q, projection=projection_name, shrink_model_covariance=shrinkage, 
                                                                              tempering_exponent=exponent_string)] = rand_index_list

                                                    # if the method glitched for some reason:
                                                    except Exception as e:
                                                        # raise the exception if we are in a reaserch/debug mode
                                                        if raise_exceptions:
                                                            raise e
                                                        # otherwise, ignore, let the data array empty and move on
                                            
                                                # 'if evaluate_stability' over
                                            # 'for K_tried in K_list' over
                                        # 'for exponent_string' over
                                    # 'for shrinkage' over
                                        
                                    # was there a reason for this to be here and not right after the projection_learning_duration is computed?
                                    # yes, otherwise it is overwritten with NaN somewhere else in the code
                                    # the projection learning duration is computed outside of the evaluate_EM function and must be updated outside too
                                    results_da.loc[dict(p=p, n=n, simu=simu, delta_mu=delta_mu, overlap_beta=overlap_beta, amplitude_beta = non_zero_abs_value,
                                                        q=q, projection=projection_name, algorithm = 'JCR_EM', 
                                                        metric = 'projection_learning_duration')] = projection_learning_duration
    
    # output dictionary
    output = {}
    output['results'] = results_da
    output['stability'] = stability_da
    output['sparsistency_roc'] = sparsistency_roc_da
    output['version'] = results_version
    
    print(f'simu {simu} over')
    print()

    return output


# this illustrative script only runs one simulation
# several simulations can be sequentially or in parallel depending on the system
# "run_one_simulation" only returns the output, you need to write the code that saves the data within your own file system
if __name__ == '__main__':
    # run the main function
    run_one_simulation(simu=0, X_full=None, labels=None)
