
import numpy as np
import scipy.stats as st
from scipy.special import logit, expit
import os,sys
import argparse
import matplotlib.pyplot as plt

def compute_confidence(metric, N_train, N_test, alpha=0.95):
    """
    Function to calculate the adjusted confidence interval
    metric: numpy array containing the result for a metric for the different cross validations
    (e.g. If 20 cross-validations are performed it is a list of length 20 with the calculated accuracy for
    each cross validation)
    N_train: Integer, number of training samples
    N_test: Integer, number of test_samples
    alpha: float ranging from 0 to 1 to calculate the alpha*100% CI, default 95%
    """

    # Convert to floats, as python 2 rounds the divisions if we have integers
    N_train = float(N_train)
    N_test = float(N_test)
    N_iterations = float(len(metric))

    if N_iterations == 1.0:
        print('[WORC Warning] Cannot compute a confidence interval for a single iteration.')
        print('[WORC Warning] CI will be set to value of single iteration.')
        metric_average = np.mean(metric)
        CI = (metric_average, metric_average)
    else:
        metric_average = np.mean(metric)
        S_uj = 1.0 / (N_iterations - 1) * np.sum((metric_average - metric)**2.0)

        metric_std = np.sqrt((1.0/N_iterations + N_test/N_train)*S_uj)

        CI = st.t.interval(alpha, N_iterations-1, loc=metric_average, scale=metric_std)

    if np.isnan(CI[0]) and np.isnan(CI[1]):
        # When we cannot compute a CI, just give the averages
        CI = (metric_average, metric_average)
    return CI


def plot_global_results(glo_result_list, figure_result_dir, metric):

    xdata = range(len(glo_result_list)-1)
    ydata = glo_result_list[1:]


    # fig = plt.figure(figsize=(7.2, 4.2))
    fig = plt.figure(figsize=(4,3))

    plt.plot(xdata, ydata, 'go-', label = "FedAvg")
    plt.axhline(y= glo_result_list[0], color='g', linestyle='--' , label = "Without aggregation")
    plt.legend(loc = "lower right", fontsize=11)
    plt.ylabel("%s" %metric, fontsize=15)
    plt.xlabel("Iteration", fontsize=15)
    # plt.xticks(xdata, range(1, len(glo_result_list)), fontsize=12, rotation=70 )
    x_for_ticks = range(1, len(glo_result_list), 2)
    plt.xticks(x_for_ticks, range(2, len(glo_result_list)+1, 2), fontsize=12)

    plt.ylim([0.74,0.8])
    yt = [0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80]
    plt.yticks(yt, fontsize=12)
    # plt.title("Global %s across update iteration" %metric)
    # plt.ylim([0,500])


    fig.savefig(os.path.join(figure_result_dir, 'lifelines_global_eval_%s.png' %metric), dpi=1500 ,bbox_inches='tight')
    fig.savefig(os.path.join(figure_result_dir, 'lifelines_global_eval_%s.eps' %metric), dpi=1500 ,bbox_inches='tight')


    plt.close()

    return


def plot_local_results(local_result_list, figure_result_dir, metric):

    xdata = range(len(local_result_list[0])-1)
    ydata = local_result_list


    # fig = plt.figure(figsize=(7.2, 4.2))
    fig = plt.figure(figsize=(4,3))
    # c_list = ['go-', 'ro-', 'bo-']
    c_list = ['go-', 'ro-', 'bo-']
    c_hl_list = ['purple', 'r', 'b']
    for index, c_i, c_hl in zip(range(len(ydata)), c_list, c_hl_list):
        plt.plot(xdata, ydata[index][1:], color = c_hl, linestyle='-', marker="o", label = 'FedAvg, client %s' %(index+1))
        plt.axhline(y= ydata[index][0], color=c_hl, linestyle='--' , label = "Without aggr., client %s " %(index+1))

    plt.legend(loc = "lower right", fontsize=8)
    plt.ylabel("%s" %metric, fontsize=15)
    plt.xlabel("Iteration", fontsize=15)
    # plt.xticks(xdata, range(1, len(local_result_list[0])), fontsize=12, rotation=70 )

    x_for_ticks = range(1, len(local_result_list[0]), 2)
    plt.xticks(x_for_ticks, range(2, len(local_result_list[0])+1, 2), fontsize=12)


    plt.ylim([0.74,0.8])
    yt = [0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80]

    plt.yticks(yt, fontsize=12)
    # plt.title("Local %s across update iteration" %metric)
    # plt.ylim([0,500])

    fig.savefig(os.path.join(figure_result_dir, 'lifelines_local_eval_%s.png' %metric), dpi=1500 ,bbox_inches='tight')
    fig.savefig(os.path.join(figure_result_dir, 'lifelines_local_eval_%s.eps' %metric), dpi=1500 ,bbox_inches='tight')


    plt.close()

    return

# get path of current directory
# current_path = Path(__file__).parent
current_dir = os.path.dirname(os.path.abspath(__file__))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_fold', default= 10, help='k of k-fold for ci', type=int)
    parser.add_argument('-n_train', default= 118584, help='k of k-fold for ci', type=int)
    parser.add_argument('-n_test', default= 14823, help='k of k-fold for ci', type=int)
    args = parser.parse_args()
    # print("Arguments:",args)
    n_fold = args.n_fold
    
    N_train = args.n_train
    N_test = args.n_test

    ttest_dir = os.path.join(current_dir, "ttest_ci")

    global_ci_list = []
    local_ci_list = []

    for k in range(n_fold):
        global_k = np.load(os.path.join(ttest_dir, "global_ci_%s.npy" %k))
        local_k = np.load(os.path.join(ttest_dir, "local_ci_%s.npy" %k))
        # print ("global_k", global_k)
        global_ci_list.append(global_k)
        local_ci_list.append(local_k)

    global_results_array = np.array(global_ci_list)
    local_results_array = np.array(local_ci_list)

    # print ("global_results_array", global_results_array)
    print ("global_results_array.shape", global_results_array.shape)

    glo_result_avg_list = []

    for u in range(global_results_array.shape[1]):
        result_u = global_results_array[:, u]
        print ("u", u)
        print ("result_u", result_u)

        result_u_avg = np.average(result_u)
        print ("result_u_avg", result_u_avg)
        glo_result_avg_list.append(result_u_avg)



        CI = compute_confidence(result_u, N_train, N_test, alpha=0.95)
        print ("CI", CI)

    plot_global_results(glo_result_avg_list, ttest_dir, "C-statistic")


    ## Local ci computation and plotting
    print ("local_results_array.shape", local_results_array.shape)

    num_train_list =[59292, 35575, 23716]
    num_test_list =[7412, 4447, 2965]


    local_result_list = []
    for o in range(local_results_array.shape[2]):
        print ("org: ", o)
        local_result_clinet = local_results_array[:,:,o]

        N_train = num_train_list[o]
        N_test = num_test_list[o]


        local_result_org_list = []
        for u in range(local_result_clinet.shape[1]):
            result_u = local_result_clinet[:, u]
            # print ("u", u)
            # print ("result_u", result_u)

            result_u_avg = np.average(result_u)
            print ("result_u_avg", result_u_avg)
            local_result_org_list.append(result_u_avg)

            CI = compute_confidence(result_u, N_train, N_test, alpha=0.95)
            print ("CI", CI)
    
        local_result_list.append(local_result_org_list)   

    plot_local_results(local_result_list, ttest_dir, "C-statistic")

if __name__ == '__main__':
    main()    

