import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import datetime #import datetime
from datetime import time
import numpy as np
import scipy
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score


extra = True # when true, plots everything for three set ups, when false only plots a scaled down diagonal GMM

data = pd.read_csv("data.csv")
#data.plot()
dd = data.values

## reformat data to have inputs day number, time, time on. As information such as year and exact, milliseconds is not needed
dt = [datetime.datetime.fromtimestamp(ts) for ts in dd[:,1]] # convert to a datetime object

#calculate elapsed times for two periods, a day and a week as these are the patterns which may emerge. within one day, or between days in a week.
times = [int(ts.strftime('%H'))*60 + int(ts.strftime('%M'))   for ts in dt] ## elapsed time in minites since day start
byweek = [ts.weekday()*24*60 + int(ts.strftime('%H'))*60 + int(ts.strftime('%M'))  for ts in dt] ## elapsed time in minites since week start

weekdays = np.zeros((len(byweek),3))
weekdays[:,0] = [ts.weekday() for ts in dt] # make fire column the weekday
weekdays[:,1] = times
weekdays[:,2] = dd[:,2]

#visulise the data
plt.figure()
plt.subplot(211)
plt.scatter(times, dd[:,2],marker = 'x')
plt.subplot(212)
plt.scatter(byweek, dd[:,2],marker = 'x')
plt.show()

def t2HM(time):  #convert elapsed time in day back to H:M
    hour = time//60
    min = time - 60*hour
    return str(int(hour))+':'+str(round(min,1))

def t2DHM(time): #convert elapsed time in week to D:H:M
    day = time//(24*60)
    hour = (time-day*24*60)//60
    min = time - 60*hour - time-day*24*60
    return [day,hour,min]

colors = ['maroon', 'red', 'salmon'
    ,'orange','goldenrod','olive','lawngreen','g','mediumseagreen','aquamarine','c','aqua','blue','navy','darkviolet','violet','purple','fushia']
def make_ellipses(gmm, ax,label=False): #edited from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
    for n, color in enumerate(colors[0:gmm.n_components]):

        #format the covariance matrices
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]

        mean = gmm.means_[n, :2]
        if label:  # label the clusters
            s = 'mean = [' + t2HM(mean[0]) + r',%.f], $\sigma$ = [%.f,%.f], Cov=%.f' % (
            mean[1], np.sqrt(covariances[0, 0]), np.sqrt(covariances[1, 1]), covariances[0, 1])
            plt.scatter(mean[0], mean[1], marker='+', color=color, label=s)
        else:
            plt.scatter(mean[0],mean[1], marker ='+', color=color)


        #compute eigenvectors/values in order to plot an elipse
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        #ax.set_aspect('equal', 'datalim')


def build_model(n_day,Xd,i,shape='full'):
    # build a gaussian mixture model with n components, and repeat this 20 times to account for variation caused by mixture initialisation
    model = GaussianMixture(n_components=n_day, covariance_type=shape,
                                      n_init=40) # want either diag covariance - assumes variation in time on during a routine is independa
    model.fit(Xd)
    h = plt.subplot(4, 2, i + 1)
    log_liklihood = model.score_samples(Xd) # get the liklihood of each data point given the model, with 0 being most likely and -inf least
    plt.scatter(Xd[:, 0], Xd[:, 1], marker='x', c=log_liklihood, cmap='cool', vmin=min(log_liklihood), vmax=max(log_liklihood) )
    plt.colorbar(label='Log Likelihood')
    make_ellipses(model, h)
    h.set_ylim([-10, max(Xd[:,1])+25])
    cluster_labels = model.predict(Xd) #get the cluster for each data point
    sil_score = silhouette_score(Xd, cluster_labels) # calculate closeness within each cluster and average
    return model,sil_score

def plot_model(model_days,Xd,ns,day,shape='full'):
    plt.figure()
    sil_score = [] # for storing the silhouette score of each number of clusters, this is the mean eucldian distance winin a cluster, a larger value indicates a possibly better fit with that number of clusters
    # a measure of closeness and cluster variation
    for i in range(len(ns)):
        if i != len(ns)-1:
            n_day = ns[i]
            model,score = build_model(n_day,Xd,i,shape)
            model_days.append(model)
            sil_score.append(score)
        else:
            plt.subplot(4, 2, i + 1)
            plt.xlabel('Number of clusters')
            plt.ylabel('Silhouette score')
            plt.plot(ns[0:-1], sil_score)
    titles = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
              'All days']
    plt.suptitle(titles[day])
    #plt.show()
    return model_days

ns = np.arange(4,12) # cluster numbers to test

def test_shape(shape,scale=1):
    #tests a gmm model with covariance matric of shape shape.
    #clusters on each day seperatly, and on all days overlapped
    # scale is to reduce the impact of duration on to encourage vertical clustering

    model_days=[] # to store all the models tested
    #cluster each day seperatly
    for day in range(7):
        index = np.where(weekdays[:,0]==day) # get the indices in the data that correspond to the current day
        Xd = np.zeros((len(index[0]), 2))
        Xd[:,0] = weekdays[index,1]
        Xd[:,1]= weekdays[index,2]/scale
        model_days = plot_model(model_days,Xd,ns,day,shape)

    # do all days overlaid
    X = np.zeros((dd[:,2].shape[0],2))
    X[:,0] = times
    X[:,1] = dd[:,2]/scale
    model_days = plot_model(model_days,X,ns,7,shape)
    plt.show()
    return model_days

def sumarise_results(model,n_opt,scale=1):
    titles = ['Monday: n=','Tuesday: n=','Wednesday: n=','Thursday: n=','Friday: n=','Saturday: n=', 'Sunday: n=', 'All days: n=']
    for i in range(len(titles)):
        if i != len(titles)-1: #format the correct inputs for each day
            index = np.where(weekdays[:, 0] == i)
            Xd = np.zeros((len(index[0]), 2))
            Xd[:, 0] = weekdays[index, 1]
            Xd[:, 1] = weekdays[index, 2]
        else:
            Xd = np.zeros((dd[:, 2].shape[0], 2))
            Xd[:, 0] = times
            Xd[:, 1] = dd[:, 2]
        h = plt.subplot(4, 2, i + 1)
        model_num = i * 7 + n_opt[i] - ns[0]  # the opt model to plot
        log_liklihood = model[model_num].score_samples(Xd / [1,scale])

        #rescale the covariances and mean
        model[model_num].means_[:,1] = model[model_num].means_[:,1]*scale

        if len(model[model_num].covariances_[0].shape)==3: #its a full cov
            for k in range(model[model_num].covariances_.shape):
                pass
                #not sure how to rescale the full matrix, but since im not using it it doesnt matter
                #model[model_num].covariances_[0][k,1,1] = model[model_num].covariances_[0][k,1,1] * scale**2
                #model[model_num].covariances_[0][k,1, 0] = model[model_num].covariances_[0][k,1, 0] * scale
                #model[model_num].covariances_[0][k,0, 1] = model[model_num].covariances_[0][k,0, 1] * scale
        else:
            model[model_num].covariances_[:, 1] = model[model_num].covariances_[:, 1] * scale ** 2

        #plot data
        plt.scatter(Xd[:, 0], Xd[:, 1], marker='x', c=log_liklihood, cmap='cool', vmin=min(log_liklihood), vmax=max(log_liklihood))
        plt.colorbar(label='Log Likelihood')


        make_ellipses(model[model_num], h,True)
        leg=plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          ncol=2, prop={"size":5})
        #colour legend
        t=0
        for text in leg.get_texts():
            plt.setp(text, color=colors[t])
            t+=1
        plt.title(titles[i]+str(n_opt[i]))
    plt.subplots_adjust(hspace=1)
    plt.show()


if extra:
 model_full = test_shape('full')
 model_diag = test_shape('diag')



n_opt_full = [7,6,5,5,6,6,5,5] # the highest sil_score for each day, indicating all days have 5-7 clusters
#sunday had the highest score at 8 but this appeared to be overfitting the data

# also good ones
# n_opt_full = [5,7,4,5/7,6,6/4,5,5]

n_opt_diag =[5,6,5,5,7,5,5,5] # smaller on average then with a full model


#other methods could be used to determine best number of clusters such as baysian information criterion (BIC), this says how good at predicting is our model for the data provided taking into account model complexity
#or testing variation between gmm's of a given cluster number

# other models such as baysian gaussian mixture model could be used, these have some model bias but with the right priors can produce better clustering without n=knowing the cluster number. incorporate regularisation and priors in addition to EM

if extra:
    sumarise_results(model_full,n_opt_full)
    sumarise_results(model_diag,n_opt_diag)

#i noticed that groups that could be considered clusters as they occured at the same time were often seperated as they happened at a very different length
#in order to make time on less important, i will shrink it by a factor to encourage vertical clustering
#the fully over lapped data had the same reults with both scalings, day patterns changed

model_diag_scaled = test_shape('diag',5)
n_opt_ds =[5,6,5,6,6,6,5,5]
sumarise_results(model_diag_scaled,n_opt_ds,5)

##plotting means of similar clusters across days, could do this automatically by grouping by cluster mean etc, but wanted to make sure the right groups were made

mean_time = np.array (
[ [datetime.time(7,59,0),datetime.time(7,52,0),datetime.time(7,50,0),datetime.time(7,38,0),datetime.time(7,25,0),datetime.time(8,18,0),datetime.time(7,52,0),datetime.time(7,52,0)],
[datetime.time(11,25,0),datetime.time(11,11,0),None,datetime.time(10,51,0),datetime.time(10,35,0),None,None,datetime.time(12,17,0)],
[None,datetime.time(12,52,0),datetime.time(12,49,0),datetime.time(13,26,0),datetime.time(13,51,0),datetime.time(12,30,0),datetime.time(13,4,0),None],
[datetime.time(15,23,0),datetime.time(15,30,0),datetime.time(16,50,0),None,datetime.time(16,11,0),datetime.time(17,3,0),datetime.time(16,31,0),datetime.time(16,28,0)],
[datetime.time(19,7,0),datetime.time(19,10,0),None,datetime.time(17,54,0),datetime.time(20,49,0),datetime.time(18,47,0),datetime.time(19,31,0),datetime.time(19,35,0)],
[datetime.time(22,35,0),datetime.time(22,57,0),datetime.time(22,9,0),datetime.time(22,44,0),datetime.time(23,21,0),datetime.time(23,4,0),datetime.time(22,32,0),datetime.time(22,51,0)]
])
mean_dur = np.array ([
[140,192,261,301,144,159,141,168],
[94,100,None,110,125,None,None,109],
[None,119,191,119,104,112,101,None],
[54,102,165,None,132,167,72,108],
[143,132,None,76,109,68,127,110],
[116,79,94,116,66,80,81,92]
])
sd_time = np.array([
[11,24,23,23,23,38,41,32],
[36,9,None,37,36,None,None,85],
[None,26,62,29,8,37,39,None],
[65,19,34,None,20,5,12,45],
[30,73,None,76,86,17,10,58],
[42,17,47,14,15,9,29,29]
])
sd_dur = np.array([
[123,121,108,19,118,118,121,125],
[65,83,None,73,58,None,None,66],
[None,36,13,6,56,73,67,None],
[41,70,12,None,48,22,49,68],
[93,71,None,49,14,32,36,62],
[51,48,58,55,48,50,52,57]
])


days =[1,2,3,4,5,6,7,8]
plt.figure()
plt.subplot(2,2,1)
plt.title('Mean time')
for k in range(mean_time.shape[0]):
    plt.plot(days,mean_time[k,:])
plt.subplot(2,2,2)
plt.title('Mean duration')
for k in range(mean_time.shape[0]):
    plt.plot(days,mean_dur[k,:])
    plt.ylabel('Time/s')
plt.subplot(2,2,3)
plt.title(r'$\sigma$ (time) ')
for k in range(mean_time.shape[0]):
    plt.plot(days,sd_time[k,:])
    plt.ylabel('Time/min')
plt.subplot(2,2,4)
plt.title(r'$\sigma$ (duration) ')
for k in range(sd_time.shape[0]):
    plt.plot(days,sd_dur[k,:])
    plt.xlabel('Day, where 8=Alldays')
    plt.ylabel('Time/s')
plt.show()

#conclusions
# the morning and evening usage of the kettle is more regular with similar clusters/routines appearing each day with similar variability (blue/brown lines are almost constant)
# routines during the day, seem to depend on the day in question and are often less distinct
#

#find outliers/annomalies
#can do this by looking for the extrames(most negative) of log liklihood as these are the furtherest away from cluster centre
#or by looking for clusters with 0 variances as this only one point within a cluster
#some have been highlighted in an attached document


#future work
# - # could plot similar group means across days to find the daily trends
# - test out baysian gausiian mixture models
# - test out periodic mixture models, taking the unformatted data as an input and letting the algothym decide the period in k clusters
# - try modelling/fitting the data with a function/probabilty of the kettle being used at certain times, and then analysing the patterns in this function
# - remove outliers from clustering to get a better estimatipon of a routines variability