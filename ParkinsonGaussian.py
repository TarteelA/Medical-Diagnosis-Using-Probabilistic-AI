#Import Libraries
import numpy as np 
import matplotlib.pyplot as plt

#Define Sample Gaussian
def sample_gaussian(mu, sigma):
    U1 = np.random.uniform(0, 1)
    U2 = np.random.uniform(0, 1)
    Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    return mu + sigma * Z0

#Define Multivariate Gaussian PDF
def multivariate_gaussian_pdf(x, mean, covariance):
    D = len(mean)
    cov_det = np.linalg.det(covariance)
    cov_inv = np.linalg.inv(covariance)
    x_no_mean = np.array(x) - np.array(mean)
    exponent = -0.5 *np.dot(x_no_mean.T, np.dot(cov_inv, x_no_mean))
    normalisation_factor = 1 / ((2*np.pi) ** (D/2) * (cov_det ** (1/2)))
    pdf_value = normalisation_factor * np.exp(exponent)
    return pdf_value

mean = [2, 2]
covariance = [[1, 0.6], [0.6, 1]]
data = {'num_samples':196, 'x1':[], 'x2':[], 'pdfs':[]}
for i in range(0, data['num_samples']):
    x1 = sample_gaussian(mean[0], np.sqrt(covariance[0][0]))
    x2 = sample_gaussian(mean[1], np.sqrt(covariance[1][1]))
    pdf_value = multivariate_gaussian_pdf([x1,x2], mean, covariance)
    print("i %s x %s sample %s" % (i, [x1,x2], pdf_value))
    data['x1'].append(x1)
    data['x2'].append(x2)
    data['pdfs'].append(pdf_value)