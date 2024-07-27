import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#Share info
S0 = 100
sigma = 0.3
T = 1
r=0.06

#Algorithm info
N = 2**10
delta = 0.25
alpha = 1.5

def log_char(u):
    return np.exp(1j*u*(np.log(S0)+(r-sigma**2/2)*T)-sigma**2*T*u**2/2)

def c_func(v):
    val1 = np.exp(-r*T)*log_char(v-(alpha+1)*1j)
    val2 = alpha**2+alpha-v**2+1j*(2*alpha+1)*v
    return val1/val2

n = np.array(range(N))
delta_k = 2*np.pi/(N*delta)
b = delta_k*(N-1)/2

log_strike = np.linspace(-b,b,N)

x = np.exp(1j*b*n*delta)*c_func(n*delta)*(delta)
x[0] = x[0]*0.5
x[-1] = x[-1]*0.5

# fft from python libraries
xhat = np.fft.fft(x).real 

fft_call = np.exp(-alpha*log_strike)*xhat/np.pi

#call price
d_1 = (np.log(S0/np.exp(log_strike))+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
d_2 = d_1 - sigma*np.sqrt(T)
analytic_callprice = S0*norm.cdf(d_1)-np.exp(log_strike)*np.exp(-r*(T))*norm.cdf(d_2)

plt.plot(np.exp(log_strike), analytic_callprice) #LINE 1
plt.plot(np.exp(log_strike), fft_call) #LINE 2
plt.axis([0,100,0,100]) #LINE 3
plt.xlabel("Strike")
plt.ylabel("Call Price")
plt.show() #LINE 4


plt.plot(log_strike, np.absolute(fft_call-analytic_callprice)) #LINE 5
plt.xlabel("Log-Strike")
plt.ylabel("Absolute Error")
plt.show()

