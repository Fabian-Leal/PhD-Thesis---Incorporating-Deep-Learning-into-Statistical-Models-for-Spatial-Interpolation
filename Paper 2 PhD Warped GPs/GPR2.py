import numpy as np
from scipy.optimize import minimize
from scipy.optimize import root

class GPR2:
    def __init__(self, cov_function_name, x_train, y_train, hyper_params, sigma_n, warp_params=None, nu=None):
        self.cov_function_name = cov_function_name
        self.x_train = x_train
        self.y_train = y_train
        self.hyper_params = hyper_params  # params[0] is signal variance, params[1] is lengthscale
        self.nu = nu
        self.K = self.cov_func(self.x_train, self.x_train)
        self.sigma_n = sigma_n
        self.K_inv = np.linalg.pinv(self.K + np.eye(self.K.shape[0]) * (self.sigma_n ** 2))
        self.warp_params = warp_params
        if self.warp_params is None:
            self.f = y_train
            self.df_dy = 1
        else:
            self.gamma = warp_params
            self.a = self.gamma[0]
            self.b = self.gamma[1]
            self.c = self.gamma[2]
            self.I = len(self.gamma[0])
            self.f = self.hyp_tan(y_train)
            self.df_dy = self.hyp_tan_df_dy(y_train)


    def hyp_tan(self, y):
        f = np.zeros_like(y)  # Initialize f as an array of zeros

        for i in range(self.I):
            f += self.a[i] * np.tanh(self.b[i] * (y+ self.c[i]))

        return f

    def hyp_tan_df_dy(self, y):
        df_dy = np.zeros_like(y)  # Initialize df_dy as an array of zeros

        for i in range(self.I):
            df_dy += self.a[i] * self.b[i] * (1 - np.tanh(self.b[i] * (y + self.c[i])) ** 2)

        return df_dy

    def inverse_hyp_tan(self, f):
        def equation_to_solve(y, f):
            result = np.zeros_like(y)
            for i in range(self.I):
                result += self.a[i] * np.tanh(self.b[i] * (y + self.c[i]))

            return result - f

        result = root(equation_to_solve, f, args=(f,))

        if result.success:
            return result.x
        else:
            raise ValueError("Failed to find the inverse")

    def predict(self,x_star):
        k_star = self.cov_func(self.x_train,x_star)
        f_mean = np.dot(np.dot(np.transpose(k_star),self.K_inv),self.f)
        f_std = self.cov_func(x_star, x_star) - np.dot(np.dot(np.transpose(k_star),self.K_inv),k_star)
        f_std = np.sqrt(np.diag(f_std))[:,np.newaxis]
        return [f_mean,f_std]


    def predict_original(self,x_star):
        k_star = self.cov_func(self.x_train,x_star)
        f_mean = np.dot(np.dot(np.transpose(k_star),self.K_inv),self.f)
        f_std = self.cov_func(x_star, x_star) - np.dot(np.dot(np.transpose(k_star),self.K_inv),k_star)
        f_std = np.sqrt(np.diag(f_std))
        y_mean = self.inverse_hyp_tan(f_mean.reshape(-1))
        y_std = self.inverse_hyp_tan(f_std)
        return [y_mean,y_std]

    def cov_func(self,x_1,x_2):
        
        x_1_sq = np.sum(np.square(x_1),1)
        x_2_sq = np.sum(np.square(x_2),1)
        d = -2.*np.dot(x_1, x_2.T) + (x_1_sq[:,None] + x_2_sq[None,:])
        d = np.sqrt(np.clip(d, 0, np.inf))

        if self.cov_function_name == "Squared Exponential":
            K = self.hyper_params[0]* np.exp(-0.5 * (d/self.hyper_params[1])**2)

        if self.cov_function_name == "Matern":

            if self.nu == 5/2:
                K = self.hyper_params[0] *(1 + np.sqrt(5) * d / self.hyper_params[1] + 5 * d ** 2 / (3 * self.hyper_params[1] ** 2)) \
                        * np.exp(-np.sqrt(5) * d / self.hyper_params[1])

            elif self.nu == 1/2:
                K = self.hyper_params[0] *np.exp(-0.5*(d/self.hyper_params[1])**2)
    
            else:
                K = self.hyper_params[0] *np.exp(-0.5*(d/self.hyper_params[1])**2)

                print('invalid nu')
                
        return K

    def set_hyper_params(self, hyper_params, sigma_n, warp_params = None):
        self.hyper_params = hyper_params
        self.K = self.cov_func(self.x_train,self.x_train)
        self.sigma_n = sigma_n
        self.K_inv = np.linalg.pinv(self.K+np.eye(self.K.shape[0])*(self.sigma_n**2))
        if warp_params is None:
            self.df_dy = 1
        else:
            self.gamma = warp_params    
            self.a = self.gamma[0]
            self.b = self.gamma[1]
            self.c = self.gamma[2]
            self.f = 0
            self.df_dy = 0
            for i in range(len(self.gamma[0])):
                self.f += self.a[i] * np.tanh(self.b[i] * (self.y_train + self.c[i])) 
                self.df_dy += self.a[i] * self.b[i] * (1 - np.tanh(self.b[i] * (self.y_train + self.c[i])) ** 2)
                
    def log_marginal_likelihood(self):
        lml1 = -.5*np.dot(np.dot(np.transpose(self.f),self.K_inv),self.f)
        lml2 = -.5*np.log(np.linalg.det(self.K+np.eye(self.K.shape[0])*(self.sigma_n**2)))
        lml3 = -.5*self.x_train.shape[0]*np.log(2*np.pi)
        lml4 = np.sum(np.log(self.df_dy))
        return lml1+lml2+lml3+lml4
    
    def fit(self):
        if self.warp_params is None:
            def obj_func(params):
                params = np.exp(params)
                self.set_hyper_params([self.hyper_params[0],params[0]], self.sigma_n)
                nlml = -self.log_marginal_likelihood()
                #print(str(params)+str(nlml))
                return nlml
 
            x0 = np.array([np.log(self.hyper_params[1])])
            self.res = minimize(obj_func, x0, method='Powell',
                        options={'disp': True})
            self.optimal_params = np.exp(self.res.x)
            self.set_hyper_params([1,self.optimal_params[0]],self.sigma_n)
        else:
            def obj_func(params):
                params = np.exp(params)
                a = [params[1+3*i] for i in range(self.I)]
                b = [params[2+3*i] for i in range(self.I)]
                c = [params[3+3*i] for i in range(self.I)]
                self.set_hyper_params([self.hyper_params[0],params[0]], params[1], [a,b,c])
                nlml = -self.log_marginal_likelihood()
                print(str(params)+str(nlml))
                return nlml

            #bnds = [(None, None)] * 2  # Example bounds for the first two variables
            #bnds += [(0, 10000), (0, 10000), (0, 10000)] * self.I  # Bounds for a, b, c

            x0 = np.array([np.log(self.hyper_params[1]), np.log(self.sigma_n)] +  [val for i in range(self.I) for val in [np.log(self.a[i]), np.log(self.b[i]), np.log(self.c[i])]])
            self.res = minimize(obj_func, x0, method='Nelder-Mead', 
                        options={'disp': True})
            
        
            
            self.optimal_params = np.exp(self.res.x)        
            a = [self.optimal_params[1+3*i] for i in range(self.I)]
            b = [self.optimal_params[2+3*i] for i in range(self.I)]
            c = [self.optimal_params[3+3*i] for i in range(self.I)]
            self.set_hyper_params([1,self.optimal_params[0]], self.sigma_n, [a,b,c])


        return self.optimal_params

