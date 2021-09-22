from active_fit.fit import fit
from active_fit.fit_environment.FitEnvironment import FitEnvironment
from active_fit.initialize import initialize

if __name__ == '__main__':
    fit_environment: FitEnvironment = initialize()
    fit(fit_environment)
