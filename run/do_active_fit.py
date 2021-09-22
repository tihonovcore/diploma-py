from active_fit.fit import fit
from active_fit.initialize import initialize, FitEnvironment

if __name__ == '__main__':
    fit_environment: FitEnvironment = initialize()
    fit(fit_environment)
