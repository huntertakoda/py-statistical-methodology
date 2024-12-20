import pandas as pd
import pymc as pm
import arviz as az
import numpy as np

def bayesian_analysis():
    
    # load the preprocessed dataset

    file_path = "C:/puredata/statistical_methodology_preprocessed.csv"
    data = pd.read_csv(file_path)

    # select relevant columns for bayesian analysis

    income = data["income"].values
    job_satisfaction = data["job_satisfaction"].values

    # build the bayesian model

    print("building bayesian model...")
    with pm.Model() as model:
        # define priors
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        slope = pm.Normal("slope", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        
        # define likelihood
        mu = intercept + slope * income
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=job_satisfaction)
        
        # perform posterior sampling with single-threaded sampling
        trace = pm.sample(1000, tune=500, return_inferencedata=True, random_seed=42, cores=1)
        print("sampling complete.")

    # summarize results

    print("model summary:")
    summary = az.summary(trace, var_names=["intercept", "slope", "sigma"])
    print(summary)

    # save posterior summary to a csv file

    output_path = "C:/puredata/statistical_methodology_bayesian_summary.csv"
    summary.to_csv(output_path)
    print(f"posterior summary saved to: {output_path}")

    # generate and save posterior plots

    print("generating posterior plots...")
    az.plot_posterior(trace, var_names=["intercept", "slope", "sigma"])
    posterior_plot_path = "C:/puredata/statistical_methodology_posterior_plot.png"
    az.plot_trace(trace).figure.savefig(posterior_plot_path)
    print(f"posterior plots saved to: {posterior_plot_path}")

    # script execution complete

    print("bayesian statistics analysis completed successfully.")

if __name__ == '__main__':
    bayesian_analysis()


