import pandas as pd
import numpy as np
import os
import datetime
import json

# Plotting libraries & functions
import plotly.offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import cufflinks as cf
from IPython.display import HTML,IFrame
import IPython.display

import cufflinks as cf ; cf.go_offline() 
from sklearn.decomposition import PCA
import plotly.graph_objects as go

import sys
# sys.path.insert(1, '../data_formatters')
# sys.path.insert(1, '../libs')
from data_formatters.base import GenericDataFormatter, DataTypes, InputTypes
from data_formatters.traffic import TrafficFormatter
from libs import utils
import sklearn.preprocessing
import numpy as np


import tensorflow as tf
from tensorflow.python.framework import ops
from libs.tft_model import TemporalFusionTransformer


# Suppress warnings in cells to improve readability
import warnings  
warnings.filterwarnings('ignore') 


def process_covid_data(df):
    df["date"] = pd.to_datetime(df["date"])
    # Aggregate county-level deaths and cases to state level
    df = df.drop(["county", "fips"], axis=1)
    df = df.groupby(["date", "state"]).sum().reset_index()
    # For simplicity, instead of imputing missing mobility and deaths data, we just remove Guam
    df = df.loc[df['state'] != 'Guam']
    return df

def process_mobility_data(df):
    df["date"] = pd.to_datetime(mobility_df["date"])
    # Aggregate county-level mobility to state level
    # It would definitely be better to take some form of mean weighted by county population here
    df = df.groupby(['admin1', 'date']).agg('mean').reset_index()
    df = df[["admin1", "date", "m50", "m50_index"]]
    df = df.rename(columns={'admin1': 'state'})
    # We make this change to be consistent with other data to make merging clean
    df.loc[df['state'] == 'Washington, D.C.', "state"] = "District of Columbia"
    return df

def process_bed_data(df):
    # Since the hospital data uses PO codes, we merge this data with a po code/state map
    # to get state name instead
    po_state_map = pd.read_json("../test_data/po_code_state_map.json", orient='records')
    df = df.merge(po_state_map, how='inner', left_on="state", right_on="postalCode")
    df = df[["bedspermille", "state_y"]]
    df = df.rename(columns={"state_y": "state"})
    return df

# Read in the data
covid_df = pd.read_csv(f"../test_data/nyt_us_counties_daily.csv")
bed_df = pd.read_csv(f"../test_data/bed_densities.csv")
mobility_df = pd.read_csv(f"../test_data/DL-us-mobility-daterow.csv")

# Apply the above processing steps
covid_df = process_covid_data(covid_df)
bed_df = process_bed_data(bed_df)
mobility_df = process_mobility_data(mobility_df)

# Right join to restrict our data to dates in the mobility dataset. Another option would be
# to use a larger date range, but that would require imputing mobility data (the repo doesn't handle nans)
# and making a larger percentage of our death data just a sequence of 0's (for states with no death/cases)
# data
df = covid_df.merge(mobility_df, how='right', on=['state', 'date'])
# Add hospital data
df = df.merge(bed_df, how='left', on='state')
# The initial right join will add nans for states without case/death data in the date range
# of the mobility dataset, so we replace with 0's
df = df.fillna(value=0)


# In[20]:


# Add an id column so our formatter class (below) can include state as both an identifier and categorical data
df['id'] = df['state']
df['day_of_week'] = df['date'].dt.dayofweek
df = df.sort_values(by='date')
df.head()


# In[21]:


df[df["state"]=="Arizona"].head()


# # Code to tell this repo how to use this data
# 
# Examples can be found in the repo in the data_formatters directory

# In[26]:

# This class must inherit from GenericDataFormatter and implement the methods given below
# or NotImplemented errors will be raised
class covidFormatter(GenericDataFormatter):
    """Defines and formats data for the covid dataset"""
    _column_definition = [
        ('id', DataTypes.CATEGORICAL, InputTypes.ID),
        ('date', DataTypes.DATE, InputTypes.TIME),
        ('deaths', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('cases', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('m50', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('m50_index', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('state', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('bedspermille', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
    ]
    
    def split_data(self, df):
        """Split data frame into training-validation data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
          df: Source data frame to split.
          valid_boundary: Starting date for validation data

        Returns:
          Tuple of transformed (train, valid) data.
        """
        print('Formatting train-valid splits.')
        
        # This function is meant to provide functionality for splitting
        # the data into train/valid/test along date boundaries. To keep consistent date ranges
        # for each identifier however, splitting the data would require
        # designating contiguous chunks of time as train/valid/test.
        # However, with such a small date range as is (not to mention
        # such a split would ensure that test/valid are not at all representative of
        # train since the date ranges would be different), splitting 
        # by date is just not feasible
        
        # Instead, we just do nothing and make no split. Since the model fit
        # function requires validation data, we just duplicate the train data.
        # This is clearly not optimal, and a clear way to improve on this simple example
        
        # The best way to split data would likely be along state levels,
        # but unfortunately this repo is very finicky with categorical data
        # and would not be happy with train and valid having different state
        # categories, so some workarounds would have to be made
        
        self.set_scalers(df)
        return (self.transform_inputs(data) for data in [df.copy(), df.copy()])


    

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

        Args:
          df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data...')
        # Code from their examples
        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        # For real-valued inputs (including our target), 
        # fit a transformation to scale to unit variance and zero mean
        # This is just the fitting step, the actual transformation can
        # be (or not be) applied in the next function
        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values)  

        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        # Fit an encoder to one-hot encode categorical inputs
        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].astype(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
              srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes
        
    def transform_inputs(self, df):
        """Performs feature transformations.

        Args:
          df: Data frame to transform.

        Returns:
          Transformed data frame.

        """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Use the previously fit StandardScaler() to transform the data if desired
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)
        output[real_inputs] = output[real_inputs]

        # Use the previously fit LabelEncoder()
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output
    

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

        Args:
          predictions: Dataframe of model predictions.

        Returns:
          Data frame of unnormalised predictions.
        """
        output = predictions.copy()

        column_names = predictions.columns
        # Use the inverse transform of our scaler to get back original scale
        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = self._target_scaler.inverse_transform(predictions[col])

        return output
    
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps':21,     # Total width of the Temporal Fusion Decoder
            'num_encoder_steps': 14,    # Length of LSTM decoder (ie. # historical inputs)
            'num_epochs': 30,            # Max number of epochs for training 
            'early_stopping_patience': 5, # Early stopping threshold for # iterations with no loss improvement
            'multiprocessing_workers': 5  # Number of multi-processing workers
        }

        return fixed_params
    


# In[27]:


# Instatiate our custom class and prepare training data
data_formatter = covidFormatter()
train, valid  = data_formatter.split_data(df)


# In[28]:


data_params = data_formatter.get_experiment_params()
# Model parameters for calibration

# Another parameter you could set here is "quantiles",
# right now it just predicts the default quantiles
model_params = {'dropout_rate': 0.1,      # Dropout discard rate
                'hidden_layer_size': 50, # Internal state size of TFT
                'learning_rate': 0.01,   # ADAM initial learning rate
                'minibatch_size': 64,    # Minibatch size for training
                'max_gradient_norm': 100.,# Max norm for gradient clipping
                'num_heads': 2,           # Number of heads for multi-head attention
                'stack_size': 1,           # Number of stacks (default 1 for interpretability)
               }

# Folder to save network weights during training.
model_folder = os.path.join('saved_models', 'covid', 'fixed')
model_params['model_folder'] = model_folder

model_params.update(data_params)


# # TFT Training

# In[34]:


# In[35]:


tf.compat.v1.reset_default_graph()
with tf.Graph().as_default(), tf.compat.v1.Session() as sess:

    tf.compat.v1.keras.backend.set_session(sess)
    
    # Create a TFT model with our parameters
    model = TemporalFusionTransformer(model_params)
                                    

    # We don't have much data so this caching functionality is really not necessary,
    # but why not. We could also just directly pass train and validation data to the model.fit() method
    if not model.training_data_cached():
        model.cache_batched_data(train, "train")
        model.cache_batched_data(valid, "valid")

    # Train and save model
    model.fit()
    model.save(model_folder)


# # TFT Prediction

# In[ ]:


tf.compat.v1.reset_default_graph()
with tf.Graph().as_default(), tf.compat.v1.Session() as sess:

    tf.compat.v1.keras.backend.set_session(sess)
    
    # Create a model with same parameters as we trained with & load weights
    model = TemporalFusionTransformer(model_params)
    model.load(model_folder)
    
    # Make forecasts
    output_map = model.predict(train, return_targets=True)

    targets = data_formatter.format_predictions(output_map["targets"])

    # Format predictions
    p50_forecast = data_formatter.format_predictions(output_map["p50"])
    p90_forecast = data_formatter.format_predictions(output_map["p90"])

    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
          col for col in data.columns
          if col not in {"forecast_time", "identifier"}
        ]]

    # Compute quantile losses using their functionality, but could easily be changed to pinball
    p50_loss = utils.numpy_normalised_quantile_loss(
        extract_numerical_data(targets), extract_numerical_data(p50_forecast),
        0.5)
    p90_loss = utils.numpy_normalised_quantile_loss(
        extract_numerical_data(targets), extract_numerical_data(p90_forecast),
        0.9)
print("Normalised quantile losses: P50={}, P90={}".format(p50_loss.mean(), p90_loss.mean()))


# # Visualizations and extracting interpretable attention weights

# In[ ]:


# Store outputs in maps
counts = 0
interpretability_weights = {k: None for k in ['decoder_self_attn', 
                                              'static_flags', 'historical_flags', 'future_flags']}
# Dictionary to hold weight information per state
per_state = {}

tf.compat.v1.reset_default_graph()
with tf.Graph().as_default(), tf.compat.v1.Session() as sess:

    tf.compat.v1.keras.backend.set_session(sess)
    
    # Create a new model with our same parameters & load weights
    model = TemporalFusionTransformer(model_params)
    model.load(model_folder)
    # Group by each state and extract the weights for each data point
    for identifier, sliced in train.groupby('state'):
        print("Getting attention weights for {}".format(identifier))
        weights = model.get_attention(sliced)
        per_state[identifier] = weights
        # We also keep a separate running weight sum in 'interpretability_weights'
        # to average across states
        for k in interpretability_weights:
            w = weights[k]
            
            # Average attention across heads if necessary
            if k == 'decoder_self_attn':
                w = w.mean(axis=0)
            
                batch_size, _, _ = w.shape                 
                counts += batch_size
            
            if interpretability_weights[k] is None:
                interpretability_weights[k] = w.sum(axis=0)
            else:
                interpretability_weights[k] += w.sum(axis=0)

interpretability_weight = {k: interpretability_weights[k] / counts for k in interpretability_weights}

print('Done.')


# In[14]:

# Some functions used in their code to look at attention weight values
def get_range(static_gate, axis=None):
    """Returns the mean, 10th, 50th and 90th percentile of variable importance weights."""
    return {'Mean': static_gate.mean(axis=axis), 
               '10%': np.quantile(static_gate, 0.1, axis=axis),
               '50%': np.quantile(static_gate, 0.5, axis=axis),
               '90%': np.quantile(static_gate, 0.9, axis=axis)}

def flatten(x):
    static_attn = x
    static_attn = static_attn.reshape([-1, static_attn.shape[-1]])
    return static_attn


# In[15]:


# Temporal Variable Importance -- static variables
static_attn = flatten(interpretability_weight['static_flags'])
m = get_range(static_attn, axis=0)
pd.DataFrame({k: pd.Series(m[k], index=['bedspermille', 'state']) for k in m})


# In[16]:


# Temporal Variable Importance -- variables known historically
x = flatten(interpretability_weight['historical_flags'])
m = get_range(x, axis=0)
pd.DataFrame({k: pd.Series(m[k], index=['deaths', 'cases', 'm50_index', 'm50', 'day_of_week']) for k in m})


# In[17]:


# Temporal Variable Importance -- variables known in the future (for this demo we have only 1, so 
# it will always be 1)
x = flatten(interpretability_weight['future_flags'])
m = get_range(x, axis=0)
pd.DataFrame({k: pd.Series(m[k], index=['day_of_week']) for k in m})


# In[18]:

init_notebook_mode(connected=False)  

def plotly_chart(df, title=None, kind='scatter', x_label=None, y_label=None, secondary_y=None, fill=None,
                shape=None, subplots=False, colors=['blue', 'red', 'purple'], fig_only=False):
    """Reusable dataframe plotting functionality"""
    fig = df.iplot(asFigure=True, title=title, kind=kind, xTitle=x_label, yTitle=y_label, secondary_y=secondary_y,
                  fill=fill, subplots=subplots, shape=shape, colors=colors)
    if fig_only:
        return fig
    else:
        iplot(fig)



def visualize_states(state_attention_weights):
    """Given the dictionary of attention weights per state, we use PCA to obtain a 3-dimensional
    representation of attention weight characteristics and scatterplot the result"""
    # Combine all states into one matrix where the ith row corresponds to the ith state 
    # under the previous one-hot encoding
    full_mat = np.stack([state_attention_weights[k]['decoder_self_attn'] for k in state_attention_weights.keys()])
    # Take the average across attention heads
    full_mat = full_mat.mean(axis=1)
    # Flatten the weight array and use PCA
    full_mat = full_mat.reshape(np.shape(full_mat)[0], -1)
    
    pca = PCA(n_components=3)
    fitted = pca.fit_transform(full_mat)
    # Convert the one-hot encoding back into state names
    encoder = data_formatter._cat_scalers['state']
    decoded = encoder.inverse_transform(np.arange(0, 51))
    
    fig = go.Figure(data=[go.Scatter3d(
        x=fitted[:, 0], 
        y=fitted[:, 1], 
        z=fitted[:, 2], 
        mode='markers', 
        text=decoded)])
    
    fig.update_layout(title='Per State Attention Weight Visualization')
                          
    fig.show()


# In[19]:


# Visualization of attention weights for each state
# The interactive plot allows you to mouse over any data point and see the state name it corresponds to
visualize_states(per_state)


# In[20]:


# Plot average attention pattern at different prediction horizons
self_attn = interpretability_weight['decoder_self_attn']
# Here we look at prediction 1, 3, 5, 7 days in the future
means = pd.DataFrame({"horizon={}".format(k): self_attn[model.num_encoder_steps + k - 1, :] 
                      for k in [1, 3, 5, 7]})
means.index -= model.num_encoder_steps

plotly_chart(means,
             x_label="Date Relative to Forecast Date",
             y_label="Mean Attention Weight",
             title="Average Attention Pattern at Various Prediction Horizons")


# In[21]:


def plot_historical_attention_weight(weights, index=0, name="deaths"):
    '''Given attention weights and the positional index of the variable among historical variables,
    plot how attention changes over time'''
    hist_weights = weights['historical_flags']
    hist_frame = pd.DataFrame({name: hist_weights[:, index]})
    hist_frame.index -= model.num_encoder_steps
    plotly_chart(hist_frame,
             x_label="Date Relative to Forecast Date",
             y_label="Mean Attention Weight",
             title=f"Attention over time for variable {name}")


# In[22]:


plot_historical_attention_weight(interpretability_weight, index=0, name="deaths")


# In[23]:


plot_historical_attention_weight(interpretability_weight, index=2, name="m50")


# In[24]:


# For the final visualizations, we're only going to look at New York data for simplicity
# What we're studying here is the idea of a "regime change" indicator, basically
# looking for a time where attention patterns change a lot which might
# indicate something interesting has changed
ny_data = df.loc[df["state"] == "New York"]
train, valid  = data_formatter.split_data(ny_data)


# In[ ]:


# Get attention weights over all the new york data
tf.compat.v1.reset_default_graph()

with tf.Graph().as_default(), tf.compat.v1.Session() as sess:

    tf.compat.v1.keras.backend.set_session(sess)
    
    # Create a new model & load weights
    model = TemporalFusionTransformer(model_params)
    model.load(model_folder)
    
    # Generate attention weights for test set
    print("Getting attention weights.")
    interpretability_weights = model.get_attention(train)
    # Average across multiple heads
    temporal_attention_weights = interpretability_weights['decoder_self_attn'].mean(axis=0)
    dates = interpretability_weights['time']
    print("Done.")


# In[26]:


# We can predict at most this many days in the future
max_forecast_horizon = model.time_steps - model.num_encoder_steps

# Attention weight by horizon
weights_by_horizon = {i + 1: temporal_attention_weights[..., model.num_encoder_steps + i, :] 
                                                      for i in range(max_forecast_horizon)}

# Compute average attention weights by horizon
average_by_horizon = {k: weights_by_horizon[k].mean(axis=0) for k in weights_by_horizon}

# Extract forecast dates
forecast_dates = dates[:, model.num_encoder_steps-1]

# Compute Bhattacharrya Coefficient-based distance metric (defined in the paper)
def compute_bhattacharyya_coeff(p, q):
    
    def norm(x):
        return x / np.sum(x)
    
    p_norm = norm(p)
    q_norm = norm(q)
    
    return np.sum(np.sqrt(p_norm * q_norm))

def compute_bhattacharyya_distance(p, q):
    
    coeff = compute_bhattacharyya_coeff(p, q)
    return np.sqrt(1 - coeff)

# Compute distance metric for each time step
distances = {}

for horizon in weights_by_horizon:
    weights = weights_by_horizon[horizon]
    aves = average_by_horizon[horizon]
    
    T = weights.shape[0]
    distance = [compute_bhattacharyya_distance(weights[t], aves)  for t in range(T)]
    distances[horizon] = pd.Series(distance, index=forecast_dates)
    
# Average distances across forecast horizons
distances = pd.DataFrame(distances, index=pd.to_datetime(forecast_dates)).mean(axis=1)


# In[27]:


# Get deaths from the original dataframe for the dates we care about
ny_deaths_cast = ny_data.loc[ny_data["date"].isin(forecast_dates)]
ny_deaths = ny_deaths_cast['deaths']
ny_deaths.index = ny_deaths_cast['date']
# Set a regime indicator -- with significant regimes = 1, and standard regimes = 0
significant_regimes = (distances > 0.15)*1

# Setup up dataframe for plotting
plot_order = ['Realized deaths', 'dist(t)', 'Significant Regimes']
plot_df = pd.DataFrame({'Realized deaths': ny_deaths,
                       'dist(t)': distances,
                       'Significant Regimes':significant_regimes}).dropna()[plot_order]


# In[28]:


# Visualise distance changes over time
fig = plotly_chart(plot_df, 
                   fig_only = True, 
                   title = 'Realized deaths versus dist(t)', 
                   secondary_y = ['dist(t)', 'Significant Regimes'],
                   y_label = 'Realized Deaths',
                   x_label = 'Forecast Date')

# Format shading for significant regimes
opacity=0.1
fig['data'][2]['line']['width']=0
fig['data'][2]['fill'] = 'tozeroy'
fig['data'][2]['fillcolor'] = fig['data'][2]['line']['color'].replace(', 1.0)', ', {})'.format(opacity))

# Format range of secondary y label
fig['layout']['yaxis2']['range'] = [0, 0.7]
iplot(fig)

fig.write_html('results/result1.html', auto_open=True)

# In[29]:


# Common functions
def extract_weights_on_date(forecast_date):
    
    for i, d in enumerate(forecast_dates):
        
        if pd.to_datetime(d) == pd.to_datetime(forecast_date):
            attention_weights = weights_by_horizon[1][i]

            return pd.Series(attention_weights, index=pd.to_datetime(dates[i, :]).date)
        
    raise ValueError("Cannot find weights on date {}".format(forecast_date))

def plot_attention_weights(weights, title):
    plot_df = pd.DataFrame({'Realized Deaths':ny_deaths,
                           'Attention Weights': weights}).dropna()[['Realized Deaths', 'Attention Weights']]

    fig=plotly_chart(plot_df, fig_only=True, shape=[2,1], subplots=True, title=title, colors=['blue', 'orange'])
    fig['data'][1]['fill'] = 'tozeroy'
    fig['layout']['yaxis2']['range'] = [0, 0.15]
    fig['layout']['yaxis1']['range'] = [0, 400]
    return fig


# In[30]:


# Forecast dates for each regime
standard_regime_date = pd.datetime(2020, 3, 15).date()
significant_regime_date = pd.datetime(2020, 3, 29).date()

# Plot representative weights for standard regime
weights = extract_weights_on_date(standard_regime_date)
weights_sig = extract_weights_on_date(significant_regime_date)

fig = plot_attention_weights(weights, 
                       title='One-step-ahead Attention Weights for Standard Regime (Forecast Date={})'.format(
                       standard_regime_date))

fig.show()


# In[31]:


# Plot representative weights for significant regime
fig2 = plot_attention_weights(weights_sig, 
                       title='One-step-ahead Attention Weights for Significant Regime (Forecast Date={})'.format(
                       significant_regime_date))

fig2.show()


# In[ ]:




