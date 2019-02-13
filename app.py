from flask import Flask, render_template, request
from bokeh.plotting import figure
from bokeh.embed import components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from bokeh.models import ColumnDataSource, NumeralTickFormatter
from bokeh.palettes import Spectral6, Pastel1
from sklearn.preprocessing import StandardScaler
from bokeh.models import ColumnDataSource
from bokeh.plotting import show, output_notebook, figure as bf
from bokeh.plotting import figure
from bokeh.models.glyphs import  Text
from bokeh.layouts import row, widgetbox
from bokeh.models import CustomJS, Slider
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models.widgets import Slider, TextInput
from bokeh.models import HoverTool
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, ranges, LabelSet
import math
import scipy.special

app = Flask(__name__)

with open("rf.pkl", "rb") as f:
    rf = pickle.load(f)

ytrain = pd.read_pickle('ytrain.pkl')
ytest = pd.read_pickle('ytest.pkl')
X_train_scaled = np.load('X_train_scaled.npy')
X_test_scaled = np.load('X_test_scaled.npy')

df = pd.read_pickle('data_for_regression_20k.pkl')

@app.route('/')  
def index():
    return render_template('index.html')

@app.route('/EDA') 
def EDA():
    # A. Distribution of destinations in rest of the world
    df['booking'] = 1
    dest = df.groupby(['country_dest_id'])['booking'].count().reset_index()
    dest['pct_of_total'] = [(x/dest['booking'].sum())*100 for x in dest['booking']]
    dest = dest.sort_values(by='pct_of_total', ascending = False)
    x_lab = {'1.0': 'USA',  
             '2.0':'Other', 
             '3.0':'France',
             '7.0':'Italy', 
             '5.0':'Great Britain',
             '6.0':'Spain',
             '4.0':'Canada',
             '10.0':'Denmark',
             '9.0':'Netherlands',
             '11.0':'Australia',
             '8.0':'Portugal'}
    dest.country_dest_id = [str(X) for X in dest.country_dest_id]
    dest['label'] = dest['country_dest_id'].map(x_lab)  
    dest_bookings = dest[(dest.country_dest_id != '1.0') & (dest.country_dest_id != '4.0')]

    colors = ['mediumvioletred', 'palevioletred', 'palevioletred', 'palevioletred','palevioletred',
            'lightpink', 'lightpink','lightpink', 'lightpink']
    source = ColumnDataSource(data=dict(x=list(dest_bookings.label), counts=list(dest_bookings.pct_of_total), colors=colors))
    
    plot = figure(x_range = source.data['x'], y_range = (0,12), title="Destinations outside of USA/Canada", 
                    plot_height=200,plot_width=500)

    plot.vbar(x = 'x', bottom=0, top='counts', width=0.5, color='colors', source=source)
    plot.add_tools(HoverTool(tooltips=[("LOCATION", "@x"), ("PCT BOOKED", "@counts %")]))
    plot.xaxis.major_label_orientation = math.pi/4
    plot.xaxis.major_label_text_font_size = "10pt"
    plot.yaxis.minor_tick_line_color = None
    #plot.xaxis.axis_label = 'Destination country'
    #plot.xaxis.axis_label_text_font_size = "20pt"
    plot.yaxis.axis_label = 'Percent of total bookings'
    #plot.xaxis.axis_label_text_font_size = "20pt"

    x_lab_2 = {'1.0': 'USA/Canada',  
             '2.0':'International', 
             '3.0':'International',
             '7.0':'International', 
             '5.0':'International',
             '6.0':'International',
             '4.0':'USA/Canada',
             '10.0':'International',
             '9.0':'International',
             '11.0':'International',
             '8.0':'International'}
    dest['label_2'] = dest['country_dest_id'].map(x_lab_2)  
    dest_bookings_2 = dest.groupby(['label_2'])['pct_of_total'].sum().reset_index()
    
    colors = ['palevioletred','olive']
    source = ColumnDataSource(data=dict(y=list(dest_bookings_2.label_2), counts=list(dest_bookings_2.pct_of_total), colors=colors))
    plot2 = figure(y_range = source.data['y'], x_range=(0,83), title="75 percent of users booked AirBnbs in the USA/Canada", 
                    plot_height=200,plot_width=500)
    plot2.hbar(y = 'y', left=0, right='counts', height=0.9, color='colors', source=source)
    plot2.add_tools(HoverTool(tooltips=[("Location", "@y"), ("Percent", "@counts{1.0}%")]))

    #plot2.yaxis.axis_label = 'Destination country'
    plot2.xaxis.axis_label = 'Percent of total bookings'
    plot2.xaxis.minor_tick_line_color = None
    plot2.yaxis.major_tick_line_color = None
    plot2.xaxis.major_tick_line_color = None
    plot2.xgrid.grid_line_color = None
    plot2.ygrid.grid_line_color = None
    layout = column(plot2, plot)   
    curdoc().add_root(layout)

    # B. Typical age of user
    def make_histogram(title, hist, edges, x, x_name, pdf):
        p = figure(title=title, tools='', background_fill_color="#fafafa", plot_height=300,plot_width=350)
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color="olive", line_color="white", alpha=0.8)
        p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend="PDF")
        p.y_range.start = 0
        p.legend.location = "bottom_right"
        p.legend.background_fill_color = "#fefefe"
        p.xaxis.axis_label = x_name
        p.yaxis.axis_label = 'Pr(x)'
        p.grid.grid_line_color="white"
        return p
    def get_pdf(measured):
        mu = measured.mean()
        sigma = np.std(list(measured))
        x = np.linspace(min(measured), max(measured), len(measured))
        pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
        cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2
        return(x, mu, sigma, pdf)

    measured = df.age
    measured_0 = df.age[df.country_USA_World_bi==0]
    measured_1 = df.age[df.country_USA_World_bi==1]
    
    x, mu, sigma, pdf = get_pdf(measured)
    x0, mu0, sigma0, pdf0 = get_pdf(measured_0)
    x1, mu1, sigma1, pdf1 = get_pdf(measured_1)
    
    age_dist = figure(title="Users to all destinations skew young (μ=%d)"%mu, tools='', 
                    background_fill_color="#fafafa", plot_height=250,plot_width=350, y_range = (0,0.04))
    age_dist.line(x1, pdf1, line_color="#ff8888", line_width=4, alpha=0.7, legend="Rest of world")
    age_dist.line(x0, pdf0, line_color="olive", line_width=4, alpha=0.7, legend="USA/Canada")
    age_dist.xaxis.axis_label = 'Age'
    age_dist.yaxis.major_tick_line_color = None
    age_dist.yaxis.major_label_text_font_size = '0pt'
    age_dist.yaxis.minor_tick_line_color = None
    age_dist.xaxis.minor_tick_line_color = None
    #Number of sessions
    measured = df.num_sessions
    measured_0 = df.num_sessions[df.country_USA_World_bi==0]
    measured_1 = df.num_sessions[df.country_USA_World_bi==1]
    
    x, mu, sigma, pdf = get_pdf(measured)
    x0, mu0, sigma0, pdf0 = get_pdf(measured_0)
    x1, mu1, sigma1, pdf1 = get_pdf(measured_1)
    
    s_dist = figure(title="Users sessions (μ=%d)"%mu, tools='', 
                    background_fill_color="#fafafa", plot_height=300,plot_width=350, y_range = (0,0.01))
    s_dist.line(x1, pdf1, line_color="#ff8888", line_width=4, alpha=0.7, legend="Rest of world")
    s_dist.line(x0, pdf0, line_color="olive", line_width=4, alpha=0.7, legend="USA/Canada")
    s_dist.xaxis.axis_label = 'Age'
    s_dist.yaxis.major_tick_line_color = None
    s_dist.yaxis.major_label_text_font_size = '10pt'
    s_dist.yaxis.minor_tick_line_color = None
    s_dist.xaxis.minor_tick_line_color = None

    def make_bar(var, title):
        label = {'1.0':'Rest of world', '0.0':'USA/Canada'}
        colors = ['olive','palevioletred']
        data2 = df.groupby('country_USA_World_bi')[var].agg('mean').reset_index()
        data2['country_USA_World_bi'] = [str(X) for X in data2.country_USA_World_bi]
        data2[var] = [int(X) for X in data2[var]]
        data2['loc'] = data2['country_USA_World_bi'].map(label)  
        source = ColumnDataSource(data=dict(loc=list(data2['loc']), amt=list(data2[var]), colors=colors))
        p = figure(x_range = source.data['loc'], y_range=(0,1.2*max(source.data['amt'])), title=str(title), plot_height=250,plot_width=300)
        p.vbar(x = 'loc', bottom=0, top='amt', width=0.4, color='colors', source=source)       
        labels = LabelSet(x='loc', y='amt', text='amt', level='glyph',
                        x_offset=-10, y_offset=0, source=source, render_mode='canvas')
        p.add_layout(labels)
        p.yaxis.minor_tick_line_color = None
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.yaxis.major_tick_line_color = None
        p.xaxis.major_label_text_font_size = "10pt"
        return(p)
    s=make_bar('num_sessions', title='Average number of sessions by destination')


    layout2 = row(age_dist, s)   
    curdoc().add_root(layout2)


    javascript, div = components(layout)
    javascript2, div2 = components(layout2)

    return render_template('EDA.html', javascript=javascript, div=div, javascript2=javascript2, div2=div2)


@app.route('/models')
def models():
    return "Model selection"


@app.route('/hand/<letter_input>', methods=["POST", "GET"])
def get_points(letter_input) -> str:
    points = 0
    for l in letter_input.upper():
        points += point_values.get(l)
    return render_template('scrabble_points.html',
    letter_input = letter_input,
    points = points)



#@app.route('/results/<thres_input>', methods=["POST", "GET"]) 
@app.route('/results', methods=["POST", "GET"])     
def results():
    t = 50
    source_df = pd.DataFrame()
    predict_probas = rf.predict_proba(X_test_scaled)[:,1]
    source_df['predict_probas'] = [X*100 for X in predict_probas]
    source_df = source_df.assign(y_true=ytest.values)
    source_df['y_pred'] = np.where(source_df.eval("predict_probas > "+str(t)), 1, 0)

    len_US = len(source_df[source_df.y_true == 0.0])
    len_RoW = len(source_df[source_df.y_true == 1.0])
    US_correct = len(source_df.y_pred[(source_df.y_true == 0) & (source_df.y_pred == 0)])
    RoW_correct = len(source_df.y_pred[(source_df.y_true == 1) & (source_df.y_pred == 1)])

    source = {}
    source['predict_probas'] = list(source_df['predict_probas'])
    source['y_true'] = list(source_df['y_true'])
    source['y_pred'] = list(source_df['y_pred'])
    source['pct_correct'] = [(US_correct/len_US)*100, (RoW_correct/len_RoW)*100]
    source['pct_incorrect'] = [(1-(US_correct/len_US))*100, (1-(RoW_correct/len_RoW))*100]
    source['x'] = ['correct','incorrect']
    source['loc'] = ['USA', 'Rest of World']
    
    source = ColumnDataSource(source)
    values = ["pct_correct", "pct_incorrect"]
    colors = ["palevioletred", "gray"]

    p = figure(x_range = source.data['loc'], y_range = (0,100), title="Classification accuracy at different thresholds", plot_height=300,plot_width=500)

    p.vbar_stack(values, x='loc', width=0.5, source=source, color=colors, legend=['Correct','Incorrect'])

    p.add_tools(HoverTool(tooltips=[("LOCATION", "@loc"), ("PCT CORRECT", "@pct_correct{1.11} %")]))
    
    callback = CustomJS(args=dict(source=source), code="""   
    var predict_probas = source.data.predict_probas
    var y_pred = source.data.y_pred
    var y_true = source.data.y_true    
    var pct_correct = source.data.pct_correct
    var pct_incorrect = source.data.pct_incorrect
    var t = amp.value
    var len_US = 3186
    var len_RoW = 1088;

    for (var i=0; i < predict_probas.length; i++){
        if (predict_probas[i] > t){
            y_pred[i] = 1
        } 
        else {
            y_pred[i] = 0
        }        
    }    
    var num_correct_US = 0
    for (var i=0; i < y_pred.length; i++){
        if(y_pred[i] == y_true[i] & y_pred[i] == 0){
          num_correct_US = num_correct_US + 1;
        }
    }
    
    pct_correct[0] = 100*num_correct_US/len_US
    pct_incorrect[0] = 100 - pct_correct[0]
    
    var num_correct_RoW = 0
    for (var i=0; i < y_pred.length; i++){
        if(y_pred[i] == y_true[i] & y_pred[i] == 1){
          num_correct_RoW = num_correct_RoW + 1;
        }
    }
    
    pct_correct[1] = 100*num_correct_RoW/len_RoW
    pct_incorrect[1] = 100 - pct_correct[1]

    console.log(y_pred)
    source.change.emit();
    """)

    p.xgrid.grid_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.yaxis.major_tick_line_color = None
    p.ygrid.grid_line_color = None
    p.y_range.start = 0

    amp_slider = Slider(start=0.0, end=100, value=50, step= 1,
                        title="Threshold", callback=callback)

    callback.args["amp"] = amp_slider

    layout = column(p,widgetbox(amp_slider))    
    curdoc().add_root(layout)

    javascript, div = components(layout)

    return render_template('results.html', javascript=javascript, div=div)

with open("rf.pkl", "rb") as f:
    rf_model = pickle.load(f)

@app.route("/predict", methods=["POST", "GET"])
def predict():

    x_input = []
    for i in range(len(rf_model.feature_names)):
        f_value = float(
            request.args.get(rf_model.feature_names[i], "0")
            )
        x_input.append(f_value)

    pred_probs = rf_model.predict_proba([x_input]).flat

    return render_template('predict.html',
    feature_names=rf_model.feature_names,
    x_input=x_input,
    prediction=np.argsort(pred_probs)[::-1][0]
    )



@app.route('/graph', methods=["POST", "GET"])
def results_rf():

    #measured = np.random.normal(0, 1, 1000)
    features = rf_model.feature_names
    importances_dict = {}
    for feature, importance in zip(features, rf_model.feature_importances_):
        importances_dict[feature] = importance
    sorted(importances_dict.items(), key=lambda kv: kv[1], reverse=True)[0:10]

    features_df = pd.DataFrame([importances_dict]).transpose().reset_index()
    features_df.columns = ['feature','importance']
    lang = features_df[features_df['feature'].str.contains("language")]
    lang = lang[lang['feature'] != 'language_en']
    lang['importance'].sum()

    features_df = features_df.append({'feature': 'language_non_english','importance': lang['importance'].sum()}, ignore_index=True)
    features_df = features_df.sort_values(by='importance', ascending = False).reset_index(drop=True)
    
    data = {
        'feature': features_df['feature'],
        'importance': features_df['importance']
        }

    y_lab = {'secs_elapsed_std': 'Session duration (std dev)',  
             'num_sessions':'Number of sessions', 
             'age':'Age of user',
             'secs_elapsed_mean':'Session duration (mean)', 
             'day_active_3.0':'First active on Thursday',
             'month_active_4.0':'First active in April',
             'num_different_actions':'Number of different actions',
             'gender_male':'Male user',
             'first_device_type_2_mac desktop':'Used Mac desktop',
             'gender_female':'Female user',
             'language_non_english':'Non-English account'}
    features_df['label'] = features_df['feature'].map(y_lab)

    bar_lengths = list(features_df.importance[0:11])[::-1]
    y = list(features_df.label[0:11])[::-1]
    colors = ['lightpink', 'lightpink', 'lightpink', 'lightpink', 
                'darkkhaki', 'darkkhaki', 'olive', 
                'palevioletred','palevioletred', 'palevioletred', 'palevioletred']

    plot = figure(y_range = y, x_range=(0, 0.13), title="Random forest feature importances", plot_height=300,plot_width=400)


    plot.hbar(y, left=0, right=bar_lengths, height=0.7, color=colors)
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.xaxis.minor_tick_line_color = None
    plot.yaxis.major_tick_line_color = None
    plot.xaxis.major_tick_line_color = None
    
    
    javascript, div = components(plot)
    

    return render_template('results_rf.html', javascript=javascript, div=div)


app.run(port=5000, debug=True)
