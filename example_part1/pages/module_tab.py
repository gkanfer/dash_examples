from dash import html, dcc, callback, Input, Output

import dash

dash.register_page(__name__,name='module', path='/module_tab')
import dash_bootstrap_components as dbc



def layout():
    return html.H1("Image display")